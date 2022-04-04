from typing import List, Optional, Tuple, cast
import gym
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.distributions import Categorical
import numpy as np

from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.models.torch.torch_action_dist import (
    TorchCategorical,
    TorchDistributionWrapper,
)
from ray.rllib.models.catalog import ModelCatalog

from mbag.environment.types import MbagAction, WorldSize
from mbag.environment.blocks import MinecraftBlocks
from .torch_models import MbagModel


def kl_categorical_categorical_no_inf(p: Categorical, q: Categorical) -> torch.Tensor:
    t: torch.Tensor = p.probs * (p.logits - q.logits)
    # t[(q.probs == 0).expand_as(t)] = inf
    t[(p.probs == 0).expand_as(t)] = 0
    return t.sum(-1)


class MbagAutoregressiveActionDistribution(TorchDistributionWrapper):
    """
    An auto-regressive action distribution for the MBAG environment. First, it samples
    a location and action type (e.g., NOOP, PLACE_BLOCK, etc.). Then, it samples a
    block type based on the action type and location.

    The distribution inputs should be of size
    (num_action_types + extra_size) x width x height x depth
    The first num_action_types channels are used as logits for the action_type and
    location. Then, the inputs at the sampled location are passed to
    model.block_id_model, except that the first num_action_types channels are replaced
    by a one-hot vector with the sampled action type.

    This is meant to be used with the models in models.py
    """

    model: MbagModel
    inputs: torch.Tensor  # type: ignore
    model_device: torch.device
    inputs_device: torch.device

    _world_obs: Optional[torch.Tensor]
    _cached_action_type = None
    _cached_block_location = None
    _cached_block_id_logits = None

    PLACEABLE_BLOCK_MASK = torch.tensor(
        [
            block_id in MinecraftBlocks.PLACEABLE_BLOCK_IDS
            for block_id in range(len(MinecraftBlocks.ID2NAME))
        ],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    def __init__(self, inputs: List[torch.Tensor], model: TorchModelV2):
        super().__init__(inputs, model)

        self._world_size = self.model.obs_space.original_space[0].shape[1:]
        self.inputs = self.inputs.reshape(self.inputs.size()[0], -1, *self._world_size)

        self._world_obs = getattr(self.model, "_world_obs", None)
        if self._world_obs is not None:
            self._world_obs = self._world_obs[: self.inputs.size()[0]]

        self.model_device = next(iter(cast(nn.Module, self.model).parameters())).device
        self.inputs_device = self.inputs.device
        self.inputs = self.inputs.to(self.model_device)

    def sample(self):
        # First, sample a block_location and action_type.
        action_type_location_dist = self._action_type_location_distribution()
        action_type, block_location = action_type_location_dist.sample()

        # Then, sample a block_id.
        block_id_dist = self._block_id_distribution(action_type, block_location)
        block_id = block_id_dist.sample()

        self._sampled_logp = self._calculate_logp(
            action_type_location_dist,
            action_type,
            block_location,
            block_id_dist,
            block_id,
        )
        return action_type, block_location, block_id

    def deterministic_sample(self):
        # First, sample a block_location and action_type.
        action_type_location_dist = self._action_type_location_distribution()
        action_type, block_location = action_type_location_dist.deterministic_sample()

        # Then, sample a block_id.
        block_id_dist = self._block_id_distribution(action_type, block_location)
        block_id = block_id_dist.deterministic_sample()

        self._sampled_logp = self._calculate_logp(
            action_type_location_dist,
            action_type,
            block_location,
            block_id_dist,
            block_id,
        )
        return action_type, block_location, block_id

    def sampled_action_logp(self):
        return self._sampled_logp

    def logp(self, actions):
        actions = actions.to(self.model_device)

        action_type = actions[:, 0].long()
        block_location = actions[:, 1].long()
        block_id = actions[:, 2].long()

        action_type_location_dist = self._action_type_location_distribution()
        block_id_dist = self._block_id_distribution(action_type, block_location)

        return self._calculate_logp(
            action_type_location_dist,
            action_type,
            block_location,
            block_id_dist,
            block_id,
        )

    def entropy(self):
        action_type_location_dist = self._action_type_location_distribution()
        if (
            self._cached_action_type is not None
            and self._cached_block_location is not None
        ):
            block_id_dist = self._block_id_distribution(
                self._cached_action_type, self._cached_block_location
            )
        else:
            action_type, block_location = action_type_location_dist.sample()
            block_id_dist = self._block_id_distribution(action_type, block_location)

        # Only count block_id and block_location entropy for actions which use them.
        block_id_use_prob = action_type_location_dist.action_type_distribution.probs[
            :, MbagAction.BLOCK_ID_ACTION_TYPES
        ].sum(1)

        entropy = (
            action_type_location_dist.entropy()
            + block_id_dist.entropy() * block_id_use_prob
        )
        return entropy

    def kl(self, other: "MbagAutoregressiveActionDistribution"):
        action_type_location_dist = self._action_type_location_distribution()
        if (
            self._cached_action_type is not None
            and self._cached_block_location is not None
        ):
            block_id_dist = self._block_id_distribution(
                self._cached_action_type, self._cached_block_location
            )
        else:
            action_type, block_location = action_type_location_dist.sample()
            block_id_dist = self._block_id_distribution(action_type, block_location)

        action_type_location_kl = action_type_location_dist.kl(
            other._action_type_location_distribution()
        )
        block_id_kl = kl_categorical_categorical_no_inf(
            block_id_dist.dist,
            other._block_id_distribution(
                action_type, block_location, skip_cache=True
            ).dist,
        )

        kl = action_type_location_kl + block_id_kl
        if ~torch.all(torch.isfinite(kl)):
            breakpoint()
        return kl

    def _action_type_location_distribution(
        self, mask_logit=-1e8
    ) -> "ActionTypeLocationDistribution":
        action_type_location_logits = self.inputs[:, : MbagAction.NUM_ACTION_TYPES]
        return ActionTypeLocationDistribution(
            action_type_location_logits, self.model, self._world_obs
        )

    def _block_id_distribution(
        self, action_type, block_location, mask_logit=-1e8, skip_cache=False
    ) -> TorchCategorical:
        if (
            self._cached_block_id_logits is None
            or skip_cache
            or not (
                torch.all(self._cached_action_type == action_type)
                and torch.all(self._cached_block_location == block_location)
            )
        ):
            location_inputs = self.inputs.flatten(start_dim=2)[
                torch.arange(self.inputs.size()[0]), :, block_location
            ]
            assert location_inputs.size() == self.inputs.size()[:2]
            hidden_state = location_inputs[:, MbagAction.NUM_ACTION_TYPES :]
            hidden_state_with_action_type = torch.cat(
                [
                    hidden_state,
                    F.one_hot(action_type, MbagAction.NUM_ACTION_TYPES),
                ],
                dim=1,
            )
            block_id_logits = self.model.block_id_model(
                hidden_state_with_action_type,
            )
            if not skip_cache:
                self._cached_block_id_logits = block_id_logits
                self._cached_action_type = action_type
                self._cached_block_location = block_location
        else:
            block_id_logits = self._cached_block_id_logits

        # Mask out logits for placing unplaceable blocks.
        block_id_logits = block_id_logits.clone()
        block_id_logits[
            (action_type == MbagAction.PLACE_BLOCK)[:, None]
            & ~MbagAutoregressiveActionDistribution.PLACEABLE_BLOCK_MASK[None, :]
        ] = mask_logit

        return TorchCategorical(block_id_logits)  # type: ignore

    def _calculate_logp(
        self,
        action_type_location_dist: "ActionTypeLocationDistribution",
        action_type,
        block_location,
        block_id_dist: TorchCategorical,
        block_id,
    ):
        """
        Carefully calculates the log probability of different actions, ignoring parts
        of the action which don't matter (for instance, block_location in a movement
        action).
        """

        action_type_location_logp = action_type_location_dist.logp(
            action_type, block_location
        )

        block_id_logp = block_id_dist.logp(block_id)
        block_id_action_types = torch.tensor(
            MbagAction.BLOCK_ID_ACTION_TYPES, device=block_id.device
        )
        block_id_mask = (action_type[:, None] == block_id_action_types).any(1)
        block_id_logp[~block_id_mask] = 0

        return (action_type_location_logp + block_id_logp).to(self.inputs_device)

    @staticmethod
    def required_model_output_shape(
        action_space: gym.Space, model_config: ModelConfigDict
    ):
        return 1


class ActionTypeLocationDistribution(TorchCategorical):
    """
    Distribution over action types and locations, parameterized by logits of size
    num_action_types x width x height x depth. Returns a tuple (action_type, location).
    """

    inputs: torch.Tensor  # type: ignore
    _world_size: WorldSize

    def __init__(
        self, inputs, model: TorchModelV2, world_obs: Optional[torch.Tensor] = None
    ):
        self._world_obs = world_obs
        assert inputs.size()[1] == MbagAction.NUM_ACTION_TYPES
        super().__init__(self._mask_logits(inputs).flatten(start_dim=1), model=model)
        self._world_size = inputs.size()[-3:]
        self._num_world_blocks = int(np.prod(self._world_size))

    def _tuple_to_index(
        self, action_type: torch.Tensor, location: torch.Tensor
    ) -> torch.Tensor:
        """
        Combine a tuple of (action_type, location) tensors into a single tensor of
        indices.
        """

        return action_type * self._num_world_blocks + location

    def _index_to_tuple(
        self, action_type_location: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Split a single tensor of indices into a tuple (action_type, location).
        """

        action_type = torch.div(
            action_type_location, self._num_world_blocks, rounding_mode="floor"
        )
        location = action_type_location % self._num_world_blocks
        return action_type, location

    def _mask_logits(
        self, location_action_type_logits: torch.Tensor, mask_logit=-1e8
    ) -> torch.Tensor:
        """
        Given location and action_type logits of size
        num_action_types x width x height x depth,
        this masks out action_type + location pairs that are impossible (for instance,
        breaking bedrock or placing a block in mid-air).
        """

        # Clone logits since we're going to modify them.
        location_action_type_logits = location_action_type_logits.clone()

        if self._world_obs is not None:
            # Mask the distribution to blocks that can actually be affected.
            # First, we can't break air or bedrock.
            location_action_type_logits[:, MbagAction.BREAK_BLOCK][
                (
                    (self._world_obs[:, 0] == MinecraftBlocks.AIR)
                    | (self._world_obs[:, 0] == MinecraftBlocks.BEDROCK)
                )
            ] = mask_logit

            # Next, we can only place in locations that are next to a solid block and
            # currently occupied by air.
            solid_block_ids = torch.tensor(
                list(MinecraftBlocks.SOLID_BLOCK_IDS),
                dtype=torch.uint8,
                device=self._world_obs.device,
            )
            solid_blocks = (
                self._world_obs[:, 0, :, :, :, None] == solid_block_ids
            ).any(-1)
            next_to_solid = (
                F.conv3d(
                    solid_blocks[:, None].float(),
                    torch.ones((1, 1, 3, 3, 3), device=solid_blocks.device),
                    padding=1,
                )
                > 0
            ).squeeze(1)
            location_action_type_logits[:, MbagAction.PLACE_BLOCK][
                (self._world_obs[:, 0] != MinecraftBlocks.AIR) | ~next_to_solid
            ] = mask_logit

        return location_action_type_logits

    def sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._index_to_tuple(super().sample())

    def deterministic_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._index_to_tuple(super().deterministic_sample())

    def logp(self, action_type, location) -> torch.Tensor:
        indices = self._tuple_to_index(action_type, location)
        logp: torch.Tensor = super().logp(indices)
        # If for some reason some of the locations given to this method are invalid
        # (i.e., a block can't be placed/broken there), then just set the log probs
        # to 0.
        logp[self.dist.probs.gather(-1, indices[:, None])[:, 0] == 0] = 0
        return logp

    @property
    def action_type_distribution(self) -> Categorical:
        """
        The marginal distribution over action types.
        """

        batch_size = self.inputs.size()[0]
        raw_probs = self.dist.probs.reshape(batch_size, MbagAction.NUM_ACTION_TYPES, -1)
        return Categorical(probs=raw_probs.flatten(start_dim=2).sum(dim=2))

    @property
    def _combined_probs(self) -> torch.Tensor:
        """
        Combines probabilities of actions which are equivalent, i.e. for different
        locations for action types which do not use a location.
        """

        batch_size = self.inputs.size()[0]
        raw_probs = self.dist.probs.reshape(batch_size, MbagAction.NUM_ACTION_TYPES, -1)

        probs_using_location = raw_probs[
            :, MbagAction.BLOCK_LOCATION_ACTION_TYPES
        ].flatten(start_dim=1)
        non_location_action_types = list(
            set(range(MbagAction.NUM_ACTION_TYPES))
            - set(MbagAction.BLOCK_LOCATION_ACTION_TYPES)
        )
        probs_not_using_location = raw_probs[:, non_location_action_types].sum(dim=2)
        combined_probs = torch.cat(
            [probs_using_location, probs_not_using_location], dim=1
        )

        assert combined_probs.size() == (
            batch_size,
            len(MbagAction.BLOCK_LOCATION_ACTION_TYPES) * (self._num_world_blocks - 1)
            + MbagAction.NUM_ACTION_TYPES,
        )
        return combined_probs

    def entropy(self) -> torch.Tensor:
        return cast(torch.Tensor, Categorical(probs=self._combined_probs).entropy())

    def kl(self, other: "ActionTypeLocationDistribution") -> torch.Tensor:
        return kl_categorical_categorical_no_inf(
            Categorical(probs=self._combined_probs),
            Categorical(probs=other._combined_probs),
        )


ModelCatalog.register_custom_action_dist(
    "mbag_autoregressive", MbagAutoregressiveActionDistribution
)
