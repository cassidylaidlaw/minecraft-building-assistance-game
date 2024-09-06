import contextlib
import copy
import warnings
from abc import ABC, abstractmethod
from typing import (
    Any,
    ContextManager,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from gymnasium import spaces
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.typing import TensorType
from torch import nn
from typing_extensions import TypedDict

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.environment.actions import MbagAction
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.mbag_env import DEFAULT_CONFIG as DEFAULT_ENV_CONFIG
from mbag.environment.mbag_env import MbagConfigDict
from mbag.environment.types import (
    CURRENT_BLOCKS,
    CURRENT_PLAYER,
    GOAL_BLOCKS,
    LAST_INTERACTED,
    NO_ONE,
    PLAYER_LOCATIONS,
    WorldSize,
)

ACTION_MASK = "action_mask"
PREV_OBS = "prev_obs"
PREV_OTHER_AGENT_ACTIONS = "prev_other_agent_actions"


class Conv3d1x1x1(nn.Module):
    """
    A 1x1x1 convolution layer.
    """

    bias: Optional[nn.Parameter]

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
        )
        self.weight = nn.Parameter(conv.weight.data)
        assert self.weight.size() == (out_channels, in_channels, 1, 1, 1)
        if conv.bias is None:
            self.bias = None
        else:
            self.bias = nn.Parameter(conv.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c_in, d, h, w = x.size()
        c_out = self.weight.size()[0]
        return (
            F.linear(
                x.permute(0, 2, 3, 4, 1).reshape(-1, c_in),
                self.weight[:, :, 0, 0, 0],
                self.bias,
            )
            .reshape(n, d, h, w, c_out)
            .permute(0, 4, 1, 2, 3)
        )


class MbagModel(ABC, TorchModelV2):
    """
    A model to be used with MbagAutoregressiveActionDistribution.
    """

    @abstractmethod
    def block_id_model(  # noqa: E704
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor: ...


class MbagModelConfig(TypedDict, total=False):
    env_config: MbagConfigDict
    """Environment configuration."""
    num_inventory_obs: int
    """Number of inventory observations to input to the model."""
    embedding_size: int
    """Block ID embedding size."""
    use_extra_features: bool
    """Use extra hand-designed features as input to the network."""
    use_fc_after_embedding: bool
    """Use a fully connected layer after the observation embedding."""
    mask_goal: bool
    """Remove goal information from observations before passing into the network."""
    mask_other_players: bool
    """
    Remove other players from observations (locations, inventories, last interacted)
    before passing into the network.
    """
    fake_state: bool
    """Whether to add fake state to this model so that it's treated as recurrent."""
    hidden_size: int
    """Size of hidden layers."""
    num_action_layers: int
    """Number of extra layers for the action head."""
    num_value_layers: int
    """Number of extra layers for the value head."""
    use_per_location_lstm: bool
    """Include a LSTM operating per-location."""
    lstm_depth: Optional[int]
    """How many layers into the backbone to run the LSTM; default is at the end."""
    num_lstm_layers: int
    """Number of LSTM layers."""
    mask_action_distribution: bool
    """Mask invalid actions in the output action distribution."""
    line_of_sight_masking: bool
    """
    Mask place/break actions which are not in the line of sight of the player.
    See MbagActionDistribution.get_mask for more details.
    """
    scale_obs: bool
    """Scale inventory and timestep observations."""
    vf_scale: float
    """Scale the value function output by this amount."""
    use_prev_blocks: bool
    """Whether to input the last different value of each block type to the network."""
    use_prev_action: bool
    """Whether to input the last action taken to the network."""
    use_prev_other_agent_action: bool
    """Whether to input the last action taken by other agents to the network."""


DEFAULT_CONFIG: MbagModelConfig = {
    "env_config": DEFAULT_ENV_CONFIG,
    "num_inventory_obs": 1,
    "embedding_size": 8,
    "use_extra_features": False,
    "use_fc_after_embedding": False,
    "mask_goal": False,
    "mask_other_players": True,
    "fake_state": False,
    "hidden_size": 16,
    "num_action_layers": 1,
    "num_value_layers": 1,
    "use_per_location_lstm": False,
    "lstm_depth": None,
    "num_lstm_layers": 1,
    "mask_action_distribution": True,
    "line_of_sight_masking": False,
    "scale_obs": False,
    "vf_scale": 1.0,
    "use_prev_blocks": False,
    "use_prev_action": False,
    "use_prev_other_agent_action": False,
}


class MbagTorchModel(TorchModelV2, nn.Module, ABC):
    """
    This base class implements common functionality for PyTorch MBAG models such
    as block type embedding, separate policy and value networks and the value head.
    """

    MASK_LOGIT = -1e8

    _logits: torch.Tensor
    _mask: torch.Tensor

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config,
        name,
        **kwargs,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        if hasattr(obs_space, "original_space"):
            obs_space = cast(Any, obs_space).original_space
        self.preprocessor = get_preprocessor(obs_space)(obs_space)
        if isinstance(obs_space, spaces.Dict):
            obs_space = obs_space.spaces["obs"]
        assert isinstance(obs_space, spaces.Tuple)
        world_obs_space = obs_space[0]
        assert isinstance(world_obs_space, spaces.Box)
        self.world_obs_space: spaces.Box = world_obs_space
        self.world_size = cast(WorldSize, self.world_obs_space.shape[-3:])

        extra_config = copy.deepcopy(DEFAULT_CONFIG)
        if "env_config" in kwargs:
            extra_config["num_inventory_obs"] = kwargs["env_config"]["num_players"]
        extra_config.update(cast(MbagModelConfig, kwargs))
        self.env_config = extra_config["env_config"]
        self.num_inventory_obs = extra_config["num_inventory_obs"]
        self.embedding_size = extra_config["embedding_size"]
        self.use_extra_features = extra_config["use_extra_features"]
        self.use_fc_after_embedding = extra_config["use_fc_after_embedding"]
        self.mask_goal = extra_config["mask_goal"]
        self.mask_other_players = extra_config["mask_other_players"]
        self.fake_state: bool = extra_config["fake_state"]
        self.hidden_size = extra_config["hidden_size"]
        self.num_action_layers = extra_config["num_action_layers"]
        self.num_value_layers = extra_config["num_value_layers"]
        self.use_per_location_lstm = extra_config.get("use_per_location_lstm", False)
        self.lstm_depth = extra_config.get("lstm_depth", None)
        self.num_lstm_layers = extra_config.get("num_lstm_layers", 1)
        self.mask_action_distribution = extra_config["mask_action_distribution"]
        self.line_of_sight_masking = extra_config["line_of_sight_masking"]
        self.scale_obs = extra_config["scale_obs"]
        self.vf_scale = extra_config["vf_scale"]
        self.use_prev_blocks = extra_config["use_prev_blocks"]
        self.use_prev_action = extra_config["use_prev_action"]
        self.use_prev_other_agent_action = extra_config["use_prev_other_agent_action"]

        self.action_mapping = torch.from_numpy(
            MbagActionDistribution.get_action_mapping(self.env_config)
        )

        self.block_id_embedding = nn.Embedding(
            num_embeddings=len(MinecraftBlocks.ID2NAME),
            embedding_dim=self.embedding_size,
        )
        self.player_id_embedding = nn.Embedding(
            # Assume there are no more than 16 players, could be an issue down the line?
            num_embeddings=16,
            embedding_dim=self.embedding_size,
        )

        if self.use_fc_after_embedding:
            self.fc_after_embedding = Conv3d1x1x1(
                self._get_embedding_channels(), self.hidden_size
            )

        self.vf_share_layers: bool = model_config["vf_share_layers"]
        if self.vf_share_layers:
            self.backbone = self._construct_backbone()
        else:
            self.action_backbone = self._construct_backbone()
            self.value_backbone = self._construct_backbone(is_value_network=True)

        self.action_head = self._construct_action_head()
        self.value_head = self._construct_value_head()
        self.goal_head = self._construct_goal_head()

        if self.use_per_location_lstm:
            assert self.vf_share_layers
            assert not self.model_config.get("_time_major", False)
            self.per_location_lstm = nn.LSTM(
                input_size=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_lstm_layers,
                batch_first=True,
            )

        if self.use_prev_blocks:
            self.view_requirements[PREV_OBS] = ViewRequirement(
                data_col=SampleBatch.OBS,
                space=self.view_requirements[SampleBatch.OBS].space,
                shift=-1,
            )

    @property
    def device(self) -> torch.device:
        return self.block_id_embedding.weight.device

    def _get_embedding_planes(self) -> int:
        """
        Return how many "planes" of data of size embedding_size are present in the
        embedded observation.
        """

        # We always have in-planes for current blocks, player locations, and
        # last interacted.
        planes = 3
        if not self.mask_goal:
            planes += 1
        if self.use_prev_blocks:
            planes += 1
        return planes

    def _get_embedding_channels(self) -> int:
        """Get the number of channels in the embedded observation."""
        in_channels = self._get_embedding_planes() * self.embedding_size
        # Add inventory observation as extra input channels.
        in_channels += MinecraftBlocks.NUM_BLOCKS * self.num_inventory_obs
        # Timestep observation
        in_channels += 1
        if self.use_extra_features:
            in_channels += 1
        if self.use_prev_action:
            in_channels += MbagActionDistribution.NUM_CHANNELS
        if self.use_prev_other_agent_action:
            in_channels += MbagActionDistribution.NUM_CHANNELS
        return in_channels

    def _get_backbone_in_channels(self) -> int:
        """Get the number of channels input to the backbone."""

        if self.use_fc_after_embedding:
            return self.hidden_size
        else:
            return self._get_embedding_channels()

    def _get_head_in_channels(self) -> int:
        """
        Get the number of channels output from the backbone which are used as input to
        the value and block ID heads.
        """

        return self.hidden_size

    def _construct_backbone(self, is_value_network=False) -> nn.Module:
        """
        Construct the main backbone of the model. This takes as input the result of
        _get_embedded_obs and should output a tensor of shape
            (batch_size, self._get_head_in_channels()) + self.world_size.
        When vf_share_layers is True, this is called once; when vf_share_layers is
        False, it is called twice, and the value backbone is passed
        is_value_network=True.
        """

        raise NotImplementedError()

    def _construct_action_head(self) -> nn.Module:
        """
        Construct the head which outputs the action distribution logits.
        """

        action_head_layers: List[nn.Module] = []
        for layer_index in range(self.num_action_layers):
            action_head_layers.append(
                Conv3d1x1x1(
                    (
                        self._get_head_in_channels()
                        if layer_index == 0
                        else self.hidden_size
                    ),
                    (
                        MbagActionDistribution.NUM_CHANNELS
                        if layer_index == self.num_action_layers - 1
                        else self.hidden_size
                    ),
                )
            )
            if layer_index < self.num_action_layers - 1:
                action_head_layers.append(nn.LeakyReLU())

        return nn.Sequential(*action_head_layers)

    def _construct_value_head(self) -> nn.Module:
        """
        Construct the head which takes in the output of the value backbone and
        outputs a one-dimensional value estimate.
        """

        value_head_layers: List[nn.Module] = [
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
        ]
        for layer_index in range(self.num_value_layers):
            value_head_layers.append(
                nn.Linear(
                    (
                        self._get_head_in_channels()
                        if layer_index == 0
                        else self.hidden_size
                    ),
                    1 if layer_index == self.num_value_layers - 1 else self.hidden_size,
                )
            )
            if layer_index < self.num_value_layers - 1:
                value_head_layers.append(nn.LeakyReLU())
        return nn.Sequential(*value_head_layers)

    def _construct_goal_head(self) -> nn.Module:
        """
        Construct the head which takes in the output of the value backbone and
        outputs a goal estimate.
        """

        return nn.Sequential(
            Conv3d1x1x1(self._get_head_in_channels(), self.hidden_size),
            nn.LeakyReLU(),
            Conv3d1x1x1(self.hidden_size, MinecraftBlocks.NUM_BLOCKS),
        )

    def _get_embedded_actions(
        self,
        actions: torch.Tensor,
    ):
        """
        Transform a raw tensor of actions into the input for the network backbone.
        actions can either be of shape (batch_size,) for flat actions or
        (batch_size, 3).
        """

        batch_size = actions.size()[0]
        width, height, depth = self.world_size

        if actions.ndim == 1:
            flat_actions = actions
        else:
            self.action_mapping = self.action_mapping.to(actions.device)
            flat_actions = torch.all(
                self.action_mapping[None, :, :] == actions[:, None, :],
                dim=2,
            ).nonzero()[:, 1]

        # A trick that gets reasonable action embeddings by taking the gradient of each
        # actions' probability with respect to arbitrary action distribution
        # probabilities.
        with torch.enable_grad():
            dummy_probs = torch.zeros(
                batch_size,
                MbagActionDistribution.NUM_CHANNELS,
                width,
                height,
                depth,
                device=actions.device,
            ).requires_grad_(True)
            dummy_flat_probs = MbagActionDistribution.to_flat_torch(
                self.env_config, dummy_probs
            )
            dummy_flat_probs[
                torch.arange(batch_size),
                flat_actions,
            ].sum().backward()
            assert dummy_probs.grad is not None
            embedded_actions = dummy_probs.grad.permute(0, 2, 3, 4, 1)

        return embedded_actions

    def _get_embedded_obs(
        self,
        world_obs: torch.Tensor,
        inventory_obs: torch.Tensor,
        timestep: torch.Tensor,
        prev_blocks: Optional[torch.Tensor] = None,
        prev_actions: Optional[torch.Tensor] = None,
        prev_other_agent_actions: Optional[torch.Tensor] = None,
        include_last_interacted: bool = True,
    ):
        """
        Transform a raw observation into the input for the network backbone.
        """

        embedded_blocks = self.block_id_embedding(world_obs[:, CURRENT_BLOCKS])
        embedded_obs_pieces = [embedded_blocks]
        if not self.mask_goal:
            embedded_goal_blocks = self.block_id_embedding(world_obs[:, GOAL_BLOCKS])
            embedded_obs_pieces.append(embedded_goal_blocks)
        if self.use_prev_blocks:
            assert prev_blocks is not None
            embedded_prev_blocks = self.block_id_embedding(prev_blocks)
            embedded_obs_pieces.append(embedded_prev_blocks)

        player_locations = world_obs[:, PLAYER_LOCATIONS].clone()
        if self.mask_other_players:
            player_locations[player_locations != CURRENT_PLAYER] = NO_ONE

        embedded_player_locations = self.player_id_embedding(player_locations)
        embedded_obs_pieces.append(embedded_player_locations)

        if self.mask_other_players and inventory_obs.ndim > 2:
            assert self.num_inventory_obs == 1
            flat_inventory_obs = inventory_obs[:, 0]
        else:
            inventory_obs = inventory_obs.flatten(start_dim=1)
            flat_inventory_obs = torch.zeros(
                inventory_obs.size()[0],
                MinecraftBlocks.NUM_BLOCKS * self.num_inventory_obs,
                device=inventory_obs.device,
            )
            flat_inventory_obs[:, : inventory_obs.size()[1]] = inventory_obs

        inventory_piece = flat_inventory_obs[:, None, None, None, :].expand(
            *embedded_obs_pieces[0].size()[:-1], -1
        )
        if self.scale_obs:
            inventory_piece = inventory_piece / 64
        embedded_obs_pieces.append(inventory_piece)
        timestep_piece = timestep[:, None, None, None, None].expand(
            *embedded_obs_pieces[0].size()[:-1], 1
        )
        if self.scale_obs:
            timestep_piece = timestep_piece / 1000
        embedded_obs_pieces.append(timestep_piece)
        if self.use_extra_features:
            # Feature for if goal block is the same as the current block at each
            # location.
            embedded_obs_pieces.append(
                (world_obs[:, 0] == world_obs[:, 2]).float()[..., None]
            )

        if include_last_interacted:
            last_interacted = world_obs[:, LAST_INTERACTED].clone()
            if self.mask_other_players:
                last_interacted[last_interacted > CURRENT_PLAYER] = CURRENT_PLAYER
            embedded_last_interacted = self.player_id_embedding(last_interacted)
            embedded_obs_pieces.append(embedded_last_interacted)

        if self.use_prev_action:
            if prev_actions is None:
                raise ValueError(
                    "prev_actions must be provided if use_prev_action is True"
                )
            embedded_prev_actions = self._get_embedded_actions(prev_actions)
            embedded_obs_pieces.append(embedded_prev_actions)
        if self.use_prev_other_agent_action:
            if prev_other_agent_actions is None:
                raise ValueError(
                    "prev_other_agent_actions must be provided if use_prev_other_agent_action is True"
                )
            embedded_prev_other_agent_actions = self._get_embedded_actions(
                prev_other_agent_actions
            )
            embedded_obs_pieces.append(embedded_prev_other_agent_actions)

        embedded_obs = torch.cat(embedded_obs_pieces, dim=-1)

        return embedded_obs.permute(0, 4, 1, 2, 3)

    def _run_lstm(
        self, backbone_out: torch.Tensor, state_in: List[torch.Tensor], seq_lens
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        flat_backbone_out = backbone_out.flatten(start_dim=1)
        flat_backone_out_with_time: torch.Tensor = add_time_dimension(
            flat_backbone_out,
            seq_lens=seq_lens,
            framework="torch",
            time_major=False,
        )
        # Should be of size (batch_size, max_seq_len, hidden_size, width, height, depth).
        backbone_out_with_time = flat_backone_out_with_time.reshape(
            *flat_backone_out_with_time.size()[:2],
            *backbone_out.size()[1:],
        )
        batch_size, max_seq_len = backbone_out_with_time.size()[:2]
        assert (
            backbone_out_with_time.size()[2:] == (self.hidden_size,) + self.world_size
        )
        backbone_out_per_location = backbone_out_with_time.permute(
            0, 3, 4, 5, 1, 2
        ).flatten(end_dim=3)
        # State in should be of size (batch_size, hidden_size, width, height, depth).
        state_in_per_location = tuple(
            state.permute(0, 2, 3, 4, 1).flatten(end_dim=3)[None].contiguous()
            for state in state_in
        )

        lstm_out_per_location: torch.Tensor
        state_out_per_location: torch.Tensor
        lstm_out_per_location, state_out_per_location = self.per_location_lstm(
            backbone_out_per_location, state_in_per_location
        )
        lstm_out_with_time = lstm_out_per_location.reshape(
            batch_size,
            *self.world_size,
            max_seq_len,
            self.hidden_size,
        ).permute(0, 4, 5, 1, 2, 3)
        state_out = [
            state.reshape(batch_size, *self.world_size, self.hidden_size).permute(
                0, 4, 1, 2, 3
            )
            for state in state_out_per_location
        ]

        lstm_out = lstm_out_with_time.flatten(end_dim=1)

        return lstm_out, state_out

    def forward(self, input_dict, state, seq_lens, mask_logits=True):
        obs = input_dict[SampleBatch.OBS]
        self._world_obs, self._inventory_obs, self._timestep = obs
        self._world_obs = self._world_obs.to(self.device).long()
        self._inventory_obs = self._inventory_obs.to(self.device).long()
        self._timestep = self._timestep.to(self.device)

        if seq_lens is not None:
            # Fix RLlib issue where sometimes seq_lens is too long.
            if self._world_obs.size()[0] == 1:
                seq_lens = torch.ones(
                    1, dtype=torch.long, device=self._world_obs.device
                )

        state = [
            state_var.to(self.device) if state_var is not None else None
            for state_var in state
        ]

        prev_blocks: Optional[torch.Tensor] = None
        if self.use_prev_blocks:
            start_prev_blocks: torch.Tensor
            state, start_prev_blocks = state[:-1], state[-1]
            start_prev_blocks = start_prev_blocks.long()  # Size (B, W, H, D)
            current_blocks_with_time: torch.Tensor = (  # Size (B, T, W, H, D)
                add_time_dimension(
                    self._world_obs[:, CURRENT_BLOCKS],
                    seq_lens=seq_lens,
                    framework="torch",
                    time_major=False,
                ).long()
            )
            prev_blocks = torch.concatenate(
                [start_prev_blocks[:, None], current_blocks_with_time[:, :-1]],
                dim=1,
            ).flatten(end_dim=1)
            end_current_blocks = current_blocks_with_time[:, -1]

        self._prev_actions: Optional[torch.Tensor] = None
        if SampleBatch.PREV_ACTIONS in input_dict:
            self._prev_actions = input_dict[SampleBatch.PREV_ACTIONS].to(self.device)
        self._prev_other_agent_actions: Optional[torch.Tensor] = None
        if PREV_OTHER_AGENT_ACTIONS in input_dict:
            self._prev_other_agent_actions = input_dict[PREV_OTHER_AGENT_ACTIONS].to(
                self.device
            )

        self._embedded_obs = self._get_embedded_obs(
            self._world_obs,
            self._inventory_obs,
            self._timestep,
            prev_blocks,
            self._prev_actions,
            self._prev_other_agent_actions,
        )
        if self.use_fc_after_embedding:
            self._backbone_in = self.fc_after_embedding(self._embedded_obs)
        else:
            self._backbone_in = self._embedded_obs

        self._amp_or_nothing: ContextManager = contextlib.nullcontext()
        if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
            self._amp_or_nothing = torch.autocast("cuda", dtype=torch.bfloat16)

        with self._amp_or_nothing:
            backbone = self.backbone if self.vf_share_layers else self.action_backbone

            if self.lstm_depth is not None:
                assert self.use_per_location_lstm
                assert not (
                    isinstance(backbone, SeparatedTransformerEncoder)
                    and backbone.interleave_lstm
                )
                lstm_in = backbone(
                    self._backbone_in, layer_indices=range(0, self.lstm_depth)
                )
                lstm_out, state = self._run_lstm(lstm_in, state, seq_lens)
                num_layers: int = self.num_layers
                self._backbone_out = backbone(
                    lstm_out, layer_indices=range(self.lstm_depth, num_layers)
                )
            else:
                if (
                    isinstance(backbone, SeparatedTransformerEncoder)
                    and backbone.interleave_lstm
                ):
                    assert not self.use_per_location_lstm
                    self._backbone_out, state = backbone(
                        self._backbone_in, state, seq_lens
                    )
                else:
                    self._backbone_out = backbone(self._backbone_in)
                    if self.use_per_location_lstm:
                        self._backbone_out, state = self._run_lstm(
                            self._backbone_out, state, seq_lens
                        )

            self._backbone_out_shape = self._backbone_out.size()[1:]
            assert self._backbone_out_shape[0] == self._get_head_in_channels()

            self._logits = self.action_head(self._backbone_out)
            self._flat_logits = MbagActionDistribution.to_flat_torch_logits(
                self.env_config, self._logits
            )

        self._logits = self._logits.float()
        self._flat_logits = self._flat_logits.float()
        state = [state_var.float() for state_var in state]

        if self.use_prev_blocks:
            state.append(end_current_blocks.clone())

        if mask_logits and self.mask_action_distribution:
            if ACTION_MASK in input_dict and torch.any(input_dict[ACTION_MASK]):
                self._mask = input_dict[ACTION_MASK]
            else:
                numpy_mask = MbagActionDistribution.get_mask_flat(
                    self.env_config,
                    convert_to_numpy(obs),
                    line_of_sight_masking=self.line_of_sight_masking,
                )
                self._mask = torch.from_numpy(numpy_mask).to(self._flat_logits.device)
            self._flat_logits[~self._mask] = MbagTorchModel.MASK_LOGIT
        else:
            self._mask = torch.ones_like(self._flat_logits, dtype=torch.bool)

        return self._flat_logits, state

    @property
    def logits(self) -> torch.Tensor:
        return self._flat_logits

    def action_mask(self):
        return self._mask

    def block_id_model(self, head_input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def value_function(self):
        with self._amp_or_nothing:
            if self.vf_share_layers:
                vf = self.value_head(self._backbone_out).squeeze(1)
            else:
                vf = self.value_head(self.value_backbone(self._backbone_in)).squeeze(1)
        return vf.float() * self.vf_scale

    def goal_predictor(self) -> torch.Tensor:
        with self._amp_or_nothing:
            goal_preds: torch.Tensor = self.goal_head(self._backbone_out)
        return goal_preds.float()

    def get_initial_state(self):
        state: List[TensorType]
        if self.use_per_location_lstm:
            state = [
                torch.zeros((self.hidden_size, *self.world_size)) for _ in range(2)
            ]
        else:
            if self.fake_state:
                state = [torch.zeros(1)]
            else:
                state = super().get_initial_state()
        if self.use_prev_blocks:
            state.append(self._get_prev_blocks_initial_state())
        return state

    def _get_prev_blocks_initial_state(self):
        return torch.zeros(self.world_size, dtype=torch.uint8)

    def compute_priors_and_value(self, obs, state_in=[], action_mask=None):
        batch_size = len(obs)
        obs = convert_to_torch_tensor(
            np.stack([self.preprocessor.transform(o) for o in obs], axis=0)
        )
        input_dict = {
            SampleBatch.OBS: restore_original_dimensions(obs, self.obs_space, "torch"),
        }

        if action_mask is not None:
            action_mask = convert_to_torch_tensor(action_mask)
            input_dict[ACTION_MASK] = action_mask

        with torch.no_grad():
            logits, state_out = self.forward(
                input_dict,
                state_in,
                np.ones(batch_size, dtype=int),
                mask_logits=False,
            )
            value = self.value_function()
            priors = logits.softmax(dim=-1)

            priors = priors.cpu().numpy()
            value = value.cpu().numpy()

            return priors, value, [state.detach() for state in state_out]


class ResidualBlock(nn.Module):
    """
    Implements a residual network block with two 3D convolutions, batch norm,
    and ReLU.
    """

    def __init__(
        self,
        channels: int,
        filter_size: int = 3,
        use_bn: bool = True,
        use_skip_connection: bool = True,
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=filter_size,
            stride=1,
            padding=(filter_size - 1) // 2,
        )
        self.bn1 = nn.BatchNorm3d(channels) if use_bn else nn.Identity()
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=filter_size,
            stride=1,
            padding=(filter_size - 1) // 2,
        )
        self.bn2 = nn.BatchNorm3d(channels) if use_bn else nn.Identity()
        self.relu2 = nn.ReLU()

        self.use_skip_connection = use_skip_connection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_skip_connection:
            out = out + x
        out = self.relu2(out)
        return out


class UNet3d(nn.Module):
    """
    Implements a model similar to U-Nets, but for 3d data.
    """

    def __init__(
        self,
        size: int,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        fc_layer: nn.Module = nn.Identity(),
        grow_factor: float = 2.0,
        use_bn: bool = False,
    ):
        """
        Expects inputs of shape
        (batch, in_channels, size, size, size)
        and produces outputs of shape
        (batch, out_channels, size, size, size)
        """
        super().__init__()

        self.size = size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.fc_layer = fc_layer

        self.down_layers: List[nn.Module] = []
        self.up_layers: List[nn.Module] = []
        layer_size = self.size
        for layer_index in range(self.num_layers):
            down_in_channels = self.in_channels * int(grow_factor**layer_index)
            down_out_channels = self.in_channels * int(grow_factor ** (layer_index + 1))
            down_layer = nn.Sequential(
                nn.Conv3d(
                    in_channels=down_in_channels,
                    out_channels=down_out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                *([nn.BatchNorm3d(down_out_channels)] if use_bn else []),
                nn.LeakyReLU(),
            )
            self.down_layers.append(down_layer)
            self.add_module(f"down_{layer_index}", down_layer)

            up_in_channels = (
                self.in_channels * 2 * int(grow_factor ** (layer_index + 1))
            )
            up_out_channels = self.in_channels * int(grow_factor**layer_index)
            up_layer = nn.Sequential(
                nn.ConvTranspose3d(
                    in_channels=up_in_channels,
                    out_channels=up_out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1 if layer_size % 2 == 0 else 0,
                ),
                *([nn.BatchNorm3d(up_out_channels)] if use_bn else []),
                nn.LeakyReLU(),
            )
            self.up_layers.append(up_layer)
            self.add_module(f"up_{layer_index}", up_layer)

            layer_size = (layer_size + 1) // 2

        self.fc_layer_size = (
            (layer_size**3) * self.in_channels * int(grow_factor**self.num_layers)
        )

        # Final 1x1x1 convolution to get the right number of out channels.
        self.final_layer = nn.Conv3d(
            in_channels=self.in_channels * 2,
            out_channels=self.out_channels,
            kernel_size=1,
        )

    def set_fc_layer(self, fc_layer: nn.Module):
        self.fc_layer = fc_layer

    def forward(self, inputs: torch.Tensor, *extra_fc_inputs):
        activations = [inputs]
        for down_layer in self.down_layers:
            activations.append(down_layer(activations[-1]))

        fc_layer_inputs = activations[-1].flatten(start_dim=1)
        outputs = self.fc_layer(fc_layer_inputs, *extra_fc_inputs)
        outputs = outputs.reshape(activations[-1].size())

        for layer_index, up_layer in reversed(list(enumerate(self.up_layers))):
            w, h, d = outputs.size()[-3:]
            layer_activations = F.pad(activations[layer_index + 1], (0, 2) * 3)[
                ..., :w, :h, :d
            ]
            layer_inputs = torch.cat([layer_activations, outputs], dim=1)
            outputs = up_layer(layer_inputs)

        w, h, d = outputs.size()[-3:]
        layer_activations = F.pad(activations[0], (0, 2) * 3)[..., :w, :h, :d]
        final_layer_inputs = torch.cat([layer_activations, outputs], dim=1)

        w, h, d = inputs.size()[-3:]
        final_layer_inputs = final_layer_inputs[..., :w, :h, :d]

        return self.final_layer(final_layer_inputs)


class MbagConvolutionalModelConfig(MbagModelConfig, total=False):
    num_conv_1_layers: int
    """Number of 1x1x1 convolutions before the main backbone."""
    num_layers: int
    use_resnet: bool
    filter_size: int
    hidden_channels: int
    num_unet_layers: int
    """Number of layers to include in a UNet3d, if any."""
    unet_grow_factor: float
    unet_use_bn: bool


CONV_DEFAULT_CONFIG: MbagConvolutionalModelConfig = {
    "env_config": DEFAULT_CONFIG["env_config"],
    "embedding_size": DEFAULT_CONFIG["embedding_size"],
    "use_extra_features": DEFAULT_CONFIG["use_extra_features"],
    "mask_goal": DEFAULT_CONFIG["mask_goal"],
    "hidden_size": DEFAULT_CONFIG["hidden_size"],
    "num_action_layers": DEFAULT_CONFIG["num_action_layers"],
    "num_value_layers": DEFAULT_CONFIG["num_value_layers"],
    "fake_state": DEFAULT_CONFIG["fake_state"],
    "num_conv_1_layers": 0,
    "num_layers": 3,
    "use_resnet": False,
    "filter_size": 3,
    "hidden_channels": 32,
    "num_unet_layers": 0,
    "unet_grow_factor": 2.0,
    "unet_use_bn": False,
}


class MbagConvolutionalModel(MbagTorchModel):
    """
    Has an all-convolutional backbone and separate heads for each part of the
    autoregressive action distribution.
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config,
        name,
        **kwargs,
    ):
        extra_config: MbagConvolutionalModelConfig = copy.deepcopy(CONV_DEFAULT_CONFIG)
        extra_config.update(cast(MbagConvolutionalModelConfig, kwargs))
        self.num_conv_1_layers = extra_config["num_conv_1_layers"]
        self.num_layers = extra_config["num_layers"]
        self.use_resnet = extra_config["use_resnet"]
        self.filter_size = extra_config["filter_size"]
        self.hidden_channels = extra_config["hidden_channels"]
        self.num_unet_layers = extra_config["num_unet_layers"]
        self.unet_grow_factor = extra_config["unet_grow_factor"]
        self.unet_use_bn = extra_config["unet_use_bn"]

        super().__init__(
            obs_space, action_space, num_outputs, model_config, name, **kwargs
        )

    def _construct_backbone(self, is_value_network=False) -> nn.Module:
        include_unet = not is_value_network
        backbone_layers: List[nn.Module] = []
        layer_index = 0
        num_layers = self.num_conv_1_layers + self.num_layers
        while layer_index < num_layers:
            if layer_index < self.num_conv_1_layers:
                filter_size = 1
            else:
                filter_size = self.filter_size
            if layer_index == 0:
                in_channels = self._get_backbone_in_channels()
            else:
                in_channels = self.hidden_channels

            if (
                self.use_resnet
                and in_channels == self.hidden_channels
                and layer_index + 2 < num_layers
            ):
                backbone_layers.append(ResidualBlock(channels=self.hidden_channels))
                layer_index += 2
            else:
                backbone_layers.append(
                    nn.Conv3d(
                        in_channels=in_channels,
                        out_channels=self.hidden_channels,
                        kernel_size=filter_size,
                        stride=1,
                        padding=(filter_size - 1) // 2,
                    )
                )
                backbone_layers.append(nn.LeakyReLU())
                layer_index += 1
        if include_unet and self.num_unet_layers > 0:
            self.unet = UNet3d(
                self.world_obs_space.shape[1],
                self.hidden_channels,
                self.hidden_channels,
                self.num_unet_layers,
                grow_factor=self.unet_grow_factor,
                use_bn=self.unet_use_bn,
            )
            backbone_layers.append(self.unet)
        else:
            backbone_layers = backbone_layers[:-1]  # Remove last ReLU.
        return nn.Sequential(*backbone_layers)


ModelCatalog.register_custom_model("mbag_convolutional_model", MbagConvolutionalModel)


class MbagRecurrentConvolutionalModelConfig(MbagConvolutionalModelConfig):
    pass


RECURRENT_CONV_DEFAULT_CONFIG: MbagRecurrentConvolutionalModelConfig = {
    **CONV_DEFAULT_CONFIG,
}


class AddTimeDimRNN(nn.Module):
    _rnn_state: Tuple[torch.Tensor, torch.Tensor]
    _outputs: torch.Tensor
    _new_state: List[torch.Tensor]

    def __init__(self, rnn: nn.Module):
        super().__init__()
        self.rnn = rnn

    def set_state_seq_lens(self, rnn_state, seq_lens):
        if isinstance(seq_lens, np.ndarray):
            seq_lens = torch.Tensor(seq_lens).int()
        self._seq_lens = seq_lens
        self._rnn_state = rnn_state

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        max_seq_len = inputs.shape[0] // self._seq_lens.shape[0]
        input_shape = inputs.size()[1:]
        inputs = inputs.reshape(-1, max_seq_len, *input_shape)
        outputs, new_state = self.rnn(
            inputs,
            [self._rnn_state[0].unsqueeze(0), self._rnn_state[1].unsqueeze(0)],
        )
        self._outputs = outputs.reshape(-1, *input_shape)
        self._new_state = [new_state[0].squeeze(0), new_state[1].squeeze(0)]
        return self._outputs

    def get_outputs(self) -> torch.Tensor:
        return self._outputs

    def get_new_state(self):
        return self._new_state


class MbagRecurrentConvolutionalModel(MbagModel, nn.Module):
    _logits: torch.Tensor

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config,
        name,
        **kwargs,
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        extra_config = RECURRENT_CONV_DEFAULT_CONFIG
        extra_config.update(kwargs)  # type: ignore

        self.conv_model = MbagConvolutionalModel(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name=f"{name}.conv_model",
            **extra_config,
        )
        assert hasattr(self.conv_model, "unet")
        unet: UNet3d = self.conv_model.unet
        self.rnn_hidden_dim = unet.fc_layer_size

        self.lstm = nn.LSTM(self.rnn_hidden_dim, self.rnn_hidden_dim, batch_first=True)
        self.rnn = AddTimeDimRNN(self.lstm)
        unet.set_fc_layer(self.rnn)

        if self.model_config["vf_share_layers"]:
            value_layers: List[nn.Module] = []
            hidden_channels = extra_config["hidden_channels"]
            for layer_index in range(extra_config["num_value_layers"]):
                value_layers.append(
                    nn.Linear(
                        self.rnn_hidden_dim if layer_index == 0 else hidden_channels,
                        hidden_channels,
                    )
                )
                value_layers.append(nn.LeakyReLU())
            self.value_head = nn.Sequential(
                *value_layers,
                nn.Linear(
                    self.rnn_hidden_dim if len(value_layers) == 0 else hidden_channels,
                    1,
                    bias=True,
                ),
            )
        else:
            warnings.warn(
                "without vf_share_layers, the value function will not be recurrent"
            )

    def get_initial_state(self):
        # Place hidden states on same device as model.
        param = next(iter(self.lstm.parameters()))
        h = [
            param.new(1, self.rnn_hidden_dim).zero_().squeeze(0),
            param.new(1, self.rnn_hidden_dim).zero_().squeeze(0),
        ]
        return h

    def value_function(self):
        if self.model_config["vf_share_layers"]:
            return self.value_head(self.rnn.get_outputs()).squeeze(1) * self.vf_scale
        else:
            return self.conv_model.value_function()

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        self.rnn.set_state_seq_lens(state, seq_lens)
        self._logits, _ = self.conv_model.forward(input_dict, state, seq_lens)
        new_state = self.rnn.get_new_state()

        return self._logits, new_state

    @property
    def logits(self) -> torch.Tensor:
        return self._logits

    def block_id_model(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv_model.block_id_model(inputs)


ModelCatalog.register_custom_model(
    "mbag_recurrent_convolutional_model", MbagRecurrentConvolutionalModel
)


class View(nn.Module):
    def __init__(self, *shape: int):
        super().__init__()
        self.shape = shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*self.shape)


class Permute(nn.Module):
    def __init__(self, *dims: int):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)


class SeparatedTransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        norm_first: bool = False,
        n_spatial_dims: int = 3,
        interleave_lstm: bool = False,
        fix_interleave_lstm: bool = False,
    ):
        super().__init__()

        self.d_model = d_model
        self.norm_first = norm_first
        self.n_spatial_dims = n_spatial_dims
        self.interleave_lstm = interleave_lstm
        self.fix_interleave_lstm = fix_interleave_lstm

        self.pre_lstm_layer_norms = nn.ModuleList()
        self.layers: List[Union[nn.TransformerEncoderLayer, nn.LSTM]] = []
        for layer_index in range(num_layers):
            layer: Union[nn.TransformerEncoderLayer, nn.LSTM]
            if (
                self.interleave_lstm
                and (layer_index + 1) % (self.n_spatial_dims + 1) == 0
            ):
                layer = nn.LSTM(
                    input_size=d_model,
                    hidden_size=d_model,
                    num_layers=1,
                    batch_first=True,
                )
                if self.norm_first:
                    pre_lstm_layer_norm = nn.LayerNorm(d_model)
                    self.pre_lstm_layer_norms.append(pre_lstm_layer_norm)
            else:
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    batch_first=True,
                    norm_first=norm_first,
                )
            self.add_module(f"layer_{layer_index}", layer)
            self.layers.append(layer)

        if self.norm_first:
            self.post_layer_norm = nn.LayerNorm(d_model)

    def _layer_forward(self, layer: nn.TransformerEncoderLayer, x: torch.Tensor):
        return layer(x)

    def _batched_layer_forward(
        self, layer: nn.TransformerEncoderLayer, x: torch.Tensor
    ):
        # PyTorch only supports batch sizes up to 2 ** 16 - 1, so we need to split
        # the batch into smaller chunks.
        batch_size = x.size()[0]
        if batch_size <= 2**16 - 1:
            return self._layer_forward(layer, x)
        else:
            minibatch_size = 2**15
            num_minibatches = (batch_size + minibatch_size - 1) // minibatch_size
            minibatch_outputs = []
            for i in range(num_minibatches):
                minibatch_outputs.append(
                    self._layer_forward(
                        layer, x[i * minibatch_size : (i + 1) * minibatch_size]
                    )
                )
            return torch.cat(minibatch_outputs, dim=0)

    def _run_layer(
        self, layer_index: int, x: torch.Tensor, state: List[torch.Tensor], seq_lens
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        assert len(x.size()) - 2 == self.n_spatial_dims
        layer = self.layers[layer_index]

        if isinstance(layer, nn.LSTM):
            lstm_index = layer_index // (self.n_spatial_dims + 1)
            lstm_state = state[2 * lstm_index : 2 * (lstm_index + 1)]
            x, state = self._run_lstm(
                layer,
                x,
                lstm_state,
                seq_lens,
                pre_layer_norm=(
                    self.pre_lstm_layer_norms[lstm_index] if self.norm_first else None
                ),
            )
        else:
            if self.interleave_lstm and self.fix_interleave_lstm:
                spatial_dim = layer_index % (self.n_spatial_dims + 1)
            else:
                # This is now incorrect with interleaved LSTMs.
                spatial_dim = layer_index % self.n_spatial_dims
            permutation = (
                (0,)
                + tuple(
                    other_spatial_dim + 2
                    for other_spatial_dim in range(self.n_spatial_dims)
                    if other_spatial_dim != spatial_dim
                )
                + (spatial_dim + 2, 1)
            )
            inverse_permutation = tuple(
                permutation.index(dim) for dim in range(len(x.size()))
            )
            x_permuted = x.permute(*permutation)
            layer_input = x_permuted.flatten(end_dim=-3)
            # We use the default (math) kernel for attention b/c the sequence lengths are
            # super short, so it ends up being faster.
            if hasattr(torch.backends.cuda, "sdp_kernel"):
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_mem_efficient=False
                ):
                    layer_output = self._batched_layer_forward(layer, layer_input)
            else:
                layer_output = self._batched_layer_forward(layer, layer_input)
            x = layer_output.reshape(x_permuted.size()).permute(*inverse_permutation)
            state = []

        if x.device.type == "cuda" and torch.cuda.is_bf16_supported():
            x = x.bfloat16()

        return x, state

    def _run_lstm(
        self,
        lstm: nn.LSTM,
        x: torch.Tensor,
        state_in: List[torch.Tensor],
        seq_lens,
        pre_layer_norm: Optional[nn.LayerNorm] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = x.float()
        world_size = x.size()[2:]
        flat_x = x.flatten(start_dim=1)
        flat_backone_out_with_time: torch.Tensor = add_time_dimension(
            flat_x,
            seq_lens=seq_lens,
            framework="torch",
            time_major=False,
        )
        # Should be of size (batch_size, max_seq_len, hidden_size, width, height, depth).
        x_with_time = flat_backone_out_with_time.reshape(
            *flat_backone_out_with_time.size()[:2],
            *x.size()[1:],
        )
        batch_size, max_seq_len = x_with_time.size()[:2]
        assert x_with_time.size()[2:] == (self.d_model,) + world_size
        # Should be of size (batch_size * width * height * depth, max_seq_len, hidden_size).
        x_per_location = x_with_time.permute(0, 3, 4, 5, 1, 2).flatten(end_dim=3)
        if pre_layer_norm:
            x_per_location = pre_layer_norm(x_per_location)
        # State in should be of size (batch_size, hidden_size, width, height, depth).
        state_in_per_location = tuple(
            state.permute(0, 2, 3, 4, 1).flatten(end_dim=3)[None].contiguous()
            for state in state_in
        )

        lstm_out_per_location: torch.Tensor
        state_out_per_location: torch.Tensor
        lstm_out_per_location, state_out_per_location = lstm(
            x_per_location, state_in_per_location
        )
        lstm_out_with_time = lstm_out_per_location.reshape(
            batch_size,
            *world_size,
            max_seq_len,
            self.d_model,
        ).permute(0, 4, 5, 1, 2, 3)
        state_out = [
            state.reshape(batch_size, *world_size, self.d_model).permute(0, 4, 1, 2, 3)
            for state in state_out_per_location
        ]

        lstm_out = lstm_out_with_time.flatten(end_dim=1)

        out = lstm_out + x  # Residual connection.

        return out, state_out

    def forward(
        self,
        x: torch.Tensor,
        state: List[torch.Tensor] = [],
        seq_lens=None,
        *,
        layer_indices: Optional[Sequence[int]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        # input shape: (batch_size, channels, spatial_dim_1, spatial_dim_2...)

        if layer_indices is None:
            layer_indices = range(len(self.layers))

        state_out: List[torch.Tensor] = []
        for layer_index in layer_indices:
            if self.training:
                from torch.utils.checkpoint import checkpoint

                x, layer_state = checkpoint(
                    lambda x, state, seq_lens, layer_index=layer_index: self._run_layer(
                        layer_index, x, state, seq_lens
                    ),
                    x,
                    state,
                    seq_lens,
                )
            else:
                x, layer_state = self._run_layer(layer_index, x, state, seq_lens)
            state_out.extend(layer_state)

        if self.norm_first:
            x = self.post_layer_norm(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)

        if self.interleave_lstm:
            return x, state_out
        else:
            assert state_out == []
            return x

    def get_initial_state(self, world_size: WorldSize) -> List[torch.Tensor]:
        if self.interleave_lstm:
            num_lstm_layers = len(self.layers) // (self.n_spatial_dims + 1)
            return [
                torch.zeros((self.d_model, *world_size))
                for _ in range(2 * num_lstm_layers)
            ]
        else:
            return []


class MbagTransformerModelConfig(MbagModelConfig, total=False):
    position_embedding_size: int
    num_layers: int
    dim_feedforward: int
    num_heads: int
    norm_first: bool
    use_separated_transformer: bool
    interleave_lstm: bool
    fix_interleave_lstm: bool


TRANSFORMER_DEFAULT_CONFIG: MbagTransformerModelConfig = {
    "env_config": DEFAULT_CONFIG["env_config"],
    "embedding_size": DEFAULT_CONFIG["embedding_size"],
    "use_extra_features": DEFAULT_CONFIG["use_extra_features"],
    "mask_goal": DEFAULT_CONFIG["mask_goal"],
    "hidden_size": DEFAULT_CONFIG["hidden_size"],
    "num_action_layers": DEFAULT_CONFIG["num_action_layers"],
    "num_value_layers": DEFAULT_CONFIG["num_value_layers"],
    "fake_state": DEFAULT_CONFIG["fake_state"],
    "position_embedding_size": 12,
    "num_layers": 3,
    "dim_feedforward": DEFAULT_CONFIG["hidden_size"],
    "num_heads": 2,
    "norm_first": False,
    "use_separated_transformer": False,
    "interleave_lstm": False,
    "fix_interleave_lstm": False,
}


class MbagTransformerModel(MbagTorchModel):
    """
    Model which uses a transformer encoder as the backbone.
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config,
        name,
        **kwargs,
    ):
        if "hidden_size" in kwargs:
            kwargs.setdefault("dim_feedforward", kwargs["hidden_size"])
        extra_config: MbagTransformerModelConfig = copy.deepcopy(
            TRANSFORMER_DEFAULT_CONFIG
        )
        extra_config.update(cast(MbagTransformerModelConfig, kwargs))
        self.position_embedding_size = extra_config["position_embedding_size"]
        self.num_layers = extra_config["num_layers"]
        self.dim_feedforward = extra_config["dim_feedforward"]
        self.num_heads = extra_config["num_heads"]
        self.norm_first = extra_config["norm_first"]
        self.use_separated_transformer = extra_config["use_separated_transformer"]
        self.interleave_lstm = extra_config["interleave_lstm"]
        self.fix_interleave_lstm = extra_config["fix_interleave_lstm"]

        super().__init__(
            obs_space, action_space, num_outputs, model_config, name, **kwargs
        )

        assert self._get_backbone_in_channels() <= self.hidden_size

        # Initialize positional embeddings along each dimension.
        self.position_embedding = nn.Parameter(
            torch.zeros(self.world_size + (self.position_embedding_size,))
        )
        dim_embedding_size = self.position_embedding_size // 6 * 2
        self.position_embedding.data[..., :dim_embedding_size] = (
            self._get_position_embedding(
                self.position_embedding.size()[0],
                dim_embedding_size,
            )[:, None, None]
        )
        self.position_embedding.data[
            ..., dim_embedding_size : dim_embedding_size * 2
        ] = self._get_position_embedding(
            self.position_embedding.size()[1],
            dim_embedding_size,
        )[
            None, :, None
        ]
        self.position_embedding.data[
            ..., dim_embedding_size * 2 : dim_embedding_size * 3
        ] = self._get_position_embedding(
            self.position_embedding.size()[2],
            dim_embedding_size,
        )[
            None, None, :
        ]

    def _get_embedding_channels(self) -> int:
        return super()._get_embedding_channels() + self.position_embedding_size

    def _get_position_embedding(self, seq_len: int, size: int) -> torch.Tensor:
        """
        Get an initial positional embedding of shape (seq_len, size) by using
        the sin/cos embedding.
        """

        embedding = torch.zeros(seq_len, size)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # TODO: improve position embedding?
        div_term = torch.exp(
            torch.arange(0, size, 2).float() * (-np.log(10000.0) / size)
        )
        embedding[:, 0::2] = torch.sin(position * div_term)
        embedding[:, 1::2] = torch.cos(position * div_term)
        return embedding

    def _pad_to_hidden_size(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pads the last dimension of the given tensor so that it is self.hidden_size.
        """

        return F.pad(x, [0, self.hidden_size - x.size()[-1]])

    def _construct_backbone(self, is_value_network=False) -> nn.Module:
        if self.use_separated_transformer:
            return SeparatedTransformerEncoder(
                num_layers=self.num_layers,
                d_model=self.hidden_size,
                nhead=self.num_heads,
                dim_feedforward=self.dim_feedforward,
                norm_first=self.norm_first,
                interleave_lstm=self.interleave_lstm,
                fix_interleave_lstm=self.fix_interleave_lstm,
            )
        else:
            assert self.interleave_lstm is False
            return nn.Sequential(
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=self.hidden_size,
                        nhead=self.num_heads,
                        dim_feedforward=self.dim_feedforward,
                        batch_first=True,
                        norm_first=self.norm_first,
                    ),
                    self.num_layers,
                ),
                View(-1, *self.world_size, self.hidden_size),
                Permute(0, 4, 1, 2, 3),
            )

    def _get_embedded_obs(
        self,
        world_obs: torch.Tensor,
        inventory_obs: torch.Tensor,
        timestep: torch.Tensor,
        prev_blocks: Optional[torch.Tensor] = None,
        prev_actions: Optional[torch.Tensor] = None,
        prev_other_agent_actions: Optional[torch.Tensor] = None,
        include_last_interacted=True,
    ):
        embedded_obs = super()._get_embedded_obs(
            world_obs,
            inventory_obs,
            timestep,
            prev_blocks,
            prev_actions,
            prev_other_agent_actions,
            include_last_interacted,
        )
        batch_size = embedded_obs.size()[0]
        embedded_obs = torch.cat(
            [
                embedded_obs.permute(0, 2, 3, 4, 1),
                self.position_embedding[None].expand(batch_size, -1, -1, -1, -1),
            ],
            dim=4,
        )
        if not self.use_fc_after_embedding:
            embedded_obs = self._pad_to_hidden_size(embedded_obs)

        embedded_obs = embedded_obs.permute(0, 4, 1, 2, 3)

        if self.use_separated_transformer:
            return embedded_obs
        else:
            return embedded_obs.flatten(start_dim=2).transpose(1, 2)

    def get_initial_state(self):
        if self.interleave_lstm:
            assert isinstance(self.backbone, SeparatedTransformerEncoder)
            state = self.backbone.get_initial_state(world_size=self.world_size)
            if self.use_prev_blocks:
                state.append(self._get_prev_blocks_initial_state())
            return state
        else:
            return super().get_initial_state()


ModelCatalog.register_custom_model("mbag_transformer_model", MbagTransformerModel)


class OtherAgentActionPredictorMixin(MbagTorchModel):
    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config,
        name,
        **kwargs,
    ):
        super().__init__(
            obs_space, action_space, num_outputs, model_config, name, **kwargs
        )

        self.other_agent_action_prediction_head = self._construct_action_head()

    def predict_other_agent_action(self) -> torch.Tensor:
        with self._amp_or_nothing:
            logits: torch.Tensor = self.other_agent_action_prediction_head(
                self._backbone_out
            )
        flat_logits = MbagActionDistribution.to_flat_torch_logits(
            self.env_config, logits
        )
        return flat_logits.float()


class MbagTransformerAlphaZeroModel(
    MbagTransformerModel, OtherAgentActionPredictorMixin
):
    pass


ModelCatalog.register_custom_model(
    "mbag_transformer_alpha_zero_model", MbagTransformerAlphaZeroModel
)


class ModelWithDiscriminator(TorchModelV2, nn.Module, ABC):
    @abstractmethod
    def discriminator(  # noqa: E704
        self, input_dict: Dict[str, TensorType]
    ) -> torch.Tensor: ...


class DiscriminatorMixin(MbagTorchModel, ModelWithDiscriminator):
    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config,
        name,
        **kwargs,
    ):
        super().__init__(
            obs_space, action_space, num_outputs, model_config, name, **kwargs
        )

        self.discriminator_backbone = self._construct_backbone()
        self.discriminator_head = self._construct_discriminator_head()

    def _get_discriminator_module_names(self) -> List[str]:
        return ["discriminator_backbone", "discriminator_head"]

    def _get_embedded_obs_and_actions(
        self,
        world_obs: torch.Tensor,
        inventory_obs: torch.Tensor,
        timestep: torch.Tensor,
        actions: torch.Tensor,
        prev_blocks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError()

    def _construct_discriminator_head(self) -> nn.Module:
        return nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(self.hidden_size, 1),
        )

    def discriminator(self, input_dict: Dict[str, TensorType]) -> torch.Tensor:
        if self.model_config.get("_disable_preprocessor_api"):
            obs = cast(
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                input_dict[SampleBatch.OBS],
            )
        else:
            obs = cast(
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                restore_original_dimensions(
                    input_dict[SampleBatch.OBS], self.obs_space, self.framework
                ),
            )

        world_obs, inventory_obs, timestep = obs
        world_obs = world_obs.to(self.device).long()
        inventory_obs = inventory_obs.to(self.device).long()
        timestep = timestep.to(self.device)
        actions = (
            cast(torch.Tensor, input_dict[SampleBatch.ACTIONS]).to(self.device).long()
        )

        embedded_obs = self._get_embedded_obs_and_actions(
            world_obs, inventory_obs, timestep, actions
        )

        with self._amp_or_nothing:
            backbone_out: torch.Tensor = self.discriminator_backbone(embedded_obs)
            logits: torch.Tensor = self.discriminator_head(backbone_out).squeeze(1)

        return logits.float()

    def load_state_dict(self, state_dict: Mapping[str, Any], strict=True, assign=False):
        # Allow loading existing model weights that don't include the discriminator
        # weights.
        new_state_dict = dict(state_dict)
        current_state_dict = self.state_dict()
        for key in current_state_dict:
            if any(
                key.startswith(module_name + ".")
                for module_name in self._get_discriminator_module_names()
            ):
                new_state_dict.setdefault(key, current_state_dict[key])

        super().load_state_dict(new_state_dict)


class MbagTransformerModelWithDiscriminator(MbagTransformerModel, DiscriminatorMixin):
    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config,
        name,
        **kwargs,
    ):
        super().__init__(
            obs_space, action_space, num_outputs, model_config, name, **kwargs
        )

        self.discriminator_first_layer = nn.Conv3d(
            in_channels=self._get_embedding_channels()
            - self.embedding_size
            + MbagActionDistribution.NUM_CHANNELS,
            out_channels=self.hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def _get_discriminator_module_names(self):
        return super()._get_discriminator_module_names() + ["discriminator_first_layer"]

    def _get_embedded_obs_and_actions(
        self,
        world_obs: torch.Tensor,
        inventory_obs: torch.Tensor,
        timestep: torch.Tensor,
        actions: torch.Tensor,
        prev_blocks: Optional[torch.Tensor] = None,
    ):
        # Ignore LAST_INTERACTED channel in the discriminator.
        embedded_obs = MbagTorchModel._get_embedded_obs(
            self,
            world_obs,
            inventory_obs,
            timestep,
            prev_blocks,
            include_last_interacted=False,
        )

        batch_size, _, width, height, depth = world_obs.size()
        embedded_actions = torch.zeros(
            (batch_size, MbagActionDistribution.NUM_CHANNELS, width, height, depth),
            dtype=embedded_obs.dtype,
            device=embedded_obs.device,
        )

        flat_action_start = 0
        for channel, (action_type, block_id) in enumerate(
            MbagActionDistribution.CHANNELS
        ):
            channel_embedded_actions = embedded_actions[:, channel].view(batch_size, -1)
            if action_type in MbagAction.BLOCK_LOCATION_ACTION_TYPES:
                flat_action_end = flat_action_start + width * height * depth

                actions_in_channel_mask = (actions >= flat_action_start) & (
                    actions < flat_action_end
                )
                channel_embedded_actions[
                    torch.arange(batch_size, device=embedded_obs.device)[
                        actions_in_channel_mask
                    ],
                    actions[actions_in_channel_mask] - flat_action_start,
                ] = 1
            else:
                actions_in_channel_mask = actions == flat_action_start
                channel_embedded_actions[actions_in_channel_mask, :] = 1
                flat_action_end = flat_action_start + 1

            flat_action_start = flat_action_end

        assert torch.any(embedded_actions.flatten(start_dim=1), dim=1).all()

        batch_size = embedded_obs.size()[0]
        embedded_obs_and_actions = torch.cat(
            [
                embedded_obs,
                embedded_actions,
                self.position_embedding[None]
                .expand(batch_size, -1, -1, -1, -1)
                .permute(0, 4, 1, 2, 3),
            ],
            dim=1,
        )
        embedded_obs_and_actions = self.discriminator_first_layer(
            embedded_obs_and_actions
        )

        if self.use_separated_transformer:
            return embedded_obs_and_actions
        else:
            return embedded_obs_and_actions.flatten(start_dim=2).transpose(1, 2)


ModelCatalog.register_custom_model(
    "mbag_transformer_with_discriminator_model", MbagTransformerModelWithDiscriminator
)
