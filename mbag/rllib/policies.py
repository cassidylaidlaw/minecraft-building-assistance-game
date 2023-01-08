from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union, cast

import gym
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from gym import spaces
from ray.rllib.agents.ppo import PPOTorchPolicy, PPOTrainer
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.typing import TensorStructType, TensorType, TrainerConfigDict
from ray.tune.registry import register_trainable
from torch import nn

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.agents.mbag_agent import MbagAgent
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.types import GOAL_BLOCKS, MbagAction, MbagActionTuple, MbagObs

from .torch_action_distributions import MbagAutoregressiveActionDistribution
from .torch_models import MbagTorchModel


class MbagAgentPolicy(Policy):
    """
    An RLlib policy that selects actions based on an MBAG agent instance.
    """

    agent: MbagAgent

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        config: TrainerConfigDict,
    ):
        super().__init__(observation_space, action_space, config)
        self.agent = config["mbag_agent"]
        self.exploration = self._create_exploration()
        self.flat_actions = isinstance(self.action_space, spaces.Discrete)

    def get_initial_state(self) -> List[TensorType]:
        self.agent.reset()
        return self.agent.get_state()

    def compute_actions(
        self,
        obs_batch: Union[List[TensorStructType], TensorStructType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
        info_batch: Optional[Dict[str, list]] = None,
        episodes=None,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        **kwargs,
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        for batch_key, view_requirement in self.view_requirements.items():
            if batch_key.startswith("state_in_"):
                view_requirement.batch_repeat_value = 1

        unflattened_obs_batch = restore_original_dimensions(
            obs_batch,
            obs_space=self.observation_space,
            tensorlib=np,
        )

        actions: List[MbagActionTuple] = []
        new_states: List[List[np.ndarray]] = []

        assert state_batches is not None

        obs: MbagObs
        prev_state: Sequence[TensorType]
        for obs, prev_state in zip(
            zip(*unflattened_obs_batch),
            zip(*state_batches)
            if state_batches != []
            else [[] for _ in range(len(obs_batch))],
        ):
            self.agent.set_state(list(prev_state))
            action = self.agent.get_action(obs)
            actions.append(action)
            new_states.append(self.agent.get_state())

        action_array: Any
        if self.flat_actions:
            action_array = np.array(
                [
                    MbagActionDistribution.get_flat_action(
                        self.config["env_config"], action
                    )
                    for action in actions
                ]
            )
        else:
            action_array = tuple(
                np.array([action[action_part] for action in actions])
                for action_part in range(3)
            )
        state_arrays = [
            np.array([new_state[state_part] for new_state in new_states])
            for state_part in range(len(state_batches))
        ]
        return action_array, state_arrays, {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass


class MbagPPOTorchPolicy(PPOTorchPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        config: TrainerConfigDict,
        **kwargs,
    ):
        self.action_mapping = torch.from_numpy(
            MbagActionDistribution.get_action_mapping(
                config["model"]["custom_model_config"]["env_config"]
            )
        )
        super().__init__(observation_space, action_space, config, **kwargs)
        self.action_mapping = self.action_mapping.to(self.device)

    def loss(
        self,
        model,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> TensorType:
        loss = super().loss(model, dist_class, train_batch)
        assert not isinstance(loss, list)

        assert isinstance(model, MbagTorchModel)

        world_obs, _, _ = restore_original_dimensions(
            train_batch[SampleBatch.OBS],
            obs_space=self.observation_space,
            tensorlib=torch,
        )

        goal = world_obs[:, GOAL_BLOCKS].long()

        loss += self.config["place_block_loss_coeff"] * self.place_block_loss(
            model, dist_class, goal, train_batch
        )
        loss += self.config["goal_loss_coeff"] * self.predict_goal_loss(
            model, goal, train_batch
        )

        return loss

    def predict_goal_loss(
        self,
        model: MbagTorchModel,
        goal: TensorType,
        train_batch: SampleBatch,
    ) -> TensorType:
        if not hasattr(model, "_backbone_out"):
            model(train_batch)
        log_odds = model.goal_function()

        ce = nn.CrossEntropyLoss()
        loss = ce(log_odds, goal)

        model.tower_stats["predict_goal_loss"] = loss

        return loss

    def place_block_loss(
        self,
        model: MbagTorchModel,
        dist_class: Type[ActionDistribution],
        goal: TensorType,
        train_batch,
    ) -> TensorType:
        """
        Add loss to minimize the cross-entropy between the block ID for a "place block" action
        and the goal block at that location, if there is any goal block there.
        """

        world_obs, _, _ = restore_original_dimensions(
            train_batch[SampleBatch.OBS],
            obs_space=self.observation_space,
            tensorlib=torch,
        )
        goal_block_ids = world_obs[:, 2].long()

        if hasattr(model, "logits"):
            # Don't recompute logits if we don't have to.
            logits = model.logits
        else:
            logits, state = model(train_batch)

        # We only care about place block actions at places where there are blocks in the
        # goal.
        place_block_mask = ~torch.any(
            goal_block_ids[..., None]
            == MbagAutoregressiveActionDistribution.PLACEABLE_BLOCK_MASK[
                None, None, None, None, :
            ],
            dim=4,
        ).flatten()

        place_block_logits = logits[
            :, self.action_mapping.to(self.device)[:, 0] == MbagAction.PLACE_BLOCK
        ].reshape((-1, MinecraftBlocks.NUM_BLOCKS) + world_obs.size()[-3:])
        place_block_logits = place_block_logits.permute((0, 2, 3, 4, 1)).flatten(
            end_dim=3
        )
        place_block_mask &= (
            place_block_logits[
                torch.arange(place_block_logits.size()[0]), goal_block_ids.flatten()
            ]
            > MbagTorchModel.MASK_LOGIT
        )

        place_block_loss = F.cross_entropy(
            place_block_logits[place_block_mask],
            goal_block_ids.flatten()[place_block_mask],
        )

        model.tower_stats["place_block_loss"] = place_block_loss

        return place_block_loss

    def log_mean_loss(self, info: Dict[str, TensorType], loss_name: str):
        try:
            info[loss_name] = torch.mean(torch.stack(self.get_tower_stats(loss_name)))
        except AssertionError:
            info[loss_name] = torch.nan

    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        info = super().stats_fn(train_batch)

        self.log_mean_loss(info, "place_block_loss")
        self.log_mean_loss(info, "predict_goal_loss")

        return cast(
            Dict[str, TensorType],
            convert_to_numpy(info),
        )


class MbagPPOTrainer(PPOTrainer):
    @classmethod
    def get_default_config(cls) -> TrainerConfigDict:
        return {
            **super().get_default_config(),
            "goal_loss_coeff": 1.0,
            "place_block_loss_coeff": 1.0,
        }

    def get_default_policy_class(self, config):
        return MbagPPOTorchPolicy


register_trainable("MbagPPO", MbagPPOTrainer)
