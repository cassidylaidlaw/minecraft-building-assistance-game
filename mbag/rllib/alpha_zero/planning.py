from typing import Dict, List, Optional, Sequence, Tuple, Union, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from ray.rllib.utils.typing import AgentID
from ray.tune.registry import register_env

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.environment.actions import MbagAction
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.mbag_env import MbagConfigDict
from mbag.environment.state import MbagStateDict, mbag_obs_to_state
from mbag.environment.types import (
    CURRENT_BLOCK_STATES,
    CURRENT_BLOCKS,
    GOAL_BLOCK_STATES,
    GOAL_BLOCKS,
    MbagInfoDict,
    MbagObs,
)

from ..rllib_env import (
    FlatActionSpaceWrapper,
    MbagMultiAgentEnv,
    MbagRllibWrapper,
    unwrap_mbag_env,
)


class MbagEnvModelStateDict(MbagStateDict, total=False):
    last_obs_dict: Dict[AgentID, MbagObs]


class MbagEnvModelInfoDict(MbagInfoDict):
    other_player_infos: List[MbagInfoDict]


class MbagEnvModel(gym.Env):
    """
    A single-agent environment model that can be used for planning in algorithms such
    as AlphaZero.
    """

    action_space: spaces.Discrete
    last_obs_dict: Dict[AgentID, MbagObs]

    def __init__(
        self,
        env: MbagRllibWrapper,
        config: MbagConfigDict,
        player_index: int = 0,
        line_of_sight_masking=False,
        expected_own_reward_scale=1.0,
        expected_reward_shift=0.0,
    ):
        super().__init__()

        self.env = env
        self.config = config
        self.set_player_index(player_index)
        self.line_of_sight_masking = line_of_sight_masking
        self.expected_own_reward_scale = expected_own_reward_scale
        self.expected_reward_shift = expected_reward_shift

        assert isinstance(self.env.action_space, spaces.Dict)
        action_space = self.env.action_space.spaces[self.agent_id]
        assert isinstance(action_space, spaces.Discrete)
        self.action_space = action_space
        self.observation_space = self.env.observation_space

    def set_player_index(self, player_index: int):
        self.player_index = player_index
        self.agent_id = f"player_{player_index}"

    def _store_last_obs_dict(self, obs_dict):
        self.last_obs_dict = obs_dict

    def reset(self, *, seed=None, options=None):
        obs_dict, info_dict = self.env.reset()
        self._store_last_obs_dict(obs_dict)
        return cast(MbagObs, obs_dict[self.agent_id]), cast(
            MbagInfoDict, info_dict[self.agent_id]
        )

    def step(
        self,
        action: int,
        goal_logits: Optional[np.ndarray] = None,
        other_player_actions: Optional[Sequence[int]] = None,
    ):
        action_dict = {self.agent_id: action}

        other_agent_ids = list(self.last_obs_dict)
        other_agent_ids.remove(self.agent_id)
        if other_player_actions is None:
            for other_agent_id in other_agent_ids:
                action_dict[other_agent_id] = 0  # NOOP
        else:
            for other_agent_id, other_player_action in zip(
                other_agent_ids, other_player_actions
            ):
                action_dict[other_agent_id] = other_player_action

        (
            obs_dict,
            reward_dict,
            terminated_dict,
            truncated_dict,
            info_dict,
        ) = self.env.step(action_dict)
        info: MbagEnvModelInfoDict = info_dict[self.agent_id]
        info["other_player_infos"] = [
            info_dict[other_agent_id] for other_agent_id in other_agent_ids
        ]

        if goal_logits is not None:
            info["goal_dependent_reward"] = self._get_predicted_goal_dependent_reward(
                self.last_obs_dict[self.agent_id], info["action"], goal_logits
            )
            info["own_reward"] = (
                info["goal_dependent_reward"] + info["goal_independent_reward"]
            )
            if (
                info["action"].action_type == MbagAction.PLACE_BLOCK
                or info["action"].action_type == MbagAction.BREAK_BLOCK
            ):
                info["own_reward"] = (
                    self.expected_reward_shift
                    + self.expected_own_reward_scale * info["own_reward"]
                )
            reward = info["own_reward"]
            reward = (
                info["own_reward_prop"] * info["own_reward"]
                + (1 - info["own_reward_prop"]) * reward
            )
        else:
            reward = reward_dict[self.agent_id]

        self._store_last_obs_dict(obs_dict)

        obs: MbagObs = obs_dict[self.agent_id]

        return (
            obs,
            reward,
            terminated_dict.get(self.agent_id, terminated_dict["__all__"]),
            truncated_dict.get(self.agent_id, truncated_dict["__all__"]),
            info,
        )

    def get_valid_actions(self, obs: MbagObs, is_batch=False) -> np.ndarray:
        if not is_batch:
            world_obs, inventory_obs, timestep = obs
            batched_obs = (
                world_obs[None],
                inventory_obs[None],
                timestep[None],
            )
        else:
            batched_obs = obs
        action_mask: np.ndarray = MbagActionDistribution.get_mask_flat(
            self.config,
            batched_obs,
            line_of_sight_masking=self.line_of_sight_masking,
        )
        if not is_batch:
            action_mask = action_mask[0]
        return action_mask

    def get_state(self) -> MbagEnvModelStateDict:
        env_state = self.env.get_state()
        return {
            "current_blocks": env_state["current_blocks"],
            "goal_blocks": env_state["goal_blocks"],
            "player_locations": env_state["player_locations"],
            "player_directions": env_state["player_directions"],
            "player_inventories": env_state["player_inventories"],
            "last_interacted": env_state["last_interacted"],
            "timestep": env_state["timestep"],
            "last_obs_dict": self.last_obs_dict,
        }

    def set_state(self, state: Union[MbagEnvModelStateDict, MbagStateDict]):
        if "last_obs_dict" in state:
            state = cast(MbagEnvModelStateDict, state)
            self.env.set_state_no_obs(state)
            self.last_obs_dict = state["last_obs_dict"]
            return self.last_obs_dict[self.agent_id]
        else:
            obs_dict = self.env.set_state(state)
            self._store_last_obs_dict(obs_dict)
            return obs_dict[self.agent_id]

    def set_state_from_obs(self, obs: MbagObs) -> Tuple[MbagStateDict, MbagObs]:
        state = mbag_obs_to_state(
            obs, self.player_index, num_players=self.config["num_players"]
        )
        obs = self.set_state(state)
        return state, obs

    def get_reward_with_other_agent_actions(
        self,
        obs: MbagObs,
        info: MbagEnvModelInfoDict,
        goal_logits: np.ndarray,
        update_own_reward: bool = False,
        average_own_reward: bool = False,
    ) -> float:
        """
        Given an observation and the info dict returned from step, returns an overall
        reward based on this and other players' actions.
        """

        other_reward = 0.0
        other_player_indices = list(range(self.config["num_players"]))
        other_player_indices.remove(self.player_index)
        for other_player_index, other_player_info in zip(
            other_player_indices, info["other_player_infos"]
        ):
            goal_dependent_reward = self._get_predicted_goal_dependent_reward(
                # I think it's okay to use this player's obs even though it should
                # technically be the other player's
                obs,
                other_player_info["action"],
                goal_logits,
                player_index=other_player_index,
            )
            other_reward += (
                goal_dependent_reward + other_player_info["goal_independent_reward"]
            )

        if update_own_reward or average_own_reward:
            prev_own_reward = info["own_reward"]
            new_goal_dependent_reward = self._get_predicted_goal_dependent_reward(
                obs,
                info["action"],
                goal_logits,
            )
            new_own_reward = info["goal_independent_reward"] + new_goal_dependent_reward
            if (
                info["action"].action_type == MbagAction.PLACE_BLOCK
                or info["action"].action_type == MbagAction.BREAK_BLOCK
            ):
                new_own_reward = (
                    self.expected_reward_shift
                    + self.expected_own_reward_scale * new_own_reward
                )
            assert not (update_own_reward and average_own_reward)
            if update_own_reward:
                info["own_reward"] = new_own_reward
            elif average_own_reward:
                info["own_reward"] = (prev_own_reward + new_own_reward) / 2

        reward = info["own_reward"] + other_reward
        reward = (
            info["own_reward_prop"] * info["own_reward"]
            + (1 - info["own_reward_prop"]) * reward
        )

        return reward

    def _get_predicted_goal_dependent_reward(
        self,
        obs: MbagObs,
        action: MbagAction,
        goal_logits: np.ndarray,
        player_index: Optional[int] = None,
    ) -> float:
        """
        Given the predicted distribution over goal blocks for each location as a logit
        array of shape (NUM_BLOCKS, width, height, depth), this gives
        the expected goal-dependent reward for an action.

        Similarly to get_all_rewards, the rewards returned here are only valid if the
        action is not effectively a no-op.
        """

        if player_index is None:
            player_index = self.player_index

        world_obs, _, _ = obs
        _, width, height, depth = world_obs.shape
        env = unwrap_mbag_env(self)

        reward = 0.0

        if (
            not self.config["abilities"]["inf_blocks"]
            and action.action_type == MbagAction.BREAK_BLOCK
            and action.block_location[0] == unwrap_mbag_env(self).palette_x
        ):
            # Breaking a palette block gives no reward.
            pass
        elif (
            action.action_type == MbagAction.PLACE_BLOCK
            or action.action_type == MbagAction.BREAK_BLOCK
        ):
            prev_block = (
                world_obs[CURRENT_BLOCKS][action.block_location],
                world_obs[CURRENT_BLOCK_STATES][action.block_location],
            )
            new_block_id = (
                action.block_id
                if action.action_type == MbagAction.PLACE_BLOCK
                else MinecraftBlocks.AIR
            )
            new_block = np.array(new_block_id), np.array(0)
            goal_block_ids = np.arange(MinecraftBlocks.NUM_BLOCKS, dtype=np.uint8)
            goal_blocks = goal_block_ids, np.zeros_like(goal_block_ids)

            goal_block_id_dist = np.exp(
                goal_logits[(slice(None),) + action.block_location]
            )
            goal_block_id_dist /= goal_block_id_dist.sum()

            prev_similarity = env._get_goal_similarity(
                prev_block, goal_blocks, partial_credit=True, player_index=player_index
            )
            new_similarity = env._get_goal_similarity(
                new_block, goal_blocks, partial_credit=True, player_index=player_index
            )
            reward = float(
                np.sum((new_similarity - prev_similarity) * goal_block_id_dist)
            )

            correct = new_similarity > prev_similarity
            reward += env._get_reward(
                player_index,
                "incorrect_action",
                env.global_timestep,
            ) * float(np.sum(~correct * goal_block_id_dist))

        return reward

    def get_all_rewards(
        self, obs_batch: MbagObs, player_index: Optional[int] = None
    ) -> np.ndarray:
        """
        Given a batch of observations, get the rewards for all possible actions
        as an array of shape (batch_size, NUM_CHANNELS, width, height, depth),
        where NUM_CHANNELS is as defined as in MbagActionDistribution.

        The rewards returned here are only valid if the action is not a no-op.
        If the action is a noop (e.g., if the action is invalid) then the actual
        reward is 0.
        """

        if player_index is None:
            player_index = self.player_index

        world_obs, _, _ = obs_batch
        batch_size, _, width, height, depth = world_obs.shape
        env = unwrap_mbag_env(self)

        rewards = np.zeros(
            (batch_size, MbagActionDistribution.NUM_CHANNELS, width, height, depth)
        )

        # We only get reward for two actions: BREAK_BLOCK and PLACE_BLOCK.
        goal = (world_obs[:, GOAL_BLOCKS], world_obs[:, GOAL_BLOCK_STATES])
        similarity_before = env._get_goal_similarity(
            (world_obs[:, CURRENT_BLOCKS], world_obs[:, CURRENT_BLOCK_STATES]),
            goal,
            partial_credit=True,
            player_index=player_index,
        )

        block_states = np.zeros_like(world_obs[:, CURRENT_BLOCK_STATES])
        for block_id in range(MinecraftBlocks.NUM_BLOCKS):
            block_ids = np.full_like(world_obs[:, CURRENT_BLOCKS], block_id)
            similarity_after = env._get_goal_similarity(
                (block_ids, block_states),
                goal,
                partial_credit=True,
                player_index=player_index,
            )
            block_id_reward = similarity_after - similarity_before
            if block_id == MinecraftBlocks.AIR:
                rewards[:, MbagActionDistribution.BREAK_BLOCK] = block_id_reward
            rewards[:, MbagActionDistribution.PLACE_BLOCK][
                :, block_id
            ] = block_id_reward

        # TODO: implement lack of reward for breaking palette blocks

        return rewards


def create_mbag_env_model(
    config: MbagConfigDict,
    player_index: int = 0,
) -> MbagEnvModel:
    # We should never use Malmo in the env model.
    config["malmo"]["use_malmo"] = False
    env = MbagMultiAgentEnv(config)
    flat_env = FlatActionSpaceWrapper(env, config)
    env_model = MbagEnvModel(
        flat_env,
        config,
        player_index=player_index,
    )
    return env_model


register_env("MBAGAlphaZeroModel-v1", create_mbag_env_model)
