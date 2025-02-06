import copy
import logging
import time
from typing import List, Optional, cast

import numpy as np
from ray.rllib.evaluation import SampleBatch
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import TensorType
from typing_extensions import TypedDict

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.agents.mbag_agent import MbagAgent
from mbag.environment.actions import MbagAction, MbagActionTuple
from mbag.environment.mbag_env import MbagConfigDict
from mbag.environment.state import MbagStateDict, mbag_obs_to_state
from mbag.environment.types import (
    CURRENT_BLOCKS,
    CURRENT_PLAYER,
    LAST_INTERACTED,
    MbagInfoDict,
    MbagObs,
)
from mbag.rllib.alpha_zero.alpha_zero_policy import C_PUCT, MbagAlphaZeroPolicy

logger = logging.getLogger(__name__)


class RllibMbagAgentConfigDict(TypedDict):
    policy: Policy

    explore: bool

    min_action_interval: float
    """
    The minimum amount of time between actions, in seconds. If the agent is asked to
    take an action less than this amount of time after the last action, it will just
    return a NOOP.
    """

    ignore_own_actions: bool
    """
    If this is True, the agent will receive observations that do not have any direct
    effects of its own place/break actions.
    """

    confidence_threshold: Optional[float]
    """
    If this is not None, the agent will only take actions if the probability of the
    chosen action is greater than this threshold. If all action probabilities are less than the
    threshold, the agent will take a NOOP action.
    """

    temperature: float
    """
    The temperature to use when sampling from the policy. If this is 0, the policy will
    always choose the action with the highest probability.
    """


class RllibMbagAgent(MbagAgent):
    agent_config: RllibMbagAgentConfigDict
    state: List[TensorType]
    last_action_time: Optional[float]
    prev_action: MbagActionTuple

    # Used when ignore_own_actions is True.
    current_blocks: Optional[np.ndarray]
    last_interacted: Optional[np.ndarray]

    def __init__(self, agent_config: MbagConfigDict, env_config: MbagConfigDict):
        super().__init__(agent_config, env_config)

        self.policy = self.agent_config["policy"]
        self.explore = self.agent_config.get("explore", False)
        self.min_action_interval = self.agent_config["min_action_interval"]
        self.confidence_threshold = self.agent_config.get("confidence_threshold", None)
        self.temperature = self.agent_config.get("temperature", 1.0)
        self.ignore_own_actions = self.agent_config.get("ignore_own_actions", False)
        self.action_mapping = MbagActionDistribution.get_action_mapping(self.env_config)

    def reset(self, **kwargs) -> None:
        super().reset(**kwargs)

        self.state = self.policy.get_initial_state()
        self.c_puct: Optional[float] = None  # Used for DiL-piKL.
        self.last_action_time = None
        self.prev_action = (0, 0, 0)

        if self.ignore_own_actions:
            self.current_blocks = None
            self.last_interacted = None

    def get_action(self, obs: MbagObs, *, compute_actions_kwargs={}) -> MbagActionTuple:
        obs = copy.deepcopy(obs)

        force_noop = False
        if self.last_action_time is not None:
            time_since_last_action = time.time() - self.last_action_time
            if time_since_last_action < self.min_action_interval:
                force_noop = True

        if not force_noop:
            self.last_action_time = time.time()

        if isinstance(self.policy, MbagAlphaZeroPolicy) and self.c_puct is not None:
            compute_actions_kwargs = {
                **compute_actions_kwargs,
                "prev_c_puct": np.array([self.c_puct]),
            }

        if self.ignore_own_actions:
            compute_actions_kwargs["env_states"] = [
                mbag_obs_to_state(obs, self.policy.config["player_index"])
            ]

            world_obs = obs[0]
            if self.current_blocks is None:
                self.current_blocks = world_obs[CURRENT_BLOCKS]
            if self.last_interacted is None:
                self.last_interacted = world_obs[LAST_INTERACTED]

            last_interacted = world_obs[LAST_INTERACTED]
            self.current_blocks[last_interacted > CURRENT_PLAYER] = world_obs[
                CURRENT_BLOCKS
            ][last_interacted > CURRENT_PLAYER]
            self.last_interacted[last_interacted > CURRENT_PLAYER] = world_obs[
                LAST_INTERACTED
            ][last_interacted > CURRENT_PLAYER]
            world_obs[CURRENT_BLOCKS] = self.current_blocks
            world_obs[LAST_INTERACTED] = self.last_interacted
            assert not np.any(obs[0][LAST_INTERACTED] == CURRENT_PLAYER)

        obs_batch = tuple(obs_piece[None] for obs_piece in obs)
        # preprocessor = ModelCatalog.get_preprocessor_for_space(self.policy.observation_space)
        # obs_batch = torch.from_numpy(preprocessor.transform(obs)[None]).to(self.policy.device)
        state_batch = [state_piece[None] for state_piece in self.state]
        state_out_batch: List[TensorType]
        action_batch: TensorType
        action_batch, state_out_batch, compute_actions_info = (
            self.policy.compute_actions(
                obs_batch,
                state_batch,
                explore=self.explore,
                force_noop=force_noop,
                prev_action_batch=np.array([list(self.prev_action)]),
                **compute_actions_kwargs,
            )
        )
        self.state = [state_piece[0] for state_piece in state_out_batch]

        if self.temperature != 1.0 and not force_noop:
            assert self.confidence_threshold is None
            logits = compute_actions_info[SampleBatch.ACTION_DIST_INPUTS][0]
            if self.temperature == 0:
                action_batch = np.argmax(logits)[None]
            else:
                probs = np.exp(logits / self.temperature)
                probs /= np.sum(probs)
                action_batch = np.random.choice(np.arange(len(probs)), p=probs)[None]

        if self.confidence_threshold is not None and not force_noop:
            assert self.temperature == 1.0
            logits = compute_actions_info[SampleBatch.ACTION_DIST_INPUTS][0]
            probs = np.exp(logits)
            probs /= np.sum(probs)
            (action_id,) = action_batch
            # normalized_probs = probs / np.mean(probs[probs != 0])
            if probs[action_id] < self.confidence_threshold:
                action_batch = np.array([0])
            # normalized_probs[normalized_probs < self.normalized_confidence_threshold] = 0
            # if np.sum(normalized_probs) == 0:
            #     normalized_probs[0] = 1
            # normalized_probs /= np.sum(normalized_probs)
            # action_batch = np.random.choice(
            #     np.arange(len(normalized_probs)), p=normalized_probs
            # )[None]

        if C_PUCT in compute_actions_info:
            self.c_puct = float(compute_actions_info[C_PUCT][0])

        self.last_info = compute_actions_info

        if isinstance(action_batch, tuple):
            action = cast(
                MbagActionTuple,
                tuple(int(action_piece[0]) for action_piece in action_batch),
            )
        else:
            # Flat actions.
            action = cast(MbagActionTuple, tuple(self.action_mapping[action_batch[0]]))

        action_type, _, _ = action
        if force_noop and action_type != MbagAction.NOOP:
            logger.warning(f"policy was passed force_noop but returned action {action}")
            action = (MbagAction.NOOP, 0, 0)

        self.prev_action = action

        return action

    def get_state(self) -> List[np.ndarray]:
        state = [np.array(state_part) for state_part in self.state] + [
            np.array(self.prev_action)
        ]
        if self.agent_config["ignore_own_actions"]:
            assert self.current_blocks is not None and self.last_interacted is not None
            state += [self.current_blocks, self.last_interacted]
        return state

    def set_state(self, state: List[np.ndarray]) -> None:
        if self.agent_config["ignore_own_actions"]:
            self.current_blocks, self.last_interacted = state[-2:]
            state = state[:-2]
        self.state = [state_part for state_part in state[:-1]]
        self.prev_action = tuple(state[-1])


class RllibAlphaZeroAgentConfigDict(RllibMbagAgentConfigDict):
    player_index: str


class FakeEpisode(object):
    def __init__(self, *, user_data):
        self.user_data = user_data
        self.length = 0


class RllibAlphaZeroAgent(RllibMbagAgent):
    agent_config: RllibAlphaZeroAgentConfigDict

    def __init__(self, agent_config: MbagConfigDict, env_config: MbagConfigDict):
        super().__init__(agent_config, env_config)

        self.policy.config["player_index"] = self.agent_config["player_index"]

    def get_action_with_info_and_env_state(
        self, obs: MbagObs, info: Optional[MbagInfoDict], env_state: MbagStateDict
    ) -> MbagActionTuple:
        return super().get_action(obs)
