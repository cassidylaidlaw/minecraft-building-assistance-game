import time
from typing import Iterable, List, Optional, cast

import numpy as np
from ray.rllib.evaluation import SampleBatch
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import TensorType
from typing_extensions import TypedDict

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.agents.mbag_agent import MbagAgent
from mbag.environment.actions import MbagAction, MbagActionTuple
from mbag.environment.mbag_env import MbagConfigDict
from mbag.environment.state import MbagStateDict
from mbag.environment.types import MbagInfoDict, MbagObs
from mbag.rllib.alpha_zero.alpha_zero_policy import C_PUCT, MbagAlphaZeroPolicy


class RllibMbagAgentConfigDict(TypedDict):
    policy: Policy

    explore: bool

    min_action_interval: float
    """
    The minimum amount of time between actions, in seconds. If the agent is asked to
    take an action less than this amount of time after the last action, it will just
    return a NOOP.
    """


class RllibMbagAgent(MbagAgent):
    agent_config: RllibMbagAgentConfigDict
    state: List[TensorType]
    last_action_time: Optional[float]
    prev_action: MbagActionTuple

    def __init__(self, agent_config: MbagConfigDict, env_config: MbagConfigDict):
        super().__init__(agent_config, env_config)

        self.policy = self.agent_config["policy"]
        self.explore = self.agent_config.get("explore", False)
        self.min_action_interval = self.agent_config["min_action_interval"]
        self.confidence_threshold = cast(
            Optional[float], self.agent_config.get("confidence_threshold", None)
        )
        self.action_mapping = MbagActionDistribution.get_action_mapping(self.env_config)

    def reset(self, **kwargs) -> None:
        super().reset(**kwargs)

        self.state = self.policy.get_initial_state()
        self.c_puct: Optional[float] = None  # Used for DiL-piKL.
        self.last_action_time = None
        self.prev_action = (0, 0, 0)

    def get_action(self, obs: MbagObs, *, compute_actions_kwargs={}) -> MbagActionTuple:
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

        obs_batch = tuple(obs_piece[None] for obs_piece in obs)
        # preprocessor = ModelCatalog.get_preprocessor_for_space(self.policy.observation_space)
        # obs_batch = torch.from_numpy(preprocessor.transform(obs)[None]).to(self.policy.device)
        state_batch = [state_piece[None] for state_piece in self.state]
        state_out_batch: List[TensorType]
        action_batch: Iterable[np.ndarray]
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

        if self.confidence_threshold is not None and not force_noop:
            logits = compute_actions_info[SampleBatch.ACTION_DIST_INPUTS][0]
            probs = np.exp(logits)
            probs /= np.sum(probs)
            (action_id,) = action_batch
            # normalized_probs = probs / np.mean(probs[probs != 0])
            if probs[action_id] < self.confidence_threshold:
                action_batch = [np.array(0)]
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
        if force_noop:
            assert action_type == MbagAction.NOOP

        return action

    def get_state(self) -> List[np.ndarray]:
        return [np.array(state_part) for state_part in self.state] + [
            np.array(self.prev_action)
        ]

    def set_state(self, state: List[np.ndarray]) -> None:
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
