from typing import List, cast
from typing_extensions import TypedDict
import numpy as np
from ray.rllib.policy import Policy

from mbag.agents.mbag_agent import MbagAgent
from mbag.environment.mbag_env import MbagConfigDict
from mbag.environment.types import MbagActionTuple, MbagObs


class RllibMbagAgentConfigDict(TypedDict):
    policy: Policy


class RllibMbagAgent(MbagAgent):
    agent_config: RllibMbagAgentConfigDict
    state: List[np.ndarray]

    def __init__(self, agent_config: MbagConfigDict, env_config: MbagConfigDict):
        super().__init__(agent_config, env_config)

        self.policy = self.agent_config["policy"]

    def reset(self) -> None:
        self.state = self.policy.get_initial_state()

    def get_action(self, obs: MbagObs) -> MbagActionTuple:
        obs_batch = tuple(obs_piece[None] for obs_piece in obs)
        state_batch = [state_piece[None] for state_piece in self.state]
        action_batch, state_out_batch, info = self.policy.compute_actions(
            obs_batch, state_batch
        )
        self.state = [state_piece[0] for state_piece in state_out_batch]
        return cast(
            MbagActionTuple,
            tuple(int(action_piece[0]) for action_piece in action_batch),
        )

    def get_state(self) -> List[np.ndarray]:
        return self.state

    def set_state(self, state: List[np.ndarray]) -> None:
        self.state = state
