from abc import ABC
from typing import Any, List, cast
from mbag.environment.blocks import MinecraftBlocks
import numpy as np

from ..environment.types import MbagAction, MbagActionType, MbagObs, MbagActionTuple
from ..environment.mbag_env import MbagConfigDict


class MbagAgent(ABC):
    """
    An MBAG agent which chooses actions based on observations.
    """

    agent_config: Any
    env_config: MbagConfigDict

    def __init__(self, agent_config: Any, env_config: MbagConfigDict):
        self.agent_config = agent_config
        self.env_config = env_config

    def reset(self) -> None:
        """
        This method is called whenever a new episode starts; it can be used to clear
        internal state or otherwise prepare for a new episode.
        """

        pass

    def get_action_type_distribution(self, obs: MbagObs) -> np.ndarray:
        """
        This should return a distribution over action types that sums to 1.
        """

        uniform_dist = np.ones(MbagAction.NUM_ACTION_TYPES)
        uniform_dist /= uniform_dist.sum()
        return uniform_dist

    def get_block_id_distribution(
        self, obs: MbagObs, action_type: MbagActionType
    ) -> np.ndarray:
        """
        This should return a distribution over block IDs that sums to 1 given the
        observation and a sampled action type.
        """

        uniform_dist = np.ones(MinecraftBlocks.NUM_PLACEABLE_BLOCKS)
        uniform_dist /= uniform_dist.sum()
        return uniform_dist

    def get_block_location_distribution(
        self, obs: MbagObs, action_type: MbagActionType, block_id: int
    ) -> np.ndarray:
        """
        This should return a distribution over block locations (3d float array) that
        sums to 1 given the observation, a sampled action type, and a block ID.
        """

        uniform_dist = np.ones(self.env_config["world_size"])
        uniform_dist /= uniform_dist.sum()
        return uniform_dist

    def get_action(self, obs: MbagObs) -> MbagActionTuple:
        """
        This should return an action ID to take in the environment. Either this or the
        get_action_*_distribution methods should be overridden.
        """

        action_type_dist = self.get_action_type_distribution(obs)
        action_type: MbagActionType = cast(
            MbagActionType, int(np.random.multinomial(1, action_type_dist).argmax())
        )

        if (
            action_type in MbagAction.BLOCK_ID_ACTION_TYPES
            or action_type in MbagAction.BLOCK_LOCATION_ACTION_TYPES
        ):
            block_id_dist = self.get_block_id_distribution(obs, action_type)
            block_id = int(np.random.multinomial(1, block_id_dist).argmax())

            if action_type in MbagAction.BLOCK_LOCATION_ACTION_TYPES:
                block_location_dist = self.get_block_location_distribution(
                    obs, action_type, block_id
                )
                block_location = int(
                    np.random.multinomial(1, block_location_dist.flat).argmax()
                )
            else:
                block_location = 0
        else:
            block_id = 0
            block_location = 0

        return action_type, block_location, block_id

    def get_state(self) -> List[np.ndarray]:
        """
        Get the current state of this agent as a list of zero or more numpy arrays.
        The agent should be able to be set back to its previous state by calling
        set_state with the return value of this method.
        """

        return []

    def set_state(self, state: List[np.ndarray]) -> None:
        """
        Restore the agent's state to what it was when get_state was called.
        """

        pass
