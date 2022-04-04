from typing import List, Tuple, cast
from typing_extensions import Literal, TypedDict
import numpy as np


WorldSize = Tuple[int, int, int]

BlockLocation = Tuple[int, int, int]

WorldLocation = Tuple[float, float, float]

FacingDirection = Tuple[float, float]  # Degrees horizontally, then vertically

MbagWorldObsArray = np.ndarray
"""
The world part of the observation in the form of a 4d numpy array of uint8. The last
three dimensions are spatial and the first is channels, each of which represents
different information about the world. The channels are
 0: current blocks
 1: current block states
 2: goal blocks
 3: goal block states
 4: player locations
"""

CURRENT_BLOCKS = 0
CURRENT_BLOCK_STATES = 1
GOAL_BLOCKS = 2
GOAL_BLOCK_STATES = 3
PLAYER_LOCATIONS = 4
num_world_obs_channels = 5

MbagObs = Tuple[MbagWorldObsArray]


MbagActionType = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8]
MbagActionTuple = Tuple[MbagActionType, int, int]
"""
An action tuple (action_type, block_location, block_id).
"""


class MbagAction(object):
    """
    An action in MBAG which may or may not operate on a particular block.
    """

    NOOP: MbagActionType = 0
    PLACE_BLOCK: MbagActionType = 1
    BREAK_BLOCK: MbagActionType = 2

    MOVE_POS_X: MbagActionType = 3
    MOVE_NEG_X: MbagActionType = 4
    MOVE_POS_Y: MbagActionType = 5
    MOVE_NEG_Y: MbagActionType = 6
    MOVE_POS_Z: MbagActionType = 7
    MOVE_NEG_Z: MbagActionType = 8

    NUM_ACTION_TYPES = 9
    ACTION_TYPE_NAMES = {
        NOOP: "NOOP",
        PLACE_BLOCK: "PLACE_BLOCK",
        BREAK_BLOCK: "BREAK_BLOCK",
        MOVE_POS_X: "MOVE_POS_X",
        MOVE_NEG_X: "MOVE_NEG_X",
        MOVE_POS_Y: "MOVE_POS_Y",
        MOVE_NEG_Y: "MOVE_NEG_Y",
        MOVE_POS_Z: "MOVE_POS_Z",
        MOVE_NEG_Z: "MOVE_NEG_Z",
    }

    action_type: MbagActionType
    block_location: BlockLocation
    block_id: int

    # Which actions require which attributes:
    BLOCK_ID_ACTION_TYPES = [PLACE_BLOCK]
    BLOCK_LOCATION_ACTION_TYPES = [PLACE_BLOCK, BREAK_BLOCK]

    def __init__(self, action_tuple: MbagActionTuple, world_size: WorldSize):
        self.action_type, block_location_index, self.block_id = action_tuple
        self.block_location = cast(
            BlockLocation, tuple(np.unravel_index(block_location_index, world_size))
        )

    def __str__(self):
        from .blocks import MinecraftBlocks

        parts: List[str] = [MbagAction.ACTION_TYPE_NAMES[self.action_type]]
        if self.action_type in MbagAction.BLOCK_ID_ACTION_TYPES:
            parts.append(MinecraftBlocks.ID2NAME[self.block_id])
        if self.action_type in MbagAction.BLOCK_LOCATION_ACTION_TYPES:
            parts.append(str(self.block_location))
        return " ".join(parts)

    def __repr__(self):
        return f"MbagAction<{self}>"


class MbagInfoDict(TypedDict):
    goal_similarity: float
    """
    Number representing how similar the current blocks in the world are to the goal
    structure. Higher is more similar. This can be used as a truer "reward" than the
    potentially shaped reward given to the agent by the environment.
    """

    own_reward: float
    """
    The reward from this step which is due to the current player's direct actions.
    """

    own_reward_prop: float
    """
    The current proportion of this player's reward which is coming from their own
    direct actions, as opposed to other agents'.
    """
