from typing import List, Mapping, Tuple, cast

import numpy as np
from typing_extensions import Literal

from .types import BlockLocation, WorldSize

MbagActionType = Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
MbagActionTuple = Tuple[MbagActionType, int, int]
"""
An action tuple (action_type, block_location, block_id).
"""
MBAG_ACTION_BREAK_PALETTE_NAME = "break_palette"


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
    GIVE_BLOCK: MbagActionType = 9

    NUM_ACTION_TYPES = 10
    ACTION_TYPES: List[MbagActionType] = [
        NOOP,
        PLACE_BLOCK,
        BREAK_BLOCK,
        MOVE_POS_X,
        MOVE_NEG_X,
        MOVE_POS_Y,
        MOVE_NEG_Y,
        MOVE_POS_Z,
        MOVE_NEG_Z,
        GIVE_BLOCK,
    ]
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
        GIVE_BLOCK: "GIVE_BLOCK",
    }

    action_type: MbagActionType
    block_location_index: int
    block_location: BlockLocation
    block_id: int

    # Which actions require which attributes:
    BLOCK_ID_ACTION_TYPES = [PLACE_BLOCK, GIVE_BLOCK]
    BLOCK_LOCATION_ACTION_TYPES = [PLACE_BLOCK, BREAK_BLOCK, GIVE_BLOCK]
    MOVE_ACTION_TYPES = [
        MOVE_POS_X,
        MOVE_NEG_X,
        MOVE_POS_Y,
        MOVE_NEG_Y,
        MOVE_POS_Z,
        MOVE_NEG_Z,
    ]

    MOVE_ACTION_DELTAS: Mapping[MbagActionType, Tuple[int, int, int]] = {
        MOVE_POS_X: (1, 0, 0),
        MOVE_NEG_X: (-1, 0, 0),
        MOVE_POS_Y: (0, 1, 0),
        MOVE_NEG_Y: (0, -1, 0),
        MOVE_POS_Z: (0, 0, 1),
        MOVE_NEG_Z: (0, 0, -1),
    }

    def __init__(self, action_tuple: MbagActionTuple, world_size: WorldSize):
        self.action_type, self.block_location_index, self.block_id = action_tuple
        self.block_location = cast(
            BlockLocation,
            tuple(np.unravel_index(self.block_location_index, world_size)),
        )
        self._world_size = world_size

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

    def to_tuple(self) -> MbagActionTuple:
        return (self.action_type, self.block_location_index, self.block_id)

    def __eq__(self, other_action: object):
        if not isinstance(other_action, MbagAction):
            return False
        return self.to_tuple() == other_action.to_tuple()

    @classmethod
    def noop_action(cls):
        return cls((MbagAction.NOOP, 0, 0), (1, 1, 1))

    def is_palette(self, inf_blocks: bool) -> bool:
        """Returns whether this action is on the palette."""
        # The action can only be on the palette if inf_blocks is False,
        # otherwise the agent does not need to collect blocks and the palette
        # does not exist.
        return (self.block_location[0] == self._world_size[0] - 1) and not inf_blocks
