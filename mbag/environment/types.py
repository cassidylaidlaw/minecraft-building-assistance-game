from datetime import datetime
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Literal, TypedDict

if TYPE_CHECKING:
    from .actions import MbagAction, MbagActionTuple
    from .malmo.malmo_client import MalmoObservationDict


WorldSize = Tuple[int, int, int]

BlockLocation = Tuple[int, int, int]

WorldLocation = Tuple[float, float, float]

MbagInventoryObs = np.ndarray
"""
1D array mapping block ids with number held in inventory.
"""

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
 5: player who last interacted with block (if any)
"""

# Channels in the MbagWorldObsArray:
CURRENT_BLOCKS = 0
CURRENT_BLOCK_STATES = 1
GOAL_BLOCKS = 2
GOAL_BLOCK_STATES = 3
PLAYER_LOCATIONS = 4
LAST_INTERACTED = 5
num_world_obs_channels = 6

# Values for the PLAYER_LOCATIONS and LAST_INTERACTED channels:
NO_ONE = 0
CURRENT_PLAYER = 1
OTHER_PLAYER = 2
NO_INTERACTION = -1

MbagObs = Tuple[MbagWorldObsArray, MbagInventoryObs, NDArray[np.int32]]
"""Tuple of (world_obs, inventory_obs, timestep)."""

MbagHumanCommandType = Literal["key", "mouse"]
MbagHumanCommand = Literal[
    "forward", "right", "left", "back", "attack", "inventory", "use"
]

MbagInventory = np.ndarray
"""
Player inventory will be stored as 2d numpy array.
Inventory slots are stored from 0 to 35 inclusive
First dimension is which inventory slot is being accessed
Second dimension is 0 for block id, 1 for block count
"""

INVENTORY_NUM_SLOTS = 36
"""The number of stacks of items a player can carry."""

STACK_SIZE = 64
"""The maximum number of blocks a player can carry in a stack."""


def get_block_counts_in_inventory(inventory: MbagInventory) -> np.ndarray:
    """
    Return a numpy array of shape (NUM_BLOCKS,) where each element is the number
    of blocks of that type in the inventory.
    """

    from .blocks import MinecraftBlocks

    counts = np.zeros((MinecraftBlocks.NUM_BLOCKS,), dtype=int)
    for slot in range(INVENTORY_NUM_SLOTS):
        block_id = inventory[slot, 0]
        counts[block_id] += inventory[slot, 1]

    return counts


class MbagInfoDict(TypedDict):
    goal_similarity: float
    """
    Number representing how similar the current blocks in the world are to the goal
    structure. Higher is more similar. This can be used as a truer "reward" than the
    potentially shaped reward given to the agent by the environment.
    """

    goal_dependent_reward: float
    """
    The reward from this step which is due to the current player's actions and which
    depends on the goal.
    """

    goal_independent_reward: float
    """
    The reward from this step which is due to the current player's actions but which
    does not depend on the goal, i.e., bonuses or penalties for no-ops and actions,
    resource gathering bonuses, etc.
    """

    own_reward: float
    """
    The reward from this step which is due to the current player's direct actions, i.e.
    the sum of goal_dependent_reward and goal_independent_reward.
    """

    noop_reward: float
    """
    The current reward for doing any action which does nothing. Note that this is
    the reward the player would receive, not necessarily the reward the player
    received at this step.
    """

    action_reward: float
    """
    The current reward for doing any action which is not a noop.  Note that this is
    the reward the player would receive, not necessarily the reward the player
    received at this step.
    """

    place_wrong_reward: float
    """
    The current reward for placing a block which is not correct, but in a place where a block
    should go. The negative of this is also given for breaking a block which is not
    correct.  Note that this is the reward the player would receive, not necessarily
    the reward the player received at this step.
    """

    own_reward_prop: float
    """
    The current proportion of this player's reward which is coming from their own
    direct actions, as opposed to other agents'.
    """

    get_resources_reward: float
    """
    The current reward for getting a resource block from the palette that the player
    did not have in their inventory previously.  Note that this is the reward the
    player would receive, not necessarily the reward the player received at this step.
    """

    attempted_action: "MbagAction"
    """
    The action that the player tried to take.
    """

    action: "MbagAction"
    """
    The action that the player effectively took. That is, if the player attempted to
    do something but it didn't actually affect the world, it is logged as NOOP.
    """

    action_correct: bool
    """
    Whether an action directly contributed to the goal, either by placing the correct
    block or breakin an incorrect block.
    """

    malmo_observations: List[Tuple[datetime, "MalmoObservationDict"]]
    """
    If this player is a human agent, then this is the full timestamped list of
    observations from Malmo since the last timestep.
    """

    human_action: "MbagActionTuple"
    """
    If this player is a human agent, then this is an action that has been deduced from
    what the human has been doing in Malmo and should be played immediately.
    """
