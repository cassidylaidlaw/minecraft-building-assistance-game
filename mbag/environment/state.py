from typing import List, Optional, TypedDict

import numpy as np

from .blocks import MinecraftBlocks
from .types import (
    CURRENT_BLOCK_STATES,
    CURRENT_BLOCKS,
    GOAL_BLOCK_STATES,
    GOAL_BLOCKS,
    INVENTORY_NUM_SLOTS,
    LAST_INTERACTED,
    NO_INTERACTION,
    PLAYER_LOCATIONS,
    FacingDirection,
    MbagInventory,
    MbagObs,
    WorldLocation,
    WorldSize,
)


class MbagStateDict(TypedDict):
    current_blocks: MinecraftBlocks
    goal_blocks: MinecraftBlocks
    player_locations: List[WorldLocation]
    player_directions: List[FacingDirection]
    player_inventories: List[MbagInventory]
    last_interacted: np.ndarray
    timestep: int


def _inventory_obs_to_inventory(
    single_player_inventory_obs: np.ndarray,
) -> MbagInventory:
    """
    Given a single player's inventory observation (i.e., an array of shape
    (NUM_BLOCKS,)), returns a MbagInventory that matches the counts. Note that since
    converting an MbagInventory to an inventory obs is lossy, the original MbagInventory
    cannot be reconstructed exactly.
    """

    inventory = np.zeros((INVENTORY_NUM_SLOTS, 2), dtype=np.int32)
    slot = 0
    for block_id, count in enumerate(single_player_inventory_obs):
        while count > 0:
            inventory[slot, 0] = block_id
            inventory[slot, 1] = min(count, 64)
            count -= inventory[slot, 1]
            slot += 1
    return inventory


def mbag_obs_to_state(
    obs: MbagObs, player_index: int, *, num_players: Optional[int] = None
) -> MbagStateDict:
    world_obs, inventory_obs, timestep = obs
    _, width, height, depth = world_obs.shape
    world_size: WorldSize = (width, height, depth)
    if num_players is None:
        num_players, _ = inventory_obs.shape

    player_inventories: List[MbagInventory] = [
        np.zeros((INVENTORY_NUM_SLOTS, 2), dtype=np.int32) for _ in range(num_players)
    ]
    player_locations: List[WorldLocation] = [
        (0.0, 0.0, 0.0) for _ in range(num_players)
    ]
    last_interacted = np.full(world_size, NO_INTERACTION, dtype=np.int8)
    for player_tag, other_player_index in enumerate(
        [player_index]
        + [
            other_player_index
            for other_player_index in range(num_players)
            if other_player_index != player_index
        ]
    ):
        player_inventories[other_player_index] = _inventory_obs_to_inventory(
            inventory_obs[player_tag]
        )

        obs_player_x, obs_player_y, obs_player_z = np.nonzero(
            world_obs[PLAYER_LOCATIONS] == player_tag + 1
        )
        assert (
            len(obs_player_x) <= 2 and len(obs_player_y) <= 2 and len(obs_player_z) <= 2
        )
        if len(obs_player_y) > 0:
            feet_index = np.argmin(obs_player_y)
            player_locations[other_player_index] = (
                obs_player_x[feet_index] + 0.5,
                obs_player_y[feet_index],
                obs_player_z[feet_index] + 0.5,
            )

        last_interacted[world_obs[LAST_INTERACTED] == player_tag + 1] = (
            other_player_index
        )

    # If this observation comes from an environment with more than the specified number
    # of players, set the last_interacted for the extra players to 0.
    last_interacted[world_obs[LAST_INTERACTED] > num_players] = 0

    player_directions = [(0.0, 0.0) for _ in range(num_players)]

    current_blocks = MinecraftBlocks(world_size)
    current_blocks.blocks[:] = world_obs[CURRENT_BLOCKS]
    current_blocks.block_states[:] = world_obs[CURRENT_BLOCK_STATES]
    goal_blocks = MinecraftBlocks(world_size)
    goal_blocks.blocks[:] = world_obs[GOAL_BLOCKS]
    goal_blocks.block_states[:] = world_obs[GOAL_BLOCK_STATES]

    return {
        "current_blocks": current_blocks,
        "goal_blocks": goal_blocks,
        "player_locations": player_locations,
        "player_directions": player_directions,
        "player_inventories": player_inventories,
        "last_interacted": last_interacted,
        "timestep": int(timestep),
    }
