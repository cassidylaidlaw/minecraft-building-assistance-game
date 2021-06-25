from typing import Dict, List, Optional, Sequence, Set, Tuple, TypeVar, cast
from typing_extensions import Literal
import numpy as np
import random

from .types import BlockLocation, MbagAction, WorldLocation, WorldSize


def cartesian_product(*arrays):
    """
    From https://stackoverflow.com/a/11146645/200508
    """

    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)


MAX_PLAYER_REACH = 3

KT = TypeVar("KT")
VT = TypeVar("VT")


def map_set_through_dict(set_to_map: Set[KT], map_dict: Dict[KT, VT]) -> Set[VT]:
    return {map_dict[key] for key in set_to_map}


class MinecraftBlocks(object):
    """
    Represents a volume of Minecraft blocks, including the blocks themselves and any
    "block state", e.g. orientation.
    """

    ID2NAME: List[str] = [
        "air",
        "bedrock",
        "dirt",
        "brick_block",
        "clay",
        "cobblestone",
        "glass",
        "gravel",
        "hardened_clay",
        "hay_block",
        "iron_block",
        "log",
        "nether_brick",
        "netherrack",
        "planks",
        "quartz_block",
        "redstone_block",
        "redstone_lamp",
        "sandstone",
        "stained_hardened_clay",
        "stone",
        "stonebrick",
        "wool",
    ]
    NAME2ID: Dict[str, int] = {
        **{block_name: block_id for block_id, block_name in enumerate(ID2NAME)},
        # Alias names:
        "grass": 2,
    }
    AIR = NAME2ID["air"]

    PLACEABLE_BLOCK_NAMES = set(ID2NAME[2:])  # Can't place air or bedrock.
    PLACEABLE_BLOCK_IDS = map_set_through_dict(PLACEABLE_BLOCK_NAMES, NAME2ID)
    NUM_PLACEABLE_BLOCKS = len(PLACEABLE_BLOCK_IDS)

    SOLID_BLOCK_NAMES: Set[str] = {
        "bedrock",
        "dirt",
        "brick_block",
        "clay",
        "cobblestone",
        "glass",
        "gravel",
        "hardened_clay",
        "hay_block",
        "iron_block",
        "log",
        "nether_brick",
        "netherrack",
        "planks",
        "quartz_block",
        "redstone_block",
        "redstone_lamp",
        "sandstone",
        "stained_hardened_clay",
        "stone",
        "stonebrick",
        "wool",
    }
    SOLID_BLOCK_IDS = map_set_through_dict(SOLID_BLOCK_NAMES, NAME2ID)

    def __init__(self, size: Tuple[int, int, int]):
        self.size = size
        self.blocks = np.zeros(self.size, np.uint8)
        self.block_states = np.zeros(self.size, np.uint8)

    def copy(self) -> "MinecraftBlocks":
        copy = MinecraftBlocks(self.size)
        copy.blocks[:] = self.blocks
        copy.block_states[:] = self.block_states
        return copy

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MinecraftBlocks):
            return bool(np.all(self.blocks == other.blocks)) and bool(
                np.all(self.block_states == other.block_states)
            )
        else:
            return super().__eq__(other)

    def __getitem__(self, location: BlockLocation) -> Tuple[np.uint8, np.uint8]:
        return (self.blocks[location], self.block_states[location])

    def is_valid_block_location(self, location: BlockLocation) -> bool:
        return (
            location[0] >= 0
            and location[0] < self.size[0]
            and location[1] >= 0
            and location[1] < self.size[1]
            and location[2] >= 0
            and location[2] < self.size[2]
        )

    def valid_block_locations(self, locations: np.ndarray) -> np.ndarray:
        return cast(
            np.ndarray,
            (locations[:, 0] >= 0)
            & (locations[:, 0] < self.size[0])
            & (locations[:, 1] >= 0)
            & (locations[:, 1] < self.size[1])
            & (locations[:, 2] >= 0)
            & (locations[:, 2] < self.size[2]),
        )

    def try_break_place(
        self,
        action_type: Literal[1, 2],
        block_location: BlockLocation,
        block_id: int = 0,
        player_location: Optional[WorldLocation] = None,
    ) -> Optional[Tuple[WorldLocation, WorldLocation]]:
        """
        Try to place or break a block (depending on action_type) at the given
        block_location. If player_location is not given, then this will search for a
        player location that is empty and where the block can be placed/broken from.
        If the block can be placed or broken, then returns a tuple with the successful
        player location and click location, and updates the blocks accordingly.
        """

        # Check if block can be placed or broken at all.
        if action_type == MbagAction.PLACE_BLOCK:
            if self.blocks[block_location] != MinecraftBlocks.AIR:
                # Can only place block in air space.
                return None
        else:
            if self.blocks[block_location] in [
                MinecraftBlocks.AIR,
                MinecraftBlocks.NAME2ID["bedrock"],
            ]:
                # Can't break these blocks.
                return None

        # Now, look for a location and viewpoint from which to place/break block.
        click_locations = np.empty((3 * 2 * 3 * 3, 3))
        shift = 1e-4 if action_type == MbagAction.BREAK_BLOCK else -1e-4
        click_location_index = 0
        for face_dim in range(3):
            for face in [0 - shift, 1 + shift]:
                # If we are placing, need to make sure that there is a solid block
                # surface to place against.
                if action_type == MbagAction.PLACE_BLOCK:
                    against_block_location_arr = np.array(block_location)
                    against_block_location_arr[face_dim] += np.sign(face - 0.5)
                    against_block_location: BlockLocation = cast(
                        BlockLocation, tuple(against_block_location_arr.astype(int))
                    )
                    if (
                        not self.is_valid_block_location(against_block_location)
                        or self.blocks[against_block_location]
                        not in MinecraftBlocks.SOLID_BLOCK_IDS
                    ):
                        continue

                for u in [0.1, 0.5, 0.9]:
                    for v in [0.1, 0.5, 0.9]:
                        click_location = click_locations[click_location_index]
                        click_location[:] = block_location
                        click_location[face_dim] += face
                        click_location[face_dim - 1] += v
                        click_location[face_dim - 2] += u
                        click_location_index += 1
        click_locations = click_locations[:click_location_index]
        click_locations = click_locations[self.valid_block_locations(click_locations)]

        player_locations: np.ndarray
        if player_location is not None:
            player_locations = np.array([player_location])
        else:
            player_deltas = cartesian_product(
                np.linspace(-4, 4, 9),
                np.linspace(-5, 3, 9),
                np.linspace(-4, 4, 9),
            )
            # Remove deltas which would put the player inside the block being placed/
            # broken.
            player_deltas = player_deltas[
                ~(
                    (player_deltas[:, 0] == 0)
                    & (player_deltas[:, 1] >= -1)
                    & (
                        player_deltas[:, 1]
                        <= (1 if action_type == MbagAction.PLACE_BLOCK else 0)
                    )
                    & (player_deltas[:, 2] == 0)
                )
            ]

            block_player_location = np.array(block_location, float)
            block_player_location[0] += 0.5
            block_player_location[2] += 0.5
            player_locations = player_deltas + block_player_location[None, :]

        # Restrict player locations to those inside the world.
        player_locations = player_locations[
            self.valid_block_locations(player_locations)
        ]

        # Make blocks array with two layers of air above to make calculations easier.
        blocks = np.concatenate(
            [self.blocks, np.zeros((self.size[0], 2, self.size[2]), np.uint8)], axis=1
        )

        # Restrict player locations to those where they aren't inside a block.
        feet_block_locations = player_locations.astype(int)
        head_block_locations = feet_block_locations.copy()
        head_block_locations[:, 1] += 1
        player_locations = player_locations[
            (
                blocks.flat[np.ravel_multi_index(feet_block_locations.T, blocks.shape)]
                == MinecraftBlocks.AIR
            )
            & (
                blocks.flat[np.ravel_multi_index(head_block_locations.T, blocks.shape)]
                == MinecraftBlocks.AIR
            )
        ]

        player_viewpoints = player_locations.copy()
        player_viewpoints[:, 1] += 1.6  # Player viewpoint is 1.6 m above feet.

        viewpoint_click_candidates: np.ndarray = np.empty(
            (len(player_viewpoints), len(click_locations), 2, 3)
        )
        viewpoint_click_candidates[:, :, 0, :] = player_viewpoints[:, None, :]
        viewpoint_click_candidates[:, :, 1, :] = click_locations[None, :, :]
        viewpoint_click_candidates = viewpoint_click_candidates.reshape(-1, 2, 3)

        # Calculate deltas and make sure that the click location is within the reachable
        # distance.
        deltas = viewpoint_click_candidates[:, 1] - viewpoint_click_candidates[:, 0]
        reachable = (deltas ** 2).sum(axis=1) ** 0.5 <= MAX_PLAYER_REACH
        viewpoint_click_candidates = viewpoint_click_candidates[reachable]
        deltas = deltas[reachable]
        viewpoints = viewpoint_click_candidates[:, 0]

        # Voxel traversal to make sure there are no blocks in between the viewpoint
        # and the click location.
        # Based on http://www.cse.yorku.ca/~amana/research/grid.pdf
        step = np.sign(deltas).astype(int)
        t_max = np.abs(((-step * viewpoints) - np.floor(-step * viewpoints)) / deltas)
        t_max[np.isnan(t_max)] = 1
        t_delta = np.abs(1 / deltas)
        t_delta[deltas == 0] = 1

        intersection = np.zeros(viewpoints.shape[0], bool)
        current_block_locations = viewpoints.astype(int)
        while np.any(t_max < 1):
            min_mask = np.zeros_like(t_max, dtype=int)
            min_mask[range(min_mask.shape[0]), np.argmin(t_max, axis=1)] = 1
            min_mask[np.all(t_max >= 1, axis=1)] = 0
            t_max += t_delta * min_mask
            current_block_locations += step * min_mask

            intersection |= (
                blocks.flat[
                    np.ravel_multi_index(current_block_locations.T, blocks.shape)
                ]
                != MinecraftBlocks.AIR
            )

        viewpoint_click_candidates = viewpoint_click_candidates[~intersection]
        if len(viewpoint_click_candidates) == 0:
            # No possible location to place/break block from.
            return None

        # Actually break/place the block.
        if action_type == MbagAction.BREAK_BLOCK:
            self.blocks[block_location] = MinecraftBlocks.AIR
            self.block_states[block_location] = 0
        else:
            self.blocks[block_location] = block_id
            self.block_states[block_location] = 0
        # TODO: add block to inventory?

        viewpoint, click_location = random.choice(
            cast(Sequence[Tuple[np.ndarray, np.ndarray]], viewpoint_click_candidates)
        )
        player_location_list = list(viewpoint)
        player_location_list[1] -= 1.6
        return (
            cast(WorldLocation, tuple(player_location_list)),
            cast(WorldLocation, tuple(click_location)),
        )

    @classmethod
    def from_malmo_grid(
        cls, size: WorldSize, block_names: List[str]
    ) -> "MinecraftBlocks":
        # import logging; logger = logging.getLogger(__name__)
        # logger.info(block_names[:800])
        block_ids = [MinecraftBlocks.NAME2ID[block_name] for block_name in block_names]
        blocks = MinecraftBlocks(size)
        np.transpose(blocks.blocks, (1, 2, 0)).flat[:] = block_ids  # type: ignore
        return blocks
