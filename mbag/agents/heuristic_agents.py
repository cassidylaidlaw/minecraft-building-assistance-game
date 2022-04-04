"""
A collection of agents following simple heuristics.
"""

from typing import Dict, List, Tuple, Type
import numpy as np
import random
import heapq

from ..environment.types import (
    BlockLocation,
    MbagActionTuple,
    MbagActionType,
    MbagObs,
    MbagAction,
)
from ..environment.blocks import MinecraftBlocks
from .mbag_agent import MbagAgent


class NoopAgent(MbagAgent):
    def get_action_type_distribution(self, obs: MbagObs) -> np.ndarray:
        action_dist = np.zeros(MbagAction.NUM_ACTION_TYPES)
        action_dist[MbagAction.NOOP] = 1
        return action_dist


class HardcodedBuilderAgent(MbagAgent):
    """
    Builds the simple goal generator using a hardcoded predetermined sequence of moves
    """

    current_command: int

    def reset(self):
        self.current_command = 0

    def get_action(self, obs: MbagObs) -> MbagActionTuple:
        command_list: List[MbagActionTuple] = [
            (MbagAction.MOVE_NEG_Y, 0, 0),
            (MbagAction.MOVE_POS_Y, 0, 0),
            (MbagAction.MOVE_NEG_X, 0, 0),
            (MbagAction.MOVE_NEG_Z, 0, 0),
            (
                MbagAction.PLACE_BLOCK,
                int(
                    np.ravel_multi_index(
                        (1, 2, 1),
                        self.env_config["world_size"],
                    )
                ),
                MinecraftBlocks.NAME2ID["cobblestone"],
            ),
            (MbagAction.MOVE_POS_X, 0, 0),
            (MbagAction.MOVE_POS_X, 0, 0),
            (
                MbagAction.PLACE_BLOCK,
                int(
                    np.ravel_multi_index(
                        (2, 2, 1),
                        self.env_config["world_size"],
                    )
                ),
                MinecraftBlocks.NAME2ID["cobblestone"],
            ),
            (
                MbagAction.PLACE_BLOCK,
                int(
                    np.ravel_multi_index(
                        (3, 2, 1),
                        self.env_config["world_size"],
                    )
                ),
                MinecraftBlocks.NAME2ID["cobblestone"],
            ),
            (MbagAction.MOVE_POS_Y, 0, 0),
            (MbagAction.MOVE_POS_Z, 0, 0),
            (MbagAction.MOVE_POS_Z, 0, 0),
            (
                MbagAction.PLACE_BLOCK,
                int(
                    np.ravel_multi_index(
                        (3, 2, 2),
                        self.env_config["world_size"],
                    )
                ),
                MinecraftBlocks.NAME2ID["cobblestone"],
            ),
            (
                MbagAction.PLACE_BLOCK,
                int(
                    np.ravel_multi_index(
                        (2, 2, 2),
                        self.env_config["world_size"],
                    )
                ),
                MinecraftBlocks.NAME2ID["cobblestone"],
            ),
            (
                MbagAction.PLACE_BLOCK,
                int(
                    np.ravel_multi_index(
                        (1, 2, 2),
                        self.env_config["world_size"],
                    )
                ),
                MinecraftBlocks.NAME2ID["cobblestone"],
            ),
            (MbagAction.MOVE_POS_Z, 0, 0),
            # (MbagAction.MOVE_NEG_Z, 0, 0),
            (
                MbagAction.PLACE_BLOCK,
                int(
                    np.ravel_multi_index(
                        (3, 2, 3),
                        self.env_config["world_size"],
                    )
                ),
                MinecraftBlocks.NAME2ID["cobblestone"],
            ),
            (
                MbagAction.PLACE_BLOCK,
                int(
                    np.ravel_multi_index(
                        (2, 2, 3),
                        self.env_config["world_size"],
                    )
                ),
                MinecraftBlocks.NAME2ID["cobblestone"],
            ),
            (
                MbagAction.PLACE_BLOCK,
                int(
                    np.ravel_multi_index(
                        (1, 2, 3),
                        self.env_config["world_size"],
                    )
                ),
                MinecraftBlocks.NAME2ID["cobblestone"],
            ),
        ]
        if self.current_command >= len(command_list):
            return (MbagAction.NOOP, 0, 0)

        action = command_list[self.current_command]
        self.current_command += 1
        return action

    def get_state(self) -> List[np.ndarray]:
        return [np.array([self.current_command])]

    def set_state(self, state: List[np.ndarray]) -> None:
        self.current_command = int(state[0][0])


class MovementAgent(MbagAgent):
    """
    Moves around randomly
    """

    def get_action_type_distribution(self, obs: MbagObs) -> np.ndarray:
        action_dist = np.zeros(MbagAction.NUM_ACTION_TYPES)
        action_dist[MbagAction.MOVE_POS_X] = 1 / 6
        action_dist[MbagAction.MOVE_NEG_X] = 1 / 6
        action_dist[MbagAction.MOVE_POS_Y] = 1 / 6
        action_dist[MbagAction.MOVE_NEG_Y] = 1 / 6
        action_dist[MbagAction.MOVE_POS_Z] = 1 / 6
        action_dist[MbagAction.MOVE_NEG_Z] = 1 / 6

        return action_dist


class LayerBuilderAgent(MbagAgent):
    """
    Builds the goal structure one layer at a time, from bottom to top.
    """

    current_layer: int

    def reset(self):
        self.current_layer = 0

    def get_action(self, obs: MbagObs) -> MbagActionTuple:
        (world_obs,) = obs

        # Check if current layer is done.
        while self.current_layer < self.env_config["world_size"][1] and np.all(
            world_obs[:2, :, self.current_layer]
            == world_obs[2:4, :, self.current_layer]
        ):
            self.current_layer += 1

        action_type: MbagActionType
        if self.current_layer == self.env_config["world_size"][1]:
            action_type = MbagAction.NOOP
            return action_type, 0, 0
        else:
            layer_blocks = world_obs[0, :, self.current_layer, :]
            goal_blocks = world_obs[2, :, self.current_layer, :]

            layer_block_location: Tuple[int, int] = tuple(
                random.choice(np.argwhere(layer_blocks != goal_blocks))  # type: ignore
            )
            block_location: BlockLocation = (
                layer_block_location[0],
                self.current_layer,
                layer_block_location[1],
            )
            block_location_id = int(
                np.ravel_multi_index(block_location, self.env_config["world_size"])
            )

            block_id: int
            if layer_blocks[layer_block_location] == MinecraftBlocks.AIR:
                action_type = MbagAction.PLACE_BLOCK
                block_id = goal_blocks[layer_block_location]
            else:
                action_type = MbagAction.BREAK_BLOCK
                block_id = 0

            return action_type, block_location_id, int(block_id)

    def get_state(self) -> List[np.ndarray]:
        return [np.array([self.current_layer])]

    def set_state(self, state: List[np.ndarray]) -> None:
        self.current_layer = int(state[0][0])


class PriorityQueueAgent(MbagAgent):
    """
    Places the block with lowest layer that is reachable
    Assumes that there is a block at layer 0, otherwise the structure is floating
    Todo: Preprocess the goal?
    """

    seeding: bool  # Whether blocks have been placed yet
    block_frontier: list  # PQ to store blocks and their layers

    def reset(self):
        self.block_frontier = []
        self.seeding = False

    def get_action(self, obs: MbagObs) -> MbagActionTuple:
        (world_obs,) = obs

        # Check if we need to seed the PQ with all placeable blocks from the initial goal state
        if not self.seeding:
            self.seeding = True

            for layer in range(3):
                goal_blocks = world_obs[2, :, layer, :]
                layer_blocks = world_obs[0, :, layer, :]
                for layer_block_location in np.argwhere(layer_blocks != goal_blocks):
                    heapq.heappush(
                        self.block_frontier, (layer, tuple(layer_block_location))
                    )

        action_type: MbagActionType
        # If there is nothing in the priority queue, run a noop
        if len(self.block_frontier) == 0:
            action_type = MbagAction.NOOP
            return action_type, 0, 0
        else:

            # Pop the lowest block location off of the priority queue
            layer, location = heapq.heappop(self.block_frontier)

            # Iterate over blocks adjacent to the current block
            axes = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]
            for direction in axes:
                x = location[0] + direction[0]
                y = layer + direction[1]
                z = location[1] + direction[2]

                if (
                    x < 0
                    or y < 0
                    or z < 0
                    or x >= world_obs.shape[1]
                    or y >= world_obs.shape[2]
                    or z >= world_obs.shape[3]
                ):
                    continue

                # Add the block location to the queue if
                # Either the goal block is different from the actual block
                # Or the world block is not air (means we need to explore it further)
                goal_block = world_obs[2, x, y, z]
                actual_block = world_obs[0, x, y, z]
                if (goal_block != actual_block) and not (
                    y,
                    (x, z),
                ) in self.block_frontier:
                    heapq.heappush(self.block_frontier, (y, (x, z)))

            # Decide if we are making a change or simply exploring the terrain
            # if (world_obs[0, location[0], layer, location[1]] == world_obs[2, location[0], layer, location[1]]):
            #     action_type = MbagAction.NOOP
            #     block_id = 0
            # Decide whether a block needs to be placed or removed
            if world_obs[0, location[0], layer, location[1]] == MinecraftBlocks.AIR:
                action_type = MbagAction.PLACE_BLOCK
                block_id = world_obs[2, location[0], layer, location[1]]
            else:
                action_type = MbagAction.BREAK_BLOCK
                block_id = 0

                # If the goal block is not air, we need to process it again in the heap
                if (
                    not world_obs[2, location[0], layer, location[1]]
                    == MinecraftBlocks.AIR
                ):
                    heapq.heappush(self.block_frontier, (layer, location))

            # Encode and return action
            block_location: BlockLocation = (
                location[0],
                layer,
                location[1],
            )
            block_location_id = int(
                np.ravel_multi_index(block_location, self.env_config["world_size"])
            )

            return action_type, block_location_id, int(block_id)

    def get_state(self) -> List[np.ndarray]:
        return [np.array([self.seeding, self.block_frontier], dtype=object)]

    def set_state(self, state: List[np.ndarray]) -> None:
        self.seeding = bool(state[0][0])
        self.block_frontier = list(map(tuple, state[0][1]))


ALL_HEURISTIC_AGENTS: Dict[str, Type[MbagAgent]] = {
    "layer_builder": LayerBuilderAgent,
    "priority_queue": PriorityQueueAgent,
}
