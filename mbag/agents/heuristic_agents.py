"""
A collection of agents following simple heuristics.
"""

import heapq
import random
from typing import Dict, List, Tuple, Type

import numpy as np

from ..environment.actions import MbagAction, MbagActionTuple, MbagActionType
from ..environment.blocks import MinecraftBlocks
from ..environment.types import CURRENT_BLOCKS, GOAL_BLOCKS, BlockLocation, MbagObs
from .action_distributions import MbagActionDistribution
from .mbag_agent import MbagAgent
from .oracle_goal_prediction_agent import OracleGoalPredictionAgent


class NoopAgent(MbagAgent):
    def get_action_distribution(self, obs: MbagObs) -> np.ndarray:
        action_dist = np.zeros(
            (MbagActionDistribution.NUM_CHANNELS,) + self.env_config["world_size"]
        )
        action_dist[0, 0, 0, 0] = 1
        return action_dist


class MovementAgent(MbagAgent):
    """
    Moves around randomly
    """

    def get_action_distribution(self, obs: MbagObs) -> np.ndarray:
        action_dist = np.zeros(
            (MbagActionDistribution.NUM_CHANNELS,) + self.env_config["world_size"]
        )
        action_dist[MbagActionDistribution.MOVE_POS_X, 0, 0, 0] = 1 / 6
        action_dist[MbagActionDistribution.MOVE_NEG_X, 0, 0, 0] = 1 / 6
        action_dist[MbagActionDistribution.MOVE_POS_Y, 0, 0, 0] = 1 / 6
        action_dist[MbagActionDistribution.MOVE_NEG_Y, 0, 0, 0] = 1 / 6
        action_dist[MbagActionDistribution.MOVE_POS_Z, 0, 0, 0] = 1 / 6
        action_dist[MbagActionDistribution.MOVE_NEG_Z, 0, 0, 0] = 1 / 6
        return action_dist


class LayerBuilderAgent(MbagAgent):
    """
    Builds the goal structure one layer at a time, from bottom to top.
    """

    current_layer: int

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.current_layer = 0

    def get_action(self, obs: MbagObs) -> MbagActionTuple:
        world_obs, _, _ = obs

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
                random.choice(np.argwhere(layer_blocks != goal_blocks))
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

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.block_frontier = []
        self.seeding = False

    def get_action(self, obs: MbagObs) -> MbagActionTuple:
        world_obs, _, _ = obs

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


class LowestBlockAgent(MbagAgent):
    """
    Builds by first breaking, then placing blocks, with the lowest possible block
    chosen first, and ties randomly broken.
    """

    def __init__(self, agent_config, env_config):
        super().__init__(agent_config, env_config)

        if not (
            self.env_config["abilities"]["inf_blocks"]
            and self.env_config["abilities"]["teleportation"]
        ):
            raise ValueError("LowestBlockAgent cannot move or collect resources")

    def get_action_distribution(self, obs: MbagObs) -> np.ndarray:
        world_obs, inventory_obs, timestep = obs
        world_obs = world_obs.astype(int)
        current_blocks: np.ndarray = world_obs[CURRENT_BLOCKS]
        goal_blocks: np.ndarray = world_obs[GOAL_BLOCKS]

        action_mask = MbagActionDistribution.get_mask(
            self.env_config,
            (world_obs[None], inventory_obs[None], timestep[None]),
        )[0]
        action_dist = np.zeros(
            (MbagActionDistribution.NUM_CHANNELS,) + self.env_config["world_size"]
        )

        action_type: MbagActionType = MbagAction.NOOP

        # First, look at all blocks that need to be broken.
        need_to_be_broken = current_blocks != goal_blocks
        can_be_broken = (
            need_to_be_broken & action_mask[MbagActionDistribution.BREAK_BLOCK]
        )
        if np.any(can_be_broken):
            possible_x, possible_y, possible_z = np.nonzero(can_be_broken)
            action_type = MbagAction.BREAK_BLOCK
        else:
            # Next, look at all blocks that need to be placed.
            needs_to_be_placed = (current_blocks == MinecraftBlocks.AIR) & (
                goal_blocks != MinecraftBlocks.AIR
            )
            # All blocks should have the same place mask, so just use dirt.
            dirt = MinecraftBlocks.NAME2ID["dirt"]
            can_be_placed = (
                needs_to_be_placed
                & action_mask[MbagActionDistribution.PLACE_BLOCK][dirt]
            )
            if np.any(can_be_placed):
                possible_x, possible_y, possible_z = np.nonzero(can_be_placed)
                action_type = MbagAction.PLACE_BLOCK

        if action_type == MbagAction.NOOP:
            action_dist[0, 0, 0, 0] = 1
        else:
            min_y = np.min(possible_y)
            in_min_layer = possible_y == min_y
            possible_x = possible_x[in_min_layer]
            possible_y = possible_y[in_min_layer]
            possible_z = possible_z[in_min_layer]

            goal_blocks_at_possible_locations = goal_blocks[
                possible_x,
                possible_y,
                possible_z,
            ]

            if action_type == MbagAction.BREAK_BLOCK:
                action_dist[MbagActionDistribution.BREAK_BLOCK][
                    possible_x,
                    possible_y,
                    possible_z,
                ] = 1
            elif action_type == MbagAction.PLACE_BLOCK:
                action_dist[MbagActionDistribution.PLACE_BLOCK][
                    goal_blocks_at_possible_locations,
                    possible_x,
                    possible_y,
                    possible_z,
                ] = 1
            else:
                assert False

            action_dist /= action_dist.sum()

        return action_dist


class MirrorBuildingAgent(MbagAgent):
    """
    Build by mirroring the other agent along the x-axis.
    """

    prev_blocks = None

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self.prev_blocks = None

    def _mirror_placed_blocks(self, blocks: np.ndarray) -> np.ndarray:
        """
        First, add all blocks from the left side to the right side (mirroring on the x-axis). Then add all blocks
        from the right side to the left side in the places where there are no blocks on the left side.

        We start with this shape,
                    ^
                ` y |      |
                    | *  * |
                    |   *  |  *
                    |  *   | - -
                    | _ _ _|_ ___ _>
                                    x
        First, wherever there's air on the right side, we replace it with whatever is on the left side
                    ^
                ` y |      |
                    | *  * | *  *
                    |   *  |  *
                    |  *   | - -
                    | _ _ _|_ ___ _>
                                    x
        Then we do the same thing on the left side.
                    ^
                ` y |      |
                    | *  * | *  *
                    |   *  |  *
                    |  * - | - -
                    | _ _ _|_ ___ _>
                                    x
        """
        mirror = blocks.copy()

        for x in range(mirror.shape[0] // 2):
            left_slice = mirror[x]
            right_slice = mirror[-1 - x]
            # First, wherever there's air on the right side, we replace it with whatever is on the left side
            right_slice[right_slice == MinecraftBlocks.AIR] = left_slice[
                right_slice == MinecraftBlocks.AIR
            ]
            # Then, we do the same thing for the left side
            left_slice[left_slice == MinecraftBlocks.AIR] = right_slice[
                left_slice == MinecraftBlocks.AIR
            ]

        return mirror

    def _diff_indices(self, a1: np.ndarray, a2: np.ndarray):
        """Get indices where two numpy arrays differ"""
        return np.transpose((a1 != a2).nonzero())

    def get_action(self, obs: MbagObs) -> MbagActionTuple:
        blocks = obs[0][0]

        blocks_mirrored = self._mirror_placed_blocks(blocks)
        differences = self._diff_indices(blocks, blocks_mirrored)
        if np.size(differences) == 0:
            return MbagAction.NOOP, 0, 0
        else:
            index = np.ravel_multi_index(
                tuple(differences[0]), self.env_config["world_size"]
            )
            return (
                MbagAction.PLACE_BLOCK,
                index,
                int(blocks_mirrored[tuple(differences[0])]),
            )


class RandomAgent(MbagAgent):
    """
    Randomly takes actions.
    """

    def get_action_distribution(self, obs: MbagObs) -> np.ndarray:
        action_dist = np.ones(
            (MbagActionDistribution.NUM_CHANNELS,) + self.env_config["world_size"]
        )
        obs_batch = obs[0][None], obs[1][None], obs[2][None]
        action_mask = MbagActionDistribution.get_mask(self.env_config, obs_batch)[0]
        action_dist[~action_mask] = 0

        # Make each possible action equally likely.
        for action_type in MbagAction.ACTION_TYPES:
            action_type_channel = MbagActionDistribution.ACTION_TYPE2CHANNEL[
                action_type
            ]
            total_action_type_prob = action_dist[action_type_channel].sum()
            if total_action_type_prob > 0:
                action_dist[action_type_channel] /= total_action_type_prob

        action_dist /= action_dist.sum()

        return action_dist


ALL_HEURISTIC_AGENTS: Dict[str, Type[MbagAgent]] = {
    "noop": NoopAgent,
    "movement": MovementAgent,
    "random": RandomAgent,
    "layer_builder": LayerBuilderAgent,
    "priority_queue": PriorityQueueAgent,
    "lowest_block": LowestBlockAgent,
    "mirror_builder": MirrorBuildingAgent,
    "oracle_goal_predictor": OracleGoalPredictionAgent,
}
