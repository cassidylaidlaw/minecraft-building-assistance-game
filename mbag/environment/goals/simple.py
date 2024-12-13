import random
from typing import List, Optional, Set, TypedDict

import cc3d
import numpy as np

from ..blocks import MinecraftBlocks
from ..types import WorldSize
from .goal_generator import GoalGenerator, GoalGeneratorConfig


class SingleGoalGenerator(GoalGenerator):
    """Abstract class for goal generators that only generate a single goal."""

    num_remaining_goals: Optional[int] = 1


class BasicGoalGenerator(SingleGoalGenerator):
    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        goal = MinecraftBlocks(size)
        goal.blocks[:, :2, :] = MinecraftBlocks.NAME2ID["cobblestone"]
        return goal


class SetGoalGeneratorConfig(TypedDict):
    goals: List[MinecraftBlocks]


class SetGoalGenerator(GoalGenerator):
    """
    Randomly chooses one of a number of structures given in the config.
    """

    config: SetGoalGeneratorConfig

    def __init__(self, config: GoalGeneratorConfig):
        super().__init__(config)
        self.num_remaining_goals = len(self.config["goals"])

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        width, height, depth = size
        while True:
            goal = random.choice(self.config["goals"])
            goal_width, goal_height, goal_depth = goal.size
            if goal_width <= width and goal_height <= height and goal_depth <= depth:
                return goal


class RandomGoalGeneratorConfig(TypedDict):
    filled_prop: float


class RandomGoalGenerator(GoalGenerator):
    """
    Generates a random structure of connected blocks.
    """

    default_config: RandomGoalGeneratorConfig = {"filled_prop": 0.6}
    config: RandomGoalGeneratorConfig

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        goal = MinecraftBlocks(size)
        structure_mask_noise = (
            np.random.rand(*goal.blocks.shape) < self.config["filled_prop"]
        )
        structure_mask_ccs = cc3d.connected_components(
            structure_mask_noise, connectivity=6
        )
        ground_ccs: Set[int] = set(structure_mask_ccs[:, 0, :].reshape(-1).tolist())
        if np.any(~structure_mask_noise[:, 0, :]):
            ground_ccs.remove(0)
        structure_mask = np.isin(structure_mask_ccs, list(ground_ccs))

        block_ids = (
            np.random.randint(len(MinecraftBlocks.ID2NAME) - 2, size=goal.blocks.shape)
            + 2
        )

        goal.blocks[structure_mask] = block_ids[structure_mask]

        return goal


class SimpleOverhangGoalGenerator(SingleGoalGenerator):
    """
    Generates a structure with an overhang
    """

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        goal = MinecraftBlocks(size)
        goal.blocks[:, 0, :] = MinecraftBlocks.NAME2ID["dirt"]
        goal.blocks[0, :, size[2] // 2] = MinecraftBlocks.NAME2ID["cobblestone"]
        goal.blocks[:, -2, size[2] // 2] = MinecraftBlocks.NAME2ID["cobblestone"]
        goal.blocks[-1:, -3, size[2] // 2] = MinecraftBlocks.NAME2ID["cobblestone"]

        return goal


class TutorialGoalGenerator(SingleGoalGenerator):
    """
    Generates a small house useful as a tutorial for human players.
    """

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        cobblestone = MinecraftBlocks.NAME2ID["cobblestone"]
        planks = MinecraftBlocks.NAME2ID["planks"]
        glass = MinecraftBlocks.NAME2ID["glass"]
        stone = MinecraftBlocks.NAME2ID["stone"]
        log = MinecraftBlocks.NAME2ID["log"]

        goal = MinecraftBlocks(size)

        # Floor made of stone.
        goal.blocks[:, 0, :] = stone

        # Walls made of planks.
        goal.blocks[1:-1, 1, 0] = planks
        goal.blocks[1:-1, 1, -1] = planks
        goal.blocks[0, 1, 1:-1] = planks

        # Corners made of logs.
        goal.blocks[0, 1:3, 0] = log
        goal.blocks[0, 1:3, -1] = log
        goal.blocks[-1, 1:3, 0] = log
        goal.blocks[-1, 1:3, -1] = log

        # Glass windows above the planks.
        goal.blocks[1:-1, 2, 0] = glass
        goal.blocks[1:-1, 2, -1] = glass
        goal.blocks[0, 2, 1:-1] = glass

        # Roof made of coblestone.
        goal.blocks[:, 3, :] = cobblestone

        return goal
