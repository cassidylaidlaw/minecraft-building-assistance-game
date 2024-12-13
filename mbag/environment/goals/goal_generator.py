from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional

import numpy as np

from ..blocks import MinecraftBlocks
from ..types import WorldSize

GoalGeneratorConfig = Mapping[str, Any]


class GoalGenerator(ABC):
    default_config: GoalGeneratorConfig = {}
    config: GoalGeneratorConfig
    # Number of remaining goals that can be generated. If None, there is no limit.
    # This number may or may not be decremented as goals are generated, depending on the
    # implementation.
    num_remaining_goals: Optional[int] = None

    def __init__(self, config: GoalGeneratorConfig):
        self.config = dict(self.default_config)
        self.config.update(config)

    @abstractmethod
    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        """
        This should generate a goal of size no larger than the size specified.
        """
        ...

    def _fill_auto_with_real_blocks(self, goal: MinecraftBlocks):
        autos = np.where(goal.blocks == MinecraftBlocks.AUTO)
        coords_list = np.asarray(autos).T
        for coords in coords_list:
            x, y, z = coords[0], coords[1], coords[2]
            goal.blocks[x, y, z] = goal.block_to_nearest_neighbors((x, y, z))
