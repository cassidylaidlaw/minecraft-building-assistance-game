"""
Various GoalTransforms which alter a goal.
"""

from typing import Tuple, TypedDict, cast
from typing_extensions import Literal
import random
import numpy as np
import logging

from ..types import WorldSize
from ..blocks import MinecraftBlocks
from .goal_transform import GoalTransform

logger = logging.getLogger(__name__)


class RandomlyPlaceTransform(GoalTransform):
    """
    Given a structure of size smaller than the given size, randomly places the
    structure along the x and z axes (but always keeps it at the same height on
    the y axis).
    """

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        structure = self.goal_generator.generate_goal(size)
        blocks = MinecraftBlocks(size)

        offset_x = random.randint(0, size[0] - structure.size[0])
        offset_z = random.randint(0, size[2] - structure.size[2])
        structure_slice = (
            slice(offset_x, offset_x + structure.size[0]),
            slice(0, structure.size[1]),
            slice(offset_z, offset_z + structure.size[2]),
        )
        blocks.blocks[structure_slice] = structure.blocks
        blocks.block_states[structure_slice] = structure.block_states

        return blocks


AddGrassMode = Literal["surround", "replace", "concatenate"]


class AddGrassTransformConfig(TypedDict):
    mode: AddGrassMode
    """
    Controls how to add grass:
        "surround" means replace all air blocks on the bottom layer with grass.
        "replace" means to replace the bottom layer with grass.
        "concatenate" means generate a 1-block shorter structure and then add a grass
        layer below it.
    """


class AddGrassTransform(GoalTransform):
    """
    Adds grass (dirt) to the bottom layer of a structure where there aren't yet
    blocks.
    """

    default_config: AddGrassTransformConfig = {"mode": "replace"}
    config: AddGrassTransformConfig

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        mode = self.config["mode"]
        if mode == "concatenate":
            width, height, depth = size
            smaller_goal = self.goal_generator.generate_goal((width, height - 1, depth))
            goal = MinecraftBlocks(
                (smaller_goal.size[0], smaller_goal.size[1] + 1, smaller_goal.size[2])
            )
            goal.blocks[:, 1:, :] = smaller_goal.blocks
            goal.block_states[:, 1:, :] = smaller_goal.block_states
            goal.blocks[:, 0, :] = MinecraftBlocks.NAME2ID["grass"]
        elif mode == "replace":
            goal = self.goal_generator.generate_goal(size)
            goal.blocks[:, 0, :] = MinecraftBlocks.NAME2ID["grass"]
        elif mode == "surround":
            goal = self.goal_generator.generate_goal(size)
            bottom_layer = goal.blocks[:, 0, :]
            bottom_layer[
                bottom_layer == MinecraftBlocks.NAME2ID["air"]
            ] = MinecraftBlocks.NAME2ID["grass"]

        return goal


class CropTransformConfig(TypedDict):
    density_threshold: float
    tethered_to_ground: bool
    wall: bool


class CropTransform(GoalTransform):
    """
    Crops large structures down to a smaller size. The crop is chosen such that the
    density difference between the original and cropped structures is no more than
    density_threshold as measured by percentage. For instance, if density_threshold is
    0.25, then the crop density will be 75-125% as dense as the original structure.
    If tethered_to_ground is True, then the crop will be taken from the bottom of the
    structure.
    """

    default_config: CropTransformConfig = {
        "density_threshold": 0.25,
        "tethered_to_ground": True,
        "wall": False,
    }
    config: CropTransformConfig

    def generate_goal(self, size: WorldSize, *, retries: int = 20) -> MinecraftBlocks:
        while True:
            # Generate a goal with effectively no size limits so we can crop it down.
            goal = self.goal_generator.generate_goal((100, 100, 100))
            struct_density = goal.density()

            crop_size = (
                1 if self.config["wall"] else min(size[0], goal.size[0]),
                min(size[1], goal.size[1]),
                min(size[2], goal.size[2]),
            )

            x_range = goal.size[0] - crop_size[0]
            y_range = (
                0 if self.config["tethered_to_ground"] else goal.size[1] - crop_size[1]
            )
            z_range = goal.size[2] - crop_size[2]

            for _ in range(retries):
                crop = MinecraftBlocks(crop_size)
                crop.blocks[:] = MinecraftBlocks.AIR
                crop.block_states[:] = 0

                x, y, z = (
                    random.randint(0, x_range),
                    random.randint(0, y_range),
                    random.randint(0, z_range),
                )
                crop.fill_from_crop(goal, (x, y, z))

                if (
                    abs(crop.density() - struct_density) / struct_density
                    > self.config["density_threshold"]
                ):
                    continue

                return crop


class SeamCarvingTransformConfig(TypedDict):
    position_coefficient: float
    density_coefficient: float


class SeamCarvingTransform(GoalTransform):
    """
    Turns larger goals into smaller ones by removing slices strategically so as
    to maintain the structure of the goal while making it smaller.
    """

    default_config: SeamCarvingTransformConfig = {
        "position_coefficient": 1,
        "density_coefficient": 1,
    }
    config: SeamCarvingTransformConfig

    def _get_relative_positions(self, size: WorldSize) -> np.ndarray:
        return cast(
            np.ndarray,
            np.stack(
                np.meshgrid(*[np.linspace(0, 1, size[axis]) for axis in range(3)]),
                axis=-1,
            ).transpose((1, 0, 2, 3)),
        )

    def _slice(self, axis: int, index: int, arr: np.ndarray) -> np.ndarray:
        return np.delete(arr, [index], axis=axis)

    def _slice_cost(
        self,
        axis: int,
        index: int,
        goal: MinecraftBlocks,
        original_positions: np.ndarray,
    ) -> float:
        density = float(
            np.take(goal.blocks != MinecraftBlocks.AIR, np.array([index]), axis=axis)
            .astype(float)
            .mean()
        )

        position_mse_before = float(
            np.sqrt(
                np.mean(
                    (original_positions - self._get_relative_positions(goal.size)) ** 2
                )
            )
        )
        original_positions_sliced = self._slice(axis, index, original_positions)
        position_mse_after = float(
            np.sqrt(
                np.mean(
                    (
                        original_positions_sliced
                        - self._get_relative_positions(
                            cast(WorldSize, original_positions_sliced.shape)
                        )
                    )
                    ** 2
                )
            )
        )
        position_mse_delta = position_mse_after - position_mse_before

        cost = (
            self.config["density_coefficient"] * density
            + self.config["position_coefficient"] * position_mse_delta
        )
        logger.debug(
            f"axis {axis} slice {index:02d}: "
            f"cost={cost:.2f}\t"
            f"density={density:.2f} position_mse_delta={position_mse_delta:.2f}"
        )
        return cost

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        # Generate a goal with effectively no size limits so we can slice it down.
        goal = self.goal_generator.generate_goal((100, 100, 100))
        original_positions = self._get_relative_positions(goal.size)

        while True:
            too_big_axes = [axis for axis in range(3) if goal.size[axis] > size[axis]]
            if len(too_big_axes) == 0:
                break

            best_slice: Tuple[int, int] = (0, 0)
            best_cost: float = np.inf
            for axis in too_big_axes:
                for index in range(goal.size[axis]):
                    cost = self._slice_cost(axis, index, goal, original_positions)
                    if cost < best_cost:
                        best_cost = cost
                        best_slice = (axis, index)
            best_axis, best_index = best_slice
            goal.blocks = self._slice(best_axis, best_index, goal.blocks)
            goal.block_states = self._slice(best_axis, best_index, goal.block_states)
            goal.size = cast(WorldSize, goal.blocks.shape)
            original_positions = self._slice(best_axis, best_index, original_positions)

        return goal


class UniformBlockTypeTransformConfig(TypedDict):
    block_type: int


class UniformBlockTypeTransform(GoalTransform):
    """
    Modify structure so that every non-air block is the same specified block type.
    """

    default_config: UniformBlockTypeTransformConfig = {
        "block_type": MinecraftBlocks.NAME2ID["grass"],
    }
    config: UniformBlockTypeTransformConfig

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        goal = self.goal_generator.generate_goal(size)
        goal.blocks[goal.blocks != MinecraftBlocks.AIR] = self.config["block_type"]
        return goal


class MirrorTransform(GoalTransform):
    """
    Mirrors a structure so it's symmetric along the X axis.
    """

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        goal = self.goal_generator.generate_goal(size)
        goal.mirror_x_axis()
        return goal
