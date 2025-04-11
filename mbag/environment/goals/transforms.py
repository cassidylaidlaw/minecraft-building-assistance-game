"""
Various GoalTransforms which alter a goal.
"""

import logging
import math
import random
from typing import List, Optional, Tuple, TypedDict, cast

import cc3d
import numpy as np
from scipy import ndimage
from typing_extensions import Literal

from ..blocks import MinecraftBlocks
from ..types import WorldSize
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
            bottom_layer[bottom_layer == MinecraftBlocks.NAME2ID["air"]] = (
                MinecraftBlocks.NAME2ID["grass"]
            )

        return goal


class RemoveInvisibleNonDirtTransform(GoalTransform):
    """
    Meant to be used after AddGrassTransform. Replaces any non-dirt blocks from the
    bottom layer with dirt if they are not visible from the top.
    """

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        goal = self.goal_generator.generate_goal(size)
        bottom_layer = goal.blocks[:, 0, :]
        next_layer = goal.blocks[:, 1, :]
        visible = (next_layer == MinecraftBlocks.AIR) | (
            next_layer == MinecraftBlocks.NAME2ID["glass"]
        )
        bottom_layer[~visible & (bottom_layer != MinecraftBlocks.AIR)] = (
            MinecraftBlocks.NAME2ID["dirt"]
        )
        return goal


class CropLowDensityBottomLayersTransformConfig(TypedDict):
    density_threshold: float


class CropLowDensityBottomLayersTransform(GoalTransform):
    """
    Crop bottom layers with density below the given threshold if the layer above is
    above the threshold.
    """

    default_config: CropLowDensityBottomLayersTransformConfig = {
        "density_threshold": 0.1
    }
    config: CropLowDensityBottomLayersTransformConfig

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        goal = self.goal_generator.generate_goal(size)
        transformed_goal = goal
        width, height, depth = goal.size
        for y in range(height - 1):
            if y >= height - 1:
                continue
            layers_below = goal.blocks[:, : y + 1, :]
            layer_above = goal.blocks[:, y + 1, :]
            density_below = np.mean(layers_below != MinecraftBlocks.AIR, axis=(0, 2))
            density_above = np.mean(layer_above != MinecraftBlocks.AIR)
            if (
                np.all(density_below < self.config["density_threshold"])
                and density_above >= self.config["density_threshold"]
            ):
                transformed_goal = MinecraftBlocks((width, height - y - 1, depth))
                transformed_goal.blocks[:, :, :] = goal.blocks[:, y + 1 :, :]
                transformed_goal.block_states[:, :, :] = goal.block_states[
                    :, y + 1 :, :
                ]
        return transformed_goal


class CropAirTransform(GoalTransform):
    """
    Crops the structure to remove any surrounding air.
    """

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        goal = self.goal_generator.generate_goal(size)
        is_air = goal.blocks == MinecraftBlocks.AIR
        x_air_slices = np.all(is_air, axis=(1, 2))
        x_air_start, x_air_end = x_air_slices.argmin(), x_air_slices[::-1].argmin()
        y_air_slices = np.all(is_air, axis=(0, 2))
        y_air_start, y_air_end = y_air_slices.argmin(), y_air_slices[::-1].argmin()
        z_air_slices = np.all(is_air, axis=(0, 1))
        z_air_start, z_air_end = z_air_slices.argmin(), z_air_slices[::-1].argmin()

        blocks, block_states = goal[
            x_air_start : -x_air_end or None,
            y_air_start : -y_air_end or None,
            z_air_start : -z_air_end or None,
        ]
        cropped_goal = MinecraftBlocks(cast(WorldSize, blocks.shape))
        cropped_goal.blocks[...] = blocks
        cropped_goal.block_states[...] = block_states
        return cropped_goal


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
            if struct_density == 0:
                continue

            crop_size = (
                min(size[0], goal.size[0]),
                min(size[1], goal.size[1]),
                1 if self.config["wall"] else min(size[2], goal.size[2]),
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

            logger.info("CropTransform was unable to find a valid crop")


class LargestConnectedComponentTransformConfig(TypedDict):
    connectivity: Literal[6, 18, 26]


class LargestConnectedComponentTransform(GoalTransform):
    """
    Filters the goal to its largest connected component
    """

    default_config: LargestConnectedComponentTransformConfig = {
        "connectivity": 6,
    }
    config: LargestConnectedComponentTransformConfig

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        goal = self.goal_generator.generate_goal(size)
        width, height, depth = goal.size
        not_air = np.zeros((width, height + 2, depth), dtype=bool)
        not_air[:, 1:-1, :] = goal.blocks != MinecraftBlocks.AIR
        not_air[:, 0, :] = True
        not_air[:, -1, :] = False
        components = cc3d.connected_components(
            not_air, connectivity=self.config["connectivity"]
        )
        assert np.all(components[:, -1, :] == 0)
        components = components[:, 1:-1, :]
        num_components = np.max(components) + 1
        component_counts = [0]
        for component_id in range(1, num_components):
            component_counts.append(np.sum(components == component_id))
        largest_component_id = np.argmax(component_counts)
        goal.blocks[components != largest_component_id] = MinecraftBlocks.AIR
        return goal


class AreaSampleTransformConfig(TypedDict):
    max_scaling_factor: float
    """Maximum factor by which goals should be scaled."""

    interpolate: bool
    """Use interpolation to scale by factors that are not a power of two."""

    interpolation_order: int
    """The spline order used to interpolate, i.e., 0 = nearest neighbor, 1 = bilinear,
    3 = bicubic, etc."""

    scale_y_independently: bool
    """If interpolate is True, then this allows the Y dimension to be scaled
    independently of the X and Z dimensions."""

    max_scaling_factor_ratio: float
    """If scale_y_independently is True, this controls the maximum ratio between the
    scale factors for the Y and X/Z dimensions."""

    preserve_paths: bool
    """If True, will try to preserve walkable paths throughout the structure
    (e.g., doors and stairs)."""


class AreaSampleTransform(GoalTransform):
    default_config: AreaSampleTransformConfig = {
        "max_scaling_factor": 4.0,
        "interpolate": True,
        "interpolation_order": 1,
        "scale_y_independently": True,
        "max_scaling_factor_ratio": 1.5,
        "preserve_paths": True,
    }
    config: AreaSampleTransformConfig

    PATH = 254

    def generate_goal(self, size: WorldSize, *, retries: int = 20) -> MinecraftBlocks:
        structure: Optional[MinecraftBlocks] = None
        while structure is None:
            structure = self.goal_generator.generate_goal((100, 100, 100))
            max_scale_down_size = (
                structure.size[0] / self.config["max_scaling_factor"],
                structure.size[1] / self.config["max_scaling_factor"],
                structure.size[2] / self.config["max_scaling_factor"],
            )

            if (
                max_scale_down_size[0] > size[0]
                or max_scale_down_size[1] > size[1]
                or max_scale_down_size[2] > size[2]
            ):
                structure = None

        return self.scale_down_structure(structure, size)

    def scale_down_structure(
        self, structure: Optional[MinecraftBlocks], size: WorldSize
    ) -> MinecraftBlocks:
        assert structure is not None, "Must pass in a valid structure to scale down"

        logger.info(f"original structure size = {structure.size}")

        if all(structure.size[axis] <= size[axis] for axis in range(3)):
            return structure

        if self.config["preserve_paths"]:
            paths = AreaSampleTransform._find_paths(structure)
            structure.blocks[paths] = AreaSampleTransform.PATH

        if self.config["interpolate"]:
            scaling_iterations = int(
                np.ceil(
                    np.log2(max(structure.size[axis] / size[axis] for axis in range(3)))
                )
            )
            assert scaling_iterations >= 1

            structure = self._interpolate_structure(
                structure,
                size,
                self._get_zoom(structure.size, size, scaling_iterations),
            )

        while (
            structure.size[0] > size[0]
            or structure.size[1] > size[1]
            or structure.size[2] > size[2]
        ):
            scaled_down_structure = MinecraftBlocks(
                (
                    int(math.ceil(0.5 * structure.size[0])),
                    int(math.ceil(0.5 * structure.size[1])),
                    int(math.ceil(0.5 * structure.size[2])),
                )
            )

            chunk_size = (2, 2, 2)
            logger.info("scaling down by 2x")

            idx = [
                (i, j, k)
                for i in range(scaled_down_structure.size[0])
                for j in range(scaled_down_structure.size[1])
                for k in range(scaled_down_structure.size[2])
            ]

            for i, chunk in enumerate(structure.get_chunks(chunk_size)):
                index = idx[i]
                scaled_down_structure.blocks[index] = self._most_common_block(
                    chunk, ignore_air=True
                )

            structure = scaled_down_structure

        self._recreate_paths(structure)

        return structure

    def _most_common_block(self, array: np.ndarray, ignore_air=False) -> int:
        if np.any(array == AreaSampleTransform.PATH):
            return AreaSampleTransform.PATH

        mask = (array != 0) & (array != -1)
        if np.sum(mask) < array[(array != -1)].size / 2 and not ignore_air:
            return 0

        flat_arr = array.flatten()
        filtered_arr = flat_arr[(flat_arr != -1) & (flat_arr != 0)]
        counts = np.bincount(filtered_arr)

        try:
            ties = np.where(counts == counts[np.argmax(counts)])[0]
        except Exception:
            return 0

        return int(max(ties))

    def _zoom_with_ids(
        self,
        input: np.ndarray,
        zoom: Tuple[float, float, float],
        ignore_air=False,
        **kwargs,
    ) -> np.ndarray:
        all_ids = np.sort(np.unique(input))
        assert all_ids[0] == MinecraftBlocks.AIR
        zoomed_per_id_list: List[np.ndarray] = []
        for id_index, id in enumerate(all_ids):
            zoomed_per_id_list.append(
                ndimage.zoom((input == id).astype(float), zoom, **kwargs)
            )
        zoomed_per_id = np.stack(zoomed_per_id_list, axis=0)
        if ignore_air:
            zoomed_per_id[0] = -1
        return cast(np.ndarray, all_ids[zoomed_per_id.argmax(axis=0)])

    def _interpolate_structure(
        self,
        structure: MinecraftBlocks,
        size: WorldSize,
        zoom: Tuple[float, float, float],
        ignore_air: bool = False,
    ) -> MinecraftBlocks:
        blocks = self._zoom_with_ids(
            structure.blocks,
            zoom,
            order=self.config["interpolation_order"],
            ignore_air=ignore_air,
        )
        block_states = self._zoom_with_ids(
            structure.block_states, zoom, order=self.config["interpolation_order"]
        )

        interpolated_structure = MinecraftBlocks(cast(WorldSize, blocks.shape))
        interpolated_structure.blocks[...] = blocks
        interpolated_structure.block_states[...] = block_states
        return interpolated_structure

    def _get_zoom(
        self,
        structure_size: WorldSize,
        target_size: WorldSize,
        scaling_iterations: int,
    ) -> Tuple[float, float, float]:
        total_scaling = 2.0**scaling_iterations

        raw_zoom = [np.nan] * 3
        for axis in range(3):
            raw_zoom[axis] = (
                min(1, target_size[axis] / structure_size[axis]) * total_scaling
            )

        if self.config["scale_y_independently"]:
            xz_zoom = min(raw_zoom[0], raw_zoom[2])
            y_zoom = raw_zoom[1]

            y_zoom = min(y_zoom, xz_zoom * self.config["max_scaling_factor_ratio"])
            xz_zoom = min(xz_zoom, y_zoom * self.config["max_scaling_factor_ratio"])
            return xz_zoom, y_zoom, xz_zoom
        else:
            xyz_zoom = min(raw_zoom)
            return xyz_zoom, xyz_zoom, xyz_zoom

    @staticmethod
    def _find_paths(structure: MinecraftBlocks):
        width, height, depth = structure.size
        pad = 2
        padded_width = width + 2 * pad
        padded_depth = depth + 2 * pad
        world = MinecraftBlocks((padded_width, height + 2, padded_depth))
        world.blocks[pad:-pad, :-2, pad:-pad] = structure.blocks

        can_stand = np.zeros_like(world.blocks, dtype=bool)
        can_stand[:, 1:-1, :] = (
            (world.blocks != MinecraftBlocks.AIR)[:, :-2, :]
            & (world.blocks == MinecraftBlocks.AIR)[:, 1:-1]
            & (world.blocks == MinecraftBlocks.AIR)[:, 2:, :]
        )
        can_stand[:, 2:-1, :] |= (
            (world.blocks != MinecraftBlocks.AIR)[:, :-3, :]
            & (world.blocks == MinecraftBlocks.AIR)[:, 1:-2]
            & (world.blocks == MinecraftBlocks.AIR)[:, 2:-1]
            & (world.blocks == MinecraftBlocks.AIR)[:, 3:, :]
        )
        can_stand[:, :3, :] = (world.blocks == MinecraftBlocks.AIR)[:, :3, :] & (
            world.blocks == MinecraftBlocks.AIR
        )[:, 1:4, :]

        component_ids = cc3d.connected_components(can_stand, connectivity=6)
        ground_component_id = component_ids[0, 1, 0]
        ground_component = (component_ids == ground_component_id)[:, :-2, :]

        paths = ground_component.copy()
        prev_paths = np.zeros_like(paths)
        while np.any(paths != prev_paths):
            prev_paths = paths.copy()
            border = np.zeros_like(paths)
            for x_slice_a, x_slice_b, z_slice_a, z_slice_b in [
                (slice(1, None), slice(None, -1), slice(None), slice(None)),
                (slice(None, -1), slice(1, None), slice(None), slice(None)),
                (slice(None), slice(None), slice(1, None), slice(None, -1)),
                (slice(None), slice(None), slice(None, -1), slice(1, None)),
            ]:
                border_in_dir = np.zeros_like(border)
                border_in_dir[x_slice_a, :, z_slice_a] = ~paths[x_slice_b, :, z_slice_b]
                border_in_dir[x_slice_a, :-1, z_slice_a] &= ~paths[
                    x_slice_b, 1:, z_slice_b
                ]
                border_in_dir[x_slice_a, 1:, z_slice_a] &= ~paths[
                    x_slice_b, :-1, z_slice_b
                ]
                border |= border_in_dir

            border &= paths
            border[0, :, :] = False
            border[-1, :, :] = False
            border[:, :, 0] = False
            border[:, :, -1] = False

            border_locations: List[Tuple[int, int, int]] = list(
                zip(*np.nonzero(border))
            )
            border_locations.sort(key=lambda loc: loc[1], reverse=True)
            for x, y, z in border_locations:
                neighborhood = paths[
                    x - 1 : x + 2, max(0, y - 1) : y + 2, z - 1 : z + 2
                ]
                neighborhood_without = neighborhood.copy()
                neighborhood_without[1, 0 if y == 0 else 1, 1] = False
                if (
                    np.max(
                        cc3d.connected_components(
                            neighborhood_without.any(axis=1), connectivity=4
                        )
                    )
                    == np.max(
                        cc3d.connected_components(
                            neighborhood.any(axis=1), connectivity=4
                        )
                    )
                    and neighborhood_without.sum() > 1
                ):
                    paths[x, y, z] = False

        return paths[pad:-pad, :, pad:-pad]

    def _recreate_paths(self, structure: MinecraftBlocks):
        width, height, depth = structure.size
        PATH = AreaSampleTransform.PATH  # noqa: N806
        AIR = MinecraftBlocks.AIR  # noqa: N806

        path_locations: List[Tuple[int, int, int]] = list(
            zip(*np.nonzero(structure.blocks == PATH))
        )
        for x, y, z in list(path_locations):
            if y == 0:
                structure.blocks[x, y, z] = MinecraftBlocks.AIR
                structure.blocks[x, y + 1, z] = PATH
                path_locations.remove((x, y, z))
                if (x, y + 1, z) not in path_locations:
                    path_locations.append((x, y + 1, z))
        path_locations.sort(key=lambda loc: loc[1])

        prev_structure = structure.copy()

        for x, y, z in path_locations:
            if y >= 1:
                if structure.blocks[x, y - 1, z] == MinecraftBlocks.AIR:
                    structure.blocks[x, y - 1, z] = MinecraftBlocks.AUTO
            structure.blocks[x, y, z] = AIR
            if y + 1 < height:
                structure.blocks[x, y + 1, z] = AIR
                if y + 2 < height and (
                    (x >= 1 and prev_structure.blocks[x - 1, y + 1, z] == PATH)
                    or (
                        x < width - 1 and prev_structure.blocks[x + 1, y + 1, z] == PATH
                    )
                    or (z >= 1 and prev_structure.blocks[x, y + 1, z - 1] == PATH)
                    or (
                        z < depth - 1 and prev_structure.blocks[x, y + 1, z + 1] == PATH
                    )
                ):
                    structure.blocks[x, y + 2, z] = AIR

        self._fill_auto_with_real_blocks(structure)

        return structure


class SeamCarvingTransformConfig(TypedDict):
    position_coefficient: float
    density_coefficient: float
    max_original_size: WorldSize


class SeamCarvingTransform(GoalTransform):
    """
    Turns larger goals into smaller ones by removing slices strategically so as
    to maintain the structure of the goal while making it smaller.
    """

    default_config: SeamCarvingTransformConfig = {
        "position_coefficient": 1,
        "density_coefficient": 1,
        "max_original_size": (100, 100, 100),
    }
    config: SeamCarvingTransformConfig

    def _get_relative_positions(self, size: WorldSize) -> np.ndarray:
        return np.stack(
            np.meshgrid(*[np.linspace(0, 1, size[axis]) for axis in range(3)]),
            axis=-1,
        ).transpose((1, 0, 2, 3))

    def _slice(self, axis: int, index: int, arr: np.ndarray) -> np.ndarray:
        return cast(np.ndarray, np.delete(arr, [index], axis=axis))

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
        goal = self.goal_generator.generate_goal(self.config["max_original_size"])
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
