import glob
import os
import json
import random
import logging
import numpy as np
import sys
from scipy import ndimage  # type: ignore
from typing import Dict, List, Optional, Tuple, Any

from typing_extensions import TypedDict, Literal

from ..blocks import MinecraftBlocks
from ..types import WorldSize
from .goal_generator import GoalGenerator

logger = logging.getLogger(__name__)


class GrabcraftGoalConfig(TypedDict):
    data_dir: str
    subset: Literal["train", "val", "test"]
    force_single_cc: bool
    use_limited_block_set: bool


class StructureMetadata(TypedDict):
    id: str
    title: str
    description: str
    category: str
    slug: str
    tags: List[str]
    url: str


class StructureBlock(TypedDict):
    x: str
    y: str
    z: str
    hex: str
    rgb: Tuple[int, int, int]
    name: str
    mat_id: str
    file: str
    transparent: bool
    opacity: float
    texture: str


StructureJson = Dict[str, Dict[str, Dict[str, StructureBlock]]]


class GrabcraftGoalGenerator(GoalGenerator):
    default_config: GrabcraftGoalConfig = {
        "data_dir": "data/grabcraft",
        "subset": "train",
        "force_single_cc": False,
        "use_limited_block_set": True,
    }

    config: GrabcraftGoalConfig
    structure_metadata: Dict[str, StructureMetadata]
    block_map: Dict[str, Tuple[str, Optional[str]]]

    def __init__(self, config: dict):
        super().__init__(config)

        self.data_dir = os.path.join(self.config["data_dir"], self.config["subset"])

        self._load_block_map()
        self._load_metadata()

    def _load_block_map(self):
        block_map_fname = os.path.join(
            os.path.dirname(__file__), "grabcraft_block_map.json"
        )
        with open(block_map_fname, "r") as block_map_file:
            self.block_map = json.load(block_map_file)

        if self.config["use_limited_block_set"]:
            limited_block_map_fname = os.path.join(
                os.path.dirname(__file__), "grabcraft_block_map_limited.json"
            )
            with open(limited_block_map_fname, "r") as block_map_file:
                limited_block_map: Dict[str, str] = json.load(block_map_file)

            for key in self.block_map:
                self.block_map[key] = (
                    limited_block_map[self.block_map[key][0]],
                    self.block_map[key][1],
                )

    def _load_metadata(self):
        self.structure_metadata = {}
        for metadata_fname in glob.glob(os.path.join(self.data_dir, "*.metadata.json")):
            with open(metadata_fname, "r") as metadata_file:
                metadata = json.load(metadata_file)
            structure_id = metadata["id"]
            if not os.path.exists(os.path.join(self.data_dir, f"{structure_id}.json")):
                continue  # Structure file does not exist.

            self.structure_metadata[structure_id] = metadata

    def _get_structure_bounds(
        self, structure_json: StructureJson
    ) -> Tuple[int, int, int, int, int, int]:
        max_x, max_y, max_z = 0, 0, 0
        min_x, min_y, min_z = sys.maxsize, sys.maxsize, sys.maxsize

        for y_str, y_layer in structure_json.items():
            y = int(y_str)
            if y > max_y:
                max_y = y
            if y < min_y:
                min_y = y
            for x_str, x_layer in y_layer.items():
                x = int(x_str)
                if x > max_x:
                    max_x = x
                if x < min_x:
                    min_x = x
                for z_str, block in x_layer.items():
                    z = int(z_str)
                    if z > max_z:
                        max_z = z
                    if z < min_z:
                        min_z = z

        return min_x, max_x, min_y, max_y, min_z, max_z

    def _get_structure_size(self, structure_json: StructureJson) -> WorldSize:
        min_x, max_x, min_y, max_y, min_z, max_z = self._get_structure_bounds(
            structure_json
        )
        return max_x - min_x + 1, max_y - min_y + 1, max_z - min_z + 1

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        success = False
        while not success:
            success = True

            structure_id = random.choice(list(self.structure_metadata.keys()))
            structure = self._get_structure(structure_id)

            # check if structure is valid and within size constraints
            if structure is None or (
                structure.size[0] > size[0]
                or structure.size[1] > size[1]
                or structure.size[2] > size[2]
            ):
                success = False
                continue

            # Randomly place structure within world.
            goal = GoalGenerator.randomly_place_structure(structure, size)

            # If we want to force the structure to be a single connected component,
            # then check here.
            if self.config["force_single_cc"]:
                if not goal.is_single_cc():
                    success = False

            # Add a layer of dirt at the bottom of the structure wherever there's still
            # air.
            bottom_layer = goal.blocks[:, 0, :]
            bottom_layer[bottom_layer == MinecraftBlocks.AIR] = MinecraftBlocks.NAME2ID[
                "dirt"
            ]

        return goal

    def _get_structure(self, structure_id: str) -> Optional[MinecraftBlocks]:
        with open(
            os.path.join(self.data_dir, f"{structure_id}.json"), "r"
        ) as structure_file:
            structure_json: StructureJson = json.load(structure_file)

        bounds = self._get_structure_bounds(structure_json)
        structure_size = (bounds[1] + 1, bounds[3] + 1, bounds[5] + 1)
        structure = MinecraftBlocks(structure_size)
        structure.blocks[:] = MinecraftBlocks.AIR
        structure.block_states[:] = 0
        for y_str, y_layer in structure_json.items():
            y = int(y_str)
            for x_str, x_layer in y_layer.items():
                x = int(x_str)
                for z_str, block in x_layer.items():
                    z = int(z_str)
                    block_variant = self.block_map.get(block["name"])
                    if block_variant is None:
                        logger.warning(f"no map entry for \"{block['name']}\"")
                        structure.blocks[
                            x - 1,
                            y - 1,
                            z - 1,
                        ] = MinecraftBlocks.AUTO
                    else:
                        block_name, variant_name = block_variant
                        block_id = MinecraftBlocks.NAME2ID.get(block_name)
                        if block_id is not None:
                            structure.blocks[
                                x - 1,
                                y - 1,
                                z - 1,
                            ] = block_id
                        else:
                            return None

        if self.config["use_limited_block_set"]:
            self._fill_auto_with_real_blocks(structure)

        metadata = self.structure_metadata[structure_id]
        logger.info(f"chose structure {structure_id} ({metadata['title']})")

        return structure

    @staticmethod
    def _fill_auto_with_real_blocks(structure: MinecraftBlocks) -> None:
        autos = np.where(structure.blocks == MinecraftBlocks.AUTO)
        coords_list = np.asarray(autos).T
        for coords in coords_list:
            x, y, z = coords[0], coords[1], coords[2]
            structure.blocks[x, y, z] = structure.block_to_nearest_neighbors((x, y, z))


class CroppedGrabcraftGoalConfig(GrabcraftGoalConfig):
    tethered_to_ground: bool
    density_threshold: float
    save_crop_dir: str


class CroppedGrabcraftGoalGenerator(GrabcraftGoalGenerator):
    default_config: CroppedGrabcraftGoalConfig = {
        "data_dir": GrabcraftGoalGenerator.default_config["data_dir"],
        "subset": GrabcraftGoalGenerator.default_config["subset"],
        "force_single_cc": GrabcraftGoalGenerator.default_config["force_single_cc"],
        "tethered_to_ground": True,
        "use_limited_block_set": GrabcraftGoalGenerator.default_config[
            "use_limited_block_set"
        ],
        "density_threshold": 0.25,
        "save_crop_dir": GrabcraftGoalGenerator.default_config["subset"],
    }

    config: CroppedGrabcraftGoalConfig

    def _generate_crop(
        self, size: WorldSize, retries: int = 5
    ) -> Tuple[str, MinecraftBlocks, Tuple[int, int, int]]:
        while True:
            structure_id = random.choice(list(self.structure_metadata.keys()))
            structure = self._get_structure(structure_id)
            if structure is None:
                continue
            struct_density = structure.density()

            crop_size = (
                min(size[0], structure.size[0]),
                min(size[1], structure.size[1]),
                min(size[2], structure.size[2]),
            )

            x_range = structure.size[0] - 1
            y_range = 0 if self.config["tethered_to_ground"] else structure.size[1] - 1
            z_range = structure.size[2] - 1

            for _ in range(retries):
                rand_crop = MinecraftBlocks(crop_size)
                rand_crop.blocks[:] = MinecraftBlocks.AIR
                rand_crop.block_states[:] = 0

                x, y, z = (
                    random.randint(0, x_range),
                    random.randint(0, y_range),
                    random.randint(0, z_range),
                )
                rand_crop.fill_from_crop(structure, (x, y, z))

                if (
                    abs(rand_crop.density() - struct_density) / struct_density
                    > self.config["density_threshold"]
                ):
                    continue

                if self.config["force_single_cc"] and not rand_crop.is_single_cc():
                    continue

                return structure_id, rand_crop, (x, y, z)

    def generate_goal(
        self, size: WorldSize, save_crop: bool = False
    ) -> MinecraftBlocks:
        structure_id, crop, location = self._generate_crop(size)

        # Randomly place structure within world.
        goal = GoalGenerator.randomly_place_structure(crop, size)

        # Add a layer of dirt at the bottom of the structure wherever there's still
        # air.
        bottom_layer = goal.blocks[:, 0, :]
        bottom_layer[bottom_layer == MinecraftBlocks.AIR] = MinecraftBlocks.NAME2ID[
            "dirt"
        ]

        if save_crop:
            self.save_crop_as_json(structure_id, crop.size, location)

        return goal

    def save_crop_as_json(
        self, structure_id: str, crop_size: WorldSize, location: Tuple[int, int, int]
    ) -> None:
        assert self.config["save_crop_dir"], "No save directory initialized!"

        with open(
            os.path.join(self.data_dir, f"{structure_id}.json"), "r"
        ) as structure_file:
            structure_json: StructureJson = json.load(structure_file)
        x_start, y_start, z_start = location
        crop_json: StructureJson = dict()

        for x in range(x_start, x_start + crop_size[0]):
            for y in range(y_start, y_start + crop_size[1]):
                for z in range(z_start, z_start + crop_size[2]):
                    str_x, str_y, str_z = str(x + 1), str(y + 1), str(z + 1)
                    if (
                        str_y in structure_json
                        and str_x in structure_json[str_y]
                        and str_z in structure_json[str_y][str_x]
                    ):
                        real_x, real_y, real_z = (
                            str(x + 1 - x_start),
                            str(y + 1 - y_start),
                            str(z + 1 - z_start),
                        )
                        if real_y not in crop_json:
                            crop_json[real_y] = dict()
                        if real_x not in crop_json[real_y]:
                            crop_json[real_y][real_x] = dict()
                        if real_z not in crop_json[real_y][real_x]:
                            crop_json[real_y][real_x][real_z] = structure_json[str_y][
                                str_x
                            ][str_z]
                        else:
                            crop_json[real_y][real_x][real_z] = structure_json[str_y][
                                str_x
                            ][str_z]

        crop_json_str = json.dumps(crop_json)
        self.structure_metadata[structure_id]["id"] = structure_id + "_crop"
        metadata_json_str = json.dumps(self.structure_metadata[structure_id])

        save_dir = os.path.join(self.config["data_dir"], self.config["save_crop_dir"])
        with open(os.path.join(save_dir, str(structure_id) + "_crop.json"), "w+") as f:
            f.write(crop_json_str)

        with open(
            os.path.join(save_dir, str(structure_id) + "_crop.metadata.json"), "w+"
        ) as f:
            f.write(metadata_json_str)


class SeamCarvingGrabcraftGoalConfig(GrabcraftGoalConfig):
    density_cost_weight: float


class SeamCarvingGrabcraftGoalGenerator(GrabcraftGoalGenerator):
    config: SeamCarvingGrabcraftGoalConfig

    default_config: SeamCarvingGrabcraftGoalConfig = {
        "data_dir": GrabcraftGoalGenerator.default_config["data_dir"],
        "subset": GrabcraftGoalGenerator.default_config["subset"],
        "force_single_cc": GrabcraftGoalGenerator.default_config["force_single_cc"],
        "use_limited_block_set": GrabcraftGoalGenerator.default_config[
            "use_limited_block_set"
        ],
        "density_cost_weight": 0.5,
    }

    @staticmethod
    def _calculate_conductivities(blocks: Any) -> Any:
        padded_blocks = np.pad(
            blocks,
            pad_width=[
                (1, 1),
                (1, 1),
                (1, 1),
            ],
            mode="constant",
            constant_values=MinecraftBlocks.AIR,
        )

        padded_blocks[padded_blocks != MinecraftBlocks.AIR] = 1

        padded_conductivites = ndimage.convolve(
            padded_blocks,
            np.ones_like(
                np.array(
                    [
                        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    ]
                )
            ),
        )

        return padded_conductivites[1:-1, 1:-1, 1:-1]

    @staticmethod
    def _calculate_conductivity_diffs(
        original_conductivity: np.ndarray,
        cropped_blocks: np.ndarray,
        axis: int,
        index: int,
    ) -> float:
        assert axis in [0, 1, 2]

        cropped_conductivity = (
            SeamCarvingGrabcraftGoalGenerator._calculate_conductivities(cropped_blocks)
        )

        original_conductivity = np.delete(original_conductivity, index, axis=axis)

        diff = original_conductivity - cropped_conductivity
        return float(np.sum(np.abs(diff)))

    def _min_cost_cut_across_axis(
        self, blocks: np.ndarray, original_conductivity: np.ndarray, axis: int
    ) -> Tuple[int, float]:
        cut_size: WorldSize = (1, blocks.shape[1], blocks.shape[2])
        if axis == 0:
            cut_size = (1, blocks.shape[1], blocks.shape[2])
        if axis == 1:
            cut_size = (blocks.shape[0], 1, blocks.shape[2])
        if axis == 2:
            cut_size = (blocks.shape[0], blocks.shape[1], 1)

        min_cut_idx: int = -1
        min_cost: float = np.inf
        cut = MinecraftBlocks(cut_size)
        for i in range(blocks.shape[axis]):
            if axis == 0:
                cut.blocks = blocks[i, :, :]
            if axis == 1:
                cut.blocks = blocks[:, i, :]
            if axis == 2:
                cut.blocks = blocks[:, :, i]

            remaining_blocks = np.delete(blocks, i, axis)
            conductivity_diff = (
                SeamCarvingGrabcraftGoalGenerator._calculate_conductivity_diffs(
                    original_conductivity, remaining_blocks, axis, i
                )
                / blocks.size
            )
            cut_cost = (
                self.config["density_cost_weight"] * cut.density()
                + (1 - self.config["density_cost_weight"]) * conductivity_diff
            )
            if cut_cost < min_cost:
                min_cost = cut_cost
                min_cut_idx = i

        if logger.isEnabledFor(logging.DEBUG):
            conductivity_diff = (
                SeamCarvingGrabcraftGoalGenerator._calculate_conductivity_diffs(
                    original_conductivity,
                    np.delete(blocks, min_cut_idx, axis),
                    axis,
                    min_cut_idx,
                )
                / blocks.size
            )
            if axis == 0:
                cut.blocks = blocks[min_cut_idx, :, :]
            if axis == 1:
                cut.blocks = blocks[:, min_cut_idx, :]
            if axis == 2:
                cut.blocks = blocks[:, :, min_cut_idx]
            logger.debug(
                f"best cut for {'XYZ'[axis]} axis is {min_cut_idx} "
                f"(density = {cut.density()}, conductivity = {conductivity_diff})"
            )

        return min_cut_idx, min_cost

    def generate_goal(self, size: WorldSize) -> MinecraftBlocks:
        structure_id = random.choice(list(self.structure_metadata.keys()))
        structure = self._get_structure(structure_id)
        assert structure is not None

        curr_blocks = np.copy(structure.blocks)
        original_conductivity = self._calculate_conductivities(curr_blocks)
        while (
            curr_blocks.shape[0] > size[0]
            or curr_blocks.shape[1] > size[1]
            or curr_blocks.shape[2] > size[2]
        ):
            cut_costs = []
            cut_locations = []
            cut_axes = []
            for axis in [0, 1, 2]:
                if curr_blocks.shape[axis] > size[axis]:
                    cut, cost = self._min_cost_cut_across_axis(
                        curr_blocks, original_conductivity, axis
                    )
                    cut_locations.append(cut)
                    cut_axes.append(axis)
                    cut_costs.append(cost)

            best_index = np.argmin(cut_costs)
            curr_blocks = np.delete(
                curr_blocks, cut_locations[best_index], cut_axes[best_index]
            )
            original_conductivity = np.delete(
                original_conductivity, cut_locations[best_index], cut_axes[best_index]
            )
            logger.debug(
                "chose to cut "
                f"{'XYZ'[cut_axes[best_index]]} = {cut_locations[best_index]} "
                f"(cost = {cut_costs[best_index]})"
            )
            assert curr_blocks.shape == original_conductivity.shape

        # Randomly place structure within world.
        carved_struct_size: Tuple[int, int, int] = (
            curr_blocks.shape[0],
            curr_blocks.shape[1],
            curr_blocks.shape[2],
        )
        carved_struct = MinecraftBlocks(carved_struct_size)
        carved_struct.blocks = curr_blocks
        goal = GoalGenerator.randomly_place_structure(carved_struct, size)

        # Add a layer of dirt at the bottom of the structure wherever there's still
        # air.
        bottom_layer = goal.blocks[:, 0, :]
        bottom_layer[bottom_layer == MinecraftBlocks.AIR] = MinecraftBlocks.NAME2ID[
            "dirt"
        ]

        return goal
