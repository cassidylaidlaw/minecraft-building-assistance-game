import glob
import json
import logging
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from typing_extensions import Literal, TypedDict

from ..blocks import MinecraftBlocks
from ..types import WorldSize
from .goal_generator import GoalGenerator

logger = logging.getLogger(__name__)


class GrabcraftGoalConfig(TypedDict):
    data_dir: str
    subset: Literal["train", "val", "test"]


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
    }

    config: GrabcraftGoalConfig
    structure_metadata: Dict[str, StructureMetadata]
    block_map: Dict[str, Tuple[str, Optional[str]]]

    def __init__(self, config: dict):
        super().__init__(config)

        self.data_dir = os.path.join(self.config["data_dir"], self.config["subset"])

        self._load_block_map()
        self._load_metadata()
        self.num_remaining_goals = len(self.structure_metadata)

    def _get_generic_block_name(self, grabcraft_block_name: str) -> str:
        """
        Remove any specific GrabCraft paranthetical
        (e.g., (Facing West, Closed, Lower)) from a block name.
        """

        if "(" in grabcraft_block_name:
            return grabcraft_block_name[: grabcraft_block_name.index("(")].rstrip()
        else:
            return grabcraft_block_name

    def _load_block_map(self):
        block_map_fname = os.path.join(
            os.path.dirname(__file__), "grabcraft_block_map.json"
        )
        with open(block_map_fname, "r") as block_map_file:
            self.block_map = json.load(block_map_file)

        # Add entries for generic block names (without variance parantheticals
        # afterwards) if all variants map to the same block type.
        generic_block_groups: Dict[str, List[str]] = defaultdict(list)
        for block_name in self.block_map:
            generic_block_groups[self._get_generic_block_name(block_name)].append(
                block_name
            )
        for generic_block_name, block_names in generic_block_groups.items():
            block_types = {self.block_map[block_name][0] for block_name in block_names}
            if len(block_types) == 1:
                (block_type,) = block_types
                self.block_map[generic_block_name] = (block_type, None)

        limited_block_map_fname = os.path.join(
            os.path.dirname(__file__), "limited_block_map.json"
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
    ) -> Tuple[WorldSize, WorldSize]:
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

        return (min_x, min_y, min_z), (max_x, max_y, max_z)

    def _get_structure_size(self, structure_json: StructureJson) -> WorldSize:
        (min_x, min_y, min_z), (max_x, max_y, max_z) = self._get_structure_bounds(
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
            else:
                goal = structure

        return goal

    def _map_grabcraft_block_name(
        self, grabcraft_block_name: str
    ) -> Optional[Tuple[str, Optional[str]]]:
        block_variant = self.block_map.get(grabcraft_block_name)
        if block_variant is None:
            # Try the block without the parentheses, which usually just provide
            # variant information.
            # TODO: remove this if we start caring about variants
            block_variant = self.block_map.get(
                self._get_generic_block_name(grabcraft_block_name)
            )

        return block_variant

    def _get_structure(self, structure_id: str) -> Optional[MinecraftBlocks]:
        with open(
            os.path.join(self.data_dir, f"{structure_id}.json"), "r"
        ) as structure_file:
            structure_json: StructureJson = json.load(structure_file)

        (min_x, min_y, min_z), (max_x, max_y, max_z) = self._get_structure_bounds(
            structure_json
        )
        structure = MinecraftBlocks(
            (max_x - min_x + 1, max_y - min_y + 1, max_z - min_z + 1)
        )
        structure.blocks[:] = MinecraftBlocks.AIR
        structure.block_states[:] = 0
        for y_str, y_layer in structure_json.items():
            y = int(y_str)
            for x_str, x_layer in y_layer.items():
                x = int(x_str)
                for z_str, block in x_layer.items():
                    z = int(z_str)
                    block_variant = self._map_grabcraft_block_name(block["name"])
                    if block_variant is None:
                        logger.warning(f"no map entry for \"{block['name']}\"")
                        structure.blocks[
                            x - min_x,
                            y - min_y,
                            z - min_z,
                        ] = MinecraftBlocks.AUTO
                    else:
                        block_name, variant_name = block_variant
                        block_id = MinecraftBlocks.NAME2ID.get(block_name)
                        if block_id is not None:
                            structure.blocks[
                                x - min_x,
                                y - min_y,
                                z - min_z,
                            ] = block_id
                        else:
                            return None

        self._fill_auto_with_real_blocks(structure)

        metadata = self.structure_metadata[structure_id]
        logger.info(f"chose structure {structure_id} ({metadata['title']})")

        return structure
