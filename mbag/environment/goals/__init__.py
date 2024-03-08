from typing import Dict, Type

from .craftassist import CraftAssistGoalGenerator
from .filters import DensityFilter, MinSizeFilter, SingleConnectedComponentFilter
from .goal_generator import GoalGenerator, GoalGeneratorConfig
from .goal_transform import GoalTransform, TransformedGoalGenerator
from .grabcraft import GrabcraftGoalGenerator
from .simple import (
    BasicGoalGenerator,
    RandomGoalGenerator,
    SetGoalGenerator,
    SimpleOverhangGoalGenerator,
    TutorialGoalGenerator,
)
from .transforms import (
    AddGrassTransform,
    AreaSampleTransform,
    CropAirTransform,
    CropLowDensityBottomLayersTransform,
    CropTransform,
    LargestConnectedComponentTransform,
    MirrorTransform,
    RandomlyPlaceTransform,
    RemoveInvisibleNonDirtTransform,
    SeamCarvingTransform,
    UniformBlockTypeTransform,
)

ALL_GOAL_GENERATORS: Dict[str, Type[GoalGenerator]] = {
    "basic": BasicGoalGenerator,
    "random": RandomGoalGenerator,
    "simple_overhang": SimpleOverhangGoalGenerator,
    "tutorial": TutorialGoalGenerator,
    "grabcraft": GrabcraftGoalGenerator,
    "craftassist": CraftAssistGoalGenerator,
    "set_goal": SetGoalGenerator,
}

ALL_GOAL_TRANSFORMS: Dict[str, Type[GoalTransform]] = {
    "single_cc_filter": SingleConnectedComponentFilter,
    "density_filter": DensityFilter,
    "min_size_filter": MinSizeFilter,
    "randomly_place": RandomlyPlaceTransform,
    "add_grass": AddGrassTransform,
    "remove_invisible_non_dirt": RemoveInvisibleNonDirtTransform,
    "largest_cc": LargestConnectedComponentTransform,
    "crop_air": CropAirTransform,
    "crop": CropTransform,
    "crop_low_density_bottom_layers": CropLowDensityBottomLayersTransform,
    "uniform_block_type": UniformBlockTypeTransform,
    "mirror": MirrorTransform,
    "area_sample": AreaSampleTransform,
    "seam_carve": SeamCarvingTransform,
}

__all__ = [
    "ALL_GOAL_GENERATORS",
    "ALL_GOAL_TRANSFORMS",
    "GoalGenerator",
    "GoalGeneratorConfig",
    "GoalTransform",
    "TransformedGoalGenerator",
    "BasicGoalGenerator",
    "RandomGoalGenerator",
    "GrabcraftGoalGenerator",
    "CraftAssistGoalGenerator",
]
