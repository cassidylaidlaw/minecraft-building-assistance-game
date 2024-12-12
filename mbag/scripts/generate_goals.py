"""Generate goals for the oracle goal predictor."""

import os
import pathlib
import pickle
from logging import Logger
from typing import Dict, List

import tqdm
from sacred import Experiment

from mbag.agents.oracle_goal_prediction_agent import (
    OracleGoalPredictor,
    get_goal_size_from_world_size,
)
from mbag.environment.goals.craftassist import (
    CraftAssistGoalGenerator,
    DEFAULT_CONFIG as CRAFTASSIST_DEFAULT_CONFIG,
    NoRemainingHouseError,
)
from mbag.environment.goals.goal_transform import TransformedGoalGenerator
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.types import WorldSize
from mbag.rllib.sacred_utils import NoTypeAnnotationsFileStorageObserver


ex = Experiment("oracle_goal_predictor")


@ex.config
def sacred_config() -> None:
    data_dir = CRAFTASSIST_DEFAULT_CONFIG["data_dir"]
    subset = CRAFTASSIST_DEFAULT_CONFIG["subset"]
    repeat = False  # noqa: F841

    world_width = 11  # noqa: F841
    world_height = 10  # noqa: F841
    world_depth = 10  # noqa: F841
    world_size = (world_width, world_height, world_depth)  # noqa: F841

    out_data_dir = data_dir
    out_dir = os.path.join(out_data_dir, f"houses/{subset}/blocks")

    goal_generator_config = {
        "goal_generator": "craftassist",
        "goal_generator_config": {
            "data_dir": data_dir,
            "subset": subset,
            "repeat": repeat,
        },
        "transforms": [
            {"config": {"connectivity": 18}, "transform": "largest_cc"},
            {"transform": "crop_air"},
            {
                "config": {"density_threshold": 0.1},
                "transform": "crop_low_density_bottom_layers",
            },
            {"config": {"min_size": [4, 4, 4]}, "transform": "min_size_filter"},
            {
                "config": {
                    "interpolate": True,
                    "interpolation_order": 1,
                    "max_scaling_factor": 2,
                    "max_scaling_factor_ratio": 1.5,
                    "preserve_paths": True,
                    "scale_y_independently": True,
                },
                "transform": "area_sample",
            },
            {
                "config": {"max_density": 1, "min_density": 0},
                "transform": "density_filter",
            },
        ],
    }

    observer = NoTypeAnnotationsFileStorageObserver(out_dir)
    ex.observers.append(observer)


@ex.automain
def main(
    goal_generator_config: Dict,
    world_size: WorldSize,
    out_dir: str,
    _log: Logger,
) -> None:
    goal_size = get_goal_size_from_world_size(world_size)
    goal_predictor = OracleGoalPredictor(
        goal_generator_config, world_size, goal_size, force_generate_goals=True
    )

    out_dir_path = pathlib.Path(out_dir)
    if not out_dir_path.exists():
        out_dir_path.mkdir(parents=True)
    save_path = out_dir_path / "goals.pkl"

    with open(save_path, "wb") as f:
        pickle.dump(goal_predictor.goals, f)

    _log.info(f"Saved {len(goal_predictor.goals)} goals to {save_path}")
