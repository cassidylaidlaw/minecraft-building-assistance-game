import os
import pathlib
import pickle
from logging import Logger
from typing import List

import tqdm
from sacred import Experiment

from mbag.environment.goals.craftassist import (
    CraftAssistGoalGenerator,
    NoRemainingHouseError,
)
from mbag.environment.goals.goal_transform import TransformedGoalGenerator
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.types import WorldSize
from mbag.rllib.sacred_utils import NoTypeAnnotationsFileStorageObserver


ex = Experiment("oracle_goal_predictor")


def generate_goals(
    data_dir: str, subset: str, repeat: bool, world_size: WorldSize
) -> List[MinecraftBlocks]:
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

    goal_generator = TransformedGoalGenerator(goal_generator_config)

    world_width, world_height, world_depth = world_size
    # The goal width and depth are smaller than the world by 2 to allow players to move
    # around the goal. The goal height is smaller by 2 because the bottom layer is
    # bedrock and the top layer must be air to allow players to stand on the goal.
    goal_size = (
        world_width - 2,
        world_height - 2,
        world_depth - 2,
    )

    goals = []

    craftassist_goal_generator: CraftAssistGoalGenerator = (
        goal_generator.base_goal_generator
    )
    num_remaining_houses = len(craftassist_goal_generator.house_ids)
    with tqdm.tqdm(total=len(craftassist_goal_generator.house_ids)) as pbar:
        while True:
            try:
                goal = goal_generator.generate_goal(goal_size)
                new_num_remaining_houses = len(
                    craftassist_goal_generator.remaining_house_ids
                )
                pbar.update(num_remaining_houses - new_num_remaining_houses)
                num_remaining_houses = new_num_remaining_houses
                goals.append(goal)
            except NoRemainingHouseError:
                break

    return goals


@ex.config
def sacred_config() -> None:
    data_dir = "/nas/ucb/cassidy/minecraft-building-assistance-game/data/craftassist"
    subset = "small200"
    repeat = False  # noqa: F841

    world_width = 11  # noqa: F841
    world_height = 10  # noqa: F841
    world_depth = 10  # noqa: F841

    out_data_dir = data_dir
    out_dir = os.path.join(out_data_dir, f"houses/{subset}/blocks")

    observer = NoTypeAnnotationsFileStorageObserver(out_dir)
    ex.observers.append(observer)


@ex.automain
def main(
    data_dir: str,
    subset: str,
    repeat: bool,
    world_width: int,
    world_height: int,
    world_depth: int,
    out_dir: str,
    _log: Logger,
) -> None:
    world_size = (world_width, world_height, world_depth)
    goals = generate_goals(data_dir, subset, repeat, world_size)

    out_dir_path = pathlib.Path(out_dir)
    if not out_dir_path.exists():
        out_dir_path.mkdir(parents=True)
    save_path = out_dir_path / "goals.pkl"

    with open(save_path, "wb") as f:
        pickle.dump(goals, f)

    _log.info(f"Saved {len(goals)} goals to {save_path}")
