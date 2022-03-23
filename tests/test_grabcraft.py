import glob
import json
import os
import tempfile

import pytest
from mbag.environment.types import WorldSize

from mbag.evaluation.evaluator import MbagEvaluator
from mbag.agents.heuristic_agents import PriorityQueueAgent
from mbag.environment.goals.grabcraft import (
    GrabcraftGoalGenerator,
    CroppedGrabcraftGoalGenerator,
    SeamCarvingGrabcraftGoalGenerator,
)


def test_goal_generator():
    evaluator = MbagEvaluator(
        {
            "world_size": (20, 20, 20),
            "num_players": 1,
            "horizon": 500,
            "goal_generator": GrabcraftGoalGenerator,
            "goal_generator_config": {
                "data_dir": "data/grabcraft",
                "subset": "train",
            },
            "goal_visibility": [True],
            "malmo": {
                "use_malmo": False,
                "use_spectator": False,
                "video_dir": None,
            },
        },
        [
            (
                PriorityQueueAgent,
                {},
            )
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward > 0


@pytest.mark.xfail(strict=False)
def test_goal_generator_in_malmo():
    evaluator = MbagEvaluator(
        {
            "world_size": (20, 20, 20),
            "num_players": 1,
            "horizon": 1000,
            "goal_generator": GrabcraftGoalGenerator,
            "goal_generator_config": {"data_dir": "data/grabcraft", "subset": "train"},
            "goal_visibility": [True],
            "malmo": {
                "use_malmo": True,
                "use_spectator": False,
                "video_dir": None,
            },
        },
        [
            (
                PriorityQueueAgent,
                {},
            ),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward > 0


def test_crop_generator():
    config = {"data_dir": "data/grabcraft", "subset": "train"}
    world_size: WorldSize = (15, 10, 15)

    generator = CroppedGrabcraftGoalGenerator(config)
    goal = generator.generate_goal(world_size)
    assert goal.size == world_size

    generator = CroppedGrabcraftGoalGenerator(
        {**config, "force_single_cc": True, "use_limited_block_set": True}
    )
    goal = generator.generate_goal(world_size)
    assert goal.is_single_cc()

    evaluator = MbagEvaluator(
        {
            "world_size": (15, 10, 15),
            "num_players": 1,
            "horizon": 1000,
            "goal_generator": CroppedGrabcraftGoalGenerator,
            "goal_generator_config": config,
            "goal_visibility": [True],
            "malmo": {
                "use_malmo": False,
                "use_spectator": False,
                "video_dir": None,
            },
        },
        [
            (
                PriorityQueueAgent,
                {},
            ),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward > 0


@pytest.mark.xfail(strict=False)
def test_crop_generator_in_malmo():
    evaluator = MbagEvaluator(
        {
            "world_size": (15, 10, 15),
            "num_players": 1,
            "horizon": 1000,
            "goal_generator": CroppedGrabcraftGoalGenerator,
            "goal_generator_config": {
                "data_dir": "data/grabcraft",
                "subset": "train",
                "use_limited_block_set": True,
            },
            "goal_visibility": [True],
            "malmo": {
                "use_malmo": True,
                "use_spectator": False,
                "video_dir": None,
            },
        },
        [
            (
                PriorityQueueAgent,
                {},
            ),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward > 0


def test_generate_crop_json():
    data_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(data_dir, "hardcoded_crops"))
    os.symlink(
        os.path.abspath("data/grabcraft/train"),
        os.path.join(data_dir, "train"),
    )

    generator = CroppedGrabcraftGoalGenerator(
        {
            "data_dir": data_dir,
            "subset": "train",
            "force_single_cc": True,
            "use_limited_block_set": True,
            "save_crop_dir": "hardcoded_crops",
        }
    )

    for _ in range(10):
        structure = generator.generate_goal((8, 9, 8), save_crop=True)
        assert (
            structure.size[0] <= 8 and structure.size[1] <= 9 and structure.size[2] <= 8
        )

    for fname in glob.glob(os.path.join(data_dir, "hardcoded_crops", "*_crop.json")):
        with open(fname) as f:
            structure_json = json.load(f)

        size = generator._get_structure_size(structure_json)
        print(size)
        assert size[0] <= 8 and size[1] <= 9 and size[2] <= 8

    evaluator = MbagEvaluator(
        {
            "world_size": (10, 10, 10),
            "num_players": 1,
            "horizon": 1000,
            "goal_generator": GrabcraftGoalGenerator,
            "goal_generator_config": {
                "data_dir": data_dir,
                "subset": "hardcoded_crops",
                "use_limited_block_set": True,
            },
            "goal_visibility": [True],
            "malmo": {
                "use_malmo": False,
                "use_spectator": False,
                "video_dir": None,
            },
        },
        [
            (
                PriorityQueueAgent,
                {},
            ),
        ],
    )

    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward > 0


def test_carving_generator():
    goal_generator = SeamCarvingGrabcraftGoalGenerator(
        {
            "subset": "train",
            "use_limited_block_set": True,
        }
    )
    size: WorldSize
    for size in [(5, 5, 5), (8, 8, 8), (10, 10, 10)]:
        blocks = goal_generator.generate_goal(size)
        assert blocks.size == size


@pytest.mark.xfail(strict=False)
def test_carving_generator_in_malmo():
    evaluator = MbagEvaluator(
        {
            "world_size": (15, 10, 15),
            "num_players": 1,
            "horizon": 1000,
            "goal_generator": SeamCarvingGrabcraftGoalGenerator,
            "goal_generator_config": {
                "data_dir": "data/grabcraft",
                "subset": "train",
                "use_limited_block_set": True,
            },
            "goal_visibility": [True],
            "malmo": {
                "use_malmo": True,
                "use_spectator": False,
                "video_dir": None,
            },
        },
        [
            (
                PriorityQueueAgent,
                {},
            ),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward > 0
