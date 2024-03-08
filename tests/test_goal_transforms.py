import random

import numpy as np

from mbag.environment import goals
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.goals.filters import DensityFilter, SingleConnectedComponentFilter
from mbag.environment.goals.simple import RandomGoalGenerator, SetGoalGenerator
from mbag.environment.goals.transforms import (
    AddGrassTransform,
    AreaSampleTransform,
    CropAirTransform,
    CropTransform,
    LargestConnectedComponentTransform,
    MirrorTransform,
    RandomlyPlaceTransform,
    UniformBlockTypeTransform,
)


def test_single_cc_filter():
    single_cc = MinecraftBlocks((5, 5, 5))
    single_cc.blocks[:, 0, :] = MinecraftBlocks.NAME2ID["dirt"]
    single_cc.blocks[1:4, 1, 1:4] = MinecraftBlocks.NAME2ID["cobblestone"]

    two_ccs = MinecraftBlocks((5, 5, 5))
    two_ccs.blocks[:, 0, :] = MinecraftBlocks.NAME2ID["dirt"]
    two_ccs.blocks[1:4, 2, 1:4] = MinecraftBlocks.NAME2ID["cobblestone"]

    filter = SingleConnectedComponentFilter(
        {},
        SetGoalGenerator({"goals": [single_cc, two_ccs]}),
    )
    for _ in range(20):
        assert filter.generate_goal((5, 5, 5)) == single_cc


def test_density_filter():
    goals = []
    for fullness in range(6):
        goal = MinecraftBlocks((5, 5, 5))
        goal.blocks[:, :fullness, :] = MinecraftBlocks.NAME2ID["dirt"]
        goals.append(goal)

    filter = DensityFilter(
        {"min_density": 0.2, "max_density": 0.8},
        SetGoalGenerator({"goals": goals}),
    )
    for _ in range(20):
        goal = filter.generate_goal((5, 5, 5))
        assert 1 <= goals.index(goal) <= 4


def test_randomly_place():
    small_goal = MinecraftBlocks((3, 3, 3))
    small_goal.blocks[:] = MinecraftBlocks.NAME2ID["cobblestone"]
    transform = RandomlyPlaceTransform({}, SetGoalGenerator({"goals": [small_goal]}))
    goal = transform.generate_goal((5, 5, 5))
    found_small_goal = False
    for x_offset in range(3):
        for y_offset in range(3):
            if np.all(
                goal.blocks[x_offset : x_offset + 3, :3, y_offset : y_offset + 3]
                == small_goal.blocks
            ):
                found_small_goal = True
    assert found_small_goal


def test_add_grass():
    goal_no_grass = MinecraftBlocks((5, 5, 5))
    goal_no_grass.blocks[1:4, :2, 1:4] = MinecraftBlocks.NAME2ID["cobblestone"]
    transform = AddGrassTransform({}, SetGoalGenerator({"goals": [goal_no_grass]}))
    goal = transform.generate_goal((5, 5, 5))
    assert np.all(goal.blocks[:, 0, :] != MinecraftBlocks.AIR)
    assert np.all(goal.blocks[:, 1:, :] == goal_no_grass.blocks[:, 1:, :])
    assert np.all(
        goal.blocks[goal.blocks != goal_no_grass.blocks]
        == MinecraftBlocks.NAME2ID["dirt"]
    )


def test_remove_invisible_non_dirt():
    goal_with_invisible = MinecraftBlocks((5, 5, 5))
    goal_with_invisible.blocks[:, 0, :] = MinecraftBlocks.NAME2ID["dirt"]
    goal_with_invisible.blocks[1:4, :2, 1] = MinecraftBlocks.NAME2ID["cobblestone"]
    goal_with_invisible.blocks[1:4, 0, 2] = MinecraftBlocks.NAME2ID["cobblestone"]
    goal_with_invisible.blocks[1:4, 0, 3] = MinecraftBlocks.NAME2ID["air"]
    transform = goals.transforms.RemoveInvisibleNonDirtTransform(
        {}, goals.simple.SetGoalGenerator({"goals": [goal_with_invisible]})
    )
    goal = transform.generate_goal((5, 5, 5))
    assert np.all(goal.blocks[1:4, 0, 1] == MinecraftBlocks.NAME2ID["dirt"])
    assert np.all(goal.blocks[1:4, 0, 2] == MinecraftBlocks.NAME2ID["cobblestone"])
    assert np.all(goal.blocks[1:4, 0, 3] == MinecraftBlocks.NAME2ID["air"])


def test_largest_cc():
    single_cc = MinecraftBlocks((5, 5, 5))
    single_cc.blocks[:, 0, :] = MinecraftBlocks.NAME2ID["dirt"]

    two_ccs = MinecraftBlocks((5, 5, 5))
    two_ccs.blocks[:, 0, :] = MinecraftBlocks.NAME2ID["dirt"]
    two_ccs.blocks[1:4, 2, 1:4] = MinecraftBlocks.NAME2ID["cobblestone"]

    filter = LargestConnectedComponentTransform(
        {},
        SetGoalGenerator({"goals": [single_cc, two_ccs]}),
    )
    for _ in range(20):
        assert np.all(filter.generate_goal((5, 5, 5)).blocks == single_cc.blocks)


def test_crop_air():
    small_goal = MinecraftBlocks((3, 3, 3))
    small_goal.blocks[:] = MinecraftBlocks.NAME2ID["cobblestone"]
    transform = CropAirTransform(
        {}, RandomlyPlaceTransform({}, SetGoalGenerator({"goals": [small_goal]}))
    )
    for _ in range(4):
        goal = transform.generate_goal((5, 5, 5))
        assert np.all(goal.blocks == MinecraftBlocks.NAME2ID["cobblestone"])
        assert goal.size == (3, 3, 3)


def test_crop():
    cobblestone = MinecraftBlocks.NAME2ID["cobblestone"]
    big_goal = MinecraftBlocks((5, 5, 5))
    big_goal.blocks[:, 0, :] = cobblestone

    # Test tethered to ground crops.
    transform = CropTransform(
        {"tethered_to_ground": True, "density_threshold": 0.7},
        SetGoalGenerator({"goals": [big_goal]}),
    )
    goal = transform.generate_goal((3, 3, 3))
    print(goal.blocks)
    assert goal.size == (3, 3, 3)
    assert np.all(goal.blocks[:, 0, :] == cobblestone)
    assert np.all(goal.blocks[:, 1:, :] == MinecraftBlocks.AIR)

    # Test untethered crops.
    transform = CropTransform(
        {"tethered_to_ground": False, "density_threshold": 1},
        SetGoalGenerator({"goals": [big_goal]}),
    )
    any_untethered = False
    for _ in range(10):
        goal = transform.generate_goal((3, 3, 3))
        assert goal.size == (3, 3, 3)
        if np.all(goal.blocks == MinecraftBlocks.AIR):
            any_untethered = True
    assert any_untethered

    # Test density threshold.
    big_goal.blocks[:] = MinecraftBlocks.AIR
    big_goal.blocks[:2, :, :] = cobblestone
    transform = CropTransform(
        {"tethered_to_ground": True, "density_threshold": 0.2},
        SetGoalGenerator({"goals": [big_goal]}),
    )
    goal = transform.generate_goal((3, 3, 3))
    assert goal.size == (3, 3, 3)
    assert np.all(goal.blocks[:1, :, :] == cobblestone)
    assert np.all(goal.blocks[1:, :, :] == MinecraftBlocks.AIR)


def test_area_sampling():
    cobblestone = MinecraftBlocks.NAME2ID["cobblestone"]
    big_goal = MinecraftBlocks((7, 7, 7))
    big_goal.blocks[:, 0, :] = cobblestone

    transform = AreaSampleTransform(
        {"preserve_paths": False},
        SetGoalGenerator({"goals": [big_goal]}),
    )
    goal = transform.generate_goal((5, 5, 5))
    print(goal.blocks)
    assert goal.size == (5, 5, 5)
    assert np.all(goal.blocks[:, 0, :] == cobblestone)
    assert np.all(goal.blocks[:, 1:, :] == MinecraftBlocks.AIR)


def test_uniform_block_type():
    block_type = random.choice(list(MinecraftBlocks.PLACEABLE_BLOCK_IDS))
    transform = UniformBlockTypeTransform(
        {"block_type": block_type},
        RandomGoalGenerator({}),
    )
    goal = transform.generate_goal((5, 5, 5))
    assert np.all(goal.blocks[goal.blocks != MinecraftBlocks.AIR] == block_type)


def test_mirror():
    transform = MirrorTransform({}, RandomGoalGenerator({}))
    goal = transform.generate_goal((5, 5, 5))
    assert np.all(goal.blocks[::-1] == goal.blocks)
