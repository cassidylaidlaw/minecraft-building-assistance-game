import numpy as np
import itertools

from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.goals.grabcraft import CroppedGrabcraftGoalGenerator


def test_not_same():
    blocks = MinecraftBlocks((3, 3, 3))
    blocks.blocks.flat[:] = np.arange(27)
    blocks.blocks[1, 1, 1] = 27
    # Shouldn't produce the same block as was there before, because we want to replace
    # it.
    assert blocks.block_to_nearest_neighbors((1, 1, 1)) != blocks.blocks[1, 1, 1]


def test_air():
    blocks = MinecraftBlocks((3, 3, 3))
    cobble = MinecraftBlocks.NAME2ID["cobblestone"]

    # A block surrounded by air should return air.
    blocks.blocks[:] = MinecraftBlocks.AIR
    assert blocks.block_to_nearest_neighbors((1, 1, 1)) == MinecraftBlocks.AIR
    blocks.blocks[1, 1, 1] = cobble
    assert blocks.block_to_nearest_neighbors((1, 1, 1)) == MinecraftBlocks.AIR

    # ...but one with a non-air block surrounding should turn to that block.
    blocks.blocks[:] = MinecraftBlocks.AIR
    blocks.blocks[0, 1, 0] = cobble
    assert blocks.block_to_nearest_neighbors((1, 1, 1)) == cobble


def test_majority():
    blocks = MinecraftBlocks((3, 3, 3))
    dirt = MinecraftBlocks.NAME2ID["dirt"]
    cobble = MinecraftBlocks.NAME2ID["cobblestone"]
    blocks.blocks[:] = MinecraftBlocks.AIR
    blocks.blocks[:, 0, :] = dirt
    blocks.blocks[:, 2, :] = cobble
    blocks.blocks[2, 1, 1] = cobble
    assert blocks.block_to_nearest_neighbors((1, 1, 1)) == cobble


def test_large():
    blocks = CroppedGrabcraftGoalGenerator({}).generate_goal((10, 10, 10))
    blocks.blocks[4:6, 4, 5] = MinecraftBlocks.AUTO
    blocks.block_to_nearest_neighbors((4, 4, 5))


def test_many_auto_blocks():
    blocks = CroppedGrabcraftGoalGenerator({}).generate_goal((10, 10, 10))
    blocks.blocks[4:7, 4:7, 4:7] = MinecraftBlocks.AUTO
    blocks.block_to_nearest_neighbors((5, 5, 5))


def test_edges():
    blocks = MinecraftBlocks((3, 3, 3))
    dirt = MinecraftBlocks.NAME2ID["dirt"]
    cobble = MinecraftBlocks.NAME2ID["cobblestone"]

    for block_position in itertools.product([0, 2], [0, 2], [0, 2]):
        blocks.blocks[:] = dirt
        x, y, z = block_position
        blocks.blocks[
            max(x - 1, 0) : min(x + 2, 3),
            max(y - 1, 0) : min(y + 2, 3),
            max(z - 1, 0) : min(z + 2, 3),
        ] = cobble
        blocks.blocks[block_position] = MinecraftBlocks.AIR
        assert blocks.block_to_nearest_neighbors(block_position) == cobble
