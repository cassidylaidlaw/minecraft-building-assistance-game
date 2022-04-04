from typing import List
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.types import MbagAction, WorldLocation


def test_place_break_through_player():
    """
    Make sure we can't place/break blocks through another player.
    """

    blocks = MinecraftBlocks((3, 3, 3))
    dirt = MinecraftBlocks.NAME2ID["dirt"]
    blocks.blocks[:, 0, :] = dirt
    player_location: WorldLocation = (0.5, 1, 1.5)
    assert (
        blocks.try_break_place(
            MbagAction.PLACE_BLOCK,
            (2, 1, 1),
            dirt,
            player_location=player_location,
            update_blocks=False,
        )
        is not None
    )

    other_player_locations: List[WorldLocation] = [(1.5, 1, 1.5)]
    assert (
        blocks.try_break_place(
            MbagAction.PLACE_BLOCK,
            (2, 1, 1),
            dirt,
            player_location=player_location,
            other_player_locations=other_player_locations,
            update_blocks=False,
        )
        is None
    )
    assert (
        blocks.try_break_place(
            MbagAction.BREAK_BLOCK,
            (2, 0, 1),
            player_location=player_location,
            other_player_locations=other_player_locations,
            update_blocks=False,
        )
        is None
    )
