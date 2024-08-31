import copy
import logging
import random
from types import ModuleType
from typing import Optional, cast

import numpy as np
import pytest

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.environment.actions import MbagAction, MbagActionTuple
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.mbag_env import DEFAULT_CONFIG, MbagEnv
from mbag.environment.types import CURRENT_BLOCKS, PLAYER_LOCATIONS

torch: Optional[ModuleType]
try:
    import torch  # noqa: E402
except ImportError:
    torch = None

logger = logging.getLogger(__name__)


def test_mapping():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["world_size"] = (5, 5, 5)

    config["abilities"] = {
        "teleportation": True,
        "flying": True,
        "inf_blocks": True,
    }
    mapping = MbagActionDistribution.get_action_mapping(config)
    assert mapping.shape == (
        1  # NOOP
        + 125 * MinecraftBlocks.NUM_BLOCKS  # PLACE_BLOCK
        + 125,  # BREAK_BLOCK
        3,
    )
    assert mapping[0].tolist() == [MbagAction.NOOP, 0, 0]
    assert mapping[1].tolist() == [MbagAction.PLACE_BLOCK, 0, 0]
    assert mapping[2].tolist() == [MbagAction.PLACE_BLOCK, 1, 0]
    assert mapping[1 + 125].tolist() == [MbagAction.PLACE_BLOCK, 0, 1]
    assert mapping[1 + 125 * MinecraftBlocks.NUM_BLOCKS].tolist() == [
        MbagAction.BREAK_BLOCK,
        0,
        0,
    ]
    assert mapping[2 + 125 * MinecraftBlocks.NUM_BLOCKS].tolist() == [
        MbagAction.BREAK_BLOCK,
        1,
        0,
    ]

    config["abilities"] = {
        "teleportation": False,
        "flying": True,
        "inf_blocks": False,
    }
    mapping_all_abilities = mapping
    mapping = MbagActionDistribution.get_action_mapping(config)
    assert mapping.shape == (
        1  # NOOP
        + 125 * MinecraftBlocks.NUM_BLOCKS  # PLACE_BLOCK
        + 125  # BREAK_BLOCK
        + 6  # movement actions
        + 125 * MinecraftBlocks.NUM_BLOCKS,  # GIVE_BLOCK
        3,
    )
    assert (
        mapping[: mapping_all_abilities.shape[0]].tolist()
        == mapping_all_abilities.tolist()
    )
    assert mapping[1 + 125 * (MinecraftBlocks.NUM_BLOCKS + 1)].tolist() == [
        MbagAction.MOVE_POS_X,
        0,
        0,
    ]
    assert mapping[6 + 125 * (MinecraftBlocks.NUM_BLOCKS + 1)].tolist() == [
        MbagAction.MOVE_NEG_Z,
        0,
        0,
    ]


def test_get_flat_action():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["world_size"] = (5, 5, 5)
    config["abilities"] = {
        "teleportation": False,
        "flying": True,
        "inf_blocks": False,
    }
    mapping = MbagActionDistribution.get_action_mapping(config)
    for flat_action_id, action in enumerate(mapping):
        action_tuple = cast(MbagActionTuple, tuple(action))
        assert (
            MbagActionDistribution.get_flat_action(config, action_tuple)
            == flat_action_id
        )


def test_mask():
    env = MbagEnv(
        {"abilities": {"teleportation": True, "inf_blocks": True, "flying": True}}
    )
    obs_list, _ = env.reset()
    world_obs, inventory_obs, timestep = obs_list[0]
    planks = MinecraftBlocks.NAME2ID["planks"]
    world_obs[CURRENT_BLOCKS, 1, 2, 1] = planks
    obs_batch = world_obs[None], inventory_obs[None], timestep[None]

    mask = MbagActionDistribution.get_mask(env.config, obs_batch)

    # Can't do invalid actions.
    assert np.all(mask[0, MbagActionDistribution.MOVE_POS_X] == False)
    assert np.all(mask[0, MbagActionDistribution.GIVE_BLOCK] == False)

    # Can place a block next to other blocks.
    assert mask[0, MbagActionDistribution.PLACE_BLOCK][planks, 1, 2, 2] == True
    # Can't place a block where there is one.
    assert mask[0, MbagActionDistribution.PLACE_BLOCK][planks, 1, 1, 1] == False
    # Can't place a block floating in midair.
    assert mask[0, MbagActionDistribution.PLACE_BLOCK][planks, 1, 4, 1] == False

    # Can't place bedrock or air.
    assert (
        mask[0, MbagActionDistribution.PLACE_BLOCK][MinecraftBlocks.AIR, 1, 2, 2]
        == False
    )
    assert (
        mask[0, MbagActionDistribution.PLACE_BLOCK][MinecraftBlocks.BEDROCK, 1, 2, 2]
        == False
    )

    # Can't break bedrock or air.
    assert mask[0, MbagActionDistribution.BREAK_BLOCK, 1, 0, 1] == False
    assert mask[0, MbagActionDistribution.BREAK_BLOCK, 1, 2, 2] == False
    # Can break dirt and planks.
    assert mask[0, MbagActionDistribution.BREAK_BLOCK, 1, 1, 1] == True
    assert mask[0, MbagActionDistribution.BREAK_BLOCK, 1, 2, 1] == True


def test_mask_c_extension():
    import _mbag  # noqa: F401

    for inf_blocks in [True, False]:
        for teleportation in [True, False]:
            for num_players in [1, 2]:
                logger.info(
                    f"testing inf_blocks={inf_blocks}, teleportation={teleportation}, "
                    f"num_players={num_players}"
                )
                env = MbagEnv(
                    {
                        "abilities": {
                            "teleportation": teleportation,
                            "inf_blocks": inf_blocks,
                            "flying": True,
                        },
                        "num_players": num_players,
                        "players": [{} for _ in range(num_players)],
                    }
                )
                action_mapping = MbagActionDistribution.get_action_mapping(env.config)
                obs_list, _ = env.reset()
                for t in range(10):
                    actions = []
                    for player_index in range(num_players):
                        world_obs, inventory_obs, timestep = obs_list[player_index]
                        obs_batch = world_obs[None], inventory_obs[None], timestep[None]
                        mask_python = MbagActionDistribution.get_mask(
                            env.config, obs_batch, force_python_impl=True
                        )
                        mask_c = MbagActionDistribution.get_mask(env.config, obs_batch)
                        if not np.all(mask_python == mask_c):
                            logger.error(f"timestep {timestep}")
                            logger.error(np.nonzero(mask_python != mask_c))
                        assert np.all(mask_python == mask_c)

                        flat_mask = MbagActionDistribution.to_flat(
                            env.config, mask_python, np.any
                        )[0]
                        possible_actions = action_mapping[flat_mask]
                        action_type = random.choice(np.unique(possible_actions[:, 0]))
                        action = random.choice(
                            possible_actions[possible_actions[:, 0] == action_type]
                        )
                        actions.append(tuple(action))
                    obs_list, _, _, _ = env.step(actions)


def test_line_of_sight_masking():
    for inf_blocks in [True, False]:
        for teleportation in [True, False]:
            for num_players in [1, 2]:
                logger.info(
                    f"testing inf_blocks={inf_blocks}, teleportation={teleportation}, "
                    f"num_players={num_players}"
                )
                env = MbagEnv(
                    {
                        "abilities": {
                            "teleportation": teleportation,
                            "inf_blocks": inf_blocks,
                            "flying": True,
                        },
                        "num_players": num_players,
                        "players": [{} for _ in range(num_players)],
                    }
                )
                action_mapping = MbagActionDistribution.get_action_mapping(env.config)
                obs_list, _ = env.reset()
                for t in range(10):
                    actions = []
                    for player_index in range(num_players):
                        world_obs, inventory_obs, timestep = obs_list[player_index]
                        obs_batch = world_obs[None], inventory_obs[None], timestep[None]
                        mask = MbagActionDistribution.get_mask(
                            env.config, obs_batch, line_of_sight_masking=True
                        )
                        flat_mask = MbagActionDistribution.to_flat(
                            env.config, mask, np.any
                        )[0]

                        possible_actions = action_mapping[flat_mask]
                        action_type = random.choice(np.unique(possible_actions[:, 0]))
                        action = random.choice(
                            possible_actions[possible_actions[:, 0] == action_type]
                        )
                        actions.append(tuple(action))
                    obs_list, _, _, info_list = env.step(actions)

                    info = info_list[0]
                    # Unintential NOOPs should be impossible with line-of-sight
                    # masking (although only for the first player).
                    assert info["action"] == info["attempted_action"]


def test_mask_no_teleportation_no_inf_blocks():
    env = MbagEnv(
        {
            "world_size": (7, 7, 7),
            "num_players": 2,
            "players": [{}, {}],
            "abilities": {
                "teleportation": False,
                "flying": True,
                "inf_blocks": False,
            },
        }
    )
    obs_list, _ = env.reset()
    world_obs, inventory_obs, timestep = obs_list[0]

    # Suppose the player has dirt in their inventory.
    dirt = MinecraftBlocks.NAME2ID["dirt"]
    planks = MinecraftBlocks.NAME2ID["planks"]
    inventory_obs[0, dirt] += 1

    obs_batch = (
        world_obs[None].repeat(2, 0),
        inventory_obs[None].repeat(2, 0),
        timestep[None].repeat(2, 0),
    )
    mask = MbagActionDistribution.get_mask(env.config, obs_batch)

    # Player 1 should be at (0, 2, 0) and player 2 at (1, 2, 0).
    assert np.all(world_obs[PLAYER_LOCATIONS][0, 2:4, 0] == 1)
    assert np.all(world_obs[PLAYER_LOCATIONS][1, 2:4, 0] == 2)

    # They shouldn't be able to place/break blocks more than 4.5 blocks away.
    assert np.all(
        mask[:, MbagActionDistribution.PLACE_BLOCK][0, dirt, 6:, :, 6:] == False
    )
    assert np.all(mask[0, MbagActionDistribution.BREAK_BLOCK, 6:, :, 6:] == False)

    # Can only place blocks we have.
    assert mask[:, MbagActionDistribution.PLACE_BLOCK][0, dirt, 0, 2, 1] == True
    assert mask[:, MbagActionDistribution.PLACE_BLOCK][0, planks, 0, 2, 1] == False

    # Should only be able to give to a location where a player is.
    assert mask[:, MbagActionDistribution.GIVE_BLOCK][0, dirt, 1, 2, 0] == True
    assert mask[:, MbagActionDistribution.GIVE_BLOCK][0, dirt, 2, 2, 0] == False

    # Can only give blocks we have.
    assert mask[:, MbagActionDistribution.GIVE_BLOCK][0, planks, 1, 2, 0] == False

    world_obs, _, _ = env.step(
        [(MbagAction.NOOP, 0, 0), (MbagAction.MOVE_POS_X, 0, 0)]
    )[0][0]
    obs_batch = world_obs[None], inventory_obs[None], timestep[None]
    mask = MbagActionDistribution.get_mask(env.config, obs_batch)

    # Player 2 should now be at (2, 2, 0).
    assert np.all(world_obs[PLAYER_LOCATIONS][2, 2:4, 0] == 2)
    # Now it should be impossible to give blocks since player 2 is too far away.
    assert np.all(mask[0, MbagAction.GIVE_BLOCK] == False)


def test_to_flat():
    c = MbagActionDistribution.NUM_CHANNELS
    probs = np.ones((1, c, 2, 2, 2))
    probs /= probs.sum()

    config = copy.deepcopy(DEFAULT_CONFIG)
    config["abilities"] = {
        "teleportation": True,
        "flying": True,
        "inf_blocks": True,
    }
    flat = MbagActionDistribution.to_flat(config, probs).flatten().tolist()
    expected_flat = (
        [1 / c]  # NOOP
        + [1 / c / 8] * 8 * MinecraftBlocks.NUM_BLOCKS  # PLACE_BLOCK
        + [1 / c / 8] * 8  # BREAK_BLOCK
    )
    assert flat == pytest.approx(expected_flat)

    if torch is not None:
        flat_torch = (
            MbagActionDistribution.to_flat_torch(config, torch.from_numpy(probs))
            .flatten()
            .tolist()
        )
        flat_torch_logits = (
            MbagActionDistribution.to_flat_torch_logits(
                config, torch.from_numpy(probs).log()
            )
            .exp()
            .flatten()
            .tolist()
        )
        assert flat_torch == pytest.approx(expected_flat)
        assert flat_torch_logits == pytest.approx(expected_flat)

    config["abilities"] = {
        "teleportation": False,
        "flying": True,
        "inf_blocks": False,
    }
    flat = MbagActionDistribution.to_flat(config, probs).flatten().tolist()
    expected_flat = (
        [1 / c]  # NOOP
        + [1 / c / 8] * 8 * MinecraftBlocks.NUM_BLOCKS  # PLACE_BLOCK
        + [1 / c / 8] * 8  # BREAK_BLOCK
        + [1 / c] * 6  # movement actions
        + [1 / c / 8] * 8 * MinecraftBlocks.NUM_BLOCKS  # GIVE_BLOCK
    )
    assert flat == pytest.approx(expected_flat)

    if torch is not None:
        flat_torch = (
            MbagActionDistribution.to_flat_torch(config, torch.from_numpy(probs))
            .flatten()
            .tolist()
        )
        flat_torch_logits = (
            MbagActionDistribution.to_flat_torch_logits(
                config, torch.from_numpy(probs).log()
            )
            .exp()
            .flatten()
            .tolist()
        )
        assert flat_torch == pytest.approx(expected_flat)
        assert flat_torch_logits == pytest.approx(expected_flat)
