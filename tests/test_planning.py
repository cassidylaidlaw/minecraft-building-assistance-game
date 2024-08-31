import copy

import numpy as np
import pytest

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.agents.heuristic_agents import LowestBlockAgent
from mbag.environment.actions import MbagAction
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.mbag_env import DEFAULT_CONFIG

try:
    from mbag.rllib.alpha_zero.planning import create_mbag_env_model
except ImportError:
    pass


@pytest.mark.uses_rllib
def test_env_model_termination():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["world_size"] = (5, 5, 5)
    config["goal_generator"] = "basic"
    config["horizon"] = 100

    for terminate_on_goal_completion in [True, False]:
        config["terminate_on_goal_completion"] = terminate_on_goal_completion
        env = create_mbag_env_model(config)
        obs, info = env.reset()
        agent = LowestBlockAgent({}, config)
        agent.reset()

        for timestep in range(100):
            action = agent.get_action(obs)
            flat_action = MbagActionDistribution.get_flat_action(config, action)
            obs, reward, terminated, truncated, info = env.step(flat_action)
            if terminated:
                timestep += 1
                break

        assert info["goal_similarity"] == 125
        if terminate_on_goal_completion:
            assert timestep < 100
        else:
            assert timestep == 100


@pytest.mark.uses_rllib
def test_get_all_rewards():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["world_size"] = (5, 5, 5)
    config["goal_generator"] = "basic"

    cobblestone = MinecraftBlocks.NAME2ID["cobblestone"]
    wool = MinecraftBlocks.NAME2ID["wool"]

    env = create_mbag_env_model(config)
    obs, info = env.reset()
    assert isinstance(obs, tuple)
    world_obs, inventory_obs, timestep = obs

    obs_batch = (
        world_obs[None].repeat(2, 0),
        inventory_obs[None].repeat(2, 0),
        timestep[None].repeat(2, 0),
    )
    rewards = env.get_all_rewards(obs_batch)

    assert rewards.shape == (2, MbagActionDistribution.NUM_CHANNELS, 5, 5, 5)
    assert np.all(rewards[0] == rewards[1])
    rewards = rewards[0]
    for action_type in MbagAction.ACTION_TYPES:
        if action_type not in [MbagAction.PLACE_BLOCK, MbagAction.BREAK_BLOCK]:
            action_type_channel = MbagActionDistribution.ACTION_TYPE2CHANNEL[
                action_type
            ]
            assert np.all(rewards[action_type_channel] == 0)

    # Placing cobblestone should give reward.
    assert np.all(
        rewards[MbagActionDistribution.PLACE_BLOCK][cobblestone][1:4, 2, 1:4] == 1
    )
    # But only in the middle...
    assert np.all(
        rewards[MbagActionDistribution.PLACE_BLOCK][cobblestone][0, 2, :] == -1
    )
    # Placing an incorrect block should not give reward.
    assert np.all(rewards[MbagActionDistribution.PLACE_BLOCK][wool][1:4, 2, 1:4] == 0)
    # Breaking one of the center dirt blocks should not give reward, but breaking
    # a different block should give -1 reward.
    assert np.all(rewards[MbagActionDistribution.BREAK_BLOCK][1:4, 1, 1:4] == 0)
    assert np.all(rewards[MbagActionDistribution.BREAK_BLOCK][0, 1, :] == -1)

    # Now test with negative place_wrong reward.
    assert isinstance(config["rewards"], dict)
    config["rewards"]["place_wrong"] = -1
    env = create_mbag_env_model(config)
    rewards = env.get_all_rewards(obs_batch)
    rewards = rewards[0]

    # Now placing an incorrect block should give negative reward.
    assert np.all(rewards[MbagActionDistribution.PLACE_BLOCK][wool][0, 2, :] == -1)
    # And, breaking one of the center dirt blocks should give reward.
    assert np.all(rewards[MbagActionDistribution.BREAK_BLOCK][1:4, 1, 1:4] == 1)
