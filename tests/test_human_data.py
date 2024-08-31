import copy
import random
from typing import cast

import numpy as np
import pytest

from mbag.environment.actions import MbagAction
from mbag.environment.mbag_env import MbagConfigDict
from mbag.environment.types import CURRENT_PLAYER, NO_ONE, PLAYER_LOCATIONS

try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
    from ray.rllib.offline import JsonReader
    from ray.rllib.policy.sample_batch import SampleBatch
    from ray.tune.registry import ENV_CREATOR, _global_registry

    from mbag.rllib.human_data import (
        PARTICIPANT_ID,
        convert_episode_to_sample_batch,
        load_episode,
        repair_missing_player_locations,
    )
    from mbag.scripts.convert_human_data_to_rllib import (
        ex as convert_human_data_to_rllib_ex,
    )
except ImportError:
    pass

TUTORIAL_BC_CHECKPOINT = (
    "data/logs/BC/sample_human_models/tutorial/2024-04-10_16-35-41/1/checkpoint_000100"
)


@pytest.mark.uses_rllib
def test_convert_episode_to_sample_batch():
    episode = load_episode(
        "data/human_data/sample_tutorial/participant_1/2023-07-18_15-41-19/1/episode.zip"
    )

    sample_batch_no_noops = convert_episode_to_sample_batch(
        episode,
        player_index=0,
        offset_rewards=True,
        include_noops=False,
    )
    np.testing.assert_array_equal(
        sample_batch_no_noops[SampleBatch.ACTIONS][:7],
        np.array(
            [
                (MbagAction.BREAK_BLOCK, 43, 0),
                (MbagAction.BREAK_BLOCK, 44, 0),
                (MbagAction.MOVE_POS_X, 0, 0),
                (MbagAction.MOVE_POS_Z, 0, 0),
                (MbagAction.MOVE_NEG_Y, 0, 0),
                (MbagAction.MOVE_POS_Z, 0, 0),
                (MbagAction.BREAK_BLOCK, 45, 0),
            ]
        ),
    )
    np.testing.assert_array_equal(
        sample_batch_no_noops[SampleBatch.ACTIONS][-3:],
        np.array(
            [
                (MbagAction.MOVE_POS_X, 0, 0),
                (MbagAction.MOVE_POS_X, 0, 0),
                (MbagAction.PLACE_BLOCK, 133, 3),
            ]
        ),
    )

    sample_batch_with_noops = convert_episode_to_sample_batch(
        episode,
        player_index=0,
        offset_rewards=True,
        include_noops=True,
        action_delay=0.8,
    )
    expected_actions_with_noops = np.array(
        [
            (MbagAction.NOOP, 0, 0),
            (MbagAction.NOOP, 0, 0),
            (MbagAction.NOOP, 0, 0),
            (MbagAction.BREAK_BLOCK, 43, 0),
            (MbagAction.BREAK_BLOCK, 44, 0),
            (MbagAction.MOVE_POS_X, 0, 0),
            (MbagAction.MOVE_POS_Z, 0, 0),
            (MbagAction.MOVE_NEG_Y, 0, 0),
            (MbagAction.NOOP, 0, 0),
            (MbagAction.MOVE_POS_Z, 0, 0),
            (MbagAction.BREAK_BLOCK, 45, 0),
            (MbagAction.MOVE_POS_Z, 0, 0),
            (MbagAction.BREAK_BLOCK, 46, 0),
            (MbagAction.BREAK_BLOCK, 82, 0),
            (MbagAction.MOVE_POS_Z, 0, 0),
            (MbagAction.BREAK_BLOCK, 81, 0),
            (MbagAction.MOVE_POS_X, 0, 0),
            (MbagAction.BREAK_BLOCK, 118, 0),
            (MbagAction.BREAK_BLOCK, 117, 0),
            (MbagAction.MOVE_NEG_Z, 0, 0),
            (MbagAction.BREAK_BLOCK, 80, 0),
            (MbagAction.BREAK_BLOCK, 116, 0),
            (MbagAction.MOVE_NEG_Z, 0, 0),
            (MbagAction.BREAK_BLOCK, 79, 0),
            (MbagAction.BREAK_BLOCK, 115, 0),
            (MbagAction.MOVE_NEG_Z, 0, 0),
            (MbagAction.MOVE_POS_X, 0, 0),
            (MbagAction.MOVE_POS_Z, 0, 0),
            (MbagAction.MOVE_POS_Y, 0, 0),
            (MbagAction.MOVE_POS_Z, 0, 0),
            (MbagAction.MOVE_POS_X, 0, 0),
            (MbagAction.NOOP, 0, 0),
            (MbagAction.NOOP, 0, 0),
            (MbagAction.BREAK_BLOCK, 195, 0),
        ]
    )
    np.testing.assert_array_equal(
        sample_batch_with_noops[SampleBatch.ACTIONS][
            : len(expected_actions_with_noops)
        ],
        expected_actions_with_noops,
    )

    not_noop = sample_batch_with_noops[SampleBatch.ACTIONS][:, 0] != MbagAction.NOOP
    np.testing.assert_array_equal(
        sample_batch_with_noops[SampleBatch.ACTIONS][not_noop],
        sample_batch_no_noops[SampleBatch.ACTIONS],
    )


@pytest.mark.timeout(120)
@pytest.mark.uses_rllib
def test_repair_missing_player_locations():
    episode = load_episode(
        "data/human_data/sample_tutorial/participant_1/2023-07-18_15-41-19/1/episode.zip"
    )
    timesteps_to_remove = random.sample(range(1, episode.length), 30)
    episode_with_missing_player_locations = copy.deepcopy(episode)
    for t in timesteps_to_remove:
        player_locations = episode_with_missing_player_locations.obs_history[t][0][0][
            PLAYER_LOCATIONS
        ]
        player_locations[player_locations == CURRENT_PLAYER] = NO_ONE

    repaired_episode = repair_missing_player_locations(
        episode_with_missing_player_locations
    )
    for t in timesteps_to_remove:
        world_obs, _, _ = repaired_episode.obs_history[t][0]
        expected_world_obs, _, _ = episode.obs_history[t][0]
        np.testing.assert_array_equal(world_obs, expected_world_obs)


@pytest.mark.timeout(120)
@pytest.mark.uses_rllib
def test_convert_human_data_consistency_with_rllib_env(tmp_path):
    for flat_actions, env_id in [
        (False, "MBAG-v1"),
        (True, "MBAGFlatActions-v1"),
    ]:
        for include_noops in [False, True]:
            for place_wrong_reward in [0, -1]:
                out_dir = str(tmp_path / f"rllib_flat_{flat_actions}")
                if include_noops:
                    out_dir += "_with_noops"
                if place_wrong_reward != 0:
                    out_dir += f"_place_wrong_{place_wrong_reward}"
                result = convert_human_data_to_rllib_ex.run(
                    config_updates={
                        "data_dir": "data/human_data/sample_tutorial/participant_1",
                        "flat_actions": flat_actions,
                        "flat_observations": False,
                        "out_dir": str(out_dir),
                        "offset_rewards": True,
                        "place_wrong_reward": place_wrong_reward,
                    }
                ).result
                reader = JsonReader(out_dir)
                episode = reader.next()
                assert isinstance(episode, SampleBatch)

                assert result is not None
                mbag_config = cast(MbagConfigDict, dict(result["mbag_config"]))
                mbag_config["malmo"]["use_malmo"] = False
                mbag_config["goal_generator"] = "tutorial"
                mbag_config["world_size"] = (6, 6, 6)
                mbag_config["random_start_locations"] = False
                assert mbag_config["rewards"]["place_wrong"] == place_wrong_reward
                env: MultiAgentEnv = _global_registry.get(ENV_CREATOR, env_id)(
                    mbag_config
                )

                obs_dict, info_dict = env.reset()
                for t in range(len(episode)):
                    for obs_piece, expected_obs_piece in zip(
                        obs_dict["player_0"][:2], episode[SampleBatch.OBS][t][:2]
                    ):
                        np.testing.assert_array_equal(obs_piece, expected_obs_piece)
                    (
                        obs_dict,
                        reward_dict,
                        terminated_dict,
                        truncated_dict,
                        info_dict,
                    ) = env.step({"player_0": episode[SampleBatch.ACTIONS][t]})
                    assert reward_dict["player_0"] == episode[SampleBatch.REWARDS][t]

                assert terminated_dict["player_0"]
                assert info_dict["player_0"]["goal_similarity"] == 216


@pytest.mark.timeout(120)
@pytest.mark.uses_rllib
def test_convert_human_data_to_rllib_participant_id(tmp_path):
    out_dir = str(tmp_path / "rllib")
    convert_human_data_to_rllib_ex.run(
        config_updates={
            "data_dir": "data/human_data/sample_tutorial",
            "flat_observations": False,
            "out_dir": out_dir,
            "offset_rewards": True,
        }
    )
    reader = JsonReader(out_dir)
    for episode_index in range(4):
        episode = reader.next()

        expected_participant_id = episode_index + 1
        assert np.all(episode[PARTICIPANT_ID] == expected_participant_id)
