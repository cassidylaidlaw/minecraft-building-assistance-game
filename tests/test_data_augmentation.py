import itertools
import random
from typing import cast

import numpy as np
import pytest

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.environment.actions import MbagActionTuple
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.mbag_env import MbagEnv
from mbag.environment.state import mbag_obs_to_state
from mbag.environment.types import MbagObs

try:
    from ray.rllib.evaluation import SampleBatch

    from mbag.rllib.data_augmentation import randomly_permute_block_types
except ImportError:
    pass


@pytest.mark.uses_rllib
def test_randomly_permute_block_types():
    for (
        teleportation,
        inf_blocks,
        flat_actions,
        flat_observations,
        keep_dirt_at_ground_level,
    ) in itertools.product(
        [False, True],
        [False, True],
        [False, True],
        [False, True],
        [False, True],
    ):
        if keep_dirt_at_ground_level and not inf_blocks:
            continue

        horizon = 100
        episodes = 2
        env = MbagEnv(
            {
                "goal_generator": "random",
                "horizon": horizon,
                "abilities": {
                    "teleportation": teleportation,
                    "flying": True,
                    "inf_blocks": inf_blocks,
                },
                "world_size": (5, 5, 5),
            }
        )
        action_map = MbagActionDistribution.get_action_mapping(env.config)

        all_obs = []
        all_actions = []
        all_rewards = []
        (obs,), _ = env.reset()
        all_obs.append(obs)
        for t in range(horizon):
            obs_batch = obs[0][None], obs[1][None], obs[2][None]
            mask = MbagActionDistribution.get_mask_flat(env.config, obs_batch)[0]
            flat_action = random.choice(mask.nonzero()[0])
            action = cast(MbagActionTuple, tuple(action_map[flat_action]))
            (obs,), rewards, _, _ = env.step([action])
            all_actions.append(flat_action if flat_actions else action)
            all_rewards.append(rewards)
            if t < horizon - 1:
                all_obs.append(obs)

        batch = SampleBatch(
            {
                SampleBatch.REWARDS: np.array(all_rewards * episodes),
                SampleBatch.SEQ_LENS: np.array([horizon] * episodes),
            }
        )
        if flat_observations:
            batch[SampleBatch.OBS] = np.stack(
                [
                    np.concatenate([obs_piece.flat for obs_piece in obs])
                    for obs in all_obs
                ]
                * episodes,
                axis=0,
            )
        else:
            batch[SampleBatch.OBS] = tuple(
                np.array([obs[obs_piece] for obs in all_obs] * episodes)
                for obs_piece in range(3)
            )
        if flat_actions:
            batch[SampleBatch.ACTIONS] = np.array(all_actions * episodes)
        else:
            batch[SampleBatch.ACTIONS] = tuple(
                np.array([action[action_part] for action in all_actions] * episodes)
                for action_part in range(3)
            )

        new_batch = randomly_permute_block_types(
            batch,
            flat_actions=flat_actions,
            flat_observations=flat_observations,
            env_config=env.config,
        )

        # Episode should be permuted so it doesn't match the original
        # (except with low probability).
        assert np.any(new_batch[SampleBatch.OBS][0] != batch[SampleBatch.OBS][0])
        # Each sequence should be independently permuted.
        assert np.any(
            new_batch[SampleBatch.OBS][0][:horizon]
            != batch[SampleBatch.OBS][0][horizon:]
        )
        np.testing.assert_equal(
            new_batch[SampleBatch.REWARDS], batch[SampleBatch.REWARDS]
        )

        for episode_index in range(episodes):
            if flat_observations:
                initial_obs = cast(
                    MbagObs,
                    (
                        new_batch[SampleBatch.OBS][
                            episode_index * horizon, :750
                        ].reshape((6, 5, 5, 5)),
                        new_batch[SampleBatch.OBS][
                            episode_index * horizon,
                            750 : 750 + MinecraftBlocks.NUM_BLOCKS,
                        ].reshape((1, -1)),
                        np.array(0),
                    ),
                )
            else:
                initial_obs = cast(
                    MbagObs,
                    (
                        new_batch[SampleBatch.OBS][0][episode_index * horizon],
                        new_batch[SampleBatch.OBS][1][episode_index * horizon],
                        new_batch[SampleBatch.OBS][2][episode_index * horizon],
                    ),
                )
            (obs,) = env.set_state(mbag_obs_to_state(initial_obs, player_index=0))
            for t in range(horizon):
                i = episode_index * horizon + t
                if flat_observations:
                    np.testing.assert_equal(
                        np.concatenate([obs_piece.flat for obs_piece in obs]),
                        new_batch[SampleBatch.OBS][i],
                    )
                else:
                    np.testing.assert_equal(obs[0], new_batch[SampleBatch.OBS][0][i])
                    np.testing.assert_equal(obs[1], new_batch[SampleBatch.OBS][1][i])
                    np.testing.assert_equal(obs[2], new_batch[SampleBatch.OBS][2][i])
                if flat_actions:
                    action = tuple(action_map[new_batch[SampleBatch.ACTIONS][i]])
                else:
                    action = cast(
                        MbagActionTuple,
                        (
                            new_batch[SampleBatch.ACTIONS][0][i],
                            new_batch[SampleBatch.ACTIONS][1][i],
                            new_batch[SampleBatch.ACTIONS][2][i],
                        ),
                    )
                (obs,), rewards, _, _ = env.step([action])
                assert rewards == new_batch[SampleBatch.REWARDS][i]
