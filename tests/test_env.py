import copy
import random
from typing import cast

import numpy as np

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.agents.heuristic_agents import LowestBlockAgent
from mbag.environment.actions import MbagAction, MbagActionTuple
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.goals import TransformedGoalGenerator
from mbag.environment.mbag_env import DEFAULT_CONFIG, MbagEnv
from mbag.environment.types import CURRENT_BLOCKS
from mbag.evaluation.evaluator import MbagEvaluator


def _convert_state(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, list):
        return [_convert_state(el) for el in data]
    elif isinstance(data, tuple):
        return tuple(_convert_state(el) for el in data)
    elif isinstance(data, dict):
        return {key: _convert_state(value) for key, value in data.items()}
    elif hasattr(data, "__dict__"):
        return _convert_state(data.__dict__)
    else:
        return data


def test_config_not_changed():
    """The env config dictionary passed should not be modified by the environment."""

    config = copy.deepcopy(DEFAULT_CONFIG)
    MbagEnv(config)
    assert config == DEFAULT_CONFIG


def test_deterministic():
    for teleportation in [False, True]:
        for inf_blocks in [False, True]:
            horizon = 100
            env_a = MbagEnv(
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
            (obs,), _ = env_a.reset()
            env_b = MbagEnv(env_a.config)
            env_b.reset()
            action_map = MbagActionDistribution.get_action_mapping(env_a.config)

            for t in range(horizon):
                env_b.set_state(env_a.get_state())
                obs_batch = obs[0][None], obs[1][None], obs[2][None]
                mask = MbagActionDistribution.get_mask_flat(env_a.config, obs_batch)[0]
                flat_action = random.choice(mask.nonzero()[0])
                action = cast(MbagActionTuple, tuple(action_map[flat_action]))
                (obs,), rewards_a, _, _ = env_a.step([action])
                state_after_a = env_a.get_state()
                (obs,), rewards_b, _, _ = env_b.step([action])
                state_after_b = env_b.get_state()
                assert _convert_state(state_after_a) == _convert_state(state_after_b)
                assert rewards_a == rewards_b


def test_goal_similarity_and_goal_percentage():
    for place_wrong_reward in [0, -1]:
        num_players = 1
        for goal_generator in ["basic", "simple_overhang", "grabcraft"]:
            if goal_generator != "grabcraft":
                transforms = []
            else:
                transforms = [
                    {
                        "transform": "crop",
                        "config": {"tethered_to_ground": True},
                        "density_threshold": 0.5,
                    },
                    {"transform": "single_cc_filter"},
                    {"transform": "randomly_place"},
                    {"transform": "add_grass"},
                ]
            evaluator = MbagEvaluator(
                {
                    "world_size": (6, 6, 6),
                    "num_players": num_players,
                    "horizon": 100,
                    "goal_generator": TransformedGoalGenerator,
                    "goal_generator_config": {
                        "goal_generator": goal_generator,
                        "goal_generator_config": {},
                        "transforms": transforms,
                    },
                    "players": [{}] * num_players,
                    "malmo": {
                        "use_malmo": False,
                        "use_spectator": False,
                        "video_dir": None,
                    },
                    "rewards": {
                        "place_wrong": place_wrong_reward,
                    },
                },
                [
                    (LowestBlockAgent, {}),
                ]
                * num_players,
            )

            _, initial_infos = evaluator.env.reset()
            episode = evaluator.rollout()

            assert initial_infos[0]["goal_percentage"] == 0
            assert episode.info_history[-1][0]["goal_similarity"] == 6 * 6 * 6
            assert episode.info_history[-1][0]["goal_percentage"] == 1


def test_truncate_on_no_progress():
    for truncate in [False, True]:
        env = MbagEnv(
            {
                "goal_generator": "basic",
                "horizon": 100,
                "abilities": {
                    "teleportation": True,
                    "flying": True,
                    "inf_blocks": True,
                },
                "rewards": {
                    "place_wrong": -1,
                },
                "world_size": (5, 5, 5),
                "truncate_on_no_progress_timesteps": 10 if truncate else None,
            }
        )

        break_action: MbagActionTuple = (
            MbagAction.BREAK_BLOCK,
            int(
                np.ravel_multi_index(
                    (2, 1, 2),
                    env.config["world_size"],
                )
            ),
            0,
        )
        place_action: MbagActionTuple = (
            MbagAction.PLACE_BLOCK,
            int(
                np.ravel_multi_index(
                    (2, 1, 2),
                    env.config["world_size"],
                )
            ),
            MinecraftBlocks.NAME2ID["cobblestone"],
        )

        _, initial_infos = env.reset()

        for _ in range(9):
            _, _, (done,), _ = env.step([(MbagAction.NOOP, 0, 0)])
            assert not done
        _, _, (done,), _ = env.step([break_action])
        assert not done

        for _ in range(9):
            _, _, (done,), _ = env.step([(MbagAction.NOOP, 0, 0)])
            assert not done
        _, _, (done,), _ = env.step([place_action])
        assert not done

        for _ in range(8):
            _, _, (done,), _ = env.step([(MbagAction.NOOP, 0, 0)])
            assert not done
        _, _, (done,), _ = env.step([break_action])
        assert not done
        _, _, (done,), _ = env.step([place_action])
        if truncate:
            assert done
        else:
            assert not done


def test_incorrect_action_reward():
    for incorrect_action_reward in [0, -0.5]:
        env = MbagEnv(
            {
                "goal_generator": "basic",
                "horizon": 100,
                "abilities": {
                    "teleportation": True,
                    "flying": True,
                    "inf_blocks": False,
                },
                "rewards": {
                    "place_wrong": -1,
                    "incorrect_action": incorrect_action_reward,
                },
                "world_size": (5, 5, 5),
            }
        )
        (obs,), (info,) = env.reset()
        world_obs, _, _ = obs

        # Incorrect break action.
        x, y, z = 2, 1, 0
        action = (
            MbagAction.BREAK_BLOCK,
            int(np.ravel_multi_index((x, y, z), env.config["world_size"])),
            0,
        )
        _, (reward,), _, (info,) = env.step([action])
        assert reward == -1 + incorrect_action_reward
        assert not info["action_correct"]

        # Correct break action.
        x, y, z = 2, 1, 2
        action = (
            MbagAction.BREAK_BLOCK,
            int(np.ravel_multi_index((x, y, z), env.config["world_size"])),
            0,
        )
        _, (reward,), _, (info,) = env.step([action])
        assert reward == 1
        assert info["action_correct"]

        # Palette action.
        cobblestone = MinecraftBlocks.NAME2ID["cobblestone"]
        palette_blocks = world_obs[CURRENT_BLOCKS, env.palette_x, 2, :]
        x, y, z = env.palette_x, 2, np.nonzero(palette_blocks == cobblestone)[0][0]
        action = (
            MbagAction.BREAK_BLOCK,
            int(np.ravel_multi_index((x, y, z), env.config["world_size"])),
            0,
        )
        _, (reward,), _, _ = env.step([action])
        assert reward == 0
        _, (reward,), _, _ = env.step([action])
        assert reward == 0

        # Incorrect place action.
        x, y, z = 2, 1, 0
        action = (
            MbagAction.PLACE_BLOCK,
            int(np.ravel_multi_index((x, y, z), env.config["world_size"])),
            cobblestone,
        )
        _, (reward,), _, (info,) = env.step([action])
        assert reward == -1 + incorrect_action_reward
        assert not info["action_correct"]

        # Correct place action.
        x, y, z = 2, 1, 2
        action = (
            MbagAction.PLACE_BLOCK,
            int(np.ravel_multi_index((x, y, z), env.config["world_size"])),
            cobblestone,
        )
        _, (reward,), _, (info,) = env.step([action])
        assert reward == 1
        assert info["action_correct"]
