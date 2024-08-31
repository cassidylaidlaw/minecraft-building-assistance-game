"""
This files contains tests that run the entire experiment pipeline (with zero actual
training). It ensures that all configs match and that the pipeline runs without errors.
"""

import glob
import json
import logging
import os
from typing import Any, Dict, List, Tuple

import pytest

import mbag.environment.goals

try:
    import torch

    from mbag.scripts.evaluate import ex as evaluate_ex
    from mbag.scripts.evaluate_human_modeling import ex as evaluate_human_modeling_ex
    from mbag.scripts.train import ex as train_ex
except ImportError:
    pass


def remove_py_id(obj):
    if isinstance(obj, dict):
        if "py/id" in obj:
            del obj["py/id"]
        for key, value in obj.items():
            obj[key] = remove_py_id(value)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            obj[i] = remove_py_id(value)
    return obj


def assert_config_matches(
    experiment_dir,
    reference_experiment_dir,
):
    with open(os.path.join(experiment_dir, "config.json")) as config_file:
        json_config = json.load(config_file)

    with open(os.path.join(reference_experiment_dir, "config.json")) as config_file:
        reference_json_config = json.load(config_file)

    json_config["config"] = remove_py_id(json_config["config"])
    reference_json_config["config"] = remove_py_id(reference_json_config["config"])

    for config in [json_config["config"], reference_json_config["config"]]:
        # We don't use rollout workers in the tests, so we remove them from the config.
        del config["num_rollout_workers"]
        del config["evaluation_num_workers"]

        # Don't report inconsistencies in the number of GPUs.
        del config["num_gpus"]
        del config["num_gpus_per_worker"]

        # Weird inconsistent JSON serialization.
        for policy_id, policy_spec in config["policies"].items():
            if "spaces" in policy_spec["observation_space"]:
                timestep_space = policy_spec["observation_space"]["spaces"]["py/tuple"][
                    2
                ]
                del timestep_space["bounded_above"]
                del timestep_space["dtype"]

    assert json_config["config"] == reference_json_config["config"]
    assert json_config["run"] == reference_json_config["run"]
    assert (
        json_config["load_policies_mapping"]
        == reference_json_config["load_policies_mapping"]
    )

    # General sanity checks to make sure the config is following best practices.
    env_config = json_config["config"]["env_config"]
    assert env_config["randomize_first_episode_length"] is True
    assert (
        json_config["evaluation_config"]["env_config"]["randomize_first_episode_length"]
        is False
    )
    assert env_config["horizon"] == 1500

    for policy_id, policy_spec in json_config["config"]["policies"].items():
        policy_config = policy_spec["config"]
        custom_model_config = policy_config.get("model", {}).get("custom_model_config")
        if custom_model_config:
            assert custom_model_config["line_of_sight_masking"]
            assert custom_model_config["mask_action_distribution"]
            assert custom_model_config["scale_obs"]
            if custom_model_config["interleave_lstm"]:
                assert custom_model_config["num_layers"] == 8
            else:
                assert custom_model_config["num_layers"] == 6
            assert custom_model_config["hidden_size"] == 64

    if json_config["run"] == "MbagAlphaZero":
        mcts_config = json_config["config"]["mcts_config"]
        assert mcts_config["use_bilevel_action_selection"]
        assert mcts_config["fix_bilevel_action_selection"]


@pytest.mark.timeout(600)
@pytest.mark.uses_rllib
@pytest.mark.slow
def test_experiments(tmp_path):
    # Supress huge number of logging messages about the goals being sampled.
    mbag.environment.goals.logger.setLevel(logging.WARNING)

    common_config_updates = {
        "num_training_iters": 0,
        "num_workers": 0,
        "evaluation_num_workers": 0,
    }
    if not torch.cuda.is_available():
        common_config_updates["num_gpus"] = 0
        common_config_updates["num_gpus_per_worker"] = 0

    # Test ppo_human
    ppo_human_result = train_ex.run(
        named_configs=["ppo_human"],
        config_updates={
            "experiment_dir": str(tmp_path / "ppo_human"),
            **common_config_updates,
        },
    ).result
    assert ppo_human_result is not None
    assert_config_matches(
        glob.glob(str(tmp_path / "ppo_human" / "[1-9]*"))[0],
        "data/testing/reference_experiments/ppo_human",
    )

    # Test alphazero_human
    alphazero_human_result = train_ex.run(
        named_configs=["alphazero_human"],
        config_updates={
            "experiment_dir": str(tmp_path / "alphazero_human"),
            **common_config_updates,
        },
    ).result
    assert alphazero_human_result is not None
    assert_config_matches(
        glob.glob(str(tmp_path / "alphazero_human" / "[1-9]*"))[0],
        "data/testing/reference_experiments/alphazero_human",
    )

    # Test bc_human
    bc_human_results: Dict[str, Any] = {}
    for bc_human_name, data_split, checkpoint_to_load_policies in [
        ("rand_init_human_alone", "human_alone", None),
        ("rand_init_human_with_assistant", "human_with_assistant", None),
        ("rand_init_combined", "combined", None),
        ("ppo_init", "human_alone", ppo_human_result["final_checkpoint"]),
        ("alphazero_init", "human_alone", alphazero_human_result["final_checkpoint"]),
    ]:
        experiment_dir = tmp_path / f"bc_human_{bc_human_name}"
        bc_human_result = train_ex.run(
            named_configs=["bc_human"],
            config_updates={
                "experiment_dir": str(experiment_dir),
                "data_split": data_split,
                "checkpoint_to_load_policies": checkpoint_to_load_policies,
                **common_config_updates,
            },
        ).result
        assert bc_human_result is not None
        assert_config_matches(
            glob.glob(str(experiment_dir / "[1-9]*"))[0],
            f"data/testing/reference_experiments/bc_human_{bc_human_name}",
        )
        bc_human_results[bc_human_name] = bc_human_result

    # Test piKL
    pikl_result = train_ex.run(
        named_configs=["pikl"],
        config_updates={
            "checkpoint_to_load_policies": bc_human_results["rand_init_human_alone"][
                "final_checkpoint"
            ],
            "checkpoint_name": "rand_init_human_alone",
            "experiment_dir": str(tmp_path / "pikl"),
            **common_config_updates,
        },
    ).result
    assert pikl_result is not None
    assert_config_matches(
        glob.glob(str(tmp_path / "pikl" / "[1-9]*"))[0],
        "data/testing/reference_experiments/pikl",
    )

    # Test all human model evals
    human_models: List[Tuple[str, str]] = (
        [
            ("MbagPPO", ppo_human_result["final_checkpoint"]),
            ("MbagAlphaZero", alphazero_human_result["final_checkpoint"]),
        ]
        + [
            ("BC", bc_human_result["final_checkpoint"])
            for bc_human_result in bc_human_results.values()
        ]
        + [
            ("MbagAlphaZero", pikl_result["final_checkpoint"]),
        ]
    )
    for human_model_run, human_model_checkpoint in human_models:
        extra_config_updates = {}
        if human_model_run == "MbagAlphaZero":
            extra_config_updates = {
                "mcts_config": {
                    "num_simulations": 1,
                }
            }

        for data_split in ["human_alone", "human_with_assistant"]:
            evaluate_human_modeling_result = evaluate_human_modeling_ex.run(
                config_updates={
                    "run": human_model_run,
                    "checkpoint": human_model_checkpoint,
                    "policy_id": "human",
                    "participant_ids": [3],
                    "max_episode_len": 10,
                    "extra_config_updates": extra_config_updates,
                    "human_data_dir": f"data/human_data_cleaned/{data_split}/infinite_blocks_true/rllib_with_own_noops_flat_actions_flat_observations_place_wrong_reward_-1_repaired_player_0",
                },
            ).result
            assert evaluate_human_modeling_result is not None

        evaluate_result = evaluate_ex.run(
            config_updates={
                "runs": [human_model_run],
                "checkpoints": [human_model_checkpoint],
                "policy_ids": ["human"],
                "num_episodes": 1,
                "algorithm_config_updates": [extra_config_updates],
                "env_config_updates": {
                    "horizon": 10,
                    "truncate_on_no_progress_timesteps": None,
                    "goal_generator_config": {
                        "goal_generator_config": {
                            "subset": "test",
                        }
                    },
                },
                "num_workers": 0,
            },
        ).result
        assert evaluate_result is not None

    # Test alphazero_assistant
    alphazero_assistant_result = train_ex.run(
        named_configs=["alphazero_assistant"],
        config_updates={
            "experiment_dir": str(tmp_path / "alphazero_assistant"),
            "checkpoint_to_load_policies": bc_human_results["rand_init_human_alone"][
                "final_checkpoint"
            ],
            "checkpoint_name": "rand_init_human_alone",
            **common_config_updates,
        },
    ).result
    assert alphazero_assistant_result is not None
    assert_config_matches(
        glob.glob(str(tmp_path / "alphazero_assistant" / "[1-9]*"))[0],
        "data/testing/reference_experiments/alphazero_assistant",
    )

    # Test ppo_assistant
    ppo_assistant_result = train_ex.run(
        named_configs=["ppo_assistant"],
        config_updates={
            "experiment_dir": str(tmp_path / "ppo_assistant"),
            "checkpoint_to_load_policies": bc_human_results["rand_init_human_alone"][
                "final_checkpoint"
            ],
            "checkpoint_name": "rand_init_human_alone",
            **common_config_updates,
        },
    ).result
    assert ppo_assistant_result is not None
    assert_config_matches(
        glob.glob(str(tmp_path / "ppo_assistant" / "[1-9]*"))[0],
        "data/testing/reference_experiments/ppo_assistant",
    )

    # TODO: test human_rollout
    # TODO: test masked_bc_human
    # TODO: test bc_assistant
