import glob
import os
import tempfile

import pytest

from mbag.rllib.rollout import ex as rollout_ex
from mbag.rllib.train import ex


@pytest.fixture(scope="session")
def default_config():
    return {
        "log_dir": tempfile.mkdtemp(),
        "width": 6,
        "depth": 6,
        "height": 6,
        "kl_target": 0.01,
        "horizon": 10,
        "num_workers": 10,
        "goal_generator": "random",
        "min_width": 0,
        "min_height": 0,
        "min_depth": 0,
        "extract_largest_cc": True,
        "extract_largest_cc_connectivity": 6,
        "area_sample": False,
        "use_extra_features": True,
        "num_training_iters": 2,
        "train_batch_size": 100,
        "sgd_minibatch_size": 5,
        "rollout_fragment_length": 10,
    }


@pytest.fixture(scope="session")
def default_alpha_zero_config():
    return {
        "run": "MbagAlphaZero",
        "goal_generator": "random",
        "use_replay_buffer": False,
        "hidden_size": 64,
        "num_simulations": 5,
        "sample_batch_size": 100,
        "train_batch_size": 1,
        "num_sgd_iter": 1,
    }


@pytest.fixture(scope="session")
def default_bc_config():
    return {
        "run": "BC",
        "num_workers": 0,
        "evaluation_num_workers": 2,
        "use_extra_features": True,
        "model": "transformer",
        "use_separated_transformer": True,
        "num_layers": 3,
        "vf_share_layers": True,
        "hidden_channels": 64,
        "num_sgd_iter": 1,
        "inf_blocks": False,
        "teleportation": False,
        "input": "data/human_data/sample_tutorial_rllib",
        "is_human": [True],
        "mask_action_distribution": False,
    }


@pytest.fixture(scope="session")
def dummy_ppo_checkpoint_fname(default_config):
    # Execute short dummy run and return the file where the checkpoint is stored.
    checkpoint_dir = tempfile.mkdtemp()
    ex.run(config_updates={**default_config, "log_dir": checkpoint_dir})

    checkpoint_fname = glob.glob(
        checkpoint_dir + "/MbagPPO/self_play/6x6x6/random/*/*/checkpoint_000002"
    )[0]
    assert os.path.exists(checkpoint_fname)
    return checkpoint_fname


@pytest.mark.uses_rllib
@pytest.mark.timeout(60)
def test_single_agent(default_config):
    result = ex.run(
        config_updates={
            **default_config,
            "num_training_iters": 10,
        }
    ).result

    assert result is not None
    assert result["custom_metrics"]["human/own_reward_mean"] > -10


@pytest.mark.uses_rllib
@pytest.mark.timeout(60)
def test_ppo_with_bilevel_categorical(default_config):
    result = ex.run(
        config_updates={
            **default_config,
            "run": "MbagPPO",
            "num_training_iters": 10,
            "custom_action_dist": "mbag_bilevel_categorical",
        }
    ).result

    assert result is not None
    assert result["custom_metrics"]["human/own_reward_mean"] > -10


@pytest.mark.uses_rllib
@pytest.mark.timeout(60)
def test_lstm(default_config):
    result = ex.run(
        config_updates={
            **default_config,
            "use_per_location_lstm": True,
            "max_seq_len": 5,
            "sgd_minibatch_size": 20,
            "vf_share_layers": True,
        }
    ).result

    assert result is not None
    assert result["custom_metrics"]["human/own_reward_mean"] > -10


@pytest.mark.uses_rllib
@pytest.mark.timeout(60)
def test_transformer(default_config):
    result = ex.run(
        config_updates={
            **default_config,
            "model": "transformer",
            "position_embedding_size": 6,
            "hidden_size": 50,
            "num_layers": 3,
            "num_heads": 1,
            "use_separated_transformer": True,
        }
    ).result

    assert result is not None
    assert result["custom_metrics"]["human/own_reward_mean"] > -10

    result = ex.run(
        config_updates={
            **default_config,
            "model": "transformer",
            "position_embedding_size": 6,
            "hidden_size": 50,
            "num_layers": 1,
            "num_heads": 1,
            "use_separated_transformer": False,
        }
    ).result

    assert result is not None
    assert result["custom_metrics"]["human/own_reward_mean"] > -10


@pytest.mark.uses_rllib
@pytest.mark.timeout(60)
def test_cross_play(default_config, dummy_ppo_checkpoint_fname):
    result = ex.run(
        config_updates={
            **default_config,
            "multiagent_mode": "cross_play",
            "num_players": 2,
            "mask_goal": True,
            "use_extra_features": False,
            "own_reward_prop": 1,
            "checkpoint_to_load_policies": dummy_ppo_checkpoint_fname,
            "load_policies_mapping": {"human": "human"},
            "policies_to_train": ["assistant"],
        }
    ).result

    assert result is not None
    assert result["custom_metrics"]["assistant/own_reward_mean"] > -10


@pytest.mark.uses_rllib
@pytest.mark.timeout(60)
def test_policy_retrieval(default_config, dummy_ppo_checkpoint_fname):
    result = ex.run(
        config_updates={
            **default_config,
            "checkpoint_path": dummy_ppo_checkpoint_fname,
        }
    ).result

    assert result is not None
    assert result["custom_metrics"]["human/own_reward_mean"] > -10


@pytest.mark.uses_rllib
@pytest.mark.timeout(60)
def test_train_together(default_config, dummy_ppo_checkpoint_fname):
    result = ex.run(
        config_updates={
            **default_config,
            "checkpoint_to_load_policies": dummy_ppo_checkpoint_fname,
            "multiagent_mode": "cross_play",
            "num_players": 2,
            "load_policies_mapping": {"human": "human"},
            "policies_to_train": ["human", "assistant"],
        }
    ).result
    assert result is not None
    assert result["custom_metrics"]["human/own_reward_mean"] > -10
    assert result["custom_metrics"]["assistant/own_reward_mean"] > -10


@pytest.mark.uses_rllib
@pytest.mark.timeout(60)
def test_alpha_zero(default_config, default_alpha_zero_config):
    result = ex.run(
        config_updates={
            **default_config,
            **default_alpha_zero_config,
        }
    ).result
    assert result is not None
    assert result["custom_metrics"]["human/own_reward_mean"] > -10


@pytest.mark.uses_rllib
@pytest.mark.timeout(60)
def test_alpha_zero_strict_mode(default_config, default_alpha_zero_config):
    result = ex.run(
        config_updates={
            **default_config,
            **default_alpha_zero_config,
            "use_goal_predictor": False,
            "num_training_iters": 2,
            "strict_mode": True,
            "action_reward": [(0, -0.2), (100_000, 0)],
            "num_workers": 0,
        }
    ).result
    assert result is not None
    assert result["custom_metrics"]["human/own_reward_mean"] > -10


@pytest.mark.uses_rllib
@pytest.mark.timeout(60)
def test_alpha_zero_multiple_envs(default_config, default_alpha_zero_config):
    result = ex.run(
        config_updates={
            **default_config,
            **default_alpha_zero_config,
            "num_envs_per_worker": 4,
            "num_workers": 0,
        }
    ).result
    assert result is not None
    assert result["custom_metrics"]["human/own_reward_mean"] > -10


@pytest.mark.uses_rllib
@pytest.mark.timeout(60)
def test_alpha_zero_assistant(
    default_config, default_alpha_zero_config, dummy_ppo_checkpoint_fname
):
    result = ex.run(
        config_updates={
            **default_config,
            **default_alpha_zero_config,
            "multiagent_mode": "cross_play",
            "num_players": 2,
            "mask_goal": True,
            "use_extra_features": False,
            "checkpoint_to_load_policies": dummy_ppo_checkpoint_fname,
            "load_policies_mapping": {"human": "human"},
            "policies_to_train": ["assistant"],
            "model": "transformer_alpha_zero",
        }
    ).result
    assert result is not None
    assert result["custom_metrics"]["human/own_reward_mean"] > -10
    assert result["custom_metrics"]["assistant/own_reward_mean"] > -10


@pytest.mark.uses_rllib
@pytest.mark.timeout(60)
def test_lstm_alpha_zero_assistant(
    default_config, default_alpha_zero_config, dummy_ppo_checkpoint_fname
):
    result = ex.run(
        config_updates={
            **default_config,
            **default_alpha_zero_config,
            "multiagent_mode": "cross_play",
            "num_players": 2,
            "mask_goal": True,
            "use_extra_features": False,
            "checkpoint_to_load_policies": dummy_ppo_checkpoint_fname,
            "load_policies_mapping": {"human": "human"},
            "policies_to_train": ["assistant"],
            "model": "transformer_alpha_zero",
            "use_per_location_lstm": True,
            "max_seq_len": 5,
            "sgd_minibatch_size": 20,
            "vf_share_layers": True,
        }
    ).result
    assert result is not None
    assert result["custom_metrics"]["human/own_reward_mean"] > -10
    assert result["custom_metrics"]["assistant/own_reward_mean"] > -10


@pytest.mark.uses_rllib
@pytest.mark.timeout(60)
def test_alpha_zero_assistant_with_lowest_block_agent(
    default_config, default_alpha_zero_config
):
    result = ex.run(
        config_updates={
            **default_config,
            **default_alpha_zero_config,
            "multiagent_mode": "cross_play",
            "num_players": 2,
            "mask_goal": True,
            "use_extra_features": False,
            "policies_to_train": ["human"],
            "model": "transformer_alpha_zero",
            "heuristic": "lowest_block",
        }
    ).result
    assert result is not None
    assert result["custom_metrics"]["human/own_reward_mean"] > -10
    assert result["custom_metrics"]["lowest_block/place_block_accuracy_mean"] == 1


@pytest.mark.uses_rllib
@pytest.mark.timeout(60)
def test_alpha_zero_assistant_pretraining(
    default_config, default_alpha_zero_config, dummy_ppo_checkpoint_fname
):
    result = ex.run(
        config_updates={
            **default_config,
            **default_alpha_zero_config,
            "multiagent_mode": "cross_play",
            "num_players": 2,
            "mask_goal": True,
            "use_extra_features": False,
            "checkpoint_to_load_policies": dummy_ppo_checkpoint_fname,
            "load_policies_mapping": {"human": "human"},
            "policies_to_train": ["assistant"],
            "model": "transformer_alpha_zero",
            "pretrain": True,
        }
    ).result
    assert result is not None
    assert result["custom_metrics"]["assistant/num_place_block_mean"] == 0
    assert result["custom_metrics"]["assistant/num_break_block_mean"] == 0


@pytest.mark.uses_rllib
@pytest.mark.timeout(60)
def test_alpha_zero_assistant_pretraining_with_alpha_zero_human(
    default_config, default_alpha_zero_config
):
    # Execute short dummy run and return the file where the checkpoint is stored.
    checkpoint_dir = tempfile.mkdtemp()
    ex.run(
        config_updates={
            **default_config,
            **default_alpha_zero_config,
            "num_training_iters": 0,
            "log_dir": checkpoint_dir,
        }
    )
    human_checkpoint_fname = glob.glob(
        checkpoint_dir + "/MbagAlphaZero/self_play/6x6x6/random/*/*/checkpoint_000000"
    )[0]
    assert os.path.exists(human_checkpoint_fname)

    config_updates = {
        **default_config,
        **default_alpha_zero_config,
        "multiagent_mode": "cross_play",
        "num_players": 2,
        "mask_goal": True,
        "use_extra_features": False,
        "checkpoint_to_load_policies": human_checkpoint_fname,
        "load_policies_mapping": {"human": "human"},
        "policies_to_train": ["assistant"],
        "model": "transformer_alpha_zero",
        "num_training_iters": 1,
        "pretrain": True,
    }

    result = ex.run(config_updates=config_updates).result
    assert result is not None
    # Make sure the assistant is doing nothing but that the human isn't!
    horizon = config_updates["horizon"]
    assert result["custom_metrics"]["human/num_noop_mean"] < horizon
    assert result["custom_metrics"]["assistant/num_noop_mean"] == horizon


@pytest.mark.uses_rllib
@pytest.mark.timeout(60)
def test_bc(default_config, default_bc_config):
    result = ex.run(
        config_updates={
            **default_config,
            **default_bc_config,
            "validation_prop": 0.1,
        }
    ).result
    assert result is not None


@pytest.mark.uses_rllib
@pytest.mark.timeout(60)
def test_pikl(default_config, default_bc_config):
    env_configs = {
        "goal_generator": "tutorial",
        "width": 6,
        "depth": 6,
        "height": 6,
    }

    bc_result = ex.run(
        config_updates={
            **default_config,
            **default_bc_config,
            **env_configs,
            "goal_generator": "tutorial",
            "validation_prop": 0.1,
        }
    ).result
    assert bc_result is not None
    bc_checkpoint = bc_result["final_checkpoint"]

    alpha_zero_result = ex.run(
        config_updates={
            **default_config,
            **default_bc_config,
            **env_configs,
            "run": "MbagAlphaZero",
            "load_policies_mapping": {"human": "human"},
            "is_human": [False],
            "checkpoint_to_load_policies": bc_checkpoint,
            "overwrite_loaded_policy_type": True,
            "num_training_iters": 0,
        }
    ).result
    assert alpha_zero_result is not None
    alpha_zero_checkpoint = alpha_zero_result["final_checkpoint"]

    pikl_result = rollout_ex.run(
        config_updates={
            "run": "MbagAlphaZero",
            "episodes": 1,
            "num_workers": 1,
            "checkpoint": alpha_zero_checkpoint,
            "extra_config_updates": {
                "evaluation_config": {
                    "explore": True,
                    "line_of_sight_masking": True,
                    "use_goal_predictor": False,
                    "mcts_config": {
                        "num_simulations": 5,
                        "argmax_tree_policy": False,
                        "temperature": 1,
                        "dirichlet_epsilon": 0,
                    },
                    "env_config": {
                        "players": [{"is_human": False}],
                        "abilities": {
                            "teleportation": False,
                        },
                    },
                }
            },
        }
    ).result
    assert pikl_result is not None
