import glob
import os
import pytest
from mbag.rllib.train import ex
import tempfile

# This is done to satisfy mypy
dummy_run: str = ""


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
        "use_extra_features": True,
        "num_training_iters": 2,
        "train_batch_size": 50,
        "sgd_minibatch_size": 5,
        "rollout_fragment_length": 10,
    }


# This runs once beore the tests run.
@pytest.fixture(scope="session", autouse=True)
def setup(default_config):
    # Execute short dummy run and return the file where the checkpoint is stored.
    checkpoint_dir = tempfile.mkdtemp()
    ex.run(config_updates={**default_config, "log_dir": checkpoint_dir})

    global dummy_run
    dummy_run = glob.glob(
        checkpoint_dir
        + "/MbagPPO/self_play/6x6x6/random/*/checkpoint_000002/checkpoint-2"
    )[0]
    assert os.path.exists(dummy_run)


@pytest.mark.uses_rllib
def test_single_agent(default_config):
    result = ex.run(
        config_updates={
            **default_config,
            "num_training_iters": 10,
        }
    ).result

    assert result["custom_metrics"]["ppo/own_reward_mean"] > -10


@pytest.mark.uses_rllib
def test_transformer(default_config):
    result = ex.run(
        config_updates={
            **default_config,
            "model": "transformer",
            "position_embedding_size": 6,
            "hidden_size": 39,
            "num_layers": 3,
            "num_heads": 1,
            "use_separated_transformer": True,
        }
    ).result

    assert result["custom_metrics"]["ppo/own_reward_mean"] > -10

    result = ex.run(
        config_updates={
            **default_config,
            "model": "transformer",
            "position_embedding_size": 6,
            "hidden_size": 39,
            "num_layers": 1,
            "num_heads": 1,
            "use_separated_transformer": False,
        }
    ).result

    assert result["custom_metrics"]["ppo/own_reward_mean"] > -10


@pytest.mark.uses_rllib
def test_cross_play(default_config):
    result = ex.run(
        config_updates={
            **default_config,
            "multiagent_mode": "cross_play",
            "num_players": 2,
            "mask_goal": True,
            "use_extra_features": False,
            "own_reward_prop": 1,
            "checkpoint_to_load_policies": dummy_run,
            "load_policies_mapping": {"ppo": "ppo_0"},
            "policies_to_train": ["ppo_1"],
        }
    ).result

    assert result["custom_metrics"]["ppo_1/own_reward_mean"] > -10


@pytest.mark.uses_rllib
def test_policy_retrieval(default_config):
    result = ex.run(
        config_updates={
            **default_config,
            "checkpoint_path": dummy_run,
        }
    ).result

    assert result["custom_metrics"]["ppo/own_reward_mean"] > -10


@pytest.mark.uses_rllib
def test_distillation(default_config):
    result = ex.run(
        config_updates={
            **default_config,
            "checkpoint_to_load_policies": dummy_run,
            "run": "distillation_prediction",
        }
    ).result

    assert result["custom_metrics"]["ppo/own_reward_mean"] > -10

    result = ex.run(
        config_updates={
            **default_config,
            "run": "distillation_prediction",
            "heuristic": "mirror_builder",
            "checkpoint_to_load_policies": dummy_run,
        }
    ).result

    assert result["custom_metrics"]["ppo/own_reward_mean"] > -10


@pytest.mark.uses_rllib
def test_train_together(default_config):
    result = ex.run(
        config_updates={
            **default_config,
            "checkpoint_to_load_policies": dummy_run,
            "multiagent_mode": "cross_play",
            "num_players": 2,
            "load_policies_mapping": {"ppo": "ppo_0"},
            "policies_to_train": ["ppo_0", "ppo_1"],
        }
    ).result
    assert result["custom_metrics"]["ppo_0/own_reward_mean"] > -10
    assert result["custom_metrics"]["ppo_1/own_reward_mean"] > -10


@pytest.mark.uses_rllib
def test_action_gate(default_config):
    result = ex.run(
        config_updates={
            **default_config,
            "model": "action_gate_transformer",
            "position_embedding_size": 6,
            "hidden_size": 39,
            "num_layers": 3,
            "num_heads": 1,
            "use_separated_transformer": True,
        }
    ).result

    assert result["custom_metrics"]["ppo/own_reward_mean"] > -10
