import glob
import os
import tempfile
from typing import Dict, Iterable, List, cast

import pytest

from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.types import GOAL_BLOCKS

try:
    import torch
    import torch.nn.functional as F  # noqa: N812
    from ray.rllib.models.catalog import ModelCatalog
    from ray.rllib.offline import JsonReader
    from ray.rllib.policy.sample_batch import MultiAgentBatch
    from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
    from torch import nn

    from mbag.rllib.alpha_zero.alpha_zero_policy import C_PUCT
    from mbag.rllib.mixture_model import MixtureModel
    from mbag.rllib.torch_models import MbagTransformerModel
    from mbag.rllib.training_utils import load_trainer
    from mbag.scripts.create_mixture_model import ex as create_mixture_model_ex
    from mbag.scripts.evaluate import ex as evaluate_ex
    from mbag.scripts.rollout import ex as rollout_ex
    from mbag.scripts.train import ex
except ImportError:
    MbagTransformerModel = object  # type: ignore


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
        "vf_share_layers": True,
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
        "train_batch_size": 914,  # total number of timesteps in the dataset
        "num_workers": 0,
        "evaluation_num_workers": 2,
        "use_extra_features": True,
        "model": "transformer",
        "use_separated_transformer": True,
        "num_layers": 3,
        "vf_share_layers": True,
        "hidden_channels": 64,
        "num_sgd_iter": 1,
        "sgd_minibatch_size": 64,
        "num_training_iters": 10,
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
        checkpoint_dir + "/MbagPPO/1_player/6x6x6/random/*/*/checkpoint_000002"
    )[0]
    assert os.path.exists(checkpoint_fname)
    return checkpoint_fname


@pytest.mark.uses_rllib
@pytest.mark.slow
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
@pytest.mark.slow
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
@pytest.mark.slow
@pytest.mark.timeout(120)
def test_kl_regularized_ppo(default_config, dummy_ppo_checkpoint_fname):
    anchor_policy_kls: Dict[float, float] = {}

    for anchor_policy_kl_coeff in [0, 10]:
        result = ex.run(
            config_updates={
                **default_config,
                "run": "MbagPPO",
                "num_training_iters": 10,
                "checkpoint_to_load_policies": dummy_ppo_checkpoint_fname,
                "load_policies_mapping": {"human": "human"},
                "use_anchor_policy": True,
                "anchor_policy_kl_coeff": anchor_policy_kl_coeff,
                "policies_to_train": ["human"],
            }
        ).result

        assert result is not None
        anchor_policy_kl = result["info"]["learner"]["human"]["learner_stats"][
            "anchor_policy_kl"
        ]
        anchor_policy_kls[anchor_policy_kl_coeff] = anchor_policy_kl

    assert anchor_policy_kls[10] < anchor_policy_kls[0]
    assert anchor_policy_kls[0] < 1


@pytest.mark.uses_rllib
@pytest.mark.slow
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
            "num_layers": 4,
            "num_heads": 1,
            "use_separated_transformer": True,
            "interleave_lstm": True,
            "max_seq_len": 5,
            "sgd_minibatch_size": 20,
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
@pytest.mark.slow
@pytest.mark.timeout(60)
def test_cross_play(default_config, dummy_ppo_checkpoint_fname):
    result = ex.run(
        config_updates={
            **default_config,
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
@pytest.mark.slow
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
@pytest.mark.slow
@pytest.mark.timeout(60)
def test_train_together(default_config, dummy_ppo_checkpoint_fname):
    result = ex.run(
        config_updates={
            **default_config,
            "checkpoint_to_load_policies": dummy_ppo_checkpoint_fname,
            "num_players": 2,
            "load_policies_mapping": {"human": "human"},
            "policies_to_train": ["human", "assistant"],
        }
    ).result
    assert result is not None
    assert result["custom_metrics"]["human/own_reward_mean"] > -10
    assert result["custom_metrics"]["assistant/own_reward_mean"] > -10


@pytest.mark.uses_rllib
@pytest.mark.slow
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
@pytest.mark.slow
@pytest.mark.timeout(120)
def test_kl_regularized_alpha_zero(
    default_config, default_alpha_zero_config, dummy_ppo_checkpoint_fname
):
    anchor_policy_kls: Dict[float, float] = {}

    for anchor_policy_kl_coeff in [0, 10]:
        result = ex.run(
            config_updates={
                **default_config,
                **default_alpha_zero_config,
                "checkpoint_to_load_policies": dummy_ppo_checkpoint_fname,
                "load_policies_mapping": {"human": "human"},
                "use_anchor_policy": True,
                "anchor_policy_kl_coeff": anchor_policy_kl_coeff,
                "policies_to_train": ["human"],
            }
        ).result

        assert result is not None
        anchor_policy_kl = result["info"]["learner"]["human"]["learner_stats"][
            "anchor_policy_kl"
        ]
        anchor_policy_kls[anchor_policy_kl_coeff] = anchor_policy_kl

    assert anchor_policy_kls[10] < anchor_policy_kls[0]
    assert anchor_policy_kls[0] < 1


@pytest.mark.uses_rllib
@pytest.mark.slow
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
    prediction_stats = result["info"]["learner"]["human"]["custom_metrics"][
        "prediction_stats"
    ]
    assert abs(prediction_stats["reward_bias"]) < 1e-4
    assert abs(prediction_stats["reward_var"]) < 1e-4
    assert abs(prediction_stats["reward_mse"]) < 1e-4
    assert abs(prediction_stats["own_reward_bias"]) < 1e-4
    assert abs(prediction_stats["own_reward_var"]) < 1e-4
    assert abs(prediction_stats["own_reward_mse"]) < 1e-4


@pytest.mark.uses_rllib
@pytest.mark.slow
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
@pytest.mark.slow
@pytest.mark.timeout(60)
def test_alpha_zero_assistant(
    default_config, default_alpha_zero_config, dummy_ppo_checkpoint_fname
):
    result = ex.run(
        config_updates={
            **default_config,
            **default_alpha_zero_config,
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
@pytest.mark.slow
@pytest.mark.timeout(6000)
def test_alpha_zero_assistant_with_bc(default_config, default_alpha_zero_config):
    result = ex.run(
        config_updates={
            **default_config,
            **default_alpha_zero_config,
            "width": 11,
            "height": 10,
            "depth": 10,
            "inf_blocks": True,
            "teleportation": False,
            "num_players": 2,
            "mask_goal": True,
            "use_extra_features": False,
            "checkpoint_to_load_policies": "data/logs/BC/sample_human_models/inf_blocks_True_teleportation_False/2024-04-10_18-51-43/1/checkpoint_000100",
            "load_policies_mapping": {"human": "human"},
            "policies_to_train": ["assistant"],
            "model": "transformer_alpha_zero",
        }
    ).result
    assert result is not None
    assert result["custom_metrics"]["assistant/own_reward_mean"] > -10
    assert result["custom_metrics"]["assistant/expected_own_reward_mean"] > -10


@pytest.mark.uses_rllib
@pytest.mark.slow
@pytest.mark.timeout(60)
def test_alpha_zero_with_model_replay_buffer(
    default_config, default_alpha_zero_config, dummy_ppo_checkpoint_fname
):
    result = ex.run(
        config_updates={
            **default_config,
            **default_alpha_zero_config,
            "num_players": 2,
            "mask_goal": True,
            "use_extra_features": False,
            "checkpoint_to_load_policies": dummy_ppo_checkpoint_fname,
            "load_policies_mapping": {"human": "human"},
            "policies_to_train": ["assistant"],
            "model": "transformer_alpha_zero",
            "interleave_lstm": True,
            "use_replay_buffer": True,
            "num_layers": 4,
            "replay_buffer_storage_unit": "sequences",
            "replay_buffer_size": 100,
            "use_model_replay_buffer": True,
            "model_replay_buffer_size": 100,
            "train_batch_size": 20,
            "max_seq_len": 5,
            "sgd_minibatch_size": 20,
            "vf_share_layers": True,
        }
    ).result
    assert result is not None
    assert result["custom_metrics"]["human/own_reward_mean"] > -10
    assert result["custom_metrics"]["assistant/own_reward_mean"] > -10


@pytest.mark.uses_rllib
@pytest.mark.slow
@pytest.mark.timeout(60)
def test_interleaved_lstm_alpha_zero_assistant(
    default_config, default_alpha_zero_config, dummy_ppo_checkpoint_fname
):
    result = ex.run(
        config_updates={
            **default_config,
            **default_alpha_zero_config,
            "num_players": 2,
            "mask_goal": True,
            "use_extra_features": False,
            "checkpoint_to_load_policies": dummy_ppo_checkpoint_fname,
            "load_policies_mapping": {"human": "human"},
            "policies_to_train": ["assistant"],
            "model": "transformer_alpha_zero",
            "use_separated_transformer": True,
            "interleave_lstm": True,
            "pretrain": True,
            "num_layers": 4,
            "max_seq_len": 5,
            "sgd_minibatch_size": 20,
            "vf_share_layers": True,
        }
    ).result
    assert result is not None
    assert result["custom_metrics"]["human/own_reward_mean"] > -10
    assert result["custom_metrics"]["assistant/own_reward_mean"] > -10


@pytest.mark.uses_rllib
@pytest.mark.slow
@pytest.mark.timeout(600)
@pytest.mark.limit_memory("600 MB")
def test_lstm_alpha_zero_memory_usage(
    default_config,
    default_alpha_zero_config,
):
    """
    There wre previously some issues with training LSTM-based agents where the memory
    usage would be very high due to saving state_out_* tensors in the replay buffer.
    This makes sure that the memory usage is reasonable.
    """

    result = ex.run(
        config_updates={
            **default_config,
            **default_alpha_zero_config,
            "ray_init_options": {
                "object_store_memory": 78643200,
            },
            "num_envs_per_worker": 8,
            "num_workers": 0,
            "num_gpus": 0,  # Trying to use CUDA with memray seems to cause an error.
            "num_simulations": 2,
            "num_training_iters": 8,
            "rollout_fragment_length": 100,
            "max_seq_len": 10,
            "sample_batch_size": 800,
            "model": "transformer",
            "num_layers": 4,
            "use_separated_transformer": True,
            "interleave_lstm": True,
            "sgd_minibatch_size": 20,
            "train_batch_size": 1,
            "vf_share_layers": True,
        }
    ).result
    assert result is not None


@pytest.mark.uses_rllib
@pytest.mark.slow
@pytest.mark.timeout(60)
def test_alpha_zero_assistant_with_lowest_block_agent(
    default_config, default_alpha_zero_config
):
    result = ex.run(
        config_updates={
            **default_config,
            **default_alpha_zero_config,
            "num_players": 2,
            "mask_goal": True,
            "use_extra_features": False,
            "policies_to_train": ["assistant"],
            "model": "transformer_alpha_zero",
            "heuristic": "lowest_block",
        }
    ).result
    assert result is not None
    assert result["custom_metrics"]["lowest_block/place_block_accuracy_mean"] == 1
    assert result["custom_metrics"]["assistant/own_reward_mean"] > -10
    assert result["custom_metrics"]["assistant/expected_own_reward_mean"] > -10


@pytest.mark.uses_rllib
@pytest.mark.slow
@pytest.mark.timeout(60)
def test_alpha_zero_assistant_pretraining(
    default_config, default_alpha_zero_config, dummy_ppo_checkpoint_fname
):
    result = ex.run(
        config_updates={
            **default_config,
            **default_alpha_zero_config,
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
@pytest.mark.slow
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
        checkpoint_dir + "/MbagAlphaZero/1_player/6x6x6/random/*/*/checkpoint_000000"
    )[0]
    assert os.path.exists(human_checkpoint_fname)

    config_updates = {
        **default_config,
        **default_alpha_zero_config,
        "num_players": 2,
        "randomize_first_episode_length": False,
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


class PerfectGoalPredictorModel(MbagTransformerModel):
    def goal_predictor(self):
        goal_blocks = self._world_obs[:, GOAL_BLOCKS]
        predicted_probs = (
            F.one_hot(goal_blocks, num_classes=MinecraftBlocks.NUM_BLOCKS)
            .permute(0, 4, 1, 2, 3)
            .to(self.device)
        )
        predicted_logits = torch.empty_like(predicted_probs, dtype=torch.float).fill_(
            -1e4
        )
        predicted_logits[predicted_probs == 1] = 0
        return predicted_logits


@pytest.mark.uses_rllib
@pytest.mark.slow
@pytest.mark.timeout(120)
def test_predicted_rewards_equal_rewards_in_alpha_zero(
    default_config, default_alpha_zero_config
):
    ModelCatalog.register_custom_model(
        "mbag_transformer_perfect_goal_predictor_model", PerfectGoalPredictorModel
    )
    result = ex.run(
        config_updates={
            **default_config,
            **default_alpha_zero_config,
            "model": "transformer_perfect_goal_predictor",
            "use_goal_predictor": True,
            "num_training_iters": 1,
            "num_simulations": 20,
            "horizon": 100,
            "rollout_fragment_length": 100,
            "sample_batch_size": 1000,
            "incorrect_action_reward": -0.3,
        }
    ).result
    assert result is not None
    prediction_stats = result["info"]["learner"]["human"]["custom_metrics"][
        "prediction_stats"
    ]
    assert abs(prediction_stats["reward_mse"]) < 1e-8
    assert abs(prediction_stats["own_reward_mse"]) < 1e-8


@pytest.mark.uses_rllib
@pytest.mark.slow
@pytest.mark.timeout(120)
def test_alpha_zero_goal_predictor_kl(default_config, default_alpha_zero_config):
    prev_goal_kls: Dict[float, float] = {}

    for prev_goal_kl_coeff in [0, 10]:
        result = ex.run(
            config_updates={
                **default_config,
                **default_alpha_zero_config,
                "use_goal_predictor": True,
                "prev_goal_kl_coeff": prev_goal_kl_coeff,
            }
        ).result

        assert result is not None
        prev_goal_kl = result["info"]["learner"]["human"]["learner_stats"][
            "prev_goal_kl"
        ]
        prev_goal_kls[prev_goal_kl_coeff] = prev_goal_kl

    assert prev_goal_kls[10] < prev_goal_kls[0]
    assert 0 < prev_goal_kls[0] < 1


@pytest.mark.uses_rllib
@pytest.mark.slow
@pytest.mark.timeout(60)
def test_bc(default_config, default_bc_config):
    result = ex.run(
        config_updates={
            **default_config,
            **default_bc_config,
            "validation_participant_ids": [4],
        }
    ).result
    assert result is not None

    # Without value loss, the value function shouldn't learn anything.
    assert result["info"]["learner"]["human"]["learner_stats"]["vf_explained_var"] < 0.1
    assert result["info"]["learner"]["human"]["learner_stats"]["vf_loss"] > 10


@pytest.mark.uses_rllib
@pytest.mark.slow
@pytest.mark.timeout(60)
def test_bc_with_value_loss(default_config, default_bc_config):
    result = ex.run(
        config_updates={
            **default_config,
            **default_bc_config,
            "validation_participant_ids": [4],
            "vf_loss_coeff": 0.01,
        }
    ).result
    assert result is not None

    assert (
        result["info"]["learner"]["human"]["learner_stats"]["vf_explained_var"] > 0.15
    )
    assert result["info"]["learner"]["human"]["learner_stats"]["vf_loss"] < 6


@pytest.mark.uses_rllib
@pytest.mark.slow
@pytest.mark.timeout(60)
def test_bc_and_evaluate_with_prev_action_input(default_config, default_bc_config):
    bc_result = ex.run(
        config_updates={
            **default_config,
            **default_bc_config,
            "validation_participant_ids": [4],
            "use_prev_action": True,
            "use_fc_after_embedding": True,
        }
    ).result
    assert bc_result is not None

    evaluate_result = evaluate_ex.run(
        config_updates={
            "runs": ["BC"],
            "checkpoints": [bc_result["final_checkpoint"]],
            "policy_ids": ["human"],
            "num_episodes": 1,
        },
    ).result
    assert evaluate_result is not None


@pytest.mark.uses_rllib
@pytest.mark.slow
@pytest.mark.timeout(60)
def test_bc_with_lstm(default_config, default_bc_config):
    result = ex.run(
        config_updates={
            **default_config,
            **default_bc_config,
            "validation_participant_ids": [4],
            "interleave_lstm": True,
            "num_layers": 4,
            "input": "data/human_data/sample_tutorial_rllib_seq_5",
            "max_seq_len": 5,
        }
    ).result
    assert result is not None


@pytest.mark.uses_rllib
@pytest.mark.slow
@pytest.mark.timeout(60)
def test_bc_with_data_augmentation(default_config, default_bc_config):
    result = ex.run(
        config_updates={
            **default_config,
            **default_bc_config,
            "validation_participant_ids": [4],
            "permute_block_types": True,
        }
    ).result
    assert result is not None


@pytest.mark.uses_rllib
@pytest.mark.slow
@pytest.mark.timeout(120)
def test_distill(default_config, default_bc_config, dummy_ppo_checkpoint_fname):
    rollout_result = rollout_ex.run(
        config_updates={
            "run": "MbagPPO",
            "num_episodes": 10,
            "num_workers": 1,
            "checkpoint": dummy_ppo_checkpoint_fname,
            "policy_ids": ["human"],
            "save_samples": True,
            "save_as_sequences": True,
            "max_seq_len": 5,
        }
    ).result
    assert rollout_result is not None
    rollout_dir = rollout_result["out_dir"]

    for use_lstm in [False, True]:
        bc_result = ex.run(
            config_updates={
                **default_config,
                **default_bc_config,
                "input": rollout_dir,
                "mask_goal": True,
                "use_extra_features": False,
                "interleave_lstm": use_lstm,
                "num_layers": 4,
                "max_seq_len": 5,
                "sgd_minibatch_size": 10,
            }
        ).result
        assert bc_result is not None


@pytest.mark.uses_rllib
@pytest.mark.slow
@pytest.mark.timeout(120)
def test_pikl(default_config, default_bc_config, default_alpha_zero_config):
    env_configs = {
        "goal_generator": "tutorial",
        "width": 6,
        "depth": 6,
        "height": 6,
        "teleportation": False,
        "inf_blocks": False,
    }

    bc_result = ex.run(
        config_updates={
            **default_config,
            **default_bc_config,
            **env_configs,
            "goal_generator": "tutorial",
            "validation_participant_ids": [4],
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
            "teleportation": False,
        }
    ).result
    assert alpha_zero_result is not None
    alpha_zero_checkpoint = alpha_zero_result["final_checkpoint"]

    pikl_result = rollout_ex.run(
        config_updates={
            "run": "MbagAlphaZero",
            "num_episodes": 1,
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

    alpha_zero_assistant_result = ex.run(
        config_updates={
            **default_config,
            **default_alpha_zero_config,
            **env_configs,
            "num_players": 2,
            "randomize_first_episode_length": False,
            "mask_goal": True,
            "use_extra_features": False,
            "checkpoint_to_load_policies": alpha_zero_checkpoint,
            "load_policies_mapping": {"human": "human"},
            "policies_to_train": ["assistant"],
            "model": "transformer_alpha_zero",
            "num_training_iters": 1,
            "rollout_fragment_length": 5,
            "pretrain": False,
        },
    ).result
    assert alpha_zero_assistant_result is not None


@pytest.mark.uses_rllib
@pytest.mark.slow
@pytest.mark.timeout(120)
def test_dil_pikl(default_config, default_bc_config):
    env_configs = {
        "goal_generator": "tutorial",
        "width": 6,
        "depth": 6,
        "height": 6,
        "teleportation": False,
        "inf_blocks": False,
    }

    bc_result = ex.run(
        config_updates={
            **default_config,
            **default_bc_config,
            **env_configs,
            "goal_generator": "tutorial",
            "validation_participant_ids": [4],
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
            "teleportation": False,
        }
    ).result
    assert alpha_zero_result is not None
    alpha_zero_checkpoint = alpha_zero_result["final_checkpoint"]

    for sample_c_puct_every_timestep in [False, True]:
        pikl_result = rollout_ex.run(
            config_updates={
                "run": "MbagAlphaZero",
                "num_episodes": 10,
                "num_workers": 0,
                "checkpoint": alpha_zero_checkpoint,
                "save_samples": True,
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
                            "puct_coefficient": [1, 10],
                            "sample_c_puct_every_timestep": sample_c_puct_every_timestep,
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
        pikl_batches = list(
            cast(
                Iterable[MultiAgentBatch],
                JsonReader(pikl_result["out_dir"]).read_all_files(),
            ),
        )

        episode_c_pucts = [
            batch.policy_batches["human"][C_PUCT] for batch in pikl_batches
        ]
        if sample_c_puct_every_timestep:
            assert any(len(set(c_puct)) == 2 for c_puct in episode_c_pucts)
        else:
            assert all(len(set(c_puct)) == 1 for c_puct in episode_c_pucts)
            assert len(set(c_puct[0] for c_puct in episode_c_pucts)) > 1

    assert pikl_result is not None


@pytest.mark.uses_rllib
@pytest.mark.slow
@pytest.mark.timeout(60)
def test_create_mixture_model(default_config, default_bc_config):
    bc_checkpoints: List[str] = []

    for seed in range(2):
        bc_result = ex.run(
            config_updates={
                **default_config,
                **default_bc_config,
                "seed": seed,
            }
        ).result
        assert bc_result is not None
        bc_checkpoints.append(bc_result["final_checkpoint"])

    create_mixture_model_result = create_mixture_model_ex.run(
        config_updates={
            "run": "BC",
            "checkpoints": bc_checkpoints,
        }
    ).result
    assert create_mixture_model_result is not None
    mixture_model_checkpoint = create_mixture_model_result["final_checkpoint"]
    mixture_model_trainer = load_trainer(
        mixture_model_checkpoint,
        "BC",
        config_updates={"num_workers": 0, "num_evaluation_workers": 0},
    )
    mixture_model: MixtureModel = cast(
        MixtureModel,
        cast(TorchPolicyV2, mixture_model_trainer.get_policy("human")).model,
    )

    for component_index, bc_checkpoint in enumerate(bc_checkpoints):
        bc_trainer = load_trainer(
            bc_checkpoint,
            "BC",
            config_updates={"num_workers": 0, "num_evaluation_workers": 0},
        )
        bc_model: nn.Module = cast(TorchPolicyV2, bc_trainer.get_policy("human")).model
        bc_state_dict = bc_model.state_dict()
        mixture_state_dict = cast(
            nn.Module, mixture_model.components[component_index]
        ).state_dict()
        for state_key in bc_state_dict:
            assert torch.allclose(
                bc_state_dict[state_key], mixture_state_dict[state_key]
            )


@pytest.mark.uses_rllib
@pytest.mark.slow
@pytest.mark.timeout(60)
def test_gail(default_config):
    result = ex.run(
        config_updates={
            **default_config,
            "run": "MbagGAIL",
            "demonstration_input": "data/human_data/sample_tutorial_rllib",
            "model": "transformer_with_discriminator",
            "position_embedding_size": 6,
            "hidden_size": 50,
            "num_layers": 3,
            "num_heads": 1,
            "use_separated_transformer": True,
        }
    ).result
    assert result is not None
