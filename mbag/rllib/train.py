from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import get_trainable_cls
from typing import Callable, List, Optional
from typing_extensions import Literal
from logging import Logger
import os
import torch

import ray
from ray.rllib.utils.typing import (
    MultiAgentPolicyConfigDict,
    TrainerConfigDict,
)

from mbag.environment.goals import ALL_GOAL_GENERATORS
from mbag.environment.mbag_env import MbagConfigDict
from mbag.agents.heuristic_agents import ALL_HEURISTIC_AGENTS
from .torch_models import MbagConvolutionalModelConfig, MbagTransformerModelConfig
from .rllib_env import MbagMultiAgentEnv
from .callbacks import MbagCallbacks
from .training_utils import build_logger_creator, load_trainer_config
from .policies import MBAG_POLICIES, MbagAgentPolicy
from .distillation_prediction import DEFAULT_CONFIG as DISTILLATION_DEFAULT_CONFIG

from sacred import Experiment
from sacred import SETTINGS as SACRED_SETTINGS

ex = Experiment("train_mbag")
SACRED_SETTINGS.CONFIG.READ_ONLY_CONFIG = False

torch.autograd.set_detect_anomaly(True)


def make_mbag_sacred_config(ex: Experiment):  # noqa
    @ex.config
    def sacred_config(_log):  # noqa
        # Environment
        goal_generator = "random"
        goal_subset: Literal["train", "val", "test"] = "train"
        horizon = 50
        num_players = 1
        height = 5
        width = 5
        depth = 5
        noop_reward = 0
        place_wrong_reward = -1
        environment_params: MbagConfigDict = {
            "num_players": num_players,
            "horizon": horizon,
            "world_size": (width, height, depth),
            "goal_generator": ALL_GOAL_GENERATORS[goal_generator],
            "goal_generator_config": {
                "data_dir": f"data/{goal_generator}",
                "subset": goal_subset,
            },
            "goal_visibility": [True] * num_players,
            "rewards": {
                "noop": noop_reward,
                "place_wrong": place_wrong_reward,
            },
        }
        env = MbagMultiAgentEnv(**environment_params)

        # Training
        run = "PPO"
        num_workers = 2
        num_aggregation_workers = min(1, num_workers)
        input = "sampler"
        seed = 0
        num_gpus = 1 if torch.cuda.is_available() else 0
        train_batch_size = 5000
        sgd_minibatch_size = 500
        rollout_fragment_length = horizon
        num_training_iters = 500  # noqa: F841
        lr = 1e-3
        grad_clip = 10
        gamma = 0.95
        gae_lambda = 0.98
        vf_share_layers = False
        vf_loss_coeff = 1e-4
        entropy_coeff_start = 0.01
        entropy_coeff_end = 0
        entropy_coeff_horizon = 1e5
        kl_coeff = 0.2
        kl_target = 0.01
        clip_param = 0.05
        num_sgd_iter = 6
        compress_observations = True

        # Model
        model: Literal[
            "convolutional", "recurrent_convolutional", "transformer"
        ] = "convolutional"
        embedding_size = 8
        position_embedding_size = 8
        use_extra_features = True
        mask_goal = False
        num_conv_1_layers = 1
        num_layers = 1
        filter_size = 3
        hidden_channels = 16
        hidden_size = hidden_channels
        num_block_id_layers = 2
        num_heads = 4
        num_unet_layers = 0
        model_config = {
            "custom_model": f"mbag_{model}_model",
            "custom_action_dist": "mbag_autoregressive",
            "vf_share_layers": vf_share_layers,
        }
        if model in ["convolutional", "recurrent_convolutional"]:
            conv_config: MbagConvolutionalModelConfig = {
                "embedding_size": embedding_size,
                "use_extra_features": use_extra_features,
                "mask_goal": mask_goal,
                "num_conv_1_layers": num_conv_1_layers,
                "num_layers": num_layers,
                "filter_size": filter_size,
                "hidden_channels": hidden_channels,
                "num_block_id_layers": num_block_id_layers,
                "num_unet_layers": num_unet_layers,
            }
            model_config["custom_model_config"] = conv_config
        elif model == "transformer":
            transformer_config: MbagTransformerModelConfig = {
                "embedding_size": embedding_size,
                "position_embedding_size": position_embedding_size,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "hidden_size": hidden_size,
                "num_block_id_layers": num_block_id_layers,
            }
            model_config["custom_model_config"] = transformer_config

        # Multiagent
        multiagent_mode: Literal["self_play", "cross_play"] = "self_play"
        policy_ids: List[str]
        policy_mapping_fn: Callable[[str], str]
        if multiagent_mode == "self_play":
            policy_ids = ["ppo"]
            policy_mapping_fn = lambda agent_id: "ppo"  # noqa: E731
        elif multiagent_mode == "cross_play":
            policy_ids = [f"ppo_{player_index}" for player_index in range(num_players)]
            policy_mapping_fn = lambda agent_id: agent_id.replace(  # noqa: E731
                "player_", "ppo_"
            )
        policies_to_train = policy_ids

        # Logging
        save_freq = 25  # noqa: F841
        log_dir = "data/logs"  # noqa: F841
        experiment_tag = None
        size_str = f"{width}x{height}x{depth}"
        experiment_name_parts = [run, multiagent_mode, size_str, goal_generator]
        if experiment_tag is not None:
            experiment_name_parts.append(experiment_tag)
        experiment_name = os.path.join(*experiment_name_parts)  # noqa: F841
        checkpoint_path = None  # noqa: F841
        checkpoint_to_load_policies = None

        policies: MultiAgentPolicyConfigDict = {}
        for policy_id in policy_ids:
            policies[policy_id] = PolicySpec(
                MBAG_POLICIES.get(run),
                env.observation_space,
                env.action_space,
                {"model": model_config},
            )

        config: TrainerConfigDict = {  # noqa: F841
            "env": "MBAG-v1",
            "env_config": environment_params,
            "multiagent": {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": policies_to_train,
            },
            "callbacks": MbagCallbacks,
            "num_workers": num_workers,
            "num_gpus": num_gpus,
            "input": input,
            "input_evaluation": [],
            "lr": lr,
            "train_batch_size": train_batch_size,
            "sgd_minibatch_size": sgd_minibatch_size,
            "num_sgd_iter": num_sgd_iter,
            "vf_loss_coeff": vf_loss_coeff,
            "compress_observations": compress_observations,
            "rollout_fragment_length": rollout_fragment_length,
            "grad_clip": grad_clip,
            "seed": seed,
            "entropy_coeff_schedule": [
                (0, entropy_coeff_start),
                (entropy_coeff_horizon, entropy_coeff_end),
            ],
            "framework": "torch",
        }
        if "PPO" in run:
            config.update(
                {
                    "gamma": gamma,
                    "lambda": gae_lambda,
                    "kl_coeff": kl_coeff,
                    "kl_target": kl_target,
                    "clip_param": clip_param,
                }
            )
        if run in ["IMPALA", "APPO"]:
            config.update(
                {
                    "num_aggregation_workers": num_aggregation_workers,
                }
            )
            config["train_batch_size"] = config.pop("sgd_minibatch_size")

        # Distillation
        if "distillation_prediction" in run:
            validation_input = None
            validation_num_batches = 1
            validation_smoothing = 0.9
            config["checkpoint_to_load_policies"] = checkpoint_to_load_policies
            if checkpoint_to_load_policies is None and validation_input is None:
                # Distill a heuristic policy.
                heuristic = "layer_builder"
                mbag_agent = ALL_HEURISTIC_AGENTS[heuristic]({}, environment_params)
                config["multiagent"]["policies"][heuristic] = (
                    MbagAgentPolicy,
                    env.observation_space,
                    env.action_space,
                    {"mbag_agent": mbag_agent},
                )
                config["multiagent"][
                    "distillation_mapping_fn"
                ] = lambda policy_id, to_policy_id=policy_ids[0]: to_policy_id
                config["multiagent"][
                    "policy_mapping_fn"
                ] = lambda agent_id, to_policy_id=heuristic: to_policy_id
            else:
                if checkpoint_to_load_policies is not None:
                    checkpoint_to_load_policies_config = load_trainer_config(
                        checkpoint_to_load_policies
                    )
                config["validation"] = {
                    "input": validation_input,
                    "num_batches": validation_num_batches,
                    "smoothing": validation_smoothing,
                }
                # Add a corresponding distilled policy for each policy in the checkpoint.
                previous_policy_ids = list(policies.keys())
                policies_to_train.clear()
                for policy_id in previous_policy_ids:
                    if checkpoint_to_load_policies is not None:
                        policies[policy_id] = checkpoint_to_load_policies_config[
                            "multiagent"
                        ]["policies"][policy_id]
                    distill_policy_id = f"{policy_id}_distilled"
                    policies[distill_policy_id] = (
                        MBAG_POLICIES.get(run),
                        env.observation_space,
                        env.action_space,
                        {"model": model_config},
                    )
                    policies_to_train.append(distill_policy_id)
                config["multiagent"][
                    "distillation_mapping_fn"
                ] = lambda policy_id: f"{policy_id}_distilled"
            # Remove extra config parameters.
            for key in list(config.keys()):
                if key not in DISTILLATION_DEFAULT_CONFIG:
                    del config[key]

        del env


make_mbag_sacred_config(ex)


@ex.automain
def main(
    config,
    log_dir,
    experiment_name,
    run,
    num_training_iters,
    save_freq,
    checkpoint_path: Optional[str],
    _log: Logger,
):
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
    )

    trainer_class = get_trainable_cls(run)
    trainer = trainer_class(
        config,
        logger_creator=build_logger_creator(
            log_dir,
            experiment_name,
        ),
    )

    if checkpoint_path is not None:
        _log.info(f"Restoring checkpoint at {checkpoint_path}")
        trainer.restore(checkpoint_path)

    result = None
    for train_iter in range(num_training_iters):
        _log.info(f"Starting training iteration {train_iter}")
        result = trainer.train()

        if trainer.iteration % save_freq == 0:
            checkpoint = trainer.save()
            _log.info(f"Saved checkpoint to {checkpoint}")

    checkpoint = trainer.save()
    _log.info(f"Saved final checkpoint to {checkpoint}")

    return result
