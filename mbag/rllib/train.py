from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import get_trainable_cls
from ray.rllib.evaluation import Episode, RolloutWorker
from typing import Callable, Dict, List, Optional
from typing_extensions import Literal
from logging import Logger
import os
import torch

import ray
from ray.rllib.utils.typing import (
    MultiAgentPolicyConfigDict,
    TrainerConfigDict,
)

from mbag.environment.goals.filters import DensityFilterConfig, MinSizeFilterConfig
from mbag.environment.goals.goal_transform import (
    GoalTransformSpec,
    TransformedGoalGeneratorConfig,
)
from mbag.environment.goals.transforms import CropTransformConfig
from mbag.environment.mbag_env import MbagConfigDict
from mbag.agents.heuristic_agents import ALL_HEURISTIC_AGENTS
from .torch_models import (
    MbagRecurrentConvolutionalModelConfig,
    MbagTransformerModelConfig,
)
from .rllib_env import MbagMultiAgentEnv
from .callbacks import MbagCallbacks
from .training_utils import (
    build_logger_creator,
    load_policies_from_checkpoint,
    load_trainer_config,
)
from mbag.rllib.policies import MbagPPOTorchPolicy, MbagAgentPolicy
from .distillation_prediction import DEFAULT_CONFIG as DISTILLATION_DEFAULT_CONFIG

from sacred import Experiment
from sacred.config.custom_containers import DogmaticDict
from sacred import SETTINGS as SACRED_SETTINGS

ex = Experiment("train_mbag")
SACRED_SETTINGS.CONFIG.READ_ONLY_CONFIG = False

torch.autograd.set_detect_anomaly(True)


def make_mbag_sacred_config(ex: Experiment):  # noqa
    @ex.config
    def sacred_config(_log):  # noqa
        # Environment
        goal_generator = "random"
        goal_subset = "train"
        horizon = 50
        num_players = 1
        height = 5
        width = 5
        depth = 5
        noop_reward = 0
        action_reward = 0
        place_wrong_reward = -1
        teleportation = True
        flying = True
        goal_visibility = [True] * num_players
        timestep_skip = [1] * num_players
        own_reward_prop = 0
        own_reward_prop_horizon: Optional[int] = None
        goal_generator_config = {"subset": goal_subset}

        goal_transforms: List[GoalTransformSpec] = []
        uniform_block_type = False
        force_single_cc = True
        min_density = 0
        max_density = 1
        crop = False
        crop_density_threshold = 0.25
        wall = False
        mirror = False
        min_width, min_height, min_depth = width // 2, height // 2, depth // 2
        if uniform_block_type:
            goal_transforms.append({"transform": "uniform_block_type"})
        if force_single_cc:
            goal_transforms.append({"transform": "single_cc_filter"})
        min_size_config: MinSizeFilterConfig = {
            "min_size": (min_width, min_height, min_depth)
        }
        goal_transforms.append(
            {"transform": "min_size_filter", "config": min_size_config}
        )
        if crop or wall:
            crop_config: CropTransformConfig = {
                "density_threshold": 1000 if wall else crop_density_threshold,
                "tethered_to_ground": True,
                "wall": wall,
            }
            goal_transforms.append({"transform": "crop", "config": crop_config})
        density_config: DensityFilterConfig = {
            "min_density": min_density,
            "max_density": max_density,
        }
        goal_transforms.append(
            {"transform": "density_filter", "config": density_config}
        )
        goal_transforms.append({"transform": "randomly_place"})
        goal_transforms.append({"transform": "add_grass"})
        if mirror:
            goal_transforms.append({"transform": "mirror"})

        transformed_goal_generator_config: TransformedGoalGeneratorConfig = {
            "goal_generator": goal_generator,
            "goal_generator_config": goal_generator_config,
            "goal_transforms": goal_transforms,
        }

        environment_params: MbagConfigDict = {
            "num_players": num_players,
            "horizon": horizon,
            "world_size": (width, height, depth),
            "goal_generator_config": transformed_goal_generator_config,
            "malmo": {
                "use_malmo": False,
                "player_names": None,
            },
            "goal_visibility": goal_visibility,
            "timestep_skip": timestep_skip,
            "rewards": {
                "noop": noop_reward,
                "action": action_reward,
                "place_wrong": place_wrong_reward,
                "own_reward_prop": own_reward_prop,
                "own_reward_prop_horizon": own_reward_prop_horizon,
            },
            "abilities": {
                "teleportation": teleportation,
                "flying": flying,
            },
        }
        env = MbagMultiAgentEnv(**environment_params)

        # Training
        run = "MbagPPO"
        num_workers = 2
        input = "sampler"
        seed = 0
        num_gpus = 1 if torch.cuda.is_available() else 0
        train_batch_size = 5000
        sgd_minibatch_size = 500
        rollout_fragment_length = horizon
        num_training_iters = 500  # noqa: F841
        lr = 1e-3
        grad_clip = 0.1
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
        goal_loss_coeff, place_block_loss_coeff = 0.5, 1

        # Model
        model: Literal[
            "convolutional", "recurrent_convolutional", "transformer"
        ] = "convolutional"
        max_seq_len = horizon
        embedding_size = 8
        position_embedding_size = 6
        mask_goal = False
        use_extra_features = not mask_goal
        num_conv_1_layers = 1
        num_layers = 1
        filter_size = 3
        hidden_channels = 16
        hidden_size = hidden_channels
        num_block_id_layers = 2
        num_heads = 4
        use_separated_transformer = False
        use_resnet = False
        num_unet_layers = 0
        unet_grow_factor = 2
        unet_use_bn = False
        num_value_layers = 0
        model_config = {
            "custom_model": f"mbag_{model}_model",
            "custom_action_dist": "mbag_autoregressive",
            "max_seq_len": max_seq_len,
            "vf_share_layers": vf_share_layers,
        }
        if model in ["convolutional", "recurrent_convolutional"]:
            conv_config: MbagRecurrentConvolutionalModelConfig = {
                "embedding_size": embedding_size,
                "use_extra_features": use_extra_features,
                "mask_goal": mask_goal,
                "num_conv_1_layers": num_conv_1_layers,
                "num_layers": num_layers,
                "use_resnet": use_resnet,
                "filter_size": filter_size,
                "hidden_channels": hidden_channels,
                "num_block_id_layers": num_block_id_layers,
                "num_unet_layers": num_unet_layers,
                "unet_grow_factor": unet_grow_factor,
                "unet_use_bn": unet_use_bn,
                "num_value_layers": num_value_layers,
            }
            model_config["custom_model_config"] = conv_config
        elif model == "transformer":
            transformer_config: MbagTransformerModelConfig = {
                "embedding_size": embedding_size,
                "use_extra_features": use_extra_features,
                "mask_goal": mask_goal,
                "position_embedding_size": position_embedding_size,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "hidden_size": hidden_size,
                "num_block_id_layers": num_block_id_layers,
                "use_separated_transformer": use_separated_transformer,
            }
            model_config["custom_model_config"] = transformer_config

        # Resume from checkpoint
        checkpoint_path = None  # noqa: F841
        checkpoint_to_load_policies = None

        # Maps policy IDs in checkpoint_to_load_policies to policy IDs here
        load_policies_mapping: Dict[str, str] = {}
        if isinstance(load_policies_mapping, DogmaticDict):
            # Weird shim for sacred
            for key in load_policies_mapping.revelation():
                load_policies_mapping[key] = load_policies_mapping[key]

        if checkpoint_to_load_policies is not None:
            checkpoint_to_load_policies_config = load_trainer_config(
                checkpoint_to_load_policies
            )

        # Multiagent
        heuristic: Optional[str] = None
        multiagent_mode: Literal["self_play", "cross_play"] = "self_play"
        policy_ids: List[str]
        policy_mapping_fn: Callable[[str, Episode, RolloutWorker], str]
        if multiagent_mode == "self_play":
            policy_ids = ["ppo"]
            policy_mapping_fn = (
                lambda agent_id, episode, worker, **kwargs: "ppo"
            )  # noqa: E731
        elif multiagent_mode == "cross_play":
            policy_ids = [f"ppo_{player_index}" for player_index in range(num_players)]
            if heuristic is not None:
                policy_ids[-1] = heuristic

            def policy_mapping_fn(
                agent_id: str, episode, worker, policy_ids=policy_ids, **kwargs
            ):
                agent_index = int(agent_id[len("player_") :])
                return policy_ids[agent_index]

        environment_params["malmo"]["player_names"] = policy_ids

        loaded_policy_dict: MultiAgentPolicyConfigDict = {}
        if checkpoint_to_load_policies is not None:
            unmapped_loaded_policy_dict = checkpoint_to_load_policies_config[
                "multiagent"
            ]["policies"]
            for old_policy_id, new_policy_id in load_policies_mapping.items():
                loaded_policy_dict[new_policy_id] = unmapped_loaded_policy_dict[
                    old_policy_id
                ]

        policies_to_train = []
        for policy_id in policy_ids:
            if policy_id.startswith("ppo") and policy_id not in loaded_policy_dict:
                policies_to_train.append(policy_id)

        policies: MultiAgentPolicyConfigDict = {}
        for policy_id in policy_ids:
            if policy_id in loaded_policy_dict:
                policies[policy_id] = loaded_policy_dict[policy_id]
            elif policy_id.startswith("ppo"):
                policies[policy_id] = PolicySpec(
                    MbagPPOTorchPolicy,
                    env.observation_space,
                    env.action_space,
                    {"model": model_config},
                )
            else:
                # Heuristic agent policy.
                mbag_agent = ALL_HEURISTIC_AGENTS[policy_id]({}, environment_params)
                policies[policy_id] = PolicySpec(
                    MbagAgentPolicy,
                    env.observation_space,
                    env.action_space,
                    {"mbag_agent": mbag_agent},
                )

        # Logging
        save_freq = 25  # noqa: F841
        log_dir = "data/logs"  # noqa: F841
        experiment_tag = None
        size_str = f"{width}x{height}x{depth}"
        experiment_name_parts = [run, multiagent_mode, size_str, goal_generator]
        if heuristic is not None:
            experiment_name_parts.append(heuristic)
        if experiment_tag is not None:
            experiment_name_parts.append(experiment_tag)
        experiment_name = os.path.join(*experiment_name_parts)  # noqa: F841

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
            "actions_in_input_normalized": input != "sampler",
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
            "goal_loss_coeff": goal_loss_coeff,
            "place_block_loss_coeff": place_block_loss_coeff,
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
        # Distillation
        if "distillation_prediction" in run:
            config["checkpoint_to_load_policies"] = checkpoint_to_load_policies
            if heuristic is not None:
                # Distill a heuristic policy.
                assert heuristic is not None
                mbag_agent = ALL_HEURISTIC_AGENTS[heuristic]({}, environment_params)
                config["multiagent"]["policies"][heuristic] = (
                    MbagAgentPolicy,
                    env.observation_space,
                    env.action_space,
                    {"mbag_agent": mbag_agent},
                )
                distill_policy_id = f"{heuristic}_distilled"
                policies[distill_policy_id] = PolicySpec(
                    None,
                    env.observation_space,
                    env.action_space,
                    {"model": model_config},
                )
                policies_to_train = [distill_policy_id]
            elif checkpoint_to_load_policies is not None:
                # Add a corresponding distilled policy for each policy in the checkpoint.
                previous_policy_ids = [
                    k for k in policies.keys() if k.startswith("ppo")
                ]
                policies_to_train.clear()
                for policy_id in previous_policy_ids:
                    load_policies_mapping[policy_id] = policy_id
                    policies[policy_id] = checkpoint_to_load_policies_config[
                        "multiagent"
                    ]["policies"][policy_id]
                    # policies[policy_id] = loaded_policy_dict[policy_id]
                    prev_model_config = policies[policy_id][3]["model"]
                    if (
                        prev_model_config.get("custom_model")
                        == "mbag_convolutional_model"
                        and model_config.get("custom_model")
                        != "mbag_convolutional_model"
                    ):
                        prev_model_config["custom_model_config"]["fake_state"] = True
                    distill_policy_id = f"{policy_id}_distilled"
                    policies[distill_policy_id] = (
                        None,
                        env.observation_space,
                        env.action_space,
                        {"model": model_config},
                    )
                    policies_to_train.append(distill_policy_id)
            config["multiagent"][
                "distillation_mapping_fn"
            ] = lambda policy_id: f"{policy_id}_distilled"
            config["multiagent"]["policies_to_train"] = policies_to_train

            # Remove extra config parameters.
            for key in list(config.keys()):
                if key not in DISTILLATION_DEFAULT_CONFIG:
                    del config[key]

        del env
        del loaded_policy_dict


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
    checkpoint_to_load_policies: Optional[str],
    load_policies_mapping: Dict[str, str],
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

    if checkpoint_to_load_policies is not None:
        _log.info(f"Initializing policies from {checkpoint_to_load_policies}")
        load_policies_from_checkpoint(
            checkpoint_to_load_policies,
            trainer,
            lambda policy_id: load_policies_mapping[policy_id],
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
