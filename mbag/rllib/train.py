import faulthandler
import os
import signal
import sys
import tempfile
from datetime import datetime
from logging import Logger
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Type, Union, cast

import ray
import torch
from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import MultiAgentEnv
from ray.rllib.evaluation import Episode
from ray.rllib.policy import TorchPolicy
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.replay_buffers import StorageUnit
from ray.rllib.utils.typing import MultiAgentPolicyConfigDict
from ray.tune.registry import ENV_CREATOR, _global_registry, get_trainable_cls
from sacred import SETTINGS as SACRED_SETTINGS
from sacred import Experiment
from sacred.config.custom_containers import DogmaticDict
from sacred.observers import FileStorageObserver
from typing_extensions import Literal

from mbag.agents.heuristic_agents import ALL_HEURISTIC_AGENTS
from mbag.environment.config import (
    MbagConfigDict,
    MbagPlayerConfigDict,
    RewardsConfigDict,
    RewardsConfigDictKey,
)
from mbag.environment.goals.filters import DensityFilterConfig, MinSizeFilterConfig
from mbag.environment.goals.goal_transform import (
    GoalTransformSpec,
    TransformedGoalGenerator,
    TransformedGoalGeneratorConfig,
)
from mbag.environment.goals.transforms import (
    AreaSampleTransformConfig,
    CropLowDensityBottomLayersTransformConfig,
    CropTransformConfig,
)
from mbag.rllib.alpha_zero import MbagAlphaZeroConfig, MbagAlphaZeroPolicy
from mbag.rllib.bc import BCConfig, BCTorchPolicy
from mbag.rllib.sacred_utils import convert_dogmatics_to_standard

from .callbacks import MbagCallbacks
from .os_utils import available_cpu_count
from .policies import MbagAgentPolicy, MbagPPOConfig, MbagPPOTorchPolicy
from .torch_models import (
    MbagRecurrentConvolutionalModelConfig,
    MbagTransformerModelConfig,
)
from .training_utils import (
    build_logger_creator,
    load_policies_from_checkpoint,
    load_trainer_config,
)

if TYPE_CHECKING:
    from typing import List
else:
    # Deal with weird sacred serialization issue.
    if sys.version_info >= (3, 9):
        List = list
    else:
        from typing import Sequence as List


ex = Experiment("train_mbag")
SACRED_SETTINGS.CONFIG.READ_ONLY_CONFIG = False

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Useful for debugging when training freezes.
faulthandler.register(signal.SIGUSR1)


@ex.config
def sacred_config(_log):  # noqa
    run = "MbagPPO"
    config: AlgorithmConfig = get_trainable_cls(run).get_default_config()

    # Environment
    environment_name = "MBAGFlatActions-v1"
    goal_generator = "craftassist"
    goal_subset = "train"
    horizon = 1000
    randomize_first_episode_length = False
    num_players = 1
    height = 12
    width = 12
    depth = 12
    random_start_locations = False
    noop_reward = 0
    get_resources_reward = 0
    action_reward = 0
    place_wrong_reward = -1
    teleportation = True
    flying = True
    inf_blocks = True
    goal_visibility = [True] * num_players
    timestep_skip = [1] * num_players
    is_human = [False] * num_players
    own_reward_prop = 0
    goal_generator_config = {"subset": goal_subset}

    goal_transforms: List[GoalTransformSpec] = []
    uniform_block_type = False
    min_density = 0
    max_density = 1
    extract_largest_cc = True
    extract_largest_cc_connectivity = 18
    force_single_cc = True
    force_single_cc_connectivity = 18
    crop_air = True
    crop_low_density_bottom_layers = True
    crop = False
    crop_density_threshold = 0.25
    area_sample = True
    wall = False
    mirror = False
    min_width, min_height, min_depth = 4, 4, 4
    remove_invisible_non_dirt = False
    if uniform_block_type:
        goal_transforms.append({"transform": "uniform_block_type"})
    if extract_largest_cc:
        goal_transforms.append(
            {
                "transform": "largest_cc",
                "config": {"connectivity": extract_largest_cc_connectivity},
            }
        )
    if crop_air:
        goal_transforms.append({"transform": "crop_air"})
    if crop_low_density_bottom_layers:
        crop_low_density_config: CropLowDensityBottomLayersTransformConfig = {
            "density_threshold": 0.1
        }
        goal_transforms.append(
            {
                "transform": "crop_low_density_bottom_layers",
                "config": crop_low_density_config,
            }
        )
    min_size_config: MinSizeFilterConfig = {
        "min_size": (min_width, min_height, min_depth)
    }
    goal_transforms.append({"transform": "min_size_filter", "config": min_size_config})
    if crop or wall:
        crop_config: CropTransformConfig = {
            "density_threshold": 1000 if wall else crop_density_threshold,
            "tethered_to_ground": True,
            "wall": wall,
        }
        goal_transforms.append({"transform": "crop", "config": crop_config})
    if area_sample:
        area_sample_config: AreaSampleTransformConfig = {
            "interpolate": True,
            "interpolation_order": 1,
            "max_scaling_factor": 2,
            "max_scaling_factor_ratio": 1.5,
            "preserve_paths": True,
            "scale_y_independently": True,
        }
        goal_transforms.append(
            {"transform": "area_sample", "config": area_sample_config}
        )
    density_config: DensityFilterConfig = {
        "min_density": min_density,
        "max_density": max_density,
    }
    goal_transforms.append({"transform": "density_filter", "config": density_config})
    goal_transforms.append({"transform": "randomly_place"})
    goal_transforms.append({"transform": "add_grass"})
    if remove_invisible_non_dirt:
        goal_transforms.append({"transform": "remove_invisible_non_dirt"})
    if mirror:
        goal_transforms.append({"transform": "mirror"})
    if force_single_cc:
        goal_transforms.append(
            {
                "transform": "single_cc_filter",
                "config": {"connectivity": force_single_cc_connectivity},
            }
        )

    transformed_goal_generator_config: TransformedGoalGeneratorConfig = {
        "goal_generator": goal_generator,
        "goal_generator_config": goal_generator_config,
        "transforms": goal_transforms,
    }

    player_configs: List[MbagPlayerConfigDict] = []
    for player_index in range(num_players):
        player_config: MbagPlayerConfigDict = {
            "goal_visible": goal_visibility[player_index],
            "timestep_skip": timestep_skip[player_index],
            "is_human": is_human[player_index],
        }
        player_configs.append(player_config)

    environment_params: MbagConfigDict = {
        "num_players": num_players,
        "horizon": horizon,
        "randomize_first_episode_length": randomize_first_episode_length,
        "world_size": (width, height, depth),
        "random_start_locations": random_start_locations,
        "goal_generator": TransformedGoalGenerator,
        "goal_generator_config": transformed_goal_generator_config,
        "malmo": {
            "use_malmo": False,
        },
        "players": player_configs,
        "rewards": {
            "noop": noop_reward,
            "action": action_reward,
            "place_wrong": place_wrong_reward,
            "own_reward_prop": own_reward_prop,
            "get_resources": get_resources_reward,
        },
        "abilities": {
            "teleportation": teleportation,
            "flying": flying,
            "inf_blocks": inf_blocks,
        },
    }
    # Convert Sacred DogmaticDicts and DogmaticLists to standard Python dicts and lists.
    environment_params = convert_dogmatics_to_standard(environment_params)
    environment_params["rewards"] = _format_reward_config(environment_params["rewards"])

    env: MultiAgentEnv = _global_registry.get(ENV_CREATOR, environment_name)(
        environment_params
    )

    # Training
    num_workers = 2
    num_cpus_per_worker = 0.5
    num_envs = max(num_workers, 1)
    assert num_envs % max(num_workers, 1) == 0
    num_envs_per_worker = num_envs // max(num_workers, 1)
    input = "sampler"
    seed = 0
    num_gpus = 1 if torch.cuda.is_available() else 0
    num_gpus_per_worker = 0
    sample_batch_size = 5000
    train_batch_size = 5000
    sgd_minibatch_size = 512
    rollout_fragment_length = horizon
    batch_mode = "truncate_episodes"
    simple_optimizer = True
    num_training_iters = 500  # noqa: F841
    lr = 1e-3
    lr_schedule = lr
    grad_clip = 10
    gamma = 0.95
    gae_lambda = 0.98
    vf_share_layers = False
    vf_loss_coeff = 1e-2
    entropy_coeff_start = 0 if "AlphaZero" in run else 0.01
    entropy_coeff_end = 0
    entropy_coeff_horizon = 1e5
    kl_coeff = 0.2
    kl_target = 0.01
    clip_param = 0.05
    num_sgd_iter = 6
    compress_observations = True
    use_replay_buffer = True
    replay_buffer_size = 10
    use_critic = True
    use_goal_predictor = True
    other_agent_action_predictor_loss_coeff = 1.0
    reward_scale = 1.0
    pretrain = False
    strict_mode = False

    # MCTS
    puct_coefficient = 1.0
    num_simulations = 30
    temperature = 1.5
    temperature_start = temperature
    temperature_end = temperature
    temperature_horizon = max(train_batch_size, sample_batch_size) * num_training_iters
    dirichlet_epsilon = 0.25
    argmax_tree_policy = False
    add_dirichlet_noise = True
    dirichlet_noise = 0.25
    # If using bi-level action selection, the alpha parameter for the Dirichlet noise
    # added to the second stage of action selection (after the action type is chosen)
    # is dynamically set to dirichlet_action_subtype_noise_multiplier / num_valid_actions,
    # where num_valid_actions is the number of valid actions at the current state.
    dirichlet_action_subtype_noise_multiplier = 10
    prior_temperature = 1.0
    init_q_with_max = False
    use_bilevel_action_selection = True
    fix_bilevel_action_selection = False
    goal_loss_coeff, place_block_loss_coeff = 0.5, 1

    # Model
    model: Literal["convolutional", "recurrent_convolutional", "transformer"] = (
        "convolutional"
    )
    max_seq_len = horizon
    embedding_size = 8
    position_embedding_size = 18
    mask_goal = False
    use_extra_features = not mask_goal
    num_conv_1_layers = 1
    num_layers = 1
    filter_size = 3
    hidden_channels = 16
    hidden_size = hidden_channels
    num_action_layers = 2
    num_value_layers = 2
    use_per_location_lstm = False
    mask_action_distribution = True
    line_of_sight_masking = False
    scale_obs = False
    num_heads = 4
    use_separated_transformer = False
    use_resnet = False
    num_unet_layers = 0
    unet_grow_factor = 2
    unet_use_bn = False
    custom_action_dist = "categorical_no_inf"
    model_config = {
        "custom_model": f"mbag_{model}_model",
        "custom_action_dist": custom_action_dist,
        "max_seq_len": max_seq_len,
        "vf_share_layers": vf_share_layers,
    }
    if "convolutional" in model:
        conv_config: MbagRecurrentConvolutionalModelConfig = {
            "env_config": cast(MbagConfigDict, dict(environment_params)),
            "embedding_size": embedding_size,
            "use_extra_features": use_extra_features,
            "mask_goal": mask_goal,
            "num_conv_1_layers": num_conv_1_layers,
            "num_layers": num_layers,
            "use_resnet": use_resnet,
            "filter_size": filter_size,
            "hidden_channels": hidden_channels,
            "num_action_layers": num_action_layers,
            "num_value_layers": num_value_layers,
            "use_per_location_lstm": use_per_location_lstm,
            "mask_action_distribution": mask_action_distribution,
            "line_of_sight_masking": line_of_sight_masking,
            "scale_obs": scale_obs,
            "num_unet_layers": num_unet_layers,
            "unet_grow_factor": unet_grow_factor,
            "unet_use_bn": unet_use_bn,
            "num_value_layers": num_value_layers,
        }
        model_config["custom_model_config"] = conv_config
    elif "transformer" in model:
        transformer_config: MbagTransformerModelConfig = {
            "env_config": cast(MbagConfigDict, dict(environment_params)),
            "embedding_size": embedding_size,
            "use_extra_features": use_extra_features,
            "mask_goal": mask_goal,
            "position_embedding_size": position_embedding_size,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "hidden_size": hidden_size,
            "num_action_layers": num_action_layers,
            "num_value_layers": num_value_layers,
            "use_per_location_lstm": use_per_location_lstm,
            "use_separated_transformer": use_separated_transformer,
            "mask_action_distribution": mask_action_distribution,
            "line_of_sight_masking": line_of_sight_masking,
            "scale_obs": scale_obs,
        }
        model_config["custom_model_config"] = transformer_config

    # Resume from checkpoint
    checkpoint_path = None  # noqa: F841
    checkpoint_to_load_policies = None

    # Maps policy IDs in checkpoint_to_load_policies to policy IDs here
    load_policies_mapping: Dict[str, str] = {}
    overwrite_loaded_policy_type = False
    load_config_from_checkpoint = not overwrite_loaded_policy_type
    if isinstance(load_policies_mapping, DogmaticDict):
        # Weird shim for sacred
        for key in load_policies_mapping.revelation():
            load_policies_mapping[key] = load_policies_mapping[key]

    if checkpoint_to_load_policies is not None:
        checkpoint_to_load_policies_config: AlgorithmConfig = load_trainer_config(
            checkpoint_to_load_policies
        )

    # Multiagent
    heuristic: Optional[str] = None
    multiagent_mode: Literal["self_play", "cross_play"] = "self_play"
    policy_ids: List[str]
    policy_mapping_fn: Callable[[str, Episode], str]
    if multiagent_mode == "self_play":
        policy_ids = ["human"]
        policy_mapping_fn = lambda agent_id, *args, **kwargs: "human"  # noqa: E731
    elif multiagent_mode == "cross_play":
        assert num_players == 2
        policy_ids = ["human", "assistant"]
        if heuristic is not None:
            policy_ids[-1] = heuristic

        def policy_mapping_fn(
            agent_id: str,
            episode=None,
            worker=None,
            policy_ids=policy_ids,
            *args,
            **kwargs,
        ):
            agent_index = int(agent_id[len("player_") :])
            return policy_ids[agent_index]

    for player_index, policy_id in enumerate(policy_ids):
        environment_params["players"][player_index]["player_name"] = policy_id

    loaded_policy_dict: MultiAgentPolicyConfigDict = {}
    if checkpoint_to_load_policies is not None:
        unmapped_loaded_policy_dict = checkpoint_to_load_policies_config["multiagent"][
            "policies"
        ]
        for old_policy_id, new_policy_id in load_policies_mapping.items():
            loaded_policy_dict[new_policy_id] = unmapped_loaded_policy_dict[
                old_policy_id
            ]

    policies_to_train = []
    for policy_id in policy_ids:
        if policy_id in ["human", "assistant"] and policy_id not in loaded_policy_dict:
            policies_to_train.append(policy_id)

    policies: MultiAgentPolicyConfigDict = {}
    policy_class: Union[None, Type[TorchPolicy], Type[TorchPolicyV2]] = None
    if "PPO" in run:
        policy_class = MbagPPOTorchPolicy
    elif "AlphaZero" in run:
        policy_class = MbagAlphaZeroPolicy
    elif run == "BC":
        policy_class = BCTorchPolicy
    policy_config: Dict[str, Any] = {
        "model": model_config,
        "goal_loss_coeff": goal_loss_coeff,
    }
    for policy_id in policy_ids:
        if policy_id in loaded_policy_dict:
            policy_spec = loaded_policy_dict[policy_id]
            if not isinstance(loaded_policy_dict, PolicySpec):
                policy_spec = PolicySpec(*cast(tuple, policy_spec))
            if load_config_from_checkpoint:
                policy_spec.config = (
                    checkpoint_to_load_policies_config.copy().update_from_dict(
                        policy_spec.config
                    )
                )
                policy_spec.config.environment(env_config=dict(environment_params))
            policies[policy_id] = policy_spec
            if overwrite_loaded_policy_type:
                policies[policy_id].policy_class = policy_class
        elif policy_id in policies_to_train:
            policies[policy_id] = PolicySpec(
                policy_class,
                env.observation_space,
                env.action_space,
                policy_config,
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

    # Evaluation
    evaluation_num_workers = num_workers
    evaluation_interval = 5
    evaluation_duration = max(evaluation_num_workers, 1)
    evaluation_duration_unit = "episodes"
    evaluation_explore = False
    evaluation_config = {
        "input": "sampler",
        "explore": evaluation_explore,
    }

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
    experiment_name_parts.append(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    experiment_dir = os.path.join(log_dir, *experiment_name_parts)

    config.framework("torch")
    config.rollouts(
        num_rollout_workers=num_workers,
        num_envs_per_worker=num_envs_per_worker,
        rollout_fragment_length=rollout_fragment_length,
        batch_mode=batch_mode,
        compress_observations=compress_observations,
    )
    config.resources(
        num_cpus_per_worker=num_cpus_per_worker,
        num_gpus=num_gpus,
        num_gpus_per_worker=num_gpus_per_worker,
    )
    config.debugging(seed=seed)
    config.environment(environment_name, env_config=dict(environment_params))
    config.multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=policies_to_train,
    )
    config.callbacks(MbagCallbacks)
    config.offline_data(
        input_=input,
        actions_in_input_normalized=input != "sampler",
    )
    config.evaluation(
        evaluation_interval=evaluation_interval,
        evaluation_num_workers=evaluation_num_workers,
        evaluation_config=evaluation_config,
        evaluation_duration=evaluation_duration,
        evaluation_duration_unit=evaluation_duration_unit,
    )
    config.rl_module(_enable_rl_module_api=False)
    config.training(
        _enable_learner_api=False,
    )
    config.simple_optimizer = simple_optimizer

    if "PPO" in run:
        assert isinstance(config, PPOConfig)
        config.training(
            lr=lr,
            lr_schedule=convert_dogmatics_to_standard(lr_schedule),
            gamma=gamma,
            train_batch_size=train_batch_size,
            sgd_minibatch_size=sgd_minibatch_size,
            num_sgd_iter=num_sgd_iter,
            vf_loss_coeff=vf_loss_coeff,
            vf_clip_param=float("inf"),
            entropy_coeff_schedule=[
                [0, entropy_coeff_start],
                [entropy_coeff_horizon, entropy_coeff_end],
            ],
            grad_clip=grad_clip,
            lambda_=gae_lambda,
            kl_coeff=kl_coeff,
            kl_target=kl_target,
            clip_param=clip_param,
        )
        if isinstance(config, MbagPPOConfig):
            config.training(
                goal_loss_coeff=goal_loss_coeff,
                place_block_loss_coeff=place_block_loss_coeff,
                reward_scale=reward_scale,
            )
    elif "AlphaZero" in run:
        assert isinstance(config, MbagAlphaZeroConfig)
        assert reward_scale == 1.0, "Reward scaling not supported for AlphaZero"
        mcts_config = {
            "puct_coefficient": puct_coefficient,
            "num_simulations": num_simulations,
            "temperature": temperature,
            "temperature_schedule": [
                (0, temperature_start),
                (temperature_horizon, temperature_end),
            ],
            "dirichlet_epsilon": dirichlet_epsilon,
            "dirichlet_noise": dirichlet_noise,
            "dirichlet_action_subtype_noise_multiplier": dirichlet_action_subtype_noise_multiplier,
            "argmax_tree_policy": argmax_tree_policy,
            "add_dirichlet_noise": add_dirichlet_noise,
            "prior_temperature": prior_temperature,
            "init_q_with_max": init_q_with_max,
            "use_bilevel_action_selection": use_bilevel_action_selection,
            "fix_bilevel_action_selection": fix_bilevel_action_selection,
        }
        config.training(
            lr=lr,
            lr_schedule=convert_dogmatics_to_standard(lr_schedule),
            gamma=gamma,
            train_batch_size=train_batch_size,
            sgd_minibatch_size=sgd_minibatch_size,
            num_sgd_iter=num_sgd_iter,
            vf_loss_coeff=vf_loss_coeff,
            entropy_coeff_schedule=[
                (0, entropy_coeff_start),
                (entropy_coeff_horizon, entropy_coeff_end),
            ],
            sample_batch_size=sample_batch_size,
            ranked_rewards={"enable": False},
            num_steps_sampled_before_learning_starts=0,
            mcts_config=mcts_config,
            use_critic=use_critic,
            use_goal_predictor=use_goal_predictor,
            other_agent_action_predictor_loss_coeff=other_agent_action_predictor_loss_coeff,
            use_replay_buffer=use_replay_buffer,
            replay_buffer_config={
                "type": "MultiAgentReplayBuffer",
                "capacity": replay_buffer_size,
                "storage_unit": StorageUnit.FRAGMENTS,
            },
            pretrain=pretrain,
            _strict_mode=strict_mode,
        )
        evaluation_mcts_config = dict(mcts_config)
        evaluation_mcts_config["argmax_tree_policy"] = True
        evaluation_mcts_config["add_dirichlet_noise"] = False
        config.evaluation(
            evaluation_config={
                "mcts_config": evaluation_mcts_config,
            }
        )
    elif run == "BC":
        assert isinstance(config, BCConfig)
        validation_prop = 0
        config.training(
            lr=lr,
            gamma=gamma,
            train_batch_size=train_batch_size,
            sgd_minibatch_size=sgd_minibatch_size,
            num_sgd_iter=num_sgd_iter,
            grad_clip=grad_clip,
            entropy_coeff=entropy_coeff_start,
            validation_prop=validation_prop,
        )

    del env
    del loaded_policy_dict

    observer = FileStorageObserver(experiment_dir)
    ex.observers.append(observer)


def _format_reward_config(reward_config: RewardsConfigDict) -> RewardsConfigDict:
    formatted_reward_config: RewardsConfigDict = {}
    for key, value in reward_config.items():
        key = cast(RewardsConfigDictKey, key)
        if isinstance(value, (list, tuple)):
            for points in value:
                if (
                    not isinstance(points, (list, tuple))
                    or len(points) != 2
                    or int(points[0]) != points[0]
                ):
                    raise ValueError(
                        f"Reward config for {key} must be a number or a "
                        "list/tuple of (timestep: int, value: float) tuples. "
                        f"Got {value}"
                    )

            formatted_reward_config[key] = [
                (int(points[0]), float(points[1])) for points in value
            ]
        elif isinstance(value, (float, int)):
            formatted_reward_config[key] = float(value)
        else:
            raise ValueError(
                f"Reward config for {key} must be a number or a "
                f"list/tuple of (timestep, value) tuples. Got {value}"
            )
    return formatted_reward_config


@ex.automain
def main(
    config: AlgorithmConfig,
    run,
    num_training_iters,
    save_freq,
    checkpoint_path: Optional[str],
    checkpoint_to_load_policies: Optional[str],
    load_policies_mapping: Dict[str, str],
    observer,
    _log: Logger,
):
    temp_dir = tempfile.mkdtemp()
    os.environ["RAY_AIR_NEW_PERSISTENCE_MODE"] = "0"
    ray.init(
        num_cpus=available_cpu_count(),
        ignore_reinit_error=True,
        include_dashboard=False,
        _temp_dir=temp_dir,
    )

    algorithm_class: Type[Algorithm] = get_trainable_cls(run)
    trainer = algorithm_class(
        config,
        logger_creator=build_logger_creator(observer.dir),
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

        old_set_state = trainer.__setstate__

        def new_set_state(checkpoint_data):
            # Remove config information from checkpoint_data so we don't override
            # the current config.
            if "config" in checkpoint_data:
                del checkpoint_data["config"]
            for policy_state in checkpoint_data["worker"]["policy_states"].values():
                if "policy_spec" in policy_state:
                    del policy_state["policy_spec"]
            return old_set_state(checkpoint_data)

        trainer.__setstate__ = new_set_state  # type: ignore

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

    trainer.stop()

    if result is None:
        result = {}
    result["final_checkpoint"] = checkpoint
    return result
