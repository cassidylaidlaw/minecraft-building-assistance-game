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
from gymnasium import spaces
from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import MultiAgentEnv
from ray.rllib.evaluation import Episode, SampleBatch
from ray.rllib.policy import TorchPolicy
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.replay_buffers import StorageUnit
from ray.rllib.utils.typing import MultiAgentPolicyConfigDict
from ray.tune.registry import ENV_CREATOR, _global_registry, get_trainable_cls
from sacred import SETTINGS as SACRED_SETTINGS
from sacred import Experiment
from sacred.config.custom_containers import DogmaticDict

import mbag
from mbag.agents.heuristic_agents import ALL_HEURISTIC_AGENTS
from mbag.environment.config import MbagConfigDict, MbagPlayerConfigDict, RewardSchedule
from mbag.environment.goals.demonstrations import DemonstrationsGoalGeneratorConfig
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
from mbag.rllib.alpha_zero.replay_buffer import (
    FixedMultiAgentReplayBuffer,
    PartialReplayBuffer,
)
from mbag.rllib.bc import BCConfig, BCTorchPolicy
from mbag.rllib.callbacks import MbagCallbacks
from mbag.rllib.data_augmentation import randomly_permute_block_types
from mbag.rllib.gail import MbagGAILConfig, MbagGAILTorchPolicy
from mbag.rllib.os_utils import available_cpu_count
from mbag.rllib.policies import MbagAgentPolicy
from mbag.rllib.ppo import MbagPPOConfig, MbagPPOTorchPolicy
from mbag.rllib.sacred_utils import (
    NoTypeAnnotationsFileStorageObserver,
    convert_dogmatics_to_standard,
)
from mbag.rllib.torch_models import (
    MbagConvolutionalModelConfig,
    MbagTransformerModelConfig,
    MbagUNetModelConfig,
)
from mbag.rllib.training_utils import (
    build_logger_creator,
    load_policies_from_checkpoint,
    load_trainer_config,
)

from .train_configs import make_named_configs

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
    randomize_first_episode_length = True
    truncate_on_no_progress_timesteps: Optional[int] = None
    num_players = 1
    evaluation_num_players = num_players
    width = 11
    height = 10
    depth = 10
    random_start_locations = True
    teleportation = True
    flying = True
    inf_blocks = True
    goal_visibility = [True] * num_players
    timestep_skip = [1] * num_players
    is_human = [False] * num_players
    goal_generator_config = {"subset": goal_subset}

    noop_reward: RewardSchedule = 0
    per_player_noop_reward: Optional[List[RewardSchedule]] = None
    get_resources_reward: RewardSchedule = 0
    per_player_get_resources_reward: Optional[List[RewardSchedule]] = None
    action_reward: RewardSchedule = 0
    per_player_action_reward: Optional[List[RewardSchedule]] = None
    incorrect_action_reward: RewardSchedule = 0
    per_player_incorrect_action_reward: Optional[List[RewardSchedule]] = None
    place_wrong_reward: RewardSchedule = -1
    per_player_place_wrong_reward: Optional[List[RewardSchedule]] = None
    own_reward_prop: RewardSchedule = 0
    per_player_own_reward_prop: Optional[List[RewardSchedule]] = None

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
            "rewards": {},
        }
        if per_player_noop_reward is not None:
            player_config["rewards"]["noop"] = per_player_noop_reward[player_index]
        if per_player_action_reward is not None:
            player_config["rewards"]["action"] = per_player_action_reward[player_index]
        if per_player_incorrect_action_reward is not None:
            player_config["rewards"]["incorrect_action"] = (
                per_player_incorrect_action_reward[player_index]
            )
        if per_player_place_wrong_reward is not None:
            player_config["rewards"]["place_wrong"] = per_player_place_wrong_reward[
                player_index
            ]
        if per_player_get_resources_reward is not None:
            player_config["rewards"]["get_resources"] = per_player_get_resources_reward[
                player_index
            ]
        if per_player_own_reward_prop is not None:
            player_config["rewards"]["own_reward_prop"] = per_player_own_reward_prop[
                player_index
            ]
        player_configs.append(player_config)

    environment_params: MbagConfigDict = {
        "num_players": num_players,
        "horizon": horizon,
        "randomize_first_episode_length": randomize_first_episode_length,
        "truncate_on_no_progress_timesteps": truncate_on_no_progress_timesteps,
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
            "incorrect_action": incorrect_action_reward,
            "place_wrong": place_wrong_reward,
            "get_resources": get_resources_reward,
            "own_reward_prop": own_reward_prop,
        },
        "abilities": {
            "teleportation": teleportation,
            "flying": flying,
            "inf_blocks": inf_blocks,
        },
    }
    # Convert Sacred DogmaticDicts and DogmaticLists to standard Python dicts and lists.
    environment_params = convert_dogmatics_to_standard(environment_params)

    env: MultiAgentEnv = _global_registry.get(ENV_CREATOR, environment_name)(
        environment_params
    )
    observation_space = cast(spaces.Dict, env.observation_space).spaces["player_0"]
    action_space = cast(spaces.Dict, env.action_space).spaces["player_0"]

    # Training
    num_workers = 2
    num_cpus_per_worker = 0.5
    num_envs = max(num_workers, 1)
    assert num_envs % max(num_workers, 1) == 0
    num_envs_per_worker = num_envs // max(num_workers, 1)
    input = "sampler"
    output = None
    seed = 0
    num_gpus = 1.0 if torch.cuda.is_available() else 0.0
    num_gpus_per_worker = 0.0
    ray_init_options = {}  # noqa: F841
    sample_freq = 1
    sample_batch_size = 5000
    train_batch_size = 5000
    sgd_minibatch_size = 512
    rollout_fragment_length = 1 if run == "BC" else horizon
    batch_mode = "truncate_episodes"
    simple_optimizer = True
    num_training_iters = 500  # noqa: F841
    lr = 1e-3
    lr_schedule = None
    weight_decay = 0
    grad_clip = 10
    gamma = 0.95
    gae_lambda = 0.98
    vf_share_layers = False
    vf_loss_coeff = 0 if run == "BC" else 1e-2
    entropy_coeff_start = 0 if "AlphaZero" in run else 0.01
    entropy_coeff_end = 0
    entropy_coeff_horizon = 1e5
    entropy_coeff_schedule = [
        [0, entropy_coeff_start],
        [entropy_coeff_horizon, entropy_coeff_end],
    ]
    kl_coeff = 0.2
    kl_target = 0.01
    clip_param = 0.05
    num_sgd_iter = 6
    anchor_policy_kl_coeff = 0.0
    anchor_policy_reverse_kl = False
    compress_observations = True
    use_replay_buffer = True
    replay_buffer_size = 10
    replay_buffer_storage_unit = StorageUnit.FRAGMENTS
    use_model_replay_buffer = False
    model_replay_buffer_size = replay_buffer_size
    model_replay_buffer_storage_probability = 0.1
    model_train_batch_size = train_batch_size
    use_critic = True
    use_goal_predictor = True
    other_agent_action_predictor_loss_coeff = 1.0
    reward_scale = 1.0
    pretrain = False
    strict_mode = False
    validation_participant_ids: List[int] = []
    validation_prop = 0
    permute_block_types: bool = False

    # MCTS
    mcts_batch_size = None
    puct_coefficient = 1.0
    sample_c_puct_every_timestep = True
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
    sample_from_full_support_policy = False
    explore_noops = True
    policy_loss_coeff = 1
    prev_policy_kl_coeff = 0
    goal_loss_coeff = 0.5
    prev_goal_kl_coeff = 0
    prev_goal_kl_coeff_schedule = None
    place_block_loss_coeff = 1
    place_block_loss_coeff_schedule = None
    predict_goal_using_next_state = False
    predict_goal_using_average = False
    predict_goal_using_future_states = False
    expected_own_reward_scale = 1.0
    expected_reward_shift = 0.0
    store_model_state_in_torch = False

    # Model
    model: str = "convolutional"
    max_seq_len = horizon
    embedding_size = 16
    position_embedding_size = 48
    position_embedding_angle = 10
    mask_goal = False
    num_inventory_obs = num_players
    mask_other_players = num_players == 1
    use_extra_features = not mask_goal
    use_fc_after_embedding = True
    num_conv_1_layers = 1
    num_layers = 1
    filter_size = 3
    hidden_channels = 16
    hidden_size = hidden_channels
    num_action_layers = 2
    num_value_layers = 2
    num_lstm_layers = 1
    mask_action_distribution = True
    # Line-of-sight masking is super slow with teleportation=True.
    line_of_sight_masking = not teleportation
    scale_obs = False
    vf_scale = 1.0
    dim_feedforward = hidden_size
    num_heads = 4
    norm_first = False
    use_separated_transformer = False
    interleave_lstm = False
    interleave_lstm_every = -1
    lstm_size = hidden_size
    use_prev_blocks = False
    use_prev_action = False
    use_prev_other_agent_action = False
    assert not use_prev_other_agent_action
    use_resnet = True
    use_groupnorm = False
    dropout = 0.0
    attention_resolutions = ()
    num_res_blocks = 1
    channel_mult = (1, 2, 4)
    use_scale_shift_norm = True
    resblock_updown = True
    use_lstm = False
    custom_action_dist = "categorical_no_inf"
    model_config = {
        "custom_model": f"mbag_{model}_model",
        "custom_action_dist": custom_action_dist,
        "max_seq_len": max_seq_len,
        "vf_share_layers": vf_share_layers,
    }
    if "convolutional" in model:
        conv_config: MbagConvolutionalModelConfig = {
            "env_config": cast(MbagConfigDict, dict(environment_params)),
            "num_inventory_obs": num_inventory_obs,
            "embedding_size": embedding_size,
            "use_extra_features": use_extra_features,
            "use_fc_after_embedding": use_fc_after_embedding,
            "mask_goal": mask_goal,
            "mask_other_players": mask_other_players,
            "num_conv_1_layers": num_conv_1_layers,
            "num_layers": num_layers,
            "use_resnet": use_resnet,
            "use_groupnorm": use_groupnorm,
            "dropout": dropout,
            "filter_size": filter_size,
            "hidden_size": hidden_size,
            "hidden_channels": hidden_channels,
            "num_action_layers": num_action_layers,
            "num_value_layers": num_value_layers,
            "use_prev_blocks": use_prev_blocks,
            "use_prev_action": use_prev_action,
            "mask_action_distribution": mask_action_distribution,
            "line_of_sight_masking": line_of_sight_masking,
            "scale_obs": scale_obs,
            "vf_scale": vf_scale,
            "num_value_layers": num_value_layers,
            "interleave_lstm_every": interleave_lstm_every,
            "lstm_size": lstm_size,
        }
        model_config["custom_model_config"] = conv_config
    elif "transformer" in model:
        transformer_config: MbagTransformerModelConfig = {
            "env_config": cast(MbagConfigDict, dict(environment_params)),
            "num_inventory_obs": num_inventory_obs,
            "embedding_size": embedding_size,
            "use_extra_features": use_extra_features,
            "use_fc_after_embedding": use_fc_after_embedding,
            "mask_goal": mask_goal,
            "mask_other_players": mask_other_players,
            "position_embedding_size": position_embedding_size,
            "position_embedding_angle": position_embedding_angle,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "num_heads": num_heads,
            "dropout": dropout,
            "norm_first": norm_first,
            "hidden_size": hidden_size,
            "num_action_layers": num_action_layers,
            "num_value_layers": num_value_layers,
            "num_lstm_layers": num_lstm_layers,
            "use_prev_blocks": use_prev_blocks,
            "use_prev_action": use_prev_action,
            "use_separated_transformer": use_separated_transformer,
            "interleave_lstm": interleave_lstm,
            "mask_action_distribution": mask_action_distribution,
            "line_of_sight_masking": line_of_sight_masking,
            "scale_obs": scale_obs,
            "vf_scale": vf_scale,
        }
        model_config["custom_model_config"] = transformer_config
    elif "unet" in model:
        unet_config: MbagUNetModelConfig = {
            "env_config": cast(MbagConfigDict, dict(environment_params)),
            "num_inventory_obs": num_inventory_obs,
            "embedding_size": embedding_size,
            "use_extra_features": use_extra_features,
            "use_fc_after_embedding": use_fc_after_embedding,
            "mask_goal": mask_goal,
            "mask_other_players": mask_other_players,
            "hidden_size": hidden_size,
            "hidden_channels": hidden_channels,
            "attention_resolutions": attention_resolutions,
            "num_res_blocks": num_res_blocks,
            "channel_mult": channel_mult,
            "num_heads": num_heads,
            "use_scale_shift_norm": use_scale_shift_norm,
            "resblock_updown": resblock_updown,
            "num_action_layers": num_action_layers,
            "num_value_layers": num_value_layers,
            "use_prev_blocks": use_prev_blocks,
            "use_prev_action": use_prev_action,
            "mask_action_distribution": mask_action_distribution,
            "line_of_sight_masking": line_of_sight_masking,
            "scale_obs": scale_obs,
            "vf_scale": vf_scale,
            "num_value_layers": num_value_layers,
            "use_lstm": use_lstm,
            "lstm_size": lstm_size,
        }
        model_config["custom_model_config"] = unet_config

    # Resume from checkpoint
    checkpoint_path = None  # noqa: F841
    checkpoint_to_load_policies = None

    # Maps policy IDs in checkpoint_to_load_policies to policy IDs here
    load_policies_mapping: Dict[str, str] = {}
    use_anchor_policy = False
    overwrite_loaded_policy_type = use_anchor_policy
    overwrite_loaded_policy_model = False
    load_config_from_checkpoint = not overwrite_loaded_policy_type
    if isinstance(load_policies_mapping, DogmaticDict):
        # Weird shim for sacred
        for key in load_policies_mapping.revelation():
            load_policies_mapping[key] = load_policies_mapping[key]

    if checkpoint_to_load_policies is not None:
        checkpoint_to_load_policies_config: AlgorithmConfig = load_trainer_config(
            checkpoint_to_load_policies
        )
        # Make sure the loaded policies use GPU if specified.
        checkpoint_to_load_policies_config.resources(
            num_gpus_per_worker=num_gpus_per_worker
        )

    # Multiagent
    heuristic: Optional[str] = None
    policy_ids: List[str]
    policy_mapping_fn: Callable[[str, Episode], str]
    if num_players == 1:
        policy_ids = ["human"]
        policy_mapping_fn = lambda agent_id, *args, **kwargs: "human"  # noqa: E731
    elif num_players == 2:
        policy_ids = ["human", "assistant"]
        if heuristic is not None:
            policy_ids[0] = heuristic

        def policy_mapping_fn(
            agent_id: str,
            episode=None,
            worker=None,
            policy_ids=convert_dogmatics_to_standard(policy_ids),
            *args,
            **kwargs,
        ):
            agent_index = int(agent_id[len("player_") :])
            if agent_index >= len(policy_ids):
                breakpoint()
            return policy_ids[agent_index]

    for player_index, policy_id in enumerate(policy_ids):
        environment_params["players"][player_index]["player_name"] = policy_id

    loaded_policy_dict: MultiAgentPolicyConfigDict = {}
    if checkpoint_to_load_policies is not None:
        unmapped_loaded_policy_dict = checkpoint_to_load_policies_config["multiagent"][
            "policies"
        ]
        for policy_id, old_policy_id in load_policies_mapping.items():
            loaded_policy_dict[policy_id] = unmapped_loaded_policy_dict[old_policy_id]

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
    elif run == "MbagGAIL":
        policy_class = MbagGAILTorchPolicy
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
            # Observation space may change from an agent trained alone to one
            # trained with other agents.
            policy_spec.observation_space = observation_space
            policies[policy_id] = policy_spec
            if overwrite_loaded_policy_type:
                policies[policy_id].policy_class = policy_class
            if overwrite_loaded_policy_model:
                policies[policy_id].config["model"] = model_config
        elif policy_id in policies_to_train:
            policies[policy_id] = PolicySpec(
                policy_class,
                observation_space,
                action_space,
                convert_dogmatics_to_standard(policy_config),
            )
        else:
            # Heuristic agent policy.
            mbag_agent = ALL_HEURISTIC_AGENTS[policy_id]({}, environment_params)
            policies[policy_id] = PolicySpec(
                MbagAgentPolicy,
                observation_space,
                action_space,
                {"mbag_agent": mbag_agent},
            )

    # Anchor policies for KL regularization
    anchor_policy_mapping = {}
    if use_anchor_policy:
        for policy_id in list(load_policies_mapping.keys()):
            anchor_policy_id = f"{policy_id}_anchor"
            policies[anchor_policy_id] = policies[policy_id]
            load_policies_mapping[anchor_policy_id] = load_policies_mapping[policy_id]
            anchor_policy_mapping[policy_id] = anchor_policy_id

    # Evaluation
    evaluation_num_workers = num_workers
    evaluation_interval = 5
    evaluation_duration = max(evaluation_num_workers, 1) * num_envs_per_worker
    evaluation_duration_unit = "episodes"
    evaluation_explore = False
    evaluation_config = {
        "input": "sampler",
        "explore": evaluation_explore,
        "env_config": {
            "randomize_first_episode_length": False,
            "num_players": evaluation_num_players,
            "players": player_configs[:evaluation_num_players],
        },
    }

    # Logging
    save_freq = 25  # noqa: F841
    log_dir = "data/logs"  # noqa: F841
    experiment_tag = None
    size_str = f"{width}x{height}x{depth}"
    players_str = "1_player" if num_players == 1 else f"{num_players}_players"
    experiment_name_parts = [run, players_str, size_str, goal_generator]
    if heuristic is not None:
        experiment_name_parts.append(heuristic)
    if experiment_tag is not None:
        experiment_name_parts.append(experiment_tag)
    if validation_participant_ids:
        experiment_name_parts.append(
            "validation_" + "_".join(map(str, validation_participant_ids))
        )
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
        policies_to_train=convert_dogmatics_to_standard(policies_to_train),
    )
    config.callbacks(MbagCallbacks)
    config.offline_data(
        input_=input,
        actions_in_input_normalized=input != "sampler",
        output=output,
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
        optimizer=dict(weight_decay=weight_decay),
        _enable_learner_api=False,
    )
    config.simple_optimizer = simple_optimizer

    if "PPO" in run or "GAIL" in run:
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
            entropy_coeff_schedule=convert_dogmatics_to_standard(
                entropy_coeff_schedule
            ),
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
                place_block_loss_coeff_schedule=convert_dogmatics_to_standard(
                    place_block_loss_coeff_schedule
                ),
                reward_scale=reward_scale,
                anchor_policy_mapping=anchor_policy_mapping,
                anchor_policy_kl_coeff=anchor_policy_kl_coeff,
                anchor_policy_reverse_kl=anchor_policy_reverse_kl,
            )
        if isinstance(config, MbagGAILConfig):
            demonstration_input = None
            train_discriminator_on_separate_batch = False
            discriminator_num_sgd_iter = num_sgd_iter

            config.training(
                demonstration_input=demonstration_input,
                validation_participant_ids=validation_participant_ids,
                train_discriminator_on_separate_batch=train_discriminator_on_separate_batch,
                discriminator_num_sgd_iter=discriminator_num_sgd_iter,
            )
        environment_params
    elif "AlphaZero" in run:
        assert isinstance(config, MbagAlphaZeroConfig)
        assert reward_scale == 1.0, "Reward scaling not supported for AlphaZero"
        mcts_config: Dict[str, Any] = {
            "puct_coefficient": puct_coefficient,
            "sample_c_puct_every_timestep": sample_c_puct_every_timestep,
            "num_simulations": num_simulations,
            "temperature": temperature,
            "temperature_schedule": None,
            "dirichlet_epsilon": dirichlet_epsilon,
            "dirichlet_noise": dirichlet_noise,
            "dirichlet_action_subtype_noise_multiplier": dirichlet_action_subtype_noise_multiplier,
            "argmax_tree_policy": argmax_tree_policy,
            "add_dirichlet_noise": add_dirichlet_noise,
            "prior_temperature": prior_temperature,
            "init_q_with_max": init_q_with_max,
            "use_bilevel_action_selection": use_bilevel_action_selection,
            "fix_bilevel_action_selection": fix_bilevel_action_selection,
            "sample_from_full_support_policy": sample_from_full_support_policy,
            "explore_noops": explore_noops,
            "predict_goal_using_next_state": predict_goal_using_next_state,
            "predict_goal_using_average": predict_goal_using_average,
            "predict_goal_using_future_states": predict_goal_using_future_states,
            "store_model_state_in_torch": store_model_state_in_torch,
        }
        if temperature_start != temperature_end:
            mcts_config["temperature_schedule"] = [
                (0, temperature_start),
                (temperature_horizon, temperature_end),
            ]
        config.training(
            lr=lr,
            lr_schedule=convert_dogmatics_to_standard(lr_schedule),
            grad_clip=grad_clip,
            gamma=gamma,
            train_batch_size=train_batch_size,
            model_train_batch_size=model_train_batch_size,
            sgd_minibatch_size=sgd_minibatch_size,
            num_sgd_iter=num_sgd_iter,
            policy_loss_coeff=policy_loss_coeff,
            prev_policy_kl_coeff=prev_policy_kl_coeff,
            vf_loss_coeff=vf_loss_coeff,
            prev_goal_kl_coeff=prev_goal_kl_coeff,
            prev_goal_kl_coeff_schedule=convert_dogmatics_to_standard(
                prev_goal_kl_coeff_schedule
            ),
            entropy_coeff_schedule=convert_dogmatics_to_standard(
                entropy_coeff_schedule
            ),
            sample_freq=sample_freq,
            sample_batch_size=sample_batch_size,
            ranked_rewards={"enable": False},
            num_steps_sampled_before_learning_starts=0,
            mcts_config=convert_dogmatics_to_standard(mcts_config),
            mcts_batch_size=mcts_batch_size,
            use_critic=use_critic,
            use_goal_predictor=use_goal_predictor,
            expected_own_reward_scale=expected_own_reward_scale,
            expected_reward_shift=expected_reward_shift,
            other_agent_action_predictor_loss_coeff=other_agent_action_predictor_loss_coeff,
            use_replay_buffer=use_replay_buffer,
            replay_buffer_config={
                "type": FixedMultiAgentReplayBuffer,
                "capacity": replay_buffer_size,
                "storage_unit": replay_buffer_storage_unit,
                "replay_sequence_override": False,
                "replay_sequence_length": 0,
                "replay_zero_init_states": False,
            },
            use_model_replay_buffer=use_model_replay_buffer,
            model_replay_buffer_config={
                "type": FixedMultiAgentReplayBuffer,
                "capacity": model_replay_buffer_size,
                "storage_unit": replay_buffer_storage_unit,
                "replay_sequence_override": False,
                "replay_sequence_length": 0,
                "replay_zero_init_states": False,
                "underlying_buffer_config": {
                    "type": PartialReplayBuffer,
                    "storage_probability": model_replay_buffer_storage_probability,
                },
            },
            pretrain=pretrain,
            anchor_policy_mapping=anchor_policy_mapping,
            anchor_policy_kl_coeff=anchor_policy_kl_coeff,
            _strict_mode=strict_mode,
        )
        evaluation_mcts_config = dict(mcts_config)
        evaluation_mcts_config["argmax_tree_policy"] = True
        evaluation_mcts_config["add_dirichlet_noise"] = False
        config.evaluation(
            evaluation_config=convert_dogmatics_to_standard(
                {
                    "mcts_config": evaluation_mcts_config,
                }
            )
        )
    elif run == "BC":
        assert isinstance(config, BCConfig)
        config.training(
            lr=lr,
            lr_schedule=convert_dogmatics_to_standard(lr_schedule),
            gamma=gamma,
            train_batch_size=train_batch_size,
            sgd_minibatch_size=sgd_minibatch_size,
            num_sgd_iter=num_sgd_iter,
            grad_clip=grad_clip,
            entropy_coeff=entropy_coeff_start,
            vf_loss_coeff=vf_loss_coeff,
            validation_participant_ids=convert_dogmatics_to_standard(
                validation_participant_ids
            ),
            validation_prop=validation_prop,
        )

    if permute_block_types:
        assert isinstance(config, BCConfig) or isinstance(config, MbagGAILConfig)
        keep_dirt_at_ground_level = inf_blocks

        def data_augmentation(
            batch: SampleBatch,
            env_config=config.env_config,
            keep_dirt_at_ground_level=keep_dirt_at_ground_level,
        ) -> SampleBatch:
            batch.decompress_if_needed()
            return randomly_permute_block_types(
                batch,
                flat_actions=True,
                flat_observations=True,
                env_config=env_config,
                keep_dirt_at_ground_level=keep_dirt_at_ground_level,
            )

        config.training(data_augmentation=data_augmentation)

    if isinstance(config, MbagGAILConfig):
        # Set goal generator to use goals from the demonstrations since otherwise
        # GAIL won't work.
        environment_params["goal_generator"] = "demonstrations"
        demonstrations_goal_generator_config: DemonstrationsGoalGeneratorConfig = {
            "demonstration_input": config.demonstration_input,
            "data_augmentation": config.data_augmentation,
        }
        environment_params["goal_generator_config"] = (
            demonstrations_goal_generator_config
        )
        config.environment(environment_name, env_config=dict(environment_params))

    del env
    del loaded_policy_dict

    observer = NoTypeAnnotationsFileStorageObserver(experiment_dir)
    ex.observers.append(observer)

    # For testing
    _no_train = False  # noqa: F841

    # Extra args that are ignored here but used in some of the named configs.
    checkpoint_name = None  # noqa: F841
    data_split = None  # noqa: F841
    lr_start = None  # noqa: F841
    puct_str = None  # noqa: F841


make_named_configs(ex)


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
    ray_init_options,
    _no_train: bool,
    _log: Logger,
):
    temp_dir = tempfile.mkdtemp()
    os.environ["RAY_AIR_NEW_PERSISTENCE_MODE"] = "0"
    ray.init(
        num_cpus=available_cpu_count(),
        ignore_reinit_error=True,
        include_dashboard=False,
        _temp_dir=temp_dir,
        **ray_init_options,
    )

    mbag.logger.setLevel(_log.getEffectiveLevel())

    algorithm_class: Type[Algorithm] = get_trainable_cls(run)
    trainer = algorithm_class(
        config,
        logger_creator=build_logger_creator(observer.dir),
    )

    # Limit CUDA memory usage based on num_gpus and num_gpus_per_worker.
    if torch.cuda.is_available():
        num_gpus_per_worker: float = config["num_gpus_per_worker"]
        if trainer.workers is not None and num_gpus_per_worker > 0:
            trainer.workers.foreach_worker(
                lambda worker: torch.cuda.set_per_process_memory_fraction(
                    float(num_gpus_per_worker)
                )
            )
        if trainer.evaluation_workers is not None and num_gpus_per_worker > 0:
            trainer.evaluation_workers.foreach_worker(
                lambda worker: torch.cuda.set_per_process_memory_fraction(
                    float(num_gpus_per_worker)
                )
            )
        torch.cuda.set_per_process_memory_fraction(float(config["num_gpus"]))

    if checkpoint_to_load_policies is not None:
        _log.info(f"Initializing policies from {checkpoint_to_load_policies}")
        load_policies_from_checkpoint(
            checkpoint_to_load_policies,
            trainer,
            lambda policy_id: load_policies_mapping.get(policy_id),
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
                if "_optimizer_variables" in policy_state:
                    del policy_state["_optimizer_variables"]
            return old_set_state(checkpoint_data)

        trainer.__setstate__ = new_set_state  # type: ignore

        trainer.restore(checkpoint_path)

    result = None
    if not _no_train:
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
