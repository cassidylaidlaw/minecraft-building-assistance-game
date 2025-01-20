import torch
from sacred import Experiment

# flake8: noqa: F841


def make_named_configs(ex: Experiment):

    @ex.named_config
    def ppo_human():
        run = "MbagPPO"
        goal_generator = "craftassist"
        width = 11
        height = 10
        depth = 10
        randomize_first_episode_length = True
        random_start_locations = True
        num_training_iters = 100
        horizon = 1500
        teleportation = False
        inf_blocks = True
        entropy_coeff_start = 0.03
        entropy_coeff_end = 0.03
        num_workers = 20
        num_envs_per_worker = 32
        num_gpus = 0.5
        num_gpus_per_worker = 0.7 / max(num_workers, 1)
        train_batch_size = 64000
        lr = 3e-4
        kl_target = 0.01
        num_sgd_iter = 3
        rollout_fragment_length = 100
        use_extra_features = True
        model = "convolutional"
        filter_size = 5
        hidden_channels = 64
        sgd_minibatch_size = 512
        num_layers = 6
        scale_obs = True
        vf_share_layers = True
        vf_loss_coeff = 0.01
        place_block_loss_coeff = 0
        evaluation_num_workers = 0
        evaluation_interval = None
        gamma = 0.95
        clip_param = 0.2
        gae_lambda = 0.95
        line_of_sight_masking = True
        custom_action_dist = "mbag_bilevel_categorical"
        experiment_tag = f"ppo_human/infinite_blocks_{str(inf_blocks).lower()}"

    @ex.named_config
    def gail_human():
        """
        This should always be used along with the ppo_human named config, like so:
            python -m mbag.scripts.train with ppo_human gail_human
        """
        run = "MbagGAIL"
        inf_blocks = True
        num_envs_per_worker = 8
        train_batch_size = 16000
        batch_mode = "truncate_episodes"
        if batch_mode == "complete_episodes":
            num_envs_per_worker = 1
        permute_block_types = True
        model = "transformer_with_discriminator"
        data_split = "human_alone"
        demonstration_input = (
            f"data/human_data_cleaned/{data_split}/"
            f"infinite_blocks_{str(inf_blocks).lower()}/"
            "rllib_with_own_noops_flat_actions_flat_observations_place_wrong_reward_-1_repaired_player_0"
        )
        train_discriminator_on_separate_batch = False
        discriminator_num_sgd_iter = 3

        checkpoint_to_load_policies = None
        checkpoint_name = None
        if checkpoint_to_load_policies is not None:
            load_policies_mapping = {"human": "human"}
            overwrite_loaded_policy_type = True
            overwrite_loaded_policy_model = True
            policies_to_train = ["human"]
        kl_target = 1e-2
        experiment_tag = (
            f"gail_human/discriminator_{discriminator_num_sgd_iter}_sgd_iter"
        )
        if train_discriminator_on_separate_batch:
            experiment_tag += "_separate_batch"
        experiment_tag += f"/{batch_mode}/kl_target_{kl_target}/infinite_blocks_{str(inf_blocks).lower()}/{data_split}/permute_{permute_block_types}"
        if checkpoint_to_load_policies is not None:
            experiment_tag += f"/init_{checkpoint_name}"

    @ex.named_config
    def alphazero_human():
        run = "MbagAlphaZero"
        goal_generator = "craftassist"
        width = 11
        height = 10
        depth = 10
        sample_batch_size = 16384
        sample_freq = 4
        rollout_fragment_length = 64
        max_seq_len = 64
        sgd_minibatch_size = 256
        random_start_locations = True
        num_training_iters = 500
        train_batch_size = 1
        use_replay_buffer = True
        replay_buffer_size = 4
        num_workers = 16
        num_envs_per_worker = 16
        evaluation_num_workers = 0
        evaluation_interval = None
        num_gpus = 1 if torch.cuda.is_available() else 0
        num_gpus_per_worker = 0.12 if torch.cuda.is_available() else 0
        model = "convolutional"
        filter_size = 5
        hidden_channels = 64
        num_layers = 6
        vf_share_layers = True
        num_simulations = 100
        num_sgd_iter = 1
        save_freq = 20
        horizon = 1500
        truncate_on_no_progress_timesteps = 100
        teleportation = False
        inf_blocks = True
        noop_reward = -0.2
        get_resources_reward = 0
        action_reward = 0
        use_goal_predictor = False
        use_bilevel_action_selection = True
        fix_bilevel_action_selection = True
        temperature = 1.5
        dirichlet_noise = 0.25
        dirichlet_action_subtype_noise_multiplier = 10
        dirichlet_epsilon = 0.25
        prior_temperature = 1.0
        init_q_with_max = False
        gamma = 0.95
        lr = 0.001
        puct_coefficient = 1
        scale_obs = True
        randomize_first_episode_length = True
        line_of_sight_masking = True
        experiment_tag = f"alphazero_human/infinite_blocks_{str(inf_blocks).lower()}"

    @ex.named_config
    def bc_human():
        run = "BC"
        data_split = "human_alone"
        inf_blocks = True
        teleportation = False
        train_batch_size = {
            True: {
                "human_alone": 9642,
                "human_with_assistant": 9759,
                "combined": 9642 + 9759,
            },
        }[inf_blocks][data_split]
        num_workers = 0
        evaluation_interval = 1
        save_freq = 1
        evaluation_num_workers = 8
        evaluation_duration = 64
        num_envs_per_worker = 8
        num_gpus_per_worker = 0.0625 if torch.cuda.is_available() else 0
        use_extra_features = True
        mask_other_players = data_split == "human_alone"
        goal_generator = "craftassist"
        width = 11
        height = 10
        depth = 10
        policy_ids = ["human"]
        if data_split != "human_alone":
            num_players = 2
        evaluation_num_players = 1
        model = "convolutional"
        dropout = 0.7
        line_of_sight_masking = True
        hidden_channels = 64
        filter_size = 5
        norm_first = False
        use_prev_action = True
        use_fc_after_embedding = True
        sgd_minibatch_size = 128
        use_separated_transformer = True
        interleave_lstm = False
        interleave_lstm_every = 4 if interleave_lstm else -1
        num_layers = 8 if interleave_lstm else 6
        vf_share_layers = True
        num_sgd_iter = 1
        inf_blocks = True
        teleportation = False
        random_start_locations = True
        policies_to_train = ["human"]
        compress_observations = True
        horizon = 1500
        mask_action_distribution = True
        num_training_iters = 40
        entropy_coeff_start = 0
        evaluation_explore = True
        checkpoint_name = None
        checkpoint_to_load_policies = None
        if checkpoint_to_load_policies is not None:
            load_policies_mapping = {"human": "human"}
        overwrite_loaded_policy_type = True
        lr_start = 1e-3 if checkpoint_to_load_policies is None else 1e-4
        lr = lr_start
        lr_schedule = [
            [0, lr_start],
            [train_batch_size * num_training_iters / 2, lr_start / 10],
        ]
        vf_loss_coeff = 0
        gamma = 0.95
        scale_obs = True
        permute_block_types = True

        input = (
            f"data/human_data_cleaned/{data_split}/"
            f"infinite_blocks_{str(inf_blocks).lower()}/"
            "rllib_with_own_noops_flat_actions_flat_observations_place_wrong_reward_-1_repaired"
        )
        if data_split == "human_alone":
            input += "_player_0_inventory_0"
        elif data_split == "human_with_assistant":
            input += "_player_1_inventory_0_1"
        elif data_split == "combined":
            input += "_player_0_inventory_0_1"
        if interleave_lstm:
            input += "_seq_64"
            max_seq_len = 64
        input_root_dir = None
        if input_root_dir is not None:
            input = f"{input_root_dir}/{input}"

        experiment_tag = f"bc_human/lr_{lr_start}/infinite_blocks_{str(inf_blocks).lower()}/{data_split}"
        if not permute_block_types:
            experiment_tag += "/no_data_augmentation"
        if not (
            ((num_layers, hidden_channels) == (6, 64) and not interleave_lstm)
            or ((num_layers, hidden_channels) == (8, 64) and interleave_lstm)
        ):
            experiment_tag += f"/model_{num_layers}x{hidden_channels}"
        if dropout != 0.7:
            experiment_tag += f"/dropout_{dropout}"
        if interleave_lstm:
            experiment_tag += "/lstm"
        if use_prev_action:
            experiment_tag += "/use_prev_action"
        if norm_first:
            experiment_tag += "/norm_first"
        if num_training_iters != 40:
            experiment_tag += f"/{num_training_iters}_iters"
        if checkpoint_to_load_policies is not None:
            experiment_tag += f"/init_{checkpoint_name}"

    @ex.named_config
    def pikl():
        run = "MbagAlphaZero"
        data_split = "human_alone"
        num_training_iters = 0
        train_batch_size = 10000
        use_extra_features = True
        num_workers = 0
        evaluation_num_workers = 0
        overwrite_loaded_policy_type = True
        goal_generator = "craftassist"
        width = 11
        height = 10
        depth = 10
        num_players = 1 if data_split == "human_alone" else 2
        gamma = 0.95
        model = "convolutional"
        filter_size = 5
        hidden_channels = 64
        sgd_minibatch_size = 128
        use_separated_transformer = True
        num_layers = 6
        vf_share_layers = True
        num_sgd_iter = 1
        inf_blocks = True
        teleportation = False
        horizon = 1500
        mask_action_distribution = True
        evaluation_explore = True
        scale_obs = True
        load_policies_mapping = {"human": "human"}
        is_human = [False] * num_players
        checkpoint_to_load_policies = None
        checkpoint_name = ""
        load_config_from_checkpoint = False
        num_simulations = 30
        puct_coefficient = 30
        sample_c_puct_every_timestep = True
        add_dirichlet_noise = False
        argmax_tree_policy = False
        explore_noops = False
        fix_bilevel_action_selection = True
        goal_subset = "test"
        horizon = 1500
        init_q_with_max = False
        prior_temperature = 1
        sample_from_full_support_policy = True
        temperature = 1
        truncate_on_no_progress_timesteps = None
        use_bilevel_action_selection = True
        use_critic = False
        use_goal_predictor = False

        if isinstance(puct_coefficient, (float, int)):
            puct_str = str(puct_coefficient)
        else:
            puct_str = "_".join(map(str, puct_coefficient))
            if sample_c_puct_every_timestep:
                puct_str += "_per_timestep"
            else:
                puct_str += "_per_episode"

        experiment_tag = (
            f"pikl/infinite_blocks_{str(inf_blocks).lower()}/"
            f"{checkpoint_name}/{num_simulations}_sims_puct_{puct_str}"
        )

    @ex.named_config
    def ppo_assistant():
        run = "MbagPPO"
        goal_generator = "craftassist"
        width = 11
        height = 10
        depth = 10
        num_players = 2
        randomize_first_episode_length = True
        random_start_locations = True
        num_training_iters = 100
        horizon = 1500
        noop_reward = 0.0
        get_resources_reward = 0.0
        teleportation = False
        inf_blocks = True
        entropy_coeff_start = 1
        entropy_coeff_end = 0.01
        entropy_coeff_horizon = 2_000_000
        own_reward_prop = 1
        train_batch_size = 16384
        num_workers = 16
        num_envs_per_worker = 16
        num_gpus = 0.5
        num_gpus_per_worker = 0.7 / max(num_workers, 1)
        lr = 0.0003
        kl_target = 0.01
        num_sgd_iter = 3
        rollout_fragment_length = 64
        batch_mode = "truncate_episodes"
        model = "convolutional"
        filter_size = 5
        hidden_channels = 64
        max_seq_len = 64
        sgd_minibatch_size = 256
        use_separated_transformer = True
        num_layers = 8
        num_heads = 4
        scale_obs = True
        vf_share_layers = True
        vf_loss_coeff = 0.01
        place_block_loss_coeff_schedule = [[0, 1], [2000000, 0]]
        evaluation_num_workers = 0
        evaluation_interval = None
        gamma = 0.95
        clip_param = 0.2
        gae_lambda = 0.95
        reward_scale = 1.0
        custom_action_dist = "mbag_bilevel_categorical"
        goal_loss_coeff = 30
        prev_goal_kl_coeff = 0
        prev_goal_kl_coeff_schedule = [
            [0, 0],
            [2_000_000, prev_goal_kl_coeff],
        ]
        mask_goal = True
        interleave_lstm_every = num_layers // 2
        policies_to_train = ["assistant"]
        checkpoint_to_load_policies = None
        checkpoint_name = ""
        load_policies_mapping = {"human": "human"}
        per_player_action_reward = [-0.2, 0]
        experiment_tag = (
            f"ppo_assistant/infinite_blocks_{str(inf_blocks).lower()}/"
            f"human_{checkpoint_name}"
        )

    @ex.named_config
    def alphazero_assistant():
        run = "MbagAlphaZero"
        goal_generator = "craftassist"
        width = 11
        height = 10
        depth = 10
        num_players = 2
        randomize_first_episode_length = True
        random_start_locations = True
        horizon = 1500
        noop_reward = 0
        get_resources_reward = 0
        per_player_action_reward = [0, 0]
        teleportation = False
        inf_blocks = True

        num_training_iters = 2000
        num_workers = 16
        num_envs_per_worker = 16
        max_seq_len = 64
        rollout_fragment_length = max_seq_len
        sample_batch_size = 16384
        sample_freq = 4
        # Train batch size is specified in terms of replay_buffer_storage_unit, i.e.,
        # sequences.
        train_batch_size = 256
        use_replay_buffer = True
        use_model_replay_buffer = False
        replay_buffer_storage_unit = "sequences"
        # Replay buffer capacities are specified in timesteps.
        replay_buffer_size = 262144
        num_gpus = 1.0
        num_gpus_per_worker = 0.12
        num_sgd_iter = 1
        batch_mode = "truncate_episodes"
        model = "convolutional_alpha_zero"
        filter_size = 5
        hidden_channels = 64
        sgd_minibatch_size = 1024
        num_layers = 8
        scale_obs = True
        vf_share_layers = True
        vf_scale = 1
        interleave_lstm_every = num_layers // 2

        num_simulations = 100
        use_bilevel_action_selection = True
        fix_bilevel_action_selection = True
        temperature = 1.5
        dirichlet_noise = 0.25
        dirichlet_action_subtype_noise_multiplier = 10
        dirichlet_epsilon = 0.25
        prior_temperature = 1.0
        init_q_with_max = False
        gamma = 0.95
        lr = 0.001
        goal_loss_coeff = 3
        prev_goal_kl_coeff = 30
        prev_goal_kl_coeff_schedule = [
            [0, 0],
            [2000 * sample_batch_size // sample_freq, prev_goal_kl_coeff],
        ]
        puct_coefficient = 1.0
        save_freq = 5
        evaluation_num_workers = 0
        evaluation_interval = None
        use_goal_predictor = True
        predict_goal_using_next_state = False
        predict_goal_using_average = False
        use_prev_blocks = False
        mask_goal = True
        pretrain = False
        policies_to_train = ["assistant"]
        checkpoint_path = None
        checkpoint_name = ""
        load_policies_mapping = {"human": "human"}
        experiment_tag = (
            f"alphazero_assistant/infinite_blocks_{str(inf_blocks).lower()}/"
            f"human_{checkpoint_name}"
        )

    @ex.named_config
    def non_goal_conditioned_human():
        run = "BC"
        inf_blocks = True
        teleportation = False
        train_batch_size = 8192
        validation_prop = 0.1
        num_workers = 0
        evaluation_interval = None
        save_freq = 100
        evaluation_num_workers = 0
        goal_generator = "craftassist"
        width = 11
        height = 10
        depth = 10
        model = "convolutional"
        hidden_channels = 64
        filter_size = 5
        dropout = 0
        interleave_lstm = True
        interleave_lstm_every = 4 if interleave_lstm else -1
        num_layers = 8 if interleave_lstm else 6
        vf_share_layers = True
        sgd_minibatch_size = 256
        max_seq_len = 64
        num_sgd_iter = 1
        compress_observations = True
        mask_action_distribution = True
        num_training_iters = 10_000
        lr = 1e-3
        scale_obs = True
        use_extra_features = False
        mask_goal = True

        # These need to be passed in as command line arguments.
        checkpoint_name = ""
        input = ""

        experiment_tag = (
            f"non_goal_conditioned_human/infinite_blocks_{str(inf_blocks).lower()}/"
            f"human_{checkpoint_name}/lstm_{interleave_lstm}"
        )
        if dropout != 0:
            experiment_tag += f"/dropout_{dropout}"
