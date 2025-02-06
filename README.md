# Minecraft Building Assistance Game

This repository contains code for the paper *Scalably Solving Assistance Games*. It includes the implementation of the Minecraft Building Assistance Game (MBAG), which we introduce to study more complex assistance games. It also includes code for running all experiments in the paper, including training assistants for MBAG with AssistanceZero.

MBAG is a multiagent environment that can run on its own within Python. It can also connect to running instances of Minecraft via [Project Malmo](https://github.com/microsoft/malmo) for visualization or for interaction with human players.

## Setup

This section describes how to set up your environment for running MBAG.

### Installing dependencies

**Installing the Python package:** first, install Python 3.8, 3.9, or 3.10. Then, run one of the following commmands:

  * Install just the environment: `pip install -e .`
  * Also install the RLlib dependencies for training and running assistants: `pip install -e .[rllib]`
  * Also install the Malmo interface to run assistants in Minecraft: `pip install -e .[rllib,malmo]`

    *Note: the interface with Minecraft via Malmo is currently only supported on macOS and Linux.*

**Installing Java:** to run assistants in Minecraft, you will also need Java 8 installed. On macOS, you specifically need to install the Java JDK 8u152, which is available [here](https://www.oracle.com/java/technologies/javase/javase8-archive-downloads.html). See [https://github.com/microsoft/malmo/issues/907](https://github.com/microsoft/malmo/issues/907) for details about why the specific version for macOS is necessary.

**Downloading the house dataset:** to install the house dataset from the [CraftAssist paper](https://arxiv.org/abs/1907.08584) that we use for training and evaluation, run:

    cd data
    wget https://minecraft-building-assistance-game.s3.us-east-1.amazonaws.com/craftassist.zip
    unzip -o craftassist.zip
    cd ..

### Linting, testing, and type checking

The project uses various tools to maintain code quality. To install them all, run

    pip install --upgrade -e .[dev]

Then, run linting and testing with the following commands:

    ./lint.sh
    pytest -m "not uses_malmo"

To run fewer tests, you can some or all of these additional filters:

    pytest -m "not uses_malmo and not slow"
    pytest -m "not uses_malmo and not uses_rllib"
    pytest -m "not uses_malmo and not uses_cuda"

## Running assistants in Minecraft

Playing with assistants in Minecraft and/or collecting data of humans playing takes two steps: first, starting Minecraft instances, and second, connecting to those instances to run an episode within the MBAG environment.

### Starting Minecraft

To start Minecraft instances, run the following command, assuming you've installed the Malmo dependencies (see setup section):

    python -m malmo.minecraft launch --num_instances 2 --goal_visibility True False

Set the number of Minecraft instances to launch with the `--num_instances` option. You need at least two instances to play with an assistant (one for the human and one for the AI assistant); if you want to record a video of the game, start an additional instance for a "spectator" player.

The `--goal_visibility` argument controls which instances show the goal house as a transparent blueprint within the game. Generally you should set this to True for the first instances and False for additional instances (e.g., `--goal_visibility True False False` for three instances).

You can tell when Minecraft instances are ready by the following signs:

 * The menu screen is displayed within Minecraft.
 * The `python -m malmo.minecraft launch` command stops producing new output, and the latest output lines show `CLIENT enter state: DORMANT`.

### Running an MBAG episode

Once the Minecraft instances are running, you can use the following command to start an episode playing with an assistant:

    python -m mbag.scripts.evaluate with human_with_assistant assistant_checkpoint=data/assistancezero_assistant/checkpoint_002000 num_simulations=1

*Note: `num_simulations` is used to control how long MCTS runs. If you are on a machine with a GPU, you can probably set `num_simulations=20`. If you do not have a GPU, you may want to set `num_simulations=1`, which will sample directly from the policy network. If you notice errors like `WARNING - mbag.environment.mbag_env - environment step took longer than Malmo action_delay; action_delay may need to be increased to achieve consistent step rate`, then reduce `num_simulations`.*

To play an episode without an assistant, run:

    python -m mbag.scripts.evaluate with human_alone assistant_checkpoint=data/assistancezero_assistant/checkpoint_002000

The `assistant_checkpoint` argument is still needed in this case to load the environment configuration.

Once the episode starts, press <kbd>Return</kbd> (<kbd>Enter</kbd>) to enable movement and <kbd>Delete</kbd> (<kbd>Fn</kbd> + <kbd>Backspace</kbd> on Mac) to enable flying.

The episode will automatically terminate when the house is completed, but if you want to end it sooner, use <kbd>Ctrl</kbd>+<kbd>C</kbd>. At the end of the episode, the episode data will be saved and metrics will be printed out.

You can additional specify `record_video=True` to record a video of the game.

## Running experiments

There are several steps in the pipeline to train and evaluate human models and assistants:

 1. **Training human models:** use the commands below to train the eight human models (PPO, AlphaZero, 3 BC, and 3 piKL):

        python -m mbag.scripts.train with ppo_human  # PPO human model
        python -m mbag.scripts.train with alphazero_human  # AlphaZero human model

        # BC human model; can also set data_split=human_with_assistant or data_split=combined
        python -m mbag.scripts.train with bc_human data_split=human_alone  

        # piKL human models; replace path/to/BC/checkpoint with the final checkpoint from the above.
        # Replace data_split=human_alone with whichever split you trained BC on.
        python -m mbag.scripts.train with pikl checkpoint_to_load_policies=path/to/BC/checkpoint \
            checkpoint_name=bc_human_alone data_split=human_alone
        
 2. **Evaluating prediction performance of human models:** for BC and piKL, we use cross validation to train human models with one participant from the data collection left out. To do this, add `validation_participant_ids='[3]'` to the `bc_human` training command. The participant IDs are 3, 4, 7, 9, and 11. Then, run this command to evaluate cross entropy and accuracy on the held-out data, replacing `participant_ids='[3]'` with the participant IDs you want to test:

        python -m mbag.scripts.evaluate_human_modeling with checkpoint=path/to/human/model/checkpoint \
            policy_id=human \
            participant_ids='[3]' \
            human_data_dir=data/human_data_cleaned/human_alone/infinite_blocks_true/rllib_with_own_noops_flat_actions_flat_observations_place_wrong_reward_-1_repaired_player_0_inventory_0
        
    Additional instructions:

      * Use the `human_data_dir` depending on the human model and which data subset you would like to evaluate on:
          * If the human model has `1_player` in the checkpoint name, use either of these to evaluate the human alone or human with assistant data, respectively:
            
                human_data_dir=data/human_data_cleaned/human_alone/infinite_blocks_true/rllib_with_own_noops_flat_actions_flat_observations_place_wrong_reward_-1_repaired_player_0_inventory_0
                human_data_dir=data/human_data_cleaned/human_with_assistant/infinite_blocks_true/rllib_with_own_noops_flat_actions_flat_observations_place_wrong_reward_-1_repaired_player_1_inventory_1

          * If the human model has `2_player` in the checkpoint name, use these:
            
                human_data_dir=data/human_data_cleaned/human_alone/infinite_blocks_true/rllib_with_own_noops_flat_actions_flat_observations_place_wrong_reward_-1_repaired_player_0_inventory_0_1
                human_data_dir=data/human_data_cleaned/human_with_assistant/infinite_blocks_true/rllib_with_own_noops_flat_actions_flat_observations_place_wrong_reward_-1_repaired_player_1_inventory_0_1

      * If you are evaluating a piKL human model, specify `minimatch_size=1`.

 3. **Evaluating performance of human models at building houses:** use the following command. Replace `runs='["BC"]'` with `runs='["MbagAlphaZero"]'` for AlphaZero/piKL or `runs='["MbagPPO"]'` for PPO.

        python -m mbag.scripts.evaluate with checkpoints='["path/to/human/model/checkpoint"]' \
            runs='["BC"]' policy_ids='["human"]' num_episodes=100 num_workers=10 \
            env_config_updates='{"truncate_on_no_progress_timesteps": None, "goal_generator_config": "goal_generator_config": {"subset": "test"}}'
    
    Specify the number of episodes to evaluate with `num_episodes` and choose how many evaluations to run in parallel by specifying `num_workers` (more workers will be faster but will use more memory/CPU/GPU).

 4. **Training an assistant with AssistanceZero:** use the following command:

        python -m mbag.scripts.train with assistancezero_assistant \
            checkpoint_to_load_policies=path/to/human/model/checkpoint \
            checkpoint_name=name_of_human_model

 5. **Testing an assistant with a human model:** use the following command:

        python -m mbag.scripts.evaluate with runs='["BC", "AlphaZero"]' \
            checkpoints='["path/to/human/model/checkpoint", "path/to/assistant/checkpoint"]' \
            policy_ids='["human", "assistant"]' \
            temperatures='[1.0, 1.0]' \
            num_episodes=100 num_workers=10 \
            algorithm_config_updates='[{},{"mcts_config": {"argmax_tree_policy": True, "add_dirichlet_noise": False, "num_simulations": 20}}]' \
            env_config_updates='{"horizon": 1500, "random_start_locations": True, "randomize_first_episode_length": False, "terminate_on_goal_completion": True, "truncate_on_no_progress_timesteps": None, "goal_generator_config": {"goal_generator_config": {"subset": "test"}}}' \
            out_dir=path/to/store/output

    Some of the options can be modified depending on the human model and assistant:
      * `runs`: these should be set to the algorithms used to train the human model and assistant. For BC, use `"BC"`; for piKL, AlphaZero, or AssistanceZero, use `"MbagAlphaZero"`; for PPO, use `"MbagPPO"`. The first element of the list should be set to the algorithm for the human model and the second should be set based on the assistant.
      * `policy_ids`: these should generally be set to `["human", "assistant"]` to use the `human` policy from the human model checkpoint and the `assistant` policy from the assistant checkpoint. However, when evaluating with the pretrained assistant, set `policy_ids='["human", "human"]` since the pretrained assistant has the policy ID `human` (it was trained by imitating human model data).
      * `temperatures`: these can be used to modify the sampling temperature for the human model and assistant. In the paper, we set the assistant's temperature to 0.3 for the pretrained assistant and SFT assistant.
      * `num_episodes` and `num_workers`: see the above instructions on *evaluating performance of human models at building houses* for the meaning of these options.
      * `algorithm_config_updates`: the options in here can be used to modify AssistanceZero's MCTS at test time. The main option of interest is `"num_simulations"`; more simulations take longer but may lead to better performance. We always use 20 simulations for evaluation in the paper.
      * `env_config_updates`: setting the `"subset"` option under `"goal_generator_config"` will choose whether to use sample goal houses from the `"train"` or `"test"` dataset.

 6. **Pretrained and SFT assistants:** to train the pretrained and SFT assistants, run the following commands:

        # Generate a dataset of episodes from the BC human model (this will take a while).
        # It will create a directory under the human model checkpoint called rollouts_...
        python -m mbag.scripts.rollout with run=BC \
            checkpoint=path/to/combined/bc/human/model/checkpoint \
            run=BC policy_ids='["human"]' \
            num_episodes=10000 num_workers=10 \
            save_samples=True save_as_sequences=True max_seq_len=64
        
        # Train pretrained assistant.
        python -m mbag.scripts.train with pretrained_assistant \
            input=path/to/rollouts
        
        # Train SFT assistant.
        python -m mbag.scripts.train with sft_assistant \
            checkpoint_to_load_policies=path/to/pretrained/assistant/checkpoint \
            checkpoint_name=pretrained_assistant
