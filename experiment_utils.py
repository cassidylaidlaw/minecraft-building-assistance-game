import copy
import glob
import json
import os
import pathlib
import re
import subprocess
import time
import typing
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from tqdm import tqdm

from mbag.environment.config import RewardScheduleEndpoints

ROOT_DIR = "/nas/ucb/ebronstein/minecraft-building-assistance-game"
ALL_DATA_SPLITS = ["combined", "human_alone", "human_with_assistant"]

DEFAULT_HUMAN_BC_ENV_VARS = dict(
    TELEPORTATION=False,
    INF_BLOCKS=True,
    # PERMUTE_BLOCK_TYPES=True,
    # NUM_TRAINING_ITERS=100,
    # TRAIN_BATCH_SIZE=9642,
    # HORIZON=1500,
    # VF_LOSS_COEFF=0,
    LR=0.001,
    # GAMMA=0.95,
    # ENTROPY=0,
    NUM_LAYERS=6,
    HIDDEN_CHANNELS=64,
    # USE_SEPARATED_TRANSFORMER=True,
    INTERLEAVE_LSTM=False,
    # NUM_SGD_ITER=1,
    # SGD_MINIBATCH_SIZE=128,
    # GRAD_CLIP=10,
    # USE_EXTRA_FEATURES=True,
    # DATA_TAG="human_alone_no_noops",
    # DATA_DIR="/nas/ucb/cassidy/minecraft-building-assistance-game/data/human_data_cleaned/human_alone/infinite_blocks_true/rllib_no_noops_flat_actions_flat_observations_player_0",
    SEED=0,
    SPLIT="human_alone",  # Training data split
    # Optional env vars
    MASK_OTHER_PLAYERS=None,
    CHECKPOINT=None,
    CHECKPOINT_NAME=None,
    VALIDATION_PARTICIPANT_IDS=None,
    TAG=None,
)

DEFAULT_HUMAN_ALPHAZERO_ENV_VARS = dict(
    TELEPORTATION=False,
    INF_BLOCKS=True,
    NOOP_REWARD=-0.2,
    GET_RESOURCES_REWARD=0,
    ACTION_REWARD=0,
    USE_GOAL_PREDICTOR=False,
    LINE_OF_SIGHT_MASKING=True,
    SCALE_OBS=True,
    USE_BILEVEL_ACTION_SELECTION=True,
    FIX_BILEVEL_ACTION_SELECTION=True,
    NUM_SIMULATIONS=100,
    TEMPERATURE=1.5,
    DIRICHLET_NOISE=0.25,
    DIRICHLET_ACTION_SUBTYPE_NOISE_MULTIPLIER=10,
    DIRICHLET_EPSILON=0.25,
    PRIOR_TEMPERATURE=1.0,
    INIT_Q_WITH_MAX=False,
    PUCT_COEFFICIENT=1,
    PUCT_COEFFICIENT_SCHEDULE=None,
    GAMMA=0.95,
    LR=0.001,
    WEIGHT_DECAY=0,
    GRAD_CLIP=10,
    # Don't set sample_batch_size so that it can be set explicitly in the
    # experiment config or computed in the script if unset.
    # SAMPLE_BATCH_SIZE=8000,
    TRAIN_BATCH_SIZE=8,
    NUM_SGD_ITER=1,
    NUM_TRAINING_ITERS=2000,
    HIDDEN_CHANNELS=64,
    NUM_LAYERS=6,
    NUM_WORKERS=8,
    NUM_ENVS_PER_WORKER=8,
    NUM_GPUS_PER_WORKER=0.1,
    HORIZON=500,
    ROLLOUT_FRAGMENT_LENGTH=256,
    MAX_SEQ_LEN=256,
    SGD_MINIBATCH_SIZE=512,
    RANDOMIZE_FIRST_EPISODE_LENGTH=True,
    BATCH_MODE="truncate_episodes",
    VF_SCALE=1,
    REPLAY_BUFFER_SIZE=20,
    USE_REPLAY_BUFFER=True,
    USE_SEPARATED_TRANSFORMER=True,
    INTERLEAVE_LSTM=False,
    PRETRAIN=False,
    SEED=0,
    # Optional env vars. These are set to None and then converted to empty
    # strings for the shell command, which makes the env var unset.
    MASK_OTHER_PLAYERS=None,
    TRUNCATE_ON_NO_PROGRESS_TIMESTEPS=None,
    CHECKPOINT=None,
    TAG=None,
)

# Default config for human MCTS.
DEFAULT_HUMAN_MCTS_CONFIG = dict(
    num_simulations=100,
    puct_coefficient=10,
    prior_temperature=1,
    explore_noops=False,
)

DEFAULT_ASSISTANT_ALPHAZERO_ENV_VARS = dict(
    INF_BLOCKS=True,
    HORIZON=1500,
    BATCH_MODE="truncate_episodes",
    RANDOMIZE_FIRST_EPISODE_LENGTH=True,
    NUM_ENVS_PER_WORKER=8,
    NUM_WORKERS=8,
    VF_SCALE=1,
    NUM_GPUS_PER_WORKER=0.08,
    USE_REPLAY_BUFFER=True,
    REPLAY_BUFFER_SIZE=32768,
    USE_MODEL_REPLAY_BUFFER=True,
    MODEL_REPLAY_BUFFER_SIZE=131072,
    GAMMA=0.95,
    LR=0.001,
    WEIGHT_DECAY=0,
    GRAD_CLIP=10,
    NUM_LAYERS=6,
    HIDDEN_CHANNELS=64,
    NUM_HEADS=4,
    NORM_FIRST=False,
    EMBEDDING_SIZE=16,
    POSITION_EMBEDDING_SIZE=48,
    POSITION_EMBEDDING_ANGLE=10,
    DIM_FEEDFORWARD=64,
    NUM_SGD_ITER=1,
    TRAIN_BATCH_SIZE=8,
    SEED=0,
    NUM_TRAINING_ITERS=100,
    # Don't set sample_batch_size so that it can be set explicitly in the
    # experiment config or computed in the script if unset.
    # SAMPLE_BATCH_SIZE=8000,
    TELEPORTATION=False,
    NOOP_REWARD=0,
    GET_RESOURCES_REWARD=0,
    ACTION_REWARD=0,
    MAX_SEQ_LEN=64,
    ROLLOUT_FRAGMENT_LENGTH=64,
    NUM_SIMULATIONS=100,
    USE_GOAL_PREDICTOR=True,
    USE_BILEVEL_ACTION_SELECTION=True,
    FIX_BILEVEL_ACTION_SELECTION=True,
    TEMPERATURE=1.5,
    DIRICHLET_NOISE=0.25,
    PRIOR_TEMPERATURE=1.0,
    INIT_Q_WITH_MAX=False,
    PRETRAIN=False,
    SGD_MINIBATCH_SIZE=1024,
    VF_LOSS_COEFF=0.01,
    GOAL_LOSS_COEFF=3,
    OTHER_AGENT_ACTION_PREDICTOR_LOSS_COEFF=1.0,
    USE_SEPARATED_TRANSFORMER=True,
    INTERLEAVE_LSTM=True,
    # Optional env vars. These are set to None and then converted to empty
    # strings for the shell command, which makes the env var unset.
    TRUNCATE_ON_NO_PROGRESS_TIMESTEPS=None,
    HUMAN_TAG=None,
    HUMAN_CHECKPOINT=None,
    HEURISTIC=None,
    CHECKPOINT=None,
    PER_PLAYER_ACTION_REWARD=None,
    TAG=None,
)

# Default PPO parameters based on Cassidy's training command:
# CLIP_PARAM=0.2 LAMBDA=0.95 ROLLOUT_FRAGMENT_LENGTH=512 NUM_SGD_ITER=3 LR=0.0003 ENTROPY_COEFF_END=0.01 ENTROPY_COEFF_HORIZON=2e6 VF_LOSS_COEFF=0.01 OWN_REWARD_PROP=1 PLACE_BLOCK_LOSS_COEFF=1 PLACE_BLOCK_LOSS_HORIZON=2e6 HORIZON=500 MAX_SEQ_LEN=512 BATCH_MODE=truncate_episodes NUM_WORKERS=8 GAMMA=0.98 GOAL_LOSS_COEFF=0.5 HUMAN_ACTION_PENALTY=0 NUM_LAYERS=6 HIDDEN_CHANNELS=64 NUM_GPUS_PER_WORKER=0.07 HEURISTIC=lowest_block TELEPORTATION=True sbatch scripts/slurm_ppo_assistant.sh
DEFAULT_ASSISTANT_PPO_ENV_VARS = dict(
    SEED=0,
    NUM_WORKERS=16,
    NUM_ENVS_PER_WORKER=16,
    NUM_TRAINING_ITERS=100,
    TELEPORTATION=False,
    PLACE_BLOCK_LOSS_COEFF=1,
    PLACE_BLOCK_LOSS_HORIZON=int(2e6),
    ENTROPY_COEFF_HORIZON=2_000_000,
    ENTROPY_COEFF_START=1,
    ENTROPY_COEFF_END=0.01,
    NOOP_REWARD=0.0,
    GET_RESOURCES_REWARD=0.0,
    GAMMA=0.95,
    HORIZON=1500,
    MODEL="convolutional",
    NUM_LAYERS=8,
    HIDDEN_CHANNELS=64,
    DIM_FEEDFORWARD=64,
    DROPOUT=0,
    NUM_SGD_ITER=3,
    LR=0.0003,
    GRAD_CLIP=10,
    CLIP_PARAM=0.2,
    LAMBDA=0.95,
    REWARD_SCALE=1.0,
    VF_LOSS_COEFF=0.01,
    INF_BLOCKS=True,
    KL_TARGET=0.01,
    BATCH_MODE="truncate_episodes",
    RANDOMIZE_FIRST_EPISODE_LENGTH=True,
    ROLLOUT_FRAGMENT_LENGTH=64,
    MAX_SEQ_LEN=64,
    SGD_MINIBATCH_SIZE=256,
    GOAL_LOSS_COEFF=30,
    PREV_GOAL_KL_COEFF=0,
    OWN_REWARD_PROP=1,
    USE_SEPARATED_TRANSFORMER=True,
    INTERLEAVE_LSTM=False,
    PRETRAIN=False,
    # Optional env vars. These are set to None and then converted to empty
    # strings for the shell command, which makes the env var unset.
    TRUNCATE_ON_NO_PROGRESS_TIMESTEPS=None,
    HUMAN_TAG=None,
    HUMAN_CHECKPOINT=None,
    HEURISTIC=None,
    CHECKPOINT=None,
    PER_PLAYER_ACTION_REWARD=None,
    TAG=None,
)


Algorithm = Literal["alphazero", "ppo", "bc"]
Agent = Literal["human", "assistant"]

ALGORITHM_TO_NAME: Dict[Algorithm, str] = {
    "alphazero": "MbagAlphaZero",
    "ppo": "MbagPPO",
    "bc": "BC",
    "pikl": "MbagAlphaZero",
}

ASSISTANT_ALGORITHM_TO_ENV_VARS: Dict[Algorithm, str] = {
    "alphazero": DEFAULT_ASSISTANT_ALPHAZERO_ENV_VARS,
    "ppo": DEFAULT_ASSISTANT_PPO_ENV_VARS,
}

HUMAN_ALGORITHM_TO_ENV_VARS: Dict[Algorithm, str] = {
    "bc": DEFAULT_HUMAN_BC_ENV_VARS,
    "alphazero": DEFAULT_HUMAN_ALPHAZERO_ENV_VARS,
}

# TODO: maybe delete this mapping because we should be using the data split rather than the agent type.
# Mapping from agent type (human or assistant) to the play mode, which is the
# directory under the run where logs are stored.
# NOTE: this used to be "self_play" and "cross_play", but is now "1_player" and
# "2_player".
AGENT_TO_PLAY_MODE: Dict[Agent, str] = {
    "human": "1_player",
    "assistant": "2_player",
}

# Mapping from the name of the data split used in the spreadsheet to the name
# used in the code and logs.
SPREADSHEET_DATA_SPLIT_MAP = {
    "none": None,
    "alone": "human_alone",
    "human_alone": "human_alone",
    "with_assistant": "human_with_assistant",
    "human_with_assistant": "human_with_assistant",
    "both": "combined",
    "combined": "combined",
}


def get_default_env_vars(algorithm: Algorithm, agent: Agent) -> dict:
    if agent == "human":
        env_vars = HUMAN_ALGORITHM_TO_ENV_VARS.get(algorithm)
    elif agent == "assistant":
        env_vars = ASSISTANT_ALGORITHM_TO_ENV_VARS.get(algorithm)
    else:
        raise ValueError(f"Invalid agent: {agent}")

    if env_vars is None:
        raise ValueError(f"Invalid algorithm: {algorithm}")

    return env_vars


def get_human_model_row(policies_df, human_model_name):
    if human_model_name == "lowest_block":
        return dict(human_model_name="lowest_block")
    human_model_row = policies_df[policies_df.human_model_name == human_model_name]
    if len(human_model_row) == 0:
        raise ValueError(f"Invalid human_model_name: {human_model_name}")
    if len(human_model_row) > 1:
        raise ValueError(f"Multiple human models with name: {human_model_name}")

    return human_model_row.iloc[0].to_dict()


def format_reward(reward):
    if isinstance(reward, (float, int)):
        return str(reward)
    elif isinstance(reward, (list, tuple)):
        reward_strs = []
        for endpoints in reward:
            assert len(endpoints) == 2
            iter, val = endpoints
            reward_strs.append(f"{iter}_{val}")
        return "-".join(reward_strs)
    else:
        raise ValueError(f"Invalid reward: {reward}")


def make_bc_common_tag(env_vars: dict, agent: Agent) -> str:
    assert agent == "human", f"Invalid BC agent: {agent}"

    tag = (
        f"teleportation_{env_vars['TELEPORTATION']}/inf_blocks_{env_vars['INF_BLOCKS']}"
    )

    data_split = env_vars["SPLIT"]
    tag += f"/split_{data_split}"
    if data_split != "human_alone":
        mask_other_players = env_vars["MASK_OTHER_PLAYERS"]
        tag += f"/mask_other_players_{mask_other_players}"

    tag += f"/lr_{env_vars['LR']}/seed_{env_vars['SEED']}"

    if env_vars.get("CHECKPOINT"):
        # Use the checkpoint name if provided, otherwise just "ckpt".
        checkpoint_name = env_vars.get("CHECKPOINT_NAME", "ckpt")
        tag += f"/init_{checkpoint_name}"

    return tag


def make_alphazero_from_bc_tag_with_human_model_name(
    env_vars: dict, human_model_name: str
) -> str:
    tag = f"bc_to_az/teleportation_{env_vars['TELEPORTATION']}/inf_blocks_{env_vars['INF_BLOCKS']}/{human_model_name}"
    num_simulations = env_vars.get("NUM_SIMULATIONS")
    if num_simulations is not None:
        tag += f"/sim_{num_simulations}"
    puct_coefficient = env_vars.get("PUCT_COEFFICIENT")
    puct_coefficient_schedule = env_vars.get("PUCT_COEFFICIENT_SCHEDULE")
    if puct_coefficient is not None:
        tag += f"/{make_puct_coeff_tag(puct_coefficient, puct_coefficient_schedule)}"

    validation_participant_ids = env_vars["VALIDATION_PARTICIPANT_IDS"]
    if validation_participant_ids is not None:
        tag += f"/validation_{validation_participant_ids}"

    return tag


def make_alphazero_from_bc_tag(env_vars: dict, human_model_name: Optional[str]) -> str:
    if human_model_name is not None:
        return make_alphazero_from_bc_tag_with_human_model_name(
            env_vars, human_model_name
        )

    tag = f"bc_to_az/teleportation_{env_vars['TELEPORTATION']}/inf_blocks_{env_vars['INF_BLOCKS']}/split_{env_vars['SPLIT']}/model_{env_vars['NUM_LAYERS']}x{env_vars['HIDDEN_CHANNELS']}"

    lr = env_vars.get("LR")
    if lr is not None:
        tag += f"/lr_{lr}"
    num_simulations = env_vars.get("NUM_SIMULATIONS")
    if num_simulations is not None:
        tag += f"/sim_{num_simulations}"
    puct_coefficient = env_vars.get("PUCT_COEFFICIENT")
    puct_coefficient_schedule = env_vars.get("PUCT_COEFFICIENT_SCHEDULE")
    if puct_coefficient is not None:
        tag += f"/{make_puct_coeff_tag(puct_coefficient, puct_coefficient_schedule)}"

    seed = env_vars.get("SEED")
    if seed is not None:
        tag += f"/seed_{seed}"

    if env_vars.get("CHECKPOINT"):
        # Use the checkpoint name if provided, otherwise just "ckpt".
        checkpoint_name = env_vars.get("CHECKPOINT_NAME", "ckpt")
        tag += f"/init_{checkpoint_name}"

    validation_participant_ids = env_vars["VALIDATION_PARTICIPANT_IDS"]
    if validation_participant_ids is not None:
        tag += f"/validation_{validation_participant_ids}"

    return tag


def make_common_tag(env_vars: dict, algorithm: Algorithm, agent: Agent) -> str:
    """Make tag that is shared between pre-training and regular training.

    The tag does not include 'pretrain_{True|False}'.
    """
    if agent not in typing.get_args(Agent):
        raise ValueError(f"Invalid agent: {agent}")
    if algorithm not in typing.get_args(Algorithm):
        raise ValueError(f"Invalid algorithm: {algorithm}")

    if algorithm == "bc":
        return make_bc_common_tag(env_vars, agent)

    tag = ""

    if agent == "assistant":
        if env_vars["HEURISTIC"] is not None:
            tag += f"{env_vars['HEURISTIC']}/"

    tag += (
        f"teleportation_{env_vars['TELEPORTATION']}/inf_blocks_{env_vars['INF_BLOCKS']}"
    )

    if agent == "assistant":
        human_tag = env_vars.get("HUMAN_TAG")
        if human_tag is not None:
            tag += f"/human_{human_tag}"

    if algorithm == "alphazero":
        tag += f"/batch_{env_vars['SAMPLE_BATCH_SIZE']}"

    tag += f"/horizon_{env_vars['HORIZON']}/{env_vars['BATCH_MODE']}"
    truncate_on_no_progress_timesteps = env_vars.get(
        "TRUNCATE_ON_NO_PROGRESS_TIMESTEPS"
    )
    if truncate_on_no_progress_timesteps is not None:
        tag += f"/trunc_no_progress_{truncate_on_no_progress_timesteps}"
    if env_vars["BATCH_MODE"] == "truncate_episodes":
        tag += f"/rollout_{env_vars['ROLLOUT_FRAGMENT_LENGTH']}"

    tag += f"/sgd_minibatch_{env_vars['SGD_MINIBATCH_SIZE']}"

    # Replay buffer(s)
    if algorithm == "alphazero":
        if env_vars["USE_REPLAY_BUFFER"]:
            replay_buffer_size = env_vars["REPLAY_BUFFER_SIZE"]
            train_batch_size = env_vars["TRAIN_BATCH_SIZE"]
        else:
            replay_buffer_size = 0
            train_batch_size = 1
        tag += f"/replay_{replay_buffer_size}/train_{env_vars['NUM_SGD_ITER']}x{train_batch_size}"
        if env_vars["USE_MODEL_REPLAY_BUFFER"]:
            model_replay_buffer_size = env_vars["MODEL_REPLAY_BUFFER_SIZE"]
        else:
            model_replay_buffer_size = 0
        tag += f"/model_replay_{model_replay_buffer_size}"

    tag += f"/max_seq_len_{env_vars['MAX_SEQ_LEN']}/gamma_{env_vars['GAMMA']}/lr_{env_vars['LR']}/weight_decay_{env_vars.get('WEIGHT_DECAY', 0)}"
    vf_scale = env_vars.get("VF_SCALE", 1)
    if vf_scale != 1:
        tag += f"/vf_scale_{vf_scale}"

    ### Model architecture
    model = env_vars["MODEL"]
    if model == "convolutional":
        tag += f"/conv_{env_vars['NUM_LAYERS']}x{env_vars['HIDDEN_CHANNELS']}/dropout_{env_vars['DROPOUT']}"
    elif model == "transformer":
        tag += f"/transformer_{env_vars['NUM_LAYERS']}x{env_vars['HIDDEN_CHANNELS']}"
        tag += (
            "/sep_transformer"
            if env_vars["USE_SEPARATED_TRANSFORMER"]
            else "/no_sep_transformer"
        )
        tag += (
            f"/dim_feedforward_{env_vars['DIM_FEEDFORWARD']}"
            f"/num_heads_{env_vars['NUM_HEADS']}"
            f"/norm_first_{env_vars['NORM_FIRST']}"
            f"/embedding_size_{env_vars['EMBEDDING_SIZE']}"
            f"/position_embedding_size_{env_vars['POSITION_EMBEDDING_SIZE']}"
            f"/position_embedding_angle_{env_vars['POSITION_EMBEDDING_ANGLE']}"
        )
        # LSTM
        interleave_lstm = env_vars["INTERLEAVE_LSTM"]
        if interleave_lstm:
            tag += "/interleave_lstm"
    else:
        raise NotImplementedError(f"Model not supported: {model}")

    tag += f"/grad_clip_{env_vars['GRAD_CLIP']}"

    # Rewards
    for reward_name in ["NOOP_REWARD", "GET_RESOURCES_REWARD", "ACTION_REWARD"]:
        reward = env_vars.get(reward_name, 0)
        if reward != 0:
            tag += f"/{reward_name.lower()}_{format_reward(reward)}"

    # Per-player action reward
    per_player_action_reward = env_vars.get("PER_PLAYER_ACTION_REWARD")
    if per_player_action_reward is not None:
        if (
            not isinstance(per_player_action_reward, list)
            or len(per_player_action_reward) != 2
        ):
            raise ValueError(
                f"Invalid per_player_action_reward: {per_player_action_reward}"
            )

        human_action_reward, assistant_action_reward = per_player_action_reward
        if human_action_reward != 0 and assistant_action_reward == 0:
            tag += f"/human_action_penalty_{human_action_reward}"
        elif human_action_reward != 0 or assistant_action_reward != 0:
            tag += f"/per_player_action_reward_{human_action_reward}_{assistant_action_reward}"

    # Goal predictor
    if env_vars.get("USE_GOAL_PREDICTOR") or (
        algorithm == "ppo" and agent == "assistant"
    ):
        prev_goal_kl_coeff = env_vars["PREV_GOAL_KL_COEFF"]
        if algorithm == "ppo" and agent == "assistant" and prev_goal_kl_coeff != 0:
            tag += f"/prev_goal_kl_schedule_0_0-2000000_{prev_goal_kl_coeff}"
        else:
            tag += f"/prev_goal_kl_{prev_goal_kl_coeff}"
        # Loss term coefficients that affect pretraining and relate to goal prediction.
        for loss_name in [
            "GOAL_LOSS_COEFF",
        ]:
            loss_coeff = env_vars[loss_name]
            tag += f"/{loss_name.lower()}_{loss_coeff}"

    # Loss term coefficients that affect pretraining but do not relate to goal prediction. Both AlphaZero and PPO use the vf_loss. Only AlphaZero uses the other_agent_action_predictor_loss.
    loss_names = ["VF_LOSS_COEFF"]
    if algorithm == "alphazero":
        loss_names.append("OTHER_AGENT_ACTION_PREDICTOR_LOSS_COEFF")
    for loss_name in loss_names:
        loss_coeff = env_vars[loss_name]
        tag += f"/{loss_name.lower()}_{loss_coeff}"

    # PPO-specific rewards
    if algorithm == "ppo":
        goal_loss_coeff = env_vars["GOAL_LOSS_COEFF"]
        if goal_loss_coeff != 0.5:
            tag += f"/goal_loss_{goal_loss_coeff}"
        own_reward_prop = env_vars["OWN_REWARD_PROP"]
        if own_reward_prop != 0:
            tag += f"/own_reward_prop_{format_reward(own_reward_prop)}"
        place_block_loss_coeff = env_vars["PLACE_BLOCK_LOSS_COEFF"]
        if place_block_loss_coeff != 0:
            tag += f"/place_block_loss_{place_block_loss_coeff}-0_{env_vars['PLACE_BLOCK_LOSS_HORIZON']}"
        tag += f"/entropy_{env_vars['ENTROPY_COEFF_START']}-{env_vars['ENTROPY_COEFF_END']}_{env_vars['ENTROPY_COEFF_HORIZON']}"

        tag += f"/{env_vars['NUM_SGD_ITER']}_sgd_iter/clip_{env_vars['CLIP_PARAM']}/vf_loss_{env_vars['VF_LOSS_COEFF']}/kl_target_{env_vars['KL_TARGET']}/seed_{env_vars['SEED']}"

    # Initialize from checkpoint. Only AlphaZero assistants are initialized from
    # a checkpoint by default. For other types of agents, if a checkpoint is
    # provided, add this to the tag.
    if (agent == "human" or algorithm != "alphazero") and env_vars["CHECKPOINT"]:
        tag += "/init_from_ckpt"

    return tag


def make_pretrain_tag_from_common_tag(
    env_vars: dict, common_tag: str, assistant_type: str
) -> str:
    if assistant_type != "alphazero":
        raise ValueError(f"assistant_type must be 'alphazero' but got {assistant_type}")
    pretrain = env_vars["PRETRAIN"]
    if not pretrain:
        raise ValueError("env_vars['PRETRAIN'] must be True to make pretrain tag.")
    return f"{common_tag}/pretrain_{pretrain}"


def make_train_tag(env_vars: dict, algorithm: Algorithm, agent: Agent) -> str:
    """Make tag for regular training."""
    tag = make_common_tag(env_vars, algorithm, agent)

    # Pretraining is only needed for the AlphaZero assistant.
    if agent == "assistant" and algorithm == "alphazero":
        pretrain = env_vars["PRETRAIN"]
        if pretrain:
            raise ValueError("env_vars['PRETRAIN'] must be False to make train tag.")

        tag += f"/pretrain_{pretrain}"

        tag += (
            f"/use_bilevel_action_selection_{env_vars['USE_BILEVEL_ACTION_SELECTION']}"
        )

    return tag


def make_schedule_str(schedule: RewardScheduleEndpoints) -> str:
    if isinstance(schedule, (float, int)):
        if int(schedule) == schedule:
            schedule = int(schedule)
        return str(schedule)
    else:
        parts = [f"{step}_{value}" for step, value in schedule]
        return "-".join(parts)


def parse_puct_coefficient_schedule_str(schedule_str: str) -> RewardScheduleEndpoints:
    parts = schedule_str.split("-")
    schedule = []
    for part in parts:
        step, value = part.split("_")
        schedule.append((int(step), float(value)))
    return schedule


def make_puct_coeff_tag(
    puct_coeff: Union[float, List[float], Tuple[float, ...]],
    puct_coeff_schedule: Optional[RewardScheduleEndpoints],
) -> str:
    if puct_coeff_schedule is not None:
        puct_coeff_str = f"schedule_{make_schedule_str(puct_coeff_schedule)}"
    elif isinstance(puct_coeff, (int, float)):
        puct_coeff_str = str(puct_coeff)
    elif isinstance(puct_coeff, (tuple, list)):
        puct_coeff_str = "mixture_" + "_".join(map(str, puct_coeff))
    else:
        raise TypeError(f"Invalid puct_coeff: {puct_coeff}")

    return f"puct_coeff_{puct_coeff_str}"


def make_mcts_eval_tag(config: Dict[str, Any]) -> str:
    # config is algorithm_config_updates
    mcts_config = config["mcts_config"]
    puct_coeff_str = make_puct_coeff_tag(
        mcts_config["puct_coefficient"], mcts_config.get("puct_coefficient_schedule")
    )

    tag = f"sim_{mcts_config['num_simulations']}/temp_{mcts_config['temperature']}/{puct_coeff_str}/prior_temp_{mcts_config['prior_temperature']}/critic_{config['use_critic']}/explore_{config['explore']}"

    explore_noops = mcts_config["explore_noops"]
    if explore_noops:
        tag += f"/explore_noops_{explore_noops}"

    return tag


def set_assistant_specific_env_vars(
    env_vars: dict,
    experiment_config: dict,
    human_model_name: str,
    human_model_row: pd.Series,
) -> None:
    """Sets environment variables specific to the assistant."""
    # Set vars based on human model.
    if human_model_name == "lowest_block":
        env_vars["HEURISTIC"] = human_model_name
        env_vars["TELEPORTATION"] = True
        env_vars["HUMAN_CHECKPOINT"] = None
        env_vars["HUMAN_TAG"] = human_model_name
        env_vars["INF_BLOCKS"] = True
    else:
        env_vars["HEURISTIC"] = None
        env_vars["TELEPORTATION"] = False
        env_vars["HUMAN_CHECKPOINT"] = human_model_row["human_model_checkpoint"]
        env_vars["HUMAN_TAG"] = human_model_row["human_model_name"]

    if "HUMAN_ACTION_PENALTY" in env_vars:
        if "PER_PLAYER_ACTION_REWARD" in experiment_config:
            raise ValueError(
                "Cannot specify both HUMAN_ACTION_PENALTY and PER_PLAYER_ACTION_REWARD"
            )
        human_action_penalty = env_vars.pop("HUMAN_ACTION_PENALTY")
        env_vars["PER_PLAYER_ACTION_REWARD"] = [-human_action_penalty, 0]


def check_train_run_exists(
    algorithm: Algorithm,
    agent: Agent,
    train_tag: Optional[str] = None,
    train_run_dir: Optional[str] = None,
    experiments_df: Optional[pd.DataFrame] = None,
    use_most_recent_run: bool = True,
) -> Tuple[bool, Optional[str]]:
    """
    Check if a training run exists for the given parameters.

    Args:
        algorithm (Algorithm): The algorithm used for training.
        agent (Agent): The agent used for training.
        train_tag (str): The tag associated with the training run.
        train_run_dir (str): The directory containing the training run.
        experiments_df (Optional[pd.DataFrame]): Optional DataFrame containing
            information about experiments. Will be used to check if a training
            run exists but was not found in the logs. Defaults to None.
        use_most_recent_run (bool): Whether to use the most recent training run
            if multiple exist. Defaults to True.

    Returns:
        tuple[bool, Optional[str]]: A tuple containing whether the training
            run exists and the path to the training run if it exists. If the
            training run is determined to exist because it is in the experiments_df but it was not found in the logs, the path will be None.

    Raises:
        ValueError: If neither or both of train_tag and train_run_dir are provided.
        ValueError: If multiple training runs are found for the given tag.
    """
    if (train_tag is None) == (train_run_dir is None):
        raise ValueError("Exactly one of train_tag and train_run_dir must be provided.")

    if train_tag is not None:
        # This pattern checks for paths where there could be something appended to
        # the experiment tag because the previous version of the code appended the
        # Slurm job ID, while the new version made a subdirectory under the train
        # tag with the Slurm job ID.
        train_run_file_pattern = (
            f"{ROOT_DIR}/data/logs/{ALGORITHM_TO_NAME[algorithm]}/{AGENT_TO_PLAY_MODE[agent]}/11x10x10/craftassist/"
            f"{train_tag}*/**/run.json"
        )
    else:
        train_run_file_pattern = os.path.join(train_run_dir, "run.json")

    train_run_fnames = list(glob.glob(train_run_file_pattern, recursive=True))

    train_run_exists = False
    train_run = None
    most_recent_datetime_run_created = None
    for run_fname in train_run_fnames:
        with open(run_fname, "r") as run_file:
            try:
                run_info = json.load(run_file)
            except json.decoder.JSONDecodeError:
                continue
        try:
            datetime_run_created = datetime.strptime(
                pathlib.Path(run_fname).parts[-3], "%Y-%m-%d_%H-%M-%S"
            )
        except ValueError as e:
            # Skip if the datetime cannot be parsed. This is likely because of
            # an eval/rollout run for this checkpoint, rather than a training run.
            continue
        # Training run exists if it is completed.
        if run_info.get("status", "COMPLETED") in ["RUNNING", "COMPLETED"]:
            if train_run_exists:
                if use_most_recent_run:
                    if datetime_run_created > most_recent_datetime_run_created:
                        most_recent_datetime_run_created = datetime_run_created
                else:
                    raise ValueError(f"Multiple training runs found for {train_tag}")
            else:
                train_run_exists = True
                most_recent_datetime_run_created = datetime_run_created
            train_run = run_fname
        elif experiments_df is not None:
            # Check if the training run is in the experiments spreadsheet.
            # The slurm job ID may be the -3rd or -4th part of the path.
            # Usually, it is the -3th, but if using BC and validation
            # participant IDs are provided, a new sub-directory is created, in
            # which case the slurm job ID is the -4th part.
            # NOTE: this check only works if the Slurm job ID is a subdirectory
            # under the experiment tag, rather than appended to the tag.
            for slurm_job_id_str in pathlib.Path(run_fname).parts[-4:-2]:
                try:
                    slurm_job_id = int(slurm_job_id_str)
                    break
                except ValueError:
                    continue
            else:
                # If no Slurm job ID is found, assume it does not exist in the spreadsheet.
                continue
            job_row = experiments_df[experiments_df["slurm_job_id"] == slurm_job_id]
            # Skip if no job is found.
            if len(job_row) == 0:
                continue
            elif len(job_row) > 1:
                raise ValueError(f"Multiple jobs found with ID {slurm_job_id}")
            else:
                status = job_row.iloc[0]["status"]
                # Training run exists if it is not recorded as failed or OOM.
                cur_train_run_exists = status not in ("failed", "oom")
                if cur_train_run_exists and train_run_exists:
                    raise ValueError(f"Multiple training runs found for {train_tag}")
                train_run_exists = cur_train_run_exists

    return train_run_exists, train_run


def get_env_vars_for_experiment(
    experiment_config: dict,
    agent: Agent,
    experiments_df: Optional[pd.DataFrame] = None,
    human_policies_df: Optional[pd.DataFrame] = None,
    use_most_recent_pretrain_checkpoint: bool = True,
    use_incomplete_pretrain_checkpoint: bool = False,
    force_pretrain: bool = False,
    force_train: bool = False,
) -> Tuple[Optional[dict], str]:
    """
    Creates environment variables for an assistant experiment.

    Args:
        experiment_config (dict): A dictionary containing the experiment
            configuration.
        agent (str): The agent for which to create the environment variables.
            One of "human" or "assistant".
        experiments_df (pd.DataFrame): A pandas DataFrame containing the experiment
            information.
        human_policies_df (pd.DataFrame, optional): A pandas DataFrame
            containing the human policies. Only used for the assistant. Defaults to None.
        use_most_recent_pretrain_checkpoint (bool, optional): Whether to use the
            most recent pretraining checkpoint if multiple exist. Defaults to True.
        use_incomplete_pretrain_checkpoint (bool, optional): Whether to use an
            incomplete pretraining checkpoint. Defaults to False.
        force_pretrain (bool, optional): Whether to force pretraining. Defaults to False.
        force_train (bool, optional): Whether to force training. Defaults to False.

    Returns:
        tuple[Optional[dict], str]: A tuple containing the environment
            variables and the algorithm. If the experiment should be
            skipped, returns (None, algorithm).
    """
    if force_pretrain and force_train:
        raise ValueError("Cannot force both pretraining and training.")
    if agent not in typing.get_args(Agent):
        raise ValueError(f"Invalid agent: {agent}")
    if agent == "assistant" and human_policies_df is None:
        raise ValueError("human_policies_df must be provided for the assistant.")

    experiment_config = copy.deepcopy(experiment_config)
    algorithm = experiment_config.pop("algorithm")
    if algorithm is None:
        raise ValueError("Algorithm must be specified in experiment_config.")
    if force_pretrain:
        if algorithm != "alphazero":
            raise ValueError(
                f"Cannot force pretraining for non-AlphaZero assistants. Got {algorithm}."
            )
        if agent != "assistant":
            raise ValueError("Cannot force pretraining for the human.")

    default_env_vars = get_default_env_vars(algorithm, agent)

    if agent == "assistant":
        human_model_name = experiment_config.pop("human_model_name")
        human_model_row = get_human_model_row(human_policies_df, human_model_name)

    # Update env vars with experiment config.
    env_vars = dict(default_env_vars)
    env_vars.update(experiment_config)

    # Set env vars specific to AlphaZero and PPO.
    if algorithm in ["alphazero", "ppo"]:
        batch_mode = env_vars["BATCH_MODE"]
        fragment_length = env_vars["ROLLOUT_FRAGMENT_LENGTH"]
        horizon = env_vars["HORIZON"]
        num_envs_per_worker = env_vars["NUM_ENVS_PER_WORKER"]

        if algorithm == "alphazero":
            sample_batch_size = env_vars.get("SAMPLE_BATCH_SIZE")
            if sample_batch_size is None:
                # Set sample batch size based on number of envs and fragment_length.
                env_vars["SAMPLE_BATCH_SIZE"] = (
                    horizon * env_vars["NUM_WORKERS"] * num_envs_per_worker
                    if batch_mode == "complete_episodes"
                    else fragment_length * env_vars["NUM_WORKERS"] * num_envs_per_worker
                )

        # Set vars based on batch_mode. If truncate_episodes, keep the
        # SGD_MINIBATCH_SIZE as is; otherwise, set it to horizon.
        if batch_mode == "truncate_episodes":
            env_vars["RANDOMIZE_FIRST_EPISODE_LENGTH"] = env_vars.get(
                "RANDOMIZE_FIRST_EPISODE_LENGTH", True
            )
        else:
            env_vars["RANDOMIZE_FIRST_EPISODE_LENGTH"] = env_vars.get(
                "RANDOMIZE_FIRST_EPISODE_LENGTH", False
            )
            env_vars["ROLLOUT_FRAGMENT_LENGTH"] = horizon
            env_vars["MAX_SEQ_LEN"] = horizon
            # Set SGD_MINIBATCH_SIZE to horizon + 1 because it must be strictly
            # greater than ROLLOUT_FRAGMENT_LENGTH and MAX_SEQ_LEN according to
            # rllib.
            env_vars["SGD_MINIBATCH_SIZE"] = horizon + 1

        # Replay buffer (only relevant to AlphaZero).
        if algorithm == "alphazero":
            use_replay_buffer = env_vars["USE_REPLAY_BUFFER"]
            replay_buffer_size = env_vars["REPLAY_BUFFER_SIZE"]
            if use_replay_buffer:
                assert (
                    replay_buffer_size > 0
                ), f"Replay buffer size must be greater than 0, got {replay_buffer_size}"
            else:
                # rllib raises an error if the replay buffer size is 0 when
                # validating the config, even if not using the replay buffer, so set
                # it to 1 arbitrarily.
                env_vars["REPLAY_BUFFER_SIZE"] = 1

    if agent == "human" and algorithm == "bc":
        # Mask other players for human BC if using human_alone data.
        mask_other_players = env_vars["MASK_OTHER_PLAYERS"]
        if mask_other_players is None:
            data_split = env_vars["SPLIT"]
            env_vars["MASK_OTHER_PLAYERS"] = data_split == "human_alone"

    # Assistant-specific env vars.
    if agent == "assistant":
        set_assistant_specific_env_vars(
            env_vars, experiment_config, human_model_name, human_model_row
        )

    common_tag = make_common_tag(env_vars, algorithm, agent)

    pretrain_run_exists = False
    pretrain_checkpoint = None
    if env_vars.get("CHECKPOINT") is not None:
        if force_pretrain:
            raise ValueError("Cannot force pretrain when a checkpoint is specified.")

        # Check if the pretraining checkpoint exists.
        pretrain_checkpoint = env_vars["CHECKPOINT"]
        if not os.path.exists(pretrain_checkpoint):
            raise ValueError(
                f"Pretraining checkpoint does not exist: {pretrain_checkpoint}"
            )
        pretrain_run_exists = True
    elif agent == "assistant" and algorithm == "alphazero":
        # If training an AlphaZero assistant, check if the pretraining run
        # exists. If it does, use it. Otherwise, start a new pretraining run.
        # The three wildcard directories are {slurm_job_id}/{timestamp}/{run_id}.
        pretrain_run_fnames = list(
            glob.glob(
                f"{ROOT_DIR}/data/logs/{ALGORITHM_TO_NAME[algorithm]}/{AGENT_TO_PLAY_MODE[agent]}/11x10x10/craftassist/"
                f"{common_tag}/pretrain_True/*/*/*/run.json"
            )
        )
        # Use the number of pretraining iterations from the environment variables
        # if it is set; otherwise, use a default value based on the sample batch
        # size.
        num_pretraining_iters = env_vars.get("NUM_PRETRAINING_ITERS")
        if num_pretraining_iters is None:
            num_pretraining_iters = (
                100 if env_vars["SAMPLE_BATCH_SIZE"] < 10_000 else 25
            )

        pretrain_checkpoints = []
        for run_fname in pretrain_run_fnames:
            with open(run_fname, "r") as run_file:
                try:
                    run_info = json.load(run_file)
                except json.decoder.JSONDecodeError:
                    continue

            # Try to get the final checkpoint from run_info.
            pretrain_checkpoint = (run_info.get("result") or {}).get("final_checkpoint")
            if pretrain_checkpoint is not None and not os.path.exists(
                pretrain_checkpoint
            ):
                pretrain_checkpoint = None

            if pretrain_checkpoint is None:
                # Otherwise, try to get the most recent checkpoint according to
                # the number of pretraining iterations.
                pretrain_checkpoint = f"{os.path.dirname(run_fname)}/checkpoint_{num_pretraining_iters:06d}"
                if not os.path.exists(pretrain_checkpoint):
                    pretrain_checkpoint = None

                # If the checkpoint does not exist, try to get the most recent
                # checkpoint.
                if use_incomplete_pretrain_checkpoint:
                    checkpoint_paths = glob.glob(
                        f"{os.path.dirname(run_fname)}/checkpoint_*"
                    )
                    checkpoint_steps = [
                        int(os.path.basename(checkpoint_path).split("_")[-1])
                        for checkpoint_path in checkpoint_paths
                    ]
                    pretrain_checkpoint = (
                        checkpoint_paths[np.argmax(checkpoint_steps)]
                        if checkpoint_steps
                        else None
                    )

                if pretrain_checkpoint is not None:
                    pretrain_checkpoint = pretrain_checkpoint[len(ROOT_DIR) + 1 :]

            if pretrain_checkpoint is not None:
                datetime_run_created = datetime.strptime(
                    pathlib.Path(pretrain_checkpoint).parts[-3], "%Y-%m-%d_%H-%M-%S"
                )
                time_ckpt_last_modified = os.path.getmtime(pretrain_checkpoint)
                pretrain_checkpoints.append(
                    {
                        "checkpoint": pretrain_checkpoint,
                        "datetime_run_created": datetime_run_created,
                        "time_ckpt_last_modified": time_ckpt_last_modified,
                    }
                )

        # Choose one checkpoint from the available checkpoints.
        if pretrain_checkpoints:
            if len(pretrain_checkpoints) == 1:
                ckpt_info = pretrain_checkpoints[0]
                pretrain_checkpoint = ckpt_info["checkpoint"]
            else:
                ckpt_paths = [ckpt["checkpoint"] for ckpt in pretrain_checkpoints]
                if use_most_recent_pretrain_checkpoint:
                    ckpt_info = max(
                        pretrain_checkpoints,
                        key=lambda x: x["datetime_run_created"],
                    )
                    pretrain_checkpoint = ckpt_info["checkpoint"]
                    print(
                        "# Multiple pretraining checkpoints found. Using the most recent one: "
                        f"{pretrain_checkpoint}. All checkpoints: {ckpt_paths}"
                    )
                else:
                    raise ValueError(
                        f"Multiple pretraining checkpoints found. Please specify which one to use from {ckpt_paths}"
                    )
            pretrain_run_exists = True

    # If pretraining run for an AlphaZero assistant does not exist, start a new
    # pretraining run. This is not needed for the human or other assistants
    # that do not require pretraining.
    if agent == "assistant" and algorithm == "alphazero" and not pretrain_run_exists:
        env_vars["PRETRAIN"] = True
        env_vars["NUM_TRAINING_ITERS"] = num_pretraining_iters
        env_vars["TAG"] = make_pretrain_tag_from_common_tag(
            env_vars, common_tag, algorithm
        )
    else:
        if pretrain_run_exists:
            if not pretrain_checkpoint:
                print(
                    "# Skipping experiment: pretraining run exists but checkpoint is not found. "
                    f"Pretraining checkpoint filepaths: {pretrain_run_fnames}",
                    end="\n\n",
                )
                return None, algorithm

            # Not pretraining, so set the pretraining checkpoint.
            env_vars["CHECKPOINT"] = pretrain_checkpoint
            env_vars["PRETRAIN"] = False

        train_tag = make_train_tag(env_vars, algorithm, agent)

        # If training run exists and training is not being forced, skip this
        # experiment.
        if not force_train:
            train_run_exists, train_run = check_train_run_exists(
                algorithm, agent, train_tag=train_tag, experiments_df=experiments_df
            )
            if train_run_exists:
                train_run_str = train_run if train_run else train_tag
                print(
                    f"# Skipping experiment because train run exists: {train_run_str}",
                    end="\n\n",
                )
                return None, algorithm

        env_vars["TAG"] = train_tag

    validate_env_vars(env_vars, algorithm)
    return env_vars, algorithm


def validate_env_vars(env_vars: dict, algorithm: Algorithm) -> None:
    # No validation currently needed for BC.
    if algorithm == "bc":
        return

    sgd_minibatch_size = env_vars["SGD_MINIBATCH_SIZE"]
    max_seq_len = env_vars["MAX_SEQ_LEN"]
    if sgd_minibatch_size <= max_seq_len:
        raise ValueError(
            f"SGD_MINIBATCH_SIZE ({sgd_minibatch_size}) must be > MAX_SEQ_LEN ({max_seq_len})"
        )
    fragment_length = env_vars["ROLLOUT_FRAGMENT_LENGTH"]
    if fragment_length < max_seq_len:
        # This is a warning because it is not necessarily an error. The fragment
        # length can be less than the max sequence length, but in practice the
        # sequences won't be longer than the fragment length, so it would make
        # sense for the sequence length to be <= the fragment length.
        print(
            f"ROLLOUT_FRAGMENT_LENGTH ({fragment_length}) should probably be >= MAX_SEQ_LEN ({max_seq_len})"
        )
    sample_batch_size_gcf = (
        env_vars["ROLLOUT_FRAGMENT_LENGTH"]
        * env_vars["NUM_WORKERS"]
        * env_vars["NUM_ENVS_PER_WORKER"]
    )
    # sample_batch_size is only used by AlphaZero.
    if algorithm == "alphazero":
        sample_batch_size = env_vars["SAMPLE_BATCH_SIZE"]
        if sample_batch_size % sample_batch_size_gcf != 0:
            raise ValueError(
                f"SAMPLE_BATCH_SIZE ({sample_batch_size}) should be divisible by {sample_batch_size_gcf} to avoid large memory usage."
            )

    hidden_size = env_vars["HIDDEN_CHANNELS"]
    model = env_vars["MODEL"]
    if model == "transformer":
        num_heads = env_vars["NUM_HEADS"]
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"HIDDEN_CHANNELS ({hidden_size}) should be divisible by NUM_HEADS ({num_heads})"
            )

    batch_mode = env_vars["BATCH_MODE"]
    if (batch_mode == "truncate_episodes") != env_vars[
        "RANDOMIZE_FIRST_EPISODE_LENGTH"
    ]:
        raise ValueError(
            "RANDOMIZE_FIRST_EPISODE_LENGTH should (probably) be True if BATCH_MODE is 'truncate_episodes' and False otherwise."
        )

    if not env_vars["USE_SEPARATED_TRANSFORMER"] and env_vars["INTERLEAVE_LSTM"]:
        raise ValueError(
            "Interleaving LSTM layers is not supported with a non-separated Transformer."
        )


def make_extra_slurm_args(env_vars: dict, algorithm: Algorithm) -> str:
    extra_slurm_args = []

    # Memory
    mem = env_vars.pop("mem", None)
    if mem is None:
        if algorithm == "alphazero" and env_vars["USE_REPLAY_BUFFER"]:
            # If using the replay buffer, increase the memory limit.
            replay_buffer_size = env_vars["REPLAY_BUFFER_SIZE"]
            if replay_buffer_size <= 5:
                mem = "120GB"
            elif replay_buffer_size <= 10:
                mem = "145GB"
            else:
                mem = "200GB"
    if mem is not None:
        extra_slurm_args.append(f"--mem={mem}")

    a100_nodelist = "--nodelist=airl.ist.berkeley.edu,sac.ist.berkeley.edu,rlhf.ist.berkeley.edu,cirl.ist.berkeley.edu"
    exclude_a4000_nodelist = "--exclude=ppo.ist.berkeley.edu,vae.ist.berkeley.edu"

    # If the model is large, run on machines with A100 GPUs.
    num_layers = env_vars["NUM_LAYERS"]
    hidden_size = env_vars["HIDDEN_CHANNELS"]
    interleave_lstm = env_vars["INTERLEAVE_LSTM"]
    use_separated_transformer = env_vars["USE_SEPARATED_TRANSFORMER"]
    if (
        algorithm == "alphazero"
        and (
            use_separated_transformer
            and (interleave_lstm and num_layers > 8)
            or (not interleave_lstm and (num_layers > 6 or hidden_size > 64))
        )
        or (not use_separated_transformer and num_layers >= 6)
    ):
        extra_slurm_args.append(a100_nodelist)
    else:
        # Otherwise, exclude machines with small GPUs. --exclude seems to be
        # mutually exclusive with --nodelist, for some reason.
        extra_slurm_args.append(exclude_a4000_nodelist)

    return extra_slurm_args


def make_env_vars_str(env_vars: dict, skip_null: bool = True) -> str:
    formatted_env_vars = []
    for key, value in sorted(env_vars.items()):
        if value is None and skip_null:
            continue

        # Format the value as a string if it is not a simple type. This is
        # needed for lists, tuples, dicts, etc. so they are properly
        # formatted for the shell.
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            value = f'"{value}"'
        formatted_env_vars.append(f"{key}={value}")

    return " ".join(formatted_env_vars)


def make_train_command_for_experiment(
    env_vars: dict, algorithm: Algorithm, agent: Agent
) -> str:
    extra_slurm_args = make_extra_slurm_args(env_vars, algorithm)
    extra_slurm_args_str = " ".join(extra_slurm_args)
    if extra_slurm_args_str:
        extra_slurm_args_str += " "

    env_vars_str = make_env_vars_str(env_vars)
    if agent == "assistant":
        if algorithm == "alphazero":
            script = "scripts/slurm_alphazero_assistant.sh"
        elif algorithm == "ppo":
            script = "scripts/slurm_ppo_assistant.sh"
        else:
            raise ValueError(f"{algorithm} for assistant is not supported.")
    elif agent == "human":
        if algorithm == "alphazero":
            script = "scripts/slurm_alphazero_human.sh"
        elif algorithm == "bc":
            script = "scripts/slurm_train_bc_human.sh"
        else:
            raise ValueError(f"{algorithm} for human is not supported.")

    return f"{env_vars_str} sbatch {extra_slurm_args_str}{script}"


EXPERIMENT_DF_DTYPES = {
    "slurm_job_id": int,
    "PRETRAIN": bool,
}


def cast_df_types(df: pd.DataFrame, dtypes: dict) -> pd.DataFrame:
    """Cast non-null columns in a DataFrame to specified data types."""
    for col, dtype in dtypes.items():
        if col in df.columns:
            df[col] = df[col].apply(lambda x: dtype(x) if not pd.isnull(x) else x)
    return df


def load_human_policies_df():
    sheet_id = "1TNVQA9KEof014eav_ymHu6T7hEct5Z_Khc-rB9HaTOk"
    gids = ["0", "754127131"]
    human_policies_df_list = []
    for gid in gids:
        df = pd.read_csv(
            f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&id={sheet_id}&gid={gid}"
        )
        # Drop empty rows. Don't consider "human_data_split" column for this because
        # its default value is "none" for empty rows.
        df = df.dropna(how="all", subset=set(df.columns) - {"human_data_split"})
        human_policies_df_list.append(df)

    return pd.concat(human_policies_df_list, ignore_index=True)


def get_experiments_df(
    agent: Agent, add_gid: bool = False, gids: Optional[List[str]] = None
) -> pd.DataFrame:
    if agent not in typing.get_args(Agent):
        raise ValueError(f"Invalid agent: {agent}")

    sheet_url_template = "https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&id={sheet_id}&gid={gid}"
    if agent == "assistant":
        sheet_id = "1kzPweoF46KcNJ8nVkcp3YlJfiM38ITVNnE0WtBeL7UI"
        gids = gids or [
            "1184545093",
            "1509291040",
            "138413330",
            "509620940",
            "409416141",
            "950110046",
            "2095034767",
            "1123701981",
            "2113927786",  # PPO assistants [old]
            "1244612052",  # PPO assistants [new]
        ]
    elif agent == "human":
        sheet_id = "1rVHrQqfjyvP93snLZ_M2az1QBmjhXA0nQzX3-xYYUqA"
        gids = gids or [
            "0",
            "1537086899",
        ]

    dfs = []
    for gid in gids:
        try:
            df = pd.read_csv(sheet_url_template.format(sheet_id=sheet_id, gid=gid))
        except pd.errors.EmptyDataError:
            # Skip empty sheets.
            continue
        # Drop empty rows without considering certain columns that can be non-empty.
        df = df.dropna(
            how="all",
            subset=set(df.columns) - {"slurm_job_id", "status", "Notes"},
        )
        if add_gid:
            df["gid"] = gid
        dfs.append(df)

    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        if "PRETRAIN" in df.columns:
            # Replace PRETRAIN NaNs with False. PRETRAIN is only defined for the AlphaZero assistant.
            df["PRETRAIN"] = df["PRETRAIN"].fillna(False)

        return cast_df_types(df, EXPERIMENT_DF_DTYPES)
    else:
        return pd.DataFrame()


def get_slurm_job_ids_regex(experiments_df):
    return "|".join(
        experiments_df["slurm_job_id"]
        .dropna()
        .astype(int)
        .unique()
        .astype(str)
        .tolist()
    )


def check_slurm_job_status(job_id):
    """
    Check the status of a Slurm job given its ID, including jobs that have finished.

    Parameters:
    job_id (str): The ID of the Slurm job.

    Returns:
    str: The status of the job (e.g., "RUNNING", "PENDING", "COMPLETED", "FAILED", etc.)
         or "NOT FOUND" if the job ID is not found.
    """
    try:
        # Execute the 'sacct' command with formatting to get job status
        result = subprocess.run(
            [
                "sacct",
                "-j",
                str(job_id),
                "-u",  # User
                "ebronstein",
                "--format=JobID,State",
                "--noheader",
                "--parsable2",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        # Process the output
        output = result.stdout.strip()
        if output:
            # Split the output by '|' and newline, and extract the job state
            # The '--parsable2' option uses '|' as a delimiter
            statuses = output.split("\n")
            job_id_status_list = [x.split("|") for x in statuses]

            # Get the status of the job.
            job_status = [
                status
                for full_job_id, status in job_id_status_list
                if full_job_id == str(job_id)
            ]
            if len(job_status) != 1:
                raise ValueError(
                    f"Unexpected number of statuses for job ID {job_id}: {job_id_status_list}"
                )
            status = job_status[0]

            # Get the status of the batch job.
            batch_job_status = [
                status
                for full_job_id, status in job_id_status_list
                if full_job_id == f"{job_id}.batch"
            ]
            if len(batch_job_status) > 1:
                raise ValueError(
                    f"Unexpected number of batch statuses for job ID {job_id}: {job_id_status_list}"
                )
            batch_status = batch_job_status[0] if batch_job_status else None

            # Check if the job OOM'd, which is indicated by the batch status.
            if batch_status == "OUT_OF_MEMORY" and status not in [
                "COMPLETED",
                "TIMEOUT",
            ]:
                return "OUT_OF_MEMORY"
            # Check if the job was canceled because the status appears as
            # "CANCELLED by X".
            elif "CANCELLED" in status:
                return "CANCELLED"
            else:
                return status
        else:
            return "NOT FOUND"
    except subprocess.CalledProcessError as e:
        # Handle errors in the subprocess
        print(f"Error checking Slurm job status: {e}")
        return None


def is_status_outdated(
    row,
    fail_is_done: bool = True,
    cancel_is_done: bool = True,
    fail_oom_cancel_overrides=True,
):
    status = row["status"]
    cur_status = row["Current status"]
    if fail_oom_cancel_overrides and status in ["failed", "oom", "canceled"]:
        return False
    # If the job failed, which is reported in the spreadsheet, but Slurm reports
    # it as done, consider the status up to date.
    if fail_is_done and cur_status == "done" and status in ["done", "failed", "oom"]:
        return False
    if cancel_is_done and cur_status == "canceled" and status in ["done", "canceled"]:
        return False
    if cur_status == "timeout" and status in ["done", "timeout"]:
        return False
    return status != cur_status


def load_result_json(path: str) -> list:
    """
    Load a JSON file and return its contents as a list of dictionaries.

    'result.json' has one JSON object per line (for each training iteration),
    so we read the file line by line.

    Args:
        path (str): The path to the JSON file.

    Returns:
        list: A list of dictionaries representing the contents of the JSON file.
    """
    result = []
    with open(path, "r") as f:
        for line in f:
            result.append(json.loads(line))
    return result


def get_metrics_from_result(result: Dict) -> Dict:
    metrics = {}
    for key in [
        "date",
        "episodes_this_iter",
        "training_iteration",
        "iterations_since_restore",
        # "agent_timesteps_total",
        "time_this_iter_s",
    ]:
        metrics[key] = result.get(key)

    sampler_results = result.get("sampler_results")
    if sampler_results is not None:
        for key in [
            "episode_reward_min",
            "episode_reward_mean",
            "episode_reward_max",
            "episode_len_mean",
            "episodes_this_iter",
        ]:
            metrics[key] = sampler_results.get(key)
        for key in [
            "policy_reward_min",
            "policy_reward_mean",
            "policy_reward_max",
        ]:
            sub_dict = sampler_results.get(key, {})
            metrics.update({f"{key}/{k}": v for k, v in sub_dict.items()})

    metrics.update(result.get("custom_metrics", {}))

    return metrics


def get_env_config_subset(env_config: Dict) -> Dict:
    config = {}
    for key in ["horizon", "random_start_locations"]:
        config[key] = env_config[key]
    config.update(env_config["abilities"])
    config.update({f"{k}_reward": v for k, v in env_config["rewards"].items()})

    for player_config in env_config["players"]:
        player = player_config["player_name"]
        player_rewards = {
            f"{player}_{k}_reward": v
            for k, v in player_config.get("rewards", {}).items()
        }
        config.update(player_rewards)

    return config


def get_policy_config_subsets(config: Dict) -> List[Dict]:
    policy_config_subsets = []

    for policy, policy_config in config["policies"].items():
        if len(policy_config) != 4:
            raise ValueError(
                f"Expected {policy} policy config to have length 4, got {len(policy_config)}."
            )
        policy_config = policy_config[-1]
        # The policy config may not be a dict if we are training an
        # assistant and this config is for the human policy. In that case,
        # it is a string like "<MbagAlphaZeroConfig object at 0x...".
        if not isinstance(policy_config, dict):
            continue

        policy_config_subset = {}
        # Model
        policy_model_config = policy_config["model"]["custom_model_config"]
        for key in [
            "num_layers",
            "hidden_channels",
            "hidden_size",
            "line_of_sight_masking",
            "scale_obs",
        ]:
            policy_config_subset[key] = policy_model_config.get(key)
        # Environment
        policy_config_subset.update(
            get_env_config_subset(policy_model_config["env_config"])
        )
        # Goal loss coefficient
        policy_config_subset["goal_loss_coeff"] = policy_config.get("goal_loss_coeff")

        policy_config_subset = {
            f"policy_{policy}/{k}": v for k, v in policy_config_subset.items()
        }
        policy_config_subsets.append(policy_config_subset)

    return policy_config_subsets


def get_result_subset(result: Dict, agent: Agent) -> Dict:
    """Return subset of a result dictionary.

    `result` is a one JSON object corresponding to a single line in the
    result.json config.
    """
    result_config = result["config"]

    config = {}
    config.update(get_env_config_subset(result_config["env_config"]))

    for key in [
        "gamma",
        "lr",
        "grad_clip",
        "policy_loss_coeff",
        "vf_loss_coeff",
        "other_agent_action_predictor_loss_coeff",
        "goal_loss_coeff",
        "pretrain",
    ]:
        config[key] = result_config.get(key)

    # TODO: update this to work with convolutional models as well.
    config["hidden_size"] = result_config["model"].get("attention_dim")
    config.update(result_config.get("mcts_config", {}))

    # Policy configs
    policy_config_subsets = get_policy_config_subsets(result_config)
    if not policy_config_subsets:
        raise ValueError("No policy config subsets found.")
    for subset in policy_config_subsets:
        config.update(subset)

    return config


def get_config_subset(config: Dict) -> Dict:
    """Return subset of config.json."""
    subset = {}
    for key in ["experiment_tag", "checkpoint_to_load_policies"]:
        subset[key] = config.get(key)
    return subset


def load_metrics(
    algorithm: Algorithm, agent: Agent, train_tag: Optional[str] = None
) -> pd.DataFrame:
    if train_tag is None:
        train_tag_filter = "**"
    else:
        train_tag_filter = f"{train_tag}/**"

    # This pattern checks for paths where there could be something appended to
    # the experiment tag because the previous version of the code appended the
    # Slurm job ID, while the new version made a subdirectory under the train
    # tag with the Slurm job ID.
    train_run_file_pattern = (
        f"{ROOT_DIR}/data/logs/{ALGORITHM_TO_NAME[algorithm]}/{AGENT_TO_PLAY_MODE[agent]}/11x10x10/craftassist/"
        f"{train_tag_filter}/result.json"
    )
    train_run_fnames = list(glob.glob(train_run_file_pattern, recursive=True))

    metrics_list = []
    for result_path in tqdm(train_run_fnames, leave=False):
        result = load_result_json(result_path)
        if not result:
            continue

        # Load metrics from the last training iteration.
        last_result = result[-1]
        config_path = pathlib.Path(
            *(pathlib.Path(result_path).parts[:-1] + ("config.json",))
        )
        if config_path.exists():
            config = json.load(config_path.open("r"))

        metrics = {}
        # Config
        metrics.update(get_config_subset(config))
        # Metrics
        metrics.update(get_metrics_from_result(last_result))
        # Metrics config
        try:
            metrics.update(get_result_subset(last_result, agent))
        except ValueError as e:
            raise ValueError(f"Failed to get config from {result_path}") from e
        metrics_list.append(metrics)

    metrics_df = pd.DataFrame(metrics_list)

    # Fill NaN reward config values with 0. If it
    reward_names = ["noop", "action", "place_wrong", "get_resources"]
    player_names = ["human", "assistant"]
    fillna_dict = {
        f"{player}_{reward}_reward": 0
        for player in player_names
        for reward in reward_names
    }
    metrics_df = metrics_df.fillna(fillna_dict)

    # Reorder columns.
    new_cols = [c for c in metrics_df.columns if c not in ["experiment_tag", "date"]]
    new_cols = ["experiment_tag", "date"] + new_cols
    metrics_df = metrics_df.reindex(columns=new_cols)

    return metrics_df


def compute_mean_cross_entropy(
    human_modeling_metrics: List[Dict[str, Any]]
) -> Tuple[float, List[float], List[int]]:
    """
    Computes the mean cross entropy based on the given human modeling metrics.

    Args:
        human_modeling_metrics (List[Dict[str, Any]]): A list of dictionaries containing the human modeling metrics.
            Each dictionary represents the metrics for a participant and should have the following keys:
            - "participant_ids" (int): The ID of the participant.
            - "episode_results" (List[Dict[str, Any]]): A list of dictionaries representing the results for each episode.
                Each episode dictionary should have the following keys:
                - "cross_entropy" (float): The cross entropy value for the episode.
                - "length" (int): The length of the episode.

    Returns:
        Tuple of float, List[float], List[int]: The mean cross entropy, the
        cross entropies for each episode, and the episode lengths.


    Raises:
        AssertionError: If the participant ID is not of type int.

    """
    cross_entropies = []
    episode_lengths = []
    for participant_metrics in human_modeling_metrics:
        assert isinstance(
            participant_metrics["participant_ids"], int
        ), f"Expected one participant ID, got {participant_metrics['participant_ids']}"
        for episode in participant_metrics["episode_results"]:
            cross_entropies.append(episode["cross_entropy"])
            episode_lengths.append(episode["length"])
    mean_cross_entropy = float(np.average(cross_entropies, weights=episode_lengths))

    return mean_cross_entropy, cross_entropies, episode_lengths


def get_data_split_from_path(path: str, subset: Literal["train", "test"]) -> str:
    """Extract the data split from the path of the experiment."""
    if subset not in ["train", "test"]:
        raise ValueError(f"subset must be 'train' or 'test', got {subset}.")
    prefix = "split_" if subset == "train" else "test_split_"
    split_dirs = [d for d in pathlib.Path(path).parts if d.startswith(prefix)]
    assert (
        len(split_dirs) == 1
    ), f"Expected one path part starting with '{prefix}', got {len(split_dirs)} for path {str(path)}"
    return split_dirs[0][len(prefix) :]


def summarize_episode_metrics(episode_metrics: Dict[str, Any]) -> pd.DataFrame:
    """Summarize episode metrics for a human model.

    Args:
        episode_metrics (pd.DataFrame): A DataFram containing metrics for a
            single episode.
    """
    # Regex pattern for per-minute metrics. Check if the key ends with "_\d+_min"
    # and capture 1) the part before the "_\d+_min" suffix, and 2) the "\d+" number.
    # E.g., "goal_percentage_5_min" matches, and the captured groups are
    # "goal_percentage" and "5", respectively.
    per_minute_pattern = r"(.*)_(\d+)_min$"
    summary_metrics = []
    # Times at which per-minute metrics are collected.
    times = set([])

    # Top-level (non-nested) metrics
    for key, value in episode_metrics.items():
        if not isinstance(value, (dict, list, tuple)):
            match = re.search(per_minute_pattern, key)
            if match:
                # Capture the key without the interval suffix
                key = match.group(1)
                time = int(match.group(2))
            else:
                time = float("nan")
            summary_metrics.append((key, value, float("nan"), time))

    # Top-level (non-nested) player metrics
    player_metrics_list = episode_metrics.get("player_metrics")
    assert player_metrics_list is not None and isinstance(player_metrics_list, list)
    for player_idx, player_metrics in enumerate(player_metrics_list):
        for key, value in player_metrics.items():
            if key != "per_minute_metrics":
                summary_metrics.append((key, value, player_idx, float("nan")))

        # Per-minute player metrics
        per_minute_player_metrics = player_metrics.get("per_minute_metrics", {})
        for key, value in per_minute_player_metrics.items():
            match = re.search(per_minute_pattern, key)
            assert match, f"Key {key} does not match the per-minute pattern."
            key = match.group(1)
            time = int(match.group(2))
            summary_metrics.append((key, value, player_idx, time))
            times.add(time)

        move_metric_keys = [
            "num_move_pos_x",
            "num_move_neg_x",
            "num_move_pos_y",
            "num_move_neg_y",
            "num_move_pos_z",
            "num_move_neg_z",
        ]
        action_or_noop_metric_keys = [
            "num_move_pos_x",
            "num_move_neg_x",
            "num_move_pos_y",
            "num_move_neg_y",
            "num_move_pos_z",
            "num_move_neg_z",
            "num_place_block",
            "num_break_block",
            "num_noop",
        ]
        place_or_break_metric_keys = ["num_place_block", "num_break_block"]

        # Compute grouped actions and action percentages.
        # Whole episode player metrics.
        num_move = sum(player_metrics[key] for key in move_metric_keys)
        num_action_or_noop = sum(
            player_metrics[key] for key in action_or_noop_metric_keys
        )
        summary_metrics.append(("num_move", num_move, player_idx, float("nan")))
        summary_metrics.append(
            (
                "num_move_percentage",
                num_move / num_action_or_noop,
                player_idx,
                float("nan"),
            )
        )
        for action in ["num_place_block", "num_break_block", "num_noop"]:
            num_action = player_metrics[action]
            summary_metrics.append(
                (
                    f"{action}_percentage",
                    num_action / num_action_or_noop,
                    player_idx,
                    float("nan"),
                )
            )
        num_place_or_break = sum(
            player_metrics[key] for key in place_or_break_metric_keys
        )
        summary_metrics.append(
            ("num_place_or_break_block", num_place_or_break, player_idx, float("nan"))
        )
        summary_metrics.append(
            (
                "num_place_or_break_block_percentage",
                num_place_or_break / num_action_or_noop,
                player_idx,
                float("nan"),
            )
        )

        # Per-minute player metrics.
        if per_minute_player_metrics:
            for time in times:
                num_move = sum(
                    per_minute_player_metrics[f"{key}_{time}_min"]
                    for key in move_metric_keys
                )
                num_action_or_noop = sum(
                    per_minute_player_metrics[f"{key}_{time}_min"]
                    for key in action_or_noop_metric_keys
                )
                summary_metrics.append(("num_move", num_move, player_idx, time))
                summary_metrics.append(
                    (
                        "num_move_percentage",
                        num_move / num_action_or_noop,
                        player_idx,
                        time,
                    )
                )
                for action in ["num_place_block", "num_break_block", "num_noop"]:
                    num_action = per_minute_player_metrics[f"{action}_{time}_min"]
                    summary_metrics.append(
                        (
                            f"{action}_percentage",
                            num_action / num_action_or_noop,
                            player_idx,
                            time,
                        )
                    )
                num_place_or_break = sum(
                    per_minute_player_metrics[f"{key}_{time}_min"]
                    for key in place_or_break_metric_keys
                )
                summary_metrics.append(
                    ("num_place_or_break_block", num_place_or_break, player_idx, time)
                )
                summary_metrics.append(
                    (
                        "num_place_or_break_block_percentage",
                        num_place_or_break / num_action_or_noop,
                        player_idx,
                        time,
                    )
                )

    return pd.DataFrame(summary_metrics, columns=["metric", "value", "player", "time"])

    # TODO: delete this alternative way of computing grouped actions and action
    # percentages because it's less efficient.
    # grouped_actions = []
    # for time, per_minute_metrics in summary_metrics_df.groupby("time"):
    #     if pd.isna(time):
    #         continue

    #     # Total movement
    #     num_move = per_minute_metrics[
    #         per_minute_metrics["metric"].isin(move_metric_keys)
    #     ]["value"].sum()
    #     grouped_actions.append(("num_move", num_move, time))

    #     # Any action or noop
    #     num_action_or_noop = per_minute_metrics[
    #         per_minute_metrics["metric"].isin(action_or_noop_metric_keys)
    #     ]["value"].sum()
    #     grouped_actions.append(("num_action_or_noop", num_action_or_noop, time))

    #     # Percentage any action or noop
    #     grouped_actions.append(
    #         ("num_move_percentage", num_move / num_action_or_noop, time)
    #     )
    #     for action in ["num_place_block", "num_break_block", "num_noop"]:
    #         num_action_df = per_minute_metrics[per_minute_metrics["metric"] == action]
    #         assert len(num_action_df) == 1
    #         action_percentage = num_action_df["value"].iloc[0] / num_action_or_noop
    #         grouped_actions.append((f"{action}_percentage", action_percentage, time))

    # grouped_actions_df = pd.DataFrame(
    #     grouped_actions, columns=["metric", "value", "time"]
    # )
    # return pd.concat([summary_metrics_df, grouped_actions_df], ignore_index=True)


def make_int_or_float(value: str) -> Union[int, float]:
    value = float(value)
    if int(value) == value:
        value = int(value)
    return value


def infer_mcts_params_from_path(eval_path: Union[str, pathlib.Path]) -> Dict[str, Any]:
    eval_path = pathlib.Path(eval_path)
    num_sims = next(p for p in eval_path.parts if p.startswith("sim_"))
    num_sims = int(num_sims[len("sim_") :])

    puct_coefficient = next(p for p in eval_path.parts if p.startswith("puct_coeff_"))
    puct_coefficient = puct_coefficient[len("puct_coeff_") :]
    if puct_coefficient.startswith("mixture_"):
        puct_coefficient = list(
            map(make_int_or_float, puct_coefficient[len("mixture_") :].split("_"))
        )
    elif puct_coefficient.startswith("schedule_"):
        puct_coefficient = puct_coefficient[len("schedule_") :]
        puct_coefficient = parse_puct_coefficient_schedule_str(puct_coefficient)
    else:
        puct_coefficient = float(puct_coefficient)
        if int(puct_coefficient) == puct_coefficient:
            puct_coefficient = int(puct_coefficient)

    return {"num_simulations": num_sims, "puct_coefficient": puct_coefficient}


def get_goal_metrics_df(
    goal_metrics_list: List[List[Dict[str, Any]]],
    algorithm: Algorithm,
    per_episode: bool = True,
):
    """Combine metrics for the human model.

    Args:
        goal_metrics_list (List[Dict[str, Any]]): A list of lists of
            dictionaries containing the goal metrics. The structure is
            similar to human_modeling_metrics_list.
        algorithm (Algorithm): The algorithm used for training.
        per_episode (bool): Whether to add the goal percentage per episode to
            the DataFrame. If False, only adds the mean goal percentage over the
            episodes for each time interval. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the combined metrics and config
            information.
    """
    dfs = []
    for i, goal_metrics in enumerate(goal_metrics_list):
        # Make placeholder goal metrics if not available.
        if not goal_metrics:
            print(f"Warning: No goal metrics found for element {i}.")
            continue
        assert (
            len(goal_metrics) == 1
        ), f"Expected goal percentage to be evaluated for one model (no cross-validation), got {len(goal_metrics)}"
        goal_metrics = goal_metrics[0]
        run_path = goal_metrics.get("run_path")
        checkpoint_dir = goal_metrics.get("checkpoint_dir")
        goal_eval_path = goal_metrics.get("eval_run_path")

        # Get the run_path from checkpoint_dir if not available.
        if not run_path and checkpoint_dir:
            run_path = pathlib.Path(checkpoint_dir).parent / "run.json"
            assert run_path.exists(), f"Run path {run_path} does not exist."
            run_path = run_path.as_posix()

        # Get the train data split.
        if "train_data_split" in goal_metrics:
            train_data_split = goal_metrics["train_data_split"]
        elif checkpoint_dir is not None:
            train_data_split = get_data_split_from_path(checkpoint_dir, "train")
        else:
            train_data_split = None

        # Get the test data split.
        if "test_data_split" in goal_metrics:
            test_data_split = goal_metrics["test_data_split"]
        elif checkpoint_dir is not None:
            test_data_split = get_data_split_from_path(checkpoint_dir, "test")
        else:
            test_data_split = None

        row = {
            "human_model_name": goal_metrics.get("human_model_name"),
            "run_path": run_path,
            "checkpoint_dir": checkpoint_dir,
            "goal_eval_path": goal_eval_path,
            "train_data_split": train_data_split,
            "test_data_split": test_data_split,
        }

        # Infer MCTS parameters.
        if algorithm in ["pikl", "alphazero"]:
            row.update(infer_mcts_params_from_path(goal_eval_path))

        # One or more rows to add to the DataFrame. If adding goal percentage
        # metrics per episode, add a row for each episode.
        if per_episode:
            goal_metrics_rows = goal_metrics["episode_metrics"]
        else:
            goal_metrics_rows = [goal_metrics["mean_metrics"]]

        for goal_metrics_row in goal_metrics_rows:
            summary_episode_metrics = summarize_episode_metrics(goal_metrics_row)
            n_metrics = len(summary_episode_metrics)
            # Add the shared row information as the first columns of the
            # summary episode metrics.
            for i, (key, value) in enumerate(row.items()):
                summary_episode_metrics.insert(i, key, [value] * n_metrics)

            dfs.append(summary_episode_metrics)

    return pd.concat(dfs, ignore_index=True)


def combine_human_model_metrics(
    human_modeling_metrics_list: List[List[Dict[str, Any]]],
    goal_metrics_list: List[List[Dict[str, Any]]],
    algorithm: Algorithm,
    add_goal_percentage_per_episode: bool = True,
    add_cross_entropy_per_episode: bool = True,
):
    """Combine metrics for the human model.

    Args:
        human_modeling_metrics_list (List[Dict[str, Any]]): A list of lists
            of dictionaries containing the human modeling metrics. Each
            outer list corresponds to a model, and each inner list
            corresponds to an episode.
        goal_metrics_list (List[Dict[str, Any]]): A list of lists of
            dictionaries containing the goal metrics. The structure is
            similar to human_modeling_metrics_list.
        algorithm (Algorithm): The algorithm used for training.
        add_goal_percentage_per_episode (bool): Whether to add the goal
            percentage per episode to the DataFrame. If False, only adds the
            mean goal percentage over the episodes for each time interval.
            Defaults to True.
        add_cross_entropy_per_episode (bool): Whether to add the cross entropy
            per participant and episode to the DataFrame. If False, only adds
            the mean cross entropy over the participants and episodes, weighted
            by the episode lengths. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the combined metrics and config
            information.
    """
    all_rows = []
    for human_modeling_metrics_per_participant, goal_metrics in zip(
        human_modeling_metrics_list, goal_metrics_list
    ):
        # Make placeholder goal metrics if not available.
        run_path = None
        checkpoint_dir = None
        goal_eval_path = None
        if not goal_metrics:
            if human_modeling_metrics_per_participant:
                hm_metrics = human_modeling_metrics_per_participant[0]
                checkpoint_dir = hm_metrics.get("checkpoint_dir")
        else:
            assert (
                len(goal_metrics) == 1
            ), f"Expected goal percentage to be evaluated for one model (no cross-validation), got {len(goal_metrics)}"
            goal_metrics = goal_metrics[0]
            run_path = goal_metrics.get("run_path")
            checkpoint_dir = goal_metrics.get("checkpoint_dir")
            goal_eval_path = goal_metrics.get("eval_run_path")

        # Get the run_path from checkpoint_dir if not available.
        if not run_path and checkpoint_dir:
            run_path = pathlib.Path(checkpoint_dir).parent / "run.json"
            assert run_path.exists(), f"Run path {run_path} does not exist."
            run_path = run_path.as_posix()

        # Make the output DataFrame row.
        if human_modeling_metrics_per_participant:
            modeling_eval_path = human_modeling_metrics_per_participant[0].get(
                "eval_run_path"
            )
            # Data split that the model was evaluated on for human modeling.
            test_data_split = get_data_split_from_path(modeling_eval_path, "test")
        else:
            modeling_eval_path = None
            test_data_split = None

        # Get the data split.
        if checkpoint_dir is not None:
            train_data_split = get_data_split_from_path(checkpoint_dir, "train")
        else:
            train_data_split = None

        row = {
            "run_path": run_path,
            "checkpoint_dir": checkpoint_dir,
            "goal_eval_path": goal_eval_path,
            "modeling_eval_path": modeling_eval_path,
            "train_data_split": train_data_split,
            "test_data_split": test_data_split,
        }

        if modeling_eval_path is not None and algorithm in ["pikl", "alphazero"]:
            config_path = pathlib.Path(modeling_eval_path).parent / "config.json"
            with open(config_path, "r") as f:
                config = json.load(f)
            mcts_config = config["extra_config_updates"]["mcts_config"]

            mcts_param_names = [
                "num_simulations",
                "puct_coefficient",
                "prior_temperature",
                "explore_noops",
                "temperature",
            ]
            for name in mcts_param_names:
                row[name] = mcts_config[name]

        # Compute mean cross-entropy.
        mean_cross_entropy, cross_entropies, episode_lengths = (
            compute_mean_cross_entropy(human_modeling_metrics_per_participant)
        )
        row["cross_entropy_mean"] = mean_cross_entropy
        if add_cross_entropy_per_episode:
            for cross_entropy, episode_length in zip(cross_entropies, episode_lengths):
                ce_row = copy.copy(row)
                ce_row["cross_entropy"] = cross_entropy
                ce_row["episode_length"] = episode_length
                all_rows.append(ce_row)

        # One or more rows to add to the DataFrame. If adding goal percentage
        # metrics per episode, add a row for each episode.
        if goal_metrics:
            if add_goal_percentage_per_episode:
                goal_metrics_rows = goal_metrics["episode_metrics"]
            else:
                goal_metrics_rows = [goal_metrics["mean_metrics"]]

            for goal_metrics_row in goal_metrics_rows:
                row_with_goal_metrics = copy.copy(row)
                row_with_goal_metrics.update(
                    summarize_episode_metrics(goal_metrics_row)
                )
                # TODO: delete
                # for k in goal_metric_keys:
                #     row_with_goal_metrics[k] = goal_metrics_row[k]
                all_rows.append(row_with_goal_metrics)
        else:
            all_rows.append(row)

    metrics_df = pd.DataFrame(all_rows)
    return metrics_df


def get_most_recent_checkpoint_path(dir_path: Union[str, pathlib.Path]) -> pathlib.Path:
    """
    Get the path of the most recent checkpoint file in the given directory.

    Args:
        dir_path (str): The directory path where the checkpoint files are located.

    Returns:
        str: The path of the most recent checkpoint file.

    Raises:
        ValueError: If no checkpoint files are found in the directory.
    """
    checkpoint_paths = list(pathlib.Path(dir_path).glob("checkpoint_*"))
    if not checkpoint_paths:
        raise ValueError(f"No checkpoints found in {dir_path}")
    return max(checkpoint_paths, key=lambda x: int(x.stem[len("checkpoint_") :]))


def get_validation_experiments(
    train_tag: str,
    algorithm: Algorithm,
    num_players: int,
    validation: Optional[str],
    includes_slurm_job_id: bool = True,
    most_recent_experiment_per_validation_participant: bool = True,
) -> List[Tuple[pathlib.Path, Algorithm, Optional[int]]]:
    if num_players == 1:
        num_players_pattern = "1_player"
    elif num_players == 2:
        num_players_pattern = "2_players"
    else:
        raise ValueError(f"Invalid number of players: {num_players}")

    # Get the paths for the cross-validation runs. The wildcards match [slurm_job_id] (optional),
    # "validation_{validation_participant_id}", timestamp, and sacred_run_id
    # if using leave-one-out validation (i.e., validation = "any" or a
    # specific participant ID); otherwise, the "validation_{validation_participant_id}"
    # subdir is not present.
    if validation == "any":
        val_pattern = "validation_*/"
    elif isinstance(validation, str):
        val_pattern = f"validation_{validation}/"
    elif validation is None:
        val_pattern = ""
    else:
        raise ValueError(f"Invalid validation: {validation}")

    # The Slurm job ID is usually included in the path, except when a BC
    # checkpoint is converted to an AlphaZero checkpoint for piKL eval.
    slurm_job_id_pattern = "*" if includes_slurm_job_id else ""

    run_paths_pattern = os.path.join(
        ROOT_DIR,
        "data/logs",
        ALGORITHM_TO_NAME[algorithm],
        num_players_pattern,
        "11x10x10/craftassist/",
        train_tag,
        slurm_job_id_pattern,  # Slurm job ID
        val_pattern,  # Validation participant ID (could be empty)
        "*",  # Timestamp
        "*",  # Sacred run ID
        "run.json",
    )
    cv_run_paths = list(glob.glob(run_paths_pattern, recursive=False))
    run_paths_algos_and_val_participant_ids: List[
        Tuple[pathlib.Path, Algorithm, Optional[int]]
    ] = []
    for run_path in cv_run_paths:
        run_path: pathlib.Path = pathlib.Path(run_path)
        if validation is None:
            val_participant_id = None
        else:
            val_participant_id_str = run_path.parts[-4]
            assert val_participant_id_str.startswith(
                "validation_"
            ), f"Unexpected path: {cv_run_paths[0]}"
            val_participant_id = int(val_participant_id_str[len("validation_") :])
        run_paths_algos_and_val_participant_ids.append(
            (run_path, algorithm, val_participant_id)
        )

    if most_recent_experiment_per_validation_participant:
        val_participant_id_to_run_path_algo = {}
        for (
            run_path,
            algo,
            val_participant_id,
        ) in run_paths_algos_and_val_participant_ids:
            if val_participant_id not in val_participant_id_to_run_path_algo:
                val_participant_id_to_run_path_algo[val_participant_id] = (
                    run_path,
                    algo,
                )
            else:
                existing_run_path, _ = val_participant_id_to_run_path_algo[
                    val_participant_id
                ]
                existing_run_datetime = datetime.strptime(
                    existing_run_path.parts[-3], "%Y-%m-%d_%H-%M-%S"
                )
                run_datetime = datetime.strptime(
                    run_path.parts[-3], "%Y-%m-%d_%H-%M-%S"
                )
                if run_datetime > existing_run_datetime:
                    val_participant_id_to_run_path_algo[val_participant_id] = (
                        run_path,
                        algo,
                    )
        run_paths_algos_and_val_participant_ids = []
        for val_participant_id, (
            run_path,
            algo,
        ) in val_participant_id_to_run_path_algo.items():
            run_paths_algos_and_val_participant_ids.append(
                (run_path, algo, val_participant_id)
            )

    return run_paths_algos_and_val_participant_ids


def get_run_info(run_path: Union[str, pathlib.Path]) -> Optional[Dict]:
    with open(run_path, "r") as run_fname:
        try:
            return cast(Dict, json.load(run_fname))
        except json.decoder.JSONDecodeError:
            return None


def get_run_status(run_path: Union[str, pathlib.Path]) -> Optional[str]:
    # If run_info is None, return None.
    run_info = get_run_info(run_path) or {}
    return run_info.get("status")


def is_run_completed_or_running(
    run_path: Union[str, pathlib.Path]
) -> Tuple[bool, Optional[str], Optional[Dict]]:
    run_info = get_run_info(run_path) or {}
    status = run_info.get("status")
    if status == "RUNNING":
        # Check if the file was modified in the past hour. If not, the run
        # likely failed and Sacred was unable to set the status to "FAILED".
        completed_or_running = time.time() - os.path.getmtime(run_path) < 60 * 60
    else:
        completed_or_running = status == "COMPLETED"

    return completed_or_running, status, run_info


def is_run_completed(run_path: Union[str, pathlib.Path]) -> bool:
    return get_run_status(run_path) == "COMPLETED"


def get_human_modeling_eval_subdir_name_or_pattern(
    experiment_tag: Optional[str],
    participant_ids: Optional[List[int]],
    timestamp: Optional[str],
):
    if experiment_tag is None:
        experiment_tag = "*"
    participant_ids_str = (
        "_".join(map(str, participant_ids)) if participant_ids is not None else "*"
    )
    if timestamp is None:
        timestamp = "*"
    return f"evaluate_human_modeling_{experiment_tag}_participants_{participant_ids_str}_{timestamp}"


# Mapping from the most up-to-date human_model_name in the data returned by
# load_human_model_df to the human_model_name used for saving results. This is needed
# when the human_model_name is updated in the spreadsheet after results have already
# been saved.
HUMAN_MODEL_NAME_MAP = {
    "bc_combined_prev_action_20250106_40": "bc_combined_lr_prev_action_20250106_40"
}


def load_human_model_df() -> pd.DataFrame:
    """Load spreadsheet with human models for comparing AlphaZero assistants."""
    sheet_id = "1TNVQA9KEof014eav_ymHu6T7hEct5Z_Khc-rB9HaTOk"
    gid = "615970647"
    df = pd.read_csv(
        f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&id={sheet_id}&gid={gid}"
    )
    df = df.dropna(how="all", subset=["human_model_checkpoint"], inplace=False)
    return df.astype(
        {
            "validation_cross_entropy_human_alone": float,
            "validation_cross_entropy_human_with_assistant": float,
        }
    )


def get_bc_to_alphazero_conversion_env_vars_for_human_model_name(
    experiment_config: Dict, human_model_df: pd.DataFrame
) -> Dict:
    """Get environment variables for converting a BC model to an AlphaZero model.

    This function is for converting a BC model corresponding to
    `experiment_config["human_model_name"]`, which must be in `human_model_df`,
    to an AlphaZero model. The main difference between this function and
    `get_bc_to_alphazero_conversion_env_vars` is the experiment tag and that it
    doesn't check for trained BC models in my data directory defined by
    `experiment_config`.
    """
    experiment_config = copy.deepcopy(experiment_config)
    # Get the row for the human model.
    human_model_name = experiment_config.pop("human_model_name")
    human_model_df = human_model_df[
        human_model_df["human_model_name"] == human_model_name
    ]
    assert (
        len(human_model_df) == 1
    ), f"Expected 1 row for human_model_name {human_model_name}, got {len(human_model_df)}"
    human_model_checkpoint = pathlib.Path(
        human_model_df["human_model_checkpoint"].iloc[0]
    )

    # Map the human model name to the name used for saving results.
    human_model_name = HUMAN_MODEL_NAME_MAP.get(human_model_name, human_model_name)

    # If using leave-one-out validation, get the checkpoint for the validation participant instead of the one trained on all of the participants.
    validation_participant_ids = experiment_config["VALIDATION_PARTICIPANT_IDS"]
    if validation_participant_ids is not None:
        validation_participant_ids = int(validation_participant_ids)
        checkpoint_dir_name = human_model_checkpoint.stem
        # Parent directory for the checkpoint trained on the participants leaving out the validation participant.
        validation_dir = (
            pathlib.Path(*human_model_checkpoint.parts[:-3])
            / f"validation_{validation_participant_ids}"
        )
        # Pattern: {validation_dir}/{timestamp}/{slurm_job_id}/{checkpoint_dir_name}
        validation_human_model_checkpoints = list(
            (validation_dir).rglob(f"*/*/{checkpoint_dir_name}")
        )
        assert len(validation_human_model_checkpoints) == 1, (
            f"Expected 1 validation checkpoint for {human_model_checkpoint}, "
            f"got {len(validation_human_model_checkpoints)}: {validation_human_model_checkpoints}"
        )
        human_model_checkpoint = validation_human_model_checkpoints[0]

    assert (
        human_model_checkpoint.exists()
    ), f"Checkpoint {human_model_checkpoint} not found"

    # Load the human model training config.
    config_path = human_model_checkpoint.parent / "config.json"
    assert config_path.exists(), f"Config {config_path} not found"
    with open(config_path, "r") as f:
        config = json.load(f)

    # Make the environment variables for the BC to AlphaZero experiment.
    bc_to_alphazero_env_vars = {
        key.upper(): config[key]
        for key in [
            "teleportation",
            "inf_blocks",
            "num_layers",
            "hidden_channels",
            "hidden_size",
        ]
    }
    bc_to_alphazero_env_vars.update(experiment_config)
    bc_to_alphazero_env_vars["CHECKPOINT"] = str(human_model_checkpoint)

    # Train data split for the human model. This is used when loading metrics for piKL eval runs to know which split the BC model was trained on.
    data_split_str = human_model_df["human_data_split"].iloc[0]
    data_split = SPREADSHEET_DATA_SPLIT_MAP.get(data_split_str)
    if data_split is None:
        raise ValueError(f"Invalid data_split: {data_split_str}")
    bc_to_alphazero_env_vars["SPLIT"] = data_split

    # Experiment tag.
    tag = make_alphazero_from_bc_tag(bc_to_alphazero_env_vars, human_model_name)

    bc_to_alphazero_env_vars["TAG"] = tag

    return bc_to_alphazero_env_vars


def get_human_modeling_eval_dir_pattern(
    checkpoint_dir: pathlib.Path,
    experiment_tag: str,
    out_parent_dir: Optional[str] = None,
    participant_ids: Optional[List[int]] = None,
    timestamp: Optional[str] = None,
) -> str:
    # evaluate_human_modeling.py makes an output subdirectory with the
    # pattern "evaluate_human_modeling_{experiment_tag}_participants_{participant_ids}_{timestamp}".
    if out_parent_dir is None:
        out_parent_dir = checkpoint_dir
    subdir = get_human_modeling_eval_subdir_name_or_pattern(
        experiment_tag, participant_ids=participant_ids, timestamp=timestamp
    )
    return os.path.join(out_parent_dir, subdir)


def get_human_data_dir_for_human_modeling_eval(
    train_split: str, test_split: str
) -> str:
    if test_split == "human_alone":
        if train_split == "human_alone":
            return "/nas/ucb/cassidy/minecraft-building-assistance-game/data/human_data_cleaned/human_alone/infinite_blocks_true/rllib_with_own_noops_flat_actions_flat_observations_place_wrong_reward_-1_repaired_player_0_inventory_0"
        elif train_split == "human_with_assistant" or train_split == "combined":
            return "/nas/ucb/cassidy/minecraft-building-assistance-game/data/human_data_cleaned/human_alone/infinite_blocks_true/rllib_with_own_noops_flat_actions_flat_observations_place_wrong_reward_-1_repaired_player_0_inventory_0_1"
    elif test_split == "human_with_assistant":
        if train_split == "human_alone":
            return "/nas/ucb/cassidy/minecraft-building-assistance-game/data/human_data_cleaned/human_with_assistant/infinite_blocks_true/rllib_with_own_noops_flat_actions_flat_observations_place_wrong_reward_-1_repaired_player_1_inventory_1"
        elif train_split == "human_with_assistant" or train_split == "combined":
            return "/nas/ucb/cassidy/minecraft-building-assistance-game/data/human_data_cleaned/human_with_assistant/infinite_blocks_true/rllib_with_own_noops_flat_actions_flat_observations_place_wrong_reward_-1_repaired_player_1_inventory_0_1"

    raise ValueError(
        f"Invalid train_split {train_split} and test_split {test_split} for human modeling eval."
    )


def get_human_modeling_eval_env_vars_and_metrics(
    checkpoint_dir: pathlib.Path,
    experiment_tag: str,
    val_participant_id: Optional[int],
    repeat_eval_if_exists: bool,
    train_data_split: str,
    test_data_split: str,
    inf_blocks: str,
    run: str,
    algorithm_config_updates: Dict,
    out_parent_dir: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    # Check if a human modeling evaluation run already exists.
    participant_ids = None if val_participant_id is None else [val_participant_id]
    # Search for the evaluation run in the output directory. Use timestamp=None
    # to match any timestamp.
    eval_dir_pattern = get_human_modeling_eval_dir_pattern(
        checkpoint_dir,
        experiment_tag,
        out_parent_dir=out_parent_dir,
        participant_ids=participant_ids,
        timestamp=None,
    )
    # print("human modeling eval_dir_pattern:", eval_dir_pattern)
    # The wildcard is for the Sacred run ID.
    evaluate_run_fnames = glob.glob(os.path.join(eval_dir_pattern, "*/run.json"))

    # Load all of the evaluation metrics (may be multiple runs).
    # List of (eval_run_fname, run_info, mtime) tuples, where mtime is the time at which
    # the file was last modified.
    completed_eval_runs = []
    for eval_run_fname in evaluate_run_fnames:
        completed_or_running, status, run_info = is_run_completed_or_running(
            eval_run_fname
        )
        if completed_or_running:
            evaluate_run_exists = True
            if status == "COMPLETED":
                completed_eval_runs.append(
                    (eval_run_fname, run_info, os.path.getmtime(eval_run_fname))
                )
    else:
        evaluate_run_exists = False

    # Get the most recent evaluation run if one exists.
    if completed_eval_runs:
        eval_run_fname, run_info, _ = max(completed_eval_runs, key=lambda x: x[2])
        metrics = {
            "checkpoint_dir": str(checkpoint_dir),
            "eval_run_path": str(eval_run_fname),
            "participant_ids": val_participant_id,
        }
        run_info = cast(Dict, run_info)
        metrics["episode_results"] = run_info["result"]["episode_results"]
    else:
        metrics = None

    # Human modeling evaluation env vars
    if not evaluate_run_exists or repeat_eval_if_exists:
        human_data_dir = get_human_data_dir_for_human_modeling_eval(
            train_data_split, test_data_split
        )
        assert os.path.exists(
            human_data_dir
        ), f"human_data_dir path does not exist: {human_data_dir}"

        env_vars = dict(
            RUN=run,
            CHECKPOINT=checkpoint_dir,
            PARTICIPANT_IDS=participant_ids,
            HUMAN_DATA_DIR=human_data_dir,
            TAG=experiment_tag,
            EXTRA_CONFIG_UPDATES=algorithm_config_updates,
        )
        if out_parent_dir is not None:
            out_dir = get_human_modeling_eval_dir_pattern(
                checkpoint_dir,
                experiment_tag,
                out_parent_dir=out_parent_dir,
                participant_ids=participant_ids,
                timestamp=timestamp,
            )
            env_vars["OUT_DIR"] = out_dir
    else:
        env_vars = None

    return env_vars, metrics


def get_human_goal_eval_subdir_name_or_pattern(
    experiment_tag: Optional[str],
    timestamp: Optional[str],
):
    if experiment_tag is None:
        experiment_tag = "*"
    if timestamp is None:
        timestamp = "*"
    return f"evaluate_{experiment_tag}_{timestamp}"


def get_human_goal_eval_dir_pattern(
    checkpoint_dir: pathlib.Path,
    experiment_tag: str,
    out_parent_dir: Optional[Union[str, pathlib.Path]] = None,
    timestamp: Optional[str] = None,
) -> str:
    """Get the glob pattern for the goal evaluation directory.

    Args:
        checkpoint_dir: Directory of the checkpoint being evaluated.
        experiment_tag: Experiment tag.
        out_parent_dir: Parent directory for the output directory. This is
            the parent of `out_dir` in `scripts/evaluate.py`.
        timestamp: Timestamp of the evaluation run.

    Returns:
        str: The glob pattern for the goal evaluation directory.
    """
    if out_parent_dir is None:
        out_parent_dir = checkpoint_dir
    subdir = get_human_goal_eval_subdir_name_or_pattern(
        experiment_tag, timestamp=timestamp
    )
    return os.path.join(out_parent_dir, subdir)


def load_episode_metrics(path):
    with open(path, "r") as f:
        metrics = json.load(f)

    dfs = []
    for i, episode_metrics in enumerate(metrics["episode_metrics"]):
        df = summarize_episode_metrics(episode_metrics)
        df["episode"] = i
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def get_human_goal_eval_env_vars_and_metrics(
    checkpoint_dir: pathlib.Path,
    run_path: pathlib.Path,
    repeat_eval_if_exists: bool,
    run: str,
    algorithm_config_updates: List[Dict],
    experiment_tag: Optional[str] = None,
    out_parent_dir: Optional[str] = None,
    timestamp: Optional[str] = None,
    num_episodes: int = 1000,
    debug: bool = False,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Get environment variables and metrics for goal evaluation.

    Args:
        checkpoint_dir: Directory of the checkpoint being evaluated.
        run_path: Path to the run.json file for the checkpoint being evaluated.
        repeat_eval_if_exists: Whether to repeat evaluation if it already exists.
        run: Name of the run as defined in train.py (e.g., "MbagAlphaZero", "BC").
        algorithm_config_updates: Updates to the algorithm configuration for each
            player. In general, the first element is for the human and the second
            is for the assistant. However, only human_alone evaluation is currently
            supported, so this should be a one-element list with config updates
            for the human.
        experiment_tag: Tag for the experiment.
        out_parent_dir: Parent directory for the output directory where
            evaluation results are saved.
        timestamp: Timestamp for the evaluation run. Used to check if an
            evaluation run already exists and to determine where to save the
            evaluation results.
        num_episodes: Number of episodes to evaluate.
        debug: Whether to print debug information.
    """
    # Goal percentage evaluation.
    subset = "test"
    if experiment_tag is None:
        experiment_tag = subset
    env_config_updates = {
        "horizon": 1500,
        "goal_generator_config": {
            "goal_generator_config": {
                "subset": subset,
            }
        },
    }

    # Check if a goal evaluation run already exists.
    eval_dir_pattern = get_human_goal_eval_dir_pattern(
        checkpoint_dir, experiment_tag, out_parent_dir=out_parent_dir, timestamp=None
    )
    evaluate_dirs = glob.glob(eval_dir_pattern)
    evaluate_run_exists = False
    metrics_fnames_and_mtimes = []
    for evaluate_dir in evaluate_dirs:
        eval_metrics_paths = list(pathlib.Path(evaluate_dir).rglob("metrics.json"))
        if len(eval_metrics_paths) > 1:
            raise ValueError(
                f"Multiple metrics.json files found in {evaluate_dir}: {[str(p) for p in eval_metrics_paths]}"
            )
        if eval_metrics_paths:
            # if evaluate_run_exists:
            #     raise ValueError(
            #         f"Multiple evaluation runs found in {checkpoint_dir}: {evaluate_dirs}"
            #     )
            evaluate_run_exists = True
            metrics_fname = eval_metrics_paths[0]
            eval_run_path = metrics_fname.with_name("run.json")
            # Add the metrics file and its modification time to the list if the evaluation run is completed.
            if is_run_completed(eval_run_path):
                mtime = os.path.getmtime(evaluate_dir)
                metrics_fnames_and_mtimes.append((metrics_fname, mtime))
        # If the directory was modified in the past two hours but metrics.json
        # does not exist, the run is likely still running.
        elif time.time() - os.path.getmtime(evaluate_dir) < 2 * 60 * 60:
            evaluate_run_exists = True

    if not evaluate_run_exists or repeat_eval_if_exists:
        # Use more workers if the run is AlphaZero or piKL since MCTS
        # requires more compute.
        if run == "MbagAlphaZero":
            num_workers = 16
        else:
            num_workers = 8
        # Make env vars for goal percentage evaluation.
        env_vars = dict(
            RUN=run,
            CHECKPOINT=str(checkpoint_dir),
            TAG=experiment_tag,
            NUM_EPISODES=num_episodes,
            ALGORITHM_CONFIG_UPDATES=algorithm_config_updates,
            ENV_CONFIG_UPDATES=env_config_updates,
            NUM_WORKERS=num_workers,
        )
        if out_parent_dir is not None:
            out_dir = get_human_goal_eval_dir_pattern(
                checkpoint_dir,
                experiment_tag,
                out_parent_dir=out_parent_dir,
                timestamp=timestamp,
            )
            env_vars["OUT_DIR"] = out_dir
    else:
        env_vars = None

    if metrics_fnames_and_mtimes:
        latest_metrics_fname, _ = max(metrics_fnames_and_mtimes, key=lambda x: x[1])
        if len(metrics_fnames_and_mtimes) > 1 and debug:
            print("Multiple metrics found. Loading metrics from", latest_metrics_fname)
        with open(latest_metrics_fname, "r") as metrics_file:
            metrics = json.load(metrics_file)
        metrics["run_path"] = str(run_path)
        metrics["checkpoint_dir"] = str(checkpoint_dir)
        metrics["eval_run_path"] = str(pathlib.Path(latest_metrics_fname).parent)
    else:
        metrics = None

    return env_vars, metrics


def filter_completed_runs(run_paths_algos_and_val_participant_ids):
    """Filter out runs that have not completed training."""
    completed = []
    for (
        run_path,
        algorithm,
        val_participant_id,
    ) in run_paths_algos_and_val_participant_ids:
        if is_run_completed(run_path):
            completed.append((run_path, algorithm, val_participant_id))
    return completed


def get_eval_dir(
    checkpoint_dir: Union[pathlib.Path, str],
    test_data_split: str,
    algorithm_config_updates: Optional[
        Union[Dict[str, Any], List[Dict[str, Any]]]
    ] = None,
):
    eval_dir = checkpoint_dir
    if algorithm_config_updates:
        if isinstance(algorithm_config_updates, list):
            assert len(algorithm_config_updates) == 1
            human_algorithm_config_updates = algorithm_config_updates[0]
        else:
            human_algorithm_config_updates = algorithm_config_updates

        mcts_eval_tag = make_mcts_eval_tag(human_algorithm_config_updates)
        eval_dir = os.path.join(eval_dir, mcts_eval_tag)

    assert test_data_split is not None
    eval_dir = os.path.join(eval_dir, f"test_split_{test_data_split}")

    return eval_dir


def get_human_eval_env_vars_and_metrics_for_experiment(
    experiment_config: Dict[str, Any],
    agent: Agent,
    goal_eval: bool = True,
    human_modeling_eval: bool = True,
    require_training_completed: bool = True,
    repeat_eval_if_exists: bool = False,
    human_model_df: Optional[pd.DataFrame] = None,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """Make environment variables for human model evaluation.

    Evaluation involves human modeling - computing cross-entropy loss between the human model and the real human data - and goal completion percentages.

    Args:
        experiment_config (Dict[str, Any]): The experiment configuration.
        agent (Agent): The agent to be evaluated.
        goal_eval (bool, optional): Whether to perform goal evaluation. Defaults
            to True.
        human_modeling_eval (bool, optional): Whether to perform human modeling
            evaluation (cross-entropy/accuracy of the human data). Defaults to
            True.
        require_training_completed (bool, optional): Whether to require training
            to be completed before evaluation. Defaults to True.
        jobs (bool, optional): Whether to create evaluation jobs. Defaults to True.
        repeat_eval_if_exists (bool, optional): Whether to repeat evaluation if it
            already exists. Defaults to False.
        human_model_df (pd.DataFrame, optional): DataFrame containing human models
            and checkpoints. Must be provided if `experiment_config` contains
            "human_model_name". Defaults to None.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any], List[Dict, str, Any]]]: A tuple containing the environment variables for human modeling evaluation, human modeling metrics, environment variables for human goal evaluation, and goal evaluation metrics.
    """
    assert (
        goal_eval or human_modeling_eval
    ), "At least one evaluation type must be enabled."
    algorithm = experiment_config["algorithm"]
    assert algorithm in [
        "bc",
        "pikl",
    ], f"Human eval only supported for BC and piKL, got {algorithm}."

    if algorithm == "bc":
        # Use force_train=True to get the environment variables since the training
        # run may exist (should exist to be able to run eval).
        env_vars, algorithm = get_env_vars_for_experiment(
            experiment_config,
            agent,
            experiments_df=None,
            use_most_recent_pretrain_checkpoint=True,
            use_incomplete_pretrain_checkpoint=True,
            force_pretrain=False,
            force_train=True,
        )
        train_data_split = env_vars["SPLIT"]
        test_data_split = env_vars["TEST_SPLIT"]
        train_tag = make_train_tag(env_vars, algorithm, agent)
        includes_slurm_job_id = True
    elif algorithm == "pikl":
        config = copy.deepcopy(DEFAULT_HUMAN_MCTS_CONFIG)
        config.update(experiment_config)
        experiment_config = config

        # Set the algorithm to AlphaZero because that the BC checkpoints were
        # converted to AlphaZero checkpoints to run piKL.
        algorithm = "alphazero"
        # The converted checkpoints don't have a Slurm job ID in the path.
        includes_slurm_job_id = False

        # Get the experiment train tag.
        # If the human model name is specified, use a different method of getting
        # the train tag. This is for human models in the spreadsheet, which may
        # have arbitrary checkpoints.
        if "human_model_name" in experiment_config:
            bc_to_alphazero_env_vars = (
                get_bc_to_alphazero_conversion_env_vars_for_human_model_name(
                    experiment_config, human_model_df
                )
            )
            train_tag = bc_to_alphazero_env_vars["TAG"]
            # Train and test data splits are optional.
            train_data_split = bc_to_alphazero_env_vars.get("SPLIT")
            test_data_split = bc_to_alphazero_env_vars.get("TEST_SPLIT")
        else:
            # Set the algorithm to BC to get the environment variables for the BC model that
            # piKL uses.
            experiment_config["algorithm"] = "bc"

            # Use force_train=True to get the environment variables since the training
            # run may exist (should exist to be able to run eval).
            bc_env_vars, _ = get_env_vars_for_experiment(
                experiment_config,
                agent,
                experiments_df=None,
                use_most_recent_pretrain_checkpoint=True,
                use_incomplete_pretrain_checkpoint=True,
                force_pretrain=False,
                force_train=True,
            )
            assert bc_env_vars is not None
            train_data_split = bc_env_vars["SPLIT"]
            test_data_split = bc_env_vars["TEST_SPLIT"]

            # Train tag for the piKL (AlphaZero) checkpoint created from the BC checkpoint.
            train_tag = make_alphazero_from_bc_tag(bc_env_vars)
    else:
        raise NotImplementedError(f"Unsupported algorithm: {algorithm}")

    # Get the paths for the cross-validation and no cross-validation runs.
    run_paths_algos_and_val_participant_ids = []
    if human_modeling_eval:
        num_players = 1 if train_data_split == "human_alone" else 2
        run_paths_algos_and_val_participant_ids.extend(
            get_validation_experiments(
                train_tag,
                algorithm,
                num_players,
                validation="any",
                includes_slurm_job_id=includes_slurm_job_id,
            )
        )
    if goal_eval:
        assert (
            test_data_split == "human_alone"
        ), f"Goal eval only currently supported for human_alone data split, got {test_data_split}."
        # NOTE: even if the checkpoint was trained with 2 players, we use num_players=1
        # for goal evaluation because the test data split is always human_alone, and we
        # initially saved the results to a path containing "1_player". num_players=1 is
        # needed to match the path for backwards compatibility.
        run_paths_algos_and_val_participant_ids.extend(
            get_validation_experiments(
                train_tag,
                algorithm,
                num_players=1,
                validation=None,
                includes_slurm_job_id=includes_slurm_job_id,
            )
        )

    # Filter out runs that have not completed training.
    if require_training_completed:
        run_paths_algos_and_val_participant_ids = filter_completed_runs(
            run_paths_algos_and_val_participant_ids
        )

    human_modeling_env_vars_list = []
    human_modeling_metrics_list = []
    human_goal_env_vars_list = []
    human_goal_metrics_list = []

    # Timestamp that may be added to the eval results directory.
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for (
        run_path,
        algorithm,
        val_participant_id,
    ) in run_paths_algos_and_val_participant_ids:
        # True if goal evaluation, False if human modeling evaluation.
        curr_goal_eval = val_participant_id is None
        curr_human_modeling_eval = not curr_goal_eval

        with open(run_path.parent / "config.json", "r") as f:
            config = json.load(f)
        inf_blocks = str(config["inf_blocks"]).lower()
        run = ALGORITHM_TO_NAME[algorithm]

        # Get the most recent checkpoint.
        checkpoint_dir = get_most_recent_checkpoint_path(run_path.parent)

        # Algorithm config updates.
        if run == "MbagAlphaZero":
            num_simulations = experiment_config["num_simulations"]
            puct_coeff = experiment_config["puct_coefficient"]
            # puct_coefficient_schedule was implemented on a feature branch but not incorporated into the main branch. Make sure we don't use it.
            puct_coefficient_schedule = experiment_config.get(
                "puct_coefficient_schedule"
            )
            assert (
                puct_coefficient_schedule is None
            ), "puct_coefficient_schedule is not implemented."
            prior_temp = experiment_config["prior_temperature"]
            explore_noops = experiment_config["explore_noops"]
            human_algorithm_config_updates = {
                "mcts_config": {
                    "num_simulations": num_simulations,
                    "temperature": 1,
                    "puct_coefficient": puct_coeff,
                    "add_dirichlet_noise": False,
                    "temperature_schedule": None,
                    "argmax_tree_policy": False,
                    "use_bilevel_action_selection": True,
                    "fix_bilevel_action_selection": True,
                    "prior_temperature": prior_temp,
                    "explore_noops": explore_noops,
                    "init_q_with_max": False,
                    "sample_from_full_support_policy": True,
                },
                "use_goal_predictor": False,
                "use_critic": False,
                "explore": True,
            }
            if puct_coefficient_schedule is not None:
                human_algorithm_config_updates["mcts_config"][
                    "puct_coefficient_schedule"
                ] = puct_coefficient_schedule
        else:
            human_algorithm_config_updates = {}

        # For goal eval, the algorithm config updates are a list because they are
        # per player. Since only human_alone goal eval is currently supported,
        # we make it a one-lement list. For human modeling eval, the updates are
        # for the human model alone, so a list is not required.
        if curr_goal_eval:
            algorithm_config_updates = [human_algorithm_config_updates]
        else:
            algorithm_config_updates = human_algorithm_config_updates

        # Use the test data split for both goal evaluation and human modeling
        # evaluation. For goal eval, the options are "human_alone" and
        # "human_with_assistant", which evaluate the human model alone and with
        # the assistant, respectively. For human modeling eval, we use the
        # provided human data split to compute the cross entropy, so "combined"
        # is also an option.
        eval_out_parent_dir = get_eval_dir(
            checkpoint_dir,
            test_data_split=test_data_split,
            algorithm_config_updates=algorithm_config_updates,
        )

        if curr_goal_eval:
            assert not curr_human_modeling_eval
            # Goal percentage evaluation. Only performed when trained on the
            # full dataset without leave-one-out validation.
            # Add a subdirectory with a timestamp to the output directory because the goal eval does not do this automatically.
            goal_env_vars, goal_metrics = get_human_goal_eval_env_vars_and_metrics(
                checkpoint_dir,
                run_path,
                repeat_eval_if_exists,
                run,
                algorithm_config_updates,
                out_parent_dir=eval_out_parent_dir,
                timestamp=timestamp,
                num_episodes=100,
            )
            if goal_env_vars is not None:
                human_goal_env_vars_list.append(goal_env_vars)
            if goal_metrics is not None:
                goal_metrics.update(
                    {
                        "train_data_split": train_data_split,
                        "test_data_split": test_data_split,
                        "human_model_name": experiment_config.get("human_model_name"),
                    }
                )
                human_goal_metrics_list.append(goal_metrics)
        else:
            assert curr_human_modeling_eval and not curr_goal_eval
            # Human modeling evaluation. Requires leave-one-out validation.
            assert (
                train_data_split in ALL_DATA_SPLITS
            ), f"Unexpected train data split: {train_data_split}"
            assert (
                test_data_split in ALL_DATA_SPLITS
            ), f"Unexpected test data split: {test_data_split}"

            experiment_tag = test_data_split
            # Use out_parent_dir for the output directory because evaluate_human_modeling.py automatically adds a new subdirectory with the participant IDs, experiment tag, and timestamp.
            human_model_env_vars, human_model_metrics = (
                get_human_modeling_eval_env_vars_and_metrics(
                    checkpoint_dir,
                    experiment_tag,
                    val_participant_id,
                    repeat_eval_if_exists,
                    train_data_split,
                    test_data_split,
                    inf_blocks,
                    run,
                    algorithm_config_updates,
                    out_parent_dir=eval_out_parent_dir,
                    timestamp=timestamp,
                )
            )
            if human_model_env_vars is not None:
                human_modeling_env_vars_list.append(human_model_env_vars)
            if human_model_metrics is not None:
                human_model_metrics.update(
                    {
                        "train_data_split": train_data_split,
                        "test_data_split": test_data_split,
                        "human_model_name": experiment_config.get("human_model_name"),
                    }
                )
                human_modeling_metrics_list.append(human_model_metrics)

    return (
        human_modeling_env_vars_list,
        human_modeling_metrics_list,
        human_goal_env_vars_list,
        human_goal_metrics_list,
    )


def make_human_modeling_eval_command(env_vars):
    env_vars_str = make_env_vars_str(env_vars)
    return f"{env_vars_str} sbatch scripts/slurm_eval_human_modeling.sh"


def make_human_goal_eval_command(env_vars):
    env_vars_str = make_env_vars_str(env_vars)
    return f"{env_vars_str} sbatch scripts/slurm_eval_human_goal.sh"


def get_bc_to_alphazero_conversion_env_vars(
    experiment_config: Dict[str, Any], agent: Agent
):
    # Use force_train=True to get the environment variables since the training
    # run exists (should exist to be able to run eval).
    bc_env_vars, algorithm = get_env_vars_for_experiment(
        experiment_config,
        agent,
        experiments_df=None,
        use_most_recent_pretrain_checkpoint=True,
        use_incomplete_pretrain_checkpoint=True,
        force_pretrain=False,
        force_train=True,
    )
    assert algorithm == "bc", f"Unexpected algorithm: {algorithm}"

    # Get the train tag for the BC checkpoint.
    bc_train_tag = make_train_tag(bc_env_vars, algorithm, agent)
    # Get the training run for the specified validation participation.
    num_players = 1 if bc_env_vars["SPLIT"] == "human_alone" else 2
    run_paths_algos_and_val_participant_ids = get_validation_experiments(
        bc_train_tag,
        algorithm,
        num_players,
        validation=bc_env_vars["VALIDATION_PARTICIPANT_IDS"],
    )
    assert (
        len(run_paths_algos_and_val_participant_ids) == 1
    ), f"Expected one BC training run, got {len(run_paths_algos_and_val_participant_ids)}: {run_paths_algos_and_val_participant_ids}"
    run_path = run_paths_algos_and_val_participant_ids[0][0]

    # Check if the training run exists.
    _, train_run_path = check_train_run_exists(
        algorithm, agent, train_run_dir=run_path.parent
    )
    if train_run_path is not None:
        checkpoint_dir = get_most_recent_checkpoint_path(
            os.path.dirname(train_run_path)
        )
        bc_to_alphazero_env_vars = {
            key: bc_env_vars[key]
            for key in ["TELEPORTATION", "INF_BLOCKS", "NUM_LAYERS", "HIDDEN_CHANNELS"]
        }
        # Optional arguments
        for key in ["NUM_SIMULATIONS", "PUCT_COEFFICIENT", "PUCT_COEFFICIENT_SCHEDULE"]:
            if key in bc_env_vars:
                bc_to_alphazero_env_vars[key] = bc_env_vars[key]
        bc_to_alphazero_env_vars["CHECKPOINT"] = checkpoint_dir
        bc_to_alphazero_env_vars["TAG"] = make_alphazero_from_bc_tag(bc_env_vars)

        # Add arguments from algorithm_config_updates and env_config_updates
        # used in goal eval.
        train_args = {
            "temperature": 1,
            "add_dirichlet_noise": False,
            "mcts_config.temperature_schedule": None,
            "argmax_tree_policy": False,
            "use_bilevel_action_selection": True,
            "fix_bilevel_action_selection": True,
            "prior_temperature": 1,
            "explore_noops": False,
            "init_q_with_max": False,
            "sample_from_full_support_policy": True,
            "use_goal_predictor": False,
            "use_critic": False,
            "horizon": 1500,
            "truncate_on_no_progress_timesteps": None,
            "goal_subset": "test",
        }
    else:
        bc_to_alphazero_env_vars = None
        train_args = None
        print(
            f"Training run for {bc_train_tag} does not exist. Skipping BC to AlphaZero conversion."
        )

    return bc_to_alphazero_env_vars, train_args


def make_bc_to_alphazero_conversion_command(env_vars, train_args=None):
    env_vars_str = make_env_vars_str(env_vars)
    train_args_str = (
        " " + make_env_vars_str(train_args, skip_null=False)
        if train_args is not None
        else ""
    )
    return f"{env_vars_str} sbatch scripts/slurm_convert_bc_to_alphazero.sh{train_args_str}"


def make_pikl_eval_goal_command(env_vars):
    env_vars_str = make_env_vars_str(env_vars)
    # Original (resulted in OOM): --cpus-per-task=32 --mem=32gb --gres=gpu:1
    extra_slurm_args = "--cpus-per-task=16 --mem=128gb --gres=gpu:1"
    return f"{env_vars_str} sbatch {extra_slurm_args} scripts/slurm_eval_human_goal.sh"


def make_pikl_eval_human_modeling_command(env_vars):
    env_vars_str = make_env_vars_str(env_vars)
    extra_slurm_args = "--cpus-per-task=32 --mem=32gb --gres=gpu:1"
    return (
        f"{env_vars_str} sbatch {extra_slurm_args} scripts/slurm_eval_human_modeling.sh"
    )


def make_human_modeling_metrics_df(
    human_modeling_metrics_list: List[List[Dict[str, Any]]],
    algorithm: Algorithm,
    per_episode: bool = True,
):
    """Make a DataFrame containing the human modeling metrics.

    Args:
        human_modeling_metrics_list (List[Dict[str, Any]]): A list of lists
            of dictionaries containing the human modeling metrics. Each
            outer list corresponds to a model, and each inner list
            corresponds to an episode.
        algorithm (Algorithm): The algorithm used for training.
        per_episode (bool): Whether to add the cross entropy
            per participant and episode to the DataFrame. If False, only adds
            the mean cross entropy over the participants and episodes, weighted
            by the episode lengths. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the combined metrics and config
            information.
    """
    all_rows = []
    for human_modeling_metrics_per_participant in human_modeling_metrics_list:
        for hm_metrics in human_modeling_metrics_per_participant:
            checkpoint_dir = pathlib.Path(hm_metrics["checkpoint_dir"])

            # Get the run_path from checkpoint_dir.
            run_path = checkpoint_dir.parent / "run.json"
            assert run_path.exists(), f"Run path {run_path} does not exist."
            run_path = run_path.as_posix()

            modeling_eval_path = hm_metrics["eval_run_path"]

            # Data split that the model was evaluated on for human modeling.
            test_data_split = hm_metrics.get(
                "test_data_split"
            ) or get_data_split_from_path(modeling_eval_path, "test")

            # Get the data split.
            train_data_split = hm_metrics.get(
                "train_data_split"
            ) or get_data_split_from_path(checkpoint_dir, "train")

            row = {
                "human_model_name": hm_metrics.get("human_model_name"),
                "train_data_split": train_data_split,
                "test_data_split": test_data_split,
                "run_path": run_path,
                "checkpoint_dir": str(checkpoint_dir),
                "modeling_eval_path": modeling_eval_path,
            }

            if algorithm in ["pikl", "alphazero"]:
                config_path = pathlib.Path(modeling_eval_path).parent / "config.json"
                with open(config_path, "r") as f:
                    config = json.load(f)
                mcts_config = config["extra_config_updates"]["mcts_config"]

                mcts_param_names = [
                    "num_simulations",
                    "puct_coefficient",
                    "prior_temperature",
                    "explore_noops",
                    "temperature",
                ]
                for name in mcts_param_names:
                    row[name] = mcts_config[name]

            # Cross-entropy
            cross_entropies = [
                ep["cross_entropy"] for ep in hm_metrics["episode_results"]
            ]
            lengths = [ep["length"] for ep in hm_metrics["episode_results"]]
            if per_episode:
                for cross_entropy, length in zip(cross_entropies, lengths):
                    ce_row = copy.copy(row)
                    ce_row["cross_entropy"] = cross_entropy
                    ce_row["episode_length"] = length
                    all_rows.append(ce_row)
            else:
                # Add the mean cross entropy, weighted by the episode lengths.
                row["cross_entropy"] = np.average(cross_entropies, weights=lengths)
                all_rows.append(row)

    metrics_df = pd.DataFrame(all_rows)
    return metrics_df


def get_human_modeling_logprobs_for_experiments(
    dil_pikl_eval_human_modeling_metrics_list,
):
    # Check that all experiments have the same participant IDs.
    participant_ids = [
        set(
            participant_metrics["participant_ids"]
            for participant_metrics in experiment_metrics
        )
        for experiment_metrics in dil_pikl_eval_human_modeling_metrics_list
    ]
    assert all(ids == participant_ids[0] for ids in participant_ids), participant_ids

    # Check that all experiments use the same test data split.
    test_data_splits = set()
    for metrics_per_participant in dil_pikl_eval_human_modeling_metrics_list:
        for metrics in metrics_per_participant:
            test_data_splits.add(
                get_data_split_from_path(metrics["eval_run_path"], "test")
            )
    assert len(test_data_splits) == 1, test_data_splits

    logprobs = []

    # metrics_per_participant is a list of metrics for each participant.
    for metrics_per_participant in dil_pikl_eval_human_modeling_metrics_list:
        # Logprobs for all participants and episodes for this experiment.
        experiment_logprobs = []

        # metrics is a dictionary of metrics for all the episodes for a single participant.
        # Loop through the participants in order of increasing participant ID.
        for metrics in sorted(
            metrics_per_participant, key=lambda x: int(x["participant_ids"])
        ):
            # Logprobs for all episodes for this participant.
            participant_logprobs = []

            for episode in metrics["episode_results"]:
                if "logprobs" in episode:
                    participant_logprobs.append(episode["logprobs"])

            experiment_logprobs.append(participant_logprobs)

        logprobs.append(experiment_logprobs)

    return logprobs


def pad_human_modeling_logprobs(logprobs) -> np.ndarray:
    # Ragged array with shape [n_experiments, n_participants, n_episodes, n_steps]
    logprobs = np.asarray(logprobs, dtype=object)

    # Compute the max numbers of episodes and episode lengths to pad the logprobs.
    max_n_episodes = 0
    max_n_steps = 0
    for exp_logprobs in logprobs:
        for participant_logprobs in exp_logprobs:
            max_n_episodes = max(max_n_episodes, len(participant_logprobs))
            for episode_logprobs in participant_logprobs:
                max_n_steps = max(max_n_steps, len(episode_logprobs))

    # Pad logprobs to have the same shape rather than being a ragged array.
    # [n_experiments, n_participants, n_episodes, n_steps]
    pad_logprobs = np.full(logprobs.shape + (max_n_episodes, max_n_steps), -np.inf)
    for i, exp_logprobs in enumerate(logprobs):
        for j, participant_logprobs in enumerate(exp_logprobs):
            for k, episode_logprobs in enumerate(participant_logprobs):
                pad_logprobs[i, j, k, : len(episode_logprobs)] = episode_logprobs

    return pad_logprobs


def valid_logprobs_mask(logprobs: np.ndarray) -> np.ndarray:
    return logprobs > np.log(1e-8)


def compute_mixture_cross_entropy(mixture_logprobs: np.ndarray) -> float:
    # Ignore probabilities that less than 1e-8 because humans sometimes somehow took
    # invalid actions, as well as -inf probabilities because they were padded.
    valid_logprobs = valid_logprobs_mask(mixture_logprobs)
    # Set the invalid logprobs to nan.
    mixture_logprobs[~valid_logprobs] = np.nan
    return -np.nanmean(mixture_logprobs.flatten())
