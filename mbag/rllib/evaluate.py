import copy
import json
import os
import pickle
import random
from datetime import datetime
from logging import Logger
from typing import List, Optional, Type

import numpy as np
import ray
import torch
import tqdm
from ray.rllib.algorithms import Algorithm
from ray.rllib.utils.typing import PolicyID
from sacred import SETTINGS, Experiment

from mbag.agents.human_agent import HumanAgent
from mbag.environment.config import DEFAULT_HUMAN_GIVE_ITEMS, merge_configs
from mbag.environment.mbag_env import MbagConfigDict
from mbag.evaluation.evaluator import EpisodeInfo, MbagAgentConfig, MbagEvaluator

from .agents import RllibAlphaZeroAgent, RllibMbagAgent
from .os_utils import available_cpu_count
from .training_utils import load_trainer, load_trainer_config

SETTINGS.CONFIG.READ_ONLY_CONFIG = False
SETTINGS.CONFIG


ex = Experiment("evaluate")


@ex.config
def sacred_config():
    runs = ["PPO"]  # noqa: F841
    checkpoints = [""]  # noqa: F841
    policy_ids = [""]  # noqa: F841
    min_action_interval = 0  # noqa: F841
    explore = False  # noqa: F841
    episodes = 100  # noqa: F841
    experiment_name = ""  # noqa: F841
    seed = 0  # noqa: F841
    record_video = False  # noqa: F841
    use_malmo = record_video  # noqa: F841
    out_dir = None  # noqa: F841

    env_config_updates = {}  # noqa: F841
    algorithm_config_updates = {}  # noqa: F841


@ex.automain
def main(
    runs: List[str],
    checkpoints: List[Optional[str]],
    policy_ids: List[Optional[PolicyID]],
    min_action_interval: float,
    explore: bool,
    episodes: int,
    experiment_name: str,
    env_config_updates: MbagConfigDict,
    algorithm_config_updates: dict,
    seed: int,
    record_video: bool,
    use_malmo: bool,
    out_dir: Optional[str],
    _log: Logger,
):
    ray.init(
        num_cpus=available_cpu_count(),
        ignore_reinit_error=True,
        include_dashboard=False,
    )

    algorithm_config_updates["seed"] = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    algorithm_config_updates.setdefault("num_workers", 0)
    algorithm_config_updates.setdefault("num_envs_per_worker", 1)
    algorithm_config_updates.setdefault("evaluation_num_workers", 0)

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not experiment_name.endswith("_") and experiment_name != "":
        experiment_name += "_"
    if out_dir is None:
        for checkpoint in checkpoints:
            if checkpoint is not None:
                out_dir = os.path.join(
                    checkpoint,
                    f"evaluate_{experiment_name}{time_str}",
                )
                break
    if out_dir is None:
        raise ValueError("out_dir must be set if no checkpoints are provided")
    os.makedirs(out_dir, exist_ok=True)

    # Try to load env config from the first checkpoint.
    env_config: Optional[MbagConfigDict] = {}
    for checkpoint in checkpoints:
        if checkpoint is not None:
            config = load_trainer_config(checkpoint)
            if env_config == {}:
                env_config = copy.deepcopy(config["env_config"])
    assert env_config is not None

    env_config["num_players"] = len(runs)
    while len(env_config["players"]) < env_config["num_players"]:
        env_config["players"].append(copy.deepcopy(env_config["players"][0]))
        player_index = len(env_config["players"]) - 1
        env_config["players"][-1]["player_name"] = f"player_{player_index}"
    for player_index, run in enumerate(runs):
        if run == "HumanAgent":
            env_config["players"][player_index]["is_human"] = True
            env_config["players"][player_index]["give_items"] = DEFAULT_HUMAN_GIVE_ITEMS
            env_config["malmo"]["restrict_players"] = True

    env_config = merge_configs(env_config, env_config_updates)

    algorithm_config_updates["env_config"] = copy.deepcopy(env_config)

    agent_configs: List[MbagAgentConfig] = []
    trainers: List[Algorithm] = []
    for player_index, (run, checkpoint, policy_id) in enumerate(
        zip(runs, checkpoints, policy_ids)
    ):
        if run == "HumanAgent":
            agent_configs.append((HumanAgent, {}))
        else:
            assert checkpoint is not None and policy_id is not None
            _log.info(f"loading policy {policy_id} from {checkpoint}...")
            trainer = load_trainer(checkpoint, run, algorithm_config_updates)
            policy = trainer.get_policy(policy_id)

            mbag_agent_class: Type[RllibMbagAgent] = RllibMbagAgent
            agent_config = {
                "policy": policy,
                "min_action_interval": min_action_interval,
                "explore": explore,
            }
            if "AlphaZero" in run:
                mbag_agent_class = RllibAlphaZeroAgent
                agent_config["player_index"] = player_index

            agent_configs.append(
                (
                    mbag_agent_class,
                    agent_config,
                )
            )
            trainers.append(trainer)

    env_config.setdefault("malmo", {})
    env_config["malmo"]["use_malmo"] = use_malmo
    if record_video:
        env_config["malmo"]["use_spectator"] = True
        env_config["malmo"]["video_dir"] = out_dir

    _log.info(f"evaluating for {episodes} episodes...")
    evaluator = MbagEvaluator(
        env_config,
        agent_configs,
    )

    episode_infos: List[EpisodeInfo] = []
    for _ in tqdm.trange(episodes):
        episode_infos.append(evaluator.rollout())

    out_pickle_fname = os.path.join(out_dir, "episode_info.pickle")
    _log.info(f"saving full results to {out_pickle_fname}")
    with open(out_pickle_fname, "wb") as out_pickle:
        pickle.dump(episode_infos, out_pickle)

    out_json_fname = os.path.join(out_dir, "episode_info.json")
    _log.info(f"saving JSON results to {out_json_fname}")
    with open(out_json_fname, "w") as out_json:
        json.dump([episode_info.to_json() for episode_info in episode_infos], out_json)

    for trainer in trainers:
        trainer.stop()
