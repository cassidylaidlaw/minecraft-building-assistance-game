import json
from logging import Logger
import os
from typing import List, Optional
import torch
import numpy as np
import random
from datetime import datetime
import ray
import tqdm
import pickle
from ray.rllib.utils.typing import PolicyID
from ray.rllib.agents import Trainer
from sacred import Experiment
from sacred import SETTINGS

from .training_utils import load_trainer
from .agents import RllibMbagAgent, RllibMbagAgentConfigDict
from mbag.evaluation.evaluator import EpisodeInfo, MbagEvaluator
from mbag.environment.mbag_env import MbagConfigDict

SETTINGS.CONFIG.READ_ONLY_CONFIG = False


ex = Experiment("evaluate")


@ex.config
def sacred_config():
    runs = ["PPO"]  # noqa: F841
    checkpoints = [""]  # noqa: F841
    policy_ids = [""]  # noqa: F841
    episodes = 100  # noqa: F841
    experiment_name = ""  # noqa: F841
    seed = 0  # noqa: F841
    record_video = False  # noqa: F841

    config_updates = {}  # noqa: F841


@ex.automain
def main(
    runs: List[str],
    checkpoints: List[str],
    policy_ids: List[PolicyID],
    episodes: int,
    experiment_name: str,
    config_updates: dict,
    seed: int,
    record_video: bool,
    _log: Logger,
):
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
    )

    config_updates["seed"] = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    config_updates["num_workers"] = 0

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not experiment_name.endswith("_") and experiment_name != "":
        experiment_name += "_"
    out_path = os.path.join(
        os.path.dirname(checkpoints[0]), f"evaluate_{experiment_name}{time_str}"
    )

    env_config: Optional[MbagConfigDict] = None
    agent_config_dicts: List[RllibMbagAgentConfigDict] = []
    trainers: List[Trainer] = []
    for run, checkpoint, policy_id in zip(runs, checkpoints, policy_ids):
        _log.info(f"loading policy {policy_id} from {checkpoint}...")
        trainer = load_trainer(checkpoint, run, config_updates)
        agent_config_dicts.append(
            {
                "policy": trainer.get_policy(policy_id),
            }
        )
        trainers.append(trainer)

        if env_config is None:
            env_config = trainer.config["env_config"]
    assert env_config is not None
    env_config["num_players"] = len(agent_config_dicts)

    _log.info(f"evaluating for {episodes} episodes...")
    evaluator = MbagEvaluator(
        env_config,
        [
            (RllibMbagAgent, agent_config_dict)
            for agent_config_dict in agent_config_dicts
        ],
    )

    episode_infos: List[EpisodeInfo] = []
    for _ in tqdm.trange(episodes):
        episode_infos.append(evaluator.rollout())

    out_pickle_fname = out_path + ".pickle"
    _log.info(f"saving full results to {out_pickle_fname}")
    with open(out_pickle_fname, "wb") as out_pickle:
        pickle.dump(episode_infos, out_pickle)

    out_json_fname = out_path + ".json"
    _log.info(f"saving JSON results to {out_json_fname}")
    with open(out_json_fname, "w") as out_json:
        json.dump([episode_info.to_json() for episode_info in episode_infos], out_json)

    for trainer in trainers:
        trainer.stop()
