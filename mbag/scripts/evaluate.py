import copy
import faulthandler
import itertools
import json
import multiprocessing as mp
import os
import pickle
import random
import signal
import time
import zipfile
from datetime import datetime
from logging import Logger
from typing import Any, Generator, List, Optional, Type, Union

import numpy as np
import torch
import tqdm
from ray.rllib.utils.typing import PolicyID
from sacred import SETTINGS, Experiment

import mbag
from mbag.agents.heuristic_agents import ALL_HEURISTIC_AGENTS
from mbag.agents.human_agent import HumanAgent
from mbag.environment.config import DEFAULT_HUMAN_GIVE_ITEMS, merge_configs
from mbag.environment.mbag_env import MbagConfigDict, MbagEnv
from mbag.evaluation.episode import MbagEpisode
from mbag.evaluation.evaluator import MbagAgentConfig, MbagEvaluator
from mbag.evaluation.metrics import (
    MbagEpisodeMetrics,
    calculate_mean_metrics,
    calculate_metrics,
)
from mbag.rllib.agents import RllibAlphaZeroAgent, RllibMbagAgent
from mbag.rllib.training_utils import load_policy, load_trainer_config

SETTINGS.CONFIG.READ_ONLY_CONFIG = False


ex = Experiment("evaluate")

# Useful for debugging when debugging freezes.
faulthandler.register(signal.SIGUSR1)


@ex.config
def sacred_config():
    runs = ["PPO"]  # noqa: F841
    checkpoints = [""]  # noqa: F841
    policy_ids = [""]  # noqa: F841
    min_action_interval = 0  # noqa: F841
    explore = [False] * len(runs)  # noqa: F841
    confidence_thresholds = None  # noqa: F841
    num_episodes = 100  # noqa: F841
    experiment_name = ""
    experiment_tag = experiment_name  # noqa: F841
    seed = 0  # noqa: F841
    record_video = False  # noqa: F841
    use_malmo = record_video  # noqa: F841
    num_workers: int = 0  # noqa: F841
    out_dir = None  # noqa: F841
    save_episodes = use_malmo  # noqa: F841

    env_config_updates = {}  # noqa: F841
    algorithm_config_updates: List[dict] = [{}]  # noqa: F841
    agent_config_updates: List[dict] = [{}]  # noqa: F841

    # Used by named configs
    assistant_checkpoint = None  # noqa: F841
    assistant_run = None  # noqa: F841
    num_simulations = None  # noqa: F841
    goal_set = None  # noqa: F841
    house_id = None  # noqa: F841


@ex.named_config
def human_alone():
    assistant_checkpoint = None
    goal_set = "test"
    house_id = None

    runs = ["HumanAgent"]  # noqa: F841
    checkpoints = [assistant_checkpoint]  # noqa: F841
    policy_ids = [None]  # noqa: F841
    num_episodes = 1  # noqa: F841
    algorithm_config_updates: List[dict] = [{}]  # noqa: F841
    env_config_updates = {  # noqa: F841
        "goal_generator_config": {
            "goal_generator_config": {"subset": goal_set, "house_id": house_id}
        },
        "malmo": {"action_delay": 0.8, "rotate_spectator": False},
        "horizon": 10000,
        "players": [{"player_name": "human"}],
    }
    min_action_interval = 0.8  # noqa: F841
    use_malmo = True  # noqa: F841


@ex.named_config
def human_with_assistant():
    assistant_checkpoint = None
    assistant_run = "MbagAlphaZero"
    num_simulations = 10
    goal_set = "test"
    house_id = None

    runs = ["HumanAgent", assistant_run]  # noqa: F841
    checkpoints = [None, assistant_checkpoint]  # noqa: F841
    policy_ids = [None, "assistant"]  # noqa: F841
    num_episodes = 1  # noqa: F841
    algorithm_config_updates = [  # noqa: F841
        {},
        {
            "num_gpus": 1 if torch.cuda.is_available() else 0,
            "num_gpus_per_worker": 0,
            "player_index": 1,
            "mcts_config": {"num_simulations": num_simulations},
        },
    ]
    env_config_updates = {  # noqa: F841
        "num_players": 2,
        "goal_generator_config": {
            "goal_generator_config": {"subset": goal_set, "house_id": house_id}
        },
        "malmo": {"action_delay": 0.8, "rotate_spectator": False},
        "horizon": 10000,
        "players": [
            {
                "player_name": "human",
            },
            {
                "player_name": "assistant",
            },
        ],
    }
    min_action_interval = 0.8  # noqa: F841
    use_malmo = True  # noqa: F841


def run_evaluation(
    *,
    runs: List[str],
    checkpoints: List[Optional[str]],
    policy_ids: List[Optional[PolicyID]],
    min_action_interval: float,
    explore: List[bool],
    confidence_thresholds: Optional[List[Optional[float]]],
    env_config_updates: MbagConfigDict,
    algorithm_config_updates: List[dict],
    agent_config_updates: List[dict],
    seed: int,
    record_video: bool,
    use_malmo: bool,
    out_dir: str,
) -> Generator[MbagEpisode, Any, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env_config_updates.setdefault("randomize_first_episode_length", False)

    # Try to load env config from the first checkpoint.
    env_config: Optional[MbagConfigDict] = None
    for checkpoint in checkpoints:
        if checkpoint is not None:
            config = load_trainer_config(checkpoint)
            if env_config is None:
                env_config = copy.deepcopy(config["env_config"])
    assert env_config is not None

    env_config["num_players"] = len(runs)
    while len(env_config["players"]) < env_config["num_players"]:
        env_config["players"].append(copy.deepcopy(env_config["players"][0]))
        player_index = len(env_config["players"]) - 1
        env_config["players"][-1]["player_name"] = f"player_{player_index}"
    env_config["players"] = env_config["players"][: env_config["num_players"]]
    for player_index, run in enumerate(runs):
        if run == "HumanAgent":
            env_config["players"][player_index]["is_human"] = True
            env_config["players"][player_index]["give_items"] = DEFAULT_HUMAN_GIVE_ITEMS
            env_config["malmo"]["restrict_players"] = True

    env_config = merge_configs(env_config, env_config_updates)

    observation_space = MbagEnv(env_config).observation_space

    agent_configs: List[MbagAgentConfig] = []
    for player_index, (run, checkpoint, policy_id, config_updates) in enumerate(
        zip(runs, checkpoints, policy_ids, algorithm_config_updates)
    ):
        if run == "HumanAgent":
            agent_configs.append((HumanAgent, {}))
        elif run in ALL_HEURISTIC_AGENTS:
            agent_configs.append((ALL_HEURISTIC_AGENTS[run], {}))
        else:
            assert checkpoint is not None and policy_id is not None

            config_updates["seed"] = seed
            config_updates.setdefault("num_workers", 0)
            config_updates.setdefault("num_envs_per_worker", 1)
            config_updates.setdefault("evaluation_num_workers", 0)
            config_updates["env_config"] = copy.deepcopy(env_config)

            policy = load_policy(
                checkpoint,
                policy_id,
                config_updates=copy.deepcopy(config_updates),
                observation_space=observation_space,
            )
            # policy = trainer.get_policy(policy_id)

            mbag_agent_class: Type[RllibMbagAgent] = RllibMbagAgent
            agent_config = {
                "policy": policy,
                "min_action_interval": min_action_interval,
                "explore": explore,
            }
            if confidence_thresholds is not None:
                agent_config["confidence_threshold"] = confidence_thresholds[
                    player_index
                ]
            if "AlphaZero" in run:
                mbag_agent_class = RllibAlphaZeroAgent
                agent_config["player_index"] = player_index

            agent_configs.append(
                (
                    mbag_agent_class,
                    agent_config,
                )
            )

    for (agent_cls, agent_config), agent_config_update in zip(
        agent_configs, agent_config_updates
    ):
        agent_config.update(agent_config_update)

    env_config.setdefault("malmo", {})
    env_config["malmo"]["use_malmo"] = use_malmo
    if record_video:
        env_config["malmo"]["use_spectator"] = True
        env_config["malmo"]["video_dir"] = out_dir

    evaluator = MbagEvaluator(
        env_config,
        agent_configs,
        return_on_exception=use_malmo,
    )

    while True:
        yield evaluator.rollout()


def evaluation_worker(
    worker_index: int,
    queue: mp.Queue,
    num_episodes: int,
    *,
    runs: List[str],
    checkpoints: List[Optional[str]],
    policy_ids: List[Optional[PolicyID]],
    min_action_interval: float,
    explore: List[bool],
    confidence_thresholds: Optional[List[Optional[float]]],
    env_config_updates: MbagConfigDict,
    algorithm_config_updates: List[dict],
    agent_config_updates: List[dict],
    seed: int,
    record_video: bool,
    use_malmo: bool,
    out_dir: str,
):
    try:
        faulthandler.register(signal.SIGUSR1)
        for episode in itertools.islice(
            run_evaluation(
                runs=runs,
                checkpoints=checkpoints,
                policy_ids=policy_ids,
                min_action_interval=min_action_interval,
                explore=explore,
                confidence_thresholds=confidence_thresholds,
                env_config_updates=env_config_updates,
                algorithm_config_updates=algorithm_config_updates,
                agent_config_updates=agent_config_updates,
                seed=seed + worker_index,
                record_video=record_video,
                use_malmo=use_malmo,
                out_dir=os.path.join(out_dir, f"worker_{worker_index}"),
            ),
            num_episodes,
        ):
            queue.put(episode)
    except Exception as error:
        queue.put(error)
        raise error


def queue_episode_generator(
    queues: List[mp.Queue],
) -> Generator[Union[MbagEpisode, Exception], Any, Any]:
    while True:
        for queue in queues:
            yield queue.get()


@ex.automain
def main(  # noqa: C901
    runs: List[str],
    checkpoints: List[Optional[str]],
    policy_ids: List[Optional[PolicyID]],
    min_action_interval: float,
    explore: List[bool],
    confidence_thresholds: Optional[List[Optional[float]]],
    num_episodes: int,
    experiment_tag: str,
    env_config_updates: MbagConfigDict,
    algorithm_config_updates: List[dict],
    agent_config_updates: List[dict],
    seed: int,
    record_video: bool,
    use_malmo: bool,
    num_workers: int,
    out_dir: Optional[str],
    save_episodes: bool,
    _log: Logger,
):
    if num_workers > 0:
        mp.set_start_method("spawn")
    mbag.logger.setLevel(_log.getEffectiveLevel())

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not experiment_tag.endswith("_") and experiment_tag != "":
        experiment_tag += "_"
    if out_dir is None:
        for checkpoint in checkpoints:
            if checkpoint is not None:
                out_dir = os.path.join(
                    checkpoint,
                    f"evaluate_{experiment_tag}{time_str}",
                )
                break
    if out_dir is None:
        raise ValueError("out_dir must be set if no checkpoints are provided")
    os.makedirs(out_dir, exist_ok=True)

    processes: List[mp.Process] = []
    queues: List[mp.Queue] = []
    episode_generator: Generator[Union[MbagEpisode, Exception], Any, Any]
    if num_workers == 0:
        episode_generator = run_evaluation(
            runs=runs,
            checkpoints=checkpoints,
            policy_ids=policy_ids,
            min_action_interval=min_action_interval,
            explore=explore,
            confidence_thresholds=confidence_thresholds,
            env_config_updates=env_config_updates,
            algorithm_config_updates=algorithm_config_updates,
            agent_config_updates=agent_config_updates,
            seed=seed,
            record_video=record_video,
            use_malmo=use_malmo,
            out_dir=out_dir,
        )
    else:
        for worker_index in range(num_workers):
            queue: mp.Queue = mp.Queue(maxsize=10)
            worker_num_episodes = num_episodes // num_workers
            if worker_index < num_episodes % num_workers:
                worker_num_episodes += 1
            process = mp.Process(
                target=evaluation_worker,
                args=(
                    worker_index,
                    queue,
                    worker_num_episodes,
                ),
                kwargs={
                    "runs": runs,
                    "checkpoints": checkpoints,
                    "policy_ids": policy_ids,
                    "min_action_interval": min_action_interval,
                    "explore": explore,
                    "confidence_thresholds": confidence_thresholds,
                    "env_config_updates": env_config_updates,
                    "algorithm_config_updates": algorithm_config_updates,
                    "agent_config_updates": agent_config_updates,
                    "seed": seed,
                    "record_video": record_video,
                    "use_malmo": use_malmo,
                    "out_dir": out_dir,
                },
                daemon=True,
            )
            process.start()
            processes.append(process)
            queues.append(queue)
            time.sleep(1)
        episode_generator = queue_episode_generator(queues)

    episodes: List[MbagEpisode] = []
    episode_metrics: List[MbagEpisodeMetrics] = []
    with tqdm.trange(num_episodes) as progress_bar:
        for _ in progress_bar:
            episode_or_exception = next(episode_generator)
            if isinstance(episode_or_exception, Exception):
                raise episode_or_exception
            else:
                episode = episode_or_exception
            if save_episodes:
                episodes.append(episode)
            episode_metrics.append(calculate_metrics(episode))
            mean_goal_percentage = np.mean(
                [metrics["goal_percentage"] for metrics in episode_metrics]
            )
            mean_goal_similarity = np.mean(
                [metrics["goal_similarity"] for metrics in episode_metrics]
            )
            progress_bar.set_postfix(
                mean_goal_similarity=f"{mean_goal_similarity:.1f}",
                mean_goal_percentage=f"{mean_goal_percentage:.3f}",
            )

    for process in processes:
        process.join(timeout=10)
        process.terminate()

    if episodes:
        out_zip_fname = os.path.join(out_dir, "episodes.zip")
        _log.info(f"saving full results to {out_zip_fname}")
        with zipfile.ZipFile(
            out_zip_fname, "w", compression=zipfile.ZIP_DEFLATED
        ) as episodes_zip:
            with episodes_zip.open("episodes.pickle", "w") as out_pickle:
                pickle.dump(episodes, out_pickle)

    metrics = {
        "mean_metrics": calculate_mean_metrics(episode_metrics),
        "episode_metrics": episode_metrics,
    }
    metrics_fname = os.path.join(out_dir, "metrics.json")
    _log.info(f"saving metrics to {metrics_fname}")
    with open(metrics_fname, "w") as metrics_file:
        json.dump(metrics, metrics_file)

    return metrics["mean_metrics"]
