import os
from typing import List, Optional
import gym
import torch
from datetime import datetime
import ray
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.rollout import rollout
from ray.rllib.utils.typing import PolicyID
from ray.rllib.agents import Trainer
from sacred import Experiment
from sacred import SETTINGS

from .training_utils import load_trainer

SETTINGS.CONFIG.READ_ONLY_CONFIG = False


ex = Experiment("rollout")


@ex.config
def sacred_config():
    run = "MbagPPO"  # noqa: F841
    checkpoint = ""  # noqa: F841
    episodes = 100  # noqa: F841
    experiment_name = ""  # noqa: F841
    policy_ids: Optional[List[str]] = None  # noqa: F841
    seed = 0

    num_workers = 4
    output_max_file_size = 64 * 1024 * 1024
    config_updates = {  # noqa: F841
        "seed": seed,
        "evaluation_num_workers": num_workers,
        "create_env_on_driver": True,
        "evaluation_num_episodes": max(num_workers, 1),
        "output_max_file_size": output_max_file_size,
        "evaluation_config": {},
        "env_config": {"malmo": {}},
        "multiagent": {},
        "num_gpus": 1 if torch.cuda.is_available() else 0,
    }
    extra_config_updates = {}  # noqa: F841

    record_video = False  # noqa: F841
    if record_video:
        num_workers = 0


@ex.automain
def main(
    run: str,
    config_updates: dict,
    extra_config_updates: dict,
    checkpoint: str,
    experiment_name: str,
    episodes: int,
    policy_ids: Optional[List[str]],
    record_video: bool,
    _log,
):
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
    )

    config_updates = Trainer.merge_trainer_configs(
        config_updates, extra_config_updates, _allow_unknown_configs=True
    )

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if not experiment_name.endswith("_") and experiment_name != "":
        experiment_name += "_"
    out_dir = os.path.join(
        os.path.dirname(checkpoint), f"rollouts_{experiment_name}{time_str}"
    )
    config_updates["output"] = out_dir

    if policy_ids is not None:

        def policy_mapping_fn(
            agent_id: str, episode, worker, policy_ids=policy_ids, **kwargs
        ):
            agent_index = int(agent_id[len("player_") :])
            return policy_ids[agent_index]

        config_updates["multiagent"]["policy_mapping_fn"] = policy_mapping_fn
        config_updates["env_config"]["num_players"] = len(policy_ids)
        config_updates["env_config"]["malmo"]["player_names"] = policy_ids

    if record_video:
        config_updates["env_config"]["malmo"].update(
            {
                "use_malmo": True,
                "use_spectator": True,
                "video_dir": out_dir,
            }
        )
        config_updates["evaluation_num_workers"] = 1
    trainer = load_trainer(checkpoint, run, config_updates)
    evaluation_workers: Optional[WorkerSet] = trainer.evaluation_workers
    assert evaluation_workers is not None

    # Remove the action_dist_inputs view requirement since it results in massive
    # (multi-gigabyte) JSON rollout files.
    def remove_action_dist_inputs_view_requirement(policy: Policy, policy_id: PolicyID):
        if SampleBatch.ACTION_DIST_INPUTS in policy.view_requirements:
            del policy.view_requirements[SampleBatch.ACTION_DIST_INPUTS]

    evaluation_workers.foreach_policy(remove_action_dist_inputs_view_requirement)

    gym.logger.set_level(gym.logger.INFO)

    rollout(
        trainer,
        None,
        num_steps=0,
        num_episodes=episodes,
    )
    trainer.stop()
