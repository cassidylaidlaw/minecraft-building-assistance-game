import os
from typing import Optional
import gym
from datetime import datetime
import ray
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.rollout import rollout
from ray.rllib.utils.typing import PolicyID
from sacred import Experiment
from sacred import SETTINGS

from .training_utils import load_trainer

SETTINGS.CONFIG.READ_ONLY_CONFIG = False


ex = Experiment("rollout")


@ex.config
def sacred_config():
    run = "PPO"  # noqa: F841
    checkpoint = ""  # noqa: F841
    episodes = 100  # noqa: F841
    goal_subset = "train"
    experiment_name = f"{goal_subset}"  # noqa: F841

    num_workers = 4
    output_max_file_size = 64 * 1024 * 1024
    config_updates = {  # noqa: F841
        "evaluation_num_workers": num_workers,
        "create_env_on_driver": True,
        "evaluation_num_episodes": 1,
        "output_max_file_size": output_max_file_size,
        "evaluation_config": {},
        "env_config": {
            "goal_generator_config": {
                "subset": goal_subset,
            }
        },
    }


@ex.automain
def main(
    run: str,
    config_updates: dict,
    checkpoint: str,
    experiment_name: str,
    episodes: int,
    _log,
):
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
    )

    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(
        os.path.dirname(checkpoint), f"rollouts_{experiment_name}_{time_str}"
    )
    config_updates.update(
        {
            "output": out_dir,
        }
    )

    trainer = load_trainer(checkpoint, run, config_updates)
    evaluation_workers: Optional[WorkerSet] = trainer.evaluation_workers
    assert evaluation_workers is not None

    # Remove the action_dist_inputs view requirement since it results in massive
    # (multi-gigabyte) JSON rollout files.
    def remove_action_dist_inputs_view_requirement(policy: Policy, policy_id: PolicyID):
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
