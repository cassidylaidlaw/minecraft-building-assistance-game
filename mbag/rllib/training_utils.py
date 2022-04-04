import os
import cloudpickle
from datetime import datetime
from typing import Any, Callable, Dict, Type, Union, cast
from ray.rllib.agents.trainer import Trainer
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.typing import PolicyID, TrainerConfigDict
from ray.tune.logger import UnifiedLogger
from ray.tune.registry import get_trainable_cls


def build_logger_creator(log_dir: str, experiment_name: str):
    experiment_dir = os.path.join(
        log_dir,
        experiment_name,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )

    def custom_logger_creator(config):
        """
        Creates a Unified logger that stores results in
        <log_dir>/<experiment_name>_<timestamp>
        """

        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir, exist_ok=True)
        return UnifiedLogger(config, experiment_dir)

    return custom_logger_creator


def load_trainer_config(checkpoint_path: str) -> TrainerConfigDict:
    # Load configuration from checkpoint file.
    config_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(config_dir, "params.pkl")
    # Try parent directory.
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")

    # If no pkl file found, require command line `--config`.
    if not os.path.exists(config_path):
        raise ValueError(
            "Could not find params.pkl in either the checkpoint dir or "
            "its parent directory!"
        )
    # Load the config from pickled.
    else:
        with open(config_path, "rb") as f:
            config: TrainerConfigDict = cloudpickle.load(f)

    return config


def load_trainer(
    checkpoint_path: str,
    run: Union[str, Type[Trainer]],
    config_updates: dict = {},
) -> Trainer:
    config = load_trainer_config(checkpoint_path)
    config_updates.setdefault("num_workers", 0)
    config = Trainer.merge_trainer_configs(
        config, config_updates, _allow_unknown_configs=True
    )

    # Create the Trainer from config.
    if isinstance(run, str):
        cls = cast(Type[Trainer], get_trainable_cls(run))
    else:
        cls = run
    trainer: Trainer = cls(config=config)
    # Load state from checkpoint.
    trainer.restore(checkpoint_path)

    return trainer


def load_policies_from_checkpoint(
    checkpoint_fname: str,
    trainer: Trainer,
    policy_map: Callable[[PolicyID], PolicyID] = lambda policy_id: policy_id,
):
    """
    Load policy model weights from a checkpoint and copy them into the given
    trainer.
    """

    with open(checkpoint_fname, "rb") as checkpoint_file:
        checkpoint_data = cloudpickle.load(checkpoint_file)
    policy_states: Dict[str, Any] = cloudpickle.loads(checkpoint_data["worker"])[
        "state"
    ]

    policy_weights = {
        policy_map(policy_id): policy_state["weights"]
        for policy_id, policy_state in policy_states.items()
    }

    def copy_policy_weights(policy: Policy, policy_id: PolicyID):
        if policy_id in policy_weights:
            policy.set_weights(policy_weights[policy_id])

    workers: WorkerSet = cast(Any, trainer).workers
    workers.foreach_policy(copy_policy_weights)
