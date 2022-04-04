"""
Distillation-prediction, where one policy is distilled into a another model
for predicting the next action given past actions.
"""

import logging
import numpy as np
from ray.rllib.execution.concurrency_ops import Concurrently, Dequeue, Enqueue
from ray.rllib.execution.learner_thread import LearnerThread
from ray.rllib.models.torch.attention_net import AttentionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.agents.trainer import Trainer
from ray.rllib.utils.sgd import minibatches
from ray.tune.registry import register_trainable
from ray.rllib.execution.common import (
    AGENT_STEPS_TRAINED_COUNTER,
    STEPS_TRAINED_COUNTER,
    _get_shared_metrics,
)
import torch
from typing import Dict, List, Type, Union, Callable, cast

from ray.rllib.agents import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.rollout_ops import (
    ParallelRollouts,
    ConcatBatches,
    SelectExperiences,
)
from ray.rllib.execution.train_ops import TrainOneStep
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.utils.typing import (
    PolicyID,
    SampleBatchType,
    TrainerConfigDict,
    TensorType,
)
from ray.util.iter import LocalIterator
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.torch_policy import (
    LearningRateSchedule,
    EntropyCoeffSchedule,
    TorchPolicy,
)
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    sequence_mask,
)
from ray.rllib.utils.numpy import convert_to_numpy

from .training_utils import load_policies_from_checkpoint

logger = logging.getLogger(__name__)

# Adds the following updates to the (base) `Trainer` config in
# rllib/agents/trainer.py (`COMMON_CONFIG` dict).
DEFAULT_CONFIG = with_common_config(
    {
        "multiagent": {
            # Function which maps the ID of a policy acting in the environment to the
            # policy ID which should be trained to mimic it.
            "distillation_mapping_fn": None,
        },
        # Optional checkpoint from which to load policy model weights to distill.
        "checkpoint_to_load_policies": None,
        # Size of batches collected from each worker.
        "rollout_fragment_length": 200,
        # Number of timesteps collected for each SGD round. This defines the size
        # of each SGD epoch.
        "train_batch_size": 4000,
        # Total SGD batch size across all devices for SGD. This defines the
        # minibatch size within each epoch.
        "sgd_minibatch_size": 128,
        # Whether to shuffle sequences in the batch when training (recommended).
        "shuffle_sequences": True,
        # Number of SGD iterations in each outer loop (i.e., number of epochs to
        # execute per train batch).
        "num_sgd_iter": 30,
        # Stepsize of SGD.
        "lr": 5e-5,
        # Learning rate schedule.
        "lr_schedule": None,
        # If specified, clip the global norm of gradients by this amount.
        "grad_clip": None,
        # Coefficient of the entropy regularizer.
        "entropy_coeff": 0.0,
        # Decay schedule for the entropy regularizer.
        "entropy_coeff_schedule": None,
        # Whether to rollout "complete_episodes" or "truncate_episodes".
        "batch_mode": "truncate_episodes",
        # Which observation filter to apply to the observation.
        "observation_filter": "NoFilter",
        # Whether to fake GPUs (using CPUs).
        # Set this to True for debugging on non-GPU machines (set `num_gpus` > 0).
        "_fake_gpus": False,
    }
)


def validate_config(config: TrainerConfigDict) -> None:
    # SGD minibatch size must be smaller than train_batch_size (b/c
    # we subsample a batch of `sgd_minibatch_size` from the train-batch for
    # each `sgd_num_iter`).
    if config["sgd_minibatch_size"] > config["train_batch_size"]:
        raise ValueError(
            "`sgd_minibatch_size` ({}) must be <= "
            "`train_batch_size` ({}).".format(
                config["sgd_minibatch_size"], config["train_batch_size"]
            )
        )

    # Multi-gpu not supported for PyTorch and tf-eager.
    if config["framework"] != "torch":
        raise ValueError("only PyTorch is supported")


class DistillationPredictionPolicy(
    TorchPolicy, LearningRateSchedule, EntropyCoeffSchedule
):
    # Cross-entropy during the first epoch of SGD. This is a more reliable metric
    # for prediction performance because the model has not yet been able to train on
    # the data.
    _initial_cross_entropy: List[float]

    def __init__(self, observation_space, action_space, config):
        config = Trainer.merge_trainer_configs(
            {**DEFAULT_CONFIG, "worker_index": None}, config
        )

        TorchPolicy.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )

        EntropyCoeffSchedule.__init__(
            self, config["entropy_coeff"], config["entropy_coeff_schedule"]
        )
        LearningRateSchedule.__init__(self, config["lr"], config["lr_schedule"])

        self._initial_cross_entropy = []

        self._initialize_loss_from_dummy_batch()

    def loss(
        self,
        model: TorchModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        # Replace initial state from teacher model with initial state from student model.
        for state_index, initial_state in enumerate(model.get_initial_state()):
            if isinstance(initial_state, np.ndarray):
                initial_state_tensor = torch.from_numpy(initial_state)
            else:
                initial_state_tensor = initial_state
            train_batch[f"state_in_{state_index}"] = initial_state_tensor[None].repeat(
                (len(train_batch[SampleBatch.SEQ_LENS]), 1)
            )

        # If the model is a transformer, we need to add additional state to the batch.
        if isinstance(model, AttentionWrapper):
            for data_col, view_req in self.view_requirements.items():
                if data_col.startswith("state_in_"):
                    train_batch[data_col] = np.zeros(
                        (
                            len(train_batch["seq_lens"]),
                            view_req.shift_to - view_req.shift_from + 1,
                        )
                        + view_req.space.shape
                    )

        # TODO: is this still an issue?
        # train_batch.dont_check_lens = True
        logits, state = model(train_batch)
        action_dist = dist_class(logits, model)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            batch_size = len(train_batch["seq_lens"])
            max_seq_len = logits.shape[0] // batch_size
            mask = sequence_mask(
                train_batch["seq_lens"], max_seq_len, time_major=model.is_time_major()
            )
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        # Compute cross entropy loss.
        cross_entropy = -action_dist.logp(train_batch[SampleBatch.ACTIONS])
        mean_cross_entropy = reduce_mean_valid(cross_entropy)

        curr_entropy = action_dist.entropy()
        mean_entropy = reduce_mean_valid(curr_entropy)

        total_loss = reduce_mean_valid(
            cross_entropy - self.entropy_coeff * curr_entropy
        )

        # Store values for stats function in model (tower), such that for
        # multi-GPU, we do not override them during the parallel loss phase.
        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["mean_cross_entropy"] = mean_cross_entropy
        model.tower_stats["mean_entropy"] = mean_entropy

        batches_per_epoch: int = (
            self.config["train_batch_size"] // self.config["sgd_minibatch_size"]
        )
        if len(self._initial_cross_entropy) < batches_per_epoch:
            self._initial_cross_entropy.append(mean_cross_entropy.item())

        return total_loss

    def extra_grad_process(self, local_optimizer, loss):
        return apply_grad_clipping(self, local_optimizer, loss)

    def extra_grad_info(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        return cast(
            Dict[str, TensorType],
            convert_to_numpy(
                {
                    "cur_lr": self.cur_lr,
                    "total_loss": torch.mean(
                        torch.stack(self.get_tower_stats("total_loss"))
                    ),
                    "cross_entropy": torch.mean(
                        torch.stack(self.get_tower_stats("mean_cross_entropy"))
                    ),
                    "entropy": torch.mean(
                        torch.stack(self.get_tower_stats("mean_entropy"))
                    ),
                    "entropy_coeff": self.entropy_coeff,
                }
            ),
        )

    def on_global_var_update(self, global_vars):
        super().on_global_var_update(global_vars)
        if self._lr_schedule:
            self.cur_lr = self._lr_schedule.value(global_vars["timestep"])
            for opt in self._optimizers:
                for p in opt.param_groups:
                    p["lr"] = self.cur_lr
        if self._entropy_coeff_schedule is not None:
            self.entropy_coeff = self._entropy_coeff_schedule.value(
                global_vars["timestep"]
            )


class MapExperiences:
    """
    Callable used to transfer experiences from policies acting in the environment
    to the policies that are being distilled onto.
    """

    def __init__(
        self,
        distillation_mapping_fn: Callable[[PolicyID], PolicyID],
        policies_recurrent: Dict[PolicyID, bool],
    ):
        self.distillation_mapping_fn = distillation_mapping_fn
        self.policies_recurrent = policies_recurrent

    def __call__(self, samples: SampleBatchType) -> SampleBatchType:
        assert isinstance(samples, MultiAgentBatch)

        policy_batches: Dict[PolicyID, SampleBatch] = {}
        batch: SampleBatch
        for policy_id, batch in samples.policy_batches.items():
            distill_policy_id = self.distillation_mapping_fn(policy_id)
            distill_batch = batch.copy(shallow=True)
            if not self.policies_recurrent[distill_policy_id]:
                # Remove state from batch for non-recurrent policies.
                for key in list(distill_batch.keys()):
                    if key.startswith("state_in_") or key.startswith("state_out_"):
                        del distill_batch[key]
            policy_batches[distill_policy_id] = distill_batch

        samples = MultiAgentBatch(policy_batches, samples.count)
        return samples


class PredictionMetrics:
    """
    Extra logging for the distillation-prediction trainer.
    """

    def __init__(self, config: TrainerConfigDict, workers: WorkerSet):
        self.config = config
        self.workers = workers

    def __call__(self, result):
        policy_ids = self.workers.trainable_policies()
        for policy_id in policy_ids:
            policy: DistillationPredictionPolicy = (
                self.workers.local_worker().get_policy(policy_id)
            )

            result[f"info/learner/{policy_id}/initial_cross_entropy"] = np.mean(
                policy._initial_cross_entropy
            )
            policy._initial_cross_entropy = []

        return result


def execution_plan(
    workers: WorkerSet, config: TrainerConfigDict
) -> LocalIterator[dict]:
    rollouts = cast(
        LocalIterator[SampleBatchType], ParallelRollouts(workers, mode="bulk_sync")
    )

    # Map batches to the policies getting distilled onto.
    policies_recurrent = dict(
        workers.foreach_policy(
            lambda policy, policy_id: (policy_id, policy.is_recurrent())
        )
    )
    rollouts = rollouts.for_each(
        MapExperiences(
            config["multiagent"]["distillation_mapping_fn"],
            policies_recurrent,
        )
    )

    # Collect batches for the trainable policies.
    rollouts = rollouts.for_each(SelectExperiences(workers.trainable_policies()))
    # Concatenate the SampleBatches into one.
    rollouts = rollouts.combine(
        ConcatBatches(
            min_batch_size=config["train_batch_size"],
            count_steps_by=config["multiagent"]["count_steps_by"],
        )
    )

    train_op = rollouts.for_each(
        TrainOneStep(
            workers,
            num_sgd_iter=config["num_sgd_iter"],
            sgd_minibatch_size=config["sgd_minibatch_size"],
        )
    )

    # Return training metrics.
    results = StandardMetricsReporting(train_op, workers, config)
    return results.for_each(PredictionMetrics(config, workers))


def after_init(trainer):
    # Add view requirements for student policies that might not already be required
    # by the teacher policies.
    workers: WorkerSet = trainer.workers
    all_view_requirements = dict(
        workers.foreach_policy(
            lambda policy, policy_id: (policy_id, policy.view_requirements)
        )
    )
    distillation_mapping_fn: Callable[[PolicyID], PolicyID] = trainer.config[
        "multiagent"
    ]["distillation_mapping_fn"]
    distillation_view_requirements = {
        policy_id: all_view_requirements[distillation_mapping_fn(policy_id)]
        for policy_id in all_view_requirements.keys()
        if distillation_mapping_fn(policy_id) in all_view_requirements
    }

    def add_trajectory_views(policy: Policy, policy_id: PolicyID):
        if policy_id in distillation_view_requirements:
            for data_col, view_requirement in distillation_view_requirements[
                policy_id
            ].items():
                # Ignore state since it may be different for teacher and student
                # policies.
                if (
                    data_col not in policy.view_requirements
                    and not data_col.startswith("state_")
                ):
                    policy.view_requirements[data_col] = view_requirement

    cast(WorkerSet, trainer.workers).foreach_policy(add_trajectory_views)

    if trainer.config["checkpoint_to_load_policies"] is not None:
        load_policies_from_checkpoint(
            trainer.config["checkpoint_to_load_policies"], trainer
        )


DistillationPredictionTrainer = build_trainer(
    name="DistillationPredictionTrainer",
    default_config=DEFAULT_CONFIG,
    validate_config=validate_config,
    default_policy=DistillationPredictionPolicy,
    get_policy_class=lambda config: DistillationPredictionPolicy,
    after_init=after_init,
    execution_plan=execution_plan,
)

register_trainable("distillation_prediction", DistillationPredictionTrainer)


ASYNC_DEFAULT_CONFIG = {
    **DEFAULT_CONFIG,
    # how many train batches should be retained for minibatching. This conf
    # only has an effect if `num_sgd_iter > 1`.
    "minibatch_buffer_size": 1,
    # max queue size for train batches feeding into the learner
    "learner_queue_size": 40,
    # wait for train batches to be available in minibatch buffer queue
    # this many seconds. This may need to be increased e.g. when training
    # with a slow environment
    "learner_queue_timeout": 300,
    # max number of workers to broadcast one set of weights to
    "broadcast_interval": 1,
}


class SplitMinibatches:
    """
    Splits larger batches into minibatches.
    """

    def __init__(self, sgd_minibatch_size: int):
        self.sgd_minibatch_size = sgd_minibatch_size

    def __call__(self, samples: SampleBatchType) -> List[SampleBatchType]:
        assert isinstance(samples, MultiAgentBatch)

        all_minibatches: List[SampleBatchType] = []
        for policy_id, policy_batch in samples.policy_batches.items():
            for minibatch in minibatches(policy_batch, self.sgd_minibatch_size):
                all_minibatches.append(
                    MultiAgentBatch(
                        {policy_id: minibatch},
                        minibatch.count,
                    )
                )

        return all_minibatches


def record_steps_trained(item):
    count, fetches = item
    metrics = _get_shared_metrics()
    # Manually update the steps trained counter since the learner thread
    # is executing outside the pipeline.
    metrics.counters[STEPS_TRAINED_COUNTER] += count
    metrics.counters[AGENT_STEPS_TRAINED_COUNTER] += count
    return item


def use_steps_trained_for_timesteps_total(result):
    metrics = _get_shared_metrics()
    result["timesteps_total"] = metrics.counters[STEPS_TRAINED_COUNTER]
    return result


def async_execution_plan(workers, config):
    rollouts = cast(
        LocalIterator[SampleBatchType],
        ParallelRollouts(workers, mode="async", num_async=2),
    )

    # Map batches to the policies getting distilled onto.
    policies_recurrent = dict(
        workers.foreach_policy(
            lambda policy, policy_id: (policy_id, policy.is_recurrent())
        )
    )
    rollouts = rollouts.for_each(
        MapExperiences(
            config["multiagent"]["distillation_mapping_fn"],
            policies_recurrent,
        )
    )

    # Collect batches for the trainable policies.
    rollouts = rollouts.for_each(SelectExperiences(workers.trainable_policies()))
    # Concatenate the SampleBatches into one.
    train_batches = rollouts.combine(
        ConcatBatches(
            min_batch_size=config["train_batch_size"],
            count_steps_by=config["multiagent"]["count_steps_by"],
        )
    )

    # Split train batches into minibatches for SGD.
    minibatches = train_batches.for_each(
        SplitMinibatches(config["sgd_minibatch_size"])
    ).flatten()

    # Start the learner thread.
    learner_thread = LearnerThread(
        workers.local_worker(),
        minibatch_buffer_size=config["minibatch_buffer_size"],
        num_sgd_iter=config["num_sgd_iter"],
        learner_queue_size=config["learner_queue_size"],
        learner_queue_timeout=config["learner_queue_timeout"],
    )
    learner_thread.start()

    # This sub-flow sends experiences to the learner.
    enqueue_op = minibatches.for_each(Enqueue(learner_thread.inqueue))

    # This sub-flow updates the steps trained counter based on learner output.
    dequeue_op = Dequeue(
        learner_thread.outqueue, check=learner_thread.is_alive
    ).for_each(record_steps_trained)

    merged_op = Concurrently([enqueue_op, dequeue_op], mode="async", output_indexes=[1])

    return (
        StandardMetricsReporting(merged_op, workers, config)
        .for_each(use_steps_trained_for_timesteps_total)
        .for_each(learner_thread.add_learner_metrics)
        .for_each(PredictionMetrics(config, workers))
    )


AsyncDistillationPredictionTrainer = build_trainer(
    name="AsyncDistillationPredictionTrainer",
    default_config=ASYNC_DEFAULT_CONFIG,
    validate_config=validate_config,
    default_policy=DistillationPredictionPolicy,
    get_policy_class=lambda config: DistillationPredictionPolicy,
    after_init=after_init,
    execution_plan=async_execution_plan,
)

register_trainable("async_distillation_prediction", AsyncDistillationPredictionTrainer)
