"""
Distillation-prediction, where one policy is distilled into a another model
for predicting the next action given past actions.
"""

import logging
import numpy as np
from ray.rllib.models.torch.attention_net import AttentionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.algorithms import Algorithm
from ray.tune.registry import register_trainable
import torch
from typing import Collection, Dict, List, Type, Union, Callable, cast, Any

from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.utils.typing import (
    PolicyID,
    TrainerConfigDict,
    TensorType,
    ResultDict,
)
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.torch_mixins import (
    LearningRateSchedule,
    EntropyCoeffSchedule,
)
from ray.rllib.policy.torch_policy_v2 import (
    TorchPolicyV2,
)
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    sequence_mask,
)
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.execution.train_ops import (
    train_one_step,
    multi_gpu_train_one_step,
)
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)

from .training_utils import load_policies_from_checkpoint

logger = logging.getLogger(__name__)


class DistillationPredictionPolicy(
    TorchPolicyV2, LearningRateSchedule, EntropyCoeffSchedule
):
    # Cross-entropy during the first epoch of SGD. This is a more reliable metric
    # for prediction performance because the model has not yet been able to train on
    # the data.
    _initial_cross_entropy: List[float]

    def __init__(self, observation_space, action_space, config):
        TorchPolicyV2.__init__(
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
                            len(train_batch[SampleBatch.SEQ_LENS]),
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
            batch_size = len(train_batch[SampleBatch.SEQ_LENS])
            max_seq_len = logits.shape[0] // batch_size
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
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
        return apply_grad_clipping(cast(Any, self), local_optimizer, loss)

    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
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


class DistillationPrediction(Algorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Add view requirements for student policies that might not already be required
        # by the teacher policies.
        assert self.workers is not None
        all_view_requirements = dict(
            self.workers.foreach_policy(
                lambda policy, policy_id: (policy_id, policy.view_requirements)
            )
        )
        self.distillation_mapping_fn: Callable[[PolicyID], PolicyID] = self.config[
            "distillation_mapping_fn"
        ]
        distillation_view_requirements = {
            policy_id: all_view_requirements[self.distillation_mapping_fn(policy_id)]
            for policy_id in all_view_requirements.keys()
            if self.distillation_mapping_fn(policy_id) in all_view_requirements
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

        self.workers.foreach_policy(add_trajectory_views)

        if self.config["checkpoint_to_load_policies"] is not None:
            load_policies_from_checkpoint(
                self.config["checkpoint_to_load_policies"], self
            )

    @classmethod
    def get_default_config(cls) -> TrainerConfigDict:
        return {
            **super().get_default_config(),
            # Function which maps the ID of a policy acting in the environment to the
            # policy ID which should be trained to mimic it.
            "distillation_mapping_fn": None,
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

    def validate_config(self, config: TrainerConfigDict) -> None:
        super().validate_config(config)

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

    def get_default_policy_class(self, config: TrainerConfigDict) -> Type[Policy]:
        return DistillationPredictionPolicy

    def map_experiences(self, train_batch: MultiAgentBatch) -> MultiAgentBatch:
        """
        Transfer experiences from policies acting in the environment to the policies
        that are being distilled onto.
        """

        policy_batches: Dict[PolicyID, SampleBatch] = {}
        for policy_id, batch in train_batch.policy_batches.items():
            is_policy_recurrent = self.get_policy(policy_id).is_recurrent()
            distill_policy_id = self.distillation_mapping_fn(policy_id)
            distill_batch = batch.copy(shallow=True)
            if not is_policy_recurrent:
                # Remove state from batch for non-recurrent policies.
                for key in list(distill_batch.keys()):
                    if key.startswith("state_in_") or key.startswith("state_out_"):
                        del distill_batch[key]
            policy_batches[distill_policy_id] = distill_batch

        return MultiAgentBatch(policy_batches, train_batch.count)

    def training_step(self) -> ResultDict:
        assert self.workers is not None

        train_batch = synchronous_parallel_sample(
            worker_set=self.workers, max_env_steps=self.config["train_batch_size"]
        )
        assert not isinstance(train_batch, list)

        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        # Map batches to the policies getting distilled onto.
        train_batch = self.map_experiences(train_batch)

        # Standardize advantages
        train_results: ResultDict
        # Train
        if self.config["simple_optimizer"]:
            train_results = train_one_step(self, train_batch)
        else:
            train_results = multi_gpu_train_one_step(self, train_batch)

        global_vars = {
            "timestep": self._counters[NUM_AGENT_STEPS_SAMPLED],
        }
        # Update weights - after learning on the local worker - on all remote
        # workers.
        if self.workers.remote_workers():
            with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                self.workers.sync_weights(global_vars=global_vars)

        # Update global vars on local worker as well.
        self.workers.local_worker().set_global_vars(global_vars)

        # Extra metrics for distillation.
        policy_ids: Collection[
            PolicyID
        ] = self.workers.local_worker().get_policies_to_train()
        for policy_id in policy_ids:
            policy = self.workers.local_worker().get_policy(policy_id)
            assert isinstance(policy, DistillationPredictionPolicy)

            train_results[f"info/learner/{policy_id}/initial_cross_entropy"] = np.mean(
                policy._initial_cross_entropy
            )
            policy._initial_cross_entropy = []

        return train_results


register_trainable("distillation_prediction", DistillationPrediction)
