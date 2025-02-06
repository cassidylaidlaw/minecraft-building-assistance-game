import logging
from collections import defaultdict
from typing import (
    Callable,
    Collection,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Type,
    cast,
)

import numpy as np
import torch
from git import TYPE_CHECKING
from gymnasium import spaces
from ray.rllib.algorithms.algorithm import Algorithm, AlgorithmConfig
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.postprocessing import Postprocessing, discount_cumsum
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import multi_gpu_train_one_step, train_one_step
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch, concat_samples
from ray.rllib.policy.torch_mixins import LearningRateSchedule
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.from_config import NotProvided
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.sgd import minibatches
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    explained_variance,
    sequence_mask,
)
from ray.rllib.utils.typing import AlgorithmConfigDict, PolicyID, ResultDict, TensorType
from ray.tune.registry import register_trainable
from torch import nn

from .human_data import PARTICIPANT_ID
from .torch_models import MbagTorchModel, OptimizerMixin

logger = logging.getLogger(__name__)


class BCTorchLossesAndStats(NamedTuple):
    bc_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    accuracy: torch.Tensor
    vf_explained_var: torch.Tensor


class BCTorchPolicy(LearningRateSchedule, OptimizerMixin, TorchPolicy):
    def __init__(self, observation_space, action_space, config):
        TorchPolicy.__init__(
            self,
            observation_space,
            action_space,
            config,
            max_seq_len=config["model"]["max_seq_len"],
        )
        LearningRateSchedule.__init__(self, config["lr"], config.get("lr_schedule"))

        self._initialize_loss_from_dummy_batch()

        # Needed for training AlphaZero assistants with BC policy.
        self.view_requirements[SampleBatch.ACTION_DIST_INPUTS] = ViewRequirement(
            space=spaces.Box(low=-np.inf, high=np.inf, shape=(action_space.n,))
        )

    def _get_losses_and_stats(
        self,
        model: TorchModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch,
    ):
        model_out, state = model(train_batch)
        values = model.value_function()
        action_dist: ActionDistribution = dist_class(model_out, model)
        actions = cast(torch.Tensor, train_batch[SampleBatch.ACTIONS])
        logprobs = cast(torch.Tensor, action_dist.logp(actions))

        mask = torch.ones(
            model_out.size()[0], dtype=torch.bool, device=model_out.device
        )
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])  # noqa: N806
            max_seq_len = model_out.shape[0] // B
            mask &= cast(
                torch.Tensor,
                sequence_mask(
                    train_batch[SampleBatch.SEQ_LENS],
                    max_seq_len,
                    time_major=model.is_time_major(),
                ),
            ).reshape(-1)
            assert isinstance(mask, torch.Tensor)
        mask &= logprobs > MbagTorchModel.MASK_LOGIT

        def reduce_mean_valid(tensor):
            return torch.mean(tensor[mask])

        bc_loss = reduce_mean_valid(-logprobs)
        accuracy = reduce_mean_valid(
            (cast(torch.Tensor, action_dist.deterministic_sample()) == actions).float()
        )

        value_loss = reduce_mean_valid(
            (values - train_batch[Postprocessing.VALUE_TARGETS]) ** 2
        )
        vf_explained_var = cast(
            torch.Tensor,
            explained_variance(train_batch[Postprocessing.VALUE_TARGETS], values),
        )

        entropy = reduce_mean_valid(action_dist.entropy())

        return BCTorchLossesAndStats(
            bc_loss=bc_loss,
            value_loss=value_loss,
            entropy=entropy,
            accuracy=accuracy,
            vf_explained_var=vf_explained_var,
        )

    def loss(
        self,
        model: ModelV2,
        dist_class: Type[TorchDistributionWrapper],
        train_batch: SampleBatch,
    ):
        assert isinstance(model, TorchModelV2)

        losses_and_stats = self._get_losses_and_stats(model, dist_class, train_batch)

        model.tower_stats["bc_loss"] = losses_and_stats.bc_loss
        model.tower_stats["accuracy"] = losses_and_stats.accuracy
        model.tower_stats["vf_loss"] = losses_and_stats.value_loss
        model.tower_stats["vf_explained_var"] = losses_and_stats.vf_explained_var
        model.tower_stats["entropy"] = losses_and_stats.entropy

        loss = (
            losses_and_stats.bc_loss
            + self.config["vf_loss_coeff"] * losses_and_stats.value_loss
            - self.config["entropy_coeff"] * losses_and_stats.entropy
        )

        return loss

    def extra_grad_process(self, optimizer, loss):
        return apply_grad_clipping(self, optimizer, loss)

    def extra_grad_info(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        assert isinstance(self.model, TorchModelV2)
        stats: Dict[str, TensorType] = {}
        for stat_key in [
            "bc_loss",
            "accuracy",
            "vf_loss",
            "vf_explained_var",
            "entropy",
        ]:
            if stat_key in self.model.tower_stats:
                stats[stat_key] = torch.mean(
                    torch.stack(
                        cast(List[torch.Tensor], self.get_tower_stats(stat_key))
                    )
                )
        stats["lr"] = self.cur_lr
        return cast(Dict[str, TensorType], convert_to_numpy(stats))

    def postprocess_trajectory(
        self,
        sample_batch: SampleBatch,
        other_agent_batches=None,
        episode: Optional[Episode] = None,
    ) -> SampleBatch:
        # This isn't actually called during BC training when trajectories are loaded
        # from disk, but it's needed to avoid errors during RLlib's
        # _initialize_loss_from_dummy_batch method.
        sample_batch = super().postprocess_trajectory(
            sample_batch, other_agent_batches, episode
        )
        sample_batch[Postprocessing.VALUE_TARGETS] = discount_cumsum(
            cast(np.ndarray, sample_batch[SampleBatch.REWARDS]),
            self.config["gamma"],
        ).astype(np.float32)
        return sample_batch

    def validation(self, validation_batch: SampleBatch) -> Dict[str, float]:
        assert isinstance(self.model, TorchModelV2)
        assert self.dist_class is not None
        assert issubclass(self.dist_class, TorchDistributionWrapper)

        validation_batch.decompress_if_needed()

        if not validation_batch.zero_padded:
            pad_batch_to_sequences_of_same_size(
                batch=validation_batch,
                max_seq_len=self.max_seq_len,
                shuffle=False,
                batch_divisibility_req=self.batch_divisibility_req,
                view_requirements=self.view_requirements,
            )

        assert isinstance(self.model, nn.Module)
        self.model.eval()

        validation_batch.set_training(False)
        self._lazy_tensor_dict(validation_batch, device=self.devices[0])
        with torch.no_grad():
            losses_and_stats = self._get_losses_and_stats(
                self.model, self.dist_class, validation_batch
            )

        return {
            "cross_entropy": losses_and_stats.bc_loss.item(),
            "accuracy": losses_and_stats.accuracy.item(),
            "vf_loss": losses_and_stats.value_loss.item(),
            "vf_explained_var": losses_and_stats.vf_explained_var.item(),
            "entropy": losses_and_stats.entropy.item(),
        }


class BCConfig(AlgorithmConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class)

        self.lr_schedule = None
        self.sgd_minibatch_size = 128
        self.num_sgd_iter = 30
        self.validation_participant_ids: Collection[int] = []
        self.validation_prop: float = 0.0
        self.entropy_coeff: float = 0.0
        self.vf_loss_coeff: float = 0.0
        self.data_augmentation: Callable[[SampleBatch], SampleBatch] = lambda x: x

        self.exploration_config = {
            "type": "StochasticSampling",
        }

    def training(
        self,
        *args,
        lr_schedule=NotProvided,
        sgd_minibatch_size=NotProvided,
        num_sgd_iter=NotProvided,
        entropy_coeff=NotProvided,
        vf_loss_coeff=NotProvided,
        validation_participant_ids=NotProvided,
        validation_prop=NotProvided,
        data_augmentation=NotProvided,
        **kwargs,
    ):
        """
        Set training parameters.
        Args:
            validation_prop (float): Proportion of data to use for validation.
            entropy_coeff (float): Coefficient for entropy regularization.
        """

        super().training(*args, **kwargs)

        if lr_schedule is not NotProvided:
            self.lr_schedule = lr_schedule
        if sgd_minibatch_size is not NotProvided:
            self.sgd_minibatch_size = sgd_minibatch_size
        if num_sgd_iter is not NotProvided:
            self.num_sgd_iter = num_sgd_iter
        if entropy_coeff is not NotProvided:
            self.entropy_coeff = entropy_coeff
        if vf_loss_coeff is not NotProvided:
            self.vf_loss_coeff = vf_loss_coeff
        if validation_participant_ids is not NotProvided:
            self.validation_participant_ids = validation_participant_ids
        if validation_prop is not NotProvided:
            self.validation_prop = validation_prop
        if data_augmentation is not NotProvided:
            self.data_augmentation = data_augmentation


class BC(Algorithm):
    config: BCConfig  # type: ignore[assignment]

    @classmethod
    def get_default_config(cls):
        return BCConfig()

    @classmethod
    def get_default_policy_class(cls, config: AlgorithmConfigDict) -> Type[Policy]:
        if config["framework"] == "torch":
            return BCTorchPolicy
        else:
            raise NotImplementedError()

    def _split_training_and_validation_data(
        self,
        train_batch: MultiAgentBatch,
    ) -> Tuple[MultiAgentBatch, Optional[MultiAgentBatch]]:
        if (
            self.config["validation_participant_ids"]
            or self.config.get("validation_prop", 0) > 0
        ):
            assert not (
                self.config["validation_participant_ids"]
                and self.config.get("validation_prop", 0) > 0
            ), "Cannot set both validation_participant_ids and validation_prop."
            train_policy_batches: Dict[PolicyID, SampleBatch] = {}
            validation_policy_batches: Dict[PolicyID, SampleBatch] = {}
            for policy_id, policy_batch in train_batch.policy_batches.items():
                train_episodes: List[SampleBatch] = []
                validation_episodes: List[SampleBatch] = []
                for episode_batch in policy_batch.split_by_episode():
                    if self.config["validation_participant_ids"]:
                        assert PARTICIPANT_ID in episode_batch
                        participant_id = episode_batch[PARTICIPANT_ID][0]
                        assert np.all(episode_batch[PARTICIPANT_ID] == participant_id)
                        if participant_id in self.config["validation_participant_ids"]:
                            validation_episodes.append(episode_batch)
                        else:
                            train_episodes.append(episode_batch)
                    else:
                        if (
                            np.random.default_rng(
                                seed=episode_batch[SampleBatch.EPS_ID][0]
                            ).random()
                            < self.config["validation_prop"]
                        ):
                            validation_episodes.append(episode_batch)
                        else:
                            train_episodes.append(episode_batch)
                train_policy_batch = concat_samples(train_episodes)
                train_policy_batches[policy_id] = train_policy_batch
                validation_policy_batch = concat_samples(validation_episodes)
                validation_policy_batches[policy_id] = validation_policy_batch
            # Approximate number of env steps in each batch.
            train_prop = train_policy_batch.count / (
                train_policy_batch.count + validation_policy_batch.count
            )
            train_env_steps = int(train_batch.count * train_prop)
            validation_env_steps = int(train_batch.count * (1 - train_prop))
            return (
                MultiAgentBatch(train_policy_batches, train_env_steps),
                MultiAgentBatch(validation_policy_batches, validation_env_steps),
            )
        else:
            return train_batch, None

    def _run_validation(self, validation_batch: MultiAgentBatch) -> dict:
        assert self.workers is not None
        validation_results: Dict[str, dict] = {}
        for policy_id, policy_batch in validation_batch.policy_batches.items():
            policy = self.workers.local_worker().get_policy(policy_id)
            validation_totals: Dict[str, float] = defaultdict(float)
            validation_counts: Dict[str, int] = defaultdict(int)
            if TYPE_CHECKING:
                assert isinstance(policy, BCTorchPolicy)
            for minibatch in minibatches(
                policy_batch, self.config["sgd_minibatch_size"]
            ):
                results = policy.validation(minibatch)
                for key, value in results.items():
                    if np.isnan(value):
                        continue
                    validation_totals[key] += value * minibatch.count
                    validation_counts[key] += minibatch.count
            validation_results[policy_id] = {
                "validation": {
                    key: value / validation_counts[key]
                    for key, value in validation_totals.items()
                }
            }
        return validation_results

    def training_step(self) -> ResultDict:
        assert self.workers is not None

        # Collect SampleBatches from sample workers until we have a full batch.
        if self.config["count_steps_by"] == "agent_steps":
            train_batch = synchronous_parallel_sample(
                worker_set=self.workers,
                max_agent_steps=self.config["train_batch_size"],
            )
        else:
            train_batch = synchronous_parallel_sample(
                worker_set=self.workers, max_env_steps=self.config["train_batch_size"]
            )
        if isinstance(train_batch, list):
            train_batch = concat_samples(train_batch)
        if not isinstance(train_batch, MultiAgentBatch):
            policy_map = self.workers.local_worker().policy_map
            assert policy_map is not None
            policy_id = next(iter(policy_map))
            train_batch = MultiAgentBatch({policy_id: train_batch}, train_batch.count)
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        for policy_id, policy_batch in train_batch.policy_batches.items():
            if "infos" in policy_batch:
                del policy_batch["infos"]

            # Add VF targets.
            episode_vf_targets: List[np.ndarray] = []
            for episode_batch in policy_batch.split_by_episode():
                episode_vf_targets.append(
                    discount_cumsum(
                        cast(np.ndarray, episode_batch[SampleBatch.REWARDS]),
                        self.config["gamma"],
                    ).astype(np.float32),
                )
            policy_batch[Postprocessing.VALUE_TARGETS] = np.concatenate(
                episode_vf_targets
            )

            # If the policy is recurrent, add initial state.
            policy = self.get_policy(policy_id)
            for state_index, initial_state in enumerate(policy.get_initial_state()):
                assert SampleBatch.SEQ_LENS in policy_batch
                num_seqs = len(policy_batch[SampleBatch.SEQ_LENS])
                policy_batch[f"state_in_{state_index}"] = np.repeat(
                    initial_state[None], num_seqs, axis=0
                )

            # If the policy isn't recurrent, remove seq_lens.
            if not policy.get_initial_state() and SampleBatch.SEQ_LENS in policy_batch:
                del policy_batch[SampleBatch.SEQ_LENS]

        train_batch, val_batch = self._split_training_and_validation_data(train_batch)

        with self._timers["data_augmentation"]:
            for policy_id in train_batch.policy_batches:
                train_batch.policy_batches[policy_id] = self.config.data_augmentation(
                    train_batch.policy_batches[policy_id]
                )

        # Train
        train_results: ResultDict
        if self.config["simple_optimizer"]:
            train_results = train_one_step(self, train_batch)
        else:
            train_results = multi_gpu_train_one_step(self, train_batch)

        # Validation
        if val_batch is not None:
            validation_results = self._run_validation(val_batch)
            for policy_id, policy_results in validation_results.items():
                train_results[policy_id].update(policy_results)

        policies_to_update = list(train_results.keys())

        policy_map = self.workers.local_worker().policy_map
        assert policy_map is not None
        global_vars = {
            "timestep": self._counters[NUM_AGENT_STEPS_SAMPLED],
            "num_grad_updates_per_policy": {
                pid: policy_map[pid].num_grad_updates for pid in policies_to_update
            },
        }

        # Update weights - after learning on the local worker - on all remote
        # workers.
        if self.workers.num_remote_workers() > 0:
            with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
                from_worker = None
                self.workers.sync_weights(
                    from_worker_or_learner_group=from_worker,
                    policies=list(train_results.keys()),
                    global_vars=global_vars,
                )

        # Update global vars on local worker as well.
        self.workers.local_worker().set_global_vars(global_vars)

        return train_results


register_trainable("BC", BC)
