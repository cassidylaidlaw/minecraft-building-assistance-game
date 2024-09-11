import contextlib
import logging
from typing import (
    Callable,
    Collection,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    cast,
)

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from ray.rllib.algorithms.ppo.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.execution.rollout_ops import (
    standardize_fields,
    synchronous_parallel_sample,
)
from ray.rllib.execution.train_ops import multi_gpu_train_one_step, train_one_step
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.offline import JsonReader
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.rnn_sequencing import pad_batch_to_sequences_of_same_size
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch, concat_samples
from ray.rllib.utils.compression import is_compressed
from ray.rllib.utils.from_config import NotProvided
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SAMPLE_TIMER,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.sgd import minibatches
from ray.rllib.utils.typing import PolicyID, ResultDict, TensorType
from ray.tune.registry import register_trainable
from ray.util.debug import log_once
from torch import nn

from .bc import BC
from .torch_models import MbagTorchModel, ModelWithDiscriminator, OptimizerMixinV2

logger = logging.getLogger(__name__)


IS_DEMONSTRATION = "is_demonstration"


class MbagGAILTorchPolicy(OptimizerMixinV2, PPOTorchPolicy):
    def __init__(self, observation_space, action_space, config):
        PPOTorchPolicy.__init__(
            self,
            observation_space,
            action_space,
            config,
        )

    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ):
        assert isinstance(model, TorchModelV2)

        if IS_DEMONSTRATION in train_batch:
            # We're just training the discriminator in this case.

            assert isinstance(model, ModelWithDiscriminator)
            is_demonstration = cast(torch.Tensor, train_batch[IS_DEMONSTRATION])
            discriminator_scores = model.discriminator(train_batch)
            discriminator_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                discriminator_scores,
                is_demonstration.float(),
            )

            if "mean_policy_loss" in model.tower_stats:
                del model.tower_stats["mean_policy_loss"]
            model.tower_stats["discriminator/loss"] = discriminator_loss
            model.tower_stats["discriminator/policy_scores"] = discriminator_scores[
                ~is_demonstration
            ].mean()
            model.tower_stats["discriminator/demonstration_scores"] = (
                discriminator_scores[is_demonstration].mean()
            )

            return discriminator_loss
        else:
            # We're just training the policy in this case.
            return super().loss(model, dist_class, train_batch)

    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        assert isinstance(self.model, TorchModelV2)
        if "mean_policy_loss" in self.model.tower_stats:
            return super().stats_fn(train_batch)
        else:
            stats: Dict[str, TensorType] = {}
            for stat_key in [
                "discriminator/loss",
                "discriminator/policy_scores",
                "discriminator/demonstration_scores",
            ]:
                if stat_key in self.model.tower_stats:
                    stats[stat_key] = torch.mean(
                        torch.stack(
                            cast(List[torch.Tensor], self.get_tower_stats(stat_key))
                        )
                    )
            return cast(Dict[str, TensorType], convert_to_numpy(stats))

    def get_discriminator_scores(self, batch: SampleBatch) -> np.ndarray:
        assert isinstance(self.model, ModelWithDiscriminator)

        batch.decompress_if_needed()

        self.model.eval()
        batch.set_training(False)
        self._lazy_tensor_dict(batch, device=self.devices[0])
        with torch.no_grad():
            discriminator_scores: np.ndarray = (
                self.model.discriminator(batch).cpu().numpy()
            )

        return discriminator_scores

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
            model_out, state = self.model(validation_batch)
            action_dist: ActionDistribution = self.dist_class(model_out, self.model)
            actions = validation_batch[SampleBatch.ACTIONS]
            logprobs = cast(torch.Tensor, action_dist.logp(actions))
            mask = logprobs > MbagTorchModel.MASK_LOGIT
            cross_entropy = -torch.mean(logprobs[mask]).item()

        return {
            "cross_entropy": cross_entropy,
        }


class MbagGAILConfig(PPOConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class)

        self.demonstration_input: str = ""
        self.demonstration_batch_size = 0
        self.validation_participant_ids: Collection[int] = []
        self.data_augmentation: Callable[[SampleBatch], SampleBatch] = lambda x: x
        self.train_discriminator_on_separate_batch = False
        self.discriminator_num_sgd_iter: Optional[int] = None

    def training(
        self,
        *args,
        demonstration_input=NotProvided,
        demonstration_batch_size=NotProvided,
        validation_participant_ids=NotProvided,
        data_augmentation=NotProvided,
        train_discriminator_on_separate_batch=NotProvided,
        discriminator_num_sgd_iter=NotProvided,
        **kwargs,
    ):
        """
        Set training parameters.
        """

        super().training(*args, **kwargs)

        if demonstration_input is not NotProvided:
            self.demonstration_input = demonstration_input
        if demonstration_batch_size is not NotProvided:
            self.demonstration_batch_size = demonstration_batch_size
        if validation_participant_ids is not NotProvided:
            self.validation_participant_ids = validation_participant_ids
        if data_augmentation is not NotProvided:
            self.data_augmentation = data_augmentation
        if train_discriminator_on_separate_batch is not NotProvided:
            self.train_discriminator_on_separate_batch = (
                train_discriminator_on_separate_batch
            )
        if discriminator_num_sgd_iter is not NotProvided:
            self.discriminator_num_sgd_iter = discriminator_num_sgd_iter

    @contextlib.contextmanager
    def _override_num_sgd_iter(self, num_sgd_iter):
        is_frozen = self._is_frozen
        prev_num_sgd_iter = self.num_sgd_iter

        self._is_frozen = False
        self.num_sgd_iter = num_sgd_iter
        self._is_frozen = is_frozen
        try:
            yield
        finally:
            self._is_frozen = False
            self.num_sgd_iter = prev_num_sgd_iter
            self._is_frozen = is_frozen


class MbagGAIL(PPO, BC):
    config: MbagGAILConfig  # type: ignore[assignment]

    def __init__(
        self,
        config=None,
        env=None,
        logger_creator=None,
        **kwargs,
    ):
        super().__init__(config, env, logger_creator, **kwargs)

        demonstration_reader = JsonReader(self.config.demonstration_input)

        single_agent_demonstration_batch = concat_samples(
            list(cast(Iterable[SampleBatch], demonstration_reader.read_all_files()))
        )
        assert isinstance(single_agent_demonstration_batch, SampleBatch)
        del single_agent_demonstration_batch[SampleBatch.INFOS]
        (policy_id,) = self.config.policies
        self.demonstration_batch = MultiAgentBatch(
            {policy_id: single_agent_demonstration_batch},
            single_agent_demonstration_batch.count,
        )
        self.demonstration_train_batch, self.demonstration_val_batch = (
            self._split_training_and_validation_data(self.demonstration_batch)
        )

    @classmethod
    def get_default_config(cls):
        return MbagGAILConfig()

    @classmethod
    def get_default_policy_class(cls, config) -> Type[Policy]:
        if config["framework"] == "torch":
            return MbagGAILTorchPolicy
        else:
            raise NotImplementedError()

    def _split_batch_in_half(
        self, batch: MultiAgentBatch
    ) -> Tuple[MultiAgentBatch, MultiAgentBatch]:
        policy_batches_1: Dict[PolicyID, SampleBatch] = {}
        policy_batches_2: Dict[PolicyID, SampleBatch] = {}
        for policy_id, policy_batch in batch.policy_batches.items():
            # Shuffle episodes.
            episode_batches = policy_batch.split_by_episode()
            np.random.shuffle(episode_batches)  # type: ignore
            policy_batch = concat_samples(episode_batches)
            batch_length = len(policy_batch)
            policy_batches_1[policy_id] = policy_batch.slice(0, batch_length // 2)
            policy_batches_2[policy_id] = policy_batch.slice(
                batch_length // 2, batch_length
            )
        return MultiAgentBatch(
            policy_batches_1,
            batch.count // 2,
        ), MultiAgentBatch(
            policy_batches_2,
            batch.count - batch.count // 2,
        )

    def _get_discriminator_train_batch(
        self, train_batch: MultiAgentBatch
    ) -> Tuple[MultiAgentBatch, MultiAgentBatch]:
        if self.config.train_discriminator_on_separate_batch:
            discriminator_policy_batch, train_batch = self._split_batch_in_half(
                train_batch
            )
        else:
            discriminator_policy_batch = train_batch.copy()

        with self._timers["data_augmentation"]:
            augmented_demonstration_train_policy_batches: Dict[
                PolicyID, SampleBatch
            ] = {}
            for (
                policy_id,
                policy_batch,
            ) in self.demonstration_train_batch.policy_batches.items():
                total_timesteps = 0
                augmented_batches: List[SampleBatch] = []
                target_timesteps = len(
                    discriminator_policy_batch.policy_batches[policy_id]
                )

                # Concatenate enough demonstration data to match the length of the
                # PPO training batch. In order to uniformly sample from the
                # demonstration data, we randomly slice the concatenated batch.
                while total_timesteps < target_timesteps + len(policy_batch):
                    augmented_batch = self.config.data_augmentation(policy_batch)
                    if self.config.compress_observations and not is_compressed(
                        augmented_batch[SampleBatch.OBS][0]
                    ):
                        augmented_batch = augmented_batch.compress()
                    augmented_batches.append(augmented_batch)
                    total_timesteps += len(augmented_batch)
                augmented_batch = concat_samples(augmented_batches)

                slice_start = np.random.randint(0, len(policy_batch))
                augmented_demonstration_train_policy_batches[policy_id] = (
                    augmented_batch.slice(slice_start, slice_start + target_timesteps)
                )

        for (
            policy_id,
            policy_batch,
        ) in discriminator_policy_batch.policy_batches.items():
            for key in list(policy_batch):
                if key not in augmented_demonstration_train_policy_batches[policy_id]:
                    del policy_batch[key]
            policy_batch[IS_DEMONSTRATION] = np.zeros(len(policy_batch), dtype=bool)
        discriminator_demonstration_batch = MultiAgentBatch(
            augmented_demonstration_train_policy_batches,
            discriminator_policy_batch.env_steps(),
        )
        for (
            policy_id,
            policy_batch,
        ) in discriminator_demonstration_batch.policy_batches.items():
            policy_batch[IS_DEMONSTRATION] = np.ones(len(policy_batch), dtype=bool)
            for key in list(policy_batch):
                if key not in discriminator_policy_batch.policy_batches[policy_id]:
                    del policy_batch[key]

        return (
            concat_samples(
                [discriminator_policy_batch, discriminator_demonstration_batch]
            ),
            train_batch,
        )

    def _replace_rewards_with_discriminator_scores(self, train_batch: MultiAgentBatch):
        assert self.workers is not None
        for policy_id in train_batch.policy_batches:
            policy_batch = train_batch.policy_batches[policy_id]
            policy = self.workers.local_worker().get_policy(policy_id)
            assert isinstance(policy, MbagGAILTorchPolicy)

            discriminator_score_batches: List[np.ndarray] = []

            for minibatch in minibatches(
                policy_batch, self.config["sgd_minibatch_size"], shuffle=False
            ):
                discriminator_score_batches.append(
                    policy.get_discriminator_scores(minibatch)
                )
            discriminator_scores = np.concatenate(discriminator_score_batches)
            rewards = -F.logsigmoid(torch.from_numpy(-discriminator_scores)).numpy()
            policy_batch[SampleBatch.REWARDS] = rewards

            # Need to recompute advantages after changing rewards.
            episode_batches: List[SampleBatch] = []
            for episode_batch in policy_batch.split_by_episode():
                episode_batch = compute_advantages(
                    rollout=episode_batch,
                    last_r=float(episode_batch[SampleBatch.VALUES_BOOTSTRAPPED][-1]),
                    gamma=policy.config["gamma"],
                    lambda_=policy.config["lambda"],
                    use_gae=policy.config["use_gae"],
                    use_critic=policy.config.get("use_critic", True),
                    vf_preds=episode_batch[SampleBatch.VF_PREDS],
                    rewards=episode_batch[SampleBatch.REWARDS],
                )
                episode_batches.append(episode_batch)
            train_batch.policy_batches[policy_id] = concat_samples(episode_batches)

    def training_step(self) -> ResultDict:  # noqa: C901
        assert self.workers is not None

        # Collect SampleBatches from sample workers until we have a full batch.
        with self._timers[SAMPLE_TIMER]:
            if self.config.count_steps_by == "agent_steps":
                train_batch = synchronous_parallel_sample(
                    worker_set=self.workers,
                    max_agent_steps=self.config.train_batch_size,
                )
            else:
                train_batch = synchronous_parallel_sample(
                    worker_set=self.workers, max_env_steps=self.config.train_batch_size
                )

        if isinstance(train_batch, list):
            train_batch = concat_samples(train_batch)
        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        if self.config.discriminator_num_sgd_iter is not None:
            discriminator_num_sgd_iter = self.config.discriminator_num_sgd_iter
        else:
            discriminator_num_sgd_iter = self.config.num_sgd_iter

        # Train discriminator
        discriminator_train_batch, train_batch = self._get_discriminator_train_batch(
            train_batch
        )
        with self.config._override_num_sgd_iter(discriminator_num_sgd_iter):
            if self.config.simple_optimizer:
                discriminator_train_results = train_one_step(
                    self, discriminator_train_batch
                )
            else:
                discriminator_train_results = multi_gpu_train_one_step(
                    self, discriminator_train_batch
                )

        with self._timers["replace_rewards"]:
            self._replace_rewards_with_discriminator_scores(train_batch)

        # Standardize advantages
        train_batch = standardize_fields(train_batch, ["advantages"])

        # Train policy
        if self.config.simple_optimizer:
            train_results = train_one_step(self, train_batch)
        else:
            train_results = multi_gpu_train_one_step(self, train_batch)

        # Combine results from discriminator and policy training.
        for policy_id in train_results:
            discriminator_policy_results = discriminator_train_results[policy_id]
            train_policy_results = train_results[policy_id]
            for key in discriminator_policy_results[LEARNER_STATS_KEY]:
                if key not in train_policy_results[LEARNER_STATS_KEY]:
                    train_policy_results[LEARNER_STATS_KEY][key] = (
                        discriminator_policy_results[LEARNER_STATS_KEY][key]
                    )
                else:
                    train_policy_results[LEARNER_STATS_KEY][key] = (
                        0.5 * train_policy_results[LEARNER_STATS_KEY][key]
                        + 0.5 * discriminator_policy_results[LEARNER_STATS_KEY][key]
                    )
            assert isinstance(train_batch, MultiAgentBatch)
            train_policy_results[LEARNER_STATS_KEY]["discriminator/rewards"] = np.mean(
                train_batch.policy_batches[policy_id][SampleBatch.REWARDS]
            )
            for key in discriminator_policy_results:
                if key == LEARNER_STATS_KEY:
                    continue
                if discriminator_policy_results[key] == {}:
                    continue

                assert key in train_policy_results
                train_policy_results[key] += discriminator_policy_results[key]

        # Validation
        if self.demonstration_val_batch is not None:
            validation_results = self._run_validation(self.demonstration_val_batch)
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
        with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
            if self.workers.num_remote_workers() > 0:
                from_worker_or_learner_group = None
                if self.config._enable_learner_api:
                    # sync weights from learner_group to all rollout workers
                    from_worker_or_learner_group = self.learner_group
                self.workers.sync_weights(
                    from_worker_or_learner_group=from_worker_or_learner_group,
                    policies=policies_to_update,
                    global_vars=global_vars,
                )
            elif self.config._enable_learner_api:
                assert self.learner_group is not None
                weights = self.learner_group.get_weights()
                self.workers.local_worker().set_weights(weights)

        # For each policy: Update KL scale and warn about possible issues
        for policy_id, policy_info in train_results.items():
            # Update KL loss with dynamic scaling
            # for each (possibly multiagent) policy we are training
            kl_divergence = policy_info[LEARNER_STATS_KEY].get("kl")
            policy = self.get_policy(policy_id)
            assert isinstance(policy, PPOTorchPolicy)
            policy.update_kl(kl_divergence)

            # Warn about excessively high value function loss
            scaled_vf_loss = (
                self.config.vf_loss_coeff * policy_info[LEARNER_STATS_KEY]["vf_loss"]
            )
            policy_loss = policy_info[LEARNER_STATS_KEY]["policy_loss"]
            if (
                log_once("ppo_warned_lr_ratio")
                and self.config.get("model", {}).get("vf_share_layers")
                and scaled_vf_loss > 100
            ):
                logger.warning(
                    "The magnitude of your value function loss for policy: {} is "
                    "extremely large ({}) compared to the policy loss ({}). This "
                    "can prevent the policy from learning. Consider scaling down "
                    "the VF loss by reducing vf_loss_coeff, or disabling "
                    "vf_share_layers.".format(policy_id, scaled_vf_loss, policy_loss)
                )
            # Warn about bad clipping configs.
            assert isinstance(train_batch, MultiAgentBatch)
            train_batch.policy_batches[policy_id].set_get_interceptor(None)
            mean_reward = train_batch.policy_batches[policy_id]["rewards"].mean()
            if (
                log_once("ppo_warned_vf_clip")
                and mean_reward > self.config.vf_clip_param
            ):
                self.warned_vf_clip = True
                logger.warning(
                    f"The mean reward returned from the environment is {mean_reward}"
                    f" but the vf_clip_param is set to {self.config['vf_clip_param']}."
                    f" Consider increasing it for policy: {policy_id} to improve"
                    " value function convergence."
                )

        # Update global vars on local worker as well.
        self.workers.local_worker().set_global_vars(global_vars)

        return train_results


register_trainable("MbagGAIL", MbagGAIL)
