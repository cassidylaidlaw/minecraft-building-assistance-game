import logging
from typing import Dict, List, Mapping, Optional, Tuple, cast

import numpy as np
from ray.rllib.algorithms.algorithm_config import NotProvided
from ray.rllib.algorithms.alpha_zero.alpha_zero import AlphaZero, AlphaZeroConfig
from ray.rllib.evaluation import SampleBatch
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import multi_gpu_train_one_step, train_one_step
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import MultiAgentBatch, concat_samples
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SAMPLE_TIMER,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.utils.replay_buffers import ReplayBuffer, StorageUnit
from ray.rllib.utils.sgd import minibatches
from ray.rllib.utils.typing import PolicyID, ResultDict, SampleBatchType
from ray.tune.registry import register_trainable

from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.types import CURRENT_BLOCKS, GOAL_BLOCKS

from ..kl_regularization import KLRegularizationMixin
from .alpha_zero_policy import (
    EXPECTED_OWN_REWARDS,
    EXPECTED_REWARDS,
    FOR_TRAINING_MODEL,
    GOAL_LOGITS,
    OWN_REWARDS,
    VALUE_ESTIMATES,
    MbagAlphaZeroPolicy,
)
from .replay_buffer import PartialReplayBuffer

logger = logging.getLogger(__name__)


class MbagAlphaZeroConfig(AlphaZeroConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class)

        self.use_model_replay_buffer = False
        self.model_replay_buffer_config: dict = {
            "type": "MultiAgentReplayBuffer",
            "capacity": 1_000,
            "storage_unit": StorageUnit.SEQUENCES,
            "underlying_buffer_config": {
                "type": PartialReplayBuffer,
                "storage_probability": 0.1,
            },
        }
        self.model_train_batch_size: int = self.train_batch_size
        self.mcts_batch_size: Optional[int] = None
        self.sample_batch_size = 1000
        self.sample_freq = 1
        self.policy_loss_coeff = 1.0
        self.prev_policy_kl_coeff = 0.0
        self.vf_loss_coeff = 1.0
        self.other_agent_action_predictor_loss_coeff = 1.0
        self.goal_loss_coeff = 1.0
        self.prev_goal_kl_coeff = 0.0
        self.entropy_coeff = 0
        self.entropy_coeff_schedule = 0
        self.use_critic = True
        self.use_goal_predictor = True
        self.use_other_agent_action_predictor = True
        self.expected_own_reward_scale = 1.0
        self.expected_reward_shift = 0.0
        self.use_replay_buffer = True
        self.num_steps_sampled_before_learning_starts: int = 0
        self.anchor_policy_mapping: Mapping[PolicyID, PolicyID] = {}
        self.anchor_policy_kl_coeff = 0.0
        self.pretrain = False
        self.player_index: Optional[int] = None
        self.strict_mode = False

        del self.vf_share_layers

    def training(  # noqa: C901
        self,
        *args,
        use_model_replay_buffer=NotProvided,
        model_replay_buffer_config=NotProvided,
        model_train_batch_size=NotProvided,
        mcts_batch_size=NotProvided,
        sample_freq=NotProvided,
        sample_batch_size=NotProvided,
        policy_loss_coeff=NotProvided,
        prev_policy_kl_coeff=NotProvided,
        vf_loss_coeff=NotProvided,
        other_agent_action_predictor_loss_coeff=NotProvided,
        goal_loss_coeff=NotProvided,
        prev_goal_kl_coeff=NotProvided,
        prev_goal_kl_coeff_schedule=NotProvided,
        entropy_coeff=NotProvided,
        entropy_coeff_schedule=NotProvided,
        use_critic=NotProvided,
        use_goal_predictor=NotProvided,
        use_other_agent_action_predictor=NotProvided,
        expected_own_reward_scale=NotProvided,
        expected_reward_shift=NotProvided,
        use_replay_buffer=NotProvided,
        num_steps_sampled_before_learning_starts=NotProvided,
        anchor_policy_mapping=NotProvided,
        anchor_policy_kl_coeff=NotProvided,
        pretrain=NotProvided,
        player_index=NotProvided,
        _strict_mode=NotProvided,
        **kwargs,
    ):
        """
        Set training parameters.
        Args:
            use_model_replay_buffer (bool): Whether to use a separate training buffer
                to train the model parts of the network (i.e., goal and other agent
                action prediction).
            model_replay_buffer_config (dict): Config for the model replay buffer.
            model_train_batch_size (int): Number of samples to include in each
                training batch from the model replay buffer.
            mcts_batch_size (int): Number of environments to run MCTS over in parallel.
                If there are more environments than this, then they will be split into
                batches of this size. If None, then this will be set to the number of
                environments.
            sample_freq (int): If > 1, then only sample new data every `sample_freq`
                iterations.
            sample_batch_size (int): Number of samples to include in each
                training batch.
            policy_loss_coeff (float): Coefficient of the policy loss.
            prev_policy_kl_coeff (float): Coefficient between the KL of the previous
                and current policy network output during training.
            vf_loss_coeff (float): Coefficient of the value function loss.
            other_agent_action_predictor_loss_coeff (float): Coefficient of the
                other agent action predictor loss.
            goal_loss_coeff (float): Coefficient of the goal predictor loss.
            prev_goal_kl_coeff (float): Coefficient between the KL of the previous
                goal predictions and the current goal predictions during training
                (encourages stability of goal predictions).
            prev_goal_kl_coeff_schedule (float): Schedule for the previous goal KL
                coefficient.
            entropy_coeff (float): Coefficient of the entropy loss.
            entropy_coeff_schedule (float): Schedule for the entropy
                coefficient.
            use_critic (bool): Whether to use a critic.
            use_goal_predictor (bool): Whether to use a goal predictor.
            use_other_agent_action_predictor (bool): Whether to use an other agent
                action predictor. If False and there are two agents, then always
                predicts NOOPs for the other player.
            expected_own_reward_scale (float): The expected own reward is scaled by
                this value before being used for planning.
            expected_reward_shift (float): The expected reward is shifted by this
                value before being used for planning.
            use_replay_buffer (bool): Whether to use a replay buffer.
            num_steps_sampled_before_learning_starts (int): Number of steps
                collected before learning starts.
            anchor_policy_mapping (dict): Mapping from policy IDs to anchor
                policies for KL regularization.
            anchor_policy_kl_coeff (float): Coefficient of the KL regularization
                loss to the anchor policy.
            pretrain (bool): If True, then this will just pretrain the AlphaZero
                predictors for goal, other agent action, etc. and take only NOOP
                actions.
            player_index (int): Override the AGENT_INDEX field in the sample
                batch with this value.
            _strict_mode (bool): Enables various assertions that may slow down or
                mess up training in practice but are useful for testing.
        """

        super().training(*args, **kwargs)

        if use_model_replay_buffer is not NotProvided:
            self.use_model_replay_buffer = use_model_replay_buffer
        if model_replay_buffer_config is not NotProvided:
            self.model_replay_buffer_config.update(model_replay_buffer_config)
        if model_train_batch_size is not NotProvided:
            self.model_train_batch_size = model_train_batch_size
        if mcts_batch_size is not NotProvided:
            self.mcts_batch_size = mcts_batch_size
        if sample_freq is not NotProvided:
            self.sample_freq = sample_freq
        if sample_batch_size is not NotProvided:
            self.sample_batch_size = sample_batch_size
        if policy_loss_coeff is not NotProvided:
            self.policy_loss_coeff = policy_loss_coeff
        if prev_policy_kl_coeff is not NotProvided:
            self.prev_policy_kl_coeff = prev_policy_kl_coeff
        if vf_loss_coeff is not NotProvided:
            self.vf_loss_coeff = vf_loss_coeff
        if other_agent_action_predictor_loss_coeff is not NotProvided:
            self.other_agent_action_predictor_loss_coeff = (
                other_agent_action_predictor_loss_coeff
            )
        if goal_loss_coeff is not NotProvided:
            self.goal_loss_coeff = goal_loss_coeff
        if prev_goal_kl_coeff is not NotProvided:
            self.prev_goal_kl_coeff = prev_goal_kl_coeff
        if prev_goal_kl_coeff_schedule is not NotProvided:
            self.prev_goal_kl_coeff_schedule = prev_goal_kl_coeff_schedule
        if entropy_coeff is not NotProvided:
            self.entropy_coeff = entropy_coeff
        if entropy_coeff_schedule is not NotProvided:
            self.entropy_coeff_schedule = entropy_coeff_schedule
        if use_critic is not NotProvided:
            self.use_critic = use_critic
        if use_goal_predictor is not NotProvided:
            self.use_goal_predictor = use_goal_predictor
        if use_other_agent_action_predictor is not NotProvided:
            self.use_other_agent_action_predictor = use_other_agent_action_predictor
        if expected_own_reward_scale is not NotProvided:
            self.expected_own_reward_scale = expected_own_reward_scale
        if expected_reward_shift is not NotProvided:
            self.expected_reward_shift = expected_reward_shift
        if use_replay_buffer is not NotProvided:
            self.use_replay_buffer = use_replay_buffer
        if num_steps_sampled_before_learning_starts is not NotProvided:
            self.num_steps_sampled_before_learning_starts = (
                num_steps_sampled_before_learning_starts
            )
        if anchor_policy_mapping is not NotProvided:
            self.anchor_policy_mapping = anchor_policy_mapping
        if anchor_policy_kl_coeff is not NotProvided:
            self.anchor_policy_kl_coeff = anchor_policy_kl_coeff
        if pretrain is not NotProvided:
            self.pretrain = pretrain
        if player_index is not NotProvided:
            self.player_index = player_index
        if _strict_mode is not NotProvided:
            self._strict_mode = _strict_mode

        return self

    def update_from_dict(self, config_dict):
        if "mcts_config" in config_dict and isinstance(config_dict, dict):
            self.mcts_config.update(config_dict["mcts_config"])
            del config_dict["mcts_config"]

        return super().update_from_dict(config_dict)


class MbagAlphaZero(AlphaZero, KLRegularizationMixin):
    local_replay_buffer: Optional[ReplayBuffer]  # type: ignore[assignment]
    model_replay_buffer: Optional[ReplayBuffer]

    def __init__(self, config: MbagAlphaZeroConfig, *args, **kwargs):
        del config.ranked_rewards

        super().__init__(config, *args, **kwargs)

        if not config.use_replay_buffer:
            self.local_replay_buffer = None
        else:
            assert self.local_replay_buffer is not None

        if config.use_model_replay_buffer:
            self.model_replay_buffer = from_config(
                ReplayBuffer, config["model_replay_buffer_config"]
            )
        else:
            self.model_replay_buffer = None

        self._have_set_policies_training = False

    @classmethod
    def get_default_config(cls):
        return MbagAlphaZeroConfig()

    @classmethod
    def get_default_policy_class(cls, config):
        return MbagAlphaZeroPolicy

    def _set_policies_training(self):
        is_policy_to_train_dict = {}
        assert self.workers is not None
        policy_ids = self.workers.local_worker().foreach_policy(
            lambda policy, policy_id, *args, **kwargs: policy_id
        )
        for policy_id in policy_ids:
            is_policy_to_train_dict[
                policy_id
            ] = self.workers.local_worker().is_policy_to_train(
                policy_id, None  # type: ignore
            )

        def set_policy_training(
            policy: Policy,
            policy_id: PolicyID,
            is_policy_to_train_dict=is_policy_to_train_dict,
        ):
            if isinstance(policy, MbagAlphaZeroPolicy):
                policy.set_training(is_policy_to_train_dict[policy_id])

        self.workers.foreach_policy_to_train(set_policy_training)

    def get_reward_and_value_prediction_metrics(
        self, sample_batch: MultiAgentBatch
    ) -> Dict[PolicyID, dict]:
        metrics_by_policy = {}

        for policy_id, policy_batch in sample_batch.policy_batches.items():
            policy = self.get_policy(policy_id)
            if not isinstance(policy, MbagAlphaZeroPolicy):
                continue

            prediction_stats = {}
            for stat_key, estimates, targets in [
                (
                    "vf",
                    policy_batch[VALUE_ESTIMATES],
                    policy_batch[Postprocessing.VALUE_TARGETS],
                ),
                (
                    "reward",
                    policy_batch[EXPECTED_REWARDS],
                    policy_batch[SampleBatch.REWARDS],
                ),
                (
                    "own_reward",
                    policy_batch[EXPECTED_OWN_REWARDS],
                    policy_batch[OWN_REWARDS],
                ),
            ]:
                bias = np.mean(estimates - targets)
                mse = np.mean((estimates - targets) ** 2)
                var = mse - bias**2
                prediction_stats[f"{stat_key}_bias"] = bias
                prediction_stats[f"{stat_key}_var"] = var
                prediction_stats[f"{stat_key}_mse"] = mse

            metrics_by_policy[policy_id] = {"prediction_stats": prediction_stats}

        return metrics_by_policy

    def get_goal_prediction_metrics(
        self, sample_batch: MultiAgentBatch
    ) -> Dict[PolicyID, dict]:
        metrics_by_policy: Dict[PolicyID, dict] = {}

        for policy_id, policy_batch in sample_batch.policy_batches.items():
            policy = self.get_policy(policy_id)
            if not isinstance(policy, MbagAlphaZeroPolicy):
                continue
            if GOAL_LOGITS not in policy_batch:
                continue

            total_blocks = 0
            total_cross_entropy = 0.0
            total_unplaced_blocks = 0
            total_unplaced_blocks_cross_entropy = 0.0

            for minibatch in minibatches(
                policy_batch, self.config["sgd_minibatch_size"]
            ):
                minibatch.decompress_if_needed()
                world_obs, _, _ = restore_original_dimensions(
                    minibatch[SampleBatch.OBS],
                    obs_space=policy.observation_space,
                    tensorlib=np,
                )
                goal_blocks: np.ndarray = world_obs[:, GOAL_BLOCKS].astype(np.uint8)

                goal_blocks_flat: np.ndarray = goal_blocks.reshape(-1)
                goal_logits_flat: np.ndarray = (
                    minibatch[GOAL_LOGITS]
                    .transpose(0, 2, 3, 4, 1)
                    .reshape(
                        -1,
                        MinecraftBlocks.NUM_BLOCKS,
                    )
                )
                goal_logprobs_flat = goal_logits_flat - np.log(
                    np.sum(np.exp(goal_logits_flat), axis=1, keepdims=True)
                )
                if self.config["_strict_mode"]:
                    assert np.allclose(
                        np.sum(np.exp(goal_logprobs_flat), axis=1),
                        np.ones(len(goal_blocks_flat)),
                    )

                goal_ce = -goal_logprobs_flat[
                    np.arange(len(goal_blocks_flat)), goal_blocks_flat
                ]
                unplaced_blocks = (goal_blocks != MinecraftBlocks.AIR) & (
                    world_obs[:, CURRENT_BLOCKS] == MinecraftBlocks.AIR
                )
                unplaced_blocks_flat = unplaced_blocks.reshape(-1)

                total_blocks += len(goal_ce)
                total_cross_entropy += np.sum(goal_ce)
                total_unplaced_blocks += np.sum(unplaced_blocks)
                total_unplaced_blocks_cross_entropy += np.sum(
                    goal_ce[unplaced_blocks_flat]
                )

            metrics_by_policy[policy_id] = {
                "goal_cross_entropy": total_cross_entropy / total_blocks,
                "unplaced_blocks_goal_cross_entropy": total_unplaced_blocks_cross_entropy
                / total_unplaced_blocks,
            }

        return metrics_by_policy

    def _combine_train_batches(
        self,
        policy_train_batch: MultiAgentBatch,
        model_train_batch: MultiAgentBatch,
    ) -> MultiAgentBatch:
        combined_policy_batches: Dict[PolicyID, SampleBatch] = {}

        for policy_id in (
            policy_train_batch.policy_batches.keys()
            | model_train_batch.policy_batches.keys()
        ):
            policy_batch = policy_train_batch.policy_batches[policy_id]
            model_batch = model_train_batch.policy_batches[policy_id]
            policy_batch[FOR_TRAINING_MODEL] = np.zeros(
                len(policy_batch[SampleBatch.REWARDS]), dtype=bool
            )
            model_batch[FOR_TRAINING_MODEL] = np.ones(
                len(model_batch[SampleBatch.REWARDS]), dtype=bool
            )

            if "seq_lens" in policy_batch and "seq_lens" in model_batch:
                policy_batch_sequences: List[SampleBatch] = []
                seq_start = 0
                assert not policy_batch.zero_padded
                for seq_len in policy_batch["seq_lens"]:
                    seq_end = seq_start + seq_len
                    policy_batch_sequences.append(
                        policy_batch.slice(seq_start, seq_end)
                    )
                    seq_start = seq_end
                assert seq_start == policy_batch.count

                model_batch_sequences: List[SampleBatch] = []
                seq_start = 0
                assert not model_batch.zero_padded
                for seq_len in model_batch["seq_lens"]:
                    seq_end = seq_start + seq_len
                    model_batch_sequences.append(model_batch.slice(seq_start, seq_end))
                    seq_start = seq_end
                assert seq_start == model_batch.count

                combined_sequences = policy_batch_sequences + model_batch_sequences
                np.random.shuffle(combined_sequences)  # type: ignore
                combined_batch = concat_samples(
                    cast(List[SampleBatchType], combined_sequences)
                )
                assert isinstance(combined_batch, SampleBatch)
            else:
                combined_batch = concat_samples([policy_batch, model_batch])
                assert isinstance(combined_batch, SampleBatch)
                combined_batch.shuffle()
            combined_policy_batches[policy_id] = combined_batch

        return MultiAgentBatch(
            combined_policy_batches,
            policy_train_batch.env_steps() + model_train_batch.env_steps(),
        )

    def _sample_and_add_to_replay_buffer(
        self,
    ) -> Tuple[MultiAgentBatch, Dict[PolicyID, dict]]:
        assert self.workers is not None

        # Sample n MultiAgentBatches from n workers.
        with self._timers[SAMPLE_TIMER]:
            new_sample_batches = synchronous_parallel_sample(
                worker_set=self.workers,
                concat=False,
                max_env_steps=self.config["sample_batch_size"],
            )

        if isinstance(new_sample_batches, list):
            new_sample_batch = concat_samples(new_sample_batches)
        else:
            new_sample_batch = new_sample_batches

        new_sample_batch = self._add_anchor_policy_action_dist_inputs_to_sample_batch(
            new_sample_batch
        )

        assert isinstance(new_sample_batch, MultiAgentBatch)
        prediction_metrics_by_policy: Dict[PolicyID, dict] = {}
        for (
            policy_id,
            prediction_metrics,
        ) in self.get_reward_and_value_prediction_metrics(new_sample_batch).items():
            prediction_metrics_by_policy.setdefault(policy_id, {}).update(
                prediction_metrics
            )
        for policy_id, prediction_metrics in self.get_goal_prediction_metrics(
            new_sample_batch
        ).items():
            prediction_metrics_by_policy.setdefault(policy_id, {}).update(
                prediction_metrics
            )

        # Update sampling step counters.
        self._counters[NUM_ENV_STEPS_SAMPLED] += new_sample_batch.env_steps()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += new_sample_batch.agent_steps()
        # Store new samples in the replay buffer.
        if self.local_replay_buffer is not None:
            with self._timers["replay_buffer"]:
                # First, remove non-trainable policies and get rid of info dicts.
                for policy_id in list(new_sample_batch.policy_batches.keys()):
                    del new_sample_batch.policy_batches[policy_id][SampleBatch.INFOS]
                    if not self.workers.local_worker().is_policy_to_train(policy_id):  # type: ignore
                        del new_sample_batch.policy_batches[policy_id]
                self.local_replay_buffer.add(new_sample_batch)

                if self.model_replay_buffer is not None:
                    self.model_replay_buffer.add(new_sample_batch)

        return new_sample_batch, prediction_metrics_by_policy

    def training_step(self) -> ResultDict:
        assert self.workers is not None
        assert isinstance(self.config, MbagAlphaZeroConfig)

        if not self._have_set_policies_training:
            # Only policies that are set as training will use reward shaping schedules;
            # others will just use the final point in the schedule.
            # We only set the policies as training once train() is actually called so
            # that if policies are loaded for evaluation then the shaped reward
            # annealing is not used.
            self._set_policies_training()
            self._have_set_policies_training = True

        new_sample_batch: Optional[MultiAgentBatch]
        prediction_metrics_by_policy: Dict[PolicyID, dict] = {}
        if self.iteration % self.config["sample_freq"] == 0:
            new_sample_batch, prediction_metrics_by_policy = (
                self._sample_and_add_to_replay_buffer()
            )
        else:
            new_sample_batch = None

        if self.local_replay_buffer is not None:
            del new_sample_batch
            with self._timers["replay_buffer"]:
                cur_ts = self._counters[
                    (
                        NUM_AGENT_STEPS_SAMPLED
                        if self.config.count_steps_by == "agent_steps"
                        else NUM_ENV_STEPS_SAMPLED
                    )
                ]

                if cur_ts > self.config.num_steps_sampled_before_learning_starts:
                    policy_train_batch = self.local_replay_buffer.sample(
                        self.config.train_batch_size
                    )
                    if self.model_replay_buffer is not None:
                        model_train_batch = self.model_replay_buffer.sample(
                            self.config.model_train_batch_size
                        )

                        assert (
                            policy_train_batch is not None
                            and model_train_batch is not None
                        )
                        train_batch = self._combine_train_batches(
                            policy_train_batch.as_multi_agent(),
                            model_train_batch.as_multi_agent(),
                        )
                        del policy_train_batch, model_train_batch
                    else:
                        train_batch = policy_train_batch
                else:
                    train_batch = None
        else:
            train_batch = new_sample_batch

        # Learn on the training batch.
        # Use simple optimizer (only for multi-agent or tf-eager; all other
        # cases should use the multi-GPU optimizer, even if only using 1 GPU)
        train_results = {}
        if train_batch is not None:
            if self.config.get("simple_optimizer") is True:
                train_results = train_one_step(self, train_batch)
            else:
                train_results = multi_gpu_train_one_step(self, train_batch)

        # Update weights and global_vars - after learning on the local worker - on all
        # remote workers.
        global_vars = {
            "timestep": self._counters[NUM_ENV_STEPS_SAMPLED],
        }
        with self._timers[SYNCH_WORKER_WEIGHTS_TIMER]:
            self.workers.sync_weights(global_vars=global_vars)

        for policy_id, prediction_metrics in prediction_metrics_by_policy.items():
            train_results.setdefault(policy_id, {}).setdefault(
                "custom_metrics", {}
            ).update(prediction_metrics)

        # Return all collected metrics for the iteration.
        return train_results


register_trainable("MbagAlphaZero", MbagAlphaZero)
