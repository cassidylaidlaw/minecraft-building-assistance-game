import logging
from typing import Dict, Optional

import numpy as np
from ray.rllib.algorithms.algorithm_config import NotProvided
from ray.rllib.algorithms.alpha_zero.alpha_zero import AlphaZero, AlphaZeroConfig
from ray.rllib.evaluation import SampleBatch
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import multi_gpu_train_one_step, train_one_step
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import MultiAgentBatch, concat_samples
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SAMPLE_TIMER,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.utils.replay_buffers import ReplayBuffer
from ray.rllib.utils.typing import PolicyID, ResultDict
from ray.tune.registry import register_trainable

from .alpha_zero_policy import (
    EXPECTED_OWN_REWARDS,
    EXPECTED_REWARDS,
    OWN_REWARDS,
    VALUE_ESTIMATES,
    MbagAlphaZeroPolicy,
)

logger = logging.getLogger(__name__)


class MbagAlphaZeroConfig(AlphaZeroConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class)

        self.sample_batch_size = 1000
        self.vf_loss_coeff = 1.0
        self.other_agent_action_predictor_loss_coeff = 1.0
        self.goal_loss_coeff = 1.0
        self.entropy_coeff = 0
        self.entropy_coeff_schedule = 0
        self.use_critic = True
        self.use_goal_predictor = True
        self.use_replay_buffer = True
        self.num_steps_sampled_before_learning_starts = 0
        self.pretrain = False
        self.player_index: Optional[int] = None
        self.strict_mode = False

        del self.vf_share_layers

    def training(
        self,
        *args,
        sample_batch_size=NotProvided,
        vf_loss_coeff=NotProvided,
        other_agent_action_predictor_loss_coeff=NotProvided,
        goal_loss_coeff=NotProvided,
        entropy_coeff=NotProvided,
        entropy_coeff_schedule=NotProvided,
        use_critic=NotProvided,
        use_goal_predictor=NotProvided,
        use_replay_buffer=NotProvided,
        num_steps_sampled_before_learning_starts=NotProvided,
        pretrain=NotProvided,
        player_index=NotProvided,
        _strict_mode=NotProvided,
        **kwargs,
    ):
        """
        Set training parameters.
        Args:
            sample_batch_size (int): Number of samples to include in each
                training batch.
            vf_loss_coeff (float): Coefficient of the value function loss.
            other_agent_action_predictor_loss_coeff (float): Coefficient of the
                other agent action predictor loss.
            goal_loss_coeff (float): Coefficient of the goal predictor loss.
            entropy_coeff (float): Coefficient of the entropy loss.
            entropy_coeff_schedule (float): Schedule for the entropy
                coefficient.
            use_critic (bool): Whether to use a critic.
            use_goal_predictor (bool): Whether to use a goal predictor.
            use_replay_buffer (bool): Whether to use a replay buffer.
            num_steps_sampled_before_learning_starts (int): Number of steps
                collected before learning starts.
            pretrain (bool): If True, then this will just pretrain the AlphaZero
                predictors for goal, other agent action, etc. and take only NOOP
                actions.
            player_index (int): Override the AGENT_INDEX field in the sample
                batch with this value.
            _strict_mode (bool): Enables various assertions that may slow down or
                mess up training in practice but are useful for testing.
        """

        super().training(*args, **kwargs)

        if sample_batch_size is not NotProvided:
            self.sample_batch_size = sample_batch_size
        if vf_loss_coeff is not NotProvided:
            self.vf_loss_coeff = vf_loss_coeff
        if other_agent_action_predictor_loss_coeff is not NotProvided:
            self.other_agent_action_predictor_loss_coeff = (
                other_agent_action_predictor_loss_coeff
            )
        if goal_loss_coeff is not NotProvided:
            self.goal_loss_coeff = goal_loss_coeff
        if entropy_coeff is not NotProvided:
            self.entropy_coeff = entropy_coeff
        if entropy_coeff_schedule is not NotProvided:
            self.entropy_coeff_schedule = entropy_coeff_schedule
        if use_critic is not NotProvided:
            self.use_critic = use_critic
        if use_goal_predictor is not NotProvided:
            self.use_goal_predictor = use_goal_predictor
        if use_replay_buffer is not NotProvided:
            self.use_replay_buffer = use_replay_buffer
        if num_steps_sampled_before_learning_starts is not NotProvided:
            self.num_steps_sampled_before_learning_starts = (
                num_steps_sampled_before_learning_starts
            )
        if pretrain is not NotProvided:
            self.pretrain = pretrain
        if player_index is not NotProvided:
            self.player_index = player_index
        if _strict_mode is not NotProvided:
            self._strict_mode = _strict_mode

    def update_from_dict(self, config_dict):
        if "mcts_config" in config_dict and isinstance(config_dict, dict):
            self.mcts_config.update(config_dict["mcts_config"])
            del config_dict["mcts_config"]

        return super().update_from_dict(config_dict)


class MbagAlphaZero(AlphaZero):
    local_replay_buffer: Optional[ReplayBuffer]  # type: ignore[assignment]

    def __init__(self, config: MbagAlphaZeroConfig, *args, **kwargs):
        del config.ranked_rewards

        super().__init__(config, *args, **kwargs)

        if not config.use_replay_buffer:
            self.local_replay_buffer = None

        self._have_set_policies_training = False

    @classmethod
    def get_default_config(cls):
        return MbagAlphaZeroConfig()

    def get_default_policy_class(self, config):
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

        assert isinstance(new_sample_batch, MultiAgentBatch)
        prediction_metrics_by_policy = self.get_reward_and_value_prediction_metrics(
            new_sample_batch
        )

        # Update sampling step counters.
        self._counters[NUM_ENV_STEPS_SAMPLED] += new_sample_batch.env_steps()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += new_sample_batch.agent_steps()
        # Store new samples in the replay buffer.
        if self.local_replay_buffer is not None:
            with self._timers["replay_buffer"]:
                # First, remove non-trainable policies.
                policy_ids_to_train = (
                    self.workers.local_worker().foreach_policy_to_train(
                        lambda policy, policy_id, **kwargs: policy_id
                    )
                )
                for policy_id in list(new_sample_batch.policy_batches.keys()):
                    if policy_id not in policy_ids_to_train:
                        del new_sample_batch.policy_batches[policy_id]
                self.local_replay_buffer.add(new_sample_batch)

                cur_ts = self._counters[
                    (
                        NUM_AGENT_STEPS_SAMPLED
                        if self.config.count_steps_by == "agent_steps"
                        else NUM_ENV_STEPS_SAMPLED
                    )
                ]

                if cur_ts > self.config.num_steps_sampled_before_learning_starts:
                    train_batch = self.local_replay_buffer.sample(
                        self.config.train_batch_size
                    )
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
