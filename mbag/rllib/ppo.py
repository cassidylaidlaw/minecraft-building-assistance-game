import logging
from typing import Any, Dict, List, Mapping, Type, Union, cast

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from gymnasium import spaces
from ray.rllib.algorithms.algorithm_config import NotProvided
from ray.rllib.algorithms.ppo import PPO, PPOConfig, PPOTorchPolicy
from ray.rllib.execution.rollout_ops import (
    standardize_fields,
    synchronous_parallel_sample,
)
from ray.rllib.execution.train_ops import multi_gpu_train_one_step, train_one_step
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.policy.sample_batch import MultiAgentBatch, SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SAMPLE_TIMER,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.utils.metrics.learner_info import LEARNER_STATS_KEY
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.schedules import PiecewiseSchedule
from ray.rllib.utils.typing import PolicyID, ResultDict, TensorType
from ray.tune.registry import register_trainable
from ray.util.debug import log_once
from torch import nn

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.environment.actions import MbagAction
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.types import GOAL_BLOCKS

from .kl_regularization import ANCHOR_POLICY_ACTION_DIST_INPUTS, KLRegularizationMixin
from .torch_action_distributions import MbagBilevelCategorical
from .torch_models import ACTION_MASK, MbagTorchModel, OptimizerMixinV2

logger = logging.getLogger(__name__)


class MbagPPOTorchPolicy(OptimizerMixinV2, PPOTorchPolicy):
    config: Dict[str, Any]

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        config,
        **kwargs,
    ):
        self._place_block_loss_coeff_schedule = None
        place_block_loss_coeff_schedule = config.get("place_block_loss_coeff_schedule")
        if place_block_loss_coeff_schedule is None:
            self.place_block_loss_coeff = config["place_block_loss_coeff"]
        else:
            # Allows for custom schedule similar to lr_schedule format
            assert isinstance(place_block_loss_coeff_schedule, list)
            self._place_block_loss_coeff_schedule = PiecewiseSchedule(
                place_block_loss_coeff_schedule,
                outside_value=place_block_loss_coeff_schedule[-1][-1],
                framework=None,
            )
            self.place_block_loss_coeff = self._place_block_loss_coeff_schedule.value(0)

        self.action_mapping = torch.from_numpy(
            MbagActionDistribution.get_action_mapping(
                config["model"]["custom_model_config"]["env_config"]
            )
        )
        super().__init__(observation_space, action_space, config, **kwargs)
        self.action_mapping = self.action_mapping.to(self.device)
        self.view_requirements[ACTION_MASK] = ViewRequirement(
            space=spaces.Box(0, 1, (len(self.action_mapping),), dtype=np.bool8)  # type: ignore
        )

    def _compute_action_helper(
        self, input_dict, state_batches, seq_lens, explore, timestep
    ):
        actions, state_out, extra_fetches = super()._compute_action_helper(
            input_dict, state_batches, seq_lens, explore, timestep
        )
        model = self.model
        assert isinstance(model, MbagTorchModel)
        extra_fetches[ACTION_MASK] = convert_to_numpy(model.action_mask())
        return actions, state_out, extra_fetches

    def loss(
        self,
        model,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> TensorType:
        loss = super().loss(model, dist_class, train_batch)
        assert isinstance(loss, torch.Tensor)

        assert isinstance(model, MbagTorchModel)

        if hasattr(model, "action_dist"):
            if isinstance(model.action_dist, MbagBilevelCategorical):
                model.tower_stats["action_type_entropy"] = (
                    model.action_dist.action_type_entropy()
                )

        world_obs, _, _ = restore_original_dimensions(
            train_batch[SampleBatch.OBS],
            obs_space=self.observation_space,
            tensorlib=torch,
        )

        goal = world_obs[:, GOAL_BLOCKS].long()

        loss += self.place_block_loss_coeff * self.place_block_loss(
            model, dist_class, goal, train_batch
        )

        loss += self.config.get("goal_loss_coeff", 0) * self.predict_goal_loss(
            model, goal, train_batch
        )
        loss += self.config.get("anchor_policy_kl_coeff", 0) * self.anchor_policy_kl(
            model, dist_class, train_batch
        )

        return loss

    def postprocess_trajectory(
        self, sample_batch, other_agent_batches=None, episode=None
    ):
        for rewards_key in [
            SampleBatch.REWARDS,
            SampleBatch.PREV_REWARDS,
        ]:
            if rewards_key in sample_batch:
                sample_batch[rewards_key] *= self.config.get("reward_scale", 1.0)
        return super().postprocess_trajectory(
            sample_batch, other_agent_batches, episode
        )

    def predict_goal_loss(
        self,
        model: MbagTorchModel,
        goal: TensorType,
        train_batch: SampleBatch,
    ) -> TensorType:
        if not hasattr(model, "_backbone_out"):
            model(train_batch)
        log_odds = model.goal_predictor()

        ce = nn.CrossEntropyLoss()
        loss: torch.Tensor = ce(log_odds, goal)

        model.tower_stats["predict_goal_loss"] = loss

        return loss

    def place_block_loss(
        self,
        model: MbagTorchModel,
        dist_class: Type[ActionDistribution],
        goal: TensorType,
        train_batch,
    ) -> TensorType:
        """
        Add loss to minimize the cross-entropy between the block ID for a "place block" action
        and the goal block at that location, if there is any goal block there.
        """

        world_obs, _, _ = restore_original_dimensions(
            train_batch[SampleBatch.OBS],
            obs_space=self.observation_space,
            tensorlib=torch,
        )
        goal_block_ids = world_obs[:, 2].long()

        if hasattr(model, "logits"):
            # Don't recompute logits if we don't have to.
            logits = model.logits
        else:
            logger.warn("recomputing logits in place_block_loss")
            logits, state = model(train_batch)

        placeable_block_mask = torch.tensor(
            [
                block_id in MinecraftBlocks.PLACEABLE_BLOCK_IDS
                for block_id in range(len(MinecraftBlocks.ID2NAME))
            ],
            device=self.device,
        )

        # We only care about place block actions at places where there are blocks in the
        # goal.
        place_block_mask = ~torch.any(
            goal_block_ids[..., None]
            == placeable_block_mask[None, None, None, None, :].to(self.device),
            dim=4,
        ).flatten()

        place_block_logits = logits[
            :, self.action_mapping.to(self.device)[:, 0] == MbagAction.PLACE_BLOCK
        ].reshape((-1, MinecraftBlocks.NUM_BLOCKS) + world_obs.size()[-3:])
        place_block_logits = place_block_logits.permute((0, 2, 3, 4, 1)).flatten(
            end_dim=3
        )
        place_block_mask &= (
            place_block_logits[
                torch.arange(place_block_logits.size()[0]), goal_block_ids.flatten()
            ]
            > MbagTorchModel.MASK_LOGIT
        )

        place_block_loss = F.cross_entropy(
            place_block_logits[place_block_mask],
            goal_block_ids.flatten()[place_block_mask],
        )

        model.tower_stats["place_block_loss"] = place_block_loss

        return place_block_loss

    def anchor_policy_kl(
        self, model, dist_class: Type[ActionDistribution], train_batch
    ) -> Union[torch.Tensor, float]:
        if ANCHOR_POLICY_ACTION_DIST_INPUTS not in train_batch:
            return 0.0

        if hasattr(model, "logits"):
            # Don't recompute logits if we don't have to.
            logits = model.logits
        else:
            logger.warn("recomputing logits in anchor_policy_kl")
            logits, state = model(train_batch)

        action_dist = dist_class(logits, model)
        anchor_policy_action_dist_inputs = train_batch[ANCHOR_POLICY_ACTION_DIST_INPUTS]
        anchor_policy_action_dist = dist_class(anchor_policy_action_dist_inputs, model)

        if self.config.get("anchor_policy_reverse_kl", False):
            anchor_policy_kl = anchor_policy_action_dist.kl(action_dist).mean()
        else:
            anchor_policy_kl = action_dist.kl(anchor_policy_action_dist).mean()
        model.tower_stats["anchor_policy_kl"] = anchor_policy_kl

        return anchor_policy_kl

    def log_mean_stat(self, info: Dict[str, TensorType], loss_name: str):
        try:
            info[loss_name] = torch.mean(
                torch.stack(cast(List[torch.Tensor], self.get_tower_stats(loss_name)))
            )
        except AssertionError:
            info[loss_name] = torch.tensor(np.nan)

    def stats_fn(self, train_batch: SampleBatch) -> Dict[str, TensorType]:
        info = super().stats_fn(train_batch)

        self.log_mean_stat(info, "place_block_loss")
        self.log_mean_stat(info, "predict_goal_loss")
        self.log_mean_stat(info, "anchor_policy_kl")
        self.log_mean_stat(info, "action_type_entropy")

        info["place_block_loss_coeff"] = self.place_block_loss_coeff

        return cast(
            Dict[str, TensorType],
            convert_to_numpy(info),
        )

    def update_kl(self, sampled_kl):
        super().update_kl(sampled_kl)
        # Don't let the KL coefficient go below 0.001.
        self.kl_coeff = max(self.kl_coeff, 1e-3)

    def on_global_var_update(self, global_vars):
        super().on_global_var_update(global_vars)
        if self._place_block_loss_coeff_schedule is not None:
            self.place_block_loss_coeff = self._place_block_loss_coeff_schedule.value(
                global_vars["timestep"]
            )


class MbagPPOConfig(PPOConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class)

        self.goal_loss_coeff = 1.0
        self.place_block_loss_coeff = 1.0
        self.place_block_loss_coeff_schedule = None
        self.reward_scale = 1.0
        self.anchor_policy_mapping: Mapping[PolicyID, PolicyID] = {}
        self.anchor_policy_kl_coeff = 0.0
        self.anchor_policy_reverse_kl = False

    def training(
        self,
        *args,
        goal_loss_coeff=NotProvided,
        place_block_loss_coeff=NotProvided,
        place_block_loss_coeff_schedule=NotProvided,
        reward_scale=NotProvided,
        anchor_policy_mapping=NotProvided,
        anchor_policy_kl_coeff=NotProvided,
        anchor_policy_reverse_kl=NotProvided,
        **kwargs,
    ):
        super().training(*args, **kwargs)

        if goal_loss_coeff is not NotProvided:
            self.goal_loss_coeff = goal_loss_coeff
        if place_block_loss_coeff is not NotProvided:
            self.place_block_loss_coeff = place_block_loss_coeff
        if place_block_loss_coeff_schedule is not NotProvided:
            self.place_block_loss_coeff_schedule = place_block_loss_coeff_schedule
        if reward_scale is not NotProvided:
            self.reward_scale = reward_scale
        if anchor_policy_mapping is not NotProvided:
            self.anchor_policy_mapping = anchor_policy_mapping
        if anchor_policy_kl_coeff is not NotProvided:
            self.anchor_policy_kl_coeff = anchor_policy_kl_coeff
        if anchor_policy_reverse_kl is not NotProvided:
            self.anchor_policy_reverse_kl = anchor_policy_reverse_kl


class MbagPPO(PPO, KLRegularizationMixin):
    config: MbagPPOConfig  # type: ignore[assignment]

    @classmethod
    def get_default_config(cls):
        return MbagPPOConfig()

    @classmethod
    def get_default_policy_class(cls, config):
        return MbagPPOTorchPolicy

    def training_step(self) -> ResultDict:
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

        assert not isinstance(train_batch, list)
        train_batch = train_batch.as_multi_agent()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += train_batch.agent_steps()
        self._counters[NUM_ENV_STEPS_SAMPLED] += train_batch.env_steps()

        # Add anchor policy action dist inputs to sample batch to be used for KL
        # regularization.
        train_batch = self._add_anchor_policy_action_dist_inputs_to_sample_batch(
            train_batch
        )

        # Standardize advantages
        train_batch = standardize_fields(train_batch, ["advantages"])
        # Train
        if self.config.simple_optimizer:
            train_results = train_one_step(self, train_batch)
        else:
            train_results = multi_gpu_train_one_step(self, train_batch)

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

        # For each policy: Update KL scale and warn about possible issues
        for policy_id, policy_info in train_results.items():
            # Update KL loss with dynamic scaling
            # for each (possibly multiagent) policy we are training
            kl_divergence = policy_info[LEARNER_STATS_KEY].get("kl")
            policy = self.get_policy(policy_id)
            assert isinstance(policy, MbagPPOTorchPolicy)
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


register_trainable("MbagPPO", MbagPPO)
