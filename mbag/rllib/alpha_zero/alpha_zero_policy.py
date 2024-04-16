import copy
import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np
import torch
from gymnasium import spaces
from ray.rllib.algorithms.alpha_zero.alpha_zero_policy import AlphaZeroPolicy
from ray.rllib.evaluation import SampleBatch
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.postprocessing import Postprocessing, discount_cumsum
from ray.rllib.models import ActionDistribution, ModelV2
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.policy.torch_mixins import EntropyCoeffSchedule, LearningRateSchedule
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    explained_variance,
    sequence_mask,
)
from ray.rllib.utils.typing import AgentID, PolicyID, TensorStructType, TensorType
from ray.tune.registry import ENV_CREATOR, _global_registry
from torch import nn

from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.config import MbagConfigDict
from mbag.environment.types import CURRENT_BLOCKS, GOAL_BLOCKS, MbagInfoDict

from ..rllib_env import unwrap_mbag_env
from ..torch_models import ACTION_MASK, MbagTorchModel, OtherAgentActionPredictorMixin
from .mcts import MbagMCTS, MbagMCTSNode, MbagRootParentNode
from .planning import MbagEnvModel

MCTS_POLICIES = "mcts_policies"
OTHER_AGENT_ACTION_DIST_INPUTS = "other_agent_action_dist_inputs"
OWN_REWARDS = "own_rewards"
EXPECTED_REWARDS = "expected_rewards"
EXPECTED_OWN_REWARDS = "expected_own_rewards"
VALUE_ESTIMATES = "value_estimates"
FORCE_NOOP = "force_noop"


logger = logging.getLogger(__name__)


class MbagAlphaZeroPolicy(EntropyCoeffSchedule, LearningRateSchedule, AlphaZeroPolicy):
    mcts: MbagMCTS
    envs: List[MbagEnvModel]
    config: Dict[str, Any]

    def __init__(
        self,
        observation_space,
        action_space,
        config,
        **kwargs,
    ):
        TorchPolicy.__init__(
            self,
            observation_space,
            action_space,
            config,
            **kwargs,
        )

        self.set_training(False)
        # We default to setting policies as not training and only update this when
        # train() is actually called. This ensures that if policies are loaded for
        # evaluation then the shaped reward annealing is not used.

        model = self.model
        assert isinstance(model, MbagTorchModel)
        line_of_sight_masking = model.line_of_sight_masking

        self.mcts = MbagMCTS(
            self.model,
            config["mcts_config"],
            config["gamma"],
            use_critic=config["use_critic"],
            use_goal_predictor=config["use_goal_predictor"],
            _strict_mode=config.get("_strict_mode", False),
        )

        def env_creator():
            env_creator = _global_registry.get(ENV_CREATOR, config["env"])
            # We should never use Malmo in the env model.
            env_config: MbagConfigDict = copy.deepcopy(config["env_config"])
            env_config["malmo"]["use_malmo"] = False
            # Don't waste time generating goals in the env model.
            env_config["goal_generator"] = "basic"
            env_config["goal_generator_config"] = {}
            # If we're using a goal predictor, then we shouldn't end the episode when
            # the goal is completed because that leaks information about the goal.
            if self.mcts.use_goal_predictor:
                env_config["terminate_on_goal_completion"] = False
            env = env_creator(env_config)
            env_model = MbagEnvModel(
                env, env_config, line_of_sight_masking=line_of_sight_masking
            )
            unwrap_mbag_env(env_model).update_global_timestep(
                self.global_timestep_for_envs
            )
            return env_model

        self.env_creator = env_creator
        self.envs = []
        self.obs_space = observation_space

        self.view_requirements[ACTION_MASK] = ViewRequirement(
            space=spaces.MultiBinary(action_space.n)
        )
        self.view_requirements[MCTS_POLICIES] = ViewRequirement(
            space=spaces.Box(low=0, high=1, shape=(action_space.n,))
        )
        self.view_requirements[SampleBatch.ACTION_DIST_INPUTS] = ViewRequirement(
            space=spaces.Box(low=-np.inf, high=np.inf, shape=(action_space.n,))
        )
        self.view_requirements[EXPECTED_REWARDS] = ViewRequirement(
            space=spaces.Box(low=-np.inf, high=np.inf, shape=())
        )
        self.view_requirements[EXPECTED_OWN_REWARDS] = ViewRequirement(
            space=spaces.Box(low=-np.inf, high=np.inf, shape=())
        )
        self.view_requirements[VALUE_ESTIMATES] = ViewRequirement(
            space=spaces.Box(low=-np.inf, high=np.inf, shape=())
        )

        EntropyCoeffSchedule.__init__(
            self, config["entropy_coeff"], config["entropy_coeff_schedule"]
        )
        LearningRateSchedule.__init__(
            self,
            config["lr"],
            config["lr_schedule"],
        )

    def set_training(self, training: bool):
        self._training = training

        if not self._training:
            # Set global timestep to a huge value so that we get whatever the reward
            # shaping schedule is at the end of training.
            self.global_timestep_for_envs = 2**63 - 1
        else:
            self.global_timestep_for_envs = getattr(self, "global_timestep", 0)

    def compute_actions(
        self,
        obs_batch: Union[List[TensorStructType], TensorStructType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Optional[
            Union[List[TensorStructType], TensorStructType]
        ] = None,
        prev_reward_batch: Optional[
            Union[List[TensorStructType], TensorStructType]
        ] = None,
        info_batch: Optional[Dict[str, list]] = None,
        episodes: Optional[List[Episode]] = None,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        *,
        force_noop=False,
        **kwargs,
    ):
        input_dict = {"obs": obs_batch}
        if prev_action_batch is not None:
            input_dict["prev_actions"] = prev_action_batch
        if prev_reward_batch is not None:
            input_dict["prev_rewards"] = prev_reward_batch
        for state_index, state_batch in enumerate(state_batches or []):
            input_dict[f"state_in_{state_index}"] = state_batch

        return self.compute_actions_from_input_dict(
            input_dict=input_dict,
            episodes=episodes,
            state_batches=state_batches,
            force_noop=force_noop,
        )

    def _run_model_on_input_dict(self, input_dict):
        input_dict = self._lazy_tensor_dict(input_dict)
        state_batches = [
            input_dict[k] for k in input_dict.keys() if "state_in" in k[:8]
        ]
        seq_lens = (
            torch.tensor(
                [1] * len(state_batches[0]),
                dtype=torch.long,
                device=state_batches[0].device,
            )
            if state_batches
            else None
        )
        assert self.model is not None
        return self.model(input_dict, state_batches, cast(torch.Tensor, seq_lens))

    def _ensure_enough_envs(self, num_envs: int):
        while len(self.envs) < num_envs:
            env = self.env_creator()
            env.reset()
            self.envs.append(env)

    def _compute_actions_with_mcts(
        self,
        input_dict,
        obs,
    ) -> Tuple[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]:
        num_envs = obs[0].shape[0]
        model_state_len = sum(k.startswith("state_in") for k in input_dict.keys())

        if self.config.get("player_index") is not None:
            for env in self.envs:
                env.set_player_index(self.config["player_index"])
        else:
            for env_index, player_index in enumerate(
                input_dict[SampleBatch.AGENT_INDEX]
            ):
                self.envs[env_index].set_player_index(player_index)

        nodes: List[MbagMCTSNode] = []
        for env_index in range(num_envs):
            env_obs = tuple(obs_piece[env_index] for obs_piece in obs)
            env_state = self.envs[env_index].set_state_from_obs(env_obs)
            model_state = [
                input_dict[f"state_in_{state_index}"][env_index]
                for state_index in range(model_state_len)
            ]
            nodes.append(
                MbagMCTSNode(
                    state=env_state,
                    obs=env_obs,
                    reward=0,
                    done=False,
                    info=None,
                    action=None,
                    parent=MbagRootParentNode(env=self.envs[env_index]),
                    model_state_in=model_state,
                    mcts=self.mcts,
                )
            )

        mcts_policies, actions = self.mcts.compute_actions(nodes)

        expected_rewards_list: List[float] = []
        expected_own_rewards_list: List[float] = []
        for node, action in zip(nodes, actions):
            if self.mcts.num_sims > 1:
                expected_reward, expected_own_reward = node.get_expected_rewards(action)
            else:
                expected_reward, expected_own_reward = 0, 0
            expected_rewards_list.append(expected_reward)
            expected_own_rewards_list.append(expected_own_reward)
        expected_rewards = np.array(expected_rewards_list)
        expected_own_rewards = np.array(expected_own_rewards_list)

        state_out = []
        for state_index in range(model_state_len):
            state_out.append(
                np.stack(
                    [node.model_state_out[state_index].cpu().numpy() for node in nodes],
                    axis=0,
                )
            )

        action_mask = np.stack([node.valid_actions for node in nodes], axis=0)

        value_estimates = np.array([node.value_estimate for node in nodes])

        extra_out = {
            ACTION_MASK: action_mask,
            MCTS_POLICIES: mcts_policies,
            EXPECTED_REWARDS: expected_rewards,
            EXPECTED_OWN_REWARDS: expected_own_rewards,
            VALUE_ESTIMATES: value_estimates,
        }

        return actions, state_out, extra_out

    def _compute_actions_noop(
        self,
        input_dict,
        obs,
    ) -> Tuple[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]:
        assert isinstance(self.action_space, spaces.Discrete)
        num_envs = obs[0].shape[0]

        actions = np.zeros(num_envs, dtype=int)
        expected_rewards = np.zeros(num_envs)
        expected_own_rewards = np.zeros(num_envs)
        mcts_policies = np.zeros((num_envs, self.action_space.n))
        mcts_policies[:, 0] = 1

        # Get action mask.
        assert isinstance(self.model, MbagTorchModel)
        action_mask = self.envs[0].get_valid_actions(obs, is_batch=True)
        input_dict[ACTION_MASK] = action_mask

        # Run inputs through the model to get state_out and value function.
        _, state_out_torch = self._run_model_on_input_dict(input_dict)
        state_out = convert_to_numpy(state_out_torch)
        value_estimates = convert_to_numpy(self.model.value_function())

        return (
            actions,
            state_out,
            {
                ACTION_MASK: action_mask,
                MCTS_POLICIES: mcts_policies,
                EXPECTED_REWARDS: expected_rewards,
                EXPECTED_OWN_REWARDS: expected_own_rewards,
                VALUE_ESTIMATES: value_estimates,
            },
        )

    def _check_expected_rewards_and_store_in_episodes(
        self,
        episodes: List[Episode],
        compute_actions_extra_out: Dict[str, np.ndarray],
        prev_rewards: np.ndarray,
    ):
        expected_rewards = compute_actions_extra_out[EXPECTED_REWARDS]
        expected_own_rewards = compute_actions_extra_out[EXPECTED_OWN_REWARDS]

        for env_index, episode in enumerate(episodes):
            player_index = self.envs[env_index].player_index

            if (
                self.config.get("_strict_mode", False)
                and self._training
                and not (
                    self.config["use_goal_predictor"]
                    or self.envs[env_index].config["num_players"] > 1
                )
            ):
                # If there was an expected reward, make sure it matches the actual
                # reward given by the environment so we're not out of sync.
                episode_expected_rewards: Dict[int, float] = episode.user_data.get(
                    EXPECTED_REWARDS, {}
                )
                prev_expected_reward = episode_expected_rewards.get(player_index)
                if prev_expected_reward is not None:
                    assert np.isclose(
                        prev_rewards[env_index],
                        prev_expected_reward,
                    )

            episode_expected_rewards = episode.user_data.setdefault(
                EXPECTED_REWARDS, {}
            )
            episode_expected_rewards[player_index] = expected_rewards[env_index]
            episode_expected_own_rewards = episode.user_data.setdefault(
                EXPECTED_OWN_REWARDS, {}
            )
            episode_expected_own_rewards[player_index] = expected_own_rewards[env_index]

    def compute_actions_from_input_dict(
        self,
        input_dict,
        explore=None,
        timestep=None,
        episodes=None,
        force_noop=False,
        **kwargs,
    ):
        if logger.isEnabledFor(logging.DEBUG):
            if episodes is not None:
                info: MbagInfoDict = episodes[0].last_info_for("player_1")
                if info is not None:
                    reward = input_dict[SampleBatch.REWARDS][0]
                    own_reward = info["own_reward"]
                    goal_similarity = info["goal_similarity"]
                    logger.debug(f"{reward=} {own_reward=} {goal_similarity=}")

        assert self.mcts.model == self.model
        cast(nn.Module, self.model).eval()

        obs = input_dict[SampleBatch.OBS]
        obs = restore_original_dimensions(obs, self.obs_space, "numpy")

        num_envs = obs[0].shape[0]
        self._ensure_enough_envs(num_envs)

        with torch.no_grad():
            if self.config["pretrain"] or force_noop:
                actions, state_out, compute_actions_extra_out = (
                    self._compute_actions_noop(input_dict, obs)
                )
            else:
                actions, state_out, compute_actions_extra_out = (
                    self._compute_actions_with_mcts(
                        input_dict,
                        obs,
                    )
                )

        if episodes is not None:
            self._check_expected_rewards_and_store_in_episodes(
                episodes, compute_actions_extra_out, input_dict[SampleBatch.REWARDS]
            )

        action_mask = compute_actions_extra_out[ACTION_MASK]
        mcts_policies = compute_actions_extra_out[MCTS_POLICIES]
        action_dist_inputs = np.log(mcts_policies)
        action_dist_inputs[mcts_policies == 0] = MbagTorchModel.MASK_LOGIT
        extra_out = {
            **self.extra_action_out(
                input_dict,
                kwargs.get("state_batches", []),
                self.model,
                cast(Any, None),
            ),
            ACTION_MASK: action_mask,
            MCTS_POLICIES: mcts_policies,
            SampleBatch.ACTION_DIST_INPUTS: action_dist_inputs,
            EXPECTED_REWARDS: compute_actions_extra_out[EXPECTED_REWARDS],
            EXPECTED_OWN_REWARDS: compute_actions_extra_out[EXPECTED_OWN_REWARDS],
            VALUE_ESTIMATES: compute_actions_extra_out[VALUE_ESTIMATES],
        }

        return np.array(actions), state_out, extra_out

    def postprocess_trajectory(
        self,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[
            Dict[AgentID, Tuple[PolicyID, Type[TorchPolicy], SampleBatch]]
        ] = None,
        episode: Optional[Episode] = None,
    ):
        with torch.no_grad():
            last_r: float
            if sample_batch[SampleBatch.TERMINATEDS][-1]:
                last_r = 0
            else:
                input_dict = sample_batch.get_single_step_input_dict(
                    self.view_requirements, index="last"
                )
                input_dict = SampleBatch(input_dict)
                assert self.model is not None
                self._run_model_on_input_dict(input_dict)
                last_r = self.model.value_function()[0].item()
            rewards_plus_v = np.concatenate(
                [sample_batch[SampleBatch.REWARDS], np.array([last_r])]
            )
            discounted_returns = discount_cumsum(rewards_plus_v, self.config["gamma"])[
                :-1
            ].astype(np.float32)
            sample_batch[Postprocessing.VALUE_TARGETS] = discounted_returns

        if other_agent_batches is not None:
            if len(other_agent_batches) > 1:
                raise RuntimeError(
                    "Training with multiple other agents is not supported."
                )
            elif len(other_agent_batches) == 1:
                other_agent_id, (_, _, other_agent_batch) = next(
                    iter(other_agent_batches.items())
                )
                if SampleBatch.ACTION_DIST_INPUTS in other_agent_batch:
                    sample_batch[OTHER_AGENT_ACTION_DIST_INPUTS] = other_agent_batch[
                        SampleBatch.ACTION_DIST_INPUTS
                    ]
                else:
                    logger.warn(
                        f"no action_dist_inputs in sample batch for {other_agent_id}"
                    )
            else:
                pass  # No need to include other agent batch for single player case.

        assert episode is not None
        infos: List[MbagInfoDict] = list(sample_batch[SampleBatch.INFOS][1:])
        agent_index = sample_batch[SampleBatch.AGENT_INDEX][0]
        agent_id = episode.get_agents()[agent_index]
        last_info = cast(MbagInfoDict, episode.last_info_for(agent_id))
        infos.append(last_info)
        sample_batch[OWN_REWARDS] = np.array([info["own_reward"] for info in infos])

        # Remove state_out_* entries from the sample batch since they aren't needed
        # for training and they take up a lot of space.
        for key in list(sample_batch.keys()):
            if key.startswith("state_out_"):
                del sample_batch[key]

        return sample_batch

    def learn_on_batch(self, postprocessed_batch):
        return TorchPolicy.learn_on_batch(self, postprocessed_batch)

    def loss(
        self,
        model: ModelV2,
        dist_class: Type[ActionDistribution],
        train_batch: SampleBatch,
    ) -> Union[TensorType, List[TensorType]]:
        assert isinstance(model, MbagTorchModel)

        # Forward pass in model.
        logits, state = model(train_batch)
        values = model.value_function()
        logits, values = torch.squeeze(logits), torch.squeeze(values)
        dist = dist_class(logits, model=model)
        assert isinstance(dist, TorchCategorical)

        # RNN case: Mask away 0-padded chunks at end of time axis.
        if state:
            B = len(train_batch[SampleBatch.SEQ_LENS])  # noqa: N806
            max_seq_len = logits.shape[0] // B
            mask = sequence_mask(
                train_batch[SampleBatch.SEQ_LENS],
                max_seq_len,
                time_major=model.is_time_major(),
            )
            assert isinstance(mask, torch.Tensor)
            mask = torch.reshape(mask, [-1])
            num_valid = torch.sum(mask)

            def reduce_mean_valid(t):
                return torch.sum(t[mask]) / num_valid

        # non-RNN case: No masking.
        else:
            mask = None
            reduce_mean_valid = torch.mean

        # Compute actor and critic losses.
        policy_loss = reduce_mean_valid(
            -torch.sum(train_batch[MCTS_POLICIES] * dist.dist.logits, dim=-1)
        )
        value_loss = reduce_mean_valid(
            (values - train_batch[Postprocessing.VALUE_TARGETS]) ** 2
        )

        entropy = reduce_mean_valid(dist.entropy())

        # Compute goal prediction loss.
        goal_logits = model.goal_predictor()
        world_obs, _, _ = restore_original_dimensions(
            train_batch[SampleBatch.OBS],
            obs_space=self.observation_space,
            tensorlib=torch,
        )
        goal = world_obs[:, GOAL_BLOCKS].long()
        ce = nn.CrossEntropyLoss(reduction="none")
        goal_ce: torch.Tensor = ce(goal_logits, goal)
        goal_loss = reduce_mean_valid(goal_ce.flatten(start_dim=1).mean(dim=1))

        unplaced_blocks = (goal != MinecraftBlocks.AIR) & (
            world_obs[:, CURRENT_BLOCKS] == MinecraftBlocks.AIR
        )
        unplaced_blocks_goal_loss = goal_ce[unplaced_blocks].mean()

        # Compute total loss.
        total_loss: torch.Tensor = (
            self.config["vf_loss_coeff"] * value_loss
            + self.config["goal_loss_coeff"] * goal_loss
            - self.entropy_coeff * entropy
        )
        if not self.config["pretrain"]:
            total_loss = total_loss + policy_loss

        if isinstance(model, OtherAgentActionPredictorMixin):
            # Compute other agent action prediction loss.
            predicted_other_agent_action_dist = dist_class(
                cast(Any, model.predict_other_agent_action()),
                model=model,
            )
            other_agent_action_dist_inputs = train_batch[OTHER_AGENT_ACTION_DIST_INPUTS]
            # Get rid of -inf action dist inputs to avoid numeric issues with
            # KL divergence.
            other_agent_action_dist_inputs[
                other_agent_action_dist_inputs == -np.inf
            ] = -1e4
            actual_other_agent_action_dist = dist_class(
                other_agent_action_dist_inputs,  # type: ignore
                model=model,
            )
            other_agent_action_predictor_loss = reduce_mean_valid(
                actual_other_agent_action_dist.kl(predicted_other_agent_action_dist)
            )

            model.tower_stats["other_agent_action_predictor_loss"] = (
                other_agent_action_predictor_loss
            )
            total_loss = (
                total_loss
                + self.config["other_agent_action_predictor_loss_coeff"]
                * other_agent_action_predictor_loss
            )

        model.tower_stats["total_loss"] = total_loss
        model.tower_stats["policy_loss"] = policy_loss
        model.tower_stats["vf_loss"] = value_loss
        model.tower_stats["vf_explained_var"] = explained_variance(
            train_batch[Postprocessing.VALUE_TARGETS], values
        )
        model.tower_stats["goal_loss"] = goal_loss
        model.tower_stats["unplaced_blocks_goal_loss"] = unplaced_blocks_goal_loss
        model.tower_stats["entropy"] = entropy

        return total_loss

    def extra_grad_process(self, optimizer, loss):
        return apply_grad_clipping(self, optimizer, loss)

    def extra_grad_info(self, train_batch: SampleBatch):
        grad_info: Dict[str, TensorType] = {
            "entropy_coeff": self.entropy_coeff,
            "cur_lr": self.cur_lr,
            "mcts/temperature": self.mcts.temperature,
        }
        for metric in [
            "total_loss",
            "policy_loss",
            "vf_loss",
            "vf_explained_var",
            "goal_loss",
            "unplaced_blocks_goal_loss",
            "entropy",
            "other_agent_action_predictor_loss",
        ]:
            try:
                grad_info[metric] = torch.mean(
                    torch.stack(cast(List[torch.Tensor], self.get_tower_stats(metric)))
                )
            except AssertionError:
                pass
        return convert_to_numpy(grad_info)

    def on_global_var_update(self, global_vars):
        super().on_global_var_update(global_vars)

        if self._training:
            self.global_timestep_for_envs = global_vars["timestep"]
        for env in self.envs:
            unwrap_mbag_env(env).update_global_timestep(self.global_timestep_for_envs)
