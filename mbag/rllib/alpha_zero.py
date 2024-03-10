import copy
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Type, Union, cast

import numpy as np
import torch
from gymnasium import spaces
from ray.rllib.algorithms.algorithm_config import NotProvided
from ray.rllib.algorithms.alpha_zero.alpha_zero import AlphaZero, AlphaZeroConfig
from ray.rllib.algorithms.alpha_zero.alpha_zero_policy import AlphaZeroPolicy
from ray.rllib.algorithms.alpha_zero.mcts import MCTS, Node, RootParentNode
from ray.rllib.evaluation import SampleBatch
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.postprocessing import Postprocessing, discount_cumsum
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.execution.train_ops import multi_gpu_train_one_step, train_one_step
from ray.rllib.models import ActionDistribution, ModelV2
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import concat_samples
from ray.rllib.policy.torch_mixins import EntropyCoeffSchedule, LearningRateSchedule
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.metrics import (
    NUM_AGENT_STEPS_SAMPLED,
    NUM_ENV_STEPS_SAMPLED,
    SAMPLE_TIMER,
    SYNCH_WORKER_WEIGHTS_TIMER,
)
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import (
    apply_grad_clipping,
    explained_variance,
    sequence_mask,
)
from ray.rllib.utils.typing import AgentID, PolicyID, ResultDict, TensorType
from ray.tune.registry import ENV_CREATOR, _global_registry, register_trainable
from torch import nn

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.environment.actions import MbagAction, MbagActionTuple
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.schedule import PiecewiseSchedule, Schedule
from mbag.environment.types import CURRENT_BLOCKS, GOAL_BLOCKS

from .planning import MbagEnvModel, MbagEnvModelInfoDict
from .rllib_env import unwrap_mbag_env
from .torch_models import ACTION_MASK, MbagTorchModel, OtherAgentActionPredictorMixin

MCTS_POLICIES = "mcts_policies"
OTHER_AGENT_ACTION_DIST_INPUTS = "other_agent_action_dist_inputs"
EXPECTED_REWARDS = "expected_rewards"


logger = logging.getLogger(__name__)


class MbagRootParentNode(RootParentNode):
    def __init__(self, env):
        super().__init__(env)
        self.child_number_visits = defaultdict(int)

        self.action_mapping = MbagActionDistribution.get_action_mapping(
            unwrap_mbag_env(self.env).config
        )
        self.action_type_slices: List[slice] = []
        for action_type in range(MbagAction.NUM_ACTION_TYPES):
            (mask,) = np.nonzero(self.action_mapping[:, 0] == action_type)
            action_type_slice = (
                slice(mask[0], mask[-1] + 1) if len(mask) > 0 else slice(0, 0)
            )
            self.action_type_slices.append(action_type_slice)
            assert len(mask) == action_type_slice.stop - action_type_slice.start


class MbagMCTSNode(Node):
    env: MbagEnvModel
    mcts: "MbagMCTS"
    children: Dict[Tuple[int, ...], "MbagMCTSNode"]
    action_mapping: np.ndarray
    action_type_slices: List[slice]
    parent: Union["MbagMCTSNode", MbagRootParentNode]
    goal_logits: Optional[np.ndarray]
    other_agent_action_dist: Optional[np.ndarray]
    other_reward: float
    model_state_in: List[np.ndarray]
    model_state_out: List[np.ndarray]

    def __init__(
        self,
        action,
        obs,
        done,
        info: Optional[MbagEnvModelInfoDict],
        reward,
        state,
        mcts,
        model_state_in,
        parent=None,
    ):
        super().__init__(action, obs, done, reward, state, mcts, parent)

        self.info = info

        self.min_value = np.inf
        self.max_value = -np.inf
        self.action_mapping = self.parent.action_mapping
        self.action_type_slices = self.parent.action_type_slices

        self.valid_action_types = np.array(
            [
                np.any(self.valid_actions[self.action_type_slices[action_type]])
                for action_type in range(MbagAction.NUM_ACTION_TYPES)
            ]
        )
        self.action_type_total_value = np.zeros(
            MbagAction.NUM_ACTION_TYPES, dtype=np.float32
        )
        self.action_type_number_visits = np.zeros(
            MbagAction.NUM_ACTION_TYPES, dtype=np.int64
        )

        self.child_number_visits = self.child_number_visits.astype(np.int64)

        self.action_type_dirichlet_noise = None
        self.dirichlet_noise = None

        self.goal_logits = None
        self.other_agent_action_dist = None

        self.model_state_in = model_state_in

    def child_Q(self, mask=slice(None)):  # noqa: N802
        Q = self.child_total_value[mask] / np.maximum(  # noqa: N806
            self.child_number_visits[mask], 1
        )

        if self.mcts.init_q_with_max:
            init_q_value = self.max_value
        else:
            if self.mcts.fix_bilevel_action_selection:
                total_visits = np.sum(self.child_number_visits[mask])
                init_q_value = (
                    np.sum(self.child_total_value[mask]) / total_visits
                    if total_visits > 0
                    else self.min_value
                )
            else:
                V = (  # noqa: N806
                    self.total_value / self.number_visits
                    if self.number_visits > 0
                    else self.min_value
                )
                init_q_value = V

        Q[self.child_number_visits[mask] == 0] = init_q_value
        Q = (Q - self.min_value) / max(  # noqa: N806
            self.max_value - self.min_value, 0.01
        )
        return Q

    def child_U(self, mask=slice(None)):  # noqa: N802
        if self.mcts.fix_bilevel_action_selection:
            number_visits = max(np.sum(self.child_number_visits[mask]), 1)
        else:
            number_visits = self.number_visits

        child_priors = self.child_priors[mask]
        if self.mcts.fix_bilevel_action_selection:
            child_priors = child_priors / child_priors.sum()

        return (
            np.sqrt(number_visits) * child_priors / (1 + self.child_number_visits[mask])
        )

    def action_type_Q(self):  # noqa: N802
        total_value = self.action_type_total_value
        number_visits = self.action_type_number_visits
        Q = total_value / np.maximum(number_visits, 1)  # noqa: N806
        V = (  # noqa: N806
            self.total_value / self.number_visits if self.number_visits > 0 else 0
        )
        Q[number_visits == 0] = self.max_value if self.mcts.init_q_with_max else V
        Q = (Q - self.min_value) / max(  # noqa: N806
            self.max_value - self.min_value, 0.01
        )
        return Q

    def action_type_U(self):  # noqa: N802
        return (
            np.sqrt(self.number_visits)
            * self.action_type_priors
            / (1 + self.action_type_number_visits)
        )

    def best_action(self, force_python_impl=False) -> int:
        if self.mcts._strict_mode:
            force_python_impl = True

        if self.mcts.init_q_with_max:
            init_q_value = self.max_value
        else:
            init_q_value = (
                self.total_value / self.number_visits if self.number_visits > 0 else 0
            )

        if self.mcts.use_bilevel_action_selection:
            return self._best_action_bilevel(
                init_q_value=init_q_value,
                force_python_impl=force_python_impl,
            )
        else:
            action: int
            action_c = None
            try:
                import _mbag

                action_c = _mbag.mcts_best_action(
                    self.child_total_value,
                    self.child_number_visits,
                    self.child_priors,
                    self.number_visits,
                    self.mcts.c_puct,
                    init_q_value,
                    self.max_value,
                    self.min_value,
                    np.nonzero(self.valid_actions)[0],
                )
            except ImportError:
                if not force_python_impl:
                    logger.warning("C implementation of best_action not found")

            if force_python_impl or action_c is None:
                action = super().best_action()
                assert action == action_c
            else:
                action = action_c

            return action

    def _best_action_bilevel(self, init_q_value: float, force_python_impl: bool) -> int:
        action_type_c = None
        try:
            import _mbag

            action_type_c = _mbag.mcts_best_action(
                self.action_type_total_value,
                self.action_type_number_visits,
                self.action_type_priors,
                self.number_visits,
                self.mcts.c_puct,
                init_q_value,
                self.max_value,
                self.min_value,
                np.nonzero(self.valid_action_types)[0],
            )
        except ImportError:
            if not force_python_impl:
                logger.warning("C implementation of best_action not found")

        if force_python_impl or action_type_c is None:
            action_type_score = (
                self.action_type_Q() + self.mcts.c_puct * self.action_type_U()
            )
            action_type_score[~self.valid_action_types] = -np.inf
            action_type = int(np.argmax(action_type_score))
            assert action_type == action_type_c
        else:
            action_type = action_type_c

        action_type_slice = self.action_type_slices[action_type]
        valid_action_indices = (
            np.nonzero(self.valid_actions[action_type_slice])[0]
            + action_type_slice.start
        )

        action_c = None
        try:
            import _mbag

            if self.mcts.fix_bilevel_action_selection:
                number_visits = max(self.action_type_number_visits[action_type], 1)
                if not self.mcts.init_q_with_max:
                    init_q_value = (
                        self.action_type_total_value[action_type] / number_visits
                        if number_visits > 0
                        else self.min_value
                    )
                prior_scale = 1.0 / self.action_type_priors[action_type]
            else:
                number_visits = self.number_visits
                prior_scale = 1.0

            action_c = _mbag.mcts_best_action(
                self.child_total_value,
                self.child_number_visits,
                self.child_priors,
                number_visits,
                self.mcts.c_puct,
                init_q_value,
                self.max_value,
                self.min_value,
                valid_action_indices,
                prior_scale=prior_scale,
            )
        except ImportError:
            if not force_python_impl:
                logger.warning("C implementation of best_action not found")

        if force_python_impl or action_c is None:
            if len(valid_action_indices) == 1:
                action = int(valid_action_indices[0])
                assert action == action_c
            else:
                child_score = self.child_Q(
                    valid_action_indices
                ) + self.mcts.c_puct * self.child_U(valid_action_indices)
                action = int(valid_action_indices[np.argmax(child_score)])
                assert (
                    child_score[np.where(valid_action_indices == action_c)[0][0]]
                    >= child_score[np.where(valid_action_indices == action)[0][0]]
                    - 1e-4
                )
        else:
            action = action_c

        return action

    def expand(
        self,
        child_priors,
        goal_logits: Optional[np.ndarray] = None,
        other_agent_action_dist: Optional[np.ndarray] = None,
        model_state_out: List[np.ndarray] = [],
        add_dirichlet_noise=False,
    ) -> None:
        super().expand(child_priors)

        self.model_state_out = model_state_out
        self.goal_logits = goal_logits
        self.other_agent_action_dist = other_agent_action_dist

        if (
            self.info is not None
            and self.goal_logits is not None
            and isinstance(self.parent, MbagMCTSNode)
        ):
            # We need to update the reward for this node based on the new goal_logits.
            self.reward = self.env.get_reward_with_other_agent_actions(
                self.parent.obs["obs"],
                self.info,
                self.goal_logits,
            )

        self.child_priors[~self.valid_actions] = 0
        self.child_priors /= self.child_priors.sum()

        if add_dirichlet_noise:
            num_action_types = self.action_mapping[-1, 0] + 1
            type_masks = np.empty(
                (num_action_types, self.valid_actions.shape[0]), dtype=bool
            )
            for action_type in range(num_action_types):
                type_masks[action_type] = (
                    self.action_mapping[:, 0] == action_type
                ) & self.valid_actions
            valid_action_types = np.any(type_masks, axis=1)

            action_type_dirichlet_noise = np.random.dirichlet(
                np.full(num_action_types, self.mcts.dir_noise)
            )
            action_type_dirichlet_noise[~valid_action_types] = 0
            action_type_dirichlet_noise /= action_type_dirichlet_noise.sum()

            for action_type in range(num_action_types):
                type_mask = type_masks[action_type]
                if not np.any(type_mask):
                    continue

                self.child_priors[type_mask] *= (
                    (1 - self.mcts.dir_epsilon) * self.child_priors[type_mask].sum()
                    + self.mcts.dir_epsilon * action_type_dirichlet_noise[action_type]
                ) / self.child_priors[type_mask].sum()

                num_valid_actions = type_mask.astype(int).sum()
                alpha = (
                    self.mcts.dirichlet_action_subtype_noise_multiplier
                    / num_valid_actions
                )
                dirichlet_noise = np.random.dirichlet(np.full(num_valid_actions, alpha))
                self.child_priors[type_mask] = (
                    1 - self.mcts.dir_epsilon
                ) * self.child_priors[
                    type_mask
                ] + self.mcts.dir_epsilon * self.child_priors[
                    type_mask
                ].sum() * dirichlet_noise

            assert abs(self.child_priors.sum() - 1) < 1e-2

        self.action_type_priors = np.array(
            [
                np.sum(self.child_priors[self.action_type_slices[action_type]])
                for action_type in range(MbagAction.NUM_ACTION_TYPES)
            ],
            dtype=np.float32,
        )

    def get_child(self, action: int) -> "MbagMCTSNode":
        all_actions: Tuple[int, ...] = (action,)
        other_agent_actions: Optional[List[int]] = None
        if self.other_agent_action_dist is not None:
            other_agent_action = np.random.choice(
                np.arange(self.action_space_size), p=self.other_agent_action_dist
            )
            all_actions = action, other_agent_action
            other_agent_actions = [other_agent_action]

        if all_actions not in self.children:
            self.env.set_state(self.state)
            obs, reward, terminated, truncated, info = self.env.step(
                action,
                goal_logits=self.goal_logits,
                other_player_actions=other_agent_actions,
            )
            next_state = self.env.get_state()
            self.children[all_actions] = MbagMCTSNode(
                state=next_state,
                action=action,
                parent=self,
                reward=reward,
                done=terminated,
                info=info,
                obs=obs,
                mcts=self.mcts,
                model_state_in=self.model_state_out,
            )
        return self.children[all_actions]

    def backup(self, value):
        current = self
        value = float(value)
        while True:
            value *= self.mcts.gamma
            value += current.reward
            current.number_visits += 1
            current.total_value += value

            if current.action is not None:
                assert isinstance(current.parent, MbagMCTSNode)
                action_type = int(self.action_mapping[current.action, 0])
                current.parent.action_type_number_visits[action_type] += 1
                current.parent.action_type_total_value[action_type] += value

            for node in [current, current.parent]:
                if isinstance(node, MbagMCTSNode):
                    node.min_value = min(node.min_value, value)
                    node.max_value = max(node.max_value, value)
            if isinstance(current.parent, MbagMCTSNode):
                current = current.parent
            else:
                break

    def get_mbag_action(self, flat_action: int) -> MbagAction:
        return MbagAction(
            cast(MbagActionTuple, tuple(self.action_mapping[flat_action])),
            self.env.config["world_size"],
        )


class MbagMCTS(MCTS):
    _temperature_schedule: Optional[Schedule]

    def __init__(
        self,
        model,
        mcts_param,
        gamma: float,
        use_critic=True,
        use_goal_predictor=True,
        _strict_mode=False,
    ):
        super().__init__(model, mcts_param)
        self.gamma = gamma  # Discount factor.
        self.use_critic = use_critic
        self.use_goal_predictor = use_goal_predictor
        self._strict_mode = _strict_mode

        self._temperature_schedule = None
        if mcts_param["temperature_schedule"] is not None:
            self._temperature_schedule = PiecewiseSchedule(
                mcts_param["temperature_schedule"],
                outside_value=mcts_param["temperature_schedule"][-1][-1],
            )
            self.temperature = self._temperature_schedule.value(0)

        self.prior_temperature = mcts_param.get("prior_temperature", 1.0)
        self.init_q_with_max = mcts_param.get("init_q_with_max", False)
        self.use_bilevel_action_selection = mcts_param.get(
            "use_bilevel_action_selection", False
        )
        self.dirichlet_action_subtype_noise_multiplier: float = mcts_param.get(
            "dirichlet_action_subtype_noise_multiplier", 10
        )

        # Previously, we used a version of bilevel action selection that wasn't
        # quite accurate. It used the number of visits to the whole state rather than
        # for the particular action type to select the action within an action type, as
        # well as initializing Q values based on the mean for the whole state rather
        # than the action type. We keep this option around to reproduce old results.
        self.fix_bilevel_action_selection = mcts_param.get(
            "fix_bilevel_action_selection", False
        )

    def update_temperature(self, global_timestep: int):
        if self._temperature_schedule is not None:
            self.temperature = self._temperature_schedule.value(global_timestep)

    def compute_action(self, node: MbagMCTSNode):
        tree_policies, actions, children = self.compute_actions([node])
        return tree_policies[0], actions[0], children[0]

    def _stack_obs(self, obs: list):
        if isinstance(obs[0], dict):
            stacked_obs = {}
            for key in obs[0].keys():
                stacked_obs[key] = self._stack_obs([o[key] for o in obs])
            return stacked_obs
        elif isinstance(obs[0], tuple):
            return tuple(
                self._stack_obs([o[i] for o in obs]) for i in range(len(obs[0]))
            )
        else:
            return np.stack(obs, axis=0)

    def compute_actions(
        self, nodes: List[MbagMCTSNode]
    ) -> Tuple[np.ndarray, np.ndarray, List[MbagMCTSNode]]:
        for _ in range(self.num_sims):
            leaves: List[MbagMCTSNode] = [node.select() for node in nodes]
            # obs = self._stack_obs([leaf.obs for leaf in leaves])
            obs = [leaf.obs["obs"] for leaf in leaves]
            model_state_len = len(leaves[0].model_state_in)
            model_state_in = [
                np.stack([leaf.model_state_in[state_index] for leaf in leaves], axis=0)
                for state_index in range(model_state_len)
            ]
            action_mask = np.stack([leaf.valid_actions for leaf in leaves])
            child_priors: np.ndarray
            values: np.ndarray
            model_state_out: List[np.ndarray]

            child_priors, values, model_state_out = self.model.compute_priors_and_value(
                obs, model_state_in, action_mask=action_mask
            )
            child_priors = child_priors**self.prior_temperature
            child_priors /= child_priors.sum(axis=1, keepdims=True)
            if not self.use_critic:
                values[:] = 0

            goal_logits: Optional[np.ndarray]
            if self.use_goal_predictor:
                goal_logits = convert_to_numpy(self.model.goal_predictor())
            else:
                goal_logits = None

            other_agent_action_dists: Optional[np.ndarray] = None
            if isinstance(self.model, OtherAgentActionPredictorMixin):
                other_agent_action_dists = convert_to_numpy(
                    self.model.predict_other_agent_action().softmax(1)
                )

            for env_index, leaf in enumerate(leaves):
                if leaf.done:
                    value = 0.0
                else:
                    value = float(values[env_index])
                    leaf.expand(
                        child_priors[env_index],
                        goal_logits=(
                            goal_logits[env_index] if goal_logits is not None else None
                        ),
                        other_agent_action_dist=(
                            other_agent_action_dists[env_index]
                            if other_agent_action_dists is not None
                            else None
                        ),
                        add_dirichlet_noise=self.add_dirichlet_noise
                        and leaf == nodes[env_index],
                        model_state_out=[state[env_index] for state in model_state_out],
                    )
                leaf.backup(value)

        # Tree policy target (TPT)
        tree_policies = np.stack(
            [node.child_number_visits / node.number_visits for node in nodes], axis=0
        )
        tree_policies = tree_policies / np.max(
            tree_policies, axis=1, keepdims=True
        )  # to avoid overflows with temperature scaling
        tree_policies = np.power(tree_policies, self.temperature)
        tree_policies = tree_policies / np.sum(tree_policies, axis=1, keepdims=True)

        if self.exploit:
            # if exploit then choose action that has the maximum
            # tree policy probability
            actions = np.argmax(tree_policies, axis=1)
        else:
            # otherwise sample an action according to tree policy probabilities
            actions = np.array(
                [
                    np.random.choice(np.arange(node.action_space_size), p=tree_policy)
                    for node, tree_policy in zip(nodes, tree_policies)
                ]
            )

        if logger.isEnabledFor(logging.DEBUG):
            node, action, tree_policy = nodes[0], actions[0], tree_policies[0]
            child = node.get_child(action)
            logger.debug(
                "\t".join(
                    map(
                        str,
                        [
                            node.get_mbag_action(action),
                            child.reward,
                            node.child_Q()[action],
                            node.child_U()[action],
                            node.child_priors[action],
                            child.number_visits,
                            tree_policy[action],
                            node.valid_actions.astype(int).sum(),
                        ],
                    )
                )
            )

            self.model.eval()
            _, values, _ = self.model.compute_priors_and_value(
                [node.obs["obs"]], [state[None] for state in node.model_state_in]
            )
            logger.debug(f"{values[0]} {node.total_value / node.number_visits}")

            plan = []
            current = node
            while len(current.children) > 0:
                plan_action = int(np.argmax(current.child_number_visits))
                current = current.get_child(plan_action)
                plan.append((node.get_mbag_action(plan_action), current.reward))
            logger.debug(plan)

        return (
            tree_policies,
            actions,
            [node.get_child(action) for node, action in zip(nodes, actions)],
        )


class MbagAlphaZeroPolicy(AlphaZeroPolicy, EntropyCoeffSchedule, LearningRateSchedule):
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

        def env_creator():
            env_creator = _global_registry.get(ENV_CREATOR, config["env"])
            # We should never use Malmo in the env model.
            env_config = copy.deepcopy(config["env_config"])
            env_config["malmo"]["use_malmo"] = False
            env = env_creator(env_config)
            env_model = MbagEnvModel(
                env, env_config, line_of_sight_masking=line_of_sight_masking
            )
            unwrap_mbag_env(env_model).update_global_timestep(
                self.global_timestep_for_envs
            )
            return env_model

        def mcts_creator():
            return MbagMCTS(
                self.model,
                config["mcts_config"],
                config["gamma"],
                use_critic=config["use_critic"],
                use_goal_predictor=config["use_goal_predictor"],
                _strict_mode=config.get("_strict_mode", False),
            )

        self.env_creator = env_creator
        self.mcts = mcts_creator()
        self.envs = []
        self.obs_space = observation_space

        self.view_requirements[ACTION_MASK] = ViewRequirement(
            space=spaces.MultiBinary(action_space.n)
        )
        self.view_requirements[SampleBatch.ACTION_DIST_INPUTS] = ViewRequirement(
            space=spaces.Box(low=-np.inf, high=np.inf, shape=(action_space.n,))
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
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
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
        )

    def compute_actions_from_input_dict(
        self, input_dict, explore=None, timestep=None, episodes=None, **kwargs
    ):
        assert self.mcts.model == self.model

        model_state_len = sum(k.startswith("state_in") for k in input_dict.keys())
        state_out = [
            np.empty_like(input_dict[f"state_in_{state_index}"])
            for state_index in range(model_state_len)
        ]
        if self.config["pretrain"]:
            for state_out_part in state_out:
                state_out_part[:] = 0

        while len(self.envs) < len(episodes):
            env = self.env_creator()
            env.reset()
            self.envs.append(env)

        if self.config.get("player_index") is not None:
            for env in self.envs:
                env.set_player_index(self.config["player_index"])
        else:
            for env_index, player_index in enumerate(
                input_dict[SampleBatch.AGENT_INDEX]
            ):
                self.envs[env_index].set_player_index(player_index)

        assert isinstance(self.action_space, spaces.Discrete)

        with torch.no_grad():
            episode: Episode

            if self.config["pretrain"]:
                actions = np.array([0] * len(episodes))
                expected_rewards = np.zeros(len(episodes))
                expected_own_rewards = np.zeros(len(episodes))
                mcts_policies = np.zeros((len(episodes), self.action_space.n))
                mcts_policies[:, 0] = 1

                # Get action mask.
                assert isinstance(self.model, MbagTorchModel)
                obs = input_dict[SampleBatch.OBS]
                obs = restore_original_dimensions(obs, self.obs_space, "numpy")
                action_mask = MbagActionDistribution.get_mask_flat(
                    self.config["env_config"],
                    obs,
                    line_of_sight_masking=self.model.line_of_sight_masking,
                )
            else:
                nodes: List[MbagMCTSNode] = []
                for env_index, episode in enumerate(episodes):
                    env_state = episode.user_data["state"]
                    model_state = [
                        input_dict[f"state_in_{state_index}"][env_index]
                        for state_index in range(model_state_len)
                    ]
                    obs = self.envs[env_index].set_state(env_state)
                    nodes.append(
                        MbagMCTSNode(
                            state=env_state,
                            obs=obs,
                            reward=0,
                            done=False,
                            info=None,
                            action=None,
                            parent=MbagRootParentNode(env=self.envs[env_index]),
                            model_state_in=model_state,
                            mcts=self.mcts,
                        )
                    )

                mcts_policies, actions, action_nodes = self.mcts.compute_actions(nodes)

                expected_rewards = np.array(
                    [action_node.reward for action_node in action_nodes]
                )
                expected_own_rewards = np.array(
                    [
                        (
                            action_node.info["own_reward"]
                            if action_node.info is not None
                            else np.nan
                        )
                        for action_node in action_nodes
                    ]
                )
                for state_index in range(model_state_len):
                    state_out[state_index] = np.stack(
                        [node.model_state_out[state_index] for node in nodes],
                        axis=0,
                    )

                action_mask = np.stack([node.valid_actions for node in nodes], axis=0)

            for env_index, episode in enumerate(episodes):
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
                    if EXPECTED_REWARDS in episode.user_data:
                        assert np.isclose(
                            input_dict[SampleBatch.REWARDS][env_index],
                            episode.user_data[EXPECTED_REWARDS],
                        )

                if episode.length == 0:
                    episode.user_data[MCTS_POLICIES] = []
                episode.user_data[MCTS_POLICIES].append(mcts_policies[env_index])
                episode.user_data[EXPECTED_REWARDS] = expected_rewards[env_index]

        return (
            np.array(actions),
            state_out,
            {
                **self.extra_action_out(
                    input_dict,
                    kwargs.get("state_batches", []),
                    self.model,
                    cast(Any, None),
                ),
                "expected_reward": expected_rewards,
                "expected_own_reward": expected_own_rewards,
                ACTION_MASK: action_mask,
                SampleBatch.ACTION_DIST_INPUTS: mcts_policies,
            },
        )

    def postprocess_trajectory(
        self,
        sample_batch: SampleBatch,
        other_agent_batches: Optional[
            Dict[AgentID, Tuple[PolicyID, Type[TorchPolicy], SampleBatch]]
        ] = None,
        episode=None,
    ):
        with torch.no_grad():
            input_dict = sample_batch.get_single_step_input_dict(
                self.view_requirements, index="last"
            )
            input_dict = SampleBatch(input_dict)
            input_dict = self._lazy_tensor_dict(input_dict)
            assert self.model is not None
            self.model(input_dict)
            last_r = self.model.value_function()[0].item()
            rewards_plus_v = np.concatenate(
                [sample_batch[SampleBatch.REWARDS], np.array([last_r])]
            )
            discounted_returns = discount_cumsum(rewards_plus_v, self.config["gamma"])[
                :-1
            ].astype(np.float32)
            sample_batch[Postprocessing.VALUE_TARGETS] = discounted_returns

        # Add MCTS policies to sample batch.
        sample_batch[MCTS_POLICIES] = np.array(episode.user_data[MCTS_POLICIES])[
            sample_batch[SampleBatch.T]
        ]

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
        if self._entropy_coeff_schedule is not None:
            self.entropy_coeff = self._entropy_coeff_schedule.value(
                global_vars["timestep"]
            )
        if self._lr_schedule is not None:
            self.cur_lr = self._lr_schedule.value(global_vars["timestep"])
            for opt in self._optimizers:
                for p in opt.param_groups:
                    p["lr"] = self.cur_lr
        self.mcts.update_temperature(global_timestep=global_vars["timestep"])

        if self._training:
            self.global_timestep_for_envs = global_vars["timestep"]
        for env in self.envs:
            unwrap_mbag_env(env).update_global_timestep(self.global_timestep_for_envs)


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

    def training_step(self) -> ResultDict:
        assert self.workers is not None

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

        # Update sampling step counters.
        self._counters[NUM_ENV_STEPS_SAMPLED] += new_sample_batch.env_steps()
        self._counters[NUM_AGENT_STEPS_SAMPLED] += new_sample_batch.agent_steps()
        # Store new samples in the replay buffer.
        if self.local_replay_buffer is not None:
            self.local_replay_buffer.add(new_sample_batch)

        if self.local_replay_buffer is not None:
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

        # Return all collected metrics for the iteration.
        return train_results


register_trainable("MbagAlphaZero", MbagAlphaZero)
