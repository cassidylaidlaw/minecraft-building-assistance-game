import logging
import numbers
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import torch
from ray.rllib.algorithms.alpha_zero.mcts import MCTS
from ray.rllib.utils.numpy import convert_to_numpy

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.environment.actions import MbagAction, MbagActionTuple
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.schedule import PiecewiseSchedule, Schedule

from ..rllib_env import unwrap_mbag_env
from ..torch_models import MbagTorchModel, OtherAgentActionPredictorMixin
from .planning import MbagEnvModel, MbagEnvModelInfoDict

logger = logging.getLogger(__name__)


class MbagRootParentNode:
    def __init__(self, env: MbagEnvModel):
        self.parent = None
        self.env = env
        self.action_total_value: Any = defaultdict(float)
        self.action_number_visits: Any = defaultdict(int)

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


class MbagMCTSNode:
    env: MbagEnvModel
    mcts: "MbagMCTS"
    c_puct: float
    children: Dict[Tuple[int, ...], "MbagMCTSNode"]
    action_mapping: np.ndarray
    action_type_slices: List[slice]
    parent: Union["MbagMCTSNode", MbagRootParentNode]
    goal_logits: Optional[np.ndarray]
    other_agent_action_dist: Optional[np.ndarray]
    other_reward: float
    model_state_in: List[torch.Tensor]
    model_state_out: List[torch.Tensor]
    action_type_priors: np.ndarray
    noop_probability: float

    def __init__(
        self,
        action,
        obs,
        done: bool,
        info: Optional[MbagEnvModelInfoDict],
        reward,
        state,
        mcts: "MbagMCTS",
        model_state_in: Union[List[np.ndarray], List[torch.Tensor]],
        parent: Union["MbagMCTSNode", MbagRootParentNode],
    ):
        self.env = parent.env
        self.action = action  # Action used to go to this state

        self.is_expanded = False
        self.parent = parent
        self.children = {}

        self.reward = reward
        self.info = info
        self.done = done
        self.state = state
        self.obs = obs

        self.mcts = mcts

        if isinstance(self.parent, MbagRootParentNode):
            if isinstance(self.mcts.c_puct, numbers.Real):
                self.c_puct = self.mcts.c_puct
            else:
                # Implement DiL-piKL by randomly choosing a c_puct value from the list.
                self.c_puct = np.random.choice(self.mcts.c_puct)
        else:
            self.c_puct = self.parent.c_puct

        self.min_value = np.inf
        self.max_value = -np.inf
        self.action_mapping = self.parent.action_mapping
        self.action_type_slices = self.parent.action_type_slices

        self.action_space_size = self.env.action_space.n
        self.action_total_value = np.zeros(
            [self.action_space_size], dtype=np.float32
        )  # Q
        self.child_priors = np.zeros([self.action_space_size], dtype=np.float32)  # P
        self.action_number_visits = np.zeros(
            [self.action_space_size], dtype=np.int64
        )  # N

        self.valid_actions = self.env.get_valid_actions(self.obs)
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

        self.action_type_dirichlet_noise = None
        self.dirichlet_noise = None

        self.goal_logits = None
        self.other_agent_action_dist = None

        self._total_value = 0.0
        self._number_visits = 0

        assert isinstance(self.mcts.model, MbagTorchModel)
        if model_state_in:
            tensor_model_state_in = [
                torch.from_numpy(state) if isinstance(state, np.ndarray) else state
                for state in model_state_in
            ]
            self.model_state_in = [
                state.to(self.mcts.model.device) for state in tensor_model_state_in
            ]
        else:
            self.model_state_in = []

    @property
    def total_value(self):
        return self._total_value

    @total_value.setter
    def total_value(self, value):
        raise RuntimeError()

    @property
    def number_visits(self):
        return self._number_visits

    @number_visits.setter
    def number_visits(self, value):
        raise RuntimeError()

    def child_Q(self, mask=slice(None)):  # noqa: N802
        Q = self.action_total_value[mask] / np.maximum(  # noqa: N806
            self.action_number_visits[mask], 1
        )

        if self.mcts.init_q_with_max:
            init_q_value = self.max_value
        else:
            if self.mcts.fix_bilevel_action_selection:
                total_visits = np.sum(self.action_number_visits[mask])
                init_q_value = (
                    np.sum(self.action_total_value[mask]) / total_visits
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

        Q[self.action_number_visits[mask] == 0] = init_q_value
        Q = (Q - self.min_value) / max(  # noqa: N806
            self.max_value - self.min_value, 0.01
        )
        return Q

    def child_U(self, mask=slice(None)):  # noqa: N802
        if self.mcts.fix_bilevel_action_selection:
            number_visits = max(np.sum(self.action_number_visits[mask]), 1)
        else:
            number_visits = self.number_visits

        child_priors = self.child_priors[mask]
        if self.mcts.fix_bilevel_action_selection:
            child_priors = child_priors / child_priors.sum()

        return (
            np.sqrt(number_visits)
            * child_priors
            / (1 + self.action_number_visits[mask])
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
            assert self.action_number_visits.sum() + 1 == self.number_visits
            assert self.action_type_number_visits.sum() + 1 == self.number_visits

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
                    self.action_total_value,
                    self.action_number_visits,
                    self.child_priors,
                    self.number_visits,
                    self.c_puct,
                    init_q_value,
                    self.max_value,
                    self.min_value,
                    np.nonzero(self.valid_actions)[0],
                )
            except ImportError:
                if not force_python_impl:
                    logger.warning("C implementation of best_action not found")

            if force_python_impl or action_c is None:
                child_score = self.child_Q() + self.c_puct * self.child_U()
                masked_child_score = child_score
                masked_child_score[~self.valid_actions] = -np.inf
                action = int(np.argmax(masked_child_score))
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
                self.c_puct,
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
                self.action_type_Q() + self.c_puct * self.action_type_U()
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
                self.action_total_value,
                self.action_number_visits,
                self.child_priors,
                number_visits,
                self.c_puct,
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
                ) + self.c_puct * self.child_U(valid_action_indices)
                action = int(valid_action_indices[np.argmax(child_score)])
                assert (
                    child_score[np.where(valid_action_indices == action_c)[0][0]]
                    >= child_score[np.where(valid_action_indices == action)[0][0]]
                    - 1e-4
                )
        else:
            action = action_c

        return action

    def select(self):
        current_node = self
        while current_node.is_expanded:
            best_action = current_node.best_action()
            current_node = current_node.get_child(best_action)
        return current_node

    def expand(
        self,
        child_priors,
        value_estimate: float,
        goal_logits: Optional[np.ndarray] = None,
        other_agent_action_dist: Optional[np.ndarray] = None,
        model_state_out: List[torch.Tensor] = [],
        add_dirichlet_noise=False,
    ) -> None:
        self.is_expanded = True
        self.child_priors = child_priors
        self.value_estimate = value_estimate

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
                self.parent.obs,
                self.info,
                self.goal_logits,
                update_own_reward=self.mcts.predict_goal_using_next_state,
                average_own_reward=self.mcts.predict_goal_using_average,
            )

        self.child_priors[~self.valid_actions] = 0
        self.child_priors /= self.child_priors.sum()
        self.noop_probability = float(self.child_priors[0])

        if not self.mcts.explore_noops:
            self.child_priors[0] = 0
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

    def get_expected_rewards(self, action: int) -> Tuple[float, float]:
        """
        Get the expected reward for the given action. The value returned is a tuple of
        (expected_reward, expected_own_reward).
        """

        if not self.mcts.use_goal_predictor and self.other_agent_action_dist is None:
            child = self.get_child(action)
            assert child.info is not None
            return child.reward, child.info["own_reward"]

        action_children = [
            child for actions, child in self.children.items() if actions[0] == action
        ]
        total_reward = 0
        total_visits = 0
        total_own_reward = 0
        total_own_reward_visits = 0
        for child in action_children:
            if (not self.mcts.use_goal_predictor) or child.is_expanded or child.done:
                total_reward += child.number_visits * child.reward
                total_visits += child.number_visits
            assert child.info is not None
            total_own_reward += child.number_visits * child.info["own_reward"]
            total_own_reward_visits += child.number_visits
        expected_own_reward = total_own_reward / total_own_reward_visits
        expected_reward = total_reward / total_visits
        return expected_reward, expected_own_reward

    def backup(self, value):
        current = self
        value = float(value)
        while True:
            value *= self.mcts.gamma
            value += current.reward
            current._number_visits += 1
            current._total_value += value

            if current.action is not None:
                assert isinstance(current.parent, MbagMCTSNode)
                current.parent.action_number_visits[current.action] += 1
                current.parent.action_total_value[current.action] += value
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

    def get_full_support_policy(self) -> np.ndarray:
        policy = np.zeros(len(self.valid_actions))
        if self.mcts.use_bilevel_action_selection:
            dist = calculate_limiting_mcts_distribution(
                self.child_priors[None, self.valid_actions],
                self.child_Q()[None, self.valid_actions],
                self.c_puct,
                self.mcts.num_sims,
            )[0]
            policy[self.valid_actions] = dist
        else:
            action_type_dist = np.zeros(len(self.valid_action_types))
            action_type_dist[self.valid_action_types] = (
                calculate_limiting_mcts_distribution(
                    self.action_type_priors[None, self.valid_action_types],
                    self.action_type_Q()[None, self.valid_action_types],
                    self.c_puct,
                    self.mcts.num_sims,
                )[0]
            )
            for action_type in range(MbagAction.NUM_ACTION_TYPES):
                if self.valid_action_types[action_type]:
                    action_type_slice = self.action_type_slices[action_type]
                    priors = self.child_priors[action_type_slice]
                    valid_actions = self.valid_actions[action_type_slice]
                    q = self.child_Q(action_type_slice)
                    policy[action_type_slice][valid_actions] = (
                        calculate_limiting_mcts_distribution(
                            priors[None, valid_actions],
                            q[None, valid_actions],
                            self.c_puct,
                            self.mcts.num_sims,
                        )[0]
                        * action_type_dist[action_type]
                    )
        assert np.abs(np.sum(policy) - 1) < 1e-2
        return policy


class MbagMCTS(MCTS):
    c_puct: Union[float, Sequence[float]]
    _temperature_schedule: Optional[Schedule]
    _puct_coefficient_schedule: Optional[Schedule]

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

        self._puct_coefficient_schedule = None
        if mcts_param["puct_coefficient_schedule"] is not None:
            self._puct_coefficient_schedule = PiecewiseSchedule(
                mcts_param["puct_coefficient_schedule"],
                outside_value=mcts_param["puct_coefficient_schedule"][-1][-1],
            )
            self.c_puct = self._puct_coefficient_schedule.value(0)

        self.prior_temperature = mcts_param.get("prior_temperature", 1.0)
        self.init_q_with_max = mcts_param.get("init_q_with_max", False)
        self.use_bilevel_action_selection = mcts_param.get(
            "use_bilevel_action_selection", False
        )
        self.dirichlet_action_subtype_noise_multiplier: float = mcts_param.get(
            "dirichlet_action_subtype_noise_multiplier", 10
        )
        self.sample_from_full_support_policy = mcts_param.get(
            "sample_from_full_support_policy", False
        )
        self.explore_noops = mcts_param.get("explore_noops", True)

        self.predict_goal_using_next_state = mcts_param.get(
            "predict_goal_using_next_state", False
        )
        self.predict_goal_using_average = mcts_param.get(
            "predict_goal_using_average", False
        )
        if self.predict_goal_using_next_state and self.predict_goal_using_average:
            raise ValueError(
                "predict_goal_using_next_state and predict_goal_using_average "
                "cannot both be True"
            )

        # Previously, we used a version of bilevel action selection that wasn't
        # quite accurate. It used the number of visits to the whole state rather than
        # for the particular action type to select the action within an action type, as
        # well as initializing Q values based on the mean for the whole state rather
        # than the action type. We keep this option around to reproduce old results.
        self.fix_bilevel_action_selection = mcts_param.get(
            "fix_bilevel_action_selection", False
        )

    def update_from_timestep(self, global_timestep: int):
        self.update_temperature(global_timestep)
        self._update_puct_coefficient(global_timestep)

    def update_temperature(self, global_timestep: int):
        if self._temperature_schedule is not None:
            self.temperature = self._temperature_schedule.value(global_timestep)

    def _update_puct_coefficient(self, global_timestep: int):
        if self._puct_coefficient_schedule is not None:
            self.c_puct = self._puct_coefficient_schedule.value(global_timestep)

    def compute_action(self, node: MbagMCTSNode) -> Tuple[np.ndarray, int]:
        tree_policies, actions = self.compute_actions([node])
        return tree_policies[0], int(actions[0])

    def compute_actions(  # noqa: C901
        self, nodes: List[MbagMCTSNode]
    ) -> Tuple[np.ndarray, np.ndarray]:
        for _ in range(self.num_sims):
            leaves: List[MbagMCTSNode] = [node.select() for node in nodes]
            obs = [leaf.obs for leaf in leaves]
            model_state_len = len(leaves[0].model_state_in)
            model_state_in = [
                torch.stack(
                    [leaf.model_state_in[state_index] for leaf in leaves], dim=0
                )
                for state_index in range(model_state_len)
            ]
            action_mask = np.stack([leaf.valid_actions for leaf in leaves])
            child_priors: np.ndarray
            values: np.ndarray
            model_state_out: List[torch.Tensor]

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
                        value_estimate=value,
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
        if self.sample_from_full_support_policy:
            tree_policies = np.stack(
                [node.get_full_support_policy() for node in nodes], axis=0
            )
        elif self.num_sims == 1:
            tree_policies = np.stack([node.child_priors for node in nodes], axis=0)
        else:
            tree_policies = np.stack(
                [
                    node.action_number_visits / (node.number_visits - 1)
                    for node in nodes
                ],
                axis=0,
            )
        if not self.explore_noops:
            for env_index, node in enumerate(nodes):
                tree_policies[env_index, 0] = node.noop_probability
                tree_policies[env_index, 1:] *= 1 - node.noop_probability
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

        for node, action in zip(nodes, actions):
            if not any(actions[0] == action for actions in node.children.keys()):
                assert (
                    (not self.explore_noops)
                    or self.sample_from_full_support_policy
                    or self.num_sims == 1
                )

        if logger.isEnabledFor(logging.DEBUG):
            node, action, tree_policy = nodes[0], actions[0], tree_policies[0]
            expected_reward, expected_own_reward = node.get_expected_rewards(action)
            logger.debug(
                "\t".join(
                    map(
                        str,
                        [
                            node.get_mbag_action(action),
                            expected_reward,
                            expected_own_reward,
                            node.child_Q()[action],
                            node.child_U()[action],
                            node.child_priors[action],
                            node.action_number_visits[action],
                            tree_policy[action],
                            node.valid_actions.astype(int).sum(),
                        ],
                    )
                )
            )
            mbag_action = node.get_mbag_action(action)
            if node.goal_logits is not None and mbag_action.action_type in [
                MbagAction.BREAK_BLOCK,
                MbagAction.PLACE_BLOCK,
            ]:
                block_location_goal_logits = node.goal_logits[
                    :,
                    mbag_action.block_location[0],
                    mbag_action.block_location[1],
                    mbag_action.block_location[2],
                ]
                block_location_goal_probs = np.exp(block_location_goal_logits)
                block_location_goal_probs /= block_location_goal_probs.sum()
                logger.debug(
                    " ".join(f"{prob:.2f}" for prob in block_location_goal_probs)
                )
                logger.debug(
                    node.obs[0][
                        :,
                        mbag_action.block_location[0],
                        mbag_action.block_location[1],
                        mbag_action.block_location[2],
                    ]
                )

            if False:
                node = nodes[0]
                flat_goal_logits = node.goal_logits.reshape(
                    (MinecraftBlocks.NUM_BLOCKS, -1)
                ).T
                flat_probs = np.exp(flat_goal_logits)
                flat_probs /= flat_probs.sum(axis=1, keepdims=True)
                goal_blocks = cast(MinecraftBlocks, node.state["goal_blocks"]).blocks
                width, height, depth = goal_blocks.shape
                flat_goal = goal_blocks.reshape(-1)
                correct_probs = flat_probs[np.arange(len(flat_goal)), flat_goal]
                correct_probs = correct_probs.reshape(goal_blocks.shape)
                for y in range(1, 4):
                    for z in range(depth):
                        logger.debug(
                            "".join(
                                *[
                                    " ░▒▓█"[int(correct_probs[x, y, z] * 4.99)]
                                    for x in range(width)
                                ]
                            ),
                        )
                    logger.debug("")

        return (
            tree_policies,
            actions,
        )


def calculate_limiting_mcts_distribution(
    priors: np.ndarray,
    q: np.ndarray,
    c_puct: float,
    num_simulations: int,
    *,
    tolerance: float = 1e-8,
):
    """
    Given arrays priors and q of shape (batch_size, num_actions), calculate the
    approximate limiting distribution of MCTS given in the paper "Monte-Carlo tree
    search as regularized policy optimization" by Grill et al.
    """

    batch_size, num_actions = priors.shape
    multiplier = c_puct * np.sqrt(num_simulations) / (num_actions + num_simulations)

    alpha_min = (q + multiplier * priors).max(axis=1)
    alpha_max = (q + multiplier).max(axis=1)

    while np.any(alpha_max - alpha_min > tolerance):
        alpha = (alpha_min + alpha_max) / 2
        mcts_policy = multiplier * priors / (alpha[:, None] - q)
        total_prob = mcts_policy.sum(axis=1)
        alpha_min[total_prob >= 1] = alpha[total_prob >= 1]
        alpha_max[total_prob < 1] = alpha[total_prob < 1]

    alpha = (alpha_min + alpha_max) / 2
    mcts_policy = multiplier * priors / (alpha[:, None] - q)
    mcts_policy /= mcts_policy.sum(axis=1, keepdims=True)

    return mcts_policy
