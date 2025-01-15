import copy
import os
import pickle
from typing import Any, List, Literal, Optional, Tuple, TypedDict, cast, get_args

import numpy as np
import tqdm

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.agents.mbag_agent import MbagAgent
from mbag.environment.actions import MbagAction, MbagActionTuple
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.goals.craftassist import NoRemainingHouseError
from mbag.environment.goals.goal_generator import GoalGeneratorConfig
from mbag.environment.goals.goal_transform import TransformedGoalGenerator
from mbag.environment.goals.transforms import AddGrassMode
from mbag.environment.mbag_env import MbagConfigDict
from mbag.environment.types import (
    CURRENT_BLOCKS,
    CURRENT_PLAYER,
    GOAL_BLOCKS,
    LAST_INTERACTED,
    OTHER_PLAYER,
    MbagObs,
    WorldSize,
)
from mbag.rllib.alpha_zero.planning import create_mbag_env_model

# Small value to add to the goal probabilities to avoid division by zero.
PROB_EPS = 1e-6


def get_goal_size_from_world_size(world_size: WorldSize) -> WorldSize:
    """Get the size of the goal from the size of the world.

    The goal width and depth are smaller than the world by 2 to allow players to move
    around the goal. The goal height is smaller by 2 because the bottom layer is
    bedrock and the top layer must be air to allow players to stand on the goal.
    """
    return (
        world_size[0] - 2,
        world_size[1] - 2,
        world_size[2] - 2,
    )


def get_padded_goals(goal: np.ndarray, size: WorldSize) -> List[np.ndarray]:
    """Get all the padded goals that fit in a world of the given size.

    The goal can be placed in any (x, z) position in the world, while the y position is
    fixed to 1. The goal is padded with air (zeros). This implementation is based on
    mbag.environment.goals.transforms.RandomlyPlaceTransform.
    """
    padded_goals = []
    for offset_x in range(0, size[0] - goal.shape[0] + 1):
        for offset_z in range(0, size[2] - goal.shape[2] + 1):
            padded_goal = np.zeros(size, np.uint8)
            goal_slice = (
                slice(offset_x, offset_x + goal.shape[0]),
                slice(0, goal.shape[1]),
                slice(offset_z, offset_z + goal.shape[2]),
            )
            padded_goal[goal_slice] = goal
            padded_goals.append(padded_goal)

    return padded_goals


def add_grass(goal: np.ndarray, mode: AddGrassMode) -> np.ndarray:
    """Add grass blocks to the goal.

    This implementation is based on mbag.environment.goals.transforms.AddGrassTransform.
    """
    assert mode in get_args(AddGrassMode), f"Invalid mode: {mode}"

    goal = goal.copy()
    if mode == "concatenate":
        smaller_goal = goal
        goal = np.zeros(
            (smaller_goal.shape[0], smaller_goal.shape[1] + 1, smaller_goal.shape[2]),
            np.uint8,
        )
        goal[:, 1:, :] = smaller_goal
        goal[:, 0, :] = MinecraftBlocks.NAME2ID["grass"]
    elif mode == "replace":
        goal[:, 0, :] = MinecraftBlocks.NAME2ID["grass"]
    elif mode == "surround":
        bottom_layer = goal[:, 0, :]
        bottom_layer[bottom_layer == MinecraftBlocks.NAME2ID["air"]] = (
            MinecraftBlocks.NAME2ID["grass"]
        )

    return goal


def place_goal_in_world(goal: np.ndarray, world_size: WorldSize) -> np.ndarray:
    """Place the goal in the world."""
    goal_in_world = np.zeros(world_size, np.uint8)
    goal_in_world[:, 0, :] = MinecraftBlocks.BEDROCK
    goal_in_world[:, 1, :] = MinecraftBlocks.NAME2ID["dirt"]
    goal_slice = (
        slice(1, 1 + goal.shape[0]),
        slice(1, 1 + goal.shape[1]),
        slice(1, 1 + goal.shape[2]),
    )
    goal_in_world[goal_slice] = goal
    return goal_in_world


# Methods of computing goal probabilities from goal distances.
GoalProbsMethod = Literal["neg", "exp_neg", "inv", "exp_inv"]


def calculate_goal_probs(
    goal_distances: np.ndarray,
    goal_probs_method: GoalProbsMethod,
    alpha: float = 1,
    beta: float = 1,
) -> np.ndarray:
    """Calculate the probabilities of the goals based on the goal distances.

    Args:
        goal_distances: Distances to the goals.
        goal_probs_method: Method of computing goal probabilities from goal distances.
        alpha: Alpha parameter for the goal probabilities method.
        beta: Beta parameter for the goal probabilities method.

    Returns:
        The probabilities of the goals.
    """
    assert goal_probs_method in get_args(
        GoalProbsMethod
    ), f"Invalid goal_probs_method: {goal_probs_method}"

    if goal_probs_method == "neg":
        # Option 1: negative goal distance.
        goal_probs = -(alpha * goal_distances**beta + PROB_EPS)
        # Shift the probabilities so that the minimum is 0.
        goal_probs -= goal_probs.min()
    elif goal_probs_method == "exp_neg":
        # Option 2: exponential of negative goal distance.
        goal_probs = np.exp(alpha * -(goal_distances**beta))
    elif goal_probs_method == "inv":
        # Option 3: inverse of goal distance.
        # Add 1 to the denominator to make the probability equal 1 when the distance is
        # 0 to match "exp_neg".
        goal_probs = 1 / (alpha * goal_distances**beta + 1)
    elif goal_probs_method == "exp_inv":
        # Option 4: exponential of inverse of goal distance. This requires handling NaNs
        goal_probs = np.exp(1 / (alpha * goal_distances**beta + 1))
        goal_probs = np.where(np.isnan(goal_probs), 0, PROB_EPS)

    # Normalize the goal probabilities.
    goal_probs_sum = goal_probs.sum()
    if goal_probs_sum > 0:
        goal_probs /= goal_probs.sum()
    else:
        # If all goal distances are 0, set the goal probabilities to a uniform
        # distribution.
        goal_probs = np.ones(len(goal_probs)) / len(goal_probs)

    return goal_probs


def cross_entropy_loss(input: np.ndarray, target: np.ndarray) -> float:
    """Compute the cross entropy loss with mean reduction.

    Args:
        input: Predicted logits of shape (N, C) where N is batch size and C is number of
            classes.
        target: Ground truth class indices of shape (N,).

    Returns:
        Mean cross entropy loss across the batch.
    """
    if input.ndim == 1:
        input = input.reshape(1, -1)
    if target.ndim == 0:
        target = target.reshape(1)
    assert (
        input.ndim == 2
    ), f"Expected input to have 1 or 2 dimensions. Got: {input.ndim}."
    assert target.ndim == 1, f"Expected target to have 1 dimension. Got: {target.ndim}."

    n_samples, n_classes = input.shape

    # Compute softmax. Subtract max for numerical stability.
    input_max = np.max(input, axis=1, keepdims=True)
    exp = np.exp(input - input_max)
    softmax = exp / np.sum(exp, axis=1, keepdims=True)

    # Convert targets to one-hot encoding.
    target_one_hot = np.zeros((n_samples, n_classes))
    target_one_hot[np.arange(n_samples), target] = 1

    # Compute cross entropy loss.
    log_softmax = np.log(softmax + 1e-12)  # Add small epsilon for numerical stability.
    loss = -np.sum(target_one_hot * log_softmax) / n_samples

    return float(loss)


def maybe_load_generated_goals(
    data_dir: str, subset: str
) -> Optional[List[MinecraftBlocks]]:
    """Load the generated goals from a file if it exists."""
    goals_dir = os.path.join(data_dir, f"houses/{subset}/blocks")
    goals_path = os.path.join(goals_dir, "goals.pkl")
    if os.path.exists(goals_path):
        with open(goals_path, "rb") as f:
            goals = pickle.load(f)
        return cast(List[MinecraftBlocks], goals)
    else:
        return None


class GoalPredictorConfig(TypedDict):
    normalize_goal_distance: bool
    """Whether to normalize the goal distance by the number of blocks that are being
    compared."""

    use_raw_goal_distance: bool
    """Whether to use the raw goal distance (number of blocks that are different from
    the current state) without considering which blocks have been interacted with."""

    predict_entire_world: bool
    """Whether to predict the entire world or only the goal volume."""

    goal_probs_method: GoalProbsMethod
    """Method of computing goal probabilities from goal distances."""

    alpha: float
    """Alpha parameter for the goal probabilities method. See calculate_goal_probs."""

    beta: float
    """Beta parameter for the goal probabilities method. See calculate_goal_probs."""

    expect_true_goal_is_generated: bool
    """Whether to expect that the true goal has been generated by the goal generator."""

    ignore_own_actions: bool
    """Whether to ignore the agent's own actions when calculating the goal distance."""

    use_num_actions_for_goal_distance: bool
    """Whether to use the number of place/break block actions needed to reach to goal as
    the goal distance. If True, 1 is added to the goal distance for each block that is
    different from thegoal and is not air (neither in the current state nor the goal).
    """


DEFAULT_ORACLE_GOAL_PREDICTOR_CONFIG = GoalPredictorConfig(
    normalize_goal_distance=False,
    use_raw_goal_distance=False,
    predict_entire_world=True,
    goal_probs_method="exp_neg",
    alpha=3,
    beta=1,
    expect_true_goal_is_generated=True,
    ignore_own_actions=True,
    use_num_actions_for_goal_distance=False,
)


class OracleGoalPredictor:
    def __init__(
        self,
        goal_generator_config: GoalGeneratorConfig,
        world_size: WorldSize,
        goal_size: WorldSize,
        force_generate_goals: bool = False,
    ) -> None:
        self.goal_generator = TransformedGoalGenerator(goal_generator_config)
        if self.goal_generator.num_remaining_goals is None:
            raise ValueError(
                "The goal generator must have a fixed number of remaining goals."
            )

        goal_generator_subconfig = goal_generator_config.get(
            "goal_generator_config", {}
        )
        data_dir = goal_generator_subconfig.get("data_dir")
        subset = goal_generator_subconfig.get("subset")
        if not force_generate_goals and data_dir is not None and subset is not None:
            maybe_goals = maybe_load_generated_goals(
                data_dir,
                subset,
            )
        else:
            maybe_goals = None
        self.goals = (
            maybe_goals if maybe_goals is not None else self._generate_goals(goal_size)
        )

        self.world_size = world_size
        self.goal_size = goal_size

    def _generate_goals(self, goal_size) -> List[MinecraftBlocks]:
        """Generate the goals using the goal generator.

        This method generates goals until there are no more houses left in the goal
        generator or for the maximum number of goals available in the goal generator
        upon initialization. The latter condition is necessary because the goal
        generator may have an infinite number of goals (e.g., random goals, repeated
        goals).
        """
        goals = []
        num_remaining_goals = self.goal_generator.num_remaining_goals
        assert num_remaining_goals is not None, "num_remaining_goals must be set."

        with tqdm.tqdm(total=num_remaining_goals) as pbar:
            for _ in range(num_remaining_goals):
                try:
                    goal = self.goal_generator.generate_goal(goal_size)
                    new_num_remaining_goals = self.goal_generator.num_remaining_goals
                    assert new_num_remaining_goals is not None, (
                        "num_remaining_goals must remain set. It was initially set but "
                        "got changed to None."
                    )
                    # Increment the progress bar by the number of goals
                    # generated/skipped. The number of remaining goals may remain the
                    # same (e.g., if the goal generator can generate an infinite number
                    # of goals); increment by 1 in that case.
                    pbar.update(max(1, num_remaining_goals - new_num_remaining_goals))
                    num_remaining_goals = new_num_remaining_goals
                    goals.append(goal)
                except NoRemainingHouseError:
                    break

        return goals

    def predict_goal(
        self,
        obs: MbagObs,
        normalize_goal_distance: bool,
        use_raw_goal_distance: bool,
        predict_entire_world: bool,
        goal_probs_method: GoalProbsMethod,
        alpha: float,
        beta: float,
        expect_true_goal_is_generated: bool,
        ignore_own_actions: bool,
        use_num_actions_for_goal_distance: bool,
    ) -> Tuple[np.ndarray, float]:
        """Predict the goal logits from the current observation and goals.

        Steps:
        1. Calculate the goal distances from the current state to each goal. The goal
            distance depends on the number of blocks that are different from the goal.
            Only blocks that have been interacted with are considered. If
            ignore_own_actions is True, only blocks that were last interacted by the
            other player are considered.
        2. Compute the goal probabilities from the goal distances using
            goal_probs_method.
        3. Set the goal block probabilities to a weighted sum of the goals according to
            the goal probabilities.
        4. Calculate the goal logits (log of the goal block probabilities).

        See GoalPredictorConfig for a description of the arguments.

        Returns:
            Tuple (goal_logits, cross_entropy) containing the goal logits array with
            shape (NUM_BLOCKS, width, height, depth), and the cross-entropy with respect
            to the true goal.
        """
        goal_pred_shape = (
            self.world_size + (MinecraftBlocks.NUM_BLOCKS,)
            if predict_entire_world
            else self.goal_size + (MinecraftBlocks.NUM_BLOCKS,)
        )

        goal_slice = (
            slice(1, 1 + self.goal_size[0]),
            slice(1, 1 + self.goal_size[1]),
            slice(1, 1 + self.goal_size[2]),
        )

        # Get the true goal blocks. Only used for computing the cross-entropy and sanity
        # checks.
        world_obs = obs[0]
        true_goal_blocks = world_obs[GOAL_BLOCKS]
        if not predict_entire_world:
            true_goal_blocks = true_goal_blocks[goal_slice]

        current_blocks = world_obs[CURRENT_BLOCKS]
        # Mask for blocks that were last interacted with by the other player.
        interacted_mask = world_obs[LAST_INTERACTED] == OTHER_PLAYER
        if not ignore_own_actions:
            # Also consider blocks last interacted with by the current player.
            interacted_mask |= world_obs[LAST_INTERACTED] == CURRENT_PLAYER

        if not predict_entire_world:
            current_blocks = current_blocks[goal_slice]
            interacted_mask = interacted_mask[goal_slice]

        # For each goal, contains the closest transformed version of the goal to the
        # current state.
        closest_transformed_goals = []
        goal_distances = []
        # Raw distances from the candidate goals to the current state.
        raw_goal_distances = []
        # Raw distances from the candidate goals to the true goal.
        # Only needed as a sanity check.
        raw_distances_goal_to_true_goal = []

        for goal in self.goals:
            # Get all the padded goals. Equivalent to the randomly_place transform.
            padded_goals = get_padded_goals(goal.blocks, self.goal_size)
            # Get the fully transformed goals by adding grass.
            transformed_goals = [
                add_grass(padded_goal, "surround") for padded_goal in padded_goals
            ]
            if predict_entire_world:
                # Place the goals in the world.
                transformed_goals = [
                    place_goal_in_world(transformed_goal, world_size=self.world_size)
                    for transformed_goal in transformed_goals
                ]

            transformed_goal_distances = []
            transformed_raw_goal_distances = []
            transformed_raw_distances_goal_to_true_goal = []

            for transformed_goal in transformed_goals:
                # Mask for current blocks that are different from the goal.
                wrong_block_mask = current_blocks != transformed_goal
                # Mask for current blocks that are different from the goal and are not
                # air (neither in the current state nor the goal).
                wrong_non_air_block_mask = (
                    wrong_block_mask
                    & (current_blocks != MinecraftBlocks.AIR)
                    & (transformed_goal != MinecraftBlocks.AIR)
                )
                # The goal distance is the number of blocks that are different from the
                # goal.
                goal_distance_array = wrong_block_mask.astype(int)
                if use_num_actions_for_goal_distance:
                    # If neither the current block nor the goal block is air, add 1 to
                    # the goal distance because the current block must be broken and a
                    # new block must be placed.
                    goal_distance_array += wrong_non_air_block_mask.astype(int)
                # Raw goal distance is the number of blocks that are different from the
                # goal.
                raw_goal_distance = goal_distance_array.sum()
                # Goal distance is the number of blocks that are different from the goal
                # and have been interacted with.
                goal_distance_array[~interacted_mask] = 0
                goal_distance = goal_distance_array.sum()

                transformed_goal_distances.append(goal_distance)
                transformed_raw_goal_distances.append(raw_goal_distance)
                # Raw distance from the padded candidate goal to the true goal.
                transformed_raw_distances_goal_to_true_goal.append(
                    (true_goal_blocks != transformed_goal).sum()
                )

            # Find the transformed goal with the smallest goal distance.
            closest_goal_idx = (
                np.argmin(transformed_raw_goal_distances)
                if use_raw_goal_distance
                else np.argmin(transformed_goal_distances)
            )
            closest_transformed_goal = transformed_goals[closest_goal_idx]
            closest_goal_distance = transformed_goal_distances[closest_goal_idx]
            # Raw goal distance to the closest goal according to the goal distance, which
            # takes into account only blocks that have been interacted with.
            closest_raw_goal_distance = transformed_raw_goal_distances[closest_goal_idx]

            # Optionally normalize the goal distance by the number of blocks considered.
            if normalize_goal_distance:
                num_blocks_considered = interacted_mask.sum()
                if num_blocks_considered > 0:
                    closest_goal_distance /= num_blocks_considered

            closest_transformed_goals.append(closest_transformed_goal)
            goal_distances.append(closest_goal_distance)
            raw_goal_distances.append(closest_raw_goal_distance)
            raw_distances_goal_to_true_goal.append(
                transformed_raw_distances_goal_to_true_goal
            )

        # Compute goal probabilities from the distances.
        goal_distances_for_goal_probs = np.array(
            raw_goal_distances if use_raw_goal_distance else goal_distances
        )
        goal_probs = calculate_goal_probs(
            goal_distances_for_goal_probs,
            goal_probs_method,
            alpha=alpha,
            beta=beta,
        )

        # Set the goal block probabilities to the weighted sum of the goals according to
        # the goal probabilities.
        goal_blocks_probs = np.zeros(goal_pred_shape)
        for transformed_goal, prob in zip(closest_transformed_goals, goal_probs):
            goal_blocks_probs[
                np.arange(goal_pred_shape[0])[:, None, None],
                np.arange(goal_pred_shape[1])[None, :, None],
                np.arange(goal_pred_shape[2])[None, None, :],
                transformed_goal,
            ] += prob

        # If a block type has a probability of 0, set it to a small value to avoid
        # infinite cross-entropy.
        goal_blocks_probs = np.where(
            goal_blocks_probs == 0, PROB_EPS, goal_blocks_probs
        )
        goal_blocks_probs = goal_blocks_probs / goal_blocks_probs.sum(
            axis=-1, keepdims=True
        )
        assert np.allclose(
            goal_blocks_probs.sum(axis=-1), 1
        ), "Goal block probabilities do not sum to 1 across block types."
        goal_logits = np.log(goal_blocks_probs)

        # If the true goal is expected to have been generated, check that the distance
        # from the closest goal to the true goal is 0.
        if expect_true_goal_is_generated:
            # Find the index of the candidate goal with the smallest raw goal distance.
            min_raw_distances_goal_to_true_goal = np.array(
                [min(x) for x in raw_distances_goal_to_true_goal]
            )
            true_goal_idx = np.argmin(min_raw_distances_goal_to_true_goal)
            assert min_raw_distances_goal_to_true_goal[true_goal_idx] == 0, (
                "The distance from the closest goal to the true goal is not 0. Got: "
                f"{min_raw_distances_goal_to_true_goal[true_goal_idx]}."
            )

        # Compute the cross-entropy between the true goal and the goal logits.
        flat_goal_logits = goal_logits.reshape(-1, goal_logits.shape[-1])
        flat_true_goal_blocks = true_goal_blocks.reshape(-1)
        cross_entropy = cross_entropy_loss(flat_goal_logits, flat_true_goal_blocks)

        # Reshape to (NUM_BLOCKS, width, height, depth).
        goal_logits = goal_logits.transpose(3, 0, 1, 2)

        return goal_logits, cross_entropy


class OracleGoalPredictionAgent(MbagAgent):
    """An agent that greedily maximizes the reward based on the oracle goal predictor."""

    def __init__(
        self,
        agent_config: Any,
        env_config: MbagConfigDict,
    ) -> None:
        """Initialize the agent.

        Args:
            agent_config: Configuration for the agent. Required key: "player_index".
                Optional keys: "line_of_sight_masking" and all the keys in
                GoalPredictorConfig.
            env_config: Configuration for the environment.
        """
        super().__init__(agent_config, env_config)

        world_size = env_config.get("world_size")
        if world_size is None:
            raise ValueError("world_size must be provided in the environment config.")
        self.world_size = cast(WorldSize, tuple(world_size))

        goal_generator_config = env_config.get("goal_generator_config")
        if goal_generator_config is None:
            raise ValueError(
                "goal_generator_config must be provided in the environment config."
            )

        player_index = agent_config.get("player_index")
        if player_index is None:
            raise ValueError("player_index must be provided in the agent config.")
        self.player_index = player_index

        self.line_of_sight_masking = agent_config.get("line_of_sight_masking", True)
        goal_size = get_goal_size_from_world_size(self.world_size)
        self.goal_predictor = OracleGoalPredictor(
            goal_generator_config, self.world_size, goal_size
        )
        # Use a deep copy of the environment config because creating the environment
        # model modifies the config.
        self.env = create_mbag_env_model(copy.deepcopy(env_config), self.player_index)

        self.goal_predictor_config = DEFAULT_ORACLE_GOAL_PREDICTOR_CONFIG.copy()
        # Update the default config with the agent_config.
        for key in self.goal_predictor_config:
            if key in agent_config:
                self.goal_predictor_config[key] = agent_config[key]  # type: ignore[literal-required]

    def get_action_distribution(self, obs: MbagObs) -> np.ndarray:
        """Get the action distribution based on the oracle goal predictor.

        Computes the expected goal-dependent reward for each valid action based on the
        goal logits predicted by the oracle goal predictor. The action distribution is
        uniformly distributed across actions with the maximum reward. If there are no
        actions that achieve a positive reward, this will typically produce a uniform
        distribution over no-op and move actions (depending on how the goal-dependent
        reward is computed).
        """
        goal_logits, cross_entropy = self.goal_predictor.predict_goal(
            obs, **self.goal_predictor_config
        )

        world_obs, inventory_obs, timestep = obs
        action_mask_flat = MbagActionDistribution.get_mask_flat(
            self.env_config,
            (world_obs[None], inventory_obs[None], timestep[None]),
            line_of_sight_masking=self.line_of_sight_masking,
        )[0]
        action_dist = np.zeros((MbagActionDistribution.NUM_CHANNELS,) + self.world_size)
        action_mapping = MbagActionDistribution.get_action_mapping(self.env_config)

        valid_flat_actions = np.nonzero(action_mask_flat)[0]
        action_tuples = []
        rewards = []
        for flat_action in valid_flat_actions:
            action_tuple = cast(MbagActionTuple, tuple(action_mapping[flat_action]))
            action_tuples.append(action_tuple)
            action = MbagAction(
                action_tuple,
                self.world_size,
            )
            reward = self.env._get_predicted_goal_dependent_reward(
                obs,
                action,
                goal_logits,
                self.player_index,
            )
            rewards.append(reward)

        # Get indices of actions with maximum reward.
        argmax_rewards = np.array(rewards) >= max(rewards)
        for valid_action_index in np.nonzero(argmax_rewards)[0]:
            # Get the corresponding action tuple.
            action_type, block_location_index, block_id = action_tuples[
                valid_action_index
            ]
            # Convert the block location index to x, y, z coordinates.
            x, y, z = np.unravel_index(block_location_index, self.world_size)
            # Add probability mass to the action type at the x, y, z location. The
            # channel corresponding to the action type is a slice if the action type has
            # a block_id (e.g., place_block, give_block) and a single channel otherwise
            # (e.g., move). If it is a channel, first index into the correct block_id
            # channel. Then, index into the x, y, z location.
            channel = MbagActionDistribution.ACTION_TYPE2CHANNEL[action_type]
            if isinstance(channel, slice):
                action_dist[channel][block_id, x, y, z] = 1
            else:
                action_dist[channel, x, y, z] = 1

        action_dist /= action_dist.sum()
        return action_dist
