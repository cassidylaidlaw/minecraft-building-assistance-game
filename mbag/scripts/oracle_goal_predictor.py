import json
import pandas
import os
import pathlib
import pickle
import torch
import zipfile
from logging import Logger
from typing import List, Literal, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tqdm

from mbag.environment.goals.craftassist import CraftAssistGoalGenerator
from mbag.environment.goals.goal_transform import TransformedGoalGenerator
from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.state import mbag_obs_to_state
from mbag.environment.types import (
    CURRENT_BLOCKS,
    GOAL_BLOCKS,
    LAST_INTERACTED,
    CURRENT_PLAYER,
    OTHER_PLAYER,
    WorldSize,
)
from mbag.evaluation.episode import MbagEpisode
from mbag.rllib.sacred_utils import NoTypeAnnotationsFileStorageObserver

from sacred import Experiment

# Methods of computing goal probabilities from goal distances.
GoalProbsMethod = Literal["neg", "exp_neg", "inv", "exp_inv"]
# Small value to add to the goal probabilities to avoid division by zero.
PROB_EPS = 1e-6

ex = Experiment("oracle_goal_predictor")


def get_padded_goals(goal: np.ndarray, size: WorldSize) -> List[np.ndarray]:
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


def add_grass(goal: np.ndarray, mode: str) -> np.ndarray:
    assert mode in ["concatenate", "replace", "surround"], f"Invalid mode: {mode}"

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
    # Place the goal in the world.
    goal_in_world = np.zeros(world_size, np.uint8)
    # Set the bottom layer to bedrock.
    goal_in_world[:, 0, :] = MinecraftBlocks.BEDROCK
    # Set the 1-th layer to dirt.
    goal_in_world[:, 1, :] = MinecraftBlocks.NAME2ID["dirt"]
    # Place the goal in the world.
    goal_slice = (
        slice(1, 1 + goal.shape[0]),
        slice(1, 1 + goal.shape[1]),
        slice(1, 1 + goal.shape[2]),
    )
    goal_in_world[goal_slice] = goal
    return goal_in_world


def calculate_goal_probs(
    goal_distances: np.ndarray,
    goal_probs_method: GoalProbsMethod,
    alpha: float = 1,
    beta: float = 1,
) -> np.ndarray:
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
    else:
        raise ValueError(f"Invalid goal_probs_method: {goal_probs_method}")

    # Normalize the goal probabilities.
    goal_probs_sum = goal_probs.sum()
    if goal_probs_sum > 0:
        goal_probs /= goal_probs.sum()
    else:
        # If all goal distances are 0, set the goal probabilities to a uniform
        # distribution.
        goal_probs = np.ones(len(goal_probs)) / len(goal_probs)

    return goal_probs


def predict_goal(
    episode: MbagEpisode,
    goals: List[MinecraftBlocks],
    world_size: WorldSize,
    goal_size: WorldSize,
    normalize_goal_distance: bool = False,
    use_raw_goal_distance: bool = False,
    predict_entire_world: bool = False,
    goal_probs_method: GoalProbsMethod = "exp_neg",
    n_steps: int = 20,
    alpha: float = 1,
    beta: float = 1,
) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray]:
    goal_pred_shape = (
        world_size + (MinecraftBlocks.NUM_BLOCKS,)
        if predict_entire_world
        else goal_size + (MinecraftBlocks.NUM_BLOCKS,)
    )

    goal_slice = (
        slice(1, 1 + goal_size[0]),
        slice(1, 1 + goal_size[1]),
        slice(1, 1 + goal_size[2]),
    )

    cross_entropy_per_step = []
    goal_probs_per_step = []
    goal_distances_per_step = []
    raw_goal_distances_per_step = []
    # Raw distances from the candidate goals to the true goal for each step.
    # Common shape: [num_steps, num_candidate_goals]. Next dimension is the number of
    # padded versions of the same candidate gaol. Only needed for debugging.
    raw_distances_goal_to_true_goal_per_step = []

    steps = list(range(0, len(episode.obs_history), n_steps))
    if steps[-1] != len(episode.obs_history) - 1:
        steps.append(len(episode.obs_history) - 1)

    for step in tqdm.tqdm(steps, desc="Steps", leave=False):
        # Get the observation at the step from one of the players; it does not matter which.
        obs = episode.obs_history[step][0]
        world_obs = obs[0]

        # Get the true goal blocks.
        true_goal_blocks = world_obs[GOAL_BLOCKS]
        if not predict_entire_world:
            true_goal_blocks = true_goal_blocks[goal_slice]

        closest_transformed_goals = []
        goal_distances = []
        # Raw distances from the candidate goals to the current state.
        raw_goal_distances = []
        # Raw distances from the candidate goals to the true goal. Only needed for debugging.
        raw_distances_goal_to_true_goal = []

        for i, goal in enumerate(goals):
            # Get all the padded goals. Equivalent to the randomly_place transform.
            padded_goals = get_padded_goals(goal.blocks, goal_size)
            # Get the fully transformed goals by adding grass.
            transformed_goals = [
                add_grass(padded_goal, "surround") for padded_goal in padded_goals
            ]
            # Place the goals in the world (if needed).
            if predict_entire_world:
                transformed_goals = [
                    place_goal_in_world(transformed_goal, world_size=world_size)
                    for transformed_goal in transformed_goals
                ]

            transformed_goal_distances = []
            transformed_raw_goal_distances = []
            transformed_raw_distances_goal_to_true_goal = []

            for transformed_goal in transformed_goals:
                current_blocks = world_obs[CURRENT_BLOCKS]
                # Mask for blocks that have been interacted with, not considering the
                # bottom bedrock layer.
                interacted_mask = (world_obs[LAST_INTERACTED] == CURRENT_PLAYER) | (
                    world_obs[LAST_INTERACTED] == OTHER_PLAYER
                )
                if not predict_entire_world:
                    current_blocks = current_blocks[goal_slice]
                    interacted_mask = interacted_mask[goal_slice]

                # Mask for current blocks that are different from the goal.
                goal_distance_array = current_blocks != transformed_goal
                # Raw goal distance is the number of blocks that are different from the
                # goal.
                raw_goal_distance = goal_distance_array.sum()
                # Goal distance is the number of blocks that are different from the goal
                # and have been interacted with.
                goal_distance_array[~interacted_mask] = 0
                goal_distance = goal_distance_array.sum()

                transformed_goal_distances.append(goal_distance)
                transformed_raw_goal_distances.append(raw_goal_distance)
                # TODO: only compute this once, not for each step, since the true goal and the candidate
                # goals do not change per step.
                # Raw distance from the padded candidate goal to the true goal.
                transformed_raw_distances_goal_to_true_goal.append(
                    (true_goal_blocks != transformed_goal).sum()
                )

            # Find the goal with the smallest goal distance.
            if use_raw_goal_distance:
                closest_goal_idx = np.argmin(transformed_raw_goal_distances)
            else:
                closest_goal_idx = np.argmin(transformed_goal_distances)
            closest_transformed_goal = transformed_goals[closest_goal_idx]
            closest_goal_distance = transformed_goal_distances[closest_goal_idx]
            # Raw goal distance to the closest goal according to the goal distance, which
            # takes into account only blocks that have been interacted with.
            closest_raw_goal_distance = transformed_raw_goal_distances[closest_goal_idx]

            # Optionally normalize the goal distance by the number of placeable blocks in
            # the goal.
            if normalize_goal_distance:
                closest_goal_distance /= (
                    (closest_transformed_goal != MinecraftBlocks.BEDROCK)
                    & (closest_transformed_goal != MinecraftBlocks.AIR)
                ).sum()

            closest_transformed_goals.append(closest_transformed_goal)
            goal_distances.append(closest_goal_distance)
            raw_goal_distances.append(closest_raw_goal_distance)
            raw_distances_goal_to_true_goal.append(
                transformed_raw_distances_goal_to_true_goal
            )

        goal_distances_per_step.append(goal_distances)
        raw_goal_distances_per_step.append(raw_goal_distances)
        raw_distances_goal_to_true_goal_per_step.append(raw_distances_goal_to_true_goal)

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
        goal_probs_per_step.append(goal_probs)

        # Set the goal block type probabilities according to the probabilities of the goals.
        goal_blocks_probs = np.zeros(goal_pred_shape)
        for goal, prob in zip(closest_transformed_goals, goal_probs):
            goal_blocks_probs[
                np.arange(goal_pred_shape[0])[:, None, None],
                np.arange(goal_pred_shape[1])[None, :, None],
                np.arange(goal_pred_shape[2])[None, None, :],
                goal,
            ] += prob

        # If a block type has a probability of 0, set it to a small value to avoid infinite cross-entropy.
        goal_blocks_probs = np.where(
            goal_blocks_probs == 0, PROB_EPS, goal_blocks_probs
        )
        goal_blocks_probs = goal_blocks_probs / goal_blocks_probs.sum(
            axis=-1, keepdims=True
        )
        assert np.allclose(
            goal_blocks_probs.sum(axis=-1), 1
        ), "Goal block probabilities do not sum to 1 across block types."

        # Compute the cross-entropy between the true goal and the goal block probabilities.
        float_goal_blocks_probs = goal_blocks_probs.reshape(
            -1, goal_blocks_probs.shape[-1]
        )
        flat_true_goal_blocks = true_goal_blocks.reshape(-1)
        cross_entropy = torch.nn.functional.cross_entropy(
            torch.from_numpy(np.log(float_goal_blocks_probs)).float(),
            torch.from_numpy(flat_true_goal_blocks).long(),
        )
        cross_entropy_per_step.append(cross_entropy.item())

    # Find the true goal index.
    min_raw_distances_goal_to_true_goal_per_step = np.array(
        [min(x) for x in raw_distances_goal_to_true_goal_per_step[-1]]
    )
    true_goal_idx = np.argmin(min_raw_distances_goal_to_true_goal_per_step)
    assert min_raw_distances_goal_to_true_goal_per_step[true_goal_idx] == 0, (
        "The distance from the closest goal to the true goal is not 0. Got: "
        f"{min_raw_distances_goal_to_true_goal_per_step[true_goal_idx]}."
    )

    # Compute the rank of the true goal at each step according to the predicted goal
    # probabilities.
    true_goal_rank_per_step = (
        len(goals)
        - np.where(np.array(goal_probs_per_step).argsort(axis=1) == true_goal_idx)[1]
    )

    return (
        steps,
        np.array(cross_entropy_per_step),
        np.array(goal_probs_per_step),
        true_goal_rank_per_step,
    )


def get_out_dir_name(
    subset: str,
    normalize_goal_distance: bool,
    use_raw_goal_distance: bool,
    predict_entire_world: bool,
    goal_probs_method: GoalProbsMethod,
    alpha: float,
    beta: float,
    n_steps: int,
) -> str:
    name = f"{subset}"
    if normalize_goal_distance:
        name += "_norm_dist"
    if use_raw_goal_distance:
        name += "_raw_dist"
    if predict_entire_world:
        name += "_entire_world"
    name += f"_{goal_probs_method}_alpha_{alpha}_beta_{beta}_nsteps_{n_steps}"
    return name


@ex.config
def sacred_config() -> None:
    episodes_eval_dir = ""
    episodes_path = os.path.join(episodes_eval_dir, "episodes.zip")
    if not os.path.exists(episodes_path):
        raise ValueError(f"Episodes file not found: {episodes_path}")

    data_dir = "/nas/ucb/ebronstein/minecraft-building-assistance-game/data/craftassist"
    subset = "small200"
    goals_dir = os.path.join(data_dir, f"houses/{subset}/blocks")
    goals_path = os.path.join(goals_dir, "goals.pkl")
    if not os.path.exists(goals_path):
        raise ValueError(f"Goals file not found: {goals_path}")

    world_width = 11  # noqa: F841
    world_height = 10  # noqa: F841
    world_depth = 10  # noqa: F841
    world_size = (world_width, world_height, world_depth)
    goal_width = world_width - 2
    goal_height = world_height - 2
    goal_depth = world_depth - 2
    goal_size = (goal_width, goal_height, goal_depth)

    # If True, the goal distance is divided by the number of blocks in the goal.
    normalize_goal_distance = False
    """If True, the raw goal distance is used to compute goal probabilities instead of
    the goal distance."""
    use_raw_goal_distance = False
    """If True, the goal probabilities are computed for the entire world instead of just
    the goal volume."""
    predict_entire_world = False
    goal_probs_method: GoalProbsMethod = "exp_neg"
    # Inverse temperature for computing the goal probabilities.
    alpha = 3
    # Exponent for computing the goal probabilities.
    beta = 1
    # Predictions are made for every n_steps steps.
    n_steps = 20

    out_dir = os.path.join(
        episodes_eval_dir,
        "oracle_goal_predictions",
        get_out_dir_name(
            subset,
            normalize_goal_distance,
            use_raw_goal_distance,
            predict_entire_world,
            goal_probs_method,
            alpha,
            beta,
            n_steps,
        ),
    )

    observer = NoTypeAnnotationsFileStorageObserver(out_dir)
    ex.observers.append(observer)


@ex.automain
def main(
    episodes_path: str,
    goals_path: str,
    world_size: WorldSize,
    goal_size: WorldSize,
    normalize_goal_distance: bool,
    use_raw_goal_distance: bool,
    predict_entire_world: bool,
    goal_probs_method: GoalProbsMethod,
    n_steps: int,
    alpha: float,
    beta: float,
    observer: NoTypeAnnotationsFileStorageObserver,
    _log: Logger,
) -> None:
    # Load the episodes.
    with zipfile.ZipFile(episodes_path, "r") as episodes_zip:
        with episodes_zip.open("episodes.pickle") as episodes_file:
            episodes = pickle.load(episodes_file)

    # Load the goals.
    with open(goals_path, "rb") as f:
        goals = pickle.load(f)

    rows = []
    for episode_idx in tqdm.trange(100, desc="Episodes", leave=False):
        episode = episodes[episode_idx]
        (
            steps,
            cross_entropy_per_step,
            goal_probs_per_step,
            true_goal_rank_per_step,
        ) = predict_goal(
            episode,
            goals,
            cast(WorldSize, tuple(world_size)),
            cast(WorldSize, tuple(goal_size)),
            normalize_goal_distance=normalize_goal_distance,
            use_raw_goal_distance=use_raw_goal_distance,
            predict_entire_world=predict_entire_world,
            goal_probs_method=goal_probs_method,
            n_steps=n_steps,
            alpha=alpha,
            beta=beta,
        )
        episode_rows = [
            {
                "episode": episode_idx,
                "step": step,
                "goal_probs_method": goal_probs_method,
                "normalize_goal_distance": normalize_goal_distance,
                "use_raw_goal_distance": use_raw_goal_distance,
                "predict_entire_world": predict_entire_world,
                "alpha": alpha,
                "beta": beta,
                "cross_entropy": cross_entropy_per_step[i],
                "goal_probs": goal_probs_per_step[i],
                "true_goal_rank": true_goal_rank_per_step[i],
            }
            for i, step in enumerate(steps)
        ]
        rows.extend(episode_rows)

    df = pandas.DataFrame(rows)
    result_path = os.path.join(observer.dir, "result.pkl")
    df.to_pickle(result_path)
    _log.info(f"Saved results to {result_path}")
