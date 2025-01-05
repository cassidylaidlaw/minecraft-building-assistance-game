from typing import Dict, List, TypedDict, cast

import numpy as np

from mbag.agents.action_distributions import MbagActionDistribution
from mbag.environment.actions import MBAG_ACTION_BREAK_PALETTE_NAME, MbagAction

from .episode import MbagEpisode


class MbagPlayerMetrics(TypedDict, total=False):
    num_noop: int
    num_unintentional_noop: int
    num_break_block: int
    num_break_palette: int
    num_place_block: int
    num_give_block: int
    num_move_pos_x: int
    num_move_neg_x: int
    num_move_pos_y: int
    num_move_neg_y: int
    num_move_pos_z: int
    num_move_neg_z: int

    num_correct_break_block: int
    num_correct_place_block: int
    break_block_accuracy: float
    place_block_accuracy: float

    own_reward: float
    own_reward_prop: float
    goal_dependent_reward: float
    goal_independent_reward: float

    per_minute_metrics: Dict[str, float]


class MbagEpisodeMetrics(TypedDict):
    player_metrics: List[MbagPlayerMetrics]
    goal_similarity: float
    goal_distance: float
    goal_percentage: float
    reward: float


def get_rounded_minutes(episode: MbagEpisode, t: int) -> int:
    info_dict = episode.info_history[min(t, episode.length - 1)][0]
    if episode.env_config["malmo"]["use_malmo"]:
        start_time = episode.info_history[0][0]["timestamp"]
        if t < episode.length:
            total_seconds = (info_dict["timestamp"] - start_time).total_seconds()
        else:
            end_time = episode.info_history[-1][0]["timestamp"]
            episode_seconds = (end_time - start_time).total_seconds()
            horizon_seconds = (
                episode.env_config["horizon"]
                * episode.env_config["malmo"]["action_delay"]
            )
            total_seconds = episode_seconds + (t - episode.length) / (
                episode.env_config["horizon"] - episode.length
            ) * (horizon_seconds - episode_seconds)
    else:
        total_seconds = (t + 1) * episode.env_config["malmo"]["action_delay"]
    return int(total_seconds // 60)


def calculate_per_player_metrics(
    episode: MbagEpisode, player_index: int
) -> MbagPlayerMetrics:
    player_metrics: MbagPlayerMetrics = {}
    per_minute_player_metrics: Dict[str, float] = {}

    for valid_action_type in MbagActionDistribution.get_valid_action_types(
        episode.env_config
    ):
        action_type_name = MbagAction.ACTION_TYPE_NAMES[valid_action_type]
        player_metrics[f"num_{action_type_name.lower()}"] = 0  # type: ignore[literal-required]
        if valid_action_type in [MbagAction.BREAK_BLOCK, MbagAction.PLACE_BLOCK]:
            player_metrics[f"num_correct_{action_type_name.lower()}"] = 0  # type: ignore[literal-required]
    player_metrics["num_unintentional_noop"] = 0
    if not episode.env_config["abilities"]["inf_blocks"]:
        player_metrics["num_break_palette"] = 0
    player_metrics["own_reward"] = 0
    player_metrics["goal_dependent_reward"] = 0
    player_metrics["goal_independent_reward"] = 0

    for t in range(episode.env_config["horizon"]):
        if t < episode.length:
            info_dict = episode.info_history[t][player_index]

            player_metrics["own_reward"] += info_dict["own_reward"]
            player_metrics["goal_dependent_reward"] += info_dict[
                "goal_dependent_reward"
            ]
            player_metrics["goal_independent_reward"] += info_dict[
                "goal_independent_reward"
            ]

            action = info_dict["action"]
            if action.action_type == MbagAction.BREAK_BLOCK and action.is_palette(
                episode.env_config["abilities"]["inf_blocks"]
            ):
                action_type_name = MBAG_ACTION_BREAK_PALETTE_NAME
            else:
                action_type_name = MbagAction.ACTION_TYPE_NAMES[action.action_type]
            player_metrics[f"num_{action_type_name.lower()}"] += 1  # type: ignore[literal-required]

            if (
                info_dict["attempted_action"].action_type != MbagAction.NOOP
                and info_dict["action"].action_type == MbagAction.NOOP
            ):
                player_metrics["num_unintentional_noop"] += 1

            if info_dict["action_correct"]:
                player_metrics[f"num_correct_{action_type_name.lower()}"] += 1  # type: ignore[literal-required]

        rounded_minutes = get_rounded_minutes(episode, t)
        for metric_key in player_metrics:
            if rounded_minutes > 0:
                metric_min_key = f"{metric_key}_{rounded_minutes}_min"
                if metric_min_key not in per_minute_player_metrics:
                    per_minute_player_metrics[metric_min_key] = player_metrics[
                        metric_key  # type: ignore[literal-required]
                    ]

    last_info_dict = episode.last_infos[player_index]
    player_metrics["own_reward_prop"] = last_info_dict["own_reward_prop"]
    player_metrics["per_minute_metrics"] = per_minute_player_metrics

    action_type_names = [
        MbagAction.ACTION_TYPE_NAMES[action_type]
        for action_type in [
            MbagAction.BREAK_BLOCK,
            MbagAction.PLACE_BLOCK,
        ]
    ]
    for action_type_name in action_type_names:
        num_correct = cast(
            int, player_metrics.get(f"num_correct_{action_type_name.lower()}", 0)
        )
        total: int = cast(int, player_metrics.get(f"num_{action_type_name.lower()}", 0))
        percent_correct = num_correct / total if total != 0 else np.nan
        player_metrics[f"{action_type_name.lower()}_accuracy"] = percent_correct  # type: ignore[literal-required]

    return player_metrics


def calculate_metrics(episode: MbagEpisode) -> MbagEpisodeMetrics:
    width, height, depth = episode.env_config["world_size"]

    goal_similarity = episode.last_infos[0]["goal_similarity"]
    goal_distance = width * height * depth - goal_similarity

    players_metrics: List[MbagPlayerMetrics] = []
    for player_index in range(episode.env_config["num_players"]):
        players_metrics.append(calculate_per_player_metrics(episode, player_index))

    metrics: MbagEpisodeMetrics = {
        "goal_similarity": episode.last_infos[0]["goal_similarity"],
        "goal_distance": goal_distance,
        "goal_percentage": episode.last_infos[0].get("goal_percentage", np.nan),
        "player_metrics": players_metrics,
        "reward": sum(episode.reward_history),
    }

    cumulative_reward = 0.0
    for t in range(episode.env_config["horizon"]):
        rounded_minutes = get_rounded_minutes(episode, t)
        # Use the first player's info dict to get metrics that should be the same for
        # all players (if there are multiple).
        info_dict = episode.info_history[min(t, episode.length - 1)][0]
        if t < len(episode.reward_history):
            cumulative_reward += episode.reward_history[t]
        goal_percentage_key = f"goal_percentage_{rounded_minutes}_min"
        if goal_percentage_key not in metrics:
            metrics[goal_percentage_key] = info_dict[  # type: ignore[literal-required]
                "goal_percentage"
            ]
        goal_similarity_key = f"goal_similarity_{rounded_minutes}_min"
        if goal_similarity_key not in metrics:
            metrics[goal_similarity_key] = info_dict["goal_similarity"]  # type: ignore[literal-required]
        goal_distance_key = f"goal_distance_{rounded_minutes}_min"
        if goal_distance_key not in metrics:
            metrics[goal_distance_key] = width * height * depth - info_dict["goal_similarity"]  # type: ignore[literal-required]
        reward_key = f"reward_{rounded_minutes}_min"
        if reward_key not in metrics:
            metrics[reward_key] = cumulative_reward  # type: ignore[literal-required]

    return metrics


def calculate_mean_metrics(
    episodes_metrics: List[MbagEpisodeMetrics],
) -> MbagEpisodeMetrics:
    num_players = len(episodes_metrics[0]["player_metrics"])
    mean_player_metrics: List[MbagPlayerMetrics] = [{} for _ in range(num_players)]
    for player_index in range(num_players):
        mean_player_metrics[player_index] = {"per_minute_metrics": {}}
        for metric_name in episodes_metrics[0]["player_metrics"][player_index]:
            if metric_name == "per_minute_metrics":
                continue
            metric_values = [
                episode_metrics["player_metrics"][player_index][metric_name]  # type: ignore[literal-required]
                for episode_metrics in episodes_metrics
            ]
            mean_player_metrics[player_index][metric_name] = np.nanmean(metric_values)  # type: ignore[literal-required]
        for metric_name in episodes_metrics[0]["player_metrics"][player_index][
            "per_minute_metrics"
        ]:
            metric_values = [
                episode_metrics["player_metrics"][player_index]["per_minute_metrics"][
                    metric_name
                ]
                for episode_metrics in episodes_metrics
            ]
            mean_player_metrics[player_index]["per_minute_metrics"][metric_name] = (
                np.nanmean(metric_values)
            )
    metrics: MbagEpisodeMetrics = {
        "goal_similarity": np.mean(
            [episode_metrics["goal_similarity"] for episode_metrics in episodes_metrics]
        ),
        "goal_distance": np.mean(
            [episode_metrics["goal_distance"] for episode_metrics in episodes_metrics]
        ),
        "goal_percentage": np.mean(
            [
                episode_metrics["goal_percentage"]
                for episode_metrics in episodes_metrics
                if "goal_percentage" in episode_metrics
            ]
        ),
        "reward": np.mean(
            [episode_metrics["reward"] for episode_metrics in episodes_metrics]
        ),
        "player_metrics": mean_player_metrics,
    }

    for key in episodes_metrics[0]:
        if key not in metrics:
            metric_values = [
                episode_metrics[key]  # type: ignore[literal-required]
                for episode_metrics in episodes_metrics
            ]
            metrics[key] = np.mean(metric_values)  # type: ignore[literal-required]

    return metrics
