import logging
import re
from itertools import product
from typing import Any, Dict, Type, cast

import numpy as np
import pytest

from mbag.agents.heuristic_agents import LowestBlockAgent, RandomAgent
from mbag.agents.mbag_agent import MbagAgent
from mbag.environment.goals.simple import BasicGoalGenerator
from mbag.environment.mbag_env import MbagConfigDict, MbagEnv
from mbag.evaluation.evaluator import MbagEvaluator
from mbag.evaluation.metrics import calculate_metrics

try:
    from ray.rllib.policy.policy import PolicySpec
    from ray.rllib.utils.typing import MultiAgentPolicyConfigDict
    from ray.tune.registry import ENV_CREATOR, _global_registry

    from mbag.rllib.callbacks import MbagCallbacks
except ImportError:
    pass

logger = logging.getLogger(__name__)


@pytest.mark.timeout(30)
@pytest.mark.uses_rllib
def test_metrics():
    from ray.rllib.algorithms.pg import PGConfig

    from mbag.rllib.policies import MbagAgentPolicy

    for num_players, teleportation, inf_blocks in product(
        [1, 2], [True, False], [True, False]
    ):
        logger.info(
            f"testing {num_players} player(s) with "
            f"teleportation={teleportation}, inf_blocks={inf_blocks}"
        )

        env_config: MbagConfigDict = MbagEnv.get_config(
            {
                "world_size": (8, 8, 8),
                "num_players": num_players,
                "players": [{} for _ in range(num_players)],
                "horizon": 1000,
                "goal_generator": BasicGoalGenerator,
                "goal_generator_config": {},
                "abilities": {
                    "teleportation": teleportation,
                    "inf_blocks": inf_blocks,
                    "flying": True,
                },
                "malmo": {
                    "use_malmo": False,
                    "use_spectator": False,
                    "video_dir": None,
                },
            }
        )

        agent_cls: Type[MbagAgent] = RandomAgent
        if num_players == 1 and teleportation and inf_blocks:
            agent_cls = LowestBlockAgent

        agents = [agent_cls({}, env_config) for _ in range(num_players)]
        env_id = "MBAGFlatActions-v1"
        env = _global_registry.get(ENV_CREATOR, env_id)(env_config)

        policies_config: MultiAgentPolicyConfigDict = {}
        for player_index in range(num_players):
            policy_id = f"player_{player_index}"
            agent_id = policy_id
            policies_config[policy_id] = PolicySpec(
                policy_class=MbagAgentPolicy,
                observation_space=env.observation_space.spaces[agent_id],
                action_space=env.action_space.spaces[agent_id],
                config={"mbag_agent": agents[player_index], "force_seed": 0},
            )

        trainer = (
            PGConfig()
            .environment(
                env_id,
                env_config=cast(Dict[Any, Any], env_config),
            )
            .callbacks(
                MbagCallbacks,
            )
            .rollouts(batch_mode="complete_episodes")
            .multi_agent(
                policies=policies_config,
                policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
                policies_to_train=[],
            )
            .evaluation(
                evaluation_duration=1,
                evaluation_duration_unit="episodes",
            )
            .framework("torch")
            .build()
        )

        evaluation_results = trainer.evaluate()["evaluation"]

        evaluator = MbagEvaluator(
            env_config,
            [(agent_cls, {}) for _ in range(num_players)],
        )
        episode = evaluator.rollout(agent_seed=0)
        metrics = calculate_metrics(episode)

        assert evaluation_results["custom_metrics"]

        for custom_metric_name, metric_value in evaluation_results[
            "custom_metrics"
        ].items():
            if not custom_metric_name.endswith("_mean"):
                continue
            metric_name = custom_metric_name[: -len("_mean")]

            if player_metric_match := re.fullmatch(
                r"player_(\d+)(/per_minute_metrics)?/(.*)", metric_name
            ):
                player_index_str, is_per_minute_metric, player_metric_name = (
                    player_metric_match.groups()
                )
                player_index = int(player_index_str)
                player_metrics = metrics["player_metrics"][player_index]
                if is_per_minute_metric:
                    other_metric_value = player_metrics["per_minute_metrics"][
                        player_metric_name
                    ]
                else:
                    other_metric_value = player_metrics[player_metric_name]  # type: ignore[literal-required]
                assert (
                    np.isnan(other_metric_value) and np.isnan(metric_value)
                ) or other_metric_value == metric_value
            else:
                assert (
                    np.isnan(metrics[metric_name]) and np.isnan(metric_value)  # type: ignore[literal-required]
                ) or metrics[
                    metric_name  # type: ignore[literal-required]
                ] == metric_value
