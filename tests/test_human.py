import pytest
import logging

from mbag.evaluation.evaluator import MbagEvaluator
from mbag.agents.heuristic_agents import (
    NoopAgent,
)
from mbag.environment.goals.simple import BasicGoalGenerator

logger = logging.getLogger(__name__)


@pytest.mark.uses_malmo
def test_human_in_malmo():
    """
    Make sure the inventory agent can place blocks
    """

    evaluator = MbagEvaluator(
        {
            "world_size": (5, 6, 5),
            "num_players": 1,
            "horizon": 25,
            "goal_generator": BasicGoalGenerator,
            "goal_generator_config": {"pallette": True},
            "malmo": {
                "use_malmo": True,
                "use_spectator": False,
                "video_dir": None,
            },
            "players": [
                {
                    "is_human": True,
                }
            ],
            "abilities": {"teleportation": False, "flying": True, "inf_blocks": False},
        },
        [
            (NoopAgent, {}),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward == -1
