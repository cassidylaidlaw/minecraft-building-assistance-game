import os
import tempfile

import pytest

from mbag.agents.heuristic_agents import LayerBuilderAgent, NoopAgent, RandomAgent
from mbag.environment.goals.simple import BasicGoalGenerator
from mbag.evaluation.evaluator import MbagEvaluator


@pytest.mark.uses_malmo
def test_malmo():
    evaluator = MbagEvaluator(
        {
            "world_size": (5, 5, 5),
            "num_players": 1,
            "horizon": 100,
            "goal_generator": BasicGoalGenerator,
            "goal_generator_config": {},
            "malmo": {
                "use_malmo": True,
                "use_spectator": False,
                "video_dir": None,
            },
        },
        [
            (NoopAgent, {}),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward == 18


@pytest.mark.uses_malmo
def test_two_agents_in_malmo():
    evaluator = MbagEvaluator(
        {
            "world_size": (5, 5, 5),
            "num_players": 2,
            "horizon": 50,
            "goal_generator": BasicGoalGenerator,
            "goal_generator_config": {},
            "players": [{}, {}],
            "malmo": {
                "use_malmo": True,
                "use_spectator": False,
                "video_dir": None,
            },
        },
        [
            (LayerBuilderAgent, {}),
            (NoopAgent, {}),
        ],
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward == 18


@pytest.mark.uses_malmo
def test_video_dir():
    with tempfile.TemporaryDirectory() as video_dir:
        evaluator = MbagEvaluator(
            {
                "world_size": (5, 5, 5),
                "num_players": 1,
                "horizon": 100,
                "goal_generator": BasicGoalGenerator,
                "goal_generator_config": {},
                "malmo": {
                    "use_malmo": True,
                    "use_spectator": True,
                    "video_dir": video_dir,
                },
            },
            [
                (LayerBuilderAgent, {}),
            ],
        )
        evaluator.rollout()
        assert os.path.exists(os.path.join(video_dir, "000000.mp4"))
