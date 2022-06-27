from mbag.environment.blocks import MinecraftBlocks
from mbag.environment.types import MbagAction
import pytest
import numpy as np
import logging
from numpy.testing import assert_array_equal

from mbag.environment.goals.grabcraft import (
    GrabcraftGoalGenerator,
    SingleWallGrabcraftGenerator,
)
from mbag.environment.mbag_env import MbagConfigDict
from mbag.evaluation.evaluator import MbagEvaluator
from mbag.agents.heuristic_agents import (
    LayerBuilderAgent,
    PriorityQueueAgent,
    MirrorBuildingAgent,
    ALL_HEURISTIC_AGENTS,
)
from mbag.environment.goals.simple import (
    BasicGoalGenerator,
    SimpleOverhangGoalGenerator,
)

logger = logging.getLogger(__name__)


def test_layer_builder_agent():
    evaluator = MbagEvaluator(
        {
            "world_size": (5, 5, 5),
            "num_players": 1,
            "horizon": 50,
            "goal_generator": (BasicGoalGenerator, {}),
            "goal_visibility": [True],
            "malmo": {
                "use_malmo": False,
                "use_spectator": False,
                "video_dir": None,
            },
        },
        [
            (
                LayerBuilderAgent,
                {},
            )
        ],
        force_get_set_state=True,
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward == 18


def test_pq_agent_basic():
    evaluator = MbagEvaluator(
        {
            "world_size": (5, 5, 5),
            "num_players": 1,
            "horizon": 50,
            "goal_generator": (BasicGoalGenerator, {}),
            "goal_visibility": [True],
            "malmo": {
                "use_malmo": False,
                "use_spectator": False,
                "video_dir": None,
            },
        },
        [
            (
                PriorityQueueAgent,
                {},
            )
        ],
        force_get_set_state=True,
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward == 18


def test_pq_agent_overhangs():
    for num_players in [1, 2]:
        evaluator = MbagEvaluator(
            {
                "world_size": (8, 8, 8),
                "num_players": num_players,
                "horizon": 100,
                "goal_generator": (SimpleOverhangGoalGenerator, {}),
                "goal_visibility": [True] * num_players,
                "malmo": {
                    "use_malmo": False,
                    "use_spectator": False,
                    "video_dir": None,
                },
            },
            [
                (PriorityQueueAgent, {}),
            ]
            * num_players,
            force_get_set_state=True,
        )
        episode_info = evaluator.rollout()
        assert episode_info.cumulative_reward == 13


def test_pq_agent_grabcraft():
    for num_players in [1, 2]:
        evaluator = MbagEvaluator(
            {
                "world_size": (12, 12, 12),
                "num_players": num_players,
                "horizon": 1000,
                "goal_generator": GrabcraftGoalGenerator,
                "goal_generator_config": {
                    "data_dir": "data/grabcraft",
                    "subset": "train",
                    "force_single_cc": True,
                },
                "goal_visibility": [True] * num_players,
                "malmo": {
                    "use_malmo": False,
                    "use_spectator": False,
                    "video_dir": None,
                },
            },
            [
                (PriorityQueueAgent, {}),
            ]
            * num_players,
        )
        episode_info = evaluator.rollout()
        (last_obs,) = episode_info.last_obs[0]
        if not np.all(last_obs[0] == last_obs[2]):
            for layer in range(12):
                if not np.all(last_obs[0, :, layer] == last_obs[2, :, layer]):
                    print(f"Mismatch in layer {layer}")
                    print("Current blocks:")
                    print(last_obs[0, :, layer])
                    print("Goal blocks:")
                    print(last_obs[2, :, layer])
                    break
        assert_array_equal(
            last_obs[0], last_obs[2], "Agent should finish building house."
        )


@pytest.mark.xfail(strict=False)
def test_malmo_pq():
    evaluator = MbagEvaluator(
        {
            "world_size": (8, 8, 8),
            "num_players": 1,
            "horizon": 100,
            "goal_generator": (SimpleOverhangGoalGenerator, {}),
            "goal_visibility": [True],
            "malmo": {
                "use_malmo": True,
                "use_spectator": False,
                "video_dir": None,
            },
        },
        [
            (PriorityQueueAgent, {}),
        ],
        force_get_set_state=True,
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward == 13


def test_rllib_heuristic_agents():
    from ray.rllib.agents.pg.pg import PGTrainer
    from ray.rllib.rollout import rollout
    from mbag.rllib.policies import MbagAgentPolicy

    env_config: MbagConfigDict = {
        "world_size": (8, 8, 8),
        "num_players": 1,
        "horizon": 100,
        "goal_generator": (BasicGoalGenerator, {}),
        "goal_visibility": [True],
        "malmo": {
            "use_malmo": False,
            "use_spectator": False,
            "video_dir": None,
        },
    }

    for heuristic_agent_id, heuristic_agent_cls in ALL_HEURISTIC_AGENTS.items():
        logger.info(f"Testing {heuristic_agent_id} agent...")
        heuristic_agent = heuristic_agent_cls({}, env_config)
        trainer = PGTrainer(
            {
                "env": "MBAG-v1",
                "env_config": env_config,
                "multiagent": {
                    "policies": {
                        "pq": (
                            MbagAgentPolicy,
                            None,
                            None,
                            {"mbag_agent": heuristic_agent},
                        )
                    },
                    "policy_mapping_fn": lambda agent_id: "pq",
                    "policies_to_train": [],
                },
                "framework": "torch",
            }
        )

        rollout(
            trainer,
            None,
            num_steps=0,
            num_episodes=2,
        )


def test_mirror_placed_blocks():
    agent = MirrorBuildingAgent({"world_size": (10, 10, 10)}, {})

    # If it's all air, nothing should be changed
    blocks = np.zeros((4, 4, 4))
    blocks[:] = MinecraftBlocks.AIR
    assert np.array_equal(agent._mirror_placed_blocks(blocks), blocks)

    # If one half is all bedrock the other half should also be all bedrock
    blocks[
        :2,
    ] = MinecraftBlocks.BEDROCK
    assert (agent._mirror_placed_blocks(blocks) == MinecraftBlocks.BEDROCK).all()
    blocks[:] = MinecraftBlocks.AIR
    blocks[
        2:,
    ] = MinecraftBlocks.BEDROCK
    assert (agent._mirror_placed_blocks(blocks) == MinecraftBlocks.BEDROCK).all()

    # Check if it can mirror one block change on the left side
    blocks[:] = MinecraftBlocks.AIR
    blocks[0, 0, 0] = MinecraftBlocks.BEDROCK
    mirror = blocks.copy()
    mirror[3, 0, 0] = MinecraftBlocks.BEDROCK
    assert np.array_equal(agent._mirror_placed_blocks(blocks), mirror)

    # Check if it can mirror one block change on the right side
    blocks[:] = MinecraftBlocks.AIR
    blocks[2, 0, 0] = MinecraftBlocks.BEDROCK
    mirror = blocks.copy()
    mirror[1, 0, 0] = MinecraftBlocks.BEDROCK
    assert np.array_equal(agent._mirror_placed_blocks(blocks), mirror)

    # If there are two different blocks on opposite side, nothing should be changed.
    blocks[:] = MinecraftBlocks.AIR
    blocks[3, 0, 0] = MinecraftBlocks.BEDROCK
    blocks[0, 0, 0] = MinecraftBlocks.NAME2ID["grass"]
    assert np.array_equal(agent._mirror_placed_blocks(blocks), blocks)

    # Test more complicated behavior
    blocks[:] = MinecraftBlocks.AIR
    blocks[2, 0, 0] = MinecraftBlocks.BEDROCK
    blocks[3, 0, 0] = MinecraftBlocks.BEDROCK
    blocks[0, 0, 0] = MinecraftBlocks.NAME2ID["grass"]
    mirror = blocks.copy()
    mirror[1, 0, 0] = MinecraftBlocks.BEDROCK
    assert np.array_equal(agent._mirror_placed_blocks(blocks), mirror)


def test_diff_indices():
    agent = MirrorBuildingAgent({"world_size": (10, 10, 10)}, {})

    a = np.zeros((5, 5, 5))
    b = np.ones((5, 5, 5))
    assert len(agent._diff_indices(a, a)) == 0

    diffs = agent._diff_indices(
        a,
        b,
    )
    assert len(diffs) == 5**3
    for x in range(5):
        for y in range(5):
            for z in range(5):
                assert np.array([x, y, z]) in diffs

    c = a.copy()
    c[0, 0, 0] = 1
    assert np.array_equal(agent._diff_indices(a, c), [[0, 0, 0]])
    c[1, 2, 3] = 1
    assert np.array_equal(agent._diff_indices(a, c), [[0, 0, 0], [1, 2, 3]])


def test_mirror_building_agent_get_action():
    agent = MirrorBuildingAgent({}, {"world_size": (4, 4, 4)})

    dim = (4, 4, 4, 4)
    a = np.zeros(dim)

    # Does it do nothing if the map is empty?
    assert agent.get_action((a,)) == (MbagAction.NOOP, 0, 0)

    # Does it copy to the right?
    a[0, 0, 0, 0] = MinecraftBlocks.BEDROCK
    assert str(agent.get_action((a,))) == str(
        (MbagAction.PLACE_BLOCK, 48, MinecraftBlocks.BEDROCK)
    )

    # Does it do nothing if there are differnt blocks on opposite sides?
    a[0, 3, 0, 0] = MinecraftBlocks.NAME2ID["grass"]
    assert str(agent.get_action((a,))) == str((MbagAction.NOOP, 0, 0))

    # Does it copy to the left?
    a[0, 0, 0, 0] = MinecraftBlocks.AIR
    assert str(agent.get_action((a,))) == str(
        (MbagAction.PLACE_BLOCK, 0, MinecraftBlocks.NAME2ID["grass"])
    )


def test_mirror_building_agent():
    evaluator = MbagEvaluator(
        {
            "world_size": (10, 10, 10),
            "num_players": 2,
            "horizon": 50,
            "goal_generator": (
                SingleWallGrabcraftGenerator,
                {"test_wall": True, "choose_densest": True, "force_bottom_grass": True},
            ),
            "goal_visibility": [True, False],
            "malmo": {
                "use_malmo": False,
                "use_spectator": False,
                "video_dir": None,
            },
        },
        [(LayerBuilderAgent, {}), (MirrorBuildingAgent, {})],
        force_get_set_state=False,
    )
    episode_info = evaluator.rollout()
    assert episode_info.cumulative_reward > 50


@pytest.mark.xfail(strict=False)
def test_mirror_building_agent_in_malmo():
    evaluator = MbagEvaluator(
        {
            "world_size": (10, 10, 10),
            "num_players": 2,
            "horizon": 100,
            "goal_generator": (
                SingleWallGrabcraftGenerator,
                {"force_bottom_grass": True},
            ),
            "goal_visibility": [True, False],
            "malmo": {
                "use_malmo": True,
                "use_spectator": True,
                "video_dir": None,
            },
        },
        [(LayerBuilderAgent, {}), (MirrorBuildingAgent, {})],
        force_get_set_state=False,
    )
    evaluator.rollout()
