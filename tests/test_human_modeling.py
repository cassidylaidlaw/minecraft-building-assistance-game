from typing import cast

import numpy as np
import pytest

try:
    from mbag.rllib.alpha_zero.mcts import calculate_limiting_mcts_distribution
    from mbag.scripts.evaluate_human_modeling import HumanModelingEvaluationResults
    from mbag.scripts.evaluate_human_modeling import ex as evaluate_human_modeling_ex
    from mbag.scripts.train import ex as train_ex
except ImportError:
    pass

from .test_train import default_bc_config, default_config  # noqa: F401

TUTORIAL_BC_CHECKPOINT = (
    "data/logs/BC/sample_human_models/tutorial/2024-04-10_16-35-41/1/checkpoint_000100"
)


@pytest.mark.uses_rllib
def test_calculate_limiting_mcts_distribution():
    priors = np.array(
        [
            [0.1, 0.1, 0.3, 0.4, 0.1],
            [0.1, 0.1, 0.3, 0.4, 0.1],
            [0.1, 0.1, 0.3, 0.4, 0.1],
        ]
    )
    q = np.array(
        [
            [0, 0, 0, 0, 0],
            [4, 4, 4, 4, 4],
            [0, 0, -2, 1, 0],
        ]
    )
    np.testing.assert_almost_equal(
        calculate_limiting_mcts_distribution(priors, q, 6, 1),
        np.array(
            [
                [0.1, 0.1, 0.3, 0.4, 0.1],
                [0.1, 0.1, 0.3, 0.4, 0.1],
                [0.0643648, 0.0643648, 0.0844204, 0.722485, 0.0643648],
            ]
        ),
        decimal=6,
    )


@pytest.mark.uses_rllib
@pytest.mark.slow
@pytest.mark.timeout(30)
def test_evaluate_human_modeling(tmp_path):
    out_dir = str(tmp_path)
    result = cast(
        HumanModelingEvaluationResults,
        evaluate_human_modeling_ex.run(
            config_updates={
                "checkpoint": TUTORIAL_BC_CHECKPOINT,
                "policy_id": "human",
                "human_data_dir": "data/human_data/sample_tutorial_rllib",
                "out_dir": out_dir,
            }
        ).result,
    )

    assert result["cross_entropy"] < 0.15
    assert result["accuracy"] > 0.9


@pytest.mark.uses_rllib
@pytest.mark.slow
@pytest.mark.timeout(60)
def test_evaluate_human_modeling_pikl(
    tmp_path, default_config, default_bc_config  # noqa: F811
):
    for num_simulations, explore_noops in [
        (1, True),
        (1, False),
        (10, False),
    ]:
        alpha_zero_result = train_ex.run(
            config_updates={
                **default_config,
                **default_bc_config,
                "teleportation": False,
                "inf_blocks": False,
                "width": 6,
                "height": 6,
                "depth": 6,
                "min_width": 0,
                "min_height": 0,
                "min_depth": 0,
                "horizon": 500,
                "goal_generator": "tutorial",
                "run": "MbagAlphaZero",
                "load_policies_mapping": {"human": "human"},
                "is_human": [False],
                "checkpoint_to_load_policies": TUTORIAL_BC_CHECKPOINT,
                "overwrite_loaded_policy_type": True,
                "num_training_iters": 0,
                "add_dirichlet_noise": False,
                "evaluation_explore": True,
                "line_of_sight_masking": True,
                "use_goal_predictor": False,
                "num_simulations": num_simulations,
                "argmax_tree_policy": False,
                "sample_from_full_support_policy": True,
                "explore_noops": explore_noops,
                "temperature": 1,
                "dirichlet_epsilon": 0,
            }
        ).result
        assert alpha_zero_result is not None
        alpha_zero_checkpoint = alpha_zero_result["final_checkpoint"]

        out_dir = str(
            tmp_path / f"eval_num_sims_{num_simulations}_explore_noops_{explore_noops}"
        )
        result = cast(
            HumanModelingEvaluationResults,
            evaluate_human_modeling_ex.run(
                config_updates={
                    "checkpoint": alpha_zero_checkpoint,
                    "policy_id": "human",
                    "human_data_dir": "data/human_data/sample_tutorial_rllib",
                    "out_dir": out_dir,
                    "minibatch_size": 1,
                    "participant_ids": [1],
                }
            ).result,
        )

        print(result)

        if num_simulations == 1:
            assert result["cross_entropy"] < 0.15
            assert result["accuracy"] > 0.9
        else:
            assert result["cross_entropy"] < 2
            assert result["accuracy"] > 0.4
