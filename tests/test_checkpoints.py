import pytest

try:
    from mbag.rllib.training_utils import load_trainer
except ImportError:
    pass


@pytest.mark.uses_rllib
def test_load_alpha_zero_assistant_checkpoint(tmp_path):
    load_trainer(
        "data/testing/checkpoints/alpha_zero_assistant/checkpoint_000100",
        "MbagAlphaZero",
        config_updates={"num_workers": 0, "num_gpus": 0, "gpus_per_worker": 0},
    )
