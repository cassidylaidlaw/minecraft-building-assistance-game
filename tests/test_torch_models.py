import pytest

try:
    import torch

    from mbag.rllib.torch_models import SeparatedTransformerEncoder
except ImportError:
    pass


@pytest.mark.uses_cuda
@pytest.mark.uses_rllib
def test_separated_transformer_batch_size():
    encoder = SeparatedTransformerEncoder(
        num_layers=3,
        d_model=4,
        nhead=2,
        dim_feedforward=4,
    )
    encoder.cuda()
    encoder_inputs = torch.rand((512, 4, 20, 20, 20), device="cuda")
    encoder_outputs = encoder(encoder_inputs)
    assert encoder_outputs.size() == encoder_inputs.size()
