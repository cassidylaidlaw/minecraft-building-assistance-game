import pytest

try:
    import torch

    from mbag.rllib.torch_models import (
        InterleavedBackbone,
        SeparatedTransformerEncoderLayer,
    )
except ImportError:
    pass


@pytest.mark.uses_cuda
@pytest.mark.uses_rllib
def test_separated_transformer_batch_size():
    encoder = InterleavedBackbone(
        num_layers=3,
        layer_creator=lambda layer_index: SeparatedTransformerEncoderLayer(
            d_model=4,
            nhead=2,
            dim_feedforward=4,
            n_spatial_dims=3,
            spatial_dim=layer_index % 3,
            batch_first=True,
        ),
    )
    encoder.cuda()
    encoder_inputs = torch.rand((512, 4, 20, 20, 20), device="cuda")
    encoder_outputs, _ = encoder(encoder_inputs)
    assert encoder_outputs.size() == encoder_inputs.size()
