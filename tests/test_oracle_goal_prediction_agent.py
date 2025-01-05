import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose

from mbag.agents.oracle_goal_prediction_agent import cross_entropy_loss


def test_perfect_prediction():
    """Test when predictions are perfect (very high confidence for correct class)"""
    logits = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0]])
    targets = np.array([0, 1])
    loss = cross_entropy_loss(logits, targets)
    assert loss < 1e-3


def test_uniform_prediction():
    """Test when predictions are uniform (equal confidence for all classes)"""
    logits = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    targets = np.array([1, 2])
    expected_loss = -np.log(1 / 3)  # -log(1/num_classes)
    loss = cross_entropy_loss(logits, targets)
    assert_allclose(loss, expected_loss, rtol=1e-5)


def test_binary_classification():
    """Test binary classification case"""
    logits = np.array([[0.6, 0.4], [0.3, 0.7]])
    targets = np.array([0, 1])

    p0 = np.exp(0.6) / (np.exp(0.6) + np.exp(0.4))
    p1 = np.exp(0.7) / (np.exp(0.3) + np.exp(0.7))
    expected_loss = -(np.log(p0) + np.log(p1)) / 2

    loss = cross_entropy_loss(logits, targets)
    assert_allclose(loss, expected_loss, rtol=1e-5)


def test_numerical_stability():
    """Test numerical stability with very large and very small numbers"""
    logits = np.array([[1000.0, -1000.0], [-1000.0, 1000.0]])
    targets = np.array([0, 1])
    loss = cross_entropy_loss(logits, targets)
    assert loss < 1e-3


def test_batch_size_one():
    """Test with batch size of 1"""
    logits = np.array([[0.6, 0.4, 0.2]])
    targets = np.array([1])
    loss = cross_entropy_loss(logits, targets)
    assert isinstance(loss, float)


def test_no_batch_dimension():
    """Test with no batch dimension"""
    logits = np.array([0.6, 0.4, 0.2])
    targets = np.array(1)
    loss = cross_entropy_loss(logits, targets)
    assert isinstance(loss, float)


def test_invalid_target_index():
    """Test input validation for invalid target index"""
    logits = np.array([[0.6, 0.4]])
    targets = np.array([2])
    with pytest.raises(IndexError):
        cross_entropy_loss(logits, targets)


@pytest.mark.parametrize("logits_shape", [(10, 2), (100, 5)])
def test_matches_pytorch(logits_shape):
    """Test that the implementation matches PyTorch for binary classification."""
    np.random.seed(42)
    torch.manual_seed(42)

    num_samples, num_classes = logits_shape
    logits_np = np.random.randn(num_samples, num_classes)
    targets_np = np.random.randint(0, num_classes, size=num_samples)

    logits_torch = torch.from_numpy(logits_np)
    targets_torch = torch.from_numpy(targets_np)

    loss = cross_entropy_loss(logits_np, targets_np)
    torch_loss = torch.nn.functional.cross_entropy(logits_torch, targets_torch).item()

    assert_allclose(loss, torch_loss, rtol=1e-5)


def test_matches_pytorch_large_logits():
    """Test against PyTorch implementation with very large logits."""
    logits_np = np.array([[1000.0, -1000.0], [-1000.0, 1000.0]])
    targets_np = np.array([0, 1])

    logits_torch = torch.from_numpy(logits_np)
    targets_torch = torch.from_numpy(targets_np)

    loss = cross_entropy_loss(logits_np, targets_np)
    torch_loss = torch.nn.functional.cross_entropy(logits_torch, targets_torch).item()

    assert_allclose(loss, torch_loss, atol=1e-10)


def test_matches_pytorch_zero_logits():
    """Test against PyTorch implementation with zero logits."""
    logits_np = np.zeros((5, 3))
    targets_np = np.array([0, 1, 2, 1, 0])

    logits_torch = torch.from_numpy(logits_np)
    targets_torch = torch.from_numpy(targets_np)

    loss = cross_entropy_loss(logits_np, targets_np)
    torch_loss = torch.nn.functional.cross_entropy(logits_torch, targets_torch).item()

    assert_allclose(loss, torch_loss, rtol=1e-5)
