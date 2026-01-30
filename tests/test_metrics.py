"""Test evaluation metrics."""

import pytest
import numpy as np
import torch

from src.evaluation.metrics import mse, mae, rmse, compute_metrics


def test_mse():
    """Test MSE calculation."""
    pred = np.array([1.0, 2.0, 3.0])
    true = np.array([1.5, 2.5, 3.5])

    result = mse(pred, true)
    expected = 0.25

    assert np.isclose(result, expected)


def test_mae():
    """Test MAE calculation."""
    pred = np.array([1.0, 2.0, 3.0])
    true = np.array([1.5, 2.5, 3.5])

    result = mae(pred, true)
    expected = 0.5

    assert np.isclose(result, expected)


def test_rmse():
    """Test RMSE calculation."""
    pred = np.array([1.0, 2.0, 3.0])
    true = np.array([1.5, 2.5, 3.5])

    result = rmse(pred, true)
    expected = 0.5

    assert np.isclose(result, expected)


def test_compute_metrics():
    """Test multiple metrics computation."""
    pred = np.random.randn(100, 10)
    true = pred + np.random.randn(100, 10) * 0.1

    results = compute_metrics(pred, true, metrics=["mse", "mae", "rmse"])

    assert "mse" in results
    assert "mae" in results
    assert "rmse" in results
    assert all(v >= 0 for v in results.values())


def test_metrics_with_torch_tensors():
    """Test metrics work with PyTorch tensors."""
    pred = torch.randn(50, 5)
    true = pred + torch.randn(50, 5) * 0.1

    result = mse(pred, true)
    assert isinstance(result, (float, np.floating))
    assert result >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
