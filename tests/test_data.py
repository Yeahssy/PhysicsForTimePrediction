"""Test data loading and preprocessing."""

import pytest
import numpy as np
import torch

from src.data.preprocessing import StandardScaler, TimeFeatures, train_val_test_split
from src.utils.config import load_config


def test_standard_scaler():
    """Test StandardScaler."""
    data = np.random.randn(100, 5)

    scaler = StandardScaler()
    scaler.fit(data)

    # Test transform
    scaled = scaler.transform(data)
    assert scaled.shape == data.shape
    assert np.abs(np.mean(scaled, axis=0)).max() < 0.1
    assert np.abs(np.std(scaled, axis=0) - 1.0).max() < 0.1

    # Test inverse transform
    recovered = scaler.inverse_transform(scaled)
    assert np.allclose(data, recovered, rtol=1e-5)


def test_time_features():
    """Test time features extraction."""
    import pandas as pd

    dates = pd.date_range("2020-01-01", periods=100, freq="H")
    df = pd.DataFrame({"date": dates})

    time_features = TimeFeatures(freq="h")
    features = time_features.transform(df)

    assert len(features) == 100
    # Should have cyclical features (sin/cos)
    assert features.shape[1] > 0


def test_train_val_test_split():
    """Test chronological split."""
    data = np.arange(100).reshape(-1, 1)

    train, val, test = train_val_test_split(data, train_ratio=0.7, val_ratio=0.1)

    assert len(train) == 70
    assert len(val) == 10
    assert len(test) == 20

    # Verify chronological order
    assert train[-1][0] < val[0][0]
    assert val[-1][0] < test[0][0]


def test_config_loading():
    """Test configuration loading."""
    config = load_config("configs/base/default.yaml")

    assert "seq_len" in config
    assert "training" in config
    assert config["seq_len"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
