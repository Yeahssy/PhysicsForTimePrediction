"""Evaluation metrics for time series forecasting."""

from typing import Dict, List, Union

import numpy as np
import torch


def mse(
    pred: Union[np.ndarray, torch.Tensor],
    true: Union[np.ndarray, torch.Tensor],
) -> float:
    """Compute Mean Squared Error.

    Args:
        pred: Predictions.
        true: Ground truth.

    Returns:
        MSE value.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()

    return np.mean((pred - true) ** 2)


def mae(
    pred: Union[np.ndarray, torch.Tensor],
    true: Union[np.ndarray, torch.Tensor],
) -> float:
    """Compute Mean Absolute Error.

    Args:
        pred: Predictions.
        true: Ground truth.

    Returns:
        MAE value.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()

    return np.mean(np.abs(pred - true))


def rmse(
    pred: Union[np.ndarray, torch.Tensor],
    true: Union[np.ndarray, torch.Tensor],
) -> float:
    """Compute Root Mean Squared Error.

    Args:
        pred: Predictions.
        true: Ground truth.

    Returns:
        RMSE value.
    """
    return np.sqrt(mse(pred, true))


def mape(
    pred: Union[np.ndarray, torch.Tensor],
    true: Union[np.ndarray, torch.Tensor],
    epsilon: float = 1e-8,
) -> float:
    """Compute Mean Absolute Percentage Error.

    Args:
        pred: Predictions.
        true: Ground truth.
        epsilon: Small value to avoid division by zero.

    Returns:
        MAPE value (as percentage).
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()

    return np.mean(np.abs((pred - true) / (np.abs(true) + epsilon))) * 100


def smape(
    pred: Union[np.ndarray, torch.Tensor],
    true: Union[np.ndarray, torch.Tensor],
    epsilon: float = 1e-8,
) -> float:
    """Compute Symmetric Mean Absolute Percentage Error.

    Args:
        pred: Predictions.
        true: Ground truth.
        epsilon: Small value to avoid division by zero.

    Returns:
        sMAPE value (as percentage).
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()

    denominator = (np.abs(pred) + np.abs(true) + epsilon)
    return np.mean(2 * np.abs(pred - true) / denominator) * 100


def compute_metrics(
    pred: Union[np.ndarray, torch.Tensor],
    true: Union[np.ndarray, torch.Tensor],
    metrics: List[str] = None,
) -> Dict[str, float]:
    """Compute multiple metrics.

    Args:
        pred: Predictions.
        true: Ground truth.
        metrics: List of metric names to compute.
                 Default: ['mse', 'mae', 'rmse'].

    Returns:
        Dictionary of metric names to values.
    """
    if metrics is None:
        metrics = ["mse", "mae", "rmse"]

    metric_funcs = {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
    }

    results = {}
    for name in metrics:
        if name in metric_funcs:
            results[name] = metric_funcs[name](pred, true)
        else:
            raise ValueError(f"Unknown metric: {name}")

    return results


def metric_per_feature(
    pred: Union[np.ndarray, torch.Tensor],
    true: Union[np.ndarray, torch.Tensor],
    metric_fn: callable,
) -> np.ndarray:
    """Compute metric per feature dimension.

    Args:
        pred: Predictions [batch, seq_len, features].
        true: Ground truth [batch, seq_len, features].
        metric_fn: Metric function to apply.

    Returns:
        Array of metric values per feature.
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()

    n_features = pred.shape[-1]
    results = []

    for i in range(n_features):
        results.append(metric_fn(pred[..., i], true[..., i]))

    return np.array(results)
