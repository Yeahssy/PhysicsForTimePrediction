"""Evaluator for time series forecasting models."""

from typing import Dict, List, Optional, Union
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_metrics


class Evaluator:
    """Unified evaluation pipeline for time series models.

    Args:
        model: Model to evaluate.
        device: Device to run evaluation on.
        pred_len: Prediction horizon length.
    """

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        pred_len: int = 96,
    ):
        self.model = model
        self.pred_len = pred_len

        if device is None:
            device = next(model.parameters()).device
        self.device = device
        self.model.to(self.device)

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        metrics: List[str] = None,
        return_predictions: bool = False,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Evaluate model on a dataset.

        Args:
            dataloader: DataLoader for evaluation data.
            metrics: List of metric names to compute.
            return_predictions: Whether to return predictions and targets.

        Returns:
            Dictionary of evaluation results.
        """
        if metrics is None:
            metrics = ["mse", "mae"]

        self.model.eval()

        all_preds = []
        all_trues = []

        for batch in tqdm(dataloader, desc="Evaluating"):
            x_enc, x_mark_enc, x_dec, x_mark_dec = [
                b.to(self.device) for b in batch[:4]
            ]
            y_true = batch[2][:, -self.pred_len:, :]

            # Forward pass
            output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)

            # Handle VAE-style output
            if isinstance(output, tuple):
                y_pred = output[0]
            else:
                y_pred = output

            all_preds.append(y_pred.cpu().numpy())
            all_trues.append(y_true.numpy())

        # Concatenate all predictions
        preds = np.concatenate(all_preds, axis=0)
        trues = np.concatenate(all_trues, axis=0)

        # Compute metrics
        results = compute_metrics(preds, trues, metrics)

        if return_predictions:
            results["predictions"] = preds
            results["targets"] = trues

        return results

    @torch.no_grad()
    def evaluate_with_inverse_transform(
        self,
        dataloader: DataLoader,
        scaler,
        metrics: List[str] = None,
    ) -> Dict[str, float]:
        """Evaluate model with inverse scaling transformation.

        Args:
            dataloader: DataLoader for evaluation data.
            scaler: Scaler object with inverse_transform method.
            metrics: List of metric names to compute.

        Returns:
            Dictionary of evaluation results.
        """
        if metrics is None:
            metrics = ["mse", "mae"]

        results = self.evaluate(dataloader, metrics, return_predictions=True)

        preds = results["predictions"]
        trues = results["targets"]

        # Inverse transform
        batch_size, seq_len, n_features = preds.shape
        preds_flat = preds.reshape(-1, n_features)
        trues_flat = trues.reshape(-1, n_features)

        preds_inv = scaler.inverse_transform(preds_flat).reshape(batch_size, seq_len, n_features)
        trues_inv = scaler.inverse_transform(trues_flat).reshape(batch_size, seq_len, n_features)

        # Recompute metrics on original scale
        results_inv = compute_metrics(preds_inv, trues_inv, metrics)

        return {f"{k}_original_scale": v for k, v in results_inv.items()}

    def benchmark(
        self,
        test_loader: DataLoader,
        horizons: List[int] = None,
        metrics: List[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Run benchmark evaluation across multiple prediction horizons.

        Args:
            test_loader: Test data loader.
            horizons: List of prediction horizons to evaluate.
            metrics: List of metric names to compute.

        Returns:
            Nested dictionary of results by horizon and metric.
        """
        if horizons is None:
            horizons = [24, 48, 96, 192]
        if metrics is None:
            metrics = ["mse", "mae"]

        original_pred_len = self.pred_len
        results = {}

        for horizon in horizons:
            if horizon > original_pred_len:
                print(f"Skipping horizon {horizon} (exceeds pred_len {original_pred_len})")
                continue

            self.pred_len = horizon
            horizon_results = self.evaluate(test_loader, metrics)
            results[f"H{horizon}"] = horizon_results

        self.pred_len = original_pred_len
        return results

    def print_results(self, results: Dict[str, float]) -> None:
        """Print evaluation results in a formatted table.

        Args:
            results: Dictionary of metric names to values.
        """
        print("\n" + "=" * 40)
        print("Evaluation Results")
        print("=" * 40)

        for name, value in results.items():
            if isinstance(value, float):
                print(f"{name:20s}: {value:.6f}")

        print("=" * 40 + "\n")

    def save_results(
        self,
        results: Dict,
        save_path: Union[str, Path],
    ) -> None:
        """Save evaluation results to file.

        Args:
            results: Results dictionary.
            save_path: Path to save results.
        """
        import json

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        serializable = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            elif isinstance(v, dict):
                serializable[k] = {
                    kk: vv.tolist() if isinstance(vv, np.ndarray) else vv
                    for kk, vv in v.items()
                }
            else:
                serializable[k] = v

        with open(save_path, "w") as f:
            json.dump(serializable, f, indent=2)

        print(f"Results saved to: {save_path}")
