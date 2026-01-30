"""Standard trainer for Transformer-based models."""

from typing import Dict, List, Optional, Callable
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.logging import get_logger, MetricTracker, format_metrics
from .callbacks import EarlyStopping, ModelCheckpoint


class StandardTrainer:
    """Training loop for standard (non-ODE) time series models.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Configuration dictionary.
        device: Device to train on.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Setup device
        if device is None:
            device = next(model.parameters()).device
        self.device = device
        self.model.to(self.device)

        # Training config
        train_config = config.get("training", {})
        self.epochs = train_config.get("epochs", 100)
        self.lr = train_config.get("learning_rate", 1e-4)
        self.weight_decay = train_config.get("weight_decay", 0.0)
        self.clip_grad = train_config.get("clip_grad", True)
        self.max_grad_norm = train_config.get("max_grad_norm", 1.0)
        self.pred_len = config.get("pred_len", 96)

        # Setup components
        self.criterion = self._get_criterion()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()

        # Callbacks
        self.early_stopping = EarlyStopping(
            patience=train_config.get("patience", 10),
            mode="min",
        )
        self.checkpoint = ModelCheckpoint(
            save_dir=config.get("logging", {}).get("save_dir", "checkpoints"),
            monitor="val_loss",
            mode="min",
        )

        self.logger = get_logger()

    def _get_criterion(self) -> nn.Module:
        """Get loss criterion."""
        return nn.MSELoss()

    def _get_optimizer(self) -> torch.optim.Optimizer:
        """Get optimizer."""
        return torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def _get_scheduler(self):
        """Get learning rate scheduler."""
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True,
        )

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        tracker = MetricTracker()

        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            # Move data to device
            x_enc, x_mark_enc, x_dec, x_mark_dec = [
                b.to(self.device) for b in batch[:4]
            ]

            # Get ground truth (last pred_len steps of decoder target)
            y_true = batch[2][:, -self.pred_len:, :].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            y_pred = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)

            # Compute loss
            loss = self.criterion(y_pred, y_true)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.clip_grad:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )

            # Update weights
            self.optimizer.step()

            # Track metrics
            tracker.update("loss", loss.item() * y_true.size(0), y_true.size(0))

            # Update progress bar
            pbar.set_postfix(loss=tracker.get_average("loss"))

        return tracker.get_all_averages()

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        tracker = MetricTracker()

        for batch in self.val_loader:
            x_enc, x_mark_enc, x_dec, x_mark_dec = [
                b.to(self.device) for b in batch[:4]
            ]
            y_true = batch[2][:, -self.pred_len:, :].to(self.device)

            y_pred = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            loss = self.criterion(y_pred, y_true)

            tracker.update("loss", loss.item() * y_true.size(0), y_true.size(0))

        return {"val_loss": tracker.get_average("loss")}

    def fit(self) -> Dict[str, List[float]]:
        """Run full training loop.

        Returns:
            Dictionary of training history.
        """
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.epochs}")

            # Train
            train_metrics = self.train_epoch()
            history["train_loss"].append(train_metrics["loss"])

            # Validate
            val_metrics = self.validate()
            history["val_loss"].append(val_metrics["val_loss"])

            # Log metrics
            self.logger.info(
                format_metrics(
                    {"train_loss": train_metrics["loss"], **val_metrics},
                    prefix=f"Epoch {epoch + 1}",
                )
            )

            # Learning rate scheduling
            self.scheduler.step(val_metrics["val_loss"])

            # Checkpointing
            self.checkpoint(
                self.model,
                epoch + 1,
                {"val_loss": val_metrics["val_loss"]},
                self.config,
            )

            # Early stopping
            if self.early_stopping(val_metrics["val_loss"]):
                self.logger.info("Early stopping triggered")
                break

        # Load best model
        try:
            self.checkpoint.load_best(self.model)
            self.logger.info("Loaded best model checkpoint")
        except ValueError:
            pass

        return history

    def save(self, path: str) -> None:
        """Save model and training state.

        Args:
            path: Path to save checkpoint.
        """
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }, path)

    def load(self, path: str) -> None:
        """Load model and training state.

        Args:
            path: Path to checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
