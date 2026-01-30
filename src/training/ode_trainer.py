"""Adjoint trainer for Neural ODE models."""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils.logging import get_logger, MetricTracker, format_metrics
from .trainer import StandardTrainer


class AdjointTrainer(StandardTrainer):
    """Training loop optimized for ODE-based models.

    Key differences from StandardTrainer:
    - Supports VAE-style training with KL divergence loss
    - Monitors NFE (number of function evaluations)
    - Uses higher gradient clipping threshold

    Args:
        model: ODE-based model to train.
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
        super().__init__(model, train_loader, val_loader, config, device)

        # VAE training settings
        train_config = config.get("training", {})
        self.kl_weight = train_config.get("kl_weight", 1.0)
        self.kl_annealing = train_config.get("kl_annealing", False)
        self.annealing_epochs = train_config.get("annealing_epochs", 20)

        # ODE-specific: higher gradient clipping threshold
        self.max_grad_norm = train_config.get("max_grad_norm", 5.0)

        self.current_epoch = 0

    def _get_kl_weight(self) -> float:
        """Get current KL weight (with annealing if enabled)."""
        if not self.kl_annealing:
            return self.kl_weight

        # Linear annealing from 0 to kl_weight
        progress = min(self.current_epoch / self.annealing_epochs, 1.0)
        return self.kl_weight * progress

    def _compute_kl_loss(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence loss for VAE.

        Args:
            mean: Mean of latent distribution.
            logvar: Log variance of latent distribution.

        Returns:
            KL divergence loss.
        """
        # KL(q(z|x) || p(z)) where p(z) = N(0, 1)
        kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        return kl

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with ODE-specific handling.

        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        tracker = MetricTracker()

        # Reset NFE counter if available
        if hasattr(self.model, "get_ode_func"):
            self.model.get_ode_func().reset_nfe()

        pbar = tqdm(self.train_loader, desc="Training (ODE)")
        total_nfe = 0

        for batch in pbar:
            # Move data to device
            x_enc, x_mark_enc, x_dec, x_mark_dec = [
                b.to(self.device) for b in batch[:4]
            ]
            y_true = batch[2][:, -self.pred_len:, :].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)

            # Handle VAE-style output (predictions, mean, logvar)
            if isinstance(output, tuple) and len(output) == 3:
                y_pred, mean, logvar = output
                recon_loss = self.criterion(y_pred, y_true)
                kl_loss = self._compute_kl_loss(mean, logvar)
                kl_weight = self._get_kl_weight()
                loss = recon_loss + kl_weight * kl_loss

                tracker.update("recon_loss", recon_loss.item(), 1)
                tracker.update("kl_loss", kl_loss.item(), 1)
            else:
                y_pred = output
                loss = self.criterion(y_pred, y_true)

            # Backward pass (adjoint method handles ODE gradients)
            loss.backward()

            # Gradient clipping (more aggressive for ODE stability)
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm,
            )

            # Update weights
            self.optimizer.step()

            # Track metrics
            tracker.update("loss", loss.item(), 1)

            # Track NFE
            if hasattr(self.model, "get_ode_func"):
                nfe = self.model.get_ode_func().nfe
                total_nfe += nfe
                self.model.get_ode_func().reset_nfe()

            pbar.set_postfix(loss=tracker.get_average("loss"))

        # Compute average NFE
        if total_nfe > 0:
            avg_nfe = total_nfe / len(self.train_loader)
            tracker.update("nfe", avg_nfe, 1)
            self.logger.info(f"Average NFE per batch: {avg_nfe:.1f}")

        return tracker.get_all_averages()

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation for ODE models.

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

            output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)

            if isinstance(output, tuple) and len(output) == 3:
                y_pred, mean, logvar = output
                loss = self.criterion(y_pred, y_true)
            else:
                y_pred = output
                loss = self.criterion(y_pred, y_true)

            tracker.update("loss", loss.item() * y_true.size(0), y_true.size(0))

        return {"val_loss": tracker.get_average("loss")}

    def fit(self) -> Dict[str, List[float]]:
        """Run full training loop for ODE models.

        Returns:
            Dictionary of training history.
        """
        history = {"train_loss": [], "val_loss": [], "nfe": []}

        for epoch in range(self.epochs):
            self.current_epoch = epoch
            self.logger.info(f"Epoch {epoch + 1}/{self.epochs}")

            if self.kl_annealing:
                kl_w = self._get_kl_weight()
                self.logger.info(f"KL weight: {kl_w:.4f}")

            # Train
            train_metrics = self.train_epoch()
            history["train_loss"].append(train_metrics["loss"])

            if "nfe" in train_metrics:
                history["nfe"].append(train_metrics["nfe"])

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
