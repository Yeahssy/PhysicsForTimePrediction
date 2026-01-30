"""Training callbacks for monitoring and checkpointing."""

from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn


class EarlyStopping:
    """Early stopping to halt training when validation loss stops improving.

    Args:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum change to qualify as an improvement.
        mode: 'min' for loss, 'max' for metrics like accuracy.
        verbose: Whether to print early stopping messages.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
        verbose: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """Check if training should stop.

        Args:
            score: Current validation score.

        Returns:
            True if training should stop.
        """
        if self.mode == "min":
            score = -score

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop

    def reset(self) -> None:
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class ModelCheckpoint:
    """Save model checkpoints based on validation performance.

    Args:
        save_dir: Directory to save checkpoints.
        filename: Checkpoint filename template.
        monitor: Metric to monitor.
        mode: 'min' or 'max'.
        save_best_only: Whether to save only the best model.
        verbose: Whether to print checkpoint messages.
    """

    def __init__(
        self,
        save_dir: str = "checkpoints",
        filename: str = "model_{epoch:03d}_{val_loss:.4f}.pt",
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
        verbose: bool = True,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.verbose = verbose

        self.best_score = None
        self.best_path = None

    def __call__(
        self,
        model: nn.Module,
        epoch: int,
        metrics: Dict[str, float],
        config: Optional[Dict] = None,
    ) -> Optional[str]:
        """Save checkpoint if conditions are met.

        Args:
            model: Model to save.
            epoch: Current epoch number.
            metrics: Dictionary of metrics.
            config: Optional config to save with checkpoint.

        Returns:
            Path to saved checkpoint or None.
        """
        score = metrics.get(self.monitor, 0.0)

        if self.mode == "min":
            is_best = self.best_score is None or score < self.best_score
        else:
            is_best = self.best_score is None or score > self.best_score

        if is_best:
            self.best_score = score

            # Format filename
            format_dict = {"epoch": epoch, **metrics}
            filename = self.filename.format(**format_dict)
            save_path = self.save_dir / filename

            # Save checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
            }
            if config is not None:
                checkpoint["config"] = config

            torch.save(checkpoint, save_path)

            if self.verbose:
                print(f"Checkpoint saved: {save_path}")

            # Remove old best if only saving best
            if self.save_best_only and self.best_path is not None:
                if self.best_path.exists() and self.best_path != save_path:
                    self.best_path.unlink()

            self.best_path = save_path
            return str(save_path)

        return None

    def load_best(self, model: nn.Module) -> Dict:
        """Load the best checkpoint into the model.

        Args:
            model: Model to load weights into.

        Returns:
            Checkpoint dictionary.
        """
        if self.best_path is None or not self.best_path.exists():
            raise ValueError("No checkpoint available to load")

        checkpoint = torch.load(self.best_path)
        model.load_state_dict(checkpoint["model_state_dict"])

        return checkpoint


class LearningRateScheduler:
    """Wrapper for PyTorch learning rate schedulers.

    Args:
        scheduler: PyTorch scheduler instance.
        monitor: Metric to monitor (for ReduceLROnPlateau).
    """

    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        monitor: str = "val_loss",
    ):
        self.scheduler = scheduler
        self.monitor = monitor
        self._is_plateau = isinstance(
            scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
        )

    def step(self, metrics: Optional[Dict[str, float]] = None) -> None:
        """Step the scheduler.

        Args:
            metrics: Dictionary of metrics (needed for ReduceLROnPlateau).
        """
        if self._is_plateau:
            if metrics is None:
                raise ValueError("Metrics required for ReduceLROnPlateau")
            self.scheduler.step(metrics[self.monitor])
        else:
            self.scheduler.step()

    def get_lr(self) -> float:
        """Get current learning rate."""
        if self._is_plateau:
            return self.scheduler.optimizer.param_groups[0]["lr"]
        return self.scheduler.get_last_lr()[0]
