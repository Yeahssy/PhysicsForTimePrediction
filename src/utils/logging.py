"""Logging utilities for experiment tracking."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime

# Global logger instance
_logger: Optional[logging.Logger] = None


def setup_logger(
    name: str = "ts_forecast",
    log_dir: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """Set up a logger with file and console handlers.

    Args:
        name: Logger name.
        log_dir: Directory to save log files (None for no file logging).
        level: Logging level.
        console: Whether to log to console.

    Returns:
        Configured logger instance.
    """
    global _logger

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers

    # Format
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """Get the global logger instance.

    Returns:
        Logger instance (creates default if not set up).
    """
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger


class MetricTracker:
    """Track and aggregate metrics during training/evaluation."""

    def __init__(self):
        self.metrics = {}
        self.counts = {}

    def update(self, name: str, value: float, count: int = 1) -> None:
        """Update a metric with a new value.

        Args:
            name: Metric name.
            value: Metric value (should be sum if count > 1).
            count: Number of samples in this update.
        """
        if name not in self.metrics:
            self.metrics[name] = 0.0
            self.counts[name] = 0

        self.metrics[name] += value
        self.counts[name] += count

    def get_average(self, name: str) -> float:
        """Get the average value of a metric.

        Args:
            name: Metric name.

        Returns:
            Average value.
        """
        if name not in self.metrics or self.counts[name] == 0:
            return 0.0
        return self.metrics[name] / self.counts[name]

    def get_all_averages(self) -> dict:
        """Get all metric averages.

        Returns:
            Dictionary of metric averages.
        """
        return {name: self.get_average(name) for name in self.metrics}

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}


def format_metrics(metrics: dict, prefix: str = "") -> str:
    """Format metrics dictionary as a string.

    Args:
        metrics: Dictionary of metric names to values.
        prefix: Optional prefix for the output.

    Returns:
        Formatted string.
    """
    parts = []
    for name, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{name}: {value:.6f}")
        else:
            parts.append(f"{name}: {value}")

    result = " | ".join(parts)
    if prefix:
        result = f"{prefix} | {result}"
    return result
