"""Reproducibility utilities for deterministic experiments."""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds for reproducibility.

    Sets seeds for Python's random, NumPy, and PyTorch.
    Optionally enables deterministic algorithms in PyTorch.

    Args:
        seed: Random seed value.
        deterministic: Whether to use deterministic algorithms (may impact performance).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # PyTorch >= 1.8
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True)
            except RuntimeError:
                # Some operations don't have deterministic implementations
                pass


def get_device(gpu_id: Optional[int] = None, use_gpu: bool = True) -> torch.device:
    """Get the appropriate device for computation.

    Args:
        gpu_id: Specific GPU ID to use (None for auto-selection).
        use_gpu: Whether to attempt using GPU.

    Returns:
        torch.device for computation.
    """
    if use_gpu and torch.cuda.is_available():
        if gpu_id is not None:
            if gpu_id < torch.cuda.device_count():
                return torch.device(f"cuda:{gpu_id}")
            else:
                print(f"Warning: GPU {gpu_id} not available, using GPU 0")
                return torch.device("cuda:0")
        return torch.device("cuda:0")

    if use_gpu and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_gpu_memory_info() -> dict:
    """Get GPU memory usage information.

    Returns:
        Dictionary with memory stats for each GPU.
    """
    if not torch.cuda.is_available():
        return {"available": False}

    info = {"available": True, "devices": []}
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        info["devices"].append({
            "id": i,
            "name": props.name,
            "total_memory_gb": props.total_memory / 1e9,
            "allocated_gb": allocated / 1e9,
            "reserved_gb": reserved / 1e9,
        })
    return info
