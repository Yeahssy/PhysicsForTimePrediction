"""Model registry for dynamic model instantiation."""

from typing import Dict, Type

import torch.nn as nn

# Global model registry
MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_model(name: str):
    """Decorator to register a model class in the global registry.

    Usage:
        @register_model("informer")
        class Informer(BaseModel):
            ...

    Args:
        name: Unique identifier for the model.

    Returns:
        Decorator function.
    """
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered.")
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str, config: Dict) -> nn.Module:
    """Instantiate a model by name from the registry.

    Args:
        name: Model identifier (must be registered).
        config: Configuration dictionary for model initialization.

    Returns:
        Instantiated model.

    Raises:
        ValueError: If model name is not found in registry.
    """
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: '{name}'. Available models: {available}")
    return MODEL_REGISTRY[name](config)


def list_models() -> list:
    """Return list of all registered model names."""
    return list(MODEL_REGISTRY.keys())
