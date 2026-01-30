"""Configuration loading and management utilities."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Supports inheritance via '_base_' key which can be a single path
    or list of paths to parent configs.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Merged configuration dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # Handle inheritance
    if "_base_" in config:
        base_paths = config.pop("_base_")
        if isinstance(base_paths, str):
            base_paths = [base_paths]

        # Load and merge base configs
        base_config = {}
        for base_path in base_paths:
            # Resolve relative paths from current config's directory
            if not os.path.isabs(base_path):
                base_path = config_path.parent / base_path
            parent_config = load_config(base_path)
            base_config = merge_configs(base_config, parent_config)

        # Merge current config on top of base
        config = merge_configs(base_config, config)

    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two configuration dictionaries.

    Override values take precedence over base values.
    Nested dictionaries are merged recursively.

    Args:
        base: Base configuration dictionary.
        override: Override configuration dictionary.

    Returns:
        Merged configuration dictionary.
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """Save configuration to a YAML file.

    Args:
        config: Configuration dictionary to save.
        save_path: Path to save the configuration.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def flatten_config(config: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten nested config dictionary for logging.

    Args:
        config: Nested configuration dictionary.
        prefix: Prefix for flattened keys.

    Returns:
        Flattened dictionary with dot-separated keys.
    """
    items = {}
    for key, value in config.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            items.update(flatten_config(value, new_key))
        else:
            items[new_key] = value
    return items


def get_nested(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get a value from nested config using dot-separated path.

    Args:
        config: Configuration dictionary.
        key_path: Dot-separated path (e.g., "model.d_model").
        default: Default value if path not found.

    Returns:
        Value at the path or default.
    """
    keys = key_path.split(".")
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value
