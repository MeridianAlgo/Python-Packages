"""
Configuration settings for the MeridianAlgo package.
"""

import os
from typing import Any, Dict

# Base configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    "data_provider": "yahoo",  # Default data provider
    "cache_enabled": True,  # Enable caching of API responses
    "cache_dir": os.path.join(os.path.expanduser("~"), ".meridianalgo", "cache"),
    "log_level": "INFO",  # Default log level
    "max_workers": None,  # Max workers for parallel processing (None = auto)
    "risk_free_rate": 0.0,  # Default risk-free rate for calculations
    "default_currency": "USD",  # Default currency for financial calculations
}

# Current configuration
current_config = DEFAULT_CONFIG.copy()


def set_config(**kwargs) -> None:
    """Update the current configuration with new values.

    Args:
        **kwargs: Configuration key-value pairs to update
    """
    current_config.update(kwargs)


def get_config(key: str = None, default: Any = None) -> Any:
    """Get a configuration value.

    Args:
        key: Configuration key to retrieve. If None, returns entire config.
        default: Default value to return if key is not found.

    Returns:
        The configuration value or the entire config if key is None.
    """
    if key is None:
        return current_config
    return current_config.get(key, default)


def reset_config() -> None:
    """Reset configuration to default values."""
    current_config.clear()
    current_config.update(DEFAULT_CONFIG)
