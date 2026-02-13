"""
Configuration loader utility.
Loads YAML config files and provides easy access to settings.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def load_config(config_name: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_name: Name of config file (with or without .yaml extension)
    
    Returns:
        Dictionary containing configuration values
    
    Example:
        >>> config = load_config('data_config')
        >>> print(config['image']['input_size'])
        [224, 224]
    """
    if not config_name.endswith('.yaml'):
        config_name = f"{config_name}.yaml"
    
    config_path = get_project_root() / 'configs' / config_name
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def load_data_config() -> Dict[str, Any]:
    """Load data configuration."""
    return load_config('data_config')


def load_model_config() -> Dict[str, Any]:
    """Load model configuration."""
    return load_config('model_config')


def get_class_names(config: Optional[Dict] = None) -> list:
    """
    Get list of class names from config.
    
    Args:
        config: Data config dict. If None, loads from file.
    
    Returns:
        List of class names
    """
    if config is None:
        config = load_data_config()
    
    classes = config.get('classes', [])
    if isinstance(classes, list) and len(classes) > 0:
        return [c['name'] for c in classes if isinstance(c, dict)]
    return []


class Config:
    """
    Configuration manager for easy attribute-style access.
    
    Example:
        >>> cfg = Config('model_config')
        >>> print(cfg.model.name)
        'EfficientNetB3'
    """
    
    def __init__(self, config_name: str):
        self._config = load_config(config_name)
        self._wrap_dict(self._config)
    
    def _wrap_dict(self, d: Dict) -> None:
        """Recursively wrap nested dicts for attribute access."""
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigDict(value))
            else:
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return raw dictionary."""
        return self._config


class ConfigDict:
    """Helper class for nested dictionary attribute access."""
    
    def __init__(self, d: Dict):
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigDict(value))
            else:
                setattr(self, key, value)


if __name__ == "__main__":
    # Test config loading
    data_cfg = load_data_config()
    print("Data config loaded:")
    print(f"  Input size: {data_cfg['image']['input_size']}")
    print(f"  Train ratio: {data_cfg['split']['train_ratio']}")
    
    model_cfg = load_model_config()
    print("\nModel config loaded:")
    print(f"  Model: {model_cfg['model']['name']}")
    print(f"  Learning rate (phase 1): {model_cfg['training']['phase1']['learning_rate']}")
