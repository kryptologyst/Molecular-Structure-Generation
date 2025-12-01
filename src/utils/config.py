"""Configuration management for molecular structure generation project."""

from typing import Any, Dict, Optional
from pathlib import Path
import yaml
from omegaconf import OmegaConf


class Config:
    """Configuration manager for the molecular generation project."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "model": {
                "type": "graph_vae",
                "z_dim": 100,
                "hidden_size": 128,
                "max_atoms": 50,
                "atom_types": 10,
                "bond_types": 4
            },
            "training": {
                "batch_size": 64,
                "learning_rate": 0.0002,
                "num_epochs": 100,
                "beta": 1.0,
                "beta_schedule": "linear",
                "gradient_clip_val": 1.0
            },
            "data": {
                "dataset": "zinc",
                "data_path": "data/",
                "max_smiles_length": 100,
                "train_split": 0.8,
                "val_split": 0.1,
                "test_split": 0.1
            },
            "evaluation": {
                "metrics": ["validity", "uniqueness", "novelty", "qed", "sa", "logp"],
                "num_samples": 1000,
                "batch_size": 100
            },
            "device": "auto",
            "seed": 42,
            "log_level": "INFO",
            "save_dir": "checkpoints/",
            "log_dir": "logs/"
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key."""
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None) -> None:
        """Save configuration to file."""
        save_path = Path(path) if path else self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self._config.copy()
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        self._config.update(updates)
