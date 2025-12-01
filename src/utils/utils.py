"""Utility functions for molecular structure generation."""

import random
import numpy as np
import torch
from typing import Optional, Union, List, Tuple
import logging
from pathlib import Path


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device: str = "auto") -> torch.device:
    """Get the appropriate device for computation.
    
    Args:
        device: Device specification. "auto" detects best available device.
        
    Returns:
        PyTorch device object.
    """
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    return torch.device(device)


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional log file path.
        
    Returns:
        Configured logger.
    """
    logger = logging.getLogger("molecular_gen")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model.
        
    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: Union[str, Path],
    **kwargs
) -> None:
    """Save model checkpoint.
    
    Args:
        model: PyTorch model to save.
        optimizer: Optimizer state.
        epoch: Current epoch.
        loss: Current loss value.
        path: Path to save checkpoint.
        **kwargs: Additional data to save.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        **kwargs
    }
    
    torch.save(checkpoint, path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: Union[str, Path],
    device: torch.device
) -> dict:
    """Load model checkpoint.
    
    Args:
        model: PyTorch model to load state into.
        optimizer: Optional optimizer to load state into.
        path: Path to checkpoint file.
        device: Device to load checkpoint on.
        
    Returns:
        Checkpoint data dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def create_directories(base_path: Union[str, Path], subdirs: List[str]) -> None:
    """Create directory structure.
    
    Args:
        base_path: Base directory path.
        subdirs: List of subdirectories to create.
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    for subdir in subdirs:
        (base_path / subdir).mkdir(parents=True, exist_ok=True)


def format_time(seconds: float) -> str:
    """Format time duration in human-readable format.
    
    Args:
        seconds: Time duration in seconds.
        
    Returns:
        Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
