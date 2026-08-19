import gc
import logging

import psutil
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def get_model_device(model: nn.Module) -> torch.device:
    """Get the device of a PyTorch model by checking its parameters.

    Args:
        model: PyTorch model

    Returns:
        torch.device: The device where the model is located

    Raises:
        ValueError: If the model has no parameters
    """
    try:
        return next(model.parameters()).device
    except StopIteration:
        # Model has no parameters, check buffers
        try:
            return next(model.buffers()).device
        except StopIteration:
            # Model has neither parameters nor buffers
            raise ValueError(
                "Cannot determine device: model has no parameters or buffers"
            )


def log_memory_usage(label: str) -> None:
    """Log current memory usage.

    Args:
        label: Label to identify the measurement point
    """
    gc.collect()
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024**2
    logger.info(f"[Memory - {label}] {mem_mb:.0f} MB")
