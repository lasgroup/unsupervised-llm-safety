"""Base class and unified output format for safety classification models."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class ModelOutput:
    """Unified output format for all safety models."""

    is_safe: torch.Tensor  # Binary predictions (True=safe, False=unsafe)
    probs: torch.Tensor  # Probability of being safe [0, 1]
    loss: Optional[torch.Tensor]  # Loss value(s) if labels provided
    violations: Optional[torch.Tensor] = None  # Constraint violations
    metrics: Optional[Dict[str, Any]] = None  # Additional metrics/info

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


class BaseSafetyModel(nn.Module, ABC):
    """Base class for all safety classification models."""

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        input_in_feature_space: bool = False,
    ) -> ModelOutput:
        """Forward pass for training and inference.

        Args:
            x: Input tensor (hidden states or pre-extracted features)
            labels: Optional labels for loss computation (0=unsafe, 1=safe)
            reduction: Loss reduction mode ('mean', 'sum', 'none')
            input_in_feature_space: If True, x is pre-extracted features

        Returns:
            ModelOutput with predictions, probabilities, and optionally loss
        """
        pass

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make binary safety predictions.

        Args:
            x: Input tensor

        Returns:
            Binary predictions tensor (True=safe, False=unsafe)
        """
        pass

    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature representations.

        Args:
            x: Input tensor

        Returns:
            Feature tensor after feature extraction layer
        """
        pass
