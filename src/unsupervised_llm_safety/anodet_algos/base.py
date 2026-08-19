"""Abstract base class for anomaly detection algorithms."""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Type, TypeVar

import numpy as np
import torch
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="AnodetMethod")


class AnodetMethod(ABC):
    """Abstract base class for all anomaly detection algorithms.

    All anomaly detection methods should inherit from this class and implement
    the required methods for fitting and scoring samples.
    """

    def __init__(self):
        """Initialize the anomaly detection method."""
        self._fitted = False

    @abstractmethod
    def fit(self, data: np.ndarray, labels: np.ndarray) -> None:
        """Fit the anomaly detection method on training data.

        Args:
            data: Training data array of shape (n_samples, n_features)
            labels: Training labels of shape (n_samples,) where:
                   0 = unsafe, 1 = safe
        """
        pass

    @abstractmethod
    def score_samples(self, data: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for samples.

        Args:
            data: Data to score of shape (n_samples, n_features)

        Returns:
            Anomaly scores of shape (n_samples,) in range [0, 1]
            where higher values indicate more anomalous samples
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this anomaly detection method.

        This name is used for logging, saving results, and identifying
        the method in outputs.

        Returns:
            A string identifier for this method (e.g., "one_class_svm")
        """
        pass

    @property
    def fitted(self) -> bool:
        """Check if the method has been fitted.

        Returns:
            True if fit() has been called, False otherwise
        """
        return self._fitted

    def _check_fitted(self) -> None:
        """Check if the method has been fitted, raise error if not."""
        if not self._fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted. "
                "Call fit() before score_samples()."
            )

    def post_training_callback(
        self, save_path: str, cfg: DictConfig, device: torch.device
    ) -> None:
        """Optional callback executed after training and saving the model.

        This method can be overridden by subclasses to perform method-specific
        post-training analysis or processing. By default, does nothing.

        Args:
            save_path: Path where the trained model was saved
            cfg: Configuration dictionary
            device: Device used for training
        """
        pass

    @staticmethod
    def _compute_batched_mean_std(
        data: np.ndarray,
        device: torch.device,
        batch_size: int = 1024,
        regularization: float = 1e-8,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute feature mean and std in GPU mini-batches.

        Avoids loading the full dataset onto GPU at once. Uses a
        numerically stable two-pass approach: pass 1 for mean,
        pass 2 for variance.

        Args:
            data: Numpy array of shape (n_samples, n_features).
            device: Torch device for computation.
            batch_size: Rows per GPU batch.
            regularization: Added to std for numerical stability.

        Returns:
            (mean, std) tensors on *device*, each of shape (n_features,).
        """
        n_samples, n_features = data.shape

        # Pass 1: compute mean
        running_sum = torch.zeros(
            n_features, device=device, dtype=torch.float64
        )
        for start in range(0, n_samples, batch_size):
            batch = torch.tensor(
                data[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            running_sum += batch.to(torch.float64).sum(dim=0)
        mean = (running_sum / n_samples).to(torch.float32)

        # Pass 2: compute variance around the mean
        running_var = torch.zeros(
            n_features, device=device, dtype=torch.float64
        )
        for start in range(0, n_samples, batch_size):
            batch = torch.tensor(
                data[start : start + batch_size],
                dtype=torch.float32,
                device=device,
            )
            diff = batch - mean
            running_var += diff.to(torch.float64).pow(2).sum(dim=0)
        std = (running_var / n_samples).to(
            torch.float32
        ).sqrt() + regularization

        logger.debug(
            f"Batched stats: {n_samples} samples, "
            f"mean range [{mean.min():.4f}, {mean.max():.4f}], "
            f"std range [{std.min():.4f}, {std.max():.4f}]"
        )
        return mean, std

    @classmethod
    def _load_helper(
        cls: Type[T],
        filepath: str,
        device: Optional[str] = None,
        method_name: Optional[str] = None,
    ) -> T:
        """Common load functionality for all anomaly detection methods.

        Args:
            filepath: Path to the saved model file
            device: Device to load the model to (None = keep original)
            method_name: Expected method name for validation (uses cls._method_name if None)

        Returns:
            Loaded instance of the anomaly detection method
        """
        checkpoint = torch.load(
            filepath, map_location="cpu", weights_only=False
        )

        # Validate method name
        expected_name = method_name or getattr(
            cls, "_method_name", cls.__name__.lower()
        )
        if checkpoint["method_name"] != expected_name:
            raise ValueError(
                f"Cannot load {checkpoint['method_name']} checkpoint into "
                f"{expected_name} class"
            )

        # Get config and override device if specified.
        #
        # Every class that reaches here with a non-None *device* accepts a
        # ``device`` constructor argument: CVDD, DeepSVDD, GMM and
        # Mahalanobis all declare one. One-Class SVM is the single loader
        # whose ``__init__`` does not, and its only caller
        # (``run_anodet_eval.load_baseline_from_checkpoint``) special-cases
        # it and passes ``device=None``, so the skip is unreachable.
        config = checkpoint["config"].copy()
        if device is not None:
            config["device"] = device

        # Create instance and load state
        instance = cls(**config)
        if hasattr(instance, "load_state_dict"):
            instance.load_state_dict(checkpoint["state"])
        else:
            raise NotImplementedError(
                f"{cls.__name__} must implement load_state_dict method"
            )

        return instance
