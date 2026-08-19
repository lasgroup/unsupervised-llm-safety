"""Mahalanobis distance-based OOD detection baseline.

Based on "A Simple Unified Framework for Detecting Out-of-Distribution
Samples and Adversarial Attacks" (Lee et al., 2018).

All fitting is done in GPU mini-batches to avoid loading the full
training dataset into VRAM at once.
"""

import logging
from typing import List, Optional

import numpy as np
import torch

from .base import AnodetMethod

logger = logging.getLogger(__name__)


class MahalanobisAnodet(AnodetMethod):
    """Mahalanobis distance-based OOD detection baseline.

    Fits a single Gaussian to the safe training features and uses the
    Mahalanobis distance to its mean as the OOD score.
    """

    _method_name = "mahalanobis"  # Class attribute for validation

    def __init__(
        self,
        regularization: float = 1e-5,
        score_type: str = "percentile",  # 'percentile' or 'sigmoid'
        batch_size: int = 1024,
        device: str = "cuda",
    ):
        """Initialize Mahalanobis detector.

        Args:
            regularization: Regularization for covariance matrix
                (default: 1e-5)
            score_type: How to map distances to scores
                ('percentile' or 'sigmoid')
            batch_size: Batch size for scoring large datasets
            device: Device for computation
        """
        super().__init__()

        # Store all init params for serialization
        self._init_config = {
            "regularization": regularization,
            "score_type": score_type,
            "batch_size": batch_size,
            "device": device,
        }

        self.regularization = regularization
        self.score_type = score_type
        self.batch_size = batch_size
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )

        # Model parameters (set during fitting)
        self.shared_precision: Optional[torch.Tensor] = None  # Sigma^(-1)
        self.global_mean: Optional[torch.Tensor] = None
        self.n_features: Optional[int] = None

        # Normalization parameters (computed during fit)
        self.dist_percentiles: Optional[List[float]] = (
            None  # For percentile-based scoring
        )
        self.dist_mean: Optional[float] = None  # For sigmoid scoring
        self.dist_std: Optional[float] = None  # For sigmoid scoring

    def fit(self, data: np.ndarray, labels: np.ndarray) -> None:
        """Fit Gaussian parameters on training data using GPU batches.

        Data stays on CPU as numpy; GPU is used only in mini-batches
        for mean, covariance, and normalization computation.

        Args:
            data: Training features of shape (n_samples, n_features)
            labels: Binary labels (1=safe, 0=unsafe)
        """
        n_samples, self.n_features = data.shape
        d = self.n_features
        safe_mask = labels == 1
        n_safe = safe_mask.sum()
        n_unsafe = (labels == 0).sum()

        logger.info(
            f"Fitting Mahalanobis detector on {n_samples} samples "
            f"with {d} features"
        )
        logger.info(
            f"  Single Gaussian mode: {n_safe} safe, "
            f"{n_unsafe} unsafe samples"
        )

        # --- Compute safe mean in GPU batches ---
        if n_safe == 0:
            logger.warning("No safe samples found, using global mean")
            mean, _ = self._compute_batched_mean_std(
                data, self.device, self.batch_size
            )
            self.global_mean = mean
        else:
            safe_data = data[safe_mask]
            mean, _ = self._compute_batched_mean_std(
                safe_data, self.device, self.batch_size
            )
            self.global_mean = mean

        # --- Compute covariance in GPU batches ---
        # Accumulate (X - mu)^T (X - mu) over all data
        cov_accum = torch.zeros(
            (d, d), device=self.device, dtype=torch.float64
        )
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            batch = torch.tensor(
                data[start:end], dtype=torch.float32, device=self.device
            )
            centered = (batch - self.global_mean).to(torch.float64)
            cov_accum += centered.T @ centered

        covariance = (cov_accum / max(n_samples - 1, 1)).to(torch.float32)

        # Regularize and invert
        self._compute_precision_matrix(covariance)

        # Compute normalization parameters in batches
        self._compute_normalization_params(data)

        self._fitted = True
        logger.info("Fitted Mahalanobis detector")

    def score_samples(self, data: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for samples.

        Scores are in [0, 1] range where:
        - 0 = very normal (close to training distribution)
        - 1 = very anomalous (far from training distribution)

        Args:
            data: Test features of shape (n_samples, n_features)

        Returns:
            Anomaly scores in [0, 1] range
        """
        self._check_fitted()

        n_samples = len(data)
        all_distances = []

        # Process in batches for memory efficiency
        for i in range(0, n_samples, self.batch_size):
            end_idx = min(i + self.batch_size, n_samples)
            batch = data[i:end_idx]

            # Convert to tensor
            batch_tensor = torch.tensor(
                batch, dtype=torch.float32, device=self.device
            )

            with torch.no_grad():
                # Compute Mahalanobis distances
                batch_distances = self._compute_min_distances(batch_tensor)
                all_distances.append(batch_distances.cpu().numpy())

        # Concatenate all distances
        distances = np.concatenate(all_distances)

        # Map distances to anomaly scores [0, 1]
        if self.score_type == "percentile":
            assert self.dist_percentiles is not None
            scores = np.clip(
                (distances - self.dist_percentiles[0])
                / (self.dist_percentiles[1] - self.dist_percentiles[0]),
                0,
                1,
            )
        else:  # sigmoid
            assert self.dist_mean is not None and self.dist_std is not None
            z_scores = (distances - self.dist_mean) / max(self.dist_std, 1e-10)
            scores = 1 / (1 + np.exp(-z_scores))

        return scores

    def _compute_precision_matrix(self, covariance: torch.Tensor) -> None:
        """Compute precision matrix (inverse covariance) with
        regularization."""

        # Check condition number before regularization
        try:
            cond_before = torch.linalg.cond(covariance).item()
            logger.info(f"  Covariance condition number: {cond_before:.2e}")
        except Exception:
            cond_before = float("inf")

        # Add regularization
        reg_covariance = covariance + self.regularization * torch.eye(
            self.n_features, device=self.device
        )

        # Compute precision matrix
        try:
            self.shared_precision = torch.linalg.inv(reg_covariance)
            if cond_before > 1e10:
                logger.info(
                    f"  Added regularization {self.regularization} to "
                    f"improve conditioning"
                )
        except torch.linalg.LinAlgError:
            logger.warning(
                "  Covariance matrix is singular, using pseudo-inverse"
            )
            self.shared_precision = torch.linalg.pinv(reg_covariance)

    def _compute_min_distances(self, data: torch.Tensor) -> torch.Tensor:
        """Compute Mahalanobis distance to the global mean."""
        assert self.shared_precision is not None
        assert self.global_mean is not None
        diff = data - self.global_mean
        dist_sq = torch.sum(diff @ self.shared_precision * diff, dim=1)
        return torch.sqrt(dist_sq + 1e-10)

    def _compute_normalization_params(self, data: np.ndarray) -> None:
        """Compute normalization parameters in GPU batches."""
        n_samples = len(data)
        all_distances = []

        with torch.no_grad():
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch = torch.tensor(
                    data[start:end],
                    dtype=torch.float32,
                    device=self.device,
                )
                dists = self._compute_min_distances(batch)
                all_distances.append(dists.cpu().numpy())

        train_distances = np.concatenate(all_distances)

        if self.score_type == "percentile":
            self.dist_percentiles = [
                np.percentile(train_distances, 5),
                np.percentile(train_distances, 95),
            ]
            logger.info(
                f"  Distance percentiles [5th, 95th]: "
                f"[{self.dist_percentiles[0]:.4f}, "
                f"{self.dist_percentiles[1]:.4f}]"
            )

            # Ensure non-zero range
            if self.dist_percentiles[1] - self.dist_percentiles[0] < 1e-10:
                logger.warning(
                    "  Very small distance range, adjusting percentiles"
                )
                self.dist_percentiles[0] = train_distances.min()
                self.dist_percentiles[1] = train_distances.max()
                if self.dist_percentiles[1] - self.dist_percentiles[0] < 1e-10:
                    self.dist_percentiles[1] = self.dist_percentiles[0] + 1.0

        else:  # sigmoid
            self.dist_mean = train_distances.mean()
            self.dist_std = train_distances.std()
            logger.info(
                f"  Distance statistics: mean={self.dist_mean:.4f}, "
                f"std={self.dist_std:.4f}"
            )

        logger.info(
            f"  Distance range: [{train_distances.min():.4f}, "
            f"{train_distances.max():.4f}]"
        )
        logger.info(
            f"  Distance quartiles: "
            f"[{np.percentile(train_distances, 25):.4f}, "
            f"{np.percentile(train_distances, 50):.4f}, "
            f"{np.percentile(train_distances, 75):.4f}]"
        )

    def _check_fitted(self):
        """Check if the model has been fitted."""
        if not hasattr(self, "_fitted") or not self._fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before scoring. "
                "Call fit() first."
            )

    @property
    def name(self) -> str:
        return "mahalanobis"

    def state_dict(self) -> dict:
        """Return complete state for serialization."""
        self._check_fitted()
        state = {
            "shared_precision": self.shared_precision,
            "global_mean": self.global_mean,
            "n_features": self.n_features,
        }

        if self.score_type == "percentile":
            state["dist_percentiles"] = self.dist_percentiles
        else:
            state["dist_mean"] = self.dist_mean
            state["dist_std"] = self.dist_std

        return state

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from dictionary with validation."""
        required_keys = ["shared_precision", "global_mean", "n_features"]
        if self.score_type == "percentile":
            required_keys.append("dist_percentiles")
        else:
            required_keys.extend(["dist_mean", "dist_std"])

        missing = set(required_keys) - set(state_dict.keys())
        if missing:
            raise KeyError(
                f"Cannot load {self.name}: missing required keys {missing}. "
                f"Available keys: {list(state_dict.keys())}"
            )

        self.shared_precision = state_dict["shared_precision"].to(self.device)
        self.global_mean = state_dict["global_mean"].to(self.device)
        self.n_features = state_dict["n_features"]

        if self.score_type == "percentile":
            self.dist_percentiles = state_dict["dist_percentiles"]
        else:
            self.dist_mean = state_dict["dist_mean"]
            self.dist_std = state_dict["dist_std"]

        self._fitted = True

    def save(self, filepath: str) -> None:
        """Save the fitted anomaly detection algorithm to disk."""
        self._check_fitted()
        checkpoint = {
            "method_name": self.name,
            "state": self.state_dict(),
            "config": self._init_config,
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def load(
        cls, filepath: str, device: Optional[str] = None
    ) -> "MahalanobisAnodet":
        """Load a anomaly detection algorithm from disk."""
        return cls._load_helper(filepath, device)
