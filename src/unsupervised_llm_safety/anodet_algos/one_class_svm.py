"""PyTorch implementation of One-Class SVM baseline for anomaly detection.

Fit uses GPU mini-batches for scaler statistics, then loads only the
(subsampled) training data onto GPU for kernel computation and
projected gradient descent.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .base import AnodetMethod

logger = logging.getLogger(__name__)


class OneClassSVM(nn.Module):
    """PyTorch implementation of One-Class SVM."""

    def __init__(
        self,
        n_features: int,
        nu: float = 0.1,
        gamma: float = 0.1,
        kernel: str = "rbf",
    ):
        super().__init__()
        self.nu = nu
        self.gamma = gamma
        self.kernel = kernel
        self.n_features = n_features

        # Support vectors will be stored after training
        self.register_buffer("support_vectors", torch.empty(0, n_features))
        self.register_buffer("alphas", torch.empty(0))
        self.register_buffer("rho", torch.zeros(1))

    def rbf_kernel(self, X, Y):
        """Compute RBF kernel between X and Y."""
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2<x, y>
        X_norm = (X**2).sum(dim=1).view(-1, 1)
        Y_norm = (Y**2).sum(dim=1).view(1, -1)
        XY = torch.mm(X, Y.t())
        dists = X_norm + Y_norm - 2 * XY
        return torch.exp(-self.gamma * dists)

    def linear_kernel(self, X, Y):
        """Compute linear kernel between X and Y.

        Linear kernel: K(x, y) = <x, y> = x^T y

        Args:
            X: Tensor of shape (n_samples_X, n_features)
            Y: Tensor of shape (n_samples_Y, n_features)

        Returns:
            Kernel matrix of shape (n_samples_X, n_samples_Y)
        """
        return torch.mm(X, Y.t())

    def compute_kernel(self, X, Y, batch_size=None):
        """Compute kernel matrix based on selected kernel type."""
        if self.kernel == "rbf":
            if batch_size and X.shape[0] * Y.shape[0] > 1e7:
                return self.rbf_kernel_batch(X, Y, batch_size)
            else:
                return self.rbf_kernel(X, Y)
        elif self.kernel == "linear":
            # Linear kernel is memory-efficient, usually no batching needed
            return self.linear_kernel(X, Y)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def rbf_kernel_batch(self, X, Y, batch_size=1000):
        """Compute RBF kernel between X and Y in batches to save memory."""
        n_x, n_y = X.shape[0], Y.shape[0]
        K = torch.zeros(n_x, n_y, device=X.device)

        for i in range(0, n_x, batch_size):
            batch_x = X[i : i + batch_size]
            X_norm = (batch_x**2).sum(dim=1).view(-1, 1)

            for j in range(0, n_y, batch_size):
                batch_y = Y[j : j + batch_size]
                Y_norm = (batch_y**2).sum(dim=1).view(1, -1)
                XY = torch.mm(batch_x, batch_y.t())
                dists = X_norm + Y_norm - 2 * XY
                K[i : i + batch_size, j : j + batch_size] = torch.exp(
                    -self.gamma * dists
                )

        return K

    def decision_function(self, X, kernel_batch_size=None):
        """Compute decision function values."""
        K = self.compute_kernel(X, self.support_vectors, kernel_batch_size)
        return torch.matmul(K, self.alphas) - self.rho


class OneClassSVMAnodet(AnodetMethod):
    """PyTorch-based One-Class SVM baseline for anomaly detection.

    This implementation uses SGD to optimize the One-Class SVM objective,
    making it suitable for GPU acceleration without external dependencies.
    """

    _method_name = "one_class_svm"  # Class attribute for validation

    def __init__(
        self,
        nu: float = 0.1,
        kernel: str = "rbf",
        gamma: float = 0.1,
        n_epochs: int = 100,
        batch_size: int = 256,
        lr: float = 0.01,
        kernel_batch_size: int = 1000,
        max_support_vectors: Optional[int] = None,
        subsample_size: Optional[int] = None,
    ):
        """Initialize One-Class SVM with specified parameters.

        Args:
            nu: An upper bound on the fraction of training errors and a lower
                bound of the fraction of support vectors. Should be in (0, 1].
            kernel: Specifies the kernel type ('rbf' or 'linear').
            gamma: Kernel coefficient for 'rbf'.
            n_epochs: Number of training epochs.
            batch_size: Batch size for training.
            lr: Learning rate for optimization.
            kernel_batch_size: Batch size for kernel computation to save memory.
            max_support_vectors: Maximum number of support vectors to keep.
            subsample_size: If set, randomly subsample training data to this size.
        """
        super().__init__()

        # Store all init params for serialization
        self._init_config = {
            "nu": nu,
            "kernel": kernel,
            "gamma": gamma,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "lr": lr,
            "kernel_batch_size": kernel_batch_size,
            "max_support_vectors": max_support_vectors,
            "subsample_size": subsample_size,
        }

        if kernel not in ["rbf", "linear"]:
            raise NotImplementedError(
                f"Kernel '{kernel}' not implemented. Only 'rbf' and 'linear' are supported."
            )

        # Log warning if gamma is specified for linear kernel
        if kernel == "linear" and gamma != 0.1:  # 0.1 is default
            logger.info(
                f"Gamma parameter ({gamma}) is ignored for linear kernel"
            )

        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.kernel_batch_size = kernel_batch_size
        self.max_support_vectors = max_support_vectors
        self.subsample_size = subsample_size

        # Auto-detect device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Using device: {self.device}")

        self.model: Optional[OneClassSVM] = None
        self.scaler_mean: Optional[torch.Tensor] = None
        self.scaler_std: Optional[torch.Tensor] = None

        # Training statistics for score normalization
        self._train_score_mean: float = 0.0
        self._train_score_std: float = 1.0

    def _project_simplex_box(
        self,
        alpha: torch.Tensor,
        upper_bound: float,
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        """Project alpha onto the intersection of simplex and box constraints.

        Constraints:
            - sum(alpha) = 1  (simplex equality)
            - 0 <= alpha_i <= upper_bound  (box constraints)

        Uses iterative Bregman/Dykstra-style projection:
        1. Find threshold theta such that sum(clip(alpha - theta, 0, upper_bound)) = 1
        2. This is solved via bisection search

        Args:
            alpha: Input tensor to project
            upper_bound: Upper bound for box constraint (1 / (nu * n))
            max_iter: Maximum iterations for bisection
            tol: Convergence tolerance

        Returns:
            Projected alpha satisfying both constraints
        """
        n = alpha.shape[0]

        # Check if projection is feasible: need n * upper_bound >= 1
        if n * upper_bound < 1 - tol:
            raise ValueError(
                f"Infeasible constraints: n={n}, upper_bound={upper_bound}, "
                f"n * upper_bound = {n * upper_bound} < 1"
            )

        # Bisection to find threshold theta such that
        # sum(clip(alpha - theta, 0, upper_bound)) = 1
        theta_low = alpha.min().item() - 1
        theta_high = alpha.max().item() + 1

        for _ in range(max_iter):
            theta = (theta_low + theta_high) / 2
            alpha_proj = torch.clamp(alpha - theta, 0, upper_bound)
            proj_sum = alpha_proj.sum().item()

            if abs(proj_sum - 1) < tol:
                return alpha_proj
            elif proj_sum > 1:
                # Need to shift down more (increase theta)
                theta_low = theta
            else:
                # Need to shift down less (decrease theta)
                theta_high = theta

        # Return best effort after max iterations.
        # Do NOT renormalise and re-clamp: dividing by sum can push values
        # above upper_bound, and the subsequent clamp would break sum=1.
        # After max_iter bisection steps the residual is < tol * 2^{-max_iter}
        # which is negligible in practice.
        theta = (theta_low + theta_high) / 2
        alpha_proj = torch.clamp(alpha - theta, 0, upper_bound)
        return alpha_proj

    def fit(self, data: np.ndarray, labels: np.ndarray) -> None:
        """Fit One-Class SVM on safe (normal) samples using projected gradient descent.

        The One-Class SVM dual problem is:
            min_alpha  (1/2) * alpha^T K alpha
            s.t.       sum(alpha) = 1
                       0 <= alpha_i <= 1/(nu * n)

        We solve this via projected gradient descent with exact projection
        onto the simplex-box constraint set.

        Scaler statistics are computed in GPU mini-batches to avoid
        loading the full dataset into VRAM.  Only the (subsampled) data
        is moved to GPU for kernel computation and optimisation.

        Args:
            data: Training data array of shape (n_samples, n_features)
            labels: Training labels of shape (n_samples,) where:
                   0 = unsafe (anomaly), 1 = safe (normal)
        """
        # Filter safe samples (label=1) for One-Class SVM training
        safe_mask = labels == 1
        safe_data = data[safe_mask]

        if len(safe_data) == 0:
            raise ValueError(
                "No safe samples (label=1) found in training data"
            )

        # Subsample if requested
        if self.subsample_size and len(safe_data) > self.subsample_size:
            indices = np.random.choice(
                len(safe_data), self.subsample_size, replace=False
            )
            safe_data = safe_data[indices]
            logger.info(f"Subsampled to {self.subsample_size} samples")

        logger.info(
            f"Training One-Class SVM on {len(safe_data)} safe samples "
            f"out of {len(data)} total samples"
        )

        # Compute scaler stats in GPU batches (avoids full-data GPU load)
        self.scaler_mean, self.scaler_std = self._compute_batched_mean_std(
            safe_data, self.device, self.batch_size, 1e-8
        )

        # Standardize on CPU
        mean_np = self.scaler_mean.cpu().numpy()
        std_np = self.scaler_std.cpu().numpy()
        safe_data = (safe_data - mean_np) / std_np

        n_samples, n_features = safe_data.shape

        # Check feasibility: need n * upper_bound >= 1, i.e., nu <= 1
        if self.nu > 1:
            raise ValueError(f"nu must be <= 1, got {self.nu}")

        # Subsample for kernel memory if needed (BEFORE GPU load)
        kernel_memory_gb = (n_samples**2 * 4) / 1e9
        max_kernel_memory_gb = 8.0

        if kernel_memory_gb > max_kernel_memory_gb:
            logger.warning(
                f"Full kernel matrix would require {kernel_memory_gb:.2f} GB, "
                f"exceeding threshold of {max_kernel_memory_gb:.2f} GB. "
                f"Using subsampling approach instead."
            )
            max_samples_for_kernel = int(
                np.sqrt(max_kernel_memory_gb * 1e9 / 4)
            )
            subsample_indices = np.random.choice(
                n_samples,
                min(max_samples_for_kernel, n_samples),
                replace=False,
            )
            safe_data = safe_data[subsample_indices]
            n_samples = len(safe_data)

        # NOW load the (possibly subsampled) data to GPU
        X = torch.tensor(safe_data, dtype=torch.float32, device=self.device)

        # Initialize model
        self.model = OneClassSVM(
            n_features, nu=self.nu, gamma=self.gamma, kernel=self.kernel
        ).to(self.device)

        # Compute upper bound for box constraint
        upper_bound = 1.0 / (self.nu * n_samples)

        logger.info(
            f"Precomputing kernel matrix ({kernel_memory_gb:.2f} GB)..."
        )
        assert self.model is not None  # For mypy
        K_full = self.model.compute_kernel(
            X,
            X,
            batch_size=(
                self.kernel_batch_size if self.kernel == "rbf" else None
            ),
        )

        # Initialize alpha on the feasible set
        # Start with uniform distribution (satisfies sum=1 and box constraints if nu <= 1)
        alpha = torch.ones(n_samples, device=self.device) / n_samples

        # Verify initial point is feasible
        assert alpha.sum().item() - 1.0 < 1e-6, "Initial alpha must sum to 1"
        assert (
            alpha <= upper_bound + 1e-6
        ).all(), "Initial alpha must satisfy box constraints"

        # Projected gradient descent
        # Objective: f(alpha) = 0.5 * alpha^T K alpha
        # Gradient: grad_f = K @ alpha

        best_alpha = alpha.clone()
        best_obj = float("inf")

        for epoch in tqdm(range(self.n_epochs), desc="Training One-Class SVM"):
            # Compute gradient: grad = K @ alpha
            grad = torch.mv(K_full, alpha)

            # Compute current objective for monitoring
            obj = 0.5 * torch.dot(alpha, grad).item()

            if obj < best_obj:
                best_obj = obj
                best_alpha = alpha.clone()

            # Gradient descent step
            alpha_new = alpha - self.lr * grad

            # Project onto simplex-box constraint set
            alpha = self._project_simplex_box(alpha_new, upper_bound)

            # Optional: adaptive learning rate decay
            if (epoch + 1) % 50 == 0:
                logger.debug(f"Epoch {epoch+1}: objective = {obj:.6f}")

        # Use best alpha found
        alpha = best_alpha

        logger.info(f"Final objective value: {best_obj:.6f}")

        # Store support vectors (those with non-zero alphas)
        sv_threshold = 1e-5
        sv_mask = alpha > sv_threshold
        sv_indices = torch.where(sv_mask)[0]

        # Limit support vectors if requested
        if (
            self.max_support_vectors
            and len(sv_indices) > self.max_support_vectors
        ):
            _, top_indices = torch.topk(alpha, self.max_support_vectors)
            sv_mask = torch.zeros_like(alpha, dtype=torch.bool)
            sv_mask[top_indices] = True
            logger.info(
                f"Limited support vectors to {self.max_support_vectors}"
            )

        assert self.model is not None  # For mypy
        self.model.support_vectors = X[sv_mask].clone()
        self.model.alphas = alpha[sv_mask].clone()

        # Compute rho (bias term) using KKT conditions
        # For support vectors strictly inside the box (0 < alpha < upper_bound),
        # they lie on the margin, so: sum_j alpha_j K(x_i, x_j) = rho
        margin_threshold = 1e-5
        margin_mask = (alpha > margin_threshold) & (
            alpha < upper_bound - margin_threshold
        )
        margin_indices = torch.where(margin_mask)[0]

        if len(margin_indices) > 0:
            # Use margin support vectors to compute rho
            K_margin = self.model.compute_kernel(
                X[margin_indices],
                self.model.support_vectors,
                batch_size=(
                    self.kernel_batch_size if self.kernel == "rbf" else None
                ),
            )
            rho_values = torch.mv(K_margin, self.model.alphas)
            self.model.rho = torch.mean(rho_values)
            logger.info(
                f"Computed rho from {len(margin_indices)} margin support vectors"
            )
        else:
            # Fallback: use median of decision values on all support vectors
            logger.warning(
                "No margin support vectors found (all at box boundaries). "
                "Using median of SV decision values for rho."
            )
            K_sv = self.model.compute_kernel(
                self.model.support_vectors,
                self.model.support_vectors,
                batch_size=(
                    self.kernel_batch_size if self.kernel == "rbf" else None
                ),
            )
            decision_values = torch.mv(K_sv, self.model.alphas)
            self.model.rho = torch.median(decision_values)

        # Store normalization statistics from training data for score_samples
        assert self.model is not None  # For mypy
        with torch.no_grad():
            train_scores = self.model.decision_function(
                X, self.kernel_batch_size
            )
            self._train_score_mean = train_scores.mean().item()
            self._train_score_std = train_scores.std().item() + 1e-8

        self._fitted = True

        logger.info(
            f"One-Class SVM trained with {len(self.model.support_vectors)} support vectors "
            f"({len(self.model.support_vectors)/n_samples*100:.1f}% of training samples)"
        )

    def score_samples(self, data: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for samples.

        Args:
            data: Data to score of shape (n_samples, n_features)

        Returns:
            Anomaly scores of shape (n_samples,) in range [0, 1]
            where higher values indicate more anomalous samples
        """
        self._check_fitted()
        assert self.model is not None

        # Convert to PyTorch tensor
        X = torch.tensor(data, dtype=torch.float32, device=self.device)

        # Standardize using the fitted scaler
        X = (X - self.scaler_mean) / self.scaler_std

        # Process in batches for memory efficiency
        scores_list = []

        for i in tqdm(
            range(0, len(X), self.batch_size),
            desc="Computing anomaly scores",
            leave=False,
            disable=len(X) <= self.batch_size,
        ):
            batch = X[i : i + self.batch_size]
            with torch.no_grad():
                scores_batch = self.model.decision_function(
                    batch, self.kernel_batch_size
                )
                scores_list.append(scores_batch)

        scores = torch.cat(scores_list)
        scores_np = scores.cpu().numpy()

        # Convert to anomaly scores in [0, 1] using TRAINING statistics
        # Negative decision values = anomalies, so we negate
        # Then standardize using training distribution and apply sigmoid
        scores_negated = -scores_np

        # Standardize using training statistics (computed during fit)
        # This ensures consistent scoring across different test batches
        scores_standardized = (
            scores_negated - (-self._train_score_mean)
        ) / self._train_score_std

        # Apply sigmoid to map to [0, 1]
        # Shift so that average training sample gets score ~0.5
        scores_normalized = 1 / (1 + np.exp(-scores_standardized))

        return scores_normalized

    @property
    def name(self) -> str:
        """Return the name of this anomaly detection algorithm."""
        return "one_class_svm"

    def state_dict(self) -> dict:
        """Return complete state for serialization."""
        self._check_fitted()
        assert self.model is not None
        return {
            "model_state_dict": self.model.state_dict(),
            "scaler_mean": self.scaler_mean,
            "scaler_std": self.scaler_std,
            "train_score_mean": self._train_score_mean,
            "train_score_std": self._train_score_std,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from dictionary with validation."""
        required_keys = ["model_state_dict", "scaler_mean", "scaler_std"]
        missing = set(required_keys) - set(state_dict.keys())
        if missing:
            raise KeyError(
                f"Cannot load {self.name}: missing required keys {missing}. "
                f"Available keys: {list(state_dict.keys())}"
            )

        if self.model is None:
            n_features = state_dict["scaler_mean"].shape[0]
            self.model = OneClassSVM(
                n_features=n_features,
                nu=self.nu,
                gamma=self.gamma,
                kernel=self.kernel,
            ).to(self.device)

        # Directly assign the buffers from state dict
        model_state = state_dict["model_state_dict"]
        for key, value in model_state.items():
            if hasattr(self.model, key):
                setattr(self.model, key, value.to(self.device))

        self.scaler_mean = state_dict["scaler_mean"].to(self.device)
        self.scaler_std = state_dict["scaler_std"].to(self.device)

        # Load training score statistics (with fallback for old checkpoints)
        self._train_score_mean = state_dict.get("train_score_mean", 0.0)
        self._train_score_std = state_dict.get("train_score_std", 1.0)

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
    ) -> "OneClassSVMAnodet":
        """Load a anomaly detection algorithm from disk."""
        return cls._load_helper(filepath, device)
