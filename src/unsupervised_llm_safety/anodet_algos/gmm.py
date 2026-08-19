"""Gaussian Mixture Model (GMM) based anomaly detection baseline.

This method assumes features follow a mixture of Gaussian distributions
and uses negative log-likelihood as the anomaly score.

All fitting is done in GPU mini-batches to avoid loading the full
training dataset (~3M samples) into VRAM at once.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .base import AnodetMethod

logger = logging.getLogger(__name__)

# Constants for numerical stability
REGULARIZATION = 1e-6
MIN_COMPONENT_SIZE = 10


class GMMAnodet(AnodetMethod):
    """GMM-based anomaly detection baseline.

    Fits a Gaussian Mixture Model on safe samples and uses negative
    log-likelihood as anomaly score.
    """

    _method_name = "gmm"  # Class attribute for validation

    def __init__(
        self,
        n_components: int = 5,
        em_max_iter: int = 100,
        em_tol: float = 1e-4,
        init_method: str = "kmeans++",
        score_normalization: str = "percentile",
        batch_size: int = 1024,
        device: Optional[str] = None,
    ):
        """Initialize GMM detector.

        Args:
            n_components: Number of Gaussian components
            em_max_iter: Maximum iterations for EM algorithm
            em_tol: Convergence tolerance for EM algorithm
            init_method: Initialization method ("kmeans++" or "random")
            score_normalization: How to normalize scores ("percentile" or "sigmoid")
            batch_size: Batch size for scoring
            device: Device for computation (auto-detect if None)
        """
        super().__init__()

        # Store all init params for serialization
        self._init_config = {
            "n_components": n_components,
            "em_max_iter": em_max_iter,
            "em_tol": em_tol,
            "init_method": init_method,
            "score_normalization": score_normalization,
            "batch_size": batch_size,
            "device": device,
        }

        self.n_components = n_components
        self.em_max_iter = em_max_iter
        self.em_tol = em_tol
        self.init_method = init_method
        self.score_normalization = score_normalization
        self.batch_size = batch_size

        # Auto-detect device if not specified
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        # Model parameters (set during fitting)
        self.means: Optional[List[torch.Tensor]] = None
        self.inv_stds: Optional[List[torch.Tensor]] = None
        self.log_dets: Optional[List[torch.Tensor]] = None
        self.log_mixture_weights: Optional[torch.Tensor] = None
        self.n_features: Optional[int] = None

        # Standardization parameters
        self.feature_mean: Optional[torch.Tensor] = None
        self.feature_std: Optional[torch.Tensor] = None

        # Score normalization parameters
        self.score_percentiles: Optional[Tuple[float, float]] = None
        self.score_mean: Optional[float] = None
        self.score_std: Optional[float] = None

    @property
    def name(self) -> str:
        """Return method name."""
        return f"gmm_k{self.n_components}"

    def fit(self, data: np.ndarray, labels: np.ndarray) -> None:
        """Fit GMM on safe (normal) samples.

        Data stays on CPU as numpy; GPU is used only in mini-batches
        for statistics, EM steps, and normalization.

        Args:
            data: Training features of shape (n_samples, n_features).
            labels: Binary labels (1=safe, 0=unsafe).
        """
        safe_mask = labels == 1
        data = data[safe_mask]

        if len(data) == 0:
            raise ValueError(
                "No safe samples (label=1) found in training data"
            )

        n_samples, self.n_features = data.shape

        if n_samples < self.n_components * MIN_COMPONENT_SIZE:
            raise ValueError(
                f"Not enough samples ({n_samples}) for "
                f"{self.n_components} components"
            )

        logger.info(
            f"Fitting GMM with {self.n_components} components on "
            f"{n_samples} safe samples (filtered from {len(safe_mask)} total)"
        )

        # Compute mean/std in GPU batches, then standardize on CPU
        self.feature_mean, self.feature_std = self._compute_batched_mean_std(
            data, self.device, self.batch_size, REGULARIZATION
        )
        mean_np = self.feature_mean.cpu().numpy()
        std_np = self.feature_std.cpu().numpy()
        data_normalized = (data - mean_np) / std_np

        # Fit GMM using batched EM algorithm
        self._fit_gmm_em(data_normalized)

        # Compute score normalization parameters in batches
        self._compute_normalization_params(data_normalized)

        self._fitted = True
        logger.info("GMM fitting completed")

    def score_samples(self, data: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for samples.

        Args:
            data: Features of shape (n_samples, n_features)

        Returns:
            Anomaly scores in [0, 1] range where higher = more anomalous
        """
        self._check_fitted()

        n_samples = len(data)
        scores = np.zeros(n_samples)

        # Process in batches
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch = data[start_idx:end_idx]

            # Convert to tensor and standardize
            batch_tensor = torch.tensor(
                batch, dtype=torch.float32, device=self.device
            )
            batch_normalized = (
                batch_tensor - self.feature_mean
            ) / self.feature_std

            # Compute negative log-likelihood under GMM
            log_probs = self._compute_log_probabilities(batch_normalized)
            raw_scores = -log_probs

            # Normalize scores to [0, 1]
            normalized_scores = self._normalize_scores(raw_scores)
            scores[start_idx:end_idx] = normalized_scores.cpu().numpy()

        return scores

    def state_dict(self) -> Dict:
        """Return state dict for saving."""
        state = {
            "means": [m.cpu() for m in self.means] if self.means else [],
            "inv_stds": (
                [s.cpu() for s in self.inv_stds] if self.inv_stds else []
            ),
            "log_dets": (
                [d.cpu() for d in self.log_dets] if self.log_dets else []
            ),
            "log_mixture_weights": (
                self.log_mixture_weights.cpu()
                if self.log_mixture_weights is not None
                else None
            ),
            "n_features": self.n_features,
            "feature_mean": (
                self.feature_mean.cpu()
                if self.feature_mean is not None
                else None
            ),
            "feature_std": (
                self.feature_std.cpu()
                if self.feature_std is not None
                else None
            ),
            "score_percentiles": self.score_percentiles,
            "score_mean": self.score_mean,
            "score_std": self.score_std,
            "init_config": self._init_config,
        }
        return state

    def load_state_dict(self, state_dict: Dict) -> None:
        """Load state dict."""
        # Restore parameters
        device = self.device
        self.means = [m.to(device) for m in state_dict["means"]]
        self.inv_stds = [s.to(device) for s in state_dict["inv_stds"]]
        self.log_dets = [d.to(device) for d in state_dict["log_dets"]]
        self.log_mixture_weights = state_dict["log_mixture_weights"].to(device)
        self.n_features = state_dict["n_features"]
        self.feature_mean = state_dict["feature_mean"].to(device)
        self.feature_std = state_dict["feature_std"].to(device)
        self.score_percentiles = state_dict["score_percentiles"]
        self.score_mean = state_dict["score_mean"]
        self.score_std = state_dict["score_std"]
        self._fitted = True

    # ------------------------------------------------------------------
    # Batched EM internals
    # ------------------------------------------------------------------

    def _fit_gmm_em(self, data: np.ndarray) -> None:
        """Fit GMM using EM algorithm on CPU numpy data.

        GPU is used in mini-batches for E-step / M-step computation.
        Responsibilities are stored on CPU between steps.
        """
        n_samples, d = data.shape

        # Initialize centers
        if self.init_method == "kmeans++":
            initial_means = self._kmeans_plus_plus_init(data)
        else:
            indices = np.random.permutation(n_samples)[: self.n_components]
            initial_means = [
                torch.tensor(
                    data[idx], dtype=torch.float32, device=self.device
                )
                for idx in indices
            ]

        # Initialize per-component parameters on GPU
        # Compute global std in batches for initialization
        _, global_std = self._compute_batched_mean_std(
            data, self.device, self.batch_size, REGULARIZATION
        )

        means = [m.clone() for m in initial_means]
        inv_stds = [1.0 / global_std for _ in range(self.n_components)]
        log_dets = [
            torch.sum(torch.log(global_std**2))
            for _ in range(self.n_components)
        ]

        log_mixture_weights = torch.log(
            torch.ones(self.n_components, device=self.device)
            / self.n_components
        )

        # EM iterations
        prev_log_likelihood = float("-inf")

        for iter_num in range(self.em_max_iter):
            # E-step (batched): responsibilities on CPU
            responsibilities, log_likelihood = self._e_step_batched(
                data, means, inv_stds, log_dets, log_mixture_weights
            )

            # Check convergence
            if iter_num > 0:
                improvement = (log_likelihood - prev_log_likelihood) / abs(
                    prev_log_likelihood
                )
                if improvement < self.em_tol:
                    logger.info(
                        f"EM converged after {iter_num + 1} iterations"
                    )
                    break

            prev_log_likelihood = log_likelihood

            # M-step (batched, two-pass)
            means, inv_stds, log_dets, log_mixture_weights = (
                self._m_step_batched(data, responsibilities)
            )

            if (iter_num + 1) % 10 == 0:
                logger.debug(
                    f"EM iteration {iter_num + 1}, "
                    f"log-likelihood: {log_likelihood:.2f}"
                )

        else:
            logger.info(f"EM reached maximum iterations ({self.em_max_iter})")

        # Store final parameters
        self.means = means
        self.inv_stds = inv_stds
        self.log_dets = log_dets
        self.log_mixture_weights = log_mixture_weights

    def _kmeans_plus_plus_init(self, data: np.ndarray) -> List[torch.Tensor]:
        """Initialize centers using k-means++ with batched distances."""
        n_samples = len(data)

        # Choose first center randomly
        first_idx = np.random.randint(n_samples)
        centers = [
            torch.tensor(
                data[first_idx], dtype=torch.float32, device=self.device
            )
        ]

        # Track min squared distances on CPU
        min_dists_sq = np.full(n_samples, np.inf, dtype=np.float32)

        for _ in range(1, self.n_components):
            new_center = centers[-1]  # on GPU

            # Update min distances in batches
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch = torch.tensor(
                    data[start:end],
                    dtype=torch.float32,
                    device=self.device,
                )
                # Squared distance to newest center
                dists = (batch - new_center).pow(2).sum(dim=1)
                dists_cpu = dists.cpu().numpy()
                min_dists_sq[start:end] = np.minimum(
                    min_dists_sq[start:end], dists_cpu
                )

            # Sample next center proportional to min distances
            probs = min_dists_sq / min_dists_sq.sum()
            next_idx = np.random.choice(n_samples, p=probs)
            centers.append(
                torch.tensor(
                    data[next_idx],
                    dtype=torch.float32,
                    device=self.device,
                )
            )

        return centers

    def _e_step_batched(
        self,
        data: np.ndarray,
        means: List[torch.Tensor],
        inv_stds: List[torch.Tensor],
        log_dets: List[torch.Tensor],
        log_mixture_weights: torch.Tensor,
    ) -> Tuple[np.ndarray, float]:
        """E-step: compute responsibilities in GPU batches.

        Returns responsibilities as CPU numpy (n_samples, n_components)
        and scalar log-likelihood.
        """
        n_samples = len(data)
        n_components = len(means)
        d = data.shape[1]

        log_2pi = np.log(2 * np.pi)

        # Allocate responsibilities on CPU
        responsibilities = np.empty(
            (n_samples, n_components), dtype=np.float32
        )
        total_log_likelihood = 0.0

        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            batch = torch.tensor(
                data[start:end], dtype=torch.float32, device=self.device
            )
            bs = batch.shape[0]

            # Log probabilities for each component
            log_probs = torch.zeros((bs, n_components), device=self.device)
            for k in range(n_components):
                diff = batch - means[k]
                normalized = diff * inv_stds[k]
                log_probs[:, k] = (
                    -0.5
                    * (
                        normalized.pow(2).sum(dim=1)
                        + log_dets[k]
                        + d * log_2pi
                    )
                    + log_mixture_weights[k]
                )

            # Responsibilities via log-sum-exp
            log_resp = log_probs - torch.logsumexp(
                log_probs, dim=1, keepdim=True
            )
            responsibilities[start:end] = torch.exp(log_resp).cpu().numpy()

            # Accumulate log-likelihood
            total_log_likelihood += (
                torch.logsumexp(log_probs, dim=1).sum().item()
            )

        return responsibilities, total_log_likelihood

    def _m_step_batched(
        self,
        data: np.ndarray,
        responsibilities: np.ndarray,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        torch.Tensor,
    ]:
        """M-step: update parameters using two-pass batched accumulation.

        Pass 1: accumulate weighted sums for means and Nk.
        Pass 2: accumulate weighted squared deviations for variances.
        """
        n_samples, d = data.shape
        n_components = responsibilities.shape[1]

        # Effective samples per component (full sum on CPU — cheap)
        Nk = responsibilities.sum(axis=0)  # (n_components,)

        # ------ Pass 1: compute means ------
        weighted_sum = torch.zeros(
            (n_components, d), device=self.device, dtype=torch.float64
        )
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            batch = torch.tensor(
                data[start:end], dtype=torch.float32, device=self.device
            )
            resp_batch = torch.tensor(
                responsibilities[start:end],
                dtype=torch.float32,
                device=self.device,
            )
            # (n_components, d) += (n_components, bs) @ (bs, d)
            weighted_sum += resp_batch.T.to(torch.float64) @ batch.to(
                torch.float64
            )

        means = []
        inv_stds = []
        log_dets = []
        Nk_tensor = torch.tensor(Nk, dtype=torch.float32, device=self.device)

        for k in range(n_components):
            if Nk[k] < MIN_COMPONENT_SIZE:
                logger.warning(
                    f"Component {k} has only {Nk[k]:.1f} effective "
                    f"samples, re-initializing"
                )
                random_idx = np.random.randint(n_samples)
                means.append(
                    torch.tensor(
                        data[random_idx],
                        dtype=torch.float32,
                        device=self.device,
                    )
                )
                inv_stds.append(torch.ones(d, device=self.device))
                log_dets.append(torch.tensor(0.0, device=self.device))
                continue

            mean_k = (weighted_sum[k] / Nk[k]).to(torch.float32)
            means.append(mean_k)
            # Placeholders — filled in pass 2
            inv_stds.append(None)  # type: ignore[arg-type]
            log_dets.append(None)  # type: ignore[arg-type]

        # ------ Pass 2: compute variances around new means ------
        weighted_var = torch.zeros(
            (n_components, d), device=self.device, dtype=torch.float64
        )
        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            batch = torch.tensor(
                data[start:end], dtype=torch.float32, device=self.device
            )
            resp_batch = torch.tensor(
                responsibilities[start:end],
                dtype=torch.float32,
                device=self.device,
            )
            for k in range(n_components):
                if Nk[k] < MIN_COMPONENT_SIZE:
                    continue
                diff = batch - means[k]
                # resp (bs,) * diff^2 (bs, d) summed over batch
                weighted_var[k] += (
                    (resp_batch[:, k].unsqueeze(1) * diff.pow(2))
                    .to(torch.float64)
                    .sum(dim=0)
                )

        for k in range(n_components):
            if Nk[k] < MIN_COMPONENT_SIZE:
                continue
            var_k = (weighted_var[k] / Nk[k]).to(
                torch.float32
            ) + REGULARIZATION
            std_k = var_k.sqrt()
            inv_stds[k] = 1.0 / std_k
            log_dets[k] = torch.sum(torch.log(var_k))

        log_mixture_weights = torch.log(Nk_tensor / n_samples + REGULARIZATION)
        return means, inv_stds, log_dets, log_mixture_weights

    # ------------------------------------------------------------------
    # Scoring helpers (unchanged interface, used by score_samples)
    # ------------------------------------------------------------------

    def _component_log_probs(self, data: torch.Tensor) -> torch.Tensor:
        """Per-component log joint ``log p(x, k)`` for a GPU batch.

        Returns a ``(n_samples, n_components)`` tensor including the log
        mixture weights, so an ``argmax`` over the component axis is the MAP
        (hard) assignment and a ``logsumexp`` is the marginal log-likelihood.
        """
        n_samples = len(data)
        d = data.shape[1]

        log_probs = torch.zeros(
            (n_samples, self.n_components), device=self.device
        )

        for k in range(self.n_components):
            diff = data - self.means[k]  # type: ignore
            normalized = diff * self.inv_stds[k]  # type: ignore

            log_probs[:, k] = (
                -0.5
                * (
                    torch.sum(normalized**2, dim=1)
                    + self.log_dets[k]  # type: ignore
                    + d
                    * torch.log(torch.tensor(2 * torch.pi, device=self.device))
                )
                + self.log_mixture_weights[k]  # type: ignore
            )

        return log_probs

    def _compute_log_probabilities(self, data: torch.Tensor) -> torch.Tensor:
        """Compute marginal log probabilities under the GMM for a GPU batch."""
        return torch.logsumexp(self._component_log_probs(data), dim=1)

    def assign_components(self, data: np.ndarray) -> np.ndarray:
        """Hard (MAP) component assignment for each sample.

        Returns ``argmax_k log p(x, k)`` per row, i.e. the component with the
        highest responsibility. Used for per-component recalibration analyses
        Does not alter scoring.
        """
        self._check_fitted()
        n_samples = len(data)
        assignments = np.zeros(n_samples, dtype=np.int64)

        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_tensor = torch.tensor(
                data[start_idx:end_idx],
                dtype=torch.float32,
                device=self.device,
            )
            batch_normalized = (
                batch_tensor - self.feature_mean
            ) / self.feature_std
            comp_log_probs = self._component_log_probs(batch_normalized)
            assignments[start_idx:end_idx] = (
                torch.argmax(comp_log_probs, dim=1).cpu().numpy()
            )

        return assignments

    def _compute_normalization_params(self, data: np.ndarray) -> None:
        """Compute score normalization parameters in GPU batches."""
        n_samples = len(data)
        all_raw_scores = []

        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            batch = torch.tensor(
                data[start:end], dtype=torch.float32, device=self.device
            )
            log_probs = self._compute_log_probabilities(batch)
            all_raw_scores.append((-log_probs).cpu())

        raw_scores = torch.cat(all_raw_scores)

        if self.score_normalization == "percentile":
            self.score_percentiles = (
                torch.quantile(raw_scores, 0.05).item(),
                torch.quantile(raw_scores, 0.95).item(),
            )
            logger.info(
                f"Score percentiles: 5th={self.score_percentiles[0]:.2f}, "
                f"95th={self.score_percentiles[1]:.2f}"
            )
        else:  # sigmoid
            self.score_mean = raw_scores.mean().item()
            self.score_std = raw_scores.std().item() + REGULARIZATION
            logger.info(
                f"Score statistics: mean={self.score_mean:.2f}, "
                f"std={self.score_std:.2f}"
            )

    def _normalize_scores(self, raw_scores: torch.Tensor) -> torch.Tensor:
        """Normalize scores to [0, 1] range."""
        if self.score_normalization == "percentile":
            p5, p95 = self.score_percentiles  # type: ignore
            normalized = (raw_scores - p5) / (p95 - p5 + REGULARIZATION)
            return torch.clamp(normalized, 0, 1)
        else:  # sigmoid
            z_scores = (raw_scores - self.score_mean) / self.score_std
            return torch.sigmoid(z_scores)

    def save(self, filepath: str) -> None:
        """Save the fitted model to disk.

        Args:
            filepath: Path to save the model
        """
        self._check_fitted()
        checkpoint = {
            "method_name": self._method_name,
            "config": self._init_config,
            "state": self.state_dict(),
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Saved GMM model to {filepath}")

    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> "GMMAnodet":
        """Load a saved model from disk.

        Args:
            filepath: Path to the saved model
            device: Device to load the model to

        Returns:
            Loaded GMMAnodet instance
        """
        return cls._load_helper(filepath, device)
