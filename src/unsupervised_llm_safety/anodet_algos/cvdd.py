"""Context Vector Data Description (CVDD) baseline for anomaly detection.

CVDD extends Deep SVDD to handle multi-modal normal data by learning K hypersphere
centers. Samples are considered normal if close to ANY center, enabling detection
of anomalies in data with multiple normal modes.
"""

import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from unsupervised_llm_safety.common.model_io import (
    reconstruct_encoder_from_state_dict,
)

from .base import AnodetMethod
from .deep_svdd import DeepSVDDNetwork

logger = logging.getLogger(__name__)


class CVDDCenters(nn.Module):
    """Module to hold CVDD centers as either fixed or learnable parameters."""

    def __init__(self, centers: torch.Tensor, learnable: bool = False):
        """Initialize centers module.

        Args:
            centers: Initial center values of shape (n_contexts, rep_dim)
            learnable: Whether centers should be learnable parameters
        """
        super().__init__()
        if learnable:
            self.centers = nn.Parameter(centers)
        else:
            self.register_buffer("centers", centers)

    def forward(self) -> torch.Tensor:
        """Return the centers tensor."""
        return self.centers


class CVDDAnodet(AnodetMethod):
    """Context Vector Data Description (CVDD) - Multi-sphere extension of Deep SVDD.

    CVDD handles multi-modal normal data by learning K hypersphere centers.
    Samples are considered normal if close to ANY center, enabling detection
    of anomalies in data with multiple normal modes.
    """

    _method_name = "cvdd"  # Class attribute for validation

    def __init__(
        self,
        n_contexts: int = 3,
        rep_dim: int = 32,
        n_epochs: int = 100,
        lr: float = 0.001,
        weight_decay: float = 1e-6,
        batch_size: int = 256,
        lr_milestones: Optional[List[int]] = None,
        device: str = "cuda",
        seed: int = 42,
        # CVDD-specific
        fixed_centers: bool = True,
        t_start: float = 5.0,
        t_end: float = 0.5,
        pretrained_encoder: Optional[nn.Module] = None,
        ortho_weight: float = 0.0001,
    ):
        """Initialize CVDD with specified parameters.

        Args:
            n_contexts: Number of context vectors (hypersphere centers)
            rep_dim: Dimensionality of the learned representation
            n_epochs: Number of training epochs
            lr: Learning rate
            weight_decay: L2 regularization weight
            batch_size: Batch size for training
            lr_milestones: Epochs at which to decrease learning rate by 10x
            device: Device to use for training ('cuda' or 'cpu')
            seed: Random seed for reproducibility
            fixed_centers: Whether to keep centers fixed after initialization
            t_start: Starting temperature for soft-min
            t_end: Final temperature for soft-min
            pretrained_encoder: Optional pretrained encoder to use instead of training new one
            ortho_weight: Weight for orthogonality regularization (|| X X^T - I ||)
        """
        super().__init__()

        # Validate parameters
        assert (
            t_start > t_end > 0
        ), f"Invalid temperature range: {t_start} -> {t_end}"

        # Store all init params for serialization
        self._init_config = {
            "n_contexts": n_contexts,
            "rep_dim": rep_dim,
            "n_epochs": n_epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "lr_milestones": lr_milestones,
            "device": device,
            "seed": seed,
            "fixed_centers": fixed_centers,
            "t_start": t_start,
            "t_end": t_end,
            "ortho_weight": ortho_weight,
        }

        self.n_contexts = n_contexts
        self.rep_dim = rep_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.lr_milestones = lr_milestones if lr_milestones else []
        self.seed = seed
        self.fixed_centers = fixed_centers
        self.t_start = t_start
        self.t_end = t_end
        self.ortho_weight = ortho_weight

        # Set device
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")

        # Components (initialized during fit)
        self.scaler = StandardScaler()
        self.net: Optional[DeepSVDDNetwork] = None
        self.pretrained_encoder = pretrained_encoder
        self.centers_module: Optional[CVDDCenters] = None
        self.input_dim: Optional[int] = None

        # Score normalization parameters
        self.score_low: Optional[float] = (
            None  # 5th percentile (more negative)
        )
        self.score_high: Optional[float] = (
            None  # 95th percentile (less negative)
        )

    def fit(self, data: np.ndarray, labels: np.ndarray) -> None:
        """Fit CVDD on safe (normal) samples.

        Args:
            data: Training data array of shape (n_samples, n_features)
            labels: Training labels of shape (n_samples,) where:
                   0 = unsafe (anomaly), 1 = safe (normal)
        """
        # Set random seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)

        # Filter and validate safe samples
        safe_mask = labels == 1
        safe_data = data[safe_mask]
        min_samples_required = self.n_contexts * 50  # At least 50 per context

        if len(safe_data) < min_samples_required:
            raise ValueError(
                f"Insufficient safe samples for {self.n_contexts} contexts: "
                f"{len(safe_data)} < {min_samples_required}"
            )

        logger.info(
            f"Training CVDD with {self.n_contexts} contexts on "
            f"{len(safe_data)} safe samples out of {len(data)} total"
        )

        # Store input dimension
        self.input_dim = safe_data.shape[1]

        # Standardize features
        safe_data_scaled = self.scaler.fit_transform(safe_data)

        # Create PyTorch dataset
        tensor_data = torch.FloatTensor(safe_data_scaled)
        dataset = TensorDataset(tensor_data)
        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Initialize network or use pretrained encoder
        assert self.input_dim is not None
        if self.pretrained_encoder is not None:
            self.net = self.pretrained_encoder.to(self.device)
            logger.info("=" * 60)
            logger.info("USING PRETRAINED ENCODER")
            logger.info(f"Encoder architecture: {self.net}")
            logger.info("=" * 60)

            if self.net is not None:
                for param in self.net.parameters():
                    param.requires_grad = True
        else:
            self.net = DeepSVDDNetwork(self.input_dim, self.rep_dim).to(
                self.device
            )
            logger.info("=" * 60)
            logger.info(
                "CREATING NEW NETWORK (no pretrained encoder provided)"
            )
            logger.info(f"New network architecture: {self.net}")
            logger.info("=" * 60)

        # Initialize centers with k-means++
        logger.info("Initializing context vectors with k-means++...")
        assert self.net is not None
        initial_centers = self._init_centers_kmeans(train_loader, self.net)
        self.centers_module = CVDDCenters(
            initial_centers, learnable=not self.fixed_centers
        ).to(self.device)

        # Verify centers are diverse
        with torch.no_grad():
            assert self.centers_module is not None
            centers = self.centers_module.centers

            if (
                self.n_contexts > 1
            ):  # Only check diversity with multiple centers
                center_dists = torch.cdist(centers, centers, p=2)
                mask = ~torch.eye(
                    self.n_contexts, device=self.device, dtype=bool
                )
                min_center_dist = center_dists[mask].min().item()
                logger.info(
                    f"Minimum distance between centers: {min_center_dist:.3f}"
                )

                if min_center_dist < 0.01:
                    logger.warning(
                        "Centers initialized very close together! "
                        "Consider increasing n_contexts or checking data diversity."
                    )
            else:
                logger.info("Using single context (Deep SVDD mode)")

        # Train CVDD
        self._train_cvdd(train_loader)

        # Compute score normalization parameters
        self._compute_score_normalization(train_loader)

        self._fitted = True

    def _collect_embeddings(
        self,
        train_loader: DataLoader,
        net: DeepSVDDNetwork,
        max_samples: int = 1000,
    ) -> torch.Tensor:
        """Collect embeddings from training data for initialization.

        Args:
            train_loader: DataLoader for training data
            net: Encoder network
            max_samples: Maximum number of samples to collect

        Returns:
            Collected embeddings of shape (n_samples, rep_dim)
        """
        embeddings = []
        net.eval()
        with torch.no_grad():
            n_collected = 0
            for batch_data in train_loader:
                batch_data = batch_data[0].to(self.device)
                batch_embeddings = net(batch_data)
                embeddings.append(batch_embeddings)
                n_collected += len(batch_data)
                if n_collected >= max_samples:
                    break

        return torch.cat(embeddings, dim=0)[:max_samples]

    def _init_centers_kmeans(
        self, train_loader: DataLoader, net: DeepSVDDNetwork
    ) -> torch.Tensor:
        """Initialize centers using k-means++ on network embeddings."""
        # Collect embeddings from a subset of data
        embeddings = self._collect_embeddings(
            train_loader, net, max_samples=1000
        )
        n_samples = len(embeddings)

        # K-means++ initialization
        centers = torch.zeros(
            self.n_contexts, self.rep_dim, device=self.device
        )

        # First center: random selection from original embeddings
        first_idx = torch.randint(n_samples, (1,)).item()
        centers[0] = embeddings[first_idx]

        # Remaining centers: probability proportional to squared distance
        for k in range(1, self.n_contexts):
            # Compute distances to existing centers
            dists = torch.cdist(embeddings, centers[:k], p=2)
            min_dists = dists.min(dim=1)[0]  # Distance to nearest center

            # Probability proportional to squared distance
            probs = min_dists.pow(2)
            # Add small epsilon to avoid all-zero probabilities (especially for binary patterns)
            probs = probs + 1e-10
            probs = probs / probs.sum()

            # Sample next center from original embeddings
            next_idx = torch.multinomial(probs, 1).item()
            centers[k] = embeddings[next_idx]

        net.train()

        # Return detached tensor (will be wrapped in CVDDCenters module)
        return centers.detach()

    def _log_training_start(self) -> None:
        """Log training start message (override for variant-specific messages)."""
        logger.info("Starting CVDD training...")

    def _get_trainable_params(self) -> list:
        """Get list of trainable parameters for optimizer.

        Returns:
            List of parameters to optimize
        """
        assert self.net is not None
        assert self.centers_module is not None

        params = list(self.net.parameters())
        if not self.fixed_centers:
            params += list(self.centers_module.parameters())

        return params

    def _setup_optimizer_and_scheduler(self, params: list):
        """Setup optimizer and learning rate scheduler.

        Args:
            params: List of parameters to optimize

        Returns:
            Tuple of (optimizer, scheduler)
        """
        optimizer = optim.Adam(
            params, lr=self.lr, weight_decay=self.weight_decay
        )

        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.lr_milestones, gamma=0.1
        )

        return optimizer, scheduler

    def _compute_batch_loss(
        self, z: torch.Tensor, epoch: int, **kwargs
    ) -> torch.Tensor:
        """Compute loss for a training batch (override for variant-specific loss).

        Args:
            z: Encoder output of shape (batch_size, rep_dim)
            epoch: Current epoch number
            **kwargs: Additional variant-specific arguments (e.g., temperature)

        Returns:
            Loss tensor (scalar)
        """
        # Get temperature from kwargs (linear schedule for CVDD)
        progress = epoch / self.n_epochs
        temperature = self.t_start - (self.t_start - self.t_end) * progress

        # Standard CVDD mode
        assert self.centers_module is not None
        centers = self.centers_module.centers
        distances = torch.cdist(z, centers, p=2).pow(2)

        # Soft-minimum loss
        # -temperature * log(sum(exp(-distances/temperature)))
        # This is a smooth approximation of min(distances)
        soft_min = -temperature * torch.logsumexp(
            -distances / temperature, dim=1
        )
        cvdd_loss = soft_min.mean()

        # Add orthogonality regularization if enabled
        if self.ortho_weight > 0:
            ortho_loss = self._compute_orthogonality_loss()
            total_loss = cvdd_loss + self.ortho_weight * ortho_loss
        else:
            total_loss = cvdd_loss

        return total_loss

    def _compute_orthogonality_loss(self) -> torch.Tensor:
        """Compute orthogonality regularization: || X X^T - I ||_F^2

        Encourages center vectors to be orthogonal to each other with unit norm.
        The loss is normalized by n_contexts^2 to be scale-independent.

        The Gram matrix X X^T computes inner products between all pairs of centers:
        - Diagonal elements: self inner products (squared norms) → should be 1
        - Off-diagonal elements: cross inner products → should be 0 (orthogonal)

        Returns:
            Orthogonality loss (scalar), normalized by n_contexts^2
        """
        assert self.centers_module is not None
        centers = self.centers_module.centers  # Shape: [n_contexts, rep_dim]

        # Use actual number of centers (may differ from n_contexts if clusters were split)
        n_actual_centers = centers.shape[0]

        # Compute Gram matrix: X X^T (shape: [n_actual_centers, n_actual_centers])
        gram = torch.mm(centers, centers.t())

        # Create identity matrix of shape [n_actual_centers, n_actual_centers]
        identity = torch.eye(n_actual_centers, device=self.device)

        # Compute Frobenius norm squared: || X X^T - I ||_F^2
        diff = gram - identity
        ortho_loss = torch.sum(diff**2)

        # Normalize by number of elements for scale-independence
        ortho_loss = ortho_loss / (n_actual_centers**2)

        return ortho_loss

    def _compute_center_utilization(self, train_loader: DataLoader) -> float:
        """Compute what fraction of centers are being used (override for variants).

        Args:
            train_loader: Training data loader

        Returns:
            Utilization ratio in [0, 1]
        """
        utilized = set()
        with torch.no_grad():
            for batch in train_loader:
                assert self.net is not None
                z = self.net(batch[0].to(self.device))
                assert self.centers_module is not None
                centers = self.centers_module.centers
                closest = torch.cdist(z, centers).argmin(dim=1)
                utilized.update(closest.cpu().numpy())
        return len(utilized) / self.n_contexts

    def _get_detailed_metrics(
        self, train_loader: DataLoader, epoch: int
    ) -> dict[str, float | str]:
        """Get detailed metrics for logging (override for variant-specific metrics).

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary of metric names and values
        """
        metrics: dict[str, float | str] = {}

        # Compute temperature
        progress = epoch / self.n_epochs
        temperature = self.t_start - (self.t_start - self.t_end) * progress
        metrics["CVDD Temp"] = temperature

        # Add orthogonality metric if enabled
        if self.ortho_weight > 0:
            with torch.no_grad():
                ortho_loss = self._compute_orthogonality_loss()
                metrics["Ortho Loss"] = ortho_loss.item()

        return metrics

    def _train_cvdd(self, train_loader: DataLoader) -> None:
        """Train CVDD with soft-minimum objective."""
        self._log_training_start()

        # Setup optimizer
        params = self._get_trainable_params()
        optimizer, scheduler = self._setup_optimizer_and_scheduler(params)

        # Training loop
        assert self.net is not None
        self.net.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_data in train_loader:
                batch_data = batch_data[0].to(self.device)
                optimizer.zero_grad()

                # Forward pass through encoder
                assert self.net is not None
                z = self.net(batch_data)

                # Compute loss (variant-specific)
                loss = self._compute_batch_loss(z, epoch)

                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)

                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # Step scheduler
            scheduler.step()

            # Logging
            if epoch % 20 == 0 or epoch == self.n_epochs - 1:
                avg_loss = epoch_loss / n_batches

                # Detailed logging with center utilization
                if epoch % 40 == 0:
                    utilization = self._compute_center_utilization(
                        train_loader
                    )
                    metrics = self._get_detailed_metrics(train_loader, epoch)

                    # Build log message
                    log_msg = (
                        f"Epoch {epoch + 1}/{self.n_epochs}, "
                        f"Loss: {avg_loss:.6f}"
                    )
                    for key, value in metrics.items():
                        if isinstance(value, float):
                            log_msg += f", {key}: {value:.2f}"
                        else:
                            log_msg += f", {key}: {value}"
                    log_msg += f", Center util: {utilization:.1%}"

                    logger.info(log_msg)
                # Simple logging
                else:
                    metrics = self._get_detailed_metrics(train_loader, epoch)
                    log_msg = (
                        f"Epoch {epoch + 1}/{self.n_epochs}, "
                        f"Loss: {avg_loss:.6f}"
                    )
                    for key, value in metrics.items():
                        if isinstance(value, float):
                            log_msg += f", {key}: {value:.2f}"
                        else:
                            log_msg += f", {key}: {value}"

                    logger.info(log_msg)

        logger.info("CVDD training complete")

    def _compute_score_normalization(self, train_loader: DataLoader) -> None:
        """Compute percentiles for score normalization on training data."""
        logger.info("Computing score normalization parameters...")

        train_scores = []
        assert self.net is not None
        self.net.eval()
        with torch.no_grad():
            for batch_data in train_loader:
                batch_data = batch_data[0].to(self.device)
                assert self.net is not None
                z = self.net(batch_data)

                # Compute soft-min distances (positive values)
                assert self.centers_module is not None
                centers = self.centers_module.centers
                distances = torch.cdist(z, centers, p=2).pow(2)
                soft_min_distances = -self.t_end * torch.logsumexp(
                    -distances / self.t_end, dim=1
                )
                train_scores.append(soft_min_distances)

        train_scores = torch.cat(train_scores).cpu().numpy()

        # For positive scores (soft-minimum produces positive values):
        # 5th percentile: smaller values (normal samples in training)
        # 95th percentile: larger values (outliers in training)
        self.score_low = float(np.percentile(train_scores, 5))
        self.score_high = float(np.percentile(train_scores, 95))

        if np.isclose(self.score_low, self.score_high):
            logger.warning(
                f"All training scores are identical: {self.score_low:.3f}. "
                "Using alternative normalization for scoring."
            )
        else:
            logger.info(
                f"Score range: [{self.score_low:.3f}, {self.score_high:.3f}] "
                f"(5th-95th percentile)"
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

        # Standardize using fitted scaler
        data_scaled = self.scaler.transform(data)

        # Convert to tensor (keep on CPU, batch-transfer to GPU)
        tensor_data = torch.FloatTensor(data_scaled)

        # Compute scores in batches
        assert self.net is not None
        self.net.eval()
        scores = []

        dataset = TensorDataset(tensor_data)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            for batch_data in loader:
                batch_data = batch_data[0].to(self.device)
                z = self.net(batch_data)

                # Standard CVDD mode
                assert self.centers_module is not None
                centers = self.centers_module.centers
                distances = torch.cdist(z, centers, p=2).pow(2)

                # Compute soft-min distances (positive values)
                soft_min_distances = -self.t_end * torch.logsumexp(
                    -distances / self.t_end, dim=1
                )
                scores.append(soft_min_distances)

        # Concatenate all scores
        scores_array = torch.cat(scores).cpu().numpy()

        # Normalize scores to [0, 1]
        # Larger values (farther from centers) → score closer to 1 (anomalous)
        # Smaller values (closer to centers) → score closer to 0 (normal)
        assert self.score_low is not None and self.score_high is not None
        # Since score_low < score_high (both positive), we need:
        # scores_array = score_low → normalized = 0 (normal)
        # scores_array = score_high → normalized = 1 (anomalous)
        # Formula: (x - score_low) / (score_high - score_low)
        # When x = score_low: (score_low - score_low) / (score_high - score_low) = 0
        # When x = score_high: (score_high - score_low) / (score_high - score_low) = 1

        # Handle edge case where all training scores are identical
        if np.isclose(self.score_high, self.score_low):
            # If all training scores are the same, use a different normalization
            # Scores equal to training score get 0, higher scores get normalized
            scores_normalized = np.where(
                scores_array > self.score_high,
                np.clip(
                    (scores_array - self.score_high)
                    / (np.abs(self.score_high) + 1e-8),
                    0,
                    1,
                ),
                0.0,
            )
        else:
            scores_normalized = np.clip(
                (scores_array - self.score_low)
                / (self.score_high - self.score_low),
                0,
                1,
            )

        return scores_normalized

    def assign_contexts(self, data: np.ndarray) -> np.ndarray:
        """Hard context assignment: nearest hypersphere center per sample.

        Mirrors ``score_samples`` (same scaler -> net -> distance-to-centers
        path) but returns ``argmin`` over the contexts instead of the soft-min
        score. Used for per-context recalibration analyses; does not affect
        scoring.
        """
        self._check_fitted()
        data_scaled = self.scaler.transform(data)
        tensor_data = torch.FloatTensor(data_scaled)
        assert self.net is not None and self.centers_module is not None
        self.net.eval()
        centers = self.centers_module.centers
        out = []
        loader = DataLoader(
            TensorDataset(tensor_data),
            batch_size=self.batch_size,
            shuffle=False,
        )
        with torch.no_grad():
            for batch_data in loader:
                z = self.net(batch_data[0].to(self.device))
                distances = torch.cdist(z, centers, p=2)
                out.append(torch.argmin(distances, dim=1).cpu().numpy())
        return np.concatenate(out).astype(np.int64)

    @property
    def name(self) -> str:
        """Return the name of this anomaly detection algorithm."""
        return "cvdd"

    def state_dict(self) -> dict:
        """Return the state dictionary for serialization.

        Returns:
            Dictionary containing all model states
        """
        self._check_fitted()
        assert self.net is not None
        assert self.centers_module is not None

        state = {
            "centers_module_state_dict": self.centers_module.state_dict(),
            "scaler_mean": self.scaler.mean_,
            "scaler_scale": self.scaler.scale_,
            "score_low": self.score_low,
            "score_high": self.score_high,
            "input_dim": self.input_dim,
            "uses_pretrained_encoder": self.pretrained_encoder is not None,
            "net_state_dict": self.net.state_dict(),
        }

        return state

    def load_state_dict(self, state_dict: dict) -> None:
        """Load model state from dictionary with validation.

        Args:
            state_dict: Dictionary containing model states
        """
        # Always require net_state_dict - it should always be saved
        required_keys = [
            "centers_module_state_dict",
            "scaler_mean",
            "scaler_scale",
            "score_low",
            "score_high",
            "input_dim",
            "net_state_dict",  # Always required
        ]

        missing = set(required_keys) - set(state_dict.keys())
        if missing:
            raise KeyError(
                f"Cannot load {self.name}: missing required keys {missing}. "
                f"Available keys: {list(state_dict.keys())}"
            )

        # Restore input dimension
        self.input_dim = state_dict["input_dim"]
        assert self.input_dim is not None

        # Always reconstruct encoder from net_state_dict
        # The checkpoint contains the final trained weights
        net_state_dict = state_dict["net_state_dict"]
        self.net = reconstruct_encoder_from_state_dict(
            net_state_dict, self.input_dim, self.rep_dim, self.device
        )
        assert self.net is not None

        # Initialize centers module
        # First create dummy centers to get the shape right
        dummy_centers = torch.zeros(
            self.n_contexts, self.rep_dim, device=self.device
        )
        self.centers_module = CVDDCenters(
            dummy_centers, learnable=not self.fixed_centers
        ).to(self.device)
        # Then load the actual state
        assert self.centers_module is not None
        self.centers_module.load_state_dict(
            state_dict["centers_module_state_dict"]
        )

        # Restore scaler
        self.scaler.mean_ = state_dict["scaler_mean"]
        self.scaler.scale_ = state_dict["scaler_scale"]

        # Restore score normalization
        self.score_low = state_dict["score_low"]
        self.score_high = state_dict["score_high"]

        self._fitted = True

    def save(self, filepath: str) -> None:
        """Save the fitted CVDD model to disk."""
        self._check_fitted()
        checkpoint = {
            "method_name": self.name,
            "state": self.state_dict(),
            "config": self._init_config,
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def load(cls, filepath: str, device: Optional[str] = None) -> "CVDDAnodet":
        """Load a CVDD model from disk."""
        return cls._load_helper(filepath, device)
