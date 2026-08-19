"""Deep SVDD baseline for anomaly detection.

Based on:
Ruff, Lukas, et al. "Deep one-class classification."
International conference on machine learning. PMLR, 2018.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .base import AnodetMethod

logger = logging.getLogger(__name__)


class DeepSVDDNetwork(nn.Module):
    """Simple MLP for Deep SVDD with hidden state features."""

    def __init__(self, input_dim: int, rep_dim: int = 32):
        super().__init__()
        self.rep_dim = rep_dim

        # Simple 3-layer MLP (encoder)
        self.fc1 = nn.Linear(input_dim, 128, bias=False)
        self.fc2 = nn.Linear(128, 64, bias=False)
        self.fc3 = nn.Linear(64, rep_dim, bias=False)

        # Batch normalization (no affine parameters)
        self.bn1 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.bn2 = nn.BatchNorm1d(64, eps=1e-04, affine=False)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


class DeepSVDDAutoencoder(nn.Module):
    """Autoencoder for pretraining Deep SVDD network."""

    def __init__(self, input_dim: int, rep_dim: int = 32):
        super().__init__()
        self.rep_dim = rep_dim

        # Encoder (must match DeepSVDDNetwork exactly)
        self.fc1 = nn.Linear(input_dim, 128, bias=False)
        self.fc2 = nn.Linear(128, 64, bias=False)
        self.fc3 = nn.Linear(64, rep_dim, bias=False)
        self.bn1 = nn.BatchNorm1d(128, eps=1e-04, affine=False)
        self.bn2 = nn.BatchNorm1d(64, eps=1e-04, affine=False)

        # Decoder (reverse of encoder)
        self.fc4 = nn.Linear(rep_dim, 64, bias=False)
        self.fc5 = nn.Linear(64, 128, bias=False)
        self.fc6 = nn.Linear(128, input_dim, bias=False)
        self.bn3 = nn.BatchNorm1d(64, eps=1e-04, affine=False)
        self.bn4 = nn.BatchNorm1d(128, eps=1e-04, affine=False)

    def forward(self, x):
        # Encode
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        # Decode
        x = F.relu(self.bn3(self.fc4(x)))
        x = F.relu(self.bn4(self.fc5(x)))
        x = self.fc6(x)
        return x


class DeepSVDDAnodet(AnodetMethod):
    """Deep SVDD baseline for anomaly detection.

    This method learns a neural network transformation that maps data
    into a hypersphere with minimal volume. Normal data is expected
    to be mapped close to the center, while anomalies are mapped
    farther away.
    """

    _method_name = "deep_svdd"  # Class attribute for validation

    def __init__(
        self,
        objective: str = "one-class",
        nu: float = 0.1,
        rep_dim: int = 32,
        n_epochs: int = 100,
        lr: float = 0.001,
        weight_decay: float = 1e-6,
        batch_size: int = 256,
        lr_milestones: Optional[list] = None,
        device: str = "cuda",
        seed: int = 42,
        pretrain: bool = True,
        ae_n_epochs: int = 100,
        ae_lr: float = 0.001,
        ae_weight_decay: float = 1e-6,
    ):
        """Initialize Deep SVDD with specified parameters.

        Args:
            objective: Deep SVDD objective ('one-class' or 'soft-boundary')
            nu: Deep SVDD hyperparameter nu (must be 0 < nu <= 1).
                For soft-boundary objective, this is the fraction of outliers
            rep_dim: Dimensionality of the learned representation
            n_epochs: Number of training epochs
            lr: Learning rate
            weight_decay: L2 regularization weight
            batch_size: Batch size for training
            lr_milestones: Epochs at which to decrease learning rate by 10x
            device: Device to use for training ('cuda' or 'cpu')
            seed: Random seed for reproducibility
            pretrain: Whether to use autoencoder pretraining
            ae_n_epochs: Number of epochs for autoencoder pretraining
            ae_lr: Learning rate for autoencoder
            ae_weight_decay: L2 regularization for autoencoder
        """
        super().__init__()

        self._init_config = {
            "objective": objective,
            "nu": nu,
            "rep_dim": rep_dim,
            "n_epochs": n_epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "lr_milestones": lr_milestones,
            "device": device,
            "seed": seed,
            "pretrain": pretrain,
            "ae_n_epochs": ae_n_epochs,
            "ae_lr": ae_lr,
            "ae_weight_decay": ae_weight_decay,
        }

        assert objective in (
            "one-class",
            "soft-boundary",
        ), "Objective must be either 'one-class' or 'soft-boundary'"
        assert 0 < nu <= 1, "For hyperparameter nu, it must hold: 0 < nu <= 1"

        self.objective = objective
        self.nu = nu
        self.rep_dim = rep_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.lr_milestones = lr_milestones if lr_milestones else []
        self.seed = seed
        self.pretrain = pretrain
        self.ae_n_epochs = ae_n_epochs
        self.ae_lr = ae_lr
        self.ae_weight_decay = ae_weight_decay

        # Set device
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")

        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Model components (initialized during fit)
        self.scaler = StandardScaler()
        self.net: Optional[DeepSVDDNetwork] = None
        self.ae_net: Optional[DeepSVDDAutoencoder] = None
        self.c: Optional[torch.Tensor] = None  # Hypersphere center
        self.R: float = 0.0  # Hypersphere radius (for soft-boundary)
        self.input_dim: Optional[int] = None

        # For soft-boundary objective
        self.warm_up_n_epochs = 10  # Epochs before updating radius

    def fit(self, data: np.ndarray, labels: np.ndarray) -> None:
        """Fit Deep SVDD on safe (normal) samples.

        Args:
            data: Training data array of shape (n_samples, n_features)
            labels: Training labels of shape (n_samples,) where:
                   0 = unsafe (anomaly), 1 = safe (normal)
        """
        # Filter safe samples (label=1)
        safe_mask = labels == 1
        safe_data = data[safe_mask]

        if len(safe_data) == 0:
            raise ValueError(
                "No safe samples (label=1) found in training data"
            )

        logger.info(
            f"Training Deep SVDD on {len(safe_data)} safe samples "
            f"out of {len(data)} total samples"
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

        # Initialize network
        assert self.input_dim is not None  # For mypy
        self.net = DeepSVDDNetwork(self.input_dim, self.rep_dim).to(
            self.device
        )

        # Pretrain with autoencoder if enabled
        if self.pretrain:
            self._pretrain_autoencoder(train_loader)

        # Initialize hypersphere center c
        logger.info("Initializing hypersphere center c...")
        assert self.net is not None  # For mypy
        self.c = self._init_center_c(train_loader, self.net)
        logger.info("Center c initialized")

        # Train Deep SVDD
        self._train_deep_svdd(train_loader)

        self._fitted = True

    def _pretrain_autoencoder(self, train_loader: DataLoader) -> None:
        """Pretrain autoencoder and transfer weights to main network."""
        logger.info("Starting autoencoder pretraining...")

        # Initialize autoencoder
        assert self.input_dim is not None  # For mypy
        self.ae_net = DeepSVDDAutoencoder(self.input_dim, self.rep_dim).to(
            self.device
        )
        assert self.ae_net is not None  # For mypy
        optimizer = torch.optim.Adam(
            self.ae_net.parameters(),
            lr=self.ae_lr,
            weight_decay=self.ae_weight_decay,
        )

        # Train autoencoder
        self.ae_net.train()
        for epoch in range(self.ae_n_epochs):
            total_loss = 0
            n_batches = 0

            for batch_data in train_loader:
                batch_data = batch_data[0].to(self.device)
                optimizer.zero_grad()

                # Reconstruction loss
                assert self.ae_net is not None  # For mypy
                outputs = self.ae_net(batch_data)
                loss = F.mse_loss(outputs, batch_data)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            if epoch % 10 == 0 or epoch == self.ae_n_epochs - 1:
                avg_loss = total_loss / n_batches
                logger.info(
                    f"AE Epoch {epoch + 1}/{self.ae_n_epochs}, "
                    f"Loss: {avg_loss:.6f}"
                )

        # Transfer encoder weights to main network
        self._transfer_weights()
        logger.info("Autoencoder pretraining complete, weights transferred")

    def _transfer_weights(self) -> None:
        """Transfer encoder weights from autoencoder to main network."""
        assert self.ae_net is not None  # For mypy
        assert self.net is not None  # For mypy
        ae_state = self.ae_net.state_dict()
        net_state = self.net.state_dict()

        # Transfer matching encoder layers
        for name in net_state:
            if name in ae_state:
                net_state[name] = ae_state[name]

        self.net.load_state_dict(net_state)

    def _init_center_c(
        self, train_loader: DataLoader, net: DeepSVDDNetwork, eps: float = 0.1
    ) -> torch.Tensor:
        """Initialize hypersphere center c as mean of network outputs."""
        n_samples = 0
        c = torch.zeros(self.rep_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for batch_data in train_loader:
                batch_data = batch_data[0].to(self.device)
                outputs = net(batch_data)
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # Avoid trivial solution
        c[(torch.abs(c) < eps) & (c < 0)] = -eps
        c[(torch.abs(c) < eps) & (c >= 0)] = eps

        return c

    def _train_deep_svdd(self, train_loader: DataLoader) -> None:
        """Train the Deep SVDD network."""
        logger.info("Starting Deep SVDD training...")

        # Set optimizer
        assert self.net is not None  # For mypy
        optimizer = optim.Adam(
            self.net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.lr_milestones, gamma=0.1
        )

        # Training
        self.net.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_data in train_loader:
                batch_data = batch_data[0].to(self.device)
                optimizer.zero_grad()

                # Forward pass
                assert self.net is not None  # For mypy
                assert self.c is not None  # For mypy
                outputs = self.net(batch_data)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)

                # Compute loss
                if self.objective == "soft-boundary":
                    scores = dist - self.R**2
                    loss = self.R**2 + (1 / self.nu) * torch.mean(
                        torch.max(torch.zeros_like(scores), scores)
                    )
                else:  # one-class
                    loss = torch.mean(dist)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # Update hypersphere radius R for soft-boundary using full dataset
            if (
                self.objective == "soft-boundary"
                and epoch >= self.warm_up_n_epochs
            ):
                # Compute radius from full dataset
                all_dist = []
                with torch.no_grad():
                    for batch_data in train_loader:
                        batch_data = batch_data[0].to(self.device)
                        outputs = self.net(batch_data)
                        dist = torch.sum((outputs - self.c) ** 2, dim=1)
                        all_dist.append(dist)
                all_dist = torch.cat(all_dist)
                self.R = self._get_radius(all_dist, self.nu)

            # Step the scheduler after epoch
            scheduler.step()

            # Log learning rate changes
            if epoch + 1 in self.lr_milestones:
                logger.info(
                    f"LR scheduler: new learning rate is "
                    f"{scheduler.get_last_lr()[0]:.6f}"
                )

            # Log epoch statistics
            if epoch % 10 == 0 or epoch == self.n_epochs - 1:
                logger.info(
                    f"Epoch {epoch + 1}/{self.n_epochs}, "
                    f"Loss: {epoch_loss / n_batches:.6f}"
                )
                if self.objective == "soft-boundary":
                    logger.info(f"Radius R: {self.R:.6f}")

        logger.info("Deep SVDD training complete")

    def _get_radius(self, dist: torch.Tensor, nu: float) -> float:
        """Get radius via the (1-nu) quantile of distances."""
        with torch.no_grad():
            # Use numpy for stable quantile computation
            distances = torch.sqrt(dist).cpu().numpy()
            radius = float(np.quantile(distances, 1 - nu))
        return radius

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

        # Compute distances from center
        assert self.net is not None  # For mypy
        self.net.eval()
        with torch.no_grad():
            # Process in batches if needed
            scores = []
            dataset = TensorDataset(tensor_data)
            loader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=False
            )

            for batch_data in loader:
                batch_data = batch_data[0].to(self.device)
                assert self.c is not None  # For mypy
                outputs = self.net(batch_data)
                dist = torch.sum((outputs - self.c) ** 2, dim=1)

                # Both objectives score with the squared distance to the
                # centre. The soft-boundary objective conventionally reports
                # ``dist - R**2``, but R is a single fitted constant and the
                # min-max normalisation below is invariant to an additive
                # shift, so the two are the same normalised score.
                scores.append(dist.cpu().numpy())

        # Concatenate all scores
        scores = np.concatenate(scores)

        # Normalize scores to [0, 1] using min-max scaling
        # Higher distances (more anomalous) -> higher scores
        if len(scores) > 1:
            scores_min = np.min(scores)
            scores_max = np.max(scores)
            if scores_max > scores_min:  # Avoid division by zero
                scores_normalized = (scores - scores_min) / (
                    scores_max - scores_min
                )
            else:
                scores_normalized = np.zeros_like(scores)
        else:
            scores_normalized = np.zeros_like(scores)

        return scores_normalized

    @property
    def name(self) -> str:
        """Return the name of this anomaly detection algorithm."""
        return "deep_svdd"

    def state_dict(self) -> dict:
        """Return complete state for serialization."""
        self._check_fitted()
        assert self.net is not None
        assert self.c is not None

        return {
            "c": self.c,
            "R": self.R,
            "scaler_mean": self.scaler.mean_,
            "scaler_scale": self.scaler.scale_,
            "input_dim": self.input_dim,
            "net_state_dict": self.net.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state from dictionary with validation."""
        required_keys = [
            "c",
            "R",
            "scaler_mean",
            "scaler_scale",
            "input_dim",
            "net_state_dict",
        ]
        missing = set(required_keys) - set(state_dict.keys())
        if missing:
            raise KeyError(
                f"Cannot load {self.name}: missing required keys {missing}. "
                f"Available keys: {list(state_dict.keys())}"
            )

        # Update rep_dim from saved center dimension
        self.rep_dim = state_dict["c"].shape[0]

        # Restore network
        self.input_dim = state_dict["input_dim"]
        assert self.input_dim is not None
        self.net = DeepSVDDNetwork(self.input_dim, self.rep_dim).to(
            self.device
        )
        assert self.net is not None
        self.net.load_state_dict(state_dict["net_state_dict"])

        self.c = state_dict["c"].to(self.device)
        self.R = state_dict["R"]

        # Restore scaler
        self.scaler.mean_ = state_dict["scaler_mean"]
        self.scaler.scale_ = state_dict["scaler_scale"]

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
    ) -> "DeepSVDDAnodet":
        """Load a anomaly detection algorithm from disk."""
        return cls._load_helper(filepath, device)
