"""Binary Cross-Entropy (BCE) based anomaly detection.

This method trains a binary classifier using BCE loss on both safe and unsafe
samples, then uses the classifier's predicted probability as the anomaly score.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from unsupervised_llm_safety.anodet_algos.base import AnodetMethod
from unsupervised_llm_safety.anodet_algos.scaler import IdentityScaler

logger = logging.getLogger(__name__)


class BCEAnodet(AnodetMethod):
    """Binary Cross-Entropy based anomaly detection.

    This method:
    1. Trains a binary classifier (encoder + classifier head) with BCE loss
       using both safe (label=1) and unsafe (label=0) samples
    2. Uses the classifier's predicted probability as the anomaly score
    """

    def __init__(
        self,
        rep_dim: int = 32,
        pretrained_encoder: Optional[nn.Module] = None,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 1024,
        analysis_batch_size: int = 256,
        device: str = "cuda",
        seed: int = 42,
        weight_decay: float = 1e-6,
        lr_milestones: Optional[list] = None,
        freeze_encoder: bool = False,
        linear_only: bool = False,
        no_bias: bool = False,
    ):
        """Initialize BCE anomaly detection method.

        Args:
            rep_dim: Representation dimension (only used if no pretrained_encoder)
            pretrained_encoder: Pretrained encoder module from run_anodet_analysis
            epochs: Number of training epochs
            lr: Learning rate
            batch_size: Batch size for training
            analysis_batch_size: Batch size for scoring
            device: Device to use for training ('cuda' or 'cpu')
            seed: Random seed for reproducibility
            weight_decay: L2 regularization weight
            lr_milestones: Epochs at which to decrease learning rate by 10x
            freeze_encoder: If True, freeze pretrained encoder (linear probe)
            linear_only: If True, use no encoder — just a linear classifier
                on the raw input features (input_dim -> 1)
        """
        super().__init__()

        self.rep_dim = rep_dim
        self.pretrained_encoder = pretrained_encoder
        self.linear_only = linear_only
        self.no_bias = no_bias
        self.n_epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.analysis_batch_size = analysis_batch_size
        self.seed = seed
        self.weight_decay = weight_decay
        self.lr_milestones = lr_milestones if lr_milestones else []
        self.freeze_encoder = freeze_encoder

        # Set device
        self.device = torch.device(
            device if torch.cuda.is_available() else "cpu"
        )
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, using CPU")

        # Components (initialized during fit)
        self.scaler = IdentityScaler()
        self.net: Optional[nn.Module] = None
        self.classifier_head: Optional[nn.Module] = None
        self.input_dim: Optional[int] = None

    @property
    def name(self) -> str:
        """Return the name of this method."""
        return "bce"

    def fit(self, data: np.ndarray, labels: np.ndarray) -> None:
        """Fit BCE classifier on both safe and unsafe samples.

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

        # Check that we have both safe and unsafe samples
        safe_mask = labels == 1
        unsafe_mask = labels == 0
        n_safe = safe_mask.sum()
        n_unsafe = unsafe_mask.sum()

        if n_safe == 0:
            raise ValueError(
                "No safe samples (label=1) found in training data. "
                "BCE training requires both safe and unsafe samples."
            )
        if n_unsafe == 0:
            raise ValueError(
                "No unsafe samples (label=0) found in training data. "
                "BCE training requires both safe and unsafe samples."
            )

        logger.info(
            f"Training BCE classifier: {n_safe} safe samples, {n_unsafe} unsafe samples "
            f"out of {len(data)} total"
        )

        # Store input dimension
        self.input_dim = data.shape[1]

        # Apply scaler (fit on all data, transform all)
        data_scaled = self.scaler.fit_transform(data)

        # Initialize network (encoder)
        if self.linear_only:
            self.net = nn.Identity()
            assert self.input_dim is not None
            self.rep_dim = self.input_dim
            logger.info(
                f"Linear-only mode: no encoder, classifier head on raw "
                f"input dim {self.input_dim}"
            )
        elif self.pretrained_encoder is not None:
            self.net = self.pretrained_encoder.to(self.device)
            logger.info("Using pretrained encoder")
            # Infer rep_dim from encoder output if needed
            with torch.no_grad():
                sample_input = torch.FloatTensor(data_scaled[:1]).to(
                    self.device
                )
                sample_output = self.net(sample_input)
                self.rep_dim = sample_output.shape[1]
            logger.info(f"Pretrained encoder output dimension: {self.rep_dim}")
        else:
            from unsupervised_llm_safety.anodet_algos.deep_svdd import (
                DeepSVDDNetwork,
            )

            self.net = DeepSVDDNetwork(self.input_dim, self.rep_dim).to(
                self.device
            )
            logger.info(
                f"Created DeepSVDD network: {self.input_dim} -> {self.rep_dim}"
            )

        # Freeze or enable encoder gradients
        if self.freeze_encoder and self.pretrained_encoder is not None:
            for param in self.net.parameters():
                param.requires_grad = False
            logger.info("Frozen pretrained encoder (linear probe mode)")
        else:
            for param in self.net.parameters():
                param.requires_grad = True

        # Create classifier head
        use_bias = not self.no_bias
        self.classifier_head = nn.Linear(self.rep_dim, 1, bias=use_bias).to(
            self.device
        )
        logger.info(
            f"Created classifier head for BCE training (bias={use_bias})"
        )

        # Create DataLoader
        tensor_data = torch.FloatTensor(data_scaled)
        tensor_labels = torch.FloatTensor(labels)
        dataset = TensorDataset(tensor_data, tensor_labels)
        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Train with BCE loss
        self._train_bce(train_loader)

        self._fitted = True
        logger.info("BCE training complete!")

    def _train_bce(self, train_loader: DataLoader) -> None:
        """Train encoder and classifier head with BCE loss.

        Args:
            train_loader: DataLoader for training data
        """
        logger.info("Starting BCE training...")

        # Setup optimizer
        assert self.net is not None
        assert self.classifier_head is not None
        trainable_params = [
            p for p in self.net.parameters() if p.requires_grad
        ] + list(self.classifier_head.parameters())
        params = trainable_params
        optimizer = optim.Adam(
            params, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.lr_milestones, gamma=0.1
        )

        # BCE loss function
        bce_loss_fn = nn.BCEWithLogitsLoss()

        # Training loop
        self.net.train()
        self.classifier_head.train()

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            n_correct = 0
            n_total = 0

            for batch_data, batch_labels in train_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()

                # Forward pass through encoder
                z = self.net(batch_data)

                # Forward pass through classifier head
                logits = self.classifier_head(z).squeeze(-1)  # [batch_size]

                # Compute BCE loss
                loss = bce_loss_fn(logits, batch_labels)

                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)

                optimizer.step()

                # Track metrics
                epoch_loss += loss.item()
                n_batches += 1

                # Compute accuracy
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    predictions = (probs > 0.5).float()
                    n_correct += (predictions == batch_labels).sum().item()
                    n_total += len(batch_labels)

            # Step scheduler
            scheduler.step()

            # Logging
            if epoch % 20 == 0 or epoch == self.n_epochs - 1:
                avg_loss = epoch_loss / n_batches
                accuracy = n_correct / n_total if n_total > 0 else 0.0

                logger.info(
                    f"Epoch {epoch + 1}/{self.n_epochs}, "
                    f"Loss: {avg_loss:.6f}, "
                    f"Accuracy: {accuracy:.4f}"
                )

    def score_samples(self, data: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for samples using classifier probability.

        Args:
            data: Data to score of shape (n_samples, n_features)

        Returns:
            Anomaly scores of shape (n_samples,) in range [0, 1]
            where higher values indicate more anomalous samples
        """
        self._check_fitted()
        assert self.net is not None
        assert self.classifier_head is not None

        self.net.eval()
        self.classifier_head.eval()

        with torch.no_grad():
            # Apply scaler (same as training)
            data_scaled = self.scaler.transform(data)

            # Process in batches
            all_scores = []
            batch_size = min(self.analysis_batch_size, len(data_scaled))

            for i in range(0, len(data_scaled), batch_size):
                batch = data_scaled[i : i + batch_size]
                batch_tensor = torch.tensor(
                    batch, dtype=torch.float32, device=self.device
                )

                # Forward pass through encoder
                z = self.net(batch_tensor)

                # Forward pass through classifier head
                logits = self.classifier_head(z).squeeze(-1)  # [batch_size]

                # Apply sigmoid to get probability of being safe (label=1)
                prob_safe = torch.sigmoid(logits)  # [batch_size]

                # Anomaly score = probability of being unsafe (label=0)
                # Higher score = more anomalous
                anomaly_scores = 1.0 - prob_safe

                all_scores.append(anomaly_scores.cpu().numpy())

            scores = np.concatenate(all_scores)

        return scores

    def save(self, path: str) -> None:
        """Save the model to disk."""
        self._check_fitted()
        assert self.net is not None
        assert self.classifier_head is not None

        state = {
            "model_type": "BCE",
            "net_state_dict": self.net.state_dict(),
            "classifier_head_state_dict": self.classifier_head.state_dict(),
            "input_dim": self.input_dim,
            "rep_dim": self.rep_dim,
            "uses_pretrained_encoder": self.pretrained_encoder is not None,
            "config": {
                "rep_dim": self.rep_dim,
                "epochs": self.n_epochs,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "analysis_batch_size": self.analysis_batch_size,
                "device": str(self.device),
                "seed": self.seed,
                "weight_decay": self.weight_decay,
                "lr_milestones": self.lr_milestones,
                "freeze_encoder": self.freeze_encoder,
                "linear_only": self.linear_only,
                "no_bias": self.no_bias,
            },
        }
        torch.save(state, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load the model from disk."""
        state = torch.load(path, map_location=self.device, weights_only=False)

        # Load config
        config = state["config"]
        self.rep_dim = config["rep_dim"]
        self.n_epochs = config["epochs"]
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
        self.analysis_batch_size = config["analysis_batch_size"]
        self.seed = config["seed"]
        self.weight_decay = config["weight_decay"]
        self.lr_milestones = config.get("lr_milestones", [])
        self.linear_only = config.get("linear_only", False)
        self.no_bias = config.get("no_bias", False)

        # Load input_dim
        self.input_dim = state["input_dim"]

        # Initialize network
        if self.linear_only:
            self.net = nn.Identity()
            assert self.input_dim is not None
            self.rep_dim = self.input_dim
        elif self.pretrained_encoder is not None:
            self.net = self.pretrained_encoder.to(self.device)
        else:
            from unsupervised_llm_safety.anodet_algos.deep_svdd import (
                DeepSVDDNetwork,
            )

            self.net = DeepSVDDNetwork(self.input_dim, self.rep_dim).to(
                self.device
            )

        # Load network weights
        net_state_dict = state.get(
            "net_state_dict", state.get("encoder_state_dict")
        )

        # Handle potential "model." prefix in keys for Sequential models
        if any(key.startswith("model.") for key in net_state_dict.keys()):
            cleaned_state_dict = {}
            for key, value in net_state_dict.items():
                if key.startswith("model."):
                    new_key = key.replace("model.", "")
                    cleaned_state_dict[new_key] = value
                else:
                    cleaned_state_dict[key] = value
            net_state_dict = cleaned_state_dict

        self.net.load_state_dict(net_state_dict)

        # Initialize and load classifier head
        use_bias = not self.no_bias
        self.classifier_head = nn.Linear(self.rep_dim, 1, bias=use_bias).to(
            self.device
        )
        self.classifier_head.load_state_dict(
            state["classifier_head_state_dict"]
        )

        self._fitted = True
        logger.info(f"Model loaded from {path}")
