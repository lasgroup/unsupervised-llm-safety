"""Autoencoder model for unsupervised feature learning with sparsity regularization."""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .base_model import BaseSafetyModel, ModelOutput

log = logging.getLogger(__name__)


class Autoencoder(BaseSafetyModel):
    """Pure autoencoder for feature learning via reconstruction with sparsity regularization.

    This model implements an autoencoder architecture that learns compressed representations
    through reconstruction loss and sparsity penalties. It maintains compatibility with the
    safety classification pipeline by returning dummy safety predictions.
    """

    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        encoder_hidden_dims: List[int] = [512, 256],
        decoder_hidden_dims: List[int] = [256, 512],
        activation: str = "relu",
        use_batch_norm: bool = False,
        dropout_rate: float = 0.0,
        sparsity_weight: float = 0.01,
        sparsity_target: float = 0.0,
        contrastive_weight: float = 0.0,
    ):
        """Initialize the autoencoder model.

        Args:
            input_dim: Input dimension (hidden state size)
            feature_dim: Bottleneck feature dimension
            encoder_hidden_dims: Hidden dimensions for encoder layers
            decoder_hidden_dims: Hidden dimensions for decoder layers
            activation: Activation function ('relu', 'tanh', 'gelu')
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate (0.0 = no dropout)
            sparsity_weight: Weight for sparsity penalty on features
            sparsity_target: Target average activation for KL sparsity (if > 0, uses KL instead of L1)
            contrastive_weight: Weight for contrastive loss penalizing pairwise
                cosine similarity between encoded samples in a batch (0.0 = disabled)
        """
        super().__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.sparsity_weight = sparsity_weight
        self.sparsity_target = sparsity_target
        self.contrastive_weight = contrastive_weight
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.activation_callback: Optional[ActivationCountingCallback] = None

        # Build encoder
        self.encoder = self._build_network(
            input_dim=input_dim,
            hidden_dims=encoder_hidden_dims,
            output_dim=feature_dim,
            activation=activation,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
            final_activation=True,  # Apply activation to features
        )

        # Build decoder
        self.decoder = self._build_network(
            input_dim=feature_dim,
            hidden_dims=decoder_hidden_dims,
            output_dim=input_dim,
            activation=activation,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate,
            final_activation=False,  # No activation on reconstruction
        )

        self.to(self.device)

    def _build_network(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str,
        use_batch_norm: bool,
        dropout_rate: float,
        final_activation: bool,
    ) -> nn.Sequential:
        """Build a multi-layer network (encoder or decoder).

        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            activation: Activation function name
            use_batch_norm: Whether to use batch normalization
            dropout_rate: Dropout rate
            final_activation: Whether to apply activation after the last layer

        Returns:
            nn.Sequential network
        """
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        # Get activation function
        activation_fn = self._get_activation(activation)

        # Build layers
        for i in range(len(dims) - 1):
            # Linear layer
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            # Add batch norm if requested (except for last layer)
            if use_batch_norm and (i < len(dims) - 2 or final_activation):
                layers.append(nn.BatchNorm1d(dims[i + 1]))

            # Add activation (except for last layer unless final_activation is True)
            if i < len(dims) - 2 or final_activation:
                layers.append(activation_fn)

                # Add dropout if requested
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))

        return nn.Sequential(*layers)

    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name.

        Args:
            activation: Name of activation function

        Returns:
            Activation function module
        """
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
        }

        if activation not in activations:
            raise ValueError(
                f"Unknown activation: {activation}. Choose from {list(activations.keys())}"
            )

        return activations[activation]

    @staticmethod
    def contrastive_term(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Compute Barlow-Twins-style contrastive loss on encoded batch.

        Penalizes pairwise cosine similarity between samples, pushing
        encoded representations toward orthogonality.

        Args:
            z: Encoded features, shape (B, m).
            eps: Epsilon for numerical stability in normalization.

        Returns:
            Scalar loss: ||S - I||_F^2 / (B * (B - 1)), where S is the
            cosine similarity matrix of the L2-normalized z.
        """
        z_norm = z / (z.norm(dim=-1, keepdim=True) + eps)
        S = z_norm @ z_norm.T  # [B, B]
        B = S.shape[0]
        off_diag = S - torch.eye(B, device=S.device, dtype=S.dtype)
        return (off_diag**2).sum() / (B * (B - 1))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature representations using the encoder.

        Args:
            x: Input tensor

        Returns:
            Feature tensor after encoding
        """
        return self.encoder(x)

    def forward(
        self,
        x: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        input_in_feature_space: bool = False,
    ) -> ModelOutput:
        """Forward pass for training and inference.

        Args:
            x: Input tensor (hidden states or pre-extracted features)
            label: Optional labels (ignored for autoencoder)
            reduction: Loss reduction mode ('mean', 'sum', 'none')
            input_in_feature_space: If True, x contains pre-extracted features;
                                    if False, x contains raw hidden states

        Returns:
            ModelOutput with dummy safety predictions and reconstruction loss
        """
        # FIXED: Use explicit parameter instead of dimension-based detection
        if input_in_feature_space:
            features = x
            # Store original input for reconstruction loss if not in feature space
            original_input = None
        else:
            features = self.encoder(x)
            original_input = x

        # Decode
        reconstructed = self.decoder(features)

        # Compute losses
        loss = None
        sparsity_loss = None

        if label is not None:
            # Skip reconstruction loss when input is already in feature space
            if input_in_feature_space:
                # Cannot compute reconstruction loss without original input
                recon_loss = torch.tensor(0.0, device=x.device)
            else:
                # Reconstruction loss
                if reduction == "none":
                    recon_loss = F.mse_loss(
                        reconstructed, original_input, reduction="none"
                    ).mean(dim=1)
                else:
                    recon_loss = F.mse_loss(
                        reconstructed, original_input, reduction=reduction
                    )

            # Sparsity loss
            if self.sparsity_weight > 0:
                if self.sparsity_target > 0:
                    # KL divergence sparsity (per feature dimension)
                    # Compute average activation per feature across batch
                    avg_activation = torch.sigmoid(features).mean(
                        dim=0
                    )  # Apply sigmoid for probabilities

                    # KL divergence from target sparsity
                    eps = 1e-8
                    kl_div = self.sparsity_target * torch.log(
                        self.sparsity_target / (avg_activation + eps)
                    ) + (1 - self.sparsity_target) * torch.log(
                        (1 - self.sparsity_target) / (1 - avg_activation + eps)
                    )
                    sparsity_loss = kl_div.sum()

                    # Handle reduction for KL loss
                    if reduction == "none":
                        # Distribute KL loss evenly across batch samples
                        batch_size = x.shape[0]
                        sparsity_loss = (
                            sparsity_loss.expand(batch_size) / batch_size
                        )
                    elif reduction == "mean":
                        sparsity_loss = (
                            sparsity_loss / x.shape[0]
                        )  # Average over batch
                else:
                    # L1 sparsity
                    if reduction == "none":
                        sparsity_loss = features.abs().mean(
                            dim=1
                        )  # Per-sample sparsity
                    else:
                        sparsity_loss = features.abs().mean()

                # Total loss
                if reduction == "none":
                    loss = recon_loss + self.sparsity_weight * sparsity_loss
                else:
                    loss = recon_loss + self.sparsity_weight * sparsity_loss
            else:
                loss = recon_loss
                sparsity_loss = torch.tensor(0.0, device=x.device)

            # Contrastive loss: penalize pairwise cosine similarity
            if self.contrastive_weight > 0 and not input_in_feature_space:
                contrast_loss = self.contrastive_term(features)
                loss = loss + self.contrastive_weight * contrast_loss

        # Dummy safety predictions (all safe)
        batch_size = x.shape[0]
        is_safe = torch.ones(batch_size, dtype=torch.bool, device=x.device)
        probs = torch.full((batch_size,), 0.99, device=x.device)
        violations = torch.zeros(batch_size, 1, device=x.device)

        # Compute metrics
        metrics = {}
        if loss is not None:
            # Compute feature sparsity (percentage of near-zero activations)
            with torch.no_grad():
                feature_sparsity = (
                    (features.abs() < 0.01).float().mean().item()
                )

            if reduction == "none":
                metrics["reconstruction_loss"] = recon_loss
                metrics["sparsity_loss"] = sparsity_loss
                metrics["total_loss"] = loss
            else:
                metrics["reconstruction_loss"] = recon_loss.item()
                metrics["sparsity_loss"] = (
                    sparsity_loss.item()
                    if isinstance(sparsity_loss, torch.Tensor)
                    else sparsity_loss
                )
                metrics["total_loss"] = loss.item()
            metrics["feature_sparsity"] = feature_sparsity

            # L0 metrics (number of active neurons per sample)
            with torch.no_grad():
                l0_per_sample = (features > 0).float().sum(dim=1)
                metrics["mean_l0"] = l0_per_sample.mean().item()
                metrics["median_l0"] = l0_per_sample.median().item()

            # Contrastive metrics
            if self.contrastive_weight > 0 and not input_in_feature_space:
                metrics["contrastive_loss"] = contrast_loss.item()

        return ModelOutput(
            is_safe=is_safe,
            probs=probs,
            loss=loss,
            violations=violations,
            metrics=metrics,
        )

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make binary safety predictions.

        Always returns True (safe) for all inputs since this is an autoencoder.

        Args:
            x: Input tensor

        Returns:
            Binary predictions tensor (all True)
        """
        # Handle single sample case
        if x.dim() == 1:
            return torch.tensor(True, device=x.device)

        batch_size = x.shape[0]
        return torch.ones(batch_size, dtype=torch.bool, device=x.device)

    @staticmethod
    def create_activation_callback(
        cfg,  # kept for signature compatibility; not required
        train_loader: DataLoader,
        device: torch.device,
        seed: int,
        batch_size: int,
        save_dir: Optional[str] = None,
    ) -> Optional["ActivationCountingCallback"]:
        """Create activation counting callback (always enabled for Autoencoder).

        Note: This uses internal defaults and does not rely on config flags.
        """
        # Internal defaults
        sample_fraction = 0.1
        max_samples = None
        eval_batch_size = batch_size
        activation_threshold = 0.0

        if save_dir is None:
            save_dir = os.getcwd()

        n_total = len(train_loader.dataset)
        n_sample = max(1, int(n_total * sample_fraction))
        if max_samples is not None:
            n_sample = min(n_sample, max_samples)

        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(n_total, generator=generator)[:n_sample]
        sample_dataset = Subset(train_loader.dataset, indices.tolist())
        sample_loader = DataLoader(
            sample_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
        )

        return ActivationCountingCallback(
            sample_loader=sample_loader,
            device=device,
            activation_threshold=activation_threshold,
            save_dir=save_dir,
        )

    # --- Model-specific training hooks used by the generic trainer ---
    def on_training_start(
        self,
        cfg,
        train_dataloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        **_,
    ):
        """Set up activation monitoring and record pre-training baseline."""
        self.activation_callback = self.create_activation_callback(
            cfg=cfg,
            train_loader=train_dataloader,
            device=device,
            seed=getattr(cfg, "seed", 0),
            batch_size=getattr(cfg, "batch_size", 1),
            save_dir=os.getcwd(),
        )

        if self.activation_callback is not None:
            self.activation_callback(
                {
                    "model": self,
                    "epoch": -1,
                    "num_epochs": num_epochs,
                }
            )

    def on_epoch_end(
        self,
        epoch: int,
        num_epochs: int,
        **_,
    ):
        """Record activation monitoring at the end of each epoch."""
        if self.activation_callback is None:
            return

        self.activation_callback(
            {
                "model": self,
                "epoch": epoch,
                "num_epochs": num_epochs,
            }
        )


class ActivationCountingCallback:
    """
    Callback to measure how many neurons are activated (any activation) on a
    fixed subset of the training data after each epoch and before training.
    """

    def __init__(
        self,
        sample_loader: DataLoader,
        device: torch.device,
        activation_threshold: float = 0.0,
        save_dir: str = ".",
        plot_filename: str = "activation_history.png",
    ):
        self.sample_loader = sample_loader
        self.device = torch.device(device)
        self.activation_threshold = activation_threshold
        self.history: list[dict[str, int]] = (
            []
        )  # List of {"epoch": int, "activated_neurons": int}
        self.save_dir = save_dir
        self.plot_filename = plot_filename
        self.plot_path: Optional[str] = None

    def _count_activated_neurons(self, model: BaseSafetyModel) -> int:
        """Count neurons that fire at least once on the sampled data."""
        was_training = model.training
        model.eval()

        activation_seen = None
        with torch.no_grad():
            for batch in self.sample_loader:
                inputs = batch[0]  # works for (inputs, labels[, idx])
                inputs = inputs.to(self.device, non_blocking=True)
                features = model.extract_features(inputs)

                if activation_seen is None:
                    activation_seen = torch.zeros(
                        features.shape[1],
                        dtype=torch.bool,
                        device=self.device,
                    )

                active = features.abs() > self.activation_threshold
                activation_seen |= active.any(dim=0)

        if was_training:
            model.train()

        return int(
            activation_seen.sum().item() if activation_seen is not None else 0
        )

    def __call__(self, callback_data):
        epoch = callback_data.get("epoch", -1)
        model = callback_data["model"]
        count = self._count_activated_neurons(model)

        # epoch index: 0 = pre-training, then 1..N after each epoch
        epoch_idx = 0 if epoch < 0 else epoch + 1
        self.history.append({"epoch": epoch_idx, "activated_neurons": count})

        log.info(
            f"[ActivationCounting] Epoch {epoch_idx}: "
            f"{count} neurons activated (threshold={self.activation_threshold})"
        )

        # Create plot and sparsity diagnostics after final epoch
        num_epochs = callback_data.get("num_epochs")
        if num_epochs is not None and epoch >= num_epochs - 1:
            self._save_plot()
            self._save_sparsity_diagnostics(model)

        return None

    def _save_plot(self) -> None:
        """Save activation history plot to disk."""
        if not self.history:
            return

        epochs = [entry["epoch"] for entry in self.history]
        counts = [entry["activated_neurons"] for entry in self.history]

        os.makedirs(self.save_dir, exist_ok=True)
        self.plot_path = os.path.join(self.save_dir, self.plot_filename)

        plt.figure(figsize=(6, 4))
        plt.plot(epochs, counts, marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Activated neurons (any activation)")
        plt.title("Neuron activations across training")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        plt.savefig(self.plot_path)
        plt.close()

        log.info(
            f"[ActivationCounting] Saved activation history plot to {self.plot_path}"
        )

    def _save_sparsity_diagnostics(self, model: BaseSafetyModel) -> None:
        """Compute and save SAE sparsity diagnostics after training.

        Produces:
        - sparsity_diagnostics.json with L0, dead neurons, concentration
        - activation_histogram.png with latent activation distribution
        - per_neuron_mean_activation.png bar chart
        - activations.npy raw activation matrix
        """
        diag_dir = os.path.join(self.save_dir, "sae_diagnostics")
        os.makedirs(diag_dir, exist_ok=True)

        was_training = model.training
        model.eval()

        all_features = []
        with torch.no_grad():
            for batch in self.sample_loader:
                inputs = batch[0]
                inputs = inputs.to(self.device, non_blocking=True)
                features = model.extract_features(inputs)
                all_features.append(features.cpu().numpy())

        if was_training:
            model.train()

        if not all_features:
            log.warning("[SparsityDiagnostics] No data to compute diagnostics")
            return

        z = np.vstack(all_features)  # [n_samples, n_neurons]
        n_samples, n_neurons = z.shape

        # L0: count of active neurons per sample (activation > threshold)
        active = (z > self.activation_threshold).astype(np.float32)
        l0_per_sample = active.sum(axis=1)

        # Dead neurons: never fire across all samples
        neuron_fire_count = active.sum(axis=0)
        dead_neurons = int((neuron_fire_count == 0).sum())

        # Activation concentration: top 5% neurons' share of total
        neuron_total = np.abs(z).sum(axis=0)
        top_k = max(1, int(0.05 * n_neurons))
        total_activation = neuron_total.sum() + 1e-10
        top5_frac = float(
            np.sort(neuron_total)[-top_k:].sum() / total_activation
        )

        metrics: Dict[str, Any] = {
            "n_samples": n_samples,
            "n_neurons": n_neurons,
            "l0_mean": float(l0_per_sample.mean()),
            "l0_std": float(l0_per_sample.std()),
            "l0_frac_mean": float(l0_per_sample.mean() / n_neurons),
            "dead_neurons": dead_neurons,
            "dead_neuron_pct": float(dead_neurons / n_neurons),
            "alive_neurons": int(n_neurons - dead_neurons),
            "top5_activation_frac": top5_frac,
            "activation_mean": float(z.mean()),
            "activation_std": float(z.std()),
            "activation_max": float(z.max()),
        }

        # Save metrics JSON
        metrics_path = os.path.join(diag_dir, "sparsity_diagnostics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Save raw activations
        np.save(os.path.join(diag_dir, "activations.npy"), z)

        # Activation histogram
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(z.flatten(), bins=100, alpha=0.7)
        ax.set_title(
            f"SAE Latent Activation Distribution "
            f"(L0={metrics['l0_mean']:.1f}, "
            f"dead={dead_neurons}/{n_neurons})"
        )
        ax.set_xlabel("Activation Value")
        ax.set_ylabel("Count")
        ax.set_yscale("log")
        plt.tight_layout()
        plt.savefig(
            os.path.join(diag_dir, "activation_histogram.png"), dpi=150
        )
        plt.close(fig)

        # Per-neuron mean activation bar chart
        neuron_means = z.mean(axis=0)
        fig, ax = plt.subplots(figsize=(max(10, n_neurons / 20), 5))
        ax.bar(np.arange(n_neurons), neuron_means, width=1.0, alpha=0.7)
        ax.set_title("Mean Activation per Neuron (Post-SAE-Training)")
        ax.set_xlabel("Neuron Index")
        ax.set_ylabel("Mean Activation")
        plt.tight_layout()
        plt.savefig(
            os.path.join(diag_dir, "per_neuron_mean_activation.png"),
            dpi=150,
        )
        plt.close(fig)

        # L0 distribution histogram
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(l0_per_sample, bins=50, alpha=0.7)
        ax.set_title(
            f"L0 Sparsity Distribution " f"(mean={metrics['l0_mean']:.1f})"
        )
        ax.set_xlabel("Number of Active Neurons per Sample")
        ax.set_ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(diag_dir, "l0_distribution.png"), dpi=150)
        plt.close(fig)

        log.info(
            f"[SparsityDiagnostics] Saved to {diag_dir}: "
            f"L0={metrics['l0_mean']:.1f}/{n_neurons}, "
            f"dead={dead_neurons} ({metrics['dead_neuron_pct']:.1%}), "
            f"top5_conc={top5_frac:.1%}"
        )
