"""SAE health diagnostics for CVDD KMeans Masked pipeline.

Computes activation histograms, sparsity metrics, cluster variance,
and inter-cluster distances before and after CVDD fine-tuning.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

logger = logging.getLogger(__name__)


class SAEHealthDiagnostics:
    """Compute and save SAE health diagnostics pre/post fine-tuning."""

    def __init__(
        self,
        pre_encoder_state_dict: dict,
        post_encoder: nn.Module,
        data: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_masks: Dict[int, torch.Tensor],
        cluster_centers: torch.Tensor,
        device: torch.device,
        output_dir: Path,
        cluster_in_encoded_space: bool = False,
        batch_size: int = 256,
    ):
        """Initialize diagnostics.

        Args:
            pre_encoder_state_dict: State dict of encoder before fine-tuning.
            post_encoder: Encoder module after fine-tuning (already loaded).
            data: Training data [n_samples, dim] (raw or encoded).
            cluster_labels: Cluster assignments [n_samples].
            cluster_masks: Per-cluster binary masks {cluster_id: tensor}.
            cluster_centers: Learned cluster centers [n_clusters, dim].
            device: Torch device.
            output_dir: Directory to save diagnostics.
            cluster_in_encoded_space: If True, data is pre-encoded.
            batch_size: Batch size for forward passes.
        """
        self.pre_encoder_state_dict = pre_encoder_state_dict
        self.post_encoder = post_encoder
        self.data = data
        self.cluster_labels = cluster_labels
        self.cluster_masks = cluster_masks
        self.cluster_centers = cluster_centers
        self.device = device
        self.output_dir = Path(output_dir)
        self.cluster_in_encoded_space = cluster_in_encoded_space
        self.batch_size = batch_size

        self.n_clusters = len(cluster_masks)

    def _encode_batched(
        self, encoder: nn.Module, data: np.ndarray
    ) -> np.ndarray:
        """Encode data through encoder in batches.

        Args:
            encoder: Encoder module.
            data: Input data [n_samples, dim].

        Returns:
            Encoded data [n_samples, encoded_dim].
        """
        encoder.eval()
        encoded_parts = []

        with torch.no_grad():
            for i in range(0, len(data), self.batch_size):
                batch = torch.tensor(
                    data[i : i + self.batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                if self.cluster_in_encoded_space:
                    z = batch  # Already encoded
                else:
                    z = encoder(batch)
                encoded_parts.append(z.cpu().numpy())

        return np.vstack(encoded_parts)

    def _build_pre_encoder(self) -> nn.Module:
        """Reconstruct pre-finetune encoder from saved state dict."""
        import copy

        pre_encoder = copy.deepcopy(self.post_encoder)
        pre_encoder.load_state_dict(self.pre_encoder_state_dict)
        pre_encoder.to(self.device)
        pre_encoder.eval()
        return pre_encoder

    def compute_activation_histograms(self) -> Dict[str, Any]:
        """Compute SAE latent activation histograms pre and post training.

        Returns:
            Dict with summary stats and paths to saved artifacts.
        """
        hist_dir = self.output_dir / "activation_histograms"
        hist_dir.mkdir(parents=True, exist_ok=True)

        pre_encoder = self._build_pre_encoder()
        z_pre = self._encode_batched(pre_encoder, self.data)
        z_post = self._encode_batched(self.post_encoder, self.data)

        # Skip saving raw activations (4.7G each, ~9.3G per run)

        # Per-neuron mean activation
        pre_neuron_mean = z_pre.mean(axis=0)
        post_neuron_mean = z_post.mean(axis=0)

        # Comparison histogram plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(
            z_pre.flatten(), bins=100, alpha=0.7, label="Pre-finetune"
        )
        axes[0].set_title("Pre-Finetune Activation Distribution")
        axes[0].set_xlabel("Activation Value")
        axes[0].set_ylabel("Count")
        axes[0].set_yscale("log")
        axes[0].legend()

        axes[1].hist(
            z_post.flatten(), bins=100, alpha=0.7, label="Post-finetune"
        )
        axes[1].set_title("Post-Finetune Activation Distribution")
        axes[1].set_xlabel("Activation Value")
        axes[1].set_ylabel("Count")
        axes[1].set_yscale("log")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(hist_dir / "activation_histogram_comparison.png", dpi=150)
        plt.close(fig)

        # Per-neuron bar charts
        n_neurons = z_pre.shape[1]
        neuron_indices = np.arange(n_neurons)

        for label, means, filename in [
            ("Pre-Finetune", pre_neuron_mean, "per_neuron_activation_pre.png"),
            (
                "Post-Finetune",
                post_neuron_mean,
                "per_neuron_activation_post.png",
            ),
        ]:
            fig, ax = plt.subplots(figsize=(max(10, n_neurons / 20), 5))
            ax.bar(neuron_indices, means, width=1.0, alpha=0.7)
            ax.set_title(f"{label} Mean Activation per Neuron")
            ax.set_xlabel("Neuron Index")
            ax.set_ylabel("Mean Activation")
            plt.tight_layout()
            plt.savefig(hist_dir / filename, dpi=150)
            plt.close(fig)

        results = {
            "pre_activation_mean": float(z_pre.mean()),
            "pre_activation_std": float(z_pre.std()),
            "post_activation_mean": float(z_post.mean()),
            "post_activation_std": float(z_post.std()),
            "n_neurons": n_neurons,
            "n_samples": len(self.data),
        }

        logger.info(
            f"Activation histograms saved to {hist_dir} "
            f"(pre_mean={results['pre_activation_mean']:.4f}, "
            f"post_mean={results['post_activation_mean']:.4f})"
        )
        return results

    def compute_sparsity_metrics(self) -> Dict[str, Any]:
        """Compute L0 sparsity (fraction of active neurons) pre/post.

        Returns:
            Dict with sparsity stats.
        """
        sparsity_dir = self.output_dir / "sparsity"
        sparsity_dir.mkdir(parents=True, exist_ok=True)

        pre_encoder = self._build_pre_encoder()
        z_pre = self._encode_batched(pre_encoder, self.data)
        z_post = self._encode_batched(self.post_encoder, self.data)

        # L0: count of nonzero activations per sample (threshold > 0)
        pre_active = (z_pre > 0).astype(np.float32)
        post_active = (z_post > 0).astype(np.float32)

        n_neurons = z_pre.shape[1]
        pre_l0 = pre_active.sum(axis=1)  # Active count per sample
        post_l0 = post_active.sum(axis=1)

        # Dead neurons: never fire across all samples
        pre_dead = (pre_active.sum(axis=0) == 0).sum()
        post_dead = (post_active.sum(axis=0) == 0).sum()

        # Activation concentration: fraction of total activation from top 5%
        pre_neuron_total = np.abs(z_pre).sum(axis=0)
        post_neuron_total = np.abs(z_post).sum(axis=0)

        top_k = max(1, int(0.05 * n_neurons))
        pre_top5_frac = np.sort(pre_neuron_total)[-top_k:].sum() / (
            pre_neuron_total.sum() + 1e-10
        )
        post_top5_frac = np.sort(post_neuron_total)[-top_k:].sum() / (
            post_neuron_total.sum() + 1e-10
        )

        results = {
            "n_neurons": n_neurons,
            "pre_l0_mean": float(pre_l0.mean()),
            "pre_l0_std": float(pre_l0.std()),
            "post_l0_mean": float(post_l0.mean()),
            "post_l0_std": float(post_l0.std()),
            "pre_l0_frac_mean": float(pre_l0.mean() / n_neurons),
            "post_l0_frac_mean": float(post_l0.mean() / n_neurons),
            "pre_dead_neurons": int(pre_dead),
            "post_dead_neurons": int(post_dead),
            "pre_dead_neuron_pct": float(pre_dead / n_neurons),
            "post_dead_neuron_pct": float(post_dead / n_neurons),
            "pre_top5_activation_frac": float(pre_top5_frac),
            "post_top5_activation_frac": float(post_top5_frac),
        }

        # Save metrics
        with open(sparsity_dir / "sparsity_metrics.json", "w") as f:
            json.dump(results, f, indent=2)

        # L0 distribution comparison plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(
            pre_l0,
            bins=50,
            alpha=0.6,
            label=f"Pre (mean={pre_l0.mean():.1f})",
        )
        ax.hist(
            post_l0,
            bins=50,
            alpha=0.6,
            label=f"Post (mean={post_l0.mean():.1f})",
        )
        ax.set_title("L0 Sparsity Distribution (Active Neurons per Sample)")
        ax.set_xlabel("Number of Active Neurons")
        ax.set_ylabel("Count")
        ax.legend()
        plt.tight_layout()
        plt.savefig(sparsity_dir / "l0_distribution_comparison.png", dpi=150)
        plt.close(fig)

        logger.info(
            f"Sparsity metrics saved to {sparsity_dir} "
            f"(pre_L0={results['pre_l0_mean']:.1f}, "
            f"post_L0={results['post_l0_mean']:.1f}, "
            f"dead: {pre_dead}->{post_dead})"
        )
        return results

    def compute_cluster_variance(self) -> Dict[str, Any]:
        """Compute inner-cluster variance before and after encoder training.

        Returns:
            Dict with per-cluster and aggregate variance stats.
        """
        cluster_dir = self.output_dir / "cluster_metrics"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        pre_encoder = self._build_pre_encoder()
        z_pre = self._encode_batched(pre_encoder, self.data)
        z_post = self._encode_batched(self.post_encoder, self.data)

        pre_variances = {}
        post_variances = {}

        for cluster_id in range(self.n_clusters):
            mask = self.cluster_labels == cluster_id
            if mask.sum() == 0:
                pre_variances[cluster_id] = 0.0
                post_variances[cluster_id] = 0.0
                continue

            z_pre_cluster = z_pre[mask]
            z_post_cluster = z_post[mask]

            # Variance = mean squared distance to centroid
            pre_centroid = z_pre_cluster.mean(axis=0)
            post_centroid = z_post_cluster.mean(axis=0)

            pre_var = float(
                ((z_pre_cluster - pre_centroid) ** 2).sum(axis=1).mean()
            )
            post_var = float(
                ((z_post_cluster - post_centroid) ** 2).sum(axis=1).mean()
            )

            pre_variances[cluster_id] = pre_var
            post_variances[cluster_id] = post_var

        results = {
            "pre_variances": {str(k): v for k, v in pre_variances.items()},
            "post_variances": {str(k): v for k, v in post_variances.items()},
            "pre_mean_variance": float(np.mean(list(pre_variances.values()))),
            "post_mean_variance": float(
                np.mean(list(post_variances.values()))
            ),
        }

        # Per-cluster change
        variance_changes = {}
        for cid in range(self.n_clusters):
            pre_v = pre_variances[cid]
            post_v = post_variances[cid]
            if pre_v > 0:
                change_pct = (post_v - pre_v) / pre_v
            else:
                change_pct = 0.0
            variance_changes[str(cid)] = float(change_pct)
        results["variance_change_pct"] = variance_changes

        # Save
        with open(cluster_dir / "inner_cluster_variance.json", "w") as f:
            json.dump(results, f, indent=2)

        # Comparison bar chart
        cluster_ids = list(range(self.n_clusters))
        pre_vals = [pre_variances[c] for c in cluster_ids]
        post_vals = [post_variances[c] for c in cluster_ids]

        x = np.arange(len(cluster_ids))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(8, len(cluster_ids) * 1.5), 5))
        ax.bar(x - width / 2, pre_vals, width, label="Pre-finetune")
        ax.bar(x + width / 2, post_vals, width, label="Post-finetune")
        ax.set_xlabel("Cluster ID")
        ax.set_ylabel("Inner-Cluster Variance")
        ax.set_title("Inner-Cluster Variance: Pre vs Post Fine-tuning")
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in cluster_ids])
        ax.legend()
        plt.tight_layout()
        plt.savefig(cluster_dir / "cluster_variance_comparison.png", dpi=150)
        plt.close(fig)

        logger.info(
            f"Cluster variance saved to {cluster_dir} "
            f"(pre_mean={results['pre_mean_variance']:.4f}, "
            f"post_mean={results['post_mean_variance']:.4f})"
        )
        return results

    def compute_inter_cluster_distances(self) -> Dict[str, Any]:
        """Compute pairwise centroid distances before and after training.

        Returns:
            Dict with distance matrices and summary stats.
        """
        cluster_dir = self.output_dir / "cluster_metrics"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        pre_encoder = self._build_pre_encoder()
        z_pre = self._encode_batched(pre_encoder, self.data)
        z_post = self._encode_batched(self.post_encoder, self.data)

        def _compute_centroid_distances(
            z: np.ndarray, labels: np.ndarray
        ) -> np.ndarray:
            """Compute pairwise L2 distances between cluster centroids."""
            centroids = []
            for cid in range(self.n_clusters):
                mask = labels == cid
                if mask.sum() > 0:
                    centroids.append(z[mask].mean(axis=0))
                else:
                    centroids.append(np.zeros(z.shape[1]))
            centroids = np.array(centroids)

            n = len(centroids)
            dist_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.linalg.norm(centroids[i] - centroids[j])
                    dist_matrix[i, j] = d
                    dist_matrix[j, i] = d
            return dist_matrix

        pre_dists = _compute_centroid_distances(z_pre, self.cluster_labels)
        post_dists = _compute_centroid_distances(z_post, self.cluster_labels)

        # Extract upper triangle (pairwise distances)
        triu_idx = np.triu_indices(self.n_clusters, k=1)
        pre_pairwise = pre_dists[triu_idx]
        post_pairwise = post_dists[triu_idx]

        results = {
            "pre_distance_matrix": pre_dists.tolist(),
            "post_distance_matrix": post_dists.tolist(),
            "pre_stats": {
                "mean": (
                    float(pre_pairwise.mean())
                    if len(pre_pairwise) > 0
                    else 0.0
                ),
                "std": (
                    float(pre_pairwise.std()) if len(pre_pairwise) > 0 else 0.0
                ),
                "min": (
                    float(pre_pairwise.min()) if len(pre_pairwise) > 0 else 0.0
                ),
                "max": (
                    float(pre_pairwise.max()) if len(pre_pairwise) > 0 else 0.0
                ),
            },
            "post_stats": {
                "mean": (
                    float(post_pairwise.mean())
                    if len(post_pairwise) > 0
                    else 0.0
                ),
                "std": (
                    float(post_pairwise.std())
                    if len(post_pairwise) > 0
                    else 0.0
                ),
                "min": (
                    float(post_pairwise.min())
                    if len(post_pairwise) > 0
                    else 0.0
                ),
                "max": (
                    float(post_pairwise.max())
                    if len(post_pairwise) > 0
                    else 0.0
                ),
            },
        }

        # Change in min distance
        if results["pre_stats"]["min"] > 0:
            results["min_distance_change_pct"] = float(
                (results["post_stats"]["min"] - results["pre_stats"]["min"])
                / results["pre_stats"]["min"]
            )
        else:
            results["min_distance_change_pct"] = 0.0

        # Save
        with open(cluster_dir / "inter_cluster_distances.json", "w") as f:
            json.dump(results, f, indent=2)

        # Distance matrix comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        im0 = axes[0].imshow(pre_dists, cmap="viridis", aspect="equal")
        axes[0].set_title("Pre-Finetune Centroid Distances")
        axes[0].set_xlabel("Cluster")
        axes[0].set_ylabel("Cluster")
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(post_dists, cmap="viridis", aspect="equal")
        axes[1].set_title("Post-Finetune Centroid Distances")
        axes[1].set_xlabel("Cluster")
        axes[1].set_ylabel("Cluster")
        plt.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        plt.savefig(
            cluster_dir / "inter_cluster_distance_comparison.png",
            dpi=150,
        )
        plt.close(fig)

        logger.info(
            f"Inter-cluster distances saved to {cluster_dir} "
            f"(pre_min={results['pre_stats']['min']:.4f}, "
            f"post_min={results['post_stats']['min']:.4f})"
        )
        return results

    def run_all(self) -> Dict[str, Any]:
        """Run all diagnostics and return combined results.

        Returns:
            Dict with all diagnostic results.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Running SAE health diagnostics -> {self.output_dir}")

        results = {}
        results["activations"] = self.compute_activation_histograms()
        results["sparsity"] = self.compute_sparsity_metrics()
        results["cluster_variance"] = self.compute_cluster_variance()
        results["inter_cluster_distances"] = (
            self.compute_inter_cluster_distances()
        )

        logger.info("All SAE health diagnostics complete")
        return results
