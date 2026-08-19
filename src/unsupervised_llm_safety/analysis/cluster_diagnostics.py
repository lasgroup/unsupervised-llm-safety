#!/usr/bin/env python3
"""Per-cluster diagnostics for CVDDKMeansMasked / LearnedMask checkpoints.

Pure-compute + plotting helpers operating on already-encoded (SAE) cluster
assignments:
  - Cluster sizes + histogram
  - Union-of-support size (|∪ supp(x_i)| per cluster)
  - Average L0 per sample per cluster
  - NxN pairwise centroid L1 and L2 distance heatmaps

Library module imported by the training pipeline's `_run_cluster_diagnostics`
hook (see `cvdd_kmeans_masked.py` / `learned_mask.py`). Data loading /
encoding is owned by the caller; this module has no CLI entry point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def compute_cluster_diagnostics(
    data_encoded: np.ndarray,
    cluster_labels: np.ndarray,
    n_clusters: int,
) -> Tuple[List[Dict], np.ndarray]:
    """Compute per-cluster diagnostics on encoded (SAE) data.

    Returns:
        cluster_table: List of dicts with per-cluster stats.
        centroids: [n_clusters, d] array of cluster centroids.
    """
    d = data_encoded.shape[1]
    cluster_table = []
    centroids = np.zeros((n_clusters, d), dtype=np.float32)

    for c in range(n_clusters):
        mask = cluster_labels == c
        cluster_data = data_encoded[mask]
        n_samples = len(cluster_data)

        if n_samples == 0:
            cluster_table.append(
                {
                    "cluster": c,
                    "n_samples": 0,
                    "union_support_size": 0,
                    "avg_l0": 0.0,
                    "std_l0": 0.0,
                    "pct_of_total": 0.0,
                }
            )
            continue

        # Binary support: which components are active (> 0)
        active = cluster_data > 0  # [n_samples, d]

        # Per-sample L0
        l0_per_sample = active.sum(axis=1).astype(np.float32)
        avg_l0 = float(l0_per_sample.mean())
        std_l0 = float(l0_per_sample.std())

        # Union of active components across all samples in cluster
        union = active.any(axis=0)
        union_support_size = int(union.sum())

        # Per-component frequency (how often each fires)
        frequency = active.sum(axis=0).astype(np.float32) / n_samples

        # Centroid (mean of activations)
        centroid = cluster_data.mean(axis=0)
        centroids[c] = centroid

        cluster_table.append(
            {
                "cluster": c,
                "n_samples": n_samples,
                "union_support_size": union_support_size,
                "avg_l0": round(avg_l0, 1),
                "std_l0": round(std_l0, 1),
                "pct_of_total": round(
                    100.0 * n_samples / len(data_encoded), 2
                ),
                "frequency_nonzero": int((frequency > 0).sum()),
                "frequency_gt50pct": int((frequency > 0.5).sum()),
                "frequency_gt90pct": int((frequency > 0.9).sum()),
            }
        )

    return cluster_table, centroids


def plot_centroid_heatmap(
    centroids: np.ndarray,
    metric: str,
    output_path: Path,
) -> np.ndarray:
    """Plot NxN pairwise centroid distance heatmap."""
    n = centroids.shape[0]
    dist_matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            diff = centroids[i] - centroids[j]
            if metric == "l1":
                dist_matrix[i, j] = np.abs(diff).sum()
            elif metric == "l2":
                dist_matrix[i, j] = np.sqrt((diff**2).sum())

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(dist_matrix, cmap="viridis", aspect="auto")
    ax.set_title(f"Pairwise centroid {metric.upper()} distance")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Cluster")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

    return dist_matrix


def plot_cluster_sizes(cluster_table: List[Dict], output_path: Path) -> None:
    """Plot histogram of cluster sizes."""
    sizes = [c["n_samples"] for c in cluster_table]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(sizes)), sorted(sizes, reverse=True), alpha=0.7)
    ax.set_xlabel("Cluster (sorted by size)")
    ax.set_ylabel("Number of samples")
    ax.set_title(f"Cluster sizes (n={len(sizes)})")
    ax.axhline(
        y=np.mean(sizes),
        color="r",
        linestyle="--",
        label=f"mean={np.mean(sizes):.0f}",
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_union_sizes(
    cluster_table: List[Dict], d_sae: int, output_path: Path
) -> None:
    """Plot union-of-support sizes per cluster."""
    unions = sorted(
        [c["union_support_size"] for c in cluster_table if c["n_samples"] > 0],
        reverse=True,
    )
    arr = np.array(unions, dtype=float)
    max_val = arr.max()
    median_val = np.median(arr)
    mean_val = arr.mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(unions)), unions, alpha=0.7)
    ax.set_xlabel("Cluster (sorted by union size)")
    ax.set_ylabel("Union of active components")
    ax.set_title("Union-of-support per cluster")
    ax.axhline(
        y=max_val,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"max={max_val:.0f}",
    )
    ax.axhline(
        y=median_val,
        color="g",
        linestyle="--",
        alpha=0.5,
        label=f"median={median_val:.0f}",
    )
    ax.axhline(
        y=mean_val,
        color="orange",
        linestyle="--",
        alpha=0.5,
        label=f"mean={mean_val:.0f}",
    )
    ax.set_ylim(0, max_val * 1.1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def format_table(cluster_table: List[Dict]) -> str:
    """Format cluster diagnostics as a markdown table."""
    lines = [
        "| cluster | n_samples | pct | union_support | avg_l0 | std_l0 | freq>50% | freq>90% |",
        "|:-------:|:---------:|:---:|:-------------:|:------:|:------:|:--------:|:--------:|",
    ]
    for row in sorted(cluster_table, key=lambda r: -r["n_samples"]):
        lines.append(
            f"| {row['cluster']} | {row['n_samples']} | {row['pct_of_total']}% "
            f"| {row['union_support_size']} | {row['avg_l0']} | {row['std_l0']} "
            f"| {row.get('frequency_gt50pct', 'N/A')} | {row.get('frequency_gt90pct', 'N/A')} |"
        )
    return "\n".join(lines)
