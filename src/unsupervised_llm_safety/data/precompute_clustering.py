#!/usr/bin/env python3
"""Pre-compute L0 clustering on cached SAE encodings.

Loads pre-encoded SAE features, runs L0 KMeans clustering with
outlier splitting, and caches the results (cluster labels, centroids,
per-cluster union-of-support indices).

Depends on: precompute_sae_encodings.py having already been run.

Output structure:
    {output_dir}/{model_key}/
        clustering_{n_clusters}_l0_ot{outlier_threshold}.pt
            {
                "cluster_labels": ndarray[N_safe],
                "centroids": ndarray[n_clusters_final, sae_dim],
                "n_clusters": int,
                "union_support": {c: ndarray of indices, ...},
                "cluster_sizes": ndarray[n_clusters_final],
                "config": {...},
                "safe_dataset_names": [...],
                "n_safe_samples": int,
            }

Usage:
    python -m unsupervised_llm_safety.data.precompute_clustering \\
        --model-key qwen2_1.5b --n-clusters 1500

    # All models with defaults:
    python -m unsupervised_llm_safety.data.precompute_clustering \\
        --model-key qwen2_1.5b ministral_8b llama3_8b
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from unsupervised_llm_safety.anodet_algos.cvdd_km_utils import (
    cluster_kmeans,
    split_large_union_clusters,
)
from unsupervised_llm_safety.data.dataset_registry import ROOT_DATA_PATH
from unsupervised_llm_safety.data.precompute_sae_encodings import (
    OUTPUT_BASE as SAE_CACHE_BASE,
)

logger = logging.getLogger(__name__)

# Where the offline k-means cache is written. Set ``ULS_CLUSTERING_CACHE`` to
# relocate it (it can grow to a few GB); unset, it sits beside the activations
# under ``ULS_ACTIVATION_ROOT``.
CLUSTERING_CACHE_BASE = Path(
    os.environ.get(
        "ULS_CLUSTERING_CACHE", f"{ROOT_DATA_PATH}/clustering_cache"
    )
)

# Default cluster counts per model (matching existing experiments)
DEFAULT_N_CLUSTERS = {
    "qwen2_1.5b": 1500,
    "ministral_8b": 2000,
    "llama3_8b": 200,
}

MODEL_KEYS = ["qwen2_1.5b", "ministral_8b", "llama3_8b"]

# IID datasets that contribute safe training data
# (same as what run_anodet_analysis loads with complex_data=True)
IID_DATASETS = [
    "GSM8K",
    "MathSFT40K",
    "HumanEval",
    "MBPP",
    "Alpaca",
    "MTBENCH",
    "MMLU",
    "HumanEval_Plus",
    "ARC",
]

COMPLEX_DATA_DATASETS = [
    "BeaverTails_safe",
    "ToxiGen_safe",
    "HarmBench-AutoDAN_safe",
    "HarmBench-AutoPrompt_safe",
    "HarmBench-DirectRequest_safe",
    "HarmBench-GBDA_safe",
    "HarmBench-GCG_safe",
    "HarmBench-HumanJailbreaks_safe",
    "HarmBench-PEZ_safe",
    "HarmBench-UAT_safe",
]


def cache_filename(
    n_clusters: int,
    outlier_threshold: float,
    l0_filter: float,
) -> str:
    """Generate cache filename from clustering parameters."""
    ot = f"ot{outlier_threshold}" if outlier_threshold > 0 else "ot0"
    lf = f"lf{l0_filter}" if l0_filter > 0 else "lf0"
    return f"clustering_{n_clusters}_l0_{ot}_{lf}.pt"


def load_safe_encoded_data(
    model_key: str,
) -> tuple:
    """Load safe training data from SAE encoding cache.

    Returns:
        (encoded_safe, dataset_names_used, n_per_dataset)
    """
    sae_dir = SAE_CACHE_BASE / model_key

    all_encoded: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    dataset_names_used: List[str] = []
    n_per_dataset: List[int] = []

    # Load IID datasets (no filter — all samples are safe)
    for ds_name in IID_DATASETS:
        path = sae_dir / "iid" / f"{ds_name}.pth"
        if not path.exists():
            logger.warning(f"Missing: {path}")
            continue
        data = torch.load(path, map_location="cpu", weights_only=False)
        # Use train split only (matching run_anodet_analysis behavior)
        if "train" in data:
            enc = data["train"]["encoded_hidden_states"].float().numpy()
            lab = data["train"]["labels"].numpy()
            all_encoded.append(enc)
            all_labels.append(lab)
            dataset_names_used.append(ds_name)
            n_per_dataset.append(len(enc))
            logger.info(f"  {ds_name}: {len(enc)} samples (train)")

    # Load complex_data safe datasets (filter safe_only: label==1)
    # Only use train split to match run_anodet_analysis behavior
    for ds_name in COMPLEX_DATA_DATASETS:
        path = sae_dir / "iid" / f"{ds_name}.pth"
        if not path.exists():
            continue
        data = torch.load(path, map_location="cpu", weights_only=False)
        # Use train split only; fall back to "all" if no train
        split = "train" if "train" in data else "all"
        if split not in data:
            continue
        enc = data[split]["encoded_hidden_states"].float().numpy()
        lab = data[split]["labels"].numpy()
        safe_mask = lab == 1
        enc_safe = enc[safe_mask]
        if len(enc_safe) > 0:
            all_encoded.append(enc_safe)
            all_labels.append(lab[safe_mask])
            dataset_names_used.append(ds_name)
            n_per_dataset.append(len(enc_safe))
            logger.info(
                f"  {ds_name} ({split}): " f"{len(enc_safe)}/{len(lab)} safe"
            )

    if not all_encoded:
        raise ValueError(f"No safe encoded data found for {model_key}")

    encoded = np.concatenate(all_encoded, axis=0)
    logger.info(
        f"Total safe encoded: {encoded.shape} from "
        f"{len(dataset_names_used)} datasets"
    )

    return encoded, dataset_names_used, n_per_dataset


def run_clustering(
    model_key: str,
    n_clusters: int,
    outlier_threshold: float = 2.0,
    l0_filter: float = 2.0,
    force: bool = False,
) -> Dict[str, Any]:
    """Run L0 clustering and cache results."""
    output_dir = CLUSTERING_CACHE_BASE / model_key
    output_dir.mkdir(parents=True, exist_ok=True)

    fname = cache_filename(n_clusters, outlier_threshold, l0_filter)
    output_path = output_dir / fname

    if output_path.exists() and not force:
        logger.info(f"Cache exists: {output_path}")
        return torch.load(output_path, map_location="cpu", weights_only=False)

    logger.info(f"\n{'='*60}")
    logger.info(f"Clustering {model_key}")
    logger.info(
        f"  n_clusters={n_clusters}, ot={outlier_threshold}, "
        f"l0_filter={l0_filter}"
    )
    logger.info(f"{'='*60}")

    # Load pre-encoded safe data
    logger.info("Loading cached SAE encodings...")
    encoded, ds_names, n_per_ds = load_safe_encoded_data(model_key)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run L0 KMeans
    logger.info("Running L0 KMeans...")
    result = cluster_kmeans(
        data=encoded,
        n_clusters=n_clusters,
        cluster_method="l0",
        l0_filter=l0_filter,
        device=device,
    )
    cluster_labels = result.labels
    centroids = result.centroids
    actual_n = int(cluster_labels.max() + 1)
    logger.info(f"Initial clustering: {actual_n} clusters")

    # Sub-cluster large-union clusters
    if outlier_threshold > 0:
        logger.info(
            f"Splitting large-union clusters (threshold={outlier_threshold}x)..."
        )
        split_result = split_large_union_clusters(
            data=encoded,
            cluster_labels=cluster_labels,
            n_clusters=actual_n,
            threshold=outlier_threshold,
        )
        cluster_labels = split_result.labels
        centroids = split_result.centroids
        actual_n = split_result.n_clusters
        logger.info(f"After splitting: {actual_n} clusters")

    # Compute per-cluster union of support
    logger.info("Computing per-cluster union of support...")
    union_support: Dict[int, np.ndarray] = {}
    cluster_sizes = np.bincount(cluster_labels, minlength=actual_n)
    for c in range(actual_n):
        c_mask = cluster_labels == c
        if c_mask.sum() == 0:
            union_support[c] = np.array([], dtype=np.int64)
            continue
        c_data = encoded[c_mask]
        active = np.any(c_data > 0, axis=0)
        union_support[c] = np.where(active)[0].astype(np.int64)

    union_sizes = [len(v) for v in union_support.values() if len(v) > 0]
    logger.info(
        f"Union of support: mean={np.mean(union_sizes):.0f}, "
        f"median={np.median(union_sizes):.0f}, "
        f"max={max(union_sizes)}"
    )

    # Save cache
    cache = {
        "cluster_labels": cluster_labels,
        "centroids": centroids,
        "n_clusters": actual_n,
        "union_support": union_support,
        "cluster_sizes": cluster_sizes,
        "config": {
            "model_key": model_key,
            "n_clusters_requested": n_clusters,
            "n_clusters_actual": actual_n,
            "cluster_method": "l0",
            "outlier_threshold": outlier_threshold,
            "l0_filter": l0_filter,
        },
        "safe_dataset_names": ds_names,
        "n_per_dataset": n_per_ds,
        "n_safe_samples": len(encoded),
    }
    torch.save(cache, output_path)
    logger.info(f"Saved: {output_path}")

    return cache


def load_clustering_cache(
    model_key: str,
    n_clusters: int = 0,
    outlier_threshold: float = 2.0,
    l0_filter: float = 2.0,
    cache_base: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load cached clustering results.

    Args:
        model_key: Model identifier.
        n_clusters: Requested cluster count (0 = use model default).
        outlier_threshold: Outlier splitting threshold.
        l0_filter: L0 filter threshold.
        cache_base: Override cache directory.

    Returns:
        Cache dict with cluster_labels, centroids, union_support, etc.
    """
    base = cache_base or CLUSTERING_CACHE_BASE
    if n_clusters == 0:
        n_clusters = DEFAULT_N_CLUSTERS.get(model_key, 1500)

    fname = cache_filename(n_clusters, outlier_threshold, l0_filter)
    path = base / model_key / fname
    if not path.exists():
        raise FileNotFoundError(
            f"Clustering cache not found: {path}. "
            f"Run precompute_clustering first."
        )
    return torch.load(path, map_location="cpu", weights_only=False)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute L0 clustering on cached SAE encodings"
    )
    parser.add_argument(
        "--model-key",
        nargs="+",
        default=None,
        help="Model keys (default: all)",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=0,
        help="Number of clusters (0 = model-specific default)",
    )
    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=2.0,
        help="Outlier splitting threshold (default: 2.0)",
    )
    parser.add_argument(
        "--l0-filter",
        type=float,
        default=2.0,
        help="L0 filter threshold (default: 2.0)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing cache",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    models = args.model_key or MODEL_KEYS

    for model_key in models:
        n_clusters = args.n_clusters or DEFAULT_N_CLUSTERS.get(model_key, 1500)
        run_clustering(
            model_key=model_key,
            n_clusters=n_clusters,
            outlier_threshold=args.outlier_threshold,
            l0_filter=args.l0_filter,
            force=args.force,
        )

    logger.info("\nAll done.")


if __name__ == "__main__":
    main()
