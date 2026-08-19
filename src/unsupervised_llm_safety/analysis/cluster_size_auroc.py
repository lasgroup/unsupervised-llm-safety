#!/usr/bin/env python3
"""Cluster size vs per-cluster AUROC analysis.

Computes three per-cluster AUROCs:
  - auroc_iid_vs_unsafe: IID-safe as negatives only
  - auroc_benigngen_vs_unsafe: benign_generalization as negatives only
  - auroc_pooled: IID + benign_gen pooled as negatives

Produces scatter plots, binned histograms, and JSON output with
Spearman correlations and bin-wise medians.

Shared utility for both learned_mask and learned_mask_lora diagnostics.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

# Size bins for histogram analysis (log-scale boundaries)
_SIZE_BINS = [
    (0, 10, "<10"),
    (10, 50, "10-50"),
    (50, 200, "50-200"),
    (200, 1000, "200-1k"),
    (1000, float("inf"), ">1k"),
]

# Minimum samples per class to compute AUROC for a cluster
_MIN_SAFE = 5
_MIN_UNSAFE = 5


def compute_per_cluster_auroc(
    cluster_assignments: Dict[str, np.ndarray],
    raw_scores: Dict[str, np.ndarray],
    categories: Dict[str, str],
    train_cluster_sizes: Dict[int, int],
    union_support_sizes: Optional[Dict[int, int]] = None,
    min_safe: int = _MIN_SAFE,
    min_unsafe: int = _MIN_UNSAFE,
) -> Dict[int, Dict[str, Any]]:
    """Compute three AUROC variants per cluster.

    Args:
        cluster_assignments: dataset_name -> (N,) int array of cluster IDs
        raw_scores: dataset_name -> (N,) float array of anomaly scores
        categories: dataset_name -> category string
        train_cluster_sizes: cluster_id -> training sample count
        union_support_sizes: cluster_id -> union support size (optional)
        min_safe: minimum safe samples per cluster to compute AUROC
        min_unsafe: minimum unsafe samples per cluster to compute AUROC

    Returns:
        dict: cluster_id -> {
            "n_train", "union_support_size",
            "n_eval_iid", "n_eval_benigngen", "n_eval_unsafe",
            "auroc_iid_vs_unsafe", "auroc_benigngen_vs_unsafe",
            "auroc_pooled",
            "mean_iid_score", "mean_benigngen_score",
            "mean_unsafe_score",
        }
    """
    # Categorise datasets
    iid_datasets = [k for k, v in categories.items() if v == "iid"]
    benigngen_datasets = [
        k for k, v in categories.items() if v == "benign_generalization"
    ]
    unsafe_datasets = [
        k for k, v in categories.items() if v in ("ood", "adversarial")
    ]

    # Collect per-cluster scores by safety class
    # cluster_id -> {"iid": [...], "benigngen": [...], "unsafe": [...]}
    cluster_scores: Dict[int, Dict[str, List[float]]] = {}

    def _accumulate(
        ds_list: List[str],
        label: str,
    ) -> None:
        for ds_name in ds_list:
            if ds_name not in cluster_assignments:
                continue
            if ds_name not in raw_scores:
                continue
            ca = cluster_assignments[ds_name]
            sc = raw_scores[ds_name]
            for i in range(len(ca)):
                cid = int(ca[i])
                if cid not in cluster_scores:
                    cluster_scores[cid] = {
                        "iid": [],
                        "benigngen": [],
                        "unsafe": [],
                    }
                cluster_scores[cid][label].append(float(sc[i]))

    _accumulate(iid_datasets, "iid")
    _accumulate(benigngen_datasets, "benigngen")
    _accumulate(unsafe_datasets, "unsafe")

    # Compute AUROCs
    result: Dict[int, Dict[str, Any]] = {}
    for cid, scores in cluster_scores.items():
        iid_arr = np.array(scores["iid"])
        bg_arr = np.array(scores["benigngen"])
        unsafe_arr = np.array(scores["unsafe"])

        entry: Dict[str, Any] = {
            "n_train": train_cluster_sizes.get(cid, -1),
            "n_eval_iid": len(iid_arr),
            "n_eval_benigngen": len(bg_arr),
            "n_eval_unsafe": len(unsafe_arr),
        }
        if union_support_sizes is not None:
            entry["union_support_size"] = union_support_sizes.get(cid, -1)

        # Mean scores
        if len(iid_arr) > 0:
            entry["mean_iid_score"] = round(float(iid_arr.mean()), 6)
        if len(bg_arr) > 0:
            entry["mean_benigngen_score"] = round(float(bg_arr.mean()), 6)
        if len(unsafe_arr) > 0:
            entry["mean_unsafe_score"] = round(float(unsafe_arr.mean()), 6)

        # AUROC 1: iid vs unsafe
        entry["auroc_iid_vs_unsafe"] = _safe_auroc(
            iid_arr, unsafe_arr, min_safe, min_unsafe
        )

        # AUROC 2: benigngen vs unsafe
        entry["auroc_benigngen_vs_unsafe"] = _safe_auroc(
            bg_arr, unsafe_arr, min_safe, min_unsafe
        )

        # AUROC 3: pooled (iid + benigngen) vs unsafe
        pooled = (
            np.concatenate([iid_arr, bg_arr]) if len(bg_arr) > 0 else iid_arr
        )
        entry["auroc_pooled"] = _safe_auroc(
            pooled, unsafe_arr, min_safe, min_unsafe
        )

        result[cid] = entry

    return result


def _safe_auroc(
    safe_scores: np.ndarray,
    unsafe_scores: np.ndarray,
    min_safe: int,
    min_unsafe: int,
) -> Optional[float]:
    """Compute AUROC or return None if insufficient samples."""
    if len(safe_scores) < min_safe or len(unsafe_scores) < min_unsafe:
        return None
    labels = np.concatenate(
        [np.zeros(len(safe_scores)), np.ones(len(unsafe_scores))]
    )
    scores = np.concatenate([safe_scores, unsafe_scores])
    try:
        return round(float(roc_auc_score(labels, scores)), 4)
    except ValueError:
        return None


def plot_cluster_size_vs_auroc(
    per_cluster_stats: Dict[int, Dict[str, Any]],
    output_path: Path,
    title: str = "",
    lora_mask: Optional[Dict[int, bool]] = None,
) -> None:
    """Scatter: x=log(n_train), y=auroc_pooled, colored by eval count."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Filter to clusters with valid pooled AUROC and known train size
    valid = {
        cid: s
        for cid, s in per_cluster_stats.items()
        if s.get("auroc_pooled") is not None and s["n_train"] > 0
    }
    if len(valid) < 3:
        logger.warning("Too few clusters for scatter plot")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    if lora_mask is not None:
        # Split by LoRA/non-LoRA
        for has_lora, marker, color, label in [
            (True, "o", "steelblue", "LoRA"),
            (False, "^", "coral", "No LoRA"),
        ]:
            subset = {
                c: s
                for c, s in valid.items()
                if lora_mask.get(c, False) == has_lora
            }
            if not subset:
                continue
            x = np.array([np.log10(s["n_train"]) for s in subset.values()])
            y = np.array([s["auroc_pooled"] for s in subset.values()])
            n_eval = np.array(
                [
                    s["n_eval_iid"]
                    + s["n_eval_benigngen"]
                    + s["n_eval_unsafe"]
                    for s in subset.values()
                ]
            )
            ax.scatter(
                x,
                y,
                c=np.log1p(n_eval),
                cmap="viridis",
                marker=marker,
                alpha=0.6,
                s=30,
                label=label,
            )
    else:
        x = np.array([np.log10(s["n_train"]) for s in valid.values()])
        y = np.array([s["auroc_pooled"] for s in valid.values()])
        n_eval = np.array(
            [
                s["n_eval_iid"] + s["n_eval_benigngen"] + s["n_eval_unsafe"]
                for s in valid.values()
            ]
        )
        sc = ax.scatter(
            x, y, c=np.log1p(n_eval), cmap="viridis", alpha=0.6, s=30
        )
        plt.colorbar(sc, ax=ax, label="log(1 + n_eval)")

    ax.axhline(
        y=0.5, color="gray", linestyle="--", alpha=0.5, label="AUROC=0.5"
    )

    # LOWESS trend if enough points
    all_x = np.array([np.log10(s["n_train"]) for s in valid.values()])
    all_y = np.array([s["auroc_pooled"] for s in valid.values()])
    if len(all_x) >= 30:
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess

            smoothed = lowess(all_y, all_x, frac=0.3, return_sorted=True)
            ax.plot(
                smoothed[:, 0],
                smoothed[:, 1],
                color="red",
                linewidth=2,
                label="LOWESS trend",
            )
        except ImportError:
            # Fallback: rolling median
            order = np.argsort(all_x)
            window = max(len(all_x) // 10, 5)
            rolling_x, rolling_y = [], []
            for i in range(0, len(order) - window + 1, max(window // 2, 1)):
                idx = order[i : i + window]
                rolling_x.append(np.median(all_x[idx]))
                rolling_y.append(np.median(all_y[idx]))
            ax.plot(
                rolling_x,
                rolling_y,
                color="red",
                linewidth=2,
                label="Rolling median",
            )

    ax.set_xlabel("log10(n_train)")
    ax.set_ylabel("AUROC (pooled)")
    ax.set_title(title or "Cluster size vs per-cluster AUROC")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_cluster_size_histogram_with_auroc_bins(
    per_cluster_stats: Dict[int, Dict[str, Any]],
    output_path: Path,
) -> None:
    """Bar chart: n_clusters per size bin, mean AUROC with std error bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = {
        cid: s
        for cid, s in per_cluster_stats.items()
        if s.get("auroc_pooled") is not None and s["n_train"] > 0
    }
    if len(valid) < 3:
        return

    bin_labels = []
    bin_counts = []
    bin_means = []
    bin_stds = []

    for lo, hi, label in _SIZE_BINS:
        aurocs = [
            s["auroc_pooled"]
            for s in valid.values()
            if lo <= s["n_train"] < hi
        ]
        bin_labels.append(label)
        bin_counts.append(len(aurocs))
        if aurocs:
            bin_means.append(float(np.mean(aurocs)))
            bin_stds.append(float(np.std(aurocs) / np.sqrt(len(aurocs))))
        else:
            bin_means.append(0.0)
            bin_stds.append(0.0)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(bin_labels))
    width = 0.35

    # Bar: cluster count
    bars = ax1.bar(
        x - width / 2,
        bin_counts,
        width,
        color="steelblue",
        alpha=0.7,
        label="Cluster count",
    )
    ax1.set_ylabel("Number of clusters", color="steelblue")
    ax1.set_xlabel("Training size bin")
    ax1.set_xticks(x)
    ax1.set_xticklabels(bin_labels)

    # Twin axis: mean AUROC
    ax2 = ax1.twinx()
    ax2.errorbar(
        x + width / 2,
        bin_means,
        yerr=bin_stds,
        fmt="o-",
        color="coral",
        capsize=5,
        label="Mean AUROC",
    )
    ax2.set_ylabel("Mean AUROC (pooled)", color="coral")
    ax2.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)

    # Annotate counts
    for i, (cnt, mean) in enumerate(zip(bin_counts, bin_means)):
        if cnt > 0:
            ax2.annotate(
                f"{mean:.3f}\n(n={cnt})",
                xy=(x[i] + width / 2, mean),
                xytext=(0, 12),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    fig.suptitle("Cluster size bins vs mean AUROC")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def compute_summary(
    per_cluster_stats: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute summary statistics: Spearman + bin-wise medians.

    Returns dict suitable for JSON serialisation.
    """
    from scipy import stats as sp_stats

    # Filter to clusters with valid pooled AUROC and known train size
    valid = [
        s
        for s in per_cluster_stats.values()
        if s.get("auroc_pooled") is not None and s["n_train"] > 0
    ]

    if not valid:
        return {"n_clusters_total": 0, "error": "no valid clusters"}

    aurocs_pooled = np.array([s["auroc_pooled"] for s in valid])
    sizes = np.array([s["n_train"] for s in valid])
    log_sizes = np.log10(sizes)

    # Spearman correlation
    rho, pval = sp_stats.spearmanr(log_sizes, aurocs_pooled)

    # Bin-wise medians
    bin_stats = []
    for lo, hi, label in _SIZE_BINS:
        bin_aurocs = [
            s["auroc_pooled"] for s in valid if lo <= s["n_train"] < hi
        ]
        entry: Dict[str, Any] = {
            "bin": label,
            "n_clusters": len(bin_aurocs),
        }
        if bin_aurocs:
            entry["median_auroc"] = round(float(np.median(bin_aurocs)), 4)
            entry["mean_auroc"] = round(float(np.mean(bin_aurocs)), 4)
            entry["std_auroc"] = round(float(np.std(bin_aurocs)), 4)
        else:
            entry["median_auroc"] = None
            entry["mean_auroc"] = None
            entry["std_auroc"] = None
        bin_stats.append(entry)

    # Per-AUROC-type summary
    auroc_types = [
        "auroc_iid_vs_unsafe",
        "auroc_benigngen_vs_unsafe",
        "auroc_pooled",
    ]
    type_summaries = {}
    for atype in auroc_types:
        vals = [
            s[atype]
            for s in per_cluster_stats.values()
            if s.get(atype) is not None
        ]
        if vals:
            type_summaries[atype] = {
                "n_clusters": len(vals),
                "median": round(float(np.median(vals)), 4),
                "mean": round(float(np.mean(vals)), 4),
            }
        else:
            type_summaries[atype] = {"n_clusters": 0}

    return {
        "n_clusters_total": len(per_cluster_stats),
        "n_clusters_with_pooled_auroc": len(valid),
        "spearman_log_size_vs_auroc": {
            "rho": round(float(rho), 4),
            "p_value": float(pval),
        },
        "bin_wise_stats": bin_stats,
        "per_auroc_type": type_summaries,
        "fraction_below_0.5": round(float((aurocs_pooled < 0.5).mean()), 4),
    }


def save_per_cluster_stats(
    per_cluster_stats: Dict[int, Dict[str, Any]],
    summary: Dict[str, Any],
    output_path: Path,
) -> None:
    """Save per-cluster stats + summary to JSON."""
    output = {
        "per_cluster": {
            str(cid): s for cid, s in sorted(per_cluster_stats.items())
        },
        "summary": summary,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved {output_path}")


def print_summary(summary: Dict[str, Any]) -> None:
    """Print human-readable summary to stdout."""
    n_valid = summary.get("n_clusters_with_pooled_auroc", 0)
    spearman = summary.get("spearman_log_size_vs_auroc", {})

    print(
        f"\nPer-cluster size vs AUROC analysis (n={n_valid} clusters with valid AUROC)"
    )
    rho = spearman.get("rho", "N/A")
    pval = spearman.get("p_value", "N/A")
    if isinstance(pval, float):
        print(
            f"Spearman correlation log(size) vs AUROC: "
            f"{rho} (p={pval:.2e})"
        )
    else:
        print("Spearman correlation log(size) vs AUROC: N/A")

    print("\nMedian AUROC by size bin:")
    for b in summary.get("bin_wise_stats", []):
        median = b.get("median_auroc")
        mean = b.get("mean_auroc")
        n = b["n_clusters"]
        if median is not None:
            print(
                f"  {b['bin']:>12s}:  median={median:.4f}  "
                f"mean={mean:.4f}  (n={n} clusters)"
            )
        else:
            print(f"  {b['bin']:>12s}:  no clusters")

    frac = summary.get("fraction_below_0.5", 0)
    print(f"\nFraction of clusters with AUROC < 0.5: {100*frac:.1f}%")

    # Per-type breakdown
    per_type = summary.get("per_auroc_type", {})
    print("\nAUROC type breakdown:")
    for atype, info in per_type.items():
        n = info.get("n_clusters", 0)
        if n > 0:
            print(
                f"  {atype}: median={info['median']:.4f}  "
                f"mean={info['mean']:.4f}  (n={n})"
            )
        else:
            print(f"  {atype}: no valid clusters")

    # Recommendation
    bins = summary.get("bin_wise_stats", [])
    small_bin = next((b for b in bins if b["bin"] == "<10"), None)
    large_bins = [b for b in bins if b["bin"] in ("200-1k", ">1k")]
    if (
        small_bin
        and small_bin.get("median_auroc") is not None
        and small_bin["median_auroc"] < 0.55
        and large_bins
        and any(
            b.get("median_auroc") is not None and b["median_auroc"] > 0.65
            for b in large_bins
        )
    ):
        print(
            "\n** RECOMMENDATION: Small clusters (<10 samples) have notably "
            "lower AUROC. Consider increasing min_cluster_size or merging "
            "small clusters. **"
        )


def plot_quality_vs_auroc(
    per_cluster_stats: Dict[int, Dict[str, Any]],
    cluster_quality: Dict[int, Dict[str, float]],
    output_path: Path,
    title: str = "",
) -> None:
    """Scatter: each quality proxy vs per-cluster AUROC.

    This is the key diagnostic plot — shows which quality proxy
    correlates best with AUROC, enabling size-independent cluster
    adoption rules.

    Args:
        per_cluster_stats: from compute_per_cluster_auroc
        cluster_quality: cluster_id -> quality metrics dict
        output_path: where to save the PNG
        title: optional suptitle
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats as sp_stats

    metrics = [
        ("mask_density", "Mask density"),
        ("train_res_std", "Train residual std"),
        ("memorization_gap", "Memorization gap"),
        ("centroid_spread", "Centroid spread"),
        ("mask_support_ratio", "Mask support ratio"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for i, (key, label) in enumerate(metrics):
        ax = axes_flat[i]
        xs = []
        ys = []
        for cid, stats in per_cluster_stats.items():
            auroc = stats.get("auroc_pooled")
            if auroc is None:
                continue
            q = cluster_quality.get(cid, {})
            val = q.get(key)
            if val is None:
                continue
            xs.append(val)
            ys.append(auroc)

        if len(xs) >= 3:
            ax.scatter(xs, ys, alpha=0.5, s=20, color="steelblue")
            ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
            if len(xs) >= 5:
                rho, pval = sp_stats.spearmanr(xs, ys)
                ax.set_title(
                    f"{label}\n(ρ={rho:.3f}, p={pval:.2e})",
                    fontsize=10,
                )
            else:
                ax.set_title(label, fontsize=10)
        else:
            ax.set_title(f"{label} (insufficient data)", fontsize=10)

        ax.set_xlabel(label)
        ax.set_ylabel("AUROC (pooled)")

    # Hide unused subplot
    for j in range(len(metrics), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title or "Quality proxies vs per-cluster AUROC", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def load_cluster_quality(
    run_dir: Path,
) -> Optional[Dict[int, Dict[str, float]]]:
    """Load cluster_quality.json from a run directory.

    Returns None if the file doesn't exist.
    """
    path = run_dir / "cluster_quality.json"
    if not path.exists():
        return None
    with open(path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def get_train_cluster_sizes_for_lora(
    run_dir: Path,
    model: Any,
) -> Dict[int, int]:
    """Get training cluster sizes for a LoRA run.

    Tries (in order):
    1. lora_diagnostic_summary.json -> train_cluster_sizes
    2. step1_run_dir -> cluster_diagnostics.json -> cluster_table
    """
    # Try lora diagnostic summary
    summary_path = (
        run_dir / "diagnostics" / "lora" / "lora_diagnostic_summary.json"
    )
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        tcs = summary.get("train_cluster_sizes", {})
        if tcs:
            return {int(k): v for k, v in tcs.items()}

    # Fallback: step1 cluster diagnostics
    step1_dir = getattr(model, "_step1_run_dir", None)
    if step1_dir is None:
        # Try loading from checkpoint
        ckpt_path = run_dir / "learned_mask_lora_trained.pt"
        if ckpt_path.exists():
            import torch

            state = torch.load(
                ckpt_path, map_location="cpu", weights_only=False
            )
            step1_dir = state.get("step1_run_dir")

    if step1_dir:
        diag_path = (
            Path(step1_dir)
            / "diagnostics"
            / "clusters"
            / "cluster_diagnostics.json"
        )
        if diag_path.exists():
            with open(diag_path) as f:
                diag = json.load(f)
            return {
                entry["cluster"]: entry["n_samples"]
                for entry in diag.get("cluster_table", [])
            }

    logger.warning("Could not find train cluster sizes")
    return {}


def get_union_support_sizes(
    run_dir: Path,
    model: Any,
    is_lora: bool = False,
) -> Optional[Dict[int, int]]:
    """Get union_support_size per cluster from cluster_diagnostics.json."""
    if is_lora:
        step1_dir = getattr(model, "_step1_run_dir", None)
        if step1_dir is None:
            ckpt_path = run_dir / "learned_mask_lora_trained.pt"
            if ckpt_path.exists():
                import torch

                state = torch.load(
                    ckpt_path, map_location="cpu", weights_only=False
                )
                step1_dir = state.get("step1_run_dir")
        if step1_dir:
            diag_path = (
                Path(step1_dir)
                / "diagnostics"
                / "clusters"
                / "cluster_diagnostics.json"
            )
        else:
            return None
    else:
        diag_path = (
            run_dir / "diagnostics" / "clusters" / "cluster_diagnostics.json"
        )

    if diag_path.exists():
        with open(diag_path) as f:
            diag = json.load(f)
        return {
            entry["cluster"]: entry.get("union_support_size", -1)
            for entry in diag.get("cluster_table", [])
        }
    return None
