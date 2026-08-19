#!/usr/bin/env python3
"""Post-hoc diagnostics for completed learned_mask runs.

Tests whether the mask creates separation between safe and unsafe
residuals, independent of the scoring function.

Works with ANY completed learned_mask run (whitened or L2, continuous
or binary STE). Diagnostics A-F recompute everything from the checkpoint
and the eval data; only G (cluster size vs AUROC) needs the
<method>_cluster_assignments.npz / _raw_scores.npz artifacts a completed
evaluation writes.

Usage:
    python scripts/anodet_algos/diagnose_learned_mask.py \
        --run-dir /path/to/learned_mask_run/0

    # Specify model key explicitly (otherwise inferred from checkpoint):
    python scripts/anodet_algos/diagnose_learned_mask.py \
        --run-dir /path/to/run --model-key qwen3_8b

Outputs go to {run_dir}/diagnostics/post_hoc/.
GPU required for forward passes (residual computation).
Estimated time: ~5-10 minutes per run (one forward pass per dataset).
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from unsupervised_llm_safety.analysis.cluster_data import (
    load_eval_data_with_sources,
)
from unsupervised_llm_safety.anodet_algos.learned_mask import LearnedMaskAnodet
from unsupervised_llm_safety.data.dataset_registry import (
    set_complex_data_mode,
    set_last_token_mode,
)

logger = logging.getLogger(__name__)

# Model keys understood by the dataset registry, used to recover a model_key
# from a run directory name when the Hydra config is unavailable.
MODEL_KEYS = (
    "qwen3_8b",
    "gpt_oss_20b",
    "gemma4_26b",
    "qwen2_1.5b",
    "ministral_8b",
    "llama3_8b",
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def load_model(run_dir: Path, device: str) -> LearnedMaskAnodet:
    """Load learned_mask or learned_mask_lora checkpoint from run dir."""
    ckpt_path = run_dir / "learned_mask_trained.pt"
    if ckpt_path.exists():
        model = LearnedMaskAnodet(device=device)
        model.load(str(ckpt_path))
        return model

    # Also support learned_mask_lora runs (delegates via __getattr__)
    lora_path = run_dir / "learned_mask_lora_trained.pt"
    if lora_path.exists():
        from unsupervised_llm_safety.anodet_algos.learned_mask_lora import (
            LearnedMaskLoRAAnodet,
        )

        model = LearnedMaskLoRAAnodet(device=device)
        model.load(str(lora_path))
        return model

    raise FileNotFoundError(f"No checkpoint at {ckpt_path} or {lora_path}")


def infer_model_key(run_dir: Path) -> str:
    """Infer model_key from Hydra config or directory name."""
    hydra_cfg = run_dir / ".hydra" / "overrides.yaml"
    if hydra_cfg.exists():
        import yaml  # type: ignore[import-untyped]

        with open(hydra_cfg) as f:
            overrides = yaml.safe_load(f) or []
        for o in overrides:
            if "model_key=" in str(o):
                return str(o).split("model_key=")[1].strip()

    # Fallback: parse from directory name
    name = run_dir.parent.name
    for key in MODEL_KEYS:
        if key in name:
            return key
    raise ValueError(
        f"Cannot infer model_key from {run_dir}. "
        "Use --model-key explicitly."
    )


def encode_and_assign_clusters(
    model: LearnedMaskAnodet,
    data: np.ndarray,
) -> tuple:
    """Encode data and assign clusters once for all diagnostics.

    Returns (encoded, clusters) where encoded is (N, sae_dim)
    and clusters is (N,) int array.
    """
    encoded = model._encode_all(data)
    _, assigns = model.kmeans.index.search(encoded.astype(np.float32), 1)
    clusters = assigns.flatten()
    return encoded, clusters


def compute_identity_norms(
    model: LearnedMaskAnodet,
    data: np.ndarray,
    encoded: np.ndarray,
) -> np.ndarray:
    """Compute ||r_identity|| = ||x - decode(encode(x))|| without mask.

    Memory-efficient: computes norms batch-by-batch, never stores
    all residual vectors at once.
    Returns (N,) norm array.
    """
    is_lora = hasattr(model, "_lora_adapters")
    decoder = model._step1.decoder if is_lora else model.decoder
    all_norms: List[np.ndarray] = []
    bs = model.analysis_batch_size

    with torch.no_grad():
        for i in range(0, len(data), bs):
            batch_raw = torch.tensor(
                data[i : i + bs],
                dtype=torch.float32,
                device=model.device,
            )
            batch_enc = torch.tensor(
                encoded[i : i + bs],
                dtype=torch.float32,
                device=model.device,
            )
            x_hat = decoder(batch_enc)
            diff = batch_raw - x_hat
            norms = torch.linalg.norm(diff, dim=1)
            all_norms.append(norms.cpu().numpy())

    return np.concatenate(all_norms, axis=0)


def compute_masked_residuals(
    model: LearnedMaskAnodet,
    data: np.ndarray,
    encoded: np.ndarray,
    clusters: np.ndarray,
) -> np.ndarray:
    """Compute masked residual vectors using pre-computed encoded data.

    Returns (N, raw_dim) residual array. For raw_dim=1536 and 367k
    samples this is ~2GB — acceptable when encoded is already freed.
    """
    # Check if model is LoRA
    is_lora = hasattr(model, "_lora_adapters")
    if is_lora:
        return model._compute_residual_vectors(data, encoded, clusters)

    # Base learned_mask path
    s1 = model
    assert s1.encoder is not None and s1.decoder is not None
    s1.encoder.eval()
    s1.decoder.eval()
    bs = s1.analysis_batch_size
    all_residuals: List[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(data), bs):
            batch_raw = torch.tensor(
                data[i : i + bs],
                dtype=torch.float32,
                device=s1.device,
            )
            batch_enc = torch.tensor(
                encoded[i : i + bs],
                dtype=torch.float32,
                device=s1.device,
            )
            batch_clusters = clusters[i : i + bs]
            B = len(batch_raw)

            z_masked = torch.zeros(B, s1.sae_dim, device=s1.device)
            for j in range(B):
                c = int(batch_clusters[j])
                cm = s1.cluster_masks.get(c)
                if cm is None or len(cm["union_indices"]) == 0:
                    z_masked[j] = batch_enc[j]
                    continue
                idx = cm["union_indices"].to(s1.device)
                vals = cm["mask_values"].to(s1.device)
                z_masked[j, idx] = batch_enc[j, idx] * vals

            x_hat = s1.decoder(z_masked)
            diff = batch_raw - x_hat
            all_residuals.append(diff.cpu().numpy())

    return np.concatenate(all_residuals, axis=0)


def compute_masked_norms(
    model: LearnedMaskAnodet,
    data: np.ndarray,
    encoded: np.ndarray,
    clusters: np.ndarray,
) -> np.ndarray:
    """Compute ||r_masked|| norms without storing full residual array.

    Memory-efficient version for diagnostics that only need norms.
    Returns (N,) norm array.
    """
    is_lora = hasattr(model, "_lora_adapters")
    if is_lora:
        s1 = model._step1
        assert s1 is not None and s1.decoder is not None
    else:
        s1 = model
        assert s1.encoder is not None and s1.decoder is not None
        s1.encoder.eval()
        s1.decoder.eval()

    bs = model.analysis_batch_size
    all_norms: List[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(data), bs):
            batch_raw = torch.tensor(
                data[i : i + bs],
                dtype=torch.float32,
                device=model.device,
            )
            batch_enc = torch.tensor(
                encoded[i : i + bs],
                dtype=torch.float32,
                device=model.device,
            )
            batch_clusters = clusters[i : i + bs]
            B = len(batch_raw)

            if is_lora:
                # LoRA path: per-cluster encode/decode
                x_hat = torch.zeros_like(batch_raw)
                for c_id in np.unique(batch_clusters):
                    c_id = int(c_id)
                    c_idx = np.where(batch_clusters == c_id)[0]
                    c_raw = batch_raw[c_idx]
                    cm = s1.cluster_masks.get(c_id)

                    if c_id in model._lora_adapters:
                        z = model._encode_with_lora(c_raw, c_id)
                    else:
                        z = batch_enc[c_idx]

                    if cm is not None and len(cm["union_indices"]) > 0:
                        idx = cm["union_indices"].to(model.device)
                        vals = cm["mask_values"].to(model.device)
                        z_m = torch.zeros(
                            len(c_idx), s1.sae_dim, device=model.device
                        )
                        z_m[:, idx] = z[:, idx] * vals
                    else:
                        z_m = z

                    if c_id in model._lora_adapters:
                        x_hat[c_idx] = model._decode_with_lora(z_m, c_id)
                    else:
                        x_hat[c_idx] = s1.decoder(z_m)
            else:
                # Base path: apply mask + decode
                z_masked = torch.zeros(B, s1.sae_dim, device=s1.device)
                for j in range(B):
                    c = int(batch_clusters[j])
                    cm = s1.cluster_masks.get(c)
                    if cm is None or len(cm["union_indices"]) == 0:
                        z_masked[j] = batch_enc[j]
                        continue
                    idx = cm["union_indices"].to(s1.device)
                    vals = cm["mask_values"].to(s1.device)
                    z_masked[j, idx] = batch_enc[j, idx] * vals
                x_hat = s1.decoder(z_masked)

            diff = batch_raw - x_hat
            norms = torch.linalg.norm(diff, dim=1)
            all_norms.append(norms.cpu().numpy())

    return np.concatenate(all_norms, axis=0)


# ------------------------------------------------------------------
# A. Mask sparsity distribution (no forward passes)
# ------------------------------------------------------------------


def diagnose_mask_sparsity(
    model: LearnedMaskAnodet,
    out_dir: Path,
) -> Dict[str, Any]:
    """Analyze mask sparsity from checkpoint. No GPU needed."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_cluster = []
    for c in range(model.n_clusters):
        cm = model.cluster_masks.get(c)
        if cm is None or len(cm["union_indices"]) == 0:
            per_cluster.append(
                {
                    "cluster": c,
                    "union_size": 0,
                    "n_above_05": 0,
                    "n_above_01": 0,
                    "n_above_001": 0,
                    "l0_at_05": 0,
                    "mask_mean": 0.0,
                }
            )
            continue
        vals = cm["mask_values"].numpy()
        per_cluster.append(
            {
                "cluster": c,
                "union_size": len(vals),
                "n_above_05": int((np.abs(vals) > 0.5).sum()),
                "n_above_01": int((np.abs(vals) > 0.1).sum()),
                "n_above_001": int((np.abs(vals) > 0.01).sum()),
                "l0_at_05": int((vals > 0.5).sum()),
                "mask_mean": round(float(np.abs(vals).mean()), 4),
            }
        )

    with open(out_dir / "mask_sparsity_per_cluster.json", "w") as f:
        json.dump(per_cluster, f, indent=2)

    # Summary stats
    nonempty = [p for p in per_cluster if p["union_size"] > 0]
    unions = [p["union_size"] for p in nonempty]
    above05 = [p["n_above_05"] for p in nonempty]
    means = [p["mask_mean"] for p in nonempty]
    pruned_50pct = sum(
        1 for p in nonempty if p["n_above_05"] < 0.5 * p["union_size"]
    )

    summary = {
        "n_clusters_total": model.n_clusters,
        "n_clusters_nonempty": len(nonempty),
        "union_size_mean": round(float(np.mean(unions)), 1),
        "union_size_median": round(float(np.median(unions)), 1),
        "n_above_05_mean": round(float(np.mean(above05)), 1),
        "n_above_05_median": round(float(np.median(above05)), 1),
        "mask_mean_mean": round(float(np.mean(means)), 4),
        "clusters_with_gt50pct_pruned_at_05": pruned_50pct,
        "clusters_with_gt50pct_pruned_pct": round(
            100 * pruned_50pct / max(len(nonempty), 1), 1
        ),
    }

    # Plot: histogram of mask mean values
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(means, bins=30, color="steelblue", edgecolor="k", alpha=0.7)
    ax.axvline(x=0.5, color="red", linestyle="--", label="0.5 threshold")
    ax.set_xlabel("Mean |mask value|")
    ax.set_ylabel("Count")
    ax.set_title(f"Mask mean value distribution (n={len(nonempty)})")
    ax.legend()

    ax = axes[1]
    ax.hist(above05, bins=30, color="coral", edgecolor="k", alpha=0.7)
    ax.set_xlabel("Features above 0.5")
    ax.set_ylabel("Count")
    ax.set_title("Features kept per cluster (at 0.5 threshold)")

    fig.tight_layout()
    fig.savefig(out_dir / "mask_sparsity_histograms.png", dpi=150)
    plt.close(fig)

    return summary


# ------------------------------------------------------------------
# B. Residual magnitude by slice (needs forward passes)
# ------------------------------------------------------------------


def diagnose_residual_magnitudes(
    model: LearnedMaskAnodet,
    eval_data: np.ndarray,
    eval_sources: np.ndarray,
    dataset_names: List[str],
    encoded: np.ndarray,
    clusters: np.ndarray,
    out_dir: Path,
) -> Dict[str, Any]:
    """Compare ||r_masked|| vs ||r_identity|| per slice."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Compute masked residual norms (memory-efficient)
    logger.info("Computing masked residual norms...")
    norms_masked = compute_masked_norms(model, eval_data, encoded, clusters)

    # Compute identity residual norms (no mask, memory-efficient)
    logger.info("Computing identity residual norms...")
    norms_identity = compute_identity_norms(model, eval_data, encoded)

    # Per-slice statistics
    per_slice = {}
    iid_mean_masked: float = 0.0
    iid_mean_identity: float = 0.0

    for sid, name in enumerate(dataset_names):
        mask = eval_sources == sid
        n = int(mask.sum())
        if n == 0:
            continue
        nm = norms_masked[mask]
        ni = norms_identity[mask]
        entry = {
            "n_samples": n,
            "masked_mean": round(float(nm.mean()), 4),
            "masked_median": round(float(np.median(nm)), 4),
            "masked_std": round(float(nm.std()), 4),
            "identity_mean": round(float(ni.mean()), 4),
            "identity_median": round(float(np.median(ni)), 4),
            "identity_std": round(float(ni.std()), 4),
        }
        per_slice[name] = entry

        if name.startswith("iid:"):
            iid_mean_masked = nm.mean()
            iid_mean_identity = ni.mean()

    # Compute ratios to IID
    if iid_mean_masked is not None:
        for name, entry in per_slice.items():
            entry["ratio_masked_to_iid"] = round(
                entry["masked_mean"] / max(iid_mean_masked, 1e-10), 4
            )
            entry["ratio_identity_to_iid"] = round(
                entry["identity_mean"] / max(iid_mean_identity, 1e-10),
                4,
            )

    with open(out_dir / "residual_magnitudes.json", "w") as f:
        json.dump(per_slice, f, indent=2)

    # Plot: KDE of masked residual norms, IID vs each unsafe slice
    unsafe_names = [n for n in dataset_names if not n.startswith("iid:")]
    iid_mask = np.zeros(len(eval_sources), dtype=bool)
    for sid, name in enumerate(dataset_names):
        if name.startswith("iid:"):
            iid_mask |= eval_sources == sid
    iid_norms = norms_masked[iid_mask]

    n_plots = min(len(unsafe_names), 12)
    if n_plots > 0:
        cols = min(3, n_plots)
        rows = (n_plots + cols - 1) // cols
        fig, axes = plt.subplots(
            rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False
        )
        for i, name in enumerate(unsafe_names[:n_plots]):
            ax = axes[i // cols][i % cols]
            sid = dataset_names.index(name)
            s_mask = eval_sources == sid
            s_norms = norms_masked[s_mask]

            ax.hist(
                iid_norms,
                bins=40,
                alpha=0.5,
                density=True,
                color="green",
                label="IID safe",
            )
            ax.hist(
                s_norms,
                bins=40,
                alpha=0.5,
                density=True,
                color="red",
                label=name.split(":")[-1],
            )
            ax.set_xlabel("||r_masked||")
            ax.set_title(name.split(":")[-1], fontsize=9)
            ax.legend(fontsize=7)

        for i in range(n_plots, rows * cols):
            axes[i // cols][i % cols].set_visible(False)
        fig.suptitle("Masked residual: IID vs unsafe", y=1.02)
        fig.tight_layout()
        fig.savefig(
            out_dir / "residual_distributions.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    return {
        "iid_mean_masked": round(float(iid_mean_masked), 4),
        "iid_mean_identity": round(float(iid_mean_identity), 4),
    }


# ------------------------------------------------------------------
# C. Identity-mask AUROC ablation
# ------------------------------------------------------------------


def diagnose_identity_auroc(
    model: LearnedMaskAnodet,
    eval_data: np.ndarray,
    eval_sources: np.ndarray,
    dataset_names: List[str],
    encoded: np.ndarray,
    run_dir: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    """AUROC using identity residual (no mask) vs actual run AUROC."""
    # Compute identity residual norms as scores (memory-efficient)
    identity_scores = compute_identity_norms(model, eval_data, encoded)

    # Pool IID scores
    iid_mask = np.zeros(len(eval_sources), dtype=bool)
    for sid, name in enumerate(dataset_names):
        if name.startswith("iid:"):
            iid_mask |= eval_sources == sid
    iid_scores = identity_scores[iid_mask]

    # Per-slice AUROC
    identity_aurocs = {}
    for sid, name in enumerate(dataset_names):
        if name.startswith("iid:"):
            continue
        s_mask = eval_sources == sid
        n = int(s_mask.sum())
        if n < 2:
            continue
        s_scores = identity_scores[s_mask]
        labels = np.concatenate(
            [np.zeros(len(iid_scores)), np.ones(len(s_scores))]
        )
        combined = np.concatenate([iid_scores, s_scores])
        try:
            auroc = float(roc_auc_score(labels, combined))
        except ValueError:
            auroc = float("nan")
        identity_aurocs[name] = round(auroc, 4)

    # Load actual run's AUROC for comparison
    results_path = run_dir / "learned_mask_analysis_results.json"
    actual_aurocs = {}
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        ps = results.get("per_slice_metrics", {}).get("per_slice", {})
        for ds_name, ds_m in ps.items():
            actual_aurocs[ds_name] = ds_m.get("auroc", float("nan"))

    # Compute deltas
    comparison = {}
    for name in sorted(
        set(list(identity_aurocs.keys()) + list(actual_aurocs.keys()))
    ):
        id_auc = identity_aurocs.get(name, float("nan"))
        act_auc = actual_aurocs.get(name, float("nan"))
        delta = (
            act_auc - id_auc
            if not (np.isnan(id_auc) or np.isnan(act_auc))
            else float("nan")
        )
        comparison[name] = {
            "identity_auroc": id_auc,
            "actual_auroc": act_auc,
            "delta": round(delta, 4) if not np.isnan(delta) else None,
        }

    with open(out_dir / "identity_mask_auroc.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # Summary
    deltas = [
        v["delta"] for v in comparison.values() if v["delta"] is not None
    ]
    id_vals = [
        v["identity_auroc"]
        for v in comparison.values()
        if not np.isnan(v["identity_auroc"])
    ]
    act_vals = [
        v["actual_auroc"]
        for v in comparison.values()
        if not np.isnan(v["actual_auroc"])
    ]

    return {
        "identity_mean_auroc": (
            round(float(np.mean(id_vals)), 4) if id_vals else None
        ),
        "actual_mean_auroc": (
            round(float(np.mean(act_vals)), 4) if act_vals else None
        ),
        "mean_delta": round(float(np.mean(deltas)), 4) if deltas else None,
    }


# ------------------------------------------------------------------
# D. Per-dimension variance: safe vs unsafe
# ------------------------------------------------------------------


def diagnose_dimension_variance(
    model: LearnedMaskAnodet,
    eval_data: np.ndarray,
    eval_sources: np.ndarray,
    dataset_names: List[str],
    encoded: np.ndarray,
    clusters: np.ndarray,
    out_dir: Path,
) -> Dict[str, Any]:
    """Per-dimension variance ratio (unsafe/safe) per cluster."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Compute masked residuals (raw_dim=1536, ~2GB for 367k samples)
    logger.info("Computing masked residuals for dimension variance...")
    r_all = compute_masked_residuals(model, eval_data, encoded, clusters)

    # Split into safe (IID) and unsafe
    safe_mask = np.zeros(len(eval_sources), dtype=bool)
    for sid, name in enumerate(dataset_names):
        if name.startswith("iid:"):
            safe_mask |= eval_sources == sid
    unsafe_mask = ~safe_mask

    safe_clusters = clusters[safe_mask]
    unsafe_clusters = clusters[unsafe_mask]
    r_safe = r_all[safe_mask]
    r_unsafe = r_all[unsafe_mask]

    # Per-cluster analysis
    min_unsafe = 20
    per_cluster_results = []
    max_ratios = []

    for c in range(model.n_clusters):
        safe_in_c = r_safe[safe_clusters == c]
        unsafe_in_c = r_unsafe[unsafe_clusters == c]
        n_safe = len(safe_in_c)
        n_unsafe = len(unsafe_in_c)

        if n_unsafe < min_unsafe or n_safe < min_unsafe:
            continue

        var_safe = safe_in_c.var(axis=0) + 1e-10
        var_unsafe = unsafe_in_c.var(axis=0) + 1e-10
        ratio = var_unsafe / var_safe

        max_r = float(ratio.max())
        max_ratios.append(max_r)
        per_cluster_results.append(
            {
                "cluster": c,
                "n_safe": n_safe,
                "n_unsafe": n_unsafe,
                "max_dim_ratio": round(max_r, 2),
                "mean_dim_ratio": round(float(ratio.mean()), 4),
                "n_dims_ratio_gt2": int((ratio > 2.0).sum()),
                "n_dims_ratio_gt5": int((ratio > 5.0).sum()),
                "n_dims_ratio_gt10": int((ratio > 10.0).sum()),
            }
        )

    with open(out_dir / "per_cluster_dimension_variance.json", "w") as f:
        json.dump(per_cluster_results, f, indent=2)

    # Summary
    n_eligible = len(per_cluster_results)
    n_max_gt2 = sum(1 for r in max_ratios if r > 2.0)
    n_max_gt5 = sum(1 for r in max_ratios if r > 5.0)
    n_max_gt10 = sum(1 for r in max_ratios if r > 10.0)

    summary = {
        "n_eligible_clusters": n_eligible,
        "n_max_ratio_gt_2": n_max_gt2,
        "n_max_ratio_gt_5": n_max_gt5,
        "n_max_ratio_gt_10": n_max_gt10,
        "frac_gt_2": round(n_max_gt2 / max(n_eligible, 1), 3),
        "frac_gt_5": round(n_max_gt5 / max(n_eligible, 1), 3),
        "frac_gt_10": round(n_max_gt10 / max(n_eligible, 1), 3),
    }

    # Plot histogram of max ratio per cluster
    if max_ratios:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(
            np.clip(max_ratios, 0, 50),
            bins=40,
            color="steelblue",
            edgecolor="k",
            alpha=0.7,
        )
        ax.axvline(x=2.0, color="orange", linestyle="--", label=">2")
        ax.axvline(x=5.0, color="red", linestyle="--", label=">5")
        ax.set_xlabel("Max dimension variance ratio (unsafe/safe)")
        ax.set_ylabel("Count")
        ax.set_title(f"Per-cluster max dim ratio (n={n_eligible} eligible)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "dimension_ratio_histogram.png", dpi=150)
        plt.close(fig)

    return summary


# ------------------------------------------------------------------
# E. Routing: where do unsafe samples go
# ------------------------------------------------------------------


def diagnose_routing(
    model: LearnedMaskAnodet,
    eval_sources: np.ndarray,
    dataset_names: List[str],
    clusters: np.ndarray,
    out_dir: Path,
) -> Dict[str, Any]:
    """Analyze which clusters absorb unsafe samples."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    safe_mask = np.zeros(len(eval_sources), dtype=bool)
    for sid, name in enumerate(dataset_names):
        if name.startswith("iid:"):
            safe_mask |= eval_sources == sid

    # Per-cluster unsafe density
    cluster_total = np.bincount(clusters, minlength=model.n_clusters)
    unsafe_clusters = clusters[~safe_mask]
    cluster_unsafe = np.bincount(unsafe_clusters, minlength=model.n_clusters)
    cluster_density = np.where(
        cluster_total > 0,
        cluster_unsafe / cluster_total,
        0.0,
    )

    # Per-slice routing
    per_slice_routing = {}
    for sid, name in enumerate(dataset_names):
        s_mask = eval_sources == sid
        n = int(s_mask.sum())
        if n == 0:
            continue
        s_clusters = clusters[s_mask]
        unique, counts = np.unique(s_clusters, return_counts=True)
        top_clusters = sorted(
            zip(unique.tolist(), counts.tolist()),
            key=lambda x: -x[1],
        )[:10]
        per_slice_routing[name] = {
            "n_samples": n,
            "n_unique_clusters": len(unique),
            "top_10_clusters": [
                {"cluster": c, "count": cnt} for c, cnt in top_clusters
            ],
        }

    # Top/bottom clusters by unsafe density
    nonempty = np.where(cluster_total > 0)[0]
    sorted_by_density = sorted(nonempty, key=lambda c: -cluster_density[c])
    top_unsafe = [
        {
            "cluster": int(c),
            "n_total": int(cluster_total[c]),
            "n_unsafe": int(cluster_unsafe[c]),
            "density": round(float(cluster_density[c]), 4),
        }
        for c in sorted_by_density[:10]
    ]
    bottom_unsafe = [
        {
            "cluster": int(c),
            "n_total": int(cluster_total[c]),
            "n_unsafe": int(cluster_unsafe[c]),
            "density": round(float(cluster_density[c]), 4),
        }
        for c in sorted_by_density[-10:]
    ]

    result = {
        "per_slice_routing": per_slice_routing,
        "top_10_unsafe_clusters": top_unsafe,
        "bottom_10_unsafe_clusters": bottom_unsafe,
    }
    with open(out_dir / "routing_stats.json", "w") as f:
        json.dump(result, f, indent=2)

    # Plot: cluster unsafe density, sorted descending
    densities_sorted = np.sort(cluster_density[nonempty])[::-1]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(
        range(len(densities_sorted)),
        densities_sorted,
        color="coral",
        alpha=0.7,
    )
    ax.set_xlabel("Cluster (sorted by unsafe density)")
    ax.set_ylabel("Fraction unsafe")
    ax.set_title(
        f"Cluster unsafe density ({len(nonempty)} non-empty clusters)"
    )
    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "cluster_unsafe_density.png", dpi=150)
    plt.close(fig)

    return {
        "n_clusters_gt50pct_unsafe": int(
            (cluster_density[nonempty] > 0.5).sum()
        ),
        "mean_unsafe_density": round(
            float(cluster_density[nonempty].mean()), 4
        ),
    }


# ------------------------------------------------------------------
# F. Top/bottom scoring unsafe samples
# ------------------------------------------------------------------


def diagnose_hard_easy_cases(
    model: LearnedMaskAnodet,
    eval_data: np.ndarray,
    eval_sources: np.ndarray,
    dataset_names: List[str],
    encoded: np.ndarray,
    clusters: np.ndarray,
    out_dir: Path,
) -> None:
    """Identify highest/lowest scoring unsafe samples per slice."""
    # Compute masked residuals and norms
    r_all = compute_masked_residuals(model, eval_data, encoded, clusters)
    norms = np.linalg.norm(r_all, axis=1)

    # Cluster sizes from training
    train_labels = model._diagnostic_cluster_labels
    train_sizes = {}
    if train_labels is not None:
        for c in range(model.n_clusters):
            train_sizes[c] = int((train_labels == c).sum())

    results: Dict[str, Any] = {}

    for sid, name in enumerate(dataset_names):
        if name.startswith("iid:"):
            continue
        s_mask = eval_sources == sid
        n = int(s_mask.sum())
        if n < 10:
            continue

        s_indices = np.where(s_mask)[0]
        s_norms = norms[s_indices]
        sorted_idx = np.argsort(s_norms)

        cases: dict = {"hardest_10": [], "easiest_10": []}
        for label, indices in [
            ("hardest_10", sorted_idx[:10]),
            ("easiest_10", sorted_idx[-10:][::-1]),
        ]:
            for idx in indices:
                global_idx = s_indices[idx]
                c = int(clusters[global_idx])
                cm = model.cluster_masks.get(c, {})
                vals = cm.get("mask_values", torch.tensor([]))
                if len(vals) > 0:
                    if hasattr(vals, "numpy"):
                        vals_np = vals.numpy()
                    else:
                        vals_np = vals
                    mask_sparsity = int((np.abs(vals_np) > 0.5).sum())
                else:
                    mask_sparsity = 0

                # Top contributing dimensions
                residual = r_all[global_idx]
                top_dims = np.argsort(np.abs(residual))[-20:][::-1]

                cases[label].append(
                    {
                        "norm": round(float(s_norms[idx]), 4),
                        "cluster": c,
                        "cluster_train_size": train_sizes.get(c, -1),
                        "mask_sparsity_05": mask_sparsity,
                        "top_20_dims": [
                            {
                                "dim": int(d),
                                "value": round(float(residual[d]), 4),
                            }
                            for d in top_dims
                        ],
                    }
                )
        results[name] = cases

    with open(out_dir / "hard_easy_unsafe_cases.json", "w") as f:
        json.dump(results, f, indent=2)


# ------------------------------------------------------------------
# G. Cluster size vs per-cluster AUROC
# ------------------------------------------------------------------


def diagnose_cluster_size_auroc(
    model: LearnedMaskAnodet,
    run_dir: Path,
    out_dir: Path,
) -> Optional[Dict[str, Any]]:
    """Cluster size vs per-cluster AUROC using saved npz artifacts."""
    from unsupervised_llm_safety.analysis.cluster_size_auroc import (
        compute_per_cluster_auroc,
        compute_summary,
        get_union_support_sizes,
        plot_cluster_size_histogram_with_auroc_bins,
        plot_cluster_size_vs_auroc,
        print_summary,
        save_per_cluster_stats,
    )

    # Detect npz prefix
    is_lora = (run_dir / "learned_mask_lora_trained.pt").exists()
    prefix = "learned_mask_lora" if is_lora else "learned_mask"

    ca_file = run_dir / f"{prefix}_cluster_assignments.npz"
    raw_file = run_dir / f"{prefix}_raw_scores.npz"
    cat_file = run_dir / f"{prefix}_categories.json"

    # Error out if artifacts missing (no fallback)
    missing = []
    for f in [ca_file, raw_file, cat_file]:
        if not f.exists():
            missing.append(str(f))
    if missing:
        logger.error(
            "Cluster size vs AUROC requires saved eval artifacts. "
            f"Missing: {missing}. Run evaluation first."
        )
        return None

    # Load artifacts
    cluster_assignments_npz = np.load(ca_file)
    cluster_assignments = {
        k: cluster_assignments_npz[k] for k in cluster_assignments_npz.files
    }
    raw_scores_npz = np.load(raw_file)
    raw_scores = {k: raw_scores_npz[k] for k in raw_scores_npz.files}
    with open(cat_file) as cat_fh:
        categories = json.load(cat_fh)

    # Get train cluster sizes
    if is_lora:
        from unsupervised_llm_safety.analysis.cluster_size_auroc import (
            get_train_cluster_sizes_for_lora,
        )

        train_sizes = get_train_cluster_sizes_for_lora(run_dir, model)
    else:
        # From cluster_diagnostics.json
        diag_path = (
            run_dir / "diagnostics" / "clusters" / "cluster_diagnostics.json"
        )
        train_sizes = {}
        if diag_path.exists():
            with open(diag_path) as diag_fh:
                diag = json.load(diag_fh)
            train_sizes = {
                e["cluster"]: e["n_samples"]
                for e in diag.get("cluster_table", [])
            }
        if not train_sizes:
            logger.warning(
                "No train cluster sizes found — "
                "size-based analysis will be limited"
            )

    # Get union support sizes
    union_sizes = get_union_support_sizes(run_dir, model, is_lora=is_lora)

    # Compute per-cluster AUROC (3 variants)
    per_cluster = compute_per_cluster_auroc(
        cluster_assignments=cluster_assignments,
        raw_scores=raw_scores,
        categories=categories,
        train_cluster_sizes=train_sizes,
        union_support_sizes=union_sizes,
    )

    # Compute summary
    summary = compute_summary(per_cluster)

    # Save JSON
    save_per_cluster_stats(
        per_cluster,
        summary,
        out_dir / "per_cluster_auroc.json",
    )

    # Plots
    plot_cluster_size_vs_auroc(
        per_cluster,
        out_dir / "cluster_size_vs_auroc.png",
        title=f"Cluster size vs AUROC ({run_dir.parent.name})",
    )
    plot_cluster_size_histogram_with_auroc_bins(
        per_cluster,
        out_dir / "cluster_size_auroc_bins.png",
    )

    # Print summary
    print_summary(summary)

    return summary


# ------------------------------------------------------------------
# Meta-summary
# ------------------------------------------------------------------


def print_meta_summary(
    sparsity_summary: Dict,
    residual_summary: Dict,
    auroc_summary: Dict,
    variance_summary: Dict,
    routing_summary: Dict,
) -> None:
    """Print one-screen summary with verdict."""
    print("\n" + "=" * 70)
    print("LEARNED MASK POST-HOC DIAGNOSTIC SUMMARY")
    print("=" * 70)

    # 1. Mask sparsity
    pct = sparsity_summary.get("clusters_with_gt50pct_pruned_pct", 0)
    print(
        f"\n1. MASK SPARSITY: "
        f"{sparsity_summary['clusters_with_gt50pct_pruned_at_05']} / "
        f"{sparsity_summary['n_clusters_nonempty']} clusters have "
        f">50% features pruned at 0.5 threshold ({pct}%)"
    )
    print(
        f"   Mean mask value: "
        f"{sparsity_summary['mask_mean_mean']:.4f}  "
        f"(1.0 = identity, 0.0 = fully pruned)"
    )

    # 2. Residual ratio
    iid_m = residual_summary.get("iid_mean_masked", 0)
    iid_i = residual_summary.get("iid_mean_identity", 0)
    print(
        f"\n2. RESIDUAL MAGNITUDE: "
        f"IID safe mean ||r_masked||={iid_m:.4f}, "
        f"||r_identity||={iid_i:.4f}"
    )

    # 3. AUROC comparison
    id_auc = auroc_summary.get("identity_mean_auroc")
    act_auc = auroc_summary.get("actual_mean_auroc")
    delta = auroc_summary.get("mean_delta")
    print(
        f"\n3. AUROC: identity_mask={id_auc}, "
        f"learned_mask={act_auc}, "
        f"delta={delta}"
    )

    # 4. Dimension variance
    n_elig = variance_summary.get("n_eligible_clusters", 0)
    frac2 = variance_summary.get("frac_gt_2", 0)
    frac5 = variance_summary.get("frac_gt_5", 0)
    print(
        f"\n4. DIMENSION VARIANCE: "
        f"{variance_summary.get('n_max_ratio_gt_2', 0)} / {n_elig} "
        f"eligible clusters have max dim ratio > 2.0 "
        f"({100*frac2:.0f}%)"
    )
    print(f"   Fraction with max ratio > 5.0: {100*frac5:.0f}%")

    # 5. Routing
    print(
        f"\n5. ROUTING: "
        f"{routing_summary.get('n_clusters_gt50pct_unsafe', 0)} "
        f"clusters have >50% unsafe samples"
    )

    # Verdict
    print("\n" + "-" * 70)
    mask_active = pct > 10
    mask_helps_auroc = delta is not None and delta > 0.005
    variance_signal = frac2 > 0.3

    if mask_active and mask_helps_auroc:
        verdict = "MASK IS ACTIVE AND HELPS AUROC"
    elif mask_active and not mask_helps_auroc:
        verdict = "MASK IS ACTIVE BUT DOES NOT HELP AUROC"
    elif not mask_active and variance_signal:
        verdict = (
            "MASK IS INERT — but per-dim variance signal exists, "
            "whitening may help even with identity mask"
        )
    elif not mask_active:
        verdict = "MASK IS INERT — no pruning, no AUROC improvement"
    else:
        verdict = "INCONCLUSIVE"

    print(f"VERDICT: {verdict}")
    print("=" * 70)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc diagnostics for learned_mask runs"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to completed learned_mask run directory",
    )
    parser.add_argument(
        "--model-key",
        type=str,
        default=None,
        help=f"Model key, one of: {', '.join(MODEL_KEYS)}. "
        "Inferred from the run dir if not specified.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: cuda if available)",
    )
    parser.add_argument(
        "--skip-forward",
        action="store_true",
        help="Skip diagnostics that need forward passes (B, C, D, F)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_key = args.model_key or infer_model_key(run_dir)
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Model key: {model_key}")
    logger.info(f"Device: {device}")

    # Output directory
    out_dir = run_dir / "diagnostics" / "post_hoc"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info("Loading checkpoint...")
    model = load_model(run_dir, device)

    # A. Mask sparsity (no GPU needed)
    logger.info("\n=== A. Mask sparsity ===")
    sparsity_summary = diagnose_mask_sparsity(model, out_dir)
    logger.info(f"Summary: {sparsity_summary}")

    if args.skip_forward:
        logger.info("Skipping forward-pass diagnostics (--skip-forward)")
        print_meta_summary(
            sparsity_summary,
            {"iid_mean_masked": 0, "iid_mean_identity": 0},
            {},
            {"n_eligible_clusters": 0, "frac_gt_2": 0, "frac_gt_5": 0},
            {"n_clusters_gt50pct_unsafe": 0},
        )
        return

    # Load eval data
    logger.info("Loading eval data...")
    set_last_token_mode(True)
    set_complex_data_mode(True)
    eval_data, eval_labels, eval_sources, dataset_names = (
        load_eval_data_with_sources(model_key)
    )

    # Encode once and assign clusters (avoids repeated 22GB allocs)
    logger.info("Encoding eval data and assigning clusters (once)...")
    encoded, clusters = encode_and_assign_clusters(model, eval_data)
    logger.info(
        f"Encoded shape: {encoded.shape}, "
        f"clusters: {len(clusters)} samples"
    )

    # B. Residual magnitudes
    logger.info("\n=== B. Residual magnitudes ===")
    residual_summary = diagnose_residual_magnitudes(
        model,
        eval_data,
        eval_sources,
        dataset_names,
        encoded,
        clusters,
        out_dir,
    )
    logger.info(f"Summary: {residual_summary}")

    # C. Identity-mask AUROC ablation
    logger.info("\n=== C. Identity-mask AUROC ablation ===")
    auroc_summary = diagnose_identity_auroc(
        model,
        eval_data,
        eval_sources,
        dataset_names,
        encoded,
        run_dir,
        out_dir,
    )
    logger.info(f"Summary: {auroc_summary}")

    # D. Per-dimension variance
    logger.info("\n=== D. Per-dimension variance ===")
    variance_summary = diagnose_dimension_variance(
        model,
        eval_data,
        eval_sources,
        dataset_names,
        encoded,
        clusters,
        out_dir,
    )
    logger.info(f"Summary: {variance_summary}")

    # Free encoded array before F (which also computes residuals)
    del encoded
    import gc

    gc.collect()

    # E. Routing
    logger.info("\n=== E. Routing ===")
    routing_summary = diagnose_routing(
        model,
        eval_sources,
        dataset_names,
        clusters,
        out_dir,
    )
    logger.info(f"Summary: {routing_summary}")

    # F. Hard/easy cases — re-encode since we freed it
    logger.info("\n=== F. Hard/easy unsafe cases ===")
    encoded_f, _ = encode_and_assign_clusters(model, eval_data)
    diagnose_hard_easy_cases(
        model,
        eval_data,
        eval_sources,
        dataset_names,
        encoded_f,
        clusters,
        out_dir,
    )
    del encoded_f
    gc.collect()
    logger.info("Saved hard_easy_unsafe_cases.json")

    # G. Cluster size vs AUROC
    logger.info("\n=== G. Cluster size vs per-cluster AUROC ===")
    cluster_size_auroc_summary = diagnose_cluster_size_auroc(
        model, run_dir, out_dir
    )
    if cluster_size_auroc_summary:
        logger.info(f"Summary: {cluster_size_auroc_summary}")

    # Save combined summary
    combined = {
        "model_key": model_key,
        "run_dir": str(run_dir),
        "mask_sparsity": sparsity_summary,
        "residual_magnitudes": residual_summary,
        "identity_auroc": auroc_summary,
        "dimension_variance": variance_summary,
        "routing": routing_summary,
        "cluster_size_auroc": cluster_size_auroc_summary,
    }
    with open(out_dir / "diagnostic_summary.json", "w") as f:
        json.dump(combined, f, indent=2)

    # Print meta-summary
    print_meta_summary(
        sparsity_summary,
        residual_summary,
        auroc_summary,
        variance_summary,
        routing_summary,
    )


if __name__ == "__main__":
    main()
