#!/usr/bin/env python3
"""Standalone evaluation for trained anomaly detection models.

Provides:
- evaluate_anodet(): shared evaluation function (used by both this script
  and run_anodet_analysis.py after training)
- load_baseline_from_checkpoint(): unified loader that auto-detects method
- main(): Hydra entry point for eval-only runs
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from unsupervised_llm_safety.analysis.anomaly_metrics import (
    compute_all_anomaly_metrics,
    compute_all_threshold_based_metrics,
    compute_anomaly_detection_metrics,
    compute_comprehensive_statistics,
    compute_per_slice_metrics,
    compute_three_slice_metrics,
)
from unsupervised_llm_safety.anodet_algos.anodet_utils import (
    BASELINE_CLASSES,
    prepare_data_for_baseline,
    resolve_dataset_configurations,
)
from unsupervised_llm_safety.common.model_io import infer_model_key_from_path
from unsupervised_llm_safety.data.adversarial_utils import (
    discover_and_load_adversarial_datasets,
    load_adversarial_dataset,
)
from unsupervised_llm_safety.data.hidden_states_loader import (
    load_dataset_single,
    split_hidden_states_data,
)

logger = logging.getLogger(__name__)

_BANNER_WIDTH = 80


def _log_eval_banner(step: str, title: str, detail: str = "") -> None:
    """Log a visually distinct evaluation section banner."""
    logger.info("")
    logger.info("-" * _BANNER_WIDTH)
    logger.info(f"[{step}] {title}")
    if detail:
        logger.info(f"  {detail}")
    logger.info("-" * _BANNER_WIDTH)


# --- Checkpoint key → BASELINE_CLASSES key mappings ---

# Pattern A: classmethod loaders via _load_helper
# (checkpoint has "method_name")
METHOD_NAME_TO_KEY = {
    "cvdd": "cvdd",
    "deep_svdd": "deep_svdd",
    "gmm": "gmm",
    "learned_mask": "learned_mask",
    "learned_mask_lora": "learned_mask_lora",
    "mahalanobis": "mahalanobis",
    "one_class_svm": "one_class_svm",
}

# Pattern B: instance loaders (checkpoint has "model_type")
MODEL_TYPE_TO_KEY = {
    "BCE": "bce",
    "CVDDKMeansMasked": "cvdd_kmeans_masked",
}


def load_baseline_from_checkpoint(
    checkpoint_path: str, device: str = "cuda"
) -> Any:
    """Load any anodet baseline from a checkpoint file.

    Auto-detects the method from the checkpoint and uses the appropriate
    loading pattern (classmethod vs instance .load()).

    Args:
        checkpoint_path: Path to the saved checkpoint
        device: Device string (e.g. "cuda", "cpu")

    Returns:
        Loaded and fitted baseline instance
    """
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )

    # Pattern A: has "method_name" key (CVDD, DeepSVDD, GMM, etc.)
    #
    # CVDDKMeansMasked.save() writes BOTH "method_name" (for
    # AnomalyDetectionDefense) and "model_type", but it is an instance
    # loader, so it must be dispatched through Pattern B. Only take
    # Pattern A when the name is one this table actually knows; an
    # unrecognised name falls through to the Pattern B check below and
    # only then becomes an error.
    if checkpoint.get("method_name") in METHOD_NAME_TO_KEY:
        method_name = checkpoint["method_name"]
        method_key = METHOD_NAME_TO_KEY[method_name]
        logger.info(f"Loading {method_key} from checkpoint")
        # Note: OneClassSVM doesn't accept device parameter in __init__,
        # so we pass None for device and it will auto-detect
        if method_key == "one_class_svm":
            return BASELINE_CLASSES[method_key].load(
                checkpoint_path, device=None
            )
        # learned_mask and learned_mask_lora use instance .load()
        if method_key in ("learned_mask", "learned_mask_lora"):
            instance = BASELINE_CLASSES[method_key](device=device)
            instance.load(checkpoint_path)
            return instance
        return BASELINE_CLASSES[method_key].load(
            checkpoint_path, device=device
        )

    # Pattern B: has "model_type" key (BCE, CVDDKMeansMasked)
    if "model_type" in checkpoint:
        model_type = checkpoint["model_type"]
        if model_type not in MODEL_TYPE_TO_KEY:
            raise ValueError(
                f"Unknown model_type in checkpoint: '{model_type}'. "
                f"Known: {list(MODEL_TYPE_TO_KEY.keys())}"
            )
        method_key = MODEL_TYPE_TO_KEY[model_type]
        logger.info(
            f"Loading {method_key} (Pattern B: instance) from checkpoint"
        )

        # Create instance
        instance = BASELINE_CLASSES[method_key](device=device)
        instance.load(checkpoint_path)
        return instance

    if "method_name" in checkpoint:
        raise ValueError(
            f"Unknown method_name in checkpoint: "
            f"'{checkpoint['method_name']}' and no usable 'model_type'. "
            f"Known method_name: {list(METHOD_NAME_TO_KEY.keys())}; "
            f"known model_type: {list(MODEL_TYPE_TO_KEY.keys())}"
        )

    raise ValueError(
        "Checkpoint has neither 'method_name' nor 'model_type' key. "
        f"Keys found: {list(checkpoint.keys())}"
    )


# --- Evaluation helper functions ---


def _get_filter_flags(dataset_cfg: DictConfig) -> Tuple[bool, bool]:
    """Extract filter flags from dataset configuration.

    Returns:
        Tuple of (filter_unsafe_only, filter_safe_only)
    """
    filter_value = getattr(dataset_cfg, "filter", None)
    if filter_value == "unsafe_only":
        return True, False
    elif filter_value == "safe_only":
        return False, True
    return False, False


def process_and_score_dataset(
    dataset_cfg: DictConfig,
    baseline: Any,
    device: torch.device,
    encoder_wrapper: Optional[nn.Module],
    cfg: DictConfig,
    category: str,
    is_adversarial: bool = False,
    split: str = "test",
) -> Tuple[str, np.ndarray]:
    """Load a dataset, compute anomaly scores, return (name, scores)."""
    cpu_device = torch.device("cpu")
    filter_unsafe_only, filter_safe_only = _get_filter_flags(dataset_cfg)

    if is_adversarial:
        max_adv = cfg.get("max_adversarial_samples", 50000)
        hidden_states, labels = load_adversarial_dataset(
            dataset_cfg.path,
            cpu_device,
            filter_unsafe_only=True,
            max_samples=max_adv,
        )
    else:
        hidden_states, labels = load_dataset_single(
            dataset_cfg,
            split=split,
            device=cpu_device,
            filter_unsafe_only=filter_unsafe_only,
            filter_safe_only=filter_safe_only,
        )

    data_np = hidden_states.cpu().numpy()
    data_processed = prepare_data_for_baseline(
        data_np, cfg.method, encoder_wrapper, cfg
    )
    scores = baseline.score_samples(data_processed)
    return dataset_cfg.name, scores


def split_and_process_dataset(
    dataset_cfg: DictConfig,
    baseline: Any,
    device: torch.device,
    encoder_wrapper: Optional[nn.Module],
    cfg: DictConfig,
    category: str,
    holdout_ratio: float,
    is_adversarial: bool = False,
) -> Tuple[str, np.ndarray, np.ndarray]:
    """Split dataset, score holdout/test, return scores."""
    cpu_device = torch.device("cpu")
    filter_unsafe_only, filter_safe_only = _get_filter_flags(dataset_cfg)

    if is_adversarial:
        max_adv = cfg.get("max_adversarial_samples", 50000)
        hidden_states, labels = load_adversarial_dataset(
            dataset_cfg.path,
            cpu_device,
            filter_unsafe_only=True,
            max_samples=max_adv,
        )
        data_dict = {"hidden_states": hidden_states, "labels": labels}
    else:
        # OOD: load both splits (no leakage risk); IID: test only
        split_to_load = "all" if category == "ood" else "test"
        if category == "ood":
            logger.info("Loading OOD dataset from both train and test splits")
        test_hs, test_lb = load_dataset_single(
            dataset_cfg,
            split=split_to_load,
            device=cpu_device,
            filter_unsafe_only=filter_unsafe_only,
            filter_safe_only=filter_safe_only,
        )
        data_dict = {"hidden_states": test_hs, "labels": test_lb}

    holdout_data, test_data = split_hidden_states_data(
        data_dict, split_ratio=holdout_ratio, seed=cfg.get("seed", 42)
    )

    # Score holdout
    holdout_np = holdout_data["hidden_states"].cpu().numpy()
    holdout_processed = prepare_data_for_baseline(
        holdout_np, cfg.method, encoder_wrapper, cfg
    )
    holdout_scores = baseline.score_samples(holdout_processed)

    # Score test
    test_np = test_data["hidden_states"].cpu().numpy()
    test_processed = prepare_data_for_baseline(
        test_np, cfg.method, encoder_wrapper, cfg
    )
    test_scores = baseline.score_samples(test_processed)

    return dataset_cfg.name, holdout_scores, test_scores


# --- Raw score distribution plot ---


def _plot_raw_score_distributions(
    all_raw_scores: Dict[str, np.ndarray],
    dataset_categories: Dict[str, str],
    method_name: str,
    output_dir: Path,
) -> None:
    """Plot reconstruction loss distributions grouped by category.

    Generates a box plot showing raw (unnormalized) score distributions
    for IID, BeaverTails, ToxiGen, and adversarial (pooled) data.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Group scores by category, with BeaverTails and ToxiGen separate
        groups: Dict[str, List[float]] = {}
        for ds_name, scores in all_raw_scores.items():
            cat = dataset_categories.get(ds_name, "unknown")
            if cat in ("iid", "benign_generalization"):
                groups.setdefault("IID (safe)", []).extend(scores.tolist())
            elif "BeaverTails" in ds_name:
                groups.setdefault("BeaverTails", []).extend(scores.tolist())
            elif "ToxiGen" in ds_name:
                groups.setdefault("ToxiGen", []).extend(scores.tolist())
            elif cat == "adversarial":
                groups.setdefault("Adversarial (pooled)", []).extend(
                    scores.tolist()
                )
            else:
                groups.setdefault(f"OOD: {ds_name}", []).extend(
                    scores.tolist()
                )

        if not groups:
            return

        # Order: IID first, then BeaverTails, ToxiGen, Adversarial, rest
        order = [
            "IID (safe)",
            "BeaverTails",
            "ToxiGen",
            "Adversarial (pooled)",
        ]
        labels = [k for k in order if k in groups]
        labels += [k for k in groups if k not in order]
        data = [np.array(groups[k]) for k in labels]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Left: box plot
        ax = axes[0]
        bp = ax.boxplot(
            data,
            labels=labels,
            showfliers=False,
            patch_artist=True,
        )
        colors = {
            "IID (safe)": "#4CAF50",
            "BeaverTails": "#FF9800",
            "ToxiGen": "#F44336",
            "Adversarial (pooled)": "#9C27B0",
        }
        for patch, label in zip(bp["boxes"], labels):
            patch.set_facecolor(colors.get(label, "#90CAF9"))
            patch.set_alpha(0.7)
        ax.set_ylabel("Raw reconstruction loss")
        ax.set_title(f"{method_name}: Raw score by category")
        ax.tick_params(axis="x", rotation=30)

        # Right: overlapping histograms
        ax = axes[1]
        for label, scores_arr in zip(labels, data):
            color = colors.get(label, "#90CAF9")
            ax.hist(
                scores_arr,
                bins=50,
                alpha=0.4,
                label=f"{label} (n={len(scores_arr)})",
                color=color,
                density=True,
            )
        ax.set_xlabel("Raw reconstruction loss")
        ax.set_ylabel("Density")
        ax.set_title(f"{method_name}: Score distributions")
        ax.legend(fontsize=8)

        fig.tight_layout()
        plot_path = output_dir / f"{method_name}_raw_score_distributions.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved raw score distribution plot to {plot_path}")

        # Log summary statistics
        logger.info("\nRaw score statistics by category:")
        for label, scores_arr in zip(labels, data):
            arr = np.array(scores_arr)
            logger.info(
                f"  {label}: n={len(arr)}, "
                f"mean={arr.mean():.4f}, "
                f"median={np.median(arr):.4f}, "
                f"p5={np.percentile(arr, 5):.4f}, "
                f"p95={np.percentile(arr, 95):.4f}"
            )

    except Exception as e:
        logger.warning(f"Failed to plot raw score distributions: {e}")


# --- Main evaluation function ---


def evaluate_anodet(
    baseline: Any,
    cfg: DictConfig,
    encoder_wrapper: Optional[nn.Module] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Evaluate a fitted anomaly detection baseline on all configured datasets.

    This is the shared evaluation function used by both the training pipeline
    (run_anodet_analysis.py) and standalone eval (this module's main()).

    Args:
        baseline: Fitted anomaly detection method instance
        cfg: Configuration with dataset configs, holdout_ratio, etc.
        encoder_wrapper: Optional encoder for data transformation
            (for GMM/Mahalanobis/OC-SVM with use_pretrained_encoder)
        output_dir: Where to save results JSON. Defaults to cwd.

    Returns:
        Dictionary with analysis results
    """
    if output_dir is None:
        output_dir = Path(os.getcwd())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    holdout_ratio = cfg.get("holdout_ratio", 0.2)
    use_threshold_evaluation = cfg.get("use_threshold_evaluation", True)
    use_holdout_split = use_threshold_evaluation and holdout_ratio > 0

    # --- Score all datasets ---
    all_scores: Dict[str, Dict[str, np.ndarray]] = {}
    all_statistics: Dict[str, Dict[str, Dict[str, float]]] = {}
    dataset_categories: Dict[str, str] = {}
    holdout_scores: Dict[str, Dict[str, np.ndarray]] = {}
    holdout_categories: Dict[str, str] = {}
    score_key = f"{baseline.name}_score"
    # Per-dataset cluster assignments and raw scores (for per-cluster analysis)
    all_cluster_assignments: Dict[str, np.ndarray] = {}
    all_raw_scores: Dict[str, np.ndarray] = {}
    has_cluster_info = hasattr(baseline, "_last_eval_clusters")
    has_raw_scores = hasattr(baseline, "_last_raw_scores")

    # IID datasets
    n_iid = len(cfg.datasets.iid)
    _log_eval_banner("EVAL 1/5", "SCORING IID DATASETS", f"{n_iid} datasets")
    iid_total_samples = 0
    for dataset_cfg in tqdm(cfg.datasets.iid, desc="IID datasets", leave=True):
        logger.info(f"  - {dataset_cfg.name}")
        if use_holdout_split:
            name, holdout_s, test_s = split_and_process_dataset(
                dataset_cfg,
                baseline,
                device,
                encoder_wrapper,
                cfg,
                "iid",
                holdout_ratio,
            )
            holdout_scores[name] = {score_key: holdout_s}
            holdout_categories[name] = "iid"
            all_scores[name] = {score_key: test_s}
        else:
            name, scores = process_and_score_dataset(
                dataset_cfg,
                baseline,
                device,
                encoder_wrapper,
                cfg,
                "iid",
            )
            all_scores[name] = {score_key: scores}
        if has_cluster_info and baseline._last_eval_clusters is not None:
            all_cluster_assignments[name] = baseline._last_eval_clusters.copy()
        if has_raw_scores and baseline._last_raw_scores is not None:
            all_raw_scores[name] = baseline._last_raw_scores.copy()
        all_statistics[name] = {
            score_key: compute_comprehensive_statistics(
                all_scores[name][score_key]
            )
        }
        dataset_categories[name] = "iid"
        iid_total_samples += len(all_scores[name][score_key])
    logger.info(
        f"  Scored {n_iid} IID datasets: {iid_total_samples:,} test samples"
    )

    # OOD datasets
    n_ood = len(cfg.datasets.ood)
    _log_eval_banner("EVAL 2/5", "SCORING OOD DATASETS", f"{n_ood} datasets")
    ood_total_samples = 0
    for dataset_cfg in tqdm(cfg.datasets.ood, desc="OOD datasets", leave=True):
        logger.info(f"  - {dataset_cfg.name}")
        name, scores = process_and_score_dataset(
            dataset_cfg,
            baseline,
            device,
            encoder_wrapper,
            cfg,
            "ood",
            split="all",
        )
        all_scores[name] = {score_key: scores}
        if has_cluster_info and baseline._last_eval_clusters is not None:
            all_cluster_assignments[name] = baseline._last_eval_clusters.copy()
        if has_raw_scores and baseline._last_raw_scores is not None:
            all_raw_scores[name] = baseline._last_raw_scores.copy()
        all_statistics[name] = {
            score_key: compute_comprehensive_statistics(
                all_scores[name][score_key]
            )
        }
        dataset_categories[name] = "ood"
        ood_total_samples += len(scores)
    logger.info(
        f"  Scored {n_ood} OOD datasets: {ood_total_samples:,} test samples"
    )

    # Benign generalization datasets
    benign_total_samples = 0
    if hasattr(cfg.datasets, "benign_generalization"):
        n_benign = len(cfg.datasets.benign_generalization)
        _log_eval_banner(
            "EVAL 3/5",
            "SCORING BENIGN GENERALIZATION DATASETS",
            f"{n_benign} datasets",
        )
        for dataset_cfg in tqdm(
            cfg.datasets.benign_generalization,
            desc="Benign gen datasets",
            leave=True,
        ):
            logger.info(f"  - {dataset_cfg.name}")
            name, scores = process_and_score_dataset(
                dataset_cfg,
                baseline,
                device,
                encoder_wrapper,
                cfg,
                "benign_generalization",
            )
            all_scores[name] = {score_key: scores}
            if has_cluster_info and baseline._last_eval_clusters is not None:
                all_cluster_assignments[name] = (
                    baseline._last_eval_clusters.copy()
                )
            if has_raw_scores and baseline._last_raw_scores is not None:
                all_raw_scores[name] = baseline._last_raw_scores.copy()
            all_statistics[name] = {
                score_key: compute_comprehensive_statistics(
                    all_scores[name][score_key]
                )
            }
            dataset_categories[name] = "benign_generalization"
            benign_total_samples += len(scores)
        logger.info(
            f"  Scored {n_benign} benign generalization datasets:"
            f" {benign_total_samples:,} test samples"
        )

    # Adversarial datasets
    adversarial_configs = _resolve_adversarial_configs(cfg)
    n_adv = len(adversarial_configs)
    _log_eval_banner(
        "EVAL 4/5", "SCORING ADVERSARIAL DATASETS", f"{n_adv} datasets"
    )
    adv_total_samples = 0
    for adv_cfg in tqdm(
        adversarial_configs, desc="Adversarial datasets", leave=True
    ):
        logger.info(f"  - {adv_cfg.name}")
        name, scores = process_and_score_dataset(
            adv_cfg,
            baseline,
            device,
            encoder_wrapper,
            cfg,
            "adversarial",
            is_adversarial=True,
        )
        all_scores[name] = {score_key: scores}
        if has_cluster_info and baseline._last_eval_clusters is not None:
            all_cluster_assignments[name] = baseline._last_eval_clusters.copy()
        if has_raw_scores and baseline._last_raw_scores is not None:
            all_raw_scores[name] = baseline._last_raw_scores.copy()
        all_statistics[name] = {
            score_key: compute_comprehensive_statistics(
                all_scores[name][score_key]
            )
        }
        dataset_categories[name] = "adversarial"
        adv_total_samples += len(scores)
    logger.info(
        f"  Scored {n_adv} adversarial datasets:"
        f" {adv_total_samples:,} test samples"
    )

    # --- Compute metrics ---
    _log_eval_banner(
        "EVAL 5/5",
        "COMPUTING METRICS",
        f"IID={iid_total_samples:,} | OOD={ood_total_samples:,}"
        f" | ADV={adv_total_samples:,}"
        f" | Benign={benign_total_samples:,}",
    )
    use_balanced_sampling = cfg.get("balanced_sampling", False)
    balance_seed = cfg.get("seed", 42)

    if use_balanced_sampling:
        logger.info("Using balanced sampling for metric computation")
    else:
        logger.info(
            "Using original imbalanced distribution for metric computation"
        )

    anomaly_metrics = compute_all_anomaly_metrics(
        all_scores,
        dataset_categories,
        score_key=score_key,
        balanced=use_balanced_sampling,
        balance_seed=balance_seed,
    )

    # Threshold-based metrics from holdout set
    threshold_based_metrics = None
    threshold_value = None

    if use_holdout_split and len(holdout_scores) > 0:
        logger.info("\nComputing threshold from holdout set...")
        holdout_metrics = compute_anomaly_detection_metrics(
            holdout_scores,
            holdout_categories,
            score_key=score_key,
            negative_categories=["iid"],
            positive_categories=["ood", "adversarial"],
            balanced=use_balanced_sampling,
            balance_seed=balance_seed,
        )
        threshold_value = holdout_metrics.get("threshold_at_95_tpr")

        if threshold_value is not None and not np.isnan(threshold_value):
            logger.info(f"Threshold at TPR=95%: {threshold_value:.6f}")
            logger.info("Computing threshold-based metrics on test set...")
            threshold_based_metrics = compute_all_threshold_based_metrics(
                all_scores,
                dataset_categories,
                threshold=threshold_value,
                score_key=score_key,
            )
        else:
            logger.warning(
                "Could not compute threshold from holdout set "
                "(TPR=95% not achieved or insufficient data)"
            )

    # --- Per-slice AUROC (equal weight per eval dataset) ---
    per_slice_metrics = compute_per_slice_metrics(
        all_scores,
        dataset_categories,
        score_key=score_key,
        negative_categories=["iid"],
    )
    logger.info(
        f"\nPer-slice mean AUROC: "
        f"{per_slice_metrics['per_slice_mean_auroc']:.4f} "
        f"({per_slice_metrics['n_slices']} slices)"
    )
    for ds_name, ds_metrics in sorted(per_slice_metrics["per_slice"].items()):
        logger.info(
            f"  {ds_name}: AUROC={ds_metrics['auroc']:.4f} "
            f"(n={ds_metrics['n']}, cat={ds_metrics['category']})"
        )

    # --- 3-Slice metrics (BeaverTails, ToxiGen, pooled adversarial) ---
    three_slice_metrics = compute_three_slice_metrics(
        all_scores,
        dataset_categories,
        score_key=score_key,
    )

    # --- Save results ---
    results = {
        "statistics": all_statistics,
        "anomaly_metrics": anomaly_metrics,
        "per_slice_metrics": per_slice_metrics,
        "three_slice_metrics": three_slice_metrics,
        "dataset_categories": dataset_categories,
        "anodet_method": baseline.name,
        "config": OmegaConf.to_container(cfg, resolve=False),
    }

    if threshold_based_metrics is not None:
        results["threshold_based_metrics"] = threshold_based_metrics
        results["threshold_at_95_tpr"] = threshold_value
        results["holdout_ratio"] = holdout_ratio

    results_file = output_dir / f"{baseline.name}_analysis_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved results to {results_file}")

    # Save raw per-dataset scores for downstream ensemble analysis
    scores_save = {}
    for ds_name, ds_scores in all_scores.items():
        scores_save[ds_name] = ds_scores[score_key]
    scores_file = output_dir / f"{baseline.name}_scores.npz"
    np.savez(scores_file, **scores_save)
    # Also save categories alongside scores
    categories_file = output_dir / f"{baseline.name}_categories.json"
    with open(categories_file, "w") as f:
        json.dump(dataset_categories, f, indent=2)
    logger.info(f"Saved raw scores to {scores_file}")

    # Save per-dataset cluster assignments (for per-cluster analysis)
    if all_cluster_assignments:
        clusters_file = output_dir / f"{baseline.name}_cluster_assignments.npz"
        np.savez(clusters_file, **all_cluster_assignments)
        logger.info(
            f"Saved cluster assignments for {len(all_cluster_assignments)} "
            f"datasets to {clusters_file}"
        )

    # Save raw (unnormalized) scores and plot distribution by category
    if all_raw_scores:
        raw_file = output_dir / f"{baseline.name}_raw_scores.npz"
        np.savez(raw_file, **all_raw_scores)
        logger.info(f"Saved raw scores to {raw_file}")

        _plot_raw_score_distributions(
            all_raw_scores, dataset_categories, baseline.name, output_dir
        )

    # --- Print summary ---
    _print_summary(
        baseline.name,
        anomaly_metrics,
        threshold_based_metrics,
        threshold_value,
        per_slice_metrics,
        three_slice_metrics,
    )

    return results


def _resolve_adversarial_configs(cfg: DictConfig) -> List[DictConfig]:
    """Resolve adversarial dataset configs from cfg."""
    adversarial_base_path = cfg.get("adversarial_base_path")
    if not adversarial_base_path:
        logger.info(
            "  No adversarial base path configured. "
            "Skipping adversarial evaluation."
        )
        return []

    if cfg.get("auto_discover_adversarial", False):
        model_key = cfg.get("model_key", None)
        if not model_key:
            model_key = infer_model_key_from_path(cfg.model_path)
        if not model_key:
            logger.error(
                "Cannot auto-discover adversarial files: "
                "model_key not specified and could not be inferred"
            )
            return []
        configs = discover_and_load_adversarial_datasets(
            model_key=model_key,
            base_path=adversarial_base_path,
            attack_filter=cfg.get("attack_filter", None),
            layer=cfg.get("layer", None),
        )
        logger.info(
            f"Auto-discovered {len(configs)} adversarial datasets "
            f"for model '{model_key}'"
        )
        return configs

    return list(cfg.datasets.adversarial)


def _print_summary(
    method_name: str,
    anomaly_metrics: Dict,
    threshold_based_metrics: Optional[Dict],
    threshold_value: Optional[float],
    per_slice_metrics: Optional[Dict] = None,
    three_slice_metrics: Optional[Dict] = None,
) -> None:
    """Print evaluation summary to log."""
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 80)

    # --- 3-Slice summary (primary metric) — printed first ---
    if three_slice_metrics is not None:
        logger.info("")
        logger.info("3-SLICE EVALUATION (primary metric)")
        tsm = three_slice_metrics.get("three_slice_mean_auroc", float("nan"))
        tfpr = three_slice_metrics.get("three_slice_mean_fpr95", float("nan"))
        logger.info(f"  Mean AUROC: {tsm:.4f}    Mean FPR@95: {tfpr:.4f}")
        for name, sm in three_slice_metrics.get("slices", {}).items():
            auroc = sm.get("auroc", float("nan"))
            fpr95 = sm.get("fpr_at_95_tpr", float("nan"))
            n = sm.get("n_positive", 0)
            logger.info(
                f"  {name + ':':<25s} AUROC={auroc:.4f}  "
                f"FPR@95={fpr95:.4f}  (n={n})"
            )

    # --- Pooled metrics (abbreviated reference) ---
    logger.info("")
    logger.info("Pooled metrics (reference):")
    for comparison, metrics in anomaly_metrics.items():
        label = comparison.replace("_", " ").replace("iid vs ", "IID vs ")
        if "error" in metrics:
            logger.info(f"  {label}: error")
            continue
        logger.info(
            f"  {label + ':':<20s} AUROC={metrics['auroc']:.4f}  "
            f"FPR@95={metrics['fpr_at_95_tpr']:.4f}"
        )

    logger.info("\nAnalysis complete!")


# --- Standalone Hydra entry point ---


@hydra.main(
    version_base="1.2",
    config_path="../../../exp_configs",
    config_name="config",
)
def main(cfg: DictConfig):
    """Standalone evaluation entry point.

    Usage:
        python -m unsupervised_llm_safety.anodet_algos.run_anodet_eval \
            anodet_algos=gmm \
            anodet_algos.load_from=/path/to/gmm_trained.pt \
            anodet_algos.model_key=qwen2_1.5b
    """
    # Extract domain
    domain = None
    if hasattr(cfg, "domain") and cfg.domain is not None:
        domain = (
            cfg.domain.name if hasattr(cfg.domain, "name") else str(cfg.domain)
        )

    # Extract anodet config
    if not (hasattr(cfg, "anodet_algos") and cfg.anodet_algos is not None):
        raise ValueError(
            "No 'anodet_algos' config group specified. "
            "Use: anodet_algos=gmm"
        )
    anodet_cfg = cfg.anodet_algos

    # Inject domain
    if domain is not None:
        OmegaConf.set_struct(anodet_cfg, False)
        anodet_cfg.domain = domain
        OmegaConf.set_struct(anodet_cfg, True)
        logger.info(f"Using domain from top-level config: {domain}")

    logging.basicConfig(level=logging.INFO)

    # Require load_from for eval-only mode
    load_from = anodet_cfg.get("load_from")
    if not load_from:
        raise ValueError(
            "anodet_algos.load_from is required for eval-only mode. "
            "Provide the path to a trained checkpoint."
        )

    # Load baseline
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading baseline from: {load_from}")
    baseline = load_baseline_from_checkpoint(load_from, device=str(device))
    logger.info(f"Loaded {baseline.name} baseline")

    # Load pretrained encoder wrapper if needed (for GMM/Mahalanobis/OC-SVM
    # that were trained with use_pretrained_encoder=true)
    encoder_wrapper = None
    model_path = anodet_cfg.get("model_path")
    if model_path and model_path != "null":
        model_path = Path(model_path)
        if model_path.exists():
            logger.info(f"Loading pretrained model from: {model_path}")
            loaded_model = torch.load(
                model_path, map_location=device, weights_only=False
            )
            if hasattr(loaded_model, "encoder"):
                pretrained_model = loaded_model.encoder
            elif isinstance(loaded_model, dict) and "encoder" in loaded_model:
                pretrained_model = loaded_model["encoder"]
            else:
                pretrained_model = loaded_model

            # Create encoder wrapper
            from unsupervised_llm_safety.anodet_algos.anodet_utils import (
                create_pretrained_encoder_wrapper,
            )

            encoder_wrapper = create_pretrained_encoder_wrapper(
                pretrained_model, baseline.name
            )
        else:
            raise ValueError(f"Model path does not exist: {model_path}")

    # Resolve datasets and run evaluation
    resolve_dataset_configurations(anodet_cfg)
    output_dir = Path(os.getcwd())
    results = evaluate_anodet(
        baseline, anodet_cfg, encoder_wrapper, output_dir
    )

    # Return 3-slice mean AUROC for Hydra optimization
    three_slice = results.get("three_slice_metrics", {})
    return three_slice.get("three_slice_mean_auroc", 0.0)


if __name__ == "__main__":
    main()
