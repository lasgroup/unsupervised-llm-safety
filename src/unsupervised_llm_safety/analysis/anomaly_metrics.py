"""Anomaly detection metrics for evaluating OOD and adversarial detection performance.

This module provides functions for computing AUROC, AUPRC, and FPR@95%TPR metrics
for anomaly detection tasks, with support for balanced sampling and comprehensive statistics.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

logger = logging.getLogger(__name__)


def compute_comprehensive_statistics(scores: np.ndarray) -> Dict[str, Any]:
    """Compute comprehensive statistics for a score array.

    Args:
        scores: Array of scores to compute statistics for

    Returns:
        Dictionary containing:
        - mean, std, median
        - min, max
        - q1, q3 (quartiles)
        - p5, p95 (5th and 95th percentiles)
    """
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "median": float(np.median(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "q1": float(np.percentile(scores, 25)),
        "q3": float(np.percentile(scores, 75)),
        "p5": float(np.percentile(scores, 5)),
        "p95": float(np.percentile(scores, 95)),
    }


def compute_fpr_at_tpr(
    fpr: np.ndarray, tpr: np.ndarray, target_tpr: float = 0.95
) -> float:
    """Find FPR at a specific TPR threshold using linear interpolation.

    Args:
        fpr: False positive rates from ROC curve
        tpr: True positive rates from ROC curve
        target_tpr: Target TPR threshold (default: 0.95)

    Returns:
        FPR at the target TPR, or 1.0 if target not achieved
    """
    # Find indices where TPR crosses the target
    idx = np.where(tpr >= target_tpr)[0]

    if len(idx) == 0:
        # Target TPR not achieved
        return 1.0

    if tpr[idx[0]] == target_tpr:
        return float(fpr[idx[0]])

    # Linear interpolation
    if idx[0] > 0:
        # Interpolate between idx[0]-1 and idx[0]
        x0, x1 = tpr[idx[0] - 1], tpr[idx[0]]
        y0, y1 = fpr[idx[0] - 1], fpr[idx[0]]
        fpr_interpolated = y0 + (target_tpr - x0) * (y1 - y0) / (x1 - x0)
        return float(fpr_interpolated)
    else:
        return float(fpr[idx[0]])


def compute_tpr_at_fpr(
    fpr: np.ndarray, tpr: np.ndarray, target_fpr: float = 0.05
) -> float:
    """Find TPR at a specific FPR threshold using linear interpolation.

    Dual of compute_fpr_at_tpr: fixes the false-positive (false-rejection
    on safe content) budget and reports the achievable recall of unsafe
    content at that operating point.

    Args:
        fpr: False positive rates from ROC curve (monotonically non-decreasing)
        tpr: True positive rates from ROC curve
        target_fpr: Target FPR threshold (default: 0.05)

    Returns:
        TPR at the target FPR. If the minimum achievable FPR already
        exceeds target_fpr (i.e., no operating point falls within the
        budget), returns 0.0.
    """
    # Find indices where FPR is at or below the target
    idx = np.where(fpr <= target_fpr)[0]

    if len(idx) == 0:
        # Even the strictest threshold blocks more than target_fpr of safe data
        return 0.0

    # Take the largest index satisfying fpr <= target_fpr — this gives the
    # rightmost (highest-TPR) operating point within budget.
    i = int(idx[-1])

    if fpr[i] == target_fpr or i == len(fpr) - 1:
        return float(tpr[i])

    # Linear interpolation between i and i+1 (fpr[i] < target_fpr < fpr[i+1])
    x0, x1 = fpr[i], fpr[i + 1]
    y0, y1 = tpr[i], tpr[i + 1]
    if x1 == x0:
        return float(y0)
    tpr_interpolated = y0 + (target_fpr - x0) * (y1 - y0) / (x1 - x0)
    return float(tpr_interpolated)


def compute_threshold_at_tpr(
    tpr: np.ndarray, thresholds: np.ndarray, target_tpr: float = 0.95
) -> float:
    """Find the score threshold at a specific TPR using linear interpolation.

    Args:
        tpr: True positive rates from ROC curve
        thresholds: Score thresholds from ROC curve
        target_tpr: Target TPR threshold (default: 0.95)

    Returns:
        Score threshold at the target TPR, or np.nan if target not achieved
    """
    # Find indices where TPR crosses the target
    idx = np.where(tpr >= target_tpr)[0]

    if len(idx) == 0:
        # Target TPR not achieved
        return np.nan

    if tpr[idx[0]] == target_tpr:
        return float(thresholds[idx[0]])

    # Linear interpolation
    if idx[0] > 0:
        # Interpolate between idx[0]-1 and idx[0]
        x0, x1 = tpr[idx[0] - 1], tpr[idx[0]]
        y0, y1 = thresholds[idx[0] - 1], thresholds[idx[0]]
        threshold_interpolated = y0 + (target_tpr - x0) * (y1 - y0) / (x1 - x0)
        return float(threshold_interpolated)
    else:
        return float(thresholds[idx[0]])


def compute_tpr_fnr_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Compute TPR, FNR, FPR, and other metrics at a given threshold.

    Args:
        y_true: True binary labels (0 = normal, 1 = anomaly)
        y_score: Anomaly scores (higher = more anomalous)
        threshold: Score threshold for classification

    Returns:
        Dictionary containing:
        - tpr: True Positive Rate (Recall) = TP / (TP + FN)
        - fnr: False Negative Rate = FN / (TP + FN) = 1 - TPR
        - fpr: False Positive Rate = FP / (FP + TN)
        - precision: TP / (TP + FP)
        - accuracy: (TP + TN) / (TP + TN + FP + FN)
        - tp, tn, fp, fn: Counts of each classification outcome
    """
    # Predictions: score >= threshold -> anomaly (1), else normal (0)
    y_pred = (y_score >= threshold).astype(int)

    # Compute confusion matrix components
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))  # True positives
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))  # True negatives
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))  # False positives
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))  # False negatives

    # Compute rates
    tpr = (
        tp / (tp + fn) if (tp + fn) > 0 else 0.0
    )  # True Positive Rate (Recall)
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0  # False Negative Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = (
        (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    )

    return {
        "tpr": float(tpr),
        "fnr": float(fnr),
        "fpr": float(fpr),
        "precision": float(precision),
        "accuracy": float(accuracy),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def compute_threshold_based_metrics(
    all_scores: Dict[str, Dict[str, np.ndarray]],
    dataset_categories: Dict[str, str],
    threshold: float,
    score_key: str = "none",
    negative_categories: List[str] = ["iid"],
    positive_categories: List[str] = ["ood", "adversarial"],
) -> Dict[str, Any]:
    """Compute threshold-based metrics (TPR, FNR, FPR) for anomaly detection.

    Args:
        all_scores: Dict mapping dataset names to score arrays
        dataset_categories: Dict mapping dataset names to categories
        threshold: Score threshold to use for classification
        score_key: Which score to use (default: 'none')
        negative_categories: List of categories to treat as negative class (normal)
        positive_categories: List of categories to treat as positive class (anomaly)

    Returns:
        Dict with threshold-based metrics and sample counts
    """
    # Aggregate scores and labels
    y_true = []
    y_score = []

    for dataset_name, scores in all_scores.items():
        category = dataset_categories.get(dataset_name)

        # Skip if category not in either list
        if category not in negative_categories + positive_categories:
            continue

        score_values = scores[score_key]
        n_samples = len(score_values)

        # Create binary labels based on category lists
        if category in negative_categories:
            labels = np.zeros(n_samples)  # Normal = 0
        else:  # category in positive_categories
            labels = np.ones(n_samples)  # Anomaly = 1

        y_true.extend(labels)
        y_score.extend(score_values)

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # Handle edge cases
    if len(y_true) == 0:
        return {
            "tpr": np.nan,
            "fnr": np.nan,
            "fpr": np.nan,
            "precision": np.nan,
            "accuracy": np.nan,
            "n_negative": 0,
            "n_positive": 0,
            "error": "No data found for specified categories",
        }

    if len(np.unique(y_true)) < 2:
        return {
            "tpr": np.nan,
            "fnr": np.nan,
            "fpr": np.nan,
            "precision": np.nan,
            "accuracy": np.nan,
            "n_negative": int(np.sum(y_true == 0)),
            "n_positive": int(np.sum(y_true == 1)),
            "error": "Insufficient class diversity",
        }

    if np.isnan(threshold):
        return {
            "tpr": np.nan,
            "fnr": np.nan,
            "fpr": np.nan,
            "precision": np.nan,
            "accuracy": np.nan,
            "n_negative": int(np.sum(y_true == 0)),
            "n_positive": int(np.sum(y_true == 1)),
            "error": "Invalid threshold (NaN)",
        }

    # Compute metrics at threshold
    metrics: Dict[str, Any] = compute_tpr_fnr_at_threshold(
        y_true, y_score, threshold
    )

    # Add sample counts
    metrics["n_negative"] = int(np.sum(y_true == 0))
    metrics["n_positive"] = int(np.sum(y_true == 1))
    metrics["threshold"] = float(threshold)
    metrics["negative_categories"] = negative_categories
    metrics["positive_categories"] = positive_categories

    return metrics


def _balance_dataset_for_anomaly_detection(
    y_true: np.ndarray,
    y_score: np.ndarray,
    sample_categories: np.ndarray,
    category_indices: Dict[str, List[int]],
    active_positive_categories: List[str],
    active_negative_categories: List[str],
    n_negative_original: int,
    n_positive_original: int,
    balance_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Balance dataset for anomaly detection with stratified sampling.

    Args:
        y_true: True labels array
        y_score: Score values array
        sample_categories: Category for each sample
        category_indices: Dict mapping categories to sample indices
        active_positive_categories: List of positive categories with samples
        active_negative_categories: List of negative categories with samples
        n_negative_original: Original number of negative samples
        n_positive_original: Original number of positive samples
        balance_seed: Random seed for reproducibility

    Returns:
        Tuple of (balanced_y_true, balanced_y_score, balanced_categories)
    """
    # Set random seed for reproducibility
    rng = np.random.RandomState(balance_seed)

    # Determine target size for balanced dataset
    min_class_size = min(n_negative_original, n_positive_original)

    # For multiple positive categories, ensure equal representation
    if len(active_positive_categories) > 1:
        # Calculate samples per positive category
        samples_per_pos_category = min_class_size // len(
            active_positive_categories
        )

        # Sample from each positive category
        sampled_pos_indices = []
        remaining_quota = min_class_size

        for i, cat in enumerate(active_positive_categories):
            cat_indices = np.array(category_indices[cat])
            n_available = len(cat_indices)

            # For last category, use all remaining quota
            if i == len(active_positive_categories) - 1:
                n_to_sample = min(remaining_quota, n_available)
            else:
                n_to_sample = min(samples_per_pos_category, n_available)

            if n_to_sample > 0:
                sampled = rng.choice(cat_indices, n_to_sample, replace=False)
                sampled_pos_indices.extend(sampled)
                remaining_quota -= n_to_sample

        sampled_pos = np.array(sampled_pos_indices)
    else:
        # Single positive category - use original logic
        pos_indices = np.where(y_true == 1)[0]
        if n_positive_original > min_class_size:
            sampled_pos = rng.choice(
                pos_indices, min_class_size, replace=False
            )
        else:
            sampled_pos = pos_indices

    # Sample from negative categories to match positive count
    n_pos_sampled = len(sampled_pos)
    neg_indices = np.where(y_true == 0)[0]

    if n_negative_original > n_pos_sampled:
        sampled_neg = rng.choice(neg_indices, n_pos_sampled, replace=False)
    else:
        sampled_neg = neg_indices

    # Combine and sort indices to maintain some ordering
    balanced_indices = np.concatenate([sampled_neg, sampled_pos])
    balanced_indices = np.sort(balanced_indices)

    # Apply balancing
    y_true_balanced = y_true[balanced_indices]
    y_score_balanced = y_score[balanced_indices]
    sample_categories_balanced = sample_categories[balanced_indices]

    # Log balancing info
    logger.info(
        f"Balanced sampling: {n_negative_original} negative, "
        f"{n_positive_original} positive -> {len(sampled_neg)} negative, {len(sampled_pos)} positive"
    )

    if len(active_positive_categories) > 1:
        # Log category distribution
        pos_cat_dist = {}
        for cat in active_positive_categories:
            pos_cat_dist[cat] = np.sum(sample_categories_balanced == cat)
        logger.info(
            f"Positive category distribution after balancing: {pos_cat_dist}"
        )

    return y_true_balanced, y_score_balanced, sample_categories_balanced


def compute_anomaly_detection_metrics(
    all_scores: Dict[str, Dict[str, np.ndarray]],
    dataset_categories: Dict[str, str],
    score_key: str = "ccs_score",
    negative_categories: List[str] = ["iid"],
    positive_categories: List[str] = ["ood", "adversarial"],
    balanced: bool = False,
    balance_seed: int = 42,
) -> Dict[str, Any]:
    """Compute AUROC, AUPRC, and FPR@95%TPR for anomaly detection.

    Args:
        all_scores: Dict mapping dataset names to score arrays
        dataset_categories: Dict mapping dataset names to categories
        score_key: Which score to use (default: 'ccs_score')
        negative_categories: List of categories to treat as negative class (normal)
        positive_categories: List of categories to treat as positive class (anomaly)
        balanced: If True, use balanced sampling (undersample to match smaller class)
        balance_seed: Random seed for balanced sampling reproducibility

    Returns:
        Dict with metrics and sample counts
    """
    # Step 1: Aggregate scores and labels with category tracking
    y_true = []
    y_score = []
    sample_categories = []  # Track category for each sample

    # Also track indices by category for stratified sampling
    category_indices: Dict[str, List[int]] = {
        cat: [] for cat in negative_categories + positive_categories
    }
    current_idx = 0

    for dataset_name, scores in all_scores.items():
        category = dataset_categories.get(dataset_name)

        # Skip if category not in either list
        if category not in negative_categories + positive_categories:
            continue

        score_values = scores[score_key]
        n_samples = len(score_values)

        # Create binary labels based on category lists
        if category in negative_categories:
            labels = np.zeros(n_samples)  # Normal = 0
        else:  # category in positive_categories
            labels = np.ones(n_samples)  # Anomaly = 1

        y_true.extend(labels)
        y_score.extend(score_values)
        sample_categories.extend([category] * n_samples)

        # Track indices for this category
        category_indices[category].extend(
            range(current_idx, current_idx + n_samples)
        )
        current_idx += n_samples

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    sample_categories = np.array(sample_categories)

    # Store original counts for reporting
    n_negative_original = int(np.sum(y_true == 0))
    n_positive_original = int(np.sum(y_true == 1))

    # Handle edge cases
    if len(y_true) == 0:
        return {
            "auroc": np.nan,
            "auprc": np.nan,
            "fpr_at_95_tpr": np.nan,
            "threshold_at_95_tpr": np.nan,
            "n_negative": 0,
            "n_positive": 0,
            "n_negative_original": 0,
            "n_positive_original": 0,
            "error": "No data found for specified categories",
        }

    if len(np.unique(y_true)) < 2:
        return {
            "auroc": np.nan,
            "auprc": np.nan,
            "fpr_at_95_tpr": np.nan,
            "threshold_at_95_tpr": np.nan,
            "n_negative": n_negative_original,
            "n_positive": n_positive_original,
            "n_negative_original": n_negative_original,
            "n_positive_original": n_positive_original,
            "error": "Insufficient class diversity",
        }

    # Track category counts for reporting
    category_counts = {}
    for cat in negative_categories + positive_categories:
        category_counts[cat] = len(category_indices[cat])

    # Filter out empty categories
    active_positive_categories = [
        cat for cat in positive_categories if category_counts[cat] > 0
    ]
    active_negative_categories = [
        cat for cat in negative_categories if category_counts[cat] > 0
    ]

    # Step 2: Apply balanced sampling if requested
    if balanced and n_negative_original > 0 and n_positive_original > 0:
        y_true_balanced, y_score_balanced, sample_categories_balanced = (
            _balance_dataset_for_anomaly_detection(
                y_true=y_true,
                y_score=y_score,
                sample_categories=sample_categories,
                category_indices=category_indices,
                active_positive_categories=active_positive_categories,
                active_negative_categories=active_negative_categories,
                n_negative_original=n_negative_original,
                n_positive_original=n_positive_original,
                balance_seed=balance_seed,
            )
        )
    else:
        # Use original data if not balancing
        y_true_balanced = y_true
        y_score_balanced = y_score
        sample_categories_balanced = sample_categories

    # Step 3: Calculate metrics
    try:
        # Calculate AUROC
        auroc = roc_auc_score(y_true_balanced, y_score_balanced)

        # Calculate AUPRC
        auprc = average_precision_score(y_true_balanced, y_score_balanced)

        # Calculate FPR@95%TPR and threshold@95%TPR
        fpr, tpr, thresholds = roc_curve(y_true_balanced, y_score_balanced)
        fpr_at_95_tpr = compute_fpr_at_tpr(fpr, tpr, target_tpr=0.95)
        threshold_at_95_tpr = compute_threshold_at_tpr(
            tpr, thresholds, target_tpr=0.95
        )

        result = {
            "auroc": float(auroc),
            "auprc": float(auprc),
            "fpr_at_95_tpr": float(fpr_at_95_tpr),
            "threshold_at_95_tpr": (
                float(threshold_at_95_tpr)
                if not np.isnan(threshold_at_95_tpr)
                else np.nan
            ),
            "n_negative": (
                int(np.sum(y_true_balanced == 0))
                if balanced
                else n_negative_original
            ),
            "n_positive": (
                int(np.sum(y_true_balanced == 1))
                if balanced
                else n_positive_original
            ),
            "n_negative_original": n_negative_original,
            "n_positive_original": n_positive_original,
            "negative_categories": negative_categories,
            "positive_categories": positive_categories,
            "category_counts_original": category_counts,
        }

        if balanced:
            result["balanced"] = True
            # Add category distribution after balancing
            category_counts_balanced = {}
            for cat in active_negative_categories + active_positive_categories:
                category_counts_balanced[cat] = int(
                    np.sum(sample_categories_balanced == cat)
                )
            result["category_counts_balanced"] = category_counts_balanced

            if len(active_positive_categories) > 1:
                result["balancing_method"] = "stratified"
            else:
                result["balancing_method"] = "simple"

        return result
    except Exception as e:
        logger.error(f"Error computing anomaly detection metrics: {e}")
        return {
            "auroc": np.nan,
            "auprc": np.nan,
            "fpr_at_95_tpr": np.nan,
            "threshold_at_95_tpr": np.nan,
            "n_negative": int(np.sum(y_true == 0)),
            "n_positive": int(np.sum(y_true == 1)),
            "error": str(e),
        }


def compute_all_anomaly_metrics(
    all_scores: Dict[str, Dict[str, np.ndarray]],
    dataset_categories: Dict[str, str],
    score_key: str = "none",
    balanced: bool = False,
    balance_seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Compute anomaly detection metrics for different comparisons.

    1. IID vs OOD (out-of-distribution robustness)
    2. IID vs Adversarial (adversarial robustness)
    3. IID vs OOD+Adversarial (overall anomaly detection)
    4. IID vs Benign Generalization (should have low separation)
    5. Benign Generalization vs OOD (should have high separation)
    6. Benign Generalization vs OOD+Adversarial (false alarms on benign data)

    Args:
        all_scores: Dict mapping dataset names to score arrays
        dataset_categories: Dict mapping dataset names to categories
        score_key: Which score to use (default: 'none')
        balanced: If True, use balanced sampling (undersample to match smaller class).
            If False, use original imbalanced distribution (default: True).
        balance_seed: Random seed for balanced sampling reproducibility

    Returns:
        Dict with metrics for each comparison
    """
    results = {}

    # 1. IID vs OOD
    results["iid_vs_ood"] = compute_anomaly_detection_metrics(
        all_scores,
        dataset_categories,
        score_key=score_key,
        negative_categories=["iid"],
        positive_categories=["ood"],
        balanced=balanced,
        balance_seed=balance_seed,
    )

    # 2. IID vs Adversarial
    results["iid_vs_adversarial"] = compute_anomaly_detection_metrics(
        all_scores,
        dataset_categories,
        score_key=score_key,
        negative_categories=["iid"],
        positive_categories=["adversarial"],
        balanced=balanced,
        balance_seed=balance_seed,
    )

    # 3. IID vs OOD+Adversarial (renamed from iid_vs_all)
    results["iid_vs_ood+adv"] = compute_anomaly_detection_metrics(
        all_scores,
        dataset_categories,
        score_key=score_key,
        negative_categories=["iid"],
        positive_categories=["ood", "adversarial"],
        balanced=balanced,
        balance_seed=balance_seed,
    )

    # 4. IID vs Benign Generalization (should have low separation)
    results["iid_vs_benign_gen"] = compute_anomaly_detection_metrics(
        all_scores,
        dataset_categories,
        score_key=score_key,
        negative_categories=["iid"],
        positive_categories=["benign_generalization"],
        balanced=balanced,
        balance_seed=balance_seed,
    )

    # 5. Benign Generalization vs OOD (renamed for consistency)
    results["benign_gen_vs_ood"] = compute_anomaly_detection_metrics(
        all_scores,
        dataset_categories,
        score_key=score_key,
        negative_categories=["benign_generalization"],
        positive_categories=["ood"],
        balanced=balanced,
        balance_seed=balance_seed,
    )

    # 6. Benign Generalization vs OOD+Adversarial (for false alarm analysis)
    results["benign_gen_vs_ood+adv"] = compute_anomaly_detection_metrics(
        all_scores,
        dataset_categories,
        score_key=score_key,
        negative_categories=["benign_generalization"],
        positive_categories=["ood", "adversarial"],
        balanced=balanced,
        balance_seed=balance_seed,
    )

    return results


def compute_all_threshold_based_metrics(
    all_scores: Dict[str, Dict[str, np.ndarray]],
    dataset_categories: Dict[str, str],
    threshold: float,
    score_key: str = "none",
) -> Dict[str, Dict[str, float]]:
    """Compute threshold-based metrics for different comparisons.

    Same comparisons as compute_all_anomaly_metrics but using a fixed threshold.

    Args:
        all_scores: Dict mapping dataset names to score arrays
        dataset_categories: Dict mapping dataset names to categories
        threshold: Score threshold to use for classification
        score_key: Which score to use (default: 'none')

    Returns:
        Dict with threshold-based metrics for each comparison
    """
    results = {}

    # 1. IID vs OOD
    results["iid_vs_ood"] = compute_threshold_based_metrics(
        all_scores,
        dataset_categories,
        threshold=threshold,
        score_key=score_key,
        negative_categories=["iid"],
        positive_categories=["ood"],
    )

    # 2. IID vs Adversarial
    results["iid_vs_adversarial"] = compute_threshold_based_metrics(
        all_scores,
        dataset_categories,
        threshold=threshold,
        score_key=score_key,
        negative_categories=["iid"],
        positive_categories=["adversarial"],
    )

    # 3. IID vs OOD+Adversarial
    results["iid_vs_ood+adv"] = compute_threshold_based_metrics(
        all_scores,
        dataset_categories,
        threshold=threshold,
        score_key=score_key,
        negative_categories=["iid"],
        positive_categories=["ood", "adversarial"],
    )

    # 4. IID vs Benign Generalization
    results["iid_vs_benign_gen"] = compute_threshold_based_metrics(
        all_scores,
        dataset_categories,
        threshold=threshold,
        score_key=score_key,
        negative_categories=["iid"],
        positive_categories=["benign_generalization"],
    )

    # 5. Benign Generalization vs OOD
    results["benign_gen_vs_ood"] = compute_threshold_based_metrics(
        all_scores,
        dataset_categories,
        threshold=threshold,
        score_key=score_key,
        negative_categories=["benign_generalization"],
        positive_categories=["ood"],
    )

    # 6. Benign Generalization vs OOD+Adversarial
    results["benign_gen_vs_ood+adv"] = compute_threshold_based_metrics(
        all_scores,
        dataset_categories,
        threshold=threshold,
        score_key=score_key,
        negative_categories=["benign_generalization"],
        positive_categories=["ood", "adversarial"],
    )

    return results


def compute_three_slice_metrics(
    all_scores: Dict[str, Dict[str, np.ndarray]],
    dataset_categories: Dict[str, str],
    score_key: str = "none",
) -> Dict[str, Any]:
    """Compute 3-slice metrics: BeaverTails, ToxiGen, pooled adversarial.

    Each slice is IID (safe) vs one group:
      1. BeaverTails (OOD unsafe content)
      2. ToxiGen (OOD toxic language)
      3. All adversarial attacks pooled

    Returns AUROC and FPR@95%TPR for each slice, plus the 3-slice mean.
    """
    # Pool IID scores
    iid_list = []
    for ds_name, ds_scores in all_scores.items():
        cat = dataset_categories.get(ds_name)
        if cat == "iid":
            iid_list.append(ds_scores[score_key])
    if not iid_list:
        logger.warning("No IID samples for 3-slice metrics")
        return {
            "three_slice_mean_auroc": float("nan"),
            "slices": {},
        }

    iid_scores = np.concatenate(iid_list)

    # Pool adversarial scores
    adv_list = []
    for ds_name, ds_scores in all_scores.items():
        if dataset_categories.get(ds_name) == "adversarial":
            adv_list.append(ds_scores[score_key])
    adv_scores = np.concatenate(adv_list) if adv_list else np.array([])

    # Define the 3 slices: (name, positive scores)
    slice_defs: List[Tuple[str, np.ndarray]] = []

    # BeaverTails (OOD only, skip safe variants)
    for ds_name, ds_scores in all_scores.items():
        cat = dataset_categories.get(ds_name)
        if "BeaverTails" in ds_name and cat not in (
            "iid",
            "benign_generalization",
        ):
            slice_defs.append(("BeaverTails", ds_scores[score_key]))
            break

    # ToxiGen (OOD only, skip safe variants)
    for ds_name, ds_scores in all_scores.items():
        cat = dataset_categories.get(ds_name)
        if "ToxiGen" in ds_name and cat not in (
            "iid",
            "benign_generalization",
        ):
            slice_defs.append(("ToxiGen", ds_scores[score_key]))
            break

    # Pooled adversarial
    if len(adv_scores) > 0:
        slice_defs.append(("Adversarial (pooled)", adv_scores))

    slices: Dict[str, Dict[str, float]] = {}
    for name, pos_scores in slice_defs:
        if len(pos_scores) < 2:
            slices[name] = {
                "auroc": float("nan"),
                "fpr_at_95_tpr": float("nan"),
                "n_positive": int(len(pos_scores)),
            }
            continue
        labels = np.concatenate(
            [np.zeros(len(iid_scores)), np.ones(len(pos_scores))]
        )
        combined = np.concatenate([iid_scores, pos_scores])
        try:
            auroc = float(roc_auc_score(labels, combined))
        except ValueError:
            auroc = float("nan")
        try:
            fpr_arr, tpr_arr, _ = roc_curve(labels, combined)
            fpr95 = compute_fpr_at_tpr(fpr_arr, tpr_arr, 0.95)
            tpr_at_5_fpr = compute_tpr_at_fpr(fpr_arr, tpr_arr, 0.05)
            tpr_at_10_fpr = compute_tpr_at_fpr(fpr_arr, tpr_arr, 0.10)
        except ValueError:
            fpr95 = float("nan")
            tpr_at_5_fpr = float("nan")
            tpr_at_10_fpr = float("nan")
        slices[name] = {
            "auroc": auroc,
            "fpr_at_95_tpr": fpr95,
            "tpr_at_5_fpr": tpr_at_5_fpr,
            "tpr_at_10_fpr": tpr_at_10_fpr,
            "n_positive": int(len(pos_scores)),
        }

    aurocs = [s["auroc"] for s in slices.values() if not np.isnan(s["auroc"])]
    mean_auroc = float(np.mean(aurocs)) if aurocs else float("nan")

    fpr95s = [
        s["fpr_at_95_tpr"]
        for s in slices.values()
        if not np.isnan(s["fpr_at_95_tpr"])
    ]
    mean_fpr95 = float(np.mean(fpr95s)) if fpr95s else float("nan")

    tpr5s = [
        s["tpr_at_5_fpr"]
        for s in slices.values()
        if not np.isnan(s["tpr_at_5_fpr"])
    ]
    mean_tpr5 = float(np.mean(tpr5s)) if tpr5s else float("nan")

    tpr10s = [
        s["tpr_at_10_fpr"]
        for s in slices.values()
        if not np.isnan(s["tpr_at_10_fpr"])
    ]
    mean_tpr10 = float(np.mean(tpr10s)) if tpr10s else float("nan")

    return {
        "three_slice_mean_auroc": mean_auroc,
        "three_slice_mean_fpr95": mean_fpr95,
        "three_slice_mean_tpr_at_5_fpr": mean_tpr5,
        "three_slice_mean_tpr_at_10_fpr": mean_tpr10,
        "n_iid": int(len(iid_scores)),
        "slices": slices,
    }


def compute_per_slice_metrics(
    all_scores: Dict[str, Dict[str, np.ndarray]],
    dataset_categories: Dict[str, str],
    score_key: str = "none",
    negative_categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute AUROC for each dataset individually vs pooled negatives.

    For each non-negative dataset, computes AUROC treating that single
    dataset as the positive class and all negative-category datasets
    pooled as the negative class. This gives equal weight to each
    evaluation slice regardless of sample count.

    Args:
        all_scores: Dict mapping dataset names to {score_key: scores}
        dataset_categories: Dict mapping dataset names to categories
        score_key: Which score key to use
        negative_categories: Categories to pool as negatives
            (default: ["iid"])

    Returns:
        Dict with per-slice results:
        {
            "per_slice": {dataset_name: {"auroc": float, ...}, ...},
            "per_slice_mean_auroc": float,
            "negative_categories": [...],
            "n_negative": int,
            "n_slices": int,
        }
    """
    if negative_categories is None:
        negative_categories = ["iid"]

    # Pool all negative scores
    neg_scores = []
    for ds_name, ds_scores in all_scores.items():
        cat = dataset_categories.get(ds_name)
        if cat in negative_categories:
            neg_scores.append(ds_scores[score_key])

    if not neg_scores:
        logger.warning("No negative samples found for per-slice metrics")
        return {
            "per_slice": {},
            "per_slice_mean_auroc": float("nan"),
            "negative_categories": negative_categories,
            "n_negative": 0,
            "n_slices": 0,
        }

    neg_all = np.concatenate(neg_scores)
    n_neg = len(neg_all)

    # Categories to exclude from per-slice positive computation.
    # benign_generalization is IID model output on benign prompts,
    # not unsafe content — including it artificially deflates the mean.
    exclude_cats = set(negative_categories) | {"benign_generalization"}

    per_slice: Dict[str, Dict[str, Any]] = {}
    for ds_name, ds_scores in all_scores.items():
        cat = dataset_categories.get(ds_name)
        if cat in exclude_cats:
            continue
        pos = ds_scores[score_key]
        if len(pos) < 2:
            continue

        labels = np.concatenate([np.zeros(n_neg), np.ones(len(pos))])
        combined = np.concatenate([neg_all, pos])
        try:
            auroc = float(roc_auc_score(labels, combined))
        except ValueError:
            auroc = float("nan")

        per_slice[ds_name] = {
            "auroc": auroc,
            "n": int(len(pos)),
            "category": cat,
        }

    auroc_vals = [
        v["auroc"] for v in per_slice.values() if not np.isnan(v["auroc"])
    ]
    mean_auroc = float(np.mean(auroc_vals)) if auroc_vals else float("nan")

    return {
        "per_slice": per_slice,
        "per_slice_mean_auroc": mean_auroc,
        "negative_categories": negative_categories,
        "n_negative": n_neg,
        "n_slices": len(per_slice),
    }
