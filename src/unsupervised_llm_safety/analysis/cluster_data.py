#!/usr/bin/env python3
"""Shared cluster-analysis data loaders.

Stable home for the registry-backed data loaders used across the
production pipeline (`run_anodet_analysis`, `per_slice_leaderboard`) and
the FreqMask / LoRA appendix analyses (`learned_mask_binary`,
`lora_resmahal_rescore`). These were originally defined inside the
exploratory `cramped_cluster_hypothesis` / `cluster_error_modes` modules
(each bundled with a one-off local CLI); they are extracted here
so production code no longer depends on those scripts.

Functions
---------
load_iid_training_data
    Reload IID safe training data from the dataset registry.
load_eval_data_with_sources
    Load IID-test + OOD + adversarial eval data with per-sample source ids.
"""

import logging
from typing import List, Tuple

import numpy as np
import torch

from unsupervised_llm_safety.data.adversarial_utils import (
    discover_and_load_adversarial_datasets,
    load_adversarial_dataset,
)
from unsupervised_llm_safety.data.dataset_registry import (
    get_adversarial_base_path,
    get_benign_ood_datasets,
    get_dataset_paths,
)
from unsupervised_llm_safety.data.hidden_states_loader import (
    load_dataset_single,
    load_datasets,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# IID training data
# ------------------------------------------------------------------


def _load_grouped_training_data(
    iid_paths: List[str],
    filter_safe_only: bool,
    filter_unsafe_only: bool,
    holdout_ratio: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load training data for one filter-group (replicates
    run_anodet_analysis._load_training_data exactly).
    """
    cpu = torch.device("cpu")
    if holdout_ratio > 0:
        split_ratio = 1.0 - holdout_ratio
        data_dict = load_datasets(
            source=iid_paths,
            split="train",
            shuffle=True,
            device=cpu,
            s1_split_ratio=split_ratio,
            filter_safe_only=filter_safe_only,
            filter_unsafe_only=filter_unsafe_only,
        )
        hs = data_dict["s1"]["hidden_states"]
        lb = data_dict["s1"]["labels"]
    else:
        data_dict = load_datasets(
            source=iid_paths,
            split="train",
            shuffle=True,
            device=cpu,
            filter_safe_only=filter_safe_only,
            filter_unsafe_only=filter_unsafe_only,
        )
        hs = data_dict["train"]["hidden_states"]
        lb = data_dict["train"]["labels"]
    if hs.dim() == 3 and hs.shape[1] == 1:
        hs = hs.squeeze(1)
    return hs.cpu().numpy(), lb.cpu().numpy()


def load_iid_training_data(
    model_key: str,
    holdout_ratio: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reload IID training data from registry.

    Groups paths by filter type and loads each group via load_datasets
    (same logic as run_anodet_analysis._train_baseline).
    """
    iid_configs = get_dataset_paths(
        model_key, "iid", domain="common_sense_safety"
    )
    logger.info(f"Loading {len(iid_configs)} IID datasets for training data")

    paths_no_filter: List[str] = []
    paths_safe_only: List[str] = []
    paths_unsafe_only: List[str] = []
    for cfg in iid_configs:
        filt = cfg.get("filter", None)
        if filt == "safe_only":
            paths_safe_only.append(cfg["path"])
        elif filt == "unsafe_only":
            paths_unsafe_only.append(cfg["path"])
        else:
            paths_no_filter.append(cfg["path"])

    logger.info(
        f"IID datasets: {len(paths_no_filter)} unfiltered, "
        f"{len(paths_safe_only)} safe_only, "
        f"{len(paths_unsafe_only)} unsafe_only"
    )

    all_data: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    if paths_no_filter:
        d, lab = _load_grouped_training_data(
            paths_no_filter, False, False, holdout_ratio
        )
        all_data.append(d)
        all_labels.append(lab)
        logger.info(f"  Loaded {len(d)} samples from unfiltered IID")

    if paths_safe_only:
        d, lab = _load_grouped_training_data(
            paths_safe_only, True, False, holdout_ratio
        )
        all_data.append(d)
        all_labels.append(lab)
        logger.info(f"  Loaded {len(d)} samples from safe_only IID")

    if paths_unsafe_only:
        d, lab = _load_grouped_training_data(
            paths_unsafe_only, False, True, holdout_ratio
        )
        all_data.append(d)
        all_labels.append(lab)
        logger.info(f"  Loaded {len(d)} samples from unsafe_only IID")

    train_data = np.concatenate(all_data)
    train_labels = np.concatenate(all_labels)

    # Filter safe only (label==1)
    safe_mask = train_labels == 1
    safe_data = train_data[safe_mask]
    logger.info(
        f"Training data: {len(train_data)} total, "
        f"{len(safe_data)} safe samples"
    )
    return safe_data, train_labels[safe_mask]


# ------------------------------------------------------------------
# Eval data with per-sample source tracking
# ------------------------------------------------------------------


def load_eval_data_with_sources(
    model_key: str,
    include_benign_ood: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load eval data and return per-sample source labels.

    Returns (data, labels, source_ids, dataset_names) where source_ids[i]
    is an int index into dataset_names.

    When ``include_benign_ood`` is True, novel-harmless-domain datasets
    (HellaSwag, HH-RLHF-helpful) are appended with source names prefixed
    ``benign_ood:`` and label 0 (safe). They form a separate group ``B`` that
    the calibration driver tracks distinctly from in-distribution safe
    (``iid``) and unsafe (``ood``/``adv``); the ``benign_ood`` prefix keeps
    them out of the iid-vs-ood/adv 3-slice AUROC so that metric is unchanged.
    """
    cpu = torch.device("cpu")
    all_data: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_sources: List[np.ndarray] = []
    dataset_names: List[str] = []

    # IID test data (safe -> label 0)
    iid_configs = get_dataset_paths(
        model_key, "iid", domain="common_sense_safety"
    )
    for ds_cfg in iid_configs:
        filt = ds_cfg.get("filter", None)
        hs, lb = load_dataset_single(
            ds_cfg,
            split="test",
            device=cpu,
            filter_safe_only=(filt == "safe_only"),
            filter_unsafe_only=(filt == "unsafe_only"),
        )
        if hs.dim() == 3 and hs.shape[1] == 1:
            hs = hs.squeeze(1)
        data_np = hs.numpy()
        name = f"iid:{ds_cfg.get('name', '?')}"
        src_id = len(dataset_names)
        dataset_names.append(name)
        all_data.append(data_np)
        all_labels.append(np.zeros(len(data_np), dtype=np.int64))
        all_sources.append(np.full(len(data_np), src_id, dtype=np.int32))

    # OOD test data (unsafe -> label 1)
    ood_configs = get_dataset_paths(
        model_key, "ood", domain="common_sense_safety"
    )
    for ds_cfg in ood_configs:
        filt = ds_cfg.get("filter", None)
        hs, lb = load_dataset_single(
            ds_cfg,
            split="all",
            device=cpu,
            filter_unsafe_only=(filt == "unsafe_only"),
            filter_safe_only=(filt == "safe_only"),
        )
        if hs.dim() == 3 and hs.shape[1] == 1:
            hs = hs.squeeze(1)
        data_np = hs.numpy()
        name = f"ood:{ds_cfg.get('name', '?')}"
        src_id = len(dataset_names)
        dataset_names.append(name)
        all_data.append(data_np)
        all_labels.append(np.ones(len(data_np), dtype=np.int64))
        all_sources.append(np.full(len(data_np), src_id, dtype=np.int32))

    # Adversarial data (unsafe -> label 1)
    adv_base = get_adversarial_base_path(
        model_key, domain="common_sense_safety"
    )
    if adv_base:
        adv_configs = discover_and_load_adversarial_datasets(
            model_key=model_key, base_path=adv_base
        )
        for adv_cfg in adv_configs:
            hs, lb = load_adversarial_dataset(
                adv_cfg.path, cpu, filter_unsafe_only=True, max_samples=50000
            )
            if hs.dim() == 3 and hs.shape[1] == 1:
                hs = hs.squeeze(1)
            data_np = hs.numpy()
            name = f"adv:{adv_cfg.name}"
            src_id = len(dataset_names)
            dataset_names.append(name)
            all_data.append(data_np)
            all_labels.append(np.ones(len(data_np), dtype=np.int64))
            all_sources.append(np.full(len(data_np), src_id, dtype=np.int32))

    # Benign-OOD data (novel harmless domains -> label 0, group B).
    if include_benign_ood:
        benign_configs = get_benign_ood_datasets(model_key)
        for ds_cfg in benign_configs:
            hs, lb = load_dataset_single(
                ds_cfg,
                split="all",
                device=cpu,
            )
            if hs.dim() == 3 and hs.shape[1] == 1:
                hs = hs.squeeze(1)
            data_np = hs.numpy()
            name = f"benign_ood:{ds_cfg.get('name', '?')}"
            src_id = len(dataset_names)
            dataset_names.append(name)
            all_data.append(data_np)
            all_labels.append(np.zeros(len(data_np), dtype=np.int64))
            all_sources.append(np.full(len(data_np), src_id, dtype=np.int32))

    eval_data = np.concatenate(all_data)
    eval_labels = np.concatenate(all_labels)
    eval_sources = np.concatenate(all_sources)
    n_safe = int((eval_labels == 0).sum())
    n_unsafe = int((eval_labels == 1).sum())
    logger.info(
        f"Eval data: {len(eval_data)} total "
        f"({n_safe} safe, {n_unsafe} unsafe), "
        f"{len(dataset_names)} source datasets"
    )
    return eval_data, eval_labels, eval_sources, dataset_names
