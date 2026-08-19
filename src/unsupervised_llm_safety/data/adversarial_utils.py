"""Utilities for loading and discovering adversarial datasets.

This module provides functions for loading adversarial datasets that don't follow
the standard train/test split structure, and for auto-discovering adversarial
attack files matching specific model configurations.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig

from unsupervised_llm_safety.models.model_registry import get_model_config

logger = logging.getLogger(__name__)


def load_adversarial_dataset(
    file_path: str,
    device: torch.device,
    filter_unsafe_only: bool = False,
    max_samples: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load adversarial dataset that doesn't have train/test splits.

    Args:
        file_path: Path to the adversarial data file
        device: Device to load data to
        filter_unsafe_only: If True, only return unsafe samples
        max_samples: Maximum number of samples to return

    Returns:
        Tuple of (hidden_states, labels)
    """
    data = torch.load(file_path, map_location="cpu", weights_only=False)

    # Handle both flat structure and train/test split structure
    if "hidden_states" in data:
        hidden_states = data["hidden_states"]
        labels = data["labels"]
    elif "train" in data and "test" in data:
        # Combine train+test for adversarial evaluation (use all data)
        hidden_states = torch.cat(
            [data["train"]["hidden_states"], data["test"]["hidden_states"]],
            dim=0,
        )
        labels = torch.cat(
            [data["train"]["labels"], data["test"]["labels"]],
            dim=0,
        )
    elif "test" in data:
        hidden_states = data["test"]["hidden_states"]
        labels = data["test"]["labels"]
    else:
        raise ValueError(
            f"Unexpected data structure in {file_path}. Keys: {list(data.keys())}"
        )

    # Filter out samples with NaN or Inf values
    finite_mask = torch.isfinite(hidden_states).all(dim=-1)
    n_original = len(hidden_states)
    n_nan_inf = (~finite_mask).sum().item()

    if n_nan_inf > 0:
        logger.warning(
            f"Filtering out {n_nan_inf}/{n_original} samples ({n_nan_inf/n_original*100:.1f}%) "
            f"with NaN/Inf values from {Path(file_path).name}"
        )
        hidden_states = hidden_states[finite_mask]
        labels = labels[finite_mask]

    # Apply filtering if requested
    if filter_unsafe_only:
        unsafe_mask = labels == 0
        hidden_states = hidden_states[unsafe_mask]
        labels = labels[unsafe_mask]
        logger.info(f"Filtered to {len(labels)} unsafe samples")

    # Limit samples if requested
    if max_samples and len(hidden_states) > max_samples:
        indices = torch.randperm(len(hidden_states))[:max_samples]
        hidden_states = hidden_states[indices]
        labels = labels[indices]

    logger.info(
        f"Loaded {len(hidden_states)} samples "
        f"(safe: {(labels == 1).sum().item()}, "
        f"unsafe: {(labels == 0).sum().item()})"
    )

    return hidden_states.to(device).float(), labels.to(device)


def discover_adversarial_files(
    model_key: str,
    attack_filter: Optional[List[str]] = None,
    layer: Optional[int] = None,
    base_path: Optional[str] = None,
    use_last_token: Optional[bool] = None,
    strict: bool = False,
) -> Dict[str, Path]:
    """Discover adversarial files matching the current model.

    Args:
        model_key: Key identifying the model in the registry
        attack_filter: Optional list of attack names to include
        layer: Specific layer number to match. When ``None`` this falls
            back to the model's active (canonical) layer rather than
            leaving the layer unconstrained -- see the note below.
        base_path: Base directory for harmbench data. Defaults to the
            registry path, which honours the ``ULS_HARMBENCH_BASE``
            environment override.
        strict: When True, raise ``FileNotFoundError`` instead of returning
            an empty dict if nothing matches. Use on evaluation paths, where
            "no adversarial data" must never pass silently as a result.

    Returns:
        Dict mapping attack names to file paths
    """
    if base_path is None:
        from unsupervised_llm_safety.data.dataset_registry import (
            get_harmbench_base_path,
        )

        base_path = get_harmbench_base_path()

    # An unconstrained layer is NOT safe here. Attacks are keyed by name
    # below, so when the harmbench directory holds several layers for one
    # model the last match in sorted order silently wins -- and "_l8_"
    # sorts after "_l25_", so a multi-layer directory resolves to layer 8.
    # The published runs predate the layer sweep and so read the canonical
    # layer; re-running today against the same code would quietly evaluate
    # layer 8 for llama3_8b, gpt_oss_20b and gemma4_26b. Default to the
    # active layer instead, matching how the IID-safe HarmBench rows are
    # already resolved in dataset_registry.get_harmbench_safe_entries.
    if layer is None:
        from unsupervised_llm_safety.data.dataset_registry import (
            get_active_layer,
        )

        layer = get_active_layer(model_key)
        if layer is not None:
            logger.info(
                f"No layer requested; pinning adversarial discovery to the "
                f"active layer for '{model_key}': l{layer}"
            )
    # Get harmbench model name
    try:
        config = get_model_config(model_key)
        harmbench_name = config.get("harmbench_name", model_key)
    except ValueError:
        logger.warning(
            f"Model key '{model_key}' not found in registry, using as-is"
        )
        harmbench_name = model_key

    # Convert harmbench_name to file format
    file_model_names = {
        "qwen_1.5b": "qwen2_1_5b",
        "mistral_8b": "ministral_8b",
    }
    file_model_name = file_model_names.get(harmbench_name, harmbench_name)

    # Resolve last-token mode: explicit parameter > global flag
    from unsupervised_llm_safety.data.dataset_registry import (
        get_last_token_mode,
    )

    last_token = (
        use_last_token if use_last_token is not None else get_last_token_mode()
    )

    # Find matching files
    base_dir = Path(base_path)
    if not base_dir.exists():
        logger.warning(f"Adversarial data directory not found: {base_path}")
        return {}

    # When last-token mode is on, use _generation_last_token.pth files
    # (one representation per task) instead of _generation.pth (per-token).
    if last_token:
        pattern = f"*_{file_model_name}_l*_generation_last_token.pth"
    else:
        pattern = f"*_{file_model_name}_l*_generation.pth"
    logger.info(f"Searching for adversarial files with pattern: {pattern}")

    all_matches = sorted(base_dir.glob(pattern))
    adversarial_files: dict[str, Path] = {}
    for file_path in all_matches:
        # Extract attack name from filename
        # Filename format: {Attack}_{model}_l{layer}[_generation][_last_token].pth
        filename = file_path.stem
        parts = filename.split("_")
        attack_name = parts[0]

        # Skip special files
        if attack_name in ["hb", "extraction"]:
            continue

        # Apply filters
        if attack_filter and attack_name not in attack_filter:
            continue

        if layer is not None:
            layer_match = re.search(r"_l(\d+)_generation", filename)
            if layer_match and int(layer_match.group(1)) != layer:
                continue

        adversarial_files[attack_name] = file_path
        logger.info(f"  Found: {attack_name} -> {file_path.name}")

    # Say plainly when the result is empty. A silently empty dict is
    # indistinguishable downstream from a clean run that changed nothing,
    # which is how an all-zero evaluation once read as "the fix did nothing".
    if not adversarial_files:
        if all_matches:
            available = sorted(
                int(m.group(1))
                for f in all_matches
                if (m := re.search(r"_l(\d+)_generation", f.stem))
            )
            msg = (
                f"No adversarial files for model '{model_key}' at layer "
                f"{layer} in {base_path}. The directory does hold this "
                f"model at layer(s) {sorted(set(available))}. Pass an "
                f"explicit `layer`, set ULS_LAYER_OVERRIDE, or point "
                f"ULS_HARMBENCH_BASE at a directory carrying that layer."
            )
        else:
            msg = (
                f"No adversarial files at all for model '{model_key}' in "
                f"{base_path} (pattern '{pattern}'). If ULS_HARMBENCH_BASE "
                f"is set, that directory likely does not cover this model."
            )
        if strict:
            raise FileNotFoundError(msg)
        logger.error(msg)

    return adversarial_files


def discover_and_load_adversarial_datasets(
    model_key: str,
    base_path: str,
    attack_filter: Optional[List[str]] = None,
    layer: Optional[int] = None,
    strict: bool = True,
) -> List[DictConfig]:
    """Create dataset configurations for all matching adversarial files.

    Args:
        model_key: Key identifying the model in the registry
        base_path: Base directory path for harmbench data
        attack_filter: Optional list of attack names to include
        layer: Specific layer to match; ``None`` resolves to the model's
            active (canonical) layer
        strict: Raise rather than return an empty list when nothing
            matches. Defaults to True because every caller of this
            function is an evaluation path, where an empty adversarial
            set silently produces a table of missing cells.

    Returns:
        List of DictConfig objects for each discovered file
    """
    adversarial_files = discover_adversarial_files(
        model_key=model_key,
        attack_filter=attack_filter,
        layer=layer,
        base_path=base_path,
        strict=strict,
    )

    if not adversarial_files:
        logger.warning("No adversarial files found matching criteria")
        return []

    # Create DictConfig for each file
    configs = []
    for attack_name, file_path in adversarial_files.items():
        config = DictConfig(
            {
                "path": str(file_path),
                "name": f"HarmBench-{attack_name}",
                "split": "test",  # Adversarial data typically doesn't have splits
            }
        )
        configs.append(config)

    return configs
