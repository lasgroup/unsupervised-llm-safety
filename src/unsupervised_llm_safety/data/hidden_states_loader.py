import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig

log = logging.getLogger(__name__)

# Fixed seed for the flat -> {train, test} split. Deliberately independent of
# any experiment/anodet seed so the test target is identical across all runs
# and seeds (e.g. the main-results 5-seed sweep 42-46).
_FLAT_SPLIT_SEED = 42


def _normalize_flat_to_split(
    data: Dict[str, Any], split_ratio: float = 0.8
) -> Dict[str, Dict[str, Any]]:
    """Wrap a flat ``{hidden_states, labels, ...}`` dict into ``{train, test}``.

    The current HarmBench generation pipeline saves a flat dict with no
    train/test split, while the strict loader path (``validate_data``) requires
    one. This wraps a flat dict using a deterministic 80/20 split so flat files
    load identically to the legacy pre-split reference files.

    Determinism: uses a *local* ``RandomState`` seeded with the fixed
    ``_FLAT_SPLIT_SEED``. This (a) does not read or mutate the global numpy RNG
    that the anodet/SAE code relies on, and (b) does not depend on any
    experiment seed, so the test partition is byte-identical on every call and
    across all experiment seeds. Already-split dicts pass through unchanged.

    Args:
        data: Loaded dataset dict, either flat or already ``{train, test}``.
        split_ratio: Fraction assigned to the train split (default 0.8).

    Returns:
        Dict with ``train``/``test`` splits, each carrying ``hidden_states``,
        ``labels``, and any per-sample list field (e.g. ``original_sentences``).
    """
    # No-op for already-split data (reference-model files).
    if "train" in data and "test" in data:
        return data

    hidden_states = data["hidden_states"]
    labels = data["labels"]
    n_samples = len(hidden_states)

    rng = np.random.RandomState(_FLAT_SPLIT_SEED)
    perm = rng.permutation(n_samples)
    split_idx = int(split_ratio * n_samples)
    train_idx = perm[:split_idx]
    test_idx = perm[split_idx:]

    # Per-sample list fields (e.g. original_sentences) are carried along; scalar
    # metadata (method/model/layer/...) is irrelevant to downstream consumers.
    per_sample_lists = {
        key: value
        for key, value in data.items()
        if isinstance(value, list) and len(value) == n_samples
    }

    def _slice(indices: np.ndarray) -> Dict[str, Any]:
        idx = torch.as_tensor(indices, dtype=torch.long)
        split = {
            "hidden_states": hidden_states[idx],
            "labels": labels[idx],
        }
        for key, value in per_sample_lists.items():
            split[key] = [value[i] for i in indices]
        return split

    return {"train": _slice(train_idx), "test": _slice(test_idx)}


def validate_data(data: Dict[str, Dict[str, Any]]) -> bool:
    """Validate the structure of loaded data.

    Args:
        data: Dictionary containing hidden states data

    Returns:
        bool: True if data is valid, False otherwise
    """
    required_keys = {"train", "test"}
    required_subkeys = {"hidden_states", "labels"}

    if not all(key in data for key in required_keys):
        log.error(
            f"Missing required keys in data. Expected {required_keys}, got {set(data.keys())}"
        )
        return False

    for split in ["train", "test"]:
        if not all(key in data[split] for key in required_subkeys):
            log.error(
                f"Missing required subkeys in {split} data. Expected {required_subkeys}, got {set(data[split].keys())}"
            )
            return False

    return True


def sample_data(
    data: Dict[str, Any], ratio: float, seed: int = 42
) -> Dict[str, Any]:
    """Sample a portion of the data based on the given ratio."""
    if ratio >= 1.0:
        return data

    n_samples = len(data["hidden_states"])
    n_selected = int(n_samples * ratio)

    # Set random seed for reproducibility
    random.seed(seed)
    indices = random.sample(range(n_samples), n_selected)

    sampled_data = {
        "hidden_states": data["hidden_states"][indices],
        "labels": data["labels"][indices],
    }

    # Handle original_sentences if present
    if "original_sentences" in data and data["original_sentences"]:
        sampled_data["original_sentences"] = [
            data["original_sentences"][i] for i in indices
        ]
    elif "original_sentences" in data:
        # If original_sentences exists but is empty, keep it empty
        sampled_data["original_sentences"] = []
    # If original_sentences doesn't exist, don't add it

    # Handle other optional fields
    for key in data:
        if key not in ["hidden_states", "labels", "original_sentences"]:
            if isinstance(data[key], list) and len(data[key]) == n_samples:
                sampled_data[key] = [data[key][i] for i in indices]
            else:
                sampled_data[key] = data[key]

    return sampled_data


def merge_hidden_states_data(
    data_list: List[Dict[str, Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """Merge multiple hidden states datasets.

    Args:
        data_list: List of hidden states datasets to merge

    Returns:
        Dict containing merged 'train' and 'test' data
    """
    if not data_list:
        raise ValueError("No data to merge")

    # Initialize merged data structure
    merged_data: Dict[str, Dict[str, Any]] = {
        "train": {
            "hidden_states": [],
            "labels": [],
            "data_source": [],  # Add data source tracking
            "original_sentences": [],
        },
        "test": {
            "hidden_states": [],
            "labels": [],
            "data_source": [],  # Add data source tracking
            "original_sentences": [],
        },
    }

    # Merge each dataset
    for data in data_list:
        for split in ["train", "test"]:
            merged_data[split]["hidden_states"].append(
                data[split]["hidden_states"]
            )
            merged_data[split]["labels"].append(data[split]["labels"])
            # Add source information for each sample
            source_path = data["source_path"]
            n_samples = len(data[split]["hidden_states"])
            merged_data[split]["data_source"].extend([source_path] * n_samples)

            # Handle text data (original_sentences or input_texts)
            if (
                "original_sentences" in data[split]
                and data[split]["original_sentences"]
            ):
                # Dataset has original_sentences with content
                merged_data[split]["original_sentences"].extend(
                    data[split]["original_sentences"]
                )
            elif "input_texts" in data[split] and data[split]["input_texts"]:
                # Dataset has input_texts instead
                merged_data[split]["original_sentences"].extend(
                    data[split]["input_texts"]
                )
            else:
                # Dataset has no text data or empty text data
                # Add empty strings to maintain alignment
                merged_data[split]["original_sentences"].extend(
                    [""] * n_samples
                )

    # Convert lists to tensors where appropriate
    for split in ["train", "test"]:
        merged_data[split]["hidden_states"] = torch.cat(
            merged_data[split]["hidden_states"]
        )
        merged_data[split]["labels"] = torch.cat(merged_data[split]["labels"])
        # Store data_source as a list of strings instead of hashed values
        merged_data[split]["data_source"] = merged_data[split][
            "data_source"
        ]  # Keep as list of strings

    return merged_data


def shuffle_hidden_states_data(
    data: Dict[str, Any], seed: int = 42
) -> Dict[str, Any]:
    """Shuffle hidden states data while maintaining correspondence between fields.

    Args:
        data: Dictionary containing hidden states data with keys like
              'hidden_states', 'labels', 'original_sentences', 'data_source'
        seed: Random seed for reproducible shuffling (default: 42)

    Returns:
        Dictionary containing shuffled data with the same keys as input
    """
    # Get the number of samples from hidden_states
    n_samples = len(data["hidden_states"])

    # Create indices and shuffle them deterministically
    random.seed(seed)
    indices = list(range(n_samples))
    random.shuffle(indices)

    # Initialize result dictionary
    shuffled_data = {}

    # Shuffle each key in the data dictionary
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            # Handle PyTorch tensors
            shuffled_data[key] = value[indices]
        elif isinstance(value, list):
            # Handle lists
            # FIXED: Check if list is empty to avoid IndexError
            if len(value) == 0:
                shuffled_data[key] = []
            # FIXED: Check if list has same length as indices to avoid IndexError
            elif len(value) == n_samples:
                shuffled_data[key] = [value[i] for i in indices]
            else:
                # FIXED: List has different length, don't shuffle it
                log.warning(
                    f"List '{key}' has length {len(value)} != {n_samples}, not shuffling"
                )
                shuffled_data[key] = value
        else:
            # For other types, copy the value as-is
            shuffled_data[key] = value

    return shuffled_data


def split_hidden_states_data(
    data: Dict[str, Any], split_ratio: float = 0.5, seed: int = 42
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Split hidden states data into two parts based on the given ratio.

    Args:
        data: Dictionary containing hidden states data with keys like
              'hidden_states', 'labels', 'original_sentences'
        split_ratio: Ratio for the first split (default: 0.5)
        seed: Random seed for reproducible shuffling (default: 42)

    Returns:
        Tuple of two dictionaries (s1_data, s2_data), each containing
        the corresponding split data with the same keys as input
    """
    if split_ratio <= 0.0 or split_ratio >= 1.0:
        raise ValueError("split_ratio must be between 0.0 and 1.0 (exclusive)")

    # Get the number of samples from hidden_states
    n_samples = len(data["hidden_states"])

    # Create indices and shuffle them deterministically
    random.seed(seed)
    indices = list(range(n_samples))
    random.shuffle(indices)

    # Calculate split point
    split_point = int(n_samples * split_ratio)
    s1_indices = indices[:split_point]
    s2_indices = indices[split_point:]

    # Initialize result dictionaries
    s1_data = {}
    s2_data = {}

    # Split each key in the data dictionary
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            # Handle PyTorch tensors
            s1_data[key] = value[s1_indices]
            s2_data[key] = value[s2_indices]
        elif isinstance(value, list):
            # Handle lists - only split if length matches n_samples
            if len(value) == n_samples:
                s1_data[key] = [value[i] for i in s1_indices]
                s2_data[key] = [value[i] for i in s2_indices]
            elif len(value) == 0:
                # Empty list - skip it (don't include in split)
                pass
            else:
                # Length mismatch - skip this field or log a warning
                log.warning(
                    f"Skipping field '{key}' in split: length {len(value)} "
                    f"doesn't match hidden_states length {n_samples}"
                )
        else:
            # For other types, copy the value to both splits
            s1_data[key] = value
            s2_data[key] = value

    return s1_data, s2_data


def _apply_label_filter(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    filter_unsafe_only: bool = False,
    filter_safe_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Apply label filtering to hidden states and labels.

    Args:
        hidden_states: Tensor of hidden states
        labels: Tensor of labels
        filter_unsafe_only: If True, only keep unsafe samples (label=0)
        filter_safe_only: If True, only keep safe samples (label=1)

    Returns:
        Tuple of (filtered hidden_states, filtered labels, mask tensor or None)
    """
    if filter_unsafe_only and filter_safe_only:
        raise ValueError("Cannot filter for both safe and unsafe only")

    mask = None
    if filter_unsafe_only:
        mask = labels == 0
        hidden_states = hidden_states[mask]
        labels = labels[mask]
        log.info(f"Filtered to {len(labels)} unsafe samples")
    elif filter_safe_only:
        mask = labels == 1
        hidden_states = hidden_states[mask]
        labels = labels[mask]
        log.info(f"Filtered to {len(labels)} safe samples")

    return hidden_states, labels, mask


def _apply_sampling(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Apply random sampling to limit number of samples.

    Args:
        hidden_states: Tensor of hidden states
        labels: Tensor of labels
        max_samples: Maximum number of samples (None = no limit)
        seed: Random seed

    Returns:
        Tuple of (sampled hidden_states, sampled labels, indices tensor or None)
    """
    indices = None
    if max_samples and len(hidden_states) > max_samples:
        torch.manual_seed(seed)
        indices = torch.randperm(len(hidden_states))[:max_samples]
        hidden_states = hidden_states[indices]
        labels = labels[indices]

    return hidden_states, labels, indices


def load_datasets(
    source: Union[str, List[str], DictConfig],
    model_key: Optional[str] = None,
    dataset_type: Optional[str] = None,
    split: str = "test",
    max_samples: Optional[int] = None,
    filter_unsafe_only: bool = False,
    filter_safe_only: bool = False,
    device: Optional[torch.device] = None,
    train_ratio: Optional[float] = None,
    test_ratio: Optional[float] = None,
    shuffle: bool = True,
    seed: int = 42,
    s1_split_ratio: Optional[float] = None,
) -> Dict[str, Dict[str, Any]]:
    """Unified function to load hidden states datasets.

    This is the single entry point for all dataset loading. It handles:
    - Direct file paths (str or List[str])
    - Registry-based loading (via model_key + dataset_type)
    - Single dataset configs (DictConfig with 'path' and 'name')

    Args:
        source: Can be:
            - File path(s): str or List[str]
            - Dataset config: DictConfig with 'path' and 'name'
            - Full config: DictConfig (requires model_key + dataset_type for registry)
        model_key: Model key for registry-based loading (e.g., "qwen2_1.5b")
        dataset_type: Dataset type for registry ("iid", "ood", or "adversarial")
        split: Which split to use: "train", "test", or "all" (combine both)
        max_samples: Maximum number of samples per dataset (None = no limit)
        filter_unsafe_only: If True, only load unsafe samples (label=0)
        filter_safe_only: If True, only load safe samples (label=1)
        device: Device to load tensors to (None = CPU)
        train_ratio: Ratio of train split to load (None = 1.0, all data)
        test_ratio: Ratio of test split to load (None = 1.0, all data)
        shuffle: Whether to shuffle data after loading
        seed: Random seed for reproducibility
        s1_split_ratio: If provided, split train into s1/s2 using this ratio

    Returns:
        Dict with structure:
        - If s1_split_ratio provided: {"s1": {...}, "s2": {...}, "test": {...}}
        - If split="all": {"all": {...}} (combined train+test)
        - Otherwise: {"train": {...}, "test": {...}}
        Each split contains: {"hidden_states": Tensor, "labels": Tensor, ...}
    """
    if device is None:
        device = torch.device("cpu")

    # Resolve paths based on source type
    paths: List[str] = []
    dataset_names: List[str] = []

    if isinstance(source, str):
        # Direct file path
        paths = [source]
        dataset_names = [source.split("/")[-1]]  # Use filename as name
    elif isinstance(source, list) and all(isinstance(p, str) for p in source):
        # List of file paths
        paths = source
        dataset_names = [p.split("/")[-1] for p in paths]
    elif isinstance(source, DictConfig) or isinstance(source, dict):
        # Config object - check if it's a dataset config or full config
        if "path" in source and "name" in source:
            # Single dataset config
            paths = [
                source["path"] if isinstance(source, dict) else source.path
            ]
            dataset_names = [
                source["name"] if isinstance(source, dict) else source.name
            ]
        elif model_key and dataset_type:
            # Full config - use registry
            from unsupervised_llm_safety.data.dataset_registry import (
                get_dataset_paths,
            )

            dataset_configs = get_dataset_paths(model_key, dataset_type)
            paths = [cfg["path"] for cfg in dataset_configs]
            dataset_names = [cfg["name"] for cfg in dataset_configs]
        else:
            raise ValueError(
                "DictConfig source requires either 'path'/'name' keys or "
                "model_key + dataset_type for registry lookup"
            )
    else:
        raise ValueError(
            f"Invalid source type: {type(source)}. "
            "Expected str, List[str], or DictConfig/dict"
        )

    # Load data from files
    if train_ratio is None:
        train_ratios = [1.0] * len(paths)
    elif isinstance(train_ratio, float):
        train_ratios = [train_ratio] * len(paths)
    else:
        train_ratios = train_ratio

    if test_ratio is None:
        test_ratios = [1.0] * len(paths)
    elif isinstance(test_ratio, float):
        test_ratios = [test_ratio] * len(paths)
    else:
        test_ratios = test_ratio

    loaded_data = []
    for p, train_r, test_r in zip(paths, train_ratios, test_ratios):
        try:
            log.info(f"Loading hidden states from {p}")
            data = torch.load(p, weights_only=False, map_location="cpu")

            # Flat files (current HarmBench generation pipeline output) carry no
            # train/test split; normalize them deterministically so the strict
            # loader path accepts them like the legacy pre-split files.
            if "hidden_states" in data and not (
                "train" in data and "test" in data
            ):
                data = _normalize_flat_to_split(data)

            if not validate_data(data):
                raise ValueError(f"Invalid data structure in {p}")

            if train_r < 1.0:
                data["train"] = sample_data(data["train"], train_r, seed)
                log.info(
                    f"Sampled {len(data['train']['hidden_states'])} training examples "
                    f"({train_r*100:.1f}%) from {p}"
                )

            if test_r < 1.0:
                data["test"] = sample_data(data["test"], test_r, seed)
                log.info(
                    f"Sampled {len(data['test']['hidden_states'])} test examples "
                    f"({test_r*100:.1f}%) from {p}"
                )

            data["source_path"] = p
            loaded_data.append(data)
        except Exception as e:
            log.error(f"Error loading hidden states from {p}: {e}")
            raise

    # Merge multiple datasets
    if len(loaded_data) > 1:
        merged_data = merge_hidden_states_data(loaded_data)
    else:
        merged_data = loaded_data[0]

    # Handle split selection/combination
    if split == "all":
        # Combine train and test
        train_hs = merged_data["train"]["hidden_states"]
        test_hs = merged_data["test"]["hidden_states"]
        train_labels = merged_data["train"]["labels"]
        test_labels = merged_data["test"]["labels"]

        # Log what's being combined
        train_count = len(train_hs)
        test_count = len(test_hs)
        total_count = train_count + test_count

        log.info(
            f"Combining splits for split='all': "
            f"train={train_count} samples, test={test_count} samples, "
            f"total={total_count} samples"
        )

        # Warn if train split is empty (only test data available)
        if train_count == 0:
            log.warning(
                f"WARNING: Train split is empty! Only test data ({test_count} samples) "
                f"will be used. The dataset file may not contain train/test splits."
            )
        elif test_count == 0:
            log.warning(
                f"WARNING: Test split is empty! Only train data ({train_count} samples) "
                f"will be used."
            )

        hidden_states = torch.cat([train_hs, test_hs], dim=0)
        labels = torch.cat([train_labels, test_labels], dim=0)

        # Collect other fields before filtering
        all_sentences = None
        if "original_sentences" in merged_data["train"]:
            train_sentences = merged_data["train"].get(
                "original_sentences", []
            )
            test_sentences = merged_data["test"].get("original_sentences", [])
            all_sentences = train_sentences + test_sentences

        # Apply filtering
        hidden_states, labels, filter_mask = _apply_label_filter(
            hidden_states, labels, filter_unsafe_only, filter_safe_only
        )

        # Apply filter mask to sentences if filtering was applied
        if filter_mask is not None and all_sentences is not None:
            if len(all_sentences) > 0:
                if len(all_sentences) == len(filter_mask):
                    # Convert boolean mask to list of indices to keep
                    mask_list = filter_mask.cpu().tolist()
                    all_sentences = [
                        all_sentences[i]
                        for i in range(len(all_sentences))
                        if mask_list[i]
                    ]
                else:
                    # If lengths don't match, don't filter sentences (shouldn't happen, but be safe)
                    log.warning(
                        f"Length mismatch: sentences={len(all_sentences)}, mask={len(filter_mask)}. Skipping original_sentences filtering."
                    )
                    all_sentences = (
                        None  # Don't include if length doesn't match
                    )
            else:
                # Empty list - set to None so we don't include it
                all_sentences = None

        # Apply sampling
        hidden_states, labels, sample_indices = _apply_sampling(
            hidden_states, labels, max_samples, seed
        )

        # Apply sampling indices to sentences if sampling was applied
        if (
            sample_indices is not None
            and all_sentences is not None
            and len(all_sentences) > 0
        ):
            all_sentences = [all_sentences[i] for i in sample_indices.tolist()]

        # Move to device
        hidden_states = hidden_states.to(device).float()
        labels = labels.to(device)

        result = {
            "all": {
                "hidden_states": hidden_states,
                "labels": labels,
            }
        }

        # Preserve other fields if they exist and have content
        if all_sentences is not None and len(all_sentences) > 0:
            result["all"]["original_sentences"] = all_sentences

        return result

    # Handle train/test splits separately
    # Always load both splits, but only apply filtering/sampling to the requested split
    result = {}
    for split_name in ["train", "test"]:
        if split_name not in merged_data:
            continue

        hidden_states = merged_data[split_name]["hidden_states"]
        labels = merged_data[split_name]["labels"]

        # Collect other fields before filtering
        original_sentences = None
        if "original_sentences" in merged_data[split_name]:
            original_sentences = merged_data[split_name]["original_sentences"]

        data_source = None
        if "data_source" in merged_data[split_name]:
            data_source = merged_data[split_name]["data_source"]

        # Only apply filtering/sampling to the requested split
        filter_mask = None
        sample_indices = None
        if split in ["train", "test"] and split_name == split:
            # Apply filtering
            hidden_states, labels, filter_mask = _apply_label_filter(
                hidden_states, labels, filter_unsafe_only, filter_safe_only
            )

            # Apply filter mask to list fields if filtering was applied
            if filter_mask is not None:
                mask_list = filter_mask.cpu().tolist()
                if (
                    original_sentences is not None
                    and len(original_sentences) > 0
                ):
                    if len(original_sentences) == len(filter_mask):
                        original_sentences = [
                            original_sentences[i]
                            for i in range(len(original_sentences))
                            if mask_list[i]
                        ]
                    else:
                        log.warning(
                            f"Length mismatch: sentences={len(original_sentences)}, mask={len(filter_mask)}. Skipping original_sentences filtering."
                        )
                        original_sentences = (
                            None  # Don't include if length doesn't match
                        )
                elif (
                    original_sentences is not None
                    and len(original_sentences) == 0
                ):
                    # Empty list - keep it as None so we don't include it
                    original_sentences = None

                if (
                    data_source is not None
                    and isinstance(data_source, list)
                    and len(data_source) > 0
                ):
                    if len(data_source) == len(filter_mask):
                        data_source = [
                            data_source[i]
                            for i in range(len(data_source))
                            if mask_list[i]
                        ]
                    else:
                        log.warning(
                            f"Length mismatch: data_source={len(data_source)}, mask={len(filter_mask)}. Skipping data_source filtering."
                        )
                        data_source = (
                            None  # Don't include if length doesn't match
                        )

            # Apply sampling
            hidden_states, labels, sample_indices = _apply_sampling(
                hidden_states, labels, max_samples, seed
            )

            # Apply sampling indices to list fields if sampling was applied
            if sample_indices is not None:
                if (
                    original_sentences is not None
                    and len(original_sentences) > 0
                ):
                    original_sentences = [
                        original_sentences[i] for i in sample_indices.tolist()
                    ]
                if (
                    data_source is not None
                    and isinstance(data_source, list)
                    and len(data_source) > 0
                ):
                    data_source = [
                        data_source[i] for i in sample_indices.tolist()
                    ]

        # Move to device
        hidden_states = hidden_states.to(device).float()
        labels = labels.to(device)

        result[split_name] = {
            "hidden_states": hidden_states,
            "labels": labels,
        }

        # Preserve other fields (only if they have content)
        if original_sentences is not None and len(original_sentences) > 0:
            result[split_name]["original_sentences"] = original_sentences
        if data_source is not None:
            result[split_name]["data_source"] = data_source

    # Handle s1/s2 split for training
    if s1_split_ratio is not None and "train" in result:
        s1_data, s2_data = split_hidden_states_data(
            result["train"], s1_split_ratio, seed
        )
        result = {"s1": s1_data, "s2": s2_data, "test": result.get("test", {})}

    # Shuffle if requested
    if shuffle:
        for split_name in result:
            if "hidden_states" in result[split_name]:
                result[split_name] = shuffle_hidden_states_data(
                    result[split_name], seed
                )

    return result


# Convenience functions for common use cases
def load_dataset_single(
    source: Union[str, DictConfig],
    split: str = "test",
    device: Optional[torch.device] = None,
    max_samples: Optional[int] = None,
    filter_unsafe_only: bool = False,
    filter_safe_only: bool = False,
    shuffle: bool = True,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a single dataset and return just the tensors for the requested split.

    Convenience wrapper around load_datasets() that returns (hidden_states, labels) tuple
    instead of the full dict structure.

    Args:
        source: File path or dataset config dict with 'path' and 'name'
        split: Which split to return: "train", "test", or "all"
        device: Device to load tensors to
        max_samples: Maximum number of samples
        filter_unsafe_only: If True, only load unsafe samples
        filter_safe_only: If True, only load safe samples
        shuffle: Whether to shuffle
        seed: Random seed

    Returns:
        Tuple of (hidden_states, labels) tensors
    """
    data = load_datasets(
        source=source,
        split=split,
        device=device,
        max_samples=max_samples,
        filter_unsafe_only=filter_unsafe_only,
        filter_safe_only=filter_safe_only,
        shuffle=shuffle,
        seed=seed,
    )

    # Extract the requested split
    if split == "all":
        key = "all"
    else:
        key = split

    if key not in data:
        raise ValueError(
            f"Split '{split}' not found in loaded data. Available: {list(data.keys())}"
        )

    return data[key]["hidden_states"], data[key]["labels"]
