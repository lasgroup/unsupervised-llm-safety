#!/usr/bin/env python3
"""
Extract activations from language models for toxicity detection datasets.

This script extracts hidden state activations from a specified layer of a language model
and saves them for downstream safety classification tasks.
"""

import gc
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.cuda import OutOfMemoryError
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from unsupervised_llm_safety.common.train_utils import set_seed
from unsupervised_llm_safety.data.dataset_processors import (
    create_dataset_processor,
)
from unsupervised_llm_safety.models.model_registry import get_model_config

log = logging.getLogger(__name__)


class ActivationExtractor:
    """Hook-based activation extractor for transformer models."""

    def __init__(self):
        self.activations = {}
        self.hooks = []

    def _create_hook(self, name: str):
        """Create a forward hook to capture activations."""

        def hook(module, input, output):
            # Handle different output types
            if isinstance(output, torch.Tensor):
                self.activations[name] = output.detach()
            elif isinstance(output, tuple) and len(output) > 0:
                # For models that return tuples (hidden_states, attention_weights, ...)
                self.activations[name] = output[0].detach()
            else:
                raise ValueError(
                    f"Unexpected output type from {name}: {type(output)}"
                )

        return hook

    def register_hook(self, model: torch.nn.Module, layer_name: str) -> None:
        """Register a forward hook on the specified layer."""
        # Clear previous activations
        self.activations = {}

        # Navigate to the target layer
        parts = layer_name.split(".")
        module = model

        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                # Handle numeric indices
                try:
                    idx = int(part)
                    module = module[idx]
                except (ValueError, TypeError, IndexError):
                    raise ValueError(
                        f"Layer '{layer_name}' not found in model"
                    )

        # Register the hook
        hook = module.register_forward_hook(self._create_hook(layer_name))
        self.hooks.append(hook)

    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Return the captured activations."""
        return self.activations

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}


def print_model_layers(model: torch.nn.Module) -> None:
    """Print the model architecture and layer names for reference."""
    log.info("\nModel Architecture:")
    log.info("=" * 80)

    # Print model type
    log.info(f"Model Type: {type(model).__name__}")

    # Check if it's a transformer model with layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        num_layers = len(model.model.layers)
        log.info(f"Number of layers: {num_layers}")
        log.info("\nAvailable layer indices for extraction:")
        for i in range(num_layers):
            log.info(f"  - Layer {i}: model.layers.{i}")

        # Print some example layer paths
        log.info("\nExample layer paths:")
        log.info("  - First layer: model.layers.0")
        log.info(f"  - Middle layer: model.layers.{num_layers // 2}")
        log.info(f"  - Last layer: model.layers.{num_layers - 1}")

    # Also print first few levels of model structure
    log.info("\nModel structure (first 2 levels):")
    for name, module in model.named_children():
        log.info(f"  {name}: {type(module).__name__}")
        if hasattr(module, "named_children"):
            for sub_name, sub_module in list(module.named_children())[:3]:
                log.info(f"    {name}.{sub_name}: {type(sub_module).__name__}")

    log.info("=" * 80)


def load_model_with_config(
    model_config: Dict[str, Any],
    device_map: str = "auto",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model and tokenizer based on model registry configuration.

    Args:
        model_config: Model configuration from the registry
        device_map: Device mapping for CUDA (from Hydra config)

    Returns:
        Tuple of (model, tokenizer)
    """

    log.info(f"Loading model: {model_config['hf_name']}")
    log.info(f"Cuda available: {torch.cuda.is_available()}")
    log.info(f"Dtype: {model_config['dtype']}")

    # Map string dtype to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(model_config["dtype"], torch.float16)

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["hf_name"],
            trust_remote_code=model_config["trust_remote_code"],
            padding_side="left",
            device_map="auto",
        )

        # Set pad token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            log.info("Set pad_token to eos_token")

        # Load model
        model_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": model_config["trust_remote_code"],
        }

        model = AutoModelForCausalLM.from_pretrained(
            model_config["hf_name"], device_map=device_map, **model_kwargs
        )
        log.info(f"Model device map: {model.hf_device_map}")

        model.eval()
        log.info(f"Model loaded successfully. Type: {type(model).__name__}")

        return model, tokenizer

    except Exception as e:
        log.error(f"Failed to load model: {str(e)}")
        raise


def compute_max_length_from_dataset(
    texts: List[str],
    tokenizer: AutoTokenizer,
    sample_size: Optional[int] = None,
    padding_margin: float = 0.1,
) -> int:
    """Compute maximum tokenized length from dataset texts.

    Args:
        texts: List of text samples
        tokenizer: Tokenizer to use for tokenization
        sample_size: If provided, only sample this many texts for efficiency
        padding_margin: Fraction to add as padding (e.g., 0.1 = 10% margin)

    Returns:
        Maximum tokenized length with padding margin
    """
    log.info("Computing max_length from dataset tokenized lengths...")

    # Sample texts if dataset is large
    if sample_size and len(texts) > sample_size:
        import random

        random.seed(42)  # For reproducibility
        sample_texts = random.sample(texts, sample_size)
        log.info(
            f"Sampling {sample_size} texts from {len(texts)} total for max_length computation"
        )
    else:
        sample_texts = texts

    max_length = 0
    total_lengths = []

    for text in tqdm(
        sample_texts, desc="Tokenizing samples", total=len(sample_texts)
    ):
        try:
            # Apply chat template (same as in extraction)
            messages = [{"role": "user", "content": text}]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=False,  # No padding for length computation
                truncation=False,  # No truncation to get actual length
            )

            # Get sequence length
            seq_length = inputs["input_ids"].shape[1]
            total_lengths.append(seq_length)
            max_length = max(max_length, seq_length)

        except Exception as e:
            log.warning(f"Error tokenizing sample: {str(e)}. Skipping...")
            continue

    if max_length == 0:
        log.warning(
            "No valid tokenized samples found. Using default max_length=8192"
        )
        return 8192

    # Add padding margin
    max_length_with_margin = int(max_length * (1 + padding_margin))

    # Log statistics
    if total_lengths:
        avg_length = np.mean(total_lengths)
        median_length = np.median(total_lengths)
        p95_length = np.percentile(total_lengths, 95)
        p99_length = np.percentile(total_lengths, 99)

        log.info(
            f"Tokenized length statistics (from {len(total_lengths)} samples):"
        )
        log.info(f"  Min: {min(total_lengths)}, Max: {max_length}")
        log.info(f"  Mean: {avg_length:.1f}, Median: {median_length:.1f}")
        log.info(
            f"  95th percentile: {p95_length:.1f}, 99th percentile: {p99_length:.1f}"
        )
        log.info(
            f"  Computed max_length (with {padding_margin*100:.0f}% margin): {max_length_with_margin}"
        )

    return max_length_with_margin


def load_dataset_with_config(
    cfg: DictConfig,
) -> Tuple[List[str], List[int], Dict]:
    """Load dataset based on configuration and return texts, labels, and metadata."""
    log.info(f"Loading dataset: {cfg.dataset.name}")

    try:
        # Create appropriate dataset processor
        processor = create_dataset_processor(cfg)

        # Load and process the dataset
        result = processor.load_and_process()

        # Handle both old (2-tuple) and new (3-tuple) return formats
        if len(result) == 2:
            texts, labels = result
            metadata = {}
        else:
            texts, labels, metadata = result

        log.info(f"Loaded {len(texts)} text samples with labels")
        if metadata:
            log.info(f"Dataset metadata: {metadata}")
        return texts, labels, metadata

    except Exception as e:
        log.error(f"Failed to load dataset: {str(e)}")
        raise


def extract_activations_with_retry(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    labels: List[int],
    cfg: DictConfig,
    max_length: int,
) -> Tuple[torch.Tensor, List[int], Dict]:
    """Extract activations with automatic batch size reduction on OOM."""
    all_activations = []
    # If the model is sharded across multiple devices via device_map, keep it
    # as-is and leave inputs on CPU (Accelerate will dispatch). If there is
    # only a single target device in the map, treat it as single-device and
    # move inputs there to avoid CPU/CUDA mismatches (e.g., index_select
    # complaining about indices on CPU).
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        unique_devices = {str(d) for d in device_map.values()}
        if len(unique_devices) == 1:
            device_token = list(unique_devices)[0]
            # Normalize device tokens like "0" -> "cuda:0"
            if device_token.isdigit():
                device_token = f"cuda:{device_token}"
            device = torch.device(device_token)
            model = model.to(device)
        else:
            device = None  # multi-device: keep inputs on CPU for dispatch
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    # Initialize batch size
    current_batch_size = cfg.memory.initial_batch_size
    processed_samples = 0

    # Statistics
    stats = {
        "total_samples": len(texts),
        "processed_samples": 0,
        "failed_samples": 0,
        "batch_size_reductions": 0,
        "final_batch_size": current_batch_size,
    }

    # Create progress bar
    pbar = tqdm(total=len(texts), desc="Extracting activations")

    i = 0
    while i < len(texts):
        batch_end = min(i + current_batch_size, len(texts))
        batch_texts = texts[i:batch_end]

        try:
            # Create activation extractor
            extractor = ActivationExtractor()
            layer_name = f"model.layers.{cfg.extraction.layer_index}"
            extractor.register_hook(model, layer_name)

            try:
                # Apply chat template to batch - process as a list of conversations
                messages_batch = [
                    [{"role": "user", "content": text}] for text in batch_texts
                ]

                # Apply chat template exactly as in official documentation
                inputs = tokenizer.apply_chat_template(
                    messages_batch,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                )

                # For single-device models, move inputs to the model device.
                if device is not None:
                    inputs = inputs.to(device)

                # Forward pass
                with torch.no_grad():
                    model(**inputs)

                # Get activations
                activations = extractor.get_activations()[layer_name]

                # Handle different activation shapes
                if (
                    len(activations.shape) == 3
                ):  # (batch, seq_len, hidden_size)
                    # Get last token activations
                    last_token_activations = activations[:, -1, :]
                elif len(activations.shape) == 2:  # (batch, hidden_size)
                    last_token_activations = activations
                else:
                    raise ValueError(
                        f"Unexpected activation shape: {activations.shape}"
                    )

                all_activations.append(last_token_activations.cpu())

                # Update progress
                processed_samples += len(batch_texts)
                pbar.update(len(batch_texts))
                stats["processed_samples"] = processed_samples

                # Move to next batch
                i = batch_end

                # Periodic garbage collection
                if (
                    processed_samples
                    % (cfg.memory.gc_frequency * current_batch_size)
                    == 0
                ):
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            finally:
                extractor.remove_hooks()

        except (OutOfMemoryError, RuntimeError) as e:
            if (
                "out of memory" in str(e).lower()
                and current_batch_size > cfg.memory.min_batch_size
            ):
                # Clear memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Reduce batch size
                current_batch_size = max(
                    int(current_batch_size * cfg.memory.reduce_factor),
                    cfg.memory.min_batch_size,
                )
                stats["batch_size_reductions"] += 1
                stats["final_batch_size"] = current_batch_size

                log.warning(
                    f"OOM error, reducing batch size to {current_batch_size}"
                )

                # Don't increment i, retry the same batch with smaller size
            else:
                # Unrecoverable error or batch size already at minimum
                log.error(f"Failed to process batch at index {i}: {str(e)}")
                stats["failed_samples"] += len(batch_texts)

                # Skip this batch
                i = batch_end
                pbar.update(len(batch_texts))

        except Exception as e:
            log.error(
                f"Unexpected error processing batch at index {i}: {str(e)}"
            )
            stats["failed_samples"] += len(batch_texts)

            # Skip this batch
            i = batch_end
            pbar.update(len(batch_texts))

    pbar.close()

    # Concatenate all batches
    if all_activations:
        final_activations = torch.cat(all_activations, dim=0)
        log.info(f"Extracted activations shape: {final_activations.shape}")
    else:
        raise ValueError("No activations were successfully extracted")

    return final_activations, labels, stats


def make_json_serializable(obj: Any) -> Any:
    """Convert object to JSON-serializable format."""
    import numpy as np
    import torch

    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {
            key: make_json_serializable(value) for key, value in obj.items()
        }
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    else:
        return obj


def save_activations_with_metadata(
    activations: torch.Tensor,
    texts: List[str],
    labels: List[int],
    cfg: DictConfig,
    stats: Dict,
    dataset_metadata: Optional[Dict] = None,
    model_short_name: Optional[str] = None,
    extract_per_token: bool = False,
    extraction_metadata: Optional[Dict] = None,
) -> None:
    """Save activations along with metadata in train/test split format.

    Args:
        activations: Tensor of activations
        texts: List of original texts
        labels: List of labels
        cfg: Configuration dict
        stats: Extraction statistics
        dataset_metadata: Optional dataset metadata
        model_short_name: Short name of the model
        extract_per_token: Whether this is per-token extraction (adds _generation suffix)
        extraction_metadata: Full extraction metadata for per-token mode
    """
    output_path = Path(cfg.extraction.output_path)

    # Ensure filename includes "_generation" suffix for per-token extraction
    if extract_per_token and "_generation" not in output_path.stem:
        # Insert "_generation" before .pth extension
        output_path = (
            output_path.parent
            / f"{output_path.stem}_generation{output_path.suffix}"
        )
        log.info(
            f"Updated output path to include '_generation' suffix: {output_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare data in the required format
    hidden_states_tensor = activations
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    n_samples = len(hidden_states_tensor)

    # Check if dataset_metadata contains pre-defined splits
    preserve_splits = (
        dataset_metadata
        and dataset_metadata.get("preserve_splits", False)
        and "split_labels" in dataset_metadata
    )

    if preserve_splits and dataset_metadata is not None:
        # Use pre-defined splits from dataset metadata
        split_labels = dataset_metadata["split_labels"]
        if len(split_labels) != len(hidden_states_tensor):
            log.warning(
                f"Split labels length ({len(split_labels)}) doesn't match "
                f"activations length ({len(hidden_states_tensor)}). "
                f"Falling back to random split."
            )
            preserve_splits = False

    if preserve_splits:
        # Use pre-defined splits
        train_indices = [
            i for i, split in enumerate(split_labels) if split == "train"
        ]
        test_indices = [
            i for i, split in enumerate(split_labels) if split == "test"
        ]
        log.info(
            f"Using pre-defined splits: {len(train_indices)} train, {len(test_indices)} test"
        )
    else:
        # Perform train/test split (80/20 by default)
        indices = list(range(n_samples))

        # Set random seed for reproducibility
        np.random.seed(42)
        np.random.shuffle(indices)

        split_ratio = getattr(cfg, "train_test_split", 0.8)
        split_idx = int(split_ratio * n_samples)
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]

    # Create train and test dictionaries
    train_data = {
        "hidden_states": hidden_states_tensor[train_indices],
        "labels": labels_tensor[train_indices],
        "original_sentences": (
            [texts[i] for i in train_indices]
            if len(texts) == n_samples
            else []
        ),
    }

    test_data = {
        "hidden_states": hidden_states_tensor[test_indices],
        "labels": labels_tensor[test_indices],
        "original_sentences": (
            [texts[i] for i in test_indices] if len(texts) == n_samples else []
        ),
    }

    # Save in the required format
    save_dict = {"train": train_data, "test": test_data}

    # Add metadata if available
    if extraction_metadata:
        # Per-token extraction mode: use full extraction metadata
        save_dict["extraction_metadata"] = extraction_metadata
    elif dataset_metadata:
        # Last-token extraction mode: use simple metadata
        save_dict["metadata"] = {
            **dataset_metadata,
            "model": model_short_name or "unknown",
            "layer": cfg.extraction.layer_index,
            "extraction_stats": stats,
        }

    log.info(
        f"Split data into {len(train_indices)} training and {len(test_indices)} test samples"
    )

    # Save the data
    torch.save(save_dict, output_path)
    log.info(f"Saved activations to {output_path}")

    # Save separate metadata JSON file if extraction_metadata is provided
    if extraction_metadata and cfg.extraction.get("save_metadata", True):
        import json

        from omegaconf import OmegaConf

        metadata_path = output_path.with_suffix(".metadata.json")
        json_metadata = {
            "model_config": extraction_metadata.get("model_config", {}),
            "extraction_stats": extraction_metadata.get(
                "extraction_stats", {}
            ),
            "task_mapping": extraction_metadata.get("task_mapping", []),
            "task_metadata": extraction_metadata.get("task_metadata", []),
            "hydra_config": OmegaConf.to_container(cfg, resolve=True),
        }

        # Ensure JSON serializability
        json_metadata = make_json_serializable(json_metadata)

        with open(metadata_path, "w") as f:
            json.dump(json_metadata, f, indent=2)

        log.info(f"Saved metadata to {metadata_path}")

    # Log statistics
    log.info("Extraction Statistics:")
    log.info(f"  Total samples: {stats['total_samples']}")
    log.info(f"  Processed samples: {stats['processed_samples']}")
    log.info(f"  Failed samples: {stats['failed_samples']}")
    log.info(f"  Batch size reductions: {stats['batch_size_reductions']}")
    log.info(f"  Final batch size: {stats['final_batch_size']}")


@hydra.main(
    version_base="1.2",
    config_path=f"{os.getcwd()}/exp_configs",
    config_name="config",
)
def main(cfg: DictConfig):
    """Main function to extract activations."""
    # Set seed for reproducibility
    cfg = cfg.analysis
    set_seed(cfg.seed)

    # Require model_key
    if not hasattr(cfg, "model_key") or cfg.model_key is None:
        raise ValueError("model_key must be specified")

    log.info(f"Using model key: {cfg.model_key}")
    model_cfg = get_model_config(cfg.model_key)

    # Set default layer index if not specified
    if (
        hasattr(cfg.extraction, "layer_index")
        and cfg.extraction.layer_index is None
    ):
        cfg.extraction.layer_index = model_cfg["default_layer"]

    # Log configuration
    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))
    log.info("Model configuration from registry:")
    log.info(f"  HF Name: {model_cfg['hf_name']}")
    log.info(f"  Short Name: {model_cfg['short_name']}")
    log.info(f"  Default Layer: {model_cfg['default_layer']}")
    log.info(f"  Trust Remote Code: {model_cfg['trust_remote_code']}")
    log.info(f"  Dtype: {model_cfg['dtype']}")

    try:
        # Load model with config from registry and device_map from Hydra config
        device_map = (
            cfg.model.device_map
            if hasattr(cfg, "model") and hasattr(cfg.model, "device_map")
            else "auto"
        )
        model, tokenizer = load_model_with_config(model_cfg, device_map)

        # Print model layers if verbose
        if cfg.verbose:
            print_model_layers(model)

        # Load dataset
        texts, labels, dataset_metadata = load_dataset_with_config(cfg)

        # Always compute max_length dynamically from dataset
        sample_size = cfg.extraction.get("auto_max_length_sample_size", 8192)
        padding_margin = cfg.extraction.get(
            "auto_max_length_padding_margin", 0.1
        )

        computed_max_length = compute_max_length_from_dataset(
            texts,
            tokenizer,
            sample_size=sample_size,
            padding_margin=padding_margin,
        )
        log.info(f"Auto-computed max_length: {computed_max_length}")

        # Check if we should use per-token generation extraction
        extract_per_token = cfg.extraction.get("extract_per_token", False)

        if extract_per_token:
            # Per-token generation extraction mode
            log.info("Starting per-token generation activation extraction...")
            from unsupervised_llm_safety.scripts.activations.extract_activations_with_generation import (
                extract_activations_with_generation,
            )

            # Determine dataset name from config
            dataset_name = cfg.dataset.name.split("/")[-1].lower()
            # Order matters: check specific names before substrings
            name_checks = [
                ("humanevalplus", "humaneval_plus"),
                ("humaneval", "humaneval"),
                ("mbpp", "mbpp"),
                ("gsm8k", "gsm8k"),
                ("math_sft_40k", "math_sft_40k"),
                ("math_sft", "math_sft_40k"),
                ("alpaca", "alpaca"),
            ]
            matched = "unknown"
            for key, val in name_checks:
                if key in dataset_name:
                    matched = val
                    break
            dataset_name = matched

            activations, labels, extraction_metadata = (
                extract_activations_with_generation(
                    model,
                    tokenizer,
                    texts,
                    labels,
                    cfg,
                    dataset_name=dataset_name,
                )
            )

            # Convert extraction_metadata to stats format for compatibility
            stats = {
                "total_samples": extraction_metadata["extraction_stats"][
                    "total_tasks"
                ],
                "processed_samples": extraction_metadata["extraction_stats"][
                    "successful_tasks"
                ],
                "failed_samples": extraction_metadata["extraction_stats"][
                    "failed_tasks"
                ],
                "batch_size_reductions": 0,
                "final_batch_size": 1,  # Per-token uses batch size 1
            }

            # Save activations with metadata
            if cfg.extraction.get("save_metadata", True):
                save_activations_with_metadata(
                    activations,
                    texts,
                    labels,
                    cfg,
                    stats,
                    dataset_metadata=dataset_metadata,
                    model_short_name=model_cfg["short_name"],
                    extract_per_token=True,
                    extraction_metadata=extraction_metadata,
                )
            else:
                # Save just the activations tensor (legacy format)
                output_path = Path(cfg.extraction.output_path)
                if "_generation" not in output_path.stem:
                    output_path = (
                        output_path.parent
                        / f"{output_path.stem}_generation{output_path.suffix}"
                    )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(activations, output_path)
                log.info(f"Saved raw activations to {output_path}")
        else:
            # Last-token extraction mode (original behavior)
            log.info("Starting activation extraction (last-token mode)...")
            activations, labels, stats = extract_activations_with_retry(
                model, tokenizer, texts, labels, cfg, computed_max_length
            )

            # Save activations with metadata
            if cfg.extraction.save_metadata:
                save_activations_with_metadata(
                    activations,
                    texts,
                    labels,
                    cfg,
                    stats,
                    dataset_metadata,
                    model_short_name=model_cfg["short_name"],
                )
            else:
                # Save just the activations tensor (legacy format)
                output_path = Path(cfg.extraction.output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(activations, output_path)
                log.info(f"Saved raw activations to {output_path}")

        # Save results summary next to the output .pth file
        output_pth = Path(cfg.extraction.output_path)
        if extract_per_token and "_generation" not in output_pth.stem:
            output_pth = (
                output_pth.parent
                / f"{output_pth.stem}_generation{output_pth.suffix}"
            )
        results_path = str(output_pth.with_suffix(".results.json"))
        import json

        # Build file structure description
        file_structure: Dict[str, Any] = {
            "format": ".pth (torch.save)",
            "top_level_keys": ["train", "test"],
        }
        if extract_per_token:
            file_structure["top_level_keys"].append("extraction_metadata")
            file_structure["train"] = {
                "hidden_states": "Tensor [n_train_tokens, hidden_dim] — per-token activations from generation",
                "labels": "Tensor [n_train_tokens] — label per token (all tokens from same task share label)",
                "original_sentences": "List[str] — original prompts",
            }
            file_structure["test"] = {
                "hidden_states": "Tensor [n_test_tokens, hidden_dim]",
                "labels": "Tensor [n_test_tokens]",
                "original_sentences": "List[str]",
            }
            file_structure["extraction_metadata"] = {
                "model_config": "dict with model_id, layer_index, max_new_tokens, dataset",
                "extraction_stats": "dict with total/successful/failed tasks, token counts",
                "task_mapping": "List[[task_idx, start_token_idx, end_token_idx]] — maps tasks to token ranges",
                "task_metadata": "List[dict] — per-task info (full prompt, full generated_solution, num_tokens, etc.)",
            }
        else:
            file_structure["train"] = {
                "hidden_states": "Tensor [n_train, hidden_dim] — last-token activations",
                "labels": "Tensor [n_train]",
                "original_sentences": "List[str]",
            }
            file_structure["test"] = {
                "hidden_states": "Tensor [n_test, hidden_dim]",
                "labels": "Tensor [n_test]",
                "original_sentences": "List[str]",
            }
            if dataset_metadata:
                file_structure["top_level_keys"].append("metadata")

        with open(results_path, "w") as f:
            # Don't resolve interpolations to avoid errors with ${model_key}
            json.dump(
                {
                    "success": True,
                    "output_path": str(cfg.extraction.output_path),
                    "extraction_mode": (
                        "per_token_generation"
                        if extract_per_token
                        else "last_token"
                    ),
                    "stats": stats,
                    "file_structure": file_structure,
                    "config": OmegaConf.to_container(cfg, resolve=False),
                },
                f,
                indent=2,
            )

        log.info("\nExtraction completed successfully!")
        log.info(f"Results saved to: {results_path}")

    except Exception as e:
        log.error(f"Extraction failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
