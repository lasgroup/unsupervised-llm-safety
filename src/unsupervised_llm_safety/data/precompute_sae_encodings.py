#!/usr/bin/env python3
"""Pre-compute SAE encodings for all datasets.

Encodes all training + test hidden states through a frozen SAE encoder
and saves the encoded features. Preserves exact 1-1 mapping with
original raw features (same ordering, same splits, no filtering).

Output structure per model:
    {output_dir}/{model_key}/
        metadata.json
        iid/{dataset_name}.pth
        ood/{dataset_name}.pth
        adversarial/{attack_name}.pth

Each .pth file contains:
    {
        "sae_path": str,
        "original_path": str,
        "train": {
            "encoded_hidden_states": Tensor[N, sae_dim],  # float16
            "labels": Tensor[N],
        },
        "test": {
            "encoded_hidden_states": Tensor[N, sae_dim],
            "labels": Tensor[N],
        },
        # Or for flat files (adversarial):
        "all": {
            "encoded_hidden_states": Tensor[N, sae_dim],
            "labels": Tensor[N],
        },
    }

1-1 mapping guarantee: encoded_hidden_states[i] is the SAE encoding
of the original file's hidden_states[i]. No shuffling, no filtering.
To reconstruct: load both original and encoded file, index by position.

Usage:
    # Single model:
    python -m unsupervised_llm_safety.data.precompute_sae_encodings \\
        --model-key qwen2_1.5b

    # All models:
    python -m unsupervised_llm_safety.data.precompute_sae_encodings \\
        --model-key qwen2_1.5b ministral_8b llama3_8b

    # Force re-encode (overwrite existing):
    python -m unsupervised_llm_safety.data.precompute_sae_encodings \\
        --model-key qwen2_1.5b --force
"""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from unsupervised_llm_safety.data.adversarial_utils import (
    discover_adversarial_files,
)
from unsupervised_llm_safety.data.dataset_registry import (
    ROOT_DATA_PATH,
    get_adversarial_base_path,
    get_dataset_paths,
    set_complex_data_mode,
    set_last_token_mode,
)

logger = logging.getLogger(__name__)

# Root of the released SAE checkpoints. Set ``SAP_SAE_ROOT`` to the ``saes/``
# directory of the downloaded artifact bundle; unset, it is assumed to sit
# beside the activations under ``SAP_ACTIVATION_ROOT``.
#
# Note on scope: the released bundle ships SAEs for the three models Table 1
# covers, named ``<model_key>__<method>.pth`` (gpt_oss_20b needs two -- one per
# method -- so there is no single file per model). The frozen encoders for the
# three earlier models were never archived, so this table cannot point at
# them; supply them explicitly with ``SAP_SAE_<MODEL_KEY>`` or ``--sae-path``.
SAE_ROOT = os.environ.get("SAP_SAE_ROOT", f"{ROOT_DATA_PATH}/saes")

MODEL_KEYS = ["qwen2_1.5b", "ministral_8b", "llama3_8b"]


def _sae_path_for(model_key: str) -> str:
    """Resolve the frozen-encoder checkpoint for *model_key*.

    Precedence: ``SAP_SAE_<MODEL_KEY>`` (uppercased, dots -> underscores),
    then ``<SAP_SAE_ROOT>/<model_key>.pth``.
    """
    env_key = "SAP_SAE_" + model_key.upper().replace(".", "_")
    override = os.environ.get(env_key)
    if override:
        return override
    return f"{SAE_ROOT}/{model_key}.pth"


class _SaePathTable(Dict[str, str]):
    """Mapping view over :func:`_sae_path_for`, resolved on each lookup.

    Kept dict-shaped because call sites subscript it directly, but resolved
    lazily so that setting ``SAP_SAE_ROOT`` after import still takes effect.
    """

    def __getitem__(self, model_key: str) -> str:
        return _sae_path_for(model_key)

    def __missing__(self, model_key: str) -> str:  # pragma: no cover
        return _sae_path_for(model_key)

    def keys(self):  # type: ignore[override]
        return iter(MODEL_KEYS)


SAE_PATHS: Dict[str, str] = _SaePathTable()

# Where the offline SAE-encoding cache is written. Set ``SAP_SAE_ENCODED`` to
# relocate it; unset, it sits beside the activations.
OUTPUT_BASE = Path(
    os.environ.get("SAP_SAE_ENCODED", f"{ROOT_DATA_PATH}/sae_encoded")
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def compute_sae_hash(sae_path: str) -> str:
    """Quick hash of SAE checkpoint for cache validation."""
    stat = Path(sae_path).stat()
    raw = f"{sae_path}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def encode_hidden_states(
    encoder: torch.nn.Module,
    hidden_states: torch.Tensor,
    device: torch.device,
    batch_size: int = 512,
) -> torch.Tensor:
    """Encode hidden states through SAE encoder in batches.

    Args:
        encoder: SAE encoder module (frozen).
        hidden_states: Raw hidden states, shape (N, raw_dim).
        device: GPU/CPU device.
        batch_size: Batch size for encoding.

    Returns:
        Encoded features as float16 tensor, shape (N, sae_dim).
    """
    # Handle 3D tensors: (N, 1, D) -> (N, D)
    if hidden_states.ndim == 3 and hidden_states.shape[1] == 1:
        hidden_states = hidden_states.squeeze(1)

    encoded_chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, len(hidden_states), batch_size):
            batch = hidden_states[i : i + batch_size].float().to(device)
            z = encoder(batch)
            encoded_chunks.append(z.cpu().half())
    return torch.cat(encoded_chunks, dim=0)


def encode_dataset_file(
    encoder: torch.nn.Module,
    sae_path: str,
    file_path: str,
    output_path: Path,
    device: torch.device,
    batch_size: int = 512,
) -> Dict[str, Any]:
    """Encode a single dataset .pth file and save results.

    Returns:
        Summary dict with split names and sample counts.
    """
    data = torch.load(file_path, map_location="cpu", weights_only=False)

    result: Dict[str, Any] = {
        "sae_path": sae_path,
        "original_path": str(file_path),
    }

    summary: Dict[str, int] = {}

    # Handle three data layouts:
    # 1. {"train": {...}, "test": {...}} — standard datasets
    # 2. {"hidden_states": ..., "labels": ...} — flat (some adversarial)
    # 3. {"test": {...}} — test-only

    if "hidden_states" in data:
        # Flat structure — encode as "all" split
        hs = data["hidden_states"]
        labels = data.get("labels")
        encoded = encode_hidden_states(encoder, hs, device, batch_size)
        split_data: Dict[str, Any] = {
            "encoded_hidden_states": encoded,
        }
        if labels is not None:
            split_data["labels"] = labels
        result["all"] = split_data
        summary["all"] = len(encoded)
    else:
        # Standard train/test structure
        for split in ["train", "test"]:
            if split not in data:
                continue
            split_raw = data[split]
            hs = split_raw["hidden_states"]
            labels = split_raw.get("labels")
            encoded = encode_hidden_states(encoder, hs, device, batch_size)
            split_data = {"encoded_hidden_states": encoded}
            if labels is not None:
                split_data["labels"] = labels
            result[split] = split_data
            summary[split] = len(encoded)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)

    return summary


# ------------------------------------------------------------------
# Per-model processing
# ------------------------------------------------------------------


def collect_all_dataset_files(
    model_key: str,
) -> List[Dict[str, str]]:
    """Collect all dataset file paths for a model.

    Returns list of {"path": ..., "name": ..., "category": ...}.
    Deduplicates by file path.
    """
    set_last_token_mode(True)
    set_complex_data_mode(True)

    seen_paths: set = set()
    datasets: List[Dict[str, str]] = []

    def add(category: str, path: str, name: str) -> None:
        real = str(Path(path).resolve())
        if real in seen_paths:
            return
        seen_paths.add(real)
        datasets.append({"path": path, "name": name, "category": category})

    # IID datasets
    for ds in get_dataset_paths(model_key, "iid"):
        add("iid", ds["path"], ds["name"])

    # OOD datasets
    for ds in get_dataset_paths(model_key, "ood"):
        add("ood", ds["path"], ds["name"])

    # Adversarial datasets
    adv_base = get_adversarial_base_path(model_key)
    if adv_base:
        adv_files = discover_adversarial_files(model_key, base_path=adv_base)
        for attack_name, file_path in sorted(adv_files.items()):
            add("adversarial", str(file_path), attack_name)

    # Also pick up non-last-token and complex-data additions
    # (complex_data is already True, which adds BeaverTails_safe etc.)
    # Re-query without complex data to get the standard OOD entries too
    set_complex_data_mode(False)
    for ds in get_dataset_paths(model_key, "ood"):
        add("ood", ds["path"], ds["name"])
    set_complex_data_mode(True)

    return datasets


def process_model(
    model_key: str,
    device: torch.device,
    force: bool = False,
    batch_size: int = 512,
) -> None:
    """Encode all datasets for one model."""
    sae_path = SAE_PATHS[model_key]
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {model_key}")
    logger.info(f"SAE: {sae_path}")
    logger.info(f"{'='*60}")

    # Load SAE
    sae = torch.load(sae_path, map_location=device, weights_only=False)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad = False
    sae.to(device)

    encoder = sae.encoder
    sae_hash = compute_sae_hash(sae_path)
    sae_dim = sae.feature_dim

    logger.info(f"SAE hash: {sae_hash}, feature_dim: {sae_dim}")

    output_dir = OUTPUT_BASE / model_key

    # Collect all datasets
    all_datasets = collect_all_dataset_files(model_key)
    logger.info(f"Found {len(all_datasets)} datasets to encode")

    encoded_manifest: List[Dict[str, Any]] = []
    n_skipped = 0
    n_encoded = 0
    total_samples = 0

    for ds in all_datasets:
        category = ds["category"]
        name = ds["name"]
        file_path = ds["path"]
        output_path = output_dir / category / f"{name}.pth"

        if not Path(file_path).exists():
            logger.warning(f"  Source file not found: {file_path}")
            continue

        if output_path.exists() and not force:
            logger.info(f"  [skip] {category}/{name} (exists)")
            # Still add to manifest
            existing = torch.load(
                output_path, map_location="cpu", weights_only=False
            )
            n_samples = 0
            for split in ["train", "test", "all"]:
                if (
                    split in existing
                    and "encoded_hidden_states" in existing[split]
                ):
                    n_samples += len(existing[split]["encoded_hidden_states"])
            encoded_manifest.append(
                {
                    "category": category,
                    "name": name,
                    "path": str(output_path),
                    "original_path": file_path,
                    "n_samples": n_samples,
                }
            )
            n_skipped += 1
            total_samples += n_samples
            continue

        logger.info(f"  Encoding {category}/{name} ...")
        try:
            summary = encode_dataset_file(
                encoder,
                sae_path,
                file_path,
                output_path,
                device,
                batch_size,
            )
            n_samples = sum(summary.values())
            logger.info(
                f"    -> {', '.join(f'{k}={v}' for k, v in summary.items())}"
            )
            encoded_manifest.append(
                {
                    "category": category,
                    "name": name,
                    "path": str(output_path),
                    "original_path": file_path,
                    "n_samples": n_samples,
                }
            )
            n_encoded += 1
            total_samples += n_samples
        except Exception as e:
            logger.error(f"    FAILED: {e}")

    # Save metadata
    metadata = {
        "model_key": model_key,
        "sae_path": sae_path,
        "sae_hash": sae_hash,
        "sae_dim": sae_dim,
        "n_datasets": len(encoded_manifest),
        "total_samples": total_samples,
        "datasets": encoded_manifest,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        f"\nDone: {n_encoded} encoded, {n_skipped} skipped, "
        f"{total_samples} total samples"
    )
    logger.info(f"Output: {output_dir}")


# ------------------------------------------------------------------
# Loading utilities (for use by other modules)
# ------------------------------------------------------------------


def load_encoded_dataset(
    model_key: str,
    category: str,
    dataset_name: str,
    split: Optional[str] = None,
    output_base: Optional[Path] = None,
) -> Dict[str, torch.Tensor]:
    """Load pre-computed SAE encodings for a dataset.

    Args:
        model_key: Model identifier (e.g., "qwen2_1.5b").
        category: "iid", "ood", or "adversarial".
        dataset_name: Dataset name matching the encoded file.
        split: Optional split to load ("train", "test", "all").
            If None, returns all available splits.
        output_base: Base directory for encoded data. Defaults to
            OUTPUT_BASE.

    Returns:
        Dict with keys like "train", "test", or "all", each
        containing "encoded_hidden_states" and "labels" tensors.
    """
    base = output_base or OUTPUT_BASE
    path = base / model_key / category / f"{dataset_name}.pth"
    if not path.exists():
        raise FileNotFoundError(
            f"Encoded dataset not found: {path}. "
            f"Run precompute_sae_encodings first."
        )

    data = torch.load(path, map_location="cpu", weights_only=False)

    if split is not None:
        if split not in data:
            raise KeyError(
                f"Split '{split}' not found in {path}. "
                f"Available: {[k for k in data if k not in ('sae_path', 'original_path')]}"
            )
        return {split: data[split]}

    # Return all splits
    result = {}
    for key in ["train", "test", "all"]:
        if key in data and isinstance(data[key], dict):
            result[key] = data[key]
    return result


def load_encoded_metadata(
    model_key: str,
    output_base: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load metadata for a model's pre-computed encodings."""
    base = output_base or OUTPUT_BASE
    meta_path = base / model_key / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {meta_path}. "
            f"Run precompute_sae_encodings first."
        )
    with open(meta_path) as f:
        return json.load(f)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute SAE encodings for all datasets"
    )
    parser.add_argument(
        "--model-key",
        nargs="+",
        default=None,
        help="Model keys to process (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-encode even if cache exists",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for encoding (default: 512)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: cuda if available)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    models = args.model_key or MODEL_KEYS

    logger.info(f"Device: {device}")
    logger.info(f"Models: {models}")
    logger.info(f"Output base: {OUTPUT_BASE}")

    for model_key in models:
        process_model(
            model_key, device, force=args.force, batch_size=args.batch_size
        )

    logger.info("\nAll done.")


if __name__ == "__main__":
    main()
