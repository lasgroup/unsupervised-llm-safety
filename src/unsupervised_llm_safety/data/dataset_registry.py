"""Central registry for dataset paths.

This module provides a centralized location for all dataset paths used in
CCS confidence analysis, organized by model and dataset type.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Repository root, derived from this file's location:
#   <repo>/src/unsupervised_llm_safety/data/dataset_registry.py
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Global root data path for most datasets.
#
# Set ``ULS_ACTIVATION_ROOT`` to the directory holding the downloaded
# activation bundle. Unset, it defaults to ``<repo>/activation_data``, which
# is the default location for cached activations. Every
# dataset path in this module is derived from this one root, so redirecting
# it moves the whole registry.
ROOT_DATA_PATH = os.environ.get(
    "ULS_ACTIVATION_ROOT", str(_REPO_ROOT / "activation_data")
)

# Default location of the HarmBench adversarial activation files.
DEFAULT_HARMBENCH_BASE_PATH = f"{ROOT_DATA_PATH}/harmbench/"

# The four earliest models' BeaverTails hidden states were produced by a
# sibling project and live outside ROOT_DATA_PATH, under a
# ``<root>/<model-slug>/all_hidden_states*.pth`` layout. Set
# ``ULS_BEAVERTAILS_HS_ROOT`` to point at that copy; unset, it is assumed to
# have been placed under the activation root alongside everything else.
BEAVER_TAILS_HS_ROOT = os.environ.get(
    "ULS_BEAVERTAILS_HS_ROOT", f"{ROOT_DATA_PATH}/beaver_tails_hs"
)


def get_harmbench_base_path() -> str:
    """Return the directory holding HarmBench activation files.

    Set ``ULS_HARMBENCH_BASE`` to redirect BOTH the adversarial eval
    slice and the HarmBench-safe rows that ``use_complex_data`` folds
    into IID training at a different copy of the data (for example a
    corrected-label copy on scratch). Unset, this is exactly the
    historical hard-coded path, so behaviour is unchanged.
    """
    return os.environ.get("ULS_HARMBENCH_BASE", DEFAULT_HARMBENCH_BASE_PATH)


def get_beaver_tails_complex_override(model_key: str) -> Optional[str]:
    """Return a replacement path for the BeaverTails-safe IID slice.

    Set ``ULS_BEAVERTAILS_COMPLEX_OVERRIDE`` to a comma separated list of
    ``model_key:/abs/path`` entries to point the ``use_complex_data``
    BeaverTails-safe rows at a different copy of the file (for example a
    text-level deduplicated copy on scratch). Only the complex-data IID
    slice is affected; the OOD/eval BeaverTails entry is untouched.
    Unset, this returns ``None`` and behaviour is unchanged.
    """
    env = os.environ.get("ULS_BEAVERTAILS_COMPLEX_OVERRIDE", "")
    if not env:
        return None
    for spec in env.split(","):
        spec = spec.strip()
        if ":" not in spec:
            continue
        mk, path = spec.split(":", 1)
        if mk.strip() == model_key:
            return path.strip()
    return None


# Default domain
DEFAULT_DOMAIN = "common_sense_safety"

DATASET_PATHS: Dict[str, Dict[str, Any]] = {
    "qwen2_1.5b": {
        "iid": [
            {
                "path": f"{ROOT_DATA_PATH}/gsm8k/gsm8k_qwen2_1.5b_layer20_full.pth",
                "name": "GSM8K",
            },
            {
                "path": f"{ROOT_DATA_PATH}/math_sft_40k/math_sft_40k_qwen2_1.5b_layer20.pth",
                "name": "MathSFT40K",
            },
            {
                "path": f"{ROOT_DATA_PATH}/humaneval/humaneval_qwen2_1.5b_layer20.pth",
                "name": "HumanEval",
            },
            {
                "path": f"{ROOT_DATA_PATH}/mbpp/mbpp_qwen2_1.5b_layer20.pth",
                "name": "MBPP",
            },
            {
                "path": f"{ROOT_DATA_PATH}/alpaca/alpaca_qwen2_1.5b_layer20_full.pth",
                "name": "Alpaca",
            },
            {
                "path": f"{ROOT_DATA_PATH}/mt_bench/mt_bench_qwen2_1.5b_layer20_full.pth",
                "name": "MTBENCH",
            },
            {
                "path": f"{ROOT_DATA_PATH}/mmlu/mmlu_qwen2_1.5b_layer20_full.pth",
                "name": "MMLU",
            },
            {
                "path": f"{ROOT_DATA_PATH}/humaneval_plus/humaneval_plus_qwen2_1.5b_layer20.pth",
                "name": "HumanEval_Plus",
            },
            {
                "path": f"{ROOT_DATA_PATH}/arc/arc_qwen2_1.5b_layer20.pth",
                "name": "ARC",
            },
        ],
        "ood": [
            {
                "path": f"{BEAVER_TAILS_HS_ROOT}/qwen2-1.5b/all_hidden_states.pth",
                "name": "BeaverTails",
                "filter": "unsafe_only",
            },
            {
                "path": f"{ROOT_DATA_PATH}/toxigen/toxigen_qwen2_1.5b_layer20_full.pth",
                "name": "ToxiGen",
                "filter": "unsafe_only",
            },
        ],
        "adversarial_base_path": f"{ROOT_DATA_PATH}/harmbench/",
    },
    "ministral_8b": {
        "iid": [
            {
                "path": f"{ROOT_DATA_PATH}/gsm8k/gsm8k_ministral_8b_layer20_full.pth",
                "name": "GSM8K",
            },
            {
                "path": f"{ROOT_DATA_PATH}/math_sft_40k/math_sft_40k_ministral_8b_layer20.pth",
                "name": "MathSFT40K",
            },
            {
                "path": f"{ROOT_DATA_PATH}/humaneval/humaneval_ministral_8b_layer20.pth",
                "name": "HumanEval",
            },
            {
                "path": f"{ROOT_DATA_PATH}/mbpp/mbpp_ministral_8b_layer20.pth",
                "name": "MBPP",
            },
            {
                "path": f"{ROOT_DATA_PATH}/alpaca/alpaca_ministral_8b_layer20_full.pth",
                "name": "Alpaca",
            },
            {
                "path": f"{ROOT_DATA_PATH}/mt_bench/mt_bench_ministral_8b_layer20_full.pth",
                "name": "MTBENCH",
            },
            {
                "path": f"{ROOT_DATA_PATH}/mmlu/mmlu_ministral_8b_layer20_full.pth",
                "name": "MMLU",
            },
            {
                "path": f"{ROOT_DATA_PATH}/humaneval_plus/humaneval_plus_ministral_8b_layer20.pth",
                "name": "HumanEval_Plus",
            },
            {
                "path": f"{ROOT_DATA_PATH}/arc/arc_ministral_8b_layer20.pth",
                "name": "ARC",
            },
        ],
        "ood": [
            {
                "path": f"{BEAVER_TAILS_HS_ROOT}/ministral-8b/all_hidden_states.pth",
                "name": "BeaverTails",
                "filter": "unsafe_only",
            },
            {
                "path": f"{ROOT_DATA_PATH}/toxigen/toxigen_ministral_8b_layer20_full.pth",
                "name": "ToxiGen",
                "filter": "unsafe_only",
            },
        ],
        "adversarial_base_path": f"{ROOT_DATA_PATH}/harmbench/",
    },
    "llama3_8b": {
        "iid": [
            {
                "path": f"{ROOT_DATA_PATH}/gsm8k/gsm8k_llama3_8b_layer25_full.pth",
                "name": "GSM8K",
            },
            {
                "path": f"{ROOT_DATA_PATH}/math_sft_40k/math_sft_40k_llama3_8b_layer25.pth",
                "name": "MathSFT40K",
            },
            {
                "path": f"{ROOT_DATA_PATH}/humaneval/humaneval_llama3_8b_layer25.pth",
                "name": "HumanEval",
            },
            {
                "path": f"{ROOT_DATA_PATH}/mbpp/mbpp_llama3_8b_layer25.pth",
                "name": "MBPP",
            },
            {
                "path": f"{ROOT_DATA_PATH}/alpaca/alpaca_llama3_8b_layer25_full.pth",
                "name": "Alpaca",
            },
            {
                "path": f"{ROOT_DATA_PATH}/mt_bench/mt_bench_llama3_8b_layer25_full.pth",
                "name": "MTBENCH",
            },
            {
                "path": f"{ROOT_DATA_PATH}/mmlu/mmlu_llama3_8b_layer25_full.pth",
                "name": "MMLU",
            },
            {
                "path": f"{ROOT_DATA_PATH}/humaneval_plus/humaneval_plus_llama3_8b_layer25.pth",
                "name": "HumanEval_Plus",
            },
            {
                "path": f"{ROOT_DATA_PATH}/arc/arc_llama3_8b_layer25.pth",
                "name": "ARC",
            },
        ],
        "ood": [
            {
                "path": f"{BEAVER_TAILS_HS_ROOT}/llama3-8b/all_hidden_states.pth",
                "name": "BeaverTails",
                "filter": "unsafe_only",
            },
            {
                "path": f"{ROOT_DATA_PATH}/toxigen/toxigen_llama3_8b_layer25_full.pth",
                "name": "ToxiGen",
                "filter": "unsafe_only",
            },
        ],
        "adversarial_base_path": f"{ROOT_DATA_PATH}/harmbench/",
    },
    "qwen3_8b": {
        "iid": [
            {
                "path": f"{ROOT_DATA_PATH}/gsm8k/gsm8k_qwen3_8b_layer30_full.pth",
                "name": "GSM8K",
            },
            {
                "path": f"{ROOT_DATA_PATH}/math_sft_40k/math_sft_40k_qwen3_8b_layer30.pth",
                "name": "MathSFT40K",
            },
            {
                "path": f"{ROOT_DATA_PATH}/humaneval/humaneval_qwen3_8b_layer30.pth",
                "name": "HumanEval",
            },
            {
                "path": f"{ROOT_DATA_PATH}/mbpp/mbpp_qwen3_8b_layer30.pth",
                "name": "MBPP",
            },
            {
                "path": f"{ROOT_DATA_PATH}/alpaca/alpaca_qwen3_8b_layer30_full.pth",
                "name": "Alpaca",
            },
            {
                "path": f"{ROOT_DATA_PATH}/mt_bench/mt_bench_qwen3_8b_layer30_full.pth",
                "name": "MTBENCH",
            },
            {
                "path": f"{ROOT_DATA_PATH}/mmlu/mmlu_qwen3_8b_layer30_full.pth",
                "name": "MMLU",
            },
            {
                "path": f"{ROOT_DATA_PATH}/humaneval_plus/humaneval_plus_qwen3_8b_layer30.pth",
                "name": "HumanEval_Plus",
            },
            {
                "path": f"{ROOT_DATA_PATH}/arc/arc_qwen3_8b_layer30.pth",
                "name": "ARC",
            },
        ],
        "ood": [
            {
                "path": f"{BEAVER_TAILS_HS_ROOT}/qwen3-8b/all_hidden_states.pth",
                "name": "BeaverTails",
                "filter": "unsafe_only",
            },
            {
                "path": f"{ROOT_DATA_PATH}/toxigen/toxigen_qwen3_8b_layer30_full.pth",
                "name": "ToxiGen",
                "filter": "unsafe_only",
            },
        ],
        "adversarial_base_path": f"{ROOT_DATA_PATH}/harmbench/",
    },
    # gpt-oss-20b (4th model). Layer 20, width 2880. All extracted on Clariden.
    # Protocol per dataset: 1-8 + ToxiGen = chat-template prompt-only last-token;
    # MT-Bench = 2-turn generation -> _full -> _last_token; BeaverTails =
    # plain-text forward-pass hook on model.layers[20] (no off-by-one). See
    "gpt_oss_20b": {
        "iid": [
            {
                "path": f"{ROOT_DATA_PATH}/gsm8k/gsm8k_gpt_oss_20b_layer20_full.pth",
                "name": "GSM8K",
            },
            {
                "path": f"{ROOT_DATA_PATH}/math_sft_40k/math_sft_40k_gpt_oss_20b_layer20.pth",
                "name": "MathSFT40K",
            },
            {
                "path": f"{ROOT_DATA_PATH}/humaneval/humaneval_gpt_oss_20b_layer20.pth",
                "name": "HumanEval",
            },
            {
                "path": f"{ROOT_DATA_PATH}/mbpp/mbpp_gpt_oss_20b_layer20.pth",
                "name": "MBPP",
            },
            {
                "path": f"{ROOT_DATA_PATH}/alpaca/alpaca_gpt_oss_20b_layer20_full.pth",
                "name": "Alpaca",
            },
            {
                "path": f"{ROOT_DATA_PATH}/mt_bench/mt_bench_gpt_oss_20b_layer20_full.pth",
                "name": "MTBENCH",
            },
            {
                "path": f"{ROOT_DATA_PATH}/mmlu/mmlu_gpt_oss_20b_layer20_full.pth",
                "name": "MMLU",
            },
            {
                "path": f"{ROOT_DATA_PATH}/humaneval_plus/humaneval_plus_gpt_oss_20b_layer20.pth",
                "name": "HumanEval_Plus",
            },
            {
                "path": f"{ROOT_DATA_PATH}/arc/arc_gpt_oss_20b_layer20.pth",
                "name": "ARC",
            },
        ],
        "ood": [
            {
                "path": f"{ROOT_DATA_PATH}/beaver_tails/beaver_tails_gpt_oss_20b_layer20.pth",
                "name": "BeaverTails",
                "filter": "unsafe_only",
            },
            {
                "path": f"{ROOT_DATA_PATH}/toxigen/toxigen_gpt_oss_20b_layer20_full.pth",
                "name": "ToxiGen",
                "filter": "unsafe_only",
            },
        ],
        "adversarial_base_path": f"{ROOT_DATA_PATH}/harmbench/",
    },
    # gemma4-26b (5th model). Layer 25, width 2816, bf16 (+ f32 for BeaverTails /
    # MT-Bench last_token). Extracted on Clariden, same dataset surface + protocol
    # as gpt-oss-20b. BeaverTails is the full-mixed forward pass (`_with_safe` ==
    # base, verified 2026-07-19), not the per-category balanced merge the original
    # 3 models use — AUROC is ratio-invariant so only the OOD category mix differs.
    "gemma4_26b": {
        "iid": [
            {
                "path": f"{ROOT_DATA_PATH}/gsm8k/gsm8k_gemma4_26b_layer25_full.pth",
                "name": "GSM8K",
            },
            {
                "path": f"{ROOT_DATA_PATH}/math_sft_40k/math_sft_40k_gemma4_26b_layer25.pth",
                "name": "MathSFT40K",
            },
            {
                "path": f"{ROOT_DATA_PATH}/humaneval/humaneval_gemma4_26b_layer25.pth",
                "name": "HumanEval",
            },
            {
                "path": f"{ROOT_DATA_PATH}/mbpp/mbpp_gemma4_26b_layer25.pth",
                "name": "MBPP",
            },
            {
                "path": f"{ROOT_DATA_PATH}/alpaca/alpaca_gemma4_26b_layer25_full.pth",
                "name": "Alpaca",
            },
            {
                "path": f"{ROOT_DATA_PATH}/mt_bench/mt_bench_gemma4_26b_layer25_full.pth",
                "name": "MTBENCH",
            },
            {
                "path": f"{ROOT_DATA_PATH}/mmlu/mmlu_gemma4_26b_layer25_full.pth",
                "name": "MMLU",
            },
            {
                "path": f"{ROOT_DATA_PATH}/humaneval_plus/humaneval_plus_gemma4_26b_layer25.pth",
                "name": "HumanEval_Plus",
            },
            {
                "path": f"{ROOT_DATA_PATH}/arc/arc_gemma4_26b_layer25.pth",
                "name": "ARC",
            },
        ],
        "ood": [
            {
                "path": f"{ROOT_DATA_PATH}/beaver_tails/beaver_tails_gemma4_26b_layer25.pth",
                "name": "BeaverTails",
                "filter": "unsafe_only",
            },
            {
                "path": f"{ROOT_DATA_PATH}/toxigen/toxigen_gemma4_26b_layer25_full.pth",
                "name": "ToxiGen",
                "filter": "unsafe_only",
            },
        ],
        "adversarial_base_path": f"{ROOT_DATA_PATH}/harmbench/",
    },
}

BENIGN_GENERALIZATION_PATHS = {
    "qwen2_1.5b": [
        {
            "path": f"{ROOT_DATA_PATH}/mbpp_plus/mbpp_plus_qwen2_1.5b_layer20.pth",
            "name": "MBPP_Plus_benign_gen",
        },
        {
            "path": f"{ROOT_DATA_PATH}/hellaswag/hellaswag_qwen2_1.5b_layer20.pth",
            "name": "HellaSwag_benign_gen",
        },
    ],
    "ministral_8b": [
        {
            "path": f"{ROOT_DATA_PATH}/mbpp_plus/mbpp_plus_ministral_8b_layer20.pth",
            "name": "MBPP_Plus_benign_gen",
        },
        {
            "path": f"{ROOT_DATA_PATH}/hellaswag/hellaswag_ministral_8b_layer20.pth",
            "name": "HellaSwag_benign_gen",
        },
    ],
    "llama3_8b": [
        {
            "path": f"{ROOT_DATA_PATH}/mbpp_plus/mbpp_plus_llama3_8b_layer25.pth",
            "name": "MBPP_Plus_benign_gen",
        },
        {
            "path": f"{ROOT_DATA_PATH}/hellaswag/hellaswag_llama3_8b_layer25.pth",
            "name": "HellaSwag_benign_gen",
        },
    ],
}

# =============================================================================
# BENIGN OUT-OF-DISTRIBUTION DATASETS (for benign-OOD calibration, exp E1)
# =============================================================================
# Novel *harmless* domains that are OOD relative to the in-distribution safe
# pool. These are labelled safe (y=0) but tracked as a separate group ``B`` so
# calibration can suppress false positives on them while preserving unsafe
# detection. Distinct from BENIGN_GENERALIZATION_PATHS (which per-slice code
# folds into the IID pool): here HH-RLHF-helpful is registered alongside
# HellaSwag as an explicit benign-OOD source. Only models with BOTH domains
# cached are listed (qwen3_8b, gpt_oss_20b and gemma4_26b added 2026-07-30
# after the Clariden extraction).
# =============================================================================
BENIGN_OOD_PATHS = {
    "qwen2_1.5b": [
        {
            "path": f"{ROOT_DATA_PATH}/hellaswag/"
            "hellaswag_qwen2_1.5b_layer20.pth",
            "name": "HellaSwag",
        },
        {
            "path": f"{ROOT_DATA_PATH}/hh_rlhf_helpful/"
            "hh_rlhf_helpful_qwen2_1.5b_layer20.pth",
            "name": "HH-RLHF-helpful",
        },
    ],
    "ministral_8b": [
        {
            "path": f"{ROOT_DATA_PATH}/hellaswag/"
            "hellaswag_ministral_8b_layer20.pth",
            "name": "HellaSwag",
        },
        {
            "path": f"{ROOT_DATA_PATH}/hh_rlhf_helpful/"
            "hh_rlhf_helpful_ministral_8b_layer20.pth",
            "name": "HH-RLHF-helpful",
        },
    ],
    "llama3_8b": [
        {
            "path": f"{ROOT_DATA_PATH}/hellaswag/"
            "hellaswag_llama3_8b_layer25.pth",
            "name": "HellaSwag",
        },
        {
            "path": f"{ROOT_DATA_PATH}/hh_rlhf_helpful/"
            "hh_rlhf_helpful_llama3_8b_layer25.pth",
            "name": "HH-RLHF-helpful",
        },
    ],
    "qwen3_8b": [
        {
            "path": f"{ROOT_DATA_PATH}/hellaswag/"
            "hellaswag_qwen3_8b_layer30.pth",
            "name": "HellaSwag",
        },
        {
            "path": f"{ROOT_DATA_PATH}/hh_rlhf_helpful/"
            "hh_rlhf_helpful_qwen3_8b_layer30.pth",
            "name": "HH-RLHF-helpful",
        },
    ],
    "gpt_oss_20b": [
        {
            "path": f"{ROOT_DATA_PATH}/hellaswag/"
            "hellaswag_gpt_oss_20b_layer20.pth",
            "name": "HellaSwag",
        },
        {
            "path": f"{ROOT_DATA_PATH}/hh_rlhf_helpful/"
            "hh_rlhf_helpful_gpt_oss_20b_layer20.pth",
            "name": "HH-RLHF-helpful",
        },
    ],
    "gemma4_26b": [
        {
            "path": f"{ROOT_DATA_PATH}/hellaswag/"
            "hellaswag_gemma4_26b_layer25.pth",
            "name": "HellaSwag",
        },
        {
            "path": f"{ROOT_DATA_PATH}/hh_rlhf_helpful/"
            "hh_rlhf_helpful_gemma4_26b_layer25.pth",
            "name": "HH-RLHF-helpful",
        },
    ],
}


# =============================================================================
# LAST-TOKEN-PER-TURN OVERRIDES
# =============================================================================
# When use_last_token=True, MT-Bench paths are swapped to _last_token versions.
# Other IID datasets are already per-sample and don't need modification.
# =============================================================================

LAST_TOKEN_MTBENCH_PATHS = {
    "qwen2_1.5b": f"{ROOT_DATA_PATH}/mt_bench/mt_bench_qwen2_1.5b_layer20_last_token.pth",
    "ministral_8b": f"{ROOT_DATA_PATH}/mt_bench/mt_bench_ministral_8b_layer20_last_token.pth",
    "llama3_8b": f"{ROOT_DATA_PATH}/mt_bench/mt_bench_llama3_8b_layer25_last_token.pth",
    "qwen3_8b": f"{ROOT_DATA_PATH}/mt_bench/mt_bench_qwen3_8b_layer30_last_token.pth",
    "gpt_oss_20b": f"{ROOT_DATA_PATH}/mt_bench/mt_bench_gpt_oss_20b_layer20_last_token.pth",
    "gemma4_26b": f"{ROOT_DATA_PATH}/mt_bench/mt_bench_gemma4_26b_layer25_last_token.pth",
}

# =============================================================================
# GENERATION TOKEN OVERRIDES
# =============================================================================
# When use_generation=True, IID paths for Alpaca, GSM8K, HumanEval, MBPP are
# swapped to _generation.pth versions (per-token generation data).
# MT-Bench _full.pth is already generation data, so it stays as-is.
# MMLU, ARC have no generation data (classification tasks), so they stay as-is.
# =============================================================================

# Datasets that have generation versions and are swapped when use_generation=True.
# Start small: only HumanEval generation (326M) to avoid OOM.
# MT-Bench _full.pth is already generation data (stays as-is).
# Alpaca (2.7G), GSM8K (3.6G) are too large to load together — scale up later.
GENERATION_DATASETS = {"HumanEval"}

# Layer mapping per model (needed to construct generation paths)
_MODEL_LAYERS = {
    "qwen2_1.5b": 20,
    "ministral_8b": 20,
    "llama3_8b": 25,
    "qwen3_8b": 30,
    "gpt_oss_20b": 20,
    "gemma4_26b": 25,
}


def _get_generation_path(dataset_name: str, model_key: str) -> Optional[str]:
    """Return the generation .pth path for a dataset+model, or None."""
    layer = _MODEL_LAYERS.get(model_key)
    if layer is None or dataset_name not in GENERATION_DATASETS:
        return None
    name_lower = dataset_name.lower()
    return (
        f"{ROOT_DATA_PATH}/{name_lower}/"
        f"{name_lower}_{model_key}_layer{layer}_generation.pth"
    )


# Global flags (last_token and complex_data are on by default)
_USE_LAST_TOKEN = True
_USE_GENERATION = False
_USE_COMPLEX_DATA = True


def set_last_token_mode(enabled: bool = True) -> None:
    """Enable or disable last-token-per-turn mode globally.

    When enabled, MT-Bench paths are replaced with _last_token versions.
    """
    global _USE_LAST_TOKEN
    _USE_LAST_TOKEN = enabled


def get_last_token_mode() -> bool:
    """Return whether last-token mode is enabled."""
    return _USE_LAST_TOKEN


def set_generation_mode(enabled: bool = True) -> None:
    """Enable or disable generation-token mode globally.

    When enabled, IID paths for Alpaca, GSM8K, HumanEval, MBPP are
    replaced with _generation.pth versions.
    """
    global _USE_GENERATION
    _USE_GENERATION = enabled


def get_generation_mode() -> bool:
    """Return whether generation mode is enabled."""
    return _USE_GENERATION


def set_complex_data_mode(enabled: bool = True) -> None:
    """Enable or disable complex-data mode globally.

    When enabled, safe portions of OOD datasets (BeaverTails, ToxiGen)
    and safe HarmBench adversarial data are added to the IID dataset list.
    This makes the anomaly detection problem harder by contaminating the
    IID training distribution with safe samples from OOD/ADV sources.
    """
    global _USE_COMPLEX_DATA
    _USE_COMPLEX_DATA = enabled


def get_complex_data_mode() -> bool:
    """Return whether complex-data mode is enabled."""
    return _USE_COMPLEX_DATA


# ── Layer override (E5 layer-depth sweep) ─────────────────────────
# The registry hardcodes each model's *published* residual-stream layer
# in the path f-strings. The layer-depth sweep (experiment E5) needs the
# same datasets extracted at a *different* layer. Rather than duplicate
# the whole registry per layer, an opt-in override rewrites the layer
# token (``_layer{canonical}`` -> ``_layer{L}``) in the paths returned by
# ``get_dataset_paths`` and filters HarmBench discovery to that layer.
# It is set through the ``ULS_LAYER_OVERRIDE`` env var
# (``"model_key:layer"``, comma-separated for multiple) so it also
# reaches SLURM subprocesses launched by the E5 driver.
#
# BeaverTails needs its own resolution rule. For the original models its
# canonical path comes from the external safety_polytope project with the
# layer baked in and no ``_layer{N}`` token to rewrite. The multi-layer
# extraction, however, wrote per-layer BeaverTails files under
# ``activation_data/beaver_tails/`` using the standard naming, so under an
# override we redirect to those instead. A BeaverTails slice is dropped
# only when the per-layer file is genuinely absent, which keeps the layer
# sweep on the same three-slice evaluation as the main results. With no
# override (or override == canonical) behaviour is byte-for-byte identical
# to before.
_CANONICAL_LAYER: Dict[str, int] = {
    "qwen2_1.5b": 20,
    "ministral_8b": 20,
    "llama3_8b": 25,
    "qwen3_8b": 30,
    "gpt_oss_20b": 20,
    "gemma4_26b": 25,
}


def get_active_layer(model_key: str) -> Optional[int]:
    """Return the layer the registry resolves paths at for a model.

    Precedence: ``ULS_LAYER_OVERRIDE`` env var > canonical layer.
    Returns ``None`` for unknown models.
    """
    env = os.environ.get("ULS_LAYER_OVERRIDE", "")
    if env:
        for spec in env.split(","):
            spec = spec.strip()
            if ":" in spec:
                mk, lyr = spec.rsplit(":", 1)
                if (
                    mk.strip() == model_key
                    and lyr.strip().lstrip("-").isdigit()
                ):
                    return int(lyr.strip())
    return _CANONICAL_LAYER.get(model_key)


def _beaver_tails_layer_path(model_key: str, layer: int, path: str) -> str:
    """Per-layer BeaverTails file written by the multi-layer extraction.

    ``path`` is the canonical entry, used only to tell the mixed file from
    its ``_with_safe`` (safe-only complex-data) variant.
    """
    suffix = "_with_safe" if path.endswith("_with_safe.pth") else ""
    return (
        f"{ROOT_DATA_PATH}/beaver_tails/"
        f"beaver_tails_{model_key}_layer{layer}{suffix}.pth"
    )


def _apply_layer_override(
    model_key: str, datasets: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Rewrite the layer token in dataset paths for a non-canonical layer.

    BeaverTails is redirected to its per-layer file; it is dropped only
    when that file does not exist. A no-op when the active layer equals
    the canonical layer.
    """
    canonical = _CANONICAL_LAYER.get(model_key)
    target = get_active_layer(model_key)
    if canonical is None or target is None or target == canonical:
        return datasets
    old = f"_layer{canonical}"
    new = f"_layer{target}"
    out: List[Dict[str, Any]] = []
    for ds in datasets:
        if str(ds.get("name", "")).startswith("BeaverTails"):
            candidate = _beaver_tails_layer_path(model_key, target, ds["path"])
            if not os.path.exists(candidate):
                # No per-layer file -> excluded from the layer sweep.
                continue
            out.append({**ds, "path": candidate})
            continue
        out.append({**ds, "path": ds["path"].replace(old, new)})
    return out


# ── Complex-data: safe OOD/ADV paths appended to IID ──────────────
# BeaverTails: qwen/ministral use merged file, llama3 uses original
COMPLEX_DATA_BEAVER_TAILS = {
    "qwen2_1.5b": {
        "path": f"{BEAVER_TAILS_HS_ROOT}/qwen2-1.5b/all_hidden_states_with_safe.pth",
        "name": "BeaverTails_safe",
        "filter": "safe_only",
    },
    "ministral_8b": {
        "path": f"{BEAVER_TAILS_HS_ROOT}/ministral-8b/all_hidden_states_with_safe.pth",
        "name": "BeaverTails_safe",
        "filter": "safe_only",
    },
    "llama3_8b": {
        "path": f"{BEAVER_TAILS_HS_ROOT}/llama3-8b/all_hidden_states_with_safe.pth",
        "name": "BeaverTails_safe",
        "filter": "safe_only",
    },
    "qwen3_8b": {
        "path": f"{BEAVER_TAILS_HS_ROOT}/qwen3-8b/all_hidden_states_with_safe.pth",
        "name": "BeaverTails_safe",
        "filter": "safe_only",
    },
    # Produced on Clariden by the hook-based extractor (model.layers[20]); the
    # `_with_safe` variant is the safe-only complex-data split.
    "gpt_oss_20b": {
        "path": f"{ROOT_DATA_PATH}/beaver_tails/beaver_tails_gpt_oss_20b_layer20_with_safe.pth",
        "name": "BeaverTails_safe",
        "filter": "safe_only",
    },
    # gemma4_26b `_with_safe` is the full-mixed file (== base), safe_only filter
    # selects the safe split. Same protocol as gpt_oss_20b.
    "gemma4_26b": {
        "path": f"{ROOT_DATA_PATH}/beaver_tails/beaver_tails_gemma4_26b_layer25_with_safe.pth",
        "name": "BeaverTails_safe",
        "filter": "safe_only",
    },
}

# ToxiGen: same files as OOD, different filter
COMPLEX_DATA_TOXIGEN = {
    "qwen2_1.5b": {
        "path": f"{ROOT_DATA_PATH}/toxigen/toxigen_qwen2_1.5b_layer20_full.pth",
        "name": "ToxiGen_safe",
        "filter": "safe_only",
    },
    "ministral_8b": {
        "path": f"{ROOT_DATA_PATH}/toxigen/toxigen_ministral_8b_layer20_full.pth",
        "name": "ToxiGen_safe",
        "filter": "safe_only",
    },
    "llama3_8b": {
        "path": f"{ROOT_DATA_PATH}/toxigen/toxigen_llama3_8b_layer25_full.pth",
        "name": "ToxiGen_safe",
        "filter": "safe_only",
    },
    "qwen3_8b": {
        "path": f"{ROOT_DATA_PATH}/toxigen/toxigen_qwen3_8b_layer30_full.pth",
        "name": "ToxiGen_safe",
        "filter": "safe_only",
    },
    "gpt_oss_20b": {
        "path": f"{ROOT_DATA_PATH}/toxigen/toxigen_gpt_oss_20b_layer20_full.pth",
        "name": "ToxiGen_safe",
        "filter": "safe_only",
    },
    "gemma4_26b": {
        "path": f"{ROOT_DATA_PATH}/toxigen/toxigen_gemma4_26b_layer25_full.pth",
        "name": "ToxiGen_safe",
        "filter": "safe_only",
    },
}


def _get_harmbench_safe_paths(model_key: str) -> List[Dict[str, Any]]:
    """Discover relabeled HarmBench generation_last_token files for IID use.

    Returns dataset configs with filter: "safe_only" so only refusal
    samples (label=1) are loaded during IID training/eval.
    """
    from unsupervised_llm_safety.data.adversarial_utils import (
        discover_adversarial_files,
    )

    # Filter HarmBench discovery to the active layer so a multi-layer
    # harmbench/ directory (populated by the E5 sweep) does not mix
    # layers into the IID-safe pool. With no override this is the
    # canonical layer -> identical to the previous behaviour.
    adv_files = discover_adversarial_files(
        model_key=model_key, layer=get_active_layer(model_key)
    )
    safe_entries = []
    for attack_name, file_path in adv_files.items():
        safe_entries.append(
            {
                "path": str(file_path),
                "name": f"HarmBench-{attack_name}_safe",
                "filter": "safe_only",
            }
        )
    return safe_entries


def get_dataset_paths(
    model_key: str,
    dataset_type: str,
    domain: str = DEFAULT_DOMAIN,
    use_last_token: Optional[bool] = None,
    use_generation: Optional[bool] = None,
    use_complex_data: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Get dataset paths for a specific model, dataset type, and domain.

    Args:
        model_key: The key identifying the model (e.g., 'qwen2_1.5b')
        dataset_type: Type of dataset ('iid', 'ood', or 'adversarial')
        domain: The domain to use (kept for backward compatibility)
        use_last_token: If True, swap MT-Bench paths to _last_token versions.
            If None, uses the global _USE_LAST_TOKEN flag.
        use_generation: If True, swap IID paths for Alpaca/GSM8K/HumanEval/MBPP
            to _generation.pth versions. If None, uses the global _USE_GENERATION flag.
        use_complex_data: If True, append safe OOD/ADV datasets to IID list.
            If None, uses the global _USE_COMPLEX_DATA flag.

    Returns:
        List of dataset configurations with 'path', 'name', and optionally 'filter' keys

    Raises:
        ValueError: If model_key is not found in the registry
    """
    dataset_registry = DATASET_PATHS

    if model_key not in dataset_registry:
        available_models = ", ".join(sorted(dataset_registry.keys()))
        raise ValueError(
            f"Unknown model '{model_key}'. "
            f"Available models: {available_models}"
        )

    if dataset_type not in ["iid", "ood", "adversarial"]:
        raise ValueError(
            f"Invalid dataset_type: {dataset_type}. Must be 'iid', 'ood', or 'adversarial'"
        )

    # For adversarial, we return empty list as they are auto-discovered
    if dataset_type == "adversarial":
        return []

    # A copy, never the live list: the complex-data block below appends to
    # this in place, and handing out the registry's own list would mutate
    # DATASET_PATHS for every later call in the same process.
    datasets = list(dataset_registry[model_key].get(dataset_type, []))

    # Apply last-token override for MT-Bench if enabled
    last_token = (
        use_last_token if use_last_token is not None else _USE_LAST_TOKEN
    )
    if (
        last_token
        and dataset_type == "iid"
        and model_key in LAST_TOKEN_MTBENCH_PATHS
    ):
        datasets = [
            (
                {**ds, "path": LAST_TOKEN_MTBENCH_PATHS[model_key]}
                if ds["name"] == "MTBENCH"
                else ds
            )
            for ds in datasets
        ]

    # Apply generation-token override for IID datasets if enabled
    # Keep only: datasets in GENERATION_DATASETS (swapped to _generation.pth),
    # plus MTBENCH (already generation), MMLU, ARC, HumanEval_Plus (input-prompt).
    # Exclude large input-prompt datasets that would dilute generation signal.
    generation = (
        use_generation if use_generation is not None else _USE_GENERATION
    )
    if generation and dataset_type == "iid":
        keep_as_is = {"MTBENCH", "MMLU", "ARC", "HumanEval_Plus"}
        new_datasets = []
        for ds in datasets:
            gen_path = _get_generation_path(ds["name"], model_key)
            if gen_path is not None:
                new_datasets.append({**ds, "path": gen_path})
            elif ds["name"] in keep_as_is:
                new_datasets.append(ds)
            # else: skip (large input-prompt datasets)
        datasets = new_datasets

    # Append safe OOD/ADV datasets to IID when complex-data mode is on
    complex_data = (
        use_complex_data if use_complex_data is not None else _USE_COMPLEX_DATA
    )
    if complex_data and dataset_type == "iid":
        if model_key in COMPLEX_DATA_BEAVER_TAILS:
            bt_entry = COMPLEX_DATA_BEAVER_TAILS[model_key]
            # Honour the ULS_BEAVERTAILS_COMPLEX_OVERRIDE redirect
            # (no-op when unset).
            bt_override = get_beaver_tails_complex_override(model_key)
            if bt_override:
                bt_entry = {**bt_entry, "path": bt_override}
            datasets.append(bt_entry)
        if model_key in COMPLEX_DATA_TOXIGEN:
            datasets.append(COMPLEX_DATA_TOXIGEN[model_key])
        # HarmBench safe data: discover relabeled generation_last_token files
        harmbench_safe = _get_harmbench_safe_paths(model_key)
        datasets.extend(harmbench_safe)

    # E5 layer-depth sweep: rewrite the layer token in the resolved paths
    # (and drop non-layer-swappable BeaverTails) when a non-canonical
    # layer override is active. No-op otherwise.
    datasets = _apply_layer_override(model_key, datasets)

    return datasets


def get_adversarial_base_path(
    model_key: str,
    domain: str = DEFAULT_DOMAIN,
) -> Optional[str]:
    """Get the base path for adversarial datasets for a specific model.

    Args:
        model_key: The key identifying the model
        domain: Kept for backward compatibility

    Returns:
        Base path for adversarial datasets, or None if not found
    """
    if model_key not in DATASET_PATHS:
        return None

    configured = DATASET_PATHS[model_key].get("adversarial_base_path", None)
    if configured is None:
        return None
    # Honour the ULS_HARMBENCH_BASE redirect (no-op when unset).
    if configured == DEFAULT_HARMBENCH_BASE_PATH:
        return get_harmbench_base_path()
    return configured


def list_available_models(domain: str = DEFAULT_DOMAIN) -> List[str]:
    """Return a list of available model keys in the dataset registry.

    Args:
        domain: Kept for backward compatibility

    Returns:
        List of available model keys
    """
    return sorted(DATASET_PATHS.keys())


def get_benign_generalization_datasets(
    model_key: str,
    domain: str = DEFAULT_DOMAIN,
) -> List[Dict[str, str]]:
    """Get benign generalization datasets for a specific model.

    Args:
        model_key: The key identifying the model
        domain: Kept for backward compatibility

    Returns:
        List of benign generalization dataset configurations
    """
    return BENIGN_GENERALIZATION_PATHS.get(model_key, [])


def get_benign_ood_datasets(
    model_key: str,
) -> List[Dict[str, str]]:
    """Get benign out-of-distribution datasets for a specific model.

    These are novel harmless domains (HellaSwag, HH-RLHF-helpful) used as the
    benign-OOD group ``B`` in benign-OOD calibration (exp E1). Returns an empty
    list for models without cached benign-OOD activations.

    Args:
        model_key: The key identifying the model

    Returns:
        List of benign-OOD dataset configurations (``path``, ``name``)
    """
    return BENIGN_OOD_PATHS.get(model_key, [])


# Dataset categories for semantic grouping in GMM clustering
DATASET_CATEGORIES = {
    # Math datasets
    "GSM8K": "math",
    "MathSFT40K": "math",
    # Coding datasets
    "HumanEval": "coding",
    "MBPP": "coding",
    "HumanEval_Plus": "coding",
    "HumanEval_Plus": "coding",
    # Knowledge/reasoning datasets
    "MMLU": "knowledge",
    "ARC": "knowledge",
    # Instruction following datasets
    "Alpaca": "instruction",
    "MTBENCH": "instruction",
    # Knowledge/reasoning datasets (benign gen)
    "HellaSwag_benign_gen": "knowledge",
    "MBPP_Plus_benign_gen": "coding",
    # Safety/harmful content datasets
    "BeaverTails": "safety_harmful",
    "ToxiGen": "safety_harmful",
}

# Map adversarial attacks to a single category
ADVERSARIAL_CATEGORY_PATTERN = "HarmBench-"


def get_dataset_category(dataset_name: str) -> str:
    """Get semantic category for a dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Category string, or "uncategorized" if not found
    """
    # Check explicit categories first
    if dataset_name in DATASET_CATEGORIES:
        return DATASET_CATEGORIES[dataset_name]

    # Check if it's an adversarial dataset
    if dataset_name.startswith(ADVERSARIAL_CATEGORY_PATTERN):
        return "adversarial"

    # Default fallback
    return "uncategorized"


def get_unique_categories() -> List[str]:
    """Get list of all unique categories."""
    categories = set(DATASET_CATEGORIES.values())
    categories.add("adversarial")
    categories.add("uncategorized")
    return sorted(categories)
