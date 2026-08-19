"""Central registry for model configurations.

This module provides a centralized location for all model-specific configurations,
including HuggingFace model names, short names, layer indices, and other
model-specific settings.
"""

from typing import Any, Dict

MODEL_CONFIGS = {
    "qwen2_1.5b": {
        "hf_name": "Qwen/Qwen2-1.5B-Instruct",
        "short_name": "qwen2-1.5b",
        "harmbench_name": "qwen_1.5b",
        "default_layer": 20,
        "trust_remote_code": True,
        "dtype": "float16",
    },
    "ministral_8b": {
        "hf_name": "mistralai/Ministral-8B-Instruct-2410",
        "short_name": "ministral_8b",
        "harmbench_name": "mistral_8b",  # HarmBench uses "mistral_8b" for Mistral models
        "default_layer": 20,
        "trust_remote_code": False,
        "dtype": "float16",
    },
    "llama3_8b": {
        "hf_name": "meta-llama/Llama-3.1-8B-Instruct",
        "short_name": "llama3_8b",
        "harmbench_name": "llama3_8b",
        "default_layer": 25,
        "trust_remote_code": False,
        "dtype": "float16",
    },
    "gpt_oss_20b": {
        "hf_name": "openai/gpt-oss-20b",
        "short_name": "gpt_oss_20b",
        "harmbench_name": "gpt_oss_20b",
        "default_layer": 20,  # of 24 blocks (~83%, matches Qwen3-8B's 30/36)
        "trust_remote_code": True,
        # bf16: container has no MXFP4/triton kernels, so the MoE weights are
        # dequantized to bf16 (~40 GB, fits one GH200); also required for the
        # white-box gradient attacks' backward pass.
        "dtype": "bfloat16",
        # gpt-oss uses the harmony format (analysis/final channels) with no true
        # "thinking off". reasoning_effort=low is the closest analog to the other
        # models' no-CoT setup. But low effort STILL opens an `analysis` channel,
        # so the last-token rep can land mid-analysis (the same contamination that
        # broke Qwen3's <think> reps) and the refusal is often decided there.
        # `assistant_prefill` forces the prompt into the `final` channel so the
        # captured rep is the final-answer last token — and matches the
        # final-channel-prefilled prompt the HarmBench attacks/completions use
        # (baselines/model_utils.py get_template "gpt_oss"), keeping attack,
        # completion, and extraction prompts identical.
        "chat_template_kwargs": {"reasoning_effort": "low"},
        "assistant_prefill": "<|channel|>final<|message|>",
    },
    "qwen3_8b": {
        "hf_name": "Qwen/Qwen3-8B",
        "short_name": "qwen3_8b",
        "harmbench_name": "qwen3_8b",
        "default_layer": 30,
        "trust_remote_code": True,
        "dtype": "float16",
    },
    # gemma4-26b (5th model). Layer 25, residual width 2816, bf16. Reps were
    # pre-extracted on Clariden; the model is NEVER loaded during replication
    # (we score saved representations), so `hf_name` matters only if you
    # re-extract.
    "gemma4_26b": {
        "hf_name": "google/gemma-4-26B-A4B-it",
        "short_name": "gemma4_26b",
        "harmbench_name": "gemma4_26b",
        "default_layer": 25,
        "trust_remote_code": False,
        "dtype": "bfloat16",
    },
}


def get_model_config(model_key: str) -> Dict[str, Any]:
    """Get model configuration by key.

    Args:
        model_key: The key identifying the model in the registry.

    Returns:
        Dictionary containing model configuration.

    Raises:
        ValueError: If the model_key is not found in the registry.
    """
    if model_key not in MODEL_CONFIGS:
        available_models = ", ".join(sorted(MODEL_CONFIGS.keys()))
        raise ValueError(
            f"Unknown model: {model_key}. Available models: {available_models}"
        )
    return MODEL_CONFIGS[model_key].copy()


def list_available_models() -> list[str]:
    """Return a list of available model keys."""
    return sorted(MODEL_CONFIGS.keys())


def get_model_by_hf_name(hf_name: str) -> tuple[str, Dict[str, Any]] | None:
    """Find a model configuration by its HuggingFace name.

    Args:
        hf_name: The HuggingFace model name to search for.

    Returns:
        Tuple of (model_key, config_dict) if found, None otherwise.
    """
    for model_key, config in MODEL_CONFIGS.items():
        if config["hf_name"] == hf_name:
            return model_key, config.copy()
    return None
