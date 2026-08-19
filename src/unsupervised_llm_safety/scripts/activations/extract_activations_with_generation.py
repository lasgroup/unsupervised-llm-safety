#!/usr/bin/env python3
"""
Extract the last-token-of-response activation during model generation.

Generates a response per prompt while capturing per-token hidden states via
ActivationCapture hooks, then keeps only the final response token's activation
(one vector per sample) under the last-token-only policy. Used for generation
datasets (GSM8K, MathSFT40K, HumanEval, MBPP, HumanEval_Plus, Alpaca).
"""

import gc
import logging
from typing import Any, Dict, List, Tuple

import torch
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from unsupervised_llm_safety.common.device_utils import get_model_device
from unsupervised_llm_safety.data.dataset_processors import ActivationCapture

log = logging.getLogger(__name__)


# Note: make_json_serializable is now in extract_activations.py
# This function is imported from there when needed


def extract_activations_with_generation(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    labels: List[int],
    cfg: DictConfig,
    dataset_name: str = "humaneval",
) -> Tuple[torch.Tensor, List[int], Dict[str, Any]]:
    """Extract the last-token-of-response activation per task.

    Generates a response per prompt and stores only the final response
    token's activation (one vector + one label per task).

    Args:
        model: Language model for generation
        tokenizer: Tokenizer for the model
        texts: List of prompt texts (one per task)
        labels: List of labels (one per task)
        cfg: Configuration dict
        dataset_name: Name of dataset ("humaneval" or "mbpp")

    Returns:
        Tuple of (activations tensor [N_tasks, H], labels list, metadata dict)
    """
    all_activations = []
    all_labels = []
    device = get_model_device(model)

    # Enhanced metadata tracking
    metadata: Dict[str, Any] = {
        "model_config": {
            "model_id": cfg.model_key,
            "layer_index": cfg.extraction.layer_index,
            "max_new_tokens": cfg.extraction.get("max_new_tokens", 512),
            "dataset": dataset_name,
        },
        "extraction_stats": {
            "total_tasks": len(texts),
            "successful_tasks": 0,
            "failed_tasks": 0,
            "total_tokens_extracted": 0,
            "avg_tokens_per_task": 0.0,
            "min_tokens_per_task": float("inf"),
            "max_tokens_per_task": 0,
        },
        "task_mapping": [],  # [(task_idx, start_token_idx, end_token_idx), ...]
        "task_metadata": [],  # List of per-task info dicts
    }

    # Track current position in activation tensor
    current_token_idx = 0

    # Generation parameters
    max_new_tokens = cfg.extraction.get("max_new_tokens", 512)
    temperature = cfg.extraction.generation.get("temperature", 0.0)
    do_sample = cfg.extraction.generation.get("do_sample", temperature > 1e-4)

    for task_idx, text in tqdm(
        enumerate(texts), desc="Generating solutions", total=len(texts)
    ):
        try:
            # Apply chat template
            messages = [{"role": "user", "content": text}]
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                truncation=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Track prompt length for proper activation capture
            prompt_length = inputs["input_ids"].shape[1]

            # Create activation capture with robust tracking
            layer_name = f"model.layers.{cfg.extraction.layer_index}"
            capturer = ActivationCapture(
                cfg.extraction.layer_index, prompt_length
            )

            try:
                # Register hook
                capturer.register(model, layer_name)

                # Generate response
                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        do_sample=do_sample,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                # Extract generated text
                generated_ids = output_ids[0][len(inputs["input_ids"][0]) :]
                generated_text = tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )

                # Get captured activations
                task_activations = capturer.get_activations()

                if len(task_activations) > 0:
                    # Log token capture for debugging
                    num_generated_tokens = len(generated_ids)
                    num_captured_tokens = len(task_activations)
                    if cfg.get("verbose", False):
                        log.info(
                            f"Task {task_idx}: Generated {num_generated_tokens} tokens, "
                            f"captured {num_captured_tokens} activations"
                        )

                    # Keep only the last-token-of-response activation (one
                    # vector per task). task_activations[-1] is the final
                    # generated token, matching the end_idx-1 selection the
                    # (now removed) create_last_token_data reduce used.
                    last_token_activation = task_activations[-1:]  # [1, H]

                    # Record metadata (one stored row per task)
                    start_idx = current_token_idx
                    end_idx = current_token_idx + 1
                    metadata["task_mapping"].append(
                        [task_idx, start_idx, end_idx]
                    )

                    # Store task metadata with full text
                    task_meta = {
                        "task_idx": task_idx,
                        "task_id": f"{dataset_name}/{task_idx}",
                        "prompt": text,
                        "generated_solution": generated_text,
                        "num_tokens": 1,
                        "num_generated_tokens": num_generated_tokens,
                        "prompt_length": prompt_length,
                    }
                    metadata["task_metadata"].append(task_meta)

                    # Store one activation + one label per task
                    all_activations.append(last_token_activation)
                    task_label = (
                        labels[task_idx] if task_idx < len(labels) else 1
                    )
                    all_labels.append(task_label)

                    # Update tracking. Stats track the generated-response
                    # length (provenance); storage is one vector per task.
                    current_token_idx = end_idx
                    metadata["extraction_stats"]["successful_tasks"] += 1
                    metadata["extraction_stats"][
                        "total_tokens_extracted"
                    ] += num_generated_tokens

                    if (
                        num_generated_tokens
                        < metadata["extraction_stats"]["min_tokens_per_task"]
                    ):
                        metadata["extraction_stats"][
                            "min_tokens_per_task"
                        ] = num_generated_tokens
                    if (
                        num_generated_tokens
                        > metadata["extraction_stats"]["max_tokens_per_task"]
                    ):
                        metadata["extraction_stats"][
                            "max_tokens_per_task"
                        ] = num_generated_tokens
                else:
                    log.warning(f"No activations captured for task {task_idx}")
                    metadata["extraction_stats"]["failed_tasks"] += 1

            except Exception as e:
                log.error(f"Failed to generate for task {task_idx}: {e}")
                metadata["extraction_stats"]["failed_tasks"] += 1
            finally:
                capturer.remove()

            # Memory management for long runs
            if (task_idx + 1) % 50 == 0:  # Every 50 tasks
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        except Exception as e:
            log.error(f"Unexpected error processing task {task_idx}: {e}")
            metadata["extraction_stats"]["failed_tasks"] += 1

    # Calculate average tokens per task
    if metadata["extraction_stats"]["successful_tasks"] > 0:
        metadata["extraction_stats"]["avg_tokens_per_task"] = (
            metadata["extraction_stats"]["total_tokens_extracted"]
            / metadata["extraction_stats"]["successful_tasks"]
        )
    else:
        metadata["extraction_stats"]["avg_tokens_per_task"] = 0.0
        metadata["extraction_stats"]["min_tokens_per_task"] = 0

    # Convert min_tokens_per_task from inf to 0 if no successful tasks
    if metadata["extraction_stats"]["min_tokens_per_task"] == float("inf"):
        metadata["extraction_stats"]["min_tokens_per_task"] = 0

    # Concatenate all activations
    if all_activations:
        final_activations = torch.cat(all_activations, dim=0)
        log.info(
            f"Extracted {len(final_activations)} last-token activations "
            f"from {len(texts)} tasks"
        )
    else:
        raise ValueError("No activations were successfully extracted")

    return final_activations, all_labels, metadata
