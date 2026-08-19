import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from unsupervised_llm_safety.common.device_utils import get_model_device


def evaluate_metrics(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = None,
) -> Dict[str, Any]:
    """
    Compute evaluation metrics for a safety model.

    Args:
        model: PyTorch model returning ModelOutput with .is_safe
        dataloader: DataLoader containing (inputs, labels) pairs

    Returns:
        Dictionary with accuracy, precision, recall, f1_score,
        false_positive_rate, false_negative_rate
    """
    if device is None:
        device = get_model_device(model)

    model.eval()
    model.to(device)
    total = 0
    correct = 0
    false_negatives = 0
    false_positives = 0
    true_positives = 0
    true_negatives = 0

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                data, target, _ = batch
            else:
                data, target = batch

            data, target = data.to(device), target.to(device)

            outputs = model(data)
            predictions = outputs.is_safe.float().to(target.device)

            correct += (predictions == target).sum().item()
            total += len(target)

            true_positives += ((predictions == 1) & (target == 1)).sum().item()
            true_negatives += ((predictions == 0) & (target == 0)).sum().item()
            false_positives += (
                ((predictions == 1) & (target == 0)).sum().item()
            )
            false_negatives += (
                ((predictions == 0) & (target == 1)).sum().item()
            )

    accuracy = correct / total if total > 0 else 0.0
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    fpr = (
        false_positives / (false_positives + true_negatives)
        if (false_positives + true_negatives) > 0
        else 0.0
    )
    fnr = (
        false_negatives / (false_negatives + true_positives)
        if (false_negatives + true_positives) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
    }


def evaluate_multi_dataset(
    model: nn.Module,
    dataset_paths: List[str],
    device: torch.device = None,
    batch_size: int = 64,
    dataset_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate a model on multiple datasets separately.

    Args:
        model: PyTorch model to evaluate
        dataset_paths: List of paths to dataset files (.pth format)
        device: Device to use for evaluation
        batch_size: Batch size for evaluation
        dataset_names: Optional names for datasets

    Returns:
        Dictionary mapping dataset names to their evaluation metrics
    """
    from unsupervised_llm_safety.data.safety_data import HiddenStatesDataset

    if device is None:
        device = get_model_device(model)

    if dataset_names is None:
        dataset_names = [
            os.path.basename(path).replace(".pth", "")
            for path in dataset_paths
        ]

    results = {}

    for dataset_path, dataset_name in zip(dataset_paths, dataset_names):
        print(f"Evaluating on dataset: {dataset_name}")

        data = torch.load(dataset_path, weights_only=False)

        if "test" in data:
            eval_data = data["test"]
        elif "train" in data:
            eval_data = data["train"]
        else:
            eval_data = data

        num_samples = len(eval_data["labels"])
        dataset_data = {
            "hidden_states": eval_data["hidden_states"],
            "labels": eval_data["labels"],
            "data_source": [f"source_{i}" for i in range(num_samples)],
        }

        dataset = HiddenStatesDataset(dataset_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        metrics = evaluate_metrics(
            model=model,
            dataloader=dataloader,
            device=device,
        )

        metrics["dataset_name"] = dataset_name
        metrics["dataset_path"] = dataset_path
        metrics["num_samples"] = len(dataset)

        results[dataset_name] = metrics

        print(
            f"  {dataset_name}: Accuracy={metrics['accuracy']:.4f}, "
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"FPR={metrics['false_positive_rate']:.4f}, "
            f"FNR={metrics['false_negative_rate']:.4f}"
        )

    return results


def save_results_to_json(results: Dict[str, float], output_path: str) -> None:
    """Save evaluation results to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            serializable_results[key] = value.tolist()
        else:
            serializable_results[key] = value

    with open(output_path, "w") as f:
        json.dump(serializable_results, f, indent=4)
    print(f"Results saved to {output_path}")
