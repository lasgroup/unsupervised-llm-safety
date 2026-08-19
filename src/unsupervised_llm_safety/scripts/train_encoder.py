import logging
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split

from unsupervised_llm_safety.common.evaluation import (
    evaluate_metrics,
    evaluate_multi_dataset,
    save_results_to_json,
)
from unsupervised_llm_safety.common.model_io import save_model
from unsupervised_llm_safety.common.train_utils import (
    create_model,
    set_seed,
    train_model,
)
from unsupervised_llm_safety.data.hidden_states_loader import load_datasets
from unsupervised_llm_safety.data.safety_data import HiddenStatesDataset

logger = logging.getLogger(__name__)


@hydra.main(
    version_base="1.2",
    config_path="../../../exp_configs",
    config_name="config",
)
def main(cfg: DictConfig):
    """Train an encoder (SAE or binary classifier) on hidden states."""

    # Extract domain from top-level config
    domain = None
    if hasattr(cfg, "domain") and cfg.domain is not None:
        domain = (
            cfg.domain.name if hasattr(cfg.domain, "name") else str(cfg.domain)
        )
        logger.info(f"Using domain: {domain}")

    # Data-mode switches -- applied in BOTH directions; see the note in
    # run_anodet_analysis. The registry globals default to last-token=True /
    # complex-data=True, so a True-only guard makes `false` silently inert.
    from unsupervised_llm_safety.data.dataset_registry import (
        set_complex_data_mode,
        set_generation_mode,
        set_last_token_mode,
    )

    last_token = bool(getattr(cfg, "use_last_token", False))
    generation = bool(getattr(cfg, "use_generation", False))
    complex_data = bool(getattr(cfg, "use_complex_data", False))

    set_last_token_mode(last_token)
    set_generation_mode(generation)
    set_complex_data_mode(complex_data)

    logger.info(
        "Data modes: last_token=%s generation=%s complex_data=%s",
        last_token,
        generation,
        complex_data,
    )

    # Extract the IID config group
    if hasattr(cfg, "iid") and cfg.iid is not None:
        iid_cfg = cfg.iid
    else:
        raise ValueError(
            "No 'iid' config group specified. Use: iid=encoder_training_config"
        )

    # --- 1. SETUP ---
    set_seed(iid_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(iid_cfg)}")

    # --- 2. LOAD DATA ---
    logger.info("Loading hidden states data...")

    from unsupervised_llm_safety.data.dataset_registry import (
        DEFAULT_DOMAIN,
        get_dataset_paths,
    )

    domain = domain or DEFAULT_DOMAIN
    logger.info(f"Loading datasets for domain: {domain}")

    iid_datasets = get_dataset_paths(iid_cfg.model_key, "iid", domain=domain)
    dataset_paths = [ds["path"] for ds in iid_datasets]
    # Group datasets by filter type for per-dataset filtering
    paths_no_filter = []
    paths_safe_only = []
    for ds in iid_datasets:
        filt = ds.get("filter")
        if filt == "safe_only":
            paths_safe_only.append(ds["path"])
        else:
            paths_no_filter.append(ds["path"])

    logger.info(
        f"IID datasets: {len(paths_no_filter)} unfiltered, "
        f"{len(paths_safe_only)} safe_only"
    )

    train_ratio = iid_cfg.datasets.iid.get("train_ratio", 1.0)
    test_ratio = iid_cfg.datasets.iid.get("test_ratio", 1.0)

    # Load unfiltered datasets
    data = load_datasets(
        source=paths_no_filter,
        split="train",
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        shuffle=True,
        seed=iid_cfg.seed,
    )

    # Load safe_only datasets and merge if present
    if paths_safe_only:
        safe_data = load_datasets(
            source=paths_safe_only,
            split="train",
            train_ratio=train_ratio,
            test_ratio=test_ratio,
            filter_safe_only=True,
            shuffle=True,
            seed=iid_cfg.seed,
        )
        # Merge train splits
        data["train"]["hidden_states"] = torch.cat(
            [
                data["train"]["hidden_states"],
                safe_data["train"]["hidden_states"],
            ],
            dim=0,
        )
        data["train"]["labels"] = torch.cat(
            [data["train"]["labels"], safe_data["train"]["labels"]],
            dim=0,
        )
        logger.info(
            f"Merged {len(safe_data['train']['labels'])} safe_only "
            f"samples into training data"
        )

    # Build dataset
    num_samples = len(data["train"]["labels"])
    data_sources = data["train"].get(
        "data_source", [f"source_{i}" for i in range(num_samples)]
    )

    full_dataset = HiddenStatesDataset(
        {
            "hidden_states": data["train"]["hidden_states"],
            "labels": data["train"]["labels"],
            "data_source": data_sources,
        }
    )
    logger.info(f"Loaded dataset with {len(full_dataset)} samples")

    # --- 3. CREATE MODEL ---
    hs_data = {"s1": data["train"]}
    model = create_model(iid_cfg, hs_data)

    # --- 4. SPLIT AND TRAIN ---
    train_split_ratio = iid_cfg.training.train_split_ratio
    n_total = len(full_dataset)
    n_train = int(n_total * train_split_ratio)
    n_holdout = n_total - n_train

    logger.info(f"Splitting: {n_train} train (S1), {n_holdout} holdout (S2)")

    s1_dataset, s2_dataset = random_split(
        full_dataset,
        [n_train, n_holdout],
        generator=torch.Generator().manual_seed(iid_cfg.seed),
    )

    train_loader = DataLoader(
        s1_dataset, batch_size=iid_cfg.batch_size, shuffle=True
    )
    holdout_loader = DataLoader(
        s2_dataset, batch_size=iid_cfg.batch_size, shuffle=False
    )

    logger.info("Starting training...")
    trained_model = train_model(
        cfg=iid_cfg,
        model=model,
        train_dataloader=train_loader,
        device=device,
    )

    # --- 5. EVALUATE ON HOLDOUT ---
    logger.info("Evaluating on holdout set S2...")
    metrics_s2 = evaluate_metrics(
        model=trained_model,
        dataloader=holdout_loader,
        device=device,
    )

    logger.info(f"Holdout error: {1.0 - metrics_s2['accuracy']:.4f}")
    logger.info(f"FPR: {metrics_s2['false_positive_rate']:.4f}")
    logger.info(f"FNR: {metrics_s2['false_negative_rate']:.4f}")
    logger.info(f"Precision: {metrics_s2['precision']:.4f}")
    logger.info(f"Recall: {metrics_s2['recall']:.4f}")
    logger.info(f"F1: {metrics_s2['f1_score']:.4f}")

    results = {
        "holdout_error": 1.0 - metrics_s2["accuracy"],
        "holdout_fpr": metrics_s2["false_positive_rate"],
        "holdout_fnr": metrics_s2["false_negative_rate"],
        "holdout_precision": metrics_s2["precision"],
        "holdout_recall": metrics_s2["recall"],
        "holdout_f1_score": metrics_s2["f1_score"],
        "n_train": n_train,
        "n_holdout": n_holdout,
    }

    # --- 6. MULTI-DATASET EVALUATION (optional) ---
    all_iid_paths = paths_no_filter + paths_safe_only
    if len(all_iid_paths) > 1:
        logger.info("Performing multi-dataset evaluation...")
        multi_dataset_results = evaluate_multi_dataset(
            model=trained_model,
            dataset_paths=all_iid_paths,
            device=device,
            batch_size=iid_cfg.batch_size,
        )
        results["multi_dataset_evaluation"] = multi_dataset_results

        for name, metrics in multi_dataset_results.items():
            logger.info(
                f"  {name}: Accuracy={metrics['accuracy']:.4f}, "
                f"Precision={metrics['precision']:.4f}, "
                f"Recall={metrics['recall']:.4f}"
            )

    # --- 7. SAVE ---
    model_path = os.path.join(os.getcwd(), "trained_model.pth")
    save_model(trained_model, model_path)
    logger.info(f"Trained model saved to: {model_path}")

    save_results_to_json(results, os.path.join(os.getcwd(), "results.json"))


if __name__ == "__main__":
    main()
