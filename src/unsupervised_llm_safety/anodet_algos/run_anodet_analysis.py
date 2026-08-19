#!/usr/bin/env python3
"""Run baseline anomaly detection: train (or load) a model, then evaluate."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from unsupervised_llm_safety.anodet_algos.anodet_utils import (
    BASELINE_CLASSES,
    initialize_anodet_method,
    prepare_data_for_baseline,
    resolve_dataset_configurations,
)
from unsupervised_llm_safety.anodet_algos.run_anodet_eval import (
    evaluate_anodet,
)
from unsupervised_llm_safety.common.device_utils import log_memory_usage
from unsupervised_llm_safety.common.tuned_defaults import apply_tuned_defaults
from unsupervised_llm_safety.data.hidden_states_loader import load_datasets

logger = logging.getLogger(__name__)

_BANNER_WIDTH = 80


def _log_banner(step: str, title: str, detail: str = "") -> None:
    """Log a visually distinct section banner."""
    logger.info("")
    logger.info("=" * _BANNER_WIDTH)
    logger.info(f"[{step}] {title}")
    if detail:
        logger.info(f"  {detail}")
    logger.info("=" * _BANNER_WIDTH)


def run_anodet_analysis(
    cfg: DictConfig,
    pretrained_model: Optional[nn.Module] = None,
    full_model: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """Run anomaly detection analysis: train/load a baseline, then evaluate.

    Args:
        cfg: Configuration for the anomaly detection analysis
        pretrained_model: Optional pretrained model (encoder) for methods
        full_model: Optional full autoencoder (encoder+decoder) for methods
            like denoised_mahalanobis that need reconstruction

    Returns:
        Dictionary with analysis results
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_device = torch.device("cpu")
    output_dir = Path(os.getcwd())

    # Set global random seed from method config (ensures reproducible
    # data loading, holdout splits, etc. even for methods that don't
    # set their own seed internally like Mahalanobis/GMM).
    method_cfg = cfg.get(cfg.method, {})
    seed = method_cfg.get("seed", 42) if method_cfg else 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    _log_banner(
        "STEP 1/4",
        "SETUP",
        f"method={cfg.method} | device={device} | output={output_dir}",
    )
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    log_memory_usage("Start")

    # Resolve dataset configurations
    resolve_dataset_configurations(cfg)

    holdout_ratio = cfg.get("holdout_ratio", 0.2)
    use_threshold_evaluation = cfg.get("use_threshold_evaluation", True)

    # Create encoder wrapper if pretrained model is provided
    encoder_wrapper = _create_encoder_wrapper(pretrained_model)

    # Train or load baseline
    if cfg.get("load_from"):
        logger.info(f"Loading pre-trained {cfg.method} from {cfg.load_from}")
        baseline_class = BASELINE_CLASSES[cfg.method]
        baseline = baseline_class.load(cfg.load_from, device=str(device))
        logger.info(f"Loaded {baseline.name} baseline from {cfg.load_from}")
    else:
        baseline = _train_baseline(
            cfg,
            pretrained_model,
            encoder_wrapper,
            data_device,
            holdout_ratio,
            use_threshold_evaluation,
            output_dir,
            full_model=full_model,
        )

    # Evaluate
    _log_banner("STEP 4/4", "EVALUATION", f"{baseline.name}")
    results = evaluate_anodet(baseline, cfg, encoder_wrapper, output_dir)

    # Post-eval diagnostics for learned_mask and learned_mask_lora
    if cfg.method in ("learned_mask", "learned_mask_lora"):
        _run_learned_mask_diagnostics(baseline, cfg, output_dir)

    return results


def _create_encoder_wrapper(
    pretrained_model: Optional[nn.Module],
) -> Optional[nn.Module]:
    """Create an encoder wrapper from a pretrained model if provided."""
    if pretrained_model is None:
        return None

    if hasattr(pretrained_model, "transform_numpy"):
        return pretrained_model

    class ModelEncoderWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.device = next(model.parameters()).device

        def forward(self, x):
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float().to(self.device)
            with torch.no_grad():
                features = self.model.extract_features(x)
            return features

        def transform_numpy(self, x_numpy):
            features = self(x_numpy)
            return features.cpu().numpy()

    wrapper = ModelEncoderWrapper(pretrained_model)
    logger.info("Created encoder wrapper for pretrained model")
    return wrapper


def _train_baseline(
    cfg: DictConfig,
    pretrained_model: Optional[nn.Module],
    encoder_wrapper: Optional[nn.Module],
    data_device: torch.device,
    holdout_ratio: float,
    use_threshold_evaluation: bool,
    output_dir: Path,
    full_model: Optional[nn.Module] = None,
) -> Any:
    """Initialize, train, and save a baseline model."""
    baseline = initialize_anodet_method(
        cfg, pretrained_model, full_model=full_model
    )
    logger.info(f"Initialized {baseline.name} anomaly detection algorithm")

    # Load training data
    _log_banner(
        "STEP 2/4",
        "LOADING TRAINING DATA",
        f"{len(cfg.datasets.iid)} IID datasets"
        f" | holdout={holdout_ratio:.0%}"
        f" | model={cfg.get('model_key', 'N/A')}",
    )

    method_requires_both_labels = cfg.method == "bce"
    max_iid_samples = cfg.get("max_iid_samples", 0)
    effective_holdout = holdout_ratio if use_threshold_evaluation else 0.0

    # Group IID datasets by filter type to handle per-dataset filtering
    paths_no_filter = []
    paths_safe_only = []
    paths_unsafe_only = []
    for iid_cfg in cfg.datasets.iid:
        filt = getattr(iid_cfg, "filter", None)
        if method_requires_both_labels or filt is None:
            paths_no_filter.append(iid_cfg.path)
        elif filt == "safe_only":
            paths_safe_only.append(iid_cfg.path)
        elif filt == "unsafe_only":
            paths_unsafe_only.append(iid_cfg.path)
        else:
            paths_no_filter.append(iid_cfg.path)

    logger.info(
        f"IID datasets: {len(paths_no_filter)} unfiltered, "
        f"{len(paths_safe_only)} safe_only, "
        f"{len(paths_unsafe_only)} unsafe_only"
    )

    # Load each group and concatenate
    all_train_data = []
    all_train_labels = []
    all_data_source = []

    if paths_no_filter:
        data, labels, dsrc = _load_training_data(
            paths_no_filter,
            data_device,
            filter_safe_only=False,
            filter_unsafe_only=False,
            holdout_ratio=effective_holdout,
            max_iid_samples=max_iid_samples,
        )
        all_train_data.append(data)
        all_train_labels.append(labels)
        if dsrc is not None:
            all_data_source.extend(dsrc)
        logger.info(f"  Loaded {len(data)} samples from unfiltered IID")

    if paths_safe_only:
        data, labels, dsrc = _load_training_data(
            paths_safe_only,
            data_device,
            filter_safe_only=True,
            filter_unsafe_only=False,
            holdout_ratio=effective_holdout,
            max_iid_samples=0,
        )
        all_train_data.append(data)
        all_train_labels.append(labels)
        if dsrc is not None:
            all_data_source.extend(dsrc)
        logger.info(f"  Loaded {len(data)} samples from safe_only IID")

    if paths_unsafe_only:
        data, labels, dsrc = _load_training_data(
            paths_unsafe_only,
            data_device,
            filter_safe_only=False,
            filter_unsafe_only=True,
            holdout_ratio=effective_holdout,
            max_iid_samples=0,
        )
        all_train_data.append(data)
        all_train_labels.append(labels)
        if dsrc is not None:
            all_data_source.extend(dsrc)
        logger.info(f"  Loaded {len(data)} samples from unsafe_only IID")

    train_data = np.concatenate(all_train_data, axis=0)
    train_labels = np.concatenate(all_train_labels, axis=0)
    train_data_source = all_data_source if all_data_source else None

    logger.info(
        f"  Total: {len(train_data):,} training samples (dim={train_data.shape[1]})"
        f" from {len(cfg.datasets.iid)} IID datasets"
        f" ({len(paths_no_filter)} unfiltered"
        f" + {len(paths_safe_only)} safe_only"
        f" + {len(paths_unsafe_only)} unsafe_only)"
    )

    # Transform data if needed
    train_data_processed = prepare_data_for_baseline(
        train_data, cfg.method, encoder_wrapper, cfg
    )

    # Fit baseline
    _log_banner(
        "STEP 3/4",
        "TRAINING",
        f"{baseline.name} | {len(train_data_processed):,} samples"
        f" | dim={train_data_processed.shape[1]}",
    )
    baseline.fit(train_data_processed, train_labels)
    log_memory_usage("After training")

    # Save if requested
    if cfg.get("save_model", False):
        save_path = output_dir / f"{baseline.name}_trained.pt"
        baseline.save(save_path)
        logger.info(f"Saved trained model to {save_path}")
        baseline.post_training_callback(
            save_path,
            cfg,
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        # Save cluster info alongside the model for post-hoc analysis.
        # Maps each training sample to its cluster and source dataset.
        cluster_labels = getattr(baseline, "cluster_labels", None)
        if cluster_labels is None:
            # learned_mask stores it under _diagnostic_cluster_labels
            cluster_labels = getattr(
                baseline, "_diagnostic_cluster_labels", None
            )
        if cluster_labels is not None:
            if isinstance(cluster_labels, torch.Tensor):
                cluster_labels = cluster_labels.cpu().numpy()
            cluster_info = {
                "cluster_labels": cluster_labels,
                "data_source": train_data_source,
                "train_labels": train_labels,
                "n_clusters": int(cluster_labels.max() + 1),
                "model_key": cfg.get("model_key"),
            }
            info_path = output_dir / "cluster_info.pt"
            torch.save(cluster_info, info_path)
            logger.info(
                f"Saved cluster_info ({len(cluster_labels)} samples, "
                f"{cluster_info['n_clusters']} clusters) to {info_path}"
            )

    return baseline


def _run_learned_mask_diagnostics(
    baseline: Any,
    cfg: DictConfig,
    output_dir: Path,
) -> None:
    """Run post-hoc diagnostics for learned_mask after evaluation.

    Loads eval data, computes identity vs masked residuals, and
    generates diagnostic plots + JSON. Results go to
    {output_dir}/diagnostics/post_hoc/.
    """
    try:
        # `scripts` is reachable as a namespace package when this entry point
        # is invoked from the repository root (which is how the example
        # drivers in scripts/ invoke it). `mod` is bound here as well as in
        # the fallback below because step B reads
        # `mod.encode_and_assign_clusters` off it; binding only the seven
        # names would leave `mod` undefined on this path.
        from scripts.anodet_algos import diagnose_learned_mask as mod
        from scripts.anodet_algos.diagnose_learned_mask import (
            diagnose_dimension_variance,
            diagnose_hard_easy_cases,
            diagnose_identity_auroc,
            diagnose_mask_sparsity,
            diagnose_residual_magnitudes,
            diagnose_routing,
            print_meta_summary,
        )
    except ImportError:
        # Fallback: load it from its path in the repository. parents[3] is the
        # repository root (<repo>/src/unsupervised_llm_safety/anodet_algos/ ->
        # <repo>); the `.exists()` guard matters because
        # spec_from_file_location happily returns a spec for a file that is
        # not there, and exec_module would then raise FileNotFoundError out of
        # this function rather than degrading to the warning below.
        import importlib.util

        diag_path = (
            Path(__file__).parents[3]
            / "scripts"
            / "anodet_algos"
            / "diagnose_learned_mask.py"
        )
        spec = (
            importlib.util.spec_from_file_location(
                "diagnose_learned_mask", diag_path
            )
            if diag_path.exists()
            else None
        )
        if spec is None or spec.loader is None:
            logger.warning(
                "Could not import diagnose_learned_mask. "
                "Skipping post-hoc diagnostics."
            )
            return
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        diagnose_mask_sparsity = mod.diagnose_mask_sparsity
        diagnose_residual_magnitudes = mod.diagnose_residual_magnitudes
        diagnose_identity_auroc = mod.diagnose_identity_auroc
        diagnose_dimension_variance = mod.diagnose_dimension_variance
        diagnose_routing = mod.diagnose_routing
        diagnose_hard_easy_cases = mod.diagnose_hard_easy_cases
        print_meta_summary = mod.print_meta_summary

    _log_banner("STEP 5/5", "POST-HOC DIAGNOSTICS", "learned_mask")

    model_key = cfg.get("model_key")
    if not model_key:
        logger.warning(
            "model_key not set in config. " "Skipping post-hoc diagnostics."
        )
        return

    out_dir = output_dir / "diagnostics" / "post_hoc"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # A. Mask sparsity (no eval data needed)
        logger.info("=== A. Mask sparsity ===")
        sparsity_summary = diagnose_mask_sparsity(baseline, out_dir)

        # Load eval data (one-time cost)
        logger.info("Loading eval data for diagnostics...")
        from unsupervised_llm_safety.analysis.cluster_data import (
            load_eval_data_with_sources,
        )

        eval_data, eval_labels, eval_sources, dataset_names = (
            load_eval_data_with_sources(model_key)
        )

        # Encode once and assign clusters (shared by B-F)
        encode_and_assign_clusters = mod.encode_and_assign_clusters
        logger.info("Encoding eval data and assigning clusters...")
        encoded, clusters = encode_and_assign_clusters(baseline, eval_data)

        # B. Residual magnitudes
        logger.info("=== B. Residual magnitudes ===")
        residual_summary = diagnose_residual_magnitudes(
            baseline,
            eval_data,
            eval_sources,
            dataset_names,
            encoded,
            clusters,
            out_dir,
        )

        # C. Identity-mask AUROC ablation
        logger.info("=== C. Identity-mask AUROC ablation ===")
        auroc_summary = diagnose_identity_auroc(
            baseline,
            eval_data,
            eval_sources,
            dataset_names,
            encoded,
            output_dir,
            out_dir,
        )

        # D. Per-dimension variance
        logger.info("=== D. Per-dimension variance ===")
        variance_summary = diagnose_dimension_variance(
            baseline,
            eval_data,
            eval_sources,
            dataset_names,
            encoded,
            clusters,
            out_dir,
        )

        # E. Routing
        logger.info("=== E. Routing ===")
        routing_summary = diagnose_routing(
            baseline,
            eval_sources,
            dataset_names,
            clusters,
            out_dir,
        )

        # F. Hard/easy cases
        logger.info("=== F. Hard/easy unsafe cases ===")
        diagnose_hard_easy_cases(
            baseline,
            eval_data,
            eval_sources,
            dataset_names,
            encoded,
            clusters,
            out_dir,
        )

        # Save combined summary
        import json as _json

        combined = {
            "model_key": model_key,
            "mask_sparsity": sparsity_summary,
            "residual_magnitudes": residual_summary,
            "identity_auroc": auroc_summary,
            "dimension_variance": variance_summary,
            "routing": routing_summary,
        }
        with open(out_dir / "diagnostic_summary.json", "w") as f:
            _json.dump(combined, f, indent=2)

        print_meta_summary(
            sparsity_summary,
            residual_summary,
            auroc_summary,
            variance_summary,
            routing_summary,
        )

    except Exception as e:
        logger.error(f"Post-hoc diagnostics failed: {e}", exc_info=True)


def _load_training_data(
    iid_paths: list,
    data_device: torch.device,
    filter_safe_only: bool,
    filter_unsafe_only: bool,
    holdout_ratio: float,
    max_iid_samples: int = 0,
) -> tuple:
    """Load training data, optionally splitting off a holdout set.

    Args:
        max_iid_samples: If > 0, limit total IID samples (tokens) loaded.

    Returns:
        Tuple of (train_data_np, train_labels_np, data_source_list)
        where data_source_list is a list of source file paths per sample
        (or None if unavailable).
    """
    max_s = max_iid_samples if max_iid_samples > 0 else None
    if max_s:
        logger.info(f"Limiting IID training data to {max_s} samples")

    if holdout_ratio > 0:
        train_split_ratio = 1.0 - holdout_ratio
        logger.info(
            f"Splitting training data: {train_split_ratio:.1%} for training, "
            f"{holdout_ratio:.1%} for holdout (threshold determination)"
        )
        train_data_dict = load_datasets(
            source=iid_paths,
            split="train",
            shuffle=True,
            device=data_device,
            s1_split_ratio=train_split_ratio,
            filter_safe_only=filter_safe_only,
            filter_unsafe_only=filter_unsafe_only,
            max_samples=max_s,
        )
        split_key = "s1"
    else:
        train_data_dict = load_datasets(
            source=iid_paths,
            split="train",
            shuffle=True,
            device=data_device,
            filter_safe_only=filter_safe_only,
            filter_unsafe_only=filter_unsafe_only,
            max_samples=max_s,
        )
        split_key = "train"

    train_hidden_states = train_data_dict[split_key]["hidden_states"]
    train_labels_tensor = train_data_dict[split_key]["labels"]
    data_source = train_data_dict[split_key].get("data_source")

    # Fix dimensions if needed
    if train_hidden_states.dim() == 3 and train_hidden_states.shape[1] == 1:
        train_hidden_states = train_hidden_states.squeeze(1)

    return (
        train_hidden_states.cpu().numpy(),
        train_labels_tensor.cpu().numpy(),
        data_source,
    )


@hydra.main(
    version_base="1.2",
    config_path="../../../exp_configs",
    config_name="config",
)
def main(cfg: DictConfig):
    """Main entry point for anomaly detection analysis."""
    # Extract domain from top-level config
    domain = None
    if hasattr(cfg, "domain") and cfg.domain is not None:
        domain = (
            cfg.domain.name if hasattr(cfg.domain, "name") else str(cfg.domain)
        )

    # Extract baselines config group
    if hasattr(cfg, "anodet_algos") and cfg.anodet_algos is not None:
        anodet_cfg = cfg.anodet_algos
    else:
        raise ValueError(
            "No 'anodet_algos' config group specified. "
            "Use: anodet_algos=cvdd"
        )

    # Inject domain into anodet config if extracted from top level
    if domain is not None:
        OmegaConf.set_struct(anodet_cfg, False)
        anodet_cfg.domain = domain
        OmegaConf.set_struct(anodet_cfg, True)
        logger.info(f"Using domain from top-level config: {domain}")

    # Apply the paper's tuned hyperparameters for this (model_key, method).
    #
    # This happens before anything is constructed or fitted, and before the
    # data-mode flags below are read, so every downstream consumer sees the
    # published configuration. An explicit command-line override still wins --
    # apply_tuned_defaults reads the running job's literal Hydra override list
    # and refuses to write any key that appears in it. Opt out entirely with
    # `anodet_algos.use_tuned_defaults=false`.
    #
    # Logging is configured here rather than further down so that the
    # `[tuned-defaults]` lines land in the run's own log file: a run has to
    # record the configuration it actually used, not the one on disk.
    logging.basicConfig(level=logging.INFO)
    apply_tuned_defaults(anodet_cfg)

    # Data-mode switches. These must be applied in BOTH directions: the
    # registry globals default to last-token=True / complex-data=True, so a
    # guard that only ever calls the setter with True makes `false` silently
    # inert -- the run would load the complex IID pool while the user believed
    # they had switched it off.
    from unsupervised_llm_safety.data.dataset_registry import (
        set_complex_data_mode,
        set_generation_mode,
        set_last_token_mode,
    )

    last_token = bool(anodet_cfg.get("use_last_token", False))
    generation = bool(anodet_cfg.get("use_generation", False))
    complex_data = bool(anodet_cfg.get("use_complex_data", False))

    set_last_token_mode(last_token)
    set_generation_mode(generation)
    set_complex_data_mode(complex_data)

    logger.info(
        "Data modes: last_token=%s generation=%s complex_data=%s",
        last_token,
        generation,
        complex_data,
    )

    # Load pretrained model if model_path is provided
    pretrained_model = None
    full_loaded_model = None
    if anodet_cfg.get("model_path") and anodet_cfg.get("model_path") != "null":
        model_path = Path(anodet_cfg.model_path)
        if model_path.exists():
            logger.info("=" * 80)
            logger.info(f"LOADING PRETRAINED MODEL FROM: {model_path}")
            logger.info("=" * 80)
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )

            loaded_model = torch.load(
                model_path, map_location=device, weights_only=False
            )
            full_loaded_model = loaded_model

            if hasattr(loaded_model, "encoder"):
                pretrained_model = loaded_model.encoder
            elif isinstance(loaded_model, dict) and "encoder" in loaded_model:
                pretrained_model = loaded_model["encoder"]
            else:
                pretrained_model = loaded_model

            logger.info(f"  Model type: {type(pretrained_model)}")
            if isinstance(pretrained_model, nn.Module):
                logger.info(
                    f"  Model parameters: {sum(p.numel() for p in pretrained_model.parameters()):,}"
                )
            logger.info("=" * 80)
        else:
            raise ValueError(f"Model path does not exist: {model_path}")

    # Run the analysis
    results = run_anodet_analysis(
        anodet_cfg,
        pretrained_model=pretrained_model,
        full_model=full_loaded_model,
    )

    # Return 3-slice mean AUROC for Hydra optimization
    three_slice = results.get("three_slice_metrics", {})
    return three_slice.get("three_slice_mean_auroc", 0.0)


if __name__ == "__main__":
    main()
