"""Utilities for anomaly detection method initialization and configuration."""

import logging
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from unsupervised_llm_safety.anodet_algos import (
    BCEAnodet,
    CVDDAnodet,
    CVDDKMeansMaskedAnodet,
    DeepSVDDAnodet,
    GMMAnodet,
    LearnedMaskAnodet,
    LearnedMaskLoRAAnodet,
    MahalanobisAnodet,
    OneClassSVMAnodet,
)
from unsupervised_llm_safety.common.model_io import infer_model_key_from_path
from unsupervised_llm_safety.data.dataset_registry import (
    DEFAULT_DOMAIN,
    get_adversarial_base_path,
    get_benign_generalization_datasets,
    get_dataset_paths,
)

logger = logging.getLogger(__name__)

# Dictionary mapping method names to baseline classes for easy loading
BASELINE_CLASSES = {
    "bce": BCEAnodet,
    "one_class_svm": OneClassSVMAnodet,
    "deep_svdd": DeepSVDDAnodet,
    "cvdd": CVDDAnodet,
    "cvdd_kmeans_masked": CVDDKMeansMaskedAnodet,
    "gmm": GMMAnodet,
    "mahalanobis": MahalanobisAnodet,
    "learned_mask": LearnedMaskAnodet,
    "learned_mask_lora": LearnedMaskLoRAAnodet,
}


def create_pretrained_encoder_wrapper(
    pretrained_model: nn.Module, method_name: str
) -> nn.Module:
    """Create a standardized encoder wrapper for pretrained models.

    This wrapper is used by all CVDD variants and Deep SVDD to transform
    input data through a pretrained encoder before anomaly detection.

    Args:
        pretrained_model: The pretrained model to wrap
        method_name: Name of the method (for logging)

    Returns:
        Encoder wrapper module with forward() and transform_numpy() methods
    """
    logger.info(f"Creating pretrained encoder wrapper for {method_name}")

    class PretrainedEncoderWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.device = next(model.parameters()).device

            # Ensure all parameters in the wrapped model require gradients
            # This is critical for CVDD training where the encoder is fine-tuned
            for param in self.model.parameters():
                param.requires_grad = True

        def forward(self, x):
            # Handle numpy input
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float().to(self.device)

            # Use extract_features if available, otherwise use forward directly
            # NOTE: No torch.no_grad() to allow gradient flow for training
            if hasattr(self.model, "extract_features"):
                features = self.model.extract_features(x)
            else:
                features = self.model(x)
            return features

        def transform_numpy(self, x_numpy):
            """Transform numpy data and return numpy (for One-Class SVM/Mahalanobis)."""
            features = self(x_numpy)
            return features.cpu().numpy()

    return PretrainedEncoderWrapper(pretrained_model)


def get_pretrained_encoder_output_dim(
    pretrained_encoder: nn.Module, pretrained_model: nn.Module, cfg: DictConfig
) -> int:
    """Infer the output dimension of a pretrained encoder.

    Args:
        pretrained_encoder: The encoder wrapper
        pretrained_model: The original pretrained model
        cfg: Configuration containing fallback values

    Returns:
        Output dimension of the encoder
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the actual input dimension from the model
    if hasattr(pretrained_model, "input_dim"):
        input_dim = pretrained_model.input_dim
    elif hasattr(pretrained_model, "hidden_size"):
        input_dim = pretrained_model.hidden_size
    else:
        # Try to infer from the first layer
        first_layer = next(pretrained_model.parameters())
        if len(first_layer.shape) >= 2:
            input_dim = first_layer.shape[
                1
            ]  # Assuming weight matrix [out, in]
        else:
            input_dim = cfg.get("input_dim", 4096)  # Default fallback

    # Run a dummy forward pass to get output dimension
    dummy_input = torch.randn(1, input_dim).to(device)
    with torch.no_grad():
        dummy_output = pretrained_encoder(dummy_input)
        if isinstance(dummy_output, torch.Tensor):
            rep_dim = dummy_output.shape[1]
        else:
            rep_dim = dummy_output.shape[1]  # Already numpy

    logger.info(f"Pretrained encoder output dimension: {rep_dim}")
    return rep_dim


def initialize_anodet_method(
    cfg: DictConfig,
    pretrained_model: Optional[nn.Module] = None,
    full_model: Optional[nn.Module] = None,
) -> Any:
    """Initialize the specified anomaly detection algorithm.

    Args:
        cfg: Configuration containing method name and parameters
        pretrained_model: Optional pretrained model (encoder) for methods
        full_model: Optional full autoencoder (encoder+decoder) for methods
            like denoised_mahalanobis that need reconstruction

    Returns:
        Initialized anomaly detection algorithm instance
    """
    method_name = cfg.method

    if method_name == "bce":
        # Handle pretrained encoder for BCE
        pretrained_encoder = None
        rep_dim = cfg.bce.get("rep_dim", 32)

        if pretrained_model is not None and cfg.bce.get(
            "use_pretrained_encoder", False
        ):
            pretrained_encoder = create_pretrained_encoder_wrapper(
                pretrained_model, "BCE"
            )
            rep_dim = get_pretrained_encoder_output_dim(
                pretrained_encoder, pretrained_model, cfg.bce
            )

        return BCEAnodet(
            rep_dim=rep_dim,
            pretrained_encoder=pretrained_encoder,
            epochs=cfg.bce.get("epochs", 50),
            lr=cfg.bce.get("lr", 0.001),
            batch_size=cfg.bce.get("batch_size", 1024),
            analysis_batch_size=cfg.bce.get("analysis_batch_size", 256),
            device=cfg.bce.get("device", "cuda"),
            seed=cfg.bce.get("seed", 42),
            weight_decay=cfg.bce.get("weight_decay", 1e-6),
            lr_milestones=cfg.bce.get("lr_milestones", []),
            freeze_encoder=cfg.bce.get("freeze_encoder", False),
            linear_only=cfg.bce.get("linear_only", False),
            no_bias=cfg.bce.get("no_bias", False),
        )
    elif method_name == "one_class_svm":
        return OneClassSVMAnodet(
            nu=cfg.one_class_svm.get("nu", 0.1),
            kernel=cfg.one_class_svm.get("kernel", "rbf"),
            gamma=cfg.one_class_svm.get("gamma", 0.1),
            n_epochs=cfg.one_class_svm.get("n_epochs", 100),
            batch_size=cfg.one_class_svm.get("batch_size", 256),
            lr=cfg.one_class_svm.get("lr", 0.01),
            kernel_batch_size=cfg.one_class_svm.get("kernel_batch_size", 1000),
            max_support_vectors=cfg.one_class_svm.get(
                "max_support_vectors", None
            ),
            subsample_size=cfg.one_class_svm.get("subsample_size", None),
        )
    elif method_name == "deep_svdd":
        return DeepSVDDAnodet(
            objective=cfg.deep_svdd.get("objective", "one-class"),
            nu=cfg.deep_svdd.get("nu", 0.1),
            rep_dim=cfg.deep_svdd.get("rep_dim", 32),
            n_epochs=cfg.deep_svdd.get("n_epochs", 100),
            lr=cfg.deep_svdd.get("lr", 0.001),
            weight_decay=cfg.deep_svdd.get("weight_decay", 1e-6),
            batch_size=cfg.deep_svdd.get("batch_size", 256),
            lr_milestones=cfg.deep_svdd.get("lr_milestones", [50]),
            device=cfg.deep_svdd.get("device", "cuda"),
            seed=cfg.deep_svdd.get("seed", 42),
            pretrain=cfg.deep_svdd.get("pretrain", True),
            ae_n_epochs=cfg.deep_svdd.get("ae_n_epochs", 100),
            ae_lr=cfg.deep_svdd.get("ae_lr", 0.001),
            ae_weight_decay=cfg.deep_svdd.get("ae_weight_decay", 1e-6),
        )
    elif method_name == "cvdd":
        # Handle pretrained encoder for CVDD
        pretrained_encoder = None
        rep_dim = cfg.cvdd.get("rep_dim", 32)

        if pretrained_model is not None and cfg.cvdd.get(
            "use_pretrained_encoder", False
        ):
            pretrained_encoder = create_pretrained_encoder_wrapper(
                pretrained_model, "CVDD"
            )
            rep_dim = get_pretrained_encoder_output_dim(
                pretrained_encoder, pretrained_model, cfg.cvdd
            )

        return CVDDAnodet(
            n_contexts=cfg.cvdd.get("n_contexts", 3),
            rep_dim=rep_dim,
            n_epochs=cfg.cvdd.get("n_epochs", 100),
            lr=cfg.cvdd.get("lr", 0.001),
            weight_decay=cfg.cvdd.get("weight_decay", 1e-6),
            batch_size=cfg.cvdd.get("batch_size", 256),
            lr_milestones=cfg.cvdd.get("lr_milestones", [50, 80]),
            device=cfg.cvdd.get("device", "cuda"),
            seed=cfg.cvdd.get("seed", 42),
            fixed_centers=cfg.cvdd.get("fixed_centers", True),
            t_start=cfg.cvdd.get("t_start", 5.0),
            t_end=cfg.cvdd.get("t_end", 0.5),
            pretrained_encoder=pretrained_encoder,
        )
    elif method_name == "cvdd_kmeans_masked":
        pretrained_encoder = None
        rep_dim = cfg.cvdd_kmeans_masked.get("rep_dim", 32)

        if pretrained_model is not None and cfg.cvdd_kmeans_masked.get(
            "use_pretrained_encoder", False
        ):
            pretrained_encoder = create_pretrained_encoder_wrapper(
                pretrained_model, "CVDDKMeansMasked"
            )
            rep_dim = get_pretrained_encoder_output_dim(
                pretrained_encoder, pretrained_model, cfg.cvdd_kmeans_masked
            )

        method = CVDDKMeansMaskedAnodet(
            n_clusters=cfg.cvdd_kmeans_masked.get("n_clusters", 3),
            k_mask=cfg.cvdd_kmeans_masked.get("k_mask", 100),
            firing_threshold=cfg.cvdd_kmeans_masked.get(
                "firing_threshold", 1e-6
            ),
            analysis_output_dir=cfg.cvdd_kmeans_masked.get(
                "analysis_output_dir", "./cvdd_kmeans_analysis/"
            ),
            epochs=cfg.cvdd_kmeans_masked.get("epochs", 50),
            lr=cfg.cvdd_kmeans_masked.get("lr", 0.001),
            batch_size=cfg.cvdd_kmeans_masked.get("batch_size", 1024),
            analysis_batch_size=cfg.cvdd_kmeans_masked.get(
                "analysis_batch_size", 256
            ),
            fixed_centers=cfg.cvdd_kmeans_masked.get("fixed_centers", False),
            rep_dim=rep_dim,
            pretrained_encoder=pretrained_encoder,
            cluster_in_encoded_space=cfg.cvdd_kmeans_masked.get(
                "cluster_in_encoded_space", False
            ),
            identity_encoder=cfg.cvdd_kmeans_masked.get(
                "identity_encoder", False
            ),
            cluster_method=cfg.cvdd_kmeans_masked.get(
                "cluster_method", "default"
            ),
            distance_metric=cfg.cvdd_kmeans_masked.get(
                "distance_metric", "l1"
            ),
            use_masking=cfg.cvdd_kmeans_masked.get("use_masking", True),
            mask_variant=cfg.cvdd_kmeans_masked.get("mask_variant", "top_k"),
            mask_coverage=cfg.cvdd_kmeans_masked.get("mask_coverage", 0.9),
            score_normalization=cfg.cvdd_kmeans_masked.get(
                "score_normalization", "percentile_train"
            ),
            l1_feature_penalty=cfg.cvdd_kmeans_masked.get(
                "l1_feature_penalty", 0.0
            ),
            train_encoder=cfg.cvdd_kmeans_masked.get("train_encoder", False),
            outlier_threshold=cfg.cvdd_kmeans_masked.get(
                "outlier_threshold", 0.0
            ),
            l0_filter=cfg.cvdd_kmeans_masked.get("l0_filter", 0.0),
            min_cluster_size=cfg.cvdd_kmeans_masked.get("min_cluster_size", 0),
            max_union_size=cfg.cvdd_kmeans_masked.get("max_union_size", 500),
            typicality_temperature=cfg.cvdd_kmeans_masked.get(
                "typicality_temperature", 1.0
            ),
            mahalanobis_reg=cfg.cvdd_kmeans_masked.get(
                "mahalanobis_reg", 1e-3
            ),
            mahalanobis_diag=cfg.cvdd_kmeans_masked.get(
                "mahalanobis_diag", False
            ),
            mahalanobis_min_across_clusters=cfg.cvdd_kmeans_masked.get(
                "mahalanobis_min_across_clusters", False
            ),
            global_mask=cfg.cvdd_kmeans_masked.get("global_mask", False),
            cluster_in_masked_space=cfg.cvdd_kmeans_masked.get(
                "cluster_in_masked_space", False
            ),
            device=cfg.cvdd_kmeans_masked.get("device", "cuda"),
            seed=cfg.cvdd_kmeans_masked.get("seed", 42),
        )

        # Clustering-basin knobs. Defaults reproduce legacy behaviour exactly
        # (cluster_seed=None -> no seed= passed to faiss.Kmeans,
        # init="faiss_random", GPU clustering), so no existing run changes
        # unless these are set explicitly.
        cluster_seed = cfg.cvdd_kmeans_masked.get("cluster_seed", None)
        method.cluster_seed = (
            None if cluster_seed is None else int(cluster_seed)
        )
        method.cluster_init = cfg.cvdd_kmeans_masked.get(
            "cluster_init", "faiss_random"
        )
        method.cluster_on_cpu = bool(
            cfg.cvdd_kmeans_masked.get("cluster_on_cpu", False)
        )
        if (
            method.cluster_seed is not None
            or method.cluster_init != "faiss_random"
            or method.cluster_on_cpu
        ):
            logger.info(
                f"Clustering basin knobs: cluster_seed={method.cluster_seed}, "
                f"cluster_init={method.cluster_init}, "
                f"cluster_on_cpu={method.cluster_on_cpu}"
            )

        # Attach full autoencoder for residual_mahalanobis distance metric
        dist_metric = cfg.cvdd_kmeans_masked.get("distance_metric", "l1")
        if dist_metric == "residual_mahalanobis" and full_model is not None:
            device = torch.device(cfg.cvdd_kmeans_masked.get("device", "cuda"))
            method.full_autoencoder = full_model.to(device)
            method.full_autoencoder.eval()
            for param in method.full_autoencoder.parameters():
                param.requires_grad = False
            logger.info(
                f"Attached full autoencoder for residual_mahalanobis: "
                f"{type(full_model).__name__}"
            )

        return method
    elif method_name == "mahalanobis":
        return MahalanobisAnodet(
            regularization=cfg.mahalanobis.get("regularization", 1e-5),
            score_type=cfg.mahalanobis.get("score_type", "percentile"),
            batch_size=cfg.mahalanobis.get("batch_size", 1024),
            device=cfg.mahalanobis.get("device", "cuda"),
        )
    elif method_name == "gmm":
        return GMMAnodet(
            n_components=cfg.gmm.get("n_components", 5),
            em_max_iter=cfg.gmm.get("em_max_iter", 100),
            em_tol=cfg.gmm.get("em_tol", 1e-4),
            init_method=cfg.gmm.get("init_method", "kmeans++"),
            score_normalization=cfg.gmm.get(
                "score_normalization", "percentile"
            ),
            batch_size=cfg.gmm.get("batch_size", 1024),
            device=cfg.gmm.get("device", "cuda"),
        )
    elif method_name == "learned_mask":
        lcfg = cfg.learned_mask
        # Pass full autoencoder (encoder+decoder) as pretrained_encoder
        autoencoder = (
            full_model if full_model is not None else pretrained_model
        )
        if autoencoder is not None:
            device = torch.device(lcfg.get("device", "cuda"))
            autoencoder = autoencoder.to(device)
            autoencoder.eval()
            for param in autoencoder.parameters():
                param.requires_grad = False

        return LearnedMaskAnodet(
            n_clusters=lcfg.get("n_clusters", 1500),
            cluster_method=lcfg.get("cluster_method", "l0"),
            outlier_threshold=lcfg.get("outlier_threshold", 2.0),
            l0_filter=lcfg.get("l0_filter", 2.0),
            min_cluster_size=lcfg.get("min_cluster_size", 0),
            max_union_size=lcfg.get("max_union_size", 500),
            use_clustering_cache=lcfg.get("use_clustering_cache", False),
            clustering_cache_model_key=(
                cfg.get("model_key")
                if lcfg.get("use_clustering_cache", False)
                else None
            ),
            # Clustering basin knobs. Defaults are byte-identical to the
            # legacy path (no seed= reaches faiss.Kmeans, GPU clustering).
            cluster_seed=(
                None
                if lcfg.get("cluster_seed", None) is None
                else int(lcfg.get("cluster_seed"))
            ),
            cluster_init=lcfg.get("cluster_init", "faiss_random"),
            cluster_on_cpu=bool(lcfg.get("cluster_on_cpu", False)),
            mask_type=lcfg.get("mask_type", "continuous"),
            residual_norm=lcfg.get("residual_norm", "l2"),
            sparsity_weighting=lcfg.get("sparsity_weighting", "uniform"),
            lambda_sparsity=lcfg.get("lambda_sparsity", 0.01),
            lambda_candidates=list(lcfg.get("lambda_candidates", [])) or None,
            n_epochs=lcfg.get("n_epochs", 50),
            lr=lcfg.get("lr", 0.01),
            batch_size=lcfg.get("batch_size", 256),
            sparsity_target=lcfg.get("sparsity_target", 50),
            train_decoder=lcfg.get("train_decoder", False),
            per_cluster_decoder=lcfg.get("per_cluster_decoder", False),
            decoder_lr=lcfg.get("decoder_lr", 1e-4),
            logit_init=lcfg.get("logit_init", 1.0),
            whitened_mode=lcfg.get("whitened_mode", "diagonal"),
            shrinkage_alpha=lcfg.get("shrinkage_alpha", 0.3),
            small_cluster_threshold=lcfg.get("small_cluster_threshold", 10),
            variance_floor=lcfg.get("variance_floor", 1e-6),
            scoring_method=lcfg.get("scoring_method", "l2_norm"),
            global_mask=lcfg.get("global_mask", False),
            noise_lr_min_cluster_size=lcfg.get(
                "noise_lr_min_cluster_size", 50
            ),
            device=lcfg.get("device", "cuda"),
            seed=lcfg.get("seed", 42),
            analysis_batch_size=lcfg.get("analysis_batch_size", 256),
            pretrained_encoder=autoencoder,
        )
    elif method_name == "learned_mask_lora":
        lcfg = cfg.learned_mask_lora
        # Pass full autoencoder as pretrained_encoder (same as learned_mask)
        autoencoder = (
            full_model if full_model is not None else pretrained_model
        )
        if autoencoder is not None:
            device = torch.device(lcfg.get("device", "cuda"))
            autoencoder = autoencoder.to(device)
            autoencoder.eval()
            for param in autoencoder.parameters():
                param.requires_grad = False

        # Collect step-1 kwargs from learned_mask config group
        step1_kwargs = {}
        if hasattr(cfg, "learned_mask"):
            lm = cfg.learned_mask
            step1_kwargs = {
                "n_clusters": lm.get("n_clusters", 1500),
                "cluster_method": lm.get("cluster_method", "l0"),
                "outlier_threshold": lm.get("outlier_threshold", 2.0),
                "l0_filter": lm.get("l0_filter", 2.0),
                "use_clustering_cache": lm.get("use_clustering_cache", False),
                "clustering_cache_model_key": (
                    cfg.get("model_key")
                    if lm.get("use_clustering_cache", False)
                    else None
                ),
                # Clustering basin knobs. Only bite in Mode B (stage 2
                # re-clusters); in Mode A the stage-1 partition is loaded from
                # step1_run_dir verbatim and these are inert. Defaults are
                # byte-identical to the legacy path.
                "cluster_seed": (
                    None
                    if lm.get("cluster_seed", None) is None
                    else int(lm.get("cluster_seed"))
                ),
                "cluster_init": lm.get("cluster_init", "faiss_random"),
                "cluster_on_cpu": bool(lm.get("cluster_on_cpu", False)),
                "mask_type": lm.get("mask_type", "continuous"),
                "residual_norm": lm.get("residual_norm", "whitened"),
                "sparsity_weighting": lm.get("sparsity_weighting", "uniform"),
                "lambda_sparsity": lm.get("lambda_sparsity", 0.01),
                "lambda_candidates": (
                    list(lm.get("lambda_candidates", [])) or None
                ),
                "n_epochs": lm.get("n_epochs", 50),
                "lr": lm.get("lr", 0.01),
                "batch_size": lm.get("batch_size", 256),
                "sparsity_target": lm.get("sparsity_target", 50),
                "train_decoder": lm.get("train_decoder", False),
                "per_cluster_decoder": lm.get("per_cluster_decoder", False),
                "decoder_lr": lm.get("decoder_lr", 1e-4),
                "logit_init": lm.get("logit_init", 1.0),
                "whitened_mode": lm.get("whitened_mode", "diagonal"),
                "shrinkage_alpha": lm.get("shrinkage_alpha", 0.3),
                "small_cluster_threshold": lm.get(
                    "small_cluster_threshold", 10
                ),
                "variance_floor": lm.get("variance_floor", 1e-6),
                # NOTE: these are forwarded verbatim to
                # LearnedMaskAnodet in LoRA Mode B, so every key here
                # must be one of its constructor arguments.
                # mahalanobis_reg is LoRA's own and is passed as an
                # explicit kwarg below.
            }

        return LearnedMaskLoRAAnodet(
            step1_run_dir=lcfg.get("step1_run_dir"),
            lora_rank=lcfg.get("lora_rank", 8),
            lora_lr=lcfg.get("lora_lr", 1e-3),
            lora_epochs=lcfg.get("lora_epochs", 30),
            lora_batch_size=lcfg.get("lora_batch_size", 256),
            lora_sparsity_lambda=lcfg.get("lora_sparsity_lambda", 1e-3),
            lora_min_cluster_size=lcfg.get("lora_min_cluster_size", 20),
            lora_rank_schedule=lcfg.get("lora_rank_schedule"),
            rank_schedule_small_max=lcfg.get("rank_schedule_small_max", 500),
            rank_schedule_medium_max=lcfg.get(
                "rank_schedule_medium_max", 2000
            ),
            run_expensive_diagnostics=lcfg.get(
                "run_expensive_diagnostics", True
            ),
            scoring_method=lcfg.get("scoring_method", "l2_norm"),
            mahalanobis_reg=lcfg.get("mahalanobis_reg", 1e-4),
            device=lcfg.get("device", "cuda"),
            seed=lcfg.get("seed", 42),
            analysis_batch_size=lcfg.get("analysis_batch_size", 256),
            pretrained_encoder=autoencoder,
            **step1_kwargs,
        )
    else:
        raise ValueError(
            f"Unknown anomaly detection algorithm: {method_name}. "
            f"Available: {list(BASELINE_CLASSES.keys())}"
        )


def prepare_data_for_baseline(
    data: np.ndarray,
    method_name: str,
    encoder_wrapper: Optional[nn.Module],
    cfg: DictConfig,
) -> np.ndarray:
    """Prepare data for anomaly detection algorithm, applying encoder transformation if needed.

    Args:
        data: Input data as numpy array
        method_name: Name of the anomaly detection algorithm
        encoder_wrapper: Optional pretrained encoder wrapper
        cfg: Configuration

    Returns:
        Processed data (either original or transformed)
    """
    # Methods that need external data transformation
    TRANSFORM_METHODS = ["one_class_svm", "mahalanobis", "gmm"]

    if encoder_wrapper is not None and method_name in TRANSFORM_METHODS:
        if cfg.get(f"{method_name}.use_pretrained_encoder", False):
            logger.info(
                f"Transforming data using pretrained encoder for {method_name}"
            )

            # Transform in batches to handle large datasets
            batch_size = cfg.get(f"{method_name}.encoder_batch_size", 1000)
            n_samples = len(data)
            transformed_data = []

            for i in tqdm(
                range(0, n_samples, batch_size),
                desc=f"Encoding data for {method_name}",
            ):
                batch = data[i : min(i + batch_size, n_samples)]
                transformed_batch = encoder_wrapper.transform_numpy(batch)
                transformed_data.append(transformed_batch)

            return np.vstack(transformed_data)

    return data


def resolve_dataset_configurations(cfg: DictConfig) -> None:
    """Resolve dataset configurations from registry (same as CCS analysis).

    Supports domain selection via cfg.domain (defaults to 'common_sense_safety').
    """
    if not cfg.get("use_centralized_datasets", True):
        return

    # Infer model key
    model_key = cfg.get("model_key", None)
    if not model_key:
        model_key = infer_model_key_from_path(cfg.model_path)

    if not model_key:
        logger.warning(
            "Could not infer model key from path. Using dataset paths from config."
        )
        return

    # Get domain (defaults to common_sense_safety for backward compatibility)
    domain = cfg.get("domain", DEFAULT_DOMAIN)
    logger.info(
        f"Loading dataset paths from registry for model: {model_key}, domain: {domain}"
    )

    # Load IID datasets
    iid_datasets = get_dataset_paths(model_key, "iid", domain=domain)

    # Apply IID dataset filter if specified
    iid_include = cfg.get("iid_include", None)
    iid_exclude = cfg.get("iid_exclude", None)
    if iid_include is not None:
        include_set = set(
            OmegaConf.to_container(iid_include)
            if not isinstance(iid_include, (list, set))
            else iid_include
        )
        before = len(iid_datasets)
        iid_datasets = [d for d in iid_datasets if d["name"] in include_set]
        logger.info(
            f"  iid_include filter: {before} -> {len(iid_datasets)} "
            f"datasets (keeping: {include_set})"
        )
    elif iid_exclude is not None:
        exclude_set = set(
            OmegaConf.to_container(iid_exclude)
            if not isinstance(iid_exclude, (list, set))
            else iid_exclude
        )
        before = len(iid_datasets)
        iid_datasets = [
            d for d in iid_datasets if d["name"] not in exclude_set
        ]
        logger.info(
            f"  iid_exclude filter: {before} -> {len(iid_datasets)} "
            f"datasets (removed: {exclude_set})"
        )

    if iid_datasets:
        cfg.datasets.iid = OmegaConf.create(iid_datasets)
        logger.info(
            f"  Loaded {len(iid_datasets)} IID dataset paths from registry"
        )

    # Load OOD datasets
    ood_datasets = get_dataset_paths(model_key, "ood", domain=domain)
    if ood_datasets:
        cfg.datasets.ood = OmegaConf.create(ood_datasets)
        logger.info(
            f"  Loaded {len(ood_datasets)} OOD dataset paths from registry"
        )

    # Set adversarial base path
    adversarial_base_path = get_adversarial_base_path(model_key, domain=domain)
    if adversarial_base_path:
        cfg.adversarial_base_path = adversarial_base_path
        logger.info("  Set adversarial base path from registry")
    else:
        cfg.adversarial_base_path = None
        logger.info(
            "  No adversarial base path found (skipping adversarial evaluation)"
        )

    # Load benign generalization datasets
    benign_gen_datasets = get_benign_generalization_datasets(
        model_key, domain=domain
    )
    if benign_gen_datasets:
        cfg.datasets.benign_generalization = OmegaConf.create(
            benign_gen_datasets
        )
        logger.info(
            f"  Loaded {len(benign_gen_datasets)} benign generalization dataset paths from registry"
        )
