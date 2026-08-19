"""CVDD with KMeans clustering on raw inputs and masked L1 distance based on firing neurons."""

import copy
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

from unsupervised_llm_safety.anodet_algos.cvdd import CVDDAnodet, CVDDCenters
from unsupervised_llm_safety.anodet_algos.cvdd_km_utils import (
    analyze_firing_patterns,
    cluster_kmeans,
    compute_cluster_covariances,
    compute_mahalanobis_distances_batch,
    compute_mask_disagreement_batch,
    create_cluster_masks,
    fit_residual_mahalanobis,
    merge_small_clusters,
    score_residual_mahalanobis,
    split_large_union_clusters,
)
from unsupervised_llm_safety.anodet_algos.scaler import IdentityScaler

logger = logging.getLogger(__name__)


class CVDDKMeansMaskedAnodet(CVDDAnodet):
    """CVDD variant using KMeans clustering and masked distance.

    This method:
    1. Clusters inputs using KMeans (raw inputs by default, or encoded space if configured)
    2. Analyzes firing neuron patterns per cluster through an encoder
    3. Creates a binary mask of top-k firing neurons (or all-ones if use_masking=False)
    4. Uses masked distance (L1 or L2, configurable) for anomaly detection

    When cluster_in_encoded_space=True:
    - Data is encoded first, then clustering happens in encoded space
    - All subsequent operations (training, scoring) work in encoded space
    - Network forward pass is skipped during training since data is pre-encoded

    Ablation flags:
    - distance_metric: "l1", "l2", or "mahalanobis" for distance computation.
      "mahalanobis" computes per-cluster Mahalanobis distance in the masked subspace.
    - use_masking: If False, uses all-ones mask (no masking)
    - score_normalization: "percentile_train" (normalize using training percentiles) or
      "minmax_test" (normalize using test batch min/max, Deep SVDD style)
    - mahalanobis_reg: Covariance regularization for Mahalanobis (only used when
      distance_metric="mahalanobis")
    - mahalanobis_min_across_clusters: If True, take min Mahalanobis distance across all
      clusters; if False, use the assigned cluster only (default)
    """

    def __init__(
        self,
        n_clusters: int = 3,
        k_mask: int = 100,
        firing_threshold: float = 0,
        analysis_output_dir: str = "./cvdd_kmeans_analysis/",
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 1024,
        analysis_batch_size: int = 256,
        fixed_centers: bool = False,
        rep_dim: int = 32,
        pretrained_encoder: Optional[nn.Module] = None,
        cluster_in_encoded_space: bool = False,
        identity_encoder: bool = False,  # If True, use nn.Identity() encoder (raw activation space; E3)
        # === ABLATION FLAGS ===
        cluster_method: str = "default",  # "default" (FAISS L2), "l0", "cosine_binary", or "l1" (k-medians)
        distance_metric: str = "l1",  # "l1", "l2", "hamming", "typicality", or "mahalanobis"
        use_masking: bool = True,  # If False, mask is all ones (no masking)
        mask_variant: str = "top_k",  # "top_k" (fixed k) or "frequency_coverage" (variable-size)
        mask_coverage: float = 0.9,  # Coverage threshold for frequency_coverage mask variant
        score_normalization: str = "percentile_train",  # "percentile_train" or "minmax_test"
        l1_feature_penalty: float = 0.0,  # L1 penalty on SAE activations (Step 3)
        train_encoder: bool = False,  # If True, train encoder even with cluster_in_encoded_space
        outlier_threshold: float = 0.0,  # Split clusters with union_support > T*median (0=disabled)
        l0_filter: float = 0.0,  # Filter samples with L0 > T*median before clustering (0=disabled)
        min_cluster_size: int = 0,  # Merge clusters < this size (0=disabled)
        max_union_size: int = 500,  # Max union-of-support after merge
        typicality_temperature: float = 1.0,  # Softmax temperature for typicality (Step 4)
        mahalanobis_reg: float = 1e-3,  # Covariance regularization (mahalanobis only)
        mahalanobis_diag: bool = False,  # If True, use diagonal covariance (saves memory)
        mahalanobis_min_across_clusters: bool = False,  # If True, min across all clusters; if False, use assigned cluster
        global_mask: bool = False,  # If True, use one global mask (pooled firing freq) for all clusters
        cluster_in_masked_space: bool = False,  # If True, apply global mask BEFORE clustering (requires global_mask=True)
        **kwargs,
    ):
        """Initialize CVDD KMeans Masked method.

        Args:
            n_clusters: Number of clusters for KMeans on raw inputs
            k_mask: Number of top neurons to keep in mask
            firing_threshold: Threshold for considering a neuron as firing (z > threshold)
            analysis_output_dir: Directory to save analysis plots
            epochs: Number of training epochs
            lr: Learning rate
            batch_size: Batch size for training
            analysis_batch_size: Batch size for scoring/analysis to control memory
            fixed_centers: Whether to fix centers during training
            rep_dim: Representation dimension (only used if no pretrained_encoder)
            pretrained_encoder: Pretrained encoder module from run_anodet_analysis
            cluster_in_encoded_space: If True, cluster in encoded space instead of raw inputs.
                Data will be encoded first, then all operations (clustering, training) happen
                in the encoded space. In training loop, data won't be passed through network again.
            distance_metric: Distance metric to use ("l1", "l2", or "mahalanobis")
            use_masking: If False, mask is all ones (no masking applied)
            score_normalization: Normalization method ("percentile_train" or "minmax_test")
            **kwargs: Additional arguments passed to parent
        """
        # Initialize parent - let it handle encoder initialization
        super().__init__(
            n_epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            n_contexts=n_clusters,
            fixed_centers=fixed_centers,
            rep_dim=rep_dim,
            pretrained_encoder=pretrained_encoder,
            **kwargs,
        )

        self.n_clusters = n_clusters
        self.k_mask = k_mask
        self.firing_threshold = firing_threshold
        self.analysis_output_dir = analysis_output_dir
        self.cluster_in_encoded_space = cluster_in_encoded_space
        self.identity_encoder = identity_encoder
        self.cluster_method = cluster_method
        self.mask_variant = mask_variant
        self.mask_coverage = mask_coverage

        # L0 clustering state: binary centroids for assignment
        self.cluster_centers_binary: Optional[np.ndarray] = None

        # Initialize attributes that will be set during clustering
        self.kmeans: Optional[Any] = None
        self.cluster_centers_raw: Optional[np.ndarray] = None

        # Clustering reproducibility knobs (default = legacy behaviour:
        # unseeded FAISS random init on GPU). Set these on the instance
        # *before* fit() to control the K-means basin — used to make the
        # bimodal Ministral clustering reproducible. None seed => exact
        # legacy call (no seed= passed to faiss.Kmeans).
        self.cluster_seed: Optional[int] = None
        self.cluster_init: str = "faiss_random"  # or kmeans++ / sklearn_ninit
        self.cluster_on_cpu: bool = False

        # Will be populated during fit
        self.cluster_masks: Dict[int, np.ndarray] = {}  # Per-cluster masks
        self.masks_tensor: Optional[torch.Tensor] = (
            None  # Cached [n_clusters, embedding_dim]
        )
        # Pre-finetune encoder state for health diagnostics
        self._pre_finetune_state_dict: Optional[dict] = None
        # Training data reference for diagnostics (set during fit)
        self._diagnostic_data: Optional[np.ndarray] = None
        self._diagnostic_cluster_labels: Optional[np.ndarray] = None

        # Use identity scaler for raw data
        self.scaler = IdentityScaler()

        # Metrics tracking
        self.metrics_file_path: Optional[Path] = None
        self.metrics_history: List[Dict[str, Any]] = []

        # Batch size used during post-training analysis / scoring to manage memory
        self.analysis_batch_size = analysis_batch_size

        # Ablation flags
        self.distance_metric = distance_metric
        self.use_masking = use_masking
        self.score_normalization = score_normalization

        # "hamming" is deliberately NOT accepted. It used to pass validation
        # here and then fall through both scoring dispatches into the
        # squared-L2 branch, silently, with nothing in the log naming Hamming.
        # Reject it rather than scoring something the caller did not ask for.
        if distance_metric not in (
            "l1",
            "l2",
            "typicality",
            "mahalanobis",
            "mask_disagreement",
            "residual_mahalanobis",
        ):
            raise ValueError(
                f"distance_metric must be 'l1', 'l2', "
                f"'typicality', 'mahalanobis', "
                f"'mask_disagreement', or 'residual_mahalanobis', "
                f"got {distance_metric}"
            )

        # Step 3: L1 feature penalty
        self.l1_feature_penalty = l1_feature_penalty
        self.train_encoder = train_encoder
        self.outlier_threshold = outlier_threshold
        self.l0_filter = l0_filter
        self.min_cluster_size = min_cluster_size
        self.max_union_size = max_union_size

        # Step 4: Typicality scoring state
        self.typicality_temperature = typicality_temperature
        # Per-cluster frequency vectors [n_clusters, d] — populated in fit()
        self.cluster_frequency_vectors: Optional[torch.Tensor] = None

        # Global mask flag
        self.global_mask = global_mask
        self.cluster_in_masked_space = cluster_in_masked_space
        if cluster_in_masked_space and not global_mask:
            raise ValueError(
                "cluster_in_masked_space=True requires global_mask=True"
            )

        # Mahalanobis-specific config
        self.mahalanobis_reg = mahalanobis_reg
        self.mahalanobis_diag = mahalanobis_diag
        self.mahalanobis_min_across_clusters = mahalanobis_min_across_clusters

        # Per-cluster Mahalanobis state (populated during fit)
        self.cluster_precisions: Optional[Dict[int, torch.Tensor]] = None
        self.cluster_means_sub: Optional[Dict[int, torch.Tensor]] = None
        self.cluster_mask_indices: Optional[Dict[int, torch.Tensor]] = None

        # Full autoencoder (encoder+decoder) for residual_mahalanobis
        self.full_autoencoder: Optional[nn.Module] = None

        # Per-cluster union support indices for Mahalanobis subspace
        # (used when use_masking=False to avoid full-dim covariances)
        self._cluster_union_indices: Optional[Dict[int, np.ndarray]] = None
        # Per-cluster normalization state (populated when mode is percentile_train_per_cluster)
        self.score_low_per_cluster: Optional[Dict[int, float]] = None
        self.score_high_per_cluster: Optional[Dict[int, float]] = None

        if score_normalization not in (
            "percentile_train",
            "percentile_train_per_cluster",
            "minmax_test",
        ):
            raise ValueError(
                f"score_normalization must be 'percentile_train', "
                f"'percentile_train_per_cluster', or 'minmax_test', "
                f"got {score_normalization}"
            )

    def _cluster_raw_inputs(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform KMeans clustering on input data.

        Delegates to :func:`cvdd_km_utils.cluster_kmeans`.
        """
        result = cluster_kmeans(
            data=data,
            n_clusters=self.n_clusters,
            cluster_method=self.cluster_method,
            l0_filter=self.l0_filter,
            device=self.device,
            seed=self.cluster_seed,
            init=self.cluster_init,
            cluster_on_cpu=self.cluster_on_cpu,
        )
        self.kmeans = result.kmeans
        self.cluster_centers_raw = result.centroids
        self.cluster_centers_binary = result.binary_centroids
        return result.labels, result.centroids

    def _split_large_union_clusters(
        self,
        data: np.ndarray,
        cluster_labels: np.ndarray,
        threshold: float,
        max_rounds: int = 10,
    ) -> np.ndarray:
        """Split clusters whose union_support exceeds threshold x median.

        Delegates to :func:`cvdd_km_utils.split_large_union_clusters`.
        """
        result = split_large_union_clusters(
            data=data,
            cluster_labels=cluster_labels,
            n_clusters=self.n_clusters,
            threshold=threshold,
            max_rounds=max_rounds,
        )
        self.n_clusters = result.n_clusters
        self.cluster_centers_raw = result.centroids

        # Rebuild FAISS index on new centroids
        centroids_f32 = result.centroids.astype(np.float32).copy()
        self.kmeans = faiss.Kmeans(
            d=centroids_f32.shape[1],
            k=self.n_clusters,
            niter=1,
            verbose=False,
        )
        self.kmeans.train(centroids_f32)
        self.kmeans.centroids = centroids_f32

        return result.labels

    def _analyze_firing_patterns(
        self,
        data: np.ndarray,
        cluster_labels: np.ndarray,
        is_encoded: bool = False,
    ) -> Dict[int, Dict[str, np.ndarray]]:
        """Analyze firing neuron patterns per cluster.

        Delegates to :func:`cvdd_km_utils.analyze_firing_patterns`.
        """
        return analyze_firing_patterns(
            data=data,
            n_clusters=self.n_clusters,
            cluster_labels=cluster_labels,
            firing_threshold=self.firing_threshold,
            sae=None if is_encoded else self.net,
            device=self.device,
        )

    def _create_cluster_masks(
        self, cluster_stats: Dict[int, Dict[str, np.ndarray]]
    ) -> Dict[int, torch.Tensor]:
        """Create binary masks of top-k firing neurons for each cluster.

        Delegates to :func:`cvdd_km_utils.create_cluster_masks`.
        """
        return create_cluster_masks(
            cluster_stats=cluster_stats,
            n_clusters=self.n_clusters,
            k_mask=self.k_mask,
            use_masking=self.use_masking,
            mask_variant=self.mask_variant,
            mask_coverage=self.mask_coverage,
            device=self.device,
            global_mask=self.global_mask,
        )

    def _compute_cluster_covariances(
        self,
        data: np.ndarray,
        cluster_labels: np.ndarray,
    ) -> None:
        """Compute per-cluster covariance and precision matrices.

        Delegates to :func:`cvdd_km_utils.compute_cluster_covariances`.
        """
        assert self.masks_tensor is not None
        assert self.net is not None

        self.net.eval()
        result = compute_cluster_covariances(
            data=data,
            cluster_labels=cluster_labels,
            n_clusters=self.n_clusters,
            masks_tensor=self.masks_tensor,
            cluster_union_indices=self._cluster_union_indices,
            use_masking=self.use_masking,
            mahalanobis_reg=self.mahalanobis_reg,
            cluster_in_encoded_space=self.cluster_in_encoded_space,
            sae=None if self.cluster_in_encoded_space else self.net,
            device=self.device,
        )
        self.cluster_precisions = result.precisions
        self.cluster_means_sub = result.means_sub
        self.cluster_mask_indices = result.mask_indices
        self.net.train()

    def _compute_mask_disagreement_batch(
        self,
        z: torch.Tensor,
        batch_cluster_assignments: np.ndarray,
    ) -> torch.Tensor:
        """Compute mask disagreement scores for a batch.

        Delegates to :func:`cvdd_km_utils.compute_mask_disagreement_batch`.
        """
        assert self.masks_tensor is not None
        return compute_mask_disagreement_batch(
            z=z,
            batch_cluster_assignments=batch_cluster_assignments,
            masks_tensor=self.masks_tensor,
            firing_threshold=self.firing_threshold,
            device=self.device,
        )

    def _compute_mahalanobis_distances_batch(
        self,
        z: torch.Tensor,
        batch_cluster_assignments: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Compute Mahalanobis distances for a batch.

        Delegates to :func:`cvdd_km_utils.compute_mahalanobis_distances_batch`.
        """
        assert self.cluster_precisions is not None
        assert self.cluster_means_sub is not None
        assert self.cluster_mask_indices is not None
        return compute_mahalanobis_distances_batch(
            z=z,
            batch_cluster_assignments=batch_cluster_assignments,
            n_clusters=self.n_clusters,
            cluster_precisions=self.cluster_precisions,
            cluster_means_sub=self.cluster_means_sub,
            cluster_mask_indices=self.cluster_mask_indices,
            min_across_clusters=self.mahalanobis_min_across_clusters,
            device=self.device,
        )

    # === Residual Mahalanobis methods ===
    # Scores anomalies using SAE reconstruction error (residual = x - decode(encode(x)))
    # with per-cluster Mahalanobis distance. Uses the full autoencoder (encoder + decoder)
    # accessed via self.net.model. Can be cleanly removed by deleting this section
    # and the "residual_mahalanobis" entries in distance_metric validation,
    # statistical_only, and freeze_encoder lists.

    def _fit_residual_mahalanobis(
        self,
        raw_data: np.ndarray,
        cluster_labels: np.ndarray,
    ) -> None:
        """Fit per-cluster Mahalanobis on SAE residuals.

        Delegates to :func:`cvdd_km_utils.fit_residual_mahalanobis`.
        """
        assert self.full_autoencoder is not None, (
            "residual_mahalanobis requires full_autoencoder "
            "(encoder+decoder). Pass full_model when initializing."
        )
        state = fit_residual_mahalanobis(
            full_autoencoder=self.full_autoencoder,
            raw_data=raw_data,
            cluster_labels=cluster_labels,
            n_clusters=self.n_clusters,
            mahalanobis_reg=self.mahalanobis_reg,
            mahalanobis_diag=self.mahalanobis_diag,
            device=self.device,
        )
        self.residual_means = state.means
        self.residual_precisions = state.precisions

    def _score_residual_mahalanobis(self, raw_data: np.ndarray) -> np.ndarray:
        """Score samples using per-cluster residual Mahalanobis.

        Delegates to :func:`cvdd_km_utils.score_residual_mahalanobis`.
        """
        assert (
            self.full_autoencoder is not None
        ), "residual_mahalanobis requires full_autoencoder"
        assert self.residual_means is not None
        return score_residual_mahalanobis(
            full_autoencoder=self.full_autoencoder,
            raw_data=raw_data,
            kmeans_index=self.kmeans,
            n_clusters=self.n_clusters,
            residual_means=self.residual_means,
            residual_precisions=self.residual_precisions,
            analysis_batch_size=self.analysis_batch_size,
            device=self.device,
        )

    def fit(self, data: np.ndarray, labels: np.ndarray) -> None:
        """Fit CVDD KMeans Masked on safe (normal) samples.

        This method overrides the parent's fit to add KMeans clustering
        and mask creation before the standard CVDD training.
        """
        # Filter safe samples
        safe_mask = labels == 1
        safe_data = data[safe_mask]
        min_samples_required = self.n_clusters

        if len(safe_data) < min_samples_required:
            raise ValueError(
                f"Insufficient safe samples for {self.n_clusters} clusters: "
                f"{len(safe_data)} < {min_samples_required}"
            )

        logger.info(
            f"Filtering safe samples: {len(safe_data)} out of {len(data)}"
        )

        # Store input dimension
        self.input_dim = safe_data.shape[1]

        # Apply scaler
        safe_data_scaled = self.scaler.fit_transform(safe_data)

        # Initialize network (encoder)
        if self.identity_encoder:
            self.net = nn.Identity()
            self.rep_dim = self.input_dim
            logger.info(
                f"Identity encoder: clustering/scoring in RAW activation "
                f"space (dim={self.input_dim})"
            )
        elif self.pretrained_encoder is not None:
            self.net = self.pretrained_encoder.to(self.device)
            logger.info("Using pretrained encoder")
        else:
            from unsupervised_llm_safety.anodet_algos.deep_svdd import (
                DeepSVDDNetwork,
            )

            self.net = DeepSVDDNetwork(self.input_dim, self.rep_dim).to(
                self.device
            )
            logger.info(
                f"Created DeepSVDD network: {self.input_dim} -> {self.rep_dim}"
            )

        # Enable gradients (but freeze encoder for Mahalanobis and
        # mask_disagreement modes — these are purely statistical methods
        # that don't benefit from encoder fine-tuning.)
        freeze_encoder = self.distance_metric in (
            "mahalanobis",
            "mask_disagreement",
            "hamming",
            "typicality",
            "residual_mahalanobis",
        )
        for param in self.net.parameters():
            param.requires_grad = not freeze_encoder
        if freeze_encoder:
            logger.info(
                "Froze encoder for Mahalanobis mode (no CVDD training)"
            )

        # Step 0 (if cluster_in_encoded_space): Encode data first
        if self.cluster_in_encoded_space:
            logger.info("Encoding data first (clustering in encoded space)...")
            with torch.no_grad():
                self.net.eval()
                encoded_data_list = []
                batch_size_encode = min(1024, len(safe_data_scaled))

                for i in range(0, len(safe_data_scaled), batch_size_encode):
                    batch = safe_data_scaled[i : i + batch_size_encode]
                    batch_tensor = torch.tensor(
                        batch, dtype=torch.float32, device=self.device
                    )
                    z = self.net(batch_tensor)
                    encoded_data_list.append(z.cpu().numpy())

                safe_data_encoded = np.vstack(encoded_data_list)
                logger.info(f"Encoded data shape: {safe_data_encoded.shape}")
                self.net.train()  # Set back to train mode for training
        else:
            safe_data_encoded = None

        # === Mask-then-cluster path ===
        # When cluster_in_masked_space=True, compute global mask FIRST
        # (from all-data firing frequencies), apply it, then cluster on
        # the masked (k-dimensional) features.
        if self.cluster_in_masked_space:
            assert (
                self.cluster_in_encoded_space
            ), "cluster_in_masked_space requires cluster_in_encoded_space"
            assert (
                self.global_mask
            ), "cluster_in_masked_space requires global_mask=True"
            logger.info(
                "=== Mask-then-cluster mode: computing global mask "
                "BEFORE clustering ==="
            )

            # Step 1a: Compute global firing frequencies on ALL encoded data
            # (no clusters needed — just count which features fire)
            binary = (safe_data_encoded > self.firing_threshold).astype(
                np.float32
            )
            firing_freq = binary.mean(axis=0)  # [embed_dim]

            # Step 1b: Create global mask (top-k by frequency)
            embed_dim = safe_data_encoded.shape[1]
            top_k_indices = np.argsort(firing_freq)[-self.k_mask :]
            mask_np = np.zeros(embed_dim, dtype=np.float32)
            mask_np[top_k_indices] = 1.0
            shared_mask = torch.tensor(mask_np, device=self.device)
            logger.info(
                f"Global mask: top-{self.k_mask} features "
                f"(top rate={firing_freq[top_k_indices[-1]]:.4f}, "
                f"kth rate={firing_freq[top_k_indices[0]]:.4f})"
            )

            # Store the mask indices for extracting masked features
            self._global_mask_indices = np.sort(top_k_indices)

            # Step 1c: Apply mask to get (n, k) features for clustering
            safe_data_masked = safe_data_encoded[:, self._global_mask_indices]
            logger.info(
                f"Masked data shape for clustering: "
                f"{safe_data_masked.shape}"
            )

            # Step 1d: Cluster on masked features
            cluster_labels, _ = self._cluster_raw_inputs(safe_data_masked)

            # Step 1e: Optional split/merge on masked features
            if self.outlier_threshold > 0:
                cluster_labels = self._split_large_union_clusters(
                    safe_data_masked,
                    cluster_labels,
                    self.outlier_threshold,
                )
            if self.min_cluster_size > 0:
                merge_result = merge_small_clusters(
                    data=safe_data_masked,
                    cluster_labels=cluster_labels,
                    n_clusters=self.n_clusters,
                    min_cluster_size=self.min_cluster_size,
                    max_union_size=self.max_union_size,
                )
                cluster_labels = merge_result.labels
                self.n_clusters = merge_result.n_clusters
                self.cluster_centers_raw = merge_result.centroids
                centroids_f32 = merge_result.centroids.astype(
                    np.float32
                ).copy()
                self.kmeans = faiss.Kmeans(
                    d=centroids_f32.shape[1],
                    k=self.n_clusters,
                    niter=1,
                    verbose=False,
                )
                self.kmeans.train(centroids_f32)
                self.kmeans.centroids = centroids_f32
                sizes = np.bincount(cluster_labels, minlength=self.n_clusters)
                logger.info(
                    f"After merge: {self.n_clusters} clusters, "
                    f"sizes min={sizes[sizes > 0].min()} "
                    f"max={sizes.max()}"
                )

            # Step 2: Set up masks (same global mask for all clusters)
            self.cluster_masks = {
                i: shared_mask for i in range(self.n_clusters)
            }
            self.masks_tensor = torch.stack(
                [self.cluster_masks[i] for i in range(self.n_clusters)]
            ).to(self.device)

            # Step 2b: Analyze firing patterns for frequency vectors
            # (needed for typicality scoring — compute on encoded data)
            cluster_stats = self._analyze_firing_patterns(
                safe_data_encoded, cluster_labels, is_encoded=True
            )
            self._cluster_union_indices = {
                cid: stats["firing_neuron_indices"]
                for cid, stats in cluster_stats.items()
            }

            # Store per-cluster frequency vectors
            freq_matrix = np.zeros(
                (self.n_clusters, embed_dim), dtype=np.float32
            )
            for cid, stats in cluster_stats.items():
                freq_matrix[cid] = stats["frequency"] / max(
                    stats["n_samples"], 1
                )
            self.cluster_frequency_vectors = torch.tensor(
                freq_matrix, dtype=torch.float32, device=self.device
            )

        else:
            # === Original path: cluster first, then mask ===

            # Step 1: Cluster inputs (raw or encoded)
            if self.cluster_in_encoded_space:
                cluster_labels, _ = self._cluster_raw_inputs(safe_data_encoded)
            else:
                cluster_labels, _ = self._cluster_raw_inputs(safe_data_scaled)

            # Step 1b: Split clusters with excessive union_support
            if self.outlier_threshold > 0:
                data_for_split = (
                    safe_data_encoded
                    if self.cluster_in_encoded_space
                    else safe_data_scaled
                )
                cluster_labels = self._split_large_union_clusters(
                    data_for_split, cluster_labels, self.outlier_threshold
                )

            # Step 1c: Merge small clusters
            if self.min_cluster_size > 0:
                data_for_merge = (
                    safe_data_encoded
                    if self.cluster_in_encoded_space
                    else safe_data_scaled
                )
                merge_result = merge_small_clusters(
                    data=data_for_merge,
                    cluster_labels=cluster_labels,
                    n_clusters=self.n_clusters,
                    min_cluster_size=self.min_cluster_size,
                    max_union_size=self.max_union_size,
                )
                cluster_labels = merge_result.labels
                self.n_clusters = merge_result.n_clusters
                self.cluster_centers_raw = merge_result.centroids

                # Rebuild FAISS index on new centroids
                centroids_f32 = merge_result.centroids.astype(
                    np.float32
                ).copy()
                self.kmeans = faiss.Kmeans(
                    d=centroids_f32.shape[1],
                    k=self.n_clusters,
                    niter=1,
                    verbose=False,
                )
                self.kmeans.train(centroids_f32)
                self.kmeans.centroids = centroids_f32

                sizes = np.bincount(cluster_labels, minlength=self.n_clusters)
                logger.info(
                    f"After merge: {self.n_clusters} clusters, "
                    f"sizes min={sizes[sizes > 0].min()} "
                    f"max={sizes.max()}"
                )

            # Step 2: Analyze firing patterns
            if self.cluster_in_encoded_space:
                cluster_stats = self._analyze_firing_patterns(
                    safe_data_encoded, cluster_labels, is_encoded=True
                )
            else:
                cluster_stats = self._analyze_firing_patterns(
                    safe_data_scaled, cluster_labels, is_encoded=False
                )

            # Store per-cluster union support indices for Mahalanobis
            self._cluster_union_indices = {
                cid: stats["firing_neuron_indices"]
                for cid, stats in cluster_stats.items()
            }

            # Step 3: Create per-cluster top-k masks and cache
            self.cluster_masks = self._create_cluster_masks(cluster_stats)
            self.masks_tensor = torch.stack(
                [self.cluster_masks[i] for i in range(self.n_clusters)]
            ).to(self.device)

            # Store per-cluster frequency vectors for typicality scoring
            first_cid = next(iter(cluster_stats))
            embedding_dim = len(cluster_stats[first_cid]["frequency"])
            freq_matrix = np.zeros(
                (self.n_clusters, embedding_dim), dtype=np.float32
            )
            for cid, stats in cluster_stats.items():
                freq_matrix[cid] = stats["frequency"] / max(
                    stats["n_samples"], 1
                )
            self.cluster_frequency_vectors = torch.tensor(
                freq_matrix, dtype=torch.float32, device=self.device
            )

        # Step 4: Initialize centers using cluster-specific masked embeddings
        print(
            "\nInitializing centers from cluster-specific masked embeddings..."
        )

        # Compute mean of masked embeddings for each cluster using
        # running sums to avoid GPU OOM with large datasets.
        with torch.no_grad():
            # Convert cluster labels to tensor
            cluster_labels_tensor = torch.tensor(
                cluster_labels, dtype=torch.long, device=self.device
            )

            # Determine which data to use and its length
            if self.cluster_in_encoded_space:
                data_to_process = safe_data_encoded
            else:
                data_to_process = safe_data_scaled

            # Determine embedding dim from a single forward pass
            sample_tensor = torch.tensor(
                data_to_process[:1], dtype=torch.float32, device=self.device
            )
            if self.cluster_in_encoded_space:
                embedding_dim = sample_tensor.shape[1]
            else:
                embedding_dim = self.net(sample_tensor).shape[1]

            # Running sums on CPU to avoid GPU OOM
            cluster_sums = torch.zeros(
                self.n_clusters, embedding_dim, dtype=torch.float64
            )
            cluster_counts = torch.zeros(self.n_clusters, dtype=torch.long)

            batch_size = min(1024, len(data_to_process))
            for i in range(0, len(data_to_process), batch_size):
                batch = data_to_process[i : i + batch_size]
                batch_clusters = cluster_labels_tensor[i : i + batch_size]
                batch_tensor = torch.tensor(
                    batch, dtype=torch.float32, device=self.device
                )

                # Encode if needed (skip if data is already encoded)
                if self.cluster_in_encoded_space:
                    z = batch_tensor  # Already encoded, no network pass
                else:
                    z = self.net(batch_tensor)  # [batch_size, embedding_dim]

                # Gather masks for this batch and apply
                batch_masks = self.masks_tensor[batch_clusters]
                z_masked = z * batch_masks  # [batch_size, embedding_dim]

                # Accumulate sums on CPU per cluster
                z_masked_cpu = z_masked.cpu().double()
                batch_clusters_cpu = batch_clusters.cpu()
                for cid in range(self.n_clusters):
                    mask = batch_clusters_cpu == cid
                    if mask.any():
                        cluster_sums[cid] += z_masked_cpu[mask].sum(dim=0)
                        cluster_counts[cid] += mask.sum()

            # Compute means
            initial_centers = []
            for cluster_id in range(self.n_clusters):
                count = int(cluster_counts[cluster_id].item())
                if count > 0:
                    cluster_mean = (
                        (cluster_sums[cluster_id] / count).float().numpy()
                    )
                    initial_centers.append(cluster_mean)
                    print(
                        f"Cluster {cluster_id}: initialized from "
                        f"{count} samples"
                    )
                else:
                    initial_centers.append(np.zeros(embedding_dim))
                    print(f"Cluster {cluster_id}: empty, using zero vector")

            initial_centers = np.array(initial_centers)

        # Convert to tensor and create centers module
        initial_centers_tensor = torch.tensor(
            initial_centers, dtype=torch.float32, device=self.device
        )
        self.centers_module = CVDDCenters(
            initial_centers_tensor, learnable=not self.fixed_centers
        ).to(self.device)

        # Step 5: Store cluster labels
        self.cluster_labels = torch.tensor(
            cluster_labels, dtype=torch.long, device=self.device
        )

        # Step 6: Create DataLoader with cluster information
        if self.cluster_in_encoded_space:
            # Use encoded data for training
            train_data = torch.tensor(safe_data_encoded, dtype=torch.float32)
        else:
            # Use raw scaled data for training
            train_data = torch.tensor(safe_data_scaled, dtype=torch.float32)

        safe_dataset = TensorDataset(
            train_data,
            torch.tensor(cluster_labels, dtype=torch.long),
        )
        train_loader = DataLoader(
            safe_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Snapshot encoder state before CVDD fine-tuning for health diagnostics
        self._pre_finetune_state_dict = copy.deepcopy(self.net.state_dict())
        logger.info("Saved pre-finetune encoder state for health diagnostics")

        # Store training data + cluster labels for diagnostics.
        # When train_encoder=True, store RAW data so health diagnostics
        # can re-encode through pre/post encoders to see actual changes.
        if self.train_encoder:
            self._diagnostic_data = safe_data_scaled
            self._diagnostic_encode_through_encoder = True
        elif self.cluster_in_encoded_space:
            self._diagnostic_data = safe_data_encoded
            self._diagnostic_encode_through_encoder = False
        else:
            self._diagnostic_data = safe_data_scaled
            self._diagnostic_encode_through_encoder = False
        self._diagnostic_cluster_labels = cluster_labels

        # Step 6b: Compute per-cluster covariances if using Mahalanobis
        if self.distance_metric == "mahalanobis":
            if not self.use_masking and self._cluster_union_indices:
                dims = [len(v) for v in self._cluster_union_indices.values()]
                logger.info(
                    f"Computing per-cluster covariance matrices "
                    f"(union support subspace: "
                    f"mean={np.mean(dims):.0f}, "
                    f"max={np.max(dims)})..."
                )
            else:
                logger.info("Computing per-cluster covariance matrices...")
            covariance_data = (
                safe_data_encoded
                if self.cluster_in_encoded_space
                else safe_data_scaled
            )
            self._compute_cluster_covariances(covariance_data, cluster_labels)

        # Step 6c: Compute residual Mahalanobis if using that metric
        if self.distance_metric == "residual_mahalanobis":
            logger.info("Computing per-cluster residual Mahalanobis...")
            self._fit_residual_mahalanobis(safe_data_scaled, cluster_labels)

        # Step 7: Train.
        # Statistical-only metrics skip training entirely.
        # When cluster_in_encoded_space + fixed_centers, there's nothing to
        # train unless train_encoder=True (variance shrinkage mode).
        statistical_only = self.distance_metric in (
            "mahalanobis",
            "mask_disagreement",
            "typicality",
            "residual_mahalanobis",
        )
        no_trainable_params = (
            self.cluster_in_encoded_space
            and self.fixed_centers
            and not self.train_encoder
        )

        if statistical_only or no_trainable_params:
            logger.info("Skipping CVDD training (no trainable parameters)")
        else:
            # When train_encoder=True with pre-encoded clustering, we need
            # raw data so the encoder gets gradients.
            if self.train_encoder and self.cluster_in_encoded_space:
                for param in self.net.parameters():
                    param.requires_grad = True
                logger.info("Training encoder (variance shrinkage mode)")
                train_data = torch.tensor(
                    safe_data_scaled, dtype=torch.float32
                )
                train_loader = DataLoader(
                    TensorDataset(
                        train_data,
                        torch.tensor(cluster_labels, dtype=torch.long),
                    ),
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=0,
                )
            self._train_cvdd(train_loader)

        # Step 8: Compute score normalization.
        # Use encoded data. If the encoder was just trained, re-encode.
        if self.cluster_in_encoded_space and self.train_encoder:
            safe_data_encoded = self._encode_data(safe_data_scaled)

        if self.cluster_in_encoded_space:
            norm_data = torch.tensor(safe_data_encoded, dtype=torch.float32)
        else:
            norm_data = torch.tensor(safe_data_scaled, dtype=torch.float32)

        norm_dataset = TensorDataset(
            norm_data,
            torch.tensor(cluster_labels, dtype=torch.long),
        )
        norm_loader = DataLoader(
            norm_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )
        # Residual Mahalanobis computes normalization from its own scores
        if self.distance_metric == "residual_mahalanobis":
            residual_scores = self._score_residual_mahalanobis(
                safe_data_scaled
            )
            self.score_low = float(np.percentile(residual_scores, 5))
            self.score_high = float(np.percentile(residual_scores, 95))
            logger.info(
                f"Residual Mahalanobis normalization: "
                f"p5={self.score_low:.4f}, p95={self.score_high:.4f}"
            )
        else:
            self._compute_score_normalization(norm_loader)

        logger.info("Training complete!")
        self._fitted = True

    def _encode_data(self, raw_data: np.ndarray) -> np.ndarray:
        """Encode raw hidden states through the encoder network."""
        self.net.eval()
        encoded_chunks = []
        batch_size = min(1024, len(raw_data))
        with torch.no_grad():
            for i in range(0, len(raw_data), batch_size):
                batch = torch.tensor(
                    raw_data[i : i + batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                encoded_chunks.append(self.net(batch).cpu().numpy())
        return np.vstack(encoded_chunks)

    def _train_cvdd(self, train_loader: DataLoader) -> None:
        """Override to properly pass cluster assignments to loss computation."""
        self._log_training_start()

        # Setup metrics file for detailed logging
        self._setup_metrics_file()

        # Setup optimizer
        params = self._get_trainable_params()
        optimizer, scheduler = self._setup_optimizer_and_scheduler(params)

        # Training loop - modified to pass cluster labels
        self.net.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch_data, batch_clusters in train_loader:
                batch_data = batch_data.to(self.device)
                batch_clusters = batch_clusters.to(self.device)

                optimizer.zero_grad()

                # Forward pass through encoder.
                # Skip if data is already encoded (pre-encoded mode)
                # unless train_encoder=True (raw data, needs encoding).
                if self.cluster_in_encoded_space and not self.train_encoder:
                    z = batch_data
                else:
                    z = self.net(batch_data)

                # Compute loss with cluster assignments
                loss = self._compute_batch_loss_with_clusters(
                    z, batch_clusters, epoch
                )

                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(params, max_norm=10.0)

                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # Step scheduler
            scheduler.step()

            avg_loss = epoch_loss / n_batches

            # Periodically recompute Mahalanobis covariances during training.
            # The encoder changes each epoch, making the precision matrices
            # stale. Recomputing every 20 epochs keeps the distance metric
            # aligned with the current encoder state so the loss signal
            # remains meaningful as clusters tighten.
            if (
                self.distance_metric == "mahalanobis"
                and self._diagnostic_data is not None
                and self._diagnostic_cluster_labels is not None
                and (epoch + 1) % 20 == 0
                and epoch < self.n_epochs - 1
            ):
                logger.info(
                    f"Epoch {epoch + 1}: recomputing Mahalanobis covariances"
                )
                self.net.eval()
                self._compute_cluster_covariances(
                    self._diagnostic_data,
                    self._diagnostic_cluster_labels,
                )
                self.net.train()

            # Only compute expensive metrics when logging
            if epoch % 20 == 0 or epoch == self.n_epochs - 1:
                metrics = self._get_detailed_metrics(train_loader, epoch)
                utilization = None

                if epoch % 40 == 0:
                    utilization = self._compute_center_utilization(
                        train_loader
                    )
                    logger.info(
                        f"Epoch {epoch + 1}/{self.n_epochs}, "
                        f"Loss: {avg_loss:.6f}, "
                        f"Center util: {utilization:.1%}, "
                        f"Union mask: {metrics.get('union_mask_size', 'N/A')}"
                    )
                else:
                    logger.info(
                        f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {avg_loss:.6f}"
                    )

                self._save_metrics_to_file(
                    epoch, avg_loss, metrics, utilization
                )

        logger.info("CVDD training complete")
        if self.metrics_file_path:
            logger.info(f"Training metrics saved to: {self.metrics_file_path}")

    def _compute_score_normalization(self, train_loader: DataLoader) -> None:
        """Compute percentiles for score normalization on training data using masked L1 distances.

        Override parent's method because:
        - Parent uses: L2 squared distance with soft-minimum
        - This class uses: masked L1 distance (matches score_samples method)
        - Normalization must match the actual scoring method for proper calibration

        Note: train_loader must provide (data, cluster_labels) tuples. Cluster labels
        are already computed during fit() and passed in, so no re-clustering is needed.
        """
        logger.info("Computing score normalization parameters...")

        train_scores = []
        assert self.net is not None
        assert self.masks_tensor is not None
        self.net.eval()

        with torch.no_grad():
            for batch_data in train_loader:
                batch_data_tensor = batch_data[0].to(self.device)
                batch_clusters = batch_data[1].to(self.device)

                if self.cluster_in_encoded_space:
                    z = batch_data_tensor
                else:
                    z = self.net(batch_data_tensor)

                if self.distance_metric == "mahalanobis":
                    batch_clusters_np = batch_clusters.cpu().numpy()
                    distances = self._compute_mahalanobis_distances_batch(
                        z, batch_clusters_np
                    )
                elif self.distance_metric == "mask_disagreement":
                    batch_clusters_np = batch_clusters.cpu().numpy()
                    distances = self._compute_mask_disagreement_batch(
                        z, batch_clusters_np
                    )
                elif self.distance_metric == "typicality":
                    assert self.cluster_frequency_vectors is not None
                    batch_freq = self.cluster_frequency_vectors[batch_clusters]
                    q = torch.softmax(
                        batch_freq / self.typicality_temperature,
                        dim=1,
                    )
                    active = (z > 0).float()
                    distances = -(q * active).sum(dim=1)
                else:
                    all_centers = self.centers_module()
                    batch_masks = self.masks_tensor[batch_clusters]
                    batch_centers = all_centers[batch_clusters]

                    z_masked = z * batch_masks
                    centers_masked = batch_centers * batch_masks

                    if self.distance_metric == "l1":
                        distances = torch.abs(z_masked - centers_masked).sum(
                            dim=1
                        )
                    else:
                        distances = ((z_masked - centers_masked) ** 2).sum(
                            dim=1
                        )
                train_scores.append(distances)

        all_distances = torch.cat(train_scores).cpu().numpy()

        # Compute global percentiles (always needed as fallback)
        self.score_low = float(np.percentile(all_distances, 5))
        self.score_high = float(np.percentile(all_distances, 95))

        if np.isclose(self.score_low, self.score_high):
            logger.warning(
                f"All training scores are identical: {self.score_low:.3f}. "
                "Using alternative normalization for scoring."
            )
        else:
            logger.info(
                f"Global score range: [{self.score_low:.3f}, {self.score_high:.3f}] "
                f"(5th-95th percentile)"
            )

        # Per-cluster percentiles
        if self.score_normalization == "percentile_train_per_cluster":
            # Collect cluster assignments matching the distances
            all_cluster_labels = []
            with torch.no_grad():
                for batch_data in train_loader:
                    batch_clusters = batch_data[1].cpu().numpy()
                    all_cluster_labels.append(batch_clusters)
            all_cluster_labels_arr = np.concatenate(all_cluster_labels)

            self.score_low_per_cluster = {}
            self.score_high_per_cluster = {}
            min_samples_per_cluster = 20

            for k in range(self.n_clusters):
                mask = all_cluster_labels_arr == k
                cluster_scores = all_distances[mask]
                if len(cluster_scores) >= min_samples_per_cluster:
                    self.score_low_per_cluster[k] = float(
                        np.percentile(cluster_scores, 5)
                    )
                    self.score_high_per_cluster[k] = float(
                        np.percentile(cluster_scores, 95)
                    )
                else:
                    # Fallback to global for small clusters
                    self.score_low_per_cluster[k] = self.score_low
                    self.score_high_per_cluster[k] = self.score_high

            n_local = sum(
                1
                for k in range(self.n_clusters)
                if k in self.score_low_per_cluster
                and self.score_low_per_cluster[k] != self.score_low
            )
            logger.info(
                f"Per-cluster normalization: {n_local}/{self.n_clusters} clusters "
                f"with local percentiles (min {min_samples_per_cluster} samples)"
            )

    def _compute_batch_loss(
        self, z: torch.Tensor, epoch: int, **kwargs
    ) -> torch.Tensor:
        """This should not be called since we override _train_cvdd."""
        raise NotImplementedError(
            "CVDDKMeansMasked requires cluster assignments. "
            "This method should not be called - _train_cvdd should be overridden."
        )

    def _compute_batch_loss_with_clusters(
        self,
        z: torch.Tensor,
        batch_clusters: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        """Compute masked loss for a batch using cluster assignments.

        Each sample is compared only to its assigned cluster center.
        No soft assignment is needed since clusters are pre-determined.
        """
        assert batch_clusters is not None, "Cluster assignments are required"
        assert self.masks_tensor is not None

        if self.distance_metric == "mahalanobis":
            batch_clusters_np = batch_clusters.cpu().numpy()
            distances = self._compute_mahalanobis_distances_batch(
                z, batch_clusters_np
            )
            return distances.mean()

        all_centers = self.centers_module()
        batch_masks = self.masks_tensor[batch_clusters]
        batch_centers = all_centers[batch_clusters]

        z_masked = z * batch_masks
        centers_masked = batch_centers * batch_masks

        if self.distance_metric == "l1":
            losses = torch.abs(z_masked - centers_masked).sum(dim=1)
        else:
            losses = ((z_masked - centers_masked) ** 2).sum(dim=1)

        loss = losses.mean()

        # Step 3: L1 feature penalty on SAE activations
        if self.l1_feature_penalty > 0:
            # z is post-ReLU SAE output (already non-negative)
            l1_term = z.abs().sum(dim=1).mean()
            loss = loss + self.l1_feature_penalty * l1_term

        return loss

    def score_samples(self, data: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for samples."""
        self.net.eval()

        with torch.no_grad():
            # Apply scaler (same as training) before clustering
            data_scaled = self.scaler.transform(data)

            # Residual Mahalanobis uses its own scoring pipeline
            if self.distance_metric == "residual_mahalanobis":
                raw_scores = self._score_residual_mahalanobis(data_scaled)
                # Normalize using percentile from training
                if self.score_normalization == "percentile_train":
                    normalized = (raw_scores - self.score_low) / max(
                        self.score_high - self.score_low, 1e-10
                    )
                    return np.clip(normalized, 0.0, 1.0)
                return raw_scores

            if self.cluster_in_encoded_space:
                # Encode data first, then cluster in encoded space
                encoded_data_list = []
                batch_size_encode = min(
                    self.analysis_batch_size, len(data_scaled)
                )

                for i in range(0, len(data_scaled), batch_size_encode):
                    batch = data_scaled[i : i + batch_size_encode]
                    batch_tensor = torch.tensor(
                        batch, dtype=torch.float32, device=self.device
                    )
                    z = self.net(batch_tensor)
                    encoded_data_list.append(z.cpu().numpy())

                data_encoded = np.vstack(encoded_data_list)
            else:
                data_encoded = None

            assert (
                self.kmeans is not None
            ), "Model must be fitted before scoring"

            # Cluster assignment always uses magnitude centroids
            # (L0 only affects initial clustering, not inference)
            if self.cluster_in_masked_space:
                # Mask-then-cluster: assign using masked features
                data_for_assign = data_encoded[
                    :, self._global_mask_indices
                ].astype(np.float32)
            elif self.cluster_in_encoded_space:
                data_for_assign = data_encoded.astype(np.float32)
            else:
                data_for_assign = data_scaled.astype(np.float32)

            _, cluster_assignments = self.kmeans.index.search(
                data_for_assign, 1
            )
            cluster_assignments = cluster_assignments.flatten()

            assert self.masks_tensor is not None

            all_scores = []
            batch_size = min(self.analysis_batch_size, len(data))
            data_to_process = (
                data_encoded if self.cluster_in_encoded_space else data_scaled
            )

            for i in range(0, len(data_to_process), batch_size):
                batch = data_to_process[i : i + batch_size]
                batch_tensor = torch.tensor(
                    batch, dtype=torch.float32, device=self.device
                )

                if self.cluster_in_encoded_space:
                    z = batch_tensor
                else:
                    z = self.net(batch_tensor)

                batch_cluster_assignments = cluster_assignments[
                    i : i + batch_size
                ]

                if self.distance_metric == "mahalanobis":
                    distances = self._compute_mahalanobis_distances_batch(
                        z, batch_cluster_assignments
                    )
                elif self.distance_metric == "mask_disagreement":
                    distances = self._compute_mask_disagreement_batch(
                        z, batch_cluster_assignments
                    )
                elif self.distance_metric == "typicality":
                    # Softmax-frequency typicality score (Step 4)
                    # Higher = more anomalous (atypical)
                    assert self.cluster_frequency_vectors is not None
                    batch_cluster_tensor = torch.tensor(
                        batch_cluster_assignments,
                        dtype=torch.long,
                        device=self.device,
                    )
                    # Get frequency vectors for assigned clusters
                    batch_freq = self.cluster_frequency_vectors[
                        batch_cluster_tensor
                    ]  # [B, d]
                    # Softmax with temperature
                    q = torch.softmax(
                        batch_freq / self.typicality_temperature, dim=1
                    )  # [B, d]
                    # Active components in sample
                    active = (z > 0).float()  # [B, d]
                    # Typicality = sum of softmax probs at active positions
                    # More typical → higher sum → lower anomaly score
                    # So distance = negative sum (higher = more anomalous)
                    typicality_sum = (q * active).sum(dim=1)
                    distances = -typicality_sum
                else:
                    all_centers = self.centers_module()
                    batch_cluster_tensor = torch.tensor(
                        batch_cluster_assignments,
                        dtype=torch.long,
                        device=self.device,
                    )

                    batch_masks = self.masks_tensor[batch_cluster_tensor]
                    batch_centers = all_centers[batch_cluster_tensor]
                    z_masked = z * batch_masks
                    centers_masked = batch_centers * batch_masks

                    if self.distance_metric == "l1":
                        distances = torch.abs(z_masked - centers_masked).sum(
                            dim=1
                        )
                    else:
                        distances = ((z_masked - centers_masked) ** 2).sum(
                            dim=1
                        )

                all_scores.append(distances.cpu().numpy())

            scores = np.concatenate(all_scores)

            # Apply configurable normalization
            if self.score_normalization == "percentile_train":
                # Global normalization: single score_low/score_high for all clusters
                assert (
                    self.score_low is not None and self.score_high is not None  # type: ignore[has-type]
                )
                score_low: float = cast(float, self.score_low)  # type: ignore[has-type]
                score_high: float = cast(float, self.score_high)  # type: ignore[has-type]

                scores = (scores - score_low) / (
                    score_high - score_low + 1e-10
                )
                scores = np.clip(scores, 0, 1)
            elif self.score_normalization == "percentile_train_per_cluster":
                # Per-cluster normalization: each cluster has its own percentiles
                assert self.score_low_per_cluster is not None
                assert self.score_high_per_cluster is not None

                # Build per-sample low/high arrays from cluster assignments
                global_low = cast(float, self.score_low)  # type: ignore[has-type]
                global_high = cast(float, self.score_high)  # type: ignore[has-type]
                lows = np.array(
                    [
                        self.score_low_per_cluster.get(int(c), global_low)
                        for c in cluster_assignments
                    ]
                )
                highs = np.array(
                    [
                        self.score_high_per_cluster.get(int(c), global_high)
                        for c in cluster_assignments
                    ]
                )
                scores = (scores - lows) / (highs - lows + 1e-10)
                scores = np.clip(scores, 0, 1)
            else:  # minmax_test
                # Deep SVDD style: normalize on current test batch
                s_min, s_max = scores.min(), scores.max()
                if s_max > s_min:
                    scores = (scores - s_min) / (s_max - s_min)
                else:
                    scores = np.zeros_like(scores)

        return scores

    def _get_detailed_metrics(
        self, train_loader: DataLoader, epoch: int
    ) -> Dict[str, Any]:
        """Get detailed metrics for logging."""
        # Get parent's metrics first
        metrics = super()._get_detailed_metrics(train_loader, epoch)

        assert self.masks_tensor is not None

        with torch.no_grad():
            # Union mask size (masks are fixed, but useful to track)
            union_mask = self.masks_tensor.max(dim=0)[0]
            metrics["union_mask_size"] = union_mask.sum().item()

            # Sample a batch to compute distance statistics
            sample_batch = None
            sample_clusters = None
            for batch_data, batch_clusters in train_loader:
                sample_batch = batch_data[: min(512, len(batch_data))].to(
                    self.device
                )
                sample_clusters = batch_clusters[
                    : min(512, len(batch_clusters))
                ].to(self.device)
                break

            if sample_batch is not None and len(sample_batch) > 0:
                if self.cluster_in_encoded_space and not self.train_encoder:
                    z = sample_batch
                else:
                    z = self.net(sample_batch)

                all_centers = self.centers_module()
                batch_masks = self.masks_tensor[sample_clusters]
                batch_centers = all_centers[sample_clusters]
                z_masked = z * batch_masks
                centers_masked = batch_centers * batch_masks

                if self.distance_metric == "l1":
                    batch_distances = torch.abs(z_masked - centers_masked).sum(
                        dim=1
                    )
                else:
                    batch_distances = ((z_masked - centers_masked) ** 2).sum(
                        dim=1
                    )

                distances_np = batch_distances.cpu().numpy()
                metrics["train_dist_mean"] = float(np.mean(distances_np))
                metrics["train_dist_std"] = float(np.std(distances_np))
                metrics["train_dist_min"] = float(np.min(distances_np))
                metrics["train_dist_max"] = float(np.max(distances_np))
                metrics["train_dist_median"] = float(np.median(distances_np))
                if metrics["train_dist_mean"] > 0:
                    metrics["train_dist_cv"] = float(
                        metrics["train_dist_std"] / metrics["train_dist_mean"]
                    )
                else:
                    metrics["train_dist_cv"] = 0.0

        return metrics

    def _compute_center_utilization(self, train_loader: DataLoader) -> float:
        """Compute what fraction of centers are being used.

        For CVDDKMeansMasked, each sample is assigned to its cluster's center,
        so we just need to check which clusters appear in the training data.
        """
        utilized_clusters = set()

        with torch.no_grad():
            # *train_loader* always wraps a two-tensor TensorDataset of
            # (data, cluster_labels) -- both loaders built in fit() are, and
            # _train_cvdd's own epoch loop unpacks exactly two per batch
            # before this is ever reached.
            for _, batch_clusters in train_loader:
                # Add all cluster IDs in this batch to the utilized set
                utilized_clusters.update(batch_clusters.cpu().numpy())

        # Since centers correspond to clusters, utilization is based on unique clusters seen
        return len(utilized_clusters) / self.n_clusters

    def _setup_metrics_file(self, output_dir: Optional[str] = None) -> None:
        """Set up metrics file for detailed logging.

        Args:
            output_dir: Directory to save metrics file. If None, uses analysis_output_dir.
        """
        if output_dir is None:
            output_dir = self.analysis_output_dir

        metrics_dir = Path(output_dir)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file_path = metrics_dir / "training_metrics.json"
        self.metrics_history = []
        logger.info(f"Metrics will be saved to: {self.metrics_file_path}")

    def _save_metrics_to_file(
        self,
        epoch: int,
        avg_loss: float,
        metrics: Dict[str, Any],
        center_utilization: Optional[float] = None,
    ) -> None:
        """Save detailed metrics to JSON file.

        Args:
            epoch: Current epoch number
            avg_loss: Average loss for this epoch
            metrics: Dictionary of detailed metrics from _get_detailed_metrics
            center_utilization: Center utilization fraction (computed separately)
        """
        if self.metrics_file_path is None:
            return

        epoch_metrics: Dict[str, Union[int, float, str, bool]] = {
            "epoch": epoch,
            "loss": float(avg_loss),
        }

        # Add all metrics, converting to JSON-serializable types
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                epoch_metrics[key] = float(value)
            elif isinstance(value, (str, bool)):
                epoch_metrics[key] = value
            elif isinstance(value, (np.integer, np.floating)):
                epoch_metrics[key] = float(value)
            elif isinstance(value, torch.Tensor):
                epoch_metrics[key] = float(value.item())
            else:
                # Skip complex types
                pass

        # Add center utilization if provided
        if center_utilization is not None:
            epoch_metrics["center_utilization"] = float(center_utilization)

        self.metrics_history.append(epoch_metrics)

        # Save to file (append mode would be better, but JSON requires full rewrite)
        # For efficiency, we could save every N epochs, but for now save every time
        try:
            with open(self.metrics_file_path, "w") as f:
                json.dump(self.metrics_history, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metrics to file: {e}")

    def _log_training_start(self) -> None:
        """Log training start message."""
        mask_status = "masked" if self.use_masking else "unmasked"
        logger.info(
            f"Starting CVDD training with {mask_status} {self.distance_metric.upper()} loss..."
        )

    def state_dict(self) -> dict:
        """Return the state dictionary for serialization.

        Always saves net_state_dict regardless of whether pretrained_encoder was used.
        The checkpoint should contain the final trained weights.
        """
        self._check_fitted()
        assert self.net is not None
        assert self.centers_module is not None

        # Always save network state - the checkpoint contains the final trained weights
        scaler_mean = getattr(self.scaler, "mean_", None)
        scaler_scale = getattr(self.scaler, "scale_", None)
        state = {
            "centers_module_state_dict": self.centers_module.state_dict(),
            "scaler_mean": scaler_mean,
            "scaler_scale": scaler_scale,
            "score_low": self.score_low,
            "score_high": self.score_high,
            "input_dim": self.input_dim,
            "uses_pretrained_encoder": self.pretrained_encoder is not None,
            "net_state_dict": self.net.state_dict(),  # Always save
        }

        return state

    def save(self, path: str) -> None:
        """Save the model to disk with all necessary information for loading."""
        assert (
            self.score_low is not None and self.score_high is not None  # type: ignore[has-type]
        ), "Model must be trained before saving"
        assert (
            self.net is not None
        ), "Network must be initialized before saving"
        assert (
            self.centers_module is not None
        ), "Centers must be initialized before saving"

        score_low = cast(float, self.score_low)  # type: ignore[has-type]
        score_high = cast(float, self.score_high)  # type: ignore[has-type]

        # Save complete state with method_name for defense loading
        state = {
            # Critical: method_name is used by AnomalyDetectionDefense to determine loading strategy
            "method_name": "cvdd_kmeans_masked",
            "model_type": "CVDDKMeansMasked",
            # Network state - always save the final trained weights
            "net_state_dict": self.net.state_dict(),
            # Model-specific components
            "centers": self.centers_module.centers.data.cpu().numpy(),
            "cluster_masks": {
                int(k): v.cpu().numpy() for k, v in self.cluster_masks.items()
            },
            "cluster_centers_raw": self.cluster_centers_raw,
            # Normalization parameters
            "score_low": score_low,
            "score_high": score_high,
            # Configuration - all parameters needed for reconstruction
            "config": {
                "n_clusters": self.n_clusters,
                "k_mask": self.k_mask,
                "firing_threshold": self.firing_threshold,
                "fixed_centers": self.fixed_centers,
                "rep_dim": self.rep_dim,
                "cluster_in_encoded_space": self.cluster_in_encoded_space,
                "cluster_method": self.cluster_method,
                "distance_metric": self.distance_metric,
                "use_masking": self.use_masking,
                "mask_variant": self.mask_variant,
                "mask_coverage": self.mask_coverage,
                "score_normalization": self.score_normalization,
                "l1_feature_penalty": self.l1_feature_penalty,
                "typicality_temperature": self.typicality_temperature,
                "input_dim": self.input_dim,  # Critical for loading
                "mahalanobis_reg": self.mahalanobis_reg,
                "mahalanobis_diag": self.mahalanobis_diag,
                "mahalanobis_min_across_clusters": self.mahalanobis_min_across_clusters,
                "global_mask": self.global_mask,
                "cluster_in_masked_space": self.cluster_in_masked_space,
                # An identity encoder has no parameters, so net_state_dict is
                # empty. load() needs this flag to tell that apart from a
                # truncated checkpoint.
                "identity_encoder": self.identity_encoder,
                # Load-bearing for reproduction, not just for memory. Scoring
                # batches the data, so the batch size changes float summation
                # order and therefore the scores. Measured: reloading a run
                # fitted at 32 with the constructor default of 256 moves scores
                # by up to 1.9e-02 and the 3-slice AUROC by ~9e-08; reloading
                # at the SAME batch size reproduces bit-exactly (max|diff| =
                # 0.0 across all 29 datasets, even on a different GPU model).
                # It was previously absent here, so every reload silently used
                # 256 while the runs used 32 from cvdd_kmeans_masked.yaml.
                "analysis_batch_size": self.analysis_batch_size,
            },
            # L0 clustering: binary centroids for cluster assignment
            "cluster_centers_binary": self.cluster_centers_binary,
            # Per-cluster frequency vectors for typicality scoring
            "cluster_frequency_vectors": (
                self.cluster_frequency_vectors.cpu().numpy()
                if self.cluster_frequency_vectors is not None
                else None
            ),
            # Global mask indices for cluster_in_masked_space mode
            "global_mask_indices": (
                self._global_mask_indices
                if hasattr(self, "_global_mask_indices")
                and self._global_mask_indices is not None
                else None
            ),
            # Training data cluster assignments (for post-hoc analysis)
            "cluster_labels": (
                self.cluster_labels.cpu().numpy()
                if self.cluster_labels is not None
                else None
            ),
            # Per-cluster normalization parameters
            "score_low_per_cluster": self.score_low_per_cluster,
            "score_high_per_cluster": self.score_high_per_cluster,
            # Pre-finetune encoder state for health diagnostics
            "pre_finetune_encoder_state": self._pre_finetune_state_dict,
            # Legacy compatibility
            "input_dim": self.input_dim,
            # Nested state dict for compatibility with some loading paths
            "state": {
                "net_state_dict": self.net.state_dict(),
                "input_dim": self.input_dim,
                "centers_module_state_dict": self.centers_module.state_dict(),
                "score_low": score_low,
                "score_high": score_high,
            },
        }

        # Save scaler parameters if using StandardScaler (not IdentityScaler)
        if (
            hasattr(self.scaler, "mean_")
            and self.scaler.mean_ is not None
            and hasattr(self.scaler, "scale_")
            and self.scaler.scale_ is not None
        ):
            state["scaler_mean"] = self.scaler.mean_
            state["scaler_scale"] = self.scaler.scale_
            state["state"]["scaler_mean"] = self.scaler.mean_
            state["state"]["scaler_scale"] = self.scaler.scale_

        # Save per-cluster Mahalanobis state if present
        if self.cluster_precisions is not None:
            state["mahalanobis_state"] = {
                "cluster_precisions": {
                    int(k): v.cpu().numpy()
                    for k, v in self.cluster_precisions.items()
                },
                "cluster_means_sub": {
                    int(k): v.cpu().numpy()
                    for k, v in (self.cluster_means_sub or {}).items()
                },
                "cluster_mask_indices": {
                    int(k): v.cpu().numpy()
                    for k, v in (self.cluster_mask_indices or {}).items()
                },
            }

        torch.save(state, path)
        logger.info(
            f"Model saved to {path} with method_name='cvdd_kmeans_masked'"
        )

    def load(self, path: str) -> None:
        """Load the model from disk - robust loading with multiple fallbacks."""
        state = torch.load(path, map_location=self.device, weights_only=False)

        # Extract config - handle multiple possible locations
        config = state.get("config", {})
        if not config and "state" in state and "config" in state["state"]:
            config = state["state"]["config"]

        # Extract input_dim - try multiple locations
        self.input_dim = None
        if "input_dim" in config:
            self.input_dim = config["input_dim"]
        elif "input_dim" in state:
            self.input_dim = state["input_dim"]
        elif "state" in state and "input_dim" in state["state"]:
            self.input_dim = state["state"]["input_dim"]

        if self.input_dim is None:
            # Try to infer from network state dict
            net_state_dict = state.get("net_state_dict", {})
            if "state" in state and "net_state_dict" in state["state"]:
                net_state_dict = state["state"]["net_state_dict"]

            for key in ("fc1.weight", "model.0.weight", "0.weight"):
                if key in net_state_dict:
                    self.input_dim = net_state_dict[key].shape[1]
                    logger.warning(
                        f"Inferred input_dim={self.input_dim} from {key}"
                    )
                    break

            if self.input_dim is None:
                raise ValueError(
                    "Cannot determine input_dim from checkpoint. "
                    "Please re-save the model with the updated save() method."
                )

        # Load config parameters
        self.n_clusters = config.get("n_clusters", 3)
        self.k_mask = config.get("k_mask", 100)
        self.firing_threshold = config.get("firing_threshold", 0)
        self.fixed_centers = config.get("fixed_centers", False)
        self.rep_dim = config.get("rep_dim", 32)
        self.cluster_in_encoded_space = config.get(
            "cluster_in_encoded_space", False
        )
        self.cluster_method = config.get("cluster_method", "default")
        self.distance_metric = config.get("distance_metric", "l1")
        self.use_masking = config.get("use_masking", True)
        self.mask_variant = config.get("mask_variant", "top_k")
        self.mask_coverage = config.get("mask_coverage", 0.9)
        self.l1_feature_penalty = config.get("l1_feature_penalty", 0.0)
        self.typicality_temperature = config.get("typicality_temperature", 1.0)
        self.score_normalization = config.get(
            "score_normalization", "percentile_train"
        )
        # These four were written by save() and never read back.
        # mahalanobis_min_across_clusters is used at scoring time, so a
        # checkpoint fitted with it True silently re-scored as False.
        self.mahalanobis_min_across_clusters = config.get(
            "mahalanobis_min_across_clusters",
            self.mahalanobis_min_across_clusters,
        )
        self.mahalanobis_reg = config.get(
            "mahalanobis_reg", self.mahalanobis_reg
        )
        self.mahalanobis_diag = config.get(
            "mahalanobis_diag", self.mahalanobis_diag
        )
        self.global_mask = config.get("global_mask", self.global_mask)
        self.identity_encoder = config.get(
            "identity_encoder", self.identity_encoder
        )
        self.cluster_in_masked_space = config.get(
            "cluster_in_masked_space", False
        )
        # Restore the fit-time scoring batch size. Scoring batches the data, so
        # this changes float summation order and therefore the scores: a run
        # fitted at 32 and reloaded at the constructor default of 256 differs by
        # up to 1.9e-02 on individual scores, whereas reloading at the SAME
        # batch size is bit-exact. Checkpoints written before this key existed
        # fall back to the current instance value, so pass the fit-time value to
        # the constructor when reproducing from an older checkpoint.
        self.analysis_batch_size = config.get(
            "analysis_batch_size", self.analysis_batch_size
        )

        # Initialize network - ALWAYS reconstruct from saved weights, never use pretrained_encoder
        # The pretrained_encoder is just the starting point; self.net contains the trained weights
        net_state_dict = state.get("net_state_dict", {})
        if (
            not net_state_dict
            and "state" in state
            and "net_state_dict" in state["state"]
        ):
            net_state_dict = state["state"]["net_state_dict"]

        if not net_state_dict and self.identity_encoder:
            # Correct, not truncated: nn.Identity() has no parameters. Without
            # this every cvdd_km_raw_no_mask checkpoint was unloadable.
            self.net = nn.Identity().to(self.device)
            logger.info("Restored identity encoder (no parameters to load)")
        elif not net_state_dict:
            raise ValueError("No net_state_dict found in checkpoint")
        # Check if it's a DeepSVDD or Sequential network
        elif "fc1.weight" in net_state_dict:
            # DeepSVDD network
            from unsupervised_llm_safety.anodet_algos.deep_svdd import (
                DeepSVDDNetwork,
            )

            self.net = DeepSVDDNetwork(self.input_dim, self.rep_dim).to(
                self.device
            )
            logger.info(
                f"Created DeepSVDDNetwork: {self.input_dim} -> {self.rep_dim}"
            )
        else:
            # Sequential/Autoencoder - reconstruct from state dict
            self.net = self._reconstruct_sequential_from_state_dict(
                net_state_dict
            )
            logger.info("Reconstructed Sequential encoder from state dict")

        # Load network weights
        net_state_dict = state.get("net_state_dict", {})
        if (
            not net_state_dict
            and "state" in state
            and "net_state_dict" in state["state"]
        ):
            net_state_dict = state["state"]["net_state_dict"]

        # Handle "model." prefix in keys for Sequential models
        if any(key.startswith("model.") for key in net_state_dict.keys()):
            cleaned_state_dict = {}
            for key, value in net_state_dict.items():
                if key.startswith("model."):
                    new_key = key.replace("model.", "")
                    cleaned_state_dict[new_key] = value
                else:
                    cleaned_state_dict[key] = value
            net_state_dict = cleaned_state_dict

        self.net.load_state_dict(net_state_dict)
        self.net.eval()

        # No need to load pre-finetune encoder state - we use the final trained weights

        # Load centers
        centers_data = state.get("centers")
        if centers_data is None and "state" in state:
            centers_data = state["state"].get("centers")

        if centers_data is None:
            raise ValueError("No centers found in checkpoint")

        centers_tensor = torch.tensor(
            centers_data, dtype=torch.float32, device=self.device
        )
        self.centers_module = CVDDCenters(
            centers_tensor, learnable=not self.fixed_centers
        ).to(self.device)

        # Load cluster masks
        cluster_masks_data = state.get("cluster_masks")
        if cluster_masks_data is None:
            raise ValueError(
                "No cluster_masks found in checkpoint. "
                "This is required for CVDDKMeansMasked."
            )

        self.cluster_masks = {
            int(k): torch.tensor(v, device=self.device)
            for k, v in cluster_masks_data.items()
        }
        self.masks_tensor = torch.stack(
            [self.cluster_masks[i] for i in range(self.n_clusters)]
        ).to(self.device)

        # Load cluster centers for KMeans
        self.cluster_centers_raw = state.get("cluster_centers_raw")
        if self.cluster_centers_raw is None:
            raise ValueError("No cluster_centers_raw found in checkpoint")

        # Load binary centroids for L0 clustering (may be None for default)
        self.cluster_centers_binary = state.get("cluster_centers_binary")

        # Load per-cluster frequency vectors for typicality scoring
        freq_data = state.get("cluster_frequency_vectors")
        if freq_data is not None:
            self.cluster_frequency_vectors = torch.tensor(
                freq_data, dtype=torch.float32, device=self.device
            )
        else:
            self.cluster_frequency_vectors = None

        # Recreate KMeans index
        if faiss is None:
            raise ImportError(
                "faiss is required for inference. "
                "Install with: pip install faiss-gpu or faiss-cpu"
            )

        # Always use magnitude centroids for assignment
        # (L0 only affects initial clustering, not inference)
        centroids = self.cluster_centers_raw.astype(np.float32).copy()
        d = centroids.shape[1]
        self.kmeans = faiss.Kmeans(
            d=d, k=self.n_clusters, niter=1, verbose=False
        )
        self.kmeans.train(centroids)
        self.kmeans.centroids = centroids

        # Load normalization parameters
        self.score_low = state.get("score_low")
        self.score_high = state.get("score_high")
        if self.score_low is None and "state" in state:
            self.score_low = state["state"].get("score_low")
            self.score_high = state["state"].get("score_high")

        if self.score_low is None or self.score_high is None:
            raise ValueError("score_low/score_high not found in checkpoint")

        # Load per-cluster normalization parameters (may be None for older models)
        self.score_low_per_cluster = state.get("score_low_per_cluster")
        self.score_high_per_cluster = state.get("score_high_per_cluster")

        # Load training data cluster assignments (may be None for older models)
        cl = state.get("cluster_labels")
        self.cluster_labels = (
            torch.tensor(cl, dtype=torch.long, device=self.device)
            if cl is not None
            else None
        )

        # Load global mask indices for cluster_in_masked_space mode
        self._global_mask_indices = state.get("global_mask_indices")
        if self.cluster_in_masked_space and self._global_mask_indices is None:
            # Derive from global mask for backwards compatibility
            mask_0 = self.cluster_masks[0]
            if isinstance(mask_0, torch.Tensor):
                self._global_mask_indices = (
                    torch.where(mask_0 > 0.5)[0].cpu().numpy()
                )
            else:
                self._global_mask_indices = np.where(np.asarray(mask_0) > 0.5)[
                    0
                ]
            logger.info(
                f"Derived global_mask_indices from mask: "
                f"{len(self._global_mask_indices)} dims"
            )

        # Load scaler
        scaler_mean = state.get("scaler_mean")
        scaler_scale = state.get("scaler_scale")
        if scaler_mean is None and "state" in state:
            scaler_mean = state["state"].get("scaler_mean")
            scaler_scale = state["state"].get("scaler_scale")

        if scaler_mean is not None and scaler_scale is not None:
            from sklearn.preprocessing import StandardScaler

            self.scaler = StandardScaler()
            self.scaler.mean_ = scaler_mean
            self.scaler.scale_ = scaler_scale
            self.scaler.n_features_in_ = len(scaler_mean)
            self.scaler.n_samples_seen_ = np.array([1000])
            logger.info(
                f"Loaded StandardScaler with {len(scaler_mean)} features"
            )
        else:
            from unsupervised_llm_safety.anodet_algos.scaler import (
                IdentityScaler,
            )

            self.scaler = IdentityScaler()
            logger.info("Using IdentityScaler (no scaling)")

        # Load per-cluster Mahalanobis state if present
        maha_state = state.get("mahalanobis_state")
        if maha_state is not None:
            self.cluster_precisions = {
                int(k): torch.tensor(v, device=self.device)
                for k, v in maha_state["cluster_precisions"].items()
            }
            self.cluster_means_sub = {
                int(k): torch.tensor(v, device=self.device)
                for k, v in maha_state["cluster_means_sub"].items()
            }
            self.cluster_mask_indices = {
                int(k): torch.tensor(v, device=self.device, dtype=torch.long)
                for k, v in maha_state["cluster_mask_indices"].items()
            }
            logger.info(
                f"Loaded Mahalanobis state for "
                f"{len(self.cluster_precisions)} clusters"
            )
        elif self.distance_metric == "mahalanobis":
            raise ValueError(
                "distance_metric='mahalanobis' but no mahalanobis_state "
                "found in checkpoint"
            )

        self._fitted = True
        logger.info(f"Model loaded successfully from {path}")

    def _reconstruct_sequential_from_state_dict(
        self, net_state_dict: dict
    ) -> nn.Module:
        """Reconstruct Sequential encoder from state dict."""
        layers = []
        i = 0

        # Try with "model." prefix first
        while (
            f"model.{i}.weight" in net_state_dict
            or f"{i}.weight" in net_state_dict
        ):
            if f"model.{i}.weight" in net_state_dict:
                weight_key = f"model.{i}.weight"
                bias_key = f"model.{i}.bias"
            else:
                weight_key = f"{i}.weight"
                bias_key = f"{i}.bias"

            weight = net_state_dict[weight_key]
            bias = net_state_dict.get(bias_key)

            in_features = weight.shape[1]
            out_features = weight.shape[0]

            linear = nn.Linear(
                in_features, out_features, bias=bias is not None
            )
            layers.append(linear)

            # Add ReLU between layers except the last one
            if i < 2:  # Assuming 3-layer network
                layers.append(nn.ReLU())

            i += 1

        if not layers:
            raise ValueError(
                f"Could not reconstruct Sequential from state dict keys: {list(net_state_dict.keys())[:10]}"
            )

        return nn.Sequential(*layers).to(self.device)

    def post_training_callback(
        self, save_path: str, cfg: DictConfig, device: torch.device
    ) -> None:
        """Run inline SAE health diagnostics after training.

        Computes activation histograms, sparsity metrics, cluster variance,
        inter-cluster distances, and generates a health report with
        pass/warn/fail flags.

        Args:
            save_path: Path where the trained model was saved
            cfg: Configuration dictionary
            device: Device used for training
        """
        # Resolve method-specific config
        method_cfg = None
        for key in (
            "cvdd_kmeans_masked",
            "cvdd_kmeans_masked_balanced",
        ):
            method_cfg = cfg.get(key)
            if method_cfg is not None:
                break

        if method_cfg is None:
            logger.info("No method config found for health check. Skipping.")
            return

        # Check if health check is enabled
        health_cfg = method_cfg.get("health_check", {})
        if not health_cfg.get("enabled", False):
            logger.info("Health check disabled in config. Skipping.")
            return

        # Verify we have the pre-finetune state
        if self._pre_finetune_state_dict is None:
            logger.warning(
                "No pre-finetune encoder state available. "
                "Skipping health diagnostics."
            )
            return

        if self._diagnostic_data is None:
            logger.warning(
                "No diagnostic data available. " "Skipping health diagnostics."
            )
            return

        logger.info("\n" + "=" * 80)
        logger.info("Running SAE health diagnostics...")
        logger.info("=" * 80)

        try:
            from unsupervised_llm_safety.analysis.health_report import (
                HealthReport,
            )
            from unsupervised_llm_safety.analysis.sae_health_diagnostics import (
                SAEHealthDiagnostics,
            )

            diagnostics_dir = Path(save_path).parent / "diagnostics"
            batch_size = health_cfg.get("batch_size", 256)

            # Run diagnostics
            # When train_encoder=True, diagnostic data is raw and must
            # be encoded through pre/post encoders (not skipped).
            diag_encoded_space = (
                self.cluster_in_encoded_space
                and not self._diagnostic_encode_through_encoder
            )
            diagnostics = SAEHealthDiagnostics(
                pre_encoder_state_dict=self._pre_finetune_state_dict,
                post_encoder=self.net,
                data=self._diagnostic_data,
                cluster_labels=self._diagnostic_cluster_labels,
                cluster_masks=self.cluster_masks,
                cluster_centers=self.centers_module.centers.data,
                device=device,
                output_dir=diagnostics_dir,
                cluster_in_encoded_space=diag_encoded_space,
                batch_size=batch_size,
            )
            results = diagnostics.run_all()

            # Generate health report
            thresholds = {}
            threshold_cfg = health_cfg.get("thresholds", {})
            if threshold_cfg:
                for k, v in threshold_cfg.items():
                    thresholds[k] = float(v)

            report_gen = HealthReport(thresholds=thresholds)
            report = report_gen.evaluate(results)
            report_gen.save_report(report, diagnostics_dir)

            # Log summary
            overall = report["overall_status"]
            n_warn = report["n_warn"]
            n_fail = report["n_fail"]
            if n_fail > 0:
                logger.warning(
                    f"Health check: {overall} "
                    f"({n_fail} FAIL, {n_warn} WARN)"
                )
            elif n_warn > 0:
                logger.info(f"Health check: {overall} ({n_warn} WARN)")
            else:
                logger.info("Health check: ALL PASS")

        except Exception as e:
            logger.error(
                f"Health diagnostics failed: {e}",
                exc_info=True,
            )
            logger.warning("Continuing despite health diagnostics error...")

        # Always run cluster diagnostics (Step 0 from plan)
        self._run_cluster_diagnostics(save_path)

    def _run_cluster_diagnostics(self, save_path: str) -> None:
        """Save per-cluster diagnostics: sizes, union-of-support, frequency
        stats, and centroid distance heatmaps.

        Always runs after training, regardless of health_check config.
        """
        if self._diagnostic_data is None or self.kmeans is None:
            logger.warning(
                "No diagnostic data or kmeans available. "
                "Skipping cluster diagnostics."
            )
            return

        try:
            from unsupervised_llm_safety.analysis.cluster_diagnostics import (
                compute_cluster_diagnostics,
                format_table,
                plot_centroid_heatmap,
                plot_cluster_sizes,
                plot_union_sizes,
            )

            diagnostics_dir = (
                Path(save_path).parent / "diagnostics" / "clusters"
            )
            diagnostics_dir.mkdir(parents=True, exist_ok=True)

            # Cluster diagnostics need encoded data. When train_encoder=True,
            # _diagnostic_data is raw — encode through post-training encoder.
            if self._diagnostic_encode_through_encoder:
                data = self._encode_data(self._diagnostic_data)
            else:
                data = self._diagnostic_data
            labels = self._diagnostic_cluster_labels

            logger.info("Running cluster diagnostics...")

            cluster_table, centroids = compute_cluster_diagnostics(
                data, labels, self.n_clusters
            )

            d_sae = data.shape[1]

            # Summary stats
            sizes = [c["n_samples"] for c in cluster_table]
            unions = [
                c["union_support_size"]
                for c in cluster_table
                if c["n_samples"] > 0
            ]
            max_pct = max(c["pct_of_total"] for c in cluster_table)

            logger.info(
                f"Cluster diagnostics: "
                f"largest={max_pct:.1f}%, "
                f"empty={sum(1 for s in sizes if s == 0)}, "
                f"union_support mean={np.mean(unions):.0f} "
                f"max={np.max(unions)} (d={d_sae})"
            )

            # Plots
            plot_cluster_sizes(
                cluster_table, diagnostics_dir / "cluster_sizes.png"
            )
            plot_union_sizes(
                cluster_table, d_sae, diagnostics_dir / "union_sizes.png"
            )
            plot_centroid_heatmap(
                centroids, "l1", diagnostics_dir / "centroid_heatmap_l1.png"
            )
            plot_centroid_heatmap(
                centroids, "l2", diagnostics_dir / "centroid_heatmap_l2.png"
            )

            # Save table and JSON
            table_md = format_table(cluster_table)
            (diagnostics_dir / "cluster_table.md").write_text(table_md)

            import json

            results = {
                "n_clusters": self.n_clusters,
                "d_sae": d_sae,
                "n_samples": len(data),
                "cluster_method": self.cluster_method,
                "distance_metric": self.distance_metric,
                "largest_cluster_pct": max_pct,
                "empty_clusters": sum(1 for s in sizes if s == 0),
                "union_support_mean": round(float(np.mean(unions)), 0),
                "union_support_max": int(np.max(unions)),
                "cluster_table": cluster_table,
            }
            with open(diagnostics_dir / "cluster_diagnostics.json", "w") as f:
                json.dump(results, f, indent=2)

            logger.info(f"Cluster diagnostics saved to {diagnostics_dir}")

        except Exception as e:
            logger.error(f"Cluster diagnostics failed: {e}", exc_info=True)
            logger.warning("Continuing despite cluster diagnostics error...")
