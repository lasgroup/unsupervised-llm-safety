"""Per-cluster learned sparse masks via SAE reconstruction.

Learns a sparse continuous mask per CVDD-KM cluster by minimizing SAE
reconstruction loss on cluster members, with L1 sparsity penalty.  The
mask is initialized from the union of support of each cluster and only
trained on those indices (all other components stay at zero).

At inference: encode -> assign cluster -> apply cluster mask -> decode ->
compute reconstruction residual as anomaly score.

Supports three residual_norm modes:
  - "l2": L2 norm of reconstruction residual (default)
  - "l1": L1 norm of reconstruction residual
  - "whitened": diagonal or full Mahalanobis distance on residual,
    with per-cluster stats estimated from safe training data.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

try:
    import faiss
except ImportError:
    faiss = None  # type: ignore

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

from unsupervised_llm_safety.anodet_algos.base import AnodetMethod
from unsupervised_llm_safety.anodet_algos.cvdd_km_utils import (
    cluster_kmeans,
    merge_small_clusters,
    split_large_union_clusters,
)

logger = logging.getLogger(__name__)


class LearnedMaskAnodet(AnodetMethod):
    """Per-cluster learned sparse mask anomaly detector.

    For each cluster, learns a continuous mask on the union-of-support
    indices that minimizes SAE reconstruction loss + L1 sparsity.
    Anomaly score = masked reconstruction residual.
    """

    def __init__(
        self,
        # Clustering
        n_clusters: int = 1500,
        cluster_method: str = "l0",
        outlier_threshold: float = 2.0,
        l0_filter: float = 2.0,
        use_clustering_cache: bool = False,
        clustering_cache_model_key: Optional[str] = None,
        # Clustering RNG control. Defaults reproduce the exact legacy
        # behaviour: cluster_seed=None means no seed= is passed to FAISS,
        # which then uses its own constant cp.seed=1234 -- NOT "unseeded".
        # The anodet `seed` above never reaches FAISS and never did.
        cluster_seed: Optional[int] = None,
        cluster_init: str = "faiss_random",
        cluster_on_cpu: bool = False,
        # Mask training
        mask_type: str = "continuous",
        residual_norm: str = "l2",
        sparsity_weighting: str = "uniform",
        lambda_sparsity: float = 0.01,
        lambda_candidates: Optional[List[float]] = None,
        n_epochs: int = 50,
        lr: float = 0.01,
        batch_size: int = 256,
        sparsity_target: int = 50,
        train_decoder: bool = False,
        per_cluster_decoder: bool = False,
        decoder_lr: float = 1e-4,
        logit_init: float = 0.0,
        # Whitened residual (used when residual_norm="whitened")
        whitened_mode: str = "diagonal",
        shrinkage_alpha: float = 0.3,
        small_cluster_threshold: int = 10,
        variance_floor: float = 1e-6,
        # Cluster merging
        min_cluster_size: int = 0,  # Merge clusters < this size (0=disabled)
        max_union_size: int = 500,  # Max union-of-support after merge
        # Global mask: train one mask on all safe data instead of per-cluster
        global_mask: bool = False,
        # Scoring method override
        scoring_method: str = "l2_norm",  # "l2_norm", "noise_lr"
        noise_lr_min_cluster_size: int = 50,
        # Infra
        device: str = "cuda",
        seed: int = 42,
        analysis_batch_size: int = 256,
        pretrained_encoder: Optional[nn.Module] = None,
        **kwargs,
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.cluster_method = cluster_method
        self.outlier_threshold = outlier_threshold
        self.l0_filter = l0_filter
        self.use_clustering_cache = use_clustering_cache
        self.clustering_cache_model_key = clustering_cache_model_key
        self.cluster_seed = cluster_seed
        self.cluster_init = cluster_init
        self.cluster_on_cpu = cluster_on_cpu

        self.mask_type = mask_type
        self.residual_norm = residual_norm
        self.sparsity_weighting = sparsity_weighting
        self.lambda_sparsity = lambda_sparsity
        self.lambda_candidates = lambda_candidates or [
            1e-4,
            3e-4,
            1e-3,
            3e-3,
            1e-2,
            3e-2,
            1e-1,
        ]
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.sparsity_target = sparsity_target
        self.train_decoder = train_decoder
        self.per_cluster_decoder = per_cluster_decoder
        self.decoder_lr = decoder_lr
        self.logit_init = logit_init

        self.whitened_mode = whitened_mode
        self.shrinkage_alpha = shrinkage_alpha
        self.small_cluster_threshold = small_cluster_threshold
        self.variance_floor = variance_floor

        self.global_mask = global_mask

        self.min_cluster_size = min_cluster_size
        self.max_union_size = max_union_size

        self.scoring_method = scoring_method
        self.noise_lr_min_cluster_size = noise_lr_min_cluster_size

        self.device = torch.device(device)
        self.seed = seed
        self.analysis_batch_size = analysis_batch_size

        # Will be set during fit or load
        self.encoder: Optional[nn.Module] = None
        self.decoder: Optional[nn.Module] = None
        self.full_autoencoder: Optional[nn.Module] = None

        # If a pretrained encoder is passed, store the full autoencoder
        # (run_anodet_analysis passes the full SAE as pretrained_encoder
        #  when model_path points to an Autoencoder checkpoint)
        if pretrained_encoder is not None:
            self._set_autoencoder(pretrained_encoder)

        # Per-cluster state (populated during fit)
        self.kmeans: Optional[Any] = None
        self.cluster_centers_raw: Optional[np.ndarray] = None
        # Per-cluster: (union_indices, mask_values)
        self.cluster_masks: Dict[int, Dict[str, torch.Tensor]] = {}
        self.sae_dim: int = 0
        self.input_dim: int = 0
        self.calibrated_lambda: float = self.lambda_sparsity

        # Whitened residual state (populated during fit when
        # residual_norm="whitened", serialized in checkpoint)
        # Diagonal mode:
        self.residual_mu_c: Optional[np.ndarray] = None
        self.residual_var_c: Optional[np.ndarray] = None
        self.residual_mu_pooled: Optional[np.ndarray] = None
        self.residual_var_pooled: Optional[np.ndarray] = None
        # Full covariance mode:
        self.residual_precisions: Optional[Dict[int, np.ndarray]] = None
        self.residual_means: Optional[Dict[int, np.ndarray]] = None

        # Noise likelihood ratio state (populated when scoring_method="noise_lr")
        self._noise_lr_precisions_safe: Dict[int, np.ndarray] = {}
        self._noise_lr_inv_diag_noise: Dict[int, np.ndarray] = {}
        self._noise_lr_means: Dict[int, np.ndarray] = {}
        self._noise_lr_log_det_diff: Dict[int, float] = {}
        self._noise_lr_fallback_clusters: set = set()

        # Score normalization
        self.score_low: Optional[float] = None
        self.score_high: Optional[float] = None

        # Diagnostics (not serialized -- set during fit for callback)
        self._diagnostic_encoded: Optional[np.ndarray] = None
        self._diagnostic_cluster_labels: Optional[np.ndarray] = None
        # Per-cluster loss curves (populated during fit step 4)
        self._loss_curves: Dict[int, List[float]] = {}
        self._last_loss_curve: List[float] = []
        self._calibration_log: Dict[str, Any] = {}
        # Eval state (set during score_samples for diagnostics)
        self._last_eval_clusters: Optional[np.ndarray] = None
        self._last_raw_scores: Optional[np.ndarray] = None

    def _set_autoencoder(self, model: nn.Module) -> None:
        """Extract encoder/decoder from a full Autoencoder model."""
        if hasattr(model, "encoder") and hasattr(model, "decoder"):
            self.full_autoencoder = model
            self.encoder = model.encoder
            self.decoder = model.decoder
        else:
            raise ValueError(
                "pretrained_encoder must be a full Autoencoder with "
                ".encoder and .decoder attributes"
            )

    @property
    def name(self) -> str:
        return "learned_mask"

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _encode_batch(self, raw: torch.Tensor) -> torch.Tensor:
        """Encode raw hidden states through the frozen SAE encoder."""
        assert self.encoder is not None, "Encoder not set"
        return self.encoder(raw.float().to(self.device))

    def _decode_batch(self, z: torch.Tensor) -> torch.Tensor:
        """Decode SAE features back to raw space."""
        assert self.decoder is not None, "Decoder not set"
        return self.decoder(z.to(self.device))

    def _encode_all(
        self, data: np.ndarray, batch_size: int = 512
    ) -> np.ndarray:
        """Encode entire dataset in batches, return numpy."""
        assert self.encoder is not None, "Encoder not set"
        self.encoder.eval()
        chunks: List[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                batch = torch.tensor(
                    data[i : i + batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                z = self.encoder(batch)  # type: ignore[misc]
                chunks.append(z.cpu().numpy())
        return np.vstack(chunks)

    # ------------------------------------------------------------------
    # Lambda calibration
    # ------------------------------------------------------------------

    def _measure_sparsity(self, mask_vals: torch.Tensor) -> int:
        """Measure achieved sparsity of a mask (features kept)."""
        if self.mask_type == "binary_ste":
            return int(mask_vals.sum().item())
        return int((mask_vals.abs() > 0.1).sum().item())

    def _calibrate_lambda_multi(
        self,
        safe_data: np.ndarray,
        encoded: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_sizes: np.ndarray,
    ) -> float:
        """Calibrate lambda using 5 clusters at size quantiles.

        Trains a mask on each of 5 calibration clusters for each
        candidate lambda, picks the lambda whose median achieved
        sparsity is closest to sparsity_target.

        Returns selected lambda. Populates self._calibration_log
        with per-lambda, per-cluster details for diagnostics.
        """
        nonzero_mask = cluster_sizes > 0
        nonzero_ids = np.where(nonzero_mask)[0]
        nonzero_sizes = cluster_sizes[nonzero_ids]

        # Pick calibration clusters at size quantiles
        if len(nonzero_ids) < 5:
            calib_ids = nonzero_ids.tolist()
        else:
            quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]
            target_sizes = np.quantile(
                nonzero_sizes, quantiles, method="nearest"
            )
            calib_ids = []
            used = set()
            for ts in target_sizes:
                # Find closest nonzero cluster not already used
                diffs = np.abs(nonzero_sizes - ts)
                for idx in np.argsort(diffs):
                    cid = int(nonzero_ids[idx])
                    if cid not in used:
                        calib_ids.append(cid)
                        used.add(cid)
                        break

        # Prepare calibration data per cluster
        calib_data = []
        for cid in calib_ids:
            c_mask = cluster_labels == cid
            c_raw = safe_data[c_mask]
            c_enc = encoded[c_mask]
            c_union = np.where(c_enc.max(axis=0) > 0)[0]
            c_z_sparse = c_enc[:, c_union]
            calib_data.append(
                {
                    "cluster": cid,
                    "size": int(cluster_sizes[cid]),
                    "union_size": len(c_union),
                    "raw": c_raw,
                    "z_sparse": c_z_sparse,
                    "union_idx": c_union,
                }
            )

        logger.info(
            f"Calibrating on {len(calib_ids)} clusters at "
            f"size quantiles: "
            + ", ".join(
                f"c{d['cluster']}(n={d['size']},u={d['union_size']})"
                for d in calib_data
            )
        )

        # Sweep lambda candidates
        calibration_log: Dict[str, Any] = {
            "calib_clusters": [
                {
                    "cluster": d["cluster"],
                    "size": d["size"],
                    "union_size": d["union_size"],
                }
                for d in calib_data
            ],
            "sparsity_target": self.sparsity_target,
            "lambda_results": [],
        }

        best_lam = self.lambda_candidates[len(self.lambda_candidates) // 2]
        best_diff = float("inf")

        # Enable decoder training during calibration if configured
        if self.train_decoder:
            assert self.decoder is not None
            self.decoder.train()
            for p in self.decoder.parameters():
                p.requires_grad = True

        for lam in self.lambda_candidates:
            per_cluster_sparsity = []
            for d in calib_data:
                mask_vals = self._train_mask_for_cluster(
                    d["raw"],
                    d["z_sparse"],
                    d["union_idx"],
                    lam,
                    epochs=50,
                    train_decoder=self.train_decoder,
                )
                sp = self._measure_sparsity(mask_vals)
                per_cluster_sparsity.append(sp)

            median_sp = float(np.median(per_cluster_sparsity))
            diff = abs(median_sp - self.sparsity_target)

            calibration_log["lambda_results"].append(
                {
                    "lambda": lam,
                    "per_cluster_sparsity": per_cluster_sparsity,
                    "median_sparsity": median_sp,
                }
            )
            logger.info(
                f"  lambda={lam:.1e} -> "
                f"sparsity={per_cluster_sparsity} "
                f"median={median_sp:.0f} "
                f"(target={self.sparsity_target})"
            )
            if diff < best_diff:
                best_diff = diff
                best_lam = lam

        calibration_log["selected_lambda"] = best_lam
        calibration_log["selected_median_sparsity"] = float(
            np.median(
                [
                    r["median_sparsity"]
                    for r in calibration_log["lambda_results"]
                    if r["lambda"] == best_lam
                ]
            )
        )
        self._calibration_log = calibration_log

        # Re-freeze decoder after calibration
        if self.train_decoder:
            assert self.decoder is not None
            self.decoder.eval()
            for p in self.decoder.parameters():
                p.requires_grad = False

        logger.info(f"Calibrated lambda={best_lam:.1e}")
        return best_lam

    # ------------------------------------------------------------------
    # Per-cluster mask training
    # ------------------------------------------------------------------

    def _train_one_mask(
        self,
        raw_cluster: np.ndarray,
        z_sparse: np.ndarray,
        union_idx: np.ndarray,
        lambda_: float,
        epochs: Optional[int] = None,
        train_decoder: bool = False,
    ) -> torch.Tensor:
        """Train mask for one cluster. Returns mask values on CPU."""
        epochs = epochs or self.n_epochs
        n = len(raw_cluster)
        if n == 0:
            return torch.ones(len(union_idx))

        # Initialize mask to 1.0 on union support
        mask = torch.ones(
            len(union_idx), device=self.device, requires_grad=True
        )
        param_groups = [{"params": [mask], "lr": self.lr}]
        if train_decoder and self.decoder is not None:
            param_groups.append(
                {"params": self.decoder.parameters(), "lr": self.decoder_lr}
            )
        optimizer = torch.optim.Adam(param_groups)

        # Compute per-feature sparsity weight from activation frequency
        if self.sparsity_weighting == "variance":
            p_cj = torch.tensor(
                (z_sparse > 0).mean(axis=0),
                dtype=torch.float32,
                device=self.device,
            )
            sparsity_weight = p_cj * (1 - p_cj)  # [0, 0.25]
        else:
            sparsity_weight = None

        # Pre-move raw data to device in chunks if small enough
        raw_t = torch.tensor(raw_cluster, dtype=torch.float32)
        z_t = torch.tensor(z_sparse, dtype=torch.float32)
        dataset = TensorDataset(raw_t, z_t)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

        union_idx_t = torch.tensor(
            union_idx, dtype=torch.long, device=self.device
        )

        assert self.decoder is not None, "Decoder not set"
        self.decoder.eval()
        epoch_losses: List[float] = []
        for _ in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch_raw, batch_z in loader:
                batch_raw = batch_raw.to(self.device)
                batch_z = batch_z.to(self.device)
                optimizer.zero_grad()

                # Construct full masked encoding
                B = batch_z.shape[0]
                z_full = torch.zeros(B, self.sae_dim, device=self.device)
                z_full[:, union_idx_t] = batch_z * mask

                # Decode and compute reconstruction loss
                x_hat = self.decoder(z_full)
                recon_loss = ((batch_raw - x_hat) ** 2).mean()
                if sparsity_weight is not None:
                    sparsity_loss = (sparsity_weight * mask.abs()).sum()
                else:
                    sparsity_loss = mask.abs().sum()
                loss = recon_loss + lambda_ * sparsity_loss

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            epoch_losses.append(epoch_loss / max(n_batches, 1))
        self._last_loss_curve = epoch_losses

        return mask.detach().cpu()

    def _train_one_mask_binary_ste(
        self,
        raw_cluster: np.ndarray,
        z_sparse: np.ndarray,
        union_idx: np.ndarray,
        lambda_: float,
        epochs: Optional[int] = None,
        train_decoder: bool = False,
    ) -> torch.Tensor:
        """Train binary mask using straight-through estimator (STE).

        Forward pass uses hard 0/1 threshold on sigmoid(logits).
        Backward pass flows gradients through the sigmoid.
        Returns binary mask values (0.0/1.0) on CPU.
        """
        epochs = epochs or self.n_epochs
        n = len(raw_cluster)
        if n == 0:
            return torch.ones(len(union_idx))

        # Logits parameterization: sigmoid(logits) -> soft probability
        logits = torch.full(
            (len(union_idx),),
            self.logit_init,
            device=self.device,
            requires_grad=True,
        )
        param_groups = [{"params": [logits], "lr": self.lr}]
        if train_decoder and self.decoder is not None:
            param_groups.append(
                {"params": self.decoder.parameters(), "lr": self.decoder_lr}
            )
        optimizer = torch.optim.Adam(param_groups)

        # Compute per-feature sparsity weight from activation frequency
        if self.sparsity_weighting == "variance":
            p_cj = torch.tensor(
                (z_sparse > 0).mean(axis=0),
                dtype=torch.float32,
                device=self.device,
            )
            sparsity_weight = p_cj * (1 - p_cj)  # [0, 0.25]
        else:
            sparsity_weight = None

        raw_t = torch.tensor(raw_cluster, dtype=torch.float32)
        z_t = torch.tensor(z_sparse, dtype=torch.float32)
        dataset = TensorDataset(raw_t, z_t)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

        union_idx_t = torch.tensor(
            union_idx, dtype=torch.long, device=self.device
        )

        assert self.decoder is not None, "Decoder not set"
        self.decoder.eval()
        epoch_losses: List[float] = []
        for _ in range(epochs):
            epoch_loss = 0.0
            n_batches = 0
            for batch_raw, batch_z in loader:
                batch_raw = batch_raw.to(self.device)
                batch_z = batch_z.to(self.device)
                optimizer.zero_grad()

                # Soft mask (for gradient)
                m_soft = torch.sigmoid(logits)
                # Hard mask (for forward) -- STE trick
                m_hard = (m_soft > 0.5).float()
                m = m_hard - m_soft.detach() + m_soft

                # Construct full masked encoding
                B = batch_z.shape[0]
                z_full = torch.zeros(B, self.sae_dim, device=self.device)
                z_full[:, union_idx_t] = batch_z * m

                # Decode and compute loss
                x_hat = self.decoder(z_full)
                recon_loss = ((batch_raw - x_hat) ** 2).mean()
                # L1 on soft mask to encourage sparsity
                if sparsity_weight is not None:
                    sparsity_loss = (sparsity_weight * m_soft).sum()
                else:
                    sparsity_loss = m_soft.sum()
                loss = recon_loss + lambda_ * sparsity_loss

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            epoch_losses.append(epoch_loss / max(n_batches, 1))
        self._last_loss_curve = epoch_losses

        # Return hard binary mask
        with torch.no_grad():
            final_mask = (torch.sigmoid(logits) > 0.5).float()
        return final_mask.cpu()

    def _train_mask_for_cluster(
        self,
        raw_cluster: np.ndarray,
        z_sparse: np.ndarray,
        union_idx: np.ndarray,
        lambda_: float,
        epochs: Optional[int] = None,
        train_decoder: bool = False,
    ) -> torch.Tensor:
        """Dispatch to the right mask training method based on mask_type."""
        if self.mask_type == "binary_ste":
            return self._train_one_mask_binary_ste(
                raw_cluster,
                z_sparse,
                union_idx,
                lambda_,
                epochs,
                train_decoder=train_decoder,
            )
        return self._train_one_mask(
            raw_cluster,
            z_sparse,
            union_idx,
            lambda_,
            epochs,
            train_decoder=train_decoder,
        )

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, data: np.ndarray, labels: np.ndarray) -> None:
        """Fit learned masks on safe training data.

        Args:
            data: Raw hidden states (N, raw_dim) from run_anodet_analysis.
            labels: 0=unsafe, 1=safe.
        """
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if self.encoder is None or self.decoder is None:
            raise RuntimeError(
                "Encoder/decoder not set. Pass a full Autoencoder as "
                "pretrained_encoder or via full_model."
            )

        # Freeze encoder and decoder
        self.encoder.eval()
        self.decoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False

        # Filter safe samples
        safe_mask = labels == 1
        safe_data = data[safe_mask]
        self.input_dim = safe_data.shape[1]

        logger.info(
            f"Fitting on {len(safe_data)} safe samples "
            f"(dim={self.input_dim})"
        )

        # Step 1: Encode all safe data
        logger.info("Encoding safe data through SAE...")
        encoded = self._encode_all(safe_data)
        self.sae_dim = encoded.shape[1]
        logger.info(
            f"Encoded: ({encoded.shape[0]}, {self.sae_dim}), "
            f"mean L0={int((encoded > 0).sum(axis=1).mean())}"
        )

        # Step 2: Cluster in encoded space
        if self.use_clustering_cache and self.clustering_cache_model_key:
            # Load pre-computed clustering (consistent with other analyses)
            from unsupervised_llm_safety.data.precompute_clustering import (
                load_clustering_cache,
            )

            logger.info(
                f"Loading clustering cache for "
                f"{self.clustering_cache_model_key}..."
            )
            cache = load_clustering_cache(
                self.clustering_cache_model_key,
                n_clusters=self.n_clusters,
                outlier_threshold=self.outlier_threshold,
                l0_filter=self.l0_filter,
            )
            self.cluster_centers_raw = cache["centroids"]
            self.n_clusters = cache["n_clusters"]

            # Build FAISS index from cached centroids
            centroids_f32 = cache["centroids"].astype(np.float32).copy()
            self.kmeans = faiss.Kmeans(
                d=centroids_f32.shape[1],
                k=self.n_clusters,
                niter=1,
                verbose=False,
            )
            self.kmeans.train(centroids_f32)
            self.kmeans.centroids = centroids_f32

            # Assign current safe data to cached clusters
            encoded_f32 = encoded.astype(np.float32)
            _, assignments = self.kmeans.index.search(encoded_f32, 1)
            cluster_labels = assignments.flatten()

            sizes = np.bincount(cluster_labels, minlength=self.n_clusters)
            logger.info(
                f"Loaded {self.n_clusters} clusters from cache, "
                f"assigned {len(encoded)} samples "
                f"(sizes min={sizes[sizes > 0].min()} "
                f"max={sizes.max()})"
            )
        else:
            # Cluster from scratch
            logger.info(
                f"Clustering: method={self.cluster_method}, "
                f"n_clusters={self.n_clusters}"
            )
            result = cluster_kmeans(
                data=encoded,
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
            cluster_labels = result.labels
            self.n_clusters = int(cluster_labels.max() + 1)

            logger.info(
                f"Initial clustering: {self.n_clusters} clusters, "
                f"sizes min={np.bincount(cluster_labels).min()} "
                f"max={np.bincount(cluster_labels).max()}"
            )

            # Step 2b: Split clusters with excessive union-of-support
            if self.outlier_threshold > 0:
                split_result = split_large_union_clusters(
                    data=encoded,
                    cluster_labels=cluster_labels,
                    n_clusters=self.n_clusters,
                    threshold=self.outlier_threshold,
                    cluster_method=self.cluster_method,
                    seed=self.cluster_seed,
                    cluster_on_cpu=self.cluster_on_cpu,
                )
                cluster_labels = split_result.labels
                self.n_clusters = split_result.n_clusters
                self.cluster_centers_raw = split_result.centroids

                # Rebuild FAISS index on new centroids
                centroids_f32 = split_result.centroids.astype(
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

                logger.info(
                    f"After sub-clustering: {self.n_clusters} clusters"
                )

            # Step 2c: Merge small clusters
            if self.min_cluster_size > 0:
                merge_result = merge_small_clusters(
                    data=encoded,
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

        # Store for diagnostics
        self._diagnostic_encoded = encoded
        self._diagnostic_cluster_labels = cluster_labels

        # Step 3: Lambda calibration on 5 clusters at size quantiles
        logger.info("Calibrating lambda (multi-cluster)...")
        cluster_sizes = np.bincount(cluster_labels, minlength=self.n_clusters)

        # Save decoder state before calibration so step 4 starts fresh
        if self.train_decoder:
            import copy

            decoder_state_backup = copy.deepcopy(self.decoder.state_dict())

        self.calibrated_lambda = self._calibrate_lambda_multi(
            safe_data, encoded, cluster_labels, cluster_sizes
        )

        # Restore original decoder after calibration
        if self.train_decoder:
            self.decoder.load_state_dict(decoder_state_backup)
            logger.info("Restored decoder to pre-calibration state")

        # Step 4: Train per-cluster masks
        if self.train_decoder:
            import copy

            mode = "per-cluster" if self.per_cluster_decoder else "shared"
            logger.info(f"Decoder training ENABLED ({mode} decoder)")
            self.decoder.train()
            for p in self.decoder.parameters():
                p.requires_grad = True
            # Save original decoder for per-cluster mode
            if self.per_cluster_decoder:
                original_decoder_state = copy.deepcopy(
                    self.decoder.state_dict()
                )
                self._cluster_decoders: Dict[int, Dict] = {}

        if self.global_mask:
            # --- Global mask: one mask trained on all safe data ---
            logger.info(
                f"Training GLOBAL mask on {len(safe_data)} samples "
                f"with lambda={self.calibrated_lambda:.1e}..."
            )
            # Global union of support
            global_union_idx = np.where(encoded.max(axis=0) > 0)[0]
            global_z_sparse = encoded[:, global_union_idx]
            logger.info(
                f"Global union of support: {len(global_union_idx)} dims"
            )

            global_mask_vals = self._train_mask_for_cluster(
                safe_data,
                global_z_sparse,
                global_union_idx,
                self.calibrated_lambda,
                train_decoder=self.train_decoder,
            )
            self._loss_curves[0] = self._last_loss_curve

            if self.mask_type == "binary_ste":
                sparsity = int(global_mask_vals.sum().item())
            else:
                sparsity = int((global_mask_vals.abs() > 0.1).sum().item())
            logger.info(
                f"Global mask: union={len(global_union_idx)}, "
                f"mask_sparsity(>0.1)={sparsity}"
            )

            # Replicate to all clusters
            global_entry = {
                "union_indices": torch.tensor(
                    global_union_idx, dtype=torch.long
                ),
                "mask_values": global_mask_vals,
            }
            for c in range(self.n_clusters):
                self.cluster_masks[c] = global_entry
        else:
            # --- Per-cluster masks (original behavior) ---
            logger.info(
                f"Training masks for {self.n_clusters} clusters "
                f"with lambda={self.calibrated_lambda:.1e}..."
            )
            for c in range(self.n_clusters):
                c_mask = cluster_labels == c
                n_c = int(c_mask.sum())
                if n_c == 0:
                    self.cluster_masks[c] = {
                        "union_indices": torch.tensor([], dtype=torch.long),
                        "mask_values": torch.tensor([]),
                    }
                    continue

                # Per-cluster decoder: reset to original before each cluster
                if self.train_decoder and self.per_cluster_decoder:
                    self.decoder.load_state_dict(original_decoder_state)

                c_raw = safe_data[c_mask]
                c_enc = encoded[c_mask]

                # Union of support
                union_idx = np.where(c_enc.max(axis=0) > 0)[0]
                z_sparse = c_enc[:, union_idx]

                mask_vals = self._train_mask_for_cluster(
                    c_raw,
                    z_sparse,
                    union_idx,
                    self.calibrated_lambda,
                    train_decoder=self.train_decoder,
                )
                self._loss_curves[c] = self._last_loss_curve

                # Save per-cluster decoder state
                if self.train_decoder and self.per_cluster_decoder:
                    self._cluster_decoders[c] = copy.deepcopy(
                        self.decoder.state_dict()
                    )

                self.cluster_masks[c] = {
                    "union_indices": torch.tensor(union_idx, dtype=torch.long),
                    "mask_values": mask_vals,
                }

                if c % 100 == 0 or c == self.n_clusters - 1:
                    if self.mask_type == "binary_ste":
                        sparsity = int(mask_vals.sum().item())
                    else:
                        sparsity = int((mask_vals.abs() > 0.1).sum().item())
                    logger.info(
                        f"  Cluster {c}/{self.n_clusters}: "
                        f"n={n_c}, union={len(union_idx)}, "
                        f"mask_sparsity(>0.1)={sparsity}"
                    )

        # Re-freeze decoder after training
        if self.train_decoder:
            self.decoder.eval()
            for p in self.decoder.parameters():
                p.requires_grad = False
            logger.info("Decoder re-frozen after mask training")

        # Step 4b: Log sparsity summary across all clusters
        all_sparsities = []
        n_identity = 0
        for c in range(self.n_clusters):
            cm = self.cluster_masks.get(c)
            if cm is None or len(cm["union_indices"]) == 0:
                continue
            sp = self._measure_sparsity(cm["mask_values"])
            all_sparsities.append(sp)
            union_sz = len(cm["union_indices"])
            if self.mask_type == "binary_ste":
                is_identity = sp > 0.8 * union_sz
            else:
                vals = cm["mask_values"]
                is_identity = int((vals.abs() > 0.1).sum()) > 0.8 * union_sz
            if is_identity:
                n_identity += 1

        n_nonempty = len(all_sparsities)
        identity_frac = n_identity / max(n_nonempty, 1)
        sparsity_arr = (
            np.array(all_sparsities) if all_sparsities else np.array([0])
        )
        self._sparsity_summary = {
            "sparsity_target": self.sparsity_target,
            "n_clusters_nonempty": n_nonempty,
            "median": round(float(np.median(sparsity_arr)), 1),
            "p10": round(float(np.percentile(sparsity_arr, 10)), 1),
            "p90": round(float(np.percentile(sparsity_arr, 90)), 1),
            "min": int(sparsity_arr.min()),
            "max": int(sparsity_arr.max()),
            "identity_fraction": round(identity_frac, 3),
            "n_identity": n_identity,
        }
        logger.info(
            f"Calibration: lambda={self.calibrated_lambda:.1e}, "
            f"target={self.sparsity_target}, "
            f"median_achieved={self._sparsity_summary['median']:.0f}, "
            f"identity_fraction={identity_frac:.3f}"
        )

        # Step 5: Compute whitened residual stats (if needed)
        if self.residual_norm == "whitened":
            self._compute_whitened_residual_stats(safe_data, cluster_labels)

        # Step 5b: Fit noise likelihood ratio scorer (if needed)
        if self.scoring_method in ("noise_lr", "noise_lr_no_logdet"):
            self._fit_noise_lr_per_cluster(safe_data, cluster_labels)

        # Step 6: Score normalization on training data
        logger.info("Computing score normalization...")
        train_scores = self._score_raw(safe_data, cluster_labels)
        self.score_low = float(np.percentile(train_scores, 5))
        self.score_high = float(np.percentile(train_scores, 95))
        logger.info(
            f"Score normalization: p5={self.score_low:.4f}, "
            f"p95={self.score_high:.4f}"
        )

        # Step 7: Mask sanity check — compare mask vs identity AUROC
        # on training data (safe=negative, uses holdout if available)
        logger.info("Mask sanity check: learned mask vs identity mask...")
        identity_scores = self._compute_identity_scores(
            safe_data, cluster_labels
        )
        mask_safe_mean = float(train_scores.mean())
        identity_safe_mean = float(identity_scores.mean())
        self._mask_sanity = {
            "mask_safe_mean_score": round(mask_safe_mean, 4),
            "identity_safe_mean_score": round(identity_safe_mean, 4),
        }
        logger.info(
            f"  Safe train scores: mask={mask_safe_mean:.4f}, "
            f"identity={identity_safe_mean:.4f}"
        )

        self._fitted = True
        logger.info("Fit complete.")

    def _compute_identity_scores(
        self,
        data: np.ndarray,
        cluster_assignments: np.ndarray,
    ) -> np.ndarray:
        """Score using identity mask (no masking) for sanity check.

        Returns L2 residual norms: ||x - decode(encode(x))||.
        """
        assert self.encoder is not None
        assert self.decoder is not None
        all_scores: List[np.ndarray] = []
        bs = self.analysis_batch_size
        with torch.no_grad():
            encoded = self._encode_all(data)
            for i in range(0, len(data), bs):
                batch_raw = torch.tensor(
                    data[i : i + bs],
                    dtype=torch.float32,
                    device=self.device,
                )
                batch_enc = torch.tensor(
                    encoded[i : i + bs],
                    dtype=torch.float32,
                    device=self.device,
                )
                x_hat = self.decoder(batch_enc)
                diff = batch_raw - x_hat
                scores = (diff**2).sum(dim=1).sqrt()
                all_scores.append(scores.cpu().numpy())
        return np.concatenate(all_scores)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_raw(
        self,
        data: np.ndarray,
        cluster_assignments: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute raw (unnormalized) anomaly scores.

        Args:
            data: Raw hidden states (N, raw_dim).
            cluster_assignments: Pre-computed cluster labels. If None,
                assigns clusters from scratch.

        Returns:
            Raw scores (N,).
        """
        assert self.encoder is not None, "Encoder not set"
        assert self.decoder is not None, "Decoder not set"
        self.encoder.eval()
        self.decoder.eval()

        all_scores: List[np.ndarray] = []
        bs = self.analysis_batch_size

        with torch.no_grad():
            # Encode all data
            encoded = self._encode_all(data)

            # Assign clusters if not provided
            if cluster_assignments is None:
                assert self.kmeans is not None, "KMeans not fitted"
                data_f32 = encoded.astype(np.float32)
                _, assignments = self.kmeans.index.search(data_f32, 1)
                cluster_assignments = assignments.flatten()

            # Store for downstream analysis
            self._last_eval_clusters = cluster_assignments.copy()

            for i in range(0, len(data), bs):
                batch_raw = torch.tensor(
                    data[i : i + bs],
                    dtype=torch.float32,
                    device=self.device,
                )
                batch_enc = torch.tensor(
                    encoded[i : i + bs],
                    dtype=torch.float32,
                    device=self.device,
                )
                batch_clusters = cluster_assignments[i : i + bs]
                B = len(batch_raw)

                # Build masked encoding per sample
                z_masked = torch.zeros(B, self.sae_dim, device=self.device)
                for j in range(B):
                    c = int(batch_clusters[j])
                    cm = self.cluster_masks.get(c)
                    if cm is None or len(cm["union_indices"]) == 0:
                        # No mask for this cluster -- use full encoding
                        z_masked[j] = batch_enc[j]
                        continue
                    idx = cm["union_indices"].to(self.device)
                    vals = cm["mask_values"].to(self.device)
                    z_masked[j, idx] = batch_enc[j, idx] * vals

                # Decode (per-cluster decoder if available)
                if (
                    self.per_cluster_decoder
                    and hasattr(self, "_cluster_decoders")
                    and self._cluster_decoders
                ):
                    x_hat = torch.zeros_like(batch_raw)
                    for c_id in np.unique(batch_clusters):
                        c_id = int(c_id)
                        c_idx = batch_clusters == c_id
                        if c_id in self._cluster_decoders:
                            self.decoder.load_state_dict(
                                self._cluster_decoders[c_id]
                            )
                        x_hat[c_idx] = self.decoder(z_masked[c_idx])
                else:
                    x_hat = self.decoder(z_masked)

                # Compute residual and score
                diff = batch_raw - x_hat
                if self.scoring_method in (
                    "noise_lr",
                    "noise_lr_no_logdet",
                ):
                    scores = self._score_noise_lr(
                        diff,
                        batch_clusters,
                        include_logdet=(self.scoring_method == "noise_lr"),
                    )
                elif (
                    self.residual_norm == "whitened"
                    and self.whitened_mode == "full"
                ):
                    scores = self._score_full_mahalanobis(diff, batch_clusters)
                elif self.residual_norm == "whitened":
                    assert self.residual_mu_c is not None
                    assert self.residual_var_c is not None
                    mu = torch.tensor(
                        self.residual_mu_c[batch_clusters],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    var = torch.tensor(
                        self.residual_var_c[batch_clusters],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    centered = diff - mu
                    scores = (centered**2 / var).sum(dim=1)
                elif self.residual_norm == "l1":
                    scores = diff.abs().sum(dim=1)
                else:
                    scores = (diff**2).sum(dim=1).sqrt()

                all_scores.append(scores.cpu().numpy())

        return np.concatenate(all_scores)

    def _score_full_mahalanobis(
        self,
        diff: torch.Tensor,
        batch_clusters: np.ndarray,
    ) -> torch.Tensor:
        """Score residuals using full per-cluster precision matrices.

        Args:
            diff: (B, raw_dim) residual vectors on device.
            batch_clusters: (B,) cluster assignments.

        Returns:
            (B,) Mahalanobis distance scores.
        """
        assert self.residual_precisions is not None
        assert self.residual_means is not None
        B = diff.shape[0]
        scores = torch.zeros(B, device=self.device)
        for j in range(B):
            c = int(batch_clusters[j])
            mu = torch.tensor(
                self.residual_means[c],
                dtype=torch.float32,
                device=self.device,
            )
            prec = torch.tensor(
                self.residual_precisions[c],
                dtype=torch.float32,
                device=self.device,
            )
            centered = diff[j] - mu
            # Mahalanobis: (x-mu)^T @ precision @ (x-mu)
            scores[j] = centered @ prec @ centered
        return scores

    def _compute_residual_vectors(
        self,
        data: np.ndarray,
        cluster_assignments: np.ndarray,
    ) -> np.ndarray:
        """Return per-sample raw residual vectors (N, raw_dim).

        Identical path to _score_raw but returns (x - x_hat) before
        the norm.  Caller is responsible for how to score these
        (L2, whitened, etc.).
        """
        assert self.encoder is not None, "Encoder not set"
        assert self.decoder is not None, "Decoder not set"
        self.encoder.eval()
        self.decoder.eval()

        all_residuals: List[np.ndarray] = []
        bs = self.analysis_batch_size

        with torch.no_grad():
            encoded = self._encode_all(data)

            for i in range(0, len(data), bs):
                batch_raw = torch.tensor(
                    data[i : i + bs],
                    dtype=torch.float32,
                    device=self.device,
                )
                batch_enc = torch.tensor(
                    encoded[i : i + bs],
                    dtype=torch.float32,
                    device=self.device,
                )
                batch_clusters = cluster_assignments[i : i + bs]
                B = len(batch_raw)

                # Build masked encoding per sample
                z_masked = torch.zeros(B, self.sae_dim, device=self.device)
                for j in range(B):
                    c = int(batch_clusters[j])
                    cm = self.cluster_masks.get(c)
                    if cm is None or len(cm["union_indices"]) == 0:
                        z_masked[j] = batch_enc[j]
                        continue
                    idx = cm["union_indices"].to(self.device)
                    vals = cm["mask_values"].to(self.device)
                    z_masked[j, idx] = batch_enc[j, idx] * vals

                # Decode (per-cluster decoder if available)
                if (
                    self.per_cluster_decoder
                    and hasattr(self, "_cluster_decoders")
                    and self._cluster_decoders
                ):
                    x_hat = torch.zeros_like(batch_raw)
                    for c_id in np.unique(batch_clusters):
                        c_id = int(c_id)
                        c_idx = batch_clusters == c_id
                        if c_id in self._cluster_decoders:
                            self.decoder.load_state_dict(
                                self._cluster_decoders[c_id]
                            )
                        x_hat[c_idx] = self.decoder(z_masked[c_idx])
                else:
                    x_hat = self.decoder(z_masked)
                diff = batch_raw - x_hat
                all_residuals.append(diff.cpu().numpy())

        return np.concatenate(all_residuals, axis=0)

    # ------------------------------------------------------------------
    # Whitened residual statistics
    # ------------------------------------------------------------------

    def _compute_whitened_residual_stats(
        self,
        safe_data: np.ndarray,
        cluster_labels: np.ndarray,
    ) -> None:
        """Compute per-cluster residual statistics for whitened scoring.

        Called at end of fit() when residual_norm="whitened".

        Diagonal mode: stores per-cluster mean and variance with
        shrinkage toward pooled variance.
        Full mode: stores per-cluster mean and precision (inverse
        covariance) using LedoitWolf shrinkage estimator.
        """
        logger.info(
            f"Computing whitened residual statistics "
            f"(mode={self.whitened_mode})..."
        )
        residuals = self._compute_residual_vectors(safe_data, cluster_labels)
        raw_dim = residuals.shape[1]
        cluster_sizes = np.bincount(cluster_labels, minlength=self.n_clusters)

        if self.whitened_mode == "full":
            self._compute_full_covariance_stats(
                residuals, cluster_labels, cluster_sizes, raw_dim
            )
        else:
            self._compute_diagonal_stats(
                residuals, cluster_labels, cluster_sizes, raw_dim
            )

        n_small = int((cluster_sizes < self.small_cluster_threshold).sum())
        n_empty = int((cluster_sizes == 0).sum())
        logger.info(
            f"Whitened stats ({self.whitened_mode}): "
            f"{self.n_clusters} clusters, "
            f"{n_empty} empty, {n_small} small "
            f"(<{self.small_cluster_threshold}), "
            f"alpha={self.shrinkage_alpha}, raw_dim={raw_dim}"
        )

    def _compute_diagonal_stats(
        self,
        residuals: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_sizes: np.ndarray,
        raw_dim: int,
    ) -> None:
        """Compute diagonal (per-dimension) variance with shrinkage."""
        # Pooled statistics
        self.residual_mu_pooled = residuals.mean(axis=0).astype(np.float32)
        self.residual_var_pooled = residuals.var(axis=0).astype(np.float32)

        mu_c = np.zeros((self.n_clusters, raw_dim), dtype=np.float32)
        var_c_raw = np.zeros((self.n_clusters, raw_dim), dtype=np.float32)

        for c in range(self.n_clusters):
            n_c = int(cluster_sizes[c])
            if n_c < self.small_cluster_threshold:
                mu_c[c] = self.residual_mu_pooled
                var_c_raw[c] = self.residual_var_pooled
            else:
                c_residuals = residuals[cluster_labels == c]
                mu_c[c] = c_residuals.mean(axis=0)
                var_c_raw[c] = c_residuals.var(axis=0)

        # Shrinkage toward pooled variance
        alpha = self.shrinkage_alpha
        var_shrunk = (
            1.0 - alpha
        ) * var_c_raw + alpha * self.residual_var_pooled[np.newaxis, :]
        self.residual_var_c = np.maximum(var_shrunk, self.variance_floor)
        self.residual_mu_c = mu_c

    def _compute_full_covariance_stats(
        self,
        residuals: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_sizes: np.ndarray,
        raw_dim: int,
    ) -> None:
        """Compute full covariance precision matrices per cluster.

        Uses LedoitWolf shrinkage estimator which handles n < p
        (more dimensions than samples) gracefully.
        """
        from sklearn.covariance import LedoitWolf

        # Pooled mean and precision (fallback for small clusters)
        pooled_mu = residuals.mean(axis=0).astype(np.float32)
        lw_pooled = LedoitWolf().fit(residuals)
        pooled_precision = lw_pooled.precision_.astype(np.float32)

        self.residual_mu_pooled = pooled_mu
        self.residual_means = {}
        self.residual_precisions = {}

        for c in range(self.n_clusters):
            n_c = int(cluster_sizes[c])
            if n_c < self.small_cluster_threshold:
                self.residual_means[c] = pooled_mu
                self.residual_precisions[c] = pooled_precision
            else:
                c_residuals = residuals[cluster_labels == c]
                c_mu = c_residuals.mean(axis=0).astype(np.float32)
                lw = LedoitWolf().fit(c_residuals)
                c_precision = lw.precision_.astype(np.float32)
                self.residual_means[c] = c_mu
                self.residual_precisions[c] = c_precision

            if c % 100 == 0 or c == self.n_clusters - 1:
                logger.info(
                    f"  Full cov cluster {c}/{self.n_clusters}: " f"n={n_c}"
                )

    # ------------------------------------------------------------------
    # Noise likelihood ratio scorer
    # ------------------------------------------------------------------

    def _fit_noise_lr_per_cluster(
        self,
        safe_data: np.ndarray,
        cluster_labels: np.ndarray,
    ) -> None:
        """Fit noise likelihood ratio scorer per cluster.

        For each cluster, estimates:
          p_safe: N(mu_s, Sigma_s) via LedoitWolf
          p_noise: N(0, diag(Sigma_s))

        Precomputes Sigma_s^{-1}, diag(Sigma_s)^{-1}, mu_s, and
        log|Sigma_s| - log|Sigma_noise| per cluster.
        """
        from sklearn.covariance import LedoitWolf

        logger.info("Fitting noise likelihood ratio scorer...")
        residuals = self._compute_residual_vectors(safe_data, cluster_labels)
        raw_dim = residuals.shape[1]
        cluster_sizes = np.bincount(cluster_labels, minlength=self.n_clusters)

        # Pooled fallback (for small clusters, use denoised_mahalanobis)
        pooled_mu = residuals.mean(axis=0).astype(np.float32)
        lw_pooled = LedoitWolf().fit(residuals)
        pooled_precision = lw_pooled.precision_.astype(np.float32)
        pooled_cov = lw_pooled.covariance_.astype(np.float32)

        # Store pooled as fallback for whitened/full Mahalanobis
        # (small clusters use this path)
        if self.residual_precisions is None:
            self.residual_precisions = {}
        if self.residual_means is None:
            self.residual_means = {}

        n_fitted = 0
        n_fallback = 0

        for c in range(self.n_clusters):
            n_c = int(cluster_sizes[c])
            if n_c == 0:
                continue

            if n_c < self.noise_lr_min_cluster_size:
                # Fallback: store full Mahalanobis stats for this
                # cluster so _score_full_mahalanobis can handle it
                self._noise_lr_fallback_clusters.add(c)
                if c not in self.residual_means:
                    self.residual_means[c] = pooled_mu
                    self.residual_precisions[c] = pooled_precision
                n_fallback += 1
                continue

            c_residuals = residuals[cluster_labels == c]
            c_mu = c_residuals.mean(axis=0).astype(np.float32)

            lw = LedoitWolf().fit(c_residuals)
            c_precision = lw.precision_.astype(np.float32)
            c_cov = lw.covariance_.astype(np.float32)

            # log|Sigma_s| via slogdet
            _, log_det_safe = np.linalg.slogdet(c_cov)

            # Sigma_noise = diag(diag(Sigma_s))
            diag_var = np.diag(c_cov).copy()
            diag_var = np.maximum(diag_var, self.variance_floor)
            inv_diag = (1.0 / diag_var).astype(np.float32)
            log_det_noise = float(np.sum(np.log(diag_var)))

            log_det_diff = float(log_det_safe) - log_det_noise

            self._noise_lr_precisions_safe[c] = c_precision
            self._noise_lr_inv_diag_noise[c] = inv_diag
            self._noise_lr_means[c] = c_mu
            self._noise_lr_log_det_diff[c] = log_det_diff

            # Also store for fallback path compatibility
            self.residual_means[c] = c_mu
            self.residual_precisions[c] = c_precision

            n_fitted += 1
            if c % 50 == 0 or c == self.n_clusters - 1:
                logger.info(
                    f"  Noise LR cluster {c}: n={n_c}, "
                    f"log_det_diff={log_det_diff:.2f}"
                )

        logger.info(
            f"Noise LR fit complete: {n_fitted} fitted, "
            f"{n_fallback} fallback (Mahalanobis)"
        )

    def _score_noise_lr(
        self,
        diff: torch.Tensor,
        batch_clusters: np.ndarray,
        include_logdet: bool = True,
    ) -> torch.Tensor:
        """Score residuals using noise likelihood ratio.

        score(r) = (r-mu)^T Sigma_s^{-1} (r-mu)
                   - r^T diag(Sigma_s)^{-1} r
                   + log|Sigma_s| - log|Sigma_noise|

        Fallback clusters use full Mahalanobis (denoised_mahalanobis).
        """
        B = diff.shape[0]
        scores = torch.zeros(B, device=self.device)

        for j in range(B):
            c = int(batch_clusters[j])

            if c not in self._noise_lr_means:
                # Cluster not fitted (empty during training or fallback)
                # Use Mahalanobis if available, else L2
                if (
                    self.residual_means is not None
                    and c in self.residual_means
                ):
                    assert self.residual_precisions is not None
                    mu = torch.tensor(
                        self.residual_means[c],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    prec = torch.tensor(
                        self.residual_precisions[c],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    centered = diff[j] - mu
                    scores[j] = centered @ prec @ centered
                else:
                    scores[j] = (diff[j] ** 2).sum().sqrt()
                continue

            if c in self._noise_lr_fallback_clusters:
                # Fallback to full Mahalanobis
                assert self.residual_means is not None
                assert self.residual_precisions is not None
                mu = torch.tensor(
                    self.residual_means[c],
                    dtype=torch.float32,
                    device=self.device,
                )
                prec = torch.tensor(
                    self.residual_precisions[c],
                    dtype=torch.float32,
                    device=self.device,
                )
                centered = diff[j] - mu
                scores[j] = centered @ prec @ centered
                continue

            mu = torch.tensor(
                self._noise_lr_means[c],
                dtype=torch.float32,
                device=self.device,
            )
            prec_safe = torch.tensor(
                self._noise_lr_precisions_safe[c],
                dtype=torch.float32,
                device=self.device,
            )
            inv_diag = torch.tensor(
                self._noise_lr_inv_diag_noise[c],
                dtype=torch.float32,
                device=self.device,
            )
            centered = diff[j] - mu
            # Term 1: Mahalanobis under safe
            term_safe = centered @ prec_safe @ centered
            # Term 2: diagonal Mahalanobis under noise (no centering)
            term_noise = (diff[j] ** 2) * inv_diag
            score = term_safe - term_noise.sum()
            if include_logdet:
                score += self._noise_lr_log_det_diff[c]
            scores[j] = score

        return scores

    def score_samples(self, data: np.ndarray) -> np.ndarray:
        """Compute normalized anomaly scores."""
        self._check_fitted()

        raw_scores = self._score_raw(data)
        # Cache raw scores for downstream diagnostics
        self._last_raw_scores = raw_scores.copy()

        # Normalize
        score_low = cast(float, self.score_low)
        score_high = cast(float, self.score_high)
        scores = (raw_scores - score_low) / (score_high - score_low + 1e-10)
        return np.clip(scores, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save trained model."""
        assert self._fitted, "Must fit before saving"
        assert self.encoder is not None

        # Serialize cluster masks
        masks_serialized = {}
        for c, cm in self.cluster_masks.items():
            masks_serialized[c] = {
                "union_indices": cm["union_indices"].numpy(),
                "mask_values": cm["mask_values"].numpy(),
            }

        state = {
            "method_name": "learned_mask",
            "config": {
                "n_clusters": self.n_clusters,
                "cluster_method": self.cluster_method,
                "outlier_threshold": self.outlier_threshold,
                "l0_filter": self.l0_filter,
                "mask_type": self.mask_type,
                "residual_norm": self.residual_norm,
                "sparsity_weighting": self.sparsity_weighting,
                "lambda_sparsity": self.calibrated_lambda,
                "n_epochs": self.n_epochs,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "sparsity_target": self.sparsity_target,
                "train_decoder": self.train_decoder,
                "per_cluster_decoder": self.per_cluster_decoder,
                "decoder_lr": self.decoder_lr,
                "logit_init": self.logit_init,
                "seed": self.seed,
                "analysis_batch_size": self.analysis_batch_size,
                "whitened_mode": self.whitened_mode,
                "shrinkage_alpha": self.shrinkage_alpha,
                "small_cluster_threshold": (self.small_cluster_threshold),
                "variance_floor": self.variance_floor,
                "scoring_method": self.scoring_method,
                "global_mask": self.global_mask,
                "noise_lr_min_cluster_size": (self.noise_lr_min_cluster_size),
            },
            "cluster_masks": masks_serialized,
            "cluster_centers_raw": self.cluster_centers_raw,
            "cluster_labels": self._diagnostic_cluster_labels,
            "score_low": self.score_low,
            "score_high": self.score_high,
            "sae_dim": self.sae_dim,
            "input_dim": self.input_dim,
            # Save full autoencoder for exact architecture preservation
            "full_autoencoder": self.full_autoencoder,
            # Whitened residual state (None if not whitened)
            "residual_mu_c": self.residual_mu_c,
            "residual_var_c": self.residual_var_c,
            "residual_mu_pooled": self.residual_mu_pooled,
            "residual_var_pooled": self.residual_var_pooled,
            # Full covariance state (None if diagonal or not whitened)
            "residual_precisions": self.residual_precisions,
            "residual_means": self.residual_means,
            # Per-cluster decoder states (None if shared or no decoder training)
            "cluster_decoders": (
                self._cluster_decoders
                if hasattr(self, "_cluster_decoders")
                else None
            ),
            # Noise LR state (empty dicts if not noise_lr)
            "noise_lr_precisions_safe": self._noise_lr_precisions_safe,
            "noise_lr_inv_diag_noise": self._noise_lr_inv_diag_noise,
            "noise_lr_means": self._noise_lr_means,
            "noise_lr_log_det_diff": self._noise_lr_log_det_diff,
            "noise_lr_fallback_clusters": list(
                self._noise_lr_fallback_clusters
            ),
        }

        torch.save(state, path)
        logger.info(f"Saved learned_mask model to {path}")

    def load(self, path: str) -> None:
        """Load trained model from disk."""
        state = torch.load(path, map_location=self.device, weights_only=False)

        config = state.get("config", {})
        self.n_clusters = config.get("n_clusters", 1500)
        self.cluster_method = config.get("cluster_method", "l0")
        self.mask_type = config.get("mask_type", "continuous")
        self.residual_norm = config.get("residual_norm", "l2")
        self.sparsity_weighting = config.get("sparsity_weighting", "uniform")
        self.calibrated_lambda = config.get("lambda_sparsity", 0.01)
        self.whitened_mode = config.get("whitened_mode", "diagonal")
        self.shrinkage_alpha = config.get("shrinkage_alpha", 0.3)
        self.small_cluster_threshold = config.get(
            "small_cluster_threshold", 10
        )
        self.variance_floor = config.get("variance_floor", 1e-6)
        self.scoring_method = config.get("scoring_method", "l2_norm")
        self.global_mask = config.get("global_mask", False)
        self.noise_lr_min_cluster_size = config.get(
            "noise_lr_min_cluster_size", 50
        )
        self.train_decoder = config.get("train_decoder", False)
        self.per_cluster_decoder = config.get("per_cluster_decoder", False)
        self.decoder_lr = config.get("decoder_lr", 1e-4)
        self.sae_dim = state["sae_dim"]
        self.input_dim = state["input_dim"]

        # Whitened residual state (backward-compat: None for old ckpts)
        self.residual_mu_c = state.get("residual_mu_c")
        self.residual_var_c = state.get("residual_var_c")
        self.residual_mu_pooled = state.get("residual_mu_pooled")
        self.residual_var_pooled = state.get("residual_var_pooled")
        self.residual_precisions = state.get("residual_precisions")
        self.residual_means = state.get("residual_means")

        # Per-cluster decoder states
        self._cluster_decoders = state.get("cluster_decoders") or {}

        # Noise LR state (backward-compat: empty for old ckpts)
        self._noise_lr_precisions_safe = state.get(
            "noise_lr_precisions_safe", {}
        )
        self._noise_lr_inv_diag_noise = state.get(
            "noise_lr_inv_diag_noise", {}
        )
        self._noise_lr_means = state.get("noise_lr_means", {})
        self._noise_lr_log_det_diff = state.get("noise_lr_log_det_diff", {})
        self._noise_lr_fallback_clusters = set(
            state.get("noise_lr_fallback_clusters", [])
        )

        # Load cluster masks
        self.cluster_masks = {}
        for c_str, cm in state["cluster_masks"].items():
            c = int(c_str)
            self.cluster_masks[c] = {
                "union_indices": torch.tensor(
                    cm["union_indices"], dtype=torch.long
                ),
                "mask_values": torch.tensor(cm["mask_values"]),
            }

        # Load cluster centers and recreate FAISS index
        self.cluster_centers_raw = state["cluster_centers_raw"]
        if faiss is None:
            raise ImportError("faiss required for inference")
        centroids = self.cluster_centers_raw.astype(np.float32).copy()
        d = centroids.shape[1]
        self.kmeans = faiss.Kmeans(
            d=d, k=self.n_clusters, niter=1, verbose=False
        )
        self.kmeans.train(centroids)
        self.kmeans.centroids = centroids

        # Score normalization
        self.score_low = state["score_low"]
        self.score_high = state["score_high"]

        # Load training cluster assignments (may be None for old models)
        self._diagnostic_cluster_labels = state.get("cluster_labels")

        # Load full autoencoder (preserves exact architecture)
        autoencoder = state["full_autoencoder"]
        autoencoder = autoencoder.to(self.device)
        autoencoder.eval()
        for p in autoencoder.parameters():
            p.requires_grad = False
        self.full_autoencoder = autoencoder
        self.encoder = autoencoder.encoder
        self.decoder = autoencoder.decoder

        self._fitted = True
        logger.info(
            f"Loaded learned_mask: {self.n_clusters} clusters, "
            f"sae_dim={self.sae_dim}"
        )

    # ------------------------------------------------------------------
    # Post-training diagnostics
    # ------------------------------------------------------------------

    def post_training_callback(
        self,
        save_path: str,
        cfg: DictConfig,
        device: torch.device,
    ) -> None:
        """Run cluster diagnostics after training."""
        if self._diagnostic_encoded is None:
            logger.warning("No diagnostic data. Skipping diagnostics.")
            return

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            from unsupervised_llm_safety.analysis.cluster_diagnostics import (
                compute_cluster_diagnostics,
                format_table,
                plot_cluster_sizes,
            )

            diagnostics_dir = (
                Path(save_path).parent / "diagnostics" / "clusters"
            )
            diagnostics_dir.mkdir(parents=True, exist_ok=True)

            # Save calibration log
            calib_log = {}
            if self._calibration_log:
                calib_log.update(self._calibration_log)
            if hasattr(self, "_sparsity_summary"):
                calib_log["sparsity_summary"] = self._sparsity_summary
            if calib_log:
                calib_path = (
                    Path(save_path).parent
                    / "diagnostics"
                    / "calibration_log.json"
                )
                calib_path.parent.mkdir(parents=True, exist_ok=True)
                with open(calib_path, "w") as f:
                    json.dump(calib_log, f, indent=2)
                logger.info(f"Saved calibration log to {calib_path}")

            # Save training cluster assignments as .npy
            labels = self._diagnostic_cluster_labels
            if labels is not None:
                train_clusters_path = (
                    Path(save_path).parent / "cluster_assignments_train.npy"
                )
                np.save(train_clusters_path, labels)
                logger.info(
                    f"Saved training cluster assignments "
                    f"({len(labels)} samples) to "
                    f"{train_clusters_path}"
                )

            data = self._diagnostic_encoded
            d_sae = data.shape[1]

            logger.info("Running cluster diagnostics...")
            cluster_table, centroids = compute_cluster_diagnostics(
                data, labels, self.n_clusters
            )

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

            plot_cluster_sizes(
                cluster_table,
                diagnostics_dir / "cluster_sizes.png",
            )

            # --- Per-cluster mask stats ---
            mask_stats = []
            for c in range(self.n_clusters):
                cm = self.cluster_masks.get(c)
                if cm is None or len(cm["union_indices"]) == 0:
                    mask_stats.append(
                        {
                            "cluster": c,
                            "union_size_before": 0,
                            "union_size_after": 0,
                            "mask_sparsity_01": 0,
                            "mask_sparsity_001": 0,
                            "mask_sparsity_05": 0,
                            "mask_mean": 0.0,
                        }
                    )
                    continue
                vals = cm["mask_values"]
                mask_stats.append(
                    {
                        "cluster": c,
                        "union_size_before": len(cm["union_indices"]),
                        "union_size_after": int((vals.abs() > 0.1).sum()),
                        "mask_sparsity_001": int((vals.abs() > 0.01).sum()),
                        "mask_sparsity_01": int((vals.abs() > 0.1).sum()),
                        "mask_sparsity_05": int((vals.abs() > 0.5).sum()),
                        "mask_mean": round(float(vals.abs().mean()), 4),
                    }
                )

            # --- Plot: union before vs after masking ---
            nonempty = [m for m in mask_stats if m["union_size_before"] > 0]
            # Sort by union_size_before (largest first)
            nonempty_sorted = sorted(
                nonempty, key=lambda x: -x["union_size_before"]
            )
            befores = [m["union_size_before"] for m in nonempty_sorted]
            afters = [m["union_size_after"] for m in nonempty_sorted]

            fig, ax = plt.subplots(figsize=(12, 5))
            x = np.arange(len(befores))
            ax.bar(
                x,
                befores,
                alpha=0.6,
                label="Before mask (union)",
                color="steelblue",
            )
            ax.bar(
                x,
                afters,
                alpha=0.8,
                label="After mask (>0.1)",
                color="coral",
            )
            ax.set_xlabel("Cluster (sorted by union size, largest first)")
            ax.set_ylabel("Number of active features")
            ax.set_title("Union of support: before vs after masking")
            ax.legend()
            fig.tight_layout()
            fig.savefig(
                diagnostics_dir / "union_before_after_masking.png",
                dpi=150,
            )
            plt.close(fig)

            # Save table and JSON
            table_md = format_table(cluster_table)
            (diagnostics_dir / "cluster_table.md").write_text(table_md)

            results = {
                "n_clusters": self.n_clusters,
                "d_sae": d_sae,
                "n_samples": len(data),
                "cluster_method": self.cluster_method,
                "calibrated_lambda": self.calibrated_lambda,
                "largest_cluster_pct": max_pct,
                "empty_clusters": sum(1 for s in sizes if s == 0),
                "union_support_mean": round(float(np.mean(unions)), 0),
                "union_support_max": int(np.max(unions)),
                "cluster_table": cluster_table,
                "mask_stats": mask_stats,
            }
            with open(diagnostics_dir / "cluster_diagnostics.json", "w") as f:
                json.dump(results, f, indent=2)

            # --- Save and plot loss curves ---
            if self._loss_curves:
                loss_curves_path = diagnostics_dir / "mask_loss_curves.json"
                # Save all curves (keyed by cluster id)
                serializable = {
                    str(c): curve for c, curve in self._loss_curves.items()
                }
                with open(loss_curves_path, "w") as f:
                    json.dump(serializable, f)
                logger.info(
                    f"Saved {len(serializable)} loss curves "
                    f"to {loss_curves_path}"
                )

                # Plot summary: sample of curves + aggregate
                sample_ids = sorted(self._loss_curves.keys())
                # Pick ~10 evenly spaced clusters
                if len(sample_ids) > 10:
                    step = len(sample_ids) // 10
                    sample_ids = sample_ids[::step][:10]
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                # Left: individual loss curves
                ax = axes[0]
                for c in sample_ids:
                    curve = self._loss_curves[c]
                    ax.plot(curve, alpha=0.6, label=f"c={c}")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title("Mask training loss (sample)")
                ax.legend(fontsize=7, ncol=2)

                # Right: aggregate (mean + p25/p75)
                ax = axes[1]
                all_curves = [
                    cv for cv in self._loss_curves.values() if len(cv) > 0
                ]
                if all_curves:
                    max_len = max(len(cv) for cv in all_curves)
                    padded = np.full((len(all_curves), max_len), np.nan)
                    for i, cv in enumerate(all_curves):
                        padded[i, : len(cv)] = cv
                    mean_curve = np.nanmean(padded, axis=0)
                    p25 = np.nanpercentile(padded, 25, axis=0)
                    p75 = np.nanpercentile(padded, 75, axis=0)
                    epochs_x = np.arange(max_len)
                    ax.plot(
                        epochs_x,
                        mean_curve,
                        color="steelblue",
                        label="Mean",
                    )
                    ax.fill_between(
                        epochs_x,
                        p25,
                        p75,
                        alpha=0.3,
                        color="steelblue",
                        label="p25-p75",
                    )
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title(f"Aggregate loss ({len(all_curves)} clusters)")
                ax.legend()

                fig.tight_layout()
                fig.savefig(
                    diagnostics_dir / "mask_loss_curves.png",
                    dpi=150,
                )
                plt.close(fig)

            logger.info(f"Diagnostics saved to {diagnostics_dir}")

        except Exception as e:
            logger.error(f"Cluster diagnostics failed: {e}", exc_info=True)
