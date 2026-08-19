"""Per-cluster LoRA adaptation on top of learned_mask.

Adds low-rank adapters (LoRA) to the SAE encoder and decoder on a
per-cluster basis, building on a trained learned_mask model.  The base
encoder/decoder and per-cluster masks are frozen; only the LoRA
parameters are optimised.

Two modes:
  Mode A  --  ``step1_run_dir`` points to a completed learned_mask run.
              Step-1 artifacts are loaded, LoRA is trained on top.
  Mode B  --  ``step1_run_dir`` is None.  The full learned_mask pipeline
              runs internally first, then LoRA is trained.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, cast

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader, TensorDataset

from unsupervised_llm_safety.anodet_algos.base import AnodetMethod
from unsupervised_llm_safety.anodet_algos.learned_mask import LearnedMaskAnodet

logger = logging.getLogger(__name__)

# -- L0-explosion thresholds --------------------------------------------------
_L0_WARN_RATIO = 1.5
_L0_LOUD_RATIO = 3.0
_L0_HARD_RATIO = 10.0
_L0_ABORT_FRAC = 0.20  # abort run if >20% of clusters hit hard error


class LearnedMaskLoRAAnodet(AnodetMethod):
    """Per-cluster LoRA adaptation on top of learned_mask."""

    def __init__(
        self,
        # LoRA parameters
        step1_run_dir: Optional[str] = None,
        lora_rank: int = 8,
        lora_lr: float = 1e-3,
        lora_epochs: int = 30,
        lora_batch_size: int = 256,
        lora_sparsity_lambda: float = 1e-3,
        lora_min_cluster_size: int = 20,
        # Size-dependent rank schedule
        lora_rank_schedule: Optional[Dict[str, int]] = None,
        rank_schedule_small_max: int = 500,
        rank_schedule_medium_max: int = 2000,
        run_expensive_diagnostics: bool = True,
        # Scoring method: "l2_norm" or "mahalanobis"
        scoring_method: str = "l2_norm",
        mahalanobis_reg: float = 1e-4,
        # Step-1 params (Mode B only -- forwarded to LearnedMaskAnodet)
        device: str = "cuda",
        seed: int = 42,
        analysis_batch_size: int = 256,
        pretrained_encoder=None,
        **step1_kwargs,
    ):
        super().__init__()
        self.step1_run_dir = step1_run_dir
        self.lora_rank = lora_rank
        self.lora_lr = lora_lr
        self.lora_epochs = lora_epochs
        self.lora_batch_size = lora_batch_size
        self.lora_sparsity_lambda = lora_sparsity_lambda
        self.lora_min_cluster_size = lora_min_cluster_size
        self.lora_rank_schedule = lora_rank_schedule
        self.rank_schedule_small_max = rank_schedule_small_max
        self.rank_schedule_medium_max = rank_schedule_medium_max
        self.run_expensive_diagnostics = run_expensive_diagnostics
        self.scoring_method = scoring_method
        self.mahalanobis_reg = mahalanobis_reg

        self.device = torch.device(device)
        self.seed = seed
        self.analysis_batch_size = analysis_batch_size

        # Store step-1 kwargs for Mode B construction
        self._step1_kwargs = dict(
            step1_kwargs,
            device=device,
            seed=seed,
            analysis_batch_size=analysis_batch_size,
        )
        self._pretrained_encoder = pretrained_encoder

        # Internal state (populated during fit / load)
        self._step1: Optional[LearnedMaskAnodet] = None
        self._lora_adapters: Dict[int, Dict[str, torch.Tensor]] = {}
        self._skipped_clusters: Set[int] = set()

        # Recomputed post-LoRA whitened stats
        self.residual_mu_c: Optional[np.ndarray] = None
        self.residual_var_c: Optional[np.ndarray] = None
        self.residual_mu_pooled: Optional[np.ndarray] = None
        self.residual_var_pooled: Optional[np.ndarray] = None

        # Per-cluster full-covariance Mahalanobis state
        # (populated when scoring_method="mahalanobis")
        self.residual_means: Optional[Dict[int, np.ndarray]] = None
        self.residual_precisions: Optional[Dict[int, np.ndarray]] = None

        # Score normalization (recomputed post-LoRA)
        self.score_low: Optional[float] = None
        self.score_high: Optional[float] = None

        # Eval state (set during score_samples for diagnostics)
        self._last_eval_clusters: Optional[np.ndarray] = None
        self._last_raw_scores: Optional[np.ndarray] = None

        # Diagnostics
        self._lora_loss_curves: Dict[int, Dict[str, List[float]]] = {}
        self._lora_diagnostic_summary: Dict[str, Any] = {}
        self._lora_delta_ratios: Dict[int, Dict[str, float]] = {}
        self._nan_events: Dict[int, Dict[str, int]] = {}
        self._train_cluster_sizes: Dict[int, int] = {}
        self._cluster_quality: Dict[int, Dict[str, float]] = {}
        self._score_samples_call_count: int = 0
        self._all_eval_cluster_assignments: List[np.ndarray] = []

    @property
    def name(self) -> str:
        return "learned_mask_lora"

    def __getattr__(self, name: str):
        """Delegate attribute lookups to step-1 for diagnostics compat.

        Attributes like n_clusters, cluster_masks, kmeans, encoder,
        decoder are accessed by diagnose_learned_mask.py on the baseline
        object.  Rather than duplicating them, we proxy to _step1.
        """
        # Avoid infinite recursion during init (before _step1 exists)
        if name == "_step1":
            raise AttributeError(name)
        step1 = self.__dict__.get("_step1")
        if step1 is not None and hasattr(step1, name):
            return getattr(step1, name)
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    # ------------------------------------------------------------------
    # LoRA helpers
    # ------------------------------------------------------------------

    def _rank_for_cluster(self, n_c: int) -> int:
        """Determine LoRA rank for a cluster based on its size."""
        if self.lora_rank_schedule is None:
            return self.lora_rank
        if n_c < self.rank_schedule_small_max:
            return self.lora_rank_schedule.get("small", self.lora_rank)
        elif n_c < self.rank_schedule_medium_max:
            return self.lora_rank_schedule.get("medium", self.lora_rank)
        else:
            return self.lora_rank_schedule.get("large", self.lora_rank)

    def _compute_rank_distribution(self) -> Dict[str, int]:
        """Count how many adapted clusters use each rank."""
        counts: Dict[int, int] = {}
        for lora in self._lora_adapters.values():
            r = lora["A_E"].shape[0]
            counts[r] = counts.get(r, 0) + 1
        return {f"rank_{r}": n for r, n in sorted(counts.items())}

    def _init_lora_for_cluster(
        self, cluster_id: int, rank: Optional[int] = None
    ) -> None:
        """Initialise LoRA matrices for one cluster (A Kaiming, B zero)."""
        s1 = self._step1
        assert s1 is not None
        in_dim = s1.input_dim  # 4096
        sae_dim = s1.sae_dim  # 16384
        r = rank if rank is not None else self.lora_rank

        A_E = torch.empty(r, in_dim, device=self.device)
        torch.nn.init.kaiming_uniform_(A_E)
        B_E = torch.zeros(sae_dim, r, device=self.device)

        A_D = torch.empty(r, sae_dim, device=self.device)
        torch.nn.init.kaiming_uniform_(A_D)
        B_D = torch.zeros(in_dim, r, device=self.device)

        self._lora_adapters[cluster_id] = {
            "A_E": A_E,
            "B_E": B_E,
            "A_D": A_D,
            "B_D": B_D,
        }

    def _encode_with_lora(
        self, x: torch.Tensor, cluster_id: int
    ) -> torch.Tensor:
        """Encode via base linear + LoRA, then ReLU.  (B, in_dim) -> (B, sae_dim)."""
        s1 = self._step1
        assert s1 is not None and s1.encoder is not None
        enc_linear = s1.encoder[0]
        lora = self._lora_adapters[cluster_id]
        z_pre = F.linear(x, enc_linear.weight, enc_linear.bias)
        z_pre = z_pre + (x @ lora["A_E"].T) @ lora["B_E"].T
        return F.relu(z_pre)

    def _decode_with_lora(
        self, z_masked: torch.Tensor, cluster_id: int
    ) -> torch.Tensor:
        """Decode via base linear + LoRA (no activation).  (B, sae_dim) -> (B, in_dim)."""
        s1 = self._step1
        assert s1 is not None and s1.decoder is not None
        dec_linear = s1.decoder[0]
        lora = self._lora_adapters[cluster_id]
        return (
            F.linear(z_masked, dec_linear.weight, dec_linear.bias)
            + (z_masked @ lora["A_D"].T) @ lora["B_D"].T
        )

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, data: np.ndarray, labels: np.ndarray) -> None:
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # --- Step A: get step-1 state ---
        if self.step1_run_dir is not None:
            ckpt_path = Path(self.step1_run_dir) / "learned_mask_trained.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Step-1 checkpoint not found: {ckpt_path}"
                )
            logger.info(f"Mode A: loading step-1 from {ckpt_path}")
            self._step1 = LearnedMaskAnodet(device=str(self.device))
            self._step1.load(str(ckpt_path))
        else:
            logger.info("Mode B: fitting step-1 from scratch")
            self._step1 = LearnedMaskAnodet(
                pretrained_encoder=self._pretrained_encoder,
                **self._step1_kwargs,
            )
            self._step1.fit(data, labels)

        s1 = self._step1
        assert s1.encoder is not None and s1.decoder is not None

        # Freeze base encoder/decoder
        s1.encoder.eval()
        s1.decoder.eval()
        for p in s1.encoder.parameters():
            p.requires_grad = False
        for p in s1.decoder.parameters():
            p.requires_grad = False

        # --- Step B: assign clusters on current data ---
        safe_mask = labels == 1
        safe_data = data[safe_mask]

        logger.info(
            f"Encoding {len(safe_data)} safe samples for "
            f"cluster assignment..."
        )
        encoded = s1._encode_all(safe_data)
        data_f32 = encoded.astype(np.float32)
        assert s1.kmeans is not None
        _, assignments = s1.kmeans.index.search(data_f32, 1)
        cluster_labels = assignments.flatten()

        n_clusters = s1.n_clusters
        cluster_sizes = np.bincount(cluster_labels, minlength=n_clusters)
        logger.info(
            f"Cluster assignment: {n_clusters} clusters, "
            f"median size={int(np.median(cluster_sizes[cluster_sizes > 0]))}"
        )

        # --- Profiling: train 3 clusters to validate timing ---
        self._profile_lora_training(
            safe_data, encoded, cluster_labels, n_clusters
        )

        # --- Step C: train per-cluster LoRA ---
        logger.info(
            f"Training LoRA (rank={self.lora_rank}) for "
            f"{n_clusters} clusters..."
        )

        # Pre-compute per-cluster L0 (base encoder) for sparsity checks
        pre_lora_l0: Dict[int, float] = {}

        # D6: holdout storage for cluster quality proxies
        holdout_raw: Dict[int, np.ndarray] = {}
        holdout_enc: Dict[int, np.ndarray] = {}

        n_hard_errors = 0
        for c in range(n_clusters):
            n_c = int(cluster_sizes[c])
            if n_c < self.lora_min_cluster_size or n_c == 0:
                self._skipped_clusters.add(c)
                continue

            c_mask = cluster_labels == c
            c_raw = safe_data[c_mask]
            c_enc = encoded[c_mask]

            # D6: 80/20 split for cluster quality proxies
            n_holdout = max(1, int(0.2 * len(c_raw)))
            rng_split = np.random.RandomState(self.seed + c)
            perm = rng_split.permutation(len(c_raw))
            holdout_idx = perm[:n_holdout]
            train_idx = perm[n_holdout:]
            holdout_raw[c] = c_raw[holdout_idx]
            holdout_enc[c] = c_enc[holdout_idx]
            c_raw_train = c_raw[train_idx]
            c_enc_train = c_enc[train_idx]

            # Pre-LoRA L0
            pre_l0 = float((c_enc > 0).sum(axis=1).astype(float).mean())
            pre_lora_l0[c] = pre_l0

            # Init and train LoRA (with size-dependent rank)
            cluster_rank = self._rank_for_cluster(n_c)
            self._init_lora_for_cluster(c, rank=cluster_rank)
            loss_curve = self._train_lora_for_cluster(
                c, c_raw_train, c_enc_train
            )
            self._lora_loss_curves[c] = loss_curve

            # Post-LoRA L0 check
            post_l0 = self._compute_post_lora_l0(c, c_raw)
            ratio = post_l0 / max(pre_l0, 1.0)

            # D3: discard LoRA if >50% of batches had NaN/Inf
            nan_info = self._nan_events.get(c, {})
            nan_total = nan_info.get("nan_loss", 0) + nan_info.get(
                "nan_grad", 0
            )
            nan_batch_total = nan_info.get("total_batches", 1)
            if nan_batch_total > 0 and nan_total / nan_batch_total > 0.50:
                logger.error(
                    f"  CRITICAL cluster {c}: {nan_total}/{nan_batch_total} "
                    f"batches NaN/Inf ({nan_total / nan_batch_total:.0%}). "
                    f"Discarding LoRA, falling back to base."
                )
                del self._lora_adapters[c]
                self._skipped_clusters.add(c)
                n_hard_errors += 1
                continue

            if ratio >= _L0_HARD_RATIO:
                logger.error(
                    f"  HARD ERROR cluster {c}: L0 ratio {ratio:.1f}x "
                    f"(pre={pre_l0:.0f}, post={post_l0:.0f}). "
                    f"Discarding LoRA, falling back to base."
                )
                del self._lora_adapters[c]
                self._skipped_clusters.add(c)
                n_hard_errors += 1
            elif ratio >= _L0_LOUD_RATIO:
                logger.error(
                    f"  LOUD WARNING cluster {c}: L0 ratio {ratio:.1f}x "
                    f"(pre={pre_l0:.0f}, post={post_l0:.0f})"
                )
            elif ratio >= _L0_WARN_RATIO:
                logger.warning(
                    f"  L0 warning cluster {c}: ratio {ratio:.1f}x "
                    f"(pre={pre_l0:.0f}, post={post_l0:.0f})"
                )

            total_curve = loss_curve["total"]
            if c % 50 == 0 or c == n_clusters - 1:
                recon_final = (
                    loss_curve["recon"][-1] if loss_curve["recon"] else 0.0
                )
                sparsity_final = (
                    loss_curve["sparsity"][-1]
                    if loss_curve["sparsity"]
                    else 0.0
                )
                logger.info(
                    f"  Cluster {c}/{n_clusters}: n={n_c}, rank={cluster_rank}, "
                    f"L0 {pre_l0:.0f}->{post_l0:.0f} "
                    f"({ratio:.2f}x), "
                    f"loss {total_curve[0]:.4f}->{total_curve[-1]:.4f} "
                    f"(recon={recon_final:.4f}, sparsity={sparsity_final:.4f})"
                )

        # Check abort condition
        n_with_lora_attempted = n_clusters - len(
            [
                c
                for c in range(n_clusters)
                if cluster_sizes[c] < self.lora_min_cluster_size
                or cluster_sizes[c] == 0
            ]
        )
        if (
            n_with_lora_attempted > 0
            and n_hard_errors / n_with_lora_attempted > _L0_ABORT_FRAC
        ):
            raise RuntimeError(
                f"ABORT: {n_hard_errors}/{n_with_lora_attempted} clusters "
                f"hit L0 hard error (>{_L0_ABORT_FRAC:.0%} threshold). "
                f"Increase lora_sparsity_lambda or reduce lora_rank."
            )

        logger.info(
            f"LoRA training complete: {len(self._lora_adapters)} adapted, "
            f"{len(self._skipped_clusters)} skipped, "
            f"{n_hard_errors} hard errors"
        )

        # --- Step D: recompute whitened stats from ALL post-LoRA residuals ---
        logger.info("Recomputing whitened residual stats (post-LoRA)...")
        self._compute_whitened_stats_post_lora(
            safe_data, encoded, cluster_labels, n_clusters
        )

        # --- Step D2: fit Mahalanobis if requested ---
        if self.scoring_method == "mahalanobis":
            logger.info("Fitting per-cluster Mahalanobis (post-LoRA)...")
            self._fit_mahalanobis_post_lora(
                safe_data, encoded, cluster_labels, n_clusters
            )

        # --- Step E: recompute score normalization ---
        logger.info("Recomputing score normalization (post-LoRA)...")
        train_scores = self._score_raw(safe_data, cluster_labels)
        self.score_low = float(np.percentile(train_scores, 5))
        self.score_high = float(np.percentile(train_scores, 95))
        logger.info(
            f"Score normalization: p5={self.score_low:.4f}, "
            f"p95={self.score_high:.4f}"
        )

        # --- Step F: reassignment diagnostic ---
        reassign_frac = self._compute_reassignment_fraction(
            safe_data, cluster_labels
        )
        logger.info(
            f"Reassignment fraction (base vs LoRA routing): "
            f"{reassign_frac:.4f}"
        )
        if reassign_frac > 0.05:
            logger.warning(
                f"Reassignment fraction {reassign_frac:.4f} > 5%! "
                f"LoRA may be changing encodings too aggressively."
            )

        # D5: save train cluster sizes for post-hoc eval analysis
        self._train_cluster_sizes = {
            int(c): int(cluster_sizes[c]) for c in range(n_clusters)
        }

        # D6: compute per-cluster quality proxies
        logger.info("Computing per-cluster quality proxies...")
        for c in range(n_clusters):
            if cluster_sizes[c] == 0 or c in self._skipped_clusters:
                continue
            quality = self._compute_cluster_quality(
                c,
                safe_data,
                encoded,
                cluster_labels,
                holdout_raw.get(c),
                holdout_enc.get(c),
            )
            self._cluster_quality[c] = quality
        logger.info(
            f"Computed quality proxies for {len(self._cluster_quality)} "
            f"clusters"
        )

        # D1: compute final loss component medians across clusters
        final_recon = [
            cv["recon"][-1]
            for cv in self._lora_loss_curves.values()
            if cv["recon"]
        ]
        final_sparsity = [
            cv["sparsity"][-1]
            for cv in self._lora_loss_curves.values()
            if cv["sparsity"]
        ]

        # D2: aggregate delta ratio stats
        enc_ratios = [r["encoder"] for r in self._lora_delta_ratios.values()]
        dec_ratios = [r["decoder"] for r in self._lora_delta_ratios.values()]
        median_enc_ratio = float(np.median(enc_ratios)) if enc_ratios else 0.0
        median_dec_ratio = float(np.median(dec_ratios)) if dec_ratios else 0.0
        if median_enc_ratio > 0.5:
            logger.warning(
                f"Median encoder delta ratio {median_enc_ratio:.4f} > 0.5 — "
                f"LoRA is no longer a small correction!"
            )

        # D3: aggregate NaN stats
        clusters_with_nan = [
            c
            for c, ev in self._nan_events.items()
            if ev["nan_loss"] + ev["nan_grad"] > 0
        ]
        total_nan_loss = sum(
            ev["nan_loss"] for ev in self._nan_events.values()
        )
        total_nan_grad = sum(
            ev["nan_grad"] for ev in self._nan_events.values()
        )

        # Build diagnostic summary
        self._lora_diagnostic_summary = {
            "n_clusters_total": n_clusters,
            "n_clusters_with_lora": len(self._lora_adapters),
            "n_clusters_skipped": len(self._skipped_clusters),
            "n_hard_errors": n_hard_errors,
            "lora_rank": self.lora_rank,
            "lora_rank_schedule": (
                dict(self.lora_rank_schedule)
                if self.lora_rank_schedule is not None
                else None
            ),
            "rank_distribution": self._compute_rank_distribution(),
            "lora_epochs": self.lora_epochs,
            "lora_sparsity_lambda": self.lora_sparsity_lambda,
            "reassignment_fraction": round(reassign_frac, 6),
            # D1: loss decomposition
            "final_recon_loss_median": (
                round(float(np.median(final_recon)), 6)
                if final_recon
                else None
            ),
            "final_sparsity_loss_median": (
                round(float(np.median(final_sparsity)), 6)
                if final_sparsity
                else None
            ),
            # D2: delta norm ratios
            "median_encoder_delta_ratio": round(median_enc_ratio, 6),
            "median_decoder_delta_ratio": round(median_dec_ratio, 6),
            "n_clusters_encoder_ratio_above_0.5": sum(
                1 for r in enc_ratios if r > 0.5
            ),
            "n_clusters_encoder_ratio_above_0.1": sum(
                1 for r in enc_ratios if r > 0.1
            ),
            # D3: NaN events
            "n_clusters_with_nan_events": len(clusters_with_nan),
            "total_nan_loss_batches": total_nan_loss,
            "total_nan_grad_batches": total_nan_grad,
            # D5: train cluster sizes + note
            "train_cluster_sizes": self._train_cluster_sizes,
            "eval_cluster_distribution_note": (
                "Eval-vs-train cluster distribution is computed by "
                "diagnose_learned_mask_lora.py, not post_training_callback. "
                "Use train_cluster_sizes here + the saved "
                "*_cluster_assignments.npz from eval."
            ),
            # D6: cluster quality proxies
            "cluster_quality": {
                str(c): q for c, q in self._cluster_quality.items()
            },
        }

        # D6: aggregate quality proxy stats
        if self._cluster_quality:
            densities = [
                q["mask_density"] for q in self._cluster_quality.values()
            ]
            train_stds = [
                q["train_res_std"] for q in self._cluster_quality.values()
            ]
            mem_gaps = [
                q["memorization_gap"] for q in self._cluster_quality.values()
            ]
            spreads = [
                q["centroid_spread"] for q in self._cluster_quality.values()
            ]
            self._lora_diagnostic_summary.update(
                {
                    "median_mask_density": round(
                        float(np.median(densities)), 6
                    ),
                    "median_train_res_std": round(
                        float(np.median(train_stds)), 6
                    ),
                    "median_memorization_gap": round(
                        float(np.median(mem_gaps)), 6
                    ),
                    "median_centroid_spread": round(
                        float(np.median(spreads)), 6
                    ),
                    "n_clusters_mask_density_above_0.5": sum(
                        1 for d in densities if d > 0.5
                    ),
                    "n_clusters_memorization_gap_above_2x": sum(
                        1
                        for c, q in self._cluster_quality.items()
                        if q["train_res_mean"] > 0
                        and q["holdout_res_mean"] > 2.0 * q["train_res_mean"]
                    ),
                }
            )

        self._fitted = True
        logger.info("LearnedMaskLoRA fit complete.")

    # ------------------------------------------------------------------
    # Per-cluster LoRA training
    # ------------------------------------------------------------------

    def _train_lora_for_cluster(
        self,
        cluster_id: int,
        raw_cluster: np.ndarray,
        enc_cluster: np.ndarray,
    ) -> Dict[str, List[float]]:
        """Train LoRA adapters for one cluster.

        Returns dict with keys "total", "recon", "sparsity", each a
        list of per-epoch average losses.
        """
        s1 = self._step1
        assert s1 is not None
        cm = s1.cluster_masks.get(cluster_id)
        lora = self._lora_adapters[cluster_id]

        # Make LoRA params require grad
        for v in lora.values():
            v.requires_grad_(True)

        optimizer = torch.optim.Adam(list(lora.values()), lr=self.lora_lr)

        raw_t = torch.tensor(raw_cluster, dtype=torch.float32)
        dataset = TensorDataset(raw_t)
        loader = DataLoader(
            dataset,
            batch_size=self.lora_batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )

        # D1: track decomposed loss curves
        epoch_total: List[float] = []
        epoch_recon: List[float] = []
        epoch_sparsity: List[float] = []

        # D3: NaN/Inf counters
        nan_loss_count = 0
        nan_grad_count = 0
        total_batch_count = 0

        for _ in range(self.lora_epochs):
            ep_total = 0.0
            ep_recon = 0.0
            ep_sparsity = 0.0
            n_batches = 0
            for (batch_raw,) in loader:
                batch_raw = batch_raw.to(self.device)
                total_batch_count += 1
                optimizer.zero_grad()

                # Encode with LoRA
                z = self._encode_with_lora(batch_raw, cluster_id)

                # Apply mask
                B = z.shape[0]
                if cm is not None and len(cm["union_indices"]) > 0:
                    idx = cm["union_indices"].to(self.device)
                    vals = cm["mask_values"].to(self.device)
                    z_masked = torch.zeros(B, s1.sae_dim, device=self.device)
                    z_masked[:, idx] = z[:, idx] * vals
                else:
                    z_masked = z

                # Decode with LoRA
                x_hat = self._decode_with_lora(z_masked, cluster_id)

                # Loss
                recon_loss = ((batch_raw - x_hat) ** 2).mean()
                sparsity_loss = z.abs().mean()
                loss = recon_loss + self.lora_sparsity_lambda * sparsity_loss

                # D3: skip batch if loss is NaN/Inf
                if not torch.isfinite(loss):
                    nan_loss_count += 1
                    continue

                loss.backward()

                # D3: skip batch if any gradient is NaN/Inf
                grad_vals = [
                    p.grad.norm().item()
                    for p in lora.values()
                    if p.grad is not None
                ]
                if any(not np.isfinite(g) for g in grad_vals):
                    nan_grad_count += 1
                    optimizer.zero_grad()
                    continue

                optimizer.step()
                ep_total += loss.item()
                ep_recon += recon_loss.item()
                ep_sparsity += sparsity_loss.item()
                n_batches += 1

            denom = max(n_batches, 1)
            epoch_total.append(ep_total / denom)
            epoch_recon.append(ep_recon / denom)
            epoch_sparsity.append(ep_sparsity / denom)

        # Detach LoRA params after training
        for v in lora.values():
            v.requires_grad_(False)

        # D3: record NaN events
        self._nan_events[cluster_id] = {
            "nan_loss": nan_loss_count,
            "nan_grad": nan_grad_count,
            "total_batches": total_batch_count,
        }
        nan_total = nan_loss_count + nan_grad_count
        if total_batch_count > 0 and nan_total > 0:
            nan_frac = nan_total / total_batch_count
            if nan_frac > 0.10:
                logger.warning(
                    f"  Cluster {cluster_id}: {nan_frac:.1%} of batches "
                    f"had NaN/Inf ({nan_loss_count} loss, "
                    f"{nan_grad_count} grad out of {total_batch_count})"
                )

        # D2: compute LoRA delta norm ratios on cluster data.
        # Ratio = mean(||delta_i||) / mean(||base_i||), computed over
        # the full cluster if n <= 256, else a random sample of 256.
        with torch.no_grad():
            n_samples = len(raw_cluster)
            if n_samples <= 256:
                sample_idx = np.arange(n_samples)
            else:
                rng = np.random.RandomState(self.seed + cluster_id)
                sample_idx = rng.choice(n_samples, 256, replace=False)

            x_sample = torch.tensor(
                raw_cluster[sample_idx],
                dtype=torch.float32,
                device=self.device,
            )

            enc_linear = s1.encoder[0]

            # Encoder ratio
            base_enc = F.linear(x_sample, enc_linear.weight, enc_linear.bias)
            delta_enc = (x_sample @ lora["A_E"].T) @ lora["B_E"].T
            mean_base_enc = base_enc.norm(dim=1).mean().item()
            mean_delta_enc = delta_enc.norm(dim=1).mean().item()
            enc_ratio = (
                mean_delta_enc / mean_base_enc
                if mean_base_enc > 1e-12
                else 0.0
            )

            # Decoder ratio: apply mask to get z_masked, then measure
            z_full = F.relu(base_enc + delta_enc)
            if cm is not None and len(cm["union_indices"]) > 0:
                idx = cm["union_indices"].to(self.device)
                vals = cm["mask_values"].to(self.device)
                z_m = torch.zeros(
                    len(sample_idx), s1.sae_dim, device=self.device
                )
                z_m[:, idx] = z_full[:, idx] * vals
            else:
                z_m = z_full

            dec_linear = s1.decoder[0]
            base_dec = F.linear(z_m, dec_linear.weight, dec_linear.bias)
            delta_dec = (z_m @ lora["A_D"].T) @ lora["B_D"].T
            mean_base_dec = base_dec.norm(dim=1).mean().item()
            mean_delta_dec = delta_dec.norm(dim=1).mean().item()
            dec_ratio = (
                mean_delta_dec / mean_base_dec
                if mean_base_dec > 1e-12
                else 0.0
            )

        self._lora_delta_ratios[cluster_id] = {
            "encoder": round(enc_ratio, 6),
            "decoder": round(dec_ratio, 6),
        }

        return {
            "total": epoch_total,
            "recon": epoch_recon,
            "sparsity": epoch_sparsity,
        }

    def _compute_post_lora_l0(
        self, cluster_id: int, raw_cluster: np.ndarray
    ) -> float:
        """Compute mean L0 of LoRA-adapted encoding on cluster data."""
        bs = self.analysis_batch_size
        l0_sum = 0.0
        n = 0
        with torch.no_grad():
            for i in range(0, len(raw_cluster), bs):
                batch = torch.tensor(
                    raw_cluster[i : i + bs],
                    dtype=torch.float32,
                    device=self.device,
                )
                z = self._encode_with_lora(batch, cluster_id)
                l0_sum += float((z > 0).sum(dim=1).float().sum().item())
                n += len(batch)
        return l0_sum / max(n, 1)

    def _profile_lora_training(
        self,
        safe_data: np.ndarray,
        encoded: np.ndarray,
        cluster_labels: np.ndarray,
        n_clusters: int,
    ) -> None:
        """Profile LoRA training on 3 clusters to validate time estimate."""
        cluster_sizes = np.bincount(cluster_labels, minlength=n_clusters)
        # Pick 3 non-trivial clusters at size quantiles
        nonzero = np.where(cluster_sizes >= self.lora_min_cluster_size)[0]
        if len(nonzero) < 3:
            logger.info("Profiling: fewer than 3 eligible clusters, skipping")
            return

        quantiles = [0.25, 0.5, 0.75]
        sizes = cluster_sizes[nonzero]
        targets = np.quantile(sizes, quantiles, method="nearest")
        profile_ids = []
        used = set()
        for ts in targets:
            diffs = np.abs(sizes - ts)
            for idx in np.argsort(diffs):
                cid = int(nonzero[idx])
                if cid not in used:
                    profile_ids.append(cid)
                    used.add(cid)
                    break

        logger.info(
            f"Profiling LoRA on {len(profile_ids)} clusters: "
            + ", ".join(f"c{c}(n={cluster_sizes[c]})" for c in profile_ids)
        )

        t0 = time.time()
        for c in profile_ids:
            c_mask = cluster_labels == c
            c_raw = safe_data[c_mask]
            c_enc = encoded[c_mask]
            rank = self._rank_for_cluster(int(cluster_sizes[c]))
            self._init_lora_for_cluster(c, rank=rank)
            self._train_lora_for_cluster(c, c_raw, c_enc)
            del self._lora_adapters[c]
        elapsed = time.time() - t0

        n_eligible = len(nonzero)
        est_total = elapsed / len(profile_ids) * n_eligible
        logger.info(
            f"Profile: {len(profile_ids)} clusters in {elapsed:.1f}s "
            f"({elapsed / len(profile_ids):.2f}s/cluster). "
            f"Estimated total for {n_eligible}: {est_total:.0f}s "
            f"({est_total / 60:.1f}min)"
        )

    # ------------------------------------------------------------------
    # D6: Cluster quality proxies
    # ------------------------------------------------------------------

    def _compute_cluster_quality(
        self,
        cluster_id: int,
        safe_data: np.ndarray,
        encoded: np.ndarray,
        cluster_labels: np.ndarray,
        holdout_raw: Optional[np.ndarray],
        holdout_enc: Optional[np.ndarray],
    ) -> Dict[str, float]:
        """Compute quality proxy metrics for one cluster.

        Metrics:
          1. mask_density — fraction of SAE dims in union mask
          2. train_res_mean, train_res_std — residual L2 norm stats
             on training portion
          3. holdout_res_mean, memorization_gap — overfitting gauge
          4. centroid_spread — L2 spread normalized by global spread
          5. mask_support_ratio — |mask support| / mean_L0
        """
        s1 = self._step1
        assert s1 is not None
        cm = s1.cluster_masks.get(cluster_id)

        quality: Dict[str, float] = {}

        # 1. Mask density
        sae_dim = s1.sae_dim
        if cm is not None and len(cm["union_indices"]) > 0:
            mask_support = len(cm["union_indices"])
            quality["mask_density"] = round(mask_support / sae_dim, 6)
        else:
            mask_support = sae_dim
            quality["mask_density"] = 1.0

        # Get cluster members (train portion = all members minus holdout)
        c_mask = cluster_labels == cluster_id
        c_raw_all = safe_data[c_mask]
        c_enc_all = encoded[c_mask]

        # Compute train residual norms via _compute_residual_vectors
        c_labels_all = np.full(len(c_raw_all), cluster_id, dtype=np.int64)
        train_residuals = self._compute_residual_vectors(
            c_raw_all, c_enc_all, c_labels_all
        )
        train_norms = np.linalg.norm(train_residuals, axis=1)

        # 2. Train residual stats
        quality["train_res_mean"] = round(float(train_norms.mean()), 6)
        quality["train_res_std"] = round(float(train_norms.std()), 6)

        # 3. Holdout residual stats and memorization gap
        if holdout_raw is not None and len(holdout_raw) > 0:
            h_labels = np.full(len(holdout_raw), cluster_id, dtype=np.int64)
            holdout_residuals = self._compute_residual_vectors(
                holdout_raw, holdout_enc, h_labels
            )
            holdout_norms = np.linalg.norm(holdout_residuals, axis=1)
            quality["holdout_res_mean"] = round(float(holdout_norms.mean()), 6)
            quality["memorization_gap"] = round(
                float(holdout_norms.mean() - train_norms.mean()), 6
            )
        else:
            quality["holdout_res_mean"] = quality["train_res_mean"]
            quality["memorization_gap"] = 0.0

        # 4. Centroid spread (L2 in raw activation space)
        # Mean distance of cluster members to their centroid,
        # normalized by global mean distance to global centroid.
        c_centroid = c_raw_all.mean(axis=0)
        c_dists = np.linalg.norm(c_raw_all - c_centroid, axis=1)
        cluster_mean_dist = float(c_dists.mean())

        global_centroid = safe_data.mean(axis=0)
        # Use a sample for efficiency if data is large
        if len(safe_data) > 5000:
            rng_g = np.random.RandomState(self.seed)
            g_idx = rng_g.choice(len(safe_data), 5000, replace=False)
            g_sample = safe_data[g_idx]
        else:
            g_sample = safe_data
        global_dists = np.linalg.norm(g_sample - global_centroid, axis=1)
        global_mean_dist = float(global_dists.mean())

        quality["centroid_spread"] = round(
            cluster_mean_dist / max(global_mean_dist, 1e-12), 6
        )

        # 5. Mask support ratio: |mask support| / mean_L0
        mean_l0 = float((c_enc_all > 0).sum(axis=1).astype(float).mean())
        quality["mask_support_ratio"] = round(
            mask_support / max(mean_l0, 1.0), 6
        )

        return quality

    # ------------------------------------------------------------------
    # Mahalanobis fitting on residuals
    # ------------------------------------------------------------------

    def _fit_mahalanobis_post_lora(
        self,
        safe_data: np.ndarray,
        encoded: np.ndarray,
        cluster_labels: np.ndarray,
        n_clusters: int,
    ) -> None:
        """Fit per-cluster full-covariance Mahalanobis on post-LoRA residuals.

        Computes residuals through the LoRA-adapted pipeline, then fits
        per-cluster mean and precision matrix.  Small clusters (< raw_dim
        samples) get higher regularization for numerical stability.
        """
        residuals = self._compute_residual_vectors(
            safe_data, encoded, cluster_labels
        )
        raw_dim = residuals.shape[1]
        reg = self.mahalanobis_reg
        cluster_sizes = np.bincount(cluster_labels, minlength=n_clusters)

        self.residual_means = {}
        self.residual_precisions = {}

        for c in range(n_clusters):
            n_c = int(cluster_sizes[c])
            if n_c < 2:
                continue
            c_resid = residuals[cluster_labels == c]
            mean = c_resid.mean(axis=0).astype(np.float32)
            self.residual_means[c] = mean

            effective_reg = max(reg, 0.1) if n_c < raw_dim else reg
            centered = (c_resid - mean).astype(np.float64)
            cov = centered.T @ centered / max(n_c - 1, 1)
            cov += effective_reg * np.eye(raw_dim)
            try:
                precision = np.linalg.inv(cov).astype(np.float32)
            except np.linalg.LinAlgError:
                precision = np.linalg.pinv(cov).astype(np.float32)
            self.residual_precisions[c] = precision

        logger.info(
            f"Fitted Mahalanobis on {len(self.residual_means)}/{n_clusters} "
            f"clusters (reg={reg}, dim={raw_dim})"
        )

    def _score_mahalanobis(
        self,
        diff: torch.Tensor,
        batch_clusters: np.ndarray,
    ) -> torch.Tensor:
        """Score residual batch using per-cluster full Mahalanobis.

        Args:
            diff: (B, raw_dim) residual vectors on device.
            batch_clusters: (B,) cluster assignments.

        Returns:
            (B,) Mahalanobis distance scores.
        """
        assert self.residual_means is not None
        assert self.residual_precisions is not None
        B = diff.shape[0]
        scores = torch.zeros(B, device=self.device)
        diff_np = diff.cpu().numpy()

        for c_id in np.unique(batch_clusters):
            c_id = int(c_id)
            c_idx = np.where(batch_clusters == c_id)[0]
            if c_id not in self.residual_means:
                # Fallback to L2 for unfitted clusters
                scores[c_idx] = diff[c_idx].norm(dim=1)
                continue
            c_diff = diff_np[c_idx] - self.residual_means[c_id]
            prec = self.residual_precisions[c_id]
            maha = np.sqrt(np.sum((c_diff @ prec) * c_diff, axis=1) + 1e-10)
            scores[c_idx] = torch.from_numpy(maha).to(self.device)

        return scores

    # ------------------------------------------------------------------
    # Whitened stats recomputation
    # ------------------------------------------------------------------

    def _compute_whitened_stats_post_lora(
        self,
        safe_data: np.ndarray,
        encoded: np.ndarray,
        cluster_labels: np.ndarray,
        n_clusters: int,
    ) -> None:
        """Recompute whitened residual stats using post-LoRA pipeline.

        Computes residuals for ALL clusters (including skipped), derives
        new pooled stats, then per-cluster with shrinkage.
        """
        s1 = self._step1
        assert s1 is not None

        # Compute residuals for all samples
        residuals = self._compute_residual_vectors(
            safe_data, encoded, cluster_labels
        )
        raw_dim = residuals.shape[1]
        cluster_sizes = np.bincount(cluster_labels, minlength=n_clusters)

        # New pooled stats from ALL residuals
        self.residual_mu_pooled = residuals.mean(axis=0).astype(np.float32)
        self.residual_var_pooled = residuals.var(axis=0).astype(np.float32)

        # Per-cluster stats
        small_thresh = s1.small_cluster_threshold
        alpha = s1.shrinkage_alpha
        variance_floor = s1.variance_floor

        mu_c = np.zeros((n_clusters, raw_dim), dtype=np.float32)
        var_c_raw = np.zeros((n_clusters, raw_dim), dtype=np.float32)

        for c in range(n_clusters):
            n_c = int(cluster_sizes[c])
            if n_c < small_thresh:
                mu_c[c] = self.residual_mu_pooled
                var_c_raw[c] = self.residual_var_pooled
            else:
                c_res = residuals[cluster_labels == c]
                mu_c[c] = c_res.mean(axis=0)
                var_c_raw[c] = c_res.var(axis=0)

        var_shrunk = (
            1.0 - alpha
        ) * var_c_raw + alpha * self.residual_var_pooled[np.newaxis, :]
        self.residual_var_c = np.maximum(var_shrunk, variance_floor)
        self.residual_mu_c = mu_c

        n_small = int((cluster_sizes < small_thresh).sum())
        logger.info(
            f"Post-LoRA whitened stats: {n_clusters} clusters, "
            f"{n_small} small (<{small_thresh}), "
            f"alpha={alpha}"
        )

    def _compute_residual_vectors(
        self,
        data: np.ndarray,
        encoded_or_clusters: np.ndarray,
        cluster_assignments: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute residual vectors using LoRA-adapted pipeline.

        Supports two calling conventions for compatibility with
        diagnose_learned_mask.py:
          (data, encoded, cluster_assignments)  -- 3-arg (internal)
          (data, cluster_assignments)           -- 2-arg (base compat)
        """
        if cluster_assignments is None:
            # 2-arg call: encoded_or_clusters is actually cluster_assignments
            cluster_assignments = encoded_or_clusters
            s1 = self._step1
            assert s1 is not None
            encoded = s1._encode_all(data)
        else:
            encoded = encoded_or_clusters
        s1 = self._step1
        assert s1 is not None and s1.decoder is not None
        bs = self.analysis_batch_size
        all_residuals: List[np.ndarray] = []

        with torch.no_grad():
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

                # Process per-cluster (LoRA vs base)
                x_hat = torch.zeros_like(batch_raw)
                for c_id in np.unique(batch_clusters):
                    c_id = int(c_id)
                    c_idx = np.where(batch_clusters == c_id)[0]
                    c_raw = batch_raw[c_idx]

                    cm = s1.cluster_masks.get(c_id)

                    if c_id in self._lora_adapters:
                        z = self._encode_with_lora(c_raw, c_id)
                    else:
                        z = batch_enc[c_idx]

                    # Apply mask
                    if cm is not None and len(cm["union_indices"]) > 0:
                        idx = cm["union_indices"].to(self.device)
                        vals = cm["mask_values"].to(self.device)
                        z_m = torch.zeros(
                            len(c_idx), s1.sae_dim, device=self.device
                        )
                        z_m[:, idx] = z[:, idx] * vals
                    else:
                        z_m = z

                    if c_id in self._lora_adapters:
                        x_hat[c_idx] = self._decode_with_lora(z_m, c_id)
                    else:
                        x_hat[c_idx] = s1.decoder(z_m)

                diff = batch_raw - x_hat
                all_residuals.append(diff.cpu().numpy())

        return np.concatenate(all_residuals, axis=0)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_raw(
        self,
        data: np.ndarray,
        cluster_assignments: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute raw anomaly scores using LoRA-adapted pipeline."""
        s1 = self._step1
        assert (
            s1 is not None
            and s1.encoder is not None
            and s1.decoder is not None
        )
        s1.encoder.eval()
        s1.decoder.eval()

        bs = self.analysis_batch_size
        all_scores: List[np.ndarray] = []

        with torch.no_grad():
            encoded = s1._encode_all(data)

            if cluster_assignments is None:
                assert s1.kmeans is not None
                data_f32 = encoded.astype(np.float32)
                _, assignments = s1.kmeans.index.search(data_f32, 1)
                cluster_assignments = assignments.flatten()

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

                x_hat = torch.zeros_like(batch_raw)
                for c_id in np.unique(batch_clusters):
                    c_id = int(c_id)
                    c_idx = np.where(batch_clusters == c_id)[0]
                    c_raw = batch_raw[c_idx]

                    cm = s1.cluster_masks.get(c_id)

                    if c_id in self._lora_adapters:
                        z = self._encode_with_lora(c_raw, c_id)
                    else:
                        z = batch_enc[c_idx]

                    if cm is not None and len(cm["union_indices"]) > 0:
                        idx = cm["union_indices"].to(self.device)
                        vals = cm["mask_values"].to(self.device)
                        z_m = torch.zeros(
                            len(c_idx), s1.sae_dim, device=self.device
                        )
                        z_m[:, idx] = z[:, idx] * vals
                    else:
                        z_m = z

                    if c_id in self._lora_adapters:
                        x_hat[c_idx] = self._decode_with_lora(z_m, c_id)
                    else:
                        x_hat[c_idx] = s1.decoder(z_m)

                diff = batch_raw - x_hat

                # Score based on scoring_method
                if self.scoring_method == "mahalanobis":
                    scores = self._score_mahalanobis(diff, batch_clusters)
                else:
                    scores = torch.norm(diff, dim=1)
                all_scores.append(scores.cpu().numpy())

        return np.concatenate(all_scores)

    def score_samples(self, data: np.ndarray) -> np.ndarray:
        """Compute normalized anomaly scores."""
        self._check_fitted()
        raw_scores = self._score_raw(data)
        self._last_raw_scores = raw_scores.copy()

        # D5: accumulate eval cluster assignments
        if self._last_eval_clusters is not None:
            self._all_eval_cluster_assignments.append(
                self._last_eval_clusters.copy()
            )

        score_low = cast(float, self.score_low)
        score_high = cast(float, self.score_high)

        # D4: log clipping fraction before normalization
        self._score_samples_call_count += 1
        n_clipped_low = int((raw_scores < score_low).sum())
        n_clipped_high = int((raw_scores > score_high).sum())
        clip_frac = (n_clipped_low + n_clipped_high) / max(len(raw_scores), 1)
        logger.info(
            f"score_samples call #{self._score_samples_call_count}: "
            f"n={len(raw_scores)}, clipped_low={n_clipped_low}, "
            f"clipped_high={n_clipped_high}, clip_frac={clip_frac:.4f}"
        )

        scores = (raw_scores - score_low) / (score_high - score_low + 1e-10)
        return np.clip(scores, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Reassignment diagnostic
    # ------------------------------------------------------------------

    def _compute_reassignment_fraction(
        self,
        safe_data: np.ndarray,
        base_assignments: np.ndarray,
    ) -> float:
        """Fraction of samples that would change cluster under LoRA encoding."""
        s1 = self._step1
        assert s1 is not None and s1.kmeans is not None

        # For each sample, encode with its assigned cluster's LoRA
        # then check if nearest centroid changes.
        bs = self.analysis_batch_size
        n_changed = 0
        n_total = 0

        with torch.no_grad():
            for i in range(0, len(safe_data), bs):
                batch = torch.tensor(
                    safe_data[i : i + bs],
                    dtype=torch.float32,
                    device=self.device,
                )
                batch_clusters = base_assignments[i : i + bs]

                # Encode each sample with its cluster's LoRA
                z_adapted = torch.zeros(
                    len(batch), s1.sae_dim, device=self.device
                )
                for c_id in np.unique(batch_clusters):
                    c_id = int(c_id)
                    c_idx = np.where(batch_clusters == c_id)[0]
                    if c_id in self._lora_adapters:
                        z_adapted[c_idx] = self._encode_with_lora(
                            batch[c_idx], c_id
                        )
                    else:
                        z_adapted[c_idx] = s1.encoder(batch[c_idx])

                # Re-assign via FAISS
                z_np = z_adapted.cpu().numpy().astype(np.float32)
                _, new_assignments = s1.kmeans.index.search(z_np, 1)
                new_assignments = new_assignments.flatten()

                n_changed += int((new_assignments != batch_clusters).sum())
                n_total += len(batch)

        return n_changed / max(n_total, 1)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        assert self._fitted, "Must fit before saving"

        # Serialize LoRA adapters
        lora_serialized = {}
        for c, lora in self._lora_adapters.items():
            lora_serialized[c] = {k: v.cpu() for k, v in lora.items()}

        is_mode_a = self.step1_run_dir is not None

        state: Dict[str, Any] = {
            "method_name": "learned_mask_lora",
            "mode": "A" if is_mode_a else "B",
            "lora_config": {
                "lora_rank": self.lora_rank,
                "lora_lr": self.lora_lr,
                "lora_epochs": self.lora_epochs,
                "lora_batch_size": self.lora_batch_size,
                "lora_sparsity_lambda": self.lora_sparsity_lambda,
                "lora_min_cluster_size": self.lora_min_cluster_size,
                "lora_rank_schedule": (
                    dict(self.lora_rank_schedule)
                    if self.lora_rank_schedule is not None
                    else None
                ),
                "rank_schedule_small_max": self.rank_schedule_small_max,
                "rank_schedule_medium_max": self.rank_schedule_medium_max,
            },
            "cluster_ranks": {
                c: lora["A_E"].shape[0]
                for c, lora in self._lora_adapters.items()
            },
            "lora_adapters": lora_serialized,
            "skipped_clusters": sorted(self._skipped_clusters),
            "residual_mu_c": self.residual_mu_c,
            "residual_var_c": self.residual_var_c,
            "residual_mu_pooled": self.residual_mu_pooled,
            "residual_var_pooled": self.residual_var_pooled,
            "score_low": self.score_low,
            "score_high": self.score_high,
            "scoring_method": self.scoring_method,
            "mahalanobis_reg": self.mahalanobis_reg,
            "residual_means": self.residual_means,
            "residual_precisions": self.residual_precisions,
        }

        if is_mode_a:
            state["step1_run_dir"] = str(self.step1_run_dir)
        else:
            # Embed step-1 state for self-contained checkpoint
            import io

            buf = io.BytesIO()
            assert self._step1 is not None
            self._step1.save(buf)
            buf.seek(0)
            state["step1_state"] = torch.load(
                buf, map_location="cpu", weights_only=False
            )

        torch.save(state, path)
        logger.info(f"Saved learned_mask_lora model to {path}")

    def load(self, path: str) -> None:
        # Load to CPU first, then move LoRA adapters to device.
        # Step-1's load() handles its own device placement.
        state = torch.load(path, map_location="cpu", weights_only=False)

        mode = state.get("mode", "A")
        lora_cfg = state.get("lora_config", {})
        self.lora_rank = lora_cfg.get("lora_rank", 8)
        self.lora_lr = lora_cfg.get("lora_lr", 1e-3)
        self.lora_epochs = lora_cfg.get("lora_epochs", 30)
        self.lora_batch_size = lora_cfg.get("lora_batch_size", 256)
        self.lora_sparsity_lambda = lora_cfg.get("lora_sparsity_lambda", 1e-3)
        self.lora_min_cluster_size = lora_cfg.get("lora_min_cluster_size", 20)
        self.lora_rank_schedule = lora_cfg.get("lora_rank_schedule")
        self.rank_schedule_small_max = lora_cfg.get(
            "rank_schedule_small_max", 500
        )
        self.rank_schedule_medium_max = lora_cfg.get(
            "rank_schedule_medium_max", 2000
        )

        # Load step-1
        self._step1 = LearnedMaskAnodet(device=str(self.device))
        if mode == "A":
            step1_dir = state["step1_run_dir"]
            ckpt_path = Path(step1_dir) / "learned_mask_trained.pt"
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    f"Step-1 checkpoint not found at {ckpt_path}. "
                    f"The Mode A checkpoint references step1_run_dir="
                    f"'{step1_dir}'. If the run was moved, provide the "
                    f"new path."
                )
            self._step1.load(str(ckpt_path))
        else:
            # Mode B: load from embedded state
            import io
            import tempfile

            buf = io.BytesIO()
            torch.save(state["step1_state"], buf)
            buf.seek(0)
            with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
                tmp.write(buf.read())
                tmp.flush()
                self._step1.load(tmp.name)

        # Load LoRA adapters
        self._lora_adapters = {}
        for c_str, lora_dict in state["lora_adapters"].items():
            c = int(c_str)
            self._lora_adapters[c] = {
                k: v.to(self.device) for k, v in lora_dict.items()
            }

        self._skipped_clusters = set(state.get("skipped_clusters", []))

        # Whitened stats
        self.residual_mu_c = state.get("residual_mu_c")
        self.residual_var_c = state.get("residual_var_c")
        self.residual_mu_pooled = state.get("residual_mu_pooled")
        self.residual_var_pooled = state.get("residual_var_pooled")

        # Score normalization
        self.score_low = state["score_low"]
        self.score_high = state["score_high"]

        # Scoring method
        self.scoring_method = state.get("scoring_method", "l2_norm")
        self.mahalanobis_reg = state.get("mahalanobis_reg", 1e-4)
        self.residual_means = state.get("residual_means")
        self.residual_precisions = state.get("residual_precisions")

        self._fitted = True
        logger.info(
            f"Loaded learned_mask_lora: mode={mode}, "
            f"{len(self._lora_adapters)} LoRA adapters, "
            f"{len(self._skipped_clusters)} skipped"
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
        """Save LoRA-specific diagnostics."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            diag_dir = Path(save_path).parent / "diagnostics" / "lora"
            diag_dir.mkdir(parents=True, exist_ok=True)

            # Save summary JSON
            summary = dict(self._lora_diagnostic_summary)
            with open(diag_dir / "lora_diagnostic_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            # Save loss curves (D1: decomposed into total/recon/sparsity)
            if self._lora_loss_curves:
                curves_serializable = {
                    str(c): curve_dict
                    for c, curve_dict in self._lora_loss_curves.items()
                }
                with open(diag_dir / "lora_loss_curves.json", "w") as f:
                    json.dump(curves_serializable, f)

                # Plot: sample of curves + aggregate
                sample_ids = sorted(self._lora_loss_curves.keys())
                if len(sample_ids) > 10:
                    step = len(sample_ids) // 10
                    sample_ids = sample_ids[::step][:10]

                # Helper to plot one loss component
                def _plot_component(
                    component: str, filename: str, title_prefix: str
                ) -> None:
                    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                    ax = axes[0]
                    for c in sample_ids:
                        cv = self._lora_loss_curves[c].get(component, [])
                        if cv:
                            ax.plot(cv, alpha=0.6, label=f"c={c}")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    ax.set_title(f"{title_prefix} (sample)")
                    ax.legend(fontsize=7, ncol=2)

                    ax = axes[1]
                    all_cv = [
                        cv[component]
                        for cv in self._lora_loss_curves.values()
                        if cv.get(component)
                    ]
                    if all_cv:
                        max_len = max(len(v) for v in all_cv)
                        padded = np.full((len(all_cv), max_len), np.nan)
                        for i, v in enumerate(all_cv):
                            padded[i, : len(v)] = v
                        mean_c = np.nanmean(padded, axis=0)
                        p25 = np.nanpercentile(padded, 25, axis=0)
                        p75 = np.nanpercentile(padded, 75, axis=0)
                        epochs_x = np.arange(max_len)
                        ax.plot(
                            epochs_x,
                            mean_c,
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
                    ax.set_title(
                        f"Aggregate {title_prefix} "
                        f"({len(all_cv)} clusters)"
                    )
                    ax.legend()

                    fig.tight_layout()
                    fig.savefig(diag_dir / filename, dpi=150)
                    plt.close(fig)

                _plot_component(
                    "total", "lora_loss_curves.png", "LoRA total loss"
                )
                _plot_component(
                    "recon", "lora_recon_curves.png", "LoRA recon loss"
                )
                _plot_component(
                    "sparsity",
                    "lora_sparsity_curves.png",
                    "LoRA sparsity loss",
                )

            # D2: save delta norm ratios
            if self._lora_delta_ratios:
                ratios_ser = {
                    str(c): r for c, r in self._lora_delta_ratios.items()
                }
                with open(diag_dir / "lora_delta_ratios.json", "w") as f:
                    json.dump(ratios_ser, f, indent=2)

            # D3: save NaN events
            if self._nan_events:
                nan_ser = {str(c): ev for c, ev in self._nan_events.items()}
                with open(diag_dir / "lora_nan_events.json", "w") as f:
                    json.dump(nan_ser, f, indent=2)

            # Save LoRA parameter norms
            norms = {}
            for c, lora in self._lora_adapters.items():
                norms[str(c)] = {
                    k: round(float(v.norm().item()), 6)
                    for k, v in lora.items()
                }
            with open(diag_dir / "lora_parameter_norms.json", "w") as f:
                json.dump(norms, f, indent=2)

            # D6: save cluster quality JSON at run dir top level
            # (alongside existing artifacts the AUROC analysis reads)
            if self._cluster_quality:
                run_dir = Path(save_path).parent
                quality_ser = {
                    str(c): q for c, q in self._cluster_quality.items()
                }
                with open(run_dir / "cluster_quality.json", "w") as f:
                    json.dump(quality_ser, f, indent=2)
                logger.info(f"Saved cluster_quality.json to {run_dir}")

                # D6 plots
                self._plot_quality_vs_size(diag_dir, plt)
                self._plot_quality_correlation_matrix(diag_dir, plt)

            logger.info(f"LoRA diagnostics saved to {diag_dir}")

        except Exception as e:
            logger.error(f"LoRA diagnostics failed: {e}", exc_info=True)

    def _plot_quality_vs_size(self, diag_dir: Path, plt: Any) -> None:
        """Scatter: each quality metric vs log10(n_train), one subplot each."""
        metrics = [
            ("mask_density", "Mask density"),
            ("train_res_std", "Train residual std"),
            ("memorization_gap", "Memorization gap"),
            ("centroid_spread", "Centroid spread"),
            ("mask_support_ratio", "Mask support ratio"),
        ]
        n_metrics = len(metrics)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for i, (key, label) in enumerate(metrics):
            ax = axes[i]
            sizes = []
            vals = []
            for c, q in self._cluster_quality.items():
                n_train = self._train_cluster_sizes.get(c, 0)
                if n_train > 0 and key in q:
                    sizes.append(np.log10(n_train))
                    vals.append(q[key])
            if sizes:
                ax.scatter(sizes, vals, alpha=0.5, s=20, color="steelblue")
                # Spearman annotation
                if len(sizes) >= 5:
                    from scipy import stats as sp_stats

                    rho, pval = sp_stats.spearmanr(sizes, vals)
                    ax.set_title(
                        f"{label}\n(ρ={rho:.3f}, p={pval:.2e})",
                        fontsize=10,
                    )
                else:
                    ax.set_title(label, fontsize=10)
            else:
                ax.set_title(f"{label} (no data)", fontsize=10)
            ax.set_xlabel("log10(n_train)")
            ax.set_ylabel(label)

        # Hide unused subplot
        if n_metrics < len(axes):
            for j in range(n_metrics, len(axes)):
                axes[j].set_visible(False)

        fig.suptitle("D6: Quality proxies vs cluster size", fontsize=13)
        fig.tight_layout()
        fig.savefig(diag_dir / "quality_vs_size.png", dpi=150)
        plt.close(fig)
        logger.info(f"Saved {diag_dir / 'quality_vs_size.png'}")

    def _plot_quality_correlation_matrix(
        self, diag_dir: Path, plt: Any
    ) -> None:
        """Pairwise correlation heatmap across quality metrics + size."""
        metric_keys = [
            "mask_density",
            "train_res_std",
            "memorization_gap",
            "centroid_spread",
            "mask_support_ratio",
        ]
        labels = [
            "mask_density",
            "train_res_std",
            "mem_gap",
            "centroid_spread",
            "mask_ratio",
            "log_size",
        ]

        # Build matrix: rows = clusters, cols = metrics + log_size
        rows = []
        for c, q in self._cluster_quality.items():
            n_train = self._train_cluster_sizes.get(c, 0)
            if n_train <= 0:
                continue
            row = [q.get(k, np.nan) for k in metric_keys]
            row.append(np.log10(n_train))
            rows.append(row)

        if len(rows) < 5:
            logger.info("Too few clusters for correlation matrix")
            return

        data = np.array(rows)
        n_cols = data.shape[1]

        # Spearman correlation matrix
        from scipy import stats as sp_stats

        corr = np.ones((n_cols, n_cols))
        for i in range(n_cols):
            for j in range(i + 1, n_cols):
                mask = np.isfinite(data[:, i]) & np.isfinite(data[:, j])
                if mask.sum() >= 5:
                    rho, _ = sp_stats.spearmanr(data[mask, i], data[mask, j])
                    corr[i, j] = rho
                    corr[j, i] = rho
                else:
                    corr[i, j] = np.nan
                    corr[j, i] = np.nan

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n_cols))
        ax.set_yticks(range(n_cols))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)

        # Annotate cells
        for i in range(n_cols):
            for j in range(n_cols):
                val = corr[i, j]
                if np.isfinite(val):
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if abs(val) > 0.6 else "black",
                    )

        fig.colorbar(im, ax=ax, label="Spearman ρ")
        ax.set_title("D6: Quality proxy correlation matrix")
        fig.tight_layout()
        fig.savefig(diag_dir / "quality_correlation_matrix.png", dpi=150)
        plt.close(fig)
        logger.info(f"Saved {diag_dir / 'quality_correlation_matrix.png'}")
