"""Pure utility functions extracted from CVDDKMeansMaskedAnodet.

All functions in this module are pure: they take numpy/torch arrays
and config values as arguments and return results without side effects.
Where a function needs an SAE or autoencoder model, it takes it as an
explicit ``nn.Module`` argument.

This module has no imports from the project — only numpy, torch, faiss,
and standard library.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return-type dataclasses (frozen for safety)
# ---------------------------------------------------------------------------


try:
    import faiss
except ImportError:
    faiss = None  # type: ignore


@dataclass(frozen=True)
class ClusterResult:
    """Return type for :func:`cluster_kmeans`."""

    labels: np.ndarray
    centroids: np.ndarray
    kmeans: Any  # faiss.Kmeans
    binary_centroids: Optional[np.ndarray]


@dataclass(frozen=True)
class SplitResult:
    """Return type for :func:`split_large_union_clusters`."""

    labels: np.ndarray
    n_clusters: int
    centroids: np.ndarray


@dataclass(frozen=True)
class ResidualMahalanobisState:
    """Per-cluster residual Mahalanobis state."""

    means: Dict[int, torch.Tensor]
    precisions: Dict[int, torch.Tensor]


# ---------------------------------------------------------------------------
# Group 1  –  Clustering utilities  (Phase 2a)
# ---------------------------------------------------------------------------


def _lloyd_l1_single(
    data_t: torch.Tensor,
    centroids: torch.Tensor,
    batch_size: int,
    n_iter: int,
    tol: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Single Lloyd's pass for L1 KMeans.

    Returns ``(labels, centroids, inertia)``.
    """
    n = data_t.shape[0]
    n_clusters = centroids.shape[0]
    labels = torch.empty(n, dtype=torch.long, device=data_t.device)
    inertia = float("inf")

    for it in range(n_iter):
        min_dists = torch.empty(n, dtype=torch.float32, device=data_t.device)
        for i in range(0, n, batch_size):
            batch = data_t[i : i + batch_size]
            dists = torch.cdist(batch, centroids, p=1)
            mn, lab = dists.min(dim=1)
            labels[i : i + batch_size] = lab
            min_dists[i : i + batch_size] = mn
        inertia = min_dists.sum().item()

        new_centroids = centroids.clone()
        for c in range(n_clusters):
            mask = labels == c
            if mask.any():
                new_centroids[c] = data_t[mask].median(dim=0).values

        shift = (new_centroids - centroids).abs().sum().item()
        denom = max(centroids.abs().sum().item(), 1e-8)
        centroids = new_centroids
        if shift / denom < tol:
            break

    return labels, centroids, inertia


def _kmedians_l1(
    data: np.ndarray,
    n_clusters: int,
    device: torch.device,
    n_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 42,
    n_restarts: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's algorithm for L1 KMeans (k-medians) with restarts.

    Optimizes ``sum_i min_c ||x_i - mu_c||_1``. The L1-optimal centroid
    for a set of points is the per-feature median; assignment uses
    Manhattan distance. Lloyd's with median centroids has many local
    minima, so we run *n_restarts* and keep the lowest-inertia result.

    Restart 0 warm-starts from FAISS L2 KMeans (niter=20) — usually a
    strong baseline. Remaining restarts use random-sample
    initialization with distinct seeds.

    GPU is used when *device* is CUDA. Distance computation is chunked
    over samples to bound memory at ``[batch, n_clusters, dim]``.

    Args:
        data: Input data ``[n_samples, dim]``.
        n_clusters: Number of clusters.
        device: Torch device.
        n_iter: Maximum Lloyd iterations per restart.
        tol: Convergence threshold on relative L1 centroid movement.
        seed: RNG seed (each restart uses ``seed + i``).
        n_restarts: Number of independent restarts.

    Returns:
        ``(labels, centroids)`` of the lowest-inertia restart.
    """
    n, d = data.shape
    data_f32 = data.astype(np.float32)
    data_t = torch.from_numpy(data_f32).to(device)

    bytes_per_elt = 4
    target_bytes = 512 * 1024 * 1024
    batch_size = max(16, target_bytes // (n_clusters * d * bytes_per_elt))

    best_labels: Optional[torch.Tensor] = None
    best_centroids: Optional[torch.Tensor] = None
    best_inertia = float("inf")

    for r in range(n_restarts):
        if r == 0 and faiss is not None:
            use_gpu = faiss.get_num_gpus() > 0 and device.type == "cuda"
            init_km = faiss.Kmeans(
                d=d,
                k=n_clusters,
                niter=20,
                verbose=False,
                gpu=use_gpu,
                seed=seed,
            )
            init_km.train(data_f32)
            init = init_km.centroids.copy()
        else:
            rng = np.random.default_rng(seed + r)
            idx = rng.choice(n, size=n_clusters, replace=False)
            init = data_f32[idx]

        centroids = torch.from_numpy(init.astype(np.float32)).to(device)
        labels, centroids, inertia = _lloyd_l1_single(
            data_t=data_t,
            centroids=centroids,
            batch_size=batch_size,
            n_iter=n_iter,
            tol=tol,
        )
        logger.info(f"  k-medians restart {r}: inertia={inertia:.3e}")
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.clone()
            best_centroids = centroids.clone()

    assert best_labels is not None and best_centroids is not None
    logger.info(
        f"  k-medians best inertia={best_inertia:.3e} "
        f"(n_restarts={n_restarts})"
    )
    return best_labels.cpu().numpy(), best_centroids.cpu().numpy()


def cluster_kmeans(
    data: np.ndarray,
    n_clusters: int,
    cluster_method: str,
    l0_filter: float,
    device: torch.device,
    seed: Optional[int] = None,
    init: str = "faiss_random",
    cluster_on_cpu: bool = False,
) -> ClusterResult:
    """Perform KMeans clustering on input data using FAISS.

    When *cluster_method* = ``"l0"``, binarizes data (``x > 0``)
    before clustering. FAISS KMeans with Euclidean distance on binary
    vectors is equivalent to Hamming distance. After clustering, also
    computes magnitude centroids per cluster for L1/L2 scoring.

    When *cluster_method* = ``"cosine_binary"``, binarizes data then
    L2-normalizes the binary vectors before KMeans. This makes
    Euclidean KMeans approximate cosine-distance clustering, which
    for binary vectors equals the Ochiai coefficient
    (``|A∩B| / sqrt(|A|·|B|)``), closely related to Jaccard
    similarity. This prevents samples with different L0 norms from
    merging into clusters with inflated union-of-support.

    When *cluster_method* = ``"l1"``, runs Lloyd's algorithm with
    Manhattan distance and per-feature median centroids (k-medians)
    on raw (non-binarized) data via :func:`_kmedians_l1`. Test-time
    cluster assignment falls back to L2 against the median centroids
    (same convention as the binary modes), which is well-defined and
    FAISS-supported.

    Args:
        data: Input data ``[n_samples, dim]``.
        n_clusters: Number of clusters.
        cluster_method: ``"default"``, ``"l0"``, ``"cosine_binary"``,
            or ``"l1"``.
        l0_filter: If > 0, filter samples with L0 > T*median before
            clustering (only used with ``"l0"``/``"cosine_binary"``).
        device: Torch device (used to decide GPU clustering).
        seed: RNG seed for the default-path ``faiss.Kmeans`` random init.
            ``None`` (default) reproduces the exact legacy call (no ``seed=``
            passed). Set an int to make the basin controllable/reproducible.
        init: Default-path init strategy. ``"faiss_random"`` (legacy: FAISS
            samples random points), ``"kmeans++"`` (sklearn k-means++ init,
            ``n_init=1``, deterministic given ``seed``), or ``"sklearn_ninit"``
            (sklearn k-means++ with ``n_init=10``, keeps lowest inertia).
        cluster_on_cpu: Force the default-path K-means onto CPU (FAISS GPU
            k-means is non-deterministic even when seeded). Encoding stays on
            GPU; only the clustering moves. Default ``False`` (legacy).

    Returns:
        :class:`ClusterResult` with labels, centroids, kmeans index,
        and optional binary centroids.
    """
    if faiss is None:
        raise ImportError(
            "faiss is required for clustering but is not installed. "
            "Install it with: pip install faiss-gpu or faiss-cpu"
        )

    logger.info(
        f"Clustering {len(data)} samples into {n_clusters} "
        f"clusters (method={cluster_method})..."
    )

    use_gpu = faiss.get_num_gpus() > 0 and device.type == "cuda"
    if use_gpu:
        logger.info("Using GPU for KMeans clustering")
    else:
        logger.info("Using CPU for KMeans clustering")

    binary_centroids: Optional[np.ndarray] = None

    if cluster_method in ("l0", "cosine_binary"):
        data_binary = (data > 0).astype(np.float32)
        per_sample_l0 = data_binary.sum(axis=1)
        logger.info(
            f"{cluster_method} mode: binarized data, "
            f"mean L0={per_sample_l0.mean():.1f}, "
            f"std={per_sample_l0.std():.1f}"
        )

        if l0_filter > 0:
            median_l0 = np.median(per_sample_l0)
            l0_cutoff = l0_filter * median_l0
            keep_mask = per_sample_l0 <= l0_cutoff
            n_filtered = (~keep_mask).sum()
            logger.info(
                f"L0 filter: keeping {keep_mask.sum()} samples "
                f"(filtered {n_filtered} with L0 > {l0_cutoff:.0f}, "
                f"median={median_l0:.0f}, threshold={l0_filter}x)"
            )
            data_binary_filtered = data_binary[keep_mask]
            filtered_indices = np.where(~keep_mask)[0]
        else:
            data_binary_filtered = data_binary
            keep_mask = np.ones(len(data), dtype=bool)
            filtered_indices = np.array([], dtype=int)

        # L2-normalize binary vectors for cosine-distance KMeans
        if cluster_method == "cosine_binary":
            norms = np.linalg.norm(data_binary_filtered, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            cluster_input = (data_binary_filtered / norms).astype(np.float32)
            logger.info(
                "cosine_binary: L2-normalized binary vectors "
                "(KMeans ≈ cosine distance)"
            )
        else:
            cluster_input = data_binary_filtered

        # Mirror the default path below: seed=None reproduces the exact
        # historical call (no seed= argument), which FAISS answers with its
        # own constant cp.seed=1234 -- not "unseeded". FAISS GPU k-means is
        # non-deterministic even when seeded, so cluster_on_cpu forces CPU.
        # Before this, `seed` was structurally unreachable from this branch,
        # so LearnedMask/LearnedMask-LoRA had no way to pin their clustering.
        effective_gpu = use_gpu and not cluster_on_cpu
        faiss_kwargs = dict(
            d=cluster_input.shape[1],
            k=n_clusters,
            niter=300,
            verbose=False,
            gpu=effective_gpu,
        )
        if seed is not None:
            faiss_kwargs["seed"] = int(seed)
        kmeans = faiss.Kmeans(**faiss_kwargs)
        kmeans.train(cluster_input)

        _, kept_labels = kmeans.index.search(cluster_input, 1)
        kept_labels = kept_labels.flatten()

        binary_centroids = kmeans.centroids.copy()

        magnitude_centroids = np.zeros(
            (n_clusters, data.shape[1]), dtype=np.float32
        )
        kept_data = data[keep_mask]
        for c in range(n_clusters):
            mask = kept_labels == c
            if mask.any():
                magnitude_centroids[c] = kept_data[mask].mean(axis=0)

        mag_centroids_f32 = magnitude_centroids.astype(np.float32).copy()
        kmeans = faiss.Kmeans(
            d=mag_centroids_f32.shape[1],
            k=n_clusters,
            niter=1,
            verbose=False,
        )
        kmeans.train(mag_centroids_f32)
        kmeans.centroids = mag_centroids_f32

        cluster_labels = np.empty(len(data), dtype=np.int64)
        cluster_labels[keep_mask] = kept_labels
        if len(filtered_indices) > 0:
            filtered_data = data[filtered_indices].astype(np.float32)
            _, filtered_labels = kmeans.index.search(filtered_data, 1)
            cluster_labels[filtered_indices] = filtered_labels.flatten()
            logger.info(
                f"Assigned {len(filtered_indices)} filtered samples "
                f"to nearest magnitude centroid"
            )

        centroids = magnitude_centroids

    elif cluster_method == "l1":
        cluster_labels, centroids = _kmedians_l1(
            data=data,
            n_clusters=n_clusters,
            device=device,
            n_iter=100,
            seed=42,
        )

        centroids_f32 = centroids.astype(np.float32).copy()
        kmeans = faiss.Kmeans(
            d=centroids_f32.shape[1],
            k=n_clusters,
            niter=1,
            verbose=False,
        )
        kmeans.train(centroids_f32)
        kmeans.centroids = centroids_f32

    else:
        data_f32 = data.astype(np.float32)

        if init in ("kmeans++", "sklearn_ninit"):
            # Deterministic basin-robust init via sklearn k-means++ (CPU only;
            # cluster_on_cpu is irrelevant here). seed=None maps to a fixed
            # random_state=42 so this path is reproducible by default, unlike
            # the legacy faiss_random path where seed=None means "unseeded".
            from sklearn.cluster import KMeans as _SKKMeans

            n_init = 10 if init == "sklearn_ninit" else 1
            logger.info(
                f"Default-path clustering via sklearn k-means++ "
                f"(init={init}, n_init={n_init}, seed={seed})"
            )
            skm = _SKKMeans(
                n_clusters=n_clusters,
                init="k-means++",
                n_init=n_init,
                max_iter=300,
                random_state=seed if seed is not None else 42,
            )
            cluster_labels = skm.fit_predict(data_f32).astype(np.int64)
            centroids = skm.cluster_centers_.astype(np.float32)
            # Wrap the sklearn centroids in a FAISS index for downstream
            # nearest-centroid assignment (same shell pattern as the l1 path).
            # train() with niter=1 on exactly k points already yields an index
            # whose centroids == centroids_f32, so no manual .centroids set.
            centroids_f32 = centroids.copy()
            kmeans = faiss.Kmeans(
                d=centroids_f32.shape[1], k=n_clusters, niter=1, verbose=False
            )
            kmeans.train(centroids_f32)
        else:
            # Legacy FAISS random-init path. seed=None reproduces the exact
            # historical call (no seed= argument passed). FAISS GPU k-means is
            # non-deterministic even when seeded, so cluster_on_cpu forces CPU.
            effective_gpu = use_gpu and not cluster_on_cpu
            faiss_kwargs = dict(
                d=data.shape[1],
                k=n_clusters,
                niter=300,
                verbose=False,
                gpu=effective_gpu,
            )
            if seed is not None:
                faiss_kwargs["seed"] = int(seed)
            kmeans = faiss.Kmeans(**faiss_kwargs)
            kmeans.train(data_f32)

            _, cluster_labels = kmeans.index.search(data_f32, 1)
            cluster_labels = cluster_labels.flatten()

            centroids = kmeans.centroids.copy()

    return ClusterResult(
        labels=cluster_labels,
        centroids=centroids,
        kmeans=kmeans,
        binary_centroids=binary_centroids,
    )


def split_large_union_clusters(
    data: np.ndarray,
    cluster_labels: np.ndarray,
    n_clusters: int,
    threshold: float,
    max_rounds: int = 10,
    cluster_method: str = "l0",
    seed: Optional[int] = None,
    cluster_on_cpu: bool = False,
) -> SplitResult:
    """Split clusters whose union_support exceeds threshold x median.

    Iteratively bisects oversized clusters using binary KMeans until
    all clusters have union_support below the cutoff or *max_rounds*
    is reached.

    Args:
        data: Encoded data ``[n_samples, dim]``.
        cluster_labels: Current cluster assignments.
        n_clusters: Current number of clusters.
        threshold: Multiplier on median union_support for the cutoff.
        max_rounds: Max splitting iterations.

    Returns:
        :class:`SplitResult` with new labels, cluster count, and
        recomputed magnitude centroids.
    """
    if faiss is None:
        raise ImportError("faiss is required for cluster splitting")

    labels = cluster_labels.copy()
    next_cid = labels.max() + 1

    for round_i in range(max_rounds):
        active_cids = np.unique(labels)
        union_sizes = {}
        for c in active_cids:
            cluster_data = data[labels == c]
            union_sizes[c] = int(np.sum(np.any(cluster_data > 0, axis=0)))

        non_zero = [u for u in union_sizes.values() if u > 0]
        if not non_zero:
            break
        median_union = float(np.median(non_zero))
        cutoff = threshold * median_union

        to_split = [
            c
            for c, u in union_sizes.items()
            if u > cutoff and (labels == c).sum() >= 4
        ]
        if not to_split:
            logger.info(
                f"Split round {round_i}: no clusters above cutoff "
                f"({cutoff:.0f}, median={median_union:.0f}). Done."
            )
            break

        logger.info(
            f"Split round {round_i}: splitting {len(to_split)} "
            f"clusters (cutoff={cutoff:.0f}, "
            f"median={median_union:.0f})"
        )

        any_improved = False
        for c in to_split:
            mask = labels == c
            n_c = mask.sum()
            sub_data = (data[mask] > 0).astype(np.float32)

            if cluster_method == "cosine_binary":
                norms = np.linalg.norm(sub_data, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-8)
                sub_input = (sub_data / norms).astype(np.float32)
            else:
                sub_input = sub_data

            # Second lottery: this bisection was also unseeded. With k=2 the
            # FAISS training subsample kicks in above 512 points, and the
            # clusters reaching here are far larger, so an unseeded draw here
            # moves the final partition on its own. seed=None reproduces the
            # exact historical call.
            sub_kwargs = dict(d=sub_input.shape[1], k=2, niter=50)
            if cluster_on_cpu:
                sub_kwargs["gpu"] = False
            if seed is not None:
                # Offset by the parent cluster id so sibling bisections in the
                # same round do not all share one seed.
                sub_kwargs["seed"] = int(seed) + int(c)
            km = faiss.Kmeans(**sub_kwargs)
            km.train(sub_input)
            _, sub_labels = km.index.search(sub_input, 1)
            sub_labels = sub_labels.flatten()

            child_unions = []
            for child in (0, 1):
                child_data = data[mask][sub_labels == child]
                if len(child_data) == 0:
                    child_unions.append(float("inf"))
                else:
                    child_unions.append(
                        int(np.sum(np.any(child_data > 0, axis=0)))
                    )

            if min(child_unions) >= union_sizes[c]:
                logger.info(
                    f"  Cluster {c}: split not useful "
                    f"(parent={union_sizes[c]}, "
                    f"children={child_unions}). Keeping."
                )
                continue

            any_improved = True
            indices = np.where(mask)[0]
            labels[indices[sub_labels == 1]] = next_cid
            logger.info(
                f"  Cluster {c} (union={union_sizes[c]}, n={n_c}) "
                f"-> {c} (union={child_unions[0]}, "
                f"n={(sub_labels == 0).sum()}) + "
                f"{next_cid} (union={child_unions[1]}, "
                f"n={(sub_labels == 1).sum()})"
            )
            next_cid += 1

        if not any_improved:
            logger.info(f"Split round {round_i}: no improvement. Done.")
            break

    # Compact labels to 0..n-1
    unique_labels = np.unique(labels)
    new_n = len(unique_labels)
    if new_n != n_clusters:
        remap = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([remap[lab] for lab in labels])
        logger.info(f"Cluster count changed: {n_clusters} -> {new_n}")

    # Recompute magnitude centroids
    centroids = np.zeros((new_n, data.shape[1]), dtype=np.float32)
    for c in range(new_n):
        mask = labels == c
        if mask.any():
            centroids[c] = data[mask].mean(axis=0)

    return SplitResult(
        labels=labels,
        n_clusters=new_n,
        centroids=centroids,
    )


@dataclass(frozen=True)
class MergeResult:
    """Return type for :func:`merge_small_clusters`."""

    labels: np.ndarray
    n_clusters: int
    centroids: np.ndarray


def merge_small_clusters(
    data: np.ndarray,
    cluster_labels: np.ndarray,
    n_clusters: int,
    min_cluster_size: int = 100,
    max_union_size: int = 500,
) -> MergeResult:
    """Merge clusters with fewer than *min_cluster_size* samples.

    Iteratively merges undersized clusters into the nearest centroid
    neighbour whose post-merge union-of-support stays below
    *max_union_size*.  Merging proceeds smallest-first so that the
    smallest clusters are absorbed first.

    Args:
        data: Encoded data ``[n_samples, dim]``.
        cluster_labels: Current cluster assignments.
        n_clusters: Current number of clusters.
        min_cluster_size: Minimum number of samples per cluster.
        max_union_size: Maximum allowed union-of-support after merge.

    Returns:
        :class:`MergeResult` with new labels, cluster count, and
        recomputed magnitude centroids.
    """
    labels = cluster_labels.copy()

    for round_i in range(50):  # safety bound
        unique_cids = np.unique(labels)
        sizes = {int(c): int((labels == c).sum()) for c in unique_cids}

        small = [c for c, s in sizes.items() if 0 < s < min_cluster_size]
        if not small:
            logger.info(
                f"Merge round {round_i}: no clusters below "
                f"{min_cluster_size}. Done."
            )
            break

        # Process smallest first
        small.sort(key=lambda c: sizes[c])

        merged_any = False
        for c in small:
            c_mask = labels == c
            if c_mask.sum() == 0:
                continue  # already merged in this round

            c_data = data[c_mask]
            c_mean = c_data.mean(axis=0)

            # Rank candidates by centroid distance (exclude self and
            # other small clusters that we haven't merged yet)
            candidates = [
                t
                for t in np.unique(labels)
                if int(t) != c and (labels == t).sum() > 0
            ]
            if not candidates:
                break

            # Sort by distance to candidate centroid
            candidate_means = np.array(
                [data[labels == t].mean(axis=0) for t in candidates]
            )
            dists = np.linalg.norm(candidate_means - c_mean, axis=1)
            order = np.argsort(dists)

            found_target = False
            for idx in order:
                t = int(candidates[idx])
                # Check union-of-support after hypothetical merge
                merged_data = np.vstack([data[labels == t], c_data])
                merged_union = int(np.sum(np.any(merged_data > 0, axis=0)))
                if merged_union > max_union_size:
                    continue  # union too big, try next

                # Accept merge
                labels[c_mask] = t
                found_target = True
                merged_any = True
                logger.info(
                    f"  Merged cluster {c} (n={sizes[c]}) "
                    f"into {t} (n={sizes[t]}), "
                    f"merged_union={merged_union}"
                )
                break

            if not found_target:
                logger.info(
                    f"  Cluster {c} (n={sizes[c]}): no valid merge "
                    f"target (max_union={max_union_size}). Keeping."
                )

        if not merged_any:
            logger.info(f"Merge round {round_i}: no progress. Done.")
            break

    # Compact labels to 0..n-1
    unique_labels = np.unique(labels)
    new_n = len(unique_labels)
    if new_n != n_clusters:
        remap = {int(old): new for new, old in enumerate(unique_labels)}
        labels = np.array([remap[int(lab)] for lab in labels])
        logger.info(f"After merge: {n_clusters} -> {new_n} clusters")

    # Recompute magnitude centroids
    centroids = np.zeros((new_n, data.shape[1]), dtype=np.float32)
    for c in range(new_n):
        mask = labels == c
        if mask.any():
            centroids[c] = data[mask].mean(axis=0)

    return MergeResult(
        labels=labels,
        n_clusters=new_n,
        centroids=centroids,
    )


# ---------------------------------------------------------------------------
# Group 2 / 3  –  Firing patterns, masks, covariances  (Phase 2b)
# ---------------------------------------------------------------------------


def analyze_firing_patterns(
    data: np.ndarray,
    n_clusters: int,
    cluster_labels: np.ndarray,
    firing_threshold: float,
    sae: Optional[nn.Module] = None,
    device: torch.device = torch.device("cpu"),
    batch_size: int = 1024,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Analyze firing neuron patterns per cluster.

    Args:
        data: Input data ``[n_samples, dim]``. If *sae* is None, data
            is assumed to be pre-encoded.
        n_clusters: Number of clusters.
        cluster_labels: ``[n_samples]`` cluster assignments.
        firing_threshold: Threshold for considering a neuron as firing.
        sae: Optional encoder model. When provided, data is encoded
            through it; when None, data is used as-is.
        device: Torch device.
        batch_size: Batch size for encoding.

    Returns:
        Dict mapping cluster_id to statistics dict.
    """
    is_encoded = sae is None
    if is_encoded:
        logger.info(
            "Analyzing firing patterns per cluster "
            "(data already encoded)..."
        )
    else:
        logger.info("Analyzing firing patterns per cluster...")

    cluster_stats: Dict[int, Dict[str, np.ndarray]] = {}

    with torch.no_grad():
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_data = data[cluster_mask]

            if len(cluster_data) == 0:
                logger.warning(f"Cluster {cluster_id} is empty!")
                continue

            print(f"Cluster {cluster_id}: " f"{len(cluster_data)} samples")

            firing_patterns_list = []
            bs = min(batch_size, len(cluster_data))

            for i in range(0, len(cluster_data), bs):
                batch = cluster_data[i : i + bs]
                batch_tensor = torch.tensor(
                    batch, dtype=torch.float32, device=device
                )

                if is_encoded:
                    z = batch_tensor
                else:
                    assert sae is not None
                    z = sae(batch_tensor)

                firing = (z > firing_threshold).cpu().numpy()
                firing_patterns_list.append(firing)

            all_firing_patterns = np.vstack(firing_patterns_list)

            n_firing_per_sample = all_firing_patterns.sum(axis=1)
            mean_firing = n_firing_per_sample.mean()
            std_firing = n_firing_per_sample.std()

            intersection = all_firing_patterns.all(axis=0)
            union = all_firing_patterns.any(axis=0)
            frequency = all_firing_patterns.sum(axis=0)
            firing_neuron_indices = np.where(union)[0].astype(np.int32)

            cluster_stats[cluster_id] = {
                "mean_firing": mean_firing,
                "std_firing": std_firing,
                "intersection": intersection,
                "union": union,
                "frequency": frequency,
                "n_samples": len(cluster_data),
                "firing_neuron_indices": firing_neuron_indices,
            }

    return cluster_stats


def create_cluster_masks(
    cluster_stats: Dict[int, Dict[str, np.ndarray]],
    n_clusters: int,
    k_mask: int,
    use_masking: bool,
    mask_variant: str,
    mask_coverage: float,
    device: torch.device,
    global_mask: bool = False,
) -> Dict[int, torch.Tensor]:
    """Create binary masks of top-k firing neurons for each cluster.

    Creates masks for ALL clusters from 0 to *n_clusters*-1. For empty
    clusters, uses the union of all masks from non-empty clusters as a
    fallback.

    Args:
        cluster_stats: Per-cluster firing statistics from
            :func:`analyze_firing_patterns`.
        n_clusters: Total number of clusters.
        k_mask: Number of top neurons for ``"top_k"`` variant.
        use_masking: If False, creates all-ones masks.
        mask_variant: ``"top_k"`` or ``"frequency_coverage"``.
        mask_coverage: Coverage threshold for ``"frequency_coverage"``.
        device: Torch device.
        global_mask: If True, pool firing frequencies across all
            clusters into one global mask and replicate it.

    Returns:
        Dict mapping cluster_id to binary mask tensor.
    """
    if not use_masking:
        logger.info("Masking disabled - using full embedding dimensions")
        if not cluster_stats:
            raise ValueError(
                "No cluster stats available - cannot create masks"
            )
        first_cluster_id = next(iter(cluster_stats))
        embedding_dim = len(cluster_stats[first_cluster_id]["frequency"])
        full_mask = torch.ones(embedding_dim, device=device)
        return {i: full_mask for i in range(n_clusters)}

    if not cluster_stats:
        raise ValueError("No cluster stats available - cannot create masks")

    first_cluster_id = next(iter(cluster_stats))
    embedding_dim = len(cluster_stats[first_cluster_id]["frequency"])

    # --- Global mask: pool all per-cluster frequencies, create one mask ---
    if global_mask:
        total_frequency = np.zeros(embedding_dim, dtype=np.float64)
        total_samples = 0
        for stats in cluster_stats.values():
            total_frequency += stats["frequency"]
            total_samples += stats["n_samples"]

        global_avg_rate = total_frequency / max(total_samples, 1)

        selected_indices = np.argsort(global_avg_rate)[-k_mask:]
        n_active = k_mask

        mask_np = np.zeros(embedding_dim, dtype=np.float32)
        mask_np[selected_indices] = 1.0
        shared_mask = torch.tensor(mask_np, device=device)

        logger.info(
            f"Created GLOBAL top-{n_active} mask from "
            f"{total_samples} pooled samples "
            f"(top rate={global_avg_rate[selected_indices[-1]]:.3f}, "
            f"kth rate={global_avg_rate[selected_indices[0]]:.3f})"
        )
        return {i: shared_mask for i in range(n_clusters)}

    # --- Per-cluster masks (original behavior) ---
    if mask_variant == "frequency_coverage":
        logger.info(
            f"Creating per-cluster frequency-coverage masks "
            f"(coverage={mask_coverage})..."
        )
    else:
        logger.info(
            f"Creating per-cluster top-{k_mask} masks "
            f"from firing patterns..."
        )

    cluster_masks: Dict[int, torch.Tensor] = {}

    for cluster_id, stats in cluster_stats.items():
        frequency = stats["frequency"]
        n_samples = stats["n_samples"]
        avg_firing_rate = frequency / n_samples

        if mask_variant == "frequency_coverage":
            sorted_idx = np.argsort(avg_firing_rate)[::-1]
            cumsum = np.cumsum(avg_firing_rate[sorted_idx])
            total = avg_firing_rate.sum()
            if total > 0:
                n_keep = int(
                    np.searchsorted(cumsum, total * mask_coverage) + 1
                )
                selected_indices = sorted_idx[:n_keep]
            else:
                selected_indices = sorted_idx[:k_mask]

            mask = np.zeros(embedding_dim, dtype=np.float32)
            mask[selected_indices] = 1.0

            cluster_masks[cluster_id] = torch.tensor(mask, device=device)
            print(
                f"Cluster {cluster_id}: freq-coverage mask with "
                f"{mask.sum():.0f} active neurons "
                f"(coverage={mask_coverage})"
            )
        else:
            top_k_indices = np.argsort(avg_firing_rate)[-k_mask:]

            mask = np.zeros(embedding_dim, dtype=np.float32)
            mask[top_k_indices] = 1.0

            cluster_masks[cluster_id] = torch.tensor(mask, device=device)

            print(
                f"Cluster {cluster_id}: mask with "
                f"{mask.sum():.0f} active neurons, "
                f"top rate="
                f"{avg_firing_rate[top_k_indices[-1]]:.3f}, "
                f"kth rate="
                f"{avg_firing_rate[top_k_indices[0]]:.3f}"
            )

    # Fallback masks for empty clusters
    if len(cluster_masks) < n_clusters:
        logger.info(
            f"Creating masks for "
            f"{n_clusters - len(cluster_masks)} empty clusters..."
        )

        # *cluster_masks* is non-empty here: the loop above writes an entry
        # for every cluster in *cluster_stats* on both of its arms, and
        # analyze_firing_patterns emits stats for every cluster that has at
        # least one member, so the dict is empty only if no sample was
        # assigned to any cluster at all.
        union_mask = torch.zeros(embedding_dim, device=device)
        for mask in cluster_masks.values():
            union_mask = torch.logical_or(union_mask, mask.bool()).float()

        union_indices = (
            torch.nonzero(union_mask, as_tuple=False).squeeze(-1).cpu().numpy()
        )
        if len(union_indices) > k_mask:
            np.random.seed(42)
            selected_indices = np.random.choice(
                union_indices, k_mask, replace=False
            )
        else:
            selected_indices = union_indices

        fallback_mask = np.zeros(embedding_dim, dtype=np.float32)
        fallback_mask[selected_indices] = 1.0
        fallback_mask_tensor = torch.tensor(fallback_mask, device=device)

        for cluster_id in range(n_clusters):
            if cluster_id not in cluster_masks:
                cluster_masks[cluster_id] = fallback_mask_tensor
                print(
                    f"  Cluster {cluster_id}: Using fallback "
                    f"mask with "
                    f"{fallback_mask.sum():.0f} active neurons"
                )

    return cluster_masks


@dataclass(frozen=True)
class CovarianceResult:
    """Return type for :func:`compute_cluster_covariances`."""

    precisions: Dict[int, torch.Tensor]
    means_sub: Dict[int, torch.Tensor]
    mask_indices: Dict[int, torch.Tensor]


def compute_cluster_covariances(
    data: np.ndarray,
    cluster_labels: np.ndarray,
    n_clusters: int,
    masks_tensor: torch.Tensor,
    cluster_union_indices: Optional[Dict[int, np.ndarray]],
    use_masking: bool,
    mahalanobis_reg: float,
    cluster_in_encoded_space: bool,
    sae: Optional[nn.Module] = None,
    device: torch.device = torch.device("cpu"),
) -> CovarianceResult:
    """Compute per-cluster covariance and precision matrices.

    For each cluster, extracts the active dimensions, computes the
    sample covariance in that subspace, regularizes, and inverts.

    Args:
        data: Input data ``[n_samples, dim]``.
        cluster_labels: ``[n_samples]`` cluster assignments.
        n_clusters: Number of clusters.
        masks_tensor: ``[n_clusters, embedding_dim]`` binary masks.
        cluster_union_indices: Per-cluster union support indices
            (used when *use_masking* is False).
        use_masking: Whether masking is enabled.
        mahalanobis_reg: Covariance regularization.
        cluster_in_encoded_space: If True, data is pre-encoded.
        sae: Encoder model (used when data is not pre-encoded).
        device: Torch device.

    Returns:
        :class:`CovarianceResult` with precisions, means, and
        active-dimension indices per cluster.
    """
    precisions: Dict[int, torch.Tensor] = {}
    means_sub: Dict[int, torch.Tensor] = {}
    mask_indices: Dict[int, torch.Tensor] = {}

    with torch.no_grad():
        for c in range(n_clusters):
            if (
                not use_masking
                and cluster_union_indices is not None
                and c in cluster_union_indices
            ):
                active_idx = torch.tensor(
                    cluster_union_indices[c],
                    dtype=torch.long,
                    device=device,
                )
            else:
                mask_c = masks_tensor[c]
                active_idx = torch.where(mask_c > 0.5)[0]
            mask_indices[c] = active_idx

            if len(active_idx) == 0:
                logger.warning(
                    f"  Cluster {c}: empty mask, " "skipping covariance"
                )
                continue

            cluster_sample_mask = cluster_labels == c
            n_c = int(cluster_sample_mask.sum())

            if n_c < 2:
                logger.warning(
                    f"  Cluster {c}: only {n_c} samples, "
                    "using identity precision"
                )
                k = len(active_idx)
                means_sub[c] = torch.zeros(k, device=device)
                precisions[c] = torch.eye(k, device=device)
                continue

            cluster_data = data[cluster_sample_mask]
            z_sub_list = []
            batch_size = min(1024, n_c)

            for i in range(0, n_c, batch_size):
                batch = cluster_data[i : i + batch_size]
                batch_tensor = torch.tensor(
                    batch,
                    dtype=torch.float32,
                    device=device,
                )
                # The caller passes ``sae=None`` exactly when
                # *cluster_in_encoded_space* is True, so the two conditions
                # are complementary and *sae* is non-None in the else arm.
                if cluster_in_encoded_space:
                    z = batch_tensor
                else:
                    assert sae is not None
                    z = sae(batch_tensor)
                z_sub = z[:, active_idx]
                z_sub_list.append(z_sub)

            z_sub_all = torch.cat(z_sub_list, dim=0)
            k_active = z_sub_all.shape[1]

            mean_c = z_sub_all.mean(dim=0)
            means_sub[c] = mean_c

            z_cpu = z_sub_all.cpu()
            mean_cpu = mean_c.cpu()
            centered = (z_cpu - mean_cpu).to(torch.float64)
            cov = (centered.T @ centered) / max(n_c - 1, 1)
            cov = cov.to(torch.float32)

            reg = mahalanobis_reg
            if n_c < k_active:
                reg = max(reg, 1e-1)
                logger.warning(
                    f"  Cluster {c}: n_samples ({n_c}) < "
                    f"k_active ({k_active}), "
                    f"increasing reg to {reg:.1e}"
                )

            cov_reg = cov + reg * torch.eye(k_active)

            try:
                cond = torch.linalg.cond(cov_reg).item()
                logger.info(
                    f"  Cluster {c}: {n_c} samples, "
                    f"dim={k_active}, cond={cond:.2e}"
                )
            except Exception:
                pass

            try:
                precision_c = torch.linalg.inv(cov_reg).to(device)
            except torch.linalg.LinAlgError:
                logger.warning(
                    f"  Cluster {c}: singular covariance, "
                    "using pseudo-inverse"
                )
                precision_c = torch.linalg.pinv(cov_reg).to(device)

            precisions[c] = precision_c

    logger.info(
        f"Computed per-cluster covariances for " f"{len(precisions)} clusters"
    )

    return CovarianceResult(
        precisions=precisions,
        means_sub=means_sub,
        mask_indices=mask_indices,
    )


# ---------------------------------------------------------------------------
# Group 4 / 5 / 6  –  Scoring helpers  (Phase 1)
# ---------------------------------------------------------------------------


def compute_mahalanobis_distances_batch(
    z: torch.Tensor,
    batch_cluster_assignments: Optional[np.ndarray],
    n_clusters: int,
    cluster_precisions: Dict[int, torch.Tensor],
    cluster_means_sub: Dict[int, torch.Tensor],
    cluster_mask_indices: Dict[int, torch.Tensor],
    min_across_clusters: bool,
    device: torch.device,
) -> torch.Tensor:
    """Compute Mahalanobis distances for a batch of embeddings.

    Args:
        z: Encoded embeddings ``[batch_size, embedding_dim]``.
        batch_cluster_assignments: Cluster assignments per sample.
            If provided and *min_across_clusters* is False, uses the
            assigned cluster only.
        n_clusters: Total number of clusters.
        cluster_precisions: ``{cluster_id: precision_matrix}``.
        cluster_means_sub: ``{cluster_id: mean_vector}``.
        cluster_mask_indices: ``{cluster_id: active_dim_indices}``.
        min_across_clusters: If True, return the minimum Mahalanobis
            distance across *all* clusters for each sample.
        device: Torch device for output tensors.

    Returns:
        distances: ``[batch_size]`` Mahalanobis distances.
    """
    batch_size = z.shape[0]

    if not min_across_clusters and batch_cluster_assignments is not None:
        # Assigned-cluster-only mode
        distances = torch.zeros(batch_size, device=device)
        for c in range(n_clusters):
            if c not in cluster_precisions:
                continue
            c_mask = batch_cluster_assignments == c
            if not c_mask.any():
                continue

            idx = cluster_mask_indices[c]
            z_sub = z[c_mask][:, idx]
            diff = z_sub - cluster_means_sub[c]
            dist_sq = torch.sum((diff @ cluster_precisions[c]) * diff, dim=1)
            distances[c_mask] = torch.sqrt(dist_sq + 1e-10)
        return distances
    else:
        # Min-across-all-clusters mode
        all_dists = torch.full(
            (batch_size, n_clusters),
            float("inf"),
            device=device,
        )
        for c in range(n_clusters):
            if c not in cluster_precisions:
                continue
            idx = cluster_mask_indices[c]
            z_sub = z[:, idx]
            diff = z_sub - cluster_means_sub[c]
            dist_sq = torch.sum((diff @ cluster_precisions[c]) * diff, dim=1)
            all_dists[:, c] = torch.sqrt(dist_sq + 1e-10)
        return all_dists.min(dim=1).values


def compute_mask_disagreement_batch(
    z: torch.Tensor,
    batch_cluster_assignments: np.ndarray,
    masks_tensor: torch.Tensor,
    firing_threshold: float,
    device: torch.device,
) -> torch.Tensor:
    """Compute mask disagreement scores for a batch.

    Measures how well each sample's firing pattern matches the expected
    mask for its assigned cluster.  Uses ``1 - Jaccard`` similarity::

        score = 1 - |M_c ∩ F| / |M_c ∪ F|

    where *M_c* is the cluster mask and *F* is the actual firing
    pattern (``z > firing_threshold``).

    High score → the sample fires different neurons than the cluster
    expects → anomalous.

    Args:
        z: Encoded embeddings ``[batch, embedding_dim]``.
        batch_cluster_assignments: ``[batch]`` cluster ids.
        masks_tensor: ``[n_clusters, embedding_dim]`` binary masks.
        firing_threshold: Activation threshold for "firing".
        device: Torch device.

    Returns:
        ``[batch]`` disagreement scores (higher = more anomalous).
    """
    firing = (z > firing_threshold).float()  # [batch, embedding_dim]

    batch_cluster_tensor = torch.tensor(
        batch_cluster_assignments, dtype=torch.long, device=device
    )
    batch_masks = masks_tensor[batch_cluster_tensor]  # [batch, embedding_dim]

    # Jaccard: |intersection| / |union|
    intersection = (batch_masks * firing).sum(dim=1)  # [batch]
    union = torch.clamp(batch_masks + firing, max=1.0).sum(dim=1)  # [batch]

    # Avoid division by zero (if both mask and firing are empty)
    jaccard = intersection / torch.clamp(union, min=1.0)
    return 1.0 - jaccard  # [batch], higher = more anomalous


def fit_residual_mahalanobis(
    full_autoencoder: nn.Module,
    raw_data: np.ndarray,
    cluster_labels: np.ndarray,
    n_clusters: int,
    mahalanobis_reg: float,
    mahalanobis_diag: bool,
    device: torch.device,
) -> ResidualMahalanobisState:
    """Fit per-cluster Mahalanobis on SAE residuals.

    For each cluster, computes residuals ``r = x - decode(encode(x))``,
    then fits Mahalanobis in the raw input space.

    Args:
        full_autoencoder: Complete encoder+decoder model with
            ``extract_features`` and ``decoder`` methods.
        raw_data: ``[n_samples, d_hidden]`` raw hidden states.
        cluster_labels: ``[n_samples]`` cluster assignments.
        n_clusters: Total number of clusters.
        mahalanobis_reg: Covariance regularization.
        mahalanobis_diag: If True, use diagonal covariance.
        device: Torch device.

    Returns:
        :class:`ResidualMahalanobisState` with per-cluster means
        and precision matrices.
    """
    full_autoencoder.eval()

    # Compute residuals in batches
    residuals_list = []
    batch_size = 1024
    with torch.no_grad():
        for i in range(0, len(raw_data), batch_size):
            batch = torch.tensor(
                raw_data[i : i + batch_size],
                dtype=torch.float32,
                device=device,
            )
            encoded = full_autoencoder.extract_features(batch)
            decoded = full_autoencoder.decoder(encoded)
            residual = (batch - decoded).cpu().numpy()
            residuals_list.append(residual)
    residuals = np.vstack(residuals_list)
    d = residuals.shape[1]

    logger.info(
        f"Residuals shape: {residuals.shape}, "
        f"MSE: {np.mean(residuals**2):.6f}"
    )

    # Per-cluster Mahalanobis on residuals
    residual_means: Dict[int, torch.Tensor] = {}
    residual_precisions: Dict[int, torch.Tensor] = {}

    if mahalanobis_diag:
        logger.info("Using diagonal covariance (memory-efficient mode)")

    for c in range(n_clusters):
        c_mask = cluster_labels == c
        n_c = int(c_mask.sum())

        if n_c < 2:
            residual_means[c] = torch.zeros(d)
            if mahalanobis_diag:
                residual_precisions[c] = torch.ones(d)
            else:
                residual_precisions[c] = torch.eye(d)
            continue

        cluster_resid = residuals[c_mask]
        mean_c = torch.tensor(
            cluster_resid.mean(axis=0),
            dtype=torch.float32,
        )
        residual_means[c] = mean_c

        reg = mahalanobis_reg
        if n_c < d:
            reg = max(reg, 1e-1)

        if mahalanobis_diag:
            # Diagonal covariance: store 1/var as a 1D vector
            var = torch.tensor(
                cluster_resid.var(axis=0, ddof=1),
                dtype=torch.float32,
            )
            var += reg
            residual_precisions[c] = 1.0 / var
        else:
            # Full covariance on CPU to avoid OOM
            centered = torch.tensor(
                cluster_resid - cluster_resid.mean(axis=0),
                dtype=torch.float64,
            )
            cov = (centered.T @ centered / max(n_c - 1, 1)).to(torch.float32)
            cov += reg * torch.eye(d)

            try:
                precision = torch.linalg.inv(cov)
            except torch.linalg.LinAlgError:
                precision = torch.linalg.pinv(cov)
            # Keep on CPU — precision matrices don't fit on GPU
            residual_precisions[c] = precision

        if c < 3 or c == n_clusters - 1:
            if mahalanobis_diag:
                logger.info(
                    f"  Cluster {c}: {n_c} samples, "
                    f"residual dim={d} (diagonal)"
                )
            else:
                cond = torch.linalg.cond(cov).item()
                logger.info(
                    f"  Cluster {c}: {n_c} samples, "
                    f"residual dim={d}, cond={cond:.2e}"
                )

    logger.info(
        f"Fitted residual Mahalanobis for "
        f"{len(residual_precisions)} clusters"
        f"{' (diagonal)' if mahalanobis_diag else ''}"
    )

    return ResidualMahalanobisState(
        means=residual_means, precisions=residual_precisions
    )


def score_residual_mahalanobis(
    full_autoencoder: nn.Module,
    raw_data: np.ndarray,
    kmeans_index: Any,
    n_clusters: int,
    residual_means: Dict[int, torch.Tensor],
    residual_precisions: Dict[int, torch.Tensor],
    analysis_batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Score samples using per-cluster residual Mahalanobis.

    1. Encode → SAE code *z*, compute residual ``r = x - decode(z)``
    2. Assign to nearest cluster via magnitude centroid (FAISS)
    3. Score = ``Mahalanobis(r, μ_c, Σ_c^{-1})``

    Args:
        full_autoencoder: Complete encoder+decoder model.
        raw_data: ``[n_samples, d_hidden]``.
        kmeans_index: FAISS KMeans object with ``.index.search()``.
        n_clusters: Total number of clusters.
        residual_means: ``{cluster_id: mean_vector}``.
        residual_precisions: ``{cluster_id: precision_matrix_or_diag}``.
        analysis_batch_size: Batch size for encoding.
        device: Torch device.

    Returns:
        ``[n_samples]`` anomaly scores.
    """
    full_autoencoder.eval()

    # Compute residuals
    residuals_list = []
    encoded_list = []
    batch_size = min(analysis_batch_size, len(raw_data))
    with torch.no_grad():
        for i in range(0, len(raw_data), batch_size):
            batch = torch.tensor(
                raw_data[i : i + batch_size],
                dtype=torch.float32,
                device=device,
            )
            encoded = full_autoencoder.extract_features(batch)
            decoded = full_autoencoder.decoder(encoded)
            residuals_list.append((batch - decoded).cpu().numpy())
            encoded_list.append(encoded.cpu().numpy())

    residuals = np.vstack(residuals_list)
    encoded_data = np.vstack(encoded_list)

    # Cluster assignment using encoded data → magnitude centroids
    _, cluster_assigns = kmeans_index.index.search(
        encoded_data.astype(np.float32), 1
    )
    cluster_assigns = cluster_assigns.flatten()

    # Per-cluster Mahalanobis on residuals (on CPU to avoid OOM)
    scores = np.zeros(len(raw_data))
    for c in range(n_clusters):
        if c not in residual_precisions:
            continue
        idx = np.where(cluster_assigns == c)[0]
        if len(idx) == 0:
            continue

        r_batch = torch.tensor(
            residuals[idx],
            dtype=torch.float32,
        )
        diff = r_batch - residual_means[c]
        prec = residual_precisions[c]
        if prec.dim() == 1:
            # Diagonal precision: element-wise multiply
            dist_sq = torch.sum(diff * diff * prec, dim=1)
        else:
            dist_sq = torch.sum((diff @ prec) * diff, dim=1)
        scores[idx] = torch.sqrt(dist_sq + 1e-10).numpy()

    return scores
