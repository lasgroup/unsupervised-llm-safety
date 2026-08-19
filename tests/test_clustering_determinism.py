"""Tests for the cluster_kmeans reproducibility knobs (seed/init/cpu).

FreqMask-KM clustering is not reproducible run to run unless these knobs are
set. The tests assert that:
- the kmeans++ path is bit-reproducible given a seed,
- it recovers obvious cluster structure,
- the default call (seed=None, faiss_random) still runs unchanged.
"""

import numpy as np
import torch

from unsupervised_llm_safety.anodet_algos.cvdd_km_utils import cluster_kmeans

_DEV = torch.device("cpu")


def _blobs(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 0.3, (200, 8))
    b = rng.normal(6.0, 0.3, (200, 8))
    c = rng.normal(-6.0, 0.3, (200, 8))
    return np.vstack([a, b, c]).astype(np.float32)


def test_kmeanspp_seed_reproducible():
    # Exercises the CPU kmeans++ path only. The GPU FAISS path
    # (cluster_on_cpu=False) is non-deterministic even when seeded, which is
    # what cluster_on_cpu exists to work around, so it cannot be asserted here.
    x = _blobs()
    r1 = cluster_kmeans(
        x,
        3,
        "default",
        0.0,
        _DEV,
        seed=7,
        init="kmeans++",
        cluster_on_cpu=True,
    )
    r2 = cluster_kmeans(
        x,
        3,
        "default",
        0.0,
        _DEV,
        seed=7,
        init="kmeans++",
        cluster_on_cpu=True,
    )
    assert np.array_equal(r1.labels, r2.labels)
    assert np.allclose(r1.centroids, r2.centroids)


def test_kmeanspp_recovers_blobs():
    x = _blobs()
    r = cluster_kmeans(
        x,
        3,
        "default",
        0.0,
        _DEV,
        seed=1,
        init="kmeans++",
        cluster_on_cpu=True,
    )
    # Three well-separated blobs -> three populated clusters.
    assert len(np.unique(r.labels)) == 3


def test_sklearn_ninit_runs_and_assigns_index():
    x = _blobs()
    r = cluster_kmeans(
        x,
        3,
        "default",
        0.0,
        _DEV,
        seed=3,
        init="sklearn_ninit",
        cluster_on_cpu=True,
    )
    assert r.kmeans is not None
    # Downstream assignment via the FAISS index shell must work.
    _, lbl = r.kmeans.index.search(x, 1)
    assert lbl.flatten().shape == (600,)


def test_unseeded_default_call_still_runs():
    x = _blobs()
    r = cluster_kmeans(x, 3, "default", 0.0, _DEV)  # seed=None, faiss_random
    assert r.labels.shape == (600,)
    assert r.centroids.shape == (3, 8)
