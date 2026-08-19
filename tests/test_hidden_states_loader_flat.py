"""Tests for flat -> {train, test} normalization in hidden_states_loader.

The current HarmBench generation pipeline (extract_harmbench_generation.py ->
create_last_token_data.convert_harmbench) saves a FLAT dict
``{hidden_states, labels, ...}`` with no train/test split. The strict
``HiddenStateLoader`` path (``validate_data``) requires top-level
``{train, test}``. ``_normalize_flat_to_split`` bridges the two by wrapping a
flat dict into a deterministic 80/20 split so flat files load identically to
files that are already split.

Determinism is the critical property: the main results table runs anodet seeds
42-46, and every run must evaluate on the SAME test samples. The split therefore
uses a dedicated fixed seed and a *local* RNG, independent of any experiment seed
and isolated from the global numpy RNG.
"""

import numpy as np
import torch

from unsupervised_llm_safety.data.hidden_states_loader import (
    _normalize_flat_to_split,
    validate_data,
)


def _make_flat(n: int = 10, dim: int = 4) -> dict:
    """Flat dict whose row i == [i]*dim so test membership is identifiable."""
    hidden_states = (
        torch.arange(n, dtype=torch.float32)
        .repeat_interleave(dim)
        .reshape(n, dim)
    )
    labels = torch.tensor([i % 2 for i in range(n)], dtype=torch.long)
    return {
        "hidden_states": hidden_states,
        "labels": labels,
        "method": "AutoDAN",
        "model": "qwen3_8b",
        "layer": 30,
    }


def _row_ids(hs: torch.Tensor) -> set:
    """Recover the original integer id of each row (row i was [i]*dim)."""
    return set(hs[:, 0].round().to(torch.long).tolist())


class TestNormalizeFlatToSplit:
    def test_produces_valid_train_test_structure(self):
        out = _normalize_flat_to_split(_make_flat(10))
        assert set(out.keys()) >= {"train", "test"}
        assert validate_data(out)

    def test_eighty_twenty_counts(self):
        out = _normalize_flat_to_split(_make_flat(10))
        assert len(out["train"]["hidden_states"]) == 8
        assert len(out["test"]["hidden_states"]) == 2
        # 400-sample real case
        out2 = _normalize_flat_to_split(_make_flat(400))
        assert len(out2["train"]["hidden_states"]) == 320
        assert len(out2["test"]["hidden_states"]) == 80

    def test_partition_is_complete_and_disjoint(self):
        out = _normalize_flat_to_split(_make_flat(10))
        train_ids = _row_ids(out["train"]["hidden_states"])
        test_ids = _row_ids(out["test"]["hidden_states"])
        assert train_ids.isdisjoint(test_ids)
        assert train_ids | test_ids == set(range(10))

    def test_labels_track_their_samples(self):
        """Label of each split row must match label of its original row id."""
        out = _normalize_flat_to_split(_make_flat(10))
        for split in ("train", "test"):
            hs = out[split]["hidden_states"]
            lbl = out[split]["labels"]
            for row, label in zip(hs, lbl):
                orig_id = int(round(row[0].item()))
                assert int(label.item()) == orig_id % 2

    def test_deterministic_across_calls(self):
        a = _normalize_flat_to_split(_make_flat(400))
        b = _normalize_flat_to_split(_make_flat(400))
        assert torch.equal(
            a["test"]["hidden_states"], b["test"]["hidden_states"]
        )
        assert torch.equal(a["test"]["labels"], b["test"]["labels"])

    def test_isolated_from_global_numpy_rng(self):
        """Split must not depend on the global numpy RNG state."""
        np.random.seed(0)
        a = _normalize_flat_to_split(_make_flat(400))
        np.random.seed(123456)
        _ = np.random.rand(97)  # perturb global state further
        b = _normalize_flat_to_split(_make_flat(400))
        assert torch.equal(
            a["test"]["hidden_states"], b["test"]["hidden_states"]
        )

    def test_idempotent_on_already_split(self):
        """Already-split dicts (reference models) must pass through unchanged."""
        flat = _make_flat(10)
        split = _normalize_flat_to_split(flat)
        again = _normalize_flat_to_split(split)
        assert torch.equal(
            split["test"]["hidden_states"], again["test"]["hidden_states"]
        )
        assert torch.equal(
            split["train"]["hidden_states"], again["train"]["hidden_states"]
        )
