#!/usr/bin/env python3
"""Recompute the paper's reported metrics from the released per-sample scores.

This is a reference calculation, not a reproduction. It reads the anomaly
scores in ``scores/`` -- which are the scores our fitted detectors actually
produced -- and recomputes AUROC and TPR@5%FPR from them, then diffs the
result against the values printed in the paper (``expected/table1.csv``).

What that establishes is narrow and worth stating plainly: that the numbers we
report follow from the scores we release. It does **not** let you check your
own implementation against ours, because the representations those scores were
computed from are not released, so there is no input you could feed to get a
comparable output.

The scores are more useful as data than as a check. They let you compute
metrics we did not report -- a different operating point, precision-recall, a
per-dataset breakdown -- without asking us for anything.

Layout under ``--scores-dir`` (default: this directory)::

    scores/<model>/<method>/seed<k>/*_scores.npz
                                    *_categories.json

Golden CSV columns (``expected/table1.csv``)::

    model,method,metric,expected,paper,std[,tol]

``method`` is the paper's label ("FreqMask-KM", "LearnedMask-LoRA", ...);
``metric`` is ``AUROC`` (the 3-slice mean), ``TPR@5FPR``, ``TPR@10FPR`` or
``FPR@95TPR``. Only the cells present in that file are recomputed and gated, so
it is also the list of what the release claims. An optional per-row ``tol``
overrides ``--tolerance``. Per-slice AUROCs are also computed and printed under
``--extra`` as ``AUROC.BeaverTails`` etc.; they never fail the run because the
reference table does not carry them.

"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from unsupervised_llm_safety.analysis.anomaly_metrics import (
    compute_three_slice_metrics,
)

# --- Table 1 shape ---------------------------------------------------------

MODELS: Tuple[str, ...] = (
    "ministral_8b",
    "llama3_8b",
    "qwen3_8b",
    "gpt_oss_20b",
    "gemma4_26b",
    "qwen2_1.5b",
)

# Directory id -> the label used in reference_scores/expected/table1.csv. The CSV is the
# authority on spelling; the directory id is the typeable form for --methods
# and for the bundle layout.
METHOD_LABELS: Dict[str, str] = {
    "bce": "BCE (linear probe, supervised)",
    "freqmask_km": "FreqMask-KM",
    "learnedmask_lora": "LearnedMask-LoRA",
    "mahalanobis": "Mahalanobis",
    "gmm": "GMM",
    "one_class_svm": "One-Class SVM",
    "deep_svdd": "Deep SVDD",
    "cvdd": "CVDD",
}
LABEL_TO_METHOD: Dict[str, str] = {v: k for k, v in METHOD_LABELS.items()}
METHODS: Tuple[str, ...] = tuple(METHOD_LABELS)

# compute_three_slice_metrics keys -> the metric names used in the reference table.
TOP_LEVEL_METRICS: Dict[str, str] = {
    "AUROC": "three_slice_mean_auroc",
    "FPR@95TPR": "three_slice_mean_fpr95",
    "TPR@5FPR": "three_slice_mean_tpr_at_5_fpr",
    "TPR@10FPR": "three_slice_mean_tpr_at_10_fpr",
}

SLICE_METRICS: Dict[str, str] = {
    "AUROC.BeaverTails": "BeaverTails",
    "AUROC.ToxiGen": "ToxiGen",
    "AUROC.Adversarial": "Adversarial (pooled)",
}

# The paper prints 3 decimals, so agreeing to +-0.0005 is agreeing to the full
# printed precision.
DEFAULT_TOLERANCE = 5e-4

EXPECTED_FIELDS = [
    "model",
    "method",
    "metric",
    "expected",
    "std",
]

_SCORE_KEY = "score"


# --- Loading one seed's scores ---------------------------------------------


def _single(paths: Sequence[Path], what: str, where: Path) -> Path:
    """Return the one matching file, or fail loudly."""
    if not paths:
        raise FileNotFoundError(f"no {what} in {where}")
    if len(paths) > 1:
        raise RuntimeError(
            f"ambiguous {what} in {where}: {[p.name for p in paths]}"
        )
    return paths[0]


def seed_dirs(root: Path) -> List[Path]:
    """Numbered/`seed*` subdirectories of *root*, in a stable order."""
    if not root.is_dir():
        return []
    out = [p for p in root.iterdir() if p.is_dir()]
    return sorted(out, key=lambda p: (len(p.name), p.name))


def load_cached_scores(seed_dir: Path) -> Tuple[Dict, Dict[str, str]]:
    """Read one seed's ``*_scores.npz`` + ``*_categories.json``."""
    npz = _single(
        sorted(seed_dir.glob("*_scores.npz")), "scores npz", seed_dir
    )
    cat = _single(
        sorted(seed_dir.glob("*_categories.json")), "categories", seed_dir
    )
    with np.load(npz) as data:
        all_scores = {
            name: {_SCORE_KEY: np.asarray(data[name])} for name in data.files
        }
    categories = json.loads(cat.read_text())
    missing = sorted(set(all_scores) - set(categories))
    if missing:
        raise RuntimeError(
            f"{seed_dir}: {len(missing)} dataset(s) in the score file have no "
            f"category: {missing[:5]}"
        )
    return all_scores, categories


# --- Metrics ----------------------------------------------------------------


def metrics_for_seed(
    all_scores: Dict, categories: Dict[str, str]
) -> Dict[str, float]:
    """Flatten compute_three_slice_metrics into the reference table's metric ids."""
    three = compute_three_slice_metrics(
        all_scores, categories, score_key=_SCORE_KEY
    )
    flat: Dict[str, float] = {}
    for metric_id, key in TOP_LEVEL_METRICS.items():
        value = three.get(key)
        if value is not None:
            flat[metric_id] = float(value)
    slices = three.get("slices", {})
    for metric_id, slice_name in SLICE_METRICS.items():
        entry = slices.get(slice_name)
        if entry is not None and "auroc" in entry:
            flat[metric_id] = float(entry["auroc"])
    n_slices = len(slices)
    if n_slices != 3:
        raise RuntimeError(
            f"expected 3 slices, got {n_slices} ({sorted(slices)}). A cell "
            "computed from fewer slices is not a Table-1 number."
        )
    return flat


def aggregate(
    per_seed: List[Dict[str, float]],
) -> Dict[str, Tuple[float, float]]:
    """Mean and population std across seeds, per metric."""
    out: Dict[str, Tuple[float, float]] = {}
    keys = sorted({k for d in per_seed for k in d})
    for k in keys:
        vals = [d[k] for d in per_seed if k in d and not math.isnan(d[k])]
        if not vals:
            continue
        out[k] = (float(np.mean(vals)), float(np.std(vals)))
    return out


# --- Golden CSV -------------------------------------------------------------


def read_expected(path: Path) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """Read the reference table keyed by (model, method-id, metric).

    Method names in the CSV are the paper's labels ("FreqMask-KM"); they are
    normalised to the directory ids used everywhere else.
    """
    expected: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        missing = [
            c
            for c in ("model", "method", "metric", "expected")
            if c not in (reader.fieldnames or [])
        ]
        if missing:
            raise RuntimeError(
                f"{path} is missing required column(s) {missing}; "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            label = row["method"].strip()
            method = LABEL_TO_METHOD.get(label, label)
            key = (row["model"].strip(), method, row["metric"].strip())
            tol_raw = (row.get("tol") or "").strip()
            std_raw = (row.get("std") or "").strip()
            expected[key] = {
                "label": label,
                "mean": float(row["expected"]),
                "std": float(std_raw) if std_raw else None,
                "tol": float(tol_raw) if tol_raw else None,
                "paper": (row.get("paper") or "").strip(),
            }
    return expected


def write_table(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write the recomputed table in the reference table's own schema."""
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=EXPECTED_FIELDS + ["n_seeds"])
        writer.writeheader()
        for row in rows:
            # Emit the paper's label so the file is drop-in compatible with
            # reference_scores/expected/table1.csv.
            writer.writerow(
                {
                    **row,
                    "method": METHOD_LABELS.get(row["method"], row["method"]),
                }
            )


# --- Driver -----------------------------------------------------------------


def collect(
    scores_dir: Path,
    cells: Sequence[Tuple[str, str]],
    extra: bool,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Recompute every requested (model, method) cell. Returns (rows, notes)."""
    rows: List[Dict[str, Any]] = []
    notes: List[str] = []

    for model, method in cells:
        cell_dir = scores_dir / "scores" / model / method
        sds = seed_dirs(cell_dir)
        if not sds:
            notes.append(
                f"MISSING {model}/{method}: no seed dirs under {cell_dir}"
            )
            continue
        per_seed: List[Dict[str, float]] = []
        for sd in sds:
            try:
                all_scores, categories = load_cached_scores(sd)
                per_seed.append(metrics_for_seed(all_scores, categories))
            except Exception as exc:  # noqa: BLE001 - reported, not hidden
                notes.append(f"FAILED {model}/{method}/{sd.name}: {exc}")
        if not per_seed:
            notes.append(
                f"EMPTY {model}/{method}: {len(sds)} seed dir(s) yielded "
                "no usable scores"
            )
            continue
        if extra:
            print(f"  {model}/{method}: {len(per_seed)} seed(s)")
        for metric, (mean, std) in aggregate(per_seed).items():
            rows.append(
                {
                    "model": model,
                    "method": method,
                    "metric": metric,
                    "expected": round(mean, 8),
                    "std": round(std, 8),
                    "n_seeds": len(per_seed),
                }
            )
    return rows, notes


def diff_against_expected(
    rows: Sequence[Dict[str, Any]],
    expected: Dict[Tuple[str, str, str], Dict[str, Any]],
    tolerance: float,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Return (failures, passes, missing, extra).

    ``missing`` are reference cells nothing was recomputed for -- a gate failure.
    ``extra`` are metrics this run computed that the reference table does not carry;
    they are informational and never fail the run.
    """
    failures: List[str] = []
    passes: List[str] = []
    extra: List[str] = []
    seen = set()

    for row in rows:
        key = (row["model"], row["method"], row["metric"])
        exp = expected.get(key)
        if exp is None:
            extra.append(f"{'/'.join(key)} = {row['expected']:.6f}")
            continue
        seen.add(key)
        tol = exp["tol"] if exp["tol"] is not None else tolerance
        delta = row["expected"] - exp["mean"]
        line = (
            f"{row['model']:<14s} {exp['label']:<18s} {row['metric']:<10s} "
            f"expected={exp['mean']:.6f} got={row['expected']:.6f} "
            f"delta={delta:+.3e} tol={tol:.1e}"
            + (f" paper={exp['paper']}" if exp.get("paper") else "")
        )
        if abs(delta) > tol:
            failures.append(line)
        else:
            passes.append(line)

    missing = ["/".join(k) for k in sorted(expected) if k not in seen]
    return failures, passes, missing, extra


def main(argv: Optional[Sequence[str]] = None) -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        prog="compute_metrics",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--scores-dir",
        type=Path,
        default=here,
        help="directory containing scores/ (default: this directory, which "
        "ships them, so the script runs with no arguments)",
    )
    parser.add_argument(
        "--models",
        default=",".join(MODELS),
        help=f"comma-separated subset of: {','.join(MODELS)}",
    )
    parser.add_argument(
        "--methods",
        default=",".join(METHODS),
        help=f"comma-separated subset of: {','.join(METHODS)}",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=here / "expected" / "table1.csv",
        help="reference table to diff against (default: reference_scores/expected/table1.csv)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"absolute tolerance per cell (default: {DEFAULT_TOLERANCE:g}). "
        "A row's own 'tol' column overrides this.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="also write the recomputed table to this CSV",
    )
    parser.add_argument(
        "--emit-expected",
        type=Path,
        default=None,
        help="write the recomputed cells out in reference-table form and skip the "
        "diff. Only for seeding a new baseline you have verified yourself.",
    )
    parser.add_argument(
        "--extra",
        action="store_true",
        help="print one line per cell -- recomputed value, the paper's "
        "printed value, and the difference -- plus any metric computed "
        "that the reference table does not carry",
    )
    args = parser.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        parser.error(f"unknown model(s): {unknown}; known: {list(MODELS)}")
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        parser.error(f"unknown method(s): {unknown}; known: {list(METHODS)}")

    expected: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    if args.emit_expected is None:
        if not args.expected.exists():
            print(
                f"ERROR: reference table not found: {args.expected}\n"
                "       Populate it (see --emit-expected) before gating "
                "on it.",
                file=sys.stderr,
            )
            return 2
        expected = {
            k: v
            for k, v in read_expected(args.expected).items()
            if k[0] in models and k[1] in methods
        }
        if not expected:
            print(
                f"ERROR: {args.expected} carries no cell for models={models} "
                f"methods={methods}. Nothing to gate on.",
                file=sys.stderr,
            )
            return 2
        # Recompute exactly the cells the reference table covers. Iterating the
        # full models x methods cross product instead would report every
        # unreleased combination as MISSING and drown the real result.
        cells = sorted({(k[0], k[1]) for k in expected})
    else:
        cells = sorted((m, meth) for m in models for meth in methods)

    print(f"scores    : {args.scores_dir}")
    print(f"cells     : {len(cells)}")

    rows, notes = collect(
        args.scores_dir,
        cells,
        args.extra,
    )

    if args.out is not None:
        write_table(args.out, rows)
        print(f"wrote {args.out}  ({len(rows)} rows)")

    if args.emit_expected is not None:
        write_table(args.emit_expected, rows)
        print(f"wrote {args.emit_expected}  ({len(rows)} rows); diff skipped")
        for note in notes:
            print(f"NOTE  {note}")
        return 0 if not notes else 1

    failures, passes, missing, extra = diff_against_expected(
        rows, expected, args.tolerance
    )

    print()
    if args.extra:
        for line in passes:
            print(f"ok      {line}")
        for line in extra:
            print(f"extra   {line}")
    for line in failures:
        print(f"FAIL    {line}")
    for line in missing:
        print(f"MISSING {line}")
    for note in notes:
        print(f"NOTE    {note}")

    print(
        f"{len(passes)} cell(s) within tolerance, {len(failures)} outside, "
        f"{len(missing)} missing, {len(notes)} note(s), "
        f"{len(extra)} extra metric(s) not in the reference table."
    )
    if failures or missing or notes:
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
