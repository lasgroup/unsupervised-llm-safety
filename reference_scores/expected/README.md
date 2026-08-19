# `expected/table1.csv` — the values `compute_metrics.py` checks against

One row per (model, method, metric). `compute_metrics.py` recomputes each from
the released scores and fails if any differs by more than the tolerance,
default 5e-4 — the precision at which the paper prints, so agreeing to that is
agreeing to every printed digit.

```
model,method,metric,expected,paper,std[,tol]
```

`method` is the paper's label; `metric` is `AUROC` (the three-slice mean) or
`TPR@5FPR`. An optional per-row `tol` overrides the default. Only the cells
listed here are checked, so this file is also the list of what the release
claims: three models times two methods, twelve rows.

`expected` is what the released scores recompute to, carried at six decimals so
the comparison is exact rather than rounded. `paper` is the value printed in
Table 1. Every row satisfies `round(expected, 3) == paper`, and
`compute_metrics.py --extra` prints both, so the comparison is visible rather
than asserted.

## What this does and does not prove

The gate recomputes the metrics from `../scores/` and compares against
`expected`. Passing therefore proves the shipped scores and the metric code are
mutually consistent — it is a regression test on this repository. The `paper`
column is what connects it to the publication, and it is checked by eye, not by
the gate: `expected` was derived from these same runs rather than transcribed
from the paper.
