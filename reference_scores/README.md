# Released anomaly scores

The per-sample anomaly scores our fitted detectors produced, for every
evaluation dataset, for three models and two methods (FreqMask-KM and LearnedMask-LoRA).
**This is reference data, not a reproduction package.**

## What it lets you do

**Recompute the reported metrics from the scores.** `compute_metrics.py` reads
the scores, recomputes AUROC and TPR@5%FPR, and diffs the result against the
values printed in the paper. That establishes one narrow thing: that the
numbers we report follow from the scores we release. It catches an aggregation
or reporting error. It does not check the method.

**Compute metrics we did not report.** A different operating point, precision
against recall, a per-dataset breakdown, a different pooling of the adversarial
attacks — additional computations are possible from these arrays.

**See the exact evaluation composition.** Which datasets are in the evaluation,
which category each falls in, and how many samples each contributes.

## What it does not let you do

**It cannot tell you whether your implementation matches ours.** Comparing your
scores against these per-sample would require running your detector on the same
representations, and those are not released: some of the upstream corpora
restrict redistribution of material derived from them. The SAE encoders and the
fitted detector checkpoints are not released either, so a per-sample comparison
is not reconstructable from this repository.

## Running it

```bash
python reference_scores/compute_metrics.py
```

No arguments needed. Expect `12 cell(s) within tolerance, 0 outside` and exit
code 0. It needs numpy and scikit-learn, no GPU, no `torch`, no network.

Useful flags: `--models` and `--methods` to narrow, `--tolerance` to change the
gate, `--out` to write the recomputed table as CSV, `--extra` to print one line
per cell (recomputed value, the paper's printed value, the difference) together
with the per-slice AUROCs the reference table does not carry.

## File format

```
scores/<model>/<method>/seed<k>/<name>_scores.npz
scores/<model>/<method>/seed<k>/<name>_categories.json
```

`<name>` is the detector's internal name, which is not always the method
directory name — `cvdd_kmeans_masked` subclasses CVDD and inherits its name, so
FreqMask-KM writes `cvdd_scores.npz`.

**`*_scores.npz`** is a zip of one `.npy` per evaluation dataset, keyed by
dataset name. Each is 1-D `float32`, one score per sample, in dataset order.
Higher means more anomalous.

```
GSM8K.npy         float32  (1196,)
MathSFT40K.npy    float32  (6381,)
Alpaca.npy        float32  (8321,)
BeaverTails.npy   float32  (185071,)
...
```

Each model carries the same 29 arrays: 19 IID, 8 adversarial and 2 OOD.

**`*_categories.json`** maps each dataset name to one of `iid`, `ood`,
`adversarial`, `benign_generalization`. That mapping is what turns the arrays
into the reported three-slice mean: the mean of the BeaverTails, ToxiGen and
pooled-adversarial AUROCs, with `iid` supplying the negatives.

## Seeds

FreqMask-KM ships one seed, because its scores vary little across seeds.
LearnedMask-LoRA ships five. `compute_metrics.py` averages across whatever
seeds it finds.

## Licence

MIT, like the rest of the repository (`../LICENSE`). These scores were computed
over upstream corpora carrying their own terms — most restrictively BeaverTails
and Alpaca (CC-BY-NC 4.0), with ARC under CC-BY-SA 4.0 — so consult those before
redistributing work derived from them.

