#!/bin/bash
# Refit every paper method on every released model, five seeds each.
#
# A thin wrapper around scripts/anodet_algos/run_baseline_experiments.py, which
# is the orchestrator: it holds the published hyperparameters per
# (method, model) and turns each into one `run_anodet_analysis` command line.
#
#   bash scripts/anodet_algos/run_paper_sweep.sh            # DRY RUN (default)
#   bash scripts/anodet_algos/run_paper_sweep.sh --run      # actually run it
#
# START WITH THE DRY RUN. It prints the exact Hydra override list for every
# published configuration without executing anything, which is both the fastest
# way to see what a cell of the table actually was, and the check that your
# $SAP_SAE_ROOT is resolving to real files (the printed model_path is the path
# that will be opened).
#
# SCOPE
# -----
# Ten methods x three released models = 30 combinations, of which the ones with
# a published tuned configuration run and the rest are skipped with a warning
# on stdout. Combined with --seeds that is up to 75 fits; on Hydra's basic
# launcher they run SEQUENTIALLY in this process. Install
# `hydra-submitit-launcher` and add your own launcher settings via --override
# to fan them out. Measured single-fit runtimes: FreqMask-KM ~8 min,
# LearnedMask stage 1 ~44 min, worst observed 1:17:31.
#
# PREREQUISITES
# -------------
#   pip install -e '.[checkpoints]'
#   $SAP_ACTIVATION_ROOT and $SAP_SAE_ROOT pointing at your extracted
#   activations and your trained SAE encoders
# Run from the repository root.

set -euo pipefail

MODE="${1:---dry-run}"
case "${MODE}" in
  --run)      DRY_FLAG=() ;;
  --dry-run)  DRY_FLAG=(--dry-run) ;;
  *) echo "usage: $0 [--dry-run|--run]" >&2; exit 2 ;;
esac

python scripts/anodet_algos/run_baseline_experiments.py \
    --best \
    --seeds 42,43,44,45,46 \
    "${DRY_FLAG[@]}"
