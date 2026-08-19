#!/bin/bash
# Train a sparse autoencoder (SAE) on one model's IID activations.
#
# WHY YOU MIGHT NEED THIS
# -----------------------
# The FreqMask-KM, LearnedMask, CVDD-SAE, Local-FreqMask and BCE detectors all
# operate in SAE feature space, not raw activation space. No trained encoders
# are shipped with this code, so run this once per model before fitting any of
# those detectors, and put the result where $ULS_SAE_ROOT can find it. You do
# NOT need an encoder to reproduce the published table:
# `reference_scores/compute_metrics.py` reads the per-sample scores shipped in
# this repository and fits nothing.
#
# WHAT IT PRODUCES
# ----------------
# `trained_model.pth` inside each Hydra run directory, under
#   $ULS_OUTPUT_ROOT/multirun/iid/<date>/<exp_ident>-<time>/<idx>/
# That file is what you then pass as `anodet_algos.model_path=...`, or copy to
# $ULS_SAE_ROOT/<model_key>__<variant>.pth so the resolver in
# scripts/anodet_algos/run_baseline_experiments.py finds it.
#
# INPUT
# -----
# The IID activations for MODEL_KEY, as listed in
# `unsupervised_llm_safety.data.dataset_registry`, rooted on $ULS_ACTIVATION_ROOT.
# Nothing is downloaded here; if those files are absent the run fails loudly in
# the loader. `use_last_token=true use_complex_data=true` is the setting every
# published number uses.
#
# THE SWEEP AXES
# --------------
# `sparsity_weight` and `num_epochs` are commas => Hydra sweep axes, so the
# command below is a 2x3 = 6-point grid in ONE invocation (`--multirun`).
# sparsity_weight is the one parameter that does not transfer between models:
# it has to be retuned per model because it trades reconstruction against L0,
# and activation scale differs by model. The paper's encoders landed at
# 0.01 (qwen3_8b), 3 (gemma4_26b) and 300/700 (gpt_oss_20b, whose activation
# RMS is ~34x the 8B models'). Widen the grid below accordingly.
#
# SCHEDULER NOTE
# --------------
# exp_configs/config.yaml ships with Hydra's basic launcher, so `--multirun`
# runs the grid points SEQUENTIALLY on this machine. For a real sweep install
# `hydra-submitit-launcher` and add `hydra/launcher=submitit_slurm` plus your
# own launcher settings to the command below.
#
# PREREQUISITES
# -------------
#   pip install -e '.[checkpoints]'
# Run from the repository root, with a GPU visible.

set -euo pipefail

# One of the released models: gpt_oss_20b, gemma4_26b, qwen3_8b.
MODEL_KEY="${1:-qwen3_8b}"
EXP_IDENT="${2:-sae_${MODEL_KEY}_complex_data}"

echo "Training SAE for ${MODEL_KEY}"
echo "  activation root : ${ULS_ACTIVATION_ROOT:-<repo>/activation_data}"
echo "  output root     : ${ULS_OUTPUT_ROOT:-./outputs}"
echo "  exp_ident       : ${EXP_IDENT}"
echo "=================================================="

# `model@iid.model=autoencoder` is load-bearing: train_encoder.py reads its
# model config from `iid.model`, and the `- /model: autoencoder` default inside
# exp_configs/iid/encoder_training_config.yaml only populates the top-level
# `model` node.
python src/unsupervised_llm_safety/scripts/train_encoder.py \
    iid=encoder_training_config \
    model@iid.model=autoencoder \
    iid.model.sparsity_weight=0.01,0.1 \
    iid.model.feature_dim=16384 \
    iid.model_key="${MODEL_KEY}" \
    iid.num_epochs=5,10,20 \
    iid.batch_size=256 \
    exp_ident="${EXP_IDENT}" \
    exp_type=iid \
    use_last_token=true \
    use_complex_data=true \
    --multirun

echo
echo "=================================================="
echo "Done. Pick the grid point with the best reconstruction at your target"
echo "L0 (results.json in each run dir), then either pass its trained_model.pth"
echo "as anodet_algos.model_path=..., or copy it to"
echo "  \${ULS_SAE_ROOT}/${MODEL_KEY}__both.pth"
