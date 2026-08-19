#!/bin/bash
# Extract ToxiGen activations for the three RELEASED models.
#
# WHY YOU MIGHT NEED THIS
# -----------------------
# No cached representations are released with this code, so anything that fits
# or evaluates a detector needs you to extract your own first. This script does
# ToxiGen for the three models the paper reports, at the layers it used.
#
# `reference_scores/compute_metrics.py` needs none of this and reproduces
# every published number exactly from the scores shipped in this repository;
# only refitting or re-scoring a detector needs activations.
#
# BEAVERTAILS IS NOT COVERED HERE. There is no BeaverTails extractor in this
# repository -- see the note at the bottom of this file.
#
# WHAT IT PRODUCES
# ----------------
# One file per model, at exactly the path `dataset_registry.py` expects,
# relative to your activation root:
#
#   <root>/toxigen/toxigen_gpt_oss_20b_layer20_full.pth
#   <root>/toxigen/toxigen_gemma4_26b_layer25_full.pth
#   <root>/toxigen/toxigen_qwen3_8b_layer30_full.pth
#
# The layer per model is not a free choice -- it is the layer the released
# detectors were fitted on. Extracting at any other layer produces a tensor of
# the wrong provenance (and, for a different width, one that will not even load
# into the detector).
#
# SETTINGS
# --------
# Last-token extraction (the default path through extract_activations.py: the
# hook takes activations[:, -1, :]) with the chat template applied and
# `add_generation_prompt=True`. That is the "last-token" half of the
# "last-token + complex-data" setting the paper reports; "complex-data" is an
# evaluation-time choice made by `use_complex_data` in exp_configs/config.yaml,
# not an extraction-time one, so there is nothing to set here for it.
#
# The `_safe` companion slice (`ToxiGen_safe`) that the manifests also list is
# the *same file*: the loader filters it by label. One extraction per model is
# enough for both rows.
#
# PREREQUISITES
# -------------
#   pip install -e '.[extract]'
#   huggingface-cli login          # ToxiGen needs an accepted licence
# and accept the terms at https://huggingface.co/datasets/toxigen/toxigen-data
#
# Run from the repository root. A 20-26B model in bf16 needs a GPU with room
# for it (or `analysis.model.device_map=auto` will shard across what you have).

set -euo pipefail

DATASET="toxigen"
# Where the extracted files land. Point this at the same directory you set
# ULS_ACTIVATION_ROOT to (default: ./activation_data next to this repo).
BASE_OUTPUT_DIR="${ULS_ACTIVATION_ROOT:-activation_data}"

echo "Extracting ToxiGen activations for the 3 released models"
echo "Output root: ${BASE_OUTPUT_DIR}"
echo "=================================================="

# model_key : canonical layer  (must match dataset_registry._CANONICAL_LAYER)
MODELS=(
  "gpt_oss_20b:20"
  "gemma4_26b:25"
  "qwen3_8b:30"
)

i=0
for spec in "${MODELS[@]}"; do
    i=$((i + 1))
    MODEL_KEY="${spec%%:*}"
    LAYER="${spec##*:}"
    OUT="${BASE_OUTPUT_DIR}/${DATASET}/${DATASET}_${MODEL_KEY}_layer${LAYER}_full.pth"

    echo
    echo "[${i}/${#MODELS[@]}] ${MODEL_KEY}, layer ${LAYER} -> ${OUT}"

    # model_key, layer_index, output_path and max_samples are all set here, so
    # exp_configs/analysis/extract_activations_config.yaml contributes only its
    # OOM-backoff and seed settings. The batch size comes from
    # analysis.memory.initial_batch_size alone.
    python src/unsupervised_llm_safety/scripts/activations/extract_activations.py \
        exp_type=extract_activations \
        exp_ident="${MODEL_KEY}-${DATASET}" \
        analysis=extract_activations_config \
        analysis.model_key="${MODEL_KEY}" \
        analysis.dataset.name="toxigen/toxigen-data" \
        analysis.dataset.subset=train \
        analysis.dataset.split=train \
        analysis.dataset.max_samples=-1 \
        analysis.extraction.layer_index="${LAYER}" \
        analysis.extraction.output_path="${OUT}" \
        analysis.extraction.save_metadata=true \
        analysis.memory.initial_batch_size=8 \
        hydra.job.chdir=False

    echo "[${i}/${#MODELS[@]}] ${MODEL_KEY} done."
done

echo
echo "=================================================="
echo "All ToxiGen extractions complete."
echo
echo "NOTE ON BEAVERTAILS"
echo "-------------------"
echo "This repository contains no BeaverTails extraction script, and this"
echo "script does not provide one. The BeaverTails representations used in the"
echo "paper were produced by code that is not checked in here, so those files"
echo "cannot be reproduced with this repository. See reference_scores/README.md."
