#!/bin/bash
# Fit and evaluate ONE detector on ONE model, from scratch.
#
# This is the smallest complete example of the main experiment entry point,
# `unsupervised_llm_safety.anodet_algos.run_anodet_analysis`. Every one of the
# nine configs in exp_configs/anodet_algos/ can be driven the same way; the
# only per-method difference is which `anodet_algos.<method>.*` keys exist and
# whether an SAE encoder is required.
#
#   bash scripts/anodet_algos/run_one_detector.sh                  # defaults
#   bash scripts/anodet_algos/run_one_detector.sh gmm qwen3_8b
#   bash scripts/anodet_algos/run_one_detector.sh learned_mask_lora gemma4_26b
#
# HYPERPARAMETERS
# ---------------
# You get the PAPER'S tuned values for the (model, method) you pick. They are
# not written out here: run_anodet_analysis reads them from
# exp_configs/best_hyperparameters.yaml and applies them before fitting, and
# logs every one it applied with a `[tuned-defaults]` prefix. Look for those
# lines in the run log to see exactly what was used.
#
# Several method YAMLs ship a default that is not the published one --
# `mahalanobis.score_type` defaults to `percentile`, which hard-clips scores at
# the 5th/95th training percentiles, and on some models that sends every
# adversarial sample to exactly 1.0 and makes all eight HarmBench attacks
# report an identical AUROC. The paper uses `sigmoid` there.
#
# To override one value, append it -- an explicit override still wins:
#   bash scripts/anodet_algos/run_one_detector.sh gmm qwen3_8b 42 '' \
#       anodet_algos.gmm.n_components=5
# To get the bare method-YAML defaults with no tuned layer at all:
#   ... anodet_algos.use_tuned_defaults=false
#
# ARGUMENTS
# ---------
#   $1 METHOD    a config name in exp_configs/anodet_algos/ (default:
#                cvdd_kmeans_masked)
#   $2 MODEL_KEY qwen3_8b | gpt_oss_20b | gemma4_26b (default: qwen3_8b)
#   $3 SEED      random seed (default: 42)
#   $4 VARIANT   OPTIONAL virtual algorithm name, when one config backs several
#                published methods. `anodet_algos=cvdd_kmeans_masked` means
#                FreqMask-KM by default; pass `local_freqmask_km`,
#                `cvdd_km_no_mask` or `cvdd_km_raw_no_mask` for the ablations,
#                or `cvdd_sae` with `anodet_algos=cvdd`. Leave empty for the
#                config's own default. The full naming table is at the top of
#                exp_configs/best_hyperparameters.yaml.
#   $5.. extra Hydra overrides, passed through verbatim.
#
# WHAT THIS IS NOT
# ----------------
# This REFITS a detector. It is not how you reproduce the published table --
# for that use `reference_scores/compute_metrics.py`, which reads the shipped
# per-sample scores and needs no training. Refitting gives numbers that agree with
# the paper to within seed noise, not bit-for-bit.
#
# THE NINE METHODS
# ----------------
#   raw activation space, no SAE needed
#     mahalanobis          single Gaussian on the safe features
#     gmm                  Gaussian mixture
#     one_class_svm        one-class SVM, RBF or linear kernel
#     deep_svdd            deep SVDD
#     cvdd                 context-vector data description (untrained encoder)
#   SAE feature space, needs anodet_algos.model_path
#     cvdd_kmeans_masked   FreqMask-KM and its ablations
#     learned_mask         LearnedMask stage 1 (clustering + binary STE masks)
#     learned_mask_lora    LearnedMask stage 2 (per-cluster LoRA)
#     bce                  supervised BCE probe (the labelled upper bound)
#
# INPUTS
# ------
#   $ULS_ACTIVATION_ROOT  the activations (dataset_registry resolves the rest)
#   $ULS_SAE_ROOT         the frozen SAE encoders, for the SAE-space methods
#   $ULS_OUTPUT_ROOT      where the run directory is created
#
# PREREQUISITES
# -------------
#   pip install -e '.[checkpoints]'
# Run from the repository root.

set -euo pipefail

METHOD="${1:-cvdd_kmeans_masked}"
MODEL_KEY="${2:-qwen3_8b}"
SEED="${3:-42}"
VARIANT="${4:-}"
shift $(( $# > 4 ? 4 : $# )) || true
EXTRA_OVERRIDES=("$@")

ACT_ROOT="${ULS_ACTIVATION_ROOT:-activation_data}"
SAE_ROOT="${ULS_SAE_ROOT:-${ACT_ROOT}/saes}"

# The frozen encoder for this model. gpt_oss_20b is the one model with two
# encoders -- __freqmask_km (sparsity weight 700) for the FreqMask-KM /
# CVDD-SAE family, __learnedmask (weight 300) for LearnedMask and BCE -- so
# pick per method. gemma4_26b and qwen3_8b use one encoder each.
# This mirrors the `sae_variant` field of exp_configs/best_hyperparameters.yaml;
# tests/test_tuned_defaults.py asserts the two agree.
case "${MODEL_KEY}:${METHOD}" in
  gpt_oss_20b:learned_mask*|gpt_oss_20b:bce) SAE_VARIANT=learnedmask ;;
  gpt_oss_20b:*)                             SAE_VARIANT=freqmask_km ;;
  *)                                         SAE_VARIANT=both ;;
esac
SAE_PATH="${SAE_ROOT}/${MODEL_KEY}__${SAE_VARIANT}.pth"

# Raw-space methods never read the encoder; passing null skips the load
# entirely rather than requiring a file that is not used.
case "${METHOD}" in
  mahalanobis|gmm|one_class_svm|deep_svdd|cvdd) MODEL_PATH="null" ;;
  *)                                            MODEL_PATH="${SAE_PATH}" ;;
esac

TUNED_OVERRIDE=()
if [ -n "${VARIANT}" ]; then
  TUNED_OVERRIDE=("anodet_algos.tuned_method=${VARIANT}")
fi

echo "method       : ${METHOD}"
echo "model_key    : ${MODEL_KEY}"
echo "seed         : ${SEED}"
echo "tuned method : ${VARIANT:-<config default>}"
echo "activations  : ${ACT_ROOT}"
echo "model_path   : ${MODEL_PATH}"
echo "output root  : ${ULS_OUTPUT_ROOT:-./outputs}"
if [ ${#EXTRA_OVERRIDES[@]} -gt 0 ]; then
  echo "extra        : ${EXTRA_OVERRIDES[*]}"
fi
echo "=================================================="

# `use_last_token=true use_complex_data=true` is the setting every published
# number uses; changing either changes which rows the registry loads.
#
# No hyperparameter is set here: they come from
# exp_configs/best_hyperparameters.yaml. Anything in "$@" is appended last and
# so overrides them.
python -m unsupervised_llm_safety.anodet_algos.run_anodet_analysis \
    exp_type=anodet_algos \
    exp_ident="${METHOD}_${MODEL_KEY}_seed${SEED}" \
    anodet_algos="${METHOD}" \
    anodet_algos.model_key="${MODEL_KEY}" \
    anodet_algos.model_path="${MODEL_PATH}" \
    anodet_algos.save_model=true \
    anodet_algos.use_last_token=true \
    anodet_algos.use_complex_data=true \
    "anodet_algos.${METHOD}.seed=${SEED}" \
    ${TUNED_OVERRIDE[@]+"${TUNED_OVERRIDE[@]}"} \
    ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}

echo
echo "=================================================="
echo "Done. The run directory holds <name>_trained.pt (the fitted detector),"
echo "results.json (metrics) and, for the mask methods, diagnostics/."
echo
echo "Grep the run log for '[tuned-defaults]' to see which published"
echo "hyperparameters were applied and which you overrode on the command line."
echo
echo "LearnedMask-LoRA is a TWO-STAGE method. Running anodet_algos="
echo "learned_mask_lora on its own refits stage 1 first. To reuse a completed"
echo "stage-1 run instead, which is what the paper does, add:"
echo "  anodet_algos.learned_mask_lora.step1_run_dir=<stage-1 run dir>"
