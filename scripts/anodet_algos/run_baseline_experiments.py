#!/usr/bin/env python3
"""Sweep orchestrator for the anomaly-detection baselines.

This builds and launches ``unsupervised_llm_safety.anodet_algos.
run_anodet_analysis`` command lines. It does not itself do any numerical work;
every invocation it prints is a command you could equally have typed by hand,
which is why ``--dry-run`` is the most useful thing here: it shows you the
exact Hydra override list for a published configuration.

    # Show every command for the paper's methods on the released models
    python scripts/anodet_algos/run_baseline_experiments.py --best --dry-run

    # One method, one model, actually run it
    python scripts/anodet_algos/run_baseline_experiments.py \\
        --best --algorithms cvdd_km_global_mask --models qwen3_8b

    # Five seeds of the two headline methods
    python scripts/anodet_algos/run_baseline_experiments.py \\
        --best --seeds 42,43,44,45,46 \\
        --algorithms cvdd_km_global_mask,learned_mask_step1

    # Hyperparameter sweep mode (no --best): sweeps the grids below
    python scripts/anodet_algos/run_baseline_experiments.py \\
        --algorithms mahalanobis,gmm --models qwen3_8b

ENCODER PATHS, UNRELEASED MODELS, LAUNCHERS
-------------------------------------------
1. ``model_path`` is resolved by :func:`sae_path`, which follows exactly the
   convention ``unsupervised_llm_safety.data.precompute_sae_encodings``
   already uses: ``$ULS_SAE_<MODEL_KEY>``, else
   ``$ULS_SAE_ROOT/<model_key>__<variant>.pth``. Name your trained encoders
   that way, or set the environment variables to point at them.

2. The models whose SAE encoders were never archived are still described, but
   cannot be run end to end. ``qwen2_1.5b``, ``ministral_8b`` and
   ``llama3_8b`` keep their tuned entries in
   exp_configs/best_hyperparameters.yaml, marked ``released: false``. Each
   SAE-dependent method of a marked model carries the sentinel
   ``model_path: unreleased``, and build_command refuses to emit a command for
   it rather than guessing a path (see reference_scores/README.md) -- each
   method used a *different* encoder, so a guessed path would produce a run
   that looks healthy and is fitted against the wrong one. Concretely,
   ``--best --models ministral_8b --dry-run`` emits 5 of 10 commands: the five
   methods that read raw representations (cvdd, deep_svdd, gmm, mahalanobis,
   one_class_svm). Four of the five it skips need an SAE encoder, and ``bce``
   was never tuned for that model.

   ``--best`` on its own iterates the three released models -- gpt_oss_20b,
   gemma4_26b, qwen3_8b -- over the ten Table-2 methods and emits 27 of those
   30 combinations, each with its published hyperparameters unchanged. The
   three it skips are ``bce``, ``cvdd_sae`` and ``local_freqmask_km`` on
   qwen3_8b, which were never tuned for that model; the warning names each one.

3. No ``hydra.launcher.*`` settings are shipped. exp_configs/config.yaml uses
   Hydra's basic launcher, against which any ``hydra.launcher.<key>`` override
   is a hard error. If you install ``hydra-submitit-launcher``, pass your own
   through verbatim with ``--override``:

       --override hydra/launcher=submitit_slurm \\
       --override hydra.launcher.mem_per_cpu=192G
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# SAE encoder resolution.
#
# Deliberately identical in shape to
# `unsupervised_llm_safety.data.precompute_sae_encodings._sae_path_for`,
# so there is one convention in the repository and not two.
# ---------------------------------------------------------------------------

# Repository root, derived from this file's location:
#   <repo>/scripts/anodet_algos/run_baseline_experiments.py
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Make the package importable when this script is run straight out of a clone
# that has not been `pip install -e`'d; the hyperparameter table is read
# through the package.
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from unsupervised_llm_safety.common.tuned_defaults import (  # noqa: E402
    UNRELEASED_SAE,
    algorithm_config_map,
    load_tuned_defaults,
)

# Same default chain as dataset_registry.ROOT_DATA_PATH -> "<root>/saes".
_ACTIVATION_ROOT = os.environ.get(
    "ULS_ACTIVATION_ROOT", str(_REPO_ROOT / "activation_data")
)


def sae_root() -> str:
    """Return the directory holding the frozen SAE encoders."""
    return os.environ.get("ULS_SAE_ROOT", f"{_ACTIVATION_ROOT}/saes")


def sae_path(model_key: str, variant: str) -> str:
    """Resolve the SAE checkpoint for *model_key* / *variant*.

    Precedence:
      1. ``$ULS_SAE_<MODEL_KEY>_<VARIANT>`` (uppercased, dots -> underscores)
      2. ``$ULS_SAE_<MODEL_KEY>``
      3. ``<sae_root()>/<model_key>__<variant>.pth``

    ``variant`` names which encoder a method was tuned against, because
    gpt_oss_20b needs two different ones: the paper's FreqMask-KM /
    CVDD-SAE / Local-FreqMask cells use the sparsity-weight-700 encoder
    (``freqmask_km``) and its BCE / LearnedMask cells use the
    sparsity-weight-300 one (``learnedmask``). gemma4_26b and qwen3_8b use a
    single encoder for every method, shipped as ``both``.
    """
    slug = model_key.upper().replace(".", "_")
    specific = os.environ.get(f"ULS_SAE_{slug}_{variant.upper()}")
    if specific:
        return specific
    generic = os.environ.get(f"ULS_SAE_{slug}")
    if generic:
        return generic
    return f"{sae_root()}/{model_key}__{variant}.pth"


# Which encoder each released model's methods were tuned against. Used for
# sweep mode (no --best), where no per-method model_path is pinned.
DEFAULT_SAE_VARIANT: Dict[str, str] = {
    "gpt_oss_20b": "freqmask_km",
    "gemma4_26b": "both",
    "qwen3_8b": "both",
}

# Virtual algorithm name -> Hydra config name.
# Algorithms not listed here use themselves as the config name.
ALGORITHM_CONFIG_MAP: Dict[str, str] = {
    "cvdd_km_global_mask": "cvdd_kmeans_masked",
    "cvdd_km_no_mask": "cvdd_kmeans_masked",
    "cvdd_km_raw_no_mask": "cvdd_kmeans_masked",
    "cvdd_sae": "cvdd",
    "local_freqmask_km": "cvdd_kmeans_masked",
    "learned_mask_step1": "learned_mask",
}

# The same mapping is carried by every entry in best_hyperparameters.yaml.
# Cross-check the two and fail loudly on disagreement rather than letting a run
# silently compose the wrong config.
_FILE_CONFIG_MAP = algorithm_config_map()
_CONFLICTS = {
    name: (ALGORITHM_CONFIG_MAP[name], cfg_name)
    for name, cfg_name in _FILE_CONFIG_MAP.items()
    if name in ALGORITHM_CONFIG_MAP and ALGORITHM_CONFIG_MAP[name] != cfg_name
}
if _CONFLICTS:
    raise ValueError(
        "ALGORITHM_CONFIG_MAP disagrees with best_hyperparameters.yaml: "
        + ", ".join(
            f"{n}: script says {a!r}, file says {b!r}"
            for n, (a, b) in sorted(_CONFLICTS.items())
        )
    )
for _name, _cfg_name in _FILE_CONFIG_MAP.items():
    if _name != _cfg_name:
        ALGORITHM_CONFIG_MAP.setdefault(_name, _cfg_name)

# Algorithms that require SAE encoder paths (in sweep mode).
# In --best mode, model_path is included in BEST_HYPERPARAMETERS directly.
ALGORITHMS_WITH_ENCODERS = {
    "cvdd",
    "cvdd_kmeans_masked",
}

# Hyperparameter sweeps for each algorithm (used in non-best mode)
HYPERPARAMETER_SWEEPS = {
    "one_class_svm": {
        "anodet_algos.one_class_svm.nu": "0.1,0.01",
        "anodet_algos.one_class_svm.kernel": "rbf,linear",
        "anodet_algos.one_class_svm.gamma": "0.0001,0.001,0.01",
        "anodet_algos.one_class_svm.lr": "0.001,0.01",
    },
    "mahalanobis": {
        "anodet_algos.mahalanobis.regularization": "1e-5,1e-4,1e-3,1e-2,1e-1",
        "anodet_algos.mahalanobis.score_type": "percentile,sigmoid",
    },
    "gmm": {
        "anodet_algos.gmm.n_components": "5,10,100",
        "anodet_algos.gmm.score_normalization": "percentile,sigmoid",
        "anodet_algos.gmm.init_method": "kmeans++,random",
    },
    "deep_svdd": {
        "anodet_algos.deep_svdd.nu": "0.05,0.1,0.2",
        "anodet_algos.deep_svdd.rep_dim": "16,64,128",
        "anodet_algos.deep_svdd.lr": "0.0001,0.001,0.01",
    },
    "cvdd": {
        "anodet_algos.cvdd.n_contexts": "10,50,100,200,250,500",
        "anodet_algos.cvdd.ortho_weight": "0.0001,0.001,0.01,0.1",
        "anodet_algos.cvdd.use_pretrained_encoder": "false",
    },
    "cvdd_kmeans_masked": {
        "anodet_algos.cvdd_kmeans_masked.n_clusters": "3,64,128,256,512",
        "anodet_algos.cvdd_kmeans_masked.k_mask": "50,100,200,300,400",
        "anodet_algos.cvdd_kmeans_masked.epochs": "50,100,200,300",
    },
}

# ---------------------------------------------------------------------------
# Best hyperparameters per (algorithm, model_key).
#
# The values live in ``exp_configs/best_hyperparameters.yaml``, which is also
# what ``run_anodet_analysis`` reads to make them the DEFAULT for a single run
# -- one table, whether you go through this orchestrator or run a detector
# directly.
#
# What this function adds on top of the file is the one thing that cannot
# live in it: turning each entry's ``sae_variant`` into a concrete
# ``anodet_algos.model_path`` for the machine it is running on. Entries whose
# encoder was never archived carry the sentinel ``unreleased``; build_command
# refuses to emit a command for those rather than guessing a path.
# ---------------------------------------------------------------------------


def _override_value(value) -> str:
    """Render a YAML value as a Hydra command-line override value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_override_value(v) for v in value) + "]"
    if isinstance(value, dict):
        inner = ",".join(f"{k}:{_override_value(v)}" for k, v in value.items())
        return "{" + inner + "}"
    return str(value)


def _load_best_hyperparameters() -> Dict[Tuple[str, str], Dict[str, str]]:
    """Build the (algorithm, model_key) -> Hydra overrides table."""
    data = load_tuned_defaults()
    table: Dict[Tuple[str, str], Dict[str, str]] = {}
    for model_key, model in data["models"].items():
        for algorithm, entry in model["methods"].items():
            overrides: Dict[str, str] = {}
            for key, value in (entry.get("params") or {}).items():
                overrides[f"anodet_algos.{key}"] = _override_value(value)
            # Sweep axes are emitted verbatim: "l1,l2" is a --multirun axis,
            # not a value, which is why run_anodet_analysis never writes them.
            for key, value in (entry.get("sweep_params") or {}).items():
                overrides[f"anodet_algos.{key}"] = str(value)
            if "sae_variant" in entry:
                variant = entry["sae_variant"]
                if variant is None:
                    overrides["anodet_algos.model_path"] = "null"
                elif variant == UNRELEASED_SAE:
                    overrides["anodet_algos.model_path"] = UNRELEASED_SAE
                else:
                    overrides["anodet_algos.model_path"] = sae_path(
                        model_key, variant
                    )
            table[(algorithm, model_key)] = overrides
    return table


BEST_HYPERPARAMETERS: Dict[Tuple[str, str], Dict[str, str]] = (
    _load_best_hyperparameters()
)

# Methods reported in the paper's main-text Table 1: the supervised BCE
# reference, the two framework instantiations, and the four activation-space
# baselines.
#
# `learned_mask_step1` is the LearnedMask-LoRA row. That row is stage-1-only on
# gpt_oss_20b (the paper disables the LoRA stage there) but stage 1 plus
# stage 2 on
# qwen3_8b and gemma4_26b. Stage 2 (`anodet_algos=learned_mask_lora`) is not in
# either list below because it is not a standalone fit: it wants
# `anodet_algos.learned_mask_lora.step1_run_dir=<a completed stage-1 run>`,
# which no orchestrator can invent. It DOES have tuned defaults -- run it with
#   run_baseline_experiments.py --best --algorithms learned_mask_lora \
#       --models gemma4_26b --override \
#       anodet_algos.learned_mask_lora.step1_run_dir=<dir>
PAPER_TABLE1_ALGORITHMS = [
    "bce",
    "cvdd_km_global_mask",
    "learned_mask_step1",
    "mahalanobis",
    "gmm",
    "one_class_svm",
    "deep_svdd",
    "cvdd",
]
# Table 2 (the decomposition table) adds the two SAE-space ablation rows.
PAPER_TABLE2_ALGORITHMS = PAPER_TABLE1_ALGORITHMS + [
    "cvdd_sae",
    "local_freqmask_km",
]

# The released models. --best iterates every (algorithm, model) combo; combos
# absent from BEST_HYPERPARAMETERS are skipped with a warning (see
# run_experiment), so listing a model here never errors -- it just runs the
# algorithms that have a tuned config for it.
ALL_MODELS = [
    "qwen3_8b",
    "gpt_oss_20b",
    "gemma4_26b",
]


def _resolve_hydra_config(algorithm: str) -> str:
    """Map virtual algorithm name to its Hydra config name."""
    return ALGORITHM_CONFIG_MAP.get(algorithm, algorithm)


def _get_seed_key(algorithm: str) -> str:
    """Get the config namespace that holds the seed for this algorithm."""
    return _resolve_hydra_config(algorithm)


def build_command(
    algorithm: str,
    model_key: str,
    encoder_path: Optional[str] = None,
    domain: Optional[str] = None,
    use_multirun: bool = True,
    use_best: bool = False,
    use_last_token: bool = False,
    use_generation: bool = False,
    use_complex_data: bool = False,
    seeds: Optional[str] = None,
    encoder_path_explicit: bool = False,
    extra_overrides: Optional[List[str]] = None,
) -> Optional[List[str]]:
    """Build the command to run an experiment.

    ``encoder_path_explicit`` marks ``encoder_path`` as coming from the
    ``--encoder-paths`` CLI flag rather than from the default resolver.
    An explicit path overrides the ``model_path`` baked into
    ``BEST_HYPERPARAMETERS``; see the comment in the body.
    """
    hydra_config = _resolve_hydra_config(algorithm)

    cmd = [
        "python",
        "-m",
        "unsupervised_llm_safety.anodet_algos.run_anodet_analysis",
        "exp_type=anodet_algos",
    ]

    # Add domain if specified
    if domain:
        cmd.append(f"domain={domain}")
        exp_ident_suffix = f"_{domain}"
    else:
        exp_ident_suffix = "_common_sense_safety"

    if use_last_token:
        exp_ident_suffix += "_last_token"
    if use_generation:
        exp_ident_suffix += "_generation"
    if use_complex_data:
        exp_ident_suffix += "_complex_data"

    # Use the virtual algorithm name in exp_ident for clarity
    cmd.extend(
        [
            f"exp_ident={algorithm}_{model_key}{exp_ident_suffix}",
            f"anodet_algos={hydra_config}",
            f"anodet_algos.model_key={model_key}",
            "anodet_algos.save_model=true",
        ]
    )

    # Name the *virtual* algorithm explicitly. Several virtual algorithms share
    # one Hydra config, so without this a direct run of `anodet_algos=
    # cvdd_kmeans_masked` and this orchestrator could disagree about which
    # published cell they mean.
    #
    # In sweep mode there is no published cell: the point is to explore the
    # grid from the config's own defaults, so the tuned layer is switched off
    # to keep sweep semantics exactly what they were.
    if use_best:
        if algorithm in _FILE_CONFIG_MAP:
            cmd.append(f"anodet_algos.tuned_method={algorithm}")
    else:
        cmd.append("anodet_algos.use_tuned_defaults=false")

    if use_last_token:
        cmd.append("anodet_algos.use_last_token=true")
    if use_generation:
        cmd.append("anodet_algos.use_generation=true")
    if use_complex_data:
        cmd.append("anodet_algos.use_complex_data=true")

    best_key = (algorithm, model_key)
    best_params = (
        dict(BEST_HYPERPARAMETERS[best_key])
        if use_best and best_key in BEST_HYPERPARAMETERS
        else {}
    )

    # An entry whose encoder was never archived cannot be turned into a
    # runnable command. Guessing a path here would produce a run that looks
    # healthy and is fitted against the wrong encoder.
    if best_params.get("anodet_algos.model_path") == UNRELEASED_SAE and not (
        encoder_path and encoder_path_explicit
    ):
        print(
            f"Warning: {algorithm}/{model_key} needs an SAE encoder that was "
            f"not released (see reference_scores/README.md). Pass "
            f"--encoder-paths to supply your own. Skipping."
        )
        return None

    # An encoder path given explicitly via --encoder-paths always wins over the
    # model_path baked into BEST_HYPERPARAMETERS. Without this, --encoder-paths
    # is silently ignored for every SAE-dependent method (they all carry their
    # own model_path), which is exactly how a rerun ends up fitting against a
    # stale encoder while still looking healthy.
    if encoder_path and encoder_path_explicit:
        best_params.pop("anodet_algos.model_path", None)
        cmd.append(f'anodet_algos.model_path="{encoder_path}"')
    elif "anodet_algos.model_path" not in best_params:
        # Add encoder path for algorithms that need it (sweep mode only).
        # In --best mode, model_path comes from BEST_HYPERPARAMETERS.
        if hydra_config in ALGORITHMS_WITH_ENCODERS:
            if encoder_path:
                cmd.append(f'anodet_algos.model_path="{encoder_path}"')
            else:
                print(
                    f"Warning: {algorithm} (config={hydra_config}) "
                    f"requires an SAE encoder but none could be resolved for "
                    f"{model_key}. Set $ULS_SAE_ROOT or pass --encoder-paths. "
                    f"Skipping."
                )
                return None

    # Add hyperparameters (best or sweep)
    if use_best:
        if best_key in BEST_HYPERPARAMETERS:
            for param, value in best_params.items():
                cmd.append(f"{param}={value}")
        else:
            print(
                f"Warning: No best hyperparameters for "
                f"{algorithm}/{model_key}. Skipping."
            )
            return None
    else:
        if (
            hydra_config in HYPERPARAMETER_SWEEPS
            and HYPERPARAMETER_SWEEPS[hydra_config]
        ):
            for param, values in HYPERPARAMETER_SWEEPS[hydra_config].items():
                cmd.append(f"{param}={values}")

    # Add seed sweep if specified
    if seeds:
        seed_key = _get_seed_key(algorithm)
        cmd.append(f"anodet_algos.{seed_key}.seed={seeds}")

    # Verbatim pass-through, last so it wins (e.g. launcher settings).
    for override in extra_overrides or []:
        cmd.append(override)

    if use_multirun:
        cmd.append("--multirun")

    return cmd


def run_experiment(
    algorithm: str,
    model_key: str,
    encoder_path: Optional[str] = None,
    domain: Optional[str] = None,
    use_multirun: bool = True,
    dry_run: bool = False,
    background: bool = False,
    use_best: bool = False,
    use_last_token: bool = False,
    use_generation: bool = False,
    use_complex_data: bool = False,
    seeds: Optional[str] = None,
    encoder_path_explicit: bool = False,
    extra_overrides: Optional[List[str]] = None,
    log_dir: Optional[Path] = None,
) -> bool:
    """Run a single experiment."""
    cmd = build_command(
        algorithm=algorithm,
        model_key=model_key,
        encoder_path=encoder_path,
        domain=domain,
        use_multirun=use_multirun,
        use_best=use_best,
        use_last_token=use_last_token,
        use_generation=use_generation,
        use_complex_data=use_complex_data,
        seeds=seeds,
        encoder_path_explicit=encoder_path_explicit,
        extra_overrides=extra_overrides,
    )

    if cmd is None:
        return False

    print(f"\n{'='*80}")
    print(f"Running: {algorithm} with model {model_key}")
    if seeds:
        print(f"Seeds: {seeds}")
    if domain:
        print(f"Domain: {domain}")
    if encoder_path:
        print(f"Encoder: {encoder_path}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    if dry_run:
        print("DRY RUN - Command not executed")
        return True

    try:
        if background:
            out_dir = (
                log_dir
                or Path(os.environ.get("ULS_OUTPUT_ROOT", "outputs"))
                / "background_logs"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            log_file = out_dir / f"{algorithm}_{model_key}.log"
            fh = open(log_file, "w")
            subprocess.Popen(cmd, stdout=fh, stderr=fh)
            print(
                f"Started background job for {algorithm} with {model_key}"
                f" (log: {log_file})"
            )
        else:
            subprocess.run(cmd, check=True)
            print(f"Completed: {algorithm} with {model_key}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {algorithm} with {model_key}: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\nInterrupted: {algorithm} with {model_key}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline anomaly detection experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Every paper method on every released model, printed not run
  python scripts/anodet_algos/run_baseline_experiments.py --best --dry-run

  # Table 1 methods only, five seeds
  python scripts/anodet_algos/run_baseline_experiments.py \\
      --best --table1-only --seeds 42,43,44,45,46 --dry-run

  # Hyperparameter sweep mode
  python scripts/anodet_algos/run_baseline_experiments.py \\
      --algorithms mahalanobis,gmm --models qwen3_8b

  # On your own SLURM, with hydra-submitit-launcher installed
  python scripts/anodet_algos/run_baseline_experiments.py --best \\
      --override hydra/launcher=submitit_slurm \\
      --override hydra.launcher.mem_per_cpu=192G
        """,
    )

    parser.add_argument(
        "--best",
        action="store_true",
        help="Run with the paper's published hyperparameters. "
        "Defaults to all paper methods and all released models.",
    )
    parser.add_argument(
        "--table1-only",
        action="store_true",
        help="When using --best, only run Table 1 methods "
        "(excludes cvdd_sae and local_freqmask_km).",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default=None,
        help="Comma-separated list of algorithms to run.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=f"Comma-separated models (released: {','.join(ALL_MODELS)}).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds for multi-seed runs "
        "(e.g., 42,43,44,45,46). Becomes a Hydra sweep axis.",
    )
    parser.add_argument(
        "--encoder-paths",
        type=str,
        default=None,
        help="Comma-separated SAE checkpoint paths, positionally matched to "
        "--models. Overrides both the resolver and any model_path pinned in "
        "the published hyperparameters.",
    )
    parser.add_argument(
        "--override",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra Hydra override appended verbatim; repeatable. Use this "
        "for launcher settings, which are not shipped (see module docstring).",
    )
    parser.add_argument("--no-multirun", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--background", action="store_true")
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--last-token", action="store_true")
    parser.add_argument("--generation", action="store_true")
    parser.add_argument("--complex-data", action="store_true")

    args = parser.parse_args()

    # Determine algorithm list
    if args.algorithms:
        algorithms = [a.strip() for a in args.algorithms.split(",")]
    elif args.best:
        if args.table1_only:
            algorithms = PAPER_TABLE1_ALGORITHMS
        else:
            algorithms = PAPER_TABLE2_ALGORITHMS
    else:
        # Default algorithm set for sweep mode.
        algorithms = [
            "one_class_svm",
            "mahalanobis",
            "gmm",
            "deep_svdd",
            "cvdd",
            "cvdd_kmeans_masked",
        ]

    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    elif args.best:
        models = ALL_MODELS
    else:
        models = []

    if not args.best and (args.algorithms is None or args.models is None):
        parser.error(
            "--algorithms and --models are required unless --best is used"
        )

    # When --best is used, force complex-data + last-token mode
    # (all paper results use this setup)
    if args.best:
        args.complex_data = True
        args.last_token = True

    # Encoder paths. Paths that came from --encoder-paths are tracked in
    # `explicit_models` and override the published model_path; paths that fell
    # back to the resolver do not.
    explicit_models = set()
    encoder_paths: Dict[str, Optional[str]] = {}
    provided_paths = (
        [p.strip() for p in args.encoder_paths.split(",")]
        if args.encoder_paths
        else []
    )
    for i, model_key in enumerate(models):
        if i < len(provided_paths):
            encoder_paths[model_key] = provided_paths[i]
            explicit_models.add(model_key)
        else:
            variant = DEFAULT_SAE_VARIANT.get(model_key)
            encoder_paths[model_key] = (
                sae_path(model_key, variant) if variant else None
            )

    # Run from the repository root so that `python -m
    # unsupervised_llm_safety...` resolves exp_configs/ and scripts/.
    os.chdir(_REPO_ROOT)

    print(f"Working directory: {os.getcwd()}")
    print(f"SAE root: {sae_root()}")
    print(f"Algorithms: {algorithms}")
    print(f"Models: {models}")
    print(f"Seeds: {args.seeds or 'default (42)'}")
    print(f"Best mode: {args.best}")
    print(f"Last-token: {args.last_token}, Complex-data: {args.complex_data}")
    print(f"Dry run: {args.dry_run}")

    n_seeds = len(args.seeds.split(",")) if args.seeds else 1
    total_experiments = len(algorithms) * len(models)
    total_jobs = total_experiments * n_seeds
    print(
        f"Total: {total_experiments} combos x {n_seeds} seeds = "
        f"{total_jobs} runs"
    )

    completed = 0
    failed = 0

    for algorithm in algorithms:
        for model_key in models:
            success = run_experiment(
                algorithm=algorithm,
                model_key=model_key,
                encoder_path=encoder_paths.get(model_key),
                domain=args.domain,
                use_multirun=not args.no_multirun,
                dry_run=args.dry_run,
                background=args.background,
                use_best=args.best,
                use_last_token=args.last_token,
                use_generation=args.generation,
                use_complex_data=args.complex_data,
                seeds=args.seeds,
                encoder_path_explicit=model_key in explicit_models,
                extra_overrides=args.overrides,
            )

            if success:
                completed += 1
            else:
                failed += 1

    print(f"\n{'='*80}")
    print(f"Summary: {completed}/{total_experiments} experiments submitted")
    if failed > 0:
        print(f"Skipped/failed: {failed} experiments")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
