# Getting started

This is the manual for running the code, in the order you would actually run
it: install, check the install, extract activations, register the datasets and
models you extracted, train a sparse autoencoder, fit a detector, sweep, and
evaluate a saved checkpoint.

Run every command from the repository root. One of the entry points depends on
that (see below), and the sweep driver `chdir`s there itself.

- [1. Install](#1-install)
- [2. Check the install against the released scores](#2-check-the-install-against-the-released-scores)
- [3. Extract activations](#3-extract-activations)
- [4. Register a dataset](#4-register-a-dataset)
- [5. Register a model](#5-register-a-model)
- [6. Train an SAE](#6-train-an-sae)
- [7. Run a detector](#7-run-a-detector)
- [8. Run a sweep](#8-run-a-sweep)
- [9. Evaluate a saved checkpoint](#9-evaluate-a-saved-checkpoint)
- [10. Where output goes](#10-where-output-goes)
- [11. Environment variables](#11-environment-variables)
- [12. What you supply](#12-what-you-supply)

## 1. Install

```bash
conda create -n unsupervised-llm-safety python=3.11 -y
conda activate unsupervised-llm-safety
pip install -e ".[all]"
```

The `-e` is required, not a convenience. `run_anodet_analysis.py`,
`run_anodet_eval.py` and `train_encoder.py` all declare
`config_path="../../../exp_configs"` in their `@hydra.main` decorator, and Hydra
resolves that relative to the module file. A non-editable install puts the
module under `site-packages`, where three directories up is not the repository,
so Hydra finds no configs at all. `extract_activations.py` is the exception: it
uses `config_path=f"{os.getcwd()}/exp_configs"`, so it needs the *current
directory* to be the repository root. An editable install plus running from the
root satisfies both.

Python 3.9 or newer. The extras, from `pyproject.toml`:

| extra | adds | you need it for |
|---|---|---|
| (base) | numpy, scikit-learn | `reference_scores/compute_metrics.py` |
| `checkpoints` | torch, omegaconf, hydra-core, faiss-cpu, scipy, tqdm, matplotlib | training SAEs, fitting and evaluating detectors, loading saved checkpoints |
| `extract` | `checkpoints` plus transformers, datasets, accelerate, psutil | activation extraction |
| `all` | `checkpoints` + `extract` | the whole pipeline |
| `dev` | pytest, black, isort, flake8 | tests and formatting |

`faiss` is imported under `try`/`except`, so the package imports without it, but
fitting FreqMask-KM or LearnedMask does not. Substitute `faiss-gpu` for
`faiss-cpu` if you want the k-means on the GPU.

## 2. Check the install against the released scores

```bash
python reference_scores/compute_metrics.py
```

No arguments needed. `--scores-dir` defaults to `reference_scores/`, which
ships `reference_scores/scores/` — 36 files of per-sample anomaly scores
committed to the repository, 23 MB in total. The script recomputes the twelve
reported cells from them and compares each against
`reference_scores/expected/table1.csv` at a tolerance of 5e-4.

Expect `12 cell(s) within tolerance, 0 outside`, then `RESULT: PASS`, exit
code 0. It needs numpy and scikit-learn only — no GPU, no `torch`, no network —
so it is the cheapest check that the install works. If it does not print that,
fix the install before going further.

This is a reference calculation, not a reproduction of the method. It confirms
the reported numbers follow from the released scores. It cannot tell you whether
your own implementation matches ours, because the representations those scores
came from are not released. See `reference_scores/README.md`.

## 3. Extract activations

Entry point:

```
src/unsupervised_llm_safety/scripts/activations/extract_activations.py
```

It is a Hydra application over `exp_configs/config.yaml`. The `analysis` config
group is optional in the base defaults list, so `analysis=extract_activations_config`
is mandatory on the command line; without it the script has no config to read
and fails immediately. Everything the script reads lives under that group, so
every override is prefixed `analysis.`.

`exp_type` and `exp_ident` are interpolated into `hydra.run.dir`, so set both or
Hydra cannot build the run directory.

### The keys you have to set

| key | meaning |
|---|---|
| `analysis.model_key` | key into `MODEL_CONFIGS` in `models/model_registry.py`; supplies `hf_name`, `dtype`, `trust_remote_code` and the fallback layer |
| `analysis.dataset.name` | the literal HuggingFace repo id, e.g. `toxigen/toxigen-data`. This same string selects the processor (section 4) |
| `analysis.dataset.subset` | passed to `load_dataset` as `name=` |
| `analysis.dataset.split` | `train` / `test` / `validation`; if the requested split is missing the loader warns and falls back to the first of those that exists |
| `analysis.dataset.max_samples` | truncation after processing. `-1` (anything `<= 0`) means all samples |
| `analysis.dataset.text_column` | only read by `GenericProcessor`; a dedicated processor decides its own text |
| `analysis.extraction.layer_index` | the hook is registered on `model.layers.<layer_index>`. If null, the registry's `default_layer` is used |
| `analysis.extraction.output_path` | the `.pth` file that gets written. Nothing derives this from the dataset registry — you type it |
| `analysis.extraction.save_metadata` | pinned to `true`; `false` writes a bare tensor instead of the `{"train": ..., "test": ...}` dict every downstream loader expects, so nothing would read the result |
| `analysis.memory.initial_batch_size` | the batch size. On CUDA OOM the loop halves it (`memory.reduce_factor`) down to `memory.min_batch_size` and retries the same batch |

Also pass `hydra.job.chdir=False`. The base config sets `hydra.job.chdir: True`,
which moves the process into the run directory, and a relative `output_path`
would then land inside it rather than in your activation tree.

`seed` is read but changes nothing here: the train/test split and the
max-length sampler both reseed to 42 themselves, and the forward pass is
deterministic.

### A working example

`scripts/extract_activations/by_dataset/toxigen.sh` extracts ToxiGen for the
three models the paper reports. One invocation from it, with the loop unrolled:

```bash
python src/unsupervised_llm_safety/scripts/activations/extract_activations.py \
    exp_type=extract_activations \
    exp_ident=qwen3_8b-toxigen \
    analysis=extract_activations_config \
    analysis.model_key=qwen3_8b \
    analysis.dataset.name=toxigen/toxigen-data \
    analysis.dataset.subset=train \
    analysis.dataset.split=train \
    analysis.dataset.max_samples=-1 \
    analysis.extraction.layer_index=30 \
    analysis.extraction.output_path="${ULS_ACTIVATION_ROOT}/toxigen/toxigen_qwen3_8b_layer30_full.pth" \
    analysis.extraction.save_metadata=true \
    analysis.memory.initial_batch_size=8 \
    hydra.job.chdir=False
```

ToxiGen requires an accepted licence on HuggingFace, so `huggingface-cli login`
and accepting the terms on the dataset page are prerequisites. A 20-26B model
in bf16 needs a GPU with room for it; `analysis.model.device_map` defaults to
`auto`, which shards across whatever devices are visible.

### What the file contains

The script saves a dict with `train` and `test` keys, each holding
`hidden_states`, `labels` and `original_sentences`. The split is 80/20 with
`np.random.seed(42)`, unless the processor returned metadata containing
`preserve_splits` and `split_labels`, in which case the upstream split is kept.

### Prompt representations versus generation representations

The default path is prompt-side. The text goes through
`tokenizer.apply_chat_template(..., add_generation_prompt=True)`, one forward
pass runs, and the hook takes `activations[:, -1, :]` — the residual stream at
the last prompt token. Every published number uses this path.

Setting `analysis.extraction.extract_per_token=true` switches to
`extract_activations_with_generation.py`, which calls `model.generate` and
captures the residual stream at every generated token. That path:

- inserts `_generation` into the output filename if it is not already there,
  and writes a `.metadata.json` next to the tensor;
- reads `analysis.extraction.max_new_tokens` (default 512) and
  `analysis.extraction.generation.temperature` / `.do_sample`. The shipped
  `extract_activations_config.yaml` has no `extraction.generation` block, so add
  one on the command line, e.g.
  `+analysis.extraction.generation.temperature=0.0`;
- maps `analysis.dataset.name` to one of `humaneval`, `humaneval_plus`, `mbpp`,
  `gsm8k`, `math_sft_40k`, `alpaca`, and records `unknown` for anything else.

Downstream, per-token files are only reached when `use_generation=true`, and the
registry currently swaps only HumanEval to its generation file
(`GENERATION_DATASETS` in `data/dataset_registry.py`).

## 4. Register a dataset

Two files, and both need an entry.

### 4a. A processor, in `src/unsupervised_llm_safety/data/dataset_processors.py`

`DATASET_PROCESSORS` maps the **literal HuggingFace repo id** — the exact string
you pass as `analysis.dataset.name` — to a `DatasetProcessor` subclass:

```python
DATASET_PROCESSORS: Dict[str, Type[DatasetProcessor]] = {
    "toxigen/toxigen-data": ToxigenProcessor,
    "openai/gsm8k": GSM8KProcessor,
    ...
}
```

A subclass implements one abstract method:

```python
def load_and_process(self) -> Tuple[List[str], List[int], Dict[str, Any]]:
```

returning the texts, one integer label per text, and a metadata dict (`{}` is
fine). The base class gives you `self._load_dataset()`, which calls
`load_dataset(path=cfg.dataset.name, name=cfg.dataset.subset)` and picks the
split, and `self._limit_samples(texts, labels)`, which applies `max_samples`.

**Label convention: 1 is safe, 0 is unsafe.** This is not cosmetic.
`hidden_states_loader` filters `labels == 1` for `safe_only` slices and
`labels == 0` for `unsafe_only`, and the complex-data machinery in the registry
relies on it. `ToxigenProcessor` writes `1 - prompt_label` for exactly this
reason.

If your dataset already carries a train/test split you want preserved, return
metadata with `preserve_splits: True` and a `split_labels` list of `"train"` /
`"test"` strings, one per sample.

**Trap.** A dataset id that is not a key in `DATASET_PROCESSORS` does not error.
`create_dataset_processor` falls back to `GenericProcessor`, which reads
`cfg.dataset.text_column` and, when the dataset has no `label` column, labels
*every* sample with `dataset.default_label`, which is **0 — unsafe**, the
opposite of what a benign IID corpus should be. The only signal is a `log.info`
saying "Using GenericProcessor" and a warning about the default label. Check the
log line after the first extraction of a new dataset.

### 4b. Paths, in `src/unsupervised_llm_safety/data/dataset_registry.py`

Every path in this module is an f-string rooted on `ROOT_DATA_PATH`
(`$ULS_ACTIVATION_ROOT`). Add your file where it belongs:

| dict | when |
|---|---|
| `DATASET_PATHS[model_key]["iid"]` | in-distribution training data |
| `DATASET_PATHS[model_key]["ood"]` | evaluation slices; entries take an optional `"filter": "unsafe_only"` |
| `COMPLEX_DATA_BEAVER_TAILS`, `COMPLEX_DATA_TOXIGEN` | the safe half of an OOD corpus, folded into IID when `use_complex_data=true` |
| `BENIGN_GENERALIZATION_PATHS`, `BENIGN_OOD_PATHS` | benign-OOD calibration slices |
| `LAST_TOKEN_MTBENCH_PATHS` | MT-Bench only; swapped in when `use_last_token=true` |
| `GENERATION_DATASETS` | datasets whose `_generation.pth` file should be used when `use_generation=true` |

An entry is `{"path": ..., "name": ..., "filter": ...}`, where `name` is the
label that appears in results and in `analysis/anomaly_metrics.py`'s slice
categories, and `filter` is optional.

HarmBench adversarial files are not listed here. They are discovered by glob
under `$ULS_HARMBENCH_BASE` using the pattern
`*_<harmbench_name>_l*_generation_last_token.pth`, filtered to the active layer.

**Trap.** The output filename is written twice and compared never: once as
`analysis.extraction.output_path` on the extraction command line, once as an
f-string here. Nothing validates that they agree. A mismatch surfaces as a
missing-file error when a detector loads its data, which is the good case; the
bad case is that a stale file with the matching name is there and gets used.
Copy the path from the registry into the extraction command rather than typing
it twice.

## 5. Register a model

`MODEL_CONFIGS` in `src/unsupervised_llm_safety/models/model_registry.py`:

```python
"qwen3_8b": {
    "hf_name": "Qwen/Qwen3-8B",
    "short_name": "qwen3_8b",
    "harmbench_name": "qwen3_8b",
    "default_layer": 30,
    "trust_remote_code": True,
    "dtype": "float16",
},
```

`dtype` is one of `float16`, `float32`, `bfloat16`; anything else silently maps
to `float16`. `short_name` is written into the extraction metadata.
`harmbench_name` is what adversarial-file discovery greps for. Two further
fields exist on `gpt_oss_20b` — `chat_template_kwargs` and `assistant_prefill` —
which record the prompt format the paper's `gpt_oss_20b` representations were
extracted with. Nothing in this repository reads them; the extractor here
applies a plain chat template, so it does not reproduce that prompt format.

Then, in lockstep, in `data/dataset_registry.py`:

- `_CANONICAL_LAYER[model_key]` — the layer baked into this model's path
  f-strings, and what the layer-override machinery rewrites away from.
- `_MODEL_LAYERS[model_key]` — same number, separate dict, used to build
  generation paths.
- `DATASET_PATHS[model_key]` — the `iid` and `ood` lists, plus
  `adversarial_base_path`.
- `COMPLEX_DATA_BEAVER_TAILS[model_key]` and `COMPLEX_DATA_TOXIGEN[model_key]` —
  needed for `use_complex_data=true`, which is what every published run uses.
- `LAST_TOKEN_MTBENCH_PATHS[model_key]` — needed for `use_last_token=true`,
  likewise.
- `BENIGN_GENERALIZATION_PATHS` / `BENIGN_OOD_PATHS` if you want those analyses.

And in `scripts/anodet_algos/run_baseline_experiments.py` if you want the sweep
driver to cover it: `ALL_MODELS`, `DEFAULT_SAE_VARIANT`, and an entry per
method in `exp_configs/best_hyperparameters.yaml`.

**Trap.** The layer number appears in at least three places per model —
`default_layer`, `_CANONICAL_LAYER`, `_MODEL_LAYERS` — and again literally
inside every path f-string. Nothing checks that they agree.
`get_dataset_paths` raises on an unknown `model_key`, but a model registered
with inconsistent layers resolves to paths that do not exist, or worse, to
paths that do.

## 6. Train an SAE

The SAE-space methods (FreqMask-KM, LearnedMask, LearnedMask-LoRA, BCE, and
CVDD when given an encoder) need a frozen encoder. No encoders ship with this
code, so this step comes before any of those methods. The path resolver looks
for `$ULS_SAE_ROOT/<model_key>__<variant>.pth`, so putting the encoder you train
there is what makes the sweep driver find it.

Entry point `src/unsupervised_llm_safety/scripts/train_encoder.py`, working
example `scripts/iid/train_sae.sh`:

```bash
python src/unsupervised_llm_safety/scripts/train_encoder.py \
    iid=encoder_training_config \
    model@iid.model=autoencoder \
    iid.model.sparsity_weight=0.01,0.1 \
    iid.model.feature_dim=16384 \
    iid.model_key=qwen3_8b \
    iid.num_epochs=5,10,20 \
    iid.batch_size=256 \
    exp_ident=sae_qwen3_8b_complex_data \
    exp_type=iid \
    use_last_token=true \
    use_complex_data=true \
    --multirun
```

`model@iid.model=autoencoder` is load-bearing. `train_encoder.py` reads its
model config from `iid.model`, and the `- /model: autoencoder` default inside
`encoder_training_config.yaml` populates only the top-level `model` node.

The values that produced the paper's encoders are the CLI overrides above, not
the YAML defaults. In particular `exp_configs/model/autoencoder.yaml` has
`feature_dim: 128`; the paper's encoders are `feature_dim=16384`. Train at the
default and you get a 128-unit bottleneck, which is not an SAE in any useful
sense and will not match anything.

`sparsity_weight` is the one parameter that does not transfer between models —
it trades reconstruction against L0, and activation scale differs per model. The
paper's encoders landed at 0.01 for `qwen3_8b`, 3 for `gemma4_26b`, and 300 and
700 for `gpt_oss_20b` (whose activation RMS is roughly 34x the 8B models').
Widen the comma-separated grid accordingly.

Commas make Hydra sweep axes, so the command above is a 2x3 grid, and
`--multirun` is required for it. `exp_configs/config.yaml` ships Hydra's basic
launcher, so the grid runs sequentially in this process. For a real sweep,
install `hydra-submitit-launcher` and add your own launcher block; there is a
template at the bottom of `exp_configs/config.yaml`.

The command above is the whole set of keys worth touching. The rest of
`encoder_training_config.yaml` is left at its defaults. Dataset selection comes
from `model_key` through `dataset_registry`, and the loss is fixed at MSE
reconstruction plus L1 sparsity inside `Autoencoder.forward`. The live keys not
shown above are `iid.seed`, `iid.datasets.iid.train_ratio` and
`iid.training.train_split_ratio` (the 80/20 train/holdout split), all left at
their defaults for every encoder the paper trained.

Each grid point writes `trained_model.pth` and `results.json` into its own run
directory. Pick the point with the best reconstruction at your target L0, then
either pass its `trained_model.pth` as `anodet_algos.model_path=`, or copy it to
`$ULS_SAE_ROOT/<model_key>__<variant>.pth` so the sweep driver's resolver finds
it.

## 7. Run a detector

```bash
python -m unsupervised_llm_safety.anodet_algos.run_anodet_analysis \
    exp_type=anodet_algos \
    exp_ident=cvdd_kmeans_masked_qwen3_8b_seed42 \
    anodet_algos=cvdd_kmeans_masked \
    anodet_algos.model_key=qwen3_8b \
    anodet_algos.model_path="${ULS_SAE_ROOT}/qwen3_8b__both.pth" \
    anodet_algos.save_model=true \
    anodet_algos.use_last_token=true \
    anodet_algos.use_complex_data=true \
    anodet_algos.cvdd_kmeans_masked.seed=42
```

`scripts/anodet_algos/run_one_detector.sh` is the same thing parameterised by
method and model.

**Every algorithm override carries the `anodet_algos.` prefix.** The whole
method config is loaded as the `anodet_algos` config group, so shared keys are
`anodet_algos.<key>` and method-specific ones are nested one level deeper,
`anodet_algos.<method>.<key>` — for example
`anodet_algos.cvdd_kmeans_masked.n_clusters=50`. Dropping the prefix produces a
Hydra error, or, with a `+`, a silently ignored key.

`anodet_algos.model_key` is `???` in every config, so it must be set.
`use_last_token=true use_complex_data=true` is the setting behind every
published number; changing either changes which rows the registry loads.

The example above is the whole command. Do not set
hyperparameters by hand: `run_anodet_analysis` reads the published value for
this `(model_key, tuned_method)` out of `exp_configs/best_hyperparameters.yaml`
before it fits anything, and logs each one it applied with a `[tuned-defaults]`
prefix. An explicit override on the command line still wins, which is how you
deviate deliberately; `anodet_algos.use_tuned_defaults=false` drops the whole
layer and gives you the bare YAML defaults, which are not the published
configuration.

Two keys look settable and are not. `adversarial_base_path` and the `datasets:`
block ship as `???` placeholders and are overwritten from `dataset_registry`
on every run, so writing a path into either of them has no effect.
`use_centralized_datasets=false` is what suppresses the overwrite, and it
leaves the `???` in place, so it raises unless you also supply the whole
`datasets:` block yourself. Move data with `$ULS_ACTIVATION_ROOT` and the
registry instead.

### The nine methods

| method | config name | space | needs `model_path` |
|---|---|---|---|
| FreqMask-KM | `cvdd_kmeans_masked` | SAE features | yes |
| LearnedMask | `learned_mask` | SAE features | yes |
| LearnedMask-LoRA | `learned_mask_lora` | SAE features | yes |
| BCE (supervised reference) | `bce` | SAE features | yes |
| CVDD | `cvdd` | activations | no |
| Deep SVDD | `deep_svdd` | activations | no |
| Gaussian mixture | `gmm` | activations | no |
| Mahalanobis | `mahalanobis` | activations | no |
| One-class SVM | `one_class_svm` | activations | no |

The raw-activation methods never read the encoder; pass
`anodet_algos.model_path=null` and the load is skipped entirely.

`gpt_oss_20b` needs two encoders, one per method family, and the tuned
configurations expect them under these names:
`gpt_oss_20b__freqmask_km.pth` (sparsity weight 700) for the FreqMask-KM and
CVDD-SAE family, `gpt_oss_20b__learnedmask.pth` (weight 300) for LearnedMask and
BCE. `gemma4_26b` and `qwen3_8b` use one encoder each, named
`<model_key>__both.pth`.

### LearnedMask-LoRA is two-stage

`anodet_algos=learned_mask_lora` inherits the whole `learned_mask` config
through its defaults list, so stage-1 parameters are set under
`anodet_algos.learned_mask.*` and LoRA parameters under
`anodet_algos.learned_mask_lora.*`. Run on its own it refits stage 1 first. To
reuse a completed stage-1 run instead, which is what the paper does, point it at
that run directory:

```bash
anodet_algos.learned_mask_lora.step1_run_dir=<stage-1 run dir>
```

## 8. Run a sweep

`scripts/anodet_algos/run_baseline_experiments.py` holds the published
hyperparameters per (method, model) and turns each into one
`run_anodet_analysis` command line. It does no numerical work itself.

Start with the dry run. It prints the exact override list for every published
configuration without executing anything, which is both the fastest way to see
what a cell of the table actually was and a check that `$ULS_SAE_ROOT` resolves
to real files — the printed `model_path` is the path that will be opened.

```bash
python scripts/anodet_algos/run_baseline_experiments.py --best --dry-run
```

Then drop `--dry-run` to run it, or narrow it first:

```bash
python scripts/anodet_algos/run_baseline_experiments.py \
    --best --algorithms cvdd_km_global_mask --models qwen3_8b --seeds 42,43,44,45,46
```

`scripts/anodet_algos/run_paper_sweep.sh` wraps the full five-seed sweep and
defaults to `--dry-run`; pass `--run` to execute.

Useful flags:

| flag | effect |
|---|---|
| `--best` | use the published hyperparameters; forces `--last-token` and `--complex-data` |
| `--table1-only` | with `--best`, the main-text methods only |
| `--algorithms`, `--models` | comma-separated; required unless `--best` |
| `--seeds` | comma-separated; becomes a Hydra sweep axis |
| `--encoder-paths` | comma-separated SAE paths, matched positionally to `--models`, overriding the resolver |
| `--override KEY=VALUE` | appended verbatim to the Hydra command line; repeatable. This is how you pass launcher settings |
| `--dry-run` | print the commands, run nothing |
| `--background` | detach each run, logging to `$ULS_OUTPUT_ROOT/background_logs/` |

Algorithm names here are partly virtual: `cvdd_km_global_mask`,
`cvdd_km_no_mask`, `cvdd_km_raw_no_mask` and `local_freqmask_km` all map to the
`cvdd_kmeans_masked` config with different overrides, `cvdd_sae` maps to `cvdd`,
and `learned_mask_step1` maps to `learned_mask`. Combinations with no tuned
configuration are skipped with a warning rather than run at defaults. `--best`
iterates the three models the paper's main tables report;
`exp_configs/best_hyperparameters.yaml` also carries tuned values for three
further models whose SAE encoders were never archived, and those SAE-space
combinations are skipped unless you pass `--encoder-paths`. The LoRA stage is
not in `--best`; run it through `run_anodet_analysis` directly as in section 7.

Without a launcher plugin the runs go sequentially in one process. For scale, a
single fit took about 8 minutes for FreqMask-KM and between 45 minutes and
1.5 hours for LearnedMask stage 1.

## 9. Evaluate a saved checkpoint

To score a fitted detector without refitting it:

```bash
python -m unsupervised_llm_safety.anodet_algos.run_anodet_eval \
    exp_type=anodet_algos \
    exp_ident=eval_gmm_qwen3_8b \
    anodet_algos=gmm \
    +anodet_algos.load_from=/path/to/gmm_trained.pt \
    anodet_algos.model_key=qwen3_8b
```

The `+` is required. `load_from` is declared in no config file, so Hydra
refuses a plain `anodet_algos.load_from=` with `Key 'load_from' is not in
struct` before the script starts. The key itself is mandatory for this entry
point: without it the script exits.

The config group you pass must match the checkpoint — `load` validates the
stored method name against the class and refuses a mismatch.

If the detector was fitted on top of an SAE, pass the same
`anodet_algos.model_path=` you fitted with, so the encoder wrapper is rebuilt.

`run_anodet_analysis` accepts `+anodet_algos.load_from` too, and takes the same
load-instead-of-fit branch, so the separate eval entry point matters mainly when
you want a clean eval-only run directory.

## 10. Where output goes

`exp_configs/config.yaml` sets:

```yaml
hydra:
  job:
    chdir: True
  run:
    dir: ${oc.env:ULS_OUTPUT_ROOT,outputs}/${exp_type}/${now:%Y-%m-%d}/${exp_ident}-${now:%H-%M-%S}
  sweep:
    dir: ${oc.env:ULS_OUTPUT_ROOT,outputs}/multirun/${exp_type}/${now:%Y-%m-%d}/${exp_ident}-${now:%H-%M-%S}
```

`chdir: True` means the process runs *inside* its run directory, so everything
each stage writes with a relative path lands there. That is why `exp_type` and
`exp_ident` are not decoration: they name the directory. Under `--multirun`,
each grid point gets a numbered subdirectory of the sweep directory.

What each stage writes into that directory:

| stage | files |
|---|---|
| extraction | whatever `analysis.extraction.output_path` says (pass `hydra.job.chdir=False` so a relative path resolves from the repo root), plus `.metadata.json` on the per-token path |
| `train_encoder.py` | `trained_model.pth`, `results.json` |
| `run_anodet_analysis.py` fit | `<name>_trained.pt` when `save_model=true`, and `cluster_info.pt` for the clustering methods |
| evaluation (both entry points) | `<name>_analysis_results.json`, `<name>_scores.npz`, `<name>_categories.json`, and optionally `<name>_cluster_assignments.npz` and `<name>_raw_scores.npz` |
| LearnedMask diagnostics | a `diagnostics/` directory and `diagnostic_summary.json` |

`<name>` is the detector's own `name` property, which is not always the config
name: `cvdd_kmeans_masked` subclasses CVDD and inherits `name == "cvdd"`, so
FreqMask-KM results are written as `cvdd_scores.npz`. This is why
`reference_scores/scores/<model>/freqmask_km/seed0/` contains `cvdd_scores.npz`.

## 11. Environment variables

All are prefixed `ULS_`, omitted in the table below. Every one has a working
default derived from the repository's own location. None are needed for
`reference_scores/compute_metrics.py`; they matter once you extract, train or
refit.

| variable | what it moves | default |
|---|---|---|
| `ACTIVATION_ROOT` | the root every path in `dataset_registry` is built from; set this one and the whole registry follows | `<repo>/activation_data` |
| `HARMBENCH_BASE` | the HarmBench adversarial slice, and the HarmBench-safe rows `use_complex_data` folds into IID | `$ULS_ACTIVATION_ROOT/harmbench/` |
| `BEAVERTAILS_HS_ROOT` | BeaverTails hidden states laid out `<root>/<model-slug>/all_hidden_states*.pth` | `$ULS_ACTIVATION_ROOT/beaver_tails_hs` |
| `BEAVERTAILS_COMPLEX_OVERRIDE` | the BeaverTails-safe IID slice only; `model_key:/abs/path`, comma-separated | unset |
| `SAE_ROOT` | the frozen SAE encoders | `$ULS_ACTIVATION_ROOT/saes` |
| `SAE_<MODEL_KEY>` | one model's SAE, overriding `ULS_SAE_ROOT` | unset |
| `SAE_<MODEL_KEY>_<VARIANT>` | one model's SAE for one variant; highest precedence | unset |
| `SAE_ENCODED` | the offline SAE-encoding cache written by `precompute_sae_encodings` | `$ULS_ACTIVATION_ROOT/sae_encoded` |
| `CLUSTERING_CACHE` | the offline k-means cache written by `precompute_clustering`; can reach a few GB | `$ULS_ACTIVATION_ROOT/clustering_cache` |
| `LAYER_OVERRIDE` | route a model's registry paths to a non-canonical layer; `model_key:layer`, comma-separated | unset (canonical layer) |
| `OUTPUT_ROOT` | where Hydra creates run and sweep directories | `./outputs` |
| `TUNED_DEFAULTS` | an alternative hyperparameter file, in place of `exp_configs/best_hyperparameters.yaml` | unset |

`ULS_SAE_<MODEL_KEY>` is name-mangled: take the model key, replace `.` with `_`,
uppercase it. So `qwen3_8b` becomes `ULS_SAE_QWEN3_8B` and `qwen2_1.5b` becomes
`ULS_SAE_QWEN2_1_5B`. The variant form appends the uppercased variant, e.g.
`ULS_SAE_GPT_OSS_20B_FREQMASK_KM`. Resolution order is
`ULS_SAE_<MODEL_KEY>_<VARIANT>`, then `ULS_SAE_<MODEL_KEY>`, then
`$ULS_SAE_ROOT/<model_key>__<variant>.pth`.

Because the default path carries the variant (`<model>__<variant>.pth`) and
`gpt_oss_20b` needs two encoders, a single `ULS_SAE_GPT_OSS_20B` cannot be right
for both of its method families — either leave it unset and let the
variant-specific default path resolve, or use the per-variant form.

## 12. What you supply

This repository ships the code and the per-sample scores under
`reference_scores/`. Everything else you generate yourself, following sections
3 to 7: extracted representations, trained SAE encoders and fitted detectors.
`reference_scores/compute_metrics.py` needs none of them.

Three things worth knowing before you start:

- **BeaverTails extraction is not implemented here.** `DATASET_PROCESSORS` has
  21 entries and BeaverTails is not among them, so you write that pass
  yourself: the residual stream at the model's layer, last token, over the
  corpus. Put the result where `data/dataset_registry.py` expects it
  (section 4b).
- **`qwen2_1.5b`, `ministral_8b` and `llama3_8b` have no SAE encoder here.**
  Their tuned entries carry `sae_variant: unreleased`, so the sweep driver
  skips the SAE-space methods for those models with a warning rather than
  guessing a path. Pass `--encoder-paths` to point it at your own.
- **No `exp_configs/dataset/` group and no SLURM launcher.** The extraction
  config inlines its `dataset` block and you override it on the command line.
  `exp_configs/config.yaml` ships Hydra's basic launcher, so `--multirun` runs
  sequentially on the local machine; the comment at the bottom of that file is
  a template for adding `hydra-submitit-launcher` with your own account and
  partition.

Tests, for what they cover:

```bash
pip install -e ".[dev]"
pytest
```

Formatting is black and isort at line length 79, configured in `pyproject.toml`.
