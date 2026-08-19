"""Tests for the published-hyperparameter defaults.

The property under test is the precedence order

    explicit CLI override  >  exp_configs/best_hyperparameters.yaml  >  the
    method YAML default

and the thing that makes those tests worth anything is that each one has a
control that pins down the *other* branch. Asserting "the CLI value came out"
proves nothing on its own if the tuned value and the method default happen to
agree; every precedence test below therefore also asserts what comes out when
the override is removed, and asserts the two differ.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from unsupervised_llm_safety.common import tuned_defaults as td

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "exp_configs"
ALGO_CONFIG_DIR = CONFIG_DIR / "anodet_algos"

RELEASED_MODELS = ("qwen3_8b", "gpt_oss_20b", "gemma4_26b")

omegaconf = pytest.importorskip("omegaconf")
hydra = pytest.importorskip("hydra")


def _compose(overrides):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.2"):
        cfg = compose(
            config_name="config",
            overrides=["exp_type=test", "exp_ident=test", *overrides],
        )
    return cfg.anodet_algos


# --------------------------------------------------------------------------
# The file itself
# --------------------------------------------------------------------------


def test_file_loads_and_has_every_paper_model():
    data = td.load_tuned_defaults()
    assert set(data["models"]) == {
        "qwen2_1.5b",
        "ministral_8b",
        "llama3_8b",
        "qwen3_8b",
        "gpt_oss_20b",
        "gemma4_26b",
    }


def test_no_value_is_a_stringified_number():
    """PyYAML resolves `1e-5` to the *string* '1e-5' but `1.0e-5` to a float.

    A stringified float would be written into the config as a string and the
    detector would then compare it against numbers, or crash. Catch it here.
    """
    numeric = re.compile(r"^[-+]?[0-9.]+([eE][-+]?[0-9]+)?$")
    offenders = []
    for model_key, model in td.load_tuned_defaults()["models"].items():
        for method, entry in model["methods"].items():
            for key, value in (entry.get("params") or {}).items():
                if isinstance(value, str) and numeric.match(value):
                    offenders.append(f"{model_key}/{method}/{key}={value!r}")
    assert offenders == []


def test_algorithm_config_map_is_unambiguous():
    mapping = td.algorithm_config_map()
    assert mapping["cvdd_km_global_mask"] == "cvdd_kmeans_masked"
    assert mapping["local_freqmask_km"] == "cvdd_kmeans_masked"
    assert mapping["cvdd_sae"] == "cvdd"
    assert mapping["learned_mask_step1"] == "learned_mask"


def test_every_method_yaml_names_a_virtual_algorithm_that_exists():
    """`tuned_method:` in a method YAML must resolve to a real entry.

    A typo there is invisible at runtime: the lookup just misses and the run
    quietly uses the untuned defaults instead of the published ones.
    """
    known = set(td.algorithm_config_map())
    missing = []
    for path in sorted(ALGO_CONFIG_DIR.glob("*.yaml")):
        text = path.read_text()
        match = re.search(r"^tuned_method:\s*([A-Za-z0-9_]+)\s*$", text, re.M)
        assert match, f"{path.name} has no tuned_method key"
        if match.group(1) not in known:
            missing.append(f"{path.name} -> {match.group(1)}")
    assert missing == []


def test_every_tuned_key_exists_in_its_method_config():
    """Every dotted key must be settable on the config it claims to belong to.

    Without this, a renamed hyperparameter would fail only at the moment
    someone ran that one cell.
    """
    from omegaconf import OmegaConf

    bad = []
    for model_key, model in td.load_tuned_defaults()["models"].items():
        for method, entry in model["methods"].items():
            cfg_name = entry["hydra_config"]
            cfg = _compose([f"anodet_algos={cfg_name}"])
            for key in entry.get("params") or {}:
                if OmegaConf.select(cfg, key, default=_MISSING) is _MISSING:
                    bad.append(f"{model_key}/{method}: {cfg_name}.{key}")
    assert bad == []


_MISSING = object()


def test_shell_driver_sae_variant_matches_the_file():
    """run_one_detector.sh encodes the encoder choice in a bash `case`.

    Two copies of the same fact drift. Assert the shell rule reproduces the
    `sae_variant` recorded for every released SAE-space cell.
    """
    data = td.load_tuned_defaults()["models"]
    mismatches = []
    for model_key in RELEASED_MODELS:
        for method, entry in data[model_key]["methods"].items():
            variant = entry.get("sae_variant")
            if variant in (None, td.UNRELEASED_SAE):
                continue
            cfg_name = entry["hydra_config"]
            if model_key == "gpt_oss_20b":
                expected = (
                    "learnedmask"
                    if cfg_name.startswith("learned_mask") or cfg_name == "bce"
                    else "freqmask_km"
                )
            else:
                expected = "both"
            if variant != expected:
                mismatches.append(
                    f"{model_key}/{method}: file={variant} shell={expected}"
                )
    assert mismatches == []


# --------------------------------------------------------------------------
# Precedence
# --------------------------------------------------------------------------


def test_tuned_value_replaces_the_method_yaml_default():
    cfg = _compose(
        ["anodet_algos=mahalanobis", "anodet_algos.model_key=gpt_oss_20b"]
    )
    assert cfg.mahalanobis.score_type == "percentile"  # the bare default
    summary = td.apply_tuned_defaults(cfg, cli_overrides=[])
    assert summary["status"] == "applied"
    assert cfg.mahalanobis.score_type == "sigmoid"
    assert summary["applied"]["mahalanobis.score_type"] == "sigmoid"


def test_cli_override_beats_the_tuned_value_with_its_control():
    """CLI wins -- and the control proves the tuned layer was live."""
    override = "anodet_algos.gmm.n_components=5"

    with_cli = _compose(
        [
            "anodet_algos=gmm",
            "anodet_algos.model_key=gpt_oss_20b",
            override,
        ]
    )
    td.apply_tuned_defaults(with_cli, cli_overrides=[override])
    assert with_cli.gmm.n_components == 5

    # CONTROL: same config, override removed. Must come out as the tuned 10,
    # not 5 and not the method YAML's 100. If this were also 5 the test above
    # would prove nothing.
    without_cli = _compose(
        ["anodet_algos=gmm", "anodet_algos.model_key=gpt_oss_20b"]
    )
    td.apply_tuned_defaults(without_cli, cli_overrides=[])
    assert without_cli.gmm.n_components == 10
    assert with_cli.gmm.n_components != without_cli.gmm.n_components


def test_opt_out_restores_the_bare_method_default():
    cfg = _compose(
        [
            "anodet_algos=gmm",
            "anodet_algos.model_key=gpt_oss_20b",
            "anodet_algos.use_tuned_defaults=false",
        ]
    )
    summary = td.apply_tuned_defaults(cfg, cli_overrides=[])
    assert summary["status"] == "disabled"
    assert cfg.gmm.n_components == 100  # exp_configs/anodet_algos/gmm.yaml
    assert cfg.gmm.init_method == "kmeans++"


def test_variant_selection_changes_which_cell_is_applied():
    """One Hydra config, two published methods, two configurations."""
    freqmask = _compose(
        [
            "anodet_algos=cvdd_kmeans_masked",
            "anodet_algos.model_key=gpt_oss_20b",
        ]
    )
    td.apply_tuned_defaults(freqmask, cli_overrides=[])

    local = _compose(
        [
            "anodet_algos=cvdd_kmeans_masked",
            "anodet_algos.model_key=gpt_oss_20b",
            "anodet_algos.tuned_method=local_freqmask_km",
        ]
    )
    td.apply_tuned_defaults(local, cli_overrides=[])

    assert freqmask.cvdd_kmeans_masked.mask_variant == "top_k"
    assert local.cvdd_kmeans_masked.mask_variant == "frequency_coverage"
    assert freqmask.cvdd_kmeans_masked.n_clusters == 50
    assert local.cvdd_kmeans_masked.n_clusters == 200


def test_sweep_axes_are_reported_but_never_written():
    cfg = _compose(
        [
            "anodet_algos=cvdd_kmeans_masked",
            "anodet_algos.model_key=gpt_oss_20b",
            "anodet_algos.tuned_method=cvdd_km_raw_no_mask",
        ]
    )
    summary = td.apply_tuned_defaults(cfg, cli_overrides=[])
    assert summary["sweep_only"] == {
        "cvdd_kmeans_masked.distance_metric": "l1,l2"
    }
    # left at the method YAML's value, not set to the literal "l1,l2"
    assert cfg.cvdd_kmeans_masked.distance_metric == "l1"


def test_unknown_model_or_method_does_not_silently_pass():
    cfg = _compose(
        ["anodet_algos=mahalanobis", "anodet_algos.model_key=not_a_model"]
    )
    summary = td.apply_tuned_defaults(cfg, cli_overrides=[])
    assert summary["status"] == "no_entry"
    assert cfg.mahalanobis.score_type == "percentile"


def test_a_hyperparameter_that_does_not_exist_raises(tmp_path):
    """Deliberate breakage: prove the applier can go red.

    If a key were silently dropped, every check above would still pass while a
    published hyperparameter quietly stopped being applied.
    """
    bogus = tmp_path / "bogus.yaml"
    bogus.write_text(
        "schema_version: 1\n"
        "models:\n"
        "  gpt_oss_20b:\n"
        "    methods:\n"
        "      mahalanobis:\n"
        "        hydra_config: mahalanobis\n"
        "        params:\n"
        "          mahalanobis.no_such_hyperparameter: 3\n"
    )
    cfg = _compose(
        ["anodet_algos=mahalanobis", "anodet_algos.model_key=gpt_oss_20b"]
    )
    with pytest.raises(ValueError, match="no_such_hyperparameter"):
        td.apply_tuned_defaults(cfg, cli_overrides=[], path=bogus)
    td.clear_cache()


def test_a_key_cannot_be_both_a_value_and_a_sweep_axis(tmp_path):
    """Deliberate breakage: the two would silently fight over dict order."""
    clash = tmp_path / "clash.yaml"
    clash.write_text(
        "schema_version: 1\n"
        "models:\n"
        "  gpt_oss_20b:\n"
        "    methods:\n"
        "      gmm:\n"
        "        hydra_config: gmm\n"
        "        params:\n"
        "          gmm.n_components: 10\n"
        "        sweep_params:\n"
        '          gmm.n_components: "5,10"\n'
    )
    with pytest.raises(ValueError, match="both 'params' and 'sweep_params'"):
        td.load_tuned_defaults(clash)
    td.clear_cache()


def test_shipped_file_has_no_value_that_is_also_a_sweep_axis():
    td.clear_cache()
    td.load_tuned_defaults()  # raises if the shipped file has a clash


def test_malformed_file_raises_rather_than_returning_nothing(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 1\nnot_models: {}\n")
    with pytest.raises(ValueError, match="no top-level 'models'"):
        td.load_tuned_defaults(bad)
    td.clear_cache()


# --------------------------------------------------------------------------
# Override parsing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("anodet_algos.gmm.n_components=10", "anodet_algos.gmm.n_components"),
        ("+anodet_algos.foo=1", "anodet_algos.foo"),
        ("++anodet_algos.foo=1", "anodet_algos.foo"),
        ("~anodet_algos.foo", "anodet_algos.foo"),
        ("anodet_algos=gmm", "anodet_algos"),
    ],
)
def test_override_key_parsing(raw, expected):
    assert td.override_keys([raw]) == {expected}


def test_override_keys_is_not_a_substring_match():
    """`...n_components_extra=1` must not shadow `...n_components`."""
    keys = td.override_keys(["anodet_algos.gmm.n_components_extra=1"])
    assert "anodet_algos.gmm.n_components" not in keys
