"""Apply the paper's per-(model, method) hyperparameters as defaults.

The values live in ``exp_configs/best_hyperparameters.yaml``; that file is the
record of what the paper reports and carries the provenance of every cell.
This module is the only thing that reads it.

The rule that matters is the precedence order::

    explicit command-line override  >  best_hyperparameters.yaml  >  the
    method YAML default in exp_configs/anodet_algos/

Getting that backwards would silently discard what a user typed, so the
command-line side is determined explicitly rather than inferred: Hydra records
the literal override strings on ``HydraConfig.get().overrides.task`` and
:func:`override_keys` parses the key out of each one. A key that appears there
is never written by this module.

Everything applied (and everything deliberately not applied) is logged with a
``[tuned-defaults]`` prefix, so a run's own log records the configuration it
actually used.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Set

logger = logging.getLogger(__name__)

#: Environment variable pointing at an alternative hyperparameter file.
CONFIG_ENV_VAR = "SAP_TUNED_DEFAULTS"

#: This file is at ``<repo>/src/unsupervised_llm_safety/common/``, so
#: ``parents[3]`` is the repository root.
_REPO_ROOT = Path(__file__).resolve().parents[3]

_DEFAULT_RELATIVE_PATH = Path("exp_configs") / "best_hyperparameters.yaml"

#: Keys this module refuses to write even if a file lists them. ``model_path``
#: is a filesystem path; it is resolved by the caller (the driver script or the
#: sweep orchestrator) from the entry's ``sae_variant``, never baked in here.
RESERVED_PARAM_KEYS = frozenset({"model_path"})

#: Marker in ``sae_variant`` meaning the encoder for this model was never
#: archived, so no honest ``model_path`` exists for it.
UNRELEASED_SAE = "unreleased"

_LOG_PREFIX = "[tuned-defaults]"

_cache: Dict[str, Dict[str, Any]] = {}


def tuned_defaults_path() -> Path:
    """Return the path of the hyperparameter file that will be read."""
    env = os.environ.get(CONFIG_ENV_VAR)
    if env:
        return Path(env)
    return _REPO_ROOT / _DEFAULT_RELATIVE_PATH


def load_tuned_defaults(
    path: Optional[os.PathLike] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Load and lightly validate the hyperparameter file.

    Raises:
        FileNotFoundError: if the file is missing.
        ValueError: if the top-level shape is not what this module expects.
            A malformed file must fail loudly; falling back to "no tuned
            values" would look exactly like a healthy run of the untuned
            configuration.
    """
    import yaml  # local: keeps this module importable without PyYAML

    resolved = Path(path) if path is not None else tuned_defaults_path()
    key = str(resolved.resolve() if resolved.exists() else resolved)
    if use_cache and key in _cache:
        return _cache[key]

    if not resolved.exists():
        raise FileNotFoundError(
            f"tuned hyperparameter file not found: {resolved}. Set "
            f"${CONFIG_ENV_VAR} to point at it, or run from the repository "
            f"root."
        )

    with open(resolved) as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict) or "models" not in data:
        raise ValueError(
            f"{resolved} has no top-level 'models' mapping; refusing to treat "
            f"it as a hyperparameter file."
        )
    for model_key, model in data["models"].items():
        if not isinstance(model, dict) or "methods" not in model:
            raise ValueError(
                f"{resolved}: model '{model_key}' has no 'methods' mapping."
            )
        for method, entry in model["methods"].items():
            if not isinstance(entry, dict) or "params" not in entry:
                raise ValueError(
                    f"{resolved}: entry '{model_key}/{method}' has no "
                    f"'params' mapping."
                )
            # A key cannot be both a single value and a sweep axis: the
            # orchestrator would emit one of them and silently drop the
            # other, and which one it dropped would depend on dict order.
            overlap = set(entry.get("params") or {}) & set(
                entry.get("sweep_params") or {}
            )
            if overlap:
                raise ValueError(
                    f"{resolved}: entry '{model_key}/{method}' lists "
                    f"{sorted(overlap)} in both 'params' and 'sweep_params'."
                )

    if use_cache:
        _cache[key] = data
    return data


def clear_cache() -> None:
    """Drop the parsed-file cache (tests reload alternate files)."""
    _cache.clear()


def tuned_entry(
    model_key: str,
    tuned_method: str,
    path: Optional[os.PathLike] = None,
) -> Optional[Dict[str, Any]]:
    """Return the raw entry for one (model, virtual algorithm), or None."""
    data = load_tuned_defaults(path)
    model = data["models"].get(model_key)
    if model is None:
        return None
    return model["methods"].get(tuned_method)


def tuned_params(
    model_key: str,
    tuned_method: str,
    path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """Return the single-value tuned parameters, keyed relative to
    ``anodet_algos``. Empty dict when the combination is not tuned."""
    entry = tuned_entry(model_key, tuned_method, path)
    if entry is None:
        return {}
    params = dict(entry.get("params") or {})
    for reserved in RESERVED_PARAM_KEYS:
        params.pop(reserved, None)
    return params


def tuned_sweep_params(
    model_key: str,
    tuned_method: str,
    path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """Return parameters that are Hydra *sweep axes* rather than single values.

    These are never written into a running config -- ``"l1,l2"`` is a
    multirun axis, not a value -- so :func:`apply_tuned_defaults` reports them
    and leaves them alone. The sweep orchestrator emits them verbatim.
    """
    entry = tuned_entry(model_key, tuned_method, path)
    if entry is None:
        return {}
    return dict(entry.get("sweep_params") or {})


def algorithm_config_map(
    path: Optional[os.PathLike] = None,
) -> Dict[str, str]:
    """Virtual algorithm name -> Hydra config name, derived from the file.

    Raises ValueError if two models disagree about which config a virtual
    algorithm maps to; that would make the name ambiguous.
    """
    data = load_tuned_defaults(path)
    mapping: Dict[str, str] = {}
    for model_key, model in data["models"].items():
        for method, entry in model["methods"].items():
            cfg_name = entry.get("hydra_config", method)
            if mapping.setdefault(method, cfg_name) != cfg_name:
                raise ValueError(
                    f"virtual algorithm '{method}' maps to both "
                    f"'{mapping[method]}' and '{cfg_name}' (at model "
                    f"'{model_key}')"
                )
    return mapping


def override_keys(task_overrides: Optional[Iterable[str]]) -> Set[str]:
    """Parse Hydra override strings into the set of keys they touch.

    ``HydraConfig.get().overrides.task`` yields strings such as
    ``anodet_algos.gmm.n_components=10``, ``+foo.bar=1`` or ``~foo.bar``. Only
    the key half matters here.
    """
    keys: Set[str] = set()
    for raw in task_overrides or ():
        text = str(raw).lstrip("+~")
        key = text.split("=", 1)[0].strip()
        if key:
            keys.add(key)
    return keys


def _hydra_task_overrides() -> Optional[Sequence[str]]:
    """Return the literal CLI overrides of the running Hydra job, if any."""
    try:
        from hydra.core.hydra_config import HydraConfig
    except Exception:  # pragma: no cover - hydra is an optional extra
        return None
    try:
        return list(HydraConfig.get().overrides.task)
    except Exception:
        # Not inside a @hydra.main job (unit tests, library use).
        return None


def resolve_tuned_method(cfg: Mapping[str, Any]) -> Optional[str]:
    """Which virtual algorithm this config asks for.

    ``tuned_method`` in the method YAML names it explicitly (the same Hydra
    config backs several virtual algorithms), falling back to ``method``.
    """
    explicit = cfg.get("tuned_method")
    if explicit:
        return str(explicit)
    method = cfg.get("method")
    return str(method) if method else None


def apply_tuned_defaults(
    cfg: Any,
    cli_overrides: Optional[Iterable[str]] = None,
    prefix: str = "anodet_algos",
    path: Optional[os.PathLike] = None,
) -> Dict[str, Any]:
    """Write the tuned values for this (model, method) into ``cfg`` in place.

    ``cfg`` is the ``anodet_algos`` config node. Keys in ``cli_overrides``
    (defaulting to the running Hydra job's own override list) are left
    untouched, so an explicit command-line value always wins.

    Returns a summary dict: ``status`` plus the ``applied`` and ``from_cli``
    mappings, so a caller (or a test) can assert on what happened rather than
    scraping the log.
    """
    from omegaconf import OmegaConf

    summary: Dict[str, Any] = {
        "status": "not_applied",
        "model_key": cfg.get("model_key"),
        "tuned_method": None,
        "applied": {},
        "from_cli": {},
        "sweep_only": {},
    }

    if not cfg.get("use_tuned_defaults", True):
        summary["status"] = "disabled"
        logger.info(
            "%s DISABLED via use_tuned_defaults=false; the method YAML "
            "defaults are used as-is.",
            _LOG_PREFIX,
        )
        return summary

    model_key = cfg.get("model_key")
    tuned_method = resolve_tuned_method(cfg)
    summary["tuned_method"] = tuned_method

    if not model_key or not tuned_method:
        summary["status"] = "no_key"
        logger.warning(
            "%s model_key=%r tuned_method=%r -- cannot look up tuned "
            "values; running with the method YAML defaults.",
            _LOG_PREFIX,
            model_key,
            tuned_method,
        )
        return summary

    entry = tuned_entry(model_key, tuned_method, path)
    if entry is None:
        summary["status"] = "no_entry"
        logger.warning(
            "%s no tuned entry for (model_key=%s, method=%s) in %s; "
            "running with the method YAML defaults. This is NOT the "
            "published configuration if the paper reports that cell.",
            _LOG_PREFIX,
            model_key,
            tuned_method,
            tuned_defaults_path(),
        )
        return summary

    if cli_overrides is None:
        cli_overrides = _hydra_task_overrides()
    cli_keys = override_keys(cli_overrides)

    params = tuned_params(model_key, tuned_method, path)
    sweeps = tuned_sweep_params(model_key, tuned_method, path)

    logger.info(
        "%s %s / %s (%s, source=%s) from %s",
        _LOG_PREFIX,
        model_key,
        tuned_method,
        entry.get("paper_name", tuned_method),
        entry.get("source", "unspecified"),
        tuned_defaults_path(),
    )

    for key in sorted(params):
        value = params[key]
        full_key = f"{prefix}.{key}" if prefix else key
        if full_key in cli_keys:
            summary["from_cli"][key] = value
            logger.info(
                "%s   %-52s SKIPPED  (explicit CLI override wins over the "
                "tuned value %r)",
                _LOG_PREFIX,
                key,
                value,
            )
            continue
        try:
            OmegaConf.update(cfg, key, value, merge=False)
        except Exception as exc:
            raise ValueError(
                f"tuned hyperparameter '{key}' for "
                f"({model_key}, {tuned_method}) does not exist in the "
                f"'{cfg.get('method')}' config. Fix "
                f"{tuned_defaults_path()} or the method YAML; refusing to "
                f"run with a hyperparameter that was silently dropped."
            ) from exc
        summary["applied"][key] = value
        logger.info("%s   %-52s = %r", _LOG_PREFIX, key, value)

    for key, value in sorted(sweeps.items()):
        summary["sweep_only"][key] = value
        logger.info(
            "%s   %-52s SWEEP AXIS %r -- not written into this run; pass it "
            "with --multirun to reproduce the published sweep.",
            _LOG_PREFIX,
            key,
            value,
        )

    summary["status"] = "applied"
    logger.info(
        "%s %d value(s) applied, %d left to explicit CLI override(s).",
        _LOG_PREFIX,
        len(summary["applied"]),
        len(summary["from_cli"]),
    )
    return summary
