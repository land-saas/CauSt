"""Declarative experiment configuration.

Every knob that changes a result lives in a YAML file under ``configs/``, never
in a script. A config can inherit from another via a ``defaults:`` key, and the
fully-resolved config is written next to the results so a run can always be
explained by the file that produced it.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .repro import DEFAULT_SEED, DEFAULT_THREADS, digest_obj

#: Maximum depth of ``defaults:`` inheritance, to catch accidental cycles.
_MAX_INHERIT_DEPTH = 10


class ConfigError(ValueError):
    """Raised when a config file is missing, malformed, or incomplete."""


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` without mutating either."""
    out: dict[str, Any] = dict(copy.deepcopy(dict(base)))
    for key, value in override.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"config must be a YAML mapping, got {type(data).__name__}: {path}"
        )
    return data


def _resolve_inheritance(path: Path, depth: int = 0) -> dict[str, Any]:
    """Load ``path``, recursively merging any ``defaults:`` parent beneath it."""
    if depth > _MAX_INHERIT_DEPTH:
        raise ConfigError(f"defaults: inheritance too deep (cycle?) at {path}")
    raw = _read_yaml(path)
    parent_ref = raw.pop("defaults", None)
    if parent_ref is None:
        return raw
    parent_path = (path.parent / str(parent_ref)).resolve()
    parent = _resolve_inheritance(parent_path, depth + 1)
    return _deep_merge(parent, raw)


def _coerce_scalar(text: str) -> Any:
    """Parse a CLI ``--set key=value`` value using YAML scalar rules."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def apply_overrides(cfg: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply ``dotted.key=value`` overrides onto a config.

    Overrides may only touch keys that already exist, so a typo fails loudly
    instead of silently adding a setting the experiment never reads.
    """
    out = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override must be key=value, got: {item!r}")
        dotted, _, raw_value = item.partition("=")
        parts = [p for p in dotted.strip().split(".") if p]
        if not parts:
            raise ConfigError(f"override has an empty key: {item!r}")
        node: Any = out
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                raise ConfigError(f"unknown config key in override: {dotted}")
            node = node[part]
        leaf = parts[-1]
        if not isinstance(node, dict) or leaf not in node:
            raise ConfigError(f"unknown config key in override: {dotted}")
        node[leaf] = _coerce_scalar(raw_value)
    return out


@dataclass(frozen=True)
class ExperimentConfig:
    """A validated, fully-resolved experiment description."""

    name: str
    seed: int = DEFAULT_SEED
    threads: int = DEFAULT_THREADS
    data: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    selection: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any]) -> ExperimentConfig:
        missing = [k for k in ("name", "data", "selection") if k not in cfg]
        if missing:
            raise ConfigError(
                f"config is missing required section(s): {', '.join(missing)}"
            )
        known = {"name", "seed", "threads", "data", "model", "selection", "evaluation"}
        unknown = sorted(set(cfg) - known)
        if unknown:
            raise ConfigError(f"unknown config section(s): {', '.join(unknown)}")
        try:
            seed = int(cfg.get("seed", DEFAULT_SEED))
            threads = int(cfg.get("threads", DEFAULT_THREADS))
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"seed and threads must be integers: {exc}") from None
        if threads < 1:
            raise ConfigError(f"threads must be >= 1, got {threads}")
        return cls(
            name=str(cfg["name"]),
            seed=seed,
            threads=threads,
            data=dict(cfg["data"]),
            model=dict(cfg.get("model", {})),
            selection=dict(cfg["selection"]),
            evaluation=dict(cfg.get("evaluation", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "seed": self.seed,
            "threads": self.threads,
            "data": dict(self.data),
            "model": dict(self.model),
            "selection": dict(self.selection),
            "evaluation": dict(self.evaluation),
        }

    @property
    def digest(self) -> str:
        """Stable content hash of this config (first 12 hex chars used as run id)."""
        return digest_obj(self.to_dict())

    @property
    def run_id(self) -> str:
        """Directory name for this config's results: ``<name>-<digest12>``.

        Content-addressed on purpose: re-running the same config writes to the
        same directory, and any change to any setting produces a new one.
        """
        return f"{self.name}-{self.digest[:12]}"


def bundled_configs() -> dict[str, Path]:
    """Configs shipped inside the package, keyed like ``experiment/smoke``."""
    from importlib.resources import files

    root = Path(str(files("caust") / "configs"))
    out: dict[str, Path] = {}
    for path in sorted(root.rglob("*.yaml")):
        out[str(path.relative_to(root).with_suffix(""))] = path
    return out


def resolve_config_path(spec: str | Path) -> Path:
    """A config path on disk, or the name of a config bundled with the package.

    ``dlpfc_holdout`` and ``experiment/dlpfc_holdout`` both resolve to the
    bundled ``configs/experiment/dlpfc_holdout.yaml`` when no such file exists
    in the working directory, so ``caust run -c dlpfc_holdout`` works from a
    plain ``pip install caust``. Run ``caust configs`` to list them.
    """
    path = Path(spec)
    if path.is_file():
        return path.resolve()
    name = str(spec)
    if name.endswith(".yaml"):
        name = name[: -len(".yaml")]
    bundled = bundled_configs()
    if name in bundled:
        return bundled[name]
    matches = [k for k in bundled if k.split("/")[-1] == name]
    if len(matches) == 1:
        return bundled[matches[0]]
    if len(matches) > 1:
        raise ConfigError(f"ambiguous config name {spec!r}: {', '.join(matches)}")
    raise ConfigError(
        f"config file not found: {spec} (bundled configs: {', '.join(bundled)})"
    )


def load_config(
    path: str | Path, overrides: Sequence[str] | None = None
) -> ExperimentConfig:
    """Load, inherit, override, and validate a config file or bundled name."""
    resolved = _resolve_inheritance(resolve_config_path(path))
    if overrides:
        resolved = apply_overrides(resolved, overrides)
    return ExperimentConfig.from_dict(resolved)


def dump_config(cfg: ExperimentConfig, path: Path) -> None:
    """Write the resolved config next to the results it produced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=True, default_flow_style=False)
