import pytest
import yaml

from caust.config import (
    ConfigError,
    ExperimentConfig,
    apply_overrides,
    dump_config,
    load_config,
)

MINIMAL = {
    "name": "t",
    "data": {"source": "synthetic", "n_slices": 2},
    "selection": {"lam": 2.0, "n_top_genes": 4},
}


def write(tmp_path, name, payload):
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return p


def test_from_dict_requires_core_sections():
    with pytest.raises(ConfigError, match="missing required section"):
        ExperimentConfig.from_dict({"name": "t"})


def test_from_dict_rejects_unknown_sections():
    with pytest.raises(ConfigError, match="unknown config section"):
        ExperimentConfig.from_dict({**MINIMAL, "typo": {}})


def test_threads_must_be_positive():
    with pytest.raises(ConfigError, match="threads must be >= 1"):
        ExperimentConfig.from_dict({**MINIMAL, "threads": 0})


def test_defaults_inheritance_merges_parent(tmp_path):
    write(tmp_path, "base.yaml", {**MINIMAL, "name": "base", "seed": 1})
    child = write(tmp_path, "child.yaml", {"defaults": "base.yaml", "name": "child"})
    cfg = load_config(child)
    assert cfg.name == "child"
    assert cfg.seed == 1  # inherited
    assert cfg.selection["n_top_genes"] == 4  # inherited


def test_inheritance_cycle_is_caught(tmp_path):
    write(tmp_path, "a.yaml", {"defaults": "b.yaml", **MINIMAL})
    write(tmp_path, "b.yaml", {"defaults": "a.yaml", **MINIMAL})
    with pytest.raises(ConfigError, match="too deep"):
        load_config(tmp_path / "a.yaml")


def test_missing_file_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_overrides_apply_and_coerce_types():
    out = apply_overrides(MINIMAL, ["selection.lam=0.5", "data.n_slices=7"])
    assert out["selection"]["lam"] == 0.5
    assert out["data"]["n_slices"] == 7


def test_override_of_unknown_key_fails_loudly():
    # A typo must not silently add a setting nothing reads.
    with pytest.raises(ConfigError, match="unknown config key"):
        apply_overrides(MINIMAL, ["selection.lamda=1.0"])
    with pytest.raises(ConfigError, match="unknown config key"):
        apply_overrides(MINIMAL, ["nosuch.key=1"])


def test_override_requires_key_equals_value():
    with pytest.raises(ConfigError, match="key=value"):
        apply_overrides(MINIMAL, ["selection.lam"])


def test_run_id_is_content_addressed():
    a = ExperimentConfig.from_dict(MINIMAL)
    b = ExperimentConfig.from_dict(
        {**MINIMAL, "selection": {"lam": 1.0, "n_top_genes": 4}}
    )
    assert a.run_id.startswith("t-")
    assert a.run_id != b.run_id
    # Same content, rebuilt independently -> same id.
    assert ExperimentConfig.from_dict(MINIMAL).run_id == a.run_id


def test_dump_config_round_trips(tmp_path):
    cfg = ExperimentConfig.from_dict(MINIMAL)
    out = tmp_path / "resolved.yaml"
    dump_config(cfg, out)
    assert load_config(out).to_dict() == cfg.to_dict()


def test_shipped_configs_are_valid():
    from pathlib import Path

    from caust.config import bundled_configs

    root = Path(bundled_configs()["experiment/smoke"]).parent
    found = sorted(root.glob("*.yaml"))
    assert found, "no experiment configs found"
    for path in found:
        cfg = load_config(path)
        assert cfg.name
        assert cfg.threads >= 1


def test_bundled_config_names_resolve():
    from caust.config import ConfigError, bundled_configs, resolve_config_path

    names = bundled_configs()
    assert "experiment/smoke" in names and "transfer/smoke" in names
    assert resolve_config_path("experiment/smoke") == names["experiment/smoke"]
    assert resolve_config_path("dlpfc_holdout").name == "dlpfc_holdout.yaml"
    with pytest.raises(ConfigError, match="ambiguous"):
        resolve_config_path("smoke")
    with pytest.raises(ConfigError, match="not found"):
        resolve_config_path("does-not-exist")
    assert load_config("synthetic_holdout").name == "synthetic_holdout"


def test_cli_configs_lists_bundled(capsys):
    from caust.cli import main

    assert main(["configs"]) == 0
    out = capsys.readouterr().out
    assert "experiment/dlpfc_holdout" in out and "transfer/dlpfc_stagate" in out
