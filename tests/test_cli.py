"""Tests for the command-line interface (caust.cli)."""

from __future__ import annotations

import json
from pathlib import Path

from caust.cli import main


def test_demo_command_runs(capsys):
    rc = main(["demo", "--slices", "3", "--n-causal", "8", "--lam", "2.0"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "invariance score" in out
    assert "ARI" in out


def test_no_command_prints_help(capsys):
    rc = main([])
    out = capsys.readouterr().out
    assert rc == 1
    assert "caust" in out.lower()


SMOKE = Path(__file__).resolve().parents[1] / "configs" / "experiment" / "smoke.yaml"


def test_run_then_verify_round_trip(tmp_path, capsys):
    rc = main(["run", "-c", str(SMOKE), "-o", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "run id" in out

    rundirs = list(tmp_path.glob("smoke-*"))
    assert len(rundirs) == 1

    rc = main(["verify", str(rundirs[0])])
    assert rc == 0
    assert "REPRODUCED" in capsys.readouterr().out


def test_run_applies_overrides_and_changes_run_id(tmp_path):
    main(["run", "-c", str(SMOKE), "-o", str(tmp_path)])
    main(["run", "-c", str(SMOKE), "-o", str(tmp_path), "--set", "selection.lam=0.0"])
    assert len(list(tmp_path.glob("smoke-*"))) == 2


def test_run_rejects_an_unknown_override_key(tmp_path, capsys):
    rc = main(
        ["run", "-c", str(SMOKE), "-o", str(tmp_path), "--set", "selection.lamda=1"]
    )
    assert rc == 2
    assert "unknown config key" in capsys.readouterr().out


def test_verify_reports_mismatch_after_tampering(tmp_path, capsys):
    main(["run", "-c", str(SMOKE), "-o", str(tmp_path)])
    rundir = next(tmp_path.glob("smoke-*"))
    metrics = json.loads((rundir / "metrics.json").read_text())
    metrics["lam"] = 99.0
    (rundir / "metrics.json").write_text(json.dumps(metrics))
    rc = main(["verify", str(rundir)])
    assert rc == 1
    assert "MISMATCH" in capsys.readouterr().out


def test_verify_on_a_missing_run_errors(tmp_path, capsys):
    rc = main(["verify", str(tmp_path / "nope")])
    assert rc == 2
    assert "error" in capsys.readouterr().out.lower()
