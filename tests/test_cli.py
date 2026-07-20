"""Tests for the command-line interface (caust.cli)."""

from __future__ import annotations

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
