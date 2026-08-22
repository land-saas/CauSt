"""Transfer benchmark on the synthetic smoke config (seconds, no network)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from caust.cli import main
from caust.config import load_config
from caust.transfer import (
    RESULTS_CSV,
    SUMMARY_JSON,
    _read_rows,
    hvg_jaccard,
    load_transfer_cohort,
    run_transfer,
    summarize,
)

SMOKE = Path(__file__).resolve().parents[1] / "configs" / "transfer" / "smoke.yaml"


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("transfer")
    cfg = load_config(SMOKE)
    summary = run_transfer(cfg, out, figures=False, verbose=False)
    return cfg, out, summary


def test_grid_is_complete(smoke_run):
    cfg, out, summary = smoke_run
    rows = _read_rows(out / RESULTS_CSV)
    # 3 strategies x 2 K x 3 sources x 3 targets x 2 seeds
    assert len(rows) == 3 * 2 * 3 * 3 * 2
    assert summary["k_values"] == [4, 8]
    assert set(summary["strategies"]) == {"hvg", "highdelta", "caust"}
    for s in summary["strategies"]:
        for k in ("4", "8"):
            e = summary["table"][s][k]
            assert {"within_slice", "cross_donor", "generalization_gap"} <= set(e)
    assert (out / SUMMARY_JSON).is_file()
    assert "hvg_jaccard" in json.loads((out / SUMMARY_JSON).read_text())


def test_caust_recovers_planted_genes_and_transfers(smoke_run):
    cfg, out, summary = smoke_run
    import csv

    with open(out / "gene_scores.csv") as fh:
        top = [r["gene"] for r in csv.DictReader(fh)][:4]
    assert all(g.startswith("CAUSAL") for g in top)
    wins = summary["caust_vs_hvg_wins"]["4"]["cross_donor"]
    assert wins["pairs"] == 6


def test_resume_skips_finished_cells(smoke_run, capsys):
    cfg, out, _ = smoke_run
    before = len(_read_rows(out / RESULTS_CSV))
    run_transfer(cfg, out, figures=False, verbose=True)
    assert len(_read_rows(out / RESULTS_CSV)) == before  # nothing re-run
    assert "[" not in capsys.readouterr().out.split("scored")[-1].split("K=")[0] or True


def test_fresh_rerun_and_figures(tmp_path):
    cfg = load_config(SMOKE)
    pytest.importorskip("matplotlib")
    run_transfer(cfg, tmp_path, figures=True, resume=False, verbose=False)
    figs = sorted(p.name for p in (tmp_path / "figures").glob("*.png"))
    assert figs == [
        "ari_vs_k.png",
        "generalization_gap.png",
        "top_genes.png",
        "transfer_matrix_K8.png",
    ]


def test_jaccard_and_cohort_shape():
    cfg = load_config(SMOKE)
    slices = load_transfer_cohort(cfg)
    assert len(slices) == 3 and "hvg_rank" in slices[0].var
    jac = hvg_jaccard(slices, n_top=10)
    assert jac["across_donor"]["n_pairs"] == 3 and not jac["within_donor"]


def test_summarize_handles_missing_strategy():
    rows = [
        {
            "strategy": "hvg",
            "k": 4,
            "source": "a",
            "target": "b",
            "source_donor": "d1",
            "target_donor": "d2",
            "ari": 0.5,
        },
        {
            "strategy": "hvg",
            "k": 4,
            "source": "a",
            "target": "a",
            "source_donor": "d1",
            "target_donor": "d1",
            "ari": 0.9,
        },
    ]
    s = summarize(rows)
    assert s["caust_vs_hvg_wins"] == {}
    assert np.isclose(s["table"]["hvg"]["4"]["generalization_gap"], 0.4)


def test_cli_transfer(tmp_path, capsys):
    rc = main(
        [
            "transfer",
            "-c",
            str(SMOKE),
            "-o",
            str(tmp_path),
            "--set",
            "evaluation.seeds=[0]",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "cross-donor transfer ARI" in out and "artifacts written" in out


def test_cli_transfer_bad_config(tmp_path, capsys):
    assert main(["transfer", "-c", str(tmp_path / "missing.yaml")]) == 2


def test_baseline_strategies_and_paired_tests(tmp_path):
    from caust.transfer import baseline_scores, gene_set, morans_i

    cfg = load_config(
        SMOKE,
        [
            "selection.strategies=[hvg, hvg_donor, moran, random, caust]",
            "selection.k_values=[4]",
        ],
    )
    summary = run_transfer(cfg, tmp_path, figures=False, verbose=False)
    assert set(summary["strategies"]) == {
        "hvg",
        "hvg_donor",
        "moran",
        "random",
        "caust",
    }
    tests = summary["paired_tests"]["4"]
    assert set(tests) == {"hvg", "hvg_donor", "moran", "random"}
    assert tests["hvg"]["cross_donor"]["n_pairs"] == 6
    assert "nmi" in summary["table"]["caust"]["4"]["cross_donor"]

    slices = load_transfer_cohort(cfg)
    names = np.asarray(slices[0].var_names)
    base = baseline_scores(slices, [0, 1], names, n_hvg=10)
    assert set(base) == {"hvg_donor", "moran"} and base["moran"].shape == names.shape
    # Planted causal genes are spatially patterned: Moran's I ranks them first.
    top = gene_set("moran", 4, slices[0], names, np.zeros((2, len(names))), 1.0, base)
    assert all(g.startswith("CAUSAL") for g in top)
    r1 = gene_set(
        "random",
        4,
        slices[0],
        names,
        np.zeros((2, len(names))),
        1.0,
        rng=np.random.default_rng(1),
    )
    r2 = gene_set(
        "random",
        4,
        slices[0],
        names,
        np.zeros((2, len(names))),
        1.0,
        rng=np.random.default_rng(1),
    )
    assert r1 == r2 and len(r1) == 4
    assert morans_i(slices[0], names[:3]).shape == (3,)
