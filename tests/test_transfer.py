"""Transfer benchmark on the synthetic smoke config (seconds, no network)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from caust.cli import main
from caust.config import load_config
from caust.experiment import ExperimentError
from caust.transfer import (
    RESULTS_CSV,
    SUMMARY_JSON,
    _read_rows,
    _write_rows,
    baseline_scores,
    default_scoring_indices,
    gene_set,
    hvg_jaccard,
    load_transfer_cohort,
    morans_i,
    run_transfer,
    summarize,
)

SMOKE = "transfer/smoke"
N_SLICES, N_SEEDS, N_K, N_STRAT = 6, 2, 2, 3


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("transfer")
    cfg = load_config(SMOKE)
    summary = run_transfer(cfg, out, figures=False, verbose=False)
    return cfg, out, summary


def test_grid_is_complete(smoke_run):
    cfg, out, summary = smoke_run
    rows = _read_rows(out / RESULTS_CSV)
    assert len(rows) == N_STRAT * N_K * N_SLICES * N_SLICES * N_SEEDS
    assert summary["k_values"] == [4, 8]
    assert summary["scoring_mode"] == "pooled"
    for s in summary["strategies"]:
        for k in ("4", "8"):
            e = summary["table"][s][k]
            assert {"within_slice", "within_donor", "cross_donor", "n_genes"} <= set(e)
            assert (
                e["cross_donor"]["n_cells"] == 6 * 4
            )  # 6 sources x 4 other-donor targets
            assert e["n_genes"]["max"] == int(k)
    assert json.loads((out / SUMMARY_JSON).read_text())["seeds"] == {
        "clustering": [0, 1],
        "training": [0],
    }


def test_caust_recovers_planted_genes_and_stats_levels(smoke_run):
    cfg, out, summary = smoke_run
    import csv

    with open(out / "gene_scores.csv") as fh:
        top = [r["gene"] for r in csv.DictReader(fh)][:4]
    assert all(g.startswith("CAUSAL") for g in top)
    wins = summary["caust_vs_hvg_wins"]["4"]
    assert wins["cross_donor"]["cells"] == 24 and wins["within_slice"]["cells"] == 6
    tests = summary["paired_tests"]["4"]["hvg"]
    assert tests["source"]["n_units"] == 6
    assert tests["donor_pair"]["n_units"] == 6  # 3 donors x 2
    assert tests["within_slice"]["n_units"] == 6


def test_resume_reruns_incomplete_cells_only(smoke_run, tmp_path):
    cfg, out, _ = smoke_run
    rows = _read_rows(out / RESULTS_CSV)
    # Drop half of one cell's rows: that cell must be re-run, the rest skipped.
    victim = ("caust", 8, "synthetic_0")
    kept = [
        r
        for r in rows
        if not (
            r["strategy"] == victim[0]
            and int(r["k"]) == victim[1]
            and r["source"] == victim[2]
        )
    ]
    partial = [r for r in rows if r not in kept][: N_SLICES * N_SEEDS // 2]
    (out / RESULTS_CSV).unlink()
    _write_rows(out / RESULTS_CSV, kept + partial)
    run_transfer(cfg, out, figures=False, verbose=False)
    again = _read_rows(out / RESULTS_CSV)
    assert len(again) == len(rows)
    assert (
        sum(1 for r in again if (r["strategy"], int(r["k"]), r["source"]) == victim)
        == N_SLICES * N_SEEDS
    )


def test_leave_target_donor_out(tmp_path):
    cfg = load_config(
        SMOKE,
        [
            "selection.scoring_mode=leave_target_donor_out",
            "selection.k_values=[4]",
            "selection.strategies=[hvg, caust, hvg_donor]",
            "selection.hvg_donor_n_top=20",
        ],
    )
    summary = run_transfer(cfg, tmp_path, figures=True, verbose=False)
    assert summary["scoring_mode"] == "leave_target_donor_out"
    ctx = summary["scoring_contexts"]
    assert set(ctx) == {"lodo_donor_0", "lodo_donor_1", "lodo_donor_2"}
    # A donor's context never scores on that donor's own slices.
    assert not {"synthetic_0", "synthetic_1"} & set(
        ctx["lodo_donor_0"]["scoring_slices"]
    )
    rows = _read_rows(tmp_path / RESULTS_CSV)
    labels = {(r["strategy"], r["target_donor"], r["scoring"]) for r in rows}
    assert ("caust", "donor_1", "lodo_donor_1") in labels
    assert all(r["scoring"] == "source" for r in rows if r["strategy"] == "hvg")
    assert sorted(p.name for p in (tmp_path / "figures").glob("*.png")) == [
        "ari_vs_k.png",
        "generalization_gap.png",
        "top_genes.png",
        "transfer_matrix_K4.png",
    ]


def test_k_above_pool_is_rejected(tmp_path):
    cfg = load_config(SMOKE, ["selection.k_values=[4, 1000]", "data.n_hvg_rank=1000"])
    with pytest.raises(ExperimentError, match="exceed"):
        run_transfer(cfg, tmp_path, figures=False, verbose=False)


def test_config_validation(tmp_path):
    cfg = load_config(SMOKE, ["selection.n_hvg_pool=999"])
    with pytest.raises(ExperimentError, match="n_hvg_pool"):
        run_transfer(cfg, tmp_path, figures=False, verbose=False)
    cfg = load_config(SMOKE, ["selection.strategies=[hvg, nope]"])
    with pytest.raises(ExperimentError, match="unknown strategies"):
        run_transfer(cfg, tmp_path, figures=False, verbose=False)


def test_train_seeds_multiply_rows(tmp_path):
    cfg = load_config(
        SMOKE,
        [
            "evaluation.train_seeds=[0, 1]",
            "selection.k_values=[4]",
            "selection.strategies=[caust]",
        ],
    )
    summary = run_transfer(cfg, tmp_path, figures=False, verbose=False)
    rows = _read_rows(tmp_path / RESULTS_CSV)
    assert len(rows) == N_SLICES * N_SLICES * N_SEEDS * 2
    assert summary["seeds"]["training"] == [0, 1]


def test_baselines_and_helpers():
    cfg = load_config(SMOKE)
    slices = load_transfer_cohort(cfg)
    assert [str(a.obs["donor"].iloc[0]) for a in slices] == [
        f"donor_{i // 2}" for i in range(6)
    ]
    assert default_scoring_indices(slices, per_donor=1) == [0, 2, 4]
    assert default_scoring_indices(slices, per_donor=2, exclude_donor="donor_1") == [
        0,
        1,
        4,
        5,
    ]
    names = np.asarray(slices[0].var_names)
    graph_before = dict(slices[0].obsp)
    base = baseline_scores(slices, [0, 2], names, n_top=10)
    assert dict(slices[0].obsp).keys() == graph_before.keys()  # no in-place graph
    assert set(base) == {"hvg_donor", "moran"} and base["moran"].shape == names.shape
    # Planted causal genes are spatially patterned: Moran's I ranks them first.
    top = gene_set("moran", 4, slices[0], names, np.zeros((2, len(names))), 1.0, base)
    assert all(g.startswith("CAUSAL") for g in top)
    # hvg_donor counts hits at the requested depth and is not degenerate.
    assert len(set(base["hvg_donor"])) > 1
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
    with pytest.raises(ExperimentError, match="exceeds"):
        gene_set("caust", 10_000, slices[0], names, np.zeros((2, len(names))), 1.0)
    assert morans_i(slices[0], names[:3]).shape == (3,)
    jac = hvg_jaccard(slices, n_top=10)
    assert jac["within_donor"]["n_pairs"] == 3 and jac["across_donor"]["n_pairs"] == 12


def test_summarize_partial_and_missing_strategy():
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
    s = summarize(rows)  # target 'b' never appears as a source: must not crash
    assert s["caust_vs_hvg_wins"] == {} and s["paired_tests"] == {}
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
    assert "cross-donor transfer ARI" in out and "one value per source" in out


def test_cli_transfer_bad_config(tmp_path):
    assert main(["transfer", "-c", str(tmp_path / "missing.yaml")]) == 2
