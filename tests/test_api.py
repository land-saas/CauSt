"""Scanpy-style API: caust.tl / caust.pl."""

from __future__ import annotations

import numpy as np
import pytest

import caust
from caust.data import make_synthetic_cohort


@pytest.fixture
def scored():
    cohort = make_synthetic_cohort(n_slices=3, grid=10, n_causal=4, seed=0)
    table = caust.tl.causal_genes(cohort, backbone="simple", lam=1.0, n_components=8)
    return cohort, table


def test_causal_genes_writes_var_and_uns(scored):
    cohort, table = scored
    assert list(table.columns) == ["gene", "score", "mean_delta", "std_delta"]
    assert table.index.name == "rank" and table.index[0] == 1
    assert all(g.startswith("CAUSAL") for g in table["gene"].head(4))
    for a in cohort:
        assert {"caust_score", "caust_delta"} <= set(a.var.columns)
        assert a.uns["caust"]["lam"] == 1.0
        assert np.asarray(a.uns["caust"]["deltas"]).shape == (3, a.n_vars)


def test_select_genes_and_relambda(scored):
    cohort, _ = scored
    chosen = caust.tl.select_genes(cohort[0], n_top_genes=4)
    assert len(chosen) == 4 and cohort[0].var["caust_selected"].sum() == 4
    again = caust.tl.select_genes(cohort[0], n_top_genes=4, lam=0.0)
    assert len(again) == 4
    with pytest.raises(KeyError, match="causal_genes first"):
        caust.tl.select_genes(make_synthetic_cohort(n_slices=1, grid=6)[0])


def test_soft_weights_and_domains(scored):
    cohort, _ = scored
    w = caust.tl.soft_gene_weights(cohort[0])
    assert w.shape == (cohort[0].n_vars,) and np.all((w >= 0) & (w <= 1))
    genes = caust.tl.select_genes(cohort[0], n_top_genes=4)
    labels = caust.tl.spatial_domains(cohort[0], 4, genes=genes, n_components=3)
    assert len(set(labels)) == 4
    assert "X_simple" in cohort[0].obsm and "caust_domain" in cohort[0].obs
    assert caust.ari(cohort[0].obs["domain"], labels) > 0.5


def test_bad_backbone():
    with pytest.raises(ValueError, match="backbone must be"):
        caust.tl.causal_genes(
            make_synthetic_cohort(n_slices=2, grid=6), backbone="nope"
        )


def test_plots(scored):
    pytest.importorskip("matplotlib")
    cohort, _ = scored
    caust.tl.spatial_domains(cohort[0], 4, n_components=3)
    ax = caust.pl.ranking(cohort[0], n=6, markers=["CAUSAL_0"])
    assert len(ax.patches) == 6
    ax2 = caust.pl.domains(cohort[0], "caust_domain")
    assert ax2.get_title() == "caust_domain"
