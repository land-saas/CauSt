"""Tests for the end-to-end pipeline and error handling (caust.pipeline)."""

from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData

from caust import CauST
from caust.data import make_synthetic_slice


def test_fit_requires_slices():
    with pytest.raises(ValueError, match="at least one"):
        CauST().fit([])


def test_fit_requires_common_genes():
    a = make_synthetic_slice(grid=8, seed=0)
    b = make_synthetic_slice(grid=8, seed=1)
    b.var_names = [f"OTHER_{i}" for i in range(b.n_vars)]
    with pytest.raises(ValueError, match="no common genes"):
        CauST().fit([a, b])


def test_methods_before_fit_raise():
    cs = CauST()
    with pytest.raises(ValueError, match="not fitted"):
        cs.select_genes(4)
    with pytest.raises(ValueError, match="not fitted"):
        cs.soft_weights()
    with pytest.raises(ValueError, match="not fitted"):
        cs.ranking()


def test_verbose_fit_prints(cohort_small, capsys):
    CauST(lam=2.0).fit(cohort_small, verbose=True)
    out = capsys.readouterr().out
    assert "training backbone" in out
    assert "knockout scoring" in out


def test_ranking_sorted_descending(cohort_small):
    cs = CauST(lam=2.0).fit(cohort_small)
    ranking = cs.ranking()
    scores = [s for _, s in ranking]
    assert scores == sorted(scores, reverse=True)
    assert len(ranking) == len(cs.common_genes_)


def test_soft_weights_shape_and_range(cohort_small):
    cs = CauST(lam=2.0).fit(cohort_small)
    w = cs.soft_weights(temperature=1.0)
    assert w.shape == cs.common_genes_.shape
    assert np.all((w >= 0) & (w <= 1))


def test_transform_subsets_to_causal_genes(cohort_small):
    cs = CauST(lam=2.0).fit(cohort_small)
    sub = cs.transform(cohort_small[0], n_top_genes=8)
    assert isinstance(sub, AnnData)
    assert sub.n_vars <= 8
    assert set(sub.var_names).issubset(set(cohort_small[0].var_names))


def test_custom_model_factory_used(cohort_small):
    from caust.models import SimpleSpatialModel

    calls = {"n": 0}

    def factory() -> SimpleSpatialModel:
        calls["n"] += 1
        return SimpleSpatialModel(n_components=5)

    CauST(model_factory=factory).fit(cohort_small)
    assert calls["n"] == len(cohort_small)
