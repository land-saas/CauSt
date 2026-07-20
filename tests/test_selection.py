"""Edge-case tests for scoring/selection primitives and synthetic data."""

from __future__ import annotations

import numpy as np
import pytest

from caust import SimpleSpatialModel
from caust.data import make_synthetic_cohort, make_synthetic_slice
from caust.intervention import knockout_scores
from caust.invariance import invariance_scores, select_causal_genes, soft_weights


def test_invariance_negative_lambda_raises():
    with pytest.raises(ValueError, match=">= 0"):
        invariance_scores(np.array([[1.0, 2.0]]), lam=-1.0)


def test_invariance_ignores_nan_slices():
    deltas = np.array([[1.0, np.nan], [3.0, np.nan]])
    s = invariance_scores(deltas, lam=0.0)
    assert s[0] == pytest.approx(2.0)
    assert s[1] == -np.inf  # gene with no valid slice


def test_select_caps_at_finite_count():
    scores = np.array([1.0, -np.inf, 2.0])
    idx = select_causal_genes(scores, n_top_genes=10)
    assert list(idx) == [2, 0]  # only 2 finite genes, best first


def test_soft_weights_all_invalid_returns_zeros():
    scores = np.array([-np.inf, -np.inf])
    w = soft_weights(scores)
    assert np.all(w == 0.0)


def test_knockout_scores_subset_leaves_nan():
    ad = make_synthetic_slice(grid=8, seed=1)
    m = SimpleSpatialModel(n_components=8).fit(ad)
    delta = knockout_scores(m, gene_indices=[0, 1, 2])
    assert np.isfinite(delta[:3]).all()
    assert np.isnan(delta[3:]).all()


def test_knockout_scores_verbose(capsys):
    ad = make_synthetic_slice(grid=8, seed=2)
    m = SimpleSpatialModel(n_components=8).fit(ad)
    knockout_scores(m, verbose=True)
    assert "knockout scoring" in capsys.readouterr().out


def test_synthetic_slice_layout_and_labels():
    ad = make_synthetic_slice(
        grid=10, n_domains=4, n_causal=5, n_noise=10, n_donor_noise=7, seed=3
    )
    assert ad.n_vars == 5 + 10 + 7
    assert ad.n_obs == 100
    assert set(ad.obs["domain"].unique()) <= {"0", "1", "2", "3"}
    assert "spatial" in ad.obsm


def test_synthetic_cohort_shares_var_names():
    cohort = make_synthetic_cohort(n_slices=3, grid=8, seed=0)
    names0 = list(cohort[0].var_names)
    for ad in cohort[1:]:
        assert list(ad.var_names) == names0
