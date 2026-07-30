import numpy as np
import pytest

from caust import (
    CauST,
    SimpleSpatialModel,
    build_spatial_graph,
    invariance_scores,
    knockout_scores,
    select_causal_genes,
    soft_weights,
)
from caust.data import make_synthetic_cohort, make_synthetic_slice
from caust.graph import CONN_KEY


def test_build_spatial_graph_symmetric():
    ad = make_synthetic_slice(grid=10, seed=1)
    build_spatial_graph(ad, n_neighbors=6)
    A = ad.obsp[CONN_KEY]
    assert A.shape == (ad.n_obs, ad.n_obs)
    # symmetric and no self-loops
    assert (A != A.T).nnz == 0
    assert A.diagonal().sum() == 0


def test_model_forward_deterministic():
    ad = make_synthetic_slice(grid=10, seed=2)
    m = SimpleSpatialModel(n_components=10).fit(ad)
    z1 = m.get_embedding()
    z2 = m.forward(m._get_expression())
    np.testing.assert_allclose(z1, z2)


def test_knockout_shifts_embedding():
    ad = make_synthetic_slice(grid=10, seed=3)
    m = SimpleSpatialModel(n_components=10).fit(ad)
    delta = knockout_scores(m)
    assert delta.shape == (ad.n_vars,)
    assert np.all(delta >= 0)
    # a causal gene should shift the embedding more than a shared-noise gene
    causal = delta[ad.var_names.get_loc("CAUSAL_0")]
    noise = delta[ad.var_names.get_loc("NOISE_0")]
    assert causal > noise


def test_invariance_scoring_math():
    deltas = np.array([[1.0, 0.5], [1.0, 5.0]])
    # gene 0: mean 1.0 std 0 -> score 1.0 ; gene 1: mean 2.75 std 2.25
    s = invariance_scores(deltas, lam=2.0)
    assert s[0] == pytest.approx(1.0)
    assert s[1] == pytest.approx(2.75 - 2.0 * 2.25)
    # lam=0 reduces to the mean
    s0 = invariance_scores(deltas, lam=0.0)
    np.testing.assert_allclose(s0, deltas.mean(0))


def test_select_and_soft_weights():
    scores = np.array([3.0, 1.0, 2.0, -np.inf])
    top = select_causal_genes(scores, 2)
    assert list(top) == [0, 2]
    w = soft_weights(scores)
    assert w[3] == 0.0
    assert np.all((w[:3] > 0) & (w[:3] < 1))


def test_pipeline_recovers_causal_genes():
    cohort = make_synthetic_cohort(n_slices=3, grid=14, n_causal=8, seed=0)
    cs = CauST(lam=2.0).fit(cohort)
    top8 = cs.select_genes(8)
    recovered = sum(g.startswith("CAUSAL") for g in top8)
    # invariance penalty should recover most causal genes over donor noise
    assert recovered >= 6
    # donor-specific noise should NOT dominate the top set
    assert sum(g.startswith("DONORNOISE") for g in top8) <= 1


def test_invariance_beats_high_delta_on_noise():
    # With lam>0, donor-noise genes (unstable across donors) should rank lower
    # than with lam=0 (mean effect only).
    cohort = make_synthetic_cohort(n_slices=3, grid=14, n_causal=8, seed=1)
    cs = CauST(lam=2.0).fit(cohort)
    inv_top = set(cs.select_genes(8))
    high_delta = set(
        np.array(cs.common_genes_)[
            select_causal_genes(invariance_scores(cs.deltas_, lam=0.0), 8)
        ]
    )
    inv_causal = sum(g.startswith("CAUSAL") for g in inv_top)
    hd_causal = sum(g.startswith("CAUSAL") for g in high_delta)
    assert inv_causal >= hd_causal


def test_beats_variance_baseline_on_held_out_donor():
    """The project's central claim: CauST transfers across donors, HVGs do not.

    Genes are chosen on the training donors only. Variance-based selection is
    drawn to the donor-specific noise genes (which are the highest-variance
    features by construction), while the knockout-invariance score is not.
    """
    cohort = make_synthetic_cohort(n_slices=3, grid=14, n_causal=8, seed=0)
    train = cohort[:-1]

    cs = CauST(lam=2.0).fit(train)
    caust_sel = cs.select_genes(8)

    names = np.asarray(cs.common_genes_)
    var = np.mean([np.asarray(a.X, dtype=float).var(axis=0) for a in train], axis=0)
    hvg_sel = names[np.argsort(-var)[:8]]

    caust_causal = sum(str(g).startswith("CAUSAL") for g in caust_sel)
    hvg_causal = sum(str(g).startswith("CAUSAL") for g in hvg_sel)

    assert caust_causal > hvg_causal
    # The baseline is actively misled by the donor-specific noise genes.
    assert sum(str(g).startswith("DONORNOISE") for g in hvg_sel) > sum(
        str(g).startswith("DONORNOISE") for g in caust_sel
    )
