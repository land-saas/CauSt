"""Tests for spatial backbones (caust.models)."""

from __future__ import annotations

import numpy as np
import pytest

from caust import SimpleSpatialModel
from caust.data import make_synthetic_slice


def test_forward_before_fit_raises():
    m = SimpleSpatialModel()
    with pytest.raises(ValueError, match="not fitted"):
        m.forward(np.zeros((4, 4)))


def test_get_embedding_on_new_adata(slice_small):
    m = SimpleSpatialModel(n_components=10).fit(slice_small)
    other = make_synthetic_slice(grid=10, seed=99)
    emb = m.get_embedding(other)
    assert emb.shape[0] == other.n_obs


def test_knockout_embedding_differs_from_base(slice_small):
    m = SimpleSpatialModel(n_components=10).fit(slice_small)
    base = m.get_embedding()
    ko = m.get_knockout_embedding(gene_idx=slice_small.var_names.get_loc("CAUSAL_0"))
    assert not np.allclose(base, ko)


def test_rank1_knockout_matches_naive_forward(slice_small):
    # The simple model's rank-1 shortcut must agree with the generic
    # zero-the-column-and-rerun path it replaces.
    from caust.models.base import BaseSpatialModel

    m = SimpleSpatialModel(n_components=10).fit(slice_small)
    for gene in ("CAUSAL_0", "NOISE_5", "DONORNOISE_3"):
        idx = slice_small.var_names.get_loc(gene)
        fast = m.get_knockout_embedding(idx)
        naive = BaseSpatialModel.get_knockout_embedding(m, idx)
        np.testing.assert_allclose(fast, naive, atol=1e-10)
