"""Tests for spatial backbones (caust.models)."""

from __future__ import annotations

import numpy as np
import pytest

from caust import SimpleSpatialModel
from caust.data import make_synthetic_slice
from caust.models import STAGATEAdapter


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


def test_stagate_adapter_is_a_stub(slice_small):
    adapter = STAGATEAdapter()
    with pytest.raises(NotImplementedError):
        adapter.fit(slice_small)
    with pytest.raises(NotImplementedError):
        adapter.get_embedding()
    with pytest.raises(NotImplementedError):
        adapter.forward(np.zeros((2, 2)))
    with pytest.raises(NotImplementedError):
        adapter._get_expression()
