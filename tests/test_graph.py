"""Tests for spatial graph construction (caust.graph)."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from caust.data import make_synthetic_slice
from caust.graph import (
    CONN_KEY,
    build_spatial_graph,
    normalized_adjacency,
)


def test_knn_graph_symmetric_no_self_loops():
    ad = make_synthetic_slice(grid=10, seed=1)
    build_spatial_graph(ad, method="knn", n_neighbors=6)
    A = ad.obsp[CONN_KEY]
    assert A.shape == (ad.n_obs, ad.n_obs)
    assert (A != A.T).nnz == 0
    assert A.diagonal().sum() == 0


def test_radius_graph_builds():
    ad = make_synthetic_slice(grid=8, seed=2)
    build_spatial_graph(ad, method="radius", radius=1.5)
    A = ad.obsp[CONN_KEY]
    assert A.shape == (ad.n_obs, ad.n_obs)
    # every spot has at least one neighbor within radius 1.5 on a unit grid
    assert np.asarray(A.sum(axis=1)).reshape(-1).min() >= 1
    assert (A != A.T).nnz == 0


def test_radius_without_radius_raises():
    ad = make_synthetic_slice(grid=8, seed=3)
    with pytest.raises(ValueError, match="radius must be given"):
        build_spatial_graph(ad, method="radius")


def test_unknown_method_raises():
    ad = make_synthetic_slice(grid=8, seed=4)
    with pytest.raises(ValueError, match="unknown method"):
        build_spatial_graph(ad, method="triangulation")


def test_missing_spatial_key_raises():
    ad = make_synthetic_slice(grid=8, seed=5)
    del ad.obsm["spatial"]
    with pytest.raises(KeyError, match="not found"):
        build_spatial_graph(ad)


def test_normalized_adjacency_rows_sum_to_one():
    ad = make_synthetic_slice(grid=8, seed=6)
    build_spatial_graph(ad, n_neighbors=6)
    norm = normalized_adjacency(ad.obsp[CONN_KEY], add_self_loops=True)
    row_sums = np.asarray(norm.sum(axis=1)).reshape(-1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-8)


def test_normalized_adjacency_isolated_node():
    # A node with no edges must not divide by zero.
    A = sp.csr_matrix(np.zeros((3, 3)))
    norm = normalized_adjacency(A, add_self_loops=False)
    assert np.isfinite(norm.toarray()).all()
