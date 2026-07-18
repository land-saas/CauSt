"""Spatial neighbor graph construction.

CauST operates on a spatial model f(X, A) where A is a spot-adjacency graph.
This module builds that graph from spot coordinates stored in
``adata.obsm['spatial']`` and stores it (row-normalized) for the model to use.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from sklearn.neighbors import NearestNeighbors

SPATIAL_KEY = "spatial"
CONN_KEY = "spatial_connectivities"


def build_spatial_graph(
    adata: AnnData,
    method: str = "knn",
    n_neighbors: int = 6,
    radius: float | None = None,
    spatial_key: str = SPATIAL_KEY,
) -> AnnData:
    """Build a symmetric spatial adjacency graph and store it in ``adata.obsp``.

    Parameters
    ----------
    adata
        AnnData with spot coordinates in ``adata.obsm[spatial_key]`` (N x 2).
    method
        ``"knn"`` for k-nearest-neighbors or ``"radius"`` for a distance cutoff.
    n_neighbors
        Number of neighbors when ``method="knn"``.
    radius
        Distance cutoff when ``method="radius"``.

    Returns
    -------
    The same ``adata``, with a binary symmetric graph in
    ``adata.obsp['spatial_connectivities']``.
    """
    if spatial_key not in adata.obsm:
        raise KeyError(
            f"adata.obsm['{spatial_key}'] not found; store spot coordinates there first."
        )
    coords = np.asarray(adata.obsm[spatial_key], dtype=float)
    n = coords.shape[0]

    if method == "knn":
        # +1 because the first neighbor of each point is itself.
        k = min(n_neighbors + 1, n)
        nn = NearestNeighbors(n_neighbors=k).fit(coords)
        _, idx = nn.kneighbors(coords)
        rows = np.repeat(np.arange(n), k - 1)
        cols = idx[:, 1:].reshape(-1)
    elif method == "radius":
        if radius is None:
            raise ValueError("radius must be given when method='radius'.")
        nn = NearestNeighbors(radius=radius).fit(coords)
        _, idxs = nn.radius_neighbors(coords)
        rows, cols = [], []
        for i, neigh in enumerate(idxs):
            for j in neigh:
                if j != i:
                    rows.append(i)
                    cols.append(j)
        rows = np.asarray(rows, dtype=int)
        cols = np.asarray(cols, dtype=int)
    else:
        raise ValueError(f"unknown method {method!r}; use 'knn' or 'radius'.")

    data = np.ones(len(rows), dtype=float)
    adj = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    # symmetrize
    adj = adj.maximum(adj.T)
    adj.setdiag(0.0)
    adj.eliminate_zeros()
    adata.obsp[CONN_KEY] = adj
    return adata


def normalized_adjacency(adj: sp.spmatrix, add_self_loops: bool = True) -> sp.csr_matrix:
    """Row-normalized adjacency ``D^{-1}(A + I)`` used for graph smoothing."""
    adj = sp.csr_matrix(adj, dtype=float)
    if add_self_loops:
        adj = adj + sp.eye(adj.shape[0], format="csr")
    deg = np.asarray(adj.sum(axis=1)).reshape(-1)
    deg[deg == 0.0] = 1.0
    dinv = sp.diags(1.0 / deg)
    return (dinv @ adj).tocsr()
