"""A lightweight, dependency-free reference spatial backbone.

This is a *working* stand-in for a full GNN autoencoder (STAGATE/GraphST). It
produces a spatial embedding by projecting standardized expression onto its top
principal components and smoothing the result over the spatial graph:

    Z = smooth_A( standardize(X) @ W )

The projection ``W``, the standardization statistics, and the graph are all
*frozen* after :meth:`fit`, so re-running :meth:`forward` on a gene-knocked-out
matrix produces a deterministic, gene-specific embedding shift -- exactly what
CauST's knockout scoring needs. Swap this out for a real backbone later via
:class:`caust.models.base.BaseSpatialModel`.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from sklearn.decomposition import PCA

from ..graph import CONN_KEY, build_spatial_graph, normalized_adjacency
from .base import BaseSpatialModel


def _dense(X) -> np.ndarray:
    return np.asarray(X.todense() if sp.issparse(X) else X, dtype=float)


class SimpleSpatialModel(BaseSpatialModel):
    """PCA + graph-smoothing reference backbone."""

    def __init__(
        self,
        n_components: int = 30,
        smooth_iters: int = 2,
        n_neighbors: int = 6,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.smooth_iters = smooth_iters
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        # frozen state, set in fit()
        self._X: np.ndarray | None = None
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._W: np.ndarray | None = None
        self._A_norm: sp.csr_matrix | None = None

    def fit(self, adata: AnnData) -> "SimpleSpatialModel":
        if CONN_KEY not in adata.obsp:
            build_spatial_graph(adata, n_neighbors=self.n_neighbors)
        self._A_norm = normalized_adjacency(adata.obsp[CONN_KEY])

        X = _dense(adata.X)
        self._X = X
        self._mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0.0] = 1.0
        self._std = std

        Xc = (X - self._mean) / self._std
        d = min(self.n_components, min(Xc.shape) - 1)
        pca = PCA(n_components=d, random_state=self.random_state)
        pca.fit(Xc)
        self._W = pca.components_.T  # (G x d)
        return self

    def _check_fitted(self) -> None:
        if self._W is None:
            raise ValueError("model is not fitted; call fit() first.")

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        # np.errstate guards against spurious FP-flag warnings raised by some
        # BLAS backends (e.g. macOS Accelerate + numpy 2.0) on valid matmuls.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            Xc = (np.asarray(X, dtype=float) - self._mean) / self._std
            Z = Xc @ self._W
            for _ in range(self.smooth_iters):
                Z = self._A_norm @ Z
        return Z

    def get_embedding(self, adata: AnnData | None = None) -> np.ndarray:
        self._check_fitted()
        X = self._X if adata is None else _dense(adata.X)
        return self.forward(X)

    def _get_expression(self) -> np.ndarray:
        self._check_fitted()
        return self._X
