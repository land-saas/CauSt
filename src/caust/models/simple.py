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
        self._Z: np.ndarray | None = None

    def fit(self, adata: AnnData) -> SimpleSpatialModel:
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
        # Same BLAS FP-flag guard as forward(): on a narrow matrix PCA takes the
        # covariance path (X.T @ X), which raises spurious divide/overflow/
        # invalid flags on some backends even though the inputs are finite.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            pca.fit(Xc)
        self._W = pca.components_.T  # (G x d)
        self._Z = None  # invalidate any embedding cached by a previous fit
        return self

    def _check_fitted(self) -> None:
        if self._W is None:
            raise ValueError("model is not fitted; call fit() first.")

    def forward(self, X: np.ndarray, A_norm: sp.csr_matrix | None = None) -> np.ndarray:
        """Frozen forward pass; ``A_norm`` defaults to the fitted slice's graph."""
        self._check_fitted()
        A = self._A_norm if A_norm is None else A_norm
        # np.errstate guards against spurious FP-flag warnings raised by some
        # BLAS backends (e.g. macOS Accelerate + numpy 2.0) on valid matmuls.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            Xc = (np.asarray(X, dtype=float) - self._mean) / self._std
            Z = Xc @ self._W
            for _ in range(self.smooth_iters):
                Z = A @ Z
        embedding: np.ndarray = np.asarray(Z)
        return embedding

    def get_embedding(self, adata: AnnData | None = None) -> np.ndarray:
        """Embed the fitted slice, or another slice zero-shot on its own graph."""
        self._check_fitted()
        if adata is None:
            return self.forward(self._get_expression())
        if CONN_KEY not in adata.obsp:
            build_spatial_graph(adata, n_neighbors=self.n_neighbors)
        return self.forward(_dense(adata.X), normalized_adjacency(adata.obsp[CONN_KEY]))

    def get_knockout_embedding(self, gene_idx: int) -> np.ndarray:
        """Knockout embedding via a rank-1 update instead of a full forward pass.

        Zeroing gene g changes exactly one column of the standardized input, by
        x_g / sigma_g, and every later step (projection, smoothing) is linear.
        The knocked-out embedding is therefore the base embedding minus a
        smoothed outer product -- O(N*d) per gene instead of O(N*G*d), which is
        what makes scoring thousands of real genes take seconds, not minutes.
        """
        self._check_fitted()
        assert self._X is not None and self._std is not None and self._W is not None
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            col = self._X[:, gene_idx] / self._std[gene_idx]
            shift = np.outer(col, self._W[gene_idx])
            for _ in range(self.smooth_iters):
                shift = self._A_norm @ shift
            embedding: np.ndarray = self._base_embedding() - shift
        return embedding

    def _base_embedding(self) -> np.ndarray:
        """Embedding of the training matrix, computed once and reused across
        the per-gene knockout loop."""
        if self._Z is None:
            self._Z = self.forward(self._get_expression())
        return self._Z

    def _get_expression(self) -> np.ndarray:
        self._check_fitted()
        assert self._X is not None  # set by fit(); _check_fitted guarantees it
        return self._X
