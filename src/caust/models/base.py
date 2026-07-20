"""Backend interface for CauST.

CauST is backbone-agnostic: it only needs a *frozen* spatial model that maps an
expression matrix (and a fixed spatial graph) to a low-dimensional embedding, and
that lets us run a forward pass on a perturbed (gene-knocked-out) input.

Any backbone (STAGATE, GraphST, SpaGCN, or the bundled reference model) can be
plugged in by implementing this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from anndata import AnnData


class BaseSpatialModel(ABC):
    """Abstract spatial-embedding backbone used by CauST."""

    @abstractmethod
    def fit(self, adata: AnnData) -> BaseSpatialModel:
        """Train the model on one tissue slice. Returns ``self``."""

    @abstractmethod
    def get_embedding(self, adata: AnnData | None = None) -> np.ndarray:
        """Return the (N x d) spatial embedding for the fitted slice."""

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Frozen forward pass f(X, A) on a (possibly perturbed) matrix X.

        The spatial graph A is fixed (captured at ``fit`` time). This is the hook
        CauST uses for in-silico knockouts: pass X with a gene column zeroed out.
        """

    def get_knockout_embedding(self, gene_idx: int) -> np.ndarray:
        """Embedding after zeroing out ``gene_idx`` (Eq. 1 in the paper)."""
        X = self._get_expression().copy()
        X[:, gene_idx] = 0.0
        return self.forward(X)

    @abstractmethod
    def _get_expression(self) -> np.ndarray:
        """Return the (N x G) dense expression matrix captured at fit time."""
