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

    #: Replacement values for a knocked-out gene: ``"zero"`` silences it (Eq. 1
    #: of the proposal); ``"mean"`` replaces it by its own mean, which removes
    #: the gene's information while keeping the input on the data manifold.
    KNOCKOUT_MODES = ("zero", "mean")

    def get_knockout_embedding(self, gene_idx: int, mode: str = "zero") -> np.ndarray:
        """Embedding after knocking out ``gene_idx`` (Eq. 1 in the paper)."""
        if mode not in self.KNOCKOUT_MODES:
            raise ValueError(f"mode must be one of {self.KNOCKOUT_MODES}, got {mode!r}")
        X = self._get_expression().copy()
        X[:, gene_idx] = 0.0 if mode == "zero" else X[:, gene_idx].mean()
        return self.forward(X)

    @abstractmethod
    def _get_expression(self) -> np.ndarray:
        """Return the (N x G) dense expression matrix captured at fit time."""
