"""Optional STAGATE backbone adapter (skeleton).

This is a placeholder to be fleshed out when wiring the real STAGATE GNN
autoencoder from ``third_party/stagate``. It conforms to
:class:`caust.models.base.BaseSpatialModel` so that, once implemented, it drops
straight into :class:`caust.pipeline.CauST` via ``model_factory``.

Until then, use the working :class:`caust.models.simple.SimpleSpatialModel`.
"""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from .base import BaseSpatialModel


class STAGATEAdapter(BaseSpatialModel):
    def __init__(
        self, n_epochs: int = 1000, device: str = "cpu", random_seed: int = 42
    ):
        self.n_epochs = n_epochs
        self.device = device
        self.random_seed = random_seed
        self.model = None
        self._X: np.ndarray | None = None

    def fit(self, adata: AnnData) -> STAGATEAdapter:
        raise NotImplementedError(
            "STAGATEAdapter is a stub. Implement it against third_party/stagate, "
            "or use caust.models.SimpleSpatialModel for a working backbone."
        )

    def get_embedding(self, adata: AnnData | None = None) -> np.ndarray:
        raise NotImplementedError

    def forward(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _get_expression(self) -> np.ndarray:
        raise NotImplementedError
