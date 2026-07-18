"""End-to-end CauST pipeline.

Given several tissue slices (one AnnData per donor/slice), CauST:

  1. trains a fresh frozen backbone on each slice,
  2. scores every common gene by in-silico knockout on each slice (Step 1),
  3. combines the per-slice effects into an invariance score (Step 2),
  4. exposes the causal gene set for retraining downstream models (Step 3).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from anndata import AnnData

from .intervention import knockout_scores
from .invariance import invariance_scores, select_causal_genes, soft_weights
from .models.base import BaseSpatialModel
from .models.simple import SimpleSpatialModel


def _default_model_factory() -> BaseSpatialModel:
    return SimpleSpatialModel()


class CauST:
    """Causal gene selection across tissue slices.

    Parameters
    ----------
    model_factory
        Zero-arg callable returning a fresh, unfitted
        :class:`~caust.models.base.BaseSpatialModel`. One is trained per slice.
        Defaults to the bundled :class:`SimpleSpatialModel`.
    lam
        Invariance penalty lambda (Eq. 3). Paper's best setting is 2.0.
    """

    def __init__(
        self,
        model_factory: Callable[[], BaseSpatialModel] = _default_model_factory,
        lam: float = 2.0,
    ):
        self.model_factory = model_factory
        self.lam = lam
        # populated by fit()
        self.common_genes_: np.ndarray | None = None
        self.deltas_: np.ndarray | None = None  # (E, G_common)
        self.scores_: np.ndarray | None = None  # (G_common,)
        self.models_: list[BaseSpatialModel] = []

    def fit(self, adatas: Sequence[AnnData], verbose: bool = False) -> "CauST":
        """Train one backbone per slice and score genes by knockout invariance."""
        if len(adatas) == 0:
            raise ValueError("provide at least one AnnData slice.")

        # Genes must be aligned across slices: use their intersection.
        common = set(adatas[0].var_names)
        for a in adatas[1:]:
            common &= set(a.var_names)
        if not common:
            raise ValueError("slices share no common genes (check var_names).")
        # Deterministic order from the first slice.
        self.common_genes_ = np.array(
            [g for g in adatas[0].var_names if g in common]
        )

        deltas = []
        self.models_ = []
        for e, adata in enumerate(adatas):
            if verbose:
                print(f"[slice {e + 1}/{len(adatas)}] training backbone...")
            sub = adata[:, self.common_genes_].copy()
            model = self.model_factory()
            model.fit(sub)
            self.models_.append(model)
            if verbose:
                print(f"[slice {e + 1}/{len(adatas)}] knockout scoring...")
            deltas.append(knockout_scores(model, verbose=verbose))

        self.deltas_ = np.vstack(deltas)  # (E, G_common)
        self.scores_ = invariance_scores(self.deltas_, lam=self.lam)
        return self

    def _check_fitted(self) -> None:
        if self.scores_ is None:
            raise ValueError("CauST is not fitted; call fit() first.")

    def select_genes(self, n_top_genes: int) -> np.ndarray:
        """Names of the top-K causally invariant genes (Step 3, hard filter)."""
        self._check_fitted()
        idx = select_causal_genes(self.scores_, n_top_genes)
        return self.common_genes_[idx]

    def soft_weights(self, temperature: float = 1.0) -> np.ndarray:
        """Sigmoid weights over common genes (Step 3, soft reweighting)."""
        self._check_fitted()
        return soft_weights(self.scores_, temperature=temperature)

    def ranking(self) -> list[tuple[str, float]]:
        """All common genes as (name, score) pairs, best first."""
        self._check_fitted()
        order = np.argsort(-self.scores_, kind="stable")
        return [
            (str(self.common_genes_[i]), float(self.scores_[i])) for i in order
        ]

    def transform(self, adata: AnnData, n_top_genes: int) -> AnnData:
        """Subset an AnnData to the causal gene set for downstream retraining."""
        genes = self.select_genes(n_top_genes)
        keep = [g for g in genes if g in set(adata.var_names)]
        return adata[:, keep].copy()
