"""Step 1 -- in-silico gene knockout scoring.

For a frozen spatial model f and a gene g, the knockout effect on slice e is the
mean per-spot shift of the embedding when g is silenced (Eq. 1-2 of the paper):

    X~^(g) = X with column g set to 0
    delta^(g,e) = (1/N) * sum_i || f(X, A)_i - f(X~^(g), A)_i ||_2
"""
from __future__ import annotations

import numpy as np

from .models.base import BaseSpatialModel


def knockout_scores(
    model: BaseSpatialModel,
    gene_indices: np.ndarray | list[int] | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """Compute delta^(g,e) for a single fitted model (one slice).

    Parameters
    ----------
    model
        A fitted :class:`~caust.models.base.BaseSpatialModel`.
    gene_indices
        Genes to score. Defaults to all genes. Genes not scored get ``nan``.
    verbose
        Print progress every 200 genes.

    Returns
    -------
    delta : np.ndarray, shape (G,)
        Mean per-spot embedding shift for each gene (``nan`` where not scored).
    """
    X = model._get_expression()
    n_spots, n_genes = X.shape
    base = model.get_embedding()

    if gene_indices is None:
        gene_indices = np.arange(n_genes)
    gene_indices = np.asarray(gene_indices, dtype=int)

    delta = np.full(n_genes, np.nan, dtype=float)
    for count, g in enumerate(gene_indices):
        Zg = model.get_knockout_embedding(int(g))
        delta[g] = np.linalg.norm(base - Zg, axis=1).mean()
        if verbose and count % 200 == 0:
            print(f"  knockout scoring {count}/{len(gene_indices)} genes")
    return delta
