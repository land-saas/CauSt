"""Steps 2 & 3 -- cross-slice invariance scoring and causal gene selection.

Step 2 combines per-slice knockout effects into a single invariance score that
rewards genes with a *large and stable* effect across donors (Eq. 3):

    s_inv^(g) = mean_e[ delta^(g,e) ] - lambda * std_e[ delta^(g,e) ]

Step 3 turns the score into a gene set: hard top-K filtering, or soft sigmoid
reweighting of the expression matrix.
"""
from __future__ import annotations

import numpy as np


def invariance_scores(deltas: np.ndarray, lam: float = 2.0) -> np.ndarray:
    """Combine per-slice knockout effects into an invariance score.

    Parameters
    ----------
    deltas : np.ndarray, shape (E, G)
        Per-slice knockout effects; row e is delta^(.,e) for slice e. ``nan``
        entries (genes not scored in a slice) are ignored.
    lam : float
        Invariance penalty weight lambda >= 0. ``lam=0`` reduces to mean effect
        ("High-delta" baseline); larger lambda demands cross-donor stability.

    Returns
    -------
    s_inv : np.ndarray, shape (G,)
        Invariance score per gene. Genes with no valid slice get ``-inf``.
    """
    deltas = np.atleast_2d(np.asarray(deltas, dtype=float))
    if lam < 0:
        raise ValueError("lam (lambda) must be >= 0.")
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(deltas, axis=0)
        std = np.nanstd(deltas, axis=0)
    s = mean - lam * std
    s[np.isnan(mean)] = -np.inf
    return s


def select_causal_genes(scores: np.ndarray, n_top_genes: int) -> np.ndarray:
    """Return indices of the top-K genes by invariance score (descending)."""
    scores = np.asarray(scores, dtype=float)
    k = int(min(n_top_genes, np.isfinite(scores).sum()))
    order = np.argsort(-scores, kind="stable")
    return order[:k]


def soft_weights(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Sigmoid gene weights in (0, 1) from standardized invariance scores.

    Used for soft reweighting: ``X_hat = X @ diag(w)`` (Step 3, soft variant).
    """
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    w = np.zeros_like(scores)
    if finite.sum() == 0:
        return w
    vals = scores[finite]
    z = (vals - vals.mean()) / (vals.std() + 1e-12)
    w[finite] = 1.0 / (1.0 + np.exp(-z / max(temperature, 1e-12)))
    return w
