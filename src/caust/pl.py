"""Scanpy-style *plots*: ``caust.pl`` (needs the ``viz`` extra)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from anndata import AnnData


def _plt() -> Any:
    try:
        import matplotlib
    except ImportError:
        raise RuntimeError('plots need matplotlib: pip install "caust[viz]"') from None
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def ranking(
    adata: AnnData,
    n: int = 20,
    *,
    key: str = "caust",
    markers: Sequence[str] | None = None,
    ax: Any = None,
    show: bool = False,
) -> Any:
    """Horizontal bar plot of the top-``n`` genes by invariance score."""
    plt = _plt()
    table = adata.uns[key]["ranking"].head(n)
    genes = list(table["gene"])[::-1]
    vals = np.asarray(table["score"], dtype=float)[::-1]
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 0.3 * n + 1))
    ax.barh(genes, vals, color="#2563eb")
    star = set(markers or [])
    for i, g in enumerate(genes):
        if g in star:
            ax.text(vals[i], i, " ★", color="#b91c1c", va="center")
    ax.set(xlabel="invariance score", title=f"Top-{n} causally invariant genes")
    if show:
        plt.show()
    return ax


def domains(
    adata: AnnData,
    key: str = "caust_domain",
    *,
    spatial_key: str = "spatial",
    ax: Any = None,
    title: str | None = None,
    s: float = 8,
    show: bool = False,
) -> Any:
    """Scatter of spots coloured by a categorical ``obs`` column."""
    plt = _plt()
    coords = np.asarray(adata.obsm[spatial_key])
    labels = adata.obs[key].astype(str).to_numpy()
    cats = sorted(set(labels))
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    cmap = plt.get_cmap("tab10" if len(cats) <= 10 else "tab20")
    for i, c in enumerate(cats):
        m = labels == c
        ax.scatter(coords[m, 0], coords[m, 1], s=s, color=cmap(i % cmap.N), label=c)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title or key)
    ax.legend(fontsize=7, markerscale=2, loc="center left", bbox_to_anchor=(1, 0.5))
    if show:
        plt.show()
    return ax
