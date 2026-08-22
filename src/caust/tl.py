"""Scanpy-style *tools*: ``caust.tl``.

Functions take AnnData objects, do the work, and write results back into
``.var`` / ``.uns`` the way ``scanpy.tl`` does, so CauST slots into an
existing scanpy / SpatialData analysis without learning a new object model.

    import caust
    caust.tl.causal_genes([slice_a, slice_b, slice_c], backbone="stagate")
    caust.tl.select_genes(slice_a, n_top_genes=100)
    sub = slice_a[:, slice_a.var["caust_selected"]]
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData

from .invariance import invariance_scores, select_causal_genes, soft_weights
from .pipeline import CauST

BACKBONES = ("simple", "stagate")


def _factory(backbone: str, **params: Any):
    if backbone == "simple":
        from .models.simple import SimpleSpatialModel

        return lambda: SimpleSpatialModel(**params)
    if backbone == "stagate":
        from .models.stagate import STAGATEModel

        return lambda: STAGATEModel(**params)
    raise ValueError(f"backbone must be one of {BACKBONES}, got {backbone!r}")


def causal_genes(
    adatas: Sequence[AnnData],
    *,
    backbone: str = "simple",
    lam: float = 2.0,
    key_added: str = "caust",
    verbose: bool = False,
    **backbone_params: Any,
) -> pd.DataFrame:
    """Score every shared gene by knockout invariance across ``adatas``.

    Writes, in every slice, ``var[f"{key_added}_score"]`` (the invariance
    score; NaN for genes not shared by all slices), ``var[f"{key_added}_delta"]``
    (that slice's own knockout effect), and ``uns[key_added]`` with the
    settings, the per-slice delta matrix, and the ranked gene table that is
    also returned.
    """
    cs = CauST(model_factory=_factory(backbone, **backbone_params), lam=lam).fit(
        adatas, verbose=verbose
    )
    names = np.asarray(cs.common_genes_)
    deltas = np.asarray(cs.deltas_)
    scores = np.asarray(cs.scores_)
    table = pd.DataFrame(
        {
            "gene": names,
            "score": scores,
            "mean_delta": np.nanmean(deltas, axis=0),
            "std_delta": np.nanstd(deltas, axis=0),
        }
    ).sort_values("score", ascending=False, kind="stable")
    table.index = np.arange(1, len(table) + 1)
    table.index.name = "rank"
    lookup = dict(zip(names, scores))
    for e, adata in enumerate(adatas):
        adata.var[f"{key_added}_score"] = [
            lookup.get(g, np.nan) for g in adata.var_names
        ]
        d = dict(zip(names, deltas[e]))
        adata.var[f"{key_added}_delta"] = [d.get(g, np.nan) for g in adata.var_names]
        adata.uns[key_added] = {
            "backbone": backbone,
            "lam": lam,
            "n_slices": len(adatas),
            "genes": list(map(str, names)),
            "deltas": deltas,
            "ranking": table.reset_index(),
        }
    return table


def select_genes(
    adata: AnnData,
    n_top_genes: int = 100,
    *,
    key: str = "caust",
    lam: float | None = None,
) -> np.ndarray:
    """Flag the top-``n_top_genes`` causal genes in ``var[f"{key}_selected"]``.

    Re-scores with a different ``lam`` from the stored deltas if given, so
    lambda can be tuned without re-running knockouts. Returns the gene names.
    """
    if key not in adata.uns:
        raise KeyError(f"run caust.tl.causal_genes first (no uns[{key!r}])")
    info = adata.uns[key]
    names = np.asarray(info["genes"])
    scores = (
        invariance_scores(np.asarray(info["deltas"]), lam=lam)
        if lam is not None
        else np.asarray(
            [
                adata.var.loc[g, f"{key}_score"] if g in adata.var_names else np.nan
                for g in names
            ]
        )
    )
    chosen = set(names[select_causal_genes(scores, n_top_genes)])
    adata.var[f"{key}_selected"] = [g in chosen for g in adata.var_names]
    return np.asarray([g for g in adata.var_names if g in chosen])


def soft_gene_weights(
    adata: AnnData, *, key: str = "caust", temperature: float = 1.0
) -> np.ndarray:
    """Sigmoid weights for soft reweighting, stored in ``var[f"{key}_weight"]``."""
    scores = adata.var[f"{key}_score"].to_numpy(dtype=float)
    w = soft_weights(scores, temperature=temperature)
    adata.var[f"{key}_weight"] = w
    return w


def spatial_domains(
    adata: AnnData,
    n_domains: int,
    *,
    backbone: str = "simple",
    genes: Sequence[str] | None = None,
    cluster_method: str = "eee",
    random_state: int = 0,
    key_added: str = "caust_domain",
    **backbone_params: Any,
) -> np.ndarray:
    """Train a backbone on (a gene subset of) one slice and cluster it.

    Writes the embedding to ``obsm[f"X_{backbone}"]`` and the labels to
    ``obs[key_added]``; returns the labels.
    """
    from .cluster import cluster_embedding

    sub = adata[:, list(genes)].copy() if genes is not None else adata
    model = _factory(backbone, random_state=random_state, **backbone_params)().fit(sub)
    Z = model.get_embedding()
    labels = cluster_embedding(
        Z, n_domains, random_state=random_state, method=cluster_method
    )
    adata.obsm[f"X_{backbone}"] = Z
    adata.obs[key_added] = pd.Categorical(labels.astype(str))
    return labels
