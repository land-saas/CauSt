"""CauST -- Causal Gene Intervention for Robust Spatial Domain Identification.

A backbone-agnostic toolkit that selects genes by *in-silico knockout* on a
frozen spatial model and *cross-donor invariance*, rather than by variance
(HVGs). See :class:`caust.pipeline.CauST` for the end-to-end entry point.
"""

from __future__ import annotations

from .cluster import ari, cluster_embedding, nmi
from .data import make_synthetic_cohort, make_synthetic_slice
from .graph import build_spatial_graph, normalized_adjacency
from .intervention import knockout_scores
from .invariance import invariance_scores, select_causal_genes, soft_weights
from .models.base import BaseSpatialModel
from .models.simple import SimpleSpatialModel
from .pipeline import CauST

__version__ = "0.1.0"

__all__ = [
    "CauST",
    "BaseSpatialModel",
    "SimpleSpatialModel",
    "knockout_scores",
    "invariance_scores",
    "select_causal_genes",
    "soft_weights",
    "build_spatial_graph",
    "normalized_adjacency",
    "cluster_embedding",
    "ari",
    "nmi",
    "make_synthetic_slice",
    "make_synthetic_cohort",
    "__version__",
]
