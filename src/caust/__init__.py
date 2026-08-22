"""CauST -- Causal Gene Intervention for Robust Spatial Domain Identification.

A backbone-agnostic toolkit that selects genes by *in-silico knockout* on a
frozen spatial model and *cross-donor invariance*, rather than by variance
(HVGs). See :class:`caust.pipeline.CauST` for the end-to-end entry point.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

from . import pl, tl
from .cluster import ari, cluster_embedding, nmi
from .config import ExperimentConfig, load_config
from .data import make_synthetic_cohort, make_synthetic_slice
from .experiment import run_experiment, verify_run
from .graph import build_spatial_graph, normalized_adjacency
from .intervention import knockout_scores
from .invariance import invariance_scores, select_causal_genes, soft_weights
from .models.base import BaseSpatialModel
from .models.simple import SimpleSpatialModel
from .pipeline import CauST
from .repro import collect_provenance, deterministic, set_global_seeds

try:
    __version__ = _dist_version("caust")
except PackageNotFoundError:  # pragma: no cover - running from a bare checkout
    __version__ = "0+unknown"

__all__ = [
    "CauST",
    "tl",
    "pl",
    "BaseSpatialModel",
    "SimpleSpatialModel",
    "ExperimentConfig",
    "load_config",
    "run_experiment",
    "verify_run",
    "deterministic",
    "set_global_seeds",
    "collect_provenance",
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
