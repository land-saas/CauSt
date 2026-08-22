"""Spatial-embedding backbones for CauST.

``SimpleSpatialModel`` needs only numpy/scikit-learn. The GNN backbones live in
``caust.models.stagate`` (STAGATE) and ``caust.models.graphst`` (GraphST) and
need the ``stagate`` extra (PyTorch); import them from their modules.
"""

from .base import BaseSpatialModel
from .simple import SimpleSpatialModel

__all__ = ["BaseSpatialModel", "SimpleSpatialModel"]
