"""Spatial-embedding backbones for CauST."""
from .base import BaseSpatialModel
from .simple import SimpleSpatialModel
from .stagate_adpater import STAGATEAdapter

__all__ = ["BaseSpatialModel", "SimpleSpatialModel", "STAGATEAdapter"]
