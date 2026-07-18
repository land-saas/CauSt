"""Clustering + evaluation helpers for spatial domain identification.

The paper uses mclust (an R Gaussian-mixture model) for clustering. To keep the
prototype pure-Python, we default to a scikit-learn Gaussian mixture, which plays
the same role. Swap in an mclust bridge later if exact parity is needed.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture


def cluster_embedding(
    embedding: np.ndarray, n_clusters: int, random_state: int = 42
) -> np.ndarray:
    """Cluster a spatial embedding into ``n_clusters`` domains (labels 0..K-1)."""
    gm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        random_state=random_state,
    )
    return gm.fit_predict(np.asarray(embedding, dtype=float))


def ari(true_labels, pred_labels) -> float:
    """Adjusted Rand Index against manual annotations."""
    return float(adjusted_rand_score(_encode(true_labels), _encode(pred_labels)))


def nmi(true_labels, pred_labels) -> float:
    """Normalized Mutual Information (supplementary metric)."""
    return float(normalized_mutual_info_score(_encode(true_labels), _encode(pred_labels)))


def _encode(labels) -> np.ndarray:
    labels = np.asarray(labels)
    _, inv = np.unique(labels, return_inverse=True)
    return inv
