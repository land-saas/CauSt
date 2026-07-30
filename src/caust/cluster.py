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
    # The k-means initialization inside GaussianMixture computes pairwise
    # distances via BLAS matmul, which raises spurious divide/overflow/invalid
    # FP flags on some backends (macOS Accelerate + numpy 2.x) even when the
    # input and the resulting labels are entirely finite.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        labels: np.ndarray = np.asarray(
            gm.fit_predict(np.asarray(embedding, dtype=float))
        )
    return labels


def ari(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Adjusted Rand Index against manual annotations."""
    return float(adjusted_rand_score(_encode(true_labels), _encode(pred_labels)))


def nmi(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """Normalized Mutual Information (supplementary metric)."""
    return float(
        normalized_mutual_info_score(_encode(true_labels), _encode(pred_labels))
    )


def _encode(labels: np.ndarray) -> np.ndarray:
    arr = np.asarray(labels)
    _, inv = np.unique(arr, return_inverse=True)
    encoded: np.ndarray = np.asarray(inv)
    return encoded
