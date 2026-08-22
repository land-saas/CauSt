"""Clustering + evaluation helpers for spatial domain identification.

The paper uses mclust (an R Gaussian-mixture model) for clustering. To keep the
prototype pure-Python, we default to a scikit-learn Gaussian mixture, which plays
the same role. Swap in an mclust bridge later if exact parity is needed.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture

#: Clustering back-ends. ``"eee"`` is the Python equivalent of mclust's EEE
#: model (equal volume, shape and orientation = one shared full covariance),
#: which STAGATE and the CauST proposal use; ``"full"`` is the per-component
#: covariance used by the synthetic demo; ``"mclust"`` calls R's mclust through
#: rpy2 for exact parity when R is available.
CLUSTER_METHODS = ("full", "eee", "mclust")


def cluster_embedding(
    embedding: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    method: str = "full",
) -> np.ndarray:
    """Cluster a spatial embedding into ``n_clusters`` domains (labels 0..K-1)."""
    if method not in CLUSTER_METHODS:
        raise ValueError(f"method must be one of {CLUSTER_METHODS}, got {method!r}")
    if method == "mclust":
        return _mclust(embedding, n_clusters, random_state)
    gm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="tied" if method == "eee" else "full",
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


def _mclust(embedding: np.ndarray, n_clusters: int, random_state: int) -> np.ndarray:
    """R mclust (model EEE) via rpy2, exactly as the STAGATE tutorials do."""
    try:
        import rpy2.robjects as robjects
        from rpy2.robjects import numpy2ri
    except ImportError:  # pragma: no cover - optional R bridge
        raise ImportError(
            "method='mclust' needs R, the mclust package, and rpy2; "
            "use method='eee' for the pure-Python equivalent"
        ) from None
    numpy2ri.activate()
    robjects.r.library("mclust")
    robjects.r["set.seed"](random_state)
    res = robjects.r["Mclust"](
        numpy2ri.numpy2rpy(np.asarray(embedding, dtype=float)), n_clusters, "EEE"
    )
    labels: np.ndarray = np.asarray(res[-2]).astype(int) - 1
    return labels
