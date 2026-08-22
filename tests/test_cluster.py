"""Tests for clustering and evaluation metrics (caust.cluster)."""

from __future__ import annotations

import numpy as np

from caust.cluster import ari, cluster_embedding, nmi


def test_cluster_embedding_recovers_separated_blobs():
    rng = np.random.default_rng(0)
    a = rng.normal(-5, 0.2, (30, 2))
    b = rng.normal(5, 0.2, (30, 2))
    emb = np.vstack([a, b])
    labels = cluster_embedding(emb, n_clusters=2, random_state=0)
    assert labels.shape == (60,)
    assert set(np.unique(labels)) == {0, 1}
    # the two blobs should land in different clusters
    assert labels[0] != labels[-1]


def test_ari_perfect_and_permutation_invariant():
    truth = ["A", "A", "B", "B"]
    same = ["X", "X", "Y", "Y"]  # relabeled but identical partition
    assert ari(truth, same) == 1.0


def test_nmi_bounds():
    truth = ["A", "A", "B", "B"]
    pred = ["A", "B", "A", "B"]
    val = nmi(truth, pred)
    assert 0.0 <= val <= 1.0


def test_ari_handles_string_and_int_labels():
    assert ari([0, 0, 1, 1], ["a", "a", "b", "b"]) == 1.0


def test_cluster_method_eee_and_validation():
    import numpy as np
    import pytest

    from caust.cluster import cluster_embedding

    rng = np.random.default_rng(0)
    emb = np.vstack([rng.normal(0, 1, (20, 3)), rng.normal(6, 1, (20, 3))])
    labels = cluster_embedding(emb, 2, random_state=0, method="eee")
    assert len(set(labels)) == 2
    with pytest.raises(ValueError, match="method must be one of"):
        cluster_embedding(emb, 2, method="nope")
