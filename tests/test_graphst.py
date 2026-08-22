"""GraphST backbone tests (CPU, tiny); skipped when torch is not installed."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from caust.data import make_synthetic_slice  # noqa: E402
from caust.models.graphst import GraphSTModel, knn_graphs, normalized_adj  # noqa: E402


@pytest.fixture(scope="module")
def fitted():
    sl = make_synthetic_slice(grid=8, seed=0)
    model = GraphSTModel(
        latent_dim=8, n_pcs=4, n_epochs=40, device="cpu", random_state=0
    ).fit(sl)
    return sl, model


def test_training_reduces_loss_and_embeds(fitted):
    sl, model = fitted
    assert model.loss_history[-1] < model.loss_history[0]
    z = model.get_embedding()
    assert z.shape == (sl.n_obs, 4) and model.get_embedding() is z


def test_rank1_knockout_matches_forward(fitted):
    sl, model = fitted
    idx = sl.var_names.get_loc("CAUSAL_1")
    X = np.asarray(sl.X, dtype=float).copy()
    X[:, idx] = 0.0
    np.testing.assert_allclose(
        model.get_knockout_embedding(idx), model.forward(X), atol=1e-4
    )
    assert not np.allclose(model.get_knockout_embedding(idx), model.get_embedding())


def test_zero_shot_transfer(fitted):
    _, model = fitted
    other = make_synthetic_slice(grid=8, seed=2)
    assert model.get_embedding(other).shape == (other.n_obs, 4)


def test_graph_helpers():
    coords = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
    directed, adj = knn_graphs(coords, 1)
    assert directed.sum() == 4 and np.array_equal(adj, adj.T)
    A = normalized_adj(adj)
    assert np.allclose(np.diag(A), 1.0) and A.shape == (4, 4)


def test_unfitted_and_backend_registration():
    with pytest.raises(ValueError, match="not fitted"):
        GraphSTModel(device="cpu").forward(np.zeros((2, 2)))
    from caust.config import ExperimentConfig
    from caust.experiment import _model_factory

    cfg = ExperimentConfig.from_dict(
        {
            "name": "t",
            "data": {},
            "selection": {},
            "model": {"backend": "graphst", "device": "cpu"},
        }
    )
    assert isinstance(_model_factory(cfg)(), GraphSTModel)
