"""STAGATE backbone tests (CPU, tiny); skipped when torch is not installed."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from caust.data import make_synthetic_slice  # noqa: E402
from caust.models.base import BaseSpatialModel  # noqa: E402
from caust.models.stagate import STAGATEModel, resolve_device  # noqa: E402


@pytest.fixture(scope="module")
def fitted():
    sl = make_synthetic_slice(grid=8, seed=0)
    model = STAGATEModel(
        hidden_dim=16, latent_dim=4, n_epochs=30, device="cpu", random_state=0
    ).fit(sl)
    return sl, model


def test_training_reduces_reconstruction_loss(fitted):
    _, model = fitted
    assert len(model.loss_history) == 30
    assert model.loss_history[-1] < model.loss_history[0]


def test_embedding_shape_and_cache(fitted):
    sl, model = fitted
    z = model.get_embedding()
    assert z.shape == (sl.n_obs, 4)
    assert model.get_embedding() is z  # cached for the knockout loop


def test_knockout_matches_generic_path_and_changes_embedding(fitted):
    sl, model = fitted
    idx = sl.var_names.get_loc("CAUSAL_0")
    fast = model.get_knockout_embedding(idx)
    generic = BaseSpatialModel.get_knockout_embedding(model, idx)
    np.testing.assert_allclose(fast, generic, atol=1e-5)
    assert not np.allclose(fast, model.get_embedding())


def test_zero_shot_transfer_uses_target_graph(fitted):
    _, model = fitted
    other = make_synthetic_slice(grid=8, seed=3)
    z = model.get_embedding(other)
    assert z.shape == (other.n_obs, 4)
    assert "spatial_connectivities" in other.obsp


def test_parameters_are_frozen_after_fit(fitted):
    _, model = fitted
    assert all(not p.requires_grad for p in model.net.parameters())


def test_seed_makes_training_deterministic_on_cpu():
    sl = make_synthetic_slice(grid=6, seed=1)
    kw = dict(hidden_dim=8, latent_dim=3, n_epochs=10, device="cpu", random_state=7)
    a = STAGATEModel(**kw).fit(sl).get_embedding()
    b = STAGATEModel(**kw).fit(sl).get_embedding()
    np.testing.assert_allclose(a, b)


def test_unfitted_raises():
    m = STAGATEModel(device="cpu")
    with pytest.raises(ValueError, match="not fitted"):
        m.forward(np.zeros((2, 2)))


def test_resolve_device_explicit():
    assert str(resolve_device("cpu")) == "cpu"


def test_experiment_runner_accepts_stagate_backend(tmp_path):
    from caust.config import ExperimentConfig
    from caust.experiment import run_experiment

    cfg = ExperimentConfig.from_dict(
        {
            "name": "stagate-tiny",
            "data": {"source": "synthetic", "n_slices": 2, "grid": 6, "n_causal": 3},
            "model": {
                "backend": "stagate",
                "hidden_dim": 8,
                "latent_dim": 3,
                "n_epochs": 5,
                "device": "cpu",
            },
            "selection": {"n_top_genes": 3, "lam": 1.0},
            "evaluation": {"n_restarts": 1, "cluster_method": "eee"},
        }
    )
    metrics = run_experiment(cfg, tmp_path, write=False)
    assert set(metrics["held_out"]) == {"all_genes", "hvg", "caust"}
