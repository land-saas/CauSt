"""Dataset registry and h5ad loader, exercised offline on a fake section."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from caust import datasets
from caust.datasets import DATASETS, DatasetError, load_dataset, load_section


def _fake(path, n=30, g=12, label="Region", with_spatial=True, nan_one=True):
    rng = np.random.default_rng(0)
    a = ad.AnnData(rng.poisson(3.0, size=(n, g)).astype(np.float32))
    a.var_names = [f"g{i}" for i in range(g)]
    a.obs_names = [f"c{i}" for i in range(n)]
    labs = np.array(["A", "B", "C"])[rng.integers(0, 3, n)].astype(object)
    if nan_one:
        labs[0] = np.nan
    a.obs[label] = pd.Categorical(labs)
    if with_spatial:
        a.obsm["spatial"] = rng.normal(size=(n, 2))
    else:
        a.obs["X"], a.obs["Y"] = rng.normal(size=n), rng.normal(size=n)
    a.write_h5ad(path)


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    ds = DATASETS["starmap_pfc"]
    for sec in ds.sections:
        _fake(tmp_path / sec.filename, with_spatial=sec.name != "BZ9")
    monkeypatch.setattr(
        datasets,
        "DATASETS",
        {
            **DATASETS,
            "starmap_pfc": ds.__class__(
                **{
                    **ds.__dict__,
                    "sections": tuple(
                        sec.__class__(**{**sec.__dict__, "sha256": ""})
                        for sec in ds.sections
                    ),
                }
            ),
        },
    )
    return tmp_path


def test_registry_is_pinned():
    for key, ds in DATASETS.items():
        assert ds.sections, key
        for sec in ds.sections:
            assert len(sec.sha256) == 64 and sec.url.startswith("https://")


def test_load_section_labels_and_coords(fake_root):
    a = load_section("starmap_pfc", "BZ5", fake_root, download=False)
    assert a.n_obs == 29  # the NaN-labelled cell is dropped
    assert set(a.obs["domain"]) <= {"A", "B", "C"}
    assert (a.obs["sample"] == "BZ5").all() and (a.obs["donor"] == "BZ5").all()
    b = load_section("starmap_pfc", "BZ9", fake_root, download=False)
    assert b.obsm["spatial"].shape == (29, 2)  # built from obs X/Y


def test_load_dataset_aligns_and_validates(fake_root):
    slices = load_dataset("starmap_pfc", root=fake_root, download=False)
    assert len(slices) == 3 and all(
        list(s.var_names) == list(slices[0].var_names) for s in slices
    )
    with pytest.raises(DatasetError, match="unknown dataset"):
        load_dataset("nope")
    with pytest.raises(DatasetError, match="unknown section"):
        load_section("starmap_pfc", "BZ99", fake_root, download=False)
    with pytest.raises(DatasetError, match="download=False"):
        load_section("merfish", "bregma_0.04", fake_root, download=False)


def test_checksum_mismatch(tmp_path):
    sec = DATASETS["starmap_vc"].sections[0]
    _fake(tmp_path / sec.filename)
    with pytest.raises(DatasetError, match="checksum mismatch"):
        load_section("starmap_vc", sec.name, tmp_path, download=False)


def test_transfer_cohort_from_registry(fake_root):
    from caust.config import ExperimentConfig
    from caust.experiment import build_cohort
    from caust.transfer import load_transfer_cohort

    cfg = ExperimentConfig.from_dict(
        {
            "name": "t",
            "data": {
                "source": "starmap_pfc",
                "root": str(fake_root),
                "download": False,
                "n_hvg_rank": 12,
            },
            "selection": {"n_top_genes": 2},
        }
    )
    slices = load_transfer_cohort(cfg)
    assert len(slices) == 3 and "hvg_rank" in slices[0].var
    cohort = build_cohort(cfg)
    assert len(cohort) == 3 and cohort[0].obs["domain"].nunique() <= 3
