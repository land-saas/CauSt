"""Offline tests for the DLPFC loader: fake 10x files, no network."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import scipy.sparse as sp

from caust import dlpfc
from caust.dlpfc import (
    DLPFCError,
    fetch_dlpfc_sample,
    load_dlpfc_cohort,
    load_dlpfc_slice,
    read_10x_h5,
)

# Real sample names so SAMPLE_DONORS lookups succeed; the files are fakes.
SAMPLES = ("151507", "151669", "151673")


def write_fake_10x(path, counts, gene_ids, symbols, barcodes):
    """Write a minimal 10x v3 HDF5 file (genes x barcodes CSC)."""
    matrix = sp.csc_matrix(np.asarray(counts))
    with h5py.File(path, "w") as fh:
        grp = fh.create_group("matrix")
        grp.create_dataset("data", data=matrix.data)
        grp.create_dataset("indices", data=matrix.indices)
        grp.create_dataset("indptr", data=matrix.indptr)
        grp.create_dataset("shape", data=np.asarray(matrix.shape, dtype=np.int32))
        grp.create_dataset("barcodes", data=np.array(barcodes, dtype="S"))
        feat = grp.create_group("features")
        feat.create_dataset("id", data=np.array(gene_ids, dtype="S"))
        feat.create_dataset("name", data=np.array(symbols, dtype="S"))


def write_fake_metadata(path, barcodes, layers):
    """Write an R-style TSV: row names in column 0, header one name short."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("layer_guess\timagerow\timagecol\n")
        for i, (bc, layer) in enumerate(zip(barcodes, layers)):
            fh.write(f"{bc}\t{layer}\t{10 + i}\t{100 + i}\n")


@pytest.fixture
def dlpfc_root(tmp_path, monkeypatch):
    """Three fake samples under a tmp data root, checksums disabled."""
    monkeypatch.setattr(dlpfc, "CHECKSUMS", {})
    rng = np.random.default_rng(0)
    gene_ids = [f"ENSG{i:05d}" for i in range(6)]
    # One duplicated symbol pair (DUP) that the cohort loader must drop.
    symbols = ["ALPHA", "BETA", "DUP", "DUP", "GAMMA", "RARE"]
    for s, sample in enumerate(SAMPLES):
        sampledir = tmp_path / sample
        sampledir.mkdir()
        barcodes = [f"BC{s}_{i}-1" for i in range(8)]
        counts = rng.poisson(5.0, size=(6, 8))
        counts[5] = 0  # RARE: never expressed in any slice
        if sample == "151673":
            counts[4] = rng.poisson(80.0, size=8)  # GAMMA: loud only held-out
        write_fake_10x(sampledir / dlpfc.H5_NAME, counts, gene_ids, symbols, barcodes)
        layers = ["Layer1", "Layer2", "Layer3", "WM", "Layer1", "Layer2", "", "WM"]
        write_fake_metadata(sampledir / dlpfc.META_NAME, barcodes, layers)
    return tmp_path


def test_read_10x_h5_transposes_and_labels(dlpfc_root):
    adata = read_10x_h5(dlpfc_root / "151507" / dlpfc.H5_NAME)
    assert adata.shape == (8, 6)  # spots x genes
    assert list(adata.var_names) == [f"ENSG{i:05d}" for i in range(6)]
    assert list(adata.var["symbol"][:2]) == ["ALPHA", "BETA"]
    assert sp.issparse(adata.X)


def test_read_10x_h5_rejects_non_10x(tmp_path):
    bogus = tmp_path / "bogus.h5"
    with h5py.File(bogus, "w") as fh:
        fh.create_dataset("stuff", data=[1, 2, 3])
    with pytest.raises(DLPFCError, match="matrix"):
        read_10x_h5(bogus)


def test_load_slice_attaches_labels_and_flips_coords(dlpfc_root):
    adata = load_dlpfc_slice("151507", dlpfc_root, download=False)
    # The barcode with an empty layer call is dropped.
    assert adata.n_obs == 7
    assert set(adata.obs["domain"]) == {"Layer1", "Layer2", "Layer3", "WM"}
    assert (adata.obs["donor"] == "Br5292").all()
    # x = imagecol, y = -imagerow so plots come out anatomically oriented.
    assert adata.obsm["spatial"][0, 0] == 100.0
    assert adata.obsm["spatial"][0, 1] == -10.0


def test_cohort_pool_is_shared_deduped_and_train_only(dlpfc_root):
    cohort = load_dlpfc_cohort(
        samples=SAMPLES, root=dlpfc_root, n_candidates=10, download=False
    )
    assert len(cohort) == 3
    names = list(cohort[0].var_names)
    assert all(list(a.var_names) == names for a in cohort)
    # Ambiguous symbol dropped entirely; unexpressed gene filtered out.
    assert "DUP" not in names
    assert "RARE" not in names
    # GAMMA is loud only in the held-out slice, which must not shape the pool;
    # it still qualifies (expressed in training) but everything present must
    # simply be expressed in every training slice.
    assert set(names) <= {"ALPHA", "BETA", "GAMMA"}


def test_cohort_respects_n_candidates(dlpfc_root):
    cohort = load_dlpfc_cohort(
        samples=SAMPLES, root=dlpfc_root, n_candidates=2, download=False
    )
    assert cohort[0].n_vars == 2


def test_missing_file_without_download_raises(dlpfc_root):
    (dlpfc_root / "151507" / dlpfc.H5_NAME).unlink()
    with pytest.raises(DLPFCError, match="download=False"):
        fetch_dlpfc_sample("151507", dlpfc_root, download=False)


def test_unknown_sample_raises(dlpfc_root):
    with pytest.raises(DLPFCError, match="unknown DLPFC sample"):
        fetch_dlpfc_sample("999999", dlpfc_root, download=False)


def test_checksum_mismatch_raises(dlpfc_root, monkeypatch):
    monkeypatch.setattr(dlpfc, "CHECKSUMS", {f"151507/{dlpfc.H5_NAME}": "0" * 64})
    with pytest.raises(DLPFCError, match="checksum mismatch"):
        fetch_dlpfc_sample("151507", dlpfc_root, download=False)


def test_cohort_needs_two_samples(dlpfc_root):
    with pytest.raises(DLPFCError, match="at least 2"):
        load_dlpfc_cohort(samples=["151507"], root=dlpfc_root, download=False)


def test_build_cohort_dispatches_dlpfc(dlpfc_root):
    from caust.config import ExperimentConfig
    from caust.experiment import build_cohort

    cfg = ExperimentConfig.from_dict(
        {
            "name": "t",
            "data": {
                "source": "dlpfc",
                "root": str(dlpfc_root),
                "samples": list(SAMPLES),
                "n_candidates": 3,
                "download": False,
            },
            "selection": {"n_top_genes": 2},
        }
    )
    cohort = build_cohort(cfg)
    assert len(cohort) == 3 and cohort[0].n_vars == 3


def test_build_cohort_wraps_dlpfc_errors(dlpfc_root):
    from caust.config import ExperimentConfig
    from caust.experiment import ExperimentError, build_cohort

    cfg = ExperimentConfig.from_dict(
        {
            "name": "t",
            "data": {"source": "dlpfc", "root": str(dlpfc_root), "bogus_key": 1},
            "selection": {"n_top_genes": 2},
        }
    )
    with pytest.raises(ExperimentError, match="invalid data"):
        build_cohort(cfg)
