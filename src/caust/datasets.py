"""Additional benchmark datasets, distributed as AnnData (``.h5ad``) files.

Every dataset here is a set of tissue sections with manual spatial-domain
labels, so it can serve the same role as DLPFC: select genes on some
sections, evaluate on others. Files come from the SDMBench (Yuan et al.,
Nat Methods 2024) figshare distribution (CC BY 4.0), are cached under a local
data root, and are verified against pinned SHA-256 checksums.

================  ========  ======  ============================================
dataset           sections  genes   labels (``obs['domain']``)
================  ========  ======  ============================================
merfish           5         155     8 hypothalamic regions (BASS annotation) --
                                    Bregma -0.04 .. -0.24 of one animal; the
                                    expression is volume-normalized, not counts
starmap_pfc       3         166     4 cortical layers (BASS annotation)
starmap_vc        1         1,020   7 domains (L1-L6, corpus callosum, HPC)
================  ========  ======  ============================================

Loaded objects carry ``obs['domain']``, ``obs['sample']``, ``obs['donor']``
(the section itself when all sections come from one animal -- documented per
dataset), and ``obsm['spatial']``; ``X`` is left as distributed.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
from anndata import AnnData


class DatasetError(RuntimeError):
    """Raised when a dataset is unknown, missing, corrupt, or unfetchable."""


@dataclass(frozen=True)
class Section:
    name: str
    filename: str
    url: str
    sha256: str
    donor: str


@dataclass(frozen=True)
class Dataset:
    key: str
    root: str
    label_key: str
    sections: tuple[Section, ...]
    citation: str
    counts_are_integers: bool = True
    notes: str = ""


_FIG = "https://ndownloader.figshare.com/files/{id}"

DATASETS: dict[str, Dataset] = {
    "merfish": Dataset(
        key="merfish",
        root="data/MERFISH",
        label_key="Region",
        counts_are_integers=False,
        citation=(
            "Moffitt et al., Science 2018 (data); Li & Zhou, Genome Biology 2022 "
            "(region annotation); SDMBench distribution, figshare 22565170"
        ),
        notes=(
            "Five consecutive Bregma sections of one animal: 'donor' is the "
            "section, so cross-donor here means cross-section."
        ),
        sections=tuple(
            Section(
                f"bregma_{b}", f"MERFISH_{b}.h5ad", _FIG.format(id=i), h, f"bregma_{b}"
            )
            for b, i, h in (
                (
                    "0.04",
                    40038526,
                    "0fa1294fa2c7c0d30770f172144c5c2fb236ab992e1f96f1c8f164c9aa049e40",
                ),
                (
                    "0.09",
                    40038529,
                    "3849b4ca283fc773592f7209553208be9a33da8a5ca76164e1398d53f81eb235",
                ),
                (
                    "0.14",
                    40038532,
                    "f6e7143cf13e135622f675b4729ac88baccbf76729ede9d7b812c7ff119526d0",
                ),
                (
                    "0.19",
                    40038535,
                    "4ef4a008a5842b547597aa58a43823bfa9847c569b29b1651e8c212408c4a6a8",
                ),
                (
                    "0.24",
                    40038538,
                    "d0816e0a71d99d8425f4090e3bf8e4827f216e8c1b62ebe7cdbadca3b814b3a2",
                ),
            )
        ),
    ),
    "starmap_pfc": Dataset(
        key="starmap_pfc",
        root="data/STARmap_PFC",
        label_key="Region",
        citation=(
            "Wang et al., Science 2018 (data); Li & Zhou, Genome Biology 2022 "
            "(layer annotation); SDMBench distribution, figshare 22565200"
        ),
        sections=(
            Section(
                "BZ5",
                "20180417_BZ5_control.h5ad",
                _FIG.format(id=40038637),
                "c95d43875740ae917c46786920c1d26bb34d3427149423258cc5654b068f2e70",
                "BZ5",
            ),
            Section(
                "BZ9",
                "20180419_BZ9_control.h5ad",
                _FIG.format(id=40038640),
                "88bee47f80aa854d7265ebc95c43758493506c0f2894273581313e2d326f6e6f",
                "BZ9",
            ),
            Section(
                "BZ14",
                "20180424_BZ14_control.h5ad",
                _FIG.format(id=40038643),
                "c8d10a8c8a7962e314367004c16748a84d12edf3fb7eeab8fdefa96a8b16c8b2",
                "BZ14",
            ),
        ),
    ),
    "starmap_vc": Dataset(
        key="starmap_vc",
        root="data/STARmap_VC",
        label_key="Region",
        citation="Wang et al., Science 2018; SDMBench distribution, figshare 22565209",
        notes="A single section: within-slice evaluation only.",
        sections=(
            Section(
                "BY3_1k",
                "STARmap_20180505_BY3_1k.h5ad",
                _FIG.format(id=40038688),
                "a62c1804f4d1d513198f2917f185dcfd066a51fc6790c8ee3fe84a5f62949688",
                "BY3",
            ),
        ),
    ),
}

_LABEL_CANDIDATES = ("Region", "domain", "label", "layer", "ground_truth")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}")
    # figshare refuses the default urllib/curl user agents.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (caust)"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp, open(part, "wb") as out:
            shutil.copyfileobj(resp, out)
    except Exception as exc:
        part.unlink(missing_ok=True)
        raise DatasetError(f"failed to download {url}: {exc}") from None
    part.replace(dest)


def _get(dataset: str, section: str) -> tuple[Dataset, Section]:
    ds = DATASETS.get(dataset)
    if ds is None:
        raise DatasetError(f"unknown dataset {dataset!r}; known: {', '.join(DATASETS)}")
    sec = next((s for s in ds.sections if s.name == section), None)
    if sec is None:
        raise DatasetError(
            f"unknown section {section!r} for {dataset}; known: "
            + ", ".join(s.name for s in ds.sections)
        )
    return ds, sec


def fetch_section(
    dataset: str, section: str, root: str | Path | None = None, *, download: bool = True
) -> Path:
    """Local path of one section's file, fetching and verifying if needed."""
    ds, sec = _get(dataset, section)
    path = Path(root or ds.root) / sec.filename
    if not path.is_file():
        if not download:
            raise DatasetError(f"{path} is missing and download=False")
        _download(sec.url, path)
    if sec.sha256:
        actual = _sha256(path)
        if actual != sec.sha256:
            raise DatasetError(
                f"checksum mismatch for {path} (expected {sec.sha256[:12]}..., "
                f"got {actual[:12]}...); delete the file and re-download"
            )
    return path


def _label_column(adata: AnnData, preferred: str) -> str:
    for c in (preferred, *_LABEL_CANDIDATES):
        if c and c in adata.obs.columns:
            return c
    raise DatasetError(
        f"no domain label column among {_LABEL_CANDIDATES}: {list(adata.obs.columns)}"
    )


def load_section(
    dataset: str, section: str, root: str | Path | None = None, *, download: bool = True
) -> AnnData:
    """One labelled section as AnnData (``obs['domain'/'sample'/'donor']``)."""
    ds, sec = _get(dataset, section)
    path = fetch_section(dataset, section, root, download=download)
    adata = ad.read_h5ad(path)
    col = _label_column(adata, ds.label_key)
    keep = adata.obs[col].notna().to_numpy()
    adata = adata[keep].copy()
    adata.obs["domain"] = adata.obs[col].astype(str).to_numpy()
    adata.obs["sample"] = sec.name
    adata.obs["donor"] = sec.donor
    if "spatial" not in adata.obsm:
        for xk, yk in (("X", "Y"), ("x", "y"), ("imagecol", "imagerow")):
            if xk in adata.obs and yk in adata.obs:
                adata.obsm["spatial"] = np.column_stack(
                    [adata.obs[xk].to_numpy(float), adata.obs[yk].to_numpy(float)]
                )
                break
        else:
            raise DatasetError(f"{path}: no spatial coordinates found")
    adata.var_names_make_unique()
    adata.var["symbol"] = adata.var_names
    return adata


def load_dataset(
    dataset: str,
    sections: Sequence[str] | None = None,
    root: str | Path | None = None,
    *,
    download: bool = True,
) -> list[AnnData]:
    """All (or the named) sections of a dataset, aligned on shared genes."""
    if dataset not in DATASETS:
        raise DatasetError(f"unknown dataset {dataset!r}; known: {', '.join(DATASETS)}")
    ds = DATASETS[dataset]
    names = (
        [s.name for s in ds.sections]
        if sections is None
        else [str(s) for s in sections]
    )
    slices = [load_section(dataset, n, root, download=download) for n in names]
    shared = set(slices[0].var_names)
    for a in slices[1:]:
        shared &= set(a.var_names)
    if not shared:
        raise DatasetError("sections share no genes")
    order = [g for g in slices[0].var_names if g in shared]
    return [a[:, order].copy() for a in slices]
