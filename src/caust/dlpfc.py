"""Real data: the spatialLIBD human DLPFC cohort (Maynard et al., 2021).

Twelve 10x Visium slices of human dorsolateral prefrontal cortex from three
donors, with manual cortical-layer annotations (L1-L6 + white matter). This is
the standard benchmark for spatial domain identification, and its multi-donor
structure is exactly what CauST needs: genes can be selected on some donors and
evaluated on a donor the selection never saw.

Two files per sample, fetched on demand and cached under a local data root:

- ``filtered_feature_bc_matrix.h5`` -- raw counts (10x Cell Ranger v3 HDF5),
  from the spatialLIBD AWS mirror.
- ``metadata.tsv`` -- per-spot manual layer labels plus image coordinates, from
  the SEDR benchmark mirror of the spatialLIBD annotations.

Downloads are verified against pinned SHA-256 checksums, so the inputs of a
DLPFC run are as fixed as its code and config.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

#: Sample -> donor for the full 12-sample cohort. Donor is the invariance unit:
#: adjacent slices from one donor are near-replicates, so a held-out *donor*
#: (not merely a held-out slice) is the meaningful generalization test.
SAMPLE_DONORS = {
    "151507": "Br5292",
    "151508": "Br5292",
    "151509": "Br5292",
    "151510": "Br5292",
    "151669": "Br5595",
    "151670": "Br5595",
    "151671": "Br5595",
    "151672": "Br5595",
    "151673": "Br8100",
    "151674": "Br8100",
    "151675": "Br8100",
    "151676": "Br8100",
}

_H5_URL = (
    "https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/"
    "{sample}_filtered_feature_bc_matrix.h5"
)
_META_URL = (
    "https://raw.githubusercontent.com/JinmiaoChenLab/SEDR_analyses/"
    "master/data/DLPFC/{sample}/metadata.tsv"
)

H5_NAME = "filtered_feature_bc_matrix.h5"
META_NAME = "metadata.tsv"

#: SHA-256 of every remote file this module will fetch, keyed by
#: ``<sample>/<filename>``. Extend when adding samples to a config.
CHECKSUMS: dict[str, str] = {
    "151507/filtered_feature_bc_matrix.h5": (
        "686521ab02a68dfdd810c3cde26daea2257d016c8047859877ffb8c7ac5bdd01"
    ),
    "151507/metadata.tsv": (
        "0198f6d887d9360918a1c4e20bf009c426af31f7f5fa7326e1ca0eb4c15027a2"
    ),
    "151508/filtered_feature_bc_matrix.h5": (
        "48ab74b340c79f19070e433b8b7db67932a6e889d421a691f70293587fff0621"
    ),
    "151508/metadata.tsv": (
        "8d54300a7a767dc2d9c10d8d194cc46e8aa2b0e7c388c89c92b6da538e3aa118"
    ),
    "151509/filtered_feature_bc_matrix.h5": (
        "35a4850aebdfdccb9fdf2f79cebeebc037684a79c03adb4bcdffa7c7c68486fc"
    ),
    "151509/metadata.tsv": (
        "20fc26f68689fb98def7f6c0b1541cfcccd2f5b368fc72453913bf6dbfd2463e"
    ),
    "151510/filtered_feature_bc_matrix.h5": (
        "421a77bad2e08326a627d91a8994e1f47afb61b75aba815204b2b37c2aed26e9"
    ),
    "151510/metadata.tsv": (
        "83c0ef6403d1cf5a2659c05cfc912e5d666c1cb21348a4d5159cdeb3da5246cd"
    ),
    "151669/filtered_feature_bc_matrix.h5": (
        "4c92c56291f22c31abdc76eb4d7e638ffc5b72e3c08de1142cd76ca731cbea14"
    ),
    "151669/metadata.tsv": (
        "82cf728854bc30f534438e34badf59c26cf07e08475f4ba2ce0368e56c572f79"
    ),
    "151670/filtered_feature_bc_matrix.h5": (
        "931f50b5de5aaa6a9e9c9d795f072a1f1f7f42a1602c849c61540b7f2258a5b1"
    ),
    "151670/metadata.tsv": (
        "050ab781bdf8abde7539fd1d75a304296f61223bb756d1f9d9bb2602370e64f3"
    ),
    "151671/filtered_feature_bc_matrix.h5": (
        "6b6b86696b377fb44fd367e80c3820209f8977cb62af4b4715c37c6faf80baa5"
    ),
    "151671/metadata.tsv": (
        "a73e9f7d4199801e2a783cc8803aa9496c568738777ee8546671969010316745"
    ),
    "151672/filtered_feature_bc_matrix.h5": (
        "a28ad2b88314d16b1bbde612fb80d12460ce84981cf4c00902ca046cafae4009"
    ),
    "151672/metadata.tsv": (
        "93f1eb6853df48c3467784f5ab81d99fcf2e94c5f4b997d16a21e8f1d10e09ea"
    ),
    "151673/filtered_feature_bc_matrix.h5": (
        "216e8010e1c313925afa0f174207da7c8947497c960e81b07edbd16176460413"
    ),
    "151673/metadata.tsv": (
        "1e6dcce1a6e0a7dcb8f0178521d3c39873be41bd5a9cdfc061f1b84a436a0d07"
    ),
    "151674/filtered_feature_bc_matrix.h5": (
        "0ecbabdb0ea98893f41bf7b557d8de50f1e9833eb2e55b69740b85b985c01880"
    ),
    "151674/metadata.tsv": (
        "a6ee7a6de23ff44db6269c399e524a6c6369522bafd1b1d876a67f56e4749af0"
    ),
    "151675/filtered_feature_bc_matrix.h5": (
        "eba1f479b85df34dd6b58144c1fcdfbf2ad8fb01fe80caabb9bf64a2ce36394a"
    ),
    "151675/metadata.tsv": (
        "75db54be0b2af812302d08934df0b5aa8a9d8a87629427fa358350df8dc6d8a7"
    ),
    "151676/filtered_feature_bc_matrix.h5": (
        "f436605010a0133c96d1135f22fdd789317779287af58ba5af764fc1b28ea041"
    ),
    "151676/metadata.tsv": (
        "d787d3e414a29c526fe07c74432f5b382c03c6ff042c2de45ad4b053c036cdd9"
    ),
}


class DLPFCError(RuntimeError):
    """Raised when DLPFC data is missing, corrupt, or cannot be fetched."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, key: str) -> None:
    expected = CHECKSUMS.get(key)
    if expected is None:
        return
    actual = _sha256(path)
    if actual != expected:
        raise DLPFCError(
            f"checksum mismatch for {path} (expected {expected[:12]}..., "
            f"got {actual[:12]}...); delete the file and re-download"
        )


def _download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` via a temp file so partial downloads never
    masquerade as complete ones."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as resp, open(part, "wb") as out:
            shutil.copyfileobj(resp, out)
    except Exception as exc:
        part.unlink(missing_ok=True)
        raise DLPFCError(f"failed to download {url}: {exc}") from None
    part.replace(dest)


def fetch_dlpfc_sample(
    sample: str, root: str | Path, *, download: bool = True
) -> tuple[Path, Path]:
    """Return local paths to a sample's matrix and metadata, fetching if needed."""
    sample = str(sample)
    if sample not in SAMPLE_DONORS:
        known = ", ".join(sorted(SAMPLE_DONORS))
        raise DLPFCError(f"unknown DLPFC sample {sample!r}; known samples: {known}")
    sampledir = Path(root) / sample
    paths = {
        H5_NAME: _H5_URL.format(sample=sample),
        META_NAME: _META_URL.format(sample=sample),
    }
    out = []
    for name, url in paths.items():
        path = sampledir / name
        if not path.is_file():
            if not download:
                raise DLPFCError(
                    f"{path} is missing and download=False; "
                    "run once with download=True to fetch it"
                )
            _download(url, path)
        _verify(path, f"{sample}/{name}")
        out.append(path)
    return out[0], out[1]


def read_10x_h5(path: str | Path) -> AnnData:
    """Read a 10x Cell Ranger v3 HDF5 matrix into spots-by-genes AnnData.

    A minimal reader (the format is a CSC matrix of genes x barcodes plus two
    label arrays) so the core package does not need scanpy.
    """
    with h5py.File(path, "r") as fh:
        if "matrix" not in fh:
            raise DLPFCError(f"{path} has no 'matrix' group; not a v3 10x HDF5 file")
        grp = fh["matrix"]
        shape = tuple(int(v) for v in grp["shape"][:])  # (n_genes, n_barcodes)
        matrix = sp.csc_matrix(
            (grp["data"][:], grp["indices"][:], grp["indptr"][:]), shape=shape
        )
        barcodes = grp["barcodes"][:].astype(str)
        symbols = grp["features/name"][:].astype(str)
        gene_ids = grp["features/id"][:].astype(str)

    # Ensembl ids are unique; symbols are not (a handful of ids share one), so
    # ids are the var index and symbols ride along until a caller resolves them.
    adata = AnnData(matrix.T.tocsr().astype(np.float32))
    adata.obs_names = barcodes
    adata.var_names = gene_ids
    adata.var["symbol"] = symbols
    return adata


def load_dlpfc_slice(
    sample: str, root: str | Path, *, download: bool = True
) -> AnnData:
    """Load one annotated slice: raw counts + layer labels + spatial coords.

    Spots without a manual layer call are dropped. ``obs['domain']`` holds the
    layer label; ``obsm['spatial']`` holds image coordinates flipped so that
    plots come out in anatomical orientation (image rows grow downward).
    """
    h5_path, meta_path = fetch_dlpfc_sample(sample, root, download=download)
    adata = read_10x_h5(h5_path)
    # The mirror is an R export: row names carry the barcodes, so the written
    # header is one column short and pandas must take column 0 as the index.
    meta = pd.read_csv(meta_path, sep="\t")
    if "layer_guess" not in meta.columns:
        raise DLPFCError(f"{meta_path} has no 'layer_guess' column")
    meta = meta[meta["layer_guess"].notna()]

    common = adata.obs_names.intersection(meta.index)
    if len(common) == 0:
        raise DLPFCError(f"no barcodes shared between {h5_path} and {meta_path}")
    adata = adata[common].copy()
    meta = meta.loc[common]

    adata.obs["domain"] = meta["layer_guess"].astype(str).to_numpy()
    adata.obs["donor"] = SAMPLE_DONORS[str(sample)]
    adata.obs["sample"] = str(sample)
    adata.obsm["spatial"] = np.column_stack(
        [meta["imagecol"].to_numpy(float), -meta["imagerow"].to_numpy(float)]
    )
    return adata


def _normalize_log1p(adata: AnnData, target_sum: float) -> None:
    """Depth-normalize each spot to ``target_sum`` counts, then log1p, in place."""
    X = sp.csr_matrix(adata.X, dtype=np.float64)
    depth = np.asarray(X.sum(axis=1)).ravel()
    depth[depth == 0.0] = 1.0
    scale = sp.diags(target_sum / depth)
    X = scale @ X
    X.data = np.log1p(X.data)
    adata.X = X


def load_dlpfc_cohort(
    samples: Sequence[str | int] = ("151507", "151669", "151673"),
    root: str | Path = "data/DLPFC",
    n_candidates: int = 2000,
    target_sum: float = 1e4,
    min_spots_frac: float = 0.05,
    download: bool = True,
) -> list[AnnData]:
    """Load a multi-donor DLPFC cohort ready for :class:`~caust.pipeline.CauST`.

    Each slice is depth-normalized and log1p-transformed, then all slices are
    subset to one shared candidate gene pool. The pool -- genes expressed in at
    least ``min_spots_frac`` of spots in every *training* slice, ranked by mean
    training-slice variance -- is chosen **without looking at the final slice**,
    which the experiment runner holds out. The variance ranking also means the
    HVG baseline competes on its home turf: CauST re-ranks exactly the genes
    variance likes best.
    """
    if len(samples) < 2:
        raise DLPFCError("a cohort needs at least 2 samples")
    if n_candidates < 1:
        raise DLPFCError(f"n_candidates must be >= 1, got {n_candidates}")

    slices = [load_dlpfc_slice(str(s), root, download=download) for s in samples]
    for adata in slices:
        _normalize_log1p(adata, target_sum=target_sum)

    # Switch the var index from Ensembl ids to gene symbols, dropping symbols
    # that are duplicated within a slice so every name is an unambiguous key,
    # then align all slices on the shared symbols.
    for i, adata in enumerate(slices):
        symbols = pd.Index(adata.var["symbol"].astype(str))
        keep = np.flatnonzero(~symbols.duplicated(keep=False))
        sub = adata[:, keep].copy()
        sub.var_names = symbols[keep]
        slices[i] = sub
    shared = slices[0].var_names
    for adata in slices[1:]:
        shared = shared.intersection(adata.var_names)
    if len(shared) == 0:
        raise DLPFCError("slices share no gene symbols")
    slices = [adata[:, shared].copy() for adata in slices]

    train = slices[:-1]
    expressed = np.ones(len(shared), dtype=bool)
    variance = np.zeros(len(shared), dtype=np.float64)
    for adata in train:
        X = sp.csr_matrix(adata.X)
        frac = np.asarray((X > 0).mean(axis=0)).ravel()
        expressed &= frac >= min_spots_frac
        mean_sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        mean = np.asarray(X.mean(axis=0)).ravel()
        variance += mean_sq - mean**2
    variance /= len(train)
    if not expressed.any():
        raise DLPFCError(
            f"no gene is expressed in >= {min_spots_frac:.0%} of spots "
            "in every training slice; lower min_spots_frac"
        )

    variance[~expressed] = -np.inf
    n_pool = min(n_candidates, int(expressed.sum()))
    # Stable sort + name tiebreak so the pool is identical across platforms.
    order = np.lexsort((shared.to_numpy(), -variance))[:n_pool]
    pool = shared[np.sort(order)]
    return [adata[:, pool].copy() for adata in slices]
