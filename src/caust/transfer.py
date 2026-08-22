"""Cross-slice transfer benchmark -- the proposal's evaluation protocol.

For each gene-selection *strategy* and gene budget *K*, every slice serves as
the source: a backbone is trained on the source restricted to the strategy's
K genes, then applied **zero-shot** (no retraining) to every slice, and each
embedding is clustered against the manual annotations. Aggregating the
(source, target) grid gives three views of the same numbers:

- **within-slice** ARI   source == target
- **within-donor** ARI   source != target, same donor
- **cross-donor** ARI    different donors -- the generalization claim

Strategies (all at equal gene count):

- ``hvg``        top-K highly variable genes of the *source* slice (the
                 standard pipeline);
- ``hvg_donor``  donor-aware HVG: genes ranked by how many scoring slices
                 call them highly variable (scanpy's ``batch_key`` rule, the
                 simplest cross-donor-stability selector -- the baseline the
                 invariance term has to beat);
- ``moran``      top-K by Moran's I spatial autocorrelation averaged over the
                 scoring slices (the best-performing SVG selector in recent
                 benchmarks);
- ``random``     K genes drawn at random from the pool (a control);
- ``highdelta``  top-K by mean knockout effect (lambda = 0, no stability);
- ``caust``      top-K by invariance score ``mean - lambda * std``.

Knockout effects are scored on a small set of *scoring slices* over a gene
pool that is the cross-donor intersection of each scoring slice's top HVGs.
Two scoring protocols:

- ``pooled`` -- one slice per donor, every donor included (the proposal's
  design). The gene set for a cross-donor evaluation has then seen one slice
  of the *target* donor, unlabelled; the per-source HVG baseline has not.
- ``leave_target_donor_out`` -- the gene set used to evaluate targets of donor
  D is scored only on the other donors' slices, so cross-donor ARI is a
  genuinely out-of-donor measurement for every strategy. Costs one backbone
  per target donor for the global strategies.

Seeds: ``evaluation.seeds`` re-initialise the clustering only;
``evaluation.train_seeds`` (default: the run seed) re-train the backbone.
Statistics are reported at the level of independent units -- sources and
donor pairs -- not the 96 shared-backbone cells. Runs are resumable: every
(strategy, K, source) cell is written when complete and skipped on restart.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from .cluster import ari, cluster_embedding, nmi
from .config import ExperimentConfig, dump_config
from .experiment import ARTIFACT_CONFIG, ExperimentError, _model_factory
from .invariance import invariance_scores, select_causal_genes
from .pipeline import CauST
from .repro import collect_provenance, deterministic, sha256_file

STRATEGIES = ("hvg", "hvg_donor", "moran", "random", "highdelta", "caust")
#: Strategies whose gene set is the same for every source slice.
GLOBAL_STRATEGIES = ("hvg_donor", "moran", "highdelta", "caust")
DEFAULT_K = (50, 100, 200, 400, 800, 1600)
SCORING_MODES = ("pooled", "leave_target_donor_out")
RESULTS_CSV = "results.csv"
SUMMARY_JSON = "summary.json"
GENES_CSV = "gene_scores.csv"
JACCARD_JSON = "hvg_jaccard.json"
SCORES_NPZ = "knockout_scores.npz"

_FIELDS = (
    "strategy",
    "k",
    "n_genes",
    "scoring",
    "source",
    "target",
    "source_donor",
    "target_donor",
    "train_seed",
    "seed",
    "ari",
    "nmi",
    "n_clusters",
)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
def rank_hvg(adata: AnnData, n_top: int, counts_layer: str | None = None) -> np.ndarray:
    """Per-gene HVG rank (0 = most variable) for one slice.

    Uses scanpy's ``seurat_v3`` flavour on raw counts when scanpy is installed
    (the ranking STAGATE and the proposal use); otherwise falls back to plain
    variance of ``adata.X``. Genes outside the top ``n_top`` get rank ``n_top``.
    """
    n_genes = adata.n_vars
    n_top = min(n_top, n_genes)
    ranks = np.full(n_genes, n_top, dtype=int)
    try:
        if counts_layer is None:
            raise ImportError  # seurat_v3 needs raw counts; use the fallback
        import scanpy as sc

        tmp = AnnData(
            adata.layers[counts_layer] if counts_layer else adata.X,
            obs=adata.obs[[]].copy(),
            var=adata.var[[]].copy(),
        )
        sc.pp.highly_variable_genes(tmp, flavor="seurat_v3", n_top_genes=n_top)
        rank = tmp.var["highly_variable_rank"].to_numpy()
        have = np.isfinite(rank)
        ranks[have] = rank[have].astype(int)
        return ranks
    except (ImportError, ValueError) as exc:
        if counts_layer is not None:
            warnings.warn(
                f"seurat_v3 HVG ranking unavailable ({exc}); falling back to "
                "variance of the log-normalized matrix",
                stacklevel=2,
            )
        X = adata.X
        var = (
            np.asarray(X.power(2).mean(0) - np.square(X.mean(0))).ravel()
            if sp.issparse(X)
            else np.asarray(X).var(0)
        )
        order = np.argsort(-var, kind="stable")[:n_top]
        ranks[order] = np.arange(n_top)
        return ranks


def _normalize_log1p(adata: AnnData, target_sum: float = 1e4) -> None:
    X = sp.csr_matrix(adata.X, dtype=np.float64)
    depth = np.asarray(X.sum(axis=1)).ravel()
    depth[depth == 0] = 1.0
    X = sp.diags(target_sum / depth) @ X
    X.data = np.log1p(X.data)
    adata.X = X.astype(np.float32)


def load_transfer_cohort(cfg: ExperimentConfig) -> list[AnnData]:
    """All slices, log-normalized over *all* shared genes, with HVG ranks."""
    spec = dict(cfg.data)
    source = spec.pop("source", "synthetic")
    n_hvg = int(spec.pop("n_hvg_rank", 8000))
    if source == "synthetic":
        from .data import make_synthetic_cohort

        spec.setdefault("seed", cfg.seed)
        per_donor = int(spec.pop("slices_per_donor", 1))
        slices = make_synthetic_cohort(**spec)
        for i, a in enumerate(slices):
            a.obs["sample"] = f"synthetic_{i}"
            a.obs["donor"] = f"donor_{i // per_donor}"
    elif source == "dlpfc":
        from .dlpfc import DLPFCError, load_dlpfc_slice

        root = spec.pop("root", "data/DLPFC")
        samples = [str(s) for s in spec.pop("samples")]
        try:
            slices = [
                load_dlpfc_slice(s, root, download=spec.get("download", True))
                for s in samples
            ]
        except DLPFCError as exc:
            raise ExperimentError(str(exc)) from None
        for a in slices:
            a.layers["counts"] = a.X.copy()
        # Unambiguous symbols shared by every slice, in one fixed order.
        shared: set[str] | None = None
        for a in slices:
            sym = a.var["symbol"].astype(str)
            ok = ~sym.duplicated(keep=False).to_numpy()
            names = set(sym[ok])
            shared = names if shared is None else shared & names
        order = sorted(shared or set())
        for i, a in enumerate(slices):
            sym = a.var["symbol"].astype(str)
            keep = ~sym.duplicated(keep=False).to_numpy()
            sub = a[:, keep].copy()
            sub.var_names = sym[keep]
            slices[i] = sub[:, order].copy()
        for a in slices:
            _normalize_log1p(a)
    else:
        from .datasets import DATASETS, DatasetError, load_dataset

        if source not in DATASETS:
            raise ExperimentError(
                f"unsupported data.source {source!r}; expected synthetic, dlpfc, "
                + ", ".join(DATASETS)
            )
        try:
            slices = load_dataset(
                source,
                spec.get("sections"),
                spec.get("root"),
                download=bool(spec.get("download", True)),
            )
        except DatasetError as exc:
            raise ExperimentError(str(exc)) from None
        if DATASETS[source].counts_are_integers:
            for a in slices:
                a.layers["counts"] = a.X.copy()
        for a in slices:
            _normalize_log1p(a)
    for a in slices:
        a.var["hvg_rank"] = rank_hvg(
            a, n_hvg, counts_layer="counts" if "counts" in a.layers else None
        )
        if "counts" in a.layers:
            del a.layers["counts"]
    return slices


def hvg_jaccard(slices: Sequence[AnnData], n_top: int = 3000) -> dict[str, Any]:
    """Pairwise Jaccard similarity of per-slice top-``n_top`` HVG sets."""
    sets = [
        set(a.var_names[np.argsort(a.var["hvg_rank"].to_numpy())[:n_top]])
        for a in slices
    ]
    within: list[float] = []
    across: list[float] = []
    for i, j in combinations(range(len(slices)), 2):
        jac = len(sets[i] & sets[j]) / len(sets[i] | sets[j])
        same = slices[i].obs["donor"].iloc[0] == slices[j].obs["donor"].iloc[0]
        (within if same else across).append(jac)

    def stat(v: list[float]) -> dict[str, float]:
        return (
            {"mean": float(np.mean(v)), "std": float(np.std(v)), "n_pairs": len(v)}
            if v
            else {}
        )

    return {"n_top": n_top, "within_donor": stat(within), "across_donor": stat(across)}


# ---------------------------------------------------------------------------
# Gene scoring and gene sets
# ---------------------------------------------------------------------------
def default_scoring_indices(
    slices: Sequence[AnnData], per_donor: int = 1, exclude_donor: str | None = None
) -> list[int]:
    """First ``per_donor`` slices of every donor except ``exclude_donor``."""
    counts: dict[str, int] = {}
    out = []
    for i, a in enumerate(slices):
        d = str(a.obs["donor"].iloc[0])
        if d == exclude_donor:
            continue
        if counts.get(d, 0) < per_donor:
            counts[d] = counts.get(d, 0) + 1
            out.append(i)
    return out


def build_pool(
    slices: Sequence[AnnData], scoring: Sequence[int], n_hvg: int
) -> list[str]:
    """Cross-donor intersection of each scoring slice's top-``n_hvg`` HVGs."""
    pool = None
    for i in scoring:
        a = slices[i]
        top = set(a.var_names[np.argsort(a.var["hvg_rank"].to_numpy())[:n_hvg]])
        pool = top if pool is None else pool & top
    assert pool is not None
    names = slices[0].var_names
    return [g for g in names if g in pool]  # fixed order = first slice's order


def score_pool(
    slices: Sequence[AnnData],
    scoring: Sequence[int],
    pool: Sequence[str],
    model_factory: Callable[[], Any],
    lam: float,
    verbose: bool = False,
    knockout_mode: str = "zero",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Knockout deltas on the scoring slices; returns (names, deltas, s_inv)."""
    cs = CauST(model_factory=model_factory, lam=lam, knockout_mode=knockout_mode).fit(
        [slices[i][:, list(pool)].copy() for i in scoring], verbose=verbose
    )
    return np.asarray(cs.common_genes_), np.asarray(cs.deltas_), np.asarray(cs.scores_)


def morans_i(adata: AnnData, genes: Sequence[str], n_neighbors: int = 6) -> np.ndarray:
    """Moran's I of each gene over a spatial kNN graph (the slice is not mutated)."""
    from .graph import CONN_KEY, build_spatial_graph

    probe = AnnData(obs=adata.obs[[]].copy())
    probe.obsm["spatial"] = np.asarray(adata.obsm["spatial"])
    build_spatial_graph(probe, n_neighbors=n_neighbors)
    W = sp.csr_matrix(probe.obsp[CONN_KEY], dtype=np.float64)
    X = adata[:, list(genes)].X
    X = np.asarray(X.todense() if sp.issparse(X) else X, dtype=np.float64)
    Z = X - X.mean(axis=0)
    num = np.einsum("ij,ij->j", Z, W @ Z)
    den = np.einsum("ij,ij->j", Z, Z)
    with np.errstate(divide="ignore", invalid="ignore"):
        out: np.ndarray = (Z.shape[0] / W.sum()) * num / den
    out[~np.isfinite(out)] = -np.inf
    return out


def baseline_scores(
    slices: Sequence[AnnData],
    scoring: Sequence[int],
    names: np.ndarray,
    n_top: int = 3000,
) -> dict[str, np.ndarray]:
    """Gene scores for the global baselines over the pool genes ``names``.

    ``hvg_donor`` follows scanpy's ``batch_key`` rule: genes are ordered by
    the number of scoring slices in which they are within the top-``n_top``
    HVGs, then by the median of their ranks in those slices. ``n_top`` must be
    smaller than the pool's own HVG depth for the count to carry information.
    """
    pool = list(names)
    ranks = np.vstack(
        [slices[i].var.loc[pool, "hvg_rank"].to_numpy(dtype=float) for i in scoring]
    )
    is_hvg = ranks < n_top
    hits = is_hvg.sum(axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median_rank = np.nanmedian(np.where(is_hvg, ranks, np.nan), axis=0)
    median_rank = np.where(np.isfinite(median_rank), median_rank, np.inf)
    hvg_donor = hits * 1e9 - np.minimum(median_rank, 1e8)
    moran = np.mean([morans_i(slices[i], pool) for i in scoring], axis=0)
    return {"hvg_donor": hvg_donor, "moran": moran}


def gene_set(
    strategy: str,
    k: int,
    source: AnnData,
    names: np.ndarray,
    deltas: np.ndarray,
    lam: float,
    baselines: dict[str, np.ndarray] | None = None,
    rng: np.random.Generator | None = None,
) -> list[str]:
    if strategy == "hvg":
        order = np.argsort(source.var["hvg_rank"].to_numpy(), kind="stable")
        return [str(g) for g in source.var_names[order[:k]]]
    if k > len(names):
        raise ExperimentError(
            f"K={k} exceeds the {len(names)}-gene knockout pool for strategy "
            f"{strategy!r}; equal-K comparisons need K <= pool size"
        )
    if strategy == "random":
        gen = rng if rng is not None else np.random.default_rng(0)
        pick = gen.choice(len(names), size=k, replace=False)
        return [str(g) for g in names[np.sort(pick)]]
    if strategy == "highdelta":
        scores = invariance_scores(deltas, lam=0.0)
    elif strategy == "caust":
        scores = invariance_scores(deltas, lam=lam)
    elif strategy in ("hvg_donor", "moran"):
        if baselines is None or strategy not in baselines:
            raise ExperimentError(f"baseline scores for {strategy!r} not computed")
        scores = np.asarray(baselines[strategy], dtype=float)
    else:
        raise ExperimentError(
            f"unknown strategy {strategy!r}; expected one of {STRATEGIES}"
        )
    return [str(g) for g in names[select_causal_genes(scores, k)]]


# ---------------------------------------------------------------------------
# Scoring contexts: which slices score the genes for which targets
# ---------------------------------------------------------------------------
class ScoringContext:
    """Pool, knockout deltas and baseline scores from one set of scoring slices."""

    def __init__(
        self, label: str, scoring: list[int], names: np.ndarray, deltas: np.ndarray
    ):
        self.label = label
        self.scoring = scoring
        self.names = names
        self.deltas = deltas
        self.baselines: dict[str, np.ndarray] = {}


def _context_path(outdir: Path, label: str) -> Path:
    return outdir / (
        SCORES_NPZ if label == "pooled" else f"knockout_scores_{label}.npz"
    )


def _save_npz(path: Path, names: np.ndarray, deltas: np.ndarray) -> None:
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, names=names, deltas=deltas)
    os.replace(tmp, path)


def build_contexts(
    cfg: ExperimentConfig,
    slices: Sequence[AnnData],
    outdir: Path,
    strategies: Sequence[str],
    *,
    resume: bool,
    verbose: bool,
) -> dict[str, ScoringContext]:
    sel = cfg.selection
    mode = str(sel.get("scoring_mode", "pooled"))
    if mode not in SCORING_MODES:
        raise ExperimentError(f"selection.scoring_mode must be one of {SCORING_MODES}")
    per_donor = int(sel.get("scoring_slices_per_donor", 1))
    lam = float(sel.get("lam", 2.0))
    n_hvg_pool = int(sel.get("n_hvg_pool", 8000))
    donors = list(dict.fromkeys(str(a.obs["donor"].iloc[0]) for a in slices))
    if mode == "leave_target_donor_out" and len(donors) < 3:
        raise ExperimentError(
            "leave_target_donor_out needs at least 3 donors (2 to score, 1 to hold out)"
        )
    labels = ["pooled"] if mode == "pooled" else [f"lodo_{d}" for d in donors]
    excluded = [None] if mode == "pooled" else donors
    contexts: dict[str, ScoringContext] = {}
    for label, exclude in zip(labels, excluded):
        if "scoring_slices" in sel and mode == "pooled":
            scoring = [int(i) for i in sel["scoring_slices"]]
        else:
            scoring = default_scoring_indices(slices, per_donor, exclude)
        path = _context_path(outdir, label)
        if resume and path.exists():
            z = np.load(path, allow_pickle=False)
            names, deltas = z["names"].astype(str), z["deltas"]
        else:
            pool = build_pool(slices, scoring, n_hvg_pool)
            if verbose:
                print(
                    f"[{label}] knockout pool: {len(pool)} genes (intersection of "
                    f"top-{n_hvg_pool} HVGs over {len(scoring)} scoring slices)"
                )
            t0 = time.time()
            names, deltas, _ = score_pool(
                slices,
                scoring,
                pool,
                _model_factory(cfg, random_state=cfg.seed),
                lam,
                verbose,
                str(sel.get("knockout_mode", "zero")),
            )
            _save_npz(path, names=np.asarray(names, dtype=str), deltas=deltas)
            if verbose:
                print(
                    f"[{label}] scored {len(names)} genes on {len(scoring)} slices "
                    f"in {time.time() - t0:.0f}s"
                )
        ctx = ScoringContext(label, scoring, names, deltas)
        if any(st in ("hvg_donor", "moran") for st in strategies):
            ctx.baselines = baseline_scores(
                slices, scoring, names, int(sel.get("hvg_donor_n_top", 3000))
            )
        contexts[label] = ctx
    return contexts


def context_for_target(
    contexts: dict[str, ScoringContext], donor: str
) -> ScoringContext:
    return contexts["pooled"] if "pooled" in contexts else contexts[f"lodo_{donor}"]


# ---------------------------------------------------------------------------
# One grid cell: train on the source, transfer everywhere
# ---------------------------------------------------------------------------
def run_cell(
    strategy: str,
    k: int,
    src_idx: int,
    slices: Sequence[AnnData],
    contexts: dict[str, ScoringContext],
    cfg: ExperimentConfig,
    seeds: Sequence[int],
    train_seeds: Sequence[int],
    cluster_method: str,
    lam: float,
) -> list[dict[str, Any]]:
    source = slices[src_idx]
    src_name = str(source.obs["sample"].iloc[0])
    src_donor = str(source.obs["donor"].iloc[0])
    # Targets grouped by the scoring context their gene set must come from.
    groups: dict[str, list[int]] = {}
    for t, target in enumerate(slices):
        ctx = context_for_target(contexts, str(target.obs["donor"].iloc[0]))
        key = ctx.label if strategy in GLOBAL_STRATEGIES else "pooled"
        groups.setdefault(key, []).append(t)
    rows = []
    for label, targets in groups.items():
        ctx = contexts[label] if label in contexts else next(iter(contexts.values()))
        genes = gene_set(
            strategy,
            k,
            source,
            ctx.names,
            ctx.deltas,
            lam,
            ctx.baselines,
            np.random.default_rng(cfg.seed + 1000 * k + src_idx),
        )
        for ts in train_seeds:
            model = _model_factory(cfg, random_state=int(ts))().fit(
                source[:, list(genes)].copy()
            )
            for t in targets:
                target = slices[t]
                Z = model.get_embedding(target[:, list(genes)].copy())
                truth = target.obs["domain"].to_numpy()
                n_clusters = int(len(np.unique(truth)))
                for seed in seeds:
                    pred = cluster_embedding(
                        Z, n_clusters, random_state=seed, method=cluster_method
                    )
                    rows.append(
                        {
                            "strategy": strategy,
                            "k": k,
                            "n_genes": len(genes),
                            "scoring": (
                                label if strategy in GLOBAL_STRATEGIES else "source"
                            ),
                            "source": src_name,
                            "target": str(target.obs["sample"].iloc[0]),
                            "source_donor": src_donor,
                            "target_donor": str(target.obs["donor"].iloc[0]),
                            "train_seed": int(ts),
                            "seed": int(seed),
                            "ari": ari(truth, pred),
                            "nmi": nmi(truth, pred),
                            "n_clusters": n_clusters,
                        }
                    )
    return rows


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _split(row: dict[str, Any]) -> str:
    if row["source"] == row["target"]:
        return "within_slice"
    if row["source_donor"] == row["target_donor"]:
        return "within_donor"
    return "cross_donor"


def _stat(v: Sequence[float]) -> dict[str, float]:
    return {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per strategy x K x split: cell-level means, spreads at the right level
    of independence, the generalization gap, win counts, and paired tests."""
    # Seed-averaged value per (strategy, k, source, target) cell.
    cells: dict[tuple[str, int, str, str], dict[str, list[float]]] = {}
    donors: dict[str, str] = {}
    for r in rows:
        key = (r["strategy"], int(r["k"]), r["source"], r["target"])
        c = cells.setdefault(key, {"ari": [], "nmi": [], "n_genes": []})
        c["ari"].append(float(r["ari"]))
        if "nmi" in r and r["nmi"] not in ("", None):
            c["nmi"].append(float(r["nmi"]))
        if "n_genes" in r and r["n_genes"] not in ("", None):
            c["n_genes"].append(float(r["n_genes"]))
        donors[r["source"]] = r["source_donor"]
        donors[r["target"]] = r["target_donor"]
    strategies = sorted({k[0] for k in cells})
    ks = sorted({k[1] for k in cells})
    table: dict[str, Any] = {}
    for s in strategies:
        for k in ks:
            entry: dict[str, Any] = {}
            for split in ("within_slice", "within_donor", "cross_donor"):
                means, seed_stds, nmis, ng = [], [], [], []
                for (ss, kk, src, tgt), c in cells.items():
                    if ss != s or kk != k:
                        continue
                    row_split = _split(
                        {
                            "source": src,
                            "target": tgt,
                            "source_donor": donors[src],
                            "target_donor": donors[tgt],
                        }
                    )
                    if row_split != split:
                        continue
                    means.append(float(np.mean(c["ari"])))
                    seed_stds.append(float(np.std(c["ari"])))
                    if c["nmi"]:
                        nmis.append(float(np.mean(c["nmi"])))
                    ng.extend(c["n_genes"])
                if means:
                    entry[split] = {
                        "mean": float(np.mean(means)),
                        "std": float(np.std(means)),  # across (source, target) cells
                        "seed_std": float(np.mean(seed_stds)),  # within a cell
                        "n_cells": len(means),
                    }
                    if nmis:
                        entry[split]["nmi"] = float(np.mean(nmis))
            if ng:
                entry["n_genes"] = {"min": int(min(ng)), "max": int(max(ng))}
            if "within_slice" in entry and "cross_donor" in entry:
                entry["generalization_gap"] = (
                    entry["within_slice"]["mean"] - entry["cross_donor"]["mean"]
                )
            table.setdefault(s, {})[str(k)] = entry
    cell_means = {key: float(np.mean(c["ari"])) for key, c in cells.items()}
    return {
        "strategies": strategies,
        "k_values": ks,
        "table": table,
        "caust_vs_hvg_wins": _wins(cell_means, donors, ks),
        "paired_tests": paired_tests(cell_means, donors, strategies, ks),
    }


def _wins(
    cell_means: dict[tuple[str, int, str, str], float],
    donors: dict[str, str],
    ks: Sequence[int],
) -> dict[str, Any]:
    """Descriptive CauST-vs-HVG win counts over cells (not independent units)."""
    out: dict[str, Any] = {}
    strategies = {k[0] for k in cell_means}
    if not {"caust", "hvg"} <= strategies:
        return out
    for k in ks:
        counts = {"within_slice": [0, 0], "cross_donor": [0, 0]}
        for (s, kk, src, tgt), v in cell_means.items():
            if s != "caust" or kk != k or ("hvg", k, src, tgt) not in cell_means:
                continue
            if src == tgt:
                split = "within_slice"
            elif donors[src] != donors[tgt]:
                split = "cross_donor"
            else:
                continue
            counts[split][1] += 1
            counts[split][0] += v > cell_means[("hvg", k, src, tgt)]
        out[str(k)] = {
            name: {"caust_wins": int(a), "cells": int(b)}
            for name, (a, b) in counts.items()
        }
    return out


def paired_tests(
    cell_means: dict[tuple[str, int, str, str], float],
    donors: dict[str, str],
    strategies: Sequence[str],
    ks: Sequence[int],
) -> dict[str, Any]:
    """Wilcoxon signed-rank tests of CauST against every other strategy.

    The (source, target) cells share backbones (one per source) and labels
    (one per target), so they are not independent. Tests are therefore run
    on aggregated units: one value per **source** (its mean cross-donor ARI,
    n = number of slices) and one per **donor pair** (n = D x (D-1)); the
    within-slice test is one value per source.
    """
    from scipy.stats import wilcoxon

    if "caust" not in strategies:
        return {}

    def unit_means(strategy: str, k: int, level: str) -> dict[str, float]:
        acc: dict[str, list[float]] = {}
        for (s, kk, src, tgt), v in cell_means.items():
            if s != strategy or kk != k:
                continue
            if level == "within_slice":
                if src != tgt:
                    continue
                acc.setdefault(src, []).append(v)
            elif donors[src] != donors[tgt]:
                unit = src if level == "source" else f"{donors[src]}->{donors[tgt]}"
                acc.setdefault(unit, []).append(v)
        return {u: float(np.mean(v)) for u, v in acc.items()}

    out: dict[str, Any] = {}
    for k in ks:
        out[str(k)] = {}
        for other in strategies:
            if other == "caust":
                continue
            res: dict[str, Any] = {}
            for level in ("source", "donor_pair", "within_slice"):
                a_map, b_map = unit_means("caust", k, level), unit_means(
                    other, k, level
                )
                units = sorted(set(a_map) & set(b_map))
                a = [a_map[u] for u in units]
                b = [b_map[u] for u in units]
                diff = np.subtract(a, b) if a else np.array([])
                if len(units) >= 2 and np.any(diff != 0):
                    p = float(wilcoxon(a, b).pvalue)
                else:
                    p = float("nan")
                res[level] = {
                    "mean_diff": float(np.mean(diff)) if len(diff) else float("nan"),
                    "caust_wins": int(np.sum(diff > 0)) if len(diff) else 0,
                    "n_units": len(units),
                    "p_value": p,
                }
            out[str(k)][other] = res
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _write_rows(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    import csv

    new = not path.exists()
    with open(path, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({f: r.get(f, "") for f in _FIELDS})
        fh.flush()
        os.fsync(fh.fileno())


def _read_rows(path: Path) -> list[dict[str, Any]]:
    import csv

    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        rows = [dict(r) for r in csv.DictReader(fh)]
    # A truncated final line shows up as a row with missing/None fields.
    return [
        r for r in rows if r.get("ari") not in (None, "") and None not in r.values()
    ]


def _cell_key(r: dict[str, Any]) -> tuple[str, int, str]:
    return (str(r["strategy"]), int(r["k"]), str(r["source"]))


def _completed_cells(
    path: Path, expected_rows: int
) -> tuple[set[tuple[str, int, str]], list[dict[str, Any]]]:
    """Cells with every expected row; incomplete cells are dropped from disk."""
    rows = _read_rows(path)
    counts: dict[tuple[str, int, str], int] = {}
    for r in rows:
        counts[_cell_key(r)] = counts.get(_cell_key(r), 0) + 1
    done = {key for key, n in counts.items() if n >= expected_rows}
    kept = [r for r in rows if _cell_key(r) in done]
    if len(kept) != len(rows):
        path.unlink()
        if kept:
            _write_rows(path, kept)
    return done, kept


def _validate(
    cfg: ExperimentConfig, ks: Sequence[int], strategies: Sequence[str]
) -> None:
    sel, ev = cfg.selection, cfg.evaluation
    n_rank = int(cfg.data.get("n_hvg_rank", 8000))
    n_pool = int(sel.get("n_hvg_pool", 8000))
    jac = int(ev.get("jaccard_n_top", 3000))
    problems = []
    if n_pool > n_rank:
        problems.append(f"selection.n_hvg_pool ({n_pool}) > data.n_hvg_rank ({n_rank})")
    if jac > n_rank:
        problems.append(
            f"evaluation.jaccard_n_top ({jac}) > data.n_hvg_rank ({n_rank})"
        )
    if "hvg" in strategies and max(ks) > n_rank:
        problems.append(
            f"max K ({max(ks)}) > data.n_hvg_rank ({n_rank}) for strategy hvg"
        )
    unknown = [s for s in strategies if s not in STRATEGIES]
    if unknown:
        problems.append(f"unknown strategies {unknown}; expected {STRATEGIES}")
    if problems:
        raise ExperimentError("; ".join(problems))


def run_transfer(
    cfg: ExperimentConfig,
    outdir: Path,
    *,
    figures: bool = False,
    resume: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run (or resume) the full transfer grid described by ``cfg``."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, outdir / ARTIFACT_CONFIG)

    sel, ev = cfg.selection, cfg.evaluation
    lam = float(sel.get("lam", 2.0))
    ks = [int(k) for k in sel.get("k_values", DEFAULT_K)]
    strategies = [str(s) for s in sel.get("strategies", STRATEGIES)]
    seeds = [int(s) for s in ev.get("seeds", range(10))]
    train_seeds = [int(s) for s in ev.get("train_seeds", [cfg.seed])]
    cluster_method = str(ev.get("cluster_method", "eee"))
    jaccard_top = int(ev.get("jaccard_n_top", 3000))
    _validate(cfg, ks, strategies)

    if not resume:
        for name in (RESULTS_CSV, SUMMARY_JSON, "manifest.json"):
            (outdir / name).unlink(missing_ok=True)
        for stale in outdir.glob("knockout_scores*.npz"):
            stale.unlink()
        for fig in (outdir / "figures").glob("*.png"):
            fig.unlink()

    with deterministic(seed=cfg.seed, threads=cfg.threads):
        provenance = collect_provenance(seed=cfg.seed, threads=cfg.threads)
        t0 = time.time()
        slices = load_transfer_cohort(cfg)
        if verbose:
            print(
                f"loaded {len(slices)} slices x {slices[0].n_vars} genes "
                f"in {time.time() - t0:.0f}s"
            )
        expected = len(slices) * len(seeds) * len(train_seeds)
        done, _ = (
            _completed_cells(outdir / RESULTS_CSV, expected) if resume else (set(), [])
        )

        jac = hvg_jaccard(slices, jaccard_top)
        (outdir / JACCARD_JSON).write_text(json.dumps(jac, indent=2) + "\n")

        contexts = build_contexts(
            cfg, slices, outdir, strategies, resume=resume, verbose=verbose
        )
        for ctx in contexts.values():
            pool_ks = [k for k in ks if k > len(ctx.names)]
            if pool_ks and any(s != "hvg" for s in strategies):
                raise ExperimentError(
                    f"k_values {pool_ks} exceed the {len(ctx.names)}-gene pool "
                    f"({ctx.label}); drop them or enlarge selection.n_hvg_pool"
                )
        first = next(iter(contexts.values()))
        _write_gene_scores(outdir / GENES_CSV, first.names, first.deltas, lam)

        total = len(strategies) * len(ks) * len(slices)
        n_done = 0
        for k in ks:
            for strategy in strategies:
                for src_idx, source in enumerate(slices):
                    key = (strategy, k, str(source.obs["sample"].iloc[0]))
                    n_done += 1
                    if key in done:
                        continue
                    t0 = time.time()
                    rows = run_cell(
                        strategy,
                        k,
                        src_idx,
                        slices,
                        contexts,
                        cfg,
                        seeds,
                        train_seeds,
                        cluster_method,
                        lam,
                    )
                    _write_rows(outdir / RESULTS_CSV, rows)
                    done.add(key)
                    if verbose:
                        within = np.mean(
                            [r["ari"] for r in rows if r["source"] == r["target"]]
                        )
                        cross = [
                            r["ari"]
                            for r in rows
                            if r["source_donor"] != r["target_donor"]
                        ]
                        msg = (
                            f"[{n_done}/{total}] {strategy:9s} K={k:<5d} "
                            f"src={key[2]}  within {within:.3f}"
                        )
                        if cross:
                            msg += f"  cross-donor {np.mean(cross):.3f}"
                        print(msg + f"  ({time.time() - t0:.0f}s)")

    rows = _read_rows(outdir / RESULTS_CSV)
    summary = summarize(rows)
    summary["hvg_jaccard"] = jac
    summary["provenance"] = provenance
    summary["scoring_mode"] = str(sel.get("scoring_mode", "pooled"))
    summary["scoring_contexts"] = {
        label: {
            "scoring_slices": [
                str(slices[i].obs["sample"].iloc[0]) for i in ctx.scoring
            ],
            "n_pool_genes": int(len(ctx.names)),
        }
        for label, ctx in contexts.items()
    }
    summary["seeds"] = {"clustering": seeds, "training": train_seeds}
    (outdir / SUMMARY_JSON).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    if figures:
        from .figures import transfer_figures

        transfer_figures(outdir, rows, summary, first.names, first.deltas, lam, cfg)
    artifacts = {
        str(p.relative_to(outdir)): sha256_file(p)
        for p in sorted(outdir.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    (outdir / "manifest.json").write_text(
        json.dumps(
            {"run_id": cfg.run_id, "provenance": provenance, "artifacts": artifacts},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return summary


def _write_gene_scores(
    path: Path, names: np.ndarray, deltas: np.ndarray, lam: float
) -> None:
    import csv

    s_inv = invariance_scores(deltas, lam=lam)
    mean = np.nanmean(deltas, axis=0)
    std = np.nanstd(deltas, axis=0)
    order = np.argsort(-s_inv, kind="stable")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["rank", "gene", "s_inv", "mean_delta", "std_delta"]
            + [f"delta_slice{i}" for i in range(deltas.shape[0])]
        )
        for rank, i in enumerate(order, 1):
            w.writerow(
                [
                    rank,
                    names[i],
                    f"{s_inv[i]:.10g}",
                    f"{mean[i]:.10g}",
                    f"{std[i]:.10g}",
                ]
                + [f"{d:.10g}" for d in deltas[:, i]]
            )
