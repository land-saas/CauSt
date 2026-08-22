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

Knockout effects are scored once, on a small set of *scoring slices* (one per
donor by default), over a gene pool that is the cross-donor intersection of
each scoring slice's top HVGs. Runs are resumable: every (strategy, K, source)
cell is written as soon as it finishes and skipped on restart.
"""

from __future__ import annotations

import json
import time
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
DEFAULT_K = (50, 100, 200, 400, 800, 1600, 3200)
RESULTS_CSV = "results.csv"
SUMMARY_JSON = "summary.json"
GENES_CSV = "gene_scores.csv"
JACCARD_JSON = "hvg_jaccard.json"
SCORES_NPZ = "knockout_scores.npz"

_FIELDS = (
    "strategy",
    "k",
    "source",
    "target",
    "source_donor",
    "target_donor",
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
    except (ImportError, ValueError):
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
        slices = make_synthetic_cohort(**spec)
        for i, a in enumerate(slices):
            a.obs["sample"] = f"synthetic_{i}"
            a.obs["donor"] = f"donor_{i}"
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
def default_scoring_indices(slices: Sequence[AnnData]) -> list[int]:
    """First slice of each donor (the proposal scores one slice per donor)."""
    seen: dict[str, int] = {}
    for i, a in enumerate(slices):
        seen.setdefault(str(a.obs["donor"].iloc[0]), i)
    return sorted(seen.values())


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
    """Moran's I of each gene over the slice's spatial kNN graph."""
    from .graph import CONN_KEY, build_spatial_graph

    if CONN_KEY not in adata.obsp:
        build_spatial_graph(adata, n_neighbors=n_neighbors)
    W = sp.csr_matrix(adata.obsp[CONN_KEY], dtype=np.float64)
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
    slices: Sequence[AnnData], scoring: Sequence[int], names: np.ndarray, n_hvg: int
) -> dict[str, np.ndarray]:
    """Gene scores for the global baselines over the pool genes ``names``."""
    pool = list(names)
    # Donor-aware HVG: number of scoring slices in which the gene is within
    # the top-n_hvg HVGs, tie-broken by mean rank (scanpy batch_key rule).
    hits = np.zeros(len(pool))
    mean_rank = np.zeros(len(pool))
    for i in scoring:
        r = slices[i].var.loc[pool, "hvg_rank"].to_numpy(dtype=float)
        hits += r < n_hvg
        mean_rank += r
    mean_rank /= len(scoring)
    hvg_donor = hits * 1e6 - mean_rank
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
    if strategy == "random":
        gen = rng if rng is not None else np.random.default_rng(0)
        pick = gen.choice(len(names), size=min(k, len(names)), replace=False)
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
# One grid cell: train on the source, transfer everywhere
# ---------------------------------------------------------------------------
def run_cell(
    strategy: str,
    k: int,
    src_idx: int,
    slices: Sequence[AnnData],
    genes: Sequence[str],
    model_factory: Callable[[], Any],
    seeds: Sequence[int],
    cluster_method: str,
) -> list[dict[str, Any]]:
    source = slices[src_idx]
    model = model_factory().fit(source[:, list(genes)].copy())
    rows = []
    for target in slices:
        sub = target[:, list(genes)].copy()
        Z = model.get_embedding(sub)
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
                    "source": str(source.obs["sample"].iloc[0]),
                    "target": str(target.obs["sample"].iloc[0]),
                    "source_donor": str(source.obs["donor"].iloc[0]),
                    "target_donor": str(target.obs["donor"].iloc[0]),
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


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Means/stds per strategy x K x split, generalization gap, and win counts."""
    by: dict[tuple[str, int, str], list[float]] = {}
    pair: dict[tuple[str, int, str, str], list[float]] = {}
    for r in rows:
        by.setdefault((r["strategy"], int(r["k"]), _split(r)), []).append(
            float(r["ari"])
        )
        pair.setdefault(
            (r["strategy"], int(r["k"]), r["source"], r["target"]), []
        ).append(float(r["ari"]))
    nmi_by: dict[tuple[str, int, str], list[float]] = {}
    for r in rows:
        if "nmi" in r:
            nmi_by.setdefault((r["strategy"], int(r["k"]), _split(r)), []).append(
                float(r["nmi"])
            )
    strategies = sorted({s for s, _, _ in by})
    ks = sorted({k for _, k, _ in by})
    table: dict[str, Any] = {}
    for s in strategies:
        for k in ks:
            entry: dict[str, Any] = {}
            for split in ("within_slice", "within_donor", "cross_donor"):
                v = by.get((s, k, split))
                if v:
                    entry[split] = {
                        "mean": float(np.mean(v)),
                        "std": float(np.std(v)),
                        "n": len(v),
                    }
                    nv = nmi_by.get((s, k, split))
                    if nv:
                        entry[split]["nmi"] = float(np.mean(nv))
            if "within_slice" in entry and "cross_donor" in entry:
                entry["generalization_gap"] = (
                    entry["within_slice"]["mean"] - entry["cross_donor"]["mean"]
                )
            table.setdefault(s, {})[str(k)] = entry
    wins: dict[str, Any] = {}
    if "caust" in strategies and "hvg" in strategies:
        for k in ks:
            counts = {"within_slice": [0, 0], "cross_donor": [0, 0]}
            donors = {r["source"]: r["source_donor"] for r in rows}
            for (s, kk, src, tgt), v in pair.items():
                if s != "caust" or kk != k or ("hvg", k, src, tgt) not in pair:
                    continue
                if src == tgt:
                    split = "within_slice"
                elif donors[src] != donors[tgt]:
                    split = "cross_donor"
                else:
                    continue
                counts[split][1] += 1
                if np.mean(v) > np.mean(pair[("hvg", k, src, tgt)]):
                    counts[split][0] += 1
            wins[str(k)] = {
                name: {"caust_wins": a, "pairs": b} for name, (a, b) in counts.items()
            }
    return {
        "strategies": strategies,
        "k_values": ks,
        "table": table,
        "caust_vs_hvg_wins": wins,
        "paired_tests": paired_tests(pair, strategies, ks, rows),
    }


def paired_tests(
    pair: dict[tuple[str, int, str, str], list[float]],
    strategies: Sequence[str],
    ks: Sequence[int],
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Wilcoxon signed-rank test of CauST against every other strategy.

    Pairs are (source, target) cells with the seed-averaged ARI; cross-donor
    pairs and within-slice pairs are tested separately. Consecutive sections
    from one donor are near-replicates, so cross-donor is the honest test.
    """
    from scipy.stats import wilcoxon

    if "caust" not in strategies:
        return {}
    donors = {r["source"]: r["source_donor"] for r in rows}
    out: dict[str, Any] = {}
    for k in ks:
        out[str(k)] = {}
        for other in strategies:
            if other == "caust":
                continue
            res: dict[str, Any] = {}
            for split in ("within_slice", "cross_donor"):
                a, b = [], []
                for (s, kk, src, tgt), v in pair.items():
                    if s != "caust" or kk != k or ("caust", k, src, tgt) not in pair:
                        continue
                    if (other, k, src, tgt) not in pair:
                        continue
                    is_within = src == tgt
                    is_cross = donors[src] != donors[tgt]
                    if (split == "within_slice" and is_within) or (
                        split == "cross_donor" and is_cross
                    ):
                        a.append(float(np.mean(v)))
                        b.append(float(np.mean(pair[(other, k, src, tgt)])))
                if len(a) >= 2 and np.any(np.subtract(a, b) != 0):
                    p = float(wilcoxon(a, b).pvalue)
                else:
                    p = float("nan")
                res[split] = {
                    "mean_diff": float(np.mean(a) - np.mean(b)) if a else float("nan"),
                    "p_value": p,
                    "n_pairs": len(a),
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
            w.writerow({f: r[f] for f in _FIELDS})


def _read_rows(path: Path) -> list[dict[str, Any]]:
    import csv

    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


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
    n_hvg_pool = int(sel.get("n_hvg_pool", 8000))
    seeds = [int(s) for s in ev.get("seeds", range(10))]
    cluster_method = str(ev.get("cluster_method", "eee"))
    jaccard_top = int(ev.get("jaccard_n_top", 3000))

    done = (
        {
            (r["strategy"], int(r["k"]), r["source"])
            for r in _read_rows(outdir / RESULTS_CSV)
        }
        if resume
        else set()
    )
    if not resume and (outdir / RESULTS_CSV).exists():
        (outdir / RESULTS_CSV).unlink()

    with deterministic(seed=cfg.seed, threads=cfg.threads):
        provenance = collect_provenance(seed=cfg.seed, threads=cfg.threads)
        t0 = time.time()
        slices = load_transfer_cohort(cfg)
        if verbose:
            print(
                f"loaded {len(slices)} slices x {slices[0].n_vars} genes "
                f"in {time.time() - t0:.0f}s"
            )
        scoring = [
            int(i) for i in sel.get("scoring_slices", default_scoring_indices(slices))
        ]

        jac = hvg_jaccard(slices, jaccard_top)
        (outdir / JACCARD_JSON).write_text(json.dumps(jac, indent=2) + "\n")

        scores_path = outdir / SCORES_NPZ
        if resume and scores_path.exists():
            z = np.load(scores_path, allow_pickle=False)
            names, deltas = z["names"].astype(str), z["deltas"]
        else:
            pool = build_pool(slices, scoring, n_hvg_pool)
            if verbose:
                print(
                    f"knockout pool: {len(pool)} genes (intersection of "
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
            np.savez(scores_path, names=np.asarray(names, dtype=str), deltas=deltas)
            if verbose:
                print(
                    f"scored {len(names)} genes on {len(scoring)} slices "
                    f"in {time.time() - t0:.0f}s"
                )
        _write_gene_scores(outdir / GENES_CSV, names, deltas, lam)

        baselines = (
            baseline_scores(slices, scoring, names, n_hvg_pool)
            if any(st in ("hvg_donor", "moran") for st in strategies)
            else {}
        )
        factory = _model_factory(cfg, random_state=cfg.seed)
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
                    genes = gene_set(
                        strategy,
                        k,
                        source,
                        names,
                        deltas,
                        lam,
                        baselines,
                        np.random.default_rng(cfg.seed + 1000 * k + src_idx),
                    )
                    rows = run_cell(
                        strategy,
                        k,
                        src_idx,
                        slices,
                        genes,
                        factory,
                        seeds,
                        cluster_method,
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
                        print(msg + f"  ({time.time()-t0:.0f}s)")

    rows = _read_rows(outdir / RESULTS_CSV)
    summary = summarize(rows)
    summary["hvg_jaccard"] = jac
    summary["provenance"] = provenance
    summary["scoring_slices"] = [str(slices[i].obs["sample"].iloc[0]) for i in scoring]
    summary["n_pool_genes"] = int(len(names))
    (outdir / SUMMARY_JSON).write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    if figures:
        from .figures import transfer_figures

        transfer_figures(outdir, rows, summary, names, deltas, lam, cfg)
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
