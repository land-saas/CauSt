"""Config-driven experiment runner.

Turns an :class:`~caust.config.ExperimentConfig` into a directory of artifacts:

.. code-block:: text

    results/<name>-<digest>/
      config.resolved.yaml   exactly what was run
      manifest.json          git commit, versions, platform, seed, checksums
      metrics.json           the numbers
      genes.csv              per-gene scores and selections

The experiment itself is the project's central comparison: select genes on the
training donors, evaluate on a held-out donor, and compare CauST against the
variance (HVG) baseline.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from anndata import AnnData

from .cluster import ari, nmi
from .config import ExperimentConfig, dump_config
from .data import make_synthetic_cohort
from .models.simple import SimpleSpatialModel
from .pipeline import CauST
from .repro import collect_provenance, deterministic, sha256_file

ARTIFACT_CONFIG = "config.resolved.yaml"
ARTIFACT_MANIFEST = "manifest.json"
ARTIFACT_METRICS = "metrics.json"
ARTIFACT_GENES = "genes.csv"


class ExperimentError(RuntimeError):
    """Raised when a config describes an experiment that cannot be run."""


def build_cohort(cfg: ExperimentConfig) -> list[AnnData]:
    """Materialize the cohort described by the ``data:`` section."""
    spec = dict(cfg.data)
    source = spec.pop("source", "synthetic")
    if source == "synthetic":
        spec.setdefault("seed", cfg.seed)
        try:
            return make_synthetic_cohort(**spec)
        except TypeError as exc:
            raise ExperimentError(f"invalid data: section -- {exc}") from None
    if source == "dlpfc":
        from .dlpfc import DLPFCError, load_dlpfc_cohort

        try:
            return load_dlpfc_cohort(**spec)
        except TypeError as exc:
            raise ExperimentError(f"invalid data: section -- {exc}") from None
        except DLPFCError as exc:
            raise ExperimentError(str(exc)) from None
    from .datasets import DATASETS, DatasetError, load_dataset

    if source in DATASETS:
        try:
            slices = load_dataset(
                source,
                spec.get("sections"),
                spec.get("root"),
                download=bool(spec.get("download", True)),
            )
        except DatasetError as exc:
            raise ExperimentError(str(exc)) from None
        from .dlpfc import _normalize_log1p

        for a in slices:
            _normalize_log1p(a, target_sum=float(spec.get("target_sum", 1e4)))
        return slices
    raise ExperimentError(
        f"unsupported data.source {source!r}; expected synthetic, dlpfc, "
        + ", ".join(DATASETS)
    )


def _model_factory(cfg: ExperimentConfig, random_state: int | None = None):
    """Zero-arg constructor for the configured backbone (``model.backend``)."""
    params = dict(cfg.model)
    backend = params.pop("backend", "simple")
    if random_state is not None:
        params["random_state"] = random_state
    if backend == "simple":

        def factory() -> SimpleSpatialModel:
            return SimpleSpatialModel(**params)

        return factory
    if backend == "stagate":
        try:
            from .models.stagate import STAGATEModel
        except ImportError as exc:
            raise ExperimentError(str(exc)) from None

        def stagate_factory() -> STAGATEModel:
            return STAGATEModel(**params)

        return stagate_factory
    raise ExperimentError(
        f"unsupported model.backend {backend!r}; expected 'simple' or 'stagate'"
    )


def hvg_scores(adatas: Sequence[AnnData]) -> np.ndarray:
    """Baseline gene score: mean expression variance across the given slices."""
    per_slice = []
    for a in adatas:
        X = a.X
        if sp.issparse(X):
            X = X.todense()
        per_slice.append(np.asarray(X, dtype=float).var(axis=0).ravel())
    scores: np.ndarray = np.mean(per_slice, axis=0)
    return scores


def _evaluate(
    adata: AnnData,
    genes: Sequence[str],
    n_domains: int,
    seed: int,
    n_restarts: int = 5,
    model_factory: Callable[[], Any] | None = None,
    cluster_method: str = "full",
) -> dict[str, Any]:
    """Score a gene set on a slice, averaged over several clustering restarts.

    The Gaussian mixture used for clustering converges to a seed-dependent local
    optimum, and a noisy gene set is far more sensitive to that than a clean one.
    Reporting a single restart would make the baseline comparison hinge on a
    lucky seed, so this averages over ``n_restarts`` and keeps the spread.
    """
    from .cluster import cluster_embedding

    keep = [g for g in genes if g in set(adata.var_names)]
    if not keep:
        raise ExperimentError("no selected gene is present in the evaluation slice")
    if n_restarts < 1:
        raise ExperimentError(f"evaluation.n_restarts must be >= 1, got {n_restarts}")

    sub = adata[:, keep].copy()
    model = (
        SimpleSpatialModel(random_state=seed)
        if model_factory is None
        else model_factory()
    ).fit(sub)
    embedding = model.get_embedding()
    truth = adata.obs["domain"]

    aris, nmis = [], []
    first_pred = None
    for r in range(n_restarts):
        pred = cluster_embedding(
            embedding,
            n_clusters=n_domains,
            random_state=seed + r,
            method=cluster_method,
        )
        if first_pred is None:
            first_pred = pred
        aris.append(ari(truth, pred))
        nmis.append(nmi(truth, pred))

    return {
        "ari": float(np.mean(aris)),
        "ari_std": float(np.std(aris)),
        "ari_restarts": [float(v) for v in aris],
        "nmi": float(np.mean(nmis)),
        "n_genes": len(keep),
        "n_restarts": n_restarts,
        # Kept out of metrics.json (see run_experiment); used only for plotting.
        "_labels": first_pred,
    }


def _causal_hits(genes: Sequence[str]) -> int:
    return int(sum(str(g).startswith("CAUSAL") for g in genes))


def run_experiment(
    cfg: ExperimentConfig,
    outdir: Path,
    *,
    figures: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    """Run one experiment and (optionally) write its artifacts.

    Returns the metrics dict. Everything stochastic happens inside
    :func:`~caust.repro.deterministic`, so two runs of the same config on the
    same environment produce byte-identical metrics.
    """
    outdir = Path(outdir)
    n_top = int(cfg.selection.get("n_top_genes", 8))
    lam = float(cfg.selection.get("lam", 2.0))
    holdout = bool(cfg.evaluation.get("holdout", True))

    with deterministic(seed=cfg.seed, threads=cfg.threads):
        # Captured inside the pinned block so the recorded BLAS thread counts
        # are the ones the run actually used, not the ambient ones.
        provenance = collect_provenance(seed=cfg.seed, threads=cfg.threads)
        cohort = build_cohort(cfg)
        if len(cohort) < 2 and holdout:
            raise ExperimentError(
                "holdout evaluation needs at least 2 slices; "
                "set evaluation.holdout=false or add slices"
            )
        train = cohort[:-1] if holdout else cohort
        held = cohort[-1]
        n_domains = int(cohort[0].obs["domain"].nunique())

        cs = CauST(model_factory=_model_factory(cfg), lam=lam).fit(train)
        names = np.asarray(cs.common_genes_)
        caust_score = np.asarray(cs.scores_)
        variance = hvg_scores(train)

        caust_sel = [str(g) for g in cs.select_genes(n_top)]
        hvg_sel = [str(g) for g in names[np.argsort(-variance)[:n_top]]]
        all_genes = [str(g) for g in names]

        arms = {
            "all_genes": all_genes,
            "hvg": hvg_sel,
            "caust": caust_sel,
        }
        n_restarts = int(cfg.evaluation.get("n_restarts", 5))
        cluster_method = str(cfg.evaluation.get("cluster_method", "full"))
        eval_factory = _model_factory(cfg, random_state=cfg.seed)
        scored = {
            arm: _evaluate(
                held,
                genes,
                n_domains,
                cfg.seed,
                n_restarts,
                eval_factory,
                cluster_method,
            )
            for arm, genes in arms.items()
        }
        # Predicted labels are plotting data, not metrics; keep them out of the
        # JSON so metrics.json stays small and byte-comparable.
        labels = {arm: res.pop("_labels") for arm, res in scored.items()}
        evaluation = scored

    metrics: dict[str, Any] = {
        "experiment": cfg.name,
        "run_id": cfg.run_id,
        "config_digest": cfg.digest,
        "n_slices": len(cohort),
        "n_train_slices": len(train),
        "n_spots": int(cohort[0].n_obs),
        "n_genes": int(len(names)),
        "n_domains": n_domains,
        "n_top_genes": n_top,
        "lam": lam,
        "causal_recovery": {
            "caust": _causal_hits(caust_sel),
            "hvg": _causal_hits(hvg_sel),
            "max": n_top,
        },
        "held_out": evaluation,
        "selected_genes": {"caust": caust_sel, "hvg": hvg_sel},
    }

    # Only when the config names known marker genes (real data): how many of
    # the markers that made the candidate pool did each method's top-K keep?
    # Config-gated so the metrics of existing synthetic runs stay byte-stable.
    marker_genes = cfg.evaluation.get("marker_genes")
    if marker_genes:
        markers = {str(g) for g in marker_genes}
        in_pool = sorted(markers & {str(g) for g in names})
        metrics["marker_recovery"] = {
            "caust": len(markers & set(caust_sel)),
            "hvg": len(markers & set(hvg_sel)),
            "max": len(in_pool),
            "markers_in_pool": in_pool,
        }

    if not write:
        return metrics

    outdir.mkdir(parents=True, exist_ok=True)
    dump_config(cfg, outdir / ARTIFACT_CONFIG)
    _write_metrics(outdir / ARTIFACT_METRICS, metrics)
    _write_genes(
        outdir / ARTIFACT_GENES, names, caust_score, variance, caust_sel, hvg_sel
    )

    if figures:
        _write_figures(
            outdir / "figures", cfg, names, caust_score, variance, metrics, held, labels
        )

    _write_manifest(outdir, cfg, provenance)
    return metrics


def _write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _write_genes(
    path: Path,
    names: np.ndarray,
    caust_score: np.ndarray,
    variance: np.ndarray,
    caust_sel: Sequence[str],
    hvg_sel: Sequence[str],
) -> None:
    caust_set, hvg_set = set(caust_sel), set(hvg_sel)
    order = np.argsort(-caust_score, kind="stable")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["gene", "caust_score", "variance", "selected_caust", "selected_hvg"]
        )
        for i in order:
            gene = str(names[i])
            writer.writerow(
                [
                    gene,
                    f"{float(caust_score[i]):.12g}",
                    f"{float(variance[i]):.12g}",
                    int(gene in caust_set),
                    int(gene in hvg_set),
                ]
            )


def _write_manifest(
    outdir: Path, cfg: ExperimentConfig, provenance: dict[str, Any]
) -> None:
    """Record provenance plus checksums of every artifact written so far.

    Written last so it can hash the other artifacts; it never hashes itself.
    """
    artifacts = {}
    for item in sorted(outdir.rglob("*")):
        if item.is_file() and item.name != ARTIFACT_MANIFEST:
            artifacts[str(item.relative_to(outdir))] = sha256_file(item)
    manifest = {
        "run_id": cfg.run_id,
        "config_digest": cfg.digest,
        "provenance": provenance,
        "artifacts": artifacts,
    }
    with open(outdir / ARTIFACT_MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")


GENE_CLASSES = (
    ("CAUSAL", "#2563eb", "causal (shared)"),
    ("NOISE_", "#9ca3af", "background noise"),
    ("DONORNOISE", "#dc2626", "donor-specific noise"),
)


def _gene_class(name: str) -> str:
    for prefix, _, _ in GENE_CLASSES:
        if str(name).startswith(prefix):
            return prefix
    return "NOISE_"


def match_labels(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Relabel ``pred`` to best match ``true`` (Hungarian) for readable plots.

    Cluster ids are arbitrary, so this only changes colors, never the ARI.
    """
    from scipy.optimize import linear_sum_assignment

    true_u, true_i = np.unique(true, return_inverse=True)
    pred_u, pred_i = np.unique(pred, return_inverse=True)
    counts = np.zeros((len(pred_u), len(true_u)), dtype=int)
    for p, t in zip(pred_i, true_i):
        counts[p, t] += 1
    rows, cols = linear_sum_assignment(-counts)
    mapping = dict(zip(rows.tolist(), cols.tolist()))
    return np.array([mapping.get(int(p), int(p)) for p in pred_i])


def _write_figures(
    figdir: Path,
    cfg: ExperimentConfig,
    names: np.ndarray,
    caust_score: np.ndarray,
    variance: np.ndarray,
    metrics: dict[str, Any],
    held: AnnData,
    labels: dict[str, np.ndarray],
) -> None:
    try:
        import matplotlib
    except ImportError:
        raise ExperimentError(
            'figures need matplotlib; install it with: pip install ".[viz]"'
        ) from None
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir.mkdir(parents=True, exist_ok=True)
    n_top = metrics["n_top_genes"]
    idx = np.arange(len(names))
    ev = metrics["held_out"]

    # Synthetic genes are classed by their name prefix; real genes by whether
    # they are a configured known marker (drawn last, so they sit on top).
    marker_genes = {str(g) for g in cfg.evaluation.get("marker_genes") or []}
    if marker_genes:
        is_marker = np.isin(names.astype(str), sorted(marker_genes))
        classes = [
            (~is_marker, "#9ca3af", "other genes"),
            (is_marker, "#2563eb", "known layer marker"),
        ]
    else:
        klass = np.array([_gene_class(n) for n in names])
        classes = [(klass == p, c, label) for p, c, label in GENE_CLASSES]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    ax = axes[0, 0]
    for m, color, label in classes:
        ax.scatter(idx[m], caust_score[m], s=18, c=color, label=label)
    ax.axhline(np.sort(caust_score)[-n_top], ls="--", c="k", lw=0.8)
    # Plain text rather than mathtext: matplotlib's mathtext parser pulls in a
    # deprecated pyparsing path that floods the test log with warnings.
    ax.set(
        title="A. CauST invariance score",
        xlabel="gene index",
        ylabel="invariance score",
    )
    ax.legend(fontsize=7, loc="center right")

    ax = axes[0, 1]
    for m, color, _ in classes:
        ax.scatter(idx[m], variance[m], s=18, c=color)
    ax.axhline(np.sort(variance)[-n_top], ls="--", c="k", lw=0.8)
    ax.set(title="B. Variance (HVG baseline)", xlabel="gene index", ylabel="variance")

    ax = axes[1, 0]
    if "marker_recovery" in metrics:
        rec = metrics["marker_recovery"]
        title = f"C. Known layer markers kept (of {rec['max']} in pool)"
        ylabel = f"markers in top-{n_top}"
    else:
        rec = metrics["causal_recovery"]
        title = f"C. Causal gene recovery (max {rec['max']})"
        ylabel = f"causal genes in top-{rec['max']}"
    top = max(1, rec["max"])
    ax.bar(["HVG", "CauST"], [rec["hvg"], rec["caust"]], color=["#dc2626", "#2563eb"])
    ax.set(ylim=(0, top * 1.18), ylabel=ylabel, title=title)
    for i, v in enumerate([rec["hvg"], rec["caust"]]):
        ax.text(
            i,
            v + top * 0.02,
            f"{v}/{rec['max']}",
            ha="center",
            fontweight="bold",
        )

    ax = axes[1, 1]
    arms = ("all_genes", "hvg", "caust")
    vals = [ev[a]["ari"] for a in arms]
    errs = [ev[a]["ari_std"] for a in arms]
    tick = [
        f"all\n({ev['all_genes']['n_genes']} genes)",
        f"HVG\n({n_top})",
        f"CauST\n({n_top})",
    ]
    ax.bar(tick, vals, yerr=errs, capsize=4, color=["#9ca3af", "#dc2626", "#2563eb"])
    ax.set(
        ylim=(0, 1.18),
        ylabel="ARI",
        title=f"D. Held-out donor ARI ({ev['caust']['n_restarts']} restarts)",
    )
    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(i, v + e + 0.03, f"{v:.3f}", ha="center", fontweight="bold")

    fig.suptitle(
        "CauST vs. variance-based gene selection (held-out donor)", fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(figdir / "benchmark.png", dpi=150)
    plt.close(fig)

    coords = np.asarray(held.obsm["spatial"])
    # Labels may be arbitrary strings ("Layer3", "WM"); encode to 0..K-1. For
    # synthetic string digits this is the identity, so colors are unchanged.
    truth = np.unique(np.asarray(held.obs["domain"]), return_inverse=True)[1]
    panels = [
        ("ground truth", truth),
        (f"HVG (ARI {ev['hvg']['ari']:.3f})", match_labels(truth, labels["hvg"])),
        (f"CauST (ARI {ev['caust']['ari']:.3f})", match_labels(truth, labels["caust"])),
    ]
    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (title, lab) in zip(axes2, panels):
        ax.scatter(coords[:, 0], coords[:, 1], c=lab, cmap="tab10", s=14)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.axis("off")
    fig2.suptitle("Spatial domains on the held-out donor", fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(figdir / "domains.png", dpi=150)
    plt.close(fig2)


def load_metrics(rundir: Path) -> dict[str, Any]:
    """Read the metrics written by a previous run."""
    path = Path(rundir) / ARTIFACT_METRICS
    if not path.is_file():
        raise ExperimentError(f"no {ARTIFACT_METRICS} in {rundir}")
    with open(path, encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


def verify_run(rundir: Path) -> tuple[bool, str | None]:
    """Re-run a recorded config and check the metrics still match.

    Returns ``(ok, detail)``. This is the check that turns "we wrote down the
    config" into "the config actually reproduces the numbers".
    """
    from .config import load_config

    rundir = Path(rundir)
    cfg = load_config(rundir / ARTIFACT_CONFIG)
    recorded = load_metrics(rundir)
    fresh = run_experiment(cfg, rundir, write=False)
    if fresh == recorded:
        return True, None
    diffs = [
        f"{key}: recorded={recorded.get(key)!r} fresh={fresh.get(key)!r}"
        for key in sorted(set(recorded) | set(fresh))
        if recorded.get(key) != fresh.get(key)
    ]
    return False, "; ".join(diffs)
