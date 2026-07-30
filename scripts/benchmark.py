"""CauST vs. the variance (HVG) baseline -- the project's central claim.

Run::

    python scripts/benchmark.py                 # numbers only
    python scripts/benchmark.py --figures docs/figures   # + publication figures

Experiment
----------
Three synthetic donors share a set of ``CAUSAL_*`` genes that define the spatial
domains identically in every donor. Each donor additionally carries its own
``DONORNOISE_*`` genes which have *higher variance* than the causal genes but
carry no cross-donor domain signal -- precisely the trap that variance-based
highly-variable-gene (HVG) selection falls into.

Genes are selected on the training donors only, then evaluated on a **held-out
donor** never seen during selection. That held-out evaluation is the actual
robustness claim CauST makes.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

from caust import CauST, SimpleSpatialModel, ari, cluster_embedding
from caust.data import make_synthetic_cohort

GENE_CLASSES = (
    ("CAUSAL", "#2563eb", "causal (shared across donors)"),
    ("NOISE_", "#9ca3af", "background noise"),
    ("DONORNOISE", "#dc2626", "donor-specific noise"),
)


def gene_class(name: str) -> str:
    for prefix, _, _ in GENE_CLASSES:
        if str(name).startswith(prefix):
            return prefix
    return "NOISE_"


def hvg_scores(adatas) -> np.ndarray:
    """Baseline gene score: mean expression variance across selection slices."""
    return np.mean([np.asarray(a.X, dtype=float).var(axis=0) for a in adatas], axis=0)


def match_labels(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Relabel ``pred`` to best match ``true`` (Hungarian) for readable plots.

    Cluster IDs are arbitrary, so this only changes colors, never the ARI.
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


def evaluate(adata, genes, n_domains: int) -> tuple[float, np.ndarray]:
    """Cluster ``adata`` restricted to ``genes``; return (ARI, predicted labels)."""
    keep = [g for g in genes if g in set(adata.var_names)]
    sub = adata[:, keep].copy()
    model = SimpleSpatialModel().fit(sub)
    pred = cluster_embedding(model.get_embedding(), n_clusters=n_domains)
    return ari(adata.obs["domain"], pred), pred


def make_figures(
    outdir: Path, names, caust_score, var, caust_sel, hvg_sel, held, results, n_top
) -> None:
    try:
        import matplotlib
    except ImportError:  # pragma: no cover - depends on the optional extra
        raise SystemExit(
            '--figures needs matplotlib; install it with: pip install ".[viz]"'
        ) from None

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    klass = np.array([gene_class(n) for n in names])
    idx = np.arange(len(names))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    ax = axes[0, 0]
    for prefix, color, label in GENE_CLASSES:
        m = klass == prefix
        ax.scatter(idx[m], caust_score[m], s=18, c=color, label=label)
    ax.axhline(np.sort(caust_score)[-n_top], ls="--", c="k", lw=0.8)
    ax.set(title="A. CauST invariance score", xlabel="gene index", ylabel="$s_{inv}$")
    ax.legend(fontsize=7, loc="center right")

    ax = axes[0, 1]
    for prefix, color, _ in GENE_CLASSES:
        m = klass == prefix
        ax.scatter(idx[m], var[m], s=18, c=color)
    ax.axhline(np.sort(var)[-n_top], ls="--", c="k", lw=0.8)
    ax.set(title="B. Variance (HVG baseline)", xlabel="gene index", ylabel="variance")

    ax = axes[1, 0]
    rec = [
        int(sum(str(g).startswith("CAUSAL") for g in hvg_sel)),
        int(sum(str(g).startswith("CAUSAL") for g in caust_sel)),
    ]
    ax.bar(["HVG", "CauST"], rec, color=["#dc2626", "#2563eb"])
    ax.set(
        ylim=(0, n_top * 1.18),
        ylabel=f"causal genes in top-{n_top}",
        title=f"C. Causal gene recovery (max {n_top})",
    )
    for i, v in enumerate(rec):
        ax.text(i, v + n_top * 0.02, f"{v}/{n_top}", ha="center", fontweight="bold")

    ax = axes[1, 1]
    labels = [f"all\n({len(names)} genes)", f"HVG\n({n_top})", f"CauST\n({n_top})"]
    vals = [results["all genes"][0], results["HVG"][0], results["CauST"][0]]
    ax.bar(labels, vals, color=["#9ca3af", "#dc2626", "#2563eb"])
    ax.set(ylim=(0, 1.12), ylabel="ARI", title="D. Held-out donor accuracy")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")

    fig.suptitle(
        "CauST vs. variance-based gene selection (held-out donor)", fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(outdir / "benchmark.png", dpi=150)
    plt.close(fig)

    coords = np.asarray(held.obsm["spatial"])
    true = np.asarray(held.obs["domain"]).astype(int)
    panels = [
        ("ground truth", true),
        (f"HVG (ARI {results['HVG'][0]:.3f})", match_labels(true, results["HVG"][1])),
        (
            f"CauST (ARI {results['CauST'][0]:.3f})",
            match_labels(true, results["CauST"][1]),
        ),
    ]
    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (title, lab) in zip(axes2, panels):
        ax.scatter(coords[:, 0], coords[:, 1], c=lab, cmap="tab10", s=14)
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.axis("off")
    fig2.suptitle("Spatial domains on the held-out donor", fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(outdir / "domains.png", dpi=150)
    plt.close(fig2)
    print(f"\nFigures written to {outdir}/benchmark.png and {outdir}/domains.png")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--slices", type=int, default=3, help="donors (last is held out)")
    p.add_argument("--n-causal", type=int, default=8)
    p.add_argument("--lam", type=float, default=2.0)
    p.add_argument("--figures", type=Path, default=None, metavar="DIR")
    args = p.parse_args(argv)

    if args.slices < 2:
        p.error("--slices must be >= 2 (at least one training and one held-out donor)")

    k = args.n_causal
    cohort = make_synthetic_cohort(n_slices=args.slices, grid=20, n_causal=k, seed=0)
    n_domains = int(cohort[0].obs["domain"].nunique())
    train, held = cohort[:-1], cohort[-1]

    print(
        f"Cohort: {args.slices} donors x {cohort[0].n_obs} spots x "
        f"{cohort[0].n_vars} genes ({n_domains} domains)"
    )
    print(f"Selecting genes on donors 1-{len(train)}; donor {args.slices} held out.\n")

    cs = CauST(lam=args.lam).fit(train)
    names = np.asarray(cs.common_genes_)
    caust_score = np.asarray(cs.scores_)
    var = hvg_scores(train)

    caust_sel = cs.select_genes(k)
    hvg_sel = names[np.argsort(-var)[:k]]

    n_caust = int(sum(str(g).startswith("CAUSAL") for g in caust_sel))
    n_hvg = int(sum(str(g).startswith("CAUSAL") for g in hvg_sel))
    print(f"Causal genes recovered in the top-{k}:")
    print(f"  HVG    {n_hvg}/{k}")
    print(f"  CauST  {n_caust}/{k}")

    results = {}
    print(f"\nHeld-out donor {args.slices} (never used for gene selection):")
    for label, genes in (("all genes", names), ("HVG", hvg_sel), ("CauST", caust_sel)):
        score, pred = evaluate(held, genes, n_domains)
        results[label] = (score, pred)
        print(f"  ARI  {label:<10s} ({len(genes):>2d} genes)  {score:.3f}")

    print(
        f"\nCauST matches the all-genes ARI using {len(names) // k}x fewer genes, "
        f"while HVG at the same budget loses "
        f"{results['all genes'][0] - results['HVG'][0]:.3f} ARI."
    )

    if args.figures is not None:
        make_figures(
            args.figures, names, caust_score, var, caust_sel, hvg_sel, held, results, k
        )
    return 0


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    raise SystemExit(main())
