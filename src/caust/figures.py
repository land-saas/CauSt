"""Figures for the transfer benchmark (matplotlib, ``viz`` extra)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .invariance import invariance_scores

#: Canonical human cortical layer markers (spatialLIBD / Allen), used to star
#: known genes in the ranking plot. Reporting only -- never used for selection.
DLPFC_LAYER_MARKERS = (
    "AQP4", "RELN", "HPCAL1", "LAMP5", "CUX2", "CARTPT", "RORB", "PCP4",
    "TRABD2A", "FEZF2", "KRT17", "NTNG2", "MOBP", "MBP", "PLP1", "CNP",
    "SNAP25", "NRGN", "CCK", "CALM1", "GPM6A", "ENC1", "NEFL", "NEFM",
)  # fmt: skip

COLORS = {
    "hvg": "#dc2626",
    "hvg_donor": "#f97316",
    "moran": "#a855f7",
    "random": "#9ca3af",
    "highdelta": "#2563eb",
    "caust": "#16a34a",
}
LABELS = {
    "hvg": "HVG",
    "hvg_donor": "Donor-aware HVG",
    "moran": "Moran's I",
    "random": "Random",
    "highdelta": "High-δ",
    "caust": "CauST",
}


def _plt() -> Any:
    try:
        import matplotlib
    except ImportError:
        raise RuntimeError(
            'figures need matplotlib; install it with: pip install "caust[viz]"'
        ) from None
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def transfer_figures(
    outdir: Path,
    rows: Sequence[dict[str, Any]],
    summary: dict[str, Any],
    names: np.ndarray,
    deltas: np.ndarray,
    lam: float,
    cfg: ExperimentConfig,
) -> list[Path]:
    plt = _plt()
    figdir = Path(outdir) / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    written = []
    table = summary["table"]
    ks = summary["k_values"]
    strategies = [s for s in COLORS if s in table]

    # 1. ARI vs K, three evaluation settings.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, split, title in zip(
        axes,
        ("within_slice", "within_donor", "cross_donor"),
        ("Within-slice ARI", "Within-donor transfer ARI", "Cross-donor transfer ARI"),
    ):
        for s in strategies:
            xs, ys, es = [], [], []
            for k in ks:
                e = table[s].get(str(k), {}).get(split)
                if e:
                    xs.append(k)
                    ys.append(e["mean"])
                    es.append(e["std"])
            if not xs:
                continue
            ax.plot(xs, ys, "o-", color=COLORS[s], label=LABELS[s])
            ax.fill_between(
                xs, np.subtract(ys, es), np.add(ys, es), color=COLORS[s], alpha=0.12
            )
        ax.set(title=title, xlabel="K (number of genes)", xscale="log")
        ax.set_xticks(ks), ax.set_xticklabels([str(k) for k in ks])
    axes[0].set_ylabel("ARI")
    axes[0].legend()
    fig.tight_layout()
    written.append(figdir / "ari_vs_k.png")
    fig.savefig(written[-1], dpi=150), plt.close(fig)

    # 2. Generalization gap (within-slice minus cross-donor).
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for s in strategies:
        xs = [k for k in ks if "generalization_gap" in table[s].get(str(k), {})]
        ys = [table[s][str(k)]["generalization_gap"] for k in xs]
        if xs:
            ax.plot(xs, ys, "o-", color=COLORS[s], label=LABELS[s])
    ax.axhline(0, color="k", lw=0.6)
    ax.set(
        title="Generalization gap",
        xlabel="K (number of genes)",
        ylabel="ARI drop (within → cross-donor)",
        xscale="log",
    )
    ax.set_xticks(ks), ax.set_xticklabels([str(k) for k in ks])
    ax.legend(), fig.tight_layout()
    written.append(figdir / "generalization_gap.png")
    fig.savefig(written[-1], dpi=150), plt.close(fig)

    # 3. Transfer matrices at the reference K (100 if present, else the closest).
    k_ref = min(ks, key=lambda k: abs(k - 100))
    samples = sorted({r["source"] for r in rows})
    donors = {r["source"]: r["source_donor"] for r in rows}
    donors.update({r["target"]: r["target_donor"] for r in rows})
    idx = {s: i for i, s in enumerate(samples)}
    mats = {}
    for s in strategies:
        m = np.full((len(samples), len(samples)), np.nan)
        acc: dict[tuple[str, str], list[float]] = {}
        for r in rows:
            if r["strategy"] == s and int(r["k"]) == k_ref:
                acc.setdefault((r["source"], r["target"]), []).append(float(r["ari"]))
        for (a, b), v in acc.items():
            m[idx[a], idx[b]] = np.mean(v)
        mats[s] = m
    fig, axes = plt.subplots(1, len(strategies), figsize=(5.2 * len(strategies), 5))
    axes = np.atleast_1d(axes)
    vmin = min(np.nanmin(m) for m in mats.values())
    vmax = max(np.nanmax(m) for m in mats.values())
    boundaries = [
        i
        for i in range(1, len(samples))
        if donors[samples[i]] != donors[samples[i - 1]]
    ]
    for ax, s in zip(axes, strategies):
        im = ax.imshow(mats[s], cmap="YlOrRd", vmin=vmin, vmax=vmax)
        ax.set(
            title=f"{LABELS[s]} (K={k_ref})",
            xlabel="Target slice",
            ylabel="Source slice",
        )
        ax.set_xticks(range(len(samples))), ax.set_xticklabels(
            samples, rotation=90, fontsize=7
        )
        ax.set_yticks(range(len(samples))), ax.set_yticklabels(samples, fontsize=7)
        for cut in boundaries:
            ax.axhline(cut - 0.5, color="white", lw=1.5)
            ax.axvline(cut - 0.5, color="white", lw=1.5)
        if len(samples) <= 12:
            for i in range(len(samples)):
                for j in range(len(samples)):
                    if np.isfinite(mats[s][i, j]):
                        ax.text(
                            j,
                            i,
                            f"{mats[s][i, j]:.2f}",
                            ha="center",
                            va="center",
                            fontsize=5,
                        )
    fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label="ARI")
    fig.suptitle("Transfer ARI matrix", fontweight="bold")
    written.append(figdir / f"transfer_matrix_K{k_ref}.png")
    fig.savefig(written[-1], dpi=150, bbox_inches="tight"), plt.close(fig)

    # 4. Top-20 invariance ranking with known markers starred.
    markers = {
        str(g) for g in cfg.evaluation.get("marker_genes") or DLPFC_LAYER_MARKERS
    }
    s_inv = invariance_scores(deltas, lam=lam)
    order = np.argsort(-s_inv, kind="stable")[:20]
    top = [str(names[i]) for i in order][::-1]
    vals = (s_inv[order] / max(s_inv[order].max(), 1e-12))[::-1]
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["#f97316" if i >= 10 else "#3b82f6" for i in range(20)]
    ax.barh(top, vals, color=colors)
    for i, g in enumerate(top):
        if g in markers:
            ax.text(vals[i] + 0.01, i, "★", color="#b91c1c", va="center", fontsize=9)
    ax.set(
        xlabel="Invariance score (normalized)",
        title=f"Top-20 causally invariant genes (λ={lam:g})",
    )
    if markers:
        ax.text(
            0.98,
            0.02,
            "★ known marker gene",
            transform=ax.transAxes,
            ha="right",
            fontsize=8,
            color="#b91c1c",
        )
    fig.tight_layout()
    written.append(figdir / "top_genes.png")
    fig.savefig(written[-1], dpi=150), plt.close(fig)
    return written
