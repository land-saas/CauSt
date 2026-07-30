"""CauST vs. the variance (HVG) baseline -- the project's central claim.

This is a thin wrapper around the config-driven runner so there is exactly one
implementation of the experiment. Prefer the CLI directly:

    caust run -c configs/experiment/synthetic_holdout.yaml --figures

Anything you can pass here can also be expressed in the config, which is what
gets recorded alongside the results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from caust.config import load_config
from caust.experiment import run_experiment

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment" / "synthetic_holdout.yaml"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--slices", type=int, default=None, help="override data.n_slices")
    p.add_argument("--n-causal", type=int, default=None, help="override data.n_causal")
    p.add_argument("--lam", type=float, default=None, help="override selection.lam")
    p.add_argument("--figures", type=Path, default=None, metavar="DIR")
    p.add_argument("-o", "--outdir", type=Path, default=REPO_ROOT / "results")
    args = p.parse_args(argv)

    overrides = []
    if args.slices is not None:
        overrides.append(f"data.n_slices={args.slices}")
    if args.n_causal is not None:
        overrides += [
            f"data.n_causal={args.n_causal}",
            f"selection.n_top_genes={args.n_causal}",
        ]
    if args.lam is not None:
        overrides.append(f"selection.lam={args.lam}")

    cfg = load_config(args.config, overrides)
    rundir = Path(args.outdir) / cfg.run_id
    metrics = run_experiment(cfg, rundir, figures=args.figures is not None)

    ev = metrics["held_out"]
    rec = metrics["causal_recovery"]
    print(
        f"Cohort: {metrics['n_slices']} donors x {metrics['n_spots']} spots x "
        f"{metrics['n_genes']} genes ({metrics['n_domains']} domains)"
    )
    print(
        f"Selecting genes on donors 1-{metrics['n_train_slices']}; "
        f"donor {metrics['n_slices']} held out.\n"
    )
    print(f"Causal genes recovered in the top-{rec['max']}:")
    print(f"  HVG    {rec['hvg']}/{rec['max']}")
    print(f"  CauST  {rec['caust']}/{rec['max']}")
    print(
        f"\nHeld-out donor {metrics['n_slices']} "
        f"(never used for gene selection, {ev['caust']['n_restarts']} restarts):"
    )
    for arm, label in (("all_genes", "all genes"), ("hvg", "HVG"), ("caust", "CauST")):
        r = ev[arm]
        print(
            f"  ARI  {label:<10s} ({r['n_genes']:>2d} genes)  "
            f"{r['ari']:.3f} +/- {r['ari_std']:.3f}"
        )

    if args.figures is not None:
        src = rundir / "figures"
        dest = Path(args.figures)
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("benchmark.png", "domains.png"):
            if (src / name).is_file():
                dest.joinpath(name).write_bytes((src / name).read_bytes())
        print(f"\nFigures written to {dest}/benchmark.png and {dest}/domains.png")
    print(f"Artifacts written to {rundir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
