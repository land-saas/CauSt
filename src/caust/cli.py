"""Command-line entry point: ``caust demo``, ``run``, ``verify``, ``transfer``."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_RESULTS_DIR = Path("results")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="caust", description="CauST toolkit")
    sub = parser.add_subparsers(dest="command")

    d = sub.add_parser("demo", help="run the synthetic multi-donor demo")
    d.add_argument("--slices", type=int, default=3)
    d.add_argument("--lam", type=float, default=2.0)
    d.add_argument("--n-causal", type=int, default=8)

    r = sub.add_parser("run", help="run a config-driven experiment")
    r.add_argument("-c", "--config", type=Path, required=True, help="YAML config path")
    r.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="parent directory for run output (default: results/)",
    )
    r.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config value, e.g. --set selection.lam=1.0",
    )
    r.add_argument("--figures", action="store_true", help="also write figures")

    v = sub.add_parser("verify", help="re-run a recorded run and compare metrics")
    v.add_argument("rundir", type=Path, help="a results/<run_id> directory")

    t = sub.add_parser(
        "transfer", help="cross-slice transfer benchmark (every slice as source)"
    )
    t.add_argument("-c", "--config", type=Path, required=True, help="YAML config path")
    t.add_argument("-o", "--outdir", type=Path, default=DEFAULT_RESULTS_DIR)
    t.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE"
    )
    t.add_argument("--figures", action="store_true", help="also write figures")
    t.add_argument(
        "--fresh",
        action="store_true",
        help="discard finished cells instead of resuming",
    )

    return parser


def _cmd_demo(args: argparse.Namespace) -> int:
    from .cluster import ari, cluster_embedding
    from .data import make_synthetic_cohort
    from .models.simple import SimpleSpatialModel
    from .pipeline import CauST

    cohort = make_synthetic_cohort(n_slices=args.slices, n_causal=args.n_causal, seed=0)
    cs = CauST(lam=args.lam).fit(cohort, verbose=True)
    print("\nTop genes by invariance score:")
    for name, score in cs.ranking()[: args.n_causal + 4]:
        print(f"  {name:14s} {score:8.4f}")
    causal_ad = cs.transform(cohort[0], n_top_genes=args.n_causal)
    model = SimpleSpatialModel().fit(causal_ad)
    n_domains = cohort[0].obs["domain"].nunique()
    pred = cluster_embedding(model.get_embedding(), n_clusters=n_domains)
    print(f"ARI (causal set, donor 0): {ari(cohort[0].obs['domain'], pred):.3f}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .config import ConfigError, load_config
    from .experiment import ExperimentError, run_experiment

    try:
        cfg = load_config(args.config, args.overrides)
    except ConfigError as exc:
        print(f"error: {exc}")
        return 2

    rundir = Path(args.outdir) / cfg.run_id
    print(f"experiment : {cfg.name}")
    print(f"run id     : {cfg.run_id}")
    print(f"seed/threads: {cfg.seed}/{cfg.threads}")
    print(f"output     : {rundir}")

    try:
        metrics = run_experiment(cfg, rundir, figures=args.figures)
    except ExperimentError as exc:
        print(f"error: {exc}")
        return 2

    if "marker_recovery" in metrics:
        rec = metrics["marker_recovery"]
        n_top = metrics["n_top_genes"]
        print(
            f"\nknown layer markers kept in the top-{n_top} "
            f"(of {rec['max']} in pool):"
        )
    else:
        rec = metrics["causal_recovery"]
        print(f"\ncausal genes recovered (top-{rec['max']}):")
    print(f"  HVG    {rec['hvg']}/{rec['max']}")
    print(f"  CauST  {rec['caust']}/{rec['max']}")
    print("\nheld-out donor:")
    for arm in ("all_genes", "hvg", "caust"):
        res = metrics["held_out"][arm]
        print(
            f"  ARI  {arm:<10s} ({res['n_genes']:>4d} genes)  "
            f"{res['ari']:.3f} +/- {res['ari_std']:.3f}"
        )
    print(f"\nartifacts written to {rundir}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from .config import ConfigError
    from .experiment import ExperimentError, verify_run

    try:
        ok, detail = verify_run(args.rundir)
    except (ConfigError, ExperimentError, OSError) as exc:
        print(f"error: {exc}")
        return 2
    if ok:
        print(f"REPRODUCED: {args.rundir} matches a fresh run of its recorded config")
        return 0
    print(f"MISMATCH: {args.rundir} does not reproduce")
    print(detail or "")
    return 1


def _cmd_transfer(args: argparse.Namespace) -> int:
    from .config import ConfigError, load_config
    from .experiment import ExperimentError
    from .transfer import run_transfer

    try:
        cfg = load_config(args.config, args.overrides)
    except ConfigError as exc:
        print(f"error: {exc}")
        return 2
    rundir = Path(args.outdir) / cfg.run_id
    print(f"benchmark : {cfg.name}\nrun id    : {cfg.run_id}\noutput    : {rundir}")
    try:
        summary = run_transfer(cfg, rundir, figures=args.figures, resume=not args.fresh)
    except ExperimentError as exc:
        print(f"error: {exc}")
        return 2
    print("\ncross-donor transfer ARI (mean over source-target pairs and seeds):")
    print("  " + "K".rjust(9) + "".join(f"{s:>12s}" for s in summary["strategies"]))
    for k in summary["k_values"]:
        cells = []
        for s in summary["strategies"]:
            e = summary["table"][s].get(str(k), {}).get("cross_donor")
            cells.append(f"{e['mean']:.3f}" if e else "-")
        print("  " + f"{k:>9d}" + "".join(f"{c:>12s}" for c in cells))
    tests = summary.get("paired_tests", {})
    if tests:
        print("\nCauST vs each strategy, cross-donor, one value per source (Wilcoxon):")
        for k in summary["k_values"]:
            for other, res in tests.get(str(k), {}).items():
                r = res["source"]
                print(
                    f"  K={k:<5d} vs {other:10s} diff {r['mean_diff']:+.3f}  "
                    f"wins {r['caust_wins']}/{r['n_units']}  p={r['p_value']:.3g}"
                )
    print(f"\nartifacts written to {rundir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "demo":
        return _cmd_demo(args)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "verify":
        return _cmd_verify(args)
    if args.command == "transfer":
        return _cmd_transfer(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
