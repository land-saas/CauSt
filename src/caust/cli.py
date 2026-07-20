"""Minimal command-line entry point: ``caust demo``."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="caust", description="CauST toolkit")
    sub = parser.add_subparsers(dest="command")
    d = sub.add_parser("demo", help="run the synthetic multi-donor demo")
    d.add_argument("--slices", type=int, default=3)
    d.add_argument("--lam", type=float, default=2.0)
    d.add_argument("--n-causal", type=int, default=8)

    args = parser.parse_args(argv)
    if args.command == "demo":
        from .cluster import ari, cluster_embedding
        from .data import make_synthetic_cohort
        from .models.simple import SimpleSpatialModel
        from .pipeline import CauST

        cohort = make_synthetic_cohort(
            n_slices=args.slices, n_causal=args.n_causal, seed=0
        )
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

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
