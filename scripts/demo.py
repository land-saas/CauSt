"""End-to-end CauST demo on synthetic multi-donor data.

Run:  python scripts/demo.py

Shows that CauST's knockout-invariance score ranks the shared CAUSAL_* genes
above the high-variance DONORNOISE_* genes, and that clustering on the causal
gene set recovers the ground-truth spatial domains.
"""
from __future__ import annotations

import numpy as np

from caust import CauST, SimpleSpatialModel, ari, cluster_embedding
from caust.data import make_synthetic_cohort


def main() -> None:
    print("Generating 3-donor synthetic cohort...")
    cohort = make_synthetic_cohort(n_slices=3, grid=20, n_causal=8, seed=0)
    n_domains = cohort[0].obs["domain"].nunique()

    print("Running CauST (lambda=2.0)...")
    cs = CauST(lam=2.0).fit(cohort, verbose=True)

    print("\nTop-12 genes by invariance score:")
    for name, score in cs.ranking()[:12]:
        tag = "  <-- causal" if name.startswith("CAUSAL") else ""
        print(f"  {name:14s} {score:8.4f}{tag}")

    # How many true causal genes land in the top-8?
    top8 = set(cs.select_genes(8))
    recovered = sum(g.startswith("CAUSAL") for g in top8)
    print(f"\nCausal genes recovered in top-8: {recovered}/8")

    # Downstream: cluster on the causal gene set vs. all genes (donor 0).
    causal_ad = cs.transform(cohort[0], n_top_genes=8)
    model = SimpleSpatialModel().fit(causal_ad)
    pred = cluster_embedding(model.get_embedding(), n_clusters=n_domains)
    score = ari(cohort[0].obs["domain"], pred)
    print(f"ARI on causal gene set (donor 0): {score:.3f}")


if __name__ == "__main__":
    main()
