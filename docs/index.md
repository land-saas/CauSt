# CauST

**Causal Gene Intervention for Robust Spatial Domain Identification**

CauST selects genes for spatial domain identification by *causal intervention*
rather than by variance (highly variable genes, HVGs). It silences each gene
in-silico on a frozen spatial model, measures how much the spatial embedding
shifts, and keeps only genes whose effect is **large and stable across donors**.
The result is a gene set that transfers across tissue slices far better than a
variance-selected one.

## Why causal gene selection?

State-of-the-art spatial-domain methods (STAGATE, GraphST, SpaGCN) achieve
strong *within-slice* accuracy but pick genes with variance heuristics that are
neither causally grounded nor stable across tissue sections. Because clinical
and translational applications need models that generalize across slices from
different donors — where batch effects and donor-specific expression shift the
gene landscape — this lack of cross-slice robustness is a critical bottleneck.

CauST reframes gene selection as an interventional question: *if we silenced a
gene entirely, would the spatial domain structure change, and would that change
hold across donors?* Genes that pass both tests are kept.

## The method in three steps

1. **Gene knockout scoring** — zero each gene's expression on a frozen model and
   measure the mean per-spot embedding shift `δ(g,e)`.
2. **Cross-slice invariance** — combine per-slice effects into
   `s_inv(g) = mean_e[δ(g,e)] − λ·std_e[δ(g,e)]`, rewarding large *and stable*
   effects.
3. **Retrain** — feed the top-*K* causal genes (or soft weights) to any
   downstream domain-identification model.

See [Methods](methods.md) for the formal treatment and [Usage](usage.md) to get
started.

## Highlights

- Backbone-agnostic: any model implementing `BaseSpatialModel` plugs in.
- Dependency-light reference backbone runs the full pipeline out of the box.
- Fully typed, linted, and tested (≥99% coverage).
