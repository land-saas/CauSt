# CauST

**Causal Gene Intervention for Robust Spatial Domain Identification**

CauST selects genes for spatial domain identification by *causal intervention*
rather than by variance (highly variable genes, HVGs). It silences each gene
in-silico on a frozen spatial model, measures how much the spatial embedding
shifts, and keeps only genes whose effect is **large and stable across donors**.
This yields gene sets that transfer across tissue slices far better than HVGs.

This repository is a **working prototype** of the method described in the CauST
proposal. It ships a dependency-light reference backbone so the full pipeline
runs end-to-end today; real GNN backbones (STAGATE / GraphST / SpaGCN) plug in
via a small interface.

## The method (3 steps)

1. **Gene knockout scoring** — for each gene *g* in slice *e*, zero its
   expression, run the frozen model, and measure the mean per-spot embedding
   shift `δ(g,e)`.
2. **Cross-slice invariance** — score each gene by
   `s_inv(g) = mean_e[δ(g,e)] − λ·std_e[δ(g,e)]`.
3. **Retrain** — feed the top-*K* causal genes (or soft sigmoid weights) to any
   downstream domain-identification model.

## Install

```bash
pip install -e .          # core (numpy / scipy / scikit-learn / anndata)
pip install -e ".[dev]"   # + pytest
```

## Quick start

```python
from caust import CauST
from caust.data import make_synthetic_cohort

cohort = make_synthetic_cohort(n_slices=3, n_causal=8, seed=0)  # 3 donors
cs = CauST(lam=2.0).fit(cohort)

cs.select_genes(8)          # top-8 causally invariant genes
cs.ranking()[:5]            # (gene, score) pairs, best first
cs.transform(cohort[0], 8)  # AnnData subset to the causal gene set
```

Or run the demo:

```bash
python scripts/demo.py
# or, after install:
caust demo --slices 3 --lam 2.0
```

## Using your own data

Provide one `AnnData` per slice/donor with:
- expression in `adata.X` (spots × genes), aligned `var_names` across slices,
- spot coordinates in `adata.obsm['spatial']`.

CauST builds the spatial graph, trains a fresh backbone per slice, and scores
genes. To use a real backbone, implement
`caust.models.base.BaseSpatialModel` and pass a `model_factory` to `CauST`.

## Package layout

```
src/caust/
  graph.py         spatial neighbor graph construction
  intervention.py  Step 1 — in-silico knockout scoring
  invariance.py    Steps 2 & 3 — invariance score + gene selection
  pipeline.py      CauST orchestrator (multi-slice)
  cluster.py       clustering + ARI / NMI metrics
  data.py          synthetic multi-donor data
  models/          backbones (SimpleSpatialModel; STAGATE stub)
```

## Status

Prototype — the pipeline, scoring, and tests work on synthetic and real
`AnnData`. Not yet wired: real STAGATE/GraphST training, integrated-gradients
attribution, and the full DLPFC benchmark. See the proposal for the roadmap.
