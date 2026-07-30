# Usage

## Installation

```bash
pip install -e .          # core runtime
pip install -e ".[dev]"   # + test / lint / type-check tooling
pip install -e ".[docs]"  # + documentation tooling
```

## Quick start

```python
from caust import CauST
from caust.data import make_synthetic_cohort

cohort = make_synthetic_cohort(n_slices=3, n_causal=8, seed=0)  # 3 donors
cs = CauST(lam=2.0).fit(cohort)

cs.select_genes(8)          # names of the top-8 causally invariant genes
cs.ranking()[:5]            # (gene, score) pairs, best first
cs.soft_weights()           # sigmoid weights over all common genes
cs.transform(cohort[0], 8)  # AnnData subset to the causal gene set
```

## Running an experiment

Experiments are described in YAML under `configs/` rather than hard-coded in a
script, and each run writes a content-addressed directory with its resolved
config, metrics, and a provenance manifest:

```bash
caust run -c configs/experiment/synthetic_holdout.yaml
caust run -c configs/experiment/lambda_sweep.yaml --set selection.lam=1.0
caust verify results/synthetic_holdout-<digest>/
```

See [REPRODUCIBILITY.md](https://github.com/land-saas/CauSt/blob/main/REPRODUCIBILITY.md)
for what is controlled (seed, BLAS thread count, environment) and how to
reproduce a published result.

## Command-line demo

```bash
caust demo --slices 3 --lam 2.0 --n-causal 8
```

The demo builds a synthetic three-donor cohort, runs CauST, prints the
top-ranked genes, reports how many ground-truth causal genes were recovered, and
clusters the causal gene set to compute an ARI.

## Bringing your own data

Provide one `AnnData` per slice/donor with:

- expression in `adata.X` (spots × genes), with **aligned** `var_names` across
  slices, and
- spot coordinates in `adata.obsm['spatial']`.

CauST intersects `var_names` across slices, builds a spatial graph per slice,
trains a fresh backbone, scores every common gene by knockout, and combines the
per-slice effects into the invariance score.

```python
import anndata as ad
from caust import CauST

slices = [ad.read_h5ad(p) for p in ("donor1.h5ad", "donor2.h5ad", "donor3.h5ad")]
cs = CauST(lam=2.0).fit(slices, verbose=True)
causal_genes = cs.select_genes(100)
```

## Plugging in a real backbone

CauST is backbone-agnostic. Implement
[`BaseSpatialModel`](api.md#caust.models.base.BaseSpatialModel) and pass a
zero-argument factory:

```python
from caust import CauST
from caust.models.base import BaseSpatialModel

class MyBackbone(BaseSpatialModel):
    def fit(self, adata): ...
    def get_embedding(self, adata=None): ...
    def forward(self, X): ...
    def _get_expression(self): ...

cs = CauST(model_factory=MyBackbone, lam=2.0).fit(slices)
```

The only requirement is a **frozen** forward pass `f(X, A)` so that knockout
scoring reflects the input perturbation, not training stochasticity.

## Choosing λ and K

- `λ` (the invariance penalty) trades effect magnitude against cross-donor
  stability. `λ=0` reduces to selecting genes by mean knockout effect alone;
  `λ≈2` is the empirically strong setting for cross-donor transfer.
- `K` is the size of the retained gene set. CauST is competitive at small `K`
  (e.g. 100 genes) precisely because it discards donor-specific artifacts.
