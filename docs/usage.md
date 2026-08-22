# Usage

## Installation

```bash
pip install caust               # core runtime (numpy / scipy / scikit-learn / anndata)
pip install "caust[stagate]"    # + PyTorch and scanpy, for the STAGATE backbone
pip install "caust[viz]"        # + matplotlib, for figures and caust.pl

# development checkout
uv sync --extra stagate
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

## Scanpy-style workflow (`caust.tl` / `caust.pl`)

The tools API takes AnnData objects and writes results back into `.var` /
`.uns`, the way `scanpy.tl` does. One AnnData per slice, each with expression
in `.X` (log-normalized), shared `var_names`, and coordinates in
`obsm["spatial"]`:

```python
import caust

# Score every shared gene by knockout invariance across donors
table = caust.tl.causal_genes(slices, backbone="stagate", lam=2.0)
table.head(20)                                     # ranked genes

# Pick a gene set (lambda can be changed without re-running knockouts)
genes = caust.tl.select_genes(slices[0], n_top_genes=100, lam=1.0)
w = caust.tl.soft_gene_weights(slices[0])          # or soft reweighting

# Domains from the causal gene set, with mclust-EEE-equivalent clustering
caust.tl.spatial_domains(slices[0], n_domains=7, backbone="stagate", genes=genes)

caust.pl.ranking(slices[0], n=20, markers=["MBP", "NRGN", "SNAP25"])
caust.pl.domains(slices[0], "caust_domain")
```

A fully executed walkthrough of this workflow on the DLPFC cohort is in
[`docs/tutorials/dlpfc_quickstart.ipynb`](https://github.com/land-saas/CauSt/blob/main/docs/tutorials/dlpfc_quickstart.ipynb).

## The STAGATE backbone

`caust.models.stagate.STAGATEModel` is a pure-PyTorch implementation of the
STAGATE graph attention autoencoder (no torch_geometric), validated against
the reference implementation on DLPFC. It runs on CUDA, Apple MPS, or CPU:

```python
from caust import CauST
from caust.models.stagate import STAGATEModel

cs = CauST(model_factory=lambda: STAGATEModel(n_epochs=1000, device="auto"), lam=2.0)
cs.fit(slices, verbose=True)
```

A fitted model embeds another slice **zero-shot** on that slice's own spatial
graph (`model.get_embedding(other_adata)`), which is what the cross-slice
transfer benchmark measures. `model.input_attribution()` returns
integrated-gradients attributions for every gene from a few backward passes;
`caust.models.stagate.attribution_fidelity` reports how well they rank genes
like exact knockouts do, so the proxy is only used where it is shown to hold.

## Cross-slice transfer benchmark

The proposal's protocol — every slice as source, zero-shot transfer to every
other slice, three gene-selection strategies at equal gene count, several
seeds — is one command, resumable across restarts:

```bash
caust transfer -c configs/transfer/dlpfc_stagate.yaml --figures    # full grid (hours)
caust transfer -c configs/transfer/dlpfc_quick.yaml --figures      # reduced grid
```

The run directory holds `results.csv` (one row per source/target/seed),
`summary.json` (within-slice, within-donor and cross-donor ARI per strategy
and K, the generalization gap, and CauST-vs-HVG win counts), `gene_scores.csv`,
`hvg_jaccard.json`, and the figures.

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
