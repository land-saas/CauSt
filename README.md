# CauST

**Causal Gene Intervention for Robust Spatial Domain Identification**

[![CI](https://github.com/land-saas/CauSt/actions/workflows/ci.yml/badge.svg)](https://github.com/land-saas/CauSt/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Coverage](https://img.shields.io/badge/coverage-%E2%89%A599%25-brightgreen.svg)](#development)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue.svg)](https://mypy-lang.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/land-saas/CauSt/blob/main/LICENSE)

CauST selects genes for spatial domain identification by *causal intervention*
rather than by variance (highly variable genes, HVGs). It silences each gene
in-silico on a frozen spatial model, measures how much the spatial embedding
shifts, and keeps only genes whose effect is **large and stable across donors**.
This yields gene sets that transfer across tissue slices far better than HVGs.

This repository ships a dependency-light reference backbone so the full pipeline
runs end-to-end today; real GNN backbones (STAGATE / GraphST / SpaGCN) plug in
via a small interface. The method and empirical results are documented in the
[technical report](https://github.com/land-saas/CauSt/blob/main/docs/report/caust_report.pdf).

## The method (3 steps)

1. **Gene knockout scoring** — for each gene *g* in slice *e*, zero its
   expression, run the frozen model, and measure the mean per-spot embedding
   shift `δ(g,e)`.
2. **Cross-slice invariance** — score each gene by
   `s_inv(g) = mean_e[δ(g,e)] − λ·std_e[δ(g,e)]`.
3. **Retrain** — feed the top-*K* causal genes (or soft sigmoid weights) to any
   downstream domain-identification model.

See [`docs/methods.md`](https://github.com/land-saas/CauSt/blob/main/docs/methods.md)
for the formal derivation and the
[technical report](https://github.com/land-saas/CauSt/blob/main/docs/report/caust_report.pdf)
for the full write-up.

## Results: CauST vs. the variance (HVG) baseline

Three synthetic donors share a set of causal genes that define the spatial
domains identically in every donor. Each donor also carries its own
donor-specific noise genes, which have **higher variance** than the causal genes
but carry no cross-donor signal — the trap that variance-based HVG selection
falls into. Genes are selected on the training donors only and evaluated on a
**held-out donor**, which is the actual robustness claim.

```
$ uv run caust run -c configs/experiment/synthetic_holdout.yaml

Causal genes recovered in the top-8:
  HVG    1/8
  CauST  8/8

Held-out donor 3 (never used for gene selection, 5 restarts):
  ARI  all genes  (68 genes)  1.000 +/- 0.000
  ARI  HVG        ( 8 genes)  0.537 +/- 0.133
  ARI  CauST      ( 8 genes)  1.000 +/- 0.000
```

CauST matches the all-genes accuracy using **8× fewer genes**, while the variance
baseline at the same budget loses roughly 0.46 ARI. ARI is averaged over five
clustering restarts because the Gaussian mixture lands in a seed-dependent local
optimum — and the spread is itself informative: the CauST gene set clusters
identically every time (±0.000), while the HVG set swings between 0.44 and 0.80
(±0.133) depending on the seed.

![CauST vs HVG benchmark](https://raw.githubusercontent.com/land-saas/CauSt/main/docs/figures/benchmark.png)

Panels A and B are the crux: the invariance score isolates the causal genes,
while variance ranks the donor-specific noise genes highest.

![Spatial domains on the held-out donor](https://raw.githubusercontent.com/land-saas/CauSt/main/docs/figures/domains.png)

Every run is config-driven and writes a content-addressed directory under
`results/` with the resolved config, the metrics, and a provenance manifest
(git commit, package versions, platform, seed, BLAS thread counts, artifact
checksums). `caust verify <rundir>` re-runs the recorded config and fails if the
numbers have moved. See [REPRODUCIBILITY.md](https://github.com/land-saas/CauSt/blob/main/REPRODUCIBILITY.md)
for the full workflow.

```bash
make repro          # run the headline experiment
make verify         # re-run every recorded result and confirm it still matches
make determinism    # prove two separate processes produce byte-identical output
```

## The same claim on real tissue

The synthetic trap is engineered; the DLPFC benchmark is not. `make dlpfc`
fetches five Visium slices of human dorsolateral prefrontal cortex from the
spatialLIBD dataset (three donors, manual cortical-layer annotations; ~60 MB,
cached and checksum-verified), selects genes on the two training donors, and
evaluates on the third donor — which the selection never saw:

```
$ uv run caust run -c configs/experiment/dlpfc_holdout.yaml --figures

held-out donor:
  ARI  all_genes  (2000 genes)  0.378 +/- 0.026
  ARI  hvg        (  25 genes)  0.351 +/- 0.034
  ARI  caust      (  25 genes)  0.459 +/- 0.069
```

Twenty-five CauST-selected genes beat both the variance baseline at the same
budget (+0.11 ARI) and the full 2,000-gene candidate pool on the unseen donor.
The knobs (lambda, gene budget) were tuned on the training donors only — the
held-out donor was evaluated exactly once.

![Spatial domains on the held-out DLPFC donor](https://raw.githubusercontent.com/land-saas/CauSt/main/docs/figures/dlpfc_domains.png)

## Install

The project is managed with [uv](https://docs.astral.sh/uv/). One command
creates `.venv` from the committed `uv.lock` — the package plus all dev
tooling, on the pinned interpreter:

```bash
uv sync
```

Prefix any command with `uv run` (no manual venv activation needed), or use
the `make` targets below. Installing with pip also works for the runtime
package:

```bash
pip install -e .          # core (numpy / scipy / scikit-learn / anndata)
pip install -e ".[viz]"   # + matplotlib, for the benchmark figures
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
uv run scripts/demo.py
# or, via the console script:
uv run caust demo --slices 3 --lam 2.0
```

See [`docs/demo.md`](https://github.com/land-saas/CauSt/blob/main/docs/demo.md)
for a guided five-minute tour of the whole project.

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

## Development

```bash
make install-dev   # uv sync (+ docs group) + pre-commit hooks
make check         # ruff + mypy + pytest with coverage (fails under 90%)
make format        # auto-format with ruff --fix and black
make docs          # build the MkDocs site
```

The project targets clean `ruff`, `black`, and `mypy`, and ≥90% test coverage,
all enforced in [CI](https://github.com/land-saas/CauSt/blob/main/.github/workflows/ci.yml).

## Documentation

- User guide and API reference: [`docs/`](https://github.com/land-saas/CauSt/blob/main/docs) (served with MkDocs).
- Formal methods: [`docs/methods.md`](https://github.com/land-saas/CauSt/blob/main/docs/methods.md).
- Technical report (PDF): [`docs/report/caust_report.pdf`](https://github.com/land-saas/CauSt/blob/main/docs/report/caust_report.pdf).

## Status

The pipeline, scoring, invariance selection, and tests work on synthetic and
real `AnnData`. Not yet wired: real STAGATE/GraphST training, integrated-
gradients attribution, and the full DLPFC benchmark. See the
[technical report](https://github.com/land-saas/CauSt/blob/main/docs/report/caust_report.pdf) for the roadmap.

## Citing

If you use CauST in academic work, please cite it (see [`CITATION.cff`](https://github.com/land-saas/CauSt/blob/main/CITATION.cff)):

```bibtex
@software{li_caust_2026,
  author  = {Li, Jiawei},
  title   = {CauST: Causal Gene Intervention for Robust Spatial Domain Identification},
  year    = {2026},
  url     = {https://github.com/land-saas/CauSt}
}
```

## License

[MIT](https://github.com/land-saas/CauSt/blob/main/LICENSE) © 2026 Jiawei Li.
