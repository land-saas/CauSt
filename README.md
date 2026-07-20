# CauST

**Causal Gene Intervention for Robust Spatial Domain Identification**

[![CI](https://github.com/land-saas/CauSt/actions/workflows/ci.yml/badge.svg)](https://github.com/land-saas/CauSt/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org)
[![Coverage](https://img.shields.io/badge/coverage-%E2%89%A599%25-brightgreen.svg)](#development)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue.svg)](https://mypy-lang.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

CauST selects genes for spatial domain identification by *causal intervention*
rather than by variance (highly variable genes, HVGs). It silences each gene
in-silico on a frozen spatial model, measures how much the spatial embedding
shifts, and keeps only genes whose effect is **large and stable across donors**.
This yields gene sets that transfer across tissue slices far better than HVGs.

This repository ships a dependency-light reference backbone so the full pipeline
runs end-to-end today; real GNN backbones (STAGATE / GraphST / SpaGCN) plug in
via a small interface. The method and empirical results are documented in the
[technical report](docs/report/caust_report.pdf).

## The method (3 steps)

1. **Gene knockout scoring** — for each gene *g* in slice *e*, zero its
   expression, run the frozen model, and measure the mean per-spot embedding
   shift `δ(g,e)`.
2. **Cross-slice invariance** — score each gene by
   `s_inv(g) = mean_e[δ(g,e)] − λ·std_e[δ(g,e)]`.
3. **Retrain** — feed the top-*K* causal genes (or soft sigmoid weights) to any
   downstream domain-identification model.

See [`docs/methods.md`](docs/methods.md) for the formal derivation and the
[technical report](docs/report/caust_report.pdf) for the full write-up.

## Install

```bash
pip install -e .          # core (numpy / scipy / scikit-learn / anndata)
pip install -e ".[dev]"   # + test, lint, type-check tooling
pip install -e ".[docs]"  # + documentation tooling
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

## Development

```bash
make install-dev   # editable install + dev/docs extras + pre-commit hooks
make check         # ruff + mypy + pytest with coverage (fails under 90%)
make format        # auto-format with ruff --fix and black
make docs          # build the MkDocs site
```

The project targets clean `ruff`, `black`, and `mypy`, and ≥90% test coverage,
all enforced in [CI](.github/workflows/ci.yml).

## Documentation

- User guide and API reference: [`docs/`](docs/) (served with MkDocs).
- Formal methods: [`docs/methods.md`](docs/methods.md).
- Technical report (PDF): [`docs/report/caust_report.pdf`](docs/report/caust_report.pdf).

## Status

The pipeline, scoring, invariance selection, and tests work on synthetic and
real `AnnData`. Not yet wired: real STAGATE/GraphST training, integrated-
gradients attribution, and the full DLPFC benchmark. See the
[technical report](docs/report/caust_report.pdf) for the roadmap.

## Citing

If you use CauST in academic work, please cite it (see [`CITATION.cff`](CITATION.cff)):

```bibtex
@software{li_caust_2026,
  author  = {Li, Jiawei},
  title   = {CauST: Causal Gene Intervention for Robust Spatial Domain Identification},
  year    = {2026},
  url     = {https://github.com/land-saas/CauSt}
}
```

## License

[MIT](LICENSE) © 2026 Jiawei Li.
