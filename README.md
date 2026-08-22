# CauST

**Causal gene selection for spatial domain identification that transfers across donors.**

[![PyPI](https://img.shields.io/pypi/v/caust.svg)](https://pypi.org/project/caust/)
[![CI](https://github.com/land-saas/CauSt/actions/workflows/ci.yml/badge.svg)](https://github.com/land-saas/CauSt/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://pypi.org/project/caust/)
[![Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen.svg)](#development)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Spatial-domain methods pick their input genes by variance (HVGs). Those genes
are unstable across donors — only ~21% of a slice's top HVGs are shared with a
slice from another donor — so a model trained on one donor is built on features
the next donor lacks. CauST selects genes differently: it silences each gene
*in silico* on a frozen spatial model, measures how far the spatial embedding
moves, and keeps the genes whose effect is **large and stable across donors**:

```
δ(g, e)   = mean per-spot embedding shift when gene g is knocked out in slice e
s_inv(g)  = mean_e δ(g, e) − λ · std_e δ(g, e)          →  keep the top-K genes
```

Any frozen backbone works. The package ships a pure-PyTorch **STAGATE**
(validated against the reference implementation), a **GraphST** port, and a
dependency-light linear reference model.

## Install

```bash
pip install caust                # core: numpy / scipy / scikit-learn / anndata
pip install "caust[stagate]"     # + PyTorch and scanpy, for the GNN backbones
pip install "caust[viz]"         # + matplotlib, for figures and caust.pl
```

## Quick start

Scanpy-style: one `AnnData` per slice (log-normalized `.X`, shared `var_names`,
coordinates in `obsm["spatial"]`), results written back into `.var` / `.uns`.

```python
import caust

table = caust.tl.causal_genes(slices, backbone="stagate", lam=2.0)  # score every shared gene
genes = caust.tl.select_genes(slices[0], n_top_genes=100)           # top-K causal gene set
caust.tl.spatial_domains(slices[0], n_domains=7, backbone="stagate", genes=genes)
caust.pl.ranking(slices[0], n=20, markers=["MBP", "NRGN", "SNAP25"])
```

Real data, one command — the experiment configs ship with the package:

```bash
caust run -c dlpfc_holdout --figures     # select on two donors, test on a third (~10 s + 60 MB download)
caust transfer -c dlpfc_quick --figures  # 12-slice cross-donor transfer benchmark (GPU, ~2 h)
caust configs                            # list everything that ships
```

## Results on real tissue

Human DLPFC, 10x Visium (spatialLIBD; 12 slices, 3 donors, manual layer
annotations), fetched from public mirrors and checksum-verified.

**Held-out donor.** Genes selected on two donors, evaluated once on a third
donor the selection never saw, at a 25-gene budget:

| Gene set | Genes | ARI on the unseen donor |
|---|---|---|
| All candidate genes | 2,000 | 0.378 ± 0.026 |
| Highly variable genes | 25 | 0.351 ± 0.034 |
| **CauST** | **25** | **0.459 ± 0.069** |

![Spatial domains on the held-out DLPFC donor](https://raw.githubusercontent.com/land-saas/CauSt/main/docs/figures/dlpfc_domains.png)

**12-slice cross-donor transfer** (STAGATE backbone, every slice as source,
zero-shot to every other slice, paired tests over sources). CauST beats HVG
selection at equal gene count on all 12 source slices at K=400
(cross-donor ARI 0.430 vs 0.374, p < 0.001); at K ≤ 100 the edge is small and
not significant. The cross-donor stability penalty is what separates CauST
from plain knockout magnitude at small budgets (11/12 sources at K=50,
p = 0.002).

![ARI vs K](https://raw.githubusercontent.com/land-saas/CauSt/main/docs/figures/dlpfc_transfer_ari_vs_k.png)

With no label supervision, the top-ranked genes include the known cortical
layer markers MBP, CCK, NRGN, CALM1 and SNAP25. Full tables, paired statistics,
and the honest comparison with the proposal's prototype figures are in
[docs/results.md](docs/results.md). An executed walkthrough is in
[docs/tutorials/dlpfc_quickstart.ipynb](docs/tutorials/dlpfc_quickstart.ipynb).

## Reproducibility

Every run is config-driven and writes a content-addressed directory with the
resolved config, metrics, and a provenance manifest (git commit, package
versions, seed, BLAS thread counts, artifact checksums). `caust verify <rundir>`
re-runs a recorded config and fails if the numbers moved; CI proves two
separate processes produce byte-identical artifacts. One committed `uv.lock`
backs local development, CI, and the Docker image. See
[REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Documentation

| | |
|---|---|
| [Five-minute demo](docs/demo.md) | Synthetic proof of concept → real tissue → `caust verify`, from a fresh clone |
| [Usage](docs/usage.md) | The `caust.tl` / `caust.pl` API, backbones, the transfer benchmark |
| [Results](docs/results.md) | Everything the code reproduces on real data, with statistics |
| [Methods](docs/methods.md) | The three steps, formally |
| [Related work](docs/related_work.md) | Where CauST sits relative to STAMarker, Geneformer, V-REx, SVG benchmarks |
| [Roadmap](docs/roadmap.md) | Proposal objectives → status |
| [Technical report](docs/report/caust_report.pdf) | Method, results, and engineering write-up |

## Development

```bash
git clone https://github.com/land-saas/CauSt.git && cd CauSt
uv sync --extra stagate      # locked environment, GNN backbones included
make check                   # ruff, black, mypy, 139 tests, 90% coverage gate
make dlpfc                   # the held-out-donor experiment with figures
make transfer-quick          # reduced 12-slice transfer grid
```

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md), which also
describes the release process (trusted publishing to PyPI, no tokens).

## Citing

```bibtex
@software{li2026caust,
  author  = {Li, Jiawei},
  title   = {CauST: Causal Gene Intervention for Robust Spatial Domain Identification},
  year    = {2026},
  url     = {https://github.com/land-saas/CauSt},
  version = {0.1.0}
}
```

Developed as an [Open Source Research Experience 2026](https://ucsc-ospo.github.io/osre26/)
project with UC Santa Cruz OSPO, mentored by Lijinghua Zhang (UC Irvine).
Data: Maynard et al., *Nature Neuroscience* 2021 (spatialLIBD); STAGATE: Dong &
Zhang, *Nature Communications* 2022; GraphST: Long et al., *Nature Communications* 2023.

## License

MIT — see [LICENSE](LICENSE).
