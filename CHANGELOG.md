# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

## [0.1.0] - 2026-08-13

First public release on PyPI (`pip install caust`).

### Added
- **Config-driven experiments and a reproducibility harness.** Experiments are
  described in YAML under `configs/` (with `defaults:` inheritance and
  `--set key=value` overrides) and run with `caust run`, which writes a
  content-addressed directory under `results/` containing the resolved config,
  metrics, per-gene scores, and a provenance manifest (git commit and dirty
  flag, package versions, platform, seed, BLAS thread counts, artifact
  checksums). `caust verify <rundir>` re-runs a recorded config and fails if the
  numbers moved. New modules: `caust.config`, `caust.experiment`, `caust.repro`.
- A `Dockerfile` for a hermetic run, `REPRODUCIBILITY.md`, and `make repro` /
  `verify` / `determinism` / `lock` targets.
- A CI job that runs the same config in two separate processes and fails unless
  the artifacts are byte-identical.
- **Real-data support: the spatialLIBD DLPFC cohort.** `caust.dlpfc` fetches
  10x Visium slices of human cortex (counts + manual layer annotations) with
  pinned SHA-256 checksums and a scanpy-free 10x HDF5 reader, and
  `configs/experiment/dlpfc_holdout.yaml` (or `make dlpfc`) runs the headline
  claim on real tissue: genes selected on two donors, evaluated once on a
  held-out third donor. 25 CauST genes reach ARI 0.459 ± 0.069 on the unseen
  donor vs 0.351 ± 0.034 for the HVG baseline at the same budget and
  0.378 ± 0.026 for the full 2,000-gene pool. Configs may list known marker
  genes (`evaluation.marker_genes`) to report marker recovery alongside.
- **Rank-1 knockout scoring in the reference backbone.** Zeroing one gene
  shifts one standardized input column, and the rest of the model is linear,
  so each knockout embedding is the base embedding minus a smoothed rank-1
  update — O(N·d) per gene instead of O(N·G·d). Scoring 2,000 real genes
  across five slices takes seconds. (Recorded results re-verify after
  regeneration; low-order float bits differ from the old full-forward path.)
- `scripts/benchmark.py` — a held-out-donor comparison against the variance
  (HVG) baseline, the project's central claim, plus the figures in
  `docs/figures/`. CauST recovers 8/8 causal genes to HVG's 1/8 and matches the
  all-genes ARI (1.000 vs 0.798) using 8x fewer genes.
- A regression test asserting CauST beats variance-based selection on a held-out
  donor; previously no test covered the baseline the project claims to beat.
- A release-workflow guard that fails if the git tag disagrees with the packaged
  version, since PyPI burns a version number permanently on first upload.
- `make benchmark`; `ruff`/`black` now also cover `scripts/`, which was unlinted.
- Packaging metadata, classifiers, and `py.typed` marker for a typed
  distribution.
- Continuous integration (lint, type-check, test matrix over Python
  3.9–3.12), release, and docs GitHub Actions workflows.
- Pre-commit configuration (ruff, black, mypy, hygiene hooks) and a developer
  `Makefile`.
- Expanded test suite (unit + pipeline + CLI) reaching ≥99% coverage.
- MkDocs documentation site, formal methods notes, `CONTRIBUTING.md`,
  `CITATION.cff`, and a LaTeX technical report.
- Initial CauST prototype: in-silico gene knockout scoring, cross-slice
  invariance selection, the multi-slice pipeline, a dependency-light reference
  backbone, synthetic multi-donor data, and a command-line demo.

### Changed
- **The project is now managed with [uv](https://docs.astral.sh/uv/).** A
  committed `uv.lock` universally pins every dependency (all platforms and
  Python versions); dev/docs tooling moved from extras to PEP 735 dependency
  groups; the Makefile, CI, and Dockerfile all run through `uv sync --locked` /
  `uv run`, so the same lockfile backs local development, CI, and the hermetic
  Docker image. `requirements.txt` and `requirements.lock` are gone —
  `uv.lock` is the single source of truth (`uv export` can regenerate a
  requirements-style file if ever needed).
- **Held-out ARI is now averaged over five clustering restarts.** The Gaussian
  mixture converges to a seed-dependent local optimum, and the HVG gene set is
  far more sensitive to that than the CauST set. The previously reported HVG ARI
  of 0.798 was the best of five restarts; the honest figure is **0.537 ± 0.133**
  against CauST's 1.000 ± 0.000. The spread is itself a result: clustering on the
  CauST gene set is stable across seeds, while the HVG set is not.
- `scripts/benchmark.py` is now a thin wrapper over `caust run`, so the script
  and the CLI can no longer report different numbers for the same experiment.
- Run outputs under `results/` are no longer blanket-ignored by git; a specific
  run can be committed with `git add -f` as a reference point.
- Adopted the PEP 639 SPDX license form (`license = "MIT"` + `license-files`) and
  dropped the deprecated License classifier, so builds are warning-free.
- Standardized the distribution name to `caust`.
- Type-annotated the public API so `mypy` passes with no errors.
- Renamed `models/stagate_adpater.py` → `models/stagate_adapter.py`.

### Fixed
- The provenance manifest recorded the machine's ambient BLAS thread count
  rather than the pinned count the run actually used.
- `caust verify` crashed instead of reporting an error when a run directory had
  no `config.resolved.yaml`.
- **`MANIFEST.in`: the source distribution was missing `tests/conftest.py`**, so
  the shipped test suite failed with 8 fixture errors (32 passed / 8 errors).
  setuptools' default sdist manifest globs `test*.py`, which does not match
  `conftest.py`. The sdist also omitted `scripts/`, `docs/`, and every top-level
  metadata file, breaking the `python scripts/demo.py` instruction in the
  shipped README. The distribution now carries the full suite (41 passed) and a
  CI job builds the sdist, installs it, and runs its tests to prevent regression.
- Silenced spurious BLAS floating-point warnings (`divide by zero` / `overflow` /
  `invalid value encountered in matmul`) raised by macOS Accelerate under
  numpy 2.x during `GaussianMixture`'s k-means init and PCA's covariance path.
  Inputs and outputs were always finite; the demo now runs with clean stderr.
- Knockout-scoring progress reported `0/68` and never advanced, because it
  printed every 200 genes on a 68-gene problem. It now reports about ten times
  across whatever number of genes are scored.
- Declared `matplotlib` (new `viz` extra) instead of relying on it being present.
- `mypy` was configured for Python 3.10 while the package supports 3.9, so 3.9
  incompatibilities could pass type-checking.

### Removed
- Stray zero-byte placeholder files and a misplaced editor settings file.

[Unreleased]: https://github.com/land-saas/CauSt/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/land-saas/CauSt/releases/tag/v0.1.0
