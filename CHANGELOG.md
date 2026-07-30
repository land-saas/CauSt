# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
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

### Added
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

### Changed
- Adopted the PEP 639 SPDX license form (`license = "MIT"` + `license-files`) and
  dropped the deprecated License classifier, so builds are warning-free.
- Standardized the distribution name to `caust`.
- Type-annotated the public API so `mypy` passes with no errors.
- Renamed `models/stagate_adpater.py` → `models/stagate_adapter.py`.

### Removed
- Stray zero-byte placeholder files and a misplaced editor settings file.

## [0.1.0] - 2026-03-01

### Added
- Initial CauST prototype: in-silico gene knockout scoring, cross-slice
  invariance selection, the multi-slice pipeline, a dependency-light reference
  backbone, synthetic multi-donor data, and a command-line demo.
