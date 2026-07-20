# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
