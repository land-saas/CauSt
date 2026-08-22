# Contributing to CauST

Thanks for your interest in improving CauST. This guide covers the development
workflow and the quality gates the project enforces.

## Development setup

The project is managed with [uv](https://docs.astral.sh/uv/)
([install instructions](https://docs.astral.sh/uv/getting-started/installation/)).

```bash
git clone https://github.com/land-saas/CauSt.git
cd CauSt
make install-dev      # uv sync (+ docs group) + pre-commit hooks
```

`uv sync` creates `.venv` from the committed `uv.lock` on the interpreter
pinned in `.python-version` — an editable install of the package plus the
`dev` dependency group. There is no venv to activate manually: every `make`
target runs through `uv run`, which keeps the environment in sync with the
lockfile. If you change dependencies in `pyproject.toml`, run `make lock` and
commit the updated `uv.lock` (CI installs with `--locked` and fails on drift).

## Quality gates

All of the following must pass before a pull request can be merged. They run
automatically in [CI](.github/workflows/ci.yml) and can be run locally:

| Check          | Command            | Requirement                     |
| -------------- | ------------------ | ------------------------------- |
| Lint           | `make lint`        | `ruff check` reports no errors  |
| Format         | `make format`      | `black` + `ruff --fix` clean    |
| Type check     | `make typecheck`   | `mypy` reports no errors        |
| Tests          | `make test`        | all tests pass                  |
| Coverage       | `make cov`         | ≥ 90% line + branch coverage    |

`make check` runs lint, type-check, and coverage together.

## Coding conventions

- Public functions and classes carry type hints and NumPy-style docstrings.
- New behavior ships with tests; keep coverage at or above 90%.
- Keep the core package dependency-light (numpy / scipy / scikit-learn /
  anndata). Heavy backbones belong behind an optional extra (e.g. `stagate`).
- Backbones implement `caust.models.base.BaseSpatialModel` so they drop into the
  pipeline via `model_factory`.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org): `feat:`,
`fix:`, `docs:`, `test:`, `refactor:`, `chore:`, etc.

## Reporting issues

Open an issue at https://github.com/land-saas/CauSt/issues with a minimal
reproducible example (ideally built on `make_synthetic_cohort`).

## Releasing

Releases are published to PyPI by `.github/workflows/release.yml` through
[trusted publishing](https://docs.pypi.org/trusted-publishers/) — no API token
is stored anywhere. The workflow runs when a GitHub release is *published*, and
refuses to upload if the release tag disagrees with `version` in
`pyproject.toml` (PyPI burns a version number permanently on first upload).

1. Bump `version` in `pyproject.toml` (the single source; `caust.__version__`
   reads it from the installed metadata), move the `[Unreleased]` entries in
   `CHANGELOG.md` under a dated heading, and update `date-released` in
   `CITATION.cff`. Commit, push, and wait for CI to be green.
2. Optional rehearsal: `make publish-test` builds, checks, and uploads to
   TestPyPI (export a TestPyPI token as `UV_PUBLISH_TOKEN` in your shell first;
   never write it to a file).
3. `gh release create vX.Y.Z --target main --title "CauST X.Y.Z" --notes-file <(...)`
   — the tag is created at the tip of `main`, the workflow builds the sdist and
   wheel, runs `twine check --strict`, publishes to PyPI, and attaches the same
   files to the GitHub release.
4. Confirm: `pip install caust==X.Y.Z` in a fresh environment and check
   https://pypi.org/project/caust/.
