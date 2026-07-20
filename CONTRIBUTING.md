# Contributing to CauST

Thanks for your interest in improving CauST. This guide covers the development
workflow and the quality gates the project enforces.

## Development setup

```bash
git clone https://github.com/land-saas/CauSt.git
cd CauSt
python -m venv venv && source venv/bin/activate
make install-dev      # editable install + dev/docs extras + pre-commit hooks
```

`make install-dev` installs the package with the `dev` and `docs` extras and
registers the pre-commit hooks so formatting and linting run on every commit.

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
