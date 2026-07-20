# CauST developer Makefile
.DEFAULT_GOAL := help
PY ?= python

.PHONY: help install install-dev lint format typecheck test cov demo docs build clean check

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package (runtime deps only)
	$(PY) -m pip install -e .

install-dev:  ## Install with dev + docs extras and pre-commit hooks
	$(PY) -m pip install -e ".[dev,docs]"
	pre-commit install

lint:  ## Run ruff linter
	ruff check src tests

format:  ## Auto-format with black and ruff --fix
	ruff check --fix src tests
	black src tests

typecheck:  ## Run mypy static type checking
	mypy

test:  ## Run the test suite
	pytest

cov:  ## Run tests with coverage (fails under 90%)
	pytest --cov=caust --cov-report=term-missing --cov-fail-under=90

demo:  ## Run the end-to-end synthetic demo
	$(PY) scripts/demo.py

docs:  ## Build the documentation site
	mkdocs build --strict

build:  ## Build sdist and wheel
	$(PY) -m build

check: lint typecheck cov  ## Lint + type-check + tests with coverage

clean:  ## Remove caches and build artifacts
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .mypy_cache \
		.ruff_cache .coverage coverage.xml htmlcov site
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
