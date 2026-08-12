# CauST developer Makefile.
# Every target runs through uv (https://docs.astral.sh/uv/), so nothing here
# requires an activated virtualenv — uv keeps .venv in sync with uv.lock.
.DEFAULT_GOAL := help

.PHONY: help install install-dev lint format typecheck test cov demo benchmark repro verify determinism lock docs build clean check

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Create .venv from uv.lock (package + dev tooling)
	uv sync

install-dev:  ## uv sync with the docs group, plus pre-commit hooks
	uv sync --group docs
	uv run pre-commit install

lint:  ## Run ruff linter
	uv run ruff check src tests scripts

format:  ## Auto-format with black and ruff --fix
	uv run ruff check --fix src tests scripts
	uv run black src tests scripts

typecheck:  ## Run mypy static type checking
	uv run mypy

test:  ## Run the test suite
	uv run pytest

cov:  ## Run tests with coverage (fails under 90%)
	uv run pytest --cov=caust --cov-report=term-missing --cov-fail-under=90

demo:  ## Run the end-to-end synthetic demo
	uv run scripts/demo.py

benchmark:  ## Run the CauST vs. HVG benchmark and regenerate the figures
	uv run scripts/benchmark.py --figures docs/figures

repro:  ## Run the headline experiment from its config into results/
	uv run caust run -c configs/experiment/synthetic_holdout.yaml

verify:  ## Re-run every recorded run and confirm the metrics still match
	@found=0; status=0; \
	for d in results/*/; do \
		[ -f "$$d/config.resolved.yaml" ] || continue; \
		found=1; uv run caust verify "$$d" || status=1; \
	done; \
	if [ "$$found" = "0" ]; then echo "no runs in results/ -- try 'make repro' first"; fi; \
	exit $$status

determinism:  ## Prove a run is byte-identical across two separate processes
	@rm -rf .determinism && mkdir -p .determinism
	uv run caust run -c configs/experiment/smoke.yaml -o .determinism/a >/dev/null
	uv run caust run -c configs/experiment/smoke.yaml -o .determinism/b >/dev/null
	@diff -r --exclude=manifest.json .determinism/a .determinism/b \
		&& echo "OK: artifacts are byte-identical across processes" \
		|| { echo "FAIL: artifacts differ across processes"; exit 1; }
	@rm -rf .determinism

lock:  ## Re-resolve uv.lock after changing dependencies in pyproject.toml
	uv lock

docs:  ## Build the documentation site
	uv run --group docs mkdocs build --strict

build:  ## Build sdist and wheel
	uv build

check: lint typecheck cov  ## Lint + type-check + tests with coverage

clean:  ## Remove caches and build artifacts
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .mypy_cache .determinism \
		.ruff_cache .coverage coverage.xml htmlcov site
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
