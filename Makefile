# CauST developer Makefile
.DEFAULT_GOAL := help
PY ?= python

.PHONY: help install install-dev lint format typecheck test cov demo benchmark repro verify determinism lock docs build clean check

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Install the package (runtime deps only)
	$(PY) -m pip install -e .

install-dev:  ## Install with dev + docs extras and pre-commit hooks
	$(PY) -m pip install -e ".[dev,docs]"
	pre-commit install

lint:  ## Run ruff linter
	ruff check src tests scripts

format:  ## Auto-format with black and ruff --fix
	ruff check --fix src tests scripts
	black src tests scripts

typecheck:  ## Run mypy static type checking
	mypy

test:  ## Run the test suite
	pytest

cov:  ## Run tests with coverage (fails under 90%)
	pytest --cov=caust --cov-report=term-missing --cov-fail-under=90

demo:  ## Run the end-to-end synthetic demo
	$(PY) scripts/demo.py

benchmark:  ## Run the CauST vs. HVG benchmark and regenerate the figures
	$(PY) scripts/benchmark.py --figures docs/figures

repro:  ## Run the headline experiment from its config into results/
	caust run -c configs/experiment/synthetic_holdout.yaml

verify:  ## Re-run every recorded run and confirm the metrics still match
	@found=0; status=0; \
	for d in results/*/; do \
		[ -f "$$d/config.resolved.yaml" ] || continue; \
		found=1; caust verify "$$d" || status=1; \
	done; \
	if [ "$$found" = "0" ]; then echo "no runs in results/ -- try 'make repro' first"; fi; \
	exit $$status

determinism:  ## Prove a run is byte-identical across two separate processes
	@rm -rf .determinism && mkdir -p .determinism
	caust run -c configs/experiment/smoke.yaml -o .determinism/a >/dev/null
	caust run -c configs/experiment/smoke.yaml -o .determinism/b >/dev/null
	@diff -r --exclude=manifest.json .determinism/a .determinism/b \
		&& echo "OK: artifacts are byte-identical across processes" \
		|| { echo "FAIL: artifacts differ across processes"; exit 1; }
	@rm -rf .determinism

lock:  ## Regenerate requirements.lock from a clean runtime-only environment
	@rm -rf .lockenv
	$(PY) -m venv .lockenv
	.lockenv/bin/pip install --quiet --upgrade pip
	.lockenv/bin/pip install --quiet .
	@{ \
		echo "# Fully pinned runtime environment for reproducing CauST results."; \
		echo "#"; \
		echo "# Regenerate with:  make lock"; \
		echo "# Install with:     pip install -r requirements.lock"; \
		echo "#"; \
		echo "# Captured on: Python $$(.lockenv/bin/python -c 'import platform;print(platform.python_version())') / $$(uname -s) $$(uname -m)"; \
		echo "# Resolved versions are platform-specific; on a different OS or Python"; \
		echo "# version pip may legitimately resolve different wheels."; \
		echo ""; \
		.lockenv/bin/pip freeze --exclude-editable | grep -v '^caust' | grep -v '^-e '; \
	} > requirements.lock
	@rm -rf .lockenv
	@echo "wrote requirements.lock"

docs:  ## Build the documentation site
	mkdocs build --strict

build:  ## Build sdist and wheel
	$(PY) -m build

check: lint typecheck cov  ## Lint + type-check + tests with coverage

clean:  ## Remove caches and build artifacts
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .mypy_cache .determinism .lockenv \
		.ruff_cache .coverage coverage.xml htmlcov site
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
