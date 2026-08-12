# Hermetic environment for reproducing CauST results.
#
#   docker build -t caust .
#   docker run --rm -v "$PWD/results:/work/results" caust \
#     caust run -c configs/experiment/synthetic_holdout.yaml
#
# The base tag pins the OS, interpreter, and uv version; uv.lock pins every
# package. Both are part of what makes a result reproducible — bump deliberately.
FROM ghcr.io/astral-sh/uv:0.12.0-python3.11-bookworm-slim

# Pin the BLAS thread pool at the image level as well as in code. The library
# calls threadpoolctl at runtime, but setting these keeps any subprocess or
# ad-hoc python -c invocation on the same footing.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy

WORKDIR /work

# Locked runtime dependencies first, so edits to the source do not invalidate
# this layer. --no-install-project defers the package itself to the next layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY README.md LICENSE MANIFEST.in ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY tests/ ./tests/
COPY scripts/ ./scripts/

RUN uv sync --locked --no-dev --no-editable

ENV PATH="/work/.venv/bin:$PATH"

CMD ["caust", "run", "-c", "configs/experiment/synthetic_holdout.yaml"]
