# Hermetic environment for reproducing CauST results.
#
#   docker build -t caust .
#   docker run --rm -v "$PWD/results:/work/results" caust \
#     caust run -c configs/experiment/synthetic_holdout.yaml
#
# Pinned to a digest-free but explicit base tag; bump deliberately, since the
# base image is part of what makes a result reproducible.
FROM python:3.11-slim-bookworm

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
    PIP_NO_CACHE_DIR=1

WORKDIR /work

# Dependencies first so edits to the source do not invalidate this layer.
COPY requirements.lock ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.lock

COPY pyproject.toml README.md LICENSE MANIFEST.in ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY tests/ ./tests/
COPY scripts/ ./scripts/

RUN python -m pip install --no-deps .

CMD ["caust", "run", "-c", "configs/experiment/synthetic_holdout.yaml"]
