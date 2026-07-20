"""Shared pytest fixtures for the CauST test suite."""

from __future__ import annotations

import pytest
from anndata import AnnData

from caust.data import make_synthetic_cohort, make_synthetic_slice


@pytest.fixture
def slice_small() -> AnnData:
    """A single small synthetic slice (deterministic)."""
    return make_synthetic_slice(grid=10, seed=0)


@pytest.fixture
def cohort_small() -> list[AnnData]:
    """A 3-donor synthetic cohort with shared causal genes."""
    return make_synthetic_cohort(n_slices=3, grid=12, n_causal=8, seed=0)
