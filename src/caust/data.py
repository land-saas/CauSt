"""Synthetic multi-donor spatial data for demos and tests.

Generates several tissue "slices" that share a common set of *causal* genes
(which define the spatial domains identically in every donor) plus, in each
donor, a distinct set of *donor-specific noise* genes that are highly variable
but carry no cross-donor domain signal. This mirrors the paper's central claim:
variance-based HVG selection latches onto donor-specific noise, while CauST's
knockout-invariance criterion recovers the stable causal genes.
"""
from __future__ import annotations

import numpy as np
from anndata import AnnData


def make_synthetic_slice(
    grid: int = 20,
    n_domains: int = 4,
    n_causal: int = 8,
    n_noise: int = 40,
    n_donor_noise: int = 20,
    causal_strength: float = 3.0,
    donor_noise_strength: float = 4.0,
    seed: int = 0,
) -> AnnData:
    """Create one synthetic slice as an AnnData.

    Genes are laid out as ``[causal | shared-noise | donor-noise]``. Causal genes
    have domain-specific means (same across donors); donor-noise genes are highly
    variable within this donor only. Ground-truth domains are in
    ``adata.obs['domain']``.
    """
    rng = np.random.default_rng(seed)

    # Spot grid + domain labels (horizontal bands).
    xs, ys = np.meshgrid(np.arange(grid), np.arange(grid))
    coords = np.column_stack([xs.reshape(-1), ys.reshape(-1)]).astype(float)
    n_spots = coords.shape[0]
    domain = (coords[:, 1] // (grid / n_domains)).astype(int)
    domain = np.clip(domain, 0, n_domains - 1)

    n_genes = n_causal + n_noise + n_donor_noise
    X = np.zeros((n_spots, n_genes), dtype=float)

    # Causal genes: each has a fixed per-domain mean profile (donor-independent).
    causal_profiles = np.zeros((n_causal, n_domains))
    base_rng = np.random.default_rng(12345)  # shared across donors
    for g in range(n_causal):
        causal_profiles[g] = base_rng.normal(0, causal_strength, n_domains)
    for g in range(n_causal):
        X[:, g] = causal_profiles[g][domain] + rng.normal(0, 0.5, n_spots)

    # Shared background noise genes (low variance, no domain signal).
    off = n_causal
    X[:, off:off + n_noise] = rng.normal(0, 0.5, (n_spots, n_noise))

    # Donor-specific noise genes: high variance, but random (no domain signal).
    off = n_causal + n_noise
    X[:, off:off + n_donor_noise] = rng.normal(
        0, donor_noise_strength, (n_spots, n_donor_noise)
    )

    var_names = (
        [f"CAUSAL_{i}" for i in range(n_causal)]
        + [f"NOISE_{i}" for i in range(n_noise)]
        + [f"DONORNOISE_{i}" for i in range(n_donor_noise)]
    )
    adata = AnnData(X)
    adata.var_names = var_names
    adata.obs_names = [f"spot_{i}" for i in range(n_spots)]
    adata.obs["domain"] = domain.astype(str)
    adata.obsm["spatial"] = coords
    return adata


def make_synthetic_cohort(
    n_slices: int = 3,
    grid: int = 20,
    n_causal: int = 8,
    seed: int = 0,
    **kwargs,
) -> list[AnnData]:
    """A cohort of slices sharing causal genes but with per-donor noise genes.

    Every slice exposes the same ``var_names`` so genes align across donors, but
    the *values* of the donor-noise genes are independent per donor.
    """
    return [
        make_synthetic_slice(grid=grid, n_causal=n_causal, seed=seed + 100 * e, **kwargs)
        for e in range(n_slices)
    ]
