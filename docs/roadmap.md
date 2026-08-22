# Roadmap: proposal objectives → status

The OSRE'26 proposal ("CauST: Causal Gene Intervention for Robust Spatial
Domain Identification", Jiawei Li, March 2026) commits to a specific method,
evaluation protocol, datasets, and deliverables. This page tracks each against
the code, so that what the package *can* reproduce is never in doubt.

## Method (Section 3 of the proposal)

| Requirement | Status | Where |
|---|---|---|
| Step 1 — in-silico knockout on a frozen spatial model, δ = mean per-spot L2 embedding shift (Eq. 1–2) | ✅ | `caust.intervention.knockout_scores` |
| Step 2 — invariance score `mean − λ·std` across slices (Eq. 3) | ✅ | `caust.invariance.invariance_scores` |
| Step 3 — hard top-K filtering and soft sigmoid reweighting | ✅ | `caust.invariance.select_causal_genes`, `soft_weights` |
| STAGATE backbone (G→512→30 graph attention autoencoder, 1000 epochs) | ✅ pure PyTorch, validated against the reference implementation (loss 0.150 vs 0.150, ARI 0.505 vs 0.495 on 151673) | `caust.models.stagate.STAGATEModel` |
| Gradient-based pre-filter / integrated gradients as a knockout proxy | ✅ with fidelity metric vs exact knockout | `STAGATEModel.input_attribution`, `attribution_fidelity` |
| mclust (model EEE) clustering | ✅ Python equivalent (`method="eee"`, tied-covariance GMM); exact R bridge via rpy2 when available | `caust.cluster.cluster_embedding` |
| GraphST / SpaGCN backbones | ⏳ planned (interface ready: `caust.models.base.BaseSpatialModel`) | — |

## Evaluation protocol (Section 4)

| Requirement | Status | Where |
|---|---|---|
| Gene pool = cross-donor intersection of per-slice top HVGs (~2,000 genes) | ✅ 2,002 genes on DLPFC | `caust.transfer.build_pool` |
| Knockout scoring on one slice per donor | ✅ | `caust.transfer.score_pool` |
| Strategies at equal K: HVG-K, High-δ-K (λ=0), CauST-K | ✅ | `caust.transfer.gene_set` |
| Every slice as source, zero-shot transfer to all others, multiple seeds | ✅ resumable grid | `caust.transfer.run_transfer`, `caust transfer` |
| Within-slice / within-donor / cross-donor ARI, generalization gap, win counts | ✅ | `caust.transfer.summarize` |
| HVG Jaccard stability (within vs across donors) | ✅ | `caust.transfer.hvg_jaccard` |
| Figures: ARI vs K (3 settings), gap, 12×12 transfer matrices, top-20 genes with markers | ✅ | `caust.figures.transfer_figures` |
| Per-slice heatmap, UMAP visualisation | ⏳ planned | — |
| Reported numbers for the full 12-slice × 7-K × 10-seed grid | 🟡 running (`make transfer`); reduced grid first (`make transfer-quick`) | `results/` |

## Datasets (Table 1)

| Dataset | Status |
|---|---|
| DLPFC, 12 Visium sections, manual layers | ✅ checksum-pinned loader, all 12 samples |
| Mouse brain (Visium, 2 sections) | ⏳ planned |
| MERFISH hypothalamus | ⏳ planned |
| STARmap visual cortex | ⏳ planned |

## Objectives & deliverables (Sections 5–6)

| Item | Status |
|---|---|
| Production-quality codebase: ≥90 % coverage, mypy, CI, PyPI release | ✅ code and release workflow; first PyPI upload pending the maintainer's PyPI setup (see CONTRIBUTING → Releasing) |
| Reproducibility: config-driven runs, provenance manifests, `caust verify`, byte-identical determinism in CI | ✅ (GPU runs are seeded; MPS is reproducible to float tolerance, recorded in provenance) |
| Gradient-based attribution | ✅ |
| Multi-backbone integration (GraphST, SpaGCN) | ⏳ |
| Full benchmark suite on four datasets | 🟡 DLPFC in progress; others planned |
| Visualisation toolkit | 🟡 benchmark figures done; UMAP/gene-overlay maps planned |
| Scanpy-style API, tutorial notebooks, ReadTheDocs | ⏳ |
| REST server | deprioritised (not needed for the research claims) |
| Technical report / preprint | 🟡 `docs/report/` to be updated with the real-data results |
