# Related work and positioning

A literature sweep (August 2026) of the areas CauST touches, with the
consequences for how the method is described and evaluated. Numbers quoted
are as reported by the cited sources; independent benchmarks that re-run
methods under a standardized pipeline report DLPFC ARIs roughly 0.1 lower than
the method papers, and CauST's own numbers fall in the standardized regime.

## What CauST's knockout score actually measures

Silencing a gene in a *frozen* model and measuring the embedding shift is
**model reliance** (Fisher, Rudin & Dominici, JMLR 2019; Breiman's permutation
importance), not a biological causal effect. Ahlmann-Eltze, Huber & Anders
(Nat Methods 2025) show that in-silico perturbation outputs of deep
single-cell models are not validated predictions of biology. CauST therefore
describes the score as *model reliance under intervention on the input*, uses
"causal" only for the invariance motivation (ICP, IRM), and validates selected
genes against independent layer markers (Maynard et al. 2021) rather than
resting the claim on ARI alone.

The combination `mean − λ·std` over environments is the V-REx criterion
(Krueger et al., ICML 2021: penalize the variance of risk across environments)
applied to per-donor reliance, with stability selection (Meinshausen &
Bühlmann 2010) as the feature-selection ancestor. Cite both.

## Closest prior art (must be cited and differentiated)

| Work | What it does | How CauST differs |
|---|---|---|
| **STAMarker** (Zhang et al., NAR 2023) | Scores genes by back-propagated saliency through an ensemble of STAGATE encoders plus a classifier head, to find domain-specific marker genes | Ablation rather than gradient; frozen *unsupervised* model, no classifier; a cross-donor invariance penalty; the ranking is used as the gene set for retraining and zero-shot transfer, not for marker discovery |
| **Geneformer in-silico deletion** (Theodoris et al., Nature 2023) | Removes a gene token from a frozen pretrained transformer and ranks genes by cosine shift of the cell embedding | Same operation on a spatial GNN, where a knockout propagates through neighbours; evaluated as feature selection for a downstream task across environments |
| **geneBasis** (Missarova et al., Genome Biology 2021) | Selects gene panels by how much the kNN manifold changes when a gene is removed | Ablation on structure, but no trained spatial model, no environment invariance |
| **Donor-aware HVG** (`scanpy.pp.highly_variable_genes(batch_key=...)`, Seurat `SelectIntegrationFeatures`) | Ranks genes by the number of batches in which they are highly variable | The simplest cross-donor-stability selector — the baseline the invariance term must beat (now `hvg_donor` in the transfer benchmark) |
| **PERSIST** (Covert et al., Nat Commun 2023), **Spapros** (Kuemmerle et al., Nat Methods 2024), **geneCover** (Genome Research 2025) | Gene-panel selection for targeted spatial assays | Supervised or cell-type-driven objectives; no frozen-model intervention, no cross-donor test |

Taxonomy: Yan, Hua & Li (Nat Commun 2025) categorize 34 SVG detectors into
overall SVGs, cell-type-specific SVGs, and *spatial-domain-marker* SVGs. The
CauST score is a domain-marker-type criterion, not an SVG test.

## Gene selection for spatial domain identification

Evidence that the choice of genes matters is real but **contested**, and the
expected effect is modest:

- Li et al. (Pinello lab), Genome Biology 2025 (14 methods, 96 datasets):
  most SVG-based selections improve clustering over HVGs; Moran's I best
  overall, then SpatialDE2 and nnSVG.
- Chen et al. (Xi lab), Bioinformatics 2025: SPARK-X top; Moran's I
  competitive; most SVG sets beat HVGs for spatial clustering.
- Chen et al. (Fan lab), iMeta 2025 (~600 datasets): 3,000 SPARK-X SVGs
  instead of HVGs improve most methods; under a standardized pipeline DLPFC
  ARIs are STAGATE 0.498, SEDR 0.485, GraphST 0.483.
- Kang et al., NAR 2025 (19 domain methods, 30 datasets): for GNN methods,
  all-genes vs top-3,000 HVGs differs only slightly.
- SVGbench (Chen, Kim & Yang, Genome Biology 2024): SVG callers disagree
  substantially with one another.

Consequence: the benchmark reports paired statistics across slices (Wilcoxon
signed-rank on per-pair ARI, `summary.json → paired_tests`) and frames the
gain as a measured effect size, not a headline.

Selectors that should appear as baselines at equal gene budget: `seurat_v3`
HVG (the STAGATE/GraphST default; `hvg`), donor-aware HVG (`hvg_donor`),
Moran's I (`moran`; Squidpy), random sets (`random`), and all genes. SPARK-X,
nnSVG, SpatialDE2 and SINFONIA (Jiang et al., Cells 2023) are the remaining
external selectors to add.

## Cross-slice and cross-donor domain identification

Single-slice backbones: STAGATE (Dong & Zhang, Nat Commun 2022), GraphST
(Long et al., Nat Commun 2023; median ARI 0.60 over 12 DLPFC slices), SpaGCN,
BayesSpace, DeepST. Multi-sample and batch-robust: STAligner (Zhou, Dong &
Zhang 2023), SLAT, PRECAST, BASS (Li & Zhou 2022), CellCharter (Varrone et
al. 2024), BANKSY (Singhal et al. 2024), STitch3D, SPIRAL, MENDER,
NicheCompass, STAIG, MaskGraphene, SpaBatch (Adv Sci 2025; one labelled slice
→ others), SpaCross, STG3Net. Transfer and zero-shot: STELLAR, Spatial-ID,
SPACEL, stGuide, STransfer (Bioinformatics 2026), and spatial foundation
models (Novae, Nat Methods 2025; Nicheformer; scGPT-spatial; CellPLM).

Positioning: CauST changes the **input gene set**, not the model. Its
"zero-shot cross-donor transfer" is transfer of a gene set plus a frozen
backbone, distinct from label transfer and from foundation-model zero-shot
inference; the within-slice-trained ARI from the same pipeline is the
ceiling. Showing the selected genes also help at least one other backbone
(GraphST, BANKSY) is what makes the result not STAGATE-specific.

Benchmarks of the field: SDMBench (Yuan et al., Nat Methods 2024), Hu et al.
(Genome Biology 2024: BASS, GraphST, ADEPT, BANKSY, STAGATE top-tier on
DLPFC), the 2025 multi-slice integration benchmark (Genome Biology 26:318),
and "Towards a better understanding of batch effects in spatial
transcriptomics" (bioRxiv 2025), whose vocabulary CauST adopts: the shift it
targets is the **inter-sample (donor)** batch effect.

## Evaluation protocol implications

- Leave-one-**donor**-out, never random slice splits: consecutive sections
  from one donor are near-replicates (the within-donor vs cross-donor split in
  the transfer benchmark).
- Report per-slice ARI and NMI, fixed k, fixed clustering, several seeds,
  paired tests.
- Three donors give a noisy std for the invariance score; spatialDLPFC
  (Huuki-Myers et al., Science 2024: 30 samples, 10 donors) is the natural
  validation set and the place to quantify how λ interacts with the number of
  environments.
- Ablations of the score itself: λ = 0 (`highdelta`), gradient attribution vs
  ablation (`attribution_fidelity`), knockout by zeroing vs mean-imputation,
  effect measured on the embedding vs on cluster assignments, with and without
  the spatial graph.

## Datasets beyond DLPFC

- **MERFISH hypothalamus** (Moffitt et al., Science 2018): 5 consecutive
  Bregma sections of one animal with 8 BASS-annotated regions (Li & Zhou 2022;
  SDMBench distribution, `obs["Region"]`). Expression is volume-normalized, not
  integer counts; 155 genes, so the pool is the whole panel.
- **STARmap mouse PFC** (Wang et al., Science 2018): 3 sections, 166 genes,
  4 BASS-annotated layers — a second imaging dataset with cross-section
  invariance and ground truth.
- **STARmap visual cortex**: 1 section, 1,020 genes, 7 labelled domains —
  within-slice evaluation only.
- **Mouse brain Visium** (10x): official counts; manual spot labels exist only
  for Sagittal-Anterior Section 1 (ConGI / BenchmarkST annotation).
