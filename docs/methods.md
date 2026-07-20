# Methods

This page gives the formal statement of the CauST method. The same material is
available as a typeset [technical report](report/caust_report.pdf).

## Setup and notation

Let a spatial transcriptomics slice from environment (donor) $e$ be described by
an expression matrix $X^{(e)} \in \mathbb{R}^{N_e \times G}$ over $N_e$ spots and
$G$ genes, together with a spatial adjacency graph $A^{(e)}$ built from spot
coordinates. A spatial-embedding backbone is a function

$$
f_\theta : (X, A) \mapsto Z \in \mathbb{R}^{N \times d},
$$

mapping expression and graph to a $d$-dimensional per-spot embedding. CauST
treats $f_\theta$ as a black box that is **frozen** after training on a slice.

## Step 1 — In-silico gene knockout

For a fitted, frozen model on slice $e$ and a gene $g$, define the knockout
expression matrix $\tilde{X}^{(g)}$ as $X^{(e)}$ with column $g$ set to zero:

$$
\tilde{X}^{(g)}_{\cdot, j} =
\begin{cases}
0 & j = g \\
X^{(e)}_{\cdot, j} & j \neq g.
\end{cases}
$$

The **knockout effect** of gene $g$ in slice $e$ is the mean per-spot shift of
the embedding:

$$
\delta(g, e) \;=\; \frac{1}{N_e} \sum_{i=1}^{N_e}
\bigl\| f_\theta(X^{(e)}, A^{(e)})_i - f_\theta(\tilde{X}^{(g)}, A^{(e)})_i \bigr\|_2 .
$$

Because $\theta$ and $A^{(e)}$ are fixed, the only source of variation is the
input perturbation, so $\delta(g,e)$ isolates the gene's causal contribution to
the embedding rather than training stochasticity. Empirically the distribution
of $\delta$ is highly skewed: a small fraction of genes drive large shifts while
most are near zero, confirming that the backbone is selective.

When $G$ is large, an optional gradient-based pre-filter (input-gradient
magnitude) can narrow the candidate set before exact knockouts are computed.

## Step 2 — Cross-slice invariance

A large $\delta(g,e)$ in a single slice is not enough: a gene may be influential
in one donor and irrelevant in others, indicating a donor-specific artifact. We
therefore score each gene by rewarding a large **and stable** effect across the
$E$ environments:

$$
s_{\mathrm{inv}}(g) \;=\;
\underbrace{\overline{\delta}(g)}_{\text{effect must be large}}
\;-\;
\lambda \cdot
\underbrace{\operatorname{std}_e\bigl[\delta(g,e)\bigr]}_{\text{effect must be stable}},
\qquad
\overline{\delta}(g) = \frac{1}{E}\sum_{e=1}^{E}\delta(g,e),
$$

with $\lambda \ge 0$ balancing magnitude against donor-to-donor variance. Setting
$\lambda = 0$ recovers a "high-$\delta$" baseline that ranks by mean effect only;
increasing $\lambda$ demands cross-donor stability. This criterion is inspired by
Invariant Causal Prediction and Invariant Risk Minimization, adapted to an
unsupervised spatial-embedding setting.

Genes are ranked by $s_{\mathrm{inv}}$; the top-$K$ form the causal gene set
$\mathcal{G}_{\text{causal}}$.

## Step 3 — Retraining with the causal gene set

$\mathcal{G}_{\text{causal}}$ replaces the standard HVG set as input to any
downstream spatial-domain model — no architectural changes required. Two
variants are supported:

- **Hard filtering:** keep the top-$K$ genes by $s_{\mathrm{inv}}$.
- **Soft reweighting:** $\hat{X} = X \operatorname{diag}(w)$, where $w$ are
  sigmoid weights derived from standardized invariance scores,
  $w_g = \sigma\!\left(z_g / \tau\right)$ with $z$ the standardized score and
  $\tau$ a temperature.

## Algorithm

```
Input:  slices {(X^(e), A^(e))}_{e=1..E}, penalty λ, size K
Output: causal gene set G_causal

1  common ← ∩_e var_names(X^(e))
2  for e = 1..E:
3      θ_e ← train backbone f on (X^(e)[:, common], A^(e))   # frozen after
4      for g in common:
5          δ(g, e) ← mean_i || f_{θ_e}(X^(e))_i − f_{θ_e}(X̃^(g))_i ||₂
6  for g in common:
7      s_inv(g) ← mean_e δ(g, e) − λ · std_e δ(g, e)
8  G_causal ← top-K genes by s_inv
9  return G_causal
```

## Evaluation

- **ARI** (Adjusted Rand Index) against manual annotations, and its cross-slice
  variant (train on slice A, evaluate zero-shot on slice B).
- **NMI** (Normalized Mutual Information) as a supplementary metric.
- **Causal gene overlap** between selected genes and known layer-marker genes.
- **Generalization gap:** the drop from within-slice to cross-donor ARI, which
  the invariance penalty is designed to minimize.
