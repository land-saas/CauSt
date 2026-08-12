# Five-minute demo

A guided tour of CauST that runs end-to-end from a fresh clone. Total compute
time is under two minutes on a laptop; the only prerequisite is
[uv](https://docs.astral.sh/uv/getting-started/installation/).

Every command below is copy-pasteable from the repository root.

## 1. Setup — one command

```bash
git clone https://github.com/land-saas/CauSt.git && cd CauSt
uv sync
```

`uv sync` reads the committed `uv.lock`, fetches the interpreter pinned in
`.python-version` if it is missing, and builds `.venv` with the package and all
dev tooling — the same locked environment CI and the Docker image use. There is
nothing to activate; `uv run` handles that from here on.

## 2. The idea, live

```bash
uv run scripts/demo.py
```

Three synthetic donors share eight `CAUSAL_*` genes that define the spatial
domains in every donor. Each donor also has private `DONORNOISE_*` genes with
**higher variance** but no cross-donor signal — exactly the trap that
variance-based gene selection falls into.

CauST silences each gene in-silico on a frozen spatial model, measures the
embedding shift per donor, and keeps genes whose effect is large **and stable
across donors**. Watch the ranking:

```text
Top-12 genes by invariance score:
  CAUSAL_7         0.9758  <-- causal
  CAUSAL_1         0.5160  <-- causal
  ...
  NOISE_13         0.1539

Causal genes recovered in top-8: 8/8
ARI on causal gene set (donor 0): 1.000
```

All eight causal genes rank above every noise gene, and clustering on just
those eight recovers the ground-truth domains perfectly.

## 3. The headline benchmark — CauST vs. HVG on a held-out donor

```bash
uv run scripts/benchmark.py --figures docs/figures
```

Genes are selected on donors 1–2 only; donor 3 is never seen during selection.
That held-out donor is the robustness claim:

```text
Causal genes recovered in the top-8:
  HVG    1/8
  CauST  8/8

Held-out donor 3 (never used for gene selection, 5 restarts):
  ARI  all genes  (68 genes)  1.000 +/- 0.000
  ARI  HVG        ( 8 genes)  0.537 +/- 0.133
  ARI  CauST      ( 8 genes)  1.000 +/- 0.000
```

CauST matches the all-genes ceiling with **8× fewer genes**, while the
variance baseline at the same budget loses ~0.46 ARI — and is unstable across
clustering seeds (±0.133) where CauST is exactly stable (±0.000).

The command also regenerates the two figures:

![CauST vs HVG benchmark](figures/benchmark.png)

Panels A and B are the crux: the invariance score isolates the causal genes,
while variance ranks the donor-specific noise genes highest.

![Spatial domains on the held-out donor](figures/domains.png)

## 4. Every number is reproducible

Each run writes a content-addressed directory under `results/` with the
resolved config, the metrics, and a provenance manifest (git commit, package
versions, platform, seed, BLAS thread counts, artifact checksums).

```bash
uv run caust verify results/synthetic_holdout-*/
```

```text
REPRODUCED: results/synthetic_holdout-0759a2e2cbad matches a fresh run of its recorded config
```

And determinism is not an accident — two separate processes produce
byte-identical artifacts (CI enforces this on every push):

```bash
make determinism
```

```text
OK: artifacts are byte-identical across processes
```

See [REPRODUCIBILITY.md](https://github.com/land-saas/CauSt/blob/main/REPRODUCIBILITY.md)
for what is controlled and how.

## 5. Quality gates

```bash
make check
```

Runs ruff, mypy, and the 85-test suite with a 90% coverage gate (currently
~97%), through the same locked environment. CI additionally proves the built
sdist/wheel install and test cleanly on Python 3.9–3.12.

## What you just saw

- **A causal claim, tested honestly** — gene selection on training donors,
  evaluation on a held-out donor, against the standard HVG baseline.
- **One lockfile everywhere** — `uv.lock` backs local dev, CI, and Docker;
  `uv sync` rebuilds the exact environment on any machine.
- **Provenance by default** — every result carries the config, seed, and
  environment that produced it, and `caust verify` re-earns it on demand.
