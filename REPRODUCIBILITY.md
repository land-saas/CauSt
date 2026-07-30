# Reproducibility

Every number CauST reports should be re-creatable by someone who has only this
repository. This document is the contract that makes that true: what is
controlled, how to run an experiment, and how to check that a result still
reproduces.

## The workflow

```
configs/*.yaml ──▶ caust run ──▶ results/<name>-<digest>/ ──▶ caust verify
   the inputs        the run          the artifacts            the proof
```

1. **Describe** the experiment in a YAML file under `configs/`. Nothing that
   changes a result lives in a script.
2. **Run** it with `caust run -c <config>`. The run is seeded, its BLAS thread
   pool is pinned, and it writes a content-addressed directory under `results/`.
3. **Verify** with `caust verify <rundir>`, which re-runs the recorded config
   and fails if the metrics have moved.

```bash
make repro                                    # run the headline experiment
caust verify results/synthetic_holdout-*/     # prove it still reproduces
```

## What is controlled

| Source of variation | How it is fixed | Where it is recorded |
|---|---|---|
| Random number generation | `seed` in the config, applied via `caust.repro.set_global_seeds` to `random`, numpy, and torch | `manifest.json → provenance.seed` |
| BLAS thread count | `threads` in the config, applied at runtime with `threadpoolctl` | `manifest.json → provenance.blas` |
| Package versions | `requirements.lock`, or the Docker image | `manifest.json → provenance.packages` |
| Code version | git commit, plus a dirty-tree flag | `manifest.json → provenance.git` |
| Platform / interpreter | recorded, not fixed | `manifest.json → provenance.platform`, `.python` |
| Experiment settings | the config file itself | `config.resolved.yaml`, hashed into the run id |

### Why thread count is on that list

BLAS chooses its reduction order based on how many threads it is using, and
floating-point addition is not associative. The same input therefore produces
bitwise-different output on machines with different core counts. On this project
that only moves the low-order bits of the invariance scores — it does not change
which genes get selected — but it is enough to break byte-level comparison of
results, which is what `caust verify` relies on.

Two things are worth knowing:

- Setting `OMP_NUM_THREADS` alone is **not** enough on macOS, where the
  Accelerate backend reads `VECLIB_MAXIMUM_THREADS`. `caust.repro.THREAD_ENV_VARS`
  lists all five variables that matter.
- Those environment variables are read by the BLAS backend when it loads, so
  setting them from Python after numpy is imported does nothing. That is why the
  runner uses `threadpoolctl`, which changes the limits at runtime.

## Run artifacts

```
results/synthetic_holdout-2b1f0c9d4e7a/
├── config.resolved.yaml   the fully-resolved config, inheritance applied
├── manifest.json          provenance + SHA-256 of every other artifact
├── metrics.json           the numbers
└── genes.csv              per-gene scores and which arm selected them
```

The directory name is `<experiment name>-<first 12 hex of the config digest>`.
Because it is content-addressed, re-running a config overwrites its own
directory, and changing *any* setting produces a new one — so a parameter sweep
accumulates instead of silently overwriting.

## Reproducing a published result

```bash
git checkout <commit from manifest.json → provenance.git.commit>
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.lock && pip install --no-deps -e .
caust verify results/<run_id>/
```

Or hermetically, which also pins the OS and interpreter:

```bash
docker build -t caust .
docker run --rm -v "$PWD/results:/work/results" caust \
  caust run -c configs/experiment/synthetic_holdout.yaml
```

If `provenance.git.dirty` is `true`, the run came from uncommitted code and the
commit alone will not reproduce it. Treat those runs as scratch.

## Adding an experiment

Inherit from `configs/base.yaml` and override only what changes:

```yaml
defaults: ../base.yaml
name: my_experiment
selection:
  lam: 1.0
```

Override from the command line for sweeps — each value gets its own run
directory:

```bash
for lam in 0 0.5 1 2 4; do
  caust run -c configs/experiment/lambda_sweep.yaml --set selection.lam=$lam
done
```

Overrides may only set keys that already exist, so a typo fails loudly instead
of quietly adding a setting nothing reads.

## What CI enforces

- `make check` — lint, type-check, and the test suite with coverage.
- **Determinism**: the smoke experiment runs twice in separate processes and the
  artifacts must be byte-identical.
- **Distribution**: the sdist is built, installed, and its tests run from the
  unpacked tarball, so the shipped package is verified rather than assumed.

## Known limits

- The lockfile is resolved for one platform. On a different OS or Python version
  pip may legitimately pick different wheels; use the Docker image when that
  matters.
- `PYTHONHASHSEED` cannot be set after the interpreter starts, so it is recorded
  rather than enforced. Nothing in CauST currently depends on hash ordering; the
  Docker image sets it anyway.
- Only `data.source: synthetic` and `model.backend: simple` are wired up. Real
  backbones (STAGATE/GraphST) and real datasets will need their own provenance
  story — in particular, checksums for input data files.
