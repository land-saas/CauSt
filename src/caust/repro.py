"""Reproducibility primitives: seeding, thread pinning, and provenance capture.

A CauST run is reproducible when three things are fixed: the **seed**, the
**BLAS thread count**, and the **environment**. This module owns the first two
and records the third.

Why thread pinning matters here: BLAS reduction order depends on how many
threads the backend uses, so the same input produces bitwise-different floats on
machines with different core counts. On this project that moves low-order bits
of the invariance scores -- not enough to change which genes are selected, but
enough to break byte-level comparison of results. :func:`deterministic` pins the
thread pool for the duration of a run so output is bitwise stable.

Note that the environment variables (``OMP_NUM_THREADS`` and friends) are read
by the BLAS backend at import time, so setting them from Python after numpy is
loaded does nothing. ``threadpoolctl`` changes the limits at runtime instead,
which is why it is used here.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

#: Environment variables that BLAS backends consult for their thread count.
#: ``VECLIB_MAXIMUM_THREADS`` is the macOS Accelerate one and is easy to miss --
#: setting only ``OMP_NUM_THREADS`` on macOS silently leaves threading unpinned.
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

#: Packages whose versions are recorded in every run manifest.
TRACKED_PACKAGES = (
    "caust",
    "numpy",
    "scipy",
    "scikit-learn",
    "anndata",
    "pandas",
    "matplotlib",
)

DEFAULT_SEED = 0
DEFAULT_THREADS = 1


def set_global_seeds(seed: int = DEFAULT_SEED) -> int:
    """Seed every global RNG CauST can reach and return the seed.

    Seeds :mod:`random`, numpy's legacy global RNG, and torch if it is
    installed. CauST's own code paths take explicit seeds instead of relying on
    global state; this is a backstop for third-party code that does not.

    ``PYTHONHASHSEED`` cannot be changed after interpreter start, so it is
    recorded in the manifest rather than set here.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:  # pragma: no cover - torch is an optional extra
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # pragma: no cover - absent or too old to configure
        pass
    return seed


@contextlib.contextmanager
def deterministic(
    seed: int = DEFAULT_SEED, threads: int = DEFAULT_THREADS
) -> Iterator[int]:
    """Run a block with seeds set and BLAS threading pinned.

    Pinning the thread count is what makes results bitwise reproducible across
    machines; without it, identical inputs differ in the low-order bits.
    """
    set_global_seeds(seed)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - threadpoolctl ships with sklearn
        yield seed
        return
    with threadpool_limits(limits=threads):
        yield seed


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_obj(obj: Any) -> str:
    """Stable hex SHA-256 of any JSON-serializable object.

    Keys are sorted so the digest depends on content, not dict ordering.
    """
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_bytes(payload.encode("utf-8"))


def _run_git(*args: str, repo: Path | None = None) -> str | None:
    root = str(repo) if repo is not None else str(Path(__file__).resolve().parents[2])
    try:
        out = subprocess.run(
            ["git", "-C", root, *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def git_info(repo: Path | None = None) -> dict[str, Any]:
    """Commit, branch, and dirty state of the working tree.

    ``dirty=True`` means the run came from uncommitted code and is therefore not
    reproducible from the recorded commit alone.
    """
    commit = _run_git("rev-parse", "HEAD", repo=repo)
    if commit is None:
        return {"available": False}
    status = _run_git("status", "--porcelain", repo=repo)
    return {
        "available": True,
        "commit": commit,
        "branch": _run_git("rev-parse", "--abbrev-ref", "HEAD", repo=repo),
        "dirty": bool(status),
        "describe": _run_git("describe", "--tags", "--always", "--dirty", repo=repo),
    }


def package_versions() -> dict[str, str | None]:
    """Installed versions of the packages that can change numerical output."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def collect_provenance(
    seed: int = DEFAULT_SEED,
    threads: int = DEFAULT_THREADS,
    repo: Path | None = None,
) -> dict[str, Any]:
    """Everything needed to explain -- or re-create -- a run.

    Deliberately excludes wall-clock duration and output paths so that two
    reproductions of the same config on the same environment produce comparable
    provenance blocks.
    """
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "threads": threads,
        "git": git_info(repo),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "hash_seed": os.environ.get("PYTHONHASHSEED"),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": package_versions(),
        "thread_env": {k: os.environ.get(k) for k in THREAD_ENV_VARS},
        "blas": _blas_info(),
    }


def _blas_info() -> Any:
    """Which BLAS backends are loaded, and at what thread limits."""
    try:
        from threadpoolctl import threadpool_info
    except ImportError:  # pragma: no cover - threadpoolctl ships with sklearn
        return None
    try:
        return [
            {
                k: info.get(k)
                for k in ("user_api", "internal_api", "version", "num_threads")
            }
            for info in threadpool_info()
        ]
    except Exception:  # pragma: no cover - backend introspection is best-effort
        return None
