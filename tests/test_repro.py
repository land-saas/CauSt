import json

import numpy as np
import pytest

from caust import repro


def test_set_global_seeds_makes_rng_reproducible():
    repro.set_global_seeds(123)
    a = np.random.rand(5)
    repro.set_global_seeds(123)
    np.testing.assert_array_equal(a, np.random.rand(5))


def test_deterministic_yields_seed_and_pins_threads():
    with repro.deterministic(seed=7, threads=1) as seed:
        assert seed == 7
        info = repro._blas_info()
    if info:  # threadpoolctl is present via scikit-learn
        assert all(entry["num_threads"] == 1 for entry in info)


def test_deterministic_block_is_reproducible():
    def draw():
        with repro.deterministic(seed=5, threads=1):
            return np.random.rand(4).tobytes()

    assert draw() == draw()


def test_digest_is_order_independent_but_content_sensitive():
    assert repro.digest_obj({"a": 1, "b": 2}) == repro.digest_obj({"b": 2, "a": 1})
    assert repro.digest_obj({"a": 1}) != repro.digest_obj({"a": 2})


def test_sha256_file_matches_bytes(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"caust")
    assert repro.sha256_file(p) == repro.sha256_bytes(b"caust")


def test_collect_provenance_records_what_is_needed_to_reproduce():
    prov = repro.collect_provenance(seed=3, threads=2)
    assert prov["seed"] == 3
    assert prov["threads"] == 2
    for key in ("timestamp_utc", "git", "python", "platform", "packages", "thread_env"):
        assert key in prov
    assert prov["packages"]["caust"] is not None
    # Must survive JSON round-tripping, since it is written to manifest.json.
    assert json.loads(json.dumps(prov, default=str))["seed"] == 3


def test_thread_env_vars_cover_the_macos_accelerate_variable():
    # Setting only OMP_NUM_THREADS silently leaves macOS threading unpinned.
    assert "VECLIB_MAXIMUM_THREADS" in repro.THREAD_ENV_VARS
    assert "OMP_NUM_THREADS" in repro.THREAD_ENV_VARS


def test_git_info_reports_availability():
    info = repro.git_info()
    assert "available" in info
    if info["available"]:
        assert isinstance(info["dirty"], bool)
        assert len(info["commit"]) == 40


@pytest.mark.parametrize("threads", [1, 2])
def test_deterministic_accepts_thread_counts(threads):
    with repro.deterministic(seed=0, threads=threads):
        pass
