import json
from pathlib import Path

import numpy as np
import pytest

from caust.config import ExperimentConfig
from caust.experiment import (
    ARTIFACT_CONFIG,
    ARTIFACT_GENES,
    ARTIFACT_MANIFEST,
    ARTIFACT_METRICS,
    ExperimentError,
    match_labels,
    run_experiment,
    verify_run,
)

TINY = {
    "name": "tiny",
    "seed": 0,
    "threads": 1,
    "data": {"source": "synthetic", "n_slices": 2, "grid": 6, "n_causal": 3},
    "model": {"backend": "simple", "n_components": 6},
    "selection": {"lam": 2.0, "n_top_genes": 3},
    "evaluation": {"holdout": True, "n_restarts": 2},
}


def cfg(**over):
    payload = {**TINY, **over}
    return ExperimentConfig.from_dict(payload)


def test_run_writes_the_expected_artifacts(tmp_path):
    out = tmp_path / "run"
    run_experiment(cfg(), out)
    for name in (ARTIFACT_CONFIG, ARTIFACT_METRICS, ARTIFACT_GENES, ARTIFACT_MANIFEST):
        assert (out / name).is_file(), f"missing {name}"


def test_manifest_records_provenance_and_checksums(tmp_path):
    out = tmp_path / "run"
    run_experiment(cfg(), out)
    manifest = json.loads((out / ARTIFACT_MANIFEST).read_text())
    assert manifest["provenance"]["seed"] == 0
    assert manifest["provenance"]["threads"] == 1
    # The manifest hashes the other artifacts but never itself.
    assert ARTIFACT_METRICS in manifest["artifacts"]
    assert ARTIFACT_MANIFEST not in manifest["artifacts"]


def test_manifest_blas_threads_reflect_the_pinned_run(tmp_path):
    # Regression: provenance used to be captured after the pinned block exited,
    # so it recorded the machine's ambient thread count instead of the run's.
    out = tmp_path / "run"
    run_experiment(cfg(), out)
    blas = json.loads((out / ARTIFACT_MANIFEST).read_text())["provenance"]["blas"]
    if blas:
        assert all(entry["num_threads"] == 1 for entry in blas)


def test_two_runs_of_one_config_are_byte_identical(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    run_experiment(cfg(), a)
    run_experiment(cfg(), b)
    for name in (ARTIFACT_METRICS, ARTIFACT_GENES, ARTIFACT_CONFIG):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_verify_run_passes_on_an_untouched_run(tmp_path):
    out = tmp_path / "run"
    run_experiment(cfg(), out)
    ok, detail = verify_run(out)
    assert ok, detail


def test_verify_run_detects_tampering(tmp_path):
    out = tmp_path / "run"
    run_experiment(cfg(), out)
    metrics = json.loads((out / ARTIFACT_METRICS).read_text())
    metrics["causal_recovery"]["caust"] = 999
    (out / ARTIFACT_METRICS).write_text(json.dumps(metrics))
    ok, detail = verify_run(out)
    assert not ok
    assert "causal_recovery" in (detail or "")


def test_run_id_changes_with_any_setting(tmp_path):
    base = cfg()
    other = cfg(selection={"lam": 1.0, "n_top_genes": 3})
    assert base.run_id != other.run_id


def test_holdout_requires_two_slices():
    bad = cfg(data={"source": "synthetic", "n_slices": 1, "grid": 6, "n_causal": 3})
    with pytest.raises(ExperimentError, match="at least 2 slices"):
        run_experiment(bad, Path("unused"), write=False)


def test_unsupported_data_source_is_rejected():
    bad = cfg(data={"source": "visium", "n_slices": 2})
    with pytest.raises(ExperimentError, match="unsupported data.source"):
        run_experiment(bad, Path("unused"), write=False)


def test_unsupported_backend_is_rejected():
    bad = cfg(model={"backend": "spagcn"})
    with pytest.raises(ExperimentError, match="unsupported model.backend"):
        run_experiment(bad, Path("unused"), write=False)


def test_metrics_report_restart_spread():
    metrics = run_experiment(cfg(), Path("unused"), write=False)
    caust = metrics["held_out"]["caust"]
    assert caust["n_restarts"] == 2
    assert len(caust["ari_restarts"]) == 2
    assert caust["ari_std"] >= 0.0
    # Plotting labels must not leak into the metrics payload.
    assert "_labels" not in caust


def test_figures_are_written_when_requested(tmp_path):
    out = tmp_path / "run"
    run_experiment(cfg(), out, figures=True)
    figdir = out / "figures"
    for name in ("benchmark.png", "domains.png"):
        assert (figdir / name).is_file(), f"missing {name}"
        assert (figdir / name).stat().st_size > 0
    # Figures are hashed into the manifest like every other artifact.
    manifest = json.loads((out / ARTIFACT_MANIFEST).read_text())
    assert "figures/benchmark.png" in manifest["artifacts"]


def test_invalid_data_section_is_reported(tmp_path):
    bad = cfg(data={"source": "synthetic", "n_slices": 2, "no_such_arg": 1})
    with pytest.raises(ExperimentError, match="invalid data"):
        run_experiment(bad, tmp_path, write=False)


def test_n_restarts_must_be_positive(tmp_path):
    bad = cfg(evaluation={"holdout": True, "n_restarts": 0})
    with pytest.raises(ExperimentError, match="n_restarts must be >= 1"):
        run_experiment(bad, tmp_path, write=False)


def test_load_metrics_requires_a_run(tmp_path):
    from caust.experiment import load_metrics

    with pytest.raises(ExperimentError, match="no metrics.json"):
        load_metrics(tmp_path)


def test_holdout_false_uses_every_slice(tmp_path):
    metrics = run_experiment(
        cfg(evaluation={"holdout": False, "n_restarts": 1}), tmp_path, write=False
    )
    assert metrics["n_train_slices"] == metrics["n_slices"]


def test_match_labels_only_permutes_colors():
    true = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([2, 2, 0, 0, 1, 1])
    out = match_labels(true, pred)
    np.testing.assert_array_equal(out, true)


def test_marker_recovery_is_config_gated(tmp_path):
    # Without evaluation.marker_genes the metrics dict must not change shape,
    # so recorded synthetic runs keep verifying byte-for-byte.
    plain = run_experiment(cfg(), tmp_path / "plain", write=False)
    assert "marker_recovery" not in plain

    marked = run_experiment(
        cfg(
            evaluation={
                "holdout": True,
                "n_restarts": 2,
                "marker_genes": ["CAUSAL_0", "CAUSAL_1", "NOT_A_GENE"],
            }
        ),
        tmp_path / "marked",
        write=False,
    )
    rec = marked["marker_recovery"]
    # NOT_A_GENE is absent from the pool, so the denominator is 2.
    assert rec["max"] == 2
    assert rec["markers_in_pool"] == ["CAUSAL_0", "CAUSAL_1"]
    assert 0 <= rec["hvg"] <= rec["max"]
    assert 0 <= rec["caust"] <= rec["max"]
