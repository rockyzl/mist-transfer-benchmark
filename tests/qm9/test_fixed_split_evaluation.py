from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

from mist_transfer_benchmark.qm9.fixed_split_evaluation import (
    SUMMARY_SCHEMA,
    FixedSplitEvaluationError,
    TargetAccessGate,
    monitor_curves,
    paired_delta_bootstrap,
    run_smoke_protocol,
)


def _config() -> dict:
    with Path("configs/qm9_fixed_split_evaluation_v2.toml").open("rb") as handle:
        value = tomllib.load(handle)
    return value


def test_gate_denies_early_test_read() -> None:
    gate = TargetAccessGate(np.ones((2, 12)))
    with pytest.raises(FixedSplitEvaluationError, match="before global freeze"):
        gate.read()


def test_validation_increase_is_marked() -> None:
    result = monitor_curves([3, 2, 1], [1.0, 1.1, 1.2], increase_mark_after=2, max_epochs=10)
    assert result["status"] == "warning"
    assert result["maximum_consecutive_validation_increases"] == 2


def test_smoke_writes_exact_v2_contract(tmp_path: Path) -> None:
    manifest = run_smoke_protocol(_config(), tmp_path)
    assert manifest["complete"] is True
    assert manifest["test_access"]["read_count"] == 1
    assert manifest["selected_seeds"] == manifest["completed_seeds"]
    assert manifest["publication_ready"] is False
    review_status = {
        review["id"]: review["status"] for review in manifest["critical_reviews"]
    }
    assert review_status == {
        "input-boundary": "automated-review-passed",
        "selection-freeze": "automated-review-passed",
        "test-unlock": "automated-review-passed",
        "publication": "independent-review-required",
    }
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["schema_version"] == SUMMARY_SCHEMA
    assert summary["fixed_mist"]["inference_only"] is True
    assert (tmp_path / "loss-monitor.html").is_file()
    for seed in manifest["selected_seeds"]:
        payload = json.loads((tmp_path / "seeds" / f"{seed}.json").read_text())
        assert payload["validation"]["mlp_monitoring"]["training_loss"]
        for model in (
            "engineered_ridge",
            "xgboost",
            "mlp",
            "traditional_ensemble",
            "all_model_ensemble",
        ):
            prediction = np.load(tmp_path / "predictions" / f"{seed}-{model}.npy")
            assert prediction.shape == (15, 12)


def test_config_mismatch_is_rejected(tmp_path: Path) -> None:
    config = _config()
    config["frozen_winners"]["ridge_alpha"] = 9.0
    with pytest.raises(FixedSplitEvaluationError, match="contract differs"):
        run_smoke_protocol(config, tmp_path)


def test_incomplete_output_is_fail_closed(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text('{"complete": false}\n')
    with pytest.raises(FixedSplitEvaluationError, match="not resumable"):
        run_smoke_protocol(_config(), tmp_path)


def test_complete_output_tamper_is_rejected(tmp_path: Path) -> None:
    run_smoke_protocol(_config(), tmp_path)
    (tmp_path / "summary.json").write_text("{}\n")
    with pytest.raises(FixedSplitEvaluationError, match="artifact changed"):
        run_smoke_protocol(_config(), tmp_path)


def test_paired_bootstrap_per_target_math() -> None:
    truth = np.zeros((8, 12))
    mist = np.ones((8, 12))
    candidate = np.full((8, 12), 0.5)
    result = paired_delta_bootstrap(
        truth, candidate, mist, np.ones(12), samples=50, seed=7, confidence=0.95
    )
    assert result["point"] == pytest.approx(-0.5)
    assert result["per_target"]["mu"]["point"] == pytest.approx(-0.5)


def test_supplemental_is_explicitly_omitted_without_mist_validation(tmp_path: Path) -> None:
    run_smoke_protocol(_config(), tmp_path, include_mist_validation=False)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["reporting_roles"]["supplemental"] == "omitted"
    assert summary["reporting_roles"]["supplemental_omission_reason"]
