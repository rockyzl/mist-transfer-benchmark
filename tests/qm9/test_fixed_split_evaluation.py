from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np
import pytest

from mist_transfer_benchmark.qm9.fixed_split_evaluation import (
    APPROVAL_SCHEMA,
    PUBLICATION_APPROVAL_SCHEMA,
    SUMMARY_SCHEMA,
    FixedSplitEvaluationError,
    LazyTestTargetGate,
    TargetAccessGate,
    approve_publication,
    file_sha256,
    monitor_curves,
    paired_delta_bootstrap,
    require_clean_run_identity,
    require_publication_ready,
    run_smoke_protocol,
)


def _config() -> dict:
    with Path("configs/qm9_fixed_split_evaluation_v2.toml").open("rb") as handle:
        value = tomllib.load(handle)
    return value


def _write_selection_approval(run: Path, path: Path, *, gate_hash: str | None = None) -> Path:
    payload = {
        "schema_version": APPROVAL_SCHEMA,
        "decision": "approve-test-unlock",
        "global_freeze_sha256": gate_hash or file_sha256(run / "global-freeze-gate.json"),
        "reviewed_manifest_sha256": file_sha256(run / "manifest.json"),
        "reviewer": "independent-smoke-reviewer",
        "reviewed_at_utc": "2026-07-18T18:00:00Z",
        "notes": "Smoke artifacts reviewed.",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _write_publication_approval(run: Path, path: Path) -> Path:
    payload = {
        "schema_version": PUBLICATION_APPROVAL_SCHEMA,
        "decision": "approve-publication",
        "reviewed_manifest_sha256": file_sha256(run / "manifest.json"),
        "summary_sha256": file_sha256(run / "summary.json"),
        "reviewer": "independent-publication-reviewer",
        "reviewed_at_utc": "2026-07-18T19:00:00Z",
        "notes": "Summary, loss monitor, manifest, and hashes reviewed.",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def test_gate_denies_early_test_read() -> None:
    gate = TargetAccessGate(np.ones((2, 12)))
    with pytest.raises(FixedSplitEvaluationError, match="before global freeze"):
        gate.read()


def test_formal_run_rejects_dirty_identity() -> None:
    with pytest.raises(FixedSplitEvaluationError, match="clean worktree"):
        require_clean_run_identity({"git": {"dirty": True}})
    require_clean_run_identity({"git": {"dirty": False}})


def test_gate_requires_matching_approval_and_allows_exactly_one_read() -> None:
    gate = TargetAccessGate(np.ones((2, 12)))
    frozen_hash = "a" * 64
    with pytest.raises(FixedSplitEvaluationError, match="approval hash"):
        gate.authorize(frozen_hash, "b" * 64)
    gate.authorize(frozen_hash, frozen_hash)
    assert gate.read().shape == (2, 12)
    with pytest.raises(FixedSplitEvaluationError, match="exactly once"):
        gate.read()

    lazy = LazyTestTargetGate(lambda: np.ones((2, 12)))
    lazy.authorize(frozen_hash, frozen_hash)
    assert lazy.read().shape == (2, 12)
    with pytest.raises(FixedSplitEvaluationError, match="exactly once"):
        lazy.read()


def test_validation_increase_is_marked() -> None:
    result = monitor_curves([3, 2, 1], [1.0, 1.1, 1.2], increase_mark_after=2, max_epochs=10)
    assert result["status"] == "warning"
    assert result["maximum_consecutive_validation_increases"] == 2


def test_smoke_writes_exact_v2_contract(tmp_path: Path) -> None:
    run = tmp_path / "run"
    stage_a = run_smoke_protocol(_config(), run)
    assert stage_a["stage"] == "AWAITING_SELECTION_REVIEW"
    assert stage_a["test_access"]["read_count"] == 0
    assert stage_a["complete"] is False
    assert stage_a["run_identity"]["git"]["commit"]
    assert stage_a["run_identity"]["git"]["branch"]
    assert isinstance(stage_a["run_identity"]["git"]["dirty"], bool)
    assert stage_a["run_identity"]["dependency_lock"]["sha256"]
    assert stage_a["run_identity"]["dependencies"]["rdkit"]
    assert stage_a["run_identity"]["repository"]["tracked_worktree_sha256"]
    assert set(stage_a["run_identity"]["dependencies"]) == {
        "numpy",
        "scipy",
        "scikit-learn",
        "rdkit",
        "torch",
        "xgboost",
    }
    freeze = json.loads((run / "global-freeze-gate.json").read_text())
    assert freeze["run_identity"] == stage_a["run_identity"]
    stage_a_reviews = {
        item["id"]: item["status"] for item in stage_a["critical_reviews"]
    }
    assert stage_a_reviews["selection-freeze"] == "awaiting-human-review"
    assert stage_a_reviews["test-unlock"] == "blocked-awaiting-selection-approval"
    approval = _write_selection_approval(run, tmp_path / "selection-approval.json")
    manifest = run_smoke_protocol(_config(), run, review_approval_path=approval)
    assert manifest["complete"] is True
    assert manifest["stage"] == "AWAITING_PUBLICATION_REVIEW"
    assert manifest["test_access"]["read_count"] == 1
    assert [item["event"] for item in manifest["events"]].count("test-labels-read-once") == 1
    assert manifest["selected_seeds"] == manifest["completed_seeds"]
    assert manifest["publication_ready"] is False
    review_status = {
        review["id"]: review["status"] for review in manifest["critical_reviews"]
    }
    assert review_status == {
        "input-boundary": "automated-review-passed",
        "selection-freeze": "human-review-passed",
        "test-unlock": "human-review-passed",
        "publication": "awaiting-independent-review",
    }
    summary = json.loads((run / "summary.json").read_text())
    assert summary["schema_version"] == SUMMARY_SCHEMA
    assert summary["fixed_mist"]["inference_only"] is True
    assert (run / "loss-monitor.html").is_file()
    for seed in manifest["selected_seeds"]:
        payload = json.loads((run / "seeds" / f"{seed}.json").read_text())
        assert payload["validation"]["mlp_monitoring"]["training_loss"]
        for model in (
            "engineered_ridge",
            "xgboost",
            "mlp",
            "traditional_ensemble",
            "all_model_ensemble",
        ):
            prediction = np.load(run / "predictions" / f"{seed}-{model}.npy")
            assert prediction.shape == (15, 12)

    with pytest.raises(FixedSplitEvaluationError, match="not approved"):
        require_publication_ready(run)
    publication_approval = _write_publication_approval(
        run, tmp_path / "publication-approval.json"
    )
    published = approve_publication(run, publication_approval)
    assert published["publication_ready"] is True
    assert published["stage"] == "PUBLICATION_APPROVED"
    require_publication_ready(run)


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
    run = tmp_path / "run"
    run_smoke_protocol(_config(), run)
    approval = _write_selection_approval(run, tmp_path / "selection.json")
    run_smoke_protocol(_config(), run, review_approval_path=approval)
    (run / "summary.json").write_text("{}\n")
    with pytest.raises(FixedSplitEvaluationError, match="artifact changed"):
        run_smoke_protocol(_config(), run)


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
    run = tmp_path / "run"
    run_smoke_protocol(_config(), run, include_mist_validation=False)
    approval = _write_selection_approval(run, tmp_path / "selection.json")
    run_smoke_protocol(
        _config(), run, include_mist_validation=False, review_approval_path=approval
    )
    summary = json.loads((run / "summary.json").read_text())
    assert summary["reporting_roles"]["supplemental"] == "omitted"
    assert summary["reporting_roles"]["supplemental_omission_reason"]


def test_invalid_selection_approval_and_freeze_tamper_are_rejected(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run_smoke_protocol(_config(), run)
    invalid = _write_selection_approval(
        run, tmp_path / "invalid-selection.json", gate_hash="f" * 64
    )
    with pytest.raises(FixedSplitEvaluationError, match="another global freeze"):
        run_smoke_protocol(_config(), run, review_approval_path=invalid)

    invalid_time = _write_selection_approval(run, tmp_path / "invalid-time.json")
    invalid_time_payload = json.loads(invalid_time.read_text())
    invalid_time_payload["reviewed_at_utc"] = "not-a-time"
    invalid_time.write_text(json.dumps(invalid_time_payload) + "\n")
    with pytest.raises(FixedSplitEvaluationError, match="ISO-8601"):
        run_smoke_protocol(_config(), run, review_approval_path=invalid_time)

    valid = _write_selection_approval(run, tmp_path / "valid-selection.json")
    (run / "global-freeze-gate.json").write_text("{}\n")
    with pytest.raises(FixedSplitEvaluationError, match="artifact changed"):
        run_smoke_protocol(_config(), run, review_approval_path=valid)


def test_manifest_tamper_invalidates_selection_approval(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run_smoke_protocol(_config(), run)
    approval = _write_selection_approval(run, tmp_path / "selection.json")
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["unexpected_tamper"] = True
    (run / "manifest.json").write_text(json.dumps(manifest) + "\n")
    with pytest.raises(FixedSplitEvaluationError, match="another manifest state"):
        run_smoke_protocol(_config(), run, review_approval_path=approval)


def test_publication_approval_and_artifact_tamper_fail_closed(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run_smoke_protocol(_config(), run)
    selection = _write_selection_approval(run, tmp_path / "selection.json")
    run_smoke_protocol(_config(), run, review_approval_path=selection)
    publication = _write_publication_approval(run, tmp_path / "publication.json")
    payload = json.loads(publication.read_text())
    payload["summary_sha256"] = "0" * 64
    publication.write_text(json.dumps(payload) + "\n")
    with pytest.raises(FixedSplitEvaluationError, match="another summary"):
        approve_publication(run, publication)

    publication = _write_publication_approval(run, tmp_path / "publication-valid.json")
    approve_publication(run, publication)
    (run / "publication-review-approval.json").write_text("{}\n")
    with pytest.raises(FixedSplitEvaluationError, match="published artifact changed"):
        require_publication_ready(run)


def test_published_manifest_tamper_is_rejected(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run_smoke_protocol(_config(), run)
    selection = _write_selection_approval(run, tmp_path / "selection.json")
    run_smoke_protocol(_config(), run, review_approval_path=selection)
    publication = _write_publication_approval(run, tmp_path / "publication.json")
    approve_publication(run, publication)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["unexpected_tamper"] = True
    (run / "manifest.json").write_text(json.dumps(manifest) + "\n")
    with pytest.raises(FixedSplitEvaluationError, match="manifest checksum"):
        require_publication_ready(run)
