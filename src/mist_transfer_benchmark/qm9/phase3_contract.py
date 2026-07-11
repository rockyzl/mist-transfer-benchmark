"""Authenticate Phase 1 and locked Phase 2 evidence before MIST execution."""

from __future__ import annotations

import json
import stat
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import sha256_file
from .phase2_contract import Phase1Evidence, verify_phase1_evidence
from .phase2_output import OWNER_MARKER as PHASE2_OWNER_MARKER
from .phase2_output import OWNER_PAYLOAD as PHASE2_OWNER_PAYLOAD


class Phase3ContractError(ValueError):
    """Raised when locked upstream evidence cannot authenticate Phase 3."""


@dataclass(frozen=True)
class Phase3Evidence:
    phase1: Phase1Evidence
    phase2_run: dict[str, object]
    phase2_metrics: dict[str, object]
    phase2_run_sha256: str
    phase2_metrics_sha256: str
    phase2_predictions_sha256: str
    locked_selection_fingerprint: str
    train_target_scale: np.ndarray


def _regular(path: Path) -> None:
    if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.lstat().st_mode):
        raise Phase3ContractError(f"required artifact is not a regular file: {path}")


def verify_phase2_evidence(
    config: dict[str, object],
    *,
    phase1_dir: str | Path,
    phase2_dir: str | Path,
    source_path: str | Path,
    cache_dir: str | Path,
) -> Phase3Evidence:
    """Verify immutable split/cohort/source plus the locked Ridge comparison artifact."""

    phase1 = verify_phase1_evidence(config, phase1_dir, source_path)
    raw_directory = Path(phase2_dir)
    if raw_directory.is_symlink():
        raise Phase3ContractError("Phase 2 evidence directory must not be a symlink")
    directory = raw_directory.resolve(strict=True)
    if not directory.is_dir():
        raise Phase3ContractError("Phase 2 evidence directory is not a directory")
    marker = directory / PHASE2_OWNER_MARKER
    _regular(marker)
    if json.loads(marker.read_text(encoding="utf-8")) != PHASE2_OWNER_PAYLOAD:
        raise Phase3ContractError("Phase 2 ownership marker is invalid")
    run_path = directory / "phase2_run.json"
    sidecar = directory / "phase2_run.sha256"
    _regular(run_path)
    _regular(sidecar)
    run_sha = sha256_file(run_path)
    if sidecar.read_text(encoding="ascii") != f"{run_sha}  phase2_run.json\n":
        raise Phase3ContractError("Phase 2 run sidecar differs")
    observation = config["phase_2_observation"]
    if run_sha != observation["locked_phase_2_run_sha256"]:
        raise Phase3ContractError("Phase 2 run differs from the frozen observation")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("schema_version") != "qm9-phase2-classical-run-v1":
        raise Phase3ContractError("Phase 2 run schema differs")
    if run.get("phase3_mist_started") is not False:
        raise Phase3ContractError("Phase 2 run claims prior MIST execution")
    if run.get("phase1_run_sha256") != phase1.phase1_run_sha256:
        raise Phase3ContractError("Phase 1 and Phase 2 evidence disagree")
    artifacts = run.get("artifacts")
    if not isinstance(artifacts, dict):
        raise Phase3ContractError("Phase 2 artifact manifest is absent")
    for name, record in artifacts.items():
        path = directory / name
        if path.parent != directory:
            raise Phase3ContractError("Phase 2 artifact traversal is not allowed")
        _regular(path)
        if path.stat().st_size != record.get("bytes"):
            raise Phase3ContractError(f"Phase 2 artifact size differs: {name}")
        if sha256_file(path) != record.get("sha256"):
            raise Phase3ContractError(f"Phase 2 artifact hash differs: {name}")

    metrics_path = directory / "test_metrics.json"
    scaler_path = directory / "scaler.json"
    selection_path = directory / "selection_lock.json"
    predictions_path = directory / "predictions.jsonl"
    for path in (metrics_path, scaler_path, selection_path, predictions_path):
        _regular(path)
    locked_artifacts = observation["locked_artifacts"]
    expected_hashes = {
        metrics_path: locked_artifacts["test_metrics_sha256"],
        selection_path: locked_artifacts["selection_lock_sha256"],
        predictions_path: locked_artifacts["predictions_sha256"],
    }
    for path, expected in expected_hashes.items():
        if sha256_file(path) != expected:
            raise Phase3ContractError(f"frozen Phase 2 artifact differs: {path.name}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    fingerprint = observation["locked_selection_fingerprint"]
    if selection.get("selection_fingerprint") != fingerprint:
        raise Phase3ContractError("Phase 2 selection fingerprint differs")
    if run.get("selection_fingerprint") != fingerprint:
        raise Phase3ContractError("Phase 2 run selection fingerprint differs")
    cache = Path(cache_dir).resolve(strict=True)
    reservation = cache / "test-locks" / f"{fingerprint}.json"
    _regular(reservation)
    reservation_record = json.loads(reservation.read_text(encoding="utf-8"))
    if reservation_record.get("status") != "completed":
        raise Phase3ContractError("Phase 2 test reservation is not completed")
    if reservation_record.get("selection") != selection:
        raise Phase3ContractError("Phase 2 reservation and selection differ")
    result_hashes = reservation_record.get("result_hashes", {})
    if result_hashes.get("phase2_run_sha256") != run_sha:
        raise Phase3ContractError("Phase 2 reservation run hash differs")
    if result_hashes.get("test_metrics_sha256") != sha256_file(metrics_path):
        raise Phase3ContractError("Phase 2 reservation metrics hash differs")
    if result_hashes.get("predictions_sha256") != sha256_file(predictions_path):
        raise Phase3ContractError("Phase 2 reservation prediction hash differs")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("schema_version") != "qm9-phase2-test-metrics-v1":
        raise Phase3ContractError("Phase 2 metrics schema differs")
    if metrics.get("test_rows") != len(phase1.split.test):
        raise Phase3ContractError("Phase 2 full-test row count differs")
    if metrics.get("duplicate_clean_test_rows") != len(phase1.duplicate_clean_test):
        raise Phase3ContractError("Phase 2 duplicate-clean row count differs")
    ridge = metrics.get("methods", {}).get("ridge")
    if not isinstance(ridge, dict):
        raise Phase3ContractError("locked Ridge metrics are absent")
    aggregates = observation["locked_test_aggregate"]
    if (
        ridge["full_test"]["mean_normalized_mae_across_12_targets"]
        != aggregates["ridge_full"]
        or ridge["duplicate_clean_test"]["mean_normalized_mae_across_12_targets"]
        != aggregates["ridge_duplicate_clean"]
    ):
        raise Phase3ContractError("locked Ridge aggregate metrics differ")
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    scale = np.asarray(scaler.get("fitted_state", {}).get("scale_"), dtype=np.float64)
    if scale.shape != (12,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise Phase3ContractError("locked training-target scale is invalid")
    if (
        scaler.get("fitted_state_canonical_json_sha256")
        != observation["scaler_fitted_state_sha256"]
    ):
        raise Phase3ContractError("locked scaler state hash differs")
    return Phase3Evidence(
        phase1=phase1,
        phase2_run=run,
        phase2_metrics=metrics,
        phase2_run_sha256=run_sha,
        phase2_metrics_sha256=sha256_file(metrics_path),
        phase2_predictions_sha256=sha256_file(predictions_path),
        locked_selection_fingerprint=fingerprint,
        train_target_scale=scale,
    )
