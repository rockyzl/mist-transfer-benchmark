"""Bounded validation-only random-forest supplement after the locked Ridge test."""

from __future__ import annotations

import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

import tomllib

from .data import load_qm9_identities
from .download import assert_source_unchanged, copy_validated_source
from .io import atomic_write_bytes, atomic_write_json, sha256_file
from .phase2_contract import verify_phase1_evidence
from .phase2_output import (
    discard_phase2_workspace,
    finalize_phase2_workspace,
    prepare_phase2_workspace,
    write_phase2_owner,
)
from .phase2_pipeline import (
    _load_authenticated_features,
    _phase1_verification_payload,
    _resolve_repo_path,
    _rss_gib,
)
from .phase2_selection import random_forest_candidates, select_first_minimum
from .provenance import assert_code_provenance_unchanged, capture_code_provenance


def _process_rss_gib(pid: int) -> float:
    status = Path(f"/proc/{pid}/status")
    try:
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / (1024**2)
    except (FileNotFoundError, ProcessLookupError):
        return 0.0
    return 0.0


def _run_candidate(
    *,
    repo_root: Path,
    config_snapshot: Path,
    private_source: Path,
    feature_path: Path,
    candidate_id: str,
    destination: Path,
    timeout_seconds: int,
    max_rss_gib: float,
    progress,
) -> tuple[dict[str, object] | None, dict[str, object]]:
    command = [
        sys.executable,
        "-m",
        "mist_transfer_benchmark.qm9.phase2_rf_worker",
        "--config",
        str(config_snapshot),
        "--source",
        str(private_source),
        "--features",
        str(feature_path),
        "--candidate",
        candidate_id,
        "--mode",
        "validation",
        "--output",
        str(destination),
    ]
    environment = {
        "HOME": os.environ.get("HOME", str(repo_root)),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(repo_root / "src"),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    process = subprocess.Popen(
        command,
        cwd=repo_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    started = time.monotonic()
    observed_peak = 0.0
    last_progress = started
    stopped_reason: str | None = None
    while process.poll() is None:
        elapsed = time.monotonic() - started
        observed_peak = max(observed_peak, _process_rss_gib(process.pid))
        if observed_peak > max_rss_gib:
            stopped_reason = "worker exceeded frozen 64 GiB RSS ceiling"
        elif elapsed > timeout_seconds:
            stopped_reason = "worker exceeded frozen 24-hour candidate ceiling"
        if stopped_reason is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            break
        if time.monotonic() - last_progress >= 10:
            progress(
                f"RF {candidate_id}: {elapsed:.0f}s elapsed, "
                f"worker RSS {observed_peak:.3f} GiB"
            )
            last_progress = time.monotonic()
        time.sleep(1)
    stdout, stderr = process.communicate()
    execution = {
        "candidate_id": candidate_id,
        "command": command,
        "exit_code": process.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "observed_worker_peak_rss_gib": observed_peak,
        "timeout_seconds": timeout_seconds,
        "max_rss_gib": max_rss_gib,
        "stdout": stdout,
        "stderr": stderr,
        "stopped_reason": stopped_reason,
    }
    if stopped_reason is not None or process.returncode != 0 or not destination.is_file():
        destination.unlink(missing_ok=True)
        return None, execution
    result = json.loads(destination.read_text(encoding="utf-8"))
    destination.unlink()
    return result, execution


def run_rf_validation_supplement(
    *,
    config_path: str | Path,
    cache_dir: str | Path,
    phase1_dir: str | Path,
    feature_dir: str | Path,
    locked_run_dir: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
    progress=print,
) -> dict[str, object]:
    """Attempt all RF validation candidates without any test-label or test-prediction access."""

    started = time.monotonic()
    repo_root = Path(__file__).resolve().parents[3]
    config_path = _resolve_repo_path(repo_root, config_path).resolve(strict=True)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    cache = _resolve_repo_path(repo_root, cache_dir).resolve(strict=True)
    source_path = cache / "qm9.csv"
    evidence = verify_phase1_evidence(
        config, _resolve_repo_path(repo_root, phase1_dir), source_path
    )
    locked_dir = _resolve_repo_path(repo_root, locked_run_dir).resolve(strict=True)
    locked_run = json.loads((locked_dir / "phase2_run.json").read_text(encoding="utf-8"))
    locked_selection = json.loads((locked_dir / "selection_lock.json").read_text(encoding="utf-8"))
    fingerprint = locked_selection["selection_fingerprint"]
    reservation = cache / "test-locks" / f"{fingerprint}.json"
    reservation_record = json.loads(reservation.read_text(encoding="utf-8"))
    if reservation_record.get("status") != "completed":
        raise ValueError("locked Ridge test reservation is not completed")
    if locked_run.get("selection_fingerprint") != fingerprint:
        raise ValueError("locked Phase 2 run and selection fingerprint disagree")

    provenance = capture_code_provenance(repo_root, config_path)
    workspace = prepare_phase2_workspace(output_dir, repo_root, overwrite=overwrite)
    try:
        protocol_snapshot = workspace.staging_dir / "protocol_config.snapshot.toml"
        atomic_write_bytes(protocol_snapshot, config_path.read_bytes(), mode=0o600)
        private_source = workspace.staging_dir / "source.snapshot.csv"
        private_snapshot = copy_validated_source(
            source_path,
            private_source,
            evidence.source_snapshot,
            expected_bytes=evidence.source_snapshot.bytes,
            expected_sha256=evidence.source_snapshot.sha256,
        )
        data = load_qm9_identities(private_source)
        feature_path = _resolve_repo_path(repo_root, feature_dir).resolve(strict=True)
        _, feature = _load_authenticated_features(
            feature_path,
            config,
            phase1_run_sha256=evidence.phase1_run_sha256,
            source_row_identity_sha256=data.row_identity_sha256,
            source_smiles_sha256=data.raw_smiles_sha256,
        )
        resources = config["resource_budget"]
        timeout_seconds = int(resources["max_wall_clock_hours_per_validation_candidate"] * 3600)
        max_rss_gib = float(resources["max_parent_process_peak_rss_gib"])
        if resources["cpu_worker_limit"] != 16:
            raise ValueError("frozen RF supplemental requires exactly 16 or fewer CPU workers")
        specs = random_forest_candidates(config)
        if any(
            candidate.parameters["n_jobs"] > resources["cpu_worker_limit"]
            for candidate in specs
        ):
            raise ValueError("RF candidate n_jobs exceeds frozen CPU limit")
        results: list[dict[str, object]] = []
        executions: list[dict[str, object]] = []
        status = "complete"
        for candidate in specs:
            progress(f"starting isolated RF validation candidate {candidate.candidate_id}")
            result, execution = _run_candidate(
                repo_root=repo_root,
                config_snapshot=protocol_snapshot,
                private_source=private_source,
                feature_path=feature_path / "feature_matrix.npz",
                candidate_id=candidate.candidate_id,
                destination=workspace.staging_dir / f".{candidate.candidate_id}.result.json",
                timeout_seconds=timeout_seconds,
                max_rss_gib=max_rss_gib,
                progress=progress,
            )
            executions.append(execution)
            if result is None:
                status = "stopped-after-bounded-validation-attempt"
                break
            results.append(result)
            progress(
                f"RF {candidate.candidate_id} validation score "
                f"{result['mean_normalized_mae_across_12_targets']:.10f}"
            )
        selected = None
        if status == "complete":
            selected = select_first_minimum(
                results, config["classical"]["random_forest"]["candidate_order"]
            )
        attempt = {
            "schema_version": "qm9-phase2-rf-validation-supplement-v1",
            "status": status,
            "scope": "validation-only-after-locked-ridge-test",
            "test_labels_loaded": False,
            "test_predictions_generated": False,
            "second_test_evaluation": False,
            "locked_selection_fingerprint_unchanged": fingerprint,
            "locked_phase2_run_sha256": sha256_file(locked_dir / "phase2_run.json"),
            "reason_for_separate_attempt": (
                "locked execution invoked run_random_forest=False to prioritize mandatory "
                "dummy/Ridge controls; this was an execution choice, not an end-user request"
            ),
            "cpu_worker_limit": resources["cpu_worker_limit"],
            "candidate_timeout_seconds": timeout_seconds,
            "candidate_max_rss_gib": max_rss_gib,
            "candidate_order": config["classical"]["random_forest"]["candidate_order"],
            "results": results,
            "executions": executions,
            "selected_on_validation_only": selected,
            "feature_canonical_csr_sha256": feature["matrix"]["canonical_csr_sha256"],
            "runtime_seconds": time.monotonic() - started,
            "parent_peak_rss_gib": _rss_gib(),
        }
        attempt_path = workspace.staging_dir / "random_forest_attempt.json"
        atomic_write_json(attempt_path, attempt, mode=0o600)
        atomic_write_json(
            workspace.staging_dir / "phase1_verification.json",
            _phase1_verification_payload(evidence),
            mode=0o600,
        )
        atomic_write_json(
            workspace.staging_dir / "code_provenance.json", provenance, mode=0o600
        )
        assert_source_unchanged(
            private_source,
            private_snapshot,
            expected_bytes=evidence.source_snapshot.bytes,
            expected_sha256=evidence.source_snapshot.sha256,
        )
        assert_source_unchanged(
            source_path,
            evidence.source_snapshot,
            expected_bytes=evidence.source_snapshot.bytes,
            expected_sha256=evidence.source_snapshot.sha256,
        )
        assert_code_provenance_unchanged(provenance, repo_root, config_path)
        private_source.unlink()
        run = {
            "schema_version": "qm9-phase2-rf-supplement-run-v1",
            "scientific_status": "validation-only-no-test-evaluation",
            "locked_selection_fingerprint_unchanged": fingerprint,
            "random_forest_attempt_sha256": sha256_file(attempt_path),
            "protocol_config_snapshot_sha256": sha256_file(protocol_snapshot),
            "runtime_seconds": time.monotonic() - started,
            "platform": platform.platform(),
            "phase3_mist_started": False,
        }
        run_path = workspace.staging_dir / "phase2_run.json"
        atomic_write_json(run_path, run, mode=0o600)
        run_sha = sha256_file(run_path)
        atomic_write_bytes(
            workspace.staging_dir / "phase2_run.sha256",
            f"{run_sha}  phase2_run.json\n".encode("ascii"),
            mode=0o600,
        )
        write_phase2_owner(workspace.staging_dir)
        finalize_phase2_workspace(workspace)
        attempt["supplement_run_sha256"] = run_sha
        attempt["output_dir"] = str(workspace.output_dir)
        return attempt
    finally:
        discard_phase2_workspace(workspace)
