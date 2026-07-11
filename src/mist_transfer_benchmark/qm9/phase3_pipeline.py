"""Guarded Phase 3 audit, smoke, one-shot MIST inference, and comparison."""

from __future__ import annotations

import json
import os
import platform
import resource
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import tomllib

from .constants import TARGET_COLUMNS
from .data import load_qm9_identities
from .download import assert_source_unchanged, copy_validated_source
from .io import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_hash,
    sha256_file,
)
from .phase2_metrics import native_metrics
from .phase2_targets import load_targets_for_indices
from .phase3_adapter import assert_no_manual_inverse_transform, validate_prediction_matrix
from .phase3_contract import Phase3Evidence, verify_phase2_evidence
from .phase3_model import (
    EXPECTED_UNITS,
    MODEL_ID,
    MODEL_REVISION,
    build_model_audit,
    validate_model_config,
)
from .phase3_output import (
    OWNER_MARKER,
    OWNER_PAYLOAD,
    discard_workspace,
    finalize_workspace,
    prepare_phase3_workspace,
    write_owner,
)
from .phase3_reservation import (
    authorize_test_inference,
    complete_inference,
    freeze_inference,
    reserve_inference_once,
)
from .phase3_runtime import verify_runtime
from .provenance import assert_code_provenance_unchanged, capture_code_provenance


class Phase3PipelineError(ValueError):
    """Raised when a Phase 3 hard gate closes."""


def _rss_gib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024**3 if sys.platform == "darwin" else 1024**2
    return float(value) / divisor


def _repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _artifact(path: Path) -> dict[str, object]:
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _freeze_bytes(freeze: list[str]) -> bytes:
    return ("\n".join(freeze) + "\n").encode("utf-8")


def run_phase3_audit(
    *,
    config_path: str | Path,
    model_dir: str | Path,
    runtime_python: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, object]:
    """Freeze model bytes, static audit, and isolated runtime before code execution."""

    started = time.monotonic()
    repo_root = Path(__file__).resolve().parents[3]
    config_path = _repo_path(repo_root, config_path).resolve(strict=True)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    validate_model_config(config)
    model_dir = _repo_path(repo_root, model_dir).resolve(strict=True)
    audit = build_model_audit(model_dir)
    runtime, freeze = verify_runtime(_repo_path(repo_root, runtime_python), config)
    provenance = capture_code_provenance(repo_root, config_path)
    workspace = prepare_phase3_workspace(output_dir, repo_root, overwrite=overwrite)
    try:
        staging = workspace.staging_dir
        protocol = staging / "protocol_config.snapshot.toml"
        atomic_write_bytes(protocol, config_path.read_bytes(), mode=0o600)
        audit_path = staging / "model_audit.json"
        runtime_path = staging / "runtime_environment.json"
        freeze_path = staging / "runtime_environment.freeze.txt"
        provenance_path = staging / "code_provenance.json"
        failure_path = staging / "failure_log.json"
        atomic_write_json(audit_path, audit, mode=0o600)
        atomic_write_json(runtime_path, runtime, mode=0o600)
        atomic_write_bytes(freeze_path, _freeze_bytes(freeze), mode=0o600)
        atomic_write_json(provenance_path, provenance, mode=0o600)
        atomic_write_json(
            failure_path,
            {
                "schema_version": "qm9-mist-phase3-failure-log-v1",
                "events": [],
                "remote_code_executed": False,
            },
            mode=0o600,
        )
        assert_code_provenance_unchanged(provenance, repo_root, config_path)
        artifact_paths = (
            audit_path,
            runtime_path,
            freeze_path,
            provenance_path,
            protocol,
            failure_path,
        )
        run = {
            "schema_version": "qm9-mist-phase3-audit-run-v1",
            "status": "hard-gate-pass-before-remote-code-execution",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_audit_sha256": sha256_file(audit_path),
            "runtime_environment_sha256": sha256_file(runtime_path),
            "runtime_freeze_file_sha256": sha256_file(freeze_path),
            "runtime_freeze_canonical_sha256": runtime["complete_freeze_canonical_sha256"],
            "protocol_config_snapshot_sha256": sha256_file(protocol),
            "code_provenance_aggregate_sha256": provenance["aggregate_sha256"],
            "remote_code_executed": False,
            "runtime_seconds": time.monotonic() - started,
            "artifacts": {path.name: _artifact(path) for path in artifact_paths},
        }
        run_path = staging / "phase3_audit_run.json"
        atomic_write_json(run_path, run, mode=0o600)
        run_sha = sha256_file(run_path)
        atomic_write_bytes(
            staging / "phase3_audit_run.sha256",
            f"{run_sha}  phase3_audit_run.json\n".encode("ascii"),
            mode=0o600,
        )
        write_owner(staging)
        finalize_workspace(workspace)
        run["phase3_audit_run_sha256"] = run_sha
        run["output_dir"] = str(workspace.output_dir)
        return run
    finally:
        discard_workspace(workspace)


def _verify_audit_dir(
    audit_dir: Path,
    *,
    current_audit: dict[str, object],
    current_runtime: dict[str, object],
) -> tuple[dict[str, object], str]:
    raw = audit_dir
    if raw.is_symlink():
        raise Phase3PipelineError("Phase 3 audit directory must not be a symlink")
    directory = raw.resolve(strict=True)
    marker = directory / OWNER_MARKER
    if json.loads(marker.read_text(encoding="utf-8")) != OWNER_PAYLOAD:
        raise Phase3PipelineError("Phase 3 audit ownership marker differs")
    run_path = directory / "phase3_audit_run.json"
    run_sha = sha256_file(run_path)
    if (directory / "phase3_audit_run.sha256").read_text(encoding="ascii") != (
        f"{run_sha}  phase3_audit_run.json\n"
    ):
        raise Phase3PipelineError("Phase 3 audit run sidecar differs")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("status") != "hard-gate-pass-before-remote-code-execution":
        raise Phase3PipelineError("Phase 3 audit did not pass the pre-execution gate")
    for name, record in run.get("artifacts", {}).items():
        path = directory / name
        if path.parent != directory or path.stat().st_size != record.get("bytes"):
            raise Phase3PipelineError(f"Phase 3 audit artifact size differs: {name}")
        if sha256_file(path) != record.get("sha256"):
            raise Phase3PipelineError(f"Phase 3 audit artifact hash differs: {name}")
    observed_audit = json.loads((directory / "model_audit.json").read_text(encoding="utf-8"))
    observed_runtime = json.loads(
        (directory / "runtime_environment.json").read_text(encoding="utf-8")
    )
    if observed_audit != current_audit:
        raise Phase3PipelineError("model audit bytes differ from the current snapshot")
    if observed_runtime != current_runtime:
        raise Phase3PipelineError("inference runtime differs from its audit")
    return run, run_sha


def _write_worker_inputs(path: Path, data, indices: np.ndarray) -> int:
    return atomic_write_jsonl(
        path,
        (
            {
                "position": position,
                "source_row_index": int(index),
                "source_smiles": data.source_smiles[int(index)],
            }
            for position, index in enumerate(indices)
        ),
        mode=0o600,
    )


def _invoke_worker(
    *,
    runtime_python: Path,
    repo_root: Path,
    snapshot: Path,
    input_path: Path,
    output_path: Path,
    report_path: Path,
    batch_size: int,
    device: str,
    timeout_seconds: int,
    isolated_home: Path,
    progress,
) -> dict[str, object]:
    command = [
        str(runtime_python.absolute()),
        "-m",
        "mist_transfer_benchmark.qm9.phase3_worker",
        "--snapshot",
        str(snapshot),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--report",
        str(report_path),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
    ]
    environment = {
        "HOME": str(isolated_home),
        "HF_HOME": str(isolated_home / "hf"),
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(repo_root / "src"),
        "PATH": os.pathsep.join((str(runtime_python.parent), "/usr/bin", "/bin")),
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TZ": "UTC",
    }
    isolated_home.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        command,
        cwd=repo_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    started = time.monotonic()
    lines: list[str] = []
    timed_out = False
    while process.poll() is None:
        if time.monotonic() - started > timeout_seconds:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            break
        for key, _ in selector.select(timeout=1.0):
            line = key.fileobj.readline()
            if line:
                lines.append(line)
                if line.startswith("MIST_INFERENCE_PROGRESS"):
                    progress(line.strip())
    remainder, _ = process.communicate()
    if remainder:
        lines.append(remainder)
    selector.close()
    execution = {
        "command": command,
        "exit_code": process.returncode,
        "timeout_seconds": timeout_seconds,
        "timed_out": timed_out,
        "combined_output": "".join(lines),
        "runtime_seconds": time.monotonic() - started,
        "environment_policy": "sanitized-offline-local-snapshot-only",
    }
    if (
        timed_out
        or process.returncode != 0
        or not output_path.is_file()
        or not report_path.is_file()
    ):
        raise Phase3PipelineError(
            f"isolated MIST worker failed: {json.dumps(execution, sort_keys=True)}"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["execution"] = execution
    return report


def _validate_worker_report(
    report: dict[str, object],
    *,
    expected_rows: int,
    expected_indices: np.ndarray,
    batch_size: int,
    device: str,
) -> None:
    required_true = (
        "model_local_files_only",
        "use_safetensors",
        "explicit_same_revision_tokenizer",
        "tokenizer_none_fails_closed",
        "model_eval",
        "torch_inference_mode",
        "model_predict_returns_native_units",
        "named_outputs_stacked_by_config_order",
    )
    if any(report.get(name) is not True for name in required_true):
        raise Phase3PipelineError("isolated worker guard attestation is incomplete")
    assert_no_manual_inverse_transform(report)
    if report.get("model_revision") != MODEL_REVISION:
        raise Phase3PipelineError("worker model revision differs")
    if report.get("input_rows") != expected_rows or report.get("output_shape") != [
        expected_rows,
        12,
    ]:
        raise Phase3PipelineError("worker row/output shape differs")
    expected_device = "cuda:0" if device == "cuda" else "cpu"
    if report.get("batch_size") != batch_size or report.get("device") != expected_device:
        raise Phase3PipelineError("worker batch/device differs")
    if report.get("channel_order") != list(TARGET_COLUMNS):
        raise Phase3PipelineError("worker channel order differs")
    if report.get("channel_units") != list(EXPECTED_UNITS):
        raise Phase3PipelineError("worker channel units differ")
    expected_index_hash = canonical_hash([int(value) for value in expected_indices])
    if report.get("input_source_indices_sha256") != expected_index_hash:
        raise Phase3PipelineError("worker source-index order differs")
def _prediction_rows(data, indices, truth, prediction, clean_indices):
    clean = {int(index) for index in clean_indices}
    for position, source_index in enumerate(indices):
        index = int(source_index)
        yield {
            "source_row_index": index,
            "record_id": data.record_id(index),
            "duplicate_clean_test": index in clean,
            "target_order": list(TARGET_COLUMNS),
            "observed": truth[position].tolist(),
            "predicted": prediction[position].tolist(),
        }


def _comparison(mist_metrics: dict[str, object], evidence: Phase3Evidence) -> dict[str, object]:
    ridge = evidence.phase2_metrics["methods"]["ridge"]
    cohorts: dict[str, object] = {}
    for cohort in ("full_test", "duplicate_clean_test"):
        mist = mist_metrics[cohort]
        locked = ridge[cohort]
        per_target = {}
        for name in TARGET_COLUMNS:
            per_target[name] = {
                metric: {
                    "mist": mist["per_target"][name][metric],
                    "locked_ridge": locked["per_target"][name][metric],
                    "mist_minus_locked_ridge": (
                        mist["per_target"][name][metric]
                        - locked["per_target"][name][metric]
                    ),
                }
                for metric in ("mae", "rmse", "r2")
            }
        cohorts[cohort] = {
            "rows": mist["rows"],
            "mist_mean_normalized_mae": mist[
                "mean_normalized_mae_across_12_targets"
            ],
            "locked_ridge_mean_normalized_mae": locked[
                "mean_normalized_mae_across_12_targets"
            ],
            "mist_minus_locked_ridge_mean_normalized_mae": (
                mist["mean_normalized_mae_across_12_targets"]
                - locked["mean_normalized_mae_across_12_targets"]
            ),
            "per_target": per_target,
        }
    return {
        "schema_version": "qm9-mist-vs-locked-ridge-comparison-v1",
        "comparison_created_after_mist_metrics_frozen": True,
        "locked_phase2_run_sha256": evidence.phase2_run_sha256,
        "locked_phase2_metrics_sha256": evidence.phase2_metrics_sha256,
        "candidate_split_reconstructed_from_public_code": True,
        "official_checkpoint_test_reproduction_claimed": False,
        "labels": "QM9 DFT-computed quantum-chemistry properties, not experiments",
        "cohorts": cohorts,
    }


def run_phase3_inference(
    *,
    config_path: str | Path,
    cache_dir: str | Path,
    phase1_dir: str | Path,
    phase2_dir: str | Path,
    audit_dir: str | Path,
    model_dir: str | Path,
    runtime_python: str | Path,
    output_dir: str | Path,
    device: str = "auto",
    initial_batch_size: int = 128,
    overwrite: bool = False,
    progress=print,
) -> dict[str, object]:
    """Smoke on train/validation, reserve, then infer the candidate test exactly once."""

    started = time.monotonic()
    repo_root = Path(__file__).resolve().parents[3]
    config_path = _repo_path(repo_root, config_path).resolve(strict=True)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    validate_model_config(config)
    cache = _repo_path(repo_root, cache_dir).resolve(strict=True)
    source_path = cache / "qm9.csv"
    evidence = verify_phase2_evidence(
        config,
        phase1_dir=_repo_path(repo_root, phase1_dir),
        phase2_dir=_repo_path(repo_root, phase2_dir),
        source_path=source_path,
        cache_dir=cache,
    )
    model_dir = _repo_path(repo_root, model_dir).resolve(strict=True)
    current_audit = build_model_audit(model_dir)
    runtime_python = _repo_path(repo_root, runtime_python).absolute()
    current_runtime, freeze = verify_runtime(runtime_python, config)
    audit_run, audit_run_sha = _verify_audit_dir(
        _repo_path(repo_root, audit_dir),
        current_audit=current_audit,
        current_runtime=current_runtime,
    )
    if device == "auto":
        selected_device = "cuda" if current_runtime["torch_cuda"]["available"] else "cpu"
    elif device in {"cuda", "cpu"}:
        selected_device = device
    else:
        raise Phase3PipelineError("device must be auto, cuda, or cpu")
    provenance = capture_code_provenance(repo_root, config_path)
    workspace = prepare_phase3_workspace(output_dir, repo_root, overwrite=overwrite)
    reservation_path: Path | None = None
    try:
        staging = workspace.staging_dir
        protocol = staging / "protocol_config.snapshot.toml"
        atomic_write_bytes(protocol, config_path.read_bytes(), mode=0o600)
        private_source = staging / "source.snapshot.csv"
        private_snapshot = copy_validated_source(
            source_path,
            private_source,
            evidence.phase1.source_snapshot,
            expected_bytes=evidence.phase1.source_snapshot.bytes,
            expected_sha256=evidence.phase1.source_snapshot.sha256,
        )
        data = load_qm9_identities(private_source)
        failures: list[dict[str, object]] = []
        smoke_indices = np.concatenate(
            (evidence.phase1.split.train[:64], evidence.phase1.split.validation[:64])
        ).astype(np.int64)
        smoke_input = staging / ".smoke-input.jsonl"
        smoke_output = staging / ".smoke-output.npy"
        smoke_report_path = staging / ".smoke-worker.json"
        _write_worker_inputs(smoke_input, data, smoke_indices)
        smoke_report = None
        chosen_batch = None
        for candidate_batch in (128, 64, 32, 16, 8, 4, 1):
            if candidate_batch > initial_batch_size:
                continue
            smoke_output.unlink(missing_ok=True)
            smoke_report_path.unlink(missing_ok=True)
            try:
                progress(
                    f"starting train/validation-only MIST smoke: device={selected_device}, "
                    f"batch_size={candidate_batch}"
                )
                report = _invoke_worker(
                    runtime_python=runtime_python,
                    repo_root=repo_root,
                    snapshot=model_dir,
                    input_path=smoke_input,
                    output_path=smoke_output,
                    report_path=smoke_report_path,
                    batch_size=candidate_batch,
                    device=selected_device,
                    timeout_seconds=1800,
                    isolated_home=staging / ".smoke-home",
                    progress=progress,
                )
                _validate_worker_report(
                    report,
                    expected_rows=len(smoke_indices),
                    expected_indices=smoke_indices,
                    batch_size=candidate_batch,
                    device=selected_device,
                )
                smoke_prediction = np.load(smoke_output, allow_pickle=False)
                validate_prediction_matrix(smoke_prediction, expected_rows=len(smoke_indices))
                if report.get("output_npy_sha256") != sha256_file(smoke_output):
                    raise Phase3PipelineError("smoke output file hash differs")
                if report.get("output_canonical_sha256") != canonical_hash(
                    smoke_prediction.tolist()
                ):
                    raise Phase3PipelineError("smoke output semantic hash differs")
                smoke_report = report
                chosen_batch = candidate_batch
                break
            except Phase3PipelineError as error:
                message = str(error)
                failures.append(
                    {
                        "stage": "train-validation-smoke",
                        "batch_size": candidate_batch,
                        "device": selected_device,
                        "error": message,
                    }
                )
                if "out of memory" not in message.lower():
                    raise
        if smoke_report is None or chosen_batch is None:
            raise Phase3PipelineError("all operational smoke batch sizes failed")
        smoke = {
            "schema_version": "qm9-mist-smoke-v1",
            "scope": "64-train-plus-64-validation-rows-no-test",
            "test_rows_in_smoke": 0,
            "selected_device": selected_device,
            "initial_batch_size": initial_batch_size,
            "selected_batch_size": chosen_batch,
            "operational_failures": failures,
            "worker": smoke_report,
        }
        atomic_write_json(staging / "smoke.json", smoke, mode=0o600)
        smoke_input.unlink()
        smoke_output.unlink()
        smoke_report_path.unlink()

        inference_lock = freeze_inference(
            {
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "model_audit_sha256": audit_run["model_audit_sha256"],
                "phase3_audit_run_sha256": audit_run_sha,
                "runtime_freeze_canonical_sha256": current_runtime[
                    "complete_freeze_canonical_sha256"
                ],
                "phase1_run_sha256": evidence.phase1.phase1_run_sha256,
                "phase2_run_sha256": evidence.phase2_run_sha256,
                "source_sha256": evidence.phase1.source_snapshot.sha256,
                "test_ordered_index_sha256": evidence.phase1.split.ordered_hashes()["test"],
                "duplicate_clean_test_index_sha256": config["phase_1_observation"][
                    "duplicates"
                ]["duplicate_clean_retained_ordered_index_sha256"],
                "raw_smiles_sequence_sha256": data.raw_smiles_sha256,
                "protocol_config_snapshot_sha256": sha256_file(protocol),
                "code_provenance_aggregate_sha256": provenance["aggregate_sha256"],
                "device": selected_device,
                "batch_size": chosen_batch,
                "test_rows": len(evidence.phase1.split.test),
                "model_count": 1,
                "validation_selection_performed": False,
            }
        )
        assert_code_provenance_unchanged(provenance, repo_root, config_path)
        reservation_path = reserve_inference_once(
            cache / "mist-phase3" / "reservations", inference_lock
        )
        authorize_test_inference(reservation_path, inference_lock)
        test_input = staging / ".test-input.jsonl"
        test_output = staging / ".test-output.npy"
        worker_report_path = staging / "worker_report.json"
        test_rows = _write_worker_inputs(test_input, data, evidence.phase1.split.test)
        if test_rows != len(evidence.phase1.split.test):
            raise Phase3PipelineError("test worker input is incomplete")
        progress(
            f"MIST inference reserved as {inference_lock['inference_fingerprint']}; "
            f"starting {test_rows} candidate-test rows"
        )
        worker_report = _invoke_worker(
            runtime_python=runtime_python,
            repo_root=repo_root,
            snapshot=model_dir,
            input_path=test_input,
            output_path=test_output,
            report_path=worker_report_path,
            batch_size=chosen_batch,
            device=selected_device,
            timeout_seconds=int(
                config["resource_budget"]["max_wall_clock_hours_for_locked_test_evaluation"]
                * 3600
            ),
            isolated_home=staging / ".test-home",
            progress=progress,
        )
        _validate_worker_report(
            worker_report,
            expected_rows=test_rows,
            expected_indices=evidence.phase1.split.test,
            batch_size=chosen_batch,
            device=selected_device,
        )
        predictions = np.load(test_output, allow_pickle=False)
        validate_prediction_matrix(predictions, expected_rows=test_rows)
        if worker_report.get("output_npy_sha256") != sha256_file(test_output):
            raise Phase3PipelineError("test output file hash differs")
        if worker_report.get("output_canonical_sha256") != canonical_hash(
            predictions.tolist()
        ):
            raise Phase3PipelineError("test output semantic hash differs")
        atomic_write_json(worker_report_path, worker_report, mode=0o600)
        test_input.unlink()
        test_output.unlink()
        y_test = load_targets_for_indices(
            private_source, evidence.phase1.split.test, data
        )
        clean = {int(index) for index in evidence.phase1.duplicate_clean_test}
        clean_mask = np.asarray(
            [int(index) in clean for index in evidence.phase1.split.test], dtype=bool
        )
        metrics = {
            "schema_version": "qm9-mist-test-metrics-v1",
            "inference_fingerprint": inference_lock["inference_fingerprint"],
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "target_order": list(TARGET_COLUMNS),
            "manual_inverse_transform_applied": False,
            "full_test": native_metrics(y_test, predictions, evidence.train_target_scale),
            "duplicate_clean_test": native_metrics(
                y_test[clean_mask], predictions[clean_mask], evidence.train_target_scale
            ),
        }
        metrics_path = staging / "mist_metrics.json"
        atomic_write_json(metrics_path, metrics, mode=0o600)
        predictions_path = staging / "mist_predictions.jsonl"
        prediction_rows = atomic_write_jsonl(
            predictions_path,
            _prediction_rows(
                data,
                evidence.phase1.split.test,
                y_test,
                predictions,
                evidence.phase1.duplicate_clean_test,
            ),
            mode=0o600,
        )
        comparison = _comparison(metrics, evidence)
        comparison["mist_metrics_sha256"] = sha256_file(metrics_path)
        comparison_path = staging / "comparison.json"
        atomic_write_json(comparison_path, comparison, mode=0o600)
        audit_path = staging / "model_audit.json"
        runtime_path = staging / "runtime_environment.json"
        freeze_path = staging / "runtime_environment.freeze.txt"
        provenance_path = staging / "code_provenance.json"
        failure_path = staging / "failure_log.json"
        atomic_write_json(audit_path, current_audit, mode=0o600)
        atomic_write_json(runtime_path, current_runtime, mode=0o600)
        atomic_write_bytes(freeze_path, _freeze_bytes(freeze), mode=0o600)
        atomic_write_json(provenance_path, provenance, mode=0o600)
        atomic_write_json(
            failure_path,
            {
                "schema_version": "qm9-mist-phase3-failure-log-v1",
                "events": failures,
                "test_inference_retries": 0,
            },
            mode=0o600,
        )
        phase1_verify = staging / "phase1_verification.json"
        phase2_verify = staging / "phase2_verification.json"
        atomic_write_json(
            phase1_verify,
            {
                "phase1_run_sha256": evidence.phase1.phase1_run_sha256,
                "source_sha256": evidence.phase1.source_snapshot.sha256,
                "split_counts": evidence.phase1.split.counts(),
                "split_ordered_sha256": evidence.phase1.split.ordered_hashes(),
                "duplicate_clean_test_rows": len(evidence.phase1.duplicate_clean_test),
            },
            mode=0o600,
        )
        atomic_write_json(
            phase2_verify,
            {
                "phase2_run_sha256": evidence.phase2_run_sha256,
                "phase2_metrics_sha256": evidence.phase2_metrics_sha256,
                "phase2_predictions_sha256": evidence.phase2_predictions_sha256,
                "locked_selection_fingerprint": evidence.locked_selection_fingerprint,
            },
            mode=0o600,
        )
        assert_source_unchanged(
            private_source,
            private_snapshot,
            expected_bytes=evidence.phase1.source_snapshot.bytes,
            expected_sha256=evidence.phase1.source_snapshot.sha256,
        )
        assert_source_unchanged(
            source_path,
            evidence.phase1.source_snapshot,
            expected_bytes=evidence.phase1.source_snapshot.bytes,
            expected_sha256=evidence.phase1.source_snapshot.sha256,
        )
        assert_code_provenance_unchanged(provenance, repo_root, config_path)
        private_source.unlink()
        for private_dir in (staging / ".smoke-home", staging / ".test-home"):
            if private_dir.exists():
                import shutil

                shutil.rmtree(private_dir)
        artifact_paths = (
            audit_path,
            runtime_path,
            freeze_path,
            provenance_path,
            failure_path,
            protocol,
            staging / "smoke.json",
            worker_report_path,
            metrics_path,
            predictions_path,
            comparison_path,
            phase1_verify,
            phase2_verify,
        )
        artifacts = {path.name: _artifact(path) for path in artifact_paths}
        run = {
            "schema_version": "qm9-mist-phase3-run-v1",
            "scientific_status": (
                "released-mist-inference-complete-on-candidate-reconstructed-split"
            ),
            "official_checkpoint_test_reproduction_claimed": False,
            "labels_are_dft_computed_not_experimental": True,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "inference_fingerprint": inference_lock["inference_fingerprint"],
            "inference_reservation": str(reservation_path),
            "phase1_run_sha256": evidence.phase1.phase1_run_sha256,
            "phase2_run_sha256": evidence.phase2_run_sha256,
            "phase3_audit_run_sha256": audit_run_sha,
            "test_rows": test_rows,
            "duplicate_clean_test_rows": int(np.sum(clean_mask)),
            "prediction_rows": prediction_rows,
            "device": selected_device,
            "batch_size": chosen_batch,
            "worker_runtime_seconds": worker_report["runtime_seconds"],
            "worker_peak_rss_gib": worker_report["parent_peak_rss_gib"],
            "gpu_peak_memory_allocated_bytes": worker_report[
                "gpu_peak_memory_allocated_bytes"
            ],
            "gpu_peak_memory_reserved_bytes": worker_report[
                "gpu_peak_memory_reserved_bytes"
            ],
            "orchestrator_peak_rss_gib": _rss_gib(),
            "platform": platform.platform(),
            "runtime_seconds": time.monotonic() - started,
            "artifacts": artifacts,
        }
        run_path = staging / "phase3_run.json"
        atomic_write_json(run_path, run, mode=0o600)
        run_sha = sha256_file(run_path)
        atomic_write_bytes(
            staging / "phase3_run.sha256",
            f"{run_sha}  phase3_run.json\n".encode("ascii"),
            mode=0o600,
        )
        write_owner(staging)
        finalize_workspace(workspace)
        complete_inference(
            reservation_path,
            {
                "phase3_run_sha256": run_sha,
                "mist_metrics_sha256": artifacts["mist_metrics.json"]["sha256"],
                "mist_predictions_sha256": artifacts["mist_predictions.jsonl"]["sha256"],
                "comparison_sha256": artifacts["comparison.json"]["sha256"],
            },
        )
        run["phase3_run_sha256"] = run_sha
        run["output_dir"] = str(workspace.output_dir)
        return run
    finally:
        discard_workspace(workspace)
