from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from mist_transfer_benchmark.qm9.model_selection_v31_backends import (
    artifact_file_sha256,
    assess_runtime_feasibility,
    build_memory_worksheet,
    freeze_cache_choice,
    freeze_feasibility_artifact,
    profile_cache_options,
    profile_real_backend,
    write_real_backend_worker_spec,
)
from mist_transfer_benchmark.qm9.model_selection_v31_checkpoint import (
    CheckpointError,
    CompletionContract,
)
from mist_transfer_benchmark.qm9.model_selection_v31_observability import (
    ObservabilityError,
    TrialIdentity,
    validate_event_stream,
)
from mist_transfer_benchmark.qm9.model_selection_v31_supervisor import (
    ExternalWorkerSidecar,
    PlannedTrial,
    RunBinding,
    RunSupervisor,
    SidecarPolicy,
    SupervisorError,
    canonical_sha256,
    detect_device_identity,
    file_sha256,
)

TARGET_ORDER = (
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "u0",
    "u298",
    "h298",
    "g298",
    "cv",
)


def planned(run_id: str = "run-supervisor", count: int = 1) -> list[PlannedTrial]:
    return [
        PlannedTrial(
            identity=TrialIdentity(
                run_id=run_id,
                wave="W1",
                strategy="identity",
                candidate_id=f"ridge|all_raw|alpha-{index + 1}",
                family="ridge",
                fold=index,
                seed=7,
                trial_index=index + 1,
                trial_total=count,
            ),
            expected_shape=(3, 12),
            target_order=TARGET_ORDER,
            prediction_order_sha256=canonical_sha256([100 + index, 200 + index, 300 + index]),
            candidate_sha256=canonical_sha256({"alpha": index + 1}),
        )
        for index in range(count)
    ]


def _memory(*, approved: bool) -> dict[str, object]:
    capacity = 100 * 1024**3 if approved else 1024
    return build_memory_worksheet(
        rows=10,
        columns=4,
        hidden_dims=[4],
        batch_size=2,
        validation_chunk_size=2,
        cache_option="streaming",
        total_ram_bytes=100 * 1024**3,
        available_ram_bytes=capacity,
        total_vram_bytes=100 * 1024**3,
        free_vram_bytes=capacity,
        measured_rss_baseline_bytes=0,
        measured_rss_peak_bytes=0,
        measured_gpu_baseline_bytes=0,
        measured_gpu_peak_bytes=0,
        sparse_matrix_bytes=1,
        python_overhead_bytes=1,
        telemetry_bytes=1,
        framework_reservation_bytes=1,
    )


def binding(
    tmp_path: Path,
    trials: list[PlannedTrial],
    *,
    approved: bool = True,
    input_sha256: str = "b" * 64,
) -> RunBinding:
    plan_sha = canonical_sha256([row.payload() for row in trials])
    actual = {
        "plan_sha256": plan_sha,
        "config_sha256": "a" * 64,
        "input_sha256": input_sha256,
        "profile_sha256": "c" * 64,
        "code_sha256": "d" * 64,
    }
    runtime = assess_runtime_feasibility(
        {"ridge|0": [0.1, 0.2], "xgboost|0": [0.1, 0.2], "mlp|0": [0.1, 0.2]},
        {"ridge|0": 1, "xgboost|0": 1, "mlp|0": 1},
    )
    device = detect_device_identity("cpu")
    path = tmp_path / f"feasibility-{trials[0].identity.run_id}.json"
    artifact = freeze_feasibility_artifact(
        path,
        _memory(approved=approved),
        runtime,
        device_identity=device,
        actual_hashes=actual,
    )
    return RunBinding(
        run_id=trials[0].identity.run_id,
        **actual,
        feasibility_path=str(path),
        feasibility_file_sha256=artifact_file_sha256(path),
        memory_feasibility_sha256=artifact["memory_sha256"],
        runtime_feasibility_sha256=artifact["runtime_sha256"],
        device_identity=device,
        device_identity_sha256=artifact["device_identity_sha256"],
    )


def policy(**overrides: object) -> SidecarPolicy:
    values: dict[str, object] = {
        "heartbeat_seconds": 0.05,
        "heartbeat_stale_after_seconds": 0.1,
        "model_progress_stale_after_seconds": 0.1,
        "soft_deadline_seconds": 1.0,
        "hard_deadline_seconds": 2.0,
        "rss_max_bytes": 2 * 1024**3,
        "disk_free_min_bytes": 1,
        "gpu_allocated_max_bytes": 2 * 1024**3,
        "gpu_reserved_max_bytes": 2 * 1024**3,
        "gpu_temperature_max_c": 90.0,
        "require_rss_sensor": True,
        "require_disk_sensor": True,
        "require_gpu_sensors": False,
        "poll_seconds": 0.02,
    }
    values.update(overrides)
    return SidecarPolicy(**values)


def write_checkpoint(supervisor: RunSupervisor, specification: PlannedTrial) -> Path:
    trial = specification.identity
    contract = supervisor.completion_contract(trial)
    return supervisor.checkpoints.mark_trial_complete(
        trial,
        record={
            "status": "COMPLETE",
            "candidate_id": trial.candidate_id,
            "family": trial.family,
            "seed": trial.seed,
            "metadata_sha256": "e" * 64,
        },
        prediction=np.arange(36, dtype=np.float64).reshape(3, 12),
        completion=contract,
    )


def complete_one(supervisor: RunSupervisor, specification: PlannedTrial) -> None:
    supervisor.authenticate_completed_checkpoint(
        specification.identity, write_checkpoint(supervisor, specification)
    )


def wait_for_path_or_process_exit(
    path: Path, process: subprocess.Popen[str], *, timeout_seconds: float
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.is_file():
            return
        returncode = process.poll()
        if returncode is not None:
            stdout, stderr = process.communicate()
            pytest.fail(
                "sidecar exited before worker readiness: "
                f"returncode={returncode}, stdout={stdout!r}, stderr={stderr!r}"
            )
        time.sleep(0.02)
    pytest.fail(f"sidecar remained alive but worker readiness was absent after {timeout_seconds}s")


def test_startup_refuses_unapproved_or_tampered_feasibility(tmp_path: Path) -> None:
    trials = planned()
    refused = binding(tmp_path, trials, approved=False)
    with pytest.raises(Exception, match="not approved"):
        RunSupervisor(tmp_path / "refused", refused, trials)

    approved = binding(tmp_path, planned("approved"))
    Path(approved.feasibility_path).write_text("{}")
    with pytest.raises(Exception, match="hash differs"):
        RunSupervisor(tmp_path / "tampered", approved, planned("approved"))

    reserve_trials = planned("reserve-check")
    reserve_binding = binding(tmp_path, reserve_trials)
    supervisor = RunSupervisor(tmp_path / "reserve-run", reserve_binding, reserve_trials)
    Path(reserve_binding.feasibility_path).write_text("{}")
    with pytest.raises(Exception, match="hash differs"):
        supervisor.reserve(reserve_trials[0].identity)
    supervisor.close()


def test_stream_closure_is_idempotent_and_refuses_all_later_appends(tmp_path: Path) -> None:
    trials = planned("closed-streams")
    supervisor = RunSupervisor(tmp_path / "run", binding(tmp_path, trials), trials)
    supervisor.close_ledger()
    supervisor.close_ledger()
    assert supervisor.ledger.closed is True
    with pytest.raises(SupervisorError, match="ledger is closed"):
        supervisor.reserve(trials[0].identity)

    supervisor.close_events()
    supervisor.close_events()
    assert supervisor.events.closed is True
    with pytest.raises(ObservabilityError, match="event stream is closed"):
        supervisor.events.emit("heartbeat", trial=None, progress={})
    supervisor.close()


def test_completion_requires_checkpoint_and_occurs_exactly_once(tmp_path: Path) -> None:
    trials = planned()
    supervisor = RunSupervisor(tmp_path / "run", binding(tmp_path, trials), trials)
    trial = trials[0].identity
    supervisor.reserve(trial)
    supervisor.start_trial(trial)
    with pytest.raises(SupervisorError, match="direct completion is forbidden"):
        supervisor.complete_trial(trial, {"invented": "f" * 64})
    with pytest.raises(SupervisorError, match="authentication failed"):
        supervisor.authenticate_completed_checkpoint(trial, tmp_path / "missing.npz")
    complete_one(supervisor, trials[0])
    with pytest.raises(SupervisorError, match="START ledger"):
        supervisor.authenticate_completed_checkpoint(trial, tmp_path / "missing.npz")
    supervisor.finalize_success()
    supervisor.assert_ranking_allowed()
    supervisor.close()

    ledger = [
        json.loads(line) for line in (tmp_path / "run/ledger/trials.jsonl").read_text().splitlines()
    ]
    assert [row["event"] for row in ledger] == ["reserve", "start", "complete"]
    events = validate_event_stream(tmp_path / "run/telemetry/events.jsonl")
    assert [row["event"] for row in events].count("trial_complete") == 1


@pytest.mark.parametrize("mutation", ["delete", "tamper"])
def test_ranking_replays_checkpoint_after_finalization(tmp_path: Path, mutation: str) -> None:
    trials = planned(run_id=f"post-final-{mutation}")
    supervisor = RunSupervisor(tmp_path / "run", binding(tmp_path, trials), trials)
    trial = trials[0].identity
    supervisor.reserve(trial)
    supervisor.start_trial(trial)
    checkpoint = write_checkpoint(supervisor, trials[0])
    supervisor.authenticate_completed_checkpoint(trial, checkpoint)
    supervisor.finalize_success()
    supervisor.assert_ranking_allowed()

    if mutation == "delete":
        checkpoint.unlink()
    else:
        checkpoint.write_bytes(checkpoint.read_bytes() + b"post-final-tamper")

    with pytest.raises(SupervisorError, match="strict completed-checkpoint replay failed"):
        supervisor.assert_ranking_allowed()
    supervisor.close()


def test_mismatched_checkpoint_shape_and_contract_fail_closed(tmp_path: Path) -> None:
    trials = planned()
    supervisor = RunSupervisor(tmp_path / "run", binding(tmp_path, trials), trials)
    trial = trials[0].identity
    supervisor.reserve(trial)
    supervisor.start_trial(trial)
    with pytest.raises(CheckpointError, match="shape"):
        supervisor.checkpoints.mark_trial_complete(
            trial,
            record={
                "status": "COMPLETE",
                "candidate_id": trial.candidate_id,
                "family": trial.family,
                "seed": trial.seed,
                "metadata_sha256": "e" * 64,
            },
            prediction=np.zeros((2, 12)),
            completion=supervisor.completion_contract(trial),
        )
    expected = supervisor.completion_contract(trial)
    wrong_order = CompletionContract(
        expected_shape=expected.expected_shape,
        target_order=expected.target_order,
        prediction_order_sha256="f" * 64,
        start_ledger_entry_sha256=expected.start_ledger_entry_sha256,
    )
    wrong_path = supervisor.checkpoints.mark_trial_complete(
        trial,
        record={
            "status": "COMPLETE",
            "candidate_id": trial.candidate_id,
            "family": trial.family,
            "seed": trial.seed,
            "metadata_sha256": "e" * 64,
        },
        prediction=np.zeros((3, 12)),
        completion=wrong_order,
    )
    with pytest.raises(SupervisorError, match="authentication failed"):
        supervisor.authenticate_completed_checkpoint(trial, wrong_path)
    supervisor.close()


def test_atomic_backend_flow_rejects_fake_or_missing_evidence(tmp_path: Path) -> None:
    trials = planned()
    supervisor = RunSupervisor(tmp_path / "run", binding(tmp_path, trials), trials)

    def missing_runner(**kwargs: object) -> tuple[np.ndarray, dict[str, object]]:
        return np.zeros((3, 12)), {"backend": "fake"}

    with pytest.raises(SupervisorError, match="checkpoint evidence"):
        supervisor.run_backend_trial(trials[0], backend_runner=missing_runner, backend_kwargs={})
    assert (tmp_path / "run/failure/manifest.json").is_file()
    supervisor.close()


def test_atomic_backend_flow_authenticates_before_complete(tmp_path: Path) -> None:
    trials = planned()
    supervisor = RunSupervisor(tmp_path / "run", binding(tmp_path, trials), trials)

    def runner(**kwargs: object) -> tuple[np.ndarray, dict[str, object]]:
        store = kwargs["checkpoint_store"]
        trial = kwargs["trial"]
        completion = kwargs["completion_contract"]
        prediction = np.arange(36, dtype=np.float64).reshape(3, 12)
        path = store.mark_trial_complete(
            trial,
            record={
                "status": "COMPLETE",
                "candidate_id": trial.candidate_id,
                "family": trial.family,
                "seed": trial.seed,
                "metadata_sha256": "e" * 64,
            },
            prediction=prediction,
            completion=completion,
        )
        evidence = store.validate_completed_checkpoint(path, trial=trial, completion=completion)
        return prediction, {"checkpoint_evidence": evidence}

    supervisor.run_backend_trial(trials[0], backend_runner=runner, backend_kwargs={})
    supervisor.finalize_success()
    supervisor.close()


def test_multi_trial_sidecar_does_not_finalize_first_worker(tmp_path: Path) -> None:
    trials = planned(count=2)
    supervisor = RunSupervisor(tmp_path / "run", binding(tmp_path, trials), trials)
    sidecar = ExternalWorkerSidecar(supervisor, policy())
    command = [sys.executable, "-c", "pass"]
    for index, specification in enumerate(trials):
        code = sidecar.run(
            specification.identity,
            command,
            heartbeat_path=tmp_path / f"run/telemetry/worker-{index}.heartbeat",
        )
        assert code == 0
        assert not (tmp_path / "run/final").exists()
        complete_one(supervisor, specification)
    supervisor.finalize_success()
    supervisor.close()
    assert (tmp_path / "run/final/manifest.json").is_file()


def test_long_blocking_stage_survives_with_fresh_independent_heartbeat(
    tmp_path: Path,
) -> None:
    trials = planned()
    run = tmp_path / "run"
    supervisor = RunSupervisor(run, binding(tmp_path, trials), trials)
    heartbeat_path = run / "telemetry/native-fit.heartbeat"
    worker = """
import json
import os
import time
from pathlib import Path

heartbeat = Path(os.environ["QM9_V31_HEARTBEAT_PATH"])
deadline = time.monotonic() + 0.45
while time.monotonic() < deadline:
    temporary = heartbeat.with_suffix(".tmp")
    temporary.write_text(json.dumps({"stage": "native_fit", "progress_sequence": 1}))
    os.replace(temporary, heartbeat)
    time.sleep(0.03)
"""
    configured = policy(
        heartbeat_seconds=0.1,
        heartbeat_stale_after_seconds=0.2,
        model_progress_stale_after_seconds=0.15,
        model_progress_stale_overrides={"ridge:native_fit": 0.8},
        soft_deadline_seconds=1.2,
        hard_deadline_seconds=2.0,
    )
    code = ExternalWorkerSidecar(supervisor, configured).run(
        trials[0].identity,
        [sys.executable, "-c", worker],
        heartbeat_path=heartbeat_path,
    )
    supervisor.close()

    assert code == 0
    assert not (run / "failure").exists()
    heartbeats = [
        row
        for row in validate_event_stream(run / "telemetry/events.jsonl")
        if row["event"] == "heartbeat"
    ]
    assert any(
        row["progress"]["model_progress_age_seconds"] > 0.15 for row in heartbeats
    )
    assert all(
        row["progress"]["model_progress_stale_limit_seconds"] == 0.8
        for row in heartbeats
        if row["progress"]["model_progress_stage"] == "native_fit"
    )
    assert all(
        row["progress"]["worker_heartbeat_age_seconds"] < 0.2
        for row in heartbeats[1:]
    )


def test_fresh_heartbeat_does_not_hide_stale_model_progress(tmp_path: Path) -> None:
    trials = planned()
    run = tmp_path / "run"
    supervisor = RunSupervisor(run, binding(tmp_path, trials), trials)
    worker = """
import json
import os
import time
from pathlib import Path

heartbeat = Path(os.environ["QM9_V31_HEARTBEAT_PATH"])
deadline = time.monotonic() + 2.0
while time.monotonic() < deadline:
    temporary = heartbeat.with_suffix(".tmp")
    temporary.write_text(json.dumps({"stage": "native_fit", "progress_sequence": 1}))
    os.replace(temporary, heartbeat)
    time.sleep(0.03)
"""
    code = ExternalWorkerSidecar(
        supervisor,
        policy(
            heartbeat_seconds=0.1,
            heartbeat_stale_after_seconds=0.2,
            model_progress_stale_after_seconds=0.25,
            soft_deadline_seconds=1.2,
            hard_deadline_seconds=2.0,
        ),
    ).run(
        trials[0].identity,
        [sys.executable, "-c", worker],
        heartbeat_path=run / "telemetry/stale-progress.heartbeat",
    )
    supervisor.close()

    assert code != 0
    failure = json.loads((run / "failure/manifest.json").read_text())
    assert failure["evidence"]["reason"] == "STALE_MODEL_PROGRESS"


@pytest.mark.parametrize(
    ("policy_overrides", "worker", "reason"),
    [
        (
            {"rss_max_bytes": 8 * 1024**2},
            "x=bytearray(64*1024*1024); import time; time.sleep(5)",
            "RSS_LIMIT",
        ),
        (
            {"disk_free_min_bytes": 10**18},
            "import time; time.sleep(5)",
            "DISK_FLOOR",
        ),
        (
            {"require_gpu_sensors": True, "gpu_device_uuid": "GPU-NOT-PRESENT"},
            "import time; time.sleep(5)",
            "GPU_SENSOR_UNAVAILABLE",
        ),
    ],
)
def test_external_resource_gates_terminate_real_child(
    tmp_path: Path,
    policy_overrides: dict[str, object],
    worker: str,
    reason: str,
) -> None:
    trials = planned()
    supervisor = RunSupervisor(tmp_path / "run", binding(tmp_path, trials), trials)
    sidecar = ExternalWorkerSidecar(supervisor, policy(**policy_overrides))
    code = sidecar.run(
        trials[0].identity,
        [sys.executable, "-c", worker],
        heartbeat_path=tmp_path / "run/telemetry/heartbeat",
    )
    supervisor.close()
    assert code != 0
    failure = json.loads((tmp_path / "run/failure/manifest.json").read_text())
    assert failure["evidence"]["reason"] == reason
    assert failure["ranking_allowed"] is False


def test_external_signal_and_absolute_time_kill_are_recorded(tmp_path: Path) -> None:
    trials = planned()
    run = tmp_path / "run"
    bound = binding(tmp_path, trials)
    spec = {
        "output": str(run),
        "binding": asdict(bound),
        "planned_trial": trials[0].payload(),
        "policy": asdict(
            policy(
                heartbeat_seconds=0.25,
                heartbeat_stale_after_seconds=0.5,
                model_progress_stale_after_seconds=0.5,
                soft_deadline_seconds=1.0,
                hard_deadline_seconds=1.5,
                rss_max_bytes=2 * 1024**3,
            )
        ),
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    worker = (
        "import os,pathlib,signal,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "pathlib.Path(os.environ['QM9_V31_HEARTBEAT_PATH']).write_text('ready'); "
        "time.sleep(30)"
    )
    sidecar: subprocess.Popen[str] = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mist_transfer_benchmark.qm9.model_selection_v31_supervisor",
            "--spec",
            str(spec_path),
            "--",
            sys.executable,
            "-c",
            worker,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    heartbeat = run / "telemetry/worker.heartbeat"
    try:
        wait_for_path_or_process_exit(heartbeat, sidecar, timeout_seconds=10)
        sidecar.send_signal(signal.SIGTERM)
        assert sidecar.wait(timeout=5) != 0
        failure = json.loads((run / "failure/manifest.json").read_text())
        assert failure["evidence"]["reason"] == "ABSOLUTE_HARD_DEADLINE"
        events = validate_event_stream(run / "telemetry/events.jsonl")
        assert any(
            row["event"] == "signal_received"
            and row["failure"]["reason"] == "EXTERNAL_SIGNAL_SIGTERM"
            for row in events
        )
    finally:
        if sidecar.poll() is None:
            sidecar.send_signal(signal.SIGTERM)
            try:
                sidecar.wait(timeout=3)
            except subprocess.TimeoutExpired:
                sidecar.kill()
                sidecar.wait(timeout=2)
        sidecar.communicate()


def test_run_start_binds_feasibility_and_device(tmp_path: Path) -> None:
    trials = planned()
    bound = binding(tmp_path, trials)
    supervisor = RunSupervisor(tmp_path / "run", bound, trials)
    start = json.loads((tmp_path / "run/run_start.json").read_text())
    supervisor.close()
    assert start["binding"]["feasibility_file_sha256"] == file_sha256(bound.feasibility_path)
    assert start["feasibility"]["device_identity"] == detect_device_identity("cpu")


def test_all_real_families_complete_once_through_supervisor(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("xgboost")
    rng = np.random.default_rng(20260719)
    values = rng.normal(size=(24, 6))
    values[np.abs(values) < 0.6] = 0
    matrix = sparse.csr_matrix(values)
    targets = rng.normal(size=(24, 12))
    candidates = [
        {"id": "ridge|tiny", "family": "ridge", "params": {"alpha": 1.0}},
        {
            "id": "xgboost|tiny",
            "family": "xgboost",
            "params": {
                "max_depth": 2,
                "learning_rate": 0.1,
                "n_estimators": 2,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "min_child_weight": 1.0,
            },
            "early_stopping_rounds": 1,
        },
        {
            "id": "mlp|tiny",
            "family": "mlp",
            "params": {
                "hidden_dims": [8],
                "dropout": 0.0,
                "learning_rate": 0.001,
                "weight_decay": 0.00001,
            },
            "training": {
                "batch_size": 18,
                "max_epochs": 1,
                "patience": 1,
                "min_delta": 0.0,
            },
            "monitoring": {},
        },
    ]
    run_id = "all-family-supervisor"
    specifications = [
        PlannedTrial(
            identity=TrialIdentity(
                run_id=run_id,
                wave="W1",
                strategy="identity",
                candidate_id=str(candidate["id"]),
                family=str(candidate["family"]),
                fold=0,
                seed=7,
                trial_index=index,
                trial_total=3,
            ),
            expected_shape=(6, 12),
            target_order=TARGET_ORDER,
            prediction_order_sha256=canonical_sha256(list(range(18, 24))),
            candidate_sha256=canonical_sha256(candidate),
        )
        for index, candidate in enumerate(candidates, start=1)
    ]
    supervisor = RunSupervisor(tmp_path / "run", binding(tmp_path, specifications), specifications)
    cache_identity = {
        "feature_artifact_sha256": "1" * 64,
        "recipe_provenance_sha256": "2" * 64,
        "strategy": "identity",
        "fold": 0,
        "train_ordered_sha256": "3" * 64,
        "tune_ordered_sha256": "4" * 64,
    }
    cache_directory = tmp_path / "run/cache"
    profiles = profile_cache_options(
        matrix[:18],
        matrix[18:],
        batch_indices=np.asarray([0, 1, 2, 3]),
        identity=cache_identity,
        device=torch.device("cpu"),
        directory=cache_directory,
        samples=3,
        epochs_cap=1,
    )
    choice_path = tmp_path / "run/cache-choice.json"
    freeze_cache_choice(
        choice_path,
        profiles,
        total_vram_bytes=0,
        free_vram_bytes=0,
        total_ram_bytes=32 * 1024**3,
        available_ram_bytes=20 * 1024**3,
        free_disk_bytes=100 * 1024**3,
        cache_bytes=matrix.shape[0] * matrix.shape[1] * 4,
        device_identity={"type": "cpu", "name": "supervisor-test"},
    )
    for specification, candidate in zip(specifications, candidates, strict=True):
        prediction, _ = supervisor.run_backend_trial(
            specification,
            backend_runner=profile_real_backend,
            backend_kwargs={
                "candidate": candidate,
                "x_train": matrix[:18],
                "y_train": targets[:18],
                "x_tune": matrix[18:],
                "y_tune": targets[18:],
                "seed": 7,
                "guard": None,
                "cache_identity": cache_identity,
                "cache_directory": cache_directory,
                "cache_choice_path": choice_path,
                "device": "cpu",
            },
        )
        assert prediction.shape == (6, 12)
        assert not (tmp_path / "run/final").exists()
    supervisor.finalize_success()
    supervisor.assert_ranking_allowed()
    supervisor.close()
    events = validate_event_stream(tmp_path / "run/telemetry/events.jsonl")
    assert [row["event"] for row in events].count("trial_complete") == 3
    assert [row["event"] for row in events].count("checkpoint_saved") == 3


def test_real_ridge_worker_sidecar_authenticates_before_completion(tmp_path: Path) -> None:
    trials = planned("real-worker")
    run = tmp_path / "run"
    data_path = tmp_path / "public-fixture.npz"
    rng = np.random.default_rng(7)
    np.savez_compressed(
        data_path,
        x_train=rng.normal(size=(12, 6)),
        y_train=rng.normal(size=(12, 12)),
        x_tune=rng.normal(size=(3, 6)),
        y_tune=rng.normal(size=(3, 12)),
    )
    bound = binding(tmp_path, trials, input_sha256=artifact_file_sha256(data_path))
    supervisor = RunSupervisor(run, bound, trials)
    trial = trials[0].identity
    supervisor.reserve(trial)
    supervisor.start_trial(trial)
    result_path = run / "worker-results/ridge.json"
    spec_path = run / "worker-specs/ridge.json"
    write_real_backend_worker_spec(
        spec_path,
        run_root=run,
        data_path=data_path,
        data_scope="public-fixture",
        result_path=result_path,
        checkpoint_root=supervisor.checkpoints.root,
        candidate={"id": trial.candidate_id, "family": "ridge", "params": {"alpha": 1.0}},
        trial=asdict(trial),
        completion=supervisor.completion_contract(trial),
        checkpoint_bindings=supervisor.checkpoints.bindings,
        device="cpu",
    )
    code = ExternalWorkerSidecar(
        supervisor,
        policy(
            heartbeat_seconds=2.5,
            heartbeat_stale_after_seconds=5.0,
            model_progress_stale_after_seconds=5.0,
            soft_deadline_seconds=8.0,
            hard_deadline_seconds=12.0,
        ),
    ).run(
        trial,
        [
            sys.executable,
            "-m",
            "mist_transfer_benchmark.qm9.model_selection_v31_backends",
            "--worker-spec",
            str(spec_path),
        ],
        heartbeat_path=run / "telemetry/real-worker.heartbeat",
        prepared=True,
        result_path=result_path,
    )
    assert code == 0
    assert supervisor.ledger.state(trial) == "complete"
    supervisor.finalize_success()
    supervisor.close()
