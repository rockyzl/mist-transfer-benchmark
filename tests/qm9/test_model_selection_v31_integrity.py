from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from mist_transfer_benchmark.qm9 import model_selection_v31_backends as backend_module
from mist_transfer_benchmark.qm9 import model_selection_v31_checkpoint as checkpoint_module
from mist_transfer_benchmark.qm9.model_selection_v31_backends import (
    BackendError,
    artifact_file_sha256,
    backend_hook_contract,
    build_feature_cache,
    freeze_cache_choice,
    load_real_backend_profile,
    load_real_backend_worker_spec,
    profile_cache_options,
    profile_real_backend,
    run_real_backend_worker,
    write_real_backend_worker_spec,
)
from mist_transfer_benchmark.qm9.model_selection_v31_checkpoint import (
    CheckpointError,
    CheckpointStore,
    CompletionContract,
    ReplayContract,
    trial_identity_sha256,
)
from mist_transfer_benchmark.qm9.model_selection_v31_observability import (
    EventSink,
    TrialIdentity,
    validate_event_stream,
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


class RecordingGuard:
    def __init__(self) -> None:
        self.safe_points: list[str] = []

    def check(self, safe_point: str) -> None:
        self.safe_points.append(safe_point)


def _store(path: Path) -> CheckpointStore:
    return CheckpointStore(
        path,
        plan_sha256="a" * 64,
        config_sha256="b" * 64,
        input_sha256="c" * 64,
        cache_sha256="d" * 64,
        run_start_sha256="e" * 64,
    )


def _trial(
    family: str = "ridge", index: int = 1, total: int = 1
) -> dict[str, object]:
    return {
        "run_id": "integrity",
        "wave": "W1",
        "strategy": "identity",
        "candidate_id": f"{family}|tiny",
        "family": family,
        "fold": 0,
        "seed": 7,
        "trial_index": index,
        "trial_total": total,
    }


def _completion(rows: int, ledger_digit: str = "9") -> CompletionContract:
    return CompletionContract(
        expected_shape=(rows, 12),
        target_order=TARGET_ORDER,
        prediction_order_sha256="f" * 64,
        start_ledger_entry_sha256=ledger_digit * 64,
    )


def _identity() -> dict[str, object]:
    return {
        "feature_artifact_sha256": "1" * 64,
        "recipe_provenance_sha256": "2" * 64,
        "strategy": "identity",
        "fold": 0,
        "train_ordered_sha256": "3" * 64,
        "tune_ordered_sha256": "4" * 64,
    }


def _data(rows: int = 24, columns: int = 6) -> tuple[sparse.csr_matrix, np.ndarray]:
    rng = np.random.default_rng(20260719)
    dense = rng.normal(size=(rows, columns))
    dense[np.abs(dense) < 0.6] = 0.0
    return sparse.csr_matrix(dense), rng.normal(size=(rows, 12))


def _complete_one(
    store: CheckpointStore, trial: dict[str, object], rows: int = 2
) -> tuple[Path, ReplayContract]:
    prediction = np.arange(rows * 12, dtype=np.float64).reshape(rows, 12)
    path = store.mark_trial_complete(
        trial,
        record={
            "status": "COMPLETE",
            "candidate_id": trial["candidate_id"],
            "family": trial["family"],
            "seed": trial["seed"],
            "metadata_sha256": "8" * 64,
        },
        prediction=prediction,
        completion=_completion(rows),
    )
    key = trial_identity_sha256(trial)
    replay = ReplayContract(
        planned_trials=(trial,),
        completed_entries={
            key: {
                "status": "COMPLETE",
                "complete_ledger_entry_sha256": "7" * 64,
                "start_ledger_entry_sha256": "9" * 64,
                "checkpoint": store.validate_completed_checkpoint(
                    path, trial=trial, completion=_completion(rows)
                ),
            }
        },
        ledger_sha256="e" * 64,
        terminal_status="COMPLETE",
    )
    return path, replay


def test_checkpoint_semantic_tamper_and_completion_schema_are_rejected(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "store")
    trial = _trial()
    state_path = store.save_tensor_state(
        trial,
        state_kind="best",
        state={"weight": np.arange(12, dtype=np.float32).reshape(3, 4)},
        payload={"epoch": 1},
    )
    with np.load(state_path, allow_pickle=False) as payload:
        arrays = {
            key: np.asarray(payload[key]).copy()
            for key in payload.files
            if key != "__manifest_json__"
        }
        manifest = json.loads(payload["__manifest_json__"].tobytes().decode())
    manifest["payload"]["epoch"] = 99
    with state_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            **arrays,
            __manifest_json__=np.frombuffer(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(),
                dtype=np.uint8,
            ),
        )
    with pytest.raises(CheckpointError, match="semantic hash differs"):
        store.load_tensor_state(state_path)

    with pytest.raises(CheckpointError, match="schema/status"):
        store.mark_trial_complete(
            trial,
            record={
                "status": "complete",
                "candidate_id": trial["candidate_id"],
                "family": trial["family"],
                "seed": trial["seed"],
                "metadata_sha256": "8" * 64,
            },
            prediction=np.zeros((2, 12)),
            completion=_completion(2),
        )
    with pytest.raises(CheckpointError, match="shape differs"):
        store.mark_trial_complete(
            trial,
            record={
                "status": "COMPLETE",
                "candidate_id": trial["candidate_id"],
                "family": trial["family"],
                "seed": trial["seed"],
                "metadata_sha256": "8" * 64,
            },
            prediction=np.zeros((2, 1)),
            completion=_completion(2),
        )


def test_completed_replay_rejects_duplicate_missing_nonterminal_and_tamper(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "store")
    trial = _trial()
    path, replay = _complete_one(store, trial)
    with pytest.raises(CheckpointError, match="terminal status COMPLETE"):
        ReplayContract(
            planned_trials=(trial,),
            completed_entries=replay.completed_entries,
            ledger_sha256="e" * 64,
            terminal_status="FAILED",
        )
    duplicate = ReplayContract(
        planned_trials=(trial, trial),
        completed_entries=replay.completed_entries,
        ledger_sha256="e" * 64,
        terminal_status="COMPLETE",
    )
    with pytest.raises(CheckpointError, match="duplicate trial"):
        store.load_completed("a" * 64, replay=duplicate)
    missing = ReplayContract(
        planned_trials=(trial,),
        completed_entries={},
        ledger_sha256="e" * 64,
        terminal_status="COMPLETE",
    )
    with pytest.raises(CheckpointError, match="missing or extra"):
        store.load_completed("a" * 64, replay=missing)

    with np.load(path, allow_pickle=False) as payload:
        prediction = np.asarray(payload["prediction"]).copy()
        manifest = json.loads(payload["__manifest_json__"].tobytes().decode())
    manifest["payload"]["record"]["status"] = "FAILED"
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            prediction=prediction,
            __manifest_json__=np.frombuffer(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(),
                dtype=np.uint8,
            ),
        )
    with pytest.raises(CheckpointError, match="semantic hash differs"):
        store.load_completed("a" * 64, replay=replay)


def test_atomic_checkpoint_and_cache_publication_fsync_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_events: list[str] = []
    original_replace = checkpoint_module.os.replace
    original_checkpoint_fsync = checkpoint_module._fsync_directory

    def recording_replace(source: str, destination: str) -> None:
        checkpoint_events.append("replace")
        original_replace(source, destination)

    def recording_checkpoint_fsync(path: Path) -> None:
        checkpoint_events.append("directory-fsync")
        original_checkpoint_fsync(path)

    monkeypatch.setattr(checkpoint_module.os, "replace", recording_replace)
    monkeypatch.setattr(checkpoint_module, "_fsync_directory", recording_checkpoint_fsync)
    _store(tmp_path / "checkpoint-root").save_trial_state(
        _trial(), state_kind="smoke", payload={"epoch": 0}, blobs={"state": b"ok"}
    )
    assert checkpoint_events.index("replace") < len(checkpoint_events) - 1
    assert checkpoint_events[-1] == "directory-fsync"

    cache_events: list[str] = []
    original_link = backend_module.os.link
    original_cache_fsync = backend_module._fsync_directory

    def recording_link(source: str, destination: str) -> None:
        cache_events.append("link")
        original_link(source, destination)

    def recording_cache_fsync(path: Path) -> None:
        cache_events.append("directory-fsync")
        original_cache_fsync(path)

    monkeypatch.setattr(backend_module.os, "link", recording_link)
    monkeypatch.setattr(backend_module, "_fsync_directory", recording_cache_fsync)
    matrix, _ = _data(rows=8)
    build_feature_cache(
        matrix[:6],
        matrix[6:],
        option="streaming",
        identity=_identity(),
        device=pytest.importorskip("torch").device("cpu"),
        directory=tmp_path / "cache-root",
    ).close()
    assert cache_events.index("link") < len(cache_events) - 1
    assert cache_events[-1] == "directory-fsync"


def test_cache_identity_is_mandatory_and_rejects_zero_digest(tmp_path: Path) -> None:
    matrix, _ = _data(rows=8)
    with pytest.raises(BackendError, match="invalid"):
        build_feature_cache(
            matrix[:6],
            matrix[6:],
            option="streaming",
            identity={**_identity(), "train_ordered_sha256": "0" * 64},
            device=pytest.importorskip("torch").device("cpu"),
            directory=tmp_path,
        )


def test_all_real_families_use_governed_hooks_and_frozen_cache_choice(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("xgboost")
    matrix, targets = _data(rows=24, columns=6)
    identity = _identity()
    cache_directory = tmp_path / "cache"
    profiles = profile_cache_options(
        matrix[:18],
        matrix[18:],
        batch_indices=np.asarray([0, 1, 2, 3]),
        identity=identity,
        device=torch.device("cpu"),
        directory=cache_directory,
        samples=3,
        epochs_cap=1,
    )
    choice_path = tmp_path / "cache-choice.json"
    freeze_cache_choice(
        choice_path,
        profiles,
        total_vram_bytes=0,
        free_vram_bytes=0,
        total_ram_bytes=32 * 1024**3,
        available_ram_bytes=20 * 1024**3,
        free_disk_bytes=100 * 1024**3,
        cache_bytes=matrix.shape[0] * matrix.shape[1] * 4,
        device_identity={"type": "cpu", "name": "governed-smoke"},
    )
    candidates = [
        {
            "id": "ridge|tiny",
            "family": "ridge",
            "params": {"alpha": 1.0},
        },
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
    store = _store(tmp_path / "checkpoints")
    guard = RecordingGuard()
    event_path = tmp_path / "events.jsonl"
    trials: list[TrialIdentity] = []
    with EventSink(event_path, "all-family") as sink:
        for index, candidate in enumerate(candidates, start=1):
            trial = TrialIdentity(
                run_id="all-family",
                wave="W1",
                strategy="identity",
                candidate_id=str(candidate["id"]),
                family=str(candidate["family"]),
                fold=0,
                seed=7,
                trial_index=index,
                trial_total=len(candidates),
            )
            trials.append(trial)
            prediction, _ = profile_real_backend(
                candidate,
                matrix[:18],
                targets[:18],
                matrix[18:],
                targets[18:],
                seed=7,
                sink=sink,
                guard=guard,
                checkpoint_store=store,
                trial=trial,
                cache_identity=identity,
                cache_directory=cache_directory,
                cache_choice_path=choice_path,
                completion_contract=_completion(6, str(index)),
                device="cpu",
            )
            assert prediction.shape == (6, 12)
            assert store.latest_safe(trial)["kind"] == "completed"
    events = validate_event_stream(event_path)
    assert not [row for row in events if row["event"] == "trial_complete"]
    assert len([row for row in events if row["event"] == "checkpoint_saved"]) == 3
    assert {row["trial"]["family"] for row in events if row["trial"] is not None} == {
        "ridge",
        "xgboost",
        "mlp",
    }
    assert any(point == "ridge-before-fit" for point in guard.safe_points)
    assert any(point.startswith("xgboost-target-") for point in guard.safe_points)
    assert any(point.startswith("mlp-epoch-") for point in guard.safe_points)
    assert set(backend_hook_contract()["families"]) == {"ridge", "xgboost", "mlp"}


def test_bounded_real_worker_publishes_only_authenticated_checkpoint_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    matrix, targets = _data(rows=24, columns=6)
    data_path = run_root / "public-synthetic.npz"
    with data_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            x_train=matrix[:18].toarray(),
            y_train=targets[:18],
            x_tune=matrix[18:].toarray(),
            y_tune=targets[18:],
        )
    trial = _trial("ridge")
    candidate = {"id": "ridge|tiny", "family": "ridge", "params": {"alpha": 1.0}}
    spec_path = run_root / "worker-spec.json"
    write_real_backend_worker_spec(
        spec_path,
        run_root=run_root,
        data_path=data_path,
        data_scope="synthetic",
        result_path="worker-results/ridge.json",
        checkpoint_root="checkpoints/ridge",
        candidate=candidate,
        trial=trial,
        completion=_completion(6),
        checkpoint_bindings={
            "plan_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "input_sha256": artifact_file_sha256(data_path),
            "cache_sha256": "d" * 64,
            "run_start_sha256": "e" * 64,
        },
    )
    monkeypatch.setenv(
        "QM9_V31_HEARTBEAT_PATH", str(run_root / "telemetry/worker.heartbeat")
    )
    result = run_real_backend_worker(spec_path)
    assert set(result) == {"status", "checkpoint_evidence", "metadata_sha256"}
    assert result["status"] == "COMPLETE"
    assert (run_root / "telemetry/worker.heartbeat").is_file()
    result_path = run_root / "worker-results/ridge.json"
    assert json.loads(result_path.read_text()) == result
    profile_path = run_root / "worker-results/ridge.profile.json"
    profile = load_real_backend_profile(
        profile_path, expected_metadata_sha256=result["metadata_sha256"]
    )
    assert profile["metadata"]["stage_seconds"]
    assert profile["scientific_result"] is False

    store = CheckpointStore(
        run_root / "checkpoints/ridge",
        plan_sha256="a" * 64,
        config_sha256="b" * 64,
        input_sha256=artifact_file_sha256(data_path),
        cache_sha256="d" * 64,
        run_start_sha256="e" * 64,
    )
    evidence = result["checkpoint_evidence"]
    assert (
        store.validate_completed_checkpoint(
            store.root / evidence["relative_path"], trial=trial, completion=_completion(6)
        )
        == evidence
    )
    with pytest.raises(CheckpointError, match="completion contract differs"):
        store.validate_completed_checkpoint(
            store.root / evidence["relative_path"],
            trial=trial,
            completion=_completion(6, "6"),
        )

    second_trial = {
        **_trial("ridge", index=2, total=2),
        "candidate_id": "ridge|tiny-second",
    }
    second_spec_path = run_root / "worker-spec-second.json"
    write_real_backend_worker_spec(
        second_spec_path,
        run_root=run_root,
        data_path=data_path,
        data_scope="synthetic",
        result_path="worker-results/ridge-second.json",
        checkpoint_root="checkpoints/ridge",
        candidate={
            "id": "ridge|tiny-second",
            "family": "ridge",
            "params": {"alpha": 10.0},
        },
        trial=second_trial,
        completion=_completion(6, "6"),
        checkpoint_bindings={
            "plan_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "input_sha256": artifact_file_sha256(data_path),
            "cache_sha256": "d" * 64,
            "run_start_sha256": "e" * 64,
        },
    )
    assert load_real_backend_worker_spec(second_spec_path)["trial"] == second_trial
