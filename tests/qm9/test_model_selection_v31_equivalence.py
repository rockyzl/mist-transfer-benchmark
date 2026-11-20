from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from sklearn.linear_model import Ridge

from mist_transfer_benchmark.qm9.model_selection_v3 import (
    fit_real_torch_mlp,
    fit_real_xgboost,
)
from mist_transfer_benchmark.qm9.model_selection_v31_backends import (
    BackendError,
    artifact_file_sha256,
    assess_runtime_feasibility,
    build_feature_cache,
    build_memory_worksheet,
    cache_identity_sha256,
    choose_cache_option,
    estimate_memory_model,
    fit_ridge_profiled,
    fit_torch_mlp_profiled,
    fit_xgboost_profiled,
    freeze_cache_choice,
    freeze_feasibility_artifact,
    load_cache_choice,
    load_feasibility_artifact,
    profile_cache_options,
    profile_real_backend,
)
from mist_transfer_benchmark.qm9.model_selection_v31_checkpoint import (
    CHECKPOINT_SCHEMA,
    TENSOR_STATE_HASH_SCHEMA,
    CheckpointError,
    CheckpointStore,
    CompletionContract,
    ReplayContract,
    canonical_tensor_state_sha256,
    trial_identity_sha256,
)
from mist_transfer_benchmark.qm9.model_selection_v31_observability import (
    EventSink,
    TrialIdentity,
    validate_event_stream,
)


def _trial(candidate: str = "mlp|all_raw|tiny") -> dict[str, object]:
    return {
        "run_id": "synthetic",
        "wave": "W1",
        "strategy": "identity",
        "candidate_id": candidate,
        "family": candidate.split("|")[0],
        "fold": 0,
        "seed": 7,
        "trial_index": 1,
        "trial_total": 1,
    }


def _store(path: Path, prefix: str = "") -> CheckpointStore:
    return CheckpointStore(
        path,
        plan_sha256=(prefix + "a" * 64)[-64:],
        config_sha256="b" * 64,
        input_sha256="c" * 64,
        cache_sha256="d" * 64,
        run_start_sha256="e" * 64,
    )


def _completion(rows: int) -> CompletionContract:
    return CompletionContract(
        expected_shape=(rows, 12),
        target_order=(
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
        ),
        prediction_order_sha256="f" * 64,
        start_ledger_entry_sha256="9" * 64,
    )


def _identity(fold: int = 0, recipe: str = "1") -> dict[str, object]:
    return {
        "feature_artifact_sha256": "a" * 64,
        "recipe_provenance_sha256": recipe * 64,
        "strategy": "identity",
        "fold": fold,
        "train_ordered_sha256": "c" * 64,
        "tune_ordered_sha256": "d" * 64,
    }


def _data(rows: int = 36, columns: int = 10) -> tuple[sparse.csr_matrix, np.ndarray]:
    rng = np.random.default_rng(20260719)
    matrix = rng.normal(size=(rows, columns))
    matrix[np.abs(matrix) < 0.7] = 0
    targets = rng.normal(size=(rows, 12))
    return sparse.csr_matrix(matrix), targets


def _candidate() -> dict[str, object]:
    return {
        "hidden_dims": [16],
        "dropout": 0.05,
        "learning_rate": 0.001,
        "weight_decay": 0.00001,
    }


def _training(*, epochs: int = 3, batch_size: int = 24) -> dict[str, object]:
    return {"max_epochs": epochs, "patience": epochs, "min_delta": 0.0, "batch_size": batch_size}


class RecordingSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def emit(self, event: str, **kwargs: object) -> dict[str, object]:
        self.events.append((event, kwargs))
        return {"event": event}


class RecordingGuard:
    def __init__(self) -> None:
        self.safe_points: list[str] = []

    def check(self, safe_point: str) -> None:
        self.safe_points.append(safe_point)


def test_tensor_state_hash_is_order_independent_sensitive_and_fast() -> None:
    state = {
        "weight": np.arange(1_900_000, dtype=np.float32).reshape(1000, 1900),
        "bias": np.arange(1900, dtype=np.float32),
    }
    started = time.perf_counter()
    digest = canonical_tensor_state_sha256(state)
    cold_elapsed = time.perf_counter() - started
    # Cold timing includes host scheduling, page faults, and CPU frequency ramp.  Keep a
    # generous containment bound, then use repeated relative evidence to detect an actual
    # regression to Python-list/JSON tensor serialization.
    assert cold_elapsed < 2.0

    def direct_bytes_hash() -> str:
        direct = hashlib.sha256()
        for name in sorted(state):
            direct.update(np.ascontiguousarray(state[name]).tobytes(order="C"))
        return direct.hexdigest()

    canonical_seconds: list[float] = []
    direct_seconds: list[float] = []
    for _ in range(5):
        started = time.perf_counter()
        direct_bytes_hash()
        direct_seconds.append(time.perf_counter() - started)
        started = time.perf_counter()
        assert canonical_tensor_state_sha256(state) == digest
        canonical_seconds.append(time.perf_counter() - started)
    canonical_median = float(np.median(canonical_seconds))
    direct_median = float(np.median(direct_seconds))
    payload_bytes = sum(value.nbytes for value in state.values())
    throughput_bytes_per_second = payload_bytes / max(canonical_median, 1e-12)
    assert canonical_median <= max(0.25, 12 * direct_median)
    assert throughput_bytes_per_second >= 20 * 1024**2
    assert digest == canonical_tensor_state_sha256(dict(reversed(list(state.items()))))
    changed = {key: value.copy() for key, value in state.items()}
    changed["weight"][0, 0] += 1
    assert canonical_tensor_state_sha256(changed) != digest


def test_checkpoint_store_is_immutable_bound_and_completed_only(tmp_path: Path) -> None:
    store = _store(tmp_path / "checkpoints")
    trial = _trial()
    state = {"weight": np.arange(12, dtype=np.float32).reshape(3, 4)}
    first = store.save_tensor_state(
        trial, state_kind="best", state=state, payload={"epoch": 2}
    )
    second = store.save_tensor_state(
        trial, state_kind="best", state=state, payload={"epoch": 2}
    )
    assert first == second
    loaded, manifest = store.load_tensor_state(first)
    assert manifest["schema_version"] == CHECKPOINT_SCHEMA
    assert manifest["payload"]["tensor_state_hash_schema"] == TENSOR_STATE_HASH_SCHEMA
    assert np.array_equal(loaded["weight"], state["weight"])
    prediction = np.arange(24, dtype=np.float64).reshape(2, 12)
    completed = store.mark_trial_complete(
        trial,
        record={
            "status": "COMPLETE",
            "candidate_id": trial["candidate_id"],
            "family": trial["family"],
            "seed": trial["seed"],
            "metadata_sha256": "8" * 64,
        },
        prediction=prediction,
        completion=_completion(2),
    )
    assert completed.name.startswith("completed-")
    key = trial_identity_sha256(trial)
    replay = ReplayContract(
        planned_trials=(trial,),
        completed_entries={
            key: {
                "status": "COMPLETE",
                "complete_ledger_entry_sha256": "7" * 64,
                "start_ledger_entry_sha256": "9" * 64,
                "checkpoint": store.validate_completed_checkpoint(
                    completed, trial=trial, completion=_completion(2)
                ),
            }
        },
        ledger_sha256="e" * 64,
        terminal_status="COMPLETE",
    )
    rows = store.load_completed("a" * 64, replay=replay)
    assert len(rows) == 1
    assert np.array_equal(rows[0]["prediction"], prediction)
    assert store.latest_safe(trial)["kind"] == "completed"

    mismatched = CheckpointStore(
        tmp_path / "checkpoints",
        plan_sha256="e" * 64,
        config_sha256="b" * 64,
        input_sha256="c" * 64,
        cache_sha256="d" * 64,
        run_start_sha256="e" * 64,
    )
    with pytest.raises(CheckpointError, match="binding differs"):
        mismatched.load_completed("e" * 64, replay=replay)


def test_cache_identity_is_fold_and_recipe_isolated(tmp_path: Path) -> None:
    matrix, _ = _data(rows=12, columns=6)
    left_identity = _identity(fold=0, recipe="1")
    right_identity = _identity(fold=1, recipe="2")
    assert cache_identity_sha256(left_identity) != cache_identity_sha256(right_identity)
    left = build_feature_cache(
        matrix[:8],
        matrix[8:],
        option="memmap",
        identity=left_identity,
        device=__import__("torch").device("cpu"),
        directory=tmp_path,
    )
    right = build_feature_cache(
        matrix[:8],
        matrix[8:],
        option="memmap",
        identity=right_identity,
        device=__import__("torch").device("cpu"),
        directory=tmp_path,
    )
    assert left.paths != right.paths
    assert np.array_equal(left.train, right.train)
    paths = (*left.paths, *right.paths)
    left.close()
    right.close()
    assert all(path.exists() for path in paths)
    changed = matrix.copy().tolil()
    changed[0, 0] = float(changed[0, 0]) + 1
    with pytest.raises(BackendError, match="immutable cache artifact differs"):
        build_feature_cache(
            changed.tocsr()[:8],
            changed.tocsr()[8:],
            option="streaming",
            identity=left_identity,
            device=__import__("torch").device("cpu"),
            directory=tmp_path,
        )


def test_memory_model_and_deterministic_cache_choice(tmp_path: Path) -> None:
    model = estimate_memory_model(rows=133_885, columns=2065)
    assert model["dense_feature_cache_bytes"] == 1_105_890_100
    assert model["w3_base_oof_bytes"] == 385_588_800
    base = {
        "schema_version": "qm9-model-selection-v31-cache-profile-v1",
        "option": "streaming",
        "equivalent": True,
        "validation_equivalent": True,
        "identity_sha256": "7" * 64,
        "build_seconds": 0.0,
        "median_batch_ms": 4.0,
        "p95_data_ready_ms": 3.0,
        "validation_data_ready_seconds": 1.0,
        "peak_incremental_bytes": 0,
        "peak_reserved_bytes": 0,
        "baseline_reserved_bytes": 0,
        "measured_reserved_delta_bytes": 0,
        "projected_savings_seconds": 0.0,
        "swap_growth_bytes": 0,
        "gpu_execution_approved": False,
    }
    gpu = {
        **base,
        "option": "gpu",
        "build_seconds": 2.0,
        "median_batch_ms": 2.0,
        "p95_data_ready_ms": 0.4,
        "peak_incremental_bytes": 2 * 1024**3,
        "peak_reserved_bytes": 3 * 1024**3,
        "measured_reserved_delta_bytes": 2 * 1024**3,
        "gpu_execution_approved": True,
    }
    resources = {
        "total_vram_bytes": 8 * 1024**3,
        "free_vram_bytes": 7 * 1024**3,
        "total_ram_bytes": 32 * 1024**3,
        "available_ram_bytes": 20 * 1024**3,
        "free_disk_bytes": 100 * 1024**3,
        "cache_bytes": 1024**3,
    }
    assert choose_cache_option({"streaming": base, "gpu": gpu}, **resources) == "gpu"
    rejected = {**gpu, "equivalent": False}
    assert choose_cache_option({"streaming": base, "gpu": rejected}, **resources) == "streaming"
    unexecuted = {**gpu, "gpu_execution_approved": False}
    assert choose_cache_option({"streaming": base, "gpu": unexecuted}, **resources) == "streaming"

    worksheet = build_memory_worksheet(
        rows=1000,
        columns=32,
        hidden_dims=[16, 8],
        batch_size=64,
        validation_chunk_size=256,
        cache_option="gpu",
        total_ram_bytes=32 * 1024**3,
        available_ram_bytes=24 * 1024**3,
        total_vram_bytes=8 * 1024**3,
        free_vram_bytes=7 * 1024**3,
        measured_rss_baseline_bytes=100_000_000,
        measured_rss_peak_bytes=110_000_000,
        measured_gpu_baseline_bytes=200_000_000,
        measured_gpu_peak_bytes=250_000_000,
    )
    assert worksheet["safety_margin_fraction"] == 0.20
    assert set(worksheet["gpu_components"]) >= {
        "parameters",
        "gradients",
        "adamw_first_moment",
        "adamw_second_moment",
        "training_activations",
        "framework_cuda_reservation",
        "validation_workspace",
    }
    assert worksheet["approved"] is True
    no_sensor = build_memory_worksheet(
        rows=1000,
        columns=32,
        hidden_dims=[16],
        batch_size=64,
        validation_chunk_size=256,
        cache_option="streaming",
        total_ram_bytes=32 * 1024**3,
        available_ram_bytes=24 * 1024**3,
        total_vram_bytes=8 * 1024**3,
        free_vram_bytes=7 * 1024**3,
        measured_rss_baseline_bytes=None,
        measured_rss_peak_bytes=None,
        measured_gpu_baseline_bytes=None,
        measured_gpu_peak_bytes=None,
    )
    assert no_sensor["approved"] is False
    cpu_worksheet = build_memory_worksheet(
        rows=1000,
        columns=32,
        hidden_dims=[16],
        batch_size=64,
        validation_chunk_size=256,
        cache_option="streaming",
        total_ram_bytes=32 * 1024**3,
        available_ram_bytes=24 * 1024**3,
        total_vram_bytes=0,
        free_vram_bytes=0,
        measured_rss_baseline_bytes=100_000_000,
        measured_rss_peak_bytes=110_000_000,
        measured_gpu_baseline_bytes=None,
        measured_gpu_peak_bytes=None,
        device_type="cpu",
    )
    assert cpu_worksheet["approved"] is True
    assert cpu_worksheet["sensor_availability"]["gpu_sensor_required"] is False
    assert all(value == 0 for value in cpu_worksheet["gpu_components"].values())
    assert cpu_worksheet["cpu_components"]["model_parameters"] > 0

    runtime = assess_runtime_feasibility(
        {
            "ridge|tier0": [1.0, 1.1],
            "xgboost|tier1": [20.0, 22.0],
            "mlp|tier1": [30.0, 31.0],
        },
        {"ridge|tier0": 60, "xgboost|tier1": 45, "mlp|tier1": 18},
    )
    assert runtime["ceiling_seconds"] == 34_080.0
    assert runtime["uncertainty_fraction"] >= 0.20
    assert runtime["approved"] is True
    too_slow = assess_runtime_feasibility(
        {
            "ridge|tier0": [100.0, 110.0],
            "xgboost|tier1": [1000.0, 1100.0],
            "mlp|tier1": [1000.0, 1100.0],
        },
        {"ridge|tier0": 60, "xgboost|tier1": 45, "mlp|tier1": 18},
    )
    assert too_slow["approved"] is False

    device_identity = {"type": "cpu", "name": "governed-test"}
    actual_hashes = {
        "plan_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "input_sha256": "3" * 64,
        "profile_sha256": "4" * 64,
        "code_sha256": "5" * 64,
    }
    feasibility_path = tmp_path / "feasibility.json"
    feasibility = freeze_feasibility_artifact(
        feasibility_path,
        worksheet,
        runtime,
        device_identity=device_identity,
        actual_hashes=actual_hashes,
    )
    loaded = load_feasibility_artifact(
        feasibility_path,
        expected_file_sha256=artifact_file_sha256(feasibility_path),
        expected_memory_sha256=worksheet["semantic_sha256"],
        expected_runtime_sha256=runtime["semantic_sha256"],
        expected_device_identity_sha256=feasibility["device_identity_sha256"],
    )
    assert loaded["approved"] is True
    refused_path = tmp_path / "feasibility-refused.json"
    refused = freeze_feasibility_artifact(
        refused_path,
        worksheet,
        too_slow,
        device_identity=device_identity,
        actual_hashes=actual_hashes,
    )
    with pytest.raises(BackendError, match="not approved"):
        load_feasibility_artifact(
            refused_path,
            expected_file_sha256=artifact_file_sha256(refused_path),
            expected_memory_sha256=worksheet["semantic_sha256"],
            expected_runtime_sha256=too_slow["semantic_sha256"],
            expected_device_identity_sha256=refused["device_identity_sha256"],
        )


def test_bounded_cpu_cache_profiler_is_target_free_and_equivalent(tmp_path: Path) -> None:
    matrix, _ = _data(rows=20, columns=8)
    profiles = profile_cache_options(
        matrix[:15],
        matrix[15:],
        batch_indices=np.asarray([0, 2, 4, 6]),
        identity=_identity(),
        device=__import__("torch").device("cpu"),
        directory=tmp_path,
        samples=3,
    )
    assert set(profiles) == {"streaming", "cpu", "memmap"}
    assert all(row["equivalent"] for row in profiles.values())
    assert all(row["validation_equivalent"] for row in profiles.values())
    assert all(row["schema_version"].endswith("profile-v1") for row in profiles.values())
    choice_path = tmp_path / "cache-choice.json"
    choice = freeze_cache_choice(
        choice_path,
        profiles,
        total_vram_bytes=0,
        free_vram_bytes=0,
        total_ram_bytes=32 * 1024**3,
        available_ram_bytes=20 * 1024**3,
        free_disk_bytes=100 * 1024**3,
        cache_bytes=matrix.shape[0] * matrix.shape[1] * 4,
        device_identity={"type": "cpu", "name": "synthetic-test"},
    )
    assert load_cache_choice(choice_path) == choice
    tampered = __import__("json").loads(choice_path.read_text())
    tampered["chosen_option"] = "gpu"
    choice_path.write_text(__import__("json").dumps(tampered))
    with pytest.raises(BackendError, match="semantic hash differs"):
        load_cache_choice(choice_path)


def test_ridge_profiled_preserves_v3_math_and_emits_stages() -> None:
    matrix, targets = _data()
    sink = RecordingSink()
    guard = RecordingGuard()
    prediction, metadata = fit_ridge_profiled(
        matrix[:28], targets[:28], matrix[28:], alpha=10.0, sink=sink, guard=guard
    )
    mean, scale = targets[:28].mean(axis=0), targets[:28].std(axis=0)
    reference = (
        Ridge(alpha=10.0, solver="lsqr", tol=1e-4, max_iter=10_000)
        .fit(matrix[:28], (targets[:28] - mean) / scale)
        .predict(matrix[28:])
        * scale
        + mean
    )
    assert np.array_equal(prediction, reference)
    assert metadata["backend"] == "sklearn-ridge"
    assert {event for event, _ in sink.events} >= {"stage_start", "stage_end"}
    assert guard.safe_points == ["ridge-before-fit", "ridge-after-fit"]


def test_profiled_backend_uses_real_observability_contract(tmp_path: Path) -> None:
    matrix, targets = _data(rows=20, columns=6)
    trial = TrialIdentity(
        run_id="integration",
        wave="W1",
        strategy="identity",
        candidate_id="ridge|all_raw|alpha-1",
        family="ridge",
        fold=0,
        seed=7,
        trial_index=1,
        trial_total=1,
    )
    path = tmp_path / "events.jsonl"
    with EventSink(path, "integration") as sink:
        fit_ridge_profiled(
            matrix[:15], targets[:15], matrix[15:], alpha=1.0, sink=sink, trial=trial
        )
    events = validate_event_stream(path)
    assert [event["event"] for event in events] == [
        "stage_start",
        "stage_end",
        "stage_start",
        "stage_end",
    ]
    assert all(event["outer_validation_targets_read"] is False for event in events)


def test_mlp_v3_streaming_and_cpu_cache_exact_equivalence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    matrix, targets = _data(rows=32, columns=10)
    monitoring = {
        "validation_increase_mark_after": 2,
        "max_validation_mnmae_after_five_epochs": 1e6,
        "minimum_relative_train_improvement_by_epoch_ten": -1.0,
        "gradient_norm_hard_max": 1e9,
        "gradient_norm_hard_min": -1.0,
        "gradient_min_consecutive_batches": 999,
        "target_degradation_warning": 1e9,
    }
    kwargs = {
        "candidate": _candidate(),
        "training": _training(epochs=3, batch_size=24),
        "monitoring": monitoring,
        "seed": 17,
        "device": "cpu",
        "cache_identity": _identity(),
        "cache_directory": tmp_path,
        "validation_chunk_size": 4,
    }
    reference, reference_curve = fit_real_torch_mlp(
        matrix[:24],
        targets[:24],
        matrix[24:],
        targets[24:],
        _candidate(),
        _training(epochs=3, batch_size=24),
        monitoring,
        seed=17,
    )
    streaming, left = fit_torch_mlp_profiled(
        matrix[:24], targets[:24], matrix[24:], targets[24:], cache_option="streaming", **kwargs
    )
    cached, right = fit_torch_mlp_profiled(
        matrix[:24], targets[:24], matrix[24:], targets[24:], cache_option="cpu", **kwargs
    )
    assert left["initial_state_sha256"] == right["initial_state_sha256"]
    assert left["first_batch"]["indices_sha256"] == right["first_batch"]["indices_sha256"]
    assert left["first_batch"]["input_sha256"] == right["first_batch"]["input_sha256"]
    assert left["first_batch"]["target_sha256"] == right["first_batch"]["target_sha256"]
    assert left["first_batch"]["loss"] == pytest.approx(
        right["first_batch"]["loss"], rel=1e-6, abs=1e-7
    )
    assert left["first_batch"]["gradient_norm"] == pytest.approx(
        right["first_batch"]["gradient_norm"], rel=1e-6
    )
    assert left["first_batch"]["post_step_state_sha256"] == right["first_batch"][
        "post_step_state_sha256"
    ]
    assert left["final_training_state_sha256"] == right["final_training_state_sha256"]
    assert left["best_epoch"] == right["best_epoch"]
    assert left["stop_reason"] == right["stop_reason"]
    assert np.allclose(reference, streaming, rtol=1e-6, atol=1e-7)
    assert np.allclose(streaming, cached, rtol=1e-6, atol=1e-7)
    for field in (
        "train_standardized_mse",
        "inner_tune_mnmae",
        "inner_tune_per_target_nmae",
    ):
        assert np.allclose(reference_curve[field], left["curve"][field], rtol=1e-6, atol=1e-7)
        assert np.allclose(left["curve"][field], right["curve"][field], rtol=1e-6, atol=1e-7)
    for field in ("best_epoch", "stop_reason", "warnings", "anomalies"):
        assert reference_curve[field] == left["curve"][field] == right["curve"][field]


def test_mlp_validation_chunking_prediction_tolerance(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    matrix, targets = _data(rows=28, columns=8)
    common = {
        "candidate": {**_candidate(), "dropout": 0.0},
        "training": _training(epochs=1, batch_size=20),
        "monitoring": {},
        "seed": 5,
        "device": "cpu",
        "cache_option": "cpu",
        "cache_identity": _identity(),
        "cache_directory": tmp_path,
    }
    full, full_metadata = fit_torch_mlp_profiled(
        matrix[:20], targets[:20], matrix[20:], targets[20:], validation_chunk_size=8, **common
    )
    chunked, chunked_metadata = fit_torch_mlp_profiled(
        matrix[:20], targets[:20], matrix[20:], targets[20:], validation_chunk_size=2, **common
    )
    assert full_metadata["final_training_state_sha256"] == chunked_metadata[
        "final_training_state_sha256"
    ]
    assert np.allclose(full, chunked, rtol=1e-6, atol=1e-7)
    with pytest.raises(BackendError, match="chunk size"):
        fit_torch_mlp_profiled(
            matrix[:20],
            targets[:20],
            matrix[20:],
            targets[20:],
            validation_chunk_size=4097,
            **common,
        )


def test_cuda_gpu_cache_matches_streaming_when_available(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    matrix, targets = _data(rows=24, columns=8)
    common = {
        "candidate": {**_candidate(), "dropout": 0.0},
        "training": _training(epochs=1, batch_size=18),
        "monitoring": {},
        "seed": 13,
        "device": "cuda",
        "cache_identity": _identity(),
        "cache_directory": tmp_path,
        "validation_chunk_size": 3,
    }
    streaming, left = fit_torch_mlp_profiled(
        matrix[:18], targets[:18], matrix[18:], targets[18:], cache_option="streaming", **common
    )
    cached, right = fit_torch_mlp_profiled(
        matrix[:18], targets[:18], matrix[18:], targets[18:], cache_option="gpu", **common
    )
    assert left["first_batch"]["input_sha256"] == right["first_batch"]["input_sha256"]
    assert left["first_batch"]["loss"] == pytest.approx(
        right["first_batch"]["loss"], rel=1e-6, abs=1e-7
    )
    assert left["first_batch"]["gradient_norm"] == pytest.approx(
        right["first_batch"]["gradient_norm"], rel=1e-6
    )
    assert left["final_training_state_sha256"] == right["final_training_state_sha256"]
    assert np.allclose(streaming, cached, rtol=1e-6, atol=1e-7)
    assert left["determinism"]["deterministic_algorithms"] is True
    assert right["determinism"]["tf32_matmul"] is False


def test_real_xgboost_instrumentation_preserves_predictions_and_rounds() -> None:
    pytest.importorskip("xgboost")
    matrix, targets = _data(rows=36, columns=8)
    candidate = {
        "id": "tiny",
        "max_depth": 2,
        "learning_rate": 0.1,
        "n_estimators": 4,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "min_child_weight": 1.0,
    }
    reference, reference_rounds = fit_real_xgboost(
        matrix[:28],
        targets[:28],
        matrix[28:],
        targets[28:],
        candidate,
        seed=3,
        early_stopping_rounds=2,
    )
    sink = RecordingSink()
    guard = RecordingGuard()
    prediction, metadata = fit_xgboost_profiled(
        matrix[:28],
        targets[:28],
        matrix[28:],
        targets[28:],
        candidate,
        seed=3,
        early_stopping_rounds=2,
        sink=sink,
        guard=guard,
    )
    assert np.allclose(prediction, reference, rtol=1e-7, atol=1e-8)
    assert metadata["per_target_rounds"] == reference_rounds
    assert len([event for event, _ in sink.events if event == "xgb_iteration"]) >= 12
    assert len([point for point in guard.safe_points if point.endswith("-complete")]) == 12


def test_profile_dispatch_completes_checkpoint_and_rejects_nonfinite(tmp_path: Path) -> None:
    matrix, targets = _data(rows=24, columns=6)
    store = _store(tmp_path / "store")
    trial = _trial("ridge|all_raw|alpha-1")
    prediction, metadata = profile_real_backend(
        {"id": "ridge|all_raw|alpha-1", "family": "ridge", "params": {"alpha": 1.0}},
        matrix[:18],
        targets[:18],
        matrix[18:],
        targets[18:],
        seed=7,
        sink=None,
        guard=None,
        checkpoint_store=store,
        trial=trial,
        completion_contract=_completion(6),
    )
    assert prediction.shape == (6, 12)
    assert len(metadata["prediction_sha256"]) == 64
    assert store.latest_safe(trial)["kind"] == "completed"
    bad = matrix.copy()
    bad.data[0] = np.inf
    with pytest.raises(BackendError, match="nonfinite"):
        fit_ridge_profiled(bad[:18], targets[:18], bad[18:], alpha=1.0)
