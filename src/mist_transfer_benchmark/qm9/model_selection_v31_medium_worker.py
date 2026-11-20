"""Authenticated worker for the fixed train-only v3.1 medium feasibility profile."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy import sparse

from .constants import TARGET_COLUMNS
from .model_selection_v31_backends import (
    fit_torch_mlp_profiled,
    profile_real_backend,
)
from .model_selection_v31_checkpoint import CheckpointStore, CompletionContract
from .model_selection_v31_medium_profile import (
    PUBLIC_FIXED_MODE,
    RUN_ID,
    _false_boundaries,
    candidates,
)
from .model_selection_v31_observability import EventSink, TrialIdentity
from .model_selection_v31_supervisor import canonical_sha256, file_sha256
from .selection_v3_data import fit_feature_recipe

WORKER_SCHEMA_PREFIX = (
    "qm9-v31-public-fixed-worker" if PUBLIC_FIXED_MODE else "qm9-v31-medium-worker"
)
SPEC_SCHEMA = f"{WORKER_SCHEMA_PREFIX}-spec-v1"
PROFILE_SCHEMA = f"{WORKER_SCHEMA_PREFIX}-profile-v1"


class MediumWorkerError(RuntimeError):
    pass


def _contained(root: Path, value: str | Path) -> Path:
    path = (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise MediumWorkerError("medium worker path escapes run root") from error
    return path


def _immutable_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        os.write(descriptor, data)
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)


class _HeartbeatState:
    def __init__(self, label_reads: Mapping[str, int]) -> None:
        self._lock = threading.Lock()
        self._sequence = 0
        self._payload: dict[str, object] = {
            "stage": "worker_start",
            "label_reads": dict(label_reads),
        }

    def update(self, *, event: str, progress: Mapping[str, object] | None = None,
               metrics: Mapping[str, object] | None = None) -> None:
        with self._lock:
            self._sequence += 1
            self._payload = {
                "stage": event,
                "progress_sequence": self._sequence,
                "last_progress_monotonic": time.monotonic(),
                "progress": dict(progress or {}),
                "metrics": dict(metrics or {}),
                "label_reads": dict(self._payload["label_reads"]),
            }

    def set_label_reads(self, label_reads: Mapping[str, int]) -> None:
        with self._lock:
            self._payload["label_reads"] = dict(label_reads)

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return dict(self._payload)


def _heartbeat(stop: threading.Event, path: Path, state: _HeartbeatState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    while not stop.is_set():
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(state.snapshot(), handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        stop.wait(1.0)


class _LabelBoundSink:
    def __init__(
        self, sink: EventSink, label_reads: Mapping[str, int], heartbeat: _HeartbeatState
    ) -> None:
        self.sink = sink
        self.label_reads = label_reads
        self.heartbeat = heartbeat

    def emit(self, event: str, **kwargs: object) -> dict[str, object]:
        progress = dict(kwargs.pop("progress", {}))
        progress["label_reads"] = dict(self.label_reads)
        metrics = kwargs.get("metrics")
        self.heartbeat.set_label_reads(self.label_reads)
        self.heartbeat.update(
            event=event,
            progress=progress,
            metrics=metrics if isinstance(metrics, Mapping) else None,
        )
        return self.sink.emit(event, progress=progress, **kwargs)  # type: ignore[arg-type]


def _candidate(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise MediumWorkerError("medium candidate is not an object")
    payload = dict(value)
    if payload not in candidates():
        raise MediumWorkerError("medium candidate differs from frozen plan")
    return payload


def write_spec(
    path: str | Path,
    *,
    run_root: str | Path,
    data_path: str | Path,
    result_path: str | Path,
    profile_path: str | Path,
    telemetry_path: str | Path,
    checkpoint_root: str | Path,
    candidate: Mapping[str, object],
    trial: TrialIdentity,
    completion: CompletionContract,
    checkpoint_bindings: Mapping[str, object],
) -> dict[str, object]:
    root = Path(run_root).resolve()
    spec_path = _contained(root, path)
    data = _contained(root, data_path)
    if data.is_symlink() or not data.is_file():
        raise MediumWorkerError("medium worker data artifact is missing")
    normalized = _candidate(candidate)
    if trial.run_id != RUN_ID or trial.candidate_id != normalized["id"]:
        raise MediumWorkerError("medium worker candidate/trial binding differs")
    required_bindings = {
        "plan_sha256", "config_sha256", "input_sha256", "cache_sha256",
        "run_start_sha256",
    }
    if set(checkpoint_bindings) != required_bindings:
        raise MediumWorkerError("medium worker checkpoint binding schema differs")
    if checkpoint_bindings["input_sha256"] != file_sha256(data):
        raise MediumWorkerError("medium worker checkpoint input binding differs")
    unsigned = {
        "schema_version": SPEC_SCHEMA,
        "run_id": RUN_ID,
        "run_root": str(root),
        "data_path": str(data.relative_to(root)),
        "data_file_sha256": file_sha256(data),
        "result_path": str(_contained(root, result_path).relative_to(root)),
        "profile_path": str(_contained(root, profile_path).relative_to(root)),
        "telemetry_path": str(_contained(root, telemetry_path).relative_to(root)),
        "checkpoint_root": str(_contained(root, checkpoint_root).relative_to(root)),
        "candidate": normalized,
        "trial": asdict(trial),
        "completion": completion.payload(),
        "checkpoint_bindings": dict(checkpoint_bindings),
        "device": "cpu",
        "fixed_train_labels_only": not PUBLIC_FIXED_MODE,
        "synthetic_targets_only": PUBLIC_FIXED_MODE,
        "outer_validation_targets_read": False,
        "test_targets_read": False,
    }
    artifact = {**unsigned, "semantic_sha256": canonical_sha256(unsigned)}
    _immutable_json(spec_path, artifact)
    load_spec(spec_path)
    return artifact


def load_spec(path: str | Path) -> dict[str, object]:
    spec_path = Path(path)
    artifact = json.loads(spec_path.read_text())
    required = {
        "schema_version", "run_id", "run_root", "data_path", "data_file_sha256",
        "result_path", "profile_path", "telemetry_path", "checkpoint_root", "candidate",
        "trial", "completion", "checkpoint_bindings", "device", "fixed_train_labels_only",
        "synthetic_targets_only",
        "outer_validation_targets_read", "test_targets_read", "semantic_sha256",
    }
    unsigned = dict(artifact)
    semantic = unsigned.pop("semantic_sha256", None)
    if (
        not isinstance(artifact, dict)
        or set(artifact) != required
        or artifact.get("schema_version") != SPEC_SCHEMA
        or artifact.get("run_id") != RUN_ID
        or semantic != canonical_sha256(unsigned)
        or artifact.get("device") != "cpu"
        or artifact.get("fixed_train_labels_only") is not (not PUBLIC_FIXED_MODE)
        or artifact.get("synthetic_targets_only") is not PUBLIC_FIXED_MODE
        or artifact.get("outer_validation_targets_read") is not False
        or artifact.get("test_targets_read") is not False
    ):
        raise MediumWorkerError("medium worker spec contract differs")
    root = Path(str(artifact["run_root"])).resolve()
    _contained(root, spec_path)
    data = _contained(root, str(artifact["data_path"]))
    if data.is_symlink() or not data.is_file() or file_sha256(data) != artifact["data_file_sha256"]:
        raise MediumWorkerError("medium worker data binding differs")
    _candidate(artifact["candidate"])
    return artifact


def validate_profile(
    path: str | Path, *, trial: TrialIdentity, expected_metadata_sha256: str
) -> dict[str, object]:
    artifact = json.loads(Path(path).read_text())
    required = {
        "schema_version", "run_id", "trial", "metadata", "metadata_sha256", "label_reads",
        "stage_timing", "outer_validation_targets_read", "test_targets_read",
        "private_evaluation_labels_read", "protected_labels_read", "ranking_allowed",
        "ensemble_allowed", "promotion_allowed", "w1_allowed", "general_training_allowed",
        "scientific_result", "publication_ready", "semantic_sha256",
    }
    unsigned = dict(artifact)
    semantic = unsigned.pop("semantic_sha256", None)
    if (
        not isinstance(artifact, dict)
        or set(artifact) != required
        or artifact.get("schema_version") != PROFILE_SCHEMA
        or artifact.get("run_id") != RUN_ID
        or artifact.get("trial") != asdict(trial)
        or artifact.get("metadata_sha256") != expected_metadata_sha256
        or canonical_sha256(artifact.get("metadata")) != expected_metadata_sha256
        or semantic != canonical_sha256(unsigned)
        or any(artifact.get(key) is not False for key in _false_boundaries())
    ):
        raise MediumWorkerError("medium worker profile semantic contract differs")
    expected_reads = {
        "source_fixed_train_target_loads": 0,
        "derived_train_target_artifact_reads": 0 if PUBLIC_FIXED_MODE else 1,
        "outer_validation": 0,
        "test": 0,
        "protected": 0,
        "private_evaluation": 0,
    }
    if PUBLIC_FIXED_MODE:
        expected_reads["synthetic_target_artifact_reads"] = 1
    if artifact.get("label_reads") != expected_reads:
        raise MediumWorkerError("medium worker profile label counters differ")
    metadata = artifact["metadata"]
    if not isinstance(metadata, Mapping) or not isinstance(
        metadata.get("checkpoint_evidence"), Mapping
    ):
        raise MediumWorkerError("medium worker profile checkpoint evidence is missing")
    stage_timing = artifact["stage_timing"]
    if (
        not isinstance(stage_timing, Mapping)
        or set(stage_timing) != {"data_load", "recipe_transform"}
        or any(
            not isinstance(row, Mapping)
            or set(row) != {"wall_seconds", "cpu_seconds"}
            or any(
                type(row[key]) not in (int, float)
                or not np.isfinite(row[key])
                or row[key] < 0
                for key in ("wall_seconds", "cpu_seconds")
            )
            for row in stage_timing.values()
        )
    ):
        raise MediumWorkerError("medium worker stage timing contract differs")
    backend_stages = metadata.get("stage_seconds")
    monitoring = metadata.get("profile_monitoring")
    if (
        not isinstance(backend_stages, Mapping)
        or not backend_stages
        or any(
            not isinstance(name, str)
            or type(value) not in (int, float)
            or not np.isfinite(value)
            or value < 0
            for name, value in backend_stages.items()
        )
        or not isinstance(monitoring, Mapping)
        or set(monitoring)
        != {"curve_type", "curves", "warnings", "anomalies", "status"}
        or not isinstance(monitoring["warnings"], list)
        or not isinstance(monitoring["anomalies"], list)
        or monitoring["status"] not in {"normal", "warning", "abnormal"}
        or (
            monitoring["status"] == "normal"
            and (monitoring["warnings"] or monitoring["anomalies"])
        )
        or (monitoring["status"] == "warning" and not monitoring["warnings"])
        or (monitoring["status"] == "abnormal" and not monitoring["anomalies"])
    ):
        raise MediumWorkerError("medium worker backend monitoring contract differs")
    if trial.family == "ridge" and (
        metadata.get("backend") != "sklearn-ridge"
        or monitoring["curve_type"] != "non-iterative"
        or monitoring["curves"] != {}
        or set(backend_stages) != {"forward", "predict"}
    ):
        raise MediumWorkerError("medium Ridge profile contract differs")
    if trial.family == "xgboost":
        rounds = metadata.get("per_target_rounds")
        if (
            not isinstance(rounds, list)
            or len(rounds) != len(TARGET_COLUMNS)
            or any(type(value) is not int or not 1 <= value <= 200 for value in rounds)
        ):
            raise MediumWorkerError("medium XGBoost per-target round evidence differs")
        if (
            metadata.get("backend") != "xgboost"
            or monitoring["curve_type"] != "per-target-best-rounds"
            or monitoring["curves"] != {"per_target_rounds": rounds}
            or set(backend_stages) != {"forward", "predict"}
        ):
            raise MediumWorkerError("medium XGBoost curve/stage contract differs")
    if trial.family == "mlp":
        curves = metadata.get("curves")
        curve = metadata.get("curve")
        if (
            not isinstance(curves, Mapping)
            or not isinstance(curve, Mapping)
            or metadata.get("best_checkpoint_sha256")
            != metadata.get("restored_checkpoint_sha256")
            or not isinstance(metadata.get("best_epoch"), int)
            or not isinstance(metadata.get("stop_reason"), str)
            or len(curves.get("train_standardized_mse", [])) < 1
            or any(
                len(row) != len(TARGET_COLUMNS)
                for row in curves.get("inner_tune_per_target_nmae", [])
            )
            or type(curve.get("maximum_consecutive_validation_increases")) is not int
            or not isinstance(curve.get("warnings"), list)
            or not isinstance(curve.get("anomalies"), list)
            or curve.get("status") not in {"normal", "warning", "abnormal"}
        ):
            raise MediumWorkerError("medium MLP curve/anomaly/checkpoint evidence differs")
        lengths = {
            len(curves[name])
            for name in (
                "train_standardized_mse", "inner_tune_mnmae",
                "inner_tune_per_target_nmae", "gradient_norm", "parameter_norm",
            )
        }
        if (
            lengths != {len(curves["train_standardized_mse"])}
            or metadata["best_epoch"] != curve.get("best_epoch")
            or metadata["best_epoch"] != curve.get("restored_epoch")
            or metadata["stop_reason"] != curve.get("stop_reason")
            or metadata["maximum_consecutive_validation_increases"]
            != curve.get("maximum_consecutive_validation_increases")
            or monitoring["curve_type"] != "epoch"
            or monitoring["curves"] != curves
            or monitoring["warnings"] != curve["warnings"]
            or monitoring["anomalies"] != curve["anomalies"]
            or monitoring["status"] != curve["status"]
            or not {
                "feature_transform", "forward", "loss", "backward", "optimizer",
                "validation",
            }.issubset(backend_stages)
        ):
            raise MediumWorkerError("medium MLP curve/stage consistency differs")
    return artifact


def run_worker(path: str | Path) -> dict[str, object]:
    spec = load_spec(path)
    root = Path(str(spec["run_root"])).resolve()
    heartbeat_value = os.environ.get("QM9_V31_HEARTBEAT_PATH")
    if not heartbeat_value:
        raise MediumWorkerError("medium worker heartbeat path is missing")
    heartbeat_path = _contained(root, heartbeat_value)
    live_label_reads = {
        "source_fixed_train_target_loads": 0,
        "derived_train_target_artifact_reads": 0,
        "outer_validation": 0,
        "test": 0,
        "protected": 0,
        "private_evaluation": 0,
    }
    if PUBLIC_FIXED_MODE:
        live_label_reads["synthetic_target_artifact_reads"] = 0
    stop = threading.Event()
    heartbeat_state = _HeartbeatState(live_label_reads)
    heartbeat_state.update(event="data_load_start")
    thread = threading.Thread(
        target=_heartbeat, args=(stop, heartbeat_path, heartbeat_state), daemon=True
    )
    thread.start()
    sink: EventSink | None = None
    try:
        load_wall = time.perf_counter()
        load_cpu = time.process_time()
        with np.load(_contained(root, str(spec["data_path"])), allow_pickle=False) as payload:
            if set(payload.files) != {
                "x_train_data", "x_train_indices", "x_train_indptr", "x_train_shape",
                "x_tune_data", "x_tune_indices", "x_tune_indptr", "x_tune_shape",
                "y_train", "y_tune",
            }:
                raise MediumWorkerError("medium worker data schema contains unexpected arrays")
            x_train_raw = sparse.csr_matrix(
                (payload["x_train_data"], payload["x_train_indices"], payload["x_train_indptr"]),
                shape=tuple(payload["x_train_shape"]),
            )
            x_tune_raw = sparse.csr_matrix(
                (payload["x_tune_data"], payload["x_tune_indices"], payload["x_tune_indptr"]),
                shape=tuple(payload["x_tune_shape"]),
            )
            y_train = np.asarray(payload["y_train"], dtype=np.float64)
            y_tune = np.asarray(payload["y_tune"], dtype=np.float64)
        live_label_reads[
            "synthetic_target_artifact_reads"
            if PUBLIC_FIXED_MODE
            else "derived_train_target_artifact_reads"
        ] = 1
        heartbeat_state.set_label_reads(live_label_reads)
        heartbeat_state.update(event="data_load_complete")
        load_timing = {
            "wall_seconds": time.perf_counter() - load_wall,
            "cpu_seconds": time.process_time() - load_cpu,
        }
        if (
            x_train_raw.shape[0] + x_tune_raw.shape[0] < 8192
            or y_train.shape != (x_train_raw.shape[0], len(TARGET_COLUMNS))
            or y_tune.shape != (x_tune_raw.shape[0], len(TARGET_COLUMNS))
            or x_train_raw.shape[1] != x_tune_raw.shape[1]
            or not np.isfinite(y_train).all()
            or not np.isfinite(y_tune).all()
        ):
            raise MediumWorkerError("medium worker selected train/tune dimensions differ")
        candidate = _candidate(spec["candidate"])
        train_rows = np.arange(x_train_raw.shape[0], dtype=np.int64)
        tune_rows = np.arange(
            x_train_raw.shape[0], x_train_raw.shape[0] + x_tune_raw.shape[0], dtype=np.int64
        )
        raw = sparse.vstack([x_train_raw, x_tune_raw], format="csr")
        transform_wall = time.perf_counter()
        transform_cpu = time.process_time()
        heartbeat_state.update(event="recipe_transform_start")
        x_train, x_tune, recipe = fit_feature_recipe(
            str(candidate["recipe"]), raw, train_rows, tune_rows
        )
        transform_timing = {
            "wall_seconds": time.perf_counter() - transform_wall,
            "cpu_seconds": time.process_time() - transform_cpu,
        }
        heartbeat_state.update(event="recipe_transform_complete")
        completion_payload = spec["completion"]
        completion = CompletionContract(
            expected_shape=tuple(completion_payload["expected_shape"]),
            target_order=tuple(completion_payload["target_order"]),
            prediction_order_sha256=completion_payload["prediction_order_sha256"],
            start_ledger_entry_sha256=completion_payload["start_ledger_entry_sha256"],
        )
        trial = TrialIdentity(**spec["trial"])
        bindings = spec["checkpoint_bindings"]
        store = CheckpointStore(
            _contained(root, str(spec["checkpoint_root"])),
            plan_sha256=bindings["plan_sha256"], config_sha256=bindings["config_sha256"],
            input_sha256=bindings["input_sha256"], cache_sha256=bindings["cache_sha256"],
            run_start_sha256=bindings["run_start_sha256"],
        )
        sink = EventSink(
            _contained(root, str(spec["telemetry_path"])), RUN_ID,
            run_start_sha256=bindings["run_start_sha256"],
        )
        governed_sink = _LabelBoundSink(sink, live_label_reads, heartbeat_state)
        if candidate["family"] == "mlp":
            identity = {
                "feature_artifact_sha256": spec["data_file_sha256"],
                "recipe_provenance_sha256": canonical_sha256(recipe),
                "strategy": "identity", "fold": 0,
                "train_ordered_sha256": canonical_sha256(train_rows.tolist()),
                "tune_ordered_sha256": canonical_sha256(tune_rows.tolist()),
            }
            prediction, metadata = fit_torch_mlp_profiled(
                x_train, y_train, x_tune, y_tune, candidate["params"],
                candidate["training"], candidate["monitoring"], seed=trial.seed,
                sink=governed_sink, checkpoint_store=store, trial=trial, cache_option="streaming",
                cache_identity=identity, cache_directory=root / "cache" / str(trial.trial_index),
                device="cpu",
            )
            completed = store.mark_trial_complete(
                trial,
                record={"status": "COMPLETE", "candidate_id": candidate["id"],
                        "family": "mlp", "seed": trial.seed,
                        "metadata_sha256": canonical_sha256(metadata)},
                prediction=prediction, completion=completion,
            )
            metadata["checkpoint_evidence"] = store.validate_completed_checkpoint(
                completed, trial=trial, completion=completion
            )
        else:
            _, metadata = profile_real_backend(
                candidate, x_train, y_train, x_tune, y_tune, seed=trial.seed,
                sink=governed_sink,
                guard=None, checkpoint_store=store, trial=trial,
                completion_contract=completion, device="cpu",
            )
        if candidate["family"] == "ridge":
            monitoring = {
                "curve_type": "non-iterative", "curves": {}, "warnings": [],
                "anomalies": [], "status": "normal",
            }
        elif candidate["family"] == "xgboost":
            monitoring = {
                "curve_type": "per-target-best-rounds",
                "curves": {"per_target_rounds": metadata["per_target_rounds"]},
                "warnings": [], "anomalies": [], "status": "normal",
            }
        else:
            curve = metadata["curve"]
            monitoring = {
                "curve_type": "epoch", "curves": metadata["curves"],
                "warnings": curve["warnings"], "anomalies": curve["anomalies"],
                "status": curve["status"],
            }
        metadata["profile_monitoring"] = monitoring
        metadata_sha256 = canonical_sha256(metadata)
        counters = {
            "source_fixed_train_target_loads": 0,
            "derived_train_target_artifact_reads": 0 if PUBLIC_FIXED_MODE else 1,
            "outer_validation": 0,
            "test": 0,
            "protected": 0,
            "private_evaluation": 0,
        }
        if PUBLIC_FIXED_MODE:
            counters["synthetic_target_artifact_reads"] = 1
        profile_unsigned = {
            "schema_version": PROFILE_SCHEMA, "run_id": RUN_ID, "trial": spec["trial"],
            "metadata": metadata, "metadata_sha256": metadata_sha256,
            "outer_validation_targets_read": False, "test_targets_read": False,
            "scientific_result": False,
            "label_reads": counters,
            "stage_timing": {"data_load": load_timing, "recipe_transform": transform_timing},
            **_false_boundaries(),
        }
        _immutable_json(
            _contained(root, str(spec["profile_path"])),
            {**profile_unsigned, "semantic_sha256": canonical_sha256(profile_unsigned)},
        )
        result = {
            "status": "COMPLETE", "checkpoint_evidence": metadata["checkpoint_evidence"],
            "metadata_sha256": metadata_sha256,
            "label_reads": counters,
            **_false_boundaries(),
        }
        _immutable_json(_contained(root, str(spec["result_path"])), result)
        return result
    finally:
        if sink is not None:
            sink.close()
        stop.set()
        thread.join(timeout=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-spec", type=Path, required=True)
    run_worker(parser.parse_args().worker_spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
