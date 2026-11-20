"""Profiled, numerically equivalent real backends for QM9 selection v3.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge

from .constants import TARGET_COLUMNS
from .model_selection_v31_checkpoint import (
    CheckpointStore,
    CompletionContract,
    canonical_prediction_sha256,
    canonical_tensor_state_sha256,
    trial_identity_sha256,
)

CACHE_SCHEMA = "qm9-model-selection-v31-feature-cache-v1"
PROFILE_SCHEMA = "qm9-model-selection-v31-cache-profile-v1"
CACHE_CHOICE_SCHEMA = "qm9-model-selection-v31-cache-choice-v1"
FEASIBILITY_SCHEMA = "qm9-model-selection-v31-feasibility-v1"
REAL_BACKEND_WORKER_SPEC_SCHEMA = "qm9-model-selection-v31-real-worker-spec-v1"
REAL_BACKEND_PROFILE_SCHEMA = "qm9-model-selection-v31-real-profile-v1"


class BackendError(ValueError):
    """Raised when a profiled backend violates finiteness or equivalence contracts."""


class EventSink(Protocol):
    def emit(self, event: str, **kwargs: object) -> Mapping[str, object]: ...


class RunGuard(Protocol):
    def check(self, safe_point: str) -> None: ...


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def artifact_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(
        _canonical_json({"shape": list(array.shape), "dtype": array.dtype.str, "order": "C"})
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sparse_semantic_hash(value: sparse.spmatrix) -> str:
    matrix = sparse.csr_matrix(value, copy=True)
    matrix.sum_duplicates()
    matrix.sort_indices()
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "kind": "csr",
                "shape": list(matrix.shape),
                "dtype": matrix.dtype.str,
                "order": "row-major",
            }
        )
    )
    for array in (matrix.indptr, matrix.indices, matrix.data):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ensure_directory(path: Path) -> None:
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    path.mkdir(parents=True, exist_ok=True)
    for created in reversed(missing):
        _fsync_directory(created.parent)


def _write_immutable(path: Path, payload: bytes) -> None:
    """Atomically create once; concurrent writers must have identical bytes."""

    _ensure_directory(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
            _fsync_directory(path.parent)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise BackendError(
                    "immutable cache artifact differs under concurrent reuse"
                ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _finite(value: object, name: str) -> None:
    if hasattr(value, "detach"):
        import torch

        if not bool(torch.isfinite(value).all()):
            raise BackendError(f"{name} contains nonfinite values")
        return
    array = value.data if sparse.issparse(value) else np.asarray(value)
    if not np.all(np.isfinite(array)):
        raise BackendError(f"{name} contains nonfinite values")


def _guard(guard: RunGuard | None, safe_point: str) -> None:
    if guard is not None:
        guard.check(safe_point)


def _sync(device: object | None) -> None:
    if device is not None and getattr(device, "type", None) == "cuda":
        import torch

        torch.cuda.synchronize(device)


def _emit(
    sink: EventSink | None,
    event: str,
    *,
    trial: object | None,
    progress: Mapping[str, object],
    metrics: Mapping[str, object] | None = None,
    timings: Mapping[str, float] | None = None,
    force_fsync: bool = False,
) -> None:
    if sink is None:
        return
    sink.emit(
        event,
        trial=trial,
        progress=dict(progress),
        metrics=dict(metrics) if metrics is not None else None,
        timings=dict(timings) if timings is not None else None,
        force_fsync=force_fsync,
    )


@contextmanager
def _stage(
    name: str,
    timings: dict[str, float],
    *,
    sink: EventSink | None,
    trial: object | None,
    device: object | None = None,
) -> Iterator[None]:
    _sync(device)
    _emit(sink, "stage_start", trial=trial, progress={"stage": name})
    started = time.perf_counter()
    try:
        yield
    finally:
        _sync(device)
        elapsed = time.perf_counter() - started
        timings[name] = timings.get(name, 0.0) + elapsed
        _emit(
            sink,
            "stage_end",
            trial=trial,
            progress={"stage": name},
            timings={"last_seconds": elapsed, "cumulative_seconds": timings[name]},
        )


def _validate_cache_identity(identity: Mapping[str, object]) -> dict[str, object]:
    required = {
        "feature_artifact_sha256",
        "recipe_provenance_sha256",
        "strategy",
        "fold",
        "train_ordered_sha256",
        "tune_ordered_sha256",
    }
    if set(identity) != required:
        raise BackendError("cache identity schema differs")
    for key in (
        "feature_artifact_sha256",
        "recipe_provenance_sha256",
        "train_ordered_sha256",
        "tune_ordered_sha256",
    ):
        value = identity[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or value == "0" * 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise BackendError(f"cache identity {key} is invalid")
    if identity["strategy"] not in {"identity", "scaffold"} or type(identity["fold"]) is not int:
        raise BackendError("cache strategy/fold identity is invalid")
    return dict(identity)


def cache_identity_sha256(identity: Mapping[str, object]) -> str:
    return _canonical_hash(
        {"schema_version": CACHE_SCHEMA, "identity": _validate_cache_identity(identity)}
    )


def _publish_npy(path: Path, values: np.ndarray) -> None:
    _ensure_directory(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".npy", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
            _fsync_directory(path.parent)
        except FileExistsError:
            existing = np.load(path, mmap_mode="r", allow_pickle=False)
            if _array_hash(existing) != _array_hash(values):
                raise BackendError("immutable NPY cache content differs") from None
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class FeatureCache:
    option: str
    train: object
    tune: object
    identity_sha256: str
    device: object | None = None
    paths: tuple[Path, ...] = ()

    def batch(
        self,
        indices: np.ndarray,
        timings: dict[str, float],
        *,
        sink: EventSink | None,
        trial: object | None,
    ) -> object:
        import torch

        if self.option == "streaming":
            with _stage("sparse_slice", timings, sink=sink, trial=trial):
                block = self.train[indices]
            with _stage("to_dense", timings, sink=sink, trial=trial):
                dense = block.toarray()
            with _stage("tensor_create", timings, sink=sink, trial=trial):
                tensor = torch.from_numpy(dense.astype(np.float32, copy=False))
            with _stage(
                "host_to_device", timings, sink=sink, trial=trial, device=self.device
            ):
                return tensor.to(self.device)
        if self.option in {"cpu", "memmap"}:
            with _stage("cache_gather", timings, sink=sink, trial=trial):
                dense = np.array(self.train[indices], dtype=np.float32, order="C", copy=True)
            with _stage("tensor_create", timings, sink=sink, trial=trial):
                tensor = torch.from_numpy(dense)
            with _stage(
                "host_to_device", timings, sink=sink, trial=trial, device=self.device
            ):
                return tensor.to(self.device)
        if self.option == "gpu":
            with _stage("cache_gather", timings, sink=sink, trial=trial, device=self.device):
                index = torch.as_tensor(indices, dtype=torch.long, device=self.device)
                return self.train[index]
        raise BackendError(f"unknown cache option: {self.option}")

    def validation_chunks(
        self,
        chunk_size: int,
        timings: dict[str, float],
        *,
        sink: EventSink | None,
        trial: object | None,
    ) -> Iterator[tuple[int, object]]:
        rows = int(self.tune.shape[0])
        for start in range(0, rows, chunk_size):
            indices = np.arange(start, min(start + chunk_size, rows), dtype=np.int64)
            if self.option == "streaming":
                with _stage("sparse_slice", timings, sink=sink, trial=trial):
                    block = self.tune[indices]
                with _stage("to_dense", timings, sink=sink, trial=trial):
                    dense = block.toarray().astype(np.float32, copy=False)
                with _stage("tensor_create", timings, sink=sink, trial=trial):
                    tensor = __import__("torch").from_numpy(dense)
                with _stage(
                    "host_to_device", timings, sink=sink, trial=trial, device=self.device
                ):
                    yield start, tensor.to(self.device)
            elif self.option in {"cpu", "memmap"}:
                with _stage("cache_gather", timings, sink=sink, trial=trial):
                    dense = np.array(
                        self.tune[start : start + chunk_size],
                        dtype=np.float32,
                        order="C",
                        copy=True,
                    )
                with _stage("tensor_create", timings, sink=sink, trial=trial):
                    tensor = __import__("torch").from_numpy(dense)
                with _stage(
                    "host_to_device", timings, sink=sink, trial=trial, device=self.device
                ):
                    yield start, tensor.to(self.device)
            else:
                yield start, self.tune[start : start + chunk_size]

    def close(self) -> None:
        # Immutable cache artifacts are run-owned evidence.  The supervisor, not
        # an individual trial, owns retention and deletion.
        self.train = None
        self.tune = None


def build_feature_cache(
    x_train: sparse.csr_matrix,
    x_tune: sparse.csr_matrix,
    *,
    option: str,
    identity: Mapping[str, object],
    device: object,
    directory: str | Path | None = None,
) -> FeatureCache:
    """Build one fold/recipe cache without changing feature mathematics."""

    import torch

    if option not in {"streaming", "cpu", "gpu", "memmap"}:
        raise BackendError("unsupported cache option")
    if directory is None:
        raise BackendError("cache_directory is mandatory for immutable cache provenance")
    declared_identity = _validate_cache_identity(identity)
    declared_hash = cache_identity_sha256(declared_identity)
    _finite(x_train, "train features")
    _finite(x_tune, "tune features")
    cache_directory = Path(directory)
    _ensure_directory(cache_directory)
    unsigned_manifest = {
        "schema_version": CACHE_SCHEMA,
        "declared_identity": declared_identity,
        "declared_identity_sha256": declared_hash,
        "train_matrix_semantic_sha256": _sparse_semantic_hash(x_train),
        "tune_matrix_semantic_sha256": _sparse_semantic_hash(x_tune),
        "train_shape": list(x_train.shape),
        "tune_shape": list(x_tune.shape),
        "dtype": str(x_train.dtype),
    }
    manifest = {
        **unsigned_manifest,
        "semantic_sha256": _canonical_hash(unsigned_manifest),
    }
    manifest_path = cache_directory / f"identity-{declared_hash}.json"
    _write_immutable(manifest_path, _canonical_json(manifest) + b"\n")
    identity_hash = str(manifest["semantic_sha256"])
    if option == "streaming":
        return FeatureCache(
            option, x_train, x_tune, identity_hash, device, (manifest_path,)
        )
    train = x_train.toarray().astype(np.float32, copy=False)
    tune = x_tune.toarray().astype(np.float32, copy=False)
    if option == "cpu":
        return FeatureCache(option, train, tune, identity_hash, device, (manifest_path,))
    if option == "gpu":
        return FeatureCache(
            option,
            torch.from_numpy(train).to(device),
            torch.from_numpy(tune).to(device),
            identity_hash,
            device,
            (manifest_path,),
        )
    train_path = cache_directory / f"{identity_hash}-train.npy"
    tune_path = cache_directory / f"{identity_hash}-tune.npy"
    for path, values in ((train_path, train), (tune_path, tune)):
        _publish_npy(path, values)
    return FeatureCache(
        option,
        np.load(train_path, mmap_mode="r", allow_pickle=False),
        np.load(tune_path, mmap_mode="r", allow_pickle=False),
        identity_hash,
        device,
        (manifest_path, train_path, tune_path),
    )


def estimate_memory_model(
    *, rows: int, columns: int, targets: int = 12, finalists: int = 3, seeds: int = 5
) -> dict[str, int]:
    if min(rows, columns, targets, finalists, seeds) <= 0:
        raise BackendError("memory-model dimensions must be positive")
    return {
        "dense_feature_cache_bytes": rows * columns * 4,
        "target_cache_bytes": rows * targets * 4,
        "w3_base_oof_bytes": finalists * 2 * seeds * rows * targets * 8,
        "w3_seed_mean_bytes": finalists * 2 * rows * targets * 8,
        "w3_ensemble_prediction_bytes": 2 * rows * targets * 8,
    }


def build_memory_worksheet(
    *,
    rows: int,
    columns: int,
    hidden_dims: Sequence[int],
    batch_size: int,
    validation_chunk_size: int,
    cache_option: str,
    total_ram_bytes: int,
    available_ram_bytes: int,
    total_vram_bytes: int,
    free_vram_bytes: int,
    measured_rss_baseline_bytes: int | None,
    measured_rss_peak_bytes: int | None,
    measured_gpu_baseline_bytes: int | None,
    measured_gpu_peak_bytes: int | None,
    device_type: str = "cuda",
    sparse_matrix_bytes: int = 64 * 1024**2,
    python_overhead_bytes: int = 512 * 1024**2,
    telemetry_bytes: int = 128 * 1024**2,
    framework_reservation_bytes: int = 512 * 1024**2,
    checkpoint_generations: int = 2,
    safety_margin_fraction: float = 0.20,
) -> dict[str, object]:
    """Return a conservative, measured-reconciled CPU/GPU memory worksheet."""

    dimensions = [rows, columns, batch_size, validation_chunk_size, *hidden_dims]
    if any(value <= 0 for value in dimensions) or cache_option not in {
        "streaming",
        "cpu",
        "gpu",
        "memmap",
    }:
        raise BackendError("memory worksheet dimensions/cache option are invalid")
    if device_type not in {"cpu", "cuda"} or (
        device_type == "cpu" and cache_option == "gpu"
    ):
        raise BackendError("memory worksheet device/cache placement is invalid")
    if safety_margin_fraction < 0.20:
        raise BackendError("memory worksheet safety margin must be at least 20%")
    widths = [columns, *hidden_dims, len(TARGET_COLUMNS)]
    parameters = sum(
        left * right + right for left, right in zip(widths[:-1], widths[1:], strict=True)
    )
    parameter_bytes = parameters * 4
    dense_cache = rows * columns * 4
    target_cache = rows * len(TARGET_COLUMNS) * 4
    dense_temporaries = (batch_size + validation_chunk_size) * columns * (8 + 4)
    activation_bytes = batch_size * (
        columns + 6 * sum(hidden_dims) + len(TARGET_COLUMNS)
    ) * 4
    validation_workspace = validation_chunk_size * (
        columns + 2 * sum(hidden_dims) + len(TARGET_COLUMNS)
    ) * 4
    checkpoint_bytes = checkpoint_generations * parameter_bytes
    shared_model_components = {
        "parameters": parameter_bytes,
        "gradients": parameter_bytes,
        "adamw_first_moment": parameter_bytes,
        "adamw_second_moment": parameter_bytes,
        "training_activations": activation_bytes,
        "validation_workspace": validation_workspace,
        "framework_reservation": int(framework_reservation_bytes),
        "checkpoint_clone": parameter_bytes,
        "target_batch": batch_size * len(TARGET_COLUMNS) * 4,
    }
    cpu_components = {
        "sparse_matrix": int(sparse_matrix_bytes),
        "dense_feature_cache": dense_cache if cache_option in {"cpu", "memmap"} else 0,
        "target_cache": target_cache,
        "dense_temporaries": dense_temporaries,
        "python_copies_overhead": int(python_overhead_bytes),
        "telemetry_buffers": int(telemetry_bytes),
        "checkpoint_generations": checkpoint_bytes,
        "validation_output": rows * len(TARGET_COLUMNS) * 8,
        **(
            {f"model_{name}": value for name, value in shared_model_components.items()}
            if device_type == "cpu"
            else {}
        ),
    }
    gpu_components = {
        "dense_feature_cache": dense_cache if cache_option == "gpu" else 0,
        "parameters": shared_model_components["parameters"],
        "gradients": shared_model_components["gradients"],
        "adamw_first_moment": shared_model_components["adamw_first_moment"],
        "adamw_second_moment": shared_model_components["adamw_second_moment"],
        "training_activations": shared_model_components["training_activations"],
        "dense_batch_temporary": batch_size * columns * 4,
        "validation_workspace": shared_model_components["validation_workspace"],
        "framework_cuda_reservation": shared_model_components["framework_reservation"],
        "checkpoint_clone": shared_model_components["checkpoint_clone"],
        "target_batch": shared_model_components["target_batch"],
    }
    if device_type == "cpu":
        gpu_components = {name: 0 for name in gpu_components}
    cpu_estimate = sum(cpu_components.values())
    gpu_estimate = sum(gpu_components.values())
    cpu_required = int(np.ceil(cpu_estimate * (1 + safety_margin_fraction)))
    gpu_required = int(np.ceil(gpu_estimate * (1 + safety_margin_fraction)))

    def measured_delta(baseline: int | None, peak: int | None) -> int | None:
        if baseline is None or peak is None or min(baseline, peak) < 0 or peak < baseline:
            return None
        return peak - baseline

    cpu_delta = measured_delta(measured_rss_baseline_bytes, measured_rss_peak_bytes)
    gpu_delta = measured_delta(measured_gpu_baseline_bytes, measured_gpu_peak_bytes)
    cpu_sensor_available = cpu_delta is not None
    gpu_sensor_required = device_type == "cuda"
    gpu_sensor_available = gpu_delta is not None
    sensors_available = cpu_sensor_available and (
        gpu_sensor_available or not gpu_sensor_required
    )
    reconciliation = {
        "measured_rss_delta_bytes": cpu_delta,
        "measured_gpu_reserved_delta_bytes": gpu_delta,
        "cpu_measured_over_estimate": (
            float(cpu_delta / cpu_estimate) if cpu_delta is not None and cpu_estimate else None
        ),
        "gpu_measured_over_estimate": (
            float(gpu_delta / gpu_estimate) if gpu_delta is not None and gpu_estimate else None
        ),
        "estimate_covers_measured": bool(
            sensors_available
            and cpu_delta <= cpu_estimate
            and (not gpu_sensor_required or gpu_delta <= gpu_estimate)
        ),
    }
    capacity = {
        "total_ram_bytes": int(total_ram_bytes),
        "available_ram_bytes": int(available_ram_bytes),
        "total_vram_bytes": int(total_vram_bytes),
        "free_vram_bytes": int(free_vram_bytes),
        "cpu_required_with_margin_bytes": cpu_required,
        "gpu_required_with_margin_bytes": gpu_required,
        "cpu_remaining_after_plan_bytes": available_ram_bytes - cpu_required,
        "gpu_remaining_after_plan_bytes": free_vram_bytes - gpu_required,
    }
    approved = bool(
        sensors_available
        and reconciliation["estimate_covers_measured"]
        and available_ram_bytes >= cpu_required
        and (device_type == "cpu" or free_vram_bytes >= gpu_required)
        and available_ram_bytes - cpu_required >= int(0.20 * total_ram_bytes)
        and (
            device_type == "cpu"
            or free_vram_bytes - gpu_required >= int(0.20 * total_vram_bytes)
        )
    )
    unsigned = {
        "schema_version": "qm9-model-selection-v31-memory-worksheet-v1",
        "device_type": device_type,
        "cache_option": cache_option,
        "parameters": parameters,
        "cpu_components": cpu_components,
        "gpu_components": gpu_components,
        "safety_margin_fraction": safety_margin_fraction,
        "reconciliation": reconciliation,
        "capacity": capacity,
        "sensor_availability": {
            "rss_baseline_and_peak": cpu_sensor_available,
            "gpu_baseline_and_peak": gpu_sensor_available,
            "gpu_sensor_required": gpu_sensor_required,
        },
        "approved": approved,
    }
    return {**unsigned, "semantic_sha256": _canonical_hash(unsigned)}


def assess_runtime_feasibility(
    family_tier_seconds: Mapping[str, Sequence[float]],
    planned_fit_counts: Mapping[str, int],
    *,
    ceiling_seconds: float = 34_080.0,
    minimum_uncertainty_fraction: float = 0.20,
) -> dict[str, object]:
    """Conservatively project all planned fits with sampling uncertainty."""

    if ceiling_seconds != 34_080.0 or minimum_uncertainty_fraction < 0.20:
        raise BackendError("runtime feasibility ceiling/uncertainty contract differs")
    if set(family_tier_seconds) != set(planned_fit_counts) or not family_tier_seconds:
        raise BackendError("runtime timing tiers and planned counts differ")
    required_families = {"ridge", "xgboost", "mlp"}
    observed_families = {key.split("|", 1)[0] for key in family_tier_seconds}
    if not required_families.issubset(observed_families):
        raise BackendError("runtime feasibility lacks a required model family")
    tiers: dict[str, object] = {}
    base_seconds = 0.0
    uncertainty = minimum_uncertainty_fraction
    for key in sorted(family_tier_seconds):
        samples = np.asarray(family_tier_seconds[key], dtype=np.float64)
        count = planned_fit_counts[key]
        if samples.ndim != 1 or len(samples) < 2 or np.any(~np.isfinite(samples)) or np.any(
            samples <= 0
        ):
            raise BackendError(f"runtime tier {key} needs at least two finite positive samples")
        if type(count) is not int or count < 1:
            raise BackendError(f"runtime planned count is invalid: {key}")
        representative = float(np.quantile(samples, 0.95))
        coefficient_of_variation = float(samples.std(ddof=1) / samples.mean())
        uncertainty = max(uncertainty, coefficient_of_variation)
        projected = representative * count
        base_seconds += projected
        tiers[key] = {
            "samples": samples.tolist(),
            "sample_count": len(samples),
            "p95_seconds": representative,
            "coefficient_of_variation": coefficient_of_variation,
            "planned_fits": count,
            "projected_seconds": projected,
        }
    upper = base_seconds * (1 + uncertainty)
    unsigned = {
        "schema_version": "qm9-model-selection-v31-runtime-feasibility-v1",
        "tiers": tiers,
        "base_projected_seconds": base_seconds,
        "uncertainty_fraction": uncertainty,
        "upper_bound_seconds": upper,
        "ceiling_seconds": ceiling_seconds,
        "approved": bool(upper <= ceiling_seconds),
    }
    return {**unsigned, "semantic_sha256": _canonical_hash(unsigned)}


def _validate_decision_artifact(
    artifact: Mapping[str, object], *, schema_version: str, name: str
) -> dict[str, object]:
    payload = dict(artifact)
    recorded = payload.pop("semantic_sha256", None)
    if artifact.get("schema_version") != schema_version or recorded != _canonical_hash(payload):
        raise BackendError(f"{name} artifact schema/semantic hash differs")
    if type(artifact.get("approved")) is not bool:
        raise BackendError(f"{name} artifact approval field differs")
    return dict(artifact)


def _validate_actual_hashes(actual_hashes: Mapping[str, object]) -> dict[str, str]:
    required = {
        "plan_sha256",
        "config_sha256",
        "input_sha256",
        "profile_sha256",
        "code_sha256",
    }
    if set(actual_hashes) != required:
        raise BackendError("feasibility actual-hash schema differs")
    validated: dict[str, str] = {}
    for name in sorted(required):
        value = actual_hashes[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or value == "0" * 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise BackendError(f"feasibility actual hash is invalid: {name}")
        validated[name] = value
    return validated


def _device_identity_sha256(device_identity: Mapping[str, object]) -> str:
    if not device_identity or any(not isinstance(key, str) for key in device_identity):
        raise BackendError("feasibility device identity is invalid")
    try:
        return _canonical_hash(dict(device_identity))
    except (TypeError, ValueError) as error:
        raise BackendError("feasibility device identity is not canonical JSON") from error


def freeze_feasibility_artifact(
    path: str | Path,
    memory_worksheet: Mapping[str, object],
    runtime_feasibility: Mapping[str, object],
    *,
    device_identity: Mapping[str, object],
    actual_hashes: Mapping[str, object],
) -> dict[str, object]:
    """Freeze reviewed memory/runtime gates and the exact device/input bindings."""

    memory = _validate_decision_artifact(
        memory_worksheet,
        schema_version="qm9-model-selection-v31-memory-worksheet-v1",
        name="memory worksheet",
    )
    runtime = _validate_decision_artifact(
        runtime_feasibility,
        schema_version="qm9-model-selection-v31-runtime-feasibility-v1",
        name="runtime feasibility",
    )
    device = dict(device_identity)
    device_sha256 = _device_identity_sha256(device)
    hashes = _validate_actual_hashes(actual_hashes)
    unsigned = {
        "schema_version": FEASIBILITY_SCHEMA,
        "memory": memory,
        "memory_sha256": memory["semantic_sha256"],
        "runtime": runtime,
        "runtime_sha256": runtime["semantic_sha256"],
        "device_identity": device,
        "device_identity_sha256": device_sha256,
        "actual_hashes": hashes,
        "approved": bool(memory["approved"] and runtime["approved"]),
        "scientific_result": False,
        "outer_validation_targets_read": False,
        "test_targets_read": False,
    }
    artifact = {**unsigned, "semantic_sha256": _canonical_hash(unsigned)}
    _write_immutable(Path(path), _canonical_json(artifact) + b"\n")
    return artifact


def load_feasibility_artifact(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_memory_sha256: str,
    expected_runtime_sha256: str,
    expected_device_identity_sha256: str,
) -> dict[str, object]:
    """Load a frozen feasibility decision and fail closed on any mismatch/refusal."""

    for name, value in {
        "expected_file_sha256": expected_file_sha256,
        "expected_memory_sha256": expected_memory_sha256,
        "expected_runtime_sha256": expected_runtime_sha256,
        "expected_device_identity_sha256": expected_device_identity_sha256,
    }.items():
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise BackendError(f"{name} is not a lowercase SHA-256 digest")
    artifact_path = Path(path)
    if artifact_file_sha256(artifact_path) != expected_file_sha256:
        raise BackendError("feasibility artifact file hash differs")
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BackendError("feasibility artifact cannot be loaded") from error
    if not isinstance(artifact, dict):
        raise BackendError("feasibility artifact schema differs")
    unsigned = dict(artifact)
    recorded = unsigned.pop("semantic_sha256", None)
    required = {
        "schema_version",
        "memory",
        "memory_sha256",
        "runtime",
        "runtime_sha256",
        "device_identity",
        "device_identity_sha256",
        "actual_hashes",
        "approved",
        "scientific_result",
        "outer_validation_targets_read",
        "test_targets_read",
        "semantic_sha256",
    }
    if (
        set(artifact) != required
        or artifact.get("schema_version") != FEASIBILITY_SCHEMA
        or recorded != _canonical_hash(unsigned)
    ):
        raise BackendError("feasibility artifact schema/semantic hash differs")
    memory = _validate_decision_artifact(
        artifact["memory"],
        schema_version="qm9-model-selection-v31-memory-worksheet-v1",
        name="memory worksheet",
    )
    runtime = _validate_decision_artifact(
        artifact["runtime"],
        schema_version="qm9-model-selection-v31-runtime-feasibility-v1",
        name="runtime feasibility",
    )
    _validate_actual_hashes(artifact["actual_hashes"])
    device_sha256 = _device_identity_sha256(artifact["device_identity"])
    if (
        artifact["memory_sha256"] != expected_memory_sha256
        or memory["semantic_sha256"] != expected_memory_sha256
        or artifact["runtime_sha256"] != expected_runtime_sha256
        or runtime["semantic_sha256"] != expected_runtime_sha256
        or artifact["device_identity_sha256"] != expected_device_identity_sha256
        or device_sha256 != expected_device_identity_sha256
    ):
        raise BackendError("feasibility artifact reviewed binding differs")
    if (
        artifact.get("approved") is not True
        or memory.get("approved") is not True
        or runtime.get("approved") is not True
    ):
        raise BackendError("feasibility artifact is not approved")
    for field in (
        "scientific_result",
        "outer_validation_targets_read",
        "test_targets_read",
    ):
        if artifact.get(field) is not False:
            raise BackendError(f"feasibility artifact {field} boundary differs")
    return artifact


def choose_cache_option(
    profiles: Mapping[str, Mapping[str, object]],
    *,
    total_vram_bytes: int,
    free_vram_bytes: int,
    total_ram_bytes: int,
    available_ram_bytes: int,
    free_disk_bytes: int,
    cache_bytes: int,
) -> str:
    """Apply the intake's deterministic target-free cache choice thresholds."""

    _validate_profiles(profiles)
    reference = profiles.get("streaming")
    if reference is None or not bool(reference.get("equivalent", False)):
        raise BackendError("streaming reference profile is missing or invalid")
    reference_median = float(reference["median_batch_ms"])

    def valid(name: str) -> bool:
        return name in profiles and bool(profiles[name].get("equivalent", False))

    if valid("gpu") and bool(profiles["gpu"].get("gpu_execution_approved", False)):
        gpu = profiles["gpu"]
        improvement = (reference_median - float(gpu["median_batch_ms"])) / reference_median
        remaining = free_vram_bytes - int(gpu["peak_incremental_bytes"])
        if (
            float(gpu["build_seconds"]) <= 30
            and float(gpu["p95_data_ready_ms"]) <= 0.50
            and improvement >= 0.25
            and int(gpu["peak_reserved_bytes"]) <= 0.70 * total_vram_bytes
            and remaining >= max(2 * 1024**3, int(0.25 * total_vram_bytes))
        ):
            return "gpu"
    if valid("cpu"):
        cpu = profiles["cpu"]
        remaining = available_ram_bytes - int(cpu["peak_incremental_bytes"])
        if (
            float(cpu["build_seconds"]) <= 30
            and float(cpu["projected_savings_seconds"])
            >= 1.25 * float(cpu["build_seconds"])
            and int(cpu.get("swap_growth_bytes", 0)) <= 256 * 1024**2
            and remaining >= max(8 * 1024**3, int(0.25 * total_ram_bytes))
        ):
            return "cpu"
    if valid("memmap"):
        memmap = profiles["memmap"]
        cpu_p95 = float(profiles.get("cpu", {}).get("p95_data_ready_ms", np.inf))
        if (
            free_disk_bytes >= int(1.25 * cache_bytes) + 2 * 1024**3
            and float(memmap["p95_data_ready_ms"]) <= 2.0 * cpu_p95
            and float(memmap["projected_savings_seconds"])
            >= 1.25 * float(memmap["build_seconds"])
        ):
            return "memmap"
    return "streaming"


def _validate_profiles(profiles: Mapping[str, Mapping[str, object]]) -> None:
    required_numeric = {
        "build_seconds",
        "median_batch_ms",
        "p95_data_ready_ms",
        "validation_data_ready_seconds",
        "peak_incremental_bytes",
        "peak_reserved_bytes",
        "baseline_reserved_bytes",
        "measured_reserved_delta_bytes",
        "projected_savings_seconds",
        "swap_growth_bytes",
    }
    if not profiles or "streaming" not in profiles:
        raise BackendError("cache profiles must include streaming")
    identities: set[str] = set()
    for name, row in profiles.items():
        if row.get("schema_version") != PROFILE_SCHEMA or row.get("option") != name:
            raise BackendError("cache profile schema/option differs")
        if type(row.get("equivalent")) is not bool or type(
            row.get("validation_equivalent")
        ) is not bool:
            raise BackendError("cache profile equivalence fields differ")
        identity = row.get("identity_sha256")
        if not isinstance(identity, str) or len(identity) != 64:
            raise BackendError("cache profile identity is invalid")
        identities.add(identity)
        for field in required_numeric:
            value = row.get(field)
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise BackendError(f"cache profile numeric field is invalid: {field}")
        if type(row.get("gpu_execution_approved")) is not bool:
            raise BackendError("cache profile GPU approval field differs")
    if len(identities) != 1:
        raise BackendError("cache profiles do not share one actual matrix identity")


def freeze_cache_choice(
    path: str | Path,
    profiles: Mapping[str, Mapping[str, object]],
    *,
    total_vram_bytes: int,
    free_vram_bytes: int,
    total_ram_bytes: int,
    available_ram_bytes: int,
    free_disk_bytes: int,
    cache_bytes: int,
    device_identity: Mapping[str, object],
) -> dict[str, object]:
    """Freeze the profile and deterministic cache choice as one immutable artifact."""

    _validate_profiles(profiles)
    resources = {
        "total_vram_bytes": int(total_vram_bytes),
        "free_vram_bytes": int(free_vram_bytes),
        "total_ram_bytes": int(total_ram_bytes),
        "available_ram_bytes": int(available_ram_bytes),
        "free_disk_bytes": int(free_disk_bytes),
        "cache_bytes": int(cache_bytes),
    }
    if any(value < 0 for value in resources.values()):
        raise BackendError("cache choice resource value is negative")
    chosen = choose_cache_option(profiles, **resources)
    unsigned = {
        "schema_version": CACHE_CHOICE_SCHEMA,
        "profiles": {name: dict(row) for name, row in sorted(profiles.items())},
        "profiles_sha256": _canonical_hash(
            {name: dict(row) for name, row in sorted(profiles.items())}
        ),
        "resources": resources,
        "device_identity": dict(device_identity),
        "chosen_option": chosen,
        "gpu_approved": bool(
            chosen == "gpu" and profiles["gpu"].get("gpu_execution_approved", False)
        ),
        "scientific_result": False,
        "outer_validation_targets_read": False,
        "test_targets_read": False,
    }
    artifact = {**unsigned, "semantic_sha256": _canonical_hash(unsigned)}
    _write_immutable(Path(path), _canonical_json(artifact) + b"\n")
    return artifact


def load_cache_choice(path: str | Path) -> dict[str, object]:
    try:
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BackendError("cache choice artifact cannot be loaded") from error
    if not isinstance(artifact, dict) or artifact.get("schema_version") != CACHE_CHOICE_SCHEMA:
        raise BackendError("cache choice schema differs")
    unsigned = dict(artifact)
    recorded = unsigned.pop("semantic_sha256", None)
    if recorded != _canonical_hash(unsigned):
        raise BackendError("cache choice semantic hash differs")
    profiles = artifact.get("profiles")
    if not isinstance(profiles, dict):
        raise BackendError("cache choice profiles are missing")
    _validate_profiles(profiles)
    if artifact.get("profiles_sha256") != _canonical_hash(
        {name: dict(row) for name, row in sorted(profiles.items())}
    ):
        raise BackendError("cache choice profile hash differs")
    expected = choose_cache_option(profiles, **artifact["resources"])
    if artifact.get("chosen_option") != expected:
        raise BackendError("cache choice differs from deterministic thresholds")
    if expected == "gpu" and artifact.get("gpu_approved") is not True:
        raise BackendError("GPU cache path was not explicitly approved")
    return artifact


def profile_cache_options(
    x_train: sparse.csr_matrix,
    x_tune: sparse.csr_matrix,
    *,
    batch_indices: np.ndarray,
    identity: Mapping[str, object],
    device: object,
    directory: str | Path,
    samples: int = 20,
    epochs_cap: int = 20,
    batches_per_epoch: int | None = None,
) -> dict[str, dict[str, object]]:
    """Run a bounded, target-free cache microprofile and equivalence check."""

    if samples < 3 or epochs_cap < 1:
        raise BackendError("cache profile sample/epoch counts are invalid")
    epoch_batches = batches_per_epoch or int(np.ceil(x_train.shape[0] / len(batch_indices)))
    if epoch_batches < 1:
        raise BackendError("cache profile batches_per_epoch is invalid")
    options = ["streaming", "cpu", "memmap"]
    if getattr(device, "type", None) == "cuda":
        options.insert(1, "gpu")
    profiles: dict[str, dict[str, object]] = {}
    reference_batches: list[str] | None = None
    reference_validation: np.ndarray | None = None
    for option in options:
        baseline_reserved = 0
        baseline_rss = _current_rss_bytes()
        if getattr(device, "type", None) == "cuda":
            import torch

            torch.cuda.synchronize(device)
            baseline_reserved = int(torch.cuda.memory_reserved(device))
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        cache = build_feature_cache(
            x_train,
            x_tune,
            option=option,
            identity=identity,
            device=device,
            directory=directory,
        )
        build_seconds = time.perf_counter() - started
        elapsed: list[float] = []
        batch_hashes: list[str] = []
        try:
            for iteration in range(samples + 3):
                timings: dict[str, float] = {}
                before = time.perf_counter()
                tensor = cache.batch(batch_indices, timings, sink=None, trial=None)
                _sync(device)
                if iteration >= 3:
                    elapsed.append((time.perf_counter() - before) * 1000)
                batch_hashes.append(_array_hash(tensor.detach().cpu().numpy()))
            if reference_batches is None:
                reference_batches = batch_hashes
            equivalent = batch_hashes == reference_batches
            validation_started = time.perf_counter()
            validation_timings: dict[str, float] = {}
            validation_values: list[np.ndarray] = []
            for _, validation_tensor in cache.validation_chunks(
                4096, validation_timings, sink=None, trial=None
            ):
                validation_values.append(validation_tensor.detach().cpu().numpy())
            _sync(device)
            validation_seconds = time.perf_counter() - validation_started
            validation = np.concatenate(validation_values, axis=0)
            if reference_validation is None:
                reference_validation = validation
            validation_equivalent = bool(np.array_equal(reference_validation, validation))
            incremental = int(x_train.shape[0] + x_tune.shape[0]) * x_train.shape[1] * 4
            peak_reserved = baseline_reserved
            if getattr(device, "type", None) == "cuda":
                import torch

                peak_reserved = int(torch.cuda.max_memory_reserved(device))
            rss_delta = max(0, _current_rss_bytes() - baseline_rss)
            profiles[option] = {
                "schema_version": PROFILE_SCHEMA,
                "option": option,
                "equivalent": equivalent,
                "validation_equivalent": validation_equivalent,
                "identity_sha256": cache.identity_sha256,
                "build_seconds": build_seconds,
                "median_batch_ms": float(np.median(elapsed)),
                "p95_data_ready_ms": float(np.quantile(elapsed, 0.95)),
                "validation_data_ready_seconds": validation_seconds,
                "peak_incremental_bytes": (
                    max(0, peak_reserved - baseline_reserved)
                    if option == "gpu"
                    else max(rss_delta, incremental if option in {"cpu", "memmap"} else 0)
                ),
                "peak_reserved_bytes": peak_reserved,
                "baseline_reserved_bytes": baseline_reserved,
                "measured_reserved_delta_bytes": max(0, peak_reserved - baseline_reserved),
                "projected_savings_seconds": 0.0,
                "swap_growth_bytes": 0,
                "gpu_execution_approved": bool(
                    option == "gpu" and getattr(device, "type", None) == "cuda"
                ),
            }
        finally:
            cache.close()
    reference_median = float(profiles["streaming"]["median_batch_ms"])
    reference_validation = float(profiles["streaming"]["validation_data_ready_seconds"])
    for row in profiles.values():
        row["projected_savings_seconds"] = max(
            0.0,
            epochs_cap
            * (
                epoch_batches
                * (reference_median - float(row["median_batch_ms"]))
                / 1000
                + reference_validation - float(row["validation_data_ready_seconds"])
            ),
        )
    return profiles


def _current_rss_bytes() -> int:
    try:
        pages = int(Path("/proc/self/statm").read_text(encoding="utf-8").split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        return 0


def _target_scale(y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(y_train, dtype=np.float64)
    _finite(values, "train targets")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    if values.ndim != 2 or values.shape[1] != len(TARGET_COLUMNS) or np.any(scale <= 0):
        raise BackendError("target scale is invalid")
    return mean, scale


def fit_ridge_profiled(
    x_train: sparse.csr_matrix,
    y_train: np.ndarray,
    x_tune: sparse.csr_matrix,
    *,
    alpha: float,
    sink: EventSink | None = None,
    guard: RunGuard | None = None,
    trial: object | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    timings: dict[str, float] = {}
    _finite(x_train, "Ridge train features")
    _finite(x_tune, "Ridge tune features")
    mean, scale = _target_scale(y_train)
    _guard(guard, "ridge-before-fit")
    with _stage("forward", timings, sink=sink, trial=trial):
        model = Ridge(alpha=alpha, solver="lsqr", tol=1e-4, max_iter=10_000).fit(
            x_train, (y_train - mean) / scale
        )
    _guard(guard, "ridge-after-fit")
    _finite(model.coef_, "ridge coefficients")
    _finite(model.intercept_, "ridge intercept")
    with _stage("predict", timings, sink=sink, trial=trial):
        prediction = model.predict(x_tune) * scale + mean
    _finite(prediction, "ridge prediction")
    return prediction, {"backend": "sklearn-ridge", "stage_seconds": timings}


def fit_xgboost_profiled(
    x_train: sparse.csr_matrix,
    y_train: np.ndarray,
    x_tune: sparse.csr_matrix,
    y_tune: np.ndarray,
    candidate: Mapping[str, Any],
    *,
    seed: int,
    early_stopping_rounds: int,
    sink: EventSink | None = None,
    guard: RunGuard | None = None,
    trial: object | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    from xgboost import XGBRegressor
    from xgboost.callback import TrainingCallback

    _finite(x_train, "XGBoost train features")
    _finite(x_tune, "XGBoost tune features")
    _finite(y_train, "XGBoost train targets")
    _finite(y_tune, "XGBoost tune targets")
    timings: dict[str, float] = {}
    predictions: list[np.ndarray] = []
    rounds: list[int] = []

    class GuardCallback(TrainingCallback):
        def __init__(self, target: int) -> None:
            self.target = target

        def after_iteration(
            self, model: object, epoch: int, evals_log: Mapping[str, Any]
        ) -> bool:
            _guard(guard, f"xgboost-target-{self.target}-round-{epoch}")
            values = [series[-1] for group in evals_log.values() for series in group.values()]
            if values and not np.all(np.isfinite(values)):
                raise BackendError("XGBoost evaluation metric is nonfinite")
            _emit(
                sink,
                "xgb_iteration",
                trial=trial,
                progress={"target": self.target, "round": epoch + 1},
                metrics={"evaluation": float(values[-1]) if values else None},
            )
            return False

    params = {key: value for key, value in candidate.items() if key != "id"}
    for target in range(len(TARGET_COLUMNS)):
        _guard(guard, f"xgboost-target-{target}-start")
        model = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=1,
            random_state=seed + target,
            early_stopping_rounds=early_stopping_rounds,
            callbacks=[GuardCallback(target)],
            **params,
        )
        with _stage("forward", timings, sink=sink, trial=trial):
            model.fit(
                x_train,
                y_train[:, target],
                eval_set=[(x_tune, y_tune[:, target])],
                verbose=False,
            )
        with _stage("predict", timings, sink=sink, trial=trial):
            prediction = np.asarray(model.predict(x_tune), dtype=np.float64)
        _finite(prediction, f"XGBoost target {target} prediction")
        predictions.append(prediction)
        rounds.append(int(getattr(model, "best_iteration", params["n_estimators"] - 1)) + 1)
        _guard(guard, f"xgboost-target-{target}-complete")
    return np.column_stack(predictions), {
        "backend": "xgboost",
        "per_target_rounds": rounds,
        "stage_seconds": timings,
    }


def _torch_state(model: object) -> dict[str, object]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _torch_predict(
    model: object,
    cache: FeatureCache,
    *,
    chunk_size: int,
    timings: dict[str, float],
    sink: EventSink | None,
    trial: object | None,
) -> np.ndarray:
    import torch

    if not 1 <= chunk_size <= 4096:
        raise BackendError("validation chunk size must be between 1 and 4096")
    output: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for _, tensor in cache.validation_chunks(
            chunk_size, timings, sink=sink, trial=trial
        ):
            _finite(tensor, "validation input")
            with _stage("validation", timings, sink=sink, trial=trial, device=cache.device):
                prediction = model(tensor)
            _finite(prediction, "standardized validation prediction")
            output.append(prediction.detach().cpu().numpy())
    return np.concatenate(output, axis=0)


def fit_torch_mlp_profiled(
    x_train: sparse.csr_matrix,
    y_train: np.ndarray,
    x_tune: sparse.csr_matrix,
    y_tune: np.ndarray,
    candidate: Mapping[str, Any],
    training: Mapping[str, Any],
    monitoring: Mapping[str, Any],
    *,
    seed: int,
    sink: EventSink | None = None,
    guard: RunGuard | None = None,
    checkpoint_store: CheckpointStore | None = None,
    trial: object | None = None,
    cache_option: str = "streaming",
    cache_identity: Mapping[str, object] | None = None,
    cache_directory: str | Path | None = None,
    device: str | object | None = None,
    validation_chunk_size: int = 4096,
    expected_cache_identity_sha256: str | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    import torch

    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    selected_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    mean, scale = _target_scale(y_train)
    _finite(y_tune, "tune targets")
    if cache_identity is None or cache_directory is None:
        raise BackendError("authenticated cache identity and directory are mandatory")
    identity = cache_identity
    timings: dict[str, float] = {}
    with _stage("feature_transform", timings, sink=sink, trial=trial):
        cache = build_feature_cache(
            x_train,
            x_tune,
            option=cache_option,
            identity=identity,
            device=selected_device,
            directory=cache_directory,
        )
    if (
        expected_cache_identity_sha256 is not None
        and cache.identity_sha256 != expected_cache_identity_sha256
    ):
        cache.close()
        raise BackendError("fit cache identity differs from frozen cache choice")
    try:
        widths = [x_train.shape[1], *candidate["hidden_dims"], len(TARGET_COLUMNS)]
        layers: list[Any] = []
        for left, right in zip(widths[:-2], widths[1:-1], strict=True):
            layers.extend(
                [
                    torch.nn.Linear(left, right),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(float(candidate["dropout"])),
                ]
            )
        layers.append(torch.nn.Linear(widths[-2], widths[-1]))
        model = torch.nn.Sequential(*layers).to(selected_device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(candidate["learning_rate"]),
            weight_decay=float(candidate["weight_decay"]),
        )
        initial_state_hash = canonical_tensor_state_sha256(model.state_dict())
        batch_size = min(int(training["batch_size"]), len(y_train))
        max_epochs = int(training["max_epochs"])
        patience = int(training["patience"])
        min_delta = float(training["min_delta"])
        curves: dict[str, list[object]] = {
            "train_standardized_mse": [],
            "inner_tune_mnmae": [],
            "inner_tune_per_target_nmae": [],
            "gradient_norm": [],
            "parameter_norm": [],
        }
        first_batch: dict[str, object] | None = None
        best_score = float("inf")
        best_epoch = 0
        best_state: dict[str, object] | None = None
        best_state_hash: str | None = None
        stale = 0
        stop_reason = "max-epochs-reached"
        validation_increases = 0
        maximum_validation_increases = 0
        low_gradient_batches = 0
        for epoch in range(max_epochs):
            _guard(guard, f"mlp-epoch-{epoch + 1}-start")
            model.train()
            order = np.random.default_rng(seed + epoch).permutation(len(y_train))
            losses: list[float] = []
            gradients: list[float] = []
            for start in range(0, len(order), batch_size):
                _guard(guard, f"mlp-epoch-{epoch + 1}-batch-{start // batch_size + 1}")
                batch = np.asarray(order[start : start + batch_size], dtype=np.int64)
                xb = cache.batch(batch, timings, sink=sink, trial=trial)
                with _stage("tensor_create", timings, sink=sink, trial=trial):
                    standardized_y = ((y_train[batch] - mean) / scale).astype(np.float32)
                    yb = torch.from_numpy(standardized_y)
                with _stage(
                    "host_to_device", timings, sink=sink, trial=trial, device=selected_device
                ):
                    yb = yb.to(selected_device)
                _finite(xb, "training input")
                _finite(yb, "standardized training target")
                optimizer.zero_grad(set_to_none=True)
                with _stage(
                    "forward", timings, sink=sink, trial=trial, device=selected_device
                ):
                    prediction = model(xb)
                with _stage("loss", timings, sink=sink, trial=trial, device=selected_device):
                    loss = torch.nn.functional.mse_loss(prediction, yb)
                _finite(loss, "MLP loss")
                with _stage(
                    "backward", timings, sink=sink, trial=trial, device=selected_device
                ):
                    loss.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        _finite(parameter.grad, "MLP gradient")
                gradient = float(
                    torch.sqrt(
                        sum(
                            (parameter.grad**2).sum()
                            for parameter in model.parameters()
                            if parameter.grad is not None
                        )
                    )
                )
                if gradient > float(monitoring.get("gradient_norm_hard_max", np.inf)):
                    raise BackendError("MLP gradient exceeds hard maximum")
                if gradient < float(monitoring.get("gradient_norm_hard_min", -np.inf)):
                    low_gradient_batches += 1
                else:
                    low_gradient_batches = 0
                if low_gradient_batches >= int(
                    monitoring.get("gradient_min_consecutive_batches", 2**31)
                ):
                    raise BackendError("MLP gradient stayed below hard minimum")
                with _stage(
                    "optimizer", timings, sink=sink, trial=trial, device=selected_device
                ):
                    optimizer.step()
                for parameter in model.parameters():
                    _finite(parameter, "MLP parameter")
                loss_value = float(loss.detach())
                losses.append(loss_value)
                gradients.append(gradient)
                if first_batch is None:
                    first_batch = {
                        "indices_sha256": _array_hash(batch),
                        "input_sha256": _array_hash(xb.detach().cpu().numpy()),
                        "target_sha256": _array_hash(yb.detach().cpu().numpy()),
                        "prediction_sha256": _array_hash(prediction.detach().cpu().numpy()),
                        "loss": loss_value,
                        "gradient_norm": gradient,
                        "post_step_state_sha256": canonical_tensor_state_sha256(
                            model.state_dict()
                        ),
                    }
            standardized_tune = _torch_predict(
                model,
                cache,
                chunk_size=validation_chunk_size,
                timings=timings,
                sink=sink,
                trial=trial,
            )
            tune_prediction = standardized_tune * scale + mean
            _finite(tune_prediction, "inverse-scaled validation prediction")
            per_target = np.mean(np.abs(tune_prediction - y_tune), axis=0) / scale
            score = float(per_target.mean())
            train_loss = float(np.mean(losses))
            curves["train_standardized_mse"].append(train_loss)
            curves["inner_tune_mnmae"].append(score)
            curves["inner_tune_per_target_nmae"].append(per_target.tolist())
            curves["gradient_norm"].append(max(gradients))
            parameter_norm = float(
                torch.sqrt(sum((parameter.detach() ** 2).sum() for parameter in model.parameters()))
            )
            curves["parameter_norm"].append(parameter_norm)
            if epoch and score > float(curves["inner_tune_mnmae"][-2]):
                validation_increases += 1
            else:
                validation_increases = 0
            maximum_validation_increases = max(
                maximum_validation_increases, validation_increases
            )
            if epoch == 4 and score > float(
                monitoring.get("max_validation_mnmae_after_five_epochs", np.inf)
            ):
                raise BackendError("MLP epoch-five MNMAE threshold exceeded")
            if epoch == 9:
                first = float(curves["train_standardized_mse"][0])
                improvement = (first - min(curves["train_standardized_mse"])) / max(
                    abs(first), 1e-12
                )
                if improvement < float(
                    monitoring.get("minimum_relative_train_improvement_by_epoch_ten", -np.inf)
                ):
                    raise BackendError("MLP epoch-ten improvement threshold missed")
            if score < best_score - min_delta:
                best_score = score
                best_epoch = epoch + 1
                stale = 0
                best_state = _torch_state(model)
                best_state_hash = canonical_tensor_state_sha256(best_state)
                if checkpoint_store is not None:
                    if trial is None:
                        raise BackendError("checkpoint store requires a trial identity")
                    with _stage("checkpoint", timings, sink=sink, trial=trial):
                        checkpoint_store.save_tensor_state(
                            trial,
                            state_kind="best",
                            state=best_state,
                            payload={
                                "epoch": best_epoch,
                                "inner_tune_mnmae": best_score,
                                "cache_identity_sha256": cache.identity_sha256,
                            },
                        )
            else:
                stale += 1
            _emit(
                sink,
                "epoch_end",
                trial=trial,
                progress={
                    "unit": "epoch",
                    "current": epoch + 1,
                    "total": max_epochs,
                    "best_epoch": best_epoch,
                    "stale": stale,
                },
                metrics={
                    "train_standardized_mse": train_loss,
                    "inner_tune_mnmae": score,
                    "per_target_nmae": per_target.tolist(),
                    "gradient_norm": max(gradients),
                    "parameter_norm": parameter_norm,
                },
                timings=timings,
            )
            _guard(guard, f"mlp-epoch-{epoch + 1}-complete")
            if stale >= patience:
                stop_reason = "patience-exhausted"
                break
        if best_state is None or best_state_hash is None:
            raise BackendError("MLP produced no best checkpoint")
        final_training_state_hash = canonical_tensor_state_sha256(model.state_dict())
        model.load_state_dict(best_state)
        restored_hash = canonical_tensor_state_sha256(model.state_dict())
        if restored_hash != best_state_hash:
            raise BackendError("restored checkpoint hash differs from best")
        standardized_final = _torch_predict(
            model,
            cache,
            chunk_size=validation_chunk_size,
            timings=timings,
            sink=sink,
            trial=trial,
        )
        final = standardized_final * scale + mean
        _finite(final, "final MLP prediction")
        from .model_selection_v3 import monitor_curve

        monitor_thresholds = {
            "validation_increase_mark_after": 2,
            "gradient_norm_hard_max": np.inf,
            "gradient_norm_hard_min": -np.inf,
            "gradient_min_consecutive_batches": 2**31,
            "target_degradation_warning": np.inf,
        }
        monitor_thresholds.update(monitoring)
        curve = monitor_curve(
            curves["train_standardized_mse"],
            curves["inner_tune_mnmae"],
            max_epochs=max_epochs,
            restored_hash=restored_hash,
            best_hash=best_state_hash,
            thresholds=monitor_thresholds,
            per_target_nmae=curves["inner_tune_per_target_nmae"],
            gradient_norm=curves["gradient_norm"],
            parameter_norm=curves["parameter_norm"],
        )
        curve["stop_reason"] = stop_reason
        curve["restored_epoch"] = best_epoch
        return final, {
            "backend": "torch",
            "device": str(selected_device),
            "cache_option": cache_option,
            "cache_identity_sha256": cache.identity_sha256,
            "validation_chunk_size": validation_chunk_size,
            "initial_state_sha256": initial_state_hash,
            "first_batch": first_batch,
            "best_epoch": best_epoch,
            "best_checkpoint_sha256": best_state_hash,
            "restored_checkpoint_sha256": restored_hash,
            "final_training_state_sha256": final_training_state_hash,
            "stop_reason": stop_reason,
            "maximum_consecutive_validation_increases": maximum_validation_increases,
            "curves": curves,
            "curve": curve,
            "determinism": {
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
                "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
            },
            "stage_seconds": timings,
        }
    finally:
        cache.close()


def fit_torch_mlp_streaming_reference(*args: object, **kwargs: object) -> tuple[np.ndarray, dict]:
    kwargs["cache_option"] = "streaming"
    return fit_torch_mlp_profiled(*args, **kwargs)  # type: ignore[arg-type]


def profile_real_backend(
    candidate: Mapping[str, Any],
    x_train: sparse.csr_matrix,
    y_train: np.ndarray,
    x_tune: sparse.csr_matrix,
    y_tune: np.ndarray,
    *,
    seed: int,
    sink: EventSink | None,
    guard: RunGuard | None,
    checkpoint_store: CheckpointStore | None,
    trial: object | None = None,
    cache_identity: Mapping[str, object] | None = None,
    cache_directory: str | Path | None = None,
    cache_choice_path: str | Path | None = None,
    completion_contract: CompletionContract | None = None,
    device: str | object | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Dispatch one real, train-only backend with v3.1 instrumentation."""

    family = str(candidate["family"])
    params = candidate.get("params", candidate)
    if family == "ridge":
        prediction, metadata = fit_ridge_profiled(
            x_train,
            y_train,
            x_tune,
            alpha=float(params["alpha"]),
            sink=sink,
            guard=guard,
            trial=trial,
        )
    elif family == "xgboost":
        prediction, metadata = fit_xgboost_profiled(
            x_train,
            y_train,
            x_tune,
            y_tune,
            params,
            seed=seed,
            early_stopping_rounds=int(candidate.get("early_stopping_rounds", 35)),
            sink=sink,
            guard=guard,
            trial=trial,
        )
    elif family == "mlp":
        if cache_choice_path is None:
            raise BackendError("MLP dispatcher requires a frozen cache-choice artifact")
        cache_choice = load_cache_choice(cache_choice_path)
        cache_option = str(cache_choice["chosen_option"])
        profile_rows = cache_choice["profiles"]
        expected_identity = str(profile_rows["streaming"]["identity_sha256"])
        prediction, metadata = fit_torch_mlp_profiled(
            x_train,
            y_train,
            x_tune,
            y_tune,
            params,
            candidate["training"],
            candidate["monitoring"],
            seed=seed,
            sink=sink,
            guard=guard,
            checkpoint_store=checkpoint_store,
            trial=trial,
            cache_option=cache_option,
            cache_identity=cache_identity,
            cache_directory=cache_directory,
            device=device,
            expected_cache_identity_sha256=expected_identity,
        )
    else:
        raise BackendError(f"unsupported backend family: {family}")
    _finite(prediction, "backend prediction")
    if prediction.shape != (len(y_tune), len(TARGET_COLUMNS)):
        raise BackendError("backend prediction shape differs")
    if checkpoint_store is not None:
        if trial is None or completion_contract is None:
            raise BackendError("checkpoint completion requires trial and completion contracts")
        completed_path = checkpoint_store.mark_trial_complete(
            trial,
            record={
                "status": "COMPLETE",
                "candidate_id": candidate.get("id", family),
                "family": family,
                "seed": seed,
                "metadata_sha256": _canonical_hash(metadata),
            },
            prediction=prediction,
            completion=completion_contract,
        )
        checkpoint_evidence = checkpoint_store.validate_completed_checkpoint(
            completed_path,
            trial=trial,
            completion=completion_contract,
        )
        _emit(
            sink,
            "checkpoint_saved",
            trial=trial,
            progress={
                "status": "CHECKPOINT_READY",
                "checkpoint": completed_path.name,
                "prediction_shape": list(prediction.shape),
            },
            metrics={"prediction_finite": True},
            force_fsync=True,
        )
        metadata["checkpoint_evidence"] = checkpoint_evidence
    metadata["prediction_sha256"] = canonical_prediction_sha256(
        np.asarray(prediction, dtype=np.float64),
        {"candidate_id": candidate.get("id", family), "family": family, "seed": seed},
    )
    return prediction, metadata


def backend_hook_contract() -> dict[str, object]:
    """Machine-readable supervisor coordination surface for every real family."""

    return {
        "schema_version": "qm9-model-selection-v31-backend-hooks-v1",
        "families": {
            "ridge": {
                "safe_points": ["before-fit", "after-fit"],
                "stages": ["forward", "predict"],
            },
            "xgboost": {
                "safe_points": ["target-start", "every-round", "target-complete"],
                "stages": ["forward", "predict"],
                "target_estimators": len(TARGET_COLUMNS),
            },
            "mlp": {
                "safe_points": ["epoch-start", "every-batch", "epoch-complete"],
                "stages": [
                    "feature_transform",
                    "sparse_slice",
                    "to_dense",
                    "tensor_create",
                    "host_to_device",
                    "cache_gather",
                    "forward",
                    "loss",
                    "backward",
                    "optimizer",
                    "validation",
                    "checkpoint",
                ],
            },
        },
        "completion_requires": [
            "durable_checkpoint",
            "CompletionContract",
            "supervisor_authenticated_complete_transition",
        ],
        "outer_validation_targets_read": False,
        "test_targets_read": False,
    }


def _worker_trial_payload(trial: Mapping[str, object]) -> dict[str, object]:
    required = {
        "run_id",
        "wave",
        "strategy",
        "candidate_id",
        "family",
        "fold",
        "seed",
        "trial_index",
        "trial_total",
    }
    payload = dict(trial)
    if set(payload) != required:
        raise BackendError("worker trial identity schema differs")
    for field in ("run_id", "wave", "strategy", "candidate_id", "family"):
        if not isinstance(payload[field], str) or not payload[field]:
            raise BackendError(f"worker trial identity is invalid: {field}")
    for field in ("fold", "seed", "trial_index", "trial_total"):
        if type(payload[field]) is not int:
            raise BackendError(f"worker trial integer is invalid: {field}")
    if (
        payload["fold"] < 0
        or payload["trial_index"] < 1
        or payload["trial_total"] < payload["trial_index"]
    ):
        raise BackendError("worker trial indices are invalid")
    return payload


def _contained_path(root: Path, path: str | Path, name: str) -> Path:
    resolved_root = root.resolve()
    candidate = Path(path)
    resolved = (
        (resolved_root / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise BackendError(f"worker {name} must be contained by run_root") from error
    return resolved


def _validate_worker_candidate(candidate: Mapping[str, object]) -> dict[str, object]:
    payload = dict(candidate)
    family = payload.get("family")
    if family not in {"ridge", "xgboost", "mlp"} or not isinstance(payload.get("id"), str):
        raise BackendError("worker candidate family/id is invalid")
    params = payload.get("params")
    if not isinstance(params, Mapping):
        raise BackendError("worker candidate params are missing")
    if family == "ridge":
        if set(params) != {"alpha"} or float(params["alpha"]) <= 0:
            raise BackendError("worker Ridge candidate exceeds bounded schema")
    elif family == "xgboost":
        estimators = params.get("n_estimators")
        if type(estimators) is not int or not 1 <= estimators <= 8:
            raise BackendError("worker XGBoost n_estimators must be in [1, 8]")
        early_stopping = payload.get("early_stopping_rounds", 2)
        if type(early_stopping) is not int or not 1 <= early_stopping <= 4:
            raise BackendError("worker XGBoost early stopping must be in [1, 4]")
    else:
        training = payload.get("training")
        monitoring = payload.get("monitoring")
        if not isinstance(training, Mapping) or not isinstance(monitoring, Mapping):
            raise BackendError("worker MLP training/monitoring schema is missing")
        epochs = training.get("max_epochs")
        batch_size = training.get("batch_size")
        if type(epochs) is not int or not 1 <= epochs <= 3:
            raise BackendError("worker MLP max_epochs must be in [1, 3]")
        if type(batch_size) is not int or not 1 <= batch_size <= 4096:
            raise BackendError("worker MLP batch_size is invalid")
    return payload


def write_real_backend_worker_spec(
    path: str | Path,
    *,
    run_root: str | Path,
    data_path: str | Path,
    data_scope: str,
    result_path: str | Path,
    profile_path: str | Path | None = None,
    checkpoint_root: str | Path,
    candidate: Mapping[str, object],
    trial: Mapping[str, object],
    completion: CompletionContract,
    checkpoint_bindings: Mapping[str, object],
    cache_identity: Mapping[str, object] | None = None,
    cache_directory: str | Path | None = None,
    cache_choice_path: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, object]:
    """Write one immutable, bounded synthetic/public real-backend worker spec."""

    root = Path(run_root).resolve()
    if not root.is_dir():
        raise BackendError("worker run_root must already exist")
    spec_path = _contained_path(root, path, "spec path")
    result = _contained_path(root, result_path, "result path")
    profile = _contained_path(
        root,
        profile_path
        if profile_path is not None
        else result.with_name(f"{result.stem}.profile.json"),
        "profile path",
    )
    checkpoints = _contained_path(root, checkpoint_root, "checkpoint root")
    cache = (
        _contained_path(root, cache_directory, "cache directory")
        if cache_directory is not None
        else None
    )
    choice = (
        _contained_path(root, cache_choice_path, "cache choice")
        if cache_choice_path is not None
        else None
    )
    source = Path(data_path).resolve()
    if data_scope not in {"synthetic", "public-fixture"}:
        raise BackendError("worker data_scope must be synthetic or public-fixture")
    if not source.is_file():
        raise BackendError("worker data artifact is missing")
    normalized_candidate = _validate_worker_candidate(candidate)
    normalized_trial = _worker_trial_payload(trial)
    if (
        normalized_trial["candidate_id"] != normalized_candidate["id"]
        or normalized_trial["family"] != normalized_candidate["family"]
    ):
        raise BackendError("worker candidate differs from trial identity")
    required_bindings = {
        "plan_sha256",
        "config_sha256",
        "input_sha256",
        "cache_sha256",
        "run_start_sha256",
    }
    if set(checkpoint_bindings) != required_bindings:
        raise BackendError("worker checkpoint binding schema differs")
    validated_bindings = _validate_actual_hashes(
        {
            "plan_sha256": checkpoint_bindings["plan_sha256"],
            "config_sha256": checkpoint_bindings["config_sha256"],
            "input_sha256": checkpoint_bindings["input_sha256"],
            "profile_sha256": checkpoint_bindings["cache_sha256"],
            "code_sha256": checkpoint_bindings["run_start_sha256"],
        }
    )
    bindings = {
        "plan_sha256": validated_bindings["plan_sha256"],
        "config_sha256": validated_bindings["config_sha256"],
        "input_sha256": validated_bindings["input_sha256"],
        "cache_sha256": validated_bindings["profile_sha256"],
        "run_start_sha256": validated_bindings["code_sha256"],
    }
    if bindings["input_sha256"] != artifact_file_sha256(source):
        raise BackendError("worker input binding differs from actual data artifact")
    family = normalized_candidate["family"]
    if family == "mlp" and (cache_identity is None or cache is None or choice is None):
        raise BackendError("worker MLP requires cache identity/directory/frozen choice")
    unsigned = {
        "schema_version": REAL_BACKEND_WORKER_SPEC_SCHEMA,
        "run_root": str(root),
        "data_path": str(source),
        "data_file_sha256": artifact_file_sha256(source),
        "data_scope": data_scope,
        "result_path": str(result.relative_to(root)),
        "profile_path": str(profile.relative_to(root)),
        "checkpoint_root": str(checkpoints.relative_to(root)),
        "candidate": normalized_candidate,
        "trial": normalized_trial,
        "completion": completion.payload(),
        "checkpoint_bindings": bindings,
        "cache_identity": dict(cache_identity) if cache_identity is not None else None,
        "cache_directory": str(cache.relative_to(root)) if cache is not None else None,
        "cache_choice_path": str(choice.relative_to(root)) if choice is not None else None,
        "cache_choice_file_sha256": artifact_file_sha256(choice) if choice is not None else None,
        "device": device,
        "protected_labels_allowed": False,
        "outer_validation_targets_read": False,
        "test_targets_read": False,
    }
    artifact = {**unsigned, "semantic_sha256": _canonical_hash(unsigned)}
    _write_immutable(spec_path, _canonical_json(artifact) + b"\n")
    return artifact


def load_real_backend_worker_spec(path: str | Path) -> dict[str, object]:
    """Authenticate a bounded worker spec and all immutable path/file bindings."""

    try:
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BackendError("worker spec cannot be loaded") from error
    if not isinstance(artifact, dict):
        raise BackendError("worker spec schema differs")
    required = {
        "schema_version",
        "run_root",
        "data_path",
        "data_file_sha256",
        "data_scope",
        "result_path",
        "profile_path",
        "checkpoint_root",
        "candidate",
        "trial",
        "completion",
        "checkpoint_bindings",
        "cache_identity",
        "cache_directory",
        "cache_choice_path",
        "cache_choice_file_sha256",
        "device",
        "protected_labels_allowed",
        "outer_validation_targets_read",
        "test_targets_read",
        "semantic_sha256",
    }
    unsigned = dict(artifact)
    recorded = unsigned.pop("semantic_sha256", None)
    if (
        set(artifact) != required
        or artifact.get("schema_version") != REAL_BACKEND_WORKER_SPEC_SCHEMA
        or recorded != _canonical_hash(unsigned)
    ):
        raise BackendError("worker spec schema/semantic hash differs")
    for field in (
        "protected_labels_allowed",
        "outer_validation_targets_read",
        "test_targets_read",
    ):
        if artifact.get(field) is not False:
            raise BackendError(f"worker spec {field} boundary differs")
    if artifact.get("data_scope") not in {"synthetic", "public-fixture"}:
        raise BackendError("worker data_scope differs")
    root = Path(artifact["run_root"]).resolve()
    _contained_path(root, path, "spec path")
    result = _contained_path(root, artifact["result_path"], "result path")
    profile = _contained_path(root, artifact["profile_path"], "profile path")
    checkpoint_root = _contained_path(root, artifact["checkpoint_root"], "checkpoint root")
    if result.exists():
        raise BackendError("worker result path must be fresh")
    if profile.exists():
        raise BackendError("worker profile path must be fresh")
    source = Path(artifact["data_path"]).resolve()
    if artifact_file_sha256(source) != artifact["data_file_sha256"]:
        raise BackendError("worker data artifact hash differs")
    candidate = _validate_worker_candidate(artifact["candidate"])
    trial = _worker_trial_payload(artifact["trial"])
    if trial["candidate_id"] != candidate["id"] or trial["family"] != candidate["family"]:
        raise BackendError("worker candidate/trial binding differs")
    trial_checkpoint_directory = checkpoint_root / trial_identity_sha256(trial)
    if trial_checkpoint_directory.exists() and any(trial_checkpoint_directory.iterdir()):
        raise BackendError("worker trial checkpoint directory must be fresh")
    completion = artifact["completion"]
    if not isinstance(completion, Mapping):
        raise BackendError("worker completion contract is missing")
    CompletionContract(
        expected_shape=tuple(completion["expected_shape"]),
        target_order=tuple(completion["target_order"]),
        prediction_order_sha256=completion["prediction_order_sha256"],
        start_ledger_entry_sha256=completion["start_ledger_entry_sha256"],
    )
    bindings = artifact["checkpoint_bindings"]
    if not isinstance(bindings, Mapping):
        raise BackendError("worker checkpoint bindings are missing")
    if set(bindings) != {
        "plan_sha256",
        "config_sha256",
        "input_sha256",
        "cache_sha256",
        "run_start_sha256",
    }:
        raise BackendError("worker checkpoint binding schema differs")
    _validate_actual_hashes(
        {
            "plan_sha256": bindings["plan_sha256"],
            "config_sha256": bindings["config_sha256"],
            "input_sha256": bindings["input_sha256"],
            "profile_sha256": bindings["cache_sha256"],
            "code_sha256": bindings["run_start_sha256"],
        }
    )
    if bindings["input_sha256"] != artifact["data_file_sha256"]:
        raise BackendError("worker input binding differs from actual data artifact")
    if candidate["family"] == "mlp":
        if artifact["cache_identity"] is None:
            raise BackendError("worker MLP cache identity is missing")
        _validate_cache_identity(artifact["cache_identity"])
        _contained_path(root, artifact["cache_directory"], "cache directory")
        choice = _contained_path(root, artifact["cache_choice_path"], "cache choice")
        if artifact_file_sha256(choice) != artifact["cache_choice_file_sha256"]:
            raise BackendError("worker frozen cache-choice hash differs")
        load_cache_choice(choice)
    return artifact


def load_real_backend_profile(
    path: str | Path, *, expected_metadata_sha256: str
) -> dict[str, object]:
    """Authenticate public/synthetic backend timing/curve evidence separately from completion."""

    try:
        artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BackendError("worker profile artifact cannot be loaded") from error
    required = {
        "schema_version",
        "trial",
        "family",
        "metadata",
        "metadata_sha256",
        "scientific_result",
        "publication_ready",
        "outer_validation_targets_read",
        "test_targets_read",
        "semantic_sha256",
    }
    if not isinstance(artifact, dict) or set(artifact) != required:
        raise BackendError("worker profile artifact schema differs")
    unsigned = dict(artifact)
    recorded = unsigned.pop("semantic_sha256")
    if (
        artifact.get("schema_version") != REAL_BACKEND_PROFILE_SCHEMA
        or recorded != _canonical_hash(unsigned)
        or artifact.get("metadata_sha256") != expected_metadata_sha256
        or _canonical_hash(artifact.get("metadata")) != expected_metadata_sha256
    ):
        raise BackendError("worker profile artifact semantic/metadata hash differs")
    for field in (
        "scientific_result",
        "publication_ready",
        "outer_validation_targets_read",
        "test_targets_read",
    ):
        if artifact.get(field) is not False:
            raise BackendError(f"worker profile {field} boundary differs")
    _worker_trial_payload(artifact["trial"])
    return artifact


def _worker_heartbeat(stop: threading.Event, path: Path) -> None:
    _ensure_directory(path.parent)
    while not stop.is_set():
        path.touch(exist_ok=True)
        stop.wait(1.0)


def run_real_backend_worker(spec_path: str | Path) -> dict[str, object]:
    """Execute one bounded real backend and publish only authenticated checkpoint evidence."""

    spec = load_real_backend_worker_spec(spec_path)
    root = Path(spec["run_root"]).resolve()
    heartbeat_value = os.environ.get("QM9_V31_HEARTBEAT_PATH")
    if not heartbeat_value:
        raise BackendError("worker heartbeat path is not configured")
    heartbeat_path = _contained_path(root, heartbeat_value, "heartbeat path")
    stop = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_worker_heartbeat, args=(stop, heartbeat_path), daemon=True
    )
    heartbeat_thread.start()
    try:
        with np.load(spec["data_path"], allow_pickle=False) as payload:
            if set(payload.files) != {"x_train", "y_train", "x_tune", "y_tune"}:
                raise BackendError("worker data schema differs or contains protected labels")
            x_train_values = np.asarray(payload["x_train"])
            y_train = np.asarray(payload["y_train"], dtype=np.float64)
            x_tune_values = np.asarray(payload["x_tune"])
            y_tune = np.asarray(payload["y_tune"], dtype=np.float64)
        if (
            x_train_values.ndim != 2
            or x_tune_values.ndim != 2
            or y_train.shape != (len(x_train_values), len(TARGET_COLUMNS))
            or y_tune.shape != (len(x_tune_values), len(TARGET_COLUMNS))
            or x_train_values.shape[1] != x_tune_values.shape[1]
            or max(len(x_train_values), len(x_tune_values)) > 4096
            or x_train_values.shape[1] > 4096
        ):
            raise BackendError("worker data dimensions exceed bounded public-fixture contract")
        completion_payload = spec["completion"]
        completion = CompletionContract(
            expected_shape=tuple(completion_payload["expected_shape"]),
            target_order=tuple(completion_payload["target_order"]),
            prediction_order_sha256=completion_payload["prediction_order_sha256"],
            start_ledger_entry_sha256=completion_payload["start_ledger_entry_sha256"],
        )
        if completion.expected_shape != (len(y_tune), len(TARGET_COLUMNS)):
            raise BackendError("worker completion shape differs from public fixture")
        bindings = spec["checkpoint_bindings"]
        store = CheckpointStore(
            _contained_path(root, spec["checkpoint_root"], "checkpoint root"),
            plan_sha256=bindings["plan_sha256"],
            config_sha256=bindings["config_sha256"],
            input_sha256=bindings["input_sha256"],
            cache_sha256=bindings["cache_sha256"],
            run_start_sha256=bindings["run_start_sha256"],
        )
        _, metadata = profile_real_backend(
            spec["candidate"],
            sparse.csr_matrix(x_train_values),
            y_train,
            sparse.csr_matrix(x_tune_values),
            y_tune,
            seed=int(spec["trial"]["seed"]),
            sink=None,
            guard=None,
            checkpoint_store=store,
            trial=spec["trial"],
            cache_identity=spec["cache_identity"],
            cache_directory=(
                _contained_path(root, spec["cache_directory"], "cache directory")
                if spec["cache_directory"] is not None
                else None
            ),
            cache_choice_path=(
                _contained_path(root, spec["cache_choice_path"], "cache choice")
                if spec["cache_choice_path"] is not None
                else None
            ),
            completion_contract=completion,
            device=spec["device"],
        )
        evidence = metadata.get("checkpoint_evidence")
        if not isinstance(evidence, Mapping):
            raise BackendError("worker backend returned no checkpoint evidence")
        checkpoint_path = Path(store.root) / str(evidence["relative_path"])
        authenticated = store.validate_completed_checkpoint(
            checkpoint_path, trial=spec["trial"], completion=completion
        )
        if dict(evidence) != authenticated:
            raise BackendError("worker checkpoint evidence changed after backend return")
        metadata_sha256 = _canonical_hash(metadata)
        profile_unsigned = {
            "schema_version": REAL_BACKEND_PROFILE_SCHEMA,
            "trial": dict(spec["trial"]),
            "family": spec["candidate"]["family"],
            "metadata": metadata,
            "metadata_sha256": metadata_sha256,
            "scientific_result": False,
            "publication_ready": False,
            "outer_validation_targets_read": False,
            "test_targets_read": False,
        }
        profile_artifact = {
            **profile_unsigned,
            "semantic_sha256": _canonical_hash(profile_unsigned),
        }
        profile_path = _contained_path(root, spec["profile_path"], "profile path")
        _write_immutable(profile_path, _canonical_json(profile_artifact) + b"\n")
        load_real_backend_profile(
            profile_path, expected_metadata_sha256=metadata_sha256
        )
        result = {
            "status": "COMPLETE",
            "checkpoint_evidence": authenticated,
            "metadata_sha256": metadata_sha256,
        }
        result_path = _contained_path(root, spec["result_path"], "result path")
        _write_immutable(result_path, _canonical_json(result) + b"\n")
        return result
    finally:
        stop.set()
        heartbeat_thread.join(timeout=2.0)


def _main() -> int:
    parser = argparse.ArgumentParser(description="Run one bounded QM9 v3.1 real-backend worker")
    parser.add_argument("--worker-spec", required=True)
    arguments = parser.parse_args()
    run_real_backend_worker(arguments.worker_spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
