"""Durable, hash-bound checkpoints for QM9 model-selection v3.1.

The store knows only trial identities, model state, predictions, and provenance
hashes.  It deliberately has no data-loader or protected-label interface.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path

import numpy as np

CHECKPOINT_SCHEMA = "qm9-model-selection-v31-checkpoint-v1"
TENSOR_STATE_HASH_SCHEMA = "qm9-model-selection-v31-tensor-state-v1"
PREDICTION_HASH_SCHEMA = "qm9-model-selection-v31-prediction-v1"
COMPLETION_STATUS = "COMPLETE"


class CheckpointError(ValueError):
    """Raised when checkpoint identity, integrity, or immutability fails."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CheckpointError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _fsync_directory(path: Path) -> None:
    """Durably commit directory-entry changes on local POSIX filesystems."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
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


def _atomic_bytes(path: Path, payload: bytes) -> None:
    _ensure_directory(path.parent)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: object) -> None:
    _atomic_bytes(path, _canonical_json(payload) + b"\n")


def _as_numpy_tensor(value: object) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()  # type: ignore[union-attr]
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise CheckpointError("tensor state cannot contain object arrays")
    if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
        raise CheckpointError("tensor state contains nonfinite values")
    dtype = array.dtype.newbyteorder("<")
    return np.ascontiguousarray(array.astype(dtype, copy=False))


def canonical_tensor_state_sha256(state: Mapping[str, object]) -> str:
    """Hash ordered tensor metadata and device-independent contiguous bytes."""

    digest = hashlib.sha256()
    digest.update(_canonical_json({"schema_version": TENSOR_STATE_HASH_SCHEMA}))
    if not state:
        raise CheckpointError("tensor state must not be empty")
    for name in sorted(state):
        if not isinstance(name, str) or not name:
            raise CheckpointError("tensor state keys must be nonempty strings")
        array = _as_numpy_tensor(state[name])
        digest.update(
            _canonical_json(
                {
                    "name": name,
                    "shape": list(array.shape),
                    "dtype": array.dtype.str,
                    "order": "C",
                }
            )
        )
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_prediction_sha256(prediction: np.ndarray, metadata: Mapping[str, object]) -> str:
    values = np.ascontiguousarray(prediction, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise CheckpointError("prediction must be a finite two-dimensional array")
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {
                "schema_version": PREDICTION_HASH_SCHEMA,
                "shape": list(values.shape),
                "dtype": values.dtype.str,
                "metadata": dict(metadata),
            }
        )
    )
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _trial_payload(trial: object) -> dict[str, object]:
    if isinstance(trial, Mapping):
        payload = dict(trial)
    elif is_dataclass(trial):
        payload = asdict(trial)
    else:
        fields = (
            "run_id",
            "wave",
            "strategy",
            "candidate_id",
            "family",
            "fold",
            "seed",
            "trial_index",
            "trial_total",
        )
        payload = {field: getattr(trial, field) for field in fields if hasattr(trial, field)}
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
    if set(payload) != required:
        raise CheckpointError("trial identity schema is incomplete or contains extras")
    for field in ("run_id", "wave", "strategy", "candidate_id", "family"):
        if not isinstance(payload[field], str) or not payload[field]:
            raise CheckpointError(f"trial identity {field} is invalid")
    for field in ("fold", "seed", "trial_index", "trial_total"):
        if type(payload[field]) is not int:
            raise CheckpointError(f"trial identity {field} must be an integer")
    if payload["fold"] < 0 or payload["trial_index"] < 1 or payload["trial_total"] < 1:
        raise CheckpointError("trial identity indices are invalid")
    if payload["trial_index"] > payload["trial_total"]:
        raise CheckpointError("trial_index exceeds trial_total")
    return payload


def trial_identity_sha256(trial: object) -> str:
    """Return the canonical key used by immutable plans, ledgers, and checkpoints."""

    return _canonical_hash(_trial_payload(trial))


@dataclass(frozen=True)
class CompletionContract:
    """Expected completed prediction identity supplied by the immutable ledger."""

    expected_shape: tuple[int, int]
    target_order: tuple[str, ...]
    prediction_order_sha256: str
    start_ledger_entry_sha256: str

    def __post_init__(self) -> None:
        if (
            len(self.expected_shape) != 2
            or self.expected_shape[0] < 1
            or self.expected_shape[1] < 1
        ):
            raise CheckpointError("completion expected_shape is invalid")
        if len(self.target_order) != self.expected_shape[1] or len(set(self.target_order)) != len(
            self.target_order
        ):
            raise CheckpointError("completion target_order is invalid")
        if any(not isinstance(target, str) or not target for target in self.target_order):
            raise CheckpointError("completion target_order entries are invalid")
        _require_sha256(self.prediction_order_sha256, "prediction_order_sha256")
        _require_sha256(
            self.start_ledger_entry_sha256, "start_ledger_entry_sha256"
        )

    def payload(self) -> dict[str, object]:
        return {
            "expected_shape": list(self.expected_shape),
            "target_order": list(self.target_order),
            "prediction_order_sha256": self.prediction_order_sha256,
            "start_ledger_entry_sha256": self.start_ledger_entry_sha256,
        }


@dataclass(frozen=True)
class ReplayContract:
    """Immutable plan/ledger/terminal evidence required for completed replay."""

    planned_trials: tuple[Mapping[str, object], ...]
    completed_entries: Mapping[str, Mapping[str, object]]
    ledger_sha256: str
    terminal_status: str

    def __post_init__(self) -> None:
        _require_sha256(self.ledger_sha256, "ledger_sha256")
        if self.terminal_status != COMPLETION_STATUS:
            raise CheckpointError("completed replay requires terminal status COMPLETE")

    def payload(self) -> dict[str, object]:
        return {
            "planned_trials": [_trial_payload(row) for row in self.planned_trials],
            "completed_entries": {
                key: dict(value) for key, value in sorted(self.completed_entries.items())
            },
            "ledger_sha256": self.ledger_sha256,
            "terminal_status": self.terminal_status,
        }


class CheckpointStore:
    """Atomic immutable best/completed checkpoint store with provenance binding."""

    def __init__(
        self,
        root: str | Path,
        *,
        plan_sha256: str,
        config_sha256: str,
        input_sha256: str,
        cache_sha256: str,
        run_start_sha256: str,
    ) -> None:
        self.root = Path(root)
        self.bindings = {
            "plan_sha256": _require_sha256(plan_sha256, "plan_sha256"),
            "config_sha256": _require_sha256(config_sha256, "config_sha256"),
            "input_sha256": _require_sha256(input_sha256, "input_sha256"),
            "cache_sha256": _require_sha256(cache_sha256, "cache_sha256"),
            "run_start_sha256": _require_sha256(run_start_sha256, "run_start_sha256"),
        }

    def _directory(self, trial: object) -> Path:
        identity = _trial_payload(trial)
        return self.root / _canonical_hash(identity)

    def _manifest(
        self, trial: object, kind: str, payload: Mapping[str, object]
    ) -> dict[str, object]:
        return {
            "schema_version": CHECKPOINT_SCHEMA,
            "kind": kind,
            "trial": _trial_payload(trial),
            "bindings": dict(self.bindings),
            "payload": dict(payload),
            "scientific_result": False,
            "publication_ready": False,
            "outer_validation_targets_read": False,
            "test_targets_read": False,
        }

    def _write_latest(self, trial: object, record: Mapping[str, object]) -> None:
        _atomic_json(self._directory(trial) / "latest.json", record)

    def save_trial_state(
        self,
        trial: object,
        *,
        state_kind: str,
        payload: Mapping[str, object],
        blobs: Mapping[str, bytes],
    ) -> Path:
        if not state_kind or not state_kind.replace("-", "").replace("_", "").isalnum():
            raise CheckpointError("state_kind is invalid")
        blob_rows = {
            name: {
                "sha256": hashlib.sha256(value).hexdigest(),
                "bytes": len(value),
                "base64": base64.b64encode(value).decode("ascii"),
            }
            for name, value in sorted(blobs.items())
        }
        manifest = self._manifest(trial, state_kind, {**dict(payload), "blobs": blob_rows})
        semantic = _canonical_hash(manifest)
        manifest["semantic_sha256"] = semantic
        path = self._directory(trial) / f"{state_kind}-{semantic}.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != manifest:
                raise CheckpointError("immutable trial-state checkpoint differs")
        else:
            _atomic_json(path, manifest)
        latest = {
            "kind": state_kind,
            "path": path.name,
            "semantic_sha256": semantic,
            "file_sha256": _file_sha256(path),
            "bindings": dict(self.bindings),
        }
        self._write_latest(trial, latest)
        return path

    def save_tensor_state(
        self,
        trial: object,
        *,
        state_kind: str,
        state: Mapping[str, object],
        payload: Mapping[str, object],
    ) -> Path:
        """Persist an immutable state dict without Python-list serialization."""

        state_hash = canonical_tensor_state_sha256(state)
        arrays = {name: _as_numpy_tensor(value) for name, value in sorted(state.items())}
        manifest = self._manifest(
            trial,
            state_kind,
            {
                **dict(payload),
                "tensor_state_hash_schema": TENSOR_STATE_HASH_SCHEMA,
                "tensor_state_sha256": state_hash,
                "tensor_keys": list(arrays),
            },
        )
        manifest["semantic_sha256"] = _canonical_hash(manifest)
        path = self._directory(trial) / f"{state_kind}-{state_hash}.npz"
        if not path.exists():
            _ensure_directory(path.parent)
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".npz", dir=path.parent
            )
            os.close(descriptor)
            try:
                np.savez_compressed(
                    temporary,
                    **arrays,
                    __manifest_json__=np.frombuffer(_canonical_json(manifest), dtype=np.uint8),
                )
                with open(temporary, "rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                _fsync_directory(path.parent)
            except BaseException:
                Path(temporary).unlink(missing_ok=True)
                raise
        loaded_state, loaded_manifest = self.load_tensor_state(path)
        if loaded_manifest != manifest or canonical_tensor_state_sha256(loaded_state) != state_hash:
            raise CheckpointError("immutable tensor checkpoint differs")
        self._write_latest(
            trial,
            {
                "kind": state_kind,
                "path": path.name,
                "semantic_sha256": manifest["semantic_sha256"],
                "tensor_state_sha256": state_hash,
                "file_sha256": _file_sha256(path),
                "bindings": dict(self.bindings),
            },
        )
        return path

    def load_tensor_state(
        self, path: str | Path
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        try:
            with np.load(path, allow_pickle=False) as payload:
                if "__manifest_json__" not in payload.files:
                    raise CheckpointError("tensor checkpoint manifest is missing")
                manifest = json.loads(payload["__manifest_json__"].tobytes().decode("utf-8"))
                state = {
                    key: np.asarray(payload[key]).copy()
                    for key in payload.files
                    if key != "__manifest_json__"
                }
        except (OSError, ValueError, UnicodeError, json.JSONDecodeError) as error:
            raise CheckpointError("tensor checkpoint cannot be loaded") from error
        self._validate_manifest(manifest)
        expected = manifest["payload"]["tensor_state_sha256"]
        if canonical_tensor_state_sha256(state) != expected:
            raise CheckpointError("tensor checkpoint state hash differs")
        return state, manifest

    def mark_trial_complete(
        self,
        trial: object,
        *,
        record: Mapping[str, object],
        prediction: np.ndarray,
        completion: CompletionContract,
    ) -> Path:
        identity = _trial_payload(trial)
        required_record = {"status", "candidate_id", "family", "seed", "metadata_sha256"}
        if set(record) != required_record or record.get("status") != COMPLETION_STATUS:
            raise CheckpointError("completed record schema/status differs")
        if (
            record.get("candidate_id") != identity["candidate_id"]
            or record.get("family") != identity["family"]
            or record.get("seed") != identity["seed"]
        ):
            raise CheckpointError("completed record differs from trial identity")
        _require_sha256(str(record.get("metadata_sha256")), "metadata_sha256")
        values = np.ascontiguousarray(prediction, dtype=np.float64)
        if values.shape != completion.expected_shape:
            raise CheckpointError("completed prediction shape differs from contract")
        metadata = self._manifest(
            trial,
            "completed",
            {"record": dict(record), "completion": completion.payload()},
        )
        prediction_hash = canonical_prediction_sha256(values, metadata)
        metadata["prediction_hash_schema"] = PREDICTION_HASH_SCHEMA
        metadata["prediction_sha256"] = prediction_hash
        metadata["prediction_shape"] = list(values.shape)
        metadata["semantic_sha256"] = _canonical_hash(metadata)
        path = self._directory(trial) / f"completed-{prediction_hash}.npz"
        if not path.exists():
            _ensure_directory(path.parent)
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".npz", dir=path.parent
            )
            os.close(descriptor)
            try:
                np.savez_compressed(
                    temporary,
                    prediction=values,
                    __manifest_json__=np.frombuffer(_canonical_json(metadata), dtype=np.uint8),
                )
                with open(temporary, "rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                _fsync_directory(path.parent)
            except BaseException:
                Path(temporary).unlink(missing_ok=True)
                raise
        loaded = self._load_completed_path(path)
        if loaded["manifest"] != metadata:
            raise CheckpointError("immutable completed checkpoint differs")
        self._write_latest(
            trial,
            {
                "kind": "completed",
                "path": path.name,
                "semantic_sha256": metadata["semantic_sha256"],
                "prediction_sha256": prediction_hash,
                "file_sha256": _file_sha256(path),
                "bindings": dict(self.bindings),
            },
        )
        return path

    def _validate_manifest(self, manifest: object) -> dict[str, object]:
        required = {
            "schema_version",
            "kind",
            "trial",
            "bindings",
            "payload",
            "scientific_result",
            "publication_ready",
            "outer_validation_targets_read",
            "test_targets_read",
            "semantic_sha256",
        }
        if isinstance(manifest, dict) and manifest.get("kind") == "completed":
            required |= {
                "prediction_hash_schema",
                "prediction_sha256",
                "prediction_shape",
            }
        if (
            not isinstance(manifest, dict)
            or set(manifest) != required
            or manifest.get("schema_version") != CHECKPOINT_SCHEMA
        ):
            raise CheckpointError("checkpoint schema differs")
        identity = _trial_payload(manifest.get("trial", {}))
        if manifest.get("bindings") != self.bindings:
            raise CheckpointError("checkpoint plan/config/input/cache binding differs")
        for field in (
            "scientific_result",
            "publication_ready",
            "outer_validation_targets_read",
            "test_targets_read",
        ):
            if manifest.get(field) is not False:
                raise CheckpointError(f"checkpoint {field} boundary differs")
        unsigned = dict(manifest)
        recorded = unsigned.pop("semantic_sha256")
        if _canonical_hash(unsigned) != recorded:
            raise CheckpointError("checkpoint manifest semantic hash differs")
        if manifest["kind"] == "completed":
            payload = manifest.get("payload")
            if not isinstance(payload, dict) or set(payload) != {"record", "completion"}:
                raise CheckpointError("completed checkpoint payload schema differs")
            record = payload.get("record")
            required_record = {"status", "candidate_id", "family", "seed", "metadata_sha256"}
            if (
                not isinstance(record, dict)
                or set(record) != required_record
                or record.get("status") != COMPLETION_STATUS
                or record.get("candidate_id") != identity["candidate_id"]
                or record.get("family") != identity["family"]
                or record.get("seed") != identity["seed"]
            ):
                raise CheckpointError("completed checkpoint record schema/status differs")
            _require_sha256(record["metadata_sha256"], "metadata_sha256")
            completion = payload.get("completion")
            required_completion = {
                "expected_shape",
                "target_order",
                "prediction_order_sha256",
                "start_ledger_entry_sha256",
            }
            if not isinstance(completion, dict) or set(completion) != required_completion:
                raise CheckpointError("completed checkpoint contract schema differs")
            try:
                contract = CompletionContract(
                    expected_shape=tuple(completion["expected_shape"]),
                    target_order=tuple(completion["target_order"]),
                    prediction_order_sha256=completion["prediction_order_sha256"],
                    start_ledger_entry_sha256=completion["start_ledger_entry_sha256"],
                )
            except (TypeError, CheckpointError) as error:
                raise CheckpointError("completed checkpoint contract differs") from error
            if (
                manifest.get("prediction_hash_schema") != PREDICTION_HASH_SCHEMA
                or list(contract.expected_shape) != manifest.get("prediction_shape")
            ):
                raise CheckpointError("completed checkpoint prediction schema differs")
            _require_sha256(manifest["prediction_sha256"], "prediction_sha256")
        return manifest

    def _load_completed_path(self, path: Path) -> dict[str, object]:
        try:
            with np.load(path, allow_pickle=False) as payload:
                prediction = np.asarray(payload["prediction"], dtype=np.float64).copy()
                manifest = json.loads(payload["__manifest_json__"].tobytes().decode("utf-8"))
        except (OSError, KeyError, ValueError, UnicodeError, json.JSONDecodeError) as error:
            raise CheckpointError("completed checkpoint cannot be loaded") from error
        self._validate_manifest(manifest)
        expected_parent = _canonical_hash(manifest["trial"])
        expected_name = f"completed-{manifest['prediction_sha256']}.npz"
        if path.parent.name != expected_parent or path.name != expected_name:
            raise CheckpointError("completed checkpoint path identity differs")
        metadata = dict(manifest)
        recorded_hash = metadata.pop("prediction_sha256")
        metadata.pop("prediction_hash_schema")
        metadata.pop("prediction_shape")
        metadata.pop("semantic_sha256")
        if canonical_prediction_sha256(prediction, metadata) != recorded_hash:
            raise CheckpointError("completed prediction hash differs")
        return {"manifest": manifest, "prediction": prediction, "path": str(path)}

    def validate_completed_checkpoint(
        self,
        path: str | Path,
        *,
        trial: object,
        completion: CompletionContract,
    ) -> dict[str, object]:
        """Authenticate one checkpoint before the supervisor completes its ledger row."""

        artifact = Path(path).resolve()
        root = self.root.resolve()
        try:
            relative_path = artifact.relative_to(root)
        except ValueError as error:
            raise CheckpointError("completed checkpoint is outside the checkpoint root") from error
        loaded = self._load_completed_path(artifact)
        manifest = loaded["manifest"]
        expected_trial = _trial_payload(trial)
        if manifest["trial"] != expected_trial:
            raise CheckpointError("completed checkpoint trial identity differs")
        if manifest["payload"]["completion"] != completion.payload():
            raise CheckpointError("completed checkpoint completion contract differs")
        evidence = {
            "status": COMPLETION_STATUS,
            "trial_key": trial_identity_sha256(expected_trial),
            "relative_path": relative_path.as_posix(),
            "file_sha256": _file_sha256(artifact),
            "semantic_sha256": manifest["semantic_sha256"],
            "prediction_sha256": manifest["prediction_sha256"],
            "shape": manifest["prediction_shape"],
            "target_order": list(completion.target_order),
            "prediction_order_sha256": completion.prediction_order_sha256,
            "start_ledger_entry_sha256": completion.start_ledger_entry_sha256,
        }
        return evidence

    def load_completed(
        self, plan_sha256: str, *, replay: ReplayContract
    ) -> list[dict[str, object]]:
        if _require_sha256(plan_sha256, "plan_sha256") != self.bindings["plan_sha256"]:
            raise CheckpointError("resume plan hash differs")
        planned = [_trial_payload(row) for row in replay.planned_trials]
        planned_keys = [_canonical_hash(row) for row in planned]
        if len(planned_keys) != len(set(planned_keys)):
            raise CheckpointError("immutable plan contains duplicate trial keys")
        ledger_keys = list(replay.completed_entries)
        if set(ledger_keys) != set(planned_keys):
            raise CheckpointError("completed ledger has missing or extra planned trials")
        completed = [
            self._load_completed_path(path)
            for path in sorted(self.root.glob("*/completed-*.npz"))
        ]
        trial_keys = [_canonical_hash(row["manifest"]["trial"]) for row in completed]
        if len(trial_keys) != len(set(trial_keys)):
            raise CheckpointError("duplicate completed trial checkpoint")
        if set(trial_keys) != set(planned_keys):
            raise CheckpointError("checkpoint replay has missing or extra trials")
        for row in completed:
            manifest = row["manifest"]
            trial_key = _canonical_hash(manifest["trial"])
            entry = dict(replay.completed_entries[trial_key])
            required_entry = {
                "status",
                "complete_ledger_entry_sha256",
                "start_ledger_entry_sha256",
                "checkpoint",
            }
            if set(entry) != required_entry or entry.get("status") != COMPLETION_STATUS:
                raise CheckpointError("completed ledger entry schema/status differs")
            _require_sha256(
                entry["complete_ledger_entry_sha256"], "complete_ledger_entry_sha256"
            )
            _require_sha256(
                entry["start_ledger_entry_sha256"], "start_ledger_entry_sha256"
            )
            payload = manifest["payload"]
            if payload["record"]["status"] != COMPLETION_STATUS:
                raise CheckpointError("replayed checkpoint is not COMPLETE")
            completion = payload["completion"]
            contract = CompletionContract(
                expected_shape=tuple(completion["expected_shape"]),
                target_order=tuple(completion["target_order"]),
                prediction_order_sha256=completion["prediction_order_sha256"],
                start_ledger_entry_sha256=completion["start_ledger_entry_sha256"],
            )
            checkpoint = self.validate_completed_checkpoint(
                row["path"], trial=manifest["trial"], completion=contract
            )
            if (
                entry["start_ledger_entry_sha256"]
                != completion["start_ledger_entry_sha256"]
                or entry["checkpoint"] != checkpoint
            ):
                raise CheckpointError("completed ledger/checkpoint evidence differs")
        return completed

    def latest_safe(self, trial: object) -> Mapping[str, object] | None:
        path = self._directory(trial) / "latest.json"
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise CheckpointError("latest checkpoint pointer is invalid") from error
        if record.get("bindings") != self.bindings:
            raise CheckpointError("latest checkpoint pointer binding differs")
        artifact = self._directory(trial) / str(record.get("path", ""))
        if not artifact.is_file() or _file_sha256(artifact) != record.get("file_sha256"):
            raise CheckpointError("latest checkpoint artifact differs")
        return record
