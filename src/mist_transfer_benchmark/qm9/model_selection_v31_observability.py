"""Live, non-scientific telemetry for QM9 model-selection v3.1.

This module deliberately has no data-loader or model-selection API.  It owns an
append-only event stream that can be inspected while a training worker is busy.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Mapping, MutableMapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

EVENT_SCHEMA = "qm9-model-selection-v31-event-v1"
EVENT_VOCABULARY = frozenset(
    {
        "run_start",
        "trial_reserved",
        "trial_start",
        "stage_start",
        "stage_end",
        "heartbeat",
        "batch_end",
        "epoch_end",
        "xgb_iteration",
        "ridge_complete",
        "checkpoint_saved",
        "warning",
        "soft_drain_requested",
        "hard_stop_requested",
        "signal_received",
        "external_termination",
        "trial_complete",
        "trial_failed",
        "failure",
        "run_complete",
    }
)
TERMINAL_EVENTS = frozenset({"failure", "run_complete"})
_DURABLE_EVENTS = {
    "failure",
    "signal_received",
    "hard_stop_requested",
    "soft_drain_requested",
    "trial_complete",
    "run_complete",
}


class ObservabilityError(ValueError):
    """Raised when telemetry would violate its durable schema."""


def fsync_parent(path: str | Path) -> None:
    """Durably commit a created/replaced directory entry on local POSIX filesystems."""

    parent = Path(path).parent
    descriptor = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_sha256(value: str, name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ObservabilityError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _assert_finite_json(value: object, path: str = "event") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ObservabilityError(f"{path} contains a nonfinite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ObservabilityError(f"{path} contains a non-string object key")
            _assert_finite_json(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_finite_json(item, f"{path}[{index}]")


def _validate_event_payload(payload: Mapping[str, object]) -> None:
    event = payload.get("event")
    trial = payload.get("trial")
    run_level = {"run_start", "run_complete", "failure"}
    trial_level = {
        "trial_reserved",
        "trial_start",
        "stage_start",
        "stage_end",
        "batch_end",
        "epoch_end",
        "xgb_iteration",
        "ridge_complete",
        "checkpoint_saved",
        "trial_complete",
        "trial_failed",
        "external_termination",
    }
    if event in run_level and trial is not None:
        raise ObservabilityError(f"{event} must be a run-level event")
    if event in trial_level and not isinstance(trial, Mapping):
        raise ObservabilityError(f"{event} requires a trial identity")
    progress = payload.get("progress")
    if not isinstance(progress, Mapping):
        raise ObservabilityError("event progress must be an object")
    if event in {"stage_start", "stage_end"} and not isinstance(progress.get("stage"), str):
        raise ObservabilityError("stage event requires a stage name")
    if event == "epoch_end":
        metrics = payload.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ObservabilityError("epoch_end requires metrics")
        per_target = metrics.get("per_target_nmae")
        if not isinstance(per_target, list) or len(per_target) != 12:
            raise ObservabilityError("epoch_end requires exactly 12 per-target metrics")
    if event in {
        "hard_stop_requested",
        "signal_received",
        "external_termination",
        "trial_failed",
        "failure",
    } and not isinstance(payload.get("failure"), Mapping):
        raise ObservabilityError(f"{event} requires failure evidence")


@dataclass(frozen=True)
class TrialIdentity:
    run_id: str
    wave: str
    strategy: str
    candidate_id: str
    family: str
    fold: int
    seed: int
    trial_index: int
    trial_total: int

    def payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("run_id")
        return payload


class ResourceProvider(Protocol):
    def sample(self) -> Mapping[str, object]: ...


class SystemResourceProvider:
    """Collect current process, disk, and optional CUDA resource values."""

    def __init__(self, disk_path: str | Path, *, gpu_temperature_timeout: float = 1.0) -> None:
        self.disk_path = Path(disk_path)
        self.gpu_temperature_timeout = float(gpu_temperature_timeout)

    @staticmethod
    def _rss_bytes() -> int:
        try:
            resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
            return resident_pages * os.sysconf("SC_PAGE_SIZE")
        except (OSError, IndexError, ValueError):
            peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            return peak * (1 if os.uname().sysname == "Darwin" else 1024)

    @staticmethod
    def _cuda_memory() -> tuple[int | None, int | None]:
        try:
            import torch

            if torch.cuda.is_available():
                return int(torch.cuda.memory_allocated()), int(torch.cuda.memory_reserved())
        except (ImportError, RuntimeError):
            pass
        return None, None

    def _gpu_temperature(self) -> float | None:
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=self.gpu_temperature_timeout,
            )
            first = completed.stdout.strip().splitlines()[0]
            return float(first)
        except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError):
            return None

    def sample(self) -> Mapping[str, object]:
        allocated, reserved = self._cuda_memory()
        disk = shutil.disk_usage(self.disk_path)
        return {
            "rss_bytes": self._rss_bytes(),
            "gpu_allocated_bytes": allocated,
            "gpu_reserved_bytes": reserved,
            "gpu_temperature_c": self._gpu_temperature(),
            "disk_free_bytes": int(disk.free),
        }


class EventSink:
    """Thread-safe, single-handle writer for a hash-chained JSONL stream."""

    def __init__(
        self,
        path: str | Path,
        run_id: str,
        *,
        run_start_sha256: str | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
        utc_now: Callable[[], datetime] | None = None,
        fsync_interval_seconds: float = 5.0,
    ) -> None:
        if not run_id:
            raise ObservabilityError("run_id must be nonempty")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.run_start_sha256 = _require_sha256(
            run_start_sha256
            or hashlib.sha256(_canonical_json({"legacy_run_id": run_id})).hexdigest(),
            "run_start_sha256",
        )
        self._monotonic_ns = monotonic_ns
        self._utc_now = utc_now or (lambda: datetime.now(UTC))
        self._fsync_interval_ns = int(float(fsync_interval_seconds) * 1_000_000_000)
        self._lock = threading.Lock()
        if self.path.is_file() and self.path.stat().st_size:
            recover_truncated_event_stream(self.path)
        existing = validate_event_stream(self.path) if self.path.is_file() else []
        if existing and any(row["run_id"] != run_id for row in existing):
            raise ObservabilityError("existing event stream belongs to another run")
        if existing and any(row["run_start_sha256"] != self.run_start_sha256 for row in existing):
            raise ObservabilityError("existing event stream has another run-start binding")
        if existing and existing[-1]["event"] in TERMINAL_EVENTS:
            raise ObservabilityError("terminal event stream is immutable")
        created = not self.path.exists()
        self._handle = self.path.open("a", encoding="utf-8")
        if created:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            fsync_parent(self.path)
        self._sequence = len(existing)
        self._previous_hash: str | None = str(existing[-1]["event_sha256"]) if existing else None
        self._last_fsync_ns = self._monotonic_ns()
        self._last_emit_ns = self._last_fsync_ns
        self._last_heartbeat_ns = self._last_fsync_ns
        self._closed = False
        self._terminal = False

    @property
    def last_emit_monotonic_ns(self) -> int:
        with self._lock:
            return self._last_emit_ns

    @property
    def last_heartbeat_monotonic_ns(self) -> int:
        with self._lock:
            return self._last_heartbeat_ns

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def emit(
        self,
        event: str,
        *,
        trial: TrialIdentity | None,
        progress: Mapping[str, object],
        metrics: Mapping[str, object] | None = None,
        timings: Mapping[str, float] | None = None,
        resources: Mapping[str, object] | None = None,
        warning: Mapping[str, object] | None = None,
        failure: Mapping[str, object] | None = None,
        force_fsync: bool = False,
    ) -> dict[str, object]:
        if event not in EVENT_VOCABULARY:
            raise ObservabilityError(f"event is outside the reviewed vocabulary: {event}")
        if trial is not None and trial.run_id != self.run_id:
            raise ObservabilityError("trial run_id differs from event stream run_id")
        with self._lock:
            if self._closed:
                raise ObservabilityError("event stream is closed")
            if self._previous_hash is not None and getattr(self, "_terminal", False):
                raise ObservabilityError("cannot append after a terminal event")
            now_ns = self._monotonic_ns()
            now = (
                self._utc_now()
                .astimezone(UTC)
                .isoformat(timespec="microseconds")
                .replace("+00:00", "Z")
            )
            payload: dict[str, object] = {
                "schema_version": EVENT_SCHEMA,
                "sequence": self._sequence + 1,
                "timestamp_utc": now,
                "monotonic_ns": now_ns,
                "run_id": self.run_id,
                "run_start_sha256": self.run_start_sha256,
                "event": event,
                "event_id": str(uuid.uuid4()),
                "prev_event_sha256": self._previous_hash,
                "scientific_result": False,
                "publication_ready": False,
                "outer_validation_targets_read": False,
                "test_targets_read": False,
                "trial": trial.payload() if trial is not None else None,
                "progress": dict(progress),
            }
            optional = {
                "metrics": metrics,
                "timings": timings,
                "resources": resources,
                "warning": warning,
                "failure": failure,
            }
            payload.update(
                {key: dict(value) for key, value in optional.items() if value is not None}
            )
            _validate_event_payload(payload)
            _assert_finite_json(payload)
            payload["event_sha256"] = hashlib.sha256(_canonical_json(payload)).hexdigest()
            line = _canonical_json(payload).decode("utf-8") + "\n"
            self._handle.write(line)
            self._handle.flush()
            should_fsync = (
                force_fsync
                or event in _DURABLE_EVENTS
                or now_ns - self._last_fsync_ns >= self._fsync_interval_ns
            )
            if should_fsync:
                os.fsync(self._handle.fileno())
                self._last_fsync_ns = now_ns
            self._sequence += 1
            self._previous_hash = str(payload["event_sha256"])
            self._last_emit_ns = now_ns
            if event == "heartbeat":
                self._last_heartbeat_ns = now_ns
            if event in TERMINAL_EVENTS:
                self._terminal = True
            return payload

    def flush(self, *, fsync: bool = True) -> None:
        with self._lock:
            if self._closed:
                return
            self._handle.flush()
            if fsync:
                os.fsync(self._handle.fileno())
                self._last_fsync_ns = self._monotonic_ns()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._closed = True

    def __enter__(self) -> EventSink:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def validate_event_stream(
    path: str | Path, *, allow_truncated_final_line: bool = False
) -> list[dict]:
    """Validate JSON syntax, sequence, invariant flags, and the complete hash chain."""

    raw = Path(path).read_bytes()
    if raw and not raw.endswith(b"\n") and not allow_truncated_final_line:
        raise ObservabilityError("event stream has a truncated final line")
    lines = raw.splitlines()
    if raw and not raw.endswith(b"\n") and allow_truncated_final_line:
        lines = lines[:-1]
    events: list[dict] = []
    previous: str | None = None
    stable_run_id: str | None = None
    stable_start: str | None = None
    terminal_seen = False
    for expected_sequence, line in enumerate(lines, start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ObservabilityError("event stream contains invalid JSON") from error
        if event.get("schema_version") != EVENT_SCHEMA:
            raise ObservabilityError("event stream schema differs")
        if event.get("event") not in EVENT_VOCABULARY:
            raise ObservabilityError("event stream contains an unreviewed event")
        run_id = event.get("run_id")
        start_hash = event.get("run_start_sha256")
        if not isinstance(run_id, str) or not run_id:
            raise ObservabilityError("event run_id is invalid")
        if not isinstance(start_hash, str):
            raise ObservabilityError("event run-start binding is missing")
        _require_sha256(start_hash, "run_start_sha256")
        stable_run_id = stable_run_id or run_id
        stable_start = stable_start or start_hash
        if run_id != stable_run_id or start_hash != stable_start:
            raise ObservabilityError("event stream mixes run identities or start bindings")
        if terminal_seen:
            raise ObservabilityError("event stream appends after a terminal event")
        if event.get("sequence") != expected_sequence:
            raise ObservabilityError("event sequence is not contiguous")
        if event.get("prev_event_sha256") != previous:
            raise ObservabilityError("event hash chain predecessor differs")
        for field in (
            "scientific_result",
            "publication_ready",
            "outer_validation_targets_read",
            "test_targets_read",
        ):
            if event.get(field) is not False:
                raise ObservabilityError(f"event invariant differs: {field}")
        _validate_event_payload(event)
        recorded = event.pop("event_sha256", None)
        _assert_finite_json(event)
        observed = hashlib.sha256(_canonical_json(event)).hexdigest()
        event["event_sha256"] = recorded
        if recorded != observed:
            raise ObservabilityError("event hash differs")
        previous = str(recorded)
        events.append(event)
        terminal_seen = event["event"] in TERMINAL_EVENTS
    return events


def recover_truncated_event_stream(path: str | Path) -> Path | None:
    """Quarantine and remove only a non-newline-terminated final fragment.

    Supported durability assumes a local POSIX filesystem honoring file and
    parent-directory ``fsync``. Network filesystems with weaker semantics are
    outside this contract.
    """

    stream = Path(path)
    raw = stream.read_bytes()
    if not raw or raw.endswith(b"\n"):
        return None
    boundary = raw.rfind(b"\n") + 1
    prefix, tail = raw[:boundary], raw[boundary:]
    if prefix:
        temporary_prefix = stream.with_name(f".{stream.name}.recovery")
        temporary_prefix.write_bytes(prefix)
        with temporary_prefix.open("rb") as handle:
            os.fsync(handle.fileno())
        validate_event_stream(temporary_prefix)
        temporary_prefix.unlink()
    digest = hashlib.sha256(tail).hexdigest()
    quarantine = stream.with_name(f"{stream.name}.truncated-{digest}.quarantine")
    if quarantine.exists() and quarantine.read_bytes() != tail:
        raise ObservabilityError("truncated-tail quarantine collision")
    if not quarantine.exists():
        with quarantine.open("xb") as handle:
            handle.write(tail)
            handle.flush()
            os.fsync(handle.fileno())
        fsync_parent(quarantine)
    with stream.open("r+b") as handle:
        handle.truncate(boundary)
        handle.flush()
        os.fsync(handle.fileno())
    fsync_parent(stream)
    return quarantine


@contextmanager
def timed_stage(
    sink: EventSink,
    trial: TrialIdentity,
    stage: str,
    accumulator: MutableMapping[str, float],
):
    """Measure one stage and emit start/end events, including exceptional exits."""

    started = time.perf_counter()
    sink.emit("stage_start", trial=trial, progress={"stage": stage})
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        seconds = time.perf_counter() - started
        accumulator[stage] = float(accumulator.get(stage, 0.0) + seconds)
        sink.emit(
            "stage_end",
            trial=trial,
            progress={"stage": stage, "failed": failed},
            timings={"last_seconds": seconds, "cumulative_seconds": accumulator[stage]},
            force_fsync=failed,
        )


class HeartbeatEmitter:
    """Emit heartbeats from a daemon thread independently of backend progress."""

    def __init__(
        self,
        sink: EventSink,
        *,
        interval_seconds: float,
        trial_provider: Callable[[], TrialIdentity | None],
        progress_provider: Callable[[], Mapping[str, object]],
        resource_provider: ResourceProvider,
    ) -> None:
        if not 0 < interval_seconds <= 30:
            raise ObservabilityError("heartbeat interval must be in (0, 30] seconds")
        self.sink = sink
        self.interval_seconds = float(interval_seconds)
        self.trial_provider = trial_provider
        self.progress_provider = progress_provider
        self.resource_provider = resource_provider
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def emit_once(self) -> dict[str, object]:
        return self.sink.emit(
            "heartbeat",
            trial=self.trial_provider(),
            progress=self.progress_provider(),
            resources=self.resource_provider.sample(),
        )

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            try:
                self.emit_once()
            except BaseException:
                # Guard detects the stale heartbeat.  The daemon must not hide the
                # training exception path or recursively attempt telemetry writes.
                return

    def start(self) -> None:
        if self._thread is not None:
            raise ObservabilityError("heartbeat emitter was already started")
        self.emit_once()
        self._thread = threading.Thread(target=self._run, name="qm9-v31-heartbeat", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds + 0.5))
