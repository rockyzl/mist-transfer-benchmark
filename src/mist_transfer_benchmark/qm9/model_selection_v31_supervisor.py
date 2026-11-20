"""Governed run state and external worker sidecar for QM9 v3.1.

The supervisor has no dataset or model API.  It binds a precomputed trial plan,
owns the append-only ledger and terminal state, and can supervise an arbitrary
real-backend smoke worker in a separate process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from pathlib import Path

from .constants import TARGET_COLUMNS
from .model_selection_v31_backends import load_feasibility_artifact
from .model_selection_v31_checkpoint import (
    COMPLETION_STATUS,
    CheckpointStore,
    CompletionContract,
    ReplayContract,
)
from .model_selection_v31_guard import SignalRelay
from .model_selection_v31_observability import (
    EventSink,
    TrialIdentity,
    fsync_parent,
    validate_event_stream,
)

RUN_START_SCHEMA = "qm9-model-selection-v31-run-start-v3"
LEDGER_SCHEMA = "qm9-model-selection-v31-trial-ledger-v1"
TERMINAL_SCHEMA = "qm9-model-selection-v31-terminal-v1"
BOUNDED_TERMINAL_SCHEMA = "qm9-v31-bounded-profile-terminal-v5"
MEDIUM_TERMINAL_SCHEMA = "qm9-v31-medium-profile-supervisor-terminal-v1"
LEDGER_EVENTS = frozenset({"reserve", "start", "complete", "fail"})


class SupervisorError(RuntimeError):
    """Raised when governed run state would become ambiguous."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SupervisorError(f"{name} must be a lowercase SHA-256 digest")
    return value


def atomic_bytes(path: str | Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        fsync_parent(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_json(path: str | Path, payload: object) -> None:
    atomic_bytes(path, _canonical_bytes(payload) + b"\n")


@dataclass(frozen=True)
class RunBinding:
    run_id: str
    plan_sha256: str
    config_sha256: str
    input_sha256: str
    profile_sha256: str
    code_sha256: str
    feasibility_path: str
    feasibility_file_sha256: str
    memory_feasibility_sha256: str
    runtime_feasibility_sha256: str
    device_identity: Mapping[str, object]
    device_identity_sha256: str

    def validate(self) -> None:
        if not self.run_id:
            raise SupervisorError("run_id must be nonempty")
        for field in (
            "plan_sha256",
            "config_sha256",
            "input_sha256",
            "profile_sha256",
            "code_sha256",
            "feasibility_file_sha256",
            "memory_feasibility_sha256",
            "runtime_feasibility_sha256",
            "device_identity_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        if not self.feasibility_path:
            raise SupervisorError("feasibility artifact path must be nonempty")
        if canonical_sha256(dict(self.device_identity)) != self.device_identity_sha256:
            raise SupervisorError("device identity semantic hash differs")


@dataclass(frozen=True)
class BoundedProfileRunBinding:
    """Narrow bootstrap binding that can authorize only the fixed CPU profile smoke."""

    run_id: str
    run_purpose: str
    plan_sha256: str
    config_sha256: str
    input_sha256: str
    profile_sha256: str
    code_sha256: str
    dependency_sha256: str
    repository_provenance: Mapping[str, object]
    repository_provenance_sha256: str
    prepare_attempt_id: str
    prepare_claim_path: str
    prepare_claim_file_sha256: str
    prepare_claim_semantic_sha256: str
    prepare_complete_path: str
    prepare_complete_file_sha256: str
    prepare_complete_semantic_sha256: str
    bootstrap_profile_path: str
    bootstrap_profile_file_sha256: str
    bootstrap_profile_semantic_sha256: str
    bootstrap_review_path: str
    bootstrap_review_file_sha256: str
    bootstrap_review_semantic_sha256: str
    bootstrap_authorization_path: str
    bootstrap_authorization_file_sha256: str
    bootstrap_authorization_semantic_sha256: str
    device_identity: Mapping[str, object]
    device_identity_sha256: str

    def validate(self) -> None:
        if (
            self.run_id != "qm9-v31-bounded-smoke-cpu-v5"
            or self.run_purpose != "bounded_profile_only"
        ):
            raise SupervisorError("bounded profile run identity/purpose differs")
        for field in (
            "plan_sha256",
            "config_sha256",
            "input_sha256",
            "profile_sha256",
            "code_sha256",
            "dependency_sha256",
            "repository_provenance_sha256",
            "prepare_claim_file_sha256",
            "prepare_claim_semantic_sha256",
            "prepare_complete_file_sha256",
            "prepare_complete_semantic_sha256",
            "bootstrap_profile_file_sha256",
            "bootstrap_profile_semantic_sha256",
            "bootstrap_review_file_sha256",
            "bootstrap_review_semantic_sha256",
            "bootstrap_authorization_file_sha256",
            "bootstrap_authorization_semantic_sha256",
            "device_identity_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        if canonical_sha256(dict(self.repository_provenance)) != self.repository_provenance_sha256:
            raise SupervisorError("bounded profile repository provenance hash differs")
        if len(self.prepare_attempt_id) != 64 or any(
            character not in "0123456789abcdef" for character in self.prepare_attempt_id
        ):
            raise SupervisorError("bounded prepare attempt id differs")
        if canonical_sha256(dict(self.device_identity)) != self.device_identity_sha256:
            raise SupervisorError("bounded profile CPU identity hash differs")


@dataclass(frozen=True)
class MediumProfileRunBinding:
    """One-purpose bootstrap binding for generating medium feasibility evidence."""

    run_id: str
    run_purpose: str
    plan_sha256: str
    config_sha256: str
    input_sha256: str
    profile_sha256: str
    code_sha256: str
    dependency_sha256: str
    authorization_path: str
    authorization_file_sha256: str
    authorization_semantic_sha256: str
    authorization_nonce: str
    authorized_argv: tuple[str, ...]
    authorized_cwd: str
    authorized_output: str
    authorized_input_sha256: str
    authorized_sample_sha256: str
    authorized_plan_sha256: str
    authorized_budget_sha256: str
    device_identity: Mapping[str, object]
    device_identity_sha256: str

    def validate(self) -> None:
        allowed = {
            ("qm9-v31-medium-profile-cpu-v1", "feasibility_evidence_generation"),
            (
                "qm9-v31-public-fixed-three-backend-cpu-v1",
                "public_synthetic_three_backend_engineering_seam",
            ),
        }
        if (self.run_id, self.run_purpose) not in allowed:
            raise SupervisorError("medium profile run identity/purpose differs")
        for field in (
            "plan_sha256",
            "config_sha256",
            "input_sha256",
            "profile_sha256",
            "code_sha256",
            "dependency_sha256",
            "authorization_file_sha256",
            "authorization_semantic_sha256",
            "authorized_input_sha256",
            "authorized_sample_sha256",
            "authorized_plan_sha256",
            "authorized_budget_sha256",
            "device_identity_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        if not self.authorization_path:
            raise SupervisorError("medium profile authorization path must be nonempty")
        if (
            len(self.authorization_nonce) != 64
            or any(character not in "0123456789abcdef" for character in self.authorization_nonce)
            or not self.authorized_argv
            or not self.authorized_cwd
            or not self.authorized_output
            or self.authorized_plan_sha256 != self.plan_sha256
        ):
            raise SupervisorError("medium profile exact authorization binding differs")
        if canonical_sha256(dict(self.device_identity)) != self.device_identity_sha256:
            raise SupervisorError("medium profile CPU identity hash differs")


@dataclass(frozen=True)
class PlannedTrial:
    identity: TrialIdentity
    expected_shape: tuple[int, int]
    target_order: tuple[str, ...]
    prediction_order_sha256: str
    candidate_sha256: str

    def validate(self) -> None:
        if (
            len(self.expected_shape) != 2
            or self.expected_shape[0] < 1
            or self.expected_shape[1] != len(TARGET_COLUMNS)
        ):
            raise SupervisorError("planned prediction shape differs")
        if self.target_order != tuple(TARGET_COLUMNS):
            raise SupervisorError("planned target order differs")
        _require_sha256(self.prediction_order_sha256, "prediction_order_sha256")
        _require_sha256(self.candidate_sha256, "candidate_sha256")

    def payload(self) -> dict[str, object]:
        self.validate()
        return {
            "identity": trial_payload(self.identity),
            "expected_shape": list(self.expected_shape),
            "target_order": list(self.target_order),
            "prediction_order_sha256": self.prediction_order_sha256,
            "candidate_sha256": self.candidate_sha256,
        }


def detect_device_identity(device_type: str) -> dict[str, object]:
    if device_type == "cpu":
        return {
            "type": "cpu",
            "machine": platform.machine(),
            "processor": platform.processor(),
        }
    if device_type != "cuda":
        raise SupervisorError("device identity type must be cpu or cuda")
    try:
        line = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=uuid,name,driver_version",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            .stdout.strip()
            .splitlines()[0]
        )
        uuid_value, name, driver = [field.strip() for field in line.split(",", 2)]
        return {"type": "cuda", "uuid": uuid_value, "name": name, "driver": driver}
    except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError) as error:
        raise SupervisorError("approved CUDA device identity is unavailable") from error


def trial_payload(trial: TrialIdentity) -> dict[str, object]:
    return asdict(trial)


def trial_key(trial: TrialIdentity | Mapping[str, object]) -> str:
    payload = asdict(trial) if isinstance(trial, TrialIdentity) else dict(trial)
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
        raise SupervisorError("trial identity schema differs")
    return canonical_sha256(payload)


class TrialLedger:
    """Single-writer hash-chained ledger with strict trial transitions."""

    def __init__(self, path: str | Path, *, run_id: str, run_start_sha256: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.run_start_sha256 = _require_sha256(run_start_sha256, "run_start_sha256")
        self.rows = self.validate(self.path) if self.path.is_file() else []
        if self.rows and any(row["run_id"] != run_id for row in self.rows):
            raise SupervisorError("ledger belongs to another run")
        if self.rows and any(row["run_start_sha256"] != run_start_sha256 for row in self.rows):
            raise SupervisorError("ledger start binding differs")
        created = not self.path.exists()
        self._handle = self.path.open("a", encoding="utf-8")
        if created:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            fsync_parent(self.path)
        self._states = self._derive_states(self.rows)
        self._closed = False

    @staticmethod
    def _derive_states(rows: Sequence[Mapping[str, object]]) -> dict[str, str]:
        states: dict[str, str] = {}
        allowed = {
            None: {"reserve"},
            "reserve": {"start"},
            "start": {"complete", "fail"},
            "complete": set(),
            "fail": set(),
        }
        for row in rows:
            key, event = str(row["trial_key"]), str(row["event"])
            if event not in allowed.get(states.get(key), set()):
                raise SupervisorError("ledger contains an invalid trial transition")
            states[key] = event
        return states

    def append(
        self, event: str, trial: TrialIdentity, *, evidence: Mapping[str, object] | None = None
    ) -> dict[str, object]:
        if self._closed:
            raise SupervisorError("trial ledger is closed")
        if event not in LEDGER_EVENTS:
            raise SupervisorError("ledger event is invalid")
        key = trial_key(trial)
        allowed = {
            None: {"reserve"},
            "reserve": {"start"},
            "start": {"complete", "fail"},
            "complete": set(),
            "fail": set(),
        }
        if event not in allowed[self._states.get(key)]:
            raise SupervisorError("invalid or duplicate trial transition")
        previous = str(self.rows[-1]["entry_sha256"]) if self.rows else None
        row: dict[str, object] = {
            "schema_version": LEDGER_SCHEMA,
            "sequence": len(self.rows) + 1,
            "timestamp_utc": datetime.now(UTC)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z"),
            "run_id": self.run_id,
            "run_start_sha256": self.run_start_sha256,
            "event": event,
            "trial_key": key,
            "trial": trial_payload(trial),
            "evidence": dict(evidence or {}),
            "prev_entry_sha256": previous,
            "scientific_result": False,
            "publication_ready": False,
            "outer_validation_targets_read": False,
            "test_targets_read": False,
        }
        row["entry_sha256"] = canonical_sha256(row)
        self._handle.write(_canonical_bytes(row).decode() + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self.rows.append(row)
        self._states[key] = event
        return row

    @staticmethod
    def validate(path: str | Path) -> list[dict[str, object]]:
        raw = Path(path).read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise SupervisorError("trial ledger has a truncated tail")
        rows: list[dict[str, object]] = []
        previous: str | None = None
        run_id: str | None = None
        run_start_sha256: str | None = None
        for sequence, line in enumerate(raw.splitlines(), start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SupervisorError("trial ledger contains invalid JSON") from error
            recorded = row.pop("entry_sha256", None)
            if (
                row.get("schema_version") != LEDGER_SCHEMA
                or row.get("sequence") != sequence
                or row.get("event") not in LEDGER_EVENTS
                or row.get("prev_entry_sha256") != previous
                or recorded != canonical_sha256(row)
            ):
                raise SupervisorError("trial ledger chain or schema differs")
            for field in (
                "scientific_result",
                "publication_ready",
                "outer_validation_targets_read",
                "test_targets_read",
            ):
                if row.get(field) is not False:
                    raise SupervisorError("trial ledger invariant differs")
            run_id = run_id or str(row.get("run_id"))
            run_start_sha256 = run_start_sha256 or str(row.get("run_start_sha256"))
            if row.get("run_id") != run_id or row.get("run_start_sha256") != run_start_sha256:
                raise SupervisorError("trial ledger mixes run identities or start bindings")
            row["entry_sha256"] = recorded
            trial_key(row["trial"])
            previous = str(recorded)
            rows.append(row)
        TrialLedger._derive_states(rows)
        return rows

    def state(self, trial: TrialIdentity) -> str | None:
        return self._states.get(trial_key(trial))

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        if self._closed:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        self._closed = True


class RunSupervisor:
    """Own immutable start, ledger, telemetry, and exclusive terminal state."""

    def __init__(
        self,
        root: str | Path,
        binding: RunBinding,
        planned_trials: Sequence[PlannedTrial],
        *,
        resume: bool = False,
    ) -> None:
        self._run_purpose = "normal"
        binding.validate()
        self.root = Path(root)
        self.binding = binding
        self.planned_trials = list(planned_trials)
        if not self.planned_trials or any(
            not isinstance(row, PlannedTrial) for row in self.planned_trials
        ):
            raise SupervisorError("supervisor requires frozen PlannedTrial records")
        for row in self.planned_trials:
            row.validate()
        if len({trial_key(row.identity) for row in self.planned_trials}) != len(
            self.planned_trials
        ):
            raise SupervisorError("planned trials must be nonempty and unique")
        if any(row.identity.run_id != binding.run_id for row in self.planned_trials):
            raise SupervisorError("planned trial belongs to another run")
        plan_payload = [row.payload() for row in self.planned_trials]
        if canonical_sha256(plan_payload) != binding.plan_sha256:
            raise SupervisorError("planned trials differ from plan_sha256")
        self.feasibility = load_feasibility_artifact(
            binding.feasibility_path,
            expected_file_sha256=binding.feasibility_file_sha256,
            expected_memory_sha256=binding.memory_feasibility_sha256,
            expected_runtime_sha256=binding.runtime_feasibility_sha256,
            expected_device_identity_sha256=binding.device_identity_sha256,
        )
        expected_actual_hashes = {
            "plan_sha256": binding.plan_sha256,
            "config_sha256": binding.config_sha256,
            "input_sha256": binding.input_sha256,
            "profile_sha256": binding.profile_sha256,
            "code_sha256": binding.code_sha256,
        }
        if self.feasibility.get("actual_hashes") != expected_actual_hashes:
            raise SupervisorError("feasibility artifact actual hashes differ from run binding")
        if self.feasibility.get("device_identity") != dict(binding.device_identity):
            raise SupervisorError("feasibility artifact device identity differs")
        observed_device = detect_device_identity(str(binding.device_identity.get("type")))
        if observed_device != dict(binding.device_identity):
            raise SupervisorError("approved feasibility device differs from actual device")
        start = {
            "schema_version": RUN_START_SCHEMA,
            "binding": asdict(binding),
            "planned_trials": plan_payload,
            "planned_trial_keys": [trial_key(row.identity) for row in self.planned_trials],
            "feasibility": {
                "artifact": self.feasibility,
                "device_identity": dict(binding.device_identity),
            },
            "scientific_result": False,
            "publication_ready": False,
            "outer_validation_targets_read": False,
            "test_targets_read": False,
        }
        start["semantic_sha256"] = canonical_sha256(start)
        self.start_path = self.root / "run_start.json"
        if self.root.exists() and any(self.root.iterdir()):
            if not resume:
                raise SupervisorError("output is nonempty; explicit resume is required")
            if not self.start_path.is_file():
                raise SupervisorError("resume directory has no immutable run start")
            observed = json.loads(self.start_path.read_text())
            if observed != start:
                raise SupervisorError("resume binding or trial plan differs")
            if (self.root / "final").exists() or (self.root / "failure").exists():
                raise SupervisorError("terminal run cannot be resumed")
        else:
            self.root.mkdir(parents=True, exist_ok=True)
            fsync_parent(self.root)
            atomic_json(self.start_path, start)
        self.run_start_sha256 = file_sha256(self.start_path)
        self.events = EventSink(
            self.root / "telemetry/events.jsonl",
            binding.run_id,
            run_start_sha256=self.run_start_sha256,
        )
        self.ledger = TrialLedger(
            self.root / "ledger/trials.jsonl",
            run_id=binding.run_id,
            run_start_sha256=self.run_start_sha256,
        )
        self.checkpoints = CheckpointStore(
            self.root / "checkpoints",
            plan_sha256=binding.plan_sha256,
            config_sha256=binding.config_sha256,
            input_sha256=binding.input_sha256,
            cache_sha256=binding.profile_sha256,
            run_start_sha256=self.run_start_sha256,
        )
        self._terminal = False
        if not resume:
            self.events.emit(
                "run_start",
                trial=None,
                progress={
                    "binding": asdict(binding),
                    "planned_trial_count": len(self.planned_trials),
                },
                force_fsync=True,
            )

    @classmethod
    def for_bounded_profile(
        cls,
        root: str | Path,
        binding: BoundedProfileRunBinding,
        planned_trials: Sequence[PlannedTrial],
    ) -> RunSupervisor:
        """Construct only the exact independently authorized bounded CPU smoke."""

        binding.validate()
        instance = cls.__new__(cls)
        instance._run_purpose = "bounded_profile_only"
        instance.root = Path(root)
        instance.binding = binding
        instance.planned_trials = list(planned_trials)
        if len(instance.planned_trials) != 3:
            raise SupervisorError("bounded profile requires exactly three trials")
        for row in instance.planned_trials:
            row.validate()
        if len({trial_key(row.identity) for row in instance.planned_trials}) != 3:
            raise SupervisorError("bounded profile trial plan differs")
        if any(row.identity.run_id != binding.run_id for row in instance.planned_trials):
            raise SupervisorError("bounded profile trial belongs to another run")
        plan_payload = [row.payload() for row in instance.planned_trials]
        if canonical_sha256(plan_payload) != binding.plan_sha256:
            raise SupervisorError("bounded profile plan hash differs")
        instance._load_bounded_profile_authorization()
        from .model_selection_v31_profile_smoke import bounded_lifecycle_start_evidence

        lifecycle = bounded_lifecycle_start_evidence(binding)
        if instance.root.exists():
            raise SupervisorError("bounded profile run root must be fresh; resume is forbidden")
        start = {
            "schema_version": RUN_START_SCHEMA,
            "binding": asdict(binding),
            "planned_trials": plan_payload,
            "planned_trial_keys": [trial_key(row.identity) for row in instance.planned_trials],
            "bootstrap_authorization": json.loads(
                Path(binding.bootstrap_authorization_path).read_text()
            ),
            "bootstrap_lifecycle": lifecycle,
            "run_purpose": "bounded_profile_only",
            "scientific_result": False,
            "publication_ready": False,
            "ranking_allowed": False,
            "promotion_allowed": False,
            "ensemble_allowed": False,
            "outer_validation_targets_read": False,
            "test_targets_read": False,
        }
        start["semantic_sha256"] = canonical_sha256(start)
        instance.root.mkdir(parents=True, exist_ok=False)
        fsync_parent(instance.root)
        instance.start_path = instance.root / "run_start.json"
        atomic_json(instance.start_path, start)
        instance.run_start_sha256 = file_sha256(instance.start_path)
        instance.events = EventSink(
            instance.root / "telemetry/events.jsonl",
            binding.run_id,
            run_start_sha256=instance.run_start_sha256,
        )
        instance.ledger = TrialLedger(
            instance.root / "ledger/trials.jsonl",
            run_id=binding.run_id,
            run_start_sha256=instance.run_start_sha256,
        )
        instance.checkpoints = CheckpointStore(
            instance.root / "checkpoints",
            plan_sha256=binding.plan_sha256,
            config_sha256=binding.config_sha256,
            input_sha256=binding.input_sha256,
            cache_sha256=binding.profile_sha256,
            run_start_sha256=instance.run_start_sha256,
        )
        instance._terminal = False
        instance.events.emit(
            "run_start",
            trial=None,
            progress={
                "binding": asdict(binding),
                "planned_trial_count": len(instance.planned_trials),
                "run_purpose": "bounded_profile_only",
            },
            force_fsync=True,
        )
        return instance

    @classmethod
    def for_medium_profile(
        cls,
        root: str | Path,
        binding: MediumProfileRunBinding,
        planned_trials: Sequence[PlannedTrial],
    ) -> RunSupervisor:
        """Construct only the independently authorized train-only feasibility profiler."""

        binding.validate()
        instance = cls.__new__(cls)
        instance._run_purpose = binding.run_purpose
        instance.root = Path(root)
        instance.binding = binding
        instance.planned_trials = list(planned_trials)
        if len(instance.planned_trials) != 3:
            raise SupervisorError("medium profile requires exactly three trials")
        for row in instance.planned_trials:
            row.validate()
        if len({trial_key(row.identity) for row in instance.planned_trials}) != 3:
            raise SupervisorError("medium profile trial plan differs")
        if any(row.identity.run_id != binding.run_id for row in instance.planned_trials):
            raise SupervisorError("medium profile trial belongs to another run")
        plan_payload = [row.payload() for row in instance.planned_trials]
        if canonical_sha256(plan_payload) != binding.plan_sha256:
            raise SupervisorError("medium profile plan hash differs")
        instance._load_medium_profile_authorization()
        if instance.root.exists():
            raise SupervisorError("medium profile run root must be fresh; resume is forbidden")
        start = {
            "schema_version": RUN_START_SCHEMA,
            "binding": asdict(binding),
            "planned_trials": plan_payload,
            "planned_trial_keys": [trial_key(row.identity) for row in instance.planned_trials],
            "run_purpose": binding.run_purpose,
            "scientific_result": False,
            "publication_ready": False,
            "ranking_allowed": False,
            "promotion_allowed": False,
            "ensemble_allowed": False,
            "outer_validation_targets_read": False,
            "test_targets_read": False,
        }
        start["semantic_sha256"] = canonical_sha256(start)
        instance.root.mkdir(parents=True, exist_ok=False)
        fsync_parent(instance.root)
        instance.start_path = instance.root / "run_start.json"
        atomic_json(instance.start_path, start)
        instance.run_start_sha256 = file_sha256(instance.start_path)
        instance.events = EventSink(
            instance.root / "telemetry/events.jsonl",
            binding.run_id,
            run_start_sha256=instance.run_start_sha256,
        )
        instance.ledger = TrialLedger(
            instance.root / "ledger/trials.jsonl",
            run_id=binding.run_id,
            run_start_sha256=instance.run_start_sha256,
        )
        instance.checkpoints = CheckpointStore(
            instance.root / "checkpoints",
            plan_sha256=binding.plan_sha256,
            config_sha256=binding.config_sha256,
            input_sha256=binding.input_sha256,
            cache_sha256=binding.profile_sha256,
            run_start_sha256=instance.run_start_sha256,
        )
        instance._terminal = False
        instance.events.emit(
            "run_start",
            trial=None,
            progress={
                "binding": asdict(binding),
                "planned_trial_count": len(instance.planned_trials),
                "run_purpose": binding.run_purpose,
            },
            force_fsync=True,
        )
        return instance

    def _load_bounded_profile_authorization(self) -> Mapping[str, object]:
        from .model_selection_v31_profile_smoke import (
            load_bounded_bootstrap_authorization,
        )

        return load_bounded_bootstrap_authorization(
            self.binding,
            planned_trials=self.planned_trials,
            run_root=self.root,
            expected_run_start_sha256=getattr(self, "run_start_sha256", None),
        )

    def _load_medium_profile_authorization(self) -> Mapping[str, object]:
        from .model_selection_v31_medium_profile import validate_supervisor_authorization

        return validate_supervisor_authorization(self.binding, self.planned_trials)

    def _require_planned(self, trial: TrialIdentity) -> None:
        if self._terminal:
            raise SupervisorError("terminal run is immutable")
        if self._run_purpose == "bounded_profile_only":
            self._load_bounded_profile_authorization()
        elif self._run_purpose in {
            "feasibility_evidence_generation",
            "public_synthetic_three_backend_engineering_seam",
        }:
            self._load_medium_profile_authorization()
        else:
            load_feasibility_artifact(
                self.binding.feasibility_path,
                expected_file_sha256=self.binding.feasibility_file_sha256,
                expected_memory_sha256=self.binding.memory_feasibility_sha256,
                expected_runtime_sha256=self.binding.runtime_feasibility_sha256,
                expected_device_identity_sha256=self.binding.device_identity_sha256,
            )
        if trial_key(trial) not in {trial_key(row.identity) for row in self.planned_trials}:
            raise SupervisorError("trial is outside the immutable plan")

    def planned_trial(self, trial: TrialIdentity) -> PlannedTrial:
        self._require_planned(trial)
        return next(
            row for row in self.planned_trials if trial_key(row.identity) == trial_key(trial)
        )

    def reserve(self, trial: TrialIdentity, evidence: Mapping[str, object] | None = None) -> None:
        self._require_planned(trial)
        reserve_evidence = dict(evidence or {})
        if self._run_purpose == "bounded_profile_only":
            from .model_selection_v31_profile_smoke import (
                validate_bounded_trial_observation,
            )

            path = reserve_evidence.get("resource_observation_path")
            if not isinstance(path, str):
                raise SupervisorError("bounded reserve requires resource observation evidence")
            expected = validate_bounded_trial_observation(
                self.binding,
                path=path,
                trial=trial,
                run_start_sha256=self.run_start_sha256,
                previous_ledger_entry_sha256=(
                    str(self.ledger.rows[-1]["entry_sha256"]) if self.ledger.rows else None
                ),
            )
            if reserve_evidence != expected:
                raise SupervisorError("bounded reserve observation evidence differs")
        self.ledger.append("reserve", trial, evidence=reserve_evidence)
        self.events.emit(
            "trial_reserved",
            trial=trial,
            progress={"state": "reserved", "evidence": reserve_evidence},
        )

    def start_trial(self, trial: TrialIdentity) -> dict[str, object]:
        self._require_planned(trial)
        if self._run_purpose == "bounded_profile_only":
            reserve_rows = [
                row
                for row in self.ledger.rows
                if row["trial_key"] == trial_key(trial) and row["event"] == "reserve"
            ]
            if len(reserve_rows) != 1 or not reserve_rows[0]["evidence"]:
                raise SupervisorError("bounded start lacks observation-bound reserve")
        row = self.ledger.append("start", trial)
        self.events.emit("trial_start", trial=trial, progress={"state": "started"})
        return row

    def completion_contract(self, trial: TrialIdentity) -> CompletionContract:
        specification = self.planned_trial(trial)
        if self.ledger.state(trial) != "start":
            raise SupervisorError("completion contract requires a durable START ledger entry")
        start_rows = [
            row
            for row in self.ledger.rows
            if row["trial_key"] == trial_key(trial) and row["event"] == "start"
        ]
        if len(start_rows) != 1:
            raise SupervisorError("trial must have exactly one START ledger entry")
        return CompletionContract(
            expected_shape=specification.expected_shape,
            target_order=specification.target_order,
            prediction_order_sha256=specification.prediction_order_sha256,
            start_ledger_entry_sha256=str(start_rows[0]["entry_sha256"]),
        )

    def complete_trial(self, trial: TrialIdentity, evidence: Mapping[str, object]) -> None:
        raise SupervisorError("direct completion is forbidden; authenticate a completed checkpoint")

    def authenticate_completed_checkpoint(
        self, trial: TrialIdentity, checkpoint_path: str | Path
    ) -> dict[str, object]:
        contract = self.completion_contract(trial)
        try:
            checkpoint = self.checkpoints.validate_completed_checkpoint(
                checkpoint_path, trial=trial, completion=contract
            )
        except Exception as error:
            raise SupervisorError("completed checkpoint authentication failed") from error
        row = self.ledger.append(
            "complete",
            trial,
            evidence={
                "status": COMPLETION_STATUS,
                "start_ledger_entry_sha256": contract.start_ledger_entry_sha256,
                "checkpoint": checkpoint,
            },
        )
        self.events.emit(
            "trial_complete",
            trial=trial,
            progress={"state": "completed"},
            metrics={
                "checkpoint_file_sha256": checkpoint["file_sha256"],
                "prediction_sha256": checkpoint["prediction_sha256"],
                "complete_ledger_entry_sha256": row["entry_sha256"],
            },
            force_fsync=True,
        )
        return checkpoint

    def authenticate_worker_result(
        self, trial: TrialIdentity, result_path: str | Path
    ) -> dict[str, object]:
        result = Path(result_path).resolve()
        try:
            result.relative_to(self.root.resolve())
        except ValueError as error:
            raise SupervisorError("worker result is outside the governed run") from error
        try:
            payload = json.loads(result.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise SupervisorError("worker result is missing or invalid") from error
        base_keys = {"status", "checkpoint_evidence", "metadata_sha256"}
        medium_keys = base_keys | {
            "label_reads",
            "outer_validation_targets_read",
            "test_targets_read",
            "private_evaluation_labels_read",
            "protected_labels_read",
            "ranking_allowed",
            "ensemble_allowed",
            "promotion_allowed",
            "w1_allowed",
            "general_training_allowed",
            "scientific_result",
            "publication_ready",
        }
        expected_keys = (
            medium_keys
            if self._run_purpose in {
                "feasibility_evidence_generation",
                "public_synthetic_three_backend_engineering_seam",
            }
            else base_keys
        )
        if (
            not isinstance(payload, dict)
            or set(payload) != expected_keys
            or payload.get("status") != COMPLETION_STATUS
        ):
            raise SupervisorError("worker result schema/status differs")
        if self._run_purpose in {
            "feasibility_evidence_generation",
            "public_synthetic_three_backend_engineering_seam",
        }:
            boundary_keys = medium_keys - base_keys - {"label_reads"}
            if any(payload.get(key) is not False for key in boundary_keys):
                raise SupervisorError("medium worker result boundary differs")
            expected_reads = {
                "source_fixed_train_target_loads": 0,
                "derived_train_target_artifact_reads": (
                    0
                    if self._run_purpose
                    == "public_synthetic_three_backend_engineering_seam"
                    else 1
                ),
                "outer_validation": 0,
                "test": 0,
                "protected": 0,
                "private_evaluation": 0,
            }
            if self._run_purpose == "public_synthetic_three_backend_engineering_seam":
                expected_reads["synthetic_target_artifact_reads"] = 1
            if payload.get("label_reads") != expected_reads:
                raise SupervisorError("medium worker result label counters differ")
        _require_sha256(str(payload.get("metadata_sha256")), "metadata_sha256")
        evidence = payload.get("checkpoint_evidence")
        if not isinstance(evidence, Mapping) or not isinstance(evidence.get("relative_path"), str):
            raise SupervisorError("worker checkpoint evidence is missing")
        checkpoint_path = self.checkpoints.root / str(evidence["relative_path"])
        contract = self.completion_contract(trial)
        try:
            authenticated = self.checkpoints.validate_completed_checkpoint(
                checkpoint_path, trial=trial, completion=contract
            )
        except Exception as error:
            raise SupervisorError("worker checkpoint authentication failed") from error
        if authenticated != dict(evidence):
            raise SupervisorError("worker checkpoint evidence differs")
        self.authenticate_completed_checkpoint(trial, checkpoint_path)
        return payload

    def run_backend_trial(
        self,
        specification: PlannedTrial,
        *,
        backend_runner: Callable[..., tuple[object, Mapping[str, object]]],
        backend_kwargs: Mapping[str, object],
    ) -> tuple[object, Mapping[str, object]]:
        """Execute backend→checkpoint→ledger completion as one governed transition."""

        trial = specification.identity
        if self.planned_trial(trial) != specification:
            raise SupervisorError("backend trial specification differs from frozen plan")
        forbidden = {"sink", "checkpoint_store", "trial", "completion_contract"}
        if forbidden & set(backend_kwargs):
            raise SupervisorError("backend caller cannot inject governance objects")
        self.reserve(trial)
        self.start_trial(trial)
        contract = self.completion_contract(trial)
        try:
            prediction, metadata = backend_runner(
                **dict(backend_kwargs),
                sink=self.events,
                checkpoint_store=self.checkpoints,
                trial=trial,
                completion_contract=contract,
            )
            if not isinstance(metadata, Mapping):
                raise SupervisorError("backend metadata is not an object")
            checkpoint_evidence = metadata.get("checkpoint_evidence")
            if not isinstance(checkpoint_evidence, Mapping):
                raise SupervisorError("backend did not return checkpoint evidence")
            relative = checkpoint_evidence.get("relative_path")
            if not isinstance(relative, str):
                raise SupervisorError("backend checkpoint evidence path is missing")
            checkpoint_path = self.checkpoints.root / relative
            prevalidated = self.checkpoints.validate_completed_checkpoint(
                checkpoint_path, trial=trial, completion=contract
            )
            if dict(checkpoint_evidence) != prevalidated:
                raise SupervisorError("backend and supervisor checkpoint evidence differ")
            self.authenticate_completed_checkpoint(trial, checkpoint_path)
            return prediction, metadata
        except BaseException as error:
            if self.ledger.state(trial) == "start":
                self.fail_trial(
                    trial,
                    {"reason": "BACKEND_OR_CHECKPOINT_FAILURE", "error_type": type(error).__name__},
                )
            if not self._terminal:
                self.finalize_failure(
                    {"reason": "BACKEND_OR_CHECKPOINT_FAILURE", "error_type": type(error).__name__}
                )
            raise

    def fail_trial(self, trial: TrialIdentity, evidence: Mapping[str, object]) -> None:
        if self.ledger.state(trial) == "start":
            self.ledger.append("fail", trial, evidence=evidence)
        self.events.emit(
            "trial_failed",
            trial=trial,
            progress={"state": "failed"},
            failure=dict(evidence),
            force_fsync=True,
        )

    def _terminal_payload(self, status: str, evidence: Mapping[str, object]) -> dict[str, object]:
        states = {
            trial_key(row.identity): self.ledger.state(row.identity) for row in self.planned_trials
        }
        return {
            "schema_version": TERMINAL_SCHEMA,
            "status": status,
            "run_id": self.binding.run_id,
            "run_start_sha256": self.run_start_sha256,
            "ledger_sha256": file_sha256(self.ledger.path),
            "trial_states": states,
            "complete": status == "complete"
            and all(value == "complete" for value in states.values()),
            "ranking_allowed": status == "complete"
            and all(value == "complete" for value in states.values()),
            "evidence": dict(evidence),
            "scientific_result": False,
            "publication_ready": False,
            "outer_validation_targets_read": False,
            "test_targets_read": False,
        }

    def _strict_checkpoint_replay(self) -> list[dict[str, object]]:
        complete_rows = [row for row in self.ledger.rows if row["event"] == "complete"]
        completed_entries = {
            str(row["trial_key"]): {
                "status": COMPLETION_STATUS,
                "complete_ledger_entry_sha256": row["entry_sha256"],
                "start_ledger_entry_sha256": row["evidence"]["start_ledger_entry_sha256"],
                "checkpoint": row["evidence"]["checkpoint"],
            }
            for row in complete_rows
        }
        replay = ReplayContract(
            planned_trials=tuple(trial_payload(row.identity) for row in self.planned_trials),
            completed_entries=completed_entries,
            ledger_sha256=file_sha256(self.ledger.path),
            terminal_status=COMPLETION_STATUS,
        )
        try:
            replayed = self.checkpoints.load_completed(self.binding.plan_sha256, replay=replay)
        except Exception as error:
            raise SupervisorError("strict completed-checkpoint replay failed") from error
        if len(replayed) != len(self.planned_trials):
            raise SupervisorError("strict replay count differs from frozen plan")
        return replayed

    def finalize_success(self, evidence: Mapping[str, object] | None = None) -> Path:
        if self._run_purpose in {
            "feasibility_evidence_generation",
            "public_synthetic_three_backend_engineering_seam",
        }:
            raise SupervisorError("medium profile cannot use scientific success finalization")
        if self._terminal:
            raise SupervisorError("terminal run is immutable")
        payload = self._terminal_payload("complete", evidence or {})
        if not payload["complete"]:
            raise SupervisorError("partial trial ledger cannot produce a ranking or final state")
        if (self.root / "failure").exists():
            raise SupervisorError("failure and final terminal states are mutually exclusive")
        replayed = self._strict_checkpoint_replay()
        payload["evidence"] = {
            **dict(payload["evidence"]),
            "checkpoint_replay_count": len(replayed),
            "checkpoint_replay_sha256": canonical_sha256(
                [row["manifest"]["prediction_sha256"] for row in replayed]
            ),
        }
        path = self.root / "final/manifest.json"
        try:
            atomic_json(path, payload)
        except BaseException as error:
            error.failure_stage = "terminal_write"  # type: ignore[attr-defined]
            raise
        try:
            self.events.emit("run_complete", trial=None, progress=payload, force_fsync=True)
        except BaseException as error:
            error.failure_stage = "run_complete_append"  # type: ignore[attr-defined]
            raise
        self._terminal = True
        return path

    def finalize_profile_success(self, evidence: Mapping[str, object] | None = None) -> Path:
        """Close a complete engineering smoke without authorizing scientific use."""

        if self._terminal:
            raise SupervisorError("terminal run is immutable")
        if not self.ledger.closed:
            raise SupervisorError("profile ledger must be durably closed before finalization")
        payload = self._terminal_payload("complete", evidence or {})
        if not payload["complete"]:
            raise SupervisorError("partial trial ledger cannot produce a profile terminal")
        if (self.root / "failure").exists():
            raise SupervisorError("failure and final terminal states are mutually exclusive")
        if self._run_purpose == "bounded_profile_only":
            from .model_selection_v31_profile_smoke import validate_bounded_terminal_chain

            validate_bounded_terminal_chain(self)
        elif self._run_purpose not in {
            "feasibility_evidence_generation",
            "public_synthetic_three_backend_engineering_seam",
        }:
            raise SupervisorError("profile terminal purpose differs")
        replayed = self._strict_checkpoint_replay()
        is_medium = self._run_purpose in {
            "feasibility_evidence_generation",
            "public_synthetic_three_backend_engineering_seam",
        }
        payload.update(
            {
                "schema_version": (
                    MEDIUM_TERMINAL_SCHEMA if is_medium else BOUNDED_TERMINAL_SCHEMA
                ),
                "terminal_mode": "profile-only",
                "purpose": self._run_purpose if is_medium else "engineering_smoke",
                "w1_feasibility_approved": False,
                "general_training_approved": False,
                "private_data_allowed": self._run_purpose == "feasibility_evidence_generation",
                "ranking_allowed": False,
                "promotion_allowed": False,
                "ensemble_allowed": False,
            }
        )
        payload["evidence"] = {
            **dict(payload["evidence"]),
            "checkpoint_replay_count": len(replayed),
            "checkpoint_replay_sha256": canonical_sha256(
                [row["manifest"]["prediction_sha256"] for row in replayed]
            ),
        }
        path = self.root / "final/manifest.json"
        try:
            atomic_json(path, payload)
        except BaseException as error:
            error.failure_stage = "terminal_write"  # type: ignore[attr-defined]
            raise
        try:
            self.events.emit("run_complete", trial=None, progress=payload, force_fsync=True)
        except BaseException as error:
            error.failure_stage = "run_complete_append"  # type: ignore[attr-defined]
            raise
        self._terminal = True
        return path

    def finalize_failure(self, evidence: Mapping[str, object]) -> Path:
        if self._terminal:
            raise SupervisorError("terminal run is immutable")
        if (self.root / "final").exists():
            raise SupervisorError("final and failure terminal states are mutually exclusive")
        payload = self._terminal_payload("failed", evidence)
        payload["complete"] = False
        payload["ranking_allowed"] = False
        path = self.root / "failure/manifest.json"
        atomic_json(path, payload)
        self.events.emit(
            "failure", trial=None, progress={"complete": False}, failure=payload, force_fsync=True
        )
        self._terminal = True
        return path

    def assert_ranking_allowed(self) -> None:
        if self._run_purpose in {
            "feasibility_evidence_generation",
            "public_synthetic_three_backend_engineering_seam",
        }:
            raise SupervisorError("medium profile can never authorize ranking")
        path = self.root / "final/manifest.json"
        if not path.is_file() or (self.root / "failure").exists():
            raise SupervisorError("partial or failed run cannot expose a ranking")
        payload = json.loads(path.read_text())
        if payload.get("ranking_allowed") is not True or payload.get("complete") is not True:
            raise SupervisorError("terminal manifest suppresses ranking")
        events = validate_event_stream(self.events.path)
        if not events or events[-1]["event"] != "run_complete":
            raise SupervisorError("terminal telemetry does not authorize ranking")
        if events[-1].get("progress") != payload:
            raise SupervisorError("terminal manifest differs from immutable telemetry")
        if payload.get("ledger_sha256") != file_sha256(self.ledger.path):
            raise SupervisorError("terminal ledger hash differs")
        replayed = self._strict_checkpoint_replay()
        evidence = payload.get("evidence")
        if not isinstance(evidence, Mapping):
            raise SupervisorError("terminal checkpoint replay evidence is missing")
        observed_replay_hash = canonical_sha256(
            [row["manifest"]["prediction_sha256"] for row in replayed]
        )
        if (
            evidence.get("checkpoint_replay_count") != len(replayed)
            or evidence.get("checkpoint_replay_sha256") != observed_replay_hash
        ):
            raise SupervisorError("terminal checkpoint replay evidence differs")

    def assert_profile_complete(self) -> None:
        """Re-authenticate a complete profile-only terminal and every checkpoint."""

        path = self.root / "final/manifest.json"
        if not path.is_file() or (self.root / "failure").exists():
            raise SupervisorError("partial or failed run cannot expose a complete profile")
        payload = json.loads(path.read_text())
        required_false = (
            "ranking_allowed",
            "promotion_allowed",
            "ensemble_allowed",
            "scientific_result",
            "publication_ready",
            "outer_validation_targets_read",
            "test_targets_read",
        )
        expected_purpose = (
            self._run_purpose
            if self._run_purpose in {
                "feasibility_evidence_generation",
                "public_synthetic_three_backend_engineering_seam",
            }
            else "engineering_smoke"
        )
        if (
            payload.get("complete") is not True
            or payload.get("terminal_mode") != "profile-only"
            or payload.get("purpose") != expected_purpose
            or any(payload.get(field) is not False for field in required_false)
        ):
            raise SupervisorError("terminal manifest is not a complete profile-only run")
        if self._run_purpose == "bounded_profile_only":
            from .model_selection_v31_profile_smoke import validate_bounded_terminal_chain

            validate_bounded_terminal_chain(self)
        events = validate_event_stream(self.events.path)
        if not events or events[-1]["event"] != "run_complete":
            raise SupervisorError("terminal telemetry does not authorize profile completion")
        if events[-1].get("progress") != payload:
            raise SupervisorError("terminal manifest differs from immutable telemetry")
        if payload.get("ledger_sha256") != file_sha256(self.ledger.path):
            raise SupervisorError("terminal ledger hash differs")
        replayed = self._strict_checkpoint_replay()
        evidence = payload.get("evidence")
        if not isinstance(evidence, Mapping):
            raise SupervisorError("terminal checkpoint replay evidence is missing")
        observed_replay_hash = canonical_sha256(
            [row["manifest"]["prediction_sha256"] for row in replayed]
        )
        if (
            evidence.get("checkpoint_replay_count") != len(replayed)
            or evidence.get("checkpoint_replay_sha256") != observed_replay_hash
        ):
            raise SupervisorError("terminal checkpoint replay evidence differs")

    def close_ledger(self) -> None:
        """Durably and idempotently close the append-only trial ledger."""

        self.ledger.close()

    def close_events(self) -> None:
        """Durably and idempotently close the append-only telemetry stream."""

        self.events.close()

    def close(self) -> None:
        self.close_ledger()
        self.close_events()


@dataclass(frozen=True)
class SidecarPolicy:
    heartbeat_seconds: float
    heartbeat_stale_after_seconds: float
    model_progress_stale_after_seconds: float
    soft_deadline_seconds: float
    hard_deadline_seconds: float
    rss_max_bytes: int
    disk_free_min_bytes: int
    gpu_allocated_max_bytes: int
    gpu_reserved_max_bytes: int
    gpu_temperature_max_c: float
    require_rss_sensor: bool
    require_disk_sensor: bool
    require_gpu_sensors: bool
    gpu_device_uuid: str | None = None
    poll_seconds: float = 0.05
    model_progress_stale_overrides: Mapping[str, float] = dataclass_field(
        default_factory=dict
    )

    def validate(self) -> None:
        if not 0 < self.heartbeat_seconds <= 30:
            raise SupervisorError("sidecar heartbeat must be in (0, 30]")
        if self.heartbeat_stale_after_seconds != 2 * self.heartbeat_seconds:
            raise SupervisorError(
                "sidecar heartbeat stale limit must be two heartbeat intervals"
            )
        if self.model_progress_stale_after_seconds <= 0:
            raise SupervisorError("sidecar model-progress stale limit must be positive")
        for selector, limit in self.model_progress_stale_overrides.items():
            parts = selector.split(":")
            if (
                len(parts) != 2
                or not all(parts)
                or not isinstance(limit, (int, float))
                or isinstance(limit, bool)
                or limit <= 0
            ):
                raise SupervisorError(
                    "sidecar model-progress override must be FAMILY:STAGE -> positive seconds"
                )
        if not 0 < self.poll_seconds <= 5:
            raise SupervisorError("sidecar poll must be in (0, 5]")
        if not 0 < self.soft_deadline_seconds < self.hard_deadline_seconds:
            raise SupervisorError("sidecar soft deadline must precede hard deadline")
        for field in (
            "rss_max_bytes",
            "disk_free_min_bytes",
            "gpu_allocated_max_bytes",
            "gpu_reserved_max_bytes",
        ):
            if getattr(self, field) <= 0:
                raise SupervisorError(f"sidecar {field} must be positive")
        if self.gpu_temperature_max_c <= 0:
            raise SupervisorError("sidecar GPU temperature limit must be positive")
        if self.require_gpu_sensors and not self.gpu_device_uuid:
            raise SupervisorError("approved GPU path requires an exact device UUID")

    def model_progress_stale_limit(self, family: str, stage: str) -> float:
        """Resolve exact, stage-wide, then family-wide progress thresholds."""

        for selector in (
            f"{family}:{stage}",
            f"*:{stage}",
            f"{family}:*",
            "*:*",
        ):
            if selector in self.model_progress_stale_overrides:
                return float(self.model_progress_stale_overrides[selector])
        return self.model_progress_stale_after_seconds


class ExternalWorkerSidecar:
    """Supervise a worker process from outside its interpreter and GIL."""

    def __init__(self, supervisor: RunSupervisor, policy: SidecarPolicy) -> None:
        policy.validate()
        self.supervisor = supervisor
        self.policy = policy

    @staticmethod
    def _process_group_pids(process_group: int) -> list[int] | None:
        try:
            pids = []
            for entry in Path("/proc").iterdir():
                if entry.name.isdigit():
                    try:
                        if os.getpgid(int(entry.name)) == process_group:
                            pids.append(int(entry.name))
                    except (ProcessLookupError, PermissionError):
                        continue
            return pids
        except OSError:
            return None

    @classmethod
    def _process_group_rss(cls, process_group: int) -> int | None:
        pids = cls._process_group_pids(process_group)
        if pids is None:
            return None
        total = 0
        observed = False
        for pid in pids:
            try:
                resident_pages = int(Path(f"/proc/{pid}/statm").read_text().split()[1])
                total += resident_pages * os.sysconf("SC_PAGE_SIZE")
                observed = True
            except (OSError, IndexError, ValueError):
                continue
        return total if observed else None

    @classmethod
    def _gpu_resources(cls, process_group: int, device_uuid: str | None) -> dict[str, object]:
        if device_uuid is None:
            return {
                "gpu_sensor_available": False,
                "gpu_allocated_bytes": None,
                "gpu_reserved_bytes": None,
                "gpu_temperature_c": None,
            }
        pids = cls._process_group_pids(process_group)
        if pids is None:
            pids = []
        try:
            applications = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_gpu_memory,gpu_uuid",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=1,
            ).stdout.splitlines()
            used_mib = 0.0
            for line in applications:
                fields = [field.strip() for field in line.split(",")]
                if len(fields) == 3 and int(fields[0]) in pids and fields[2] == device_uuid:
                    used_mib += float(fields[1])
            devices = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=uuid,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=1,
            ).stdout.splitlines()
            temperature = next(
                float(fields[1])
                for fields in ([field.strip() for field in line.split(",")] for line in devices)
                if len(fields) == 2 and fields[0] == device_uuid
            )
            used_bytes = int(used_mib * 1024**2)
            return {
                "gpu_sensor_available": True,
                # NVML process-used memory is a conservative external proxy for
                # both framework allocated and reserved memory.
                "gpu_allocated_bytes": used_bytes,
                "gpu_reserved_bytes": used_bytes,
                "gpu_temperature_c": temperature,
                "gpu_memory_semantics": "nvidia-smi-process-used-memory-upper-bound",
            }
        except (FileNotFoundError, StopIteration, ValueError, subprocess.SubprocessError):
            return {
                "gpu_sensor_available": False,
                "gpu_allocated_bytes": None,
                "gpu_reserved_bytes": None,
                "gpu_temperature_c": None,
            }

    def _sample_external_resources(self, process_group: int) -> dict[str, object]:
        rss = self._process_group_rss(process_group)
        try:
            disk_free: int | None = int(shutil.disk_usage(self.supervisor.root).free)
        except OSError:
            disk_free = None
        return {
            "rss_sensor_available": rss is not None,
            "rss_bytes": rss,
            "disk_sensor_available": disk_free is not None,
            "disk_free_bytes": disk_free,
            **self._gpu_resources(process_group, self.policy.gpu_device_uuid),
        }

    def _resource_breach(self, resources: Mapping[str, object]) -> str | None:
        policy = self.policy
        if policy.require_rss_sensor and resources.get("rss_sensor_available") is not True:
            return "RSS_SENSOR_UNAVAILABLE"
        if policy.require_disk_sensor and resources.get("disk_sensor_available") is not True:
            return "DISK_SENSOR_UNAVAILABLE"
        if policy.require_gpu_sensors and resources.get("gpu_sensor_available") is not True:
            return "GPU_SENSOR_UNAVAILABLE"
        gates = (
            ("RSS_LIMIT", "rss_bytes", policy.rss_max_bytes, lambda value, limit: value > limit),
            (
                "DISK_FLOOR",
                "disk_free_bytes",
                policy.disk_free_min_bytes,
                lambda value, limit: value < limit,
            ),
            (
                "GPU_ALLOCATED_LIMIT",
                "gpu_allocated_bytes",
                policy.gpu_allocated_max_bytes,
                lambda value, limit: value > limit,
            ),
            (
                "GPU_RESERVED_LIMIT",
                "gpu_reserved_bytes",
                policy.gpu_reserved_max_bytes,
                lambda value, limit: value > limit,
            ),
            (
                "GPU_TEMPERATURE",
                "gpu_temperature_c",
                policy.gpu_temperature_max_c,
                lambda value, limit: value > limit,
            ),
        )
        for code, field, limit, breached in gates:
            value = resources.get(field)
            if value is not None and breached(value, limit):
                return code
        return None

    @staticmethod
    def _signal_group(process: subprocess.Popen[bytes], signum: int) -> None:
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass

    def run(
        self,
        trial: TrialIdentity,
        command: Sequence[str],
        *,
        heartbeat_path: str | Path,
        environment: Mapping[str, str] | None = None,
        signal_relay: SignalRelay | None = None,
        prepared: bool = False,
        result_path: str | Path | None = None,
    ) -> int:
        if prepared:
            if self.supervisor.ledger.state(trial) != "start":
                raise SupervisorError("prepared sidecar trial lacks a START ledger entry")
        else:
            self.supervisor.reserve(trial)
            self.supervisor.start_trial(trial)
        heartbeat = Path(heartbeat_path)
        heartbeat.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(environment or {})
        env["QM9_V31_HEARTBEAT_PATH"] = str(heartbeat)
        process = subprocess.Popen(list(command), env=env, start_new_session=True)
        started = time.monotonic()
        last_heartbeat = started
        last_heartbeat_mtime_ns: int | None = None
        heartbeat_payload: dict[str, object] = {}
        last_progress_sequence: object = None
        last_progress_stage: str | None = None
        last_model_progress = started
        next_telemetry = started
        soft_sent = False
        reason: str | None = None
        last_resources: dict[str, object] = {}
        while process.poll() is None:
            now = time.monotonic()
            if heartbeat.is_file():
                observed_mtime_ns = heartbeat.stat().st_mtime_ns
                if observed_mtime_ns != last_heartbeat_mtime_ns:
                    last_heartbeat = now
                    last_heartbeat_mtime_ns = observed_mtime_ns
                    try:
                        loaded_heartbeat = json.loads(heartbeat.read_text())
                        if isinstance(loaded_heartbeat, dict):
                            heartbeat_payload = loaded_heartbeat
                            sequence = heartbeat_payload.get("progress_sequence")
                            stage_value = heartbeat_payload.get("stage")
                            stage = stage_value if isinstance(stage_value, str) else "unknown"
                            if (
                                sequence != last_progress_sequence
                                or stage != last_progress_stage
                            ):
                                last_progress_sequence = sequence
                                last_progress_stage = stage
                                last_model_progress = now
                    except (OSError, UnicodeError, json.JSONDecodeError):
                        heartbeat_payload = {}
            elapsed = now - started
            heartbeat_age = now - last_heartbeat
            model_progress_age = now - last_model_progress
            progress_stage = last_progress_stage or "unknown"
            model_progress_stale_limit = self.policy.model_progress_stale_limit(
                trial.family, progress_stage
            )
            last_resources = self._sample_external_resources(process.pid)
            if now >= next_telemetry:
                progress = {
                    "process_alive": True,
                    "worker_heartbeat_age_seconds": heartbeat_age,
                    "worker_heartbeat_stale_limit_seconds": (
                        self.policy.heartbeat_stale_after_seconds
                    ),
                    "model_progress_age_seconds": model_progress_age,
                    "model_progress_stale_limit_seconds": model_progress_stale_limit,
                    "model_progress_stage": progress_stage,
                    "elapsed_seconds": elapsed,
                    "worker": heartbeat_payload,
                }
                if self.supervisor._run_purpose in {
                    "feasibility_evidence_generation",
                    "public_synthetic_three_backend_engineering_seam",
                }:
                    progress["label_reads"] = heartbeat_payload.get(
                        "label_reads",
                        {
                            "source_fixed_train_target_loads": 0,
                            "derived_train_target_artifact_reads": 0,
                            "outer_validation": 0,
                            "test": 0,
                            "protected": 0,
                            "private_evaluation": 0,
                        },
                    )
                self.supervisor.events.emit(
                    "heartbeat",
                    trial=trial,
                    progress=progress,
                    resources=last_resources,
                )
                next_telemetry = now + self.policy.heartbeat_seconds
            received_signal = signal_relay.poll() if signal_relay is not None else None
            if received_signal is not None and not soft_sent:
                reason = f"EXTERNAL_SIGNAL_{signal.Signals(received_signal).name}"
                self._signal_group(process, signal.SIGTERM)
                self.supervisor.events.emit(
                    "signal_received",
                    trial=trial,
                    progress={"direction": "external-to-sidecar-to-worker"},
                    failure={"signal": signal.Signals(received_signal).name, "reason": reason},
                    force_fsync=True,
                )
                soft_sent = True
            resource_reason = self._resource_breach(last_resources)
            if resource_reason is not None and not soft_sent:
                reason = resource_reason
                self._signal_group(process, signal.SIGTERM)
                self.supervisor.events.emit(
                    "hard_stop_requested",
                    trial=trial,
                    progress={"direction": "sidecar-to-worker", "reason": reason},
                    failure={"reason": reason, "resources": last_resources},
                    resources=last_resources,
                    force_fsync=True,
                )
                soft_sent = True
            if not soft_sent and (
                elapsed >= self.policy.soft_deadline_seconds
                or heartbeat_age >= self.policy.heartbeat_stale_after_seconds
                or model_progress_age >= model_progress_stale_limit
            ):
                reason = (
                    "STALE_HEARTBEAT"
                    if heartbeat_age >= self.policy.heartbeat_stale_after_seconds
                    else "STALE_MODEL_PROGRESS"
                    if model_progress_age >= model_progress_stale_limit
                    else "SOFT_DEADLINE"
                )
                self.supervisor.events.emit(
                    "soft_drain_requested",
                    trial=trial,
                    progress={"reason": reason, "elapsed_seconds": elapsed},
                    force_fsync=True,
                )
                self._signal_group(process, signal.SIGTERM)
                self.supervisor.events.emit(
                    "signal_received",
                    trial=trial,
                    progress={"direction": "sidecar-to-worker"},
                    failure={"signal": "SIGTERM", "reason": reason},
                    force_fsync=True,
                )
                soft_sent = True
            if elapsed >= self.policy.hard_deadline_seconds:
                reason = "ABSOLUTE_HARD_DEADLINE"
                self._signal_group(process, signal.SIGKILL)
                process.wait(timeout=2)
                record = {
                    "reason": reason,
                    "signal": "SIGKILL",
                    "worker_pid": process.pid,
                    "elapsed_seconds": elapsed,
                    "returncode": process.returncode,
                    "resources": last_resources,
                }
                self.supervisor.events.emit(
                    "external_termination",
                    trial=trial,
                    progress={"process_alive": False},
                    failure=record,
                    force_fsync=True,
                )
                self.supervisor.fail_trial(trial, record)
                self.supervisor.finalize_failure(record)
                return int(process.returncode or -signal.SIGKILL)
            time.sleep(self.policy.poll_seconds)
        returncode = int(process.returncode or 0)
        if returncode != 0 or soft_sent:
            record = {
                "reason": reason or "WORKER_EXIT_NONZERO",
                "worker_pid": process.pid,
                "returncode": returncode,
                "resources": last_resources,
            }
            self.supervisor.events.emit(
                "external_termination",
                trial=trial,
                progress={"process_alive": False},
                failure=record,
                force_fsync=True,
            )
            self.supervisor.fail_trial(trial, record)
            self.supervisor.finalize_failure(record)
        elif result_path is not None:
            try:
                self.supervisor.authenticate_worker_result(trial, result_path)
            except BaseException as error:
                record = {
                    "reason": "WORKER_RESULT_AUTHENTICATION_FAILED",
                    "worker_pid": process.pid,
                    "returncode": returncode,
                    "error_type": type(error).__name__,
                }
                self.supervisor.fail_trial(trial, record)
                self.supervisor.finalize_failure(record)
                raise
        return returncode


def run_governed_smoke_scaffold(
    root: str | Path,
    binding: RunBinding,
    specification: PlannedTrial,
    command: Sequence[str],
    policy: SidecarPolicy,
    *,
    signal_relay: SignalRelay | None = None,
) -> int:
    """Run one public/synthetic real-backend worker through every governance state."""

    supervisor = RunSupervisor(root, binding, [specification])
    try:
        return ExternalWorkerSidecar(supervisor, policy).run(
            specification.identity,
            command,
            heartbeat_path=Path(root) / "telemetry/worker.heartbeat",
            signal_relay=signal_relay,
        )
    finally:
        supervisor.close()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def sidecar_main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    spec = json.loads(args.spec.read_text())
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SupervisorError("sidecar command is required")
    binding = RunBinding(**spec["binding"])
    planned_payload = spec["planned_trial"]
    specification = PlannedTrial(
        identity=TrialIdentity(**planned_payload["identity"]),
        expected_shape=tuple(planned_payload["expected_shape"]),
        target_order=tuple(planned_payload["target_order"]),
        prediction_order_sha256=planned_payload["prediction_order_sha256"],
        candidate_sha256=planned_payload["candidate_sha256"],
    )
    policy = SidecarPolicy(**spec["policy"])
    with SignalRelay() as relay:
        return run_governed_smoke_scaffold(
            spec["output"], binding, specification, command, policy, signal_relay=relay
        )


if __name__ == "__main__":
    raise SystemExit(sidecar_main())
