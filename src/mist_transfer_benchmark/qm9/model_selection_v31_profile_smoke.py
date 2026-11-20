"""Two-stage, CPU-only engineering profile smoke for the QM9 v3.1 backends.

This module deliberately cannot rank, promote, compare, or ensemble models.  Its
bootstrap authorization exists only to measure the three real backend paths once.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import stat
import subprocess
import sys
import time
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

from .constants import TARGET_COLUMNS
from .model_selection_v31_backends import (
    artifact_file_sha256,
    build_memory_worksheet,
    cache_identity_sha256,
    freeze_cache_choice,
    load_real_backend_profile,
    profile_cache_options,
    write_real_backend_worker_spec,
)
from .model_selection_v31_checkpoint import (
    COMPLETION_STATUS,
    CheckpointStore,
    ReplayContract,
)
from .model_selection_v31_observability import (
    TrialIdentity,
    fsync_parent,
    validate_event_stream,
)
from .model_selection_v31_supervisor import (
    BOUNDED_TERMINAL_SCHEMA,
    RUN_START_SCHEMA,
    BoundedProfileRunBinding,
    ExternalWorkerSidecar,
    PlannedTrial,
    RunSupervisor,
    SidecarPolicy,
    SupervisorError,
    TrialLedger,
    atomic_bytes,
    atomic_json,
    canonical_sha256,
    detect_device_identity,
    file_sha256,
)

if TYPE_CHECKING:
    pass

RUN_ID = "qm9-v31-bounded-smoke-cpu-v5"
PURPOSE = "bounded_profile_only"
EXACT_BASE = Path("/tmp/qm9-v31-bounded-smoke-v5")
V1_INCIDENT_BASE = Path("/tmp/qm9-v31-bounded-smoke-v1")
V2_COMPLETED_BASE = Path("/tmp/qm9-v31-bounded-smoke-v2")
V3_REJECTED_BASE = Path("/tmp/qm9-v31-bounded-smoke-v3")
V4_READ_ONLY_BASE = Path("/tmp/qm9-v31-bounded-smoke-v4")
DEFAULT_RUNNER_IDENTITY = "qm9-v31-bounded-runner-v5"
SEED = 20260719
ROWS = 96
FEATURES = 64
TRAIN_ROWS = 72
TUNE_ROWS = 24
RSS_LIMIT = 2 * 1024**3
DISK_FLOOR = 1024**3
BOOTSTRAP_MAX_AGE_SECONDS = 30
AUTHORIZATION_MAX_LIFETIME_SECONDS = 1800
MINIMUM_EXECUTION_VALIDITY_SECONDS = 25 + 45 + 45 + 60
PROFILE_SCHEMA = "qm9-v31-bounded-bootstrap-profile-v5"
REVIEW_SCHEMA = "qm9-v31-bounded-bootstrap-review-v5"
AUTH_SCHEMA = "qm9-v31-bounded-bootstrap-authorization-v5"
SUMMARY_SCHEMA = "qm9-v31-bounded-profile-summary-v5"
POST_PROFILE_SCHEMA = "qm9-v31-post-profile-proposal-v5"
POSTRUN_MANIFEST_SCHEMA = "qm9-v31-bounded-postrun-manifest-v5"
PREPARE_FAILURE_SCHEMA = "qm9-v31-bounded-prepare-failed-v5"
PREPARE_CLAIM_SCHEMA = "qm9-v31-bounded-prepare-claim-v5"
PREPARE_COMPLETE_SCHEMA = "qm9-v31-bounded-prepare-complete-v5"
PREPARE_CLAIM_FAILURE_SCHEMA = "qm9-v31-bounded-prepare-claim-failed-v5"
EXECUTION_INCIDENT_SCHEMA = "qm9-v31-bounded-execution-incident-v5"
EXPECTED_PROJECT_NAME = "mist-transfer-benchmark"


def _now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _parse_time(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise SupervisorError(f"bounded bootstrap {name} is invalid")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise SupervisorError(f"bounded bootstrap {name} is invalid") from error


def _false_boundaries() -> dict[str, bool]:
    return {
        "w1_feasibility_approved": False,
        "general_training_approved": False,
        "private_data_allowed": False,
        "ranking_allowed": False,
        "promotion_allowed": False,
        "ensemble_allowed": False,
        "scientific_result": False,
        "publication_ready": False,
        "outer_validation_targets_read": False,
        "test_targets_read": False,
    }


def _read_json(path: str | Path) -> dict[str, object]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SupervisorError("bounded bootstrap artifact cannot be read") from error
    if not isinstance(value, dict):
        raise SupervisorError("bounded bootstrap artifact must be an object")
    return value


def _validate_semantic_artifact(
    path: str | Path, schema: str, *, expected_file_sha256: str | None = None
) -> dict[str, object]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise SupervisorError("bounded artifact must be a regular non-symlink file")
    artifact = _read_json(path)
    unsigned = dict(artifact)
    semantic = unsigned.pop("semantic_sha256", None)
    if artifact.get("schema_version") != schema or semantic != canonical_sha256(unsigned):
        raise SupervisorError("bounded bootstrap semantic hash differs")
    false_keys = set(_false_boundaries())
    schema_keys = {
        PROFILE_SCHEMA: {
            "schema_version",
            "purpose",
            "data_scope",
            "device_type",
            "run_id",
            "runner_identity",
            "seed",
            "shape",
            "input_file_sha256",
            "prediction_order_sha256",
            "candidate_sha256",
            "plan_sha256",
            "config_sha256",
            "code_sha256",
            "dependency_sha256",
            "repository_provenance",
            "repository_provenance_sha256",
            "prepare_attempt_id",
            "prepare_claim_file_sha256",
            "prepare_claim_semantic_sha256",
            "cache_choice_file_sha256",
            "prepare_tree_manifest",
            "prepare_tree_manifest_sha256",
            "device_identity",
            "device_identity_sha256",
            "memory_worksheet",
            "memory_worksheet_sha256",
            "observed_at",
            "observation_max_age_seconds",
            "available_ram_bytes",
            "free_disk_bytes",
            "peak_rss_measurement_status",
            "runtime_measurement_status",
            "policies",
            "bounded_profile_approved",
            "semantic_sha256",
        }
        | false_keys,
        REVIEW_SCHEMA: {
            "schema_version",
            "decision",
            "reviewed_at_utc",
            "reviewer_identity",
            "review_tool_identity",
            "review_tool_code_sha256",
            "runner_identity",
            "profile_path",
            "profile_file_sha256",
            "profile_semantic_sha256",
            "prepare_complete_path",
            "prepare_complete_file_sha256",
            "prepare_complete_semantic_sha256",
            "purpose",
            "device_type",
            "data_scope",
            "shape",
            "candidate_sha256",
            "policies",
            "semantic_sha256",
        }
        | false_keys,
        AUTH_SCHEMA: {
            "schema_version",
            "purpose",
            "decision",
            "nonce",
            "issued_at_utc",
            "expires_at_utc",
            "profile_path",
            "profile_file_sha256",
            "profile_semantic_sha256",
            "prepare_complete_path",
            "prepare_complete_file_sha256",
            "prepare_complete_semantic_sha256",
            "review_path",
            "review_file_sha256",
            "review_semantic_sha256",
            "reviewer_identity",
            "review_tool_identity",
            "review_tool_code_sha256",
            "runner_identity",
            "semantic_sha256",
        }
        | false_keys,
        SUMMARY_SCHEMA: {
            "schema_version",
            "purpose",
            "status",
            "run_id",
            "trial_states",
            "event_count",
            "checkpoint_count",
            "replay_count",
            "replay_sha256",
            "ledger_sha256",
            "duration_seconds",
            "resource_peaks",
            "profiles",
            "mlp_cache",
            "protected_label_reads",
            "semantic_sha256",
        }
        | false_keys,
        POST_PROFILE_SCHEMA: {
            "schema_version",
            "measurement_status",
            "decision",
            "observations_per_family",
            "families",
            "w1_feasibility_approved",
            "general_training_approved",
            "scientific_result",
            "publication_ready",
            "semantic_sha256",
        },
        POSTRUN_MANIFEST_SCHEMA: {
            "schema_version",
            "status",
            "run_id",
            "purpose",
            "created_at_utc",
            "success_lock",
            "streams_closed",
            "expected_final_path_types",
            "expected_final_path_types_sha256",
            "run_start_path",
            "run_start_file_sha256",
            "run_start_semantic_sha256",
            "ledger_path",
            "ledger_file_sha256",
            "terminal_path",
            "terminal_file_sha256",
            "summary_path",
            "summary_file_sha256",
            "summary_semantic_sha256",
            "proposal_path",
            "proposal_file_sha256",
            "proposal_semantic_sha256",
            "execution_repository_provenance_sha256",
            "final_base_inventory",
            "final_base_inventory_sha256",
            "semantic_sha256",
        }
        | false_keys,
        PREPARE_FAILURE_SCHEMA: {
            "schema_version",
            "status",
            "stage",
            "reason",
            "error_type",
            "error_message",
            "failure_code",
            "failed_at_utc",
            "prepare_attempt_id",
            "prepare_claim_path",
            "prepare_claim_file_sha256",
            "prepare_claim_semantic_sha256",
            "last_completed_checkpoint",
            "partial_tree_inventory",
            "partial_tree_inventory_sha256",
            "run_id",
            "purpose",
            "runner_identity",
            "prerequisite_provenance",
            "prerequisite_provenance_sha256",
            "resumable",
            "authorization_allowed",
            "execution_allowed",
            "semantic_sha256",
        }
        | false_keys,
        PREPARE_CLAIM_SCHEMA: {
            "schema_version",
            "prepare_attempt_id",
            "claimed_at_utc",
            "base_path",
            "run_id",
            "purpose",
            "runner_identity",
            "prerequisite_provenance",
            "prerequisite_provenance_sha256",
            "resumable",
            "authorization_allowed",
            "execution_allowed",
            "semantic_sha256",
        }
        | false_keys,
        PREPARE_CLAIM_FAILURE_SCHEMA: {
            "schema_version",
            "status",
            "prepare_attempt_id",
            "failed_at_utc",
            "failure_code",
            "error_type",
            "error_message",
            "base_path",
            "base_inventory",
            "base_inventory_sha256",
            "run_id",
            "purpose",
            "runner_identity",
            "prerequisite_provenance",
            "prerequisite_provenance_sha256",
            "resumable",
            "authorization_allowed",
            "execution_allowed",
            "semantic_sha256",
        }
        | false_keys,
        EXECUTION_INCIDENT_SCHEMA: {
            "schema_version",
            "status",
            "run_id",
            "purpose",
            "failed_at_utc",
            "failure_stage",
            "exception_type",
            "exception_message_sha256",
            "terminal_state",
            "event_state",
            "ledger_state",
            "postrun_state",
            "resumable",
            "execution_allowed",
            "ranking_allowed",
            "semantic_sha256",
        }
        | false_keys,
        PREPARE_COMPLETE_SCHEMA: {
            "schema_version",
            "status",
            "prepare_attempt_id",
            "prepare_claim_path",
            "prepare_claim_file_sha256",
            "prepare_claim_semantic_sha256",
            "completed_at_utc",
            "run_id",
            "purpose",
            "runner_identity",
            "preflight_path",
            "preflight_inventory",
            "preflight_inventory_sha256",
            "base_inventory",
            "base_inventory_sha256",
            "profile_path",
            "profile_file_sha256",
            "profile_semantic_sha256",
            "prerequisite_provenance",
            "prerequisite_provenance_sha256",
            "resumable",
            "authorization_allowed",
            "execution_allowed",
            "semantic_sha256",
        }
        | false_keys,
    }
    resource_keys = {
        "schema_version",
        "observed_at",
        "available_ram_bytes",
        "current_rss_bytes",
        "free_disk_bytes",
        "static_required_with_margin_bytes",
        "rss_limit_bytes",
        "disk_floor_bytes",
        "reserved_artifact_allowance_bytes",
        "inequalities",
        "approved",
        "authorization_nonce",
        "device_identity",
        "device_identity_sha256",
        "run_id",
        "purpose",
        "runner_identity",
        "semantic_sha256",
    } | false_keys
    schema_keys.update(
        {
            "qm9-v31-bounded-attempt-claim-v5": {
                "schema_version",
                "nonce",
                "authorization_file_sha256",
                "authorization_semantic_sha256",
                "runner_identity",
                "run_id",
                "purpose",
                "claimed_at_utc",
                "semantic_sha256",
            }
            | false_keys,
            "qm9-v31-bounded-resource-observation-v5": resource_keys,
            "qm9-v31-bounded-trial-resource-observation-v5": resource_keys
            | {
                "trial",
                "trial_key",
                "policy_sha256",
                "run_start_sha256",
                "receipt_file_sha256",
                "previous_ledger_entry_sha256",
            },
            "qm9-v31-bounded-authorization-consumed-v5": {
                "schema_version",
                "authorization_semantic_sha256",
                "authorization_file_sha256",
                "nonce",
                "run_start_sha256",
                "run_start_file_sha256",
                "run_start_semantic_sha256",
                "claim_file_sha256",
                "observation_file_sha256",
                "runner_identity",
                "run_id",
                "purpose",
                "consumed_at_utc",
                "semantic_sha256",
            }
            | false_keys,
        }
    )
    if schema in schema_keys and set(artifact) != schema_keys[schema]:
        raise SupervisorError("bounded bootstrap exact key schema differs")
    if schema in {
        PREPARE_FAILURE_SCHEMA,
        PREPARE_CLAIM_SCHEMA,
        PREPARE_CLAIM_FAILURE_SCHEMA,
        PREPARE_COMPLETE_SCHEMA,
    }:
        provenance = artifact.get("prerequisite_provenance")
        identity = dict(provenance) if isinstance(provenance, dict) else {}
        observed_identity_sha256 = identity.pop("repository_identity_sha256", None)
        if (
            not isinstance(provenance, dict)
            or set(provenance) != _repository_provenance_keys()
            or artifact.get("prerequisite_provenance_sha256") != canonical_sha256(provenance)
            or observed_identity_sha256 != canonical_sha256(identity)
            or provenance.get("project_name") != EXPECTED_PROJECT_NAME
            or len(str(artifact.get("prepare_attempt_id"))) != 64
            or any(
                character not in "0123456789abcdef"
                for character in str(artifact.get("prepare_attempt_id"))
            )
            or any(artifact.get(field) is not False for field in false_keys)
        ):
            raise SupervisorError("bounded prepare lifecycle provenance differs")
    if schema == PREPARE_FAILURE_SCHEMA and (
        artifact.get("status") != "PREPARE_FAILED"
        or artifact.get("stage") != "prepare-preflight"
        or artifact.get("reason") != "PREPARE_EXCEPTION"
        or artifact.get("resumable") is not False
        or artifact.get("authorization_allowed") is not False
        or artifact.get("execution_allowed") is not False
        or artifact.get("partial_tree_inventory_sha256")
        != canonical_sha256(artifact.get("partial_tree_inventory"))
    ):
        raise SupervisorError("bounded prepare-failure values differ")
    if schema == PREPARE_CLAIM_SCHEMA and (
        artifact.get("resumable") is not False
        or artifact.get("authorization_allowed") is not False
        or artifact.get("execution_allowed") is not False
    ):
        raise SupervisorError("bounded prepare-claim values differ")
    if schema == PREPARE_CLAIM_FAILURE_SCHEMA and (
        artifact.get("status") != "PREPARE_CLAIM_FAILED"
        or artifact.get("failure_code") != "PREPARE_CLAIM_WRITE_FAILURE"
        or artifact.get("resumable") is not False
        or artifact.get("authorization_allowed") is not False
        or artifact.get("execution_allowed") is not False
        or artifact.get("base_inventory_sha256") != canonical_sha256(artifact.get("base_inventory"))
    ):
        raise SupervisorError("bounded prepare-claim-failure values differ")
    if schema == PREPARE_COMPLETE_SCHEMA and (
        artifact.get("status") != "PREPARE_COMPLETE"
        or artifact.get("resumable") is not False
        or artifact.get("authorization_allowed") is not True
        or artifact.get("execution_allowed") is not True
        or artifact.get("preflight_inventory_sha256")
        != canonical_sha256(artifact.get("preflight_inventory"))
        or artifact.get("base_inventory_sha256") != canonical_sha256(artifact.get("base_inventory"))
    ):
        raise SupervisorError("bounded prepare-complete values differ")
    if schema == POSTRUN_MANIFEST_SCHEMA and (
        artifact.get("status") != "PROFILE_COMPLETE"
        or artifact.get("run_id") != RUN_ID
        or artifact.get("purpose") != PURPOSE
        or artifact.get("success_lock") is not True
        or artifact.get("streams_closed") != {"ledger": True, "telemetry": True}
        or artifact.get("expected_final_path_types_sha256")
        != canonical_sha256(artifact.get("expected_final_path_types"))
        or artifact.get("final_base_inventory_sha256")
        != canonical_sha256(artifact.get("final_base_inventory"))
        or any(artifact.get(field) is not False for field in false_keys)
    ):
        raise SupervisorError("bounded post-run manifest values differ")
    if schema == EXECUTION_INCIDENT_SCHEMA and (
        artifact.get("status") != "EXECUTION_INCIDENT"
        or artifact.get("run_id") != RUN_ID
        or artifact.get("purpose") != PURPOSE
        or artifact.get("resumable") is not False
        or artifact.get("execution_allowed") is not False
        or artifact.get("ranking_allowed") is not False
        or any(artifact.get(field) is not False for field in false_keys)
    ):
        raise SupervisorError("bounded execution-incident values differ")
    if expected_file_sha256 is not None and file_sha256(path) != expected_file_sha256:
        raise SupervisorError("bounded bootstrap physical hash differs")
    return artifact


def _write_immutable_json(path: Path, payload: Mapping[str, object]) -> dict[str, object]:
    if path.exists():
        raise SupervisorError(f"immutable artifact already exists: {path}")
    unsigned = dict(payload)
    artifact = {**unsigned, "semantic_sha256": canonical_sha256(unsigned)}
    atomic_json(path, artifact)
    return artifact


def _write_exclusive_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    fsync_parent(path)


def _exclusive_artifact(
    path: Path, schema_version: str, payload: Mapping[str, object]
) -> dict[str, object]:
    unsigned = {"schema_version": schema_version, **dict(payload)}
    artifact = {**unsigned, "semantic_sha256": canonical_sha256(unsigned)}
    _write_exclusive_json(path, artifact)
    return artifact


def _available_ram_bytes() -> int:
    try:
        rows = Path("/proc/meminfo").read_text().splitlines()
        return int(next(row.split()[1] for row in rows if row.startswith("MemAvailable:"))) * 1024
    except (OSError, StopIteration, ValueError) as error:
        raise SupervisorError("current available RAM sensor is unavailable") from error


def _current_rss_bytes() -> int:
    try:
        return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError) as error:
        raise SupervisorError("current RSS sensor is unavailable") from error


def _regular_lstat(path: Path, name: str) -> os.stat_result:
    try:
        observed = path.lstat()
    except OSError as error:
        raise SupervisorError(f"repository prerequisite is unavailable: {name}") from error
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        raise SupervisorError(f"repository prerequisite is not a regular file: {name}")
    return observed


def _project_name(pyproject: Path) -> str:
    try:
        parsed = tomllib.loads(pyproject.read_text())
        name = parsed["project"]["name"]
    except (OSError, UnicodeError, tomllib.TOMLDecodeError, KeyError, TypeError) as error:
        raise SupervisorError("repository project identity is unavailable") from error
    if name != EXPECTED_PROJECT_NAME:
        raise SupervisorError("repository project identity differs")
    return str(name)


def _repository_root(source_path: Path | None = None) -> Path:
    source = (source_path or Path(__file__)).resolve()
    _regular_lstat(source, "profile runner source")
    candidates: list[Path] = []
    for ancestor in (source.parent, *source.parents):
        pyproject = ancestor / "pyproject.toml"
        lock = ancestor / "uv.lock"
        package = ancestor / "src/mist_transfer_benchmark"
        try:
            pyproject_mode = pyproject.lstat().st_mode
            lock_mode = lock.lstat().st_mode
            package_mode = package.lstat().st_mode
            source.relative_to(ancestor)
        except (OSError, ValueError):
            continue
        if (
            stat.S_ISREG(pyproject_mode)
            and not stat.S_ISLNK(pyproject_mode)
            and stat.S_ISREG(lock_mode)
            and not stat.S_ISLNK(lock_mode)
            and stat.S_ISDIR(package_mode)
            and not stat.S_ISLNK(package_mode)
        ):
            candidates.append(ancestor.resolve())
    unique_candidates = sorted(set(candidates))
    if len(unique_candidates) != 1:
        raise SupervisorError("repository ancestor markers are missing or ambiguous")
    selected = unique_candidates[0]
    _project_name(selected / "pyproject.toml")
    try:
        git_root = Path(
            subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=source.parent,
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        ).resolve()
    except (OSError, subprocess.SubprocessError) as error:
        raise SupervisorError("verified git repository root is unavailable") from error
    if git_root != selected:
        raise SupervisorError("verified git top-level differs from unique project ancestor")
    return selected


def _code_state_sha256(repository: Path | None = None) -> str:
    repository = (repository or _repository_root()).resolve()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty_patch = subprocess.run(
            ["git", "diff", "HEAD", "--binary", "--no-ext-diff"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        source_paths = subprocess.run(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "src/mist_transfer_benchmark/qm9/model_selection_v31_*.py",
            ],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError) as error:
        raise SupervisorError("git code/dirty-patch provenance is unavailable") from error
    return canonical_sha256(
        {
            "commit": commit,
            "dirty_patch": dirty_patch,
            "status": status,
            "v31_source_manifest": [
                {"path": relative, "file_sha256": file_sha256(repository / relative)}
                for relative in sorted(source_paths)
            ],
            "profile_runner_file_sha256": file_sha256(__file__),
        }
    )


def _repository_provenance_keys() -> set[str]:
    return {
        "repository_root",
        "git_top_level",
        "project_name",
        "repository_identity_sha256",
        "runner_source_path",
        "runner_source_relative_path",
        "runner_source_size",
        "runner_source_mode",
        "runner_source_sha256",
        "pyproject_path",
        "pyproject_size",
        "pyproject_mode",
        "pyproject_sha256",
        "dependency_lock_path",
        "dependency_lock_size",
        "dependency_lock_mode",
        "dependency_sha256",
        "code_state_sha256",
    }


def _repository_preflight() -> dict[str, object]:
    source = Path(__file__).resolve()
    repository = _repository_root(source)
    project = repository / "pyproject.toml"
    lock = repository / "uv.lock"
    source_stat = _regular_lstat(source, "profile runner source")
    project_stat = _regular_lstat(project, "pyproject.toml")
    lock_stat = _regular_lstat(lock, "uv.lock")
    project_name = _project_name(project)
    source_relative = str(source.relative_to(repository))
    provenance: dict[str, object] = {
        "repository_root": str(repository),
        "git_top_level": str(repository),
        "project_name": project_name,
        "runner_source_relative_path": source_relative,
        "runner_source_path": str(source),
        "runner_source_size": source_stat.st_size,
        "runner_source_mode": stat.S_IMODE(source_stat.st_mode),
        "runner_source_sha256": file_sha256(source),
        "pyproject_path": str(project),
        "pyproject_size": project_stat.st_size,
        "pyproject_mode": stat.S_IMODE(project_stat.st_mode),
        "pyproject_sha256": file_sha256(project),
        "dependency_lock_path": str(lock),
        "dependency_lock_size": lock_stat.st_size,
        "dependency_lock_mode": stat.S_IMODE(lock_stat.st_mode),
        "dependency_sha256": file_sha256(lock),
        "code_state_sha256": _code_state_sha256(repository),
    }
    provenance["repository_identity_sha256"] = canonical_sha256(provenance)
    if set(provenance) != _repository_provenance_keys():
        raise SupervisorError("repository provenance schema differs")
    return provenance


def _lstat_tree(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not root.is_dir() or root.is_symlink():
        raise SupervisorError("prepare inventory root must be a regular directory")
    for candidate in sorted(root.rglob("*")):
        observed = candidate.lstat()
        if stat.S_ISREG(observed.st_mode):
            entry_type = "file"
            digest: str | None = file_sha256(candidate)
        elif stat.S_ISDIR(observed.st_mode):
            entry_type = "directory"
            digest = None
        elif stat.S_ISLNK(observed.st_mode):
            entry_type = "symlink"
            digest = None
        else:
            entry_type = "special"
            digest = None
        rows.append(
            {
                "relative_path": str(candidate.relative_to(root)),
                "entry_type": entry_type,
                "mode": stat.S_IMODE(observed.st_mode),
                "bytes": observed.st_size,
                "file_sha256": digest,
            }
        )
    return rows


def _validate_lstat_tree(
    root: Path,
    inventory: object,
    *,
    allowed_later: set[Path] | None = None,
    allowed_prefixes: set[Path] | None = None,
) -> None:
    if not isinstance(inventory, list):
        raise SupervisorError("prepare inventory is unavailable")
    expected_paths: set[Path] = set()
    for row in inventory:
        if not isinstance(row, Mapping) or set(row) != {
            "relative_path",
            "entry_type",
            "mode",
            "bytes",
            "file_sha256",
        }:
            raise SupervisorError("prepare inventory row schema differs")
        relative = Path(str(row["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise SupervisorError("prepare inventory path escapes root")
        candidate = root / relative
        try:
            observed = candidate.lstat()
        except OSError as error:
            raise SupervisorError("prepare inventory entry is missing") from error
        observed_type = (
            "file"
            if stat.S_ISREG(observed.st_mode)
            else "directory"
            if stat.S_ISDIR(observed.st_mode)
            else "symlink"
            if stat.S_ISLNK(observed.st_mode)
            else "special"
        )
        if (
            observed_type != row["entry_type"]
            or stat.S_IMODE(observed.st_mode) != row["mode"]
            or observed.st_size != row["bytes"]
            or (observed_type == "file" and file_sha256(candidate) != row["file_sha256"])
            or (observed_type != "file" and row["file_sha256"] is not None)
        ):
            raise SupervisorError("prepare inventory physical state differs")
        expected_paths.add(candidate)
    actual_paths = set(root.rglob("*"))
    extras = actual_paths - expected_paths - (allowed_later or set())
    prefixes = allowed_prefixes or set()
    if any(
        not any(candidate == prefix or prefix in candidate.parents for prefix in prefixes)
        for candidate in extras
    ):
        raise SupervisorError("prepare inventory contains unexpected entries")
    for candidate in extras:
        mode = candidate.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise SupervisorError("prepare transition contains a nonregular entry")


def _lexists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise SupervisorError("prepare terminal state cannot be inspected") from error
    return True


def _validate_prepare_state(
    base: Path,
    *,
    allow_execute_transition: bool = False,
    allow_post_run: bool = False,
    revalidate_repository: bool = True,
) -> tuple[dict[str, object], dict[str, object]]:
    if base.resolve() == V1_INCIDENT_BASE.resolve():
        raise SupervisorError("V1_INCIDENT_INCOMPLETE: v1 is permanently nonresumable")
    if base.resolve() != EXACT_BASE.resolve():
        raise SupervisorError("prepare state belongs to another base")
    failure_path = base / "PREPARE_FAILED.json"
    claim_failure_path = base / "PREPARE_CLAIM_FAILED.json"
    if _lexists(failure_path) or _lexists(claim_failure_path):
        raise SupervisorError("prepare failed; authorization and execution are forbidden")
    claim_path = base / "prepare-claim.json"
    complete_path = base / "PREPARE_COMPLETE.json"
    claim = _validate_semantic_artifact(claim_path, PREPARE_CLAIM_SCHEMA)
    complete = _validate_semantic_artifact(complete_path, PREPARE_COMPLETE_SCHEMA)
    profile_path = base / "preflight/bootstrap-profile.json"
    profile = _validate_semantic_artifact(profile_path, PROFILE_SCHEMA)
    frozen_provenance = complete.get("prerequisite_provenance")
    if not isinstance(frozen_provenance, dict):
        raise SupervisorError("prepare repository provenance is unavailable")
    provenance = _repository_preflight() if revalidate_repository else frozen_provenance
    allowed_later = {
        base / "preflight/independent-review.json",
        base / "preflight/bootstrap-authorization.json",
    }
    _validate_lstat_tree(
        base / "preflight",
        complete.get("preflight_inventory"),
        allowed_later={path for path in allowed_later if path.exists()},
    )
    base_allowed_later = {
        base / "PREPARE_COMPLETE.json",
        *{path for path in allowed_later if path.exists()},
    }
    if allow_post_run:
        base_allowed_later.update(
            {
                base / "profile-summary.json",
                base / "post-profile",
                base / "post-profile/proposal.json",
            }
        )
        if _lexists(base / "POSTRUN_MANIFEST.json"):
            base_allowed_later.add(base / "POSTRUN_MANIFEST.json")
    execute_prefixes = {base / "attempt", base / "run"} if allow_execute_transition else set()
    _validate_lstat_tree(
        base,
        complete.get("base_inventory"),
        allowed_later=base_allowed_later,
        allowed_prefixes=execute_prefixes,
    )
    if allow_post_run:
        _validate_post_run_outputs(base)
    if (
        claim.get("base_path") != str(base.resolve())
        or claim.get("run_id") != RUN_ID
        or claim.get("purpose") != PURPOSE
        or claim.get("prerequisite_provenance") != provenance
        or complete.get("prepare_attempt_id") != claim.get("prepare_attempt_id")
        or complete.get("prepare_claim_path") != str(claim_path.resolve())
        or complete.get("prepare_claim_file_sha256") != file_sha256(claim_path)
        or complete.get("prepare_claim_semantic_sha256") != claim.get("semantic_sha256")
        or complete.get("run_id") != RUN_ID
        or complete.get("purpose") != PURPOSE
        or complete.get("runner_identity") != claim.get("runner_identity")
        or complete.get("preflight_path") != str((base / "preflight").resolve())
        or complete.get("profile_path") != str(profile_path.resolve())
        or complete.get("profile_file_sha256") != file_sha256(profile_path)
        or complete.get("profile_semantic_sha256") != profile.get("semantic_sha256")
        or complete.get("prerequisite_provenance") != provenance
        or profile.get("repository_provenance") != provenance
        or profile.get("repository_provenance_sha256") != canonical_sha256(provenance)
        or profile.get("prepare_attempt_id") != claim.get("prepare_attempt_id")
        or profile.get("prepare_claim_file_sha256") != file_sha256(claim_path)
        or profile.get("prepare_claim_semantic_sha256") != claim.get("semantic_sha256")
    ):
        raise SupervisorError("prepare claim/complete provenance chain differs")
    return claim, complete


def _validate_post_run_outputs(base: Path) -> tuple[dict[str, object], dict[str, object]]:
    summary = _validate_semantic_artifact(base / "profile-summary.json", SUMMARY_SCHEMA)
    proposal = _validate_semantic_artifact(base / "post-profile/proposal.json", POST_PROFILE_SCHEMA)
    if (
        summary.get("purpose") != "engineering_smoke"
        or summary.get("status") != "complete"
        or summary.get("run_id") != RUN_ID
        or summary.get("checkpoint_count") != 3
        or summary.get("protected_label_reads") != 0
        or any(summary.get(field) is not False for field in _false_boundaries())
        or proposal.get("measurement_status") != "INSUFFICIENT_OBSERVATIONS"
        or proposal.get("decision") != "NOT_APPROVED"
        or proposal.get("observations_per_family") != 1
        or proposal.get("families") != ["ridge", "xgboost", "mlp"]
        or proposal.get("w1_feasibility_approved") is not False
        or proposal.get("general_training_approved") is not False
        or proposal.get("scientific_result") is not False
        or proposal.get("publication_ready") is not False
    ):
        raise SupervisorError("bounded post-run output scope differs")
    return summary, proposal


def _validate_postrun_manifest_bindings(
    base: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    summary, proposal = _validate_post_run_outputs(base)
    postrun_path = base / "POSTRUN_MANIFEST.json"
    postrun = _validate_semantic_artifact(postrun_path, POSTRUN_MANIFEST_SCHEMA)
    _validate_path_type_contract(
        base,
        _path_type_contract_for_phase(postrun.get("expected_final_path_types"), "postrun"),
    )
    _validate_lstat_tree(
        base,
        postrun.get("final_base_inventory"),
        allowed_later={postrun_path},
    )
    run_start_path = base / "run/run_start.json"
    run_start = _read_json(run_start_path)
    unsigned_start = dict(run_start)
    observed_start_semantic = unsigned_start.pop("semantic_sha256", None)
    terminal_path = base / "run/final/manifest.json"
    ledger_path = base / "run/ledger/trials.jsonl"
    if (
        observed_start_semantic != canonical_sha256(unsigned_start)
        or terminal_path.is_symlink()
        or not terminal_path.is_file()
        or ledger_path.is_symlink()
        or not ledger_path.is_file()
        or postrun.get("run_start_path") != str(run_start_path.resolve())
        or postrun.get("run_start_file_sha256") != file_sha256(run_start_path)
        or postrun.get("run_start_semantic_sha256") != observed_start_semantic
        or postrun.get("ledger_path") != str(ledger_path.resolve())
        or postrun.get("ledger_file_sha256") != file_sha256(ledger_path)
        or postrun.get("terminal_path") != str(terminal_path.resolve())
        or postrun.get("terminal_file_sha256") != file_sha256(terminal_path)
        or postrun.get("summary_path") != str((base / "profile-summary.json").resolve())
        or postrun.get("summary_file_sha256") != file_sha256(base / "profile-summary.json")
        or postrun.get("summary_semantic_sha256") != summary.get("semantic_sha256")
        or postrun.get("proposal_path") != str((base / "post-profile/proposal.json").resolve())
        or postrun.get("proposal_file_sha256") != file_sha256(base / "post-profile/proposal.json")
        or postrun.get("proposal_semantic_sha256") != proposal.get("semantic_sha256")
    ):
        raise SupervisorError("bounded post-run manifest binding differs")
    return postrun, summary, proposal


def _path_type_contract_for_phase(contract: object, phase: str) -> list[dict[str, str]]:
    if not isinstance(contract, list):
        raise SupervisorError("bounded final path/type contract schema differs")
    terminal = [dict(row) for row in contract if isinstance(row, Mapping)]
    if len(terminal) != len(contract):
        raise SupervisorError("bounded final path/type contract schema differs")
    if phase == "preterminal":
        return [
            row
            for row in terminal
            if row.get("relative_path") not in {"run/final", "run/final/manifest.json"}
        ]
    if phase == "terminal":
        return terminal
    if phase == "postrun":
        return [
            *terminal,
            {"relative_path": "POSTRUN_MANIFEST.json", "entry_type": "file"},
        ]
    raise SupervisorError("bounded final path/type contract phase differs")


def _validate_path_type_contract(root: Path, contract: object) -> None:
    if not isinstance(contract, list) or any(
        not isinstance(row, Mapping)
        or set(row) != {"relative_path", "entry_type"}
        or row.get("entry_type") not in {"file", "directory"}
        for row in contract
    ):
        raise SupervisorError("bounded final path/type contract schema differs")
    expected = {(str(row["relative_path"]), str(row["entry_type"])) for row in contract}
    if len(expected) != len(contract):
        raise SupervisorError("bounded final path/type contract contains duplicates")
    actual = {
        (
            str(path.relative_to(root)),
            "file" if stat.S_ISREG(path.lstat().st_mode) else "directory",
        )
        for path in root.rglob("*")
        if stat.S_ISREG(path.lstat().st_mode) or stat.S_ISDIR(path.lstat().st_mode)
    }
    if actual != expected or any(
        stat.S_ISLNK(path.lstat().st_mode)
        or not (stat.S_ISREG(path.lstat().st_mode) or stat.S_ISDIR(path.lstat().st_mode))
        for path in root.rglob("*")
    ):
        raise SupervisorError("bounded final path/type contract differs")


def _expected_final_path_types(
    base: Path,
    specifications: Sequence[PlannedTrial],
    replayed: Sequence[Mapping[str, object]],
    worker_profiles: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    """Derive the exact final tree from frozen prepare evidence and authenticated outputs."""

    prepare_complete = _validate_semantic_artifact(
        base / "PREPARE_COMPLETE.json", PREPARE_COMPLETE_SCHEMA
    )
    rows = prepare_complete.get("base_inventory")
    if not isinstance(rows, list):
        raise SupervisorError("bounded frozen prepare inventory is unavailable")
    expected = {
        (str(row["relative_path"]), str(row["entry_type"]))
        for row in rows
        if isinstance(row, Mapping)
    }
    fixed_directories = {
        "attempt",
        "post-profile",
        "run",
        "run/cache",
        "run/cache/features",
        "run/checkpoints",
        "run/final",
        "run/ledger",
        "run/telemetry",
        "run/worker-profiles",
        "run/worker-results",
        "run/worker-specs",
    }
    fixed_files = {
        "PREPARE_COMPLETE.json",
        "preflight/independent-review.json",
        "preflight/bootstrap-authorization.json",
        "attempt/authorization-consumed.json",
        "attempt/nonce-claim.json",
        "attempt/resource-observation.json",
        "post-profile/proposal.json",
        "profile-summary.json",
        "run/cache/cache-choice.json",
        "run/final/manifest.json",
        "run/ledger/trials.jsonl",
        "run/run_start.json",
        "run/telemetry/events.jsonl",
    }
    cache_identity = {
        "feature_artifact_sha256": artifact_file_sha256(base / "preflight/public-synthetic.npz"),
        "recipe_provenance_sha256": canonical_sha256({"threshold": 0.70, "dtype": "float64"}),
        "strategy": "identity",
        "fold": 0,
        "train_ordered_sha256": canonical_sha256(list(range(TRAIN_ROWS))),
        "tune_ordered_sha256": canonical_sha256(list(range(TRAIN_ROWS, ROWS))),
    }
    fixed_files.add(_run_cache_manifest_relative_path(cache_identity))
    by_family = {str(row["family"]): row for row in worker_profiles}
    if len(replayed) != len(specifications):
        raise SupervisorError("bounded replay count differs from final path contract")
    for specification in specifications:
        family = specification.identity.family
        key = canonical_sha256(asdict(specification.identity))
        fixed_directories.add(f"run/checkpoints/{key}")
        fixed_files.update(
            {
                f"attempt/resource-observation-{specification.identity.trial_index}-{family}.json",
                f"run/telemetry/{family}.heartbeat",
                f"run/worker-profiles/{family}.json",
                f"run/worker-results/{family}.json",
                f"run/worker-specs/{family}.json",
                f"run/checkpoints/{key}/latest.json",
            }
        )
        evidence = by_family[family]["metadata"]["checkpoint_evidence"]
        fixed_files.add(f"run/checkpoints/{evidence['relative_path']}")
    mlp_key = canonical_sha256(
        asdict(next(row.identity for row in specifications if row.identity.family == "mlp"))
    )
    scores = by_family["mlp"]["metadata"]["curve"]["inner_tune_mnmae"]
    expected_best = {
        index + 1: float(score)
        for index, score in enumerate(scores)
        if index == 0 or float(score) < min(float(value) for value in scores[:index])
    }
    best_paths = sorted((base / f"run/checkpoints/{mlp_key}").glob("best-*.npz"))
    bindings = replayed[0]["manifest"]["bindings"]
    store = CheckpointStore(
        base / "run/checkpoints",
        plan_sha256=str(bindings["plan_sha256"]),
        config_sha256=str(bindings["config_sha256"]),
        input_sha256=str(bindings["input_sha256"]),
        cache_sha256=str(bindings["cache_sha256"]),
        run_start_sha256=str(bindings["run_start_sha256"]),
    )
    authenticated_epochs: set[int] = set()
    mlp_identity = asdict(
        next(row.identity for row in specifications if row.identity.family == "mlp")
    )
    for best_path in best_paths:
        _, manifest = store.load_tensor_state(best_path)
        payload = manifest.get("payload")
        epoch = payload.get("epoch") if isinstance(payload, Mapping) else None
        if (
            manifest.get("kind") != "best"
            or manifest.get("trial") != mlp_identity
            or not isinstance(epoch, int)
            or epoch in authenticated_epochs
            or expected_best.get(epoch) != payload.get("inner_tune_mnmae")
            or payload.get("cache_identity_sha256")
            != by_family["mlp"]["metadata"].get("cache_identity_sha256")
            or payload.get("tensor_state_sha256") != best_path.stem.removeprefix("best-")
        ):
            raise SupervisorError("bounded MLP best-checkpoint authentication differs")
        authenticated_epochs.add(epoch)
    if authenticated_epochs != set(expected_best):
        raise SupervisorError("bounded MLP best-checkpoint path contract differs")
    fixed_files.update(str(path.relative_to(base)) for path in best_paths)
    expected.update((path, "directory") for path in fixed_directories)
    expected.update((path, "file") for path in fixed_files)
    return [
        {"relative_path": path, "entry_type": entry_type} for path, entry_type in sorted(expected)
    ]


def _run_cache_manifest_relative_path(identity: Mapping[str, object]) -> str:
    return f"run/cache/features/identity-{cache_identity_sha256(identity)}.json"


def _resource_observation(base: Path, static_required_bytes: int) -> dict[str, object]:
    available = _available_ram_bytes()
    rss = _current_rss_bytes()
    disk = int(shutil.disk_usage(base).free)
    if (
        static_required_bytes > int(0.8 * available)
        or static_required_bytes > int(0.8 * RSS_LIMIT)
        or rss + static_required_bytes > RSS_LIMIT
        or disk < DISK_FLOOR + 256 * 1024**2
    ):
        raise SupervisorError("bounded bootstrap live RAM/RSS/disk gate refused")
    inequalities = {
        "static_within_available_ram": static_required_bytes <= int(0.8 * available),
        "static_within_rss_fraction": static_required_bytes <= int(0.8 * RSS_LIMIT),
        "rss_plus_static_within_limit": rss + static_required_bytes <= RSS_LIMIT,
        "disk_within_floor_and_allowance": disk >= DISK_FLOOR + 256 * 1024**2,
    }
    return {
        "observed_at": _iso(_now()),
        "available_ram_bytes": available,
        "current_rss_bytes": rss,
        "free_disk_bytes": disk,
        "static_required_with_margin_bytes": static_required_bytes,
        "rss_limit_bytes": RSS_LIMIT,
        "disk_floor_bytes": DISK_FLOOR,
        "reserved_artifact_allowance_bytes": 256 * 1024**2,
        "inequalities": inequalities,
        "approved": all(inequalities.values()),
    }


def _validate_stored_resource_observation(artifact: Mapping[str, object]) -> None:
    required = int(artifact["static_required_with_margin_bytes"])
    available = int(artifact["available_ram_bytes"])
    rss = int(artifact["current_rss_bytes"])
    disk = int(artifact["free_disk_bytes"])
    expected = {
        "static_within_available_ram": required <= int(0.8 * available),
        "static_within_rss_fraction": required <= int(0.8 * RSS_LIMIT),
        "rss_plus_static_within_limit": rss + required <= RSS_LIMIT,
        "disk_within_floor_and_allowance": disk >= DISK_FLOOR + 256 * 1024**2,
    }
    if artifact.get("inequalities") != expected or artifact.get("approved") is not True:
        raise SupervisorError("stored bounded resource inequalities differ")


def _bound_resource_observation(
    base: Path, static_required_bytes: int, authorization: Mapping[str, object]
) -> dict[str, object]:
    device = detect_device_identity("cpu")
    return {
        **_resource_observation(base, static_required_bytes),
        "authorization_nonce": authorization["nonce"],
        "device_identity": device,
        "device_identity_sha256": canonical_sha256(device),
        "run_id": RUN_ID,
        "purpose": PURPOSE,
        "runner_identity": authorization["runner_identity"],
        **_false_boundaries(),
    }


def candidates() -> list[dict[str, object]]:
    return [
        {"id": "ridge|smoke|alpha-1", "family": "ridge", "params": {"alpha": 1.0}},
        {
            "id": "xgboost|smoke|tiny",
            "family": "xgboost",
            "params": {
                "max_depth": 2,
                "learning_rate": 0.1,
                "n_estimators": 4,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "min_child_weight": 1.0,
            },
            "early_stopping_rounds": 2,
        },
        {
            "id": "mlp|smoke|tiny",
            "family": "mlp",
            "params": {
                "hidden_dims": [16],
                "dropout": 0.0,
                "learning_rate": 0.001,
                "weight_decay": 0.00001,
            },
            "training": {
                "batch_size": 24,
                "max_epochs": 2,
                "patience": 2,
                "min_delta": 0.0,
            },
            "monitoring": {
                "validation_increase_mark_after": 1,
                "gradient_norm_hard_max": 1.0e6,
                "gradient_norm_hard_min": 0.0,
                "gradient_min_consecutive_batches": 100,
                "target_degradation_warning": 10.0,
            },
        },
    ]


def planned_trials() -> list[PlannedTrial]:
    rows = []
    for index, candidate in enumerate(candidates(), start=1):
        rows.append(
            PlannedTrial(
                identity=TrialIdentity(
                    run_id=RUN_ID,
                    wave="PROFILE_ONLY",
                    strategy="identity",
                    candidate_id=str(candidate["id"]),
                    family=str(candidate["family"]),
                    fold=0,
                    seed=SEED,
                    trial_index=index,
                    trial_total=3,
                ),
                expected_shape=(TUNE_ROWS, len(TARGET_COLUMNS)),
                target_order=tuple(TARGET_COLUMNS),
                prediction_order_sha256=canonical_sha256(list(range(TRAIN_ROWS, ROWS))),
                candidate_sha256=canonical_sha256(candidate),
            )
        )
    return rows


def sidecar_policy(family: str) -> SidecarPolicy:
    soft, hard = (15.0, 25.0) if family == "ridge" else (30.0, 45.0)
    return SidecarPolicy(
        heartbeat_seconds=2.0,
        heartbeat_stale_after_seconds=4.0,
        model_progress_stale_after_seconds=4.0,
        soft_deadline_seconds=soft,
        hard_deadline_seconds=hard,
        rss_max_bytes=RSS_LIMIT,
        disk_free_min_bytes=DISK_FLOOR,
        gpu_allocated_max_bytes=1,
        gpu_reserved_max_bytes=1,
        gpu_temperature_max_c=1.0,
        require_rss_sensor=True,
        require_disk_sensor=True,
        require_gpu_sensors=False,
        gpu_device_uuid=None,
        poll_seconds=0.05,
    )


def write_synthetic_fixture(path: str | Path) -> Path:
    destination = Path(path)
    if destination.exists():
        raise SupervisorError("synthetic fixture path must be fresh")
    rng = np.random.default_rng(SEED)
    features = rng.standard_normal((ROWS, FEATURES)).astype(np.float64)
    features[np.abs(features) < 0.70] = 0.0
    coefficients = rng.standard_normal((FEATURES, len(TARGET_COLUMNS)))
    noise = rng.standard_normal((ROWS, len(TARGET_COLUMNS)))
    targets = features @ coefficients + 0.05 * noise
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(targets)):
        raise SupervisorError("synthetic fixture contains nonfinite values")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        x_train=features[:TRAIN_ROWS],
        y_train=targets[:TRAIN_ROWS],
        x_tune=features[TRAIN_ROWS:],
        y_tune=targets[TRAIN_ROWS:],
    )
    return destination


def freeze_bounded_bootstrap_profile(
    path: str | Path,
    *,
    input_path: str | Path,
    cache_choice_path: str | Path,
    memory_worksheet: Mapping[str, object],
    config_sha256: str,
    code_sha256: str,
    dependency_sha256: str,
    runner_identity: str,
    repository_provenance: Mapping[str, object],
    prepare_attempt_id: str,
    prepare_claim_file_sha256: str,
    prepare_claim_semantic_sha256: str,
) -> dict[str, object]:
    now = _now()
    available = _available_ram_bytes()
    disk = int(shutil.disk_usage(Path(path).parent).free)
    required = int(memory_worksheet["capacity"]["cpu_required_with_margin_bytes"])
    approved = required <= int(0.8 * available) and required <= int(0.8 * RSS_LIMIT)
    trials = planned_trials()
    preflight = Path(path).parent.resolve()
    prepare_tree_manifest = []
    for candidate in sorted(preflight.rglob("*")):
        mode = candidate.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise SupervisorError("bounded prepare tree contains a nonregular entry")
        prepare_tree_manifest.append(
            {
                "relative_path": str(candidate.relative_to(preflight)),
                "entry_type": "file" if stat.S_ISREG(mode) else "directory",
                "bytes": candidate.lstat().st_size if stat.S_ISREG(mode) else 0,
                "file_sha256": file_sha256(candidate) if stat.S_ISREG(mode) else None,
            }
        )
    payload: dict[str, object] = {
        "schema_version": PROFILE_SCHEMA,
        "purpose": PURPOSE,
        "data_scope": "synthetic_only",
        "device_type": "cpu",
        "run_id": RUN_ID,
        "runner_identity": runner_identity,
        "seed": SEED,
        "shape": {
            "rows": ROWS,
            "features": FEATURES,
            "train_rows": TRAIN_ROWS,
            "tune_rows": TUNE_ROWS,
            "targets": 12,
        },
        "input_file_sha256": artifact_file_sha256(input_path),
        "prediction_order_sha256": canonical_sha256(list(range(TRAIN_ROWS, ROWS))),
        "candidate_sha256": [row.candidate_sha256 for row in trials],
        "plan_sha256": canonical_sha256([row.payload() for row in trials]),
        "config_sha256": config_sha256,
        "code_sha256": code_sha256,
        "dependency_sha256": dependency_sha256,
        "repository_provenance": dict(repository_provenance),
        "repository_provenance_sha256": canonical_sha256(dict(repository_provenance)),
        "prepare_attempt_id": prepare_attempt_id,
        "prepare_claim_file_sha256": prepare_claim_file_sha256,
        "prepare_claim_semantic_sha256": prepare_claim_semantic_sha256,
        "cache_choice_file_sha256": artifact_file_sha256(cache_choice_path),
        "prepare_tree_manifest": prepare_tree_manifest,
        "prepare_tree_manifest_sha256": canonical_sha256(prepare_tree_manifest),
        "device_identity": detect_device_identity("cpu"),
        "device_identity_sha256": canonical_sha256(detect_device_identity("cpu")),
        "memory_worksheet": dict(memory_worksheet),
        "memory_worksheet_sha256": canonical_sha256(dict(memory_worksheet)),
        "observed_at": _iso(now),
        "observation_max_age_seconds": BOOTSTRAP_MAX_AGE_SECONDS,
        "available_ram_bytes": available,
        "free_disk_bytes": disk,
        "peak_rss_measurement_status": "absent_pre_measurement",
        "runtime_measurement_status": "absent_pre_measurement",
        "policies": {
            family: asdict(sidecar_policy(family)) for family in ("ridge", "xgboost", "mlp")
        },
        "bounded_profile_approved": approved,
        **_false_boundaries(),
    }
    if not approved or disk < DISK_FLOOR:
        raise SupervisorError("bounded bootstrap static RAM/disk gate refused")
    return _write_immutable_json(Path(path), payload)


def freeze_bounded_bootstrap_review(
    path: str | Path,
    *,
    profile_path: str | Path,
    reviewer_identity: str,
    review_tool_identity: str,
    review_tool_code_sha256: str,
    runner_identity: str,
) -> dict[str, object]:
    if (
        not reviewer_identity
        or not review_tool_identity
        or len(review_tool_code_sha256) != 64
        or any(character not in "0123456789abcdef" for character in review_tool_code_sha256)
        or reviewer_identity == runner_identity
    ):
        raise SupervisorError("bootstrap reviewer must be independent from runner")
    base = Path(profile_path).resolve().parent.parent
    _, prepare_complete = _validate_prepare_state(base)
    profile = _validate_semantic_artifact(profile_path, PROFILE_SCHEMA)
    payload = {
        "schema_version": REVIEW_SCHEMA,
        "decision": "APPROVE_BOUNDED_PROFILE_ONLY",
        "reviewed_at_utc": _iso(_now()),
        "reviewer_identity": reviewer_identity,
        "review_tool_identity": review_tool_identity,
        "review_tool_code_sha256": review_tool_code_sha256,
        "runner_identity": runner_identity,
        "profile_path": str(Path(profile_path).resolve()),
        "profile_file_sha256": file_sha256(profile_path),
        "profile_semantic_sha256": profile["semantic_sha256"],
        "prepare_complete_path": str((base / "PREPARE_COMPLETE.json").resolve()),
        "prepare_complete_file_sha256": file_sha256(base / "PREPARE_COMPLETE.json"),
        "prepare_complete_semantic_sha256": prepare_complete["semantic_sha256"],
        "purpose": PURPOSE,
        "device_type": "cpu",
        "data_scope": "synthetic_only",
        "shape": profile["shape"],
        "candidate_sha256": profile["candidate_sha256"],
        "policies": profile["policies"],
        **_false_boundaries(),
    }
    return _write_immutable_json(Path(path), payload)


def freeze_bounded_bootstrap_authorization(
    path: str | Path,
    *,
    profile_path: str | Path,
    review_path: str | Path,
    lifetime_seconds: int = AUTHORIZATION_MAX_LIFETIME_SECONDS,
) -> dict[str, object]:
    if not 1 <= lifetime_seconds <= AUTHORIZATION_MAX_LIFETIME_SECONDS:
        raise SupervisorError("bootstrap authorization lifetime exceeds cap")
    base = Path(profile_path).resolve().parent.parent
    _, prepare_complete = _validate_prepare_state(base)
    profile = _validate_semantic_artifact(profile_path, PROFILE_SCHEMA)
    review = _validate_semantic_artifact(review_path, REVIEW_SCHEMA)
    if (
        review.get("decision") != "APPROVE_BOUNDED_PROFILE_ONLY"
        or review.get("profile_file_sha256") != file_sha256(profile_path)
        or review.get("profile_semantic_sha256") != profile["semantic_sha256"]
        or review.get("profile_path") != str(Path(profile_path).resolve())
        or review.get("runner_identity") != profile.get("runner_identity")
        or review.get("reviewer_identity") == profile.get("runner_identity")
        or review.get("prepare_complete_file_sha256") != file_sha256(base / "PREPARE_COMPLETE.json")
        or review.get("prepare_complete_semantic_sha256") != prepare_complete["semantic_sha256"]
        or review.get("prepare_complete_path") != str((base / "PREPARE_COMPLETE.json").resolve())
    ):
        raise SupervisorError("bootstrap review does not bind exact proposal")
    issued = _now()
    payload = {
        "schema_version": AUTH_SCHEMA,
        "purpose": PURPOSE,
        "decision": "AUTHORIZE_ONE_BOUNDED_PROFILE",
        "nonce": secrets.token_hex(32),
        "issued_at_utc": _iso(issued),
        "expires_at_utc": _iso(issued + timedelta(seconds=lifetime_seconds)),
        "profile_path": str(Path(profile_path).resolve()),
        "profile_file_sha256": file_sha256(profile_path),
        "profile_semantic_sha256": profile["semantic_sha256"],
        "prepare_complete_path": str((base / "PREPARE_COMPLETE.json").resolve()),
        "prepare_complete_file_sha256": file_sha256(base / "PREPARE_COMPLETE.json"),
        "prepare_complete_semantic_sha256": prepare_complete["semantic_sha256"],
        "review_path": str(Path(review_path).resolve()),
        "review_file_sha256": file_sha256(review_path),
        "review_semantic_sha256": review["semantic_sha256"],
        "reviewer_identity": review["reviewer_identity"],
        "review_tool_identity": review["review_tool_identity"],
        "review_tool_code_sha256": review["review_tool_code_sha256"],
        "runner_identity": review["runner_identity"],
        **_false_boundaries(),
    }
    return _write_immutable_json(Path(path), payload)


def _validate_claim_values(
    binding: BoundedProfileRunBinding, authorization: Mapping[str, object]
) -> dict[str, object]:
    path = EXACT_BASE / "attempt/nonce-claim.json"
    claim = _validate_semantic_artifact(path, "qm9-v31-bounded-attempt-claim-v5")
    claimed = _parse_time(claim.get("claimed_at_utc"), "claim time")
    issued = _parse_time(authorization.get("issued_at_utc"), "issued time")
    if (
        claim.get("nonce") != authorization.get("nonce")
        or claim.get("authorization_file_sha256") != binding.bootstrap_authorization_file_sha256
        or claim.get("authorization_semantic_sha256")
        != binding.bootstrap_authorization_semantic_sha256
        or claim.get("runner_identity") != authorization.get("runner_identity")
        or claim.get("run_id") != RUN_ID
        or claim.get("purpose") != PURPOSE
        or any(claim.get(field) is not False for field in _false_boundaries())
        or claimed < issued
        or claimed > _now() + timedelta(seconds=30)
    ):
        raise SupervisorError("bounded bootstrap claim values differ")
    return claim


def _validate_initial_observation_values(
    binding: BoundedProfileRunBinding,
    authorization: Mapping[str, object],
    claim: Mapping[str, object],
) -> dict[str, object]:
    path = EXACT_BASE / "attempt/resource-observation.json"
    observation = _validate_semantic_artifact(path, "qm9-v31-bounded-resource-observation-v5")
    observed = _parse_time(observation.get("observed_at"), "resource observation")
    claimed = _parse_time(claim.get("claimed_at_utc"), "claim time")
    profile = _validate_semantic_artifact(
        binding.bootstrap_profile_path,
        PROFILE_SCHEMA,
        expected_file_sha256=binding.bootstrap_profile_file_sha256,
    )
    reviewed_required = int(
        profile["memory_worksheet"]["capacity"]["cpu_required_with_margin_bytes"]
    )
    if (
        observation.get("authorization_nonce") != authorization.get("nonce")
        or observation.get("device_identity") != dict(binding.device_identity)
        or observation.get("device_identity_sha256") != binding.device_identity_sha256
        or observation.get("runner_identity") != authorization.get("runner_identity")
        or observation.get("run_id") != RUN_ID
        or observation.get("purpose") != PURPOSE
        or observation.get("rss_limit_bytes") != RSS_LIMIT
        or observation.get("disk_floor_bytes") != DISK_FLOOR
        or observation.get("reserved_artifact_allowance_bytes") != 256 * 1024**2
        or observation.get("static_required_with_margin_bytes") != reviewed_required
        or any(observation.get(field) is not False for field in _false_boundaries())
        or observed < claimed
        or observed > _now() + timedelta(seconds=30)
    ):
        raise SupervisorError("bounded initial resource observation values differ")
    _validate_stored_resource_observation(observation)
    return observation


def _validate_receipt_values(
    binding: BoundedProfileRunBinding,
    authorization: Mapping[str, object],
    claim: Mapping[str, object],
    observation: Mapping[str, object],
    *,
    expected_run_start_sha256: str,
) -> dict[str, object]:
    receipt_path = EXACT_BASE / "attempt/authorization-consumed.json"
    run_start_path = EXACT_BASE / "run/run_start.json"
    receipt = _validate_semantic_artifact(receipt_path, "qm9-v31-bounded-authorization-consumed-v5")
    run_start = _read_json(run_start_path)
    run_start_unsigned = dict(run_start)
    run_start_semantic = run_start_unsigned.pop("semantic_sha256", None)
    expected_run_start_keys = {
        "schema_version",
        "binding",
        "planned_trials",
        "planned_trial_keys",
        "bootstrap_authorization",
        "bootstrap_lifecycle",
        "run_purpose",
        "scientific_result",
        "publication_ready",
        "ranking_allowed",
        "promotion_allowed",
        "ensemble_allowed",
        "outer_validation_targets_read",
        "test_targets_read",
        "semantic_sha256",
    }
    consumed = _parse_time(receipt.get("consumed_at_utc"), "consumption time")
    observed = _parse_time(observation.get("observed_at"), "resource observation")
    if (
        receipt.get("authorization_semantic_sha256")
        != binding.bootstrap_authorization_semantic_sha256
        or receipt.get("authorization_file_sha256") != binding.bootstrap_authorization_file_sha256
        or receipt.get("nonce") != authorization.get("nonce")
        or receipt.get("run_start_sha256") != expected_run_start_sha256
        or receipt.get("run_start_file_sha256") != file_sha256(run_start_path)
        or file_sha256(run_start_path) != expected_run_start_sha256
        or receipt.get("run_start_semantic_sha256") != run_start.get("semantic_sha256")
        or set(run_start) != expected_run_start_keys
        or run_start.get("schema_version") != RUN_START_SCHEMA
        or run_start_semantic != canonical_sha256(run_start_unsigned)
        or receipt.get("claim_file_sha256") != file_sha256(EXACT_BASE / "attempt/nonce-claim.json")
        or receipt.get("observation_file_sha256")
        != file_sha256(EXACT_BASE / "attempt/resource-observation.json")
        or receipt.get("runner_identity") != authorization.get("runner_identity")
        or receipt.get("run_id") != RUN_ID
        or receipt.get("purpose") != PURPOSE
        or any(receipt.get(field) is not False for field in _false_boundaries())
        or consumed < observed
        or consumed > _now() + timedelta(seconds=30)
    ):
        raise SupervisorError("bounded bootstrap consumption receipt values differ")
    return receipt


def load_bounded_bootstrap_authorization(
    binding: BoundedProfileRunBinding,
    *,
    planned_trials: Sequence[PlannedTrial],
    run_root: str | Path,
    expected_run_start_sha256: str | None = None,
) -> dict[str, object]:
    binding.validate()
    prepare_claim, prepare_complete = _validate_prepare_state(
        EXACT_BASE,
        allow_execute_transition=True,
        allow_post_run=(EXACT_BASE / "profile-summary.json").is_file(),
    )
    root = Path(run_root).resolve()
    if root != (EXACT_BASE / "run").resolve():
        raise SupervisorError("bounded profile output must be the exact frozen /tmp run root")
    profile = _validate_semantic_artifact(
        binding.bootstrap_profile_path,
        PROFILE_SCHEMA,
        expected_file_sha256=binding.bootstrap_profile_file_sha256,
    )
    review = _validate_semantic_artifact(
        binding.bootstrap_review_path,
        REVIEW_SCHEMA,
        expected_file_sha256=binding.bootstrap_review_file_sha256,
    )
    authorization = _validate_semantic_artifact(
        binding.bootstrap_authorization_path,
        AUTH_SCHEMA,
        expected_file_sha256=binding.bootstrap_authorization_file_sha256,
    )
    preflight = EXACT_BASE / "preflight"
    manifest = profile.get("prepare_tree_manifest")
    if not isinstance(manifest, list) or profile.get(
        "prepare_tree_manifest_sha256"
    ) != canonical_sha256(manifest):
        raise SupervisorError("bounded prepare-tree manifest differs")
    expected_entries: dict[str, str] = {}
    for row in manifest:
        if not isinstance(row, Mapping) or set(row) != {
            "relative_path",
            "entry_type",
            "bytes",
            "file_sha256",
        }:
            raise SupervisorError("bounded prepare-tree row schema differs")
        relative = Path(str(row["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise SupervisorError("bounded prepare-tree path escapes preflight")
        candidate = preflight / relative
        mode = candidate.lstat().st_mode
        observed_type = (
            "file" if stat.S_ISREG(mode) else "directory" if stat.S_ISDIR(mode) else "invalid"
        )
        if observed_type != row["entry_type"] or observed_type == "invalid":
            raise SupervisorError("bounded prepare-tree physical bytes differ")
        if observed_type == "file" and (
            candidate.lstat().st_size != row["bytes"]
            or file_sha256(candidate) != row["file_sha256"]
        ):
            raise SupervisorError("bounded prepare-tree physical bytes differ")
        expected_entries[str(relative)] = observed_type
    allowed_later = {
        Path(binding.bootstrap_profile_path).resolve(),
        Path(binding.bootstrap_review_path).resolve(),
        Path(binding.bootstrap_authorization_path).resolve(),
    }
    actual_entries = {candidate for candidate in preflight.rglob("*")}
    expected_paths = {preflight / relative for relative in expected_entries}
    if actual_entries != expected_paths | allowed_later:
        raise SupervisorError("bounded preflight contains unexpected or missing files")
    if (
        profile["semantic_sha256"] != binding.bootstrap_profile_semantic_sha256
        or review["semantic_sha256"] != binding.bootstrap_review_semantic_sha256
        or authorization["semantic_sha256"] != binding.bootstrap_authorization_semantic_sha256
        or authorization.get("profile_file_sha256") != file_sha256(binding.bootstrap_profile_path)
        or authorization.get("review_file_sha256") != file_sha256(binding.bootstrap_review_path)
        or authorization.get("reviewer_identity") == authorization.get("runner_identity")
        or authorization.get("profile_path") != str(Path(binding.bootstrap_profile_path).resolve())
        or authorization.get("review_path") != str(Path(binding.bootstrap_review_path).resolve())
        or authorization.get("review_semantic_sha256") != review["semantic_sha256"]
        or authorization.get("reviewer_identity") != review.get("reviewer_identity")
        or authorization.get("review_tool_identity") != review.get("review_tool_identity")
        or authorization.get("review_tool_code_sha256") != review.get("review_tool_code_sha256")
        or review.get("prepare_complete_file_sha256") != binding.prepare_complete_file_sha256
        or review.get("prepare_complete_semantic_sha256")
        != binding.prepare_complete_semantic_sha256
        or review.get("prepare_complete_path") != binding.prepare_complete_path
        or authorization.get("prepare_complete_file_sha256") != binding.prepare_complete_file_sha256
        or authorization.get("prepare_complete_semantic_sha256")
        != binding.prepare_complete_semantic_sha256
        or authorization.get("prepare_complete_path") != binding.prepare_complete_path
        or review.get("profile_path") != str(Path(binding.bootstrap_profile_path).resolve())
        or profile.get("config_sha256") != binding.config_sha256
        or profile.get("input_file_sha256") != binding.input_sha256
        or profile.get("cache_choice_file_sha256") != binding.profile_sha256
        or profile.get("code_sha256") != binding.code_sha256
        or profile.get("dependency_sha256") != binding.dependency_sha256
        or profile.get("repository_provenance") != dict(binding.repository_provenance)
        or profile.get("repository_provenance_sha256") != binding.repository_provenance_sha256
        or profile.get("prepare_attempt_id") != binding.prepare_attempt_id
        or prepare_claim.get("prepare_attempt_id") != binding.prepare_attempt_id
        or prepare_complete.get("prepare_attempt_id") != binding.prepare_attempt_id
        or profile.get("device_identity") != dict(binding.device_identity)
        or detect_device_identity("cpu") != dict(binding.device_identity)
        or artifact_file_sha256(EXACT_BASE / "preflight/public-synthetic.npz")
        != binding.input_sha256
        or file_sha256(EXACT_BASE / "preflight/config.json") != binding.config_sha256
        or artifact_file_sha256(EXACT_BASE / "preflight/cache-choice.json")
        != binding.profile_sha256
        or _repository_preflight() != dict(binding.repository_provenance)
    ):
        raise SupervisorError("bounded bootstrap authorization chain differs")
    now = _now()
    issued = _parse_time(authorization["issued_at_utc"], "issued time")
    reviewed = _parse_time(review["reviewed_at_utc"], "review time")
    prepared = _parse_time(profile["observed_at"], "prepare time")
    expiry = _parse_time(authorization["expires_at_utc"], "expiry")
    if (
        issued > now
        or prepared > reviewed
        or reviewed > issued
        or expiry <= issued
        or expiry - issued > timedelta(seconds=AUTHORIZATION_MAX_LIFETIME_SECONDS)
        or expiry - now < timedelta(seconds=MINIMUM_EXECUTION_VALIDITY_SECONDS)
    ):
        raise SupervisorError("bounded bootstrap authorization validity is too short")
    if now - _parse_time(profile["observed_at"], "observation time") > timedelta(hours=24):
        raise SupervisorError("bounded bootstrap authorization/observation is stale")
    required = int(profile["memory_worksheet"]["capacity"]["cpu_required_with_margin_bytes"])
    _resource_observation(EXACT_BASE.parent, required)
    expected_plan = canonical_sha256([row.payload() for row in planned_trials])
    exact_policies = {
        family: asdict(sidecar_policy(family)) for family in ("ridge", "xgboost", "mlp")
    }
    if (
        profile.get("purpose") != PURPOSE
        or profile.get("run_id") != RUN_ID
        or profile.get("seed") != SEED
        or profile.get("data_scope") != "synthetic_only"
        or profile.get("device_type") != "cpu"
        or profile.get("prediction_order_sha256") != canonical_sha256(list(range(TRAIN_ROWS, ROWS)))
        or profile.get("runtime_measurement_status") != "absent_pre_measurement"
        or profile.get("peak_rss_measurement_status") != "absent_pre_measurement"
        or profile.get("bounded_profile_approved") is not True
        or profile.get("shape")
        != {
            "rows": ROWS,
            "features": FEATURES,
            "train_rows": TRAIN_ROWS,
            "tune_rows": TUNE_ROWS,
            "targets": 12,
        }
        or profile.get("policies") != exact_policies
        or review.get("purpose") != PURPOSE
        or review.get("data_scope") != "synthetic_only"
        or review.get("device_type") != "cpu"
        or review.get("shape") != profile.get("shape")
        or review.get("candidate_sha256") != profile.get("candidate_sha256")
        or review.get("runner_identity") != profile.get("runner_identity")
        or review.get("policies") != exact_policies
        or authorization.get("purpose") != PURPOSE
        or authorization.get("decision") != "AUTHORIZE_ONE_BOUNDED_PROFILE"
        or not isinstance(authorization.get("nonce"), str)
        or len(str(authorization.get("nonce"))) != 64
        or any(character not in "0123456789abcdef" for character in str(authorization.get("nonce")))
    ):
        raise SupervisorError("bounded bootstrap exact scope differs")
    if expected_plan != binding.plan_sha256 or profile.get("plan_sha256") != expected_plan:
        raise SupervisorError("bounded bootstrap planned trials differ")
    if profile.get("candidate_sha256") != [row.candidate_sha256 for row in planned_trials]:
        raise SupervisorError("bounded bootstrap candidate hashes differ")
    if [row.identity.family for row in planned_trials] != ["ridge", "xgboost", "mlp"]:
        raise SupervisorError("bounded bootstrap family order differs")
    for field in _false_boundaries():
        if (
            profile.get(field) is not False
            or review.get(field) is not False
            or authorization.get(field) is not False
        ):
            raise SupervisorError("bounded bootstrap exclusion flag differs")
    attempt = EXACT_BASE / "attempt"
    claim_path = attempt / "nonce-claim.json"
    observation_path = attempt / "resource-observation.json"
    receipt = attempt / "authorization-consumed.json"
    if not claim_path.is_file() or not observation_path.is_file():
        raise SupervisorError("bounded bootstrap claim/resource observation is missing")
    claim = _validate_claim_values(binding, authorization)
    observation = _validate_initial_observation_values(binding, authorization, claim)
    if expected_run_start_sha256 is None and _now() - _parse_time(
        observation.get("observed_at"), "resource observation"
    ) > timedelta(seconds=30):
        raise SupervisorError("bounded initial resource observation is stale")
    if receipt.exists():
        if expected_run_start_sha256 is None:
            raise SupervisorError("bounded bootstrap authorization was already consumed")
        _validate_receipt_values(
            binding,
            authorization,
            claim,
            observation,
            expected_run_start_sha256=expected_run_start_sha256,
        )
    elif expected_run_start_sha256 is not None:
        raise SupervisorError("bounded bootstrap consumption receipt is missing")
    return authorization


def validate_bounded_trial_observation(
    binding: BoundedProfileRunBinding,
    *,
    path: str | Path,
    trial: TrialIdentity,
    run_start_sha256: str,
    previous_ledger_entry_sha256: str | None,
    require_fresh: bool = True,
) -> dict[str, object]:
    artifact = _validate_semantic_artifact(path, "qm9-v31-bounded-trial-resource-observation-v5")
    authorization = _validate_semantic_artifact(
        binding.bootstrap_authorization_path,
        AUTH_SCHEMA,
        expected_file_sha256=binding.bootstrap_authorization_file_sha256,
    )
    profile = _validate_semantic_artifact(
        binding.bootstrap_profile_path,
        PROFILE_SCHEMA,
        expected_file_sha256=binding.bootstrap_profile_file_sha256,
    )
    reviewed_required = int(
        profile["memory_worksheet"]["capacity"]["cpu_required_with_margin_bytes"]
    )
    expected_path = (
        EXACT_BASE / f"attempt/resource-observation-{trial.trial_index}-{trial.family}.json"
    ).resolve()
    if (
        (
            require_fresh
            and _now() - _parse_time(artifact.get("observed_at"), "trial observation")
            > timedelta(seconds=5)
        )
        or artifact.get("trial") != asdict(trial)
        or artifact.get("trial_key") != canonical_sha256(asdict(trial))
        or artifact.get("policy_sha256") != canonical_sha256(asdict(sidecar_policy(trial.family)))
        or artifact.get("run_start_sha256") != run_start_sha256
        or artifact.get("previous_ledger_entry_sha256") != previous_ledger_entry_sha256
        or artifact.get("device_identity") != dict(binding.device_identity)
        or artifact.get("device_identity_sha256") != binding.device_identity_sha256
        or artifact.get("authorization_nonce") != authorization["nonce"]
        or artifact.get("runner_identity") != authorization["runner_identity"]
        or Path(path).resolve() != expected_path
        or artifact.get("receipt_file_sha256")
        != file_sha256(EXACT_BASE / "attempt/authorization-consumed.json")
        or artifact.get("static_required_with_margin_bytes") != reviewed_required
        or artifact.get("run_id") != RUN_ID
        or artifact.get("purpose") != PURPOSE
        or any(artifact.get(field) is not False for field in _false_boundaries())
    ):
        raise SupervisorError("bounded per-trial resource observation differs")
    _validate_stored_resource_observation(artifact)
    return {
        "resource_observation_path": str(Path(path).resolve()),
        "resource_observation_file_sha256": file_sha256(path),
        "resource_observation_semantic_sha256": artifact["semantic_sha256"],
    }


def consume_bounded_bootstrap_authorization(
    binding: BoundedProfileRunBinding, *, run_start_sha256: str
) -> Path:
    authorization = _validate_semantic_artifact(
        binding.bootstrap_authorization_path,
        AUTH_SCHEMA,
        expected_file_sha256=binding.bootstrap_authorization_file_sha256,
    )
    claim_path = EXACT_BASE / "attempt/nonce-claim.json"
    observation_path = EXACT_BASE / "attempt/resource-observation.json"
    receipt = EXACT_BASE / "attempt/authorization-consumed.json"
    run_start = _read_json(EXACT_BASE / "run/run_start.json")
    _exclusive_artifact(
        receipt,
        "qm9-v31-bounded-authorization-consumed-v5",
        {
            "authorization_semantic_sha256": authorization["semantic_sha256"],
            "authorization_file_sha256": file_sha256(binding.bootstrap_authorization_path),
            "nonce": authorization["nonce"],
            "run_start_sha256": run_start_sha256,
            "run_start_file_sha256": run_start_sha256,
            "run_start_semantic_sha256": run_start["semantic_sha256"],
            "claim_file_sha256": file_sha256(claim_path),
            "observation_file_sha256": file_sha256(observation_path),
            "runner_identity": authorization["runner_identity"],
            "run_id": RUN_ID,
            "purpose": PURPOSE,
            "consumed_at_utc": _iso(_now()),
            **_false_boundaries(),
        },
    )
    return receipt


def bounded_lifecycle_start_evidence(
    binding: BoundedProfileRunBinding,
) -> dict[str, object]:
    claim_path = EXACT_BASE / "attempt/nonce-claim.json"
    observation_path = EXACT_BASE / "attempt/resource-observation.json"
    claim = _validate_semantic_artifact(claim_path, "qm9-v31-bounded-attempt-claim-v5")
    observation = _validate_semantic_artifact(
        observation_path, "qm9-v31-bounded-resource-observation-v5"
    )
    _validate_stored_resource_observation(observation)
    return {
        "claim_path": str(claim_path.resolve()),
        "claim_file_sha256": file_sha256(claim_path),
        "claim_semantic_sha256": claim["semantic_sha256"],
        "initial_observation_path": str(observation_path.resolve()),
        "initial_observation_file_sha256": file_sha256(observation_path),
        "initial_observation_semantic_sha256": observation["semantic_sha256"],
        "authorization_file_sha256": binding.bootstrap_authorization_file_sha256,
        "authorization_semantic_sha256": binding.bootstrap_authorization_semantic_sha256,
    }


def validate_bounded_terminal_chain(
    supervisor: RunSupervisor,
) -> None:
    supervisor._load_bounded_profile_authorization()
    run_start = _read_json(supervisor.start_path)
    if run_start.get("bootstrap_lifecycle") != bounded_lifecycle_start_evidence(supervisor.binding):
        raise SupervisorError("bounded terminal run-start lifecycle binding differs")
    parsed_ledger = supervisor.ledger.validate(supervisor.ledger.path)
    if parsed_ledger != supervisor.ledger.rows:
        raise SupervisorError("bounded terminal durable ledger differs")
    for specification in supervisor.planned_trials:
        rows = [
            row
            for row in supervisor.ledger.rows
            if row["trial_key"] == canonical_sha256(asdict(specification.identity))
            and row["event"] == "reserve"
        ]
        if len(rows) != 1:
            raise SupervisorError("bounded terminal reserve chain differs")
        evidence = rows[0]["evidence"]
        path = evidence.get("resource_observation_path")
        if not isinstance(path, str) or file_sha256(path) != evidence.get(
            "resource_observation_file_sha256"
        ):
            raise SupervisorError("bounded terminal observation bytes differ")
        artifact = _validate_semantic_artifact(
            path, "qm9-v31-bounded-trial-resource-observation-v5"
        )
        if artifact.get("semantic_sha256") != evidence.get("resource_observation_semantic_sha256"):
            raise SupervisorError("bounded terminal observation meaning differs")
        validate_bounded_trial_observation(
            supervisor.binding,
            path=path,
            trial=specification.identity,
            run_start_sha256=supervisor.run_start_sha256,
            previous_ledger_entry_sha256=rows[0]["prev_entry_sha256"],
            require_fresh=False,
        )


def build_binding(
    preflight: Path,
    *,
    expected_review_file_sha256: str,
    expected_authorization_file_sha256: str,
    expected_reviewer_identity: str,
    post_run_audit: bool = False,
    revalidate_repository: bool = True,
) -> BoundedProfileRunBinding:
    base = preflight.resolve().parent
    prepare_claim, prepare_complete = _validate_prepare_state(
        base,
        allow_execute_transition=post_run_audit,
        allow_post_run=post_run_audit,
        revalidate_repository=revalidate_repository,
    )
    profile_path = preflight / "bootstrap-profile.json"
    review_path = preflight / "independent-review.json"
    authorization_path = preflight / "bootstrap-authorization.json"
    if (
        file_sha256(review_path) != expected_review_file_sha256
        or file_sha256(authorization_path) != expected_authorization_file_sha256
    ):
        raise SupervisorError("out-of-band review/authorization hash differs")
    profile = _validate_semantic_artifact(profile_path, PROFILE_SCHEMA)
    review = _validate_semantic_artifact(review_path, REVIEW_SCHEMA)
    authorization = _validate_semantic_artifact(authorization_path, AUTH_SCHEMA)
    if (
        review.get("reviewer_identity") != expected_reviewer_identity
        or authorization.get("reviewer_identity") != expected_reviewer_identity
        or review.get("prepare_complete_file_sha256") != file_sha256(base / "PREPARE_COMPLETE.json")
        or review.get("prepare_complete_path") != str((base / "PREPARE_COMPLETE.json").resolve())
        or review.get("prepare_complete_semantic_sha256") != prepare_complete["semantic_sha256"]
        or authorization.get("prepare_complete_file_sha256")
        != file_sha256(base / "PREPARE_COMPLETE.json")
        or authorization.get("prepare_complete_path")
        != str((base / "PREPARE_COMPLETE.json").resolve())
        or authorization.get("prepare_complete_semantic_sha256")
        != prepare_complete["semantic_sha256"]
    ):
        raise SupervisorError("out-of-band reviewer identity differs")
    return BoundedProfileRunBinding(
        run_id=RUN_ID,
        run_purpose=PURPOSE,
        plan_sha256=str(profile["plan_sha256"]),
        config_sha256=str(profile["config_sha256"]),
        input_sha256=str(profile["input_file_sha256"]),
        profile_sha256=str(profile["cache_choice_file_sha256"]),
        code_sha256=str(profile["code_sha256"]),
        dependency_sha256=str(profile["dependency_sha256"]),
        repository_provenance=dict(profile["repository_provenance"]),
        repository_provenance_sha256=str(profile["repository_provenance_sha256"]),
        prepare_attempt_id=str(prepare_claim["prepare_attempt_id"]),
        prepare_claim_path=str((base / "prepare-claim.json").resolve()),
        prepare_claim_file_sha256=file_sha256(base / "prepare-claim.json"),
        prepare_claim_semantic_sha256=str(prepare_claim["semantic_sha256"]),
        prepare_complete_path=str((base / "PREPARE_COMPLETE.json").resolve()),
        prepare_complete_file_sha256=file_sha256(base / "PREPARE_COMPLETE.json"),
        prepare_complete_semantic_sha256=str(prepare_complete["semantic_sha256"]),
        bootstrap_profile_path=str(profile_path.resolve()),
        bootstrap_profile_file_sha256=file_sha256(profile_path),
        bootstrap_profile_semantic_sha256=str(profile["semantic_sha256"]),
        bootstrap_review_path=str(review_path.resolve()),
        bootstrap_review_file_sha256=file_sha256(review_path),
        bootstrap_review_semantic_sha256=str(review["semantic_sha256"]),
        bootstrap_authorization_path=str(authorization_path.resolve()),
        bootstrap_authorization_file_sha256=file_sha256(authorization_path),
        bootstrap_authorization_semantic_sha256=str(authorization["semantic_sha256"]),
        device_identity=dict(profile["device_identity"]),
        device_identity_sha256=str(profile["device_identity_sha256"]),
    )


def _finite_evidence(value: object) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return bool(np.isfinite(value))
    if isinstance(value, Mapping):
        return all(_finite_evidence(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_finite_evidence(item) for item in value)
    return False


def _is_lower_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _complete_state_domains(
    specifications: Sequence[PlannedTrial],
) -> tuple[dict[str, str], dict[str, str]]:
    """Return the intentionally distinct terminal and human-summary key domains."""

    terminal = {
        canonical_sha256(asdict(specification.identity)): "complete"
        for specification in specifications
    }
    summary = {specification.identity.family: "complete" for specification in specifications}
    return terminal, summary


def _validate_completed_worker_evidence(
    base: Path,
    specification: PlannedTrial,
    checkpoint: Mapping[str, object],
) -> dict[str, object]:
    family = specification.identity.family
    result_path = base / f"run/worker-results/{family}.json"
    profile_path = base / f"run/worker-profiles/{family}.json"
    result = _read_json(result_path)
    if (
        set(result) != {"status", "checkpoint_evidence", "metadata_sha256"}
        or result.get("status") != COMPLETION_STATUS
        or result.get("checkpoint_evidence") != dict(checkpoint)
    ):
        raise SupervisorError("bounded worker result/checkpoint chain differs")
    profile = load_real_backend_profile(
        profile_path, expected_metadata_sha256=str(result.get("metadata_sha256"))
    )
    metadata = profile.get("metadata")
    if (
        profile.get("trial") != asdict(specification.identity)
        or profile.get("family") != family
        or not isinstance(metadata, Mapping)
        or metadata.get("checkpoint_evidence") != dict(checkpoint)
        or not _is_lower_sha256(metadata.get("prediction_sha256"))
        or not isinstance(metadata.get("stage_seconds"), Mapping)
        or not _finite_evidence(metadata)
    ):
        raise SupervisorError("bounded worker profile/metadata chain differs")
    if family == "mlp":
        curve = metadata.get("curve")
        curves = metadata.get("curves")
        if (
            not isinstance(curve, Mapping)
            or not isinstance(curves, Mapping)
            or curve.get("status") not in {"normal", "warning"}
            or not isinstance(curve.get("anomalies"), list)
            or not isinstance(curve.get("train_standardized_mse"), list)
            or not curve.get("train_standardized_mse")
            or not isinstance(curve.get("inner_tune_mnmae"), list)
            or len(curve["inner_tune_mnmae"]) != len(curve["train_standardized_mse"])
        ):
            raise SupervisorError("bounded MLP loss/anomaly evidence differs")
    return profile


def _completed_resource_peaks(events: Sequence[Mapping[str, object]]) -> dict[str, int]:
    observed: list[Mapping[str, object]] = []
    for event in events:
        resources = event.get("resources")
        if resources is None:
            continue
        if (
            not isinstance(resources, Mapping)
            or resources.get("rss_sensor_available") is not True
            or resources.get("disk_sensor_available") is not True
            or resources.get("gpu_sensor_available") is not False
            or not isinstance(resources.get("rss_bytes"), int)
            or int(resources["rss_bytes"]) > RSS_LIMIT
            or not isinstance(resources.get("disk_free_bytes"), int)
            or int(resources["disk_free_bytes"]) < DISK_FLOOR
            or resources.get("gpu_allocated_bytes") is not None
            or resources.get("gpu_reserved_bytes") is not None
            or resources.get("gpu_temperature_c") is not None
        ):
            raise SupervisorError("bounded telemetry resource chain differs")
        observed.append(resources)
    if not observed:
        raise SupervisorError("bounded telemetry has no authenticated resource observations")
    return {
        "rss_bytes": max(int(row["rss_bytes"]) for row in observed),
        "minimum_disk_free_bytes": min(int(row["disk_free_bytes"]) for row in observed),
    }


def _validate_terminal_event_chain_evidence(
    events: Sequence[Mapping[str, object]], evidence: Mapping[str, object]
) -> None:
    preterminal_count = evidence.get("event_count")
    expected_sequence = evidence.get("expected_run_complete_sequence")
    if (
        not isinstance(preterminal_count, int)
        or preterminal_count < 1
        or len(events) != preterminal_count + 1
        or len(events) < 2
        or evidence.get("event_head_sha256") != events[-2].get("event_sha256")
        or expected_sequence != preterminal_count + 1
        or events[-1].get("sequence") != expected_sequence
        or events[-1].get("event") != "run_complete"
    ):
        raise SupervisorError("bounded terminal event-chain evidence differs")


def _strict_postrun_checkpoint_replay(
    root: Path,
    binding: BoundedProfileRunBinding,
    specifications: Sequence[PlannedTrial],
    completed_entries: Mapping[str, Mapping[str, object]],
    *,
    run_start_sha256: str,
    ledger_sha256: str,
) -> list[dict[str, object]]:
    store = CheckpointStore(
        root,
        plan_sha256=binding.plan_sha256,
        config_sha256=binding.config_sha256,
        input_sha256=binding.input_sha256,
        cache_sha256=binding.profile_sha256,
        run_start_sha256=run_start_sha256,
    )
    replayed = store.load_completed(
        binding.plan_sha256,
        replay=ReplayContract(
            planned_trials=tuple(asdict(row.identity) for row in specifications),
            completed_entries=completed_entries,
            ledger_sha256=ledger_sha256,
            terminal_status=COMPLETION_STATUS,
        ),
    )
    replay_by_key = {canonical_sha256(row["manifest"]["trial"]): row for row in replayed}
    for specification in specifications:
        key = canonical_sha256(asdict(specification.identity))
        latest = store.latest_safe(specification.identity)
        replayed_path = Path(str(replay_by_key[key]["path"]))
        if (
            not isinstance(latest, Mapping)
            or latest.get("kind") != "completed"
            or latest.get("path") != replayed_path.name
            or latest.get("file_sha256") != file_sha256(replayed_path)
            or latest.get("semantic_sha256") != replay_by_key[key]["manifest"]["semantic_sha256"]
            or latest.get("prediction_sha256")
            != replay_by_key[key]["manifest"]["prediction_sha256"]
        ):
            raise SupervisorError("bounded checkpoint latest pointer differs")
    return replayed


def assert_profile_complete_postrun(
    output: str | Path,
    *,
    expected_review_file_sha256: str,
    expected_authorization_file_sha256: str,
    expected_reviewer_identity: str,
    revalidate_repository: bool = False,
) -> dict[str, object]:
    """Read-only reauthentication of the immutable completed profile evidence."""

    base = _require_v5_base(output)
    binding = build_binding(
        base / "preflight",
        expected_review_file_sha256=expected_review_file_sha256,
        expected_authorization_file_sha256=expected_authorization_file_sha256,
        expected_reviewer_identity=expected_reviewer_identity,
        post_run_audit=True,
        revalidate_repository=revalidate_repository,
    )
    summary, proposal = _validate_post_run_outputs(base)
    postrun_path = base / "POSTRUN_MANIFEST.json"
    postrun = _validate_semantic_artifact(postrun_path, POSTRUN_MANIFEST_SCHEMA)
    _validate_lstat_tree(
        base,
        postrun.get("final_base_inventory"),
        allowed_later={postrun_path},
    )
    run_start_path = base / "run/run_start.json"
    run_start = _read_json(run_start_path)
    unsigned_start = dict(run_start)
    start_semantic = unsigned_start.pop("semantic_sha256", None)
    specifications = planned_trials()
    if (
        run_start.get("schema_version") != RUN_START_SCHEMA
        or start_semantic != canonical_sha256(unsigned_start)
        or run_start.get("binding") != asdict(binding)
        or run_start.get("planned_trials") != [row.payload() for row in specifications]
        or run_start.get("bootstrap_lifecycle") != bounded_lifecycle_start_evidence(binding)
        or run_start.get("run_purpose") != PURPOSE
        or any(
            run_start.get(field) is not False
            for field in (
                "ranking_allowed",
                "promotion_allowed",
                "ensemble_allowed",
                "scientific_result",
                "publication_ready",
                "outer_validation_targets_read",
                "test_targets_read",
            )
        )
    ):
        raise SupervisorError("bounded completed run-start chain differs")
    run_start_sha256 = file_sha256(run_start_path)
    authorization = _validate_semantic_artifact(
        binding.bootstrap_authorization_path,
        AUTH_SCHEMA,
        expected_file_sha256=binding.bootstrap_authorization_file_sha256,
    )
    lifecycle_claim = _validate_claim_values(binding, authorization)
    lifecycle_observation = _validate_initial_observation_values(
        binding, authorization, lifecycle_claim
    )
    _validate_receipt_values(
        binding,
        authorization,
        lifecycle_claim,
        lifecycle_observation,
        expected_run_start_sha256=run_start_sha256,
    )
    ledger_path = base / "run/ledger/trials.jsonl"
    ledger_rows = TrialLedger.validate(ledger_path)
    completed_entries: dict[str, dict[str, object]] = {}
    expected_terminal_states, expected_summary_states = _complete_state_domains(specifications)
    worker_profiles: dict[str, dict[str, object]] = {}
    for specification in specifications:
        key = canonical_sha256(asdict(specification.identity))
        rows = [row for row in ledger_rows if row.get("trial_key") == key]
        if [row.get("event") for row in rows] != ["reserve", "start", "complete"]:
            raise SupervisorError("bounded completed trial lifecycle differs")
        reserve = rows[0]
        validate_bounded_trial_observation(
            binding,
            path=str(reserve["evidence"]["resource_observation_path"]),
            trial=specification.identity,
            run_start_sha256=run_start_sha256,
            previous_ledger_entry_sha256=reserve.get("prev_entry_sha256"),
            require_fresh=False,
        )
        checkpoint = rows[2]["evidence"]["checkpoint"]
        completed_entries[key] = {
            "status": COMPLETION_STATUS,
            "complete_ledger_entry_sha256": rows[2]["entry_sha256"],
            "start_ledger_entry_sha256": rows[2]["evidence"]["start_ledger_entry_sha256"],
            "checkpoint": checkpoint,
        }
        worker_profiles[specification.identity.family] = _validate_completed_worker_evidence(
            base, specification, checkpoint
        )
    replayed = _strict_postrun_checkpoint_replay(
        base / "run/checkpoints",
        binding,
        specifications,
        completed_entries,
        run_start_sha256=run_start_sha256,
        ledger_sha256=file_sha256(ledger_path),
    )
    expected_final_path_types = _expected_final_path_types(
        base, specifications, replayed, list(worker_profiles.values())
    )
    if postrun.get("expected_final_path_types") != expected_final_path_types or postrun.get(
        "expected_final_path_types_sha256"
    ) != canonical_sha256(expected_final_path_types):
        raise SupervisorError("bounded final path/type contract binding differs")
    checkpoint_prediction_hashes = [str(row["manifest"]["prediction_sha256"]) for row in replayed]
    events = validate_event_stream(base / "run/telemetry/events.jsonl")
    expected_resource_peaks = _completed_resource_peaks(events)
    terminal = _read_json(base / "run/final/manifest.json")
    expected_terminal_keys = {
        "schema_version",
        "status",
        "run_id",
        "run_start_sha256",
        "ledger_sha256",
        "trial_states",
        "complete",
        "ranking_allowed",
        "evidence",
        "scientific_result",
        "publication_ready",
        "outer_validation_targets_read",
        "test_targets_read",
        "terminal_mode",
        "purpose",
        "promotion_allowed",
        "ensemble_allowed",
        "w1_feasibility_approved",
        "general_training_approved",
        "private_data_allowed",
    }
    replay_sha256 = canonical_sha256(checkpoint_prediction_hashes)
    expected_summary_profiles = [
        {
            "family": row.identity.family,
            "stage_seconds": worker_profiles[row.identity.family]["metadata"].get(
                "stage_seconds", {}
            ),
            "loss_curves": worker_profiles[row.identity.family]["metadata"].get(
                "curves", {"available": False}
            ),
            "anomaly_flags": worker_profiles[row.identity.family]["metadata"].get(
                "curve", {"status": "unavailable"}
            ),
        }
        for row in specifications
    ]
    terminal_evidence = terminal.get("evidence", {})
    if not isinstance(terminal_evidence, Mapping):
        raise SupervisorError("bounded terminal evidence is unavailable")
    _validate_terminal_event_chain_evidence(events, terminal_evidence)
    expected_result_hashes = {
        family: file_sha256(base / f"run/worker-results/{family}.json")
        for family in ("ridge", "xgboost", "mlp")
    }
    expected_profile_hashes = {
        family: file_sha256(base / f"run/worker-profiles/{family}.json")
        for family in ("ridge", "xgboost", "mlp")
    }
    expected_observation_hashes = {
        row.identity.family: file_sha256(
            base
            / f"attempt/resource-observation-{row.identity.trial_index}-{row.identity.family}.json"
        )
        for row in specifications
    }
    receipt_path = base / "attempt/authorization-consumed.json"
    receipt_artifact = _validate_semantic_artifact(
        receipt_path, "qm9-v31-bounded-authorization-consumed-v5"
    )
    if (
        set(terminal) != expected_terminal_keys
        or terminal.get("schema_version") != BOUNDED_TERMINAL_SCHEMA
        or terminal.get("status") != "complete"
        or terminal.get("complete") is not True
        or terminal.get("run_id") != RUN_ID
        or terminal.get("run_start_sha256") != run_start_sha256
        or terminal.get("ledger_sha256") != file_sha256(ledger_path)
        or terminal.get("trial_states") != expected_terminal_states
        or terminal.get("terminal_mode") != "profile-only"
        or terminal.get("purpose") != "engineering_smoke"
        or terminal.get("ranking_allowed") is not False
        or terminal.get("promotion_allowed") is not False
        or terminal.get("ensemble_allowed") is not False
        or any(terminal.get(field) is not False for field in _false_boundaries())
        or terminal_evidence.get("checkpoint_replay_count") != 3
        or terminal_evidence.get("checkpoint_replay_sha256") != replay_sha256
        or terminal_evidence.get("worker_result_file_sha256") != expected_result_hashes
        or terminal_evidence.get("worker_profile_file_sha256") != expected_profile_hashes
        or terminal_evidence.get("trial_resource_observation_file_sha256")
        != expected_observation_hashes
        or terminal_evidence.get("authorization_consumed_file_sha256") != file_sha256(receipt_path)
        or terminal_evidence.get("authorization_consumed_semantic_sha256")
        != receipt_artifact.get("semantic_sha256")
        or terminal_evidence.get("summary_file_sha256")
        != file_sha256(base / "profile-summary.json")
        or terminal_evidence.get("summary_semantic_sha256") != summary.get("semantic_sha256")
        or terminal_evidence.get("proposal_file_sha256")
        != file_sha256(base / "post-profile/proposal.json")
        or terminal_evidence.get("proposal_semantic_sha256") != proposal.get("semantic_sha256")
        or terminal_evidence.get("expected_final_path_types_sha256")
        != canonical_sha256(expected_final_path_types)
        or not events
        or events[-1].get("event") != "run_complete"
        or events[-1].get("progress") != terminal
        or summary.get("trial_states") != expected_summary_states
        or summary.get("profiles") != expected_summary_profiles
        or summary.get("resource_peaks") != expected_resource_peaks
        or summary.get("event_count") != len(events)
        or summary.get("ledger_sha256") != file_sha256(ledger_path)
        or summary.get("replay_count") != 3
        or summary.get("replay_sha256") != replay_sha256
        or postrun.get("run_start_path") != str(run_start_path.resolve())
        or postrun.get("run_start_file_sha256") != run_start_sha256
        or postrun.get("run_start_semantic_sha256") != run_start.get("semantic_sha256")
        or postrun.get("ledger_path") != str(ledger_path.resolve())
        or postrun.get("ledger_file_sha256") != file_sha256(ledger_path)
        or postrun.get("terminal_path") != str((base / "run/final/manifest.json").resolve())
        or postrun.get("terminal_file_sha256") != file_sha256(base / "run/final/manifest.json")
        or postrun.get("summary_path") != str((base / "profile-summary.json").resolve())
        or postrun.get("summary_file_sha256") != file_sha256(base / "profile-summary.json")
        or postrun.get("summary_semantic_sha256") != summary.get("semantic_sha256")
        or postrun.get("proposal_path") != str((base / "post-profile/proposal.json").resolve())
        or postrun.get("proposal_file_sha256") != file_sha256(base / "post-profile/proposal.json")
        or postrun.get("proposal_semantic_sha256") != proposal.get("semantic_sha256")
        or postrun.get("execution_repository_provenance_sha256")
        != binding.repository_provenance_sha256
    ):
        raise SupervisorError("bounded completed terminal/summary chain differs")
    current_identity_matches = _repository_preflight() == dict(binding.repository_provenance)
    return {
        "evidence_reauthenticated": True,
        "execution_identity_matches_current_code": current_identity_matches,
        "run_start_sha256": run_start_sha256,
        "ledger_sha256": file_sha256(ledger_path),
        "summary_file_sha256": file_sha256(base / "profile-summary.json"),
        "proposal_file_sha256": file_sha256(base / "post-profile/proposal.json"),
        "proposal_decision": proposal["decision"],
        "audit_code_sha256": _code_state_sha256(),
        "scientific_result": False,
        "publication_ready": False,
    }


def _require_v5_base(output: str | Path) -> Path:
    base = Path(output).resolve()
    if base == V1_INCIDENT_BASE.resolve():
        raise SupervisorError("V1_INCIDENT_INCOMPLETE: v1 is permanently nonresumable")
    if base == V2_COMPLETED_BASE.resolve():
        raise SupervisorError("V2_EVIDENCE_READ_ONLY: v2 cannot be executed or mutated")
    if base == V3_REJECTED_BASE.resolve():
        raise SupervisorError("V3_EVIDENCE_READ_ONLY: v3 cannot be executed or mutated")
    if base == V4_READ_ONLY_BASE.resolve():
        raise SupervisorError("V4_EVIDENCE_READ_ONLY: v4 cannot be executed or mutated")
    if base != EXACT_BASE.resolve():
        raise SupervisorError("operation requires the exact v5 bounded-smoke base")
    return base


def _prepare_failure_code(error: BaseException) -> str:
    if isinstance(error, SupervisorError):
        return "PREPARE_CONTRACT_REFUSAL"
    if isinstance(error, OSError):
        return "PREPARE_IO_FAILURE"
    return "PREPARE_INTERNAL_FAILURE"


def _safe_error_message(error: BaseException) -> str:
    normalized = " ".join(str(error).split())
    safe = "".join(character if 32 <= ord(character) < 127 else "?" for character in normalized)
    return safe[:256] or "prepare failed without a safe message"


def prepare(output: str | Path, *, runner_identity: str) -> Path:
    base = _require_v5_base(output)
    prerequisites = _repository_preflight()
    attempt_id = secrets.token_hex(32)
    try:
        base.mkdir(exist_ok=False)
        fsync_parent(base)
    except FileExistsError as error:
        raise SupervisorError(
            "prepare requires the exact fresh bounded-smoke /tmp output"
        ) from error
    claim_path = base / "prepare-claim.json"
    try:
        claim = _exclusive_artifact(
            claim_path,
            PREPARE_CLAIM_SCHEMA,
            {
                "prepare_attempt_id": attempt_id,
                "claimed_at_utc": _iso(_now()),
                "base_path": str(base),
                "run_id": RUN_ID,
                "purpose": PURPOSE,
                "runner_identity": runner_identity,
                "prerequisite_provenance": dict(prerequisites),
                "prerequisite_provenance_sha256": canonical_sha256(dict(prerequisites)),
                "resumable": False,
                "authorization_allowed": False,
                "execution_allowed": False,
                **_false_boundaries(),
            },
        )
    except BaseException as error:
        base_inventory = _lstat_tree(base)
        try:
            _exclusive_artifact(
                base / "PREPARE_CLAIM_FAILED.json",
                PREPARE_CLAIM_FAILURE_SCHEMA,
                {
                    "status": "PREPARE_CLAIM_FAILED",
                    "prepare_attempt_id": attempt_id,
                    "failed_at_utc": _iso(_now()),
                    "failure_code": "PREPARE_CLAIM_WRITE_FAILURE",
                    "error_type": type(error).__name__,
                    "error_message": _safe_error_message(error),
                    "base_path": str(base),
                    "base_inventory": base_inventory,
                    "base_inventory_sha256": canonical_sha256(base_inventory),
                    "run_id": RUN_ID,
                    "purpose": PURPOSE,
                    "runner_identity": runner_identity,
                    "prerequisite_provenance": dict(prerequisites),
                    "prerequisite_provenance_sha256": canonical_sha256(dict(prerequisites)),
                    "resumable": False,
                    "authorization_allowed": False,
                    "execution_allowed": False,
                    **_false_boundaries(),
                },
            )
        except BaseException as marker_error:
            raise SupervisorError(
                "prepare claim and failure-marker writes failed; claimed base is permanently burned"
            ) from marker_error
        raise
    progress: dict[str, str] = {"last_completed_checkpoint": "prepare-claim-written"}
    try:
        profile_path = _prepare_claimed(
            base,
            runner_identity=runner_identity,
            prerequisites=prerequisites,
            claim=claim,
            claim_path=claim_path,
            progress=progress,
        )
        profile = _validate_semantic_artifact(profile_path, PROFILE_SCHEMA)
        inventory = _lstat_tree(base / "preflight")
        base_inventory = _lstat_tree(base)
        _exclusive_artifact(
            base / "PREPARE_COMPLETE.json",
            PREPARE_COMPLETE_SCHEMA,
            {
                "status": "PREPARE_COMPLETE",
                "prepare_attempt_id": attempt_id,
                "prepare_claim_path": str(claim_path.resolve()),
                "prepare_claim_file_sha256": file_sha256(claim_path),
                "prepare_claim_semantic_sha256": claim["semantic_sha256"],
                "completed_at_utc": _iso(_now()),
                "run_id": RUN_ID,
                "purpose": PURPOSE,
                "runner_identity": runner_identity,
                "preflight_path": str((base / "preflight").resolve()),
                "preflight_inventory": inventory,
                "preflight_inventory_sha256": canonical_sha256(inventory),
                "base_inventory": base_inventory,
                "base_inventory_sha256": canonical_sha256(base_inventory),
                "profile_path": str(profile_path.resolve()),
                "profile_file_sha256": file_sha256(profile_path),
                "profile_semantic_sha256": profile["semantic_sha256"],
                "prerequisite_provenance": dict(prerequisites),
                "prerequisite_provenance_sha256": canonical_sha256(dict(prerequisites)),
                "resumable": False,
                "authorization_allowed": True,
                "execution_allowed": True,
                **_false_boundaries(),
            },
        )
        return profile_path
    except BaseException as error:
        partial_tree = _lstat_tree(base)
        try:
            _exclusive_artifact(
                base / "PREPARE_FAILED.json",
                PREPARE_FAILURE_SCHEMA,
                {
                    "status": "PREPARE_FAILED",
                    "stage": "prepare-preflight",
                    "reason": "PREPARE_EXCEPTION",
                    "error_type": type(error).__name__,
                    "error_message": _safe_error_message(error),
                    "failure_code": _prepare_failure_code(error),
                    "failed_at_utc": _iso(_now()),
                    "prepare_attempt_id": attempt_id,
                    "prepare_claim_path": str(claim_path.resolve()),
                    "prepare_claim_file_sha256": file_sha256(claim_path),
                    "prepare_claim_semantic_sha256": claim["semantic_sha256"],
                    "last_completed_checkpoint": progress["last_completed_checkpoint"],
                    "partial_tree_inventory": partial_tree,
                    "partial_tree_inventory_sha256": canonical_sha256(partial_tree),
                    "run_id": RUN_ID,
                    "purpose": PURPOSE,
                    "runner_identity": runner_identity,
                    "prerequisite_provenance": dict(prerequisites),
                    "prerequisite_provenance_sha256": canonical_sha256(dict(prerequisites)),
                    "resumable": False,
                    "authorization_allowed": False,
                    "execution_allowed": False,
                    **_false_boundaries(),
                },
            )
        except BaseException as marker_error:
            raise SupervisorError(
                "prepare failure-marker write failed; claimed base is permanently burned"
            ) from marker_error
        raise


def _prepare_claimed(
    base: Path,
    *,
    runner_identity: str,
    prerequisites: Mapping[str, object],
    claim: Mapping[str, object],
    claim_path: Path,
    progress: dict[str, str],
) -> Path:
    preflight = base / "preflight"
    preflight.mkdir()
    progress["last_completed_checkpoint"] = "preflight-directory-created"
    fixture = write_synthetic_fixture(preflight / "public-synthetic.npz")
    progress["last_completed_checkpoint"] = "synthetic-fixture-written"
    config = {
        "seed": SEED,
        "candidates": candidates(),
        "policies": {
            family: asdict(sidecar_policy(family)) for family in ("ridge", "xgboost", "mlp")
        },
    }
    atomic_json(preflight / "config.json", config)
    progress["last_completed_checkpoint"] = "config-written"
    with np.load(fixture, allow_pickle=False) as data:
        x_train = sparse.csr_matrix(data["x_train"])
        x_tune = sparse.csr_matrix(data["x_tune"])
    identity = {
        "feature_artifact_sha256": artifact_file_sha256(fixture),
        "recipe_provenance_sha256": canonical_sha256({"threshold": 0.70, "dtype": "float64"}),
        "strategy": "identity",
        "fold": 0,
        "train_ordered_sha256": canonical_sha256(list(range(TRAIN_ROWS))),
        "tune_ordered_sha256": canonical_sha256(list(range(TRAIN_ROWS, ROWS))),
    }
    profiles = profile_cache_options(
        x_train,
        x_tune,
        batch_indices=np.arange(24),
        identity=identity,
        device=__import__("torch").device("cpu"),
        directory=preflight / "cache",
        samples=3,
        epochs_cap=2,
    )
    progress["last_completed_checkpoint"] = "cache-options-profiled"
    choice_path = preflight / "cache-choice.json"
    freeze_cache_choice(
        choice_path,
        profiles,
        total_vram_bytes=0,
        free_vram_bytes=0,
        total_ram_bytes=os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
        available_ram_bytes=_available_ram_bytes(),
        free_disk_bytes=shutil.disk_usage(base.parent).free,
        cache_bytes=ROWS * FEATURES * 4,
        device_identity=detect_device_identity("cpu"),
    )
    progress["last_completed_checkpoint"] = "cache-choice-frozen"
    worksheet = build_memory_worksheet(
        rows=ROWS,
        columns=FEATURES,
        hidden_dims=[16],
        batch_size=24,
        validation_chunk_size=24,
        cache_option="streaming",
        total_ram_bytes=os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
        available_ram_bytes=_available_ram_bytes(),
        total_vram_bytes=0,
        free_vram_bytes=0,
        measured_rss_baseline_bytes=None,
        measured_rss_peak_bytes=None,
        measured_gpu_baseline_bytes=None,
        measured_gpu_peak_bytes=None,
        device_type="cpu",
        sparse_matrix_bytes=x_train.data.nbytes + x_train.indices.nbytes + x_train.indptr.nbytes,
    )
    atomic_json(preflight / "memory-worksheet.json", worksheet)
    progress["last_completed_checkpoint"] = "memory-worksheet-written"
    freeze_bounded_bootstrap_profile(
        preflight / "bootstrap-profile.json",
        input_path=fixture,
        cache_choice_path=choice_path,
        memory_worksheet=worksheet,
        config_sha256=file_sha256(preflight / "config.json"),
        code_sha256=str(prerequisites["code_state_sha256"]),
        dependency_sha256=str(prerequisites["dependency_sha256"]),
        runner_identity=runner_identity,
        repository_provenance=prerequisites,
        prepare_attempt_id=str(claim["prepare_attempt_id"]),
        prepare_claim_file_sha256=file_sha256(claim_path),
        prepare_claim_semantic_sha256=str(claim["semantic_sha256"]),
    )
    progress["last_completed_checkpoint"] = "bootstrap-profile-frozen"
    return preflight / "bootstrap-profile.json"


def _execution_lifecycle_state(base: Path) -> dict[str, str]:
    terminal = base / "run/final/manifest.json"
    ledger = base / "run/ledger/trials.jsonl"
    events_path = base / "run/telemetry/events.jsonl"
    postrun = base / "POSTRUN_MANIFEST.json"
    terminal_state = "manifest_present" if terminal.is_file() else "absent"
    ledger_state = "present" if ledger.is_file() else "absent"
    event_state = "absent"
    if events_path.is_file():
        try:
            events = validate_event_stream(events_path)
        except Exception:
            event_state = "invalid"
        else:
            run_complete_positions = [
                index for index, event in enumerate(events) if event.get("event") == "run_complete"
            ]
            if not run_complete_positions:
                event_state = "preterminal_only"
            elif run_complete_positions == [len(events) - 1]:
                event_state = "run_complete_last"
            else:
                event_state = "run_complete_partial_or_not_last"
    postrun_state = "absent"
    if postrun.exists():
        try:
            _validate_postrun_manifest_bindings(base)
        except Exception:
            postrun_state = "present_invalid"
        else:
            postrun_state = "valid_success_lock"
    return {
        "terminal_state": terminal_state,
        "event_state": event_state,
        "ledger_state": ledger_state,
        "postrun_state": postrun_state,
    }


def _lifecycle_seam(stage: str, operation: object, *args: object, **kwargs: object) -> object:
    try:
        return operation(*args, **kwargs)  # type: ignore[operator]
    except BaseException as error:
        if not hasattr(error, "failure_stage"):
            error.failure_stage = stage  # type: ignore[attr-defined]
        raise


def _record_v5_execution_incident(base: Path, error: BaseException) -> Path | None:
    state = _execution_lifecycle_state(base)
    if state["postrun_state"] == "valid_success_lock":
        return None
    path = base / "EXECUTION_INCIDENT.json"
    if path.exists():
        return path
    for durable_path in (
        base / "run/ledger/trials.jsonl",
        base / "run/telemetry/events.jsonl",
    ):
        if durable_path.is_file() and not durable_path.is_symlink():
            try:
                with durable_path.open("rb") as handle:
                    os.fsync(handle.fileno())
                fsync_parent(durable_path)
            except OSError:
                pass
    stage = getattr(error, "failure_stage", "unclassified")
    try:
        return _exclusive_artifact(
            path,
            EXECUTION_INCIDENT_SCHEMA,
            {
                "status": "EXECUTION_INCIDENT",
                "run_id": RUN_ID,
                "purpose": PURPOSE,
                "failed_at_utc": _iso(_now()),
                "failure_stage": str(stage),
                "exception_type": type(error).__name__,
                "exception_message_sha256": canonical_sha256(str(error)),
                **state,
                "resumable": False,
                "execution_allowed": False,
                "ranking_allowed": False,
                **_false_boundaries(),
            },
        )
    except BaseException:
        return None


def _execute_v5_body(
    output: str | Path,
    *,
    expected_review_file_sha256: str,
    expected_authorization_file_sha256: str,
    expected_reviewer_identity: str,
) -> Path:
    base = _require_v5_base(output)
    forbidden_outputs = (
        base / "run",
        base / "attempt",
        base / "profile-summary.json",
        base / "post-profile",
    )
    if not base.is_dir() or any(path.exists() for path in forbidden_outputs):
        raise SupervisorError("execute requires reviewed preflight and a fresh run directory")
    binding = build_binding(
        base / "preflight",
        expected_review_file_sha256=expected_review_file_sha256,
        expected_authorization_file_sha256=expected_authorization_file_sha256,
        expected_reviewer_identity=expected_reviewer_identity,
    )
    authorization = _validate_semantic_artifact(
        binding.bootstrap_authorization_path,
        AUTH_SCHEMA,
        expected_file_sha256=binding.bootstrap_authorization_file_sha256,
    )
    attempt = base / "attempt"
    attempt.mkdir(exist_ok=False)
    claim = {
        "nonce": authorization["nonce"],
        "authorization_file_sha256": file_sha256(binding.bootstrap_authorization_path),
        "authorization_semantic_sha256": authorization["semantic_sha256"],
        "runner_identity": authorization["runner_identity"],
        "run_id": RUN_ID,
        "purpose": PURPOSE,
        "claimed_at_utc": _iso(_now()),
        **_false_boundaries(),
    }
    _exclusive_artifact(attempt / "nonce-claim.json", "qm9-v31-bounded-attempt-claim-v5", claim)
    profile = _validate_semantic_artifact(binding.bootstrap_profile_path, PROFILE_SCHEMA)
    observation = _bound_resource_observation(
        base.parent,
        int(profile["memory_worksheet"]["capacity"]["cpu_required_with_margin_bytes"]),
        authorization,
    )
    _exclusive_artifact(
        attempt / "resource-observation.json",
        "qm9-v31-bounded-resource-observation-v5",
        observation,
    )
    specifications = planned_trials()
    supervisor = RunSupervisor.for_bounded_profile(base / "run", binding, specifications)
    consume_bounded_bootstrap_authorization(binding, run_start_sha256=supervisor.run_start_sha256)
    preflight_choice = base / "preflight/cache-choice.json"
    run_choice = base / "run/cache/cache-choice.json"
    atomic_bytes(run_choice, preflight_choice.read_bytes())
    if file_sha256(run_choice) != file_sha256(preflight_choice):
        raise SupervisorError("run cache-choice copy differs from reviewed source")
    profile_rows: list[dict[str, object]] = []
    started = time.monotonic()
    for specification, candidate in zip(specifications, candidates(), strict=True):
        trial = specification.identity
        if file_sha256(run_choice) != binding.profile_sha256:
            raise SupervisorError("run cache-choice bytes changed before scheduling")
        trial_observation = _bound_resource_observation(
            base.parent,
            int(profile["memory_worksheet"]["capacity"]["cpu_required_with_margin_bytes"]),
            authorization,
        )
        trial_observation_path = (
            attempt / f"resource-observation-{trial.trial_index}-{trial.family}.json"
        )
        _exclusive_artifact(
            trial_observation_path,
            "qm9-v31-bounded-trial-resource-observation-v5",
            {
                **trial_observation,
                "trial": asdict(trial),
                "trial_key": canonical_sha256(asdict(trial)),
                "policy_sha256": canonical_sha256(asdict(sidecar_policy(trial.family))),
                "run_start_sha256": supervisor.run_start_sha256,
                "receipt_file_sha256": file_sha256(attempt / "authorization-consumed.json"),
                "previous_ledger_entry_sha256": (
                    supervisor.ledger.rows[-1]["entry_sha256"] if supervisor.ledger.rows else None
                ),
            },
        )
        reserve_evidence = validate_bounded_trial_observation(
            binding,
            path=trial_observation_path,
            trial=trial,
            run_start_sha256=supervisor.run_start_sha256,
            previous_ledger_entry_sha256=(
                str(supervisor.ledger.rows[-1]["entry_sha256"]) if supervisor.ledger.rows else None
            ),
        )
        supervisor.reserve(trial, reserve_evidence)
        supervisor.start_trial(trial)
        result_path = base / f"run/worker-results/{trial.family}.json"
        worker_profile = base / f"run/worker-profiles/{trial.family}.json"
        spec_path = base / f"run/worker-specs/{trial.family}.json"
        kwargs: dict[str, object] = {}
        if trial.family == "mlp":
            kwargs["cache_identity"] = {
                "feature_artifact_sha256": artifact_file_sha256(
                    base / "preflight/public-synthetic.npz"
                ),
                "recipe_provenance_sha256": canonical_sha256(
                    {"threshold": 0.70, "dtype": "float64"}
                ),
                "strategy": "identity",
                "fold": 0,
                "train_ordered_sha256": canonical_sha256(list(range(TRAIN_ROWS))),
                "tune_ordered_sha256": canonical_sha256(list(range(TRAIN_ROWS, ROWS))),
            }
            kwargs.update(cache_directory=base / "run/cache/features", cache_choice_path=run_choice)
        write_real_backend_worker_spec(
            spec_path,
            run_root=base / "run",
            data_path=base / "preflight/public-synthetic.npz",
            data_scope="synthetic",
            result_path=result_path,
            profile_path=worker_profile,
            checkpoint_root=supervisor.checkpoints.root,
            candidate=candidate,
            trial=asdict(trial),
            completion=supervisor.completion_contract(trial),
            checkpoint_bindings=supervisor.checkpoints.bindings,
            device="cpu",
            **kwargs,
        )
        code = ExternalWorkerSidecar(supervisor, sidecar_policy(trial.family)).run(
            trial,
            [
                sys.executable,
                "-m",
                "mist_transfer_benchmark.qm9.model_selection_v31_backends",
                "--worker-spec",
                str(spec_path),
            ],
            heartbeat_path=base / f"run/telemetry/{trial.family}.heartbeat",
            environment={"CUDA_VISIBLE_DEVICES": ""},
            prepared=True,
            result_path=result_path,
        )
        if code != 0:
            raise SupervisorError(f"bounded profile worker failed: {trial.family}")
        result = _read_json(result_path)
        profile_rows.append(
            load_real_backend_profile(
                worker_profile, expected_metadata_sha256=str(result["metadata_sha256"])
            )
        )
    _lifecycle_seam("ledger_close", supervisor.close_ledger)
    replayed = supervisor._strict_checkpoint_replay()
    replay_sha256 = canonical_sha256([row["manifest"]["prediction_sha256"] for row in replayed])
    events = validate_event_stream(supervisor.events.path)
    cache_choice = _read_json(preflight_choice)
    resource_peaks = _completed_resource_peaks(events)
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "purpose": "engineering_smoke",
        "status": "complete",
        "run_id": RUN_ID,
        "trial_states": {
            row.identity.family: supervisor.ledger.state(row.identity) for row in specifications
        },
        "event_count": len(events) + 1,
        "checkpoint_count": 3,
        "replay_count": len(replayed),
        "replay_sha256": replay_sha256,
        "ledger_sha256": file_sha256(supervisor.ledger.path),
        "duration_seconds": time.monotonic() - started,
        "resource_peaks": resource_peaks,
        "profiles": [
            {
                "family": row["family"],
                "stage_seconds": row["metadata"].get("stage_seconds", {}),
                "loss_curves": row["metadata"].get("curves", {"available": False}),
                "anomaly_flags": row["metadata"].get("curve", {"status": "unavailable"}),
            }
            for row in profile_rows
        ],
        "mlp_cache": {
            "choice_file_sha256": file_sha256(preflight_choice),
            "choice_semantic_sha256": cache_choice["semantic_sha256"],
            "profiles_sha256": cache_choice["profiles_sha256"],
            "chosen_option": cache_choice["chosen_option"],
        },
        "protected_label_reads": 0,
        **_false_boundaries(),
    }
    path = base / "profile-summary.json"
    _write_immutable_json(path, summary)
    post_profile = {
        "schema_version": POST_PROFILE_SCHEMA,
        "measurement_status": "INSUFFICIENT_OBSERVATIONS",
        "decision": "NOT_APPROVED",
        "observations_per_family": 1,
        "families": ["ridge", "xgboost", "mlp"],
        "w1_feasibility_approved": False,
        "general_training_approved": False,
        "scientific_result": False,
        "publication_ready": False,
    }
    _write_immutable_json(base / "post-profile/proposal.json", post_profile)
    summary_artifact = _validate_semantic_artifact(path, SUMMARY_SCHEMA)
    proposal_path = base / "post-profile/proposal.json"
    proposal_artifact = _validate_semantic_artifact(proposal_path, POST_PROFILE_SCHEMA)
    receipt_path = base / "attempt/authorization-consumed.json"
    receipt_artifact = _validate_semantic_artifact(
        receipt_path, "qm9-v31-bounded-authorization-consumed-v5"
    )
    expected_final_path_types = _expected_final_path_types(
        base, specifications, replayed, profile_rows
    )
    _lifecycle_seam(
        "preterminal_contract",
        _validate_path_type_contract,
        base,
        _path_type_contract_for_phase(expected_final_path_types, "preterminal"),
    )
    _lifecycle_seam(
        "terminal_finalize",
        supervisor.finalize_profile_success,
        {
            "worker_profile_count": 3,
            "worker_result_file_sha256": {
                family: file_sha256(base / f"run/worker-results/{family}.json")
                for family in ("ridge", "xgboost", "mlp")
            },
            "worker_profile_file_sha256": {
                family: file_sha256(base / f"run/worker-profiles/{family}.json")
                for family in ("ridge", "xgboost", "mlp")
            },
            "trial_resource_observation_file_sha256": {
                row.identity.family: file_sha256(
                    base
                    / (
                        "attempt/resource-observation-"
                        f"{row.identity.trial_index}-{row.identity.family}.json"
                    )
                )
                for row in specifications
            },
            "authorization_consumed_file_sha256": file_sha256(receipt_path),
            "authorization_consumed_semantic_sha256": receipt_artifact["semantic_sha256"],
            "summary_file_sha256": file_sha256(path),
            "summary_semantic_sha256": summary_artifact["semantic_sha256"],
            "proposal_file_sha256": file_sha256(proposal_path),
            "proposal_semantic_sha256": proposal_artifact["semantic_sha256"],
            "expected_final_path_types_sha256": canonical_sha256(expected_final_path_types),
            "event_count": len(events),
            "event_head_sha256": events[-1]["event_sha256"],
            "expected_run_complete_sequence": len(events) + 1,
        },
    )
    _lifecycle_seam(
        "terminal_tree",
        _validate_path_type_contract,
        base,
        _path_type_contract_for_phase(expected_final_path_types, "terminal"),
    )
    _lifecycle_seam("events_close", supervisor.close_events)
    supervisor.close()
    terminal_path = base / "run/final/manifest.json"
    final_base_inventory = _lstat_tree(base)
    run_start_artifact = _read_json(base / "run/run_start.json")
    _lifecycle_seam(
        "postrun_write",
        _exclusive_artifact,
        base / "POSTRUN_MANIFEST.json",
        POSTRUN_MANIFEST_SCHEMA,
        {
            "status": "PROFILE_COMPLETE",
            "run_id": RUN_ID,
            "purpose": PURPOSE,
            "created_at_utc": _iso(_now()),
            "success_lock": True,
            "streams_closed": {"ledger": True, "telemetry": True},
            "expected_final_path_types": expected_final_path_types,
            "expected_final_path_types_sha256": canonical_sha256(expected_final_path_types),
            "run_start_path": str((base / "run/run_start.json").resolve()),
            "run_start_file_sha256": file_sha256(base / "run/run_start.json"),
            "run_start_semantic_sha256": run_start_artifact["semantic_sha256"],
            "ledger_path": str((base / "run/ledger/trials.jsonl").resolve()),
            "ledger_file_sha256": file_sha256(base / "run/ledger/trials.jsonl"),
            "terminal_path": str(terminal_path.resolve()),
            "terminal_file_sha256": file_sha256(terminal_path),
            "summary_path": str(path.resolve()),
            "summary_file_sha256": file_sha256(path),
            "summary_semantic_sha256": summary_artifact["semantic_sha256"],
            "proposal_path": str(proposal_path.resolve()),
            "proposal_file_sha256": file_sha256(proposal_path),
            "proposal_semantic_sha256": proposal_artifact["semantic_sha256"],
            "execution_repository_provenance_sha256": binding.repository_provenance_sha256,
            "final_base_inventory": final_base_inventory,
            "final_base_inventory_sha256": canonical_sha256(final_base_inventory),
            **_false_boundaries(),
        },
    )
    _lifecycle_seam(
        "public_verify",
        assert_profile_complete_postrun,
        base,
        expected_review_file_sha256=expected_review_file_sha256,
        expected_authorization_file_sha256=expected_authorization_file_sha256,
        expected_reviewer_identity=expected_reviewer_identity,
        revalidate_repository=True,
    )
    return path


def execute(
    output: str | Path,
    *,
    expected_review_file_sha256: str,
    expected_authorization_file_sha256: str,
    expected_reviewer_identity: str,
) -> Path:
    base = Path(output).resolve()
    try:
        return _execute_v5_body(
            output,
            expected_review_file_sha256=expected_review_file_sha256,
            expected_authorization_file_sha256=expected_authorization_file_sha256,
            expected_reviewer_identity=expected_reviewer_identity,
        )
    except BaseException as error:
        if (base / "attempt/nonce-claim.json").is_file():
            _record_v5_execution_incident(base, error)
        raise


def _main() -> int:
    parser = argparse.ArgumentParser(description="Bounded CPU-only QM9 v3.1 engineering profile")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--stage",
        choices=("dry-run", "prepare", "authorize", "execute", "verify"),
        required=True,
    )
    parser.add_argument("--runner-identity", default=DEFAULT_RUNNER_IDENTITY)
    parser.add_argument("--expected-review-sha256")
    parser.add_argument("--expected-authorization-sha256")
    parser.add_argument("--expected-reviewer-identity")
    args = parser.parse_args()
    if Path(args.output).resolve() != EXACT_BASE.resolve():
        parser.error("--output must be /tmp/qm9-v31-bounded-smoke-v5")
    if args.stage == "dry-run":
        if Path(args.output).exists():
            parser.error("dry-run requires a fresh output path")
        print(
            json.dumps(
                {
                    "run_id": RUN_ID,
                    "rows": ROWS,
                    "features": FEATURES,
                    "families": ["ridge", "xgboost", "mlp"],
                    **_false_boundaries(),
                },
                sort_keys=True,
            )
        )
    elif args.stage == "prepare":
        prepare(args.output, runner_identity=args.runner_identity)
    elif args.stage == "authorize":
        preflight = Path(args.output) / "preflight"
        freeze_bounded_bootstrap_authorization(
            preflight / "bootstrap-authorization.json",
            profile_path=preflight / "bootstrap-profile.json",
            review_path=preflight / "independent-review.json",
        )
    else:
        if not all(
            (
                args.expected_review_sha256,
                args.expected_authorization_sha256,
                args.expected_reviewer_identity,
            )
        ):
            parser.error("execute/verify requires out-of-band hashes and reviewer identity")
        kwargs = {
            "expected_review_file_sha256": args.expected_review_sha256,
            "expected_authorization_file_sha256": args.expected_authorization_sha256,
            "expected_reviewer_identity": args.expected_reviewer_identity,
        }
        if args.stage == "verify":
            print(
                json.dumps(
                    assert_profile_complete_postrun(
                        args.output,
                        **kwargs,
                        revalidate_repository=False,
                    ),
                    sort_keys=True,
                )
            )
        else:
            execute(args.output, **kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
