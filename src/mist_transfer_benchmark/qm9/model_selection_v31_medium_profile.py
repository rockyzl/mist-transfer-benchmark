"""Governed train-only medium profiler for the QM9 v3.1 pipeline.

This is intentionally a separate lineage from the fixed synthetic v5 smoke.  It can read only
authenticated fixed-train rows and has no API for outer-validation or test targets.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from scipy import sparse

from .constants import TARGET_COLUMNS
from .data import load_qm9_identities
from .model_selection_v3 import group_safe_inner_split
from .model_selection_v31_checkpoint import (
    COMPLETION_STATUS,
    CheckpointStore,
    ReplayContract,
)
from .model_selection_v31_guard import GuardThresholds, RunGuard
from .model_selection_v31_observability import (
    EventSink,
    HeartbeatEmitter,
    SystemResourceProvider,
    TrialIdentity,
    fsync_parent,
    validate_event_stream,
)
from .model_selection_v31_supervisor import (
    ExternalWorkerSidecar,
    MediumProfileRunBinding,
    PlannedTrial,
    RunSupervisor,
    SidecarPolicy,
    SupervisorError,
    TrialLedger,
    atomic_json,
    canonical_sha256,
    detect_device_identity,
    file_sha256,
    trial_payload,
)
from .phase2_contract import verify_phase1_evidence
from .phase2_targets import load_targets_for_indices
from .selection_v3_data import (
    authenticate_feature_artifact,
    canonical_connectivity_groups,
)

PRODUCTION_RUN_ID = "qm9-v31-medium-profile-cpu-v1"
PUBLIC_SYNTHETIC_RUN_ID = "qm9-v31-medium-profile-public-synthetic-v1"
PUBLIC_FIXED_RUN_ID = "qm9-v31-public-fixed-three-backend-cpu-v1"
PUBLIC_SYNTHETIC_MODE = os.environ.get("QM9_MEDIUM_PUBLIC_SYNTHETIC_SEAM") == "1"
PUBLIC_FIXED_MODE = os.environ.get("QM9_V31_PUBLIC_FIXED_BACKENDS") == "1"
if PUBLIC_SYNTHETIC_MODE and PUBLIC_FIXED_MODE:
    raise RuntimeError("public marker and public fixed backend modes are mutually exclusive")
RUN_ID = (
    PUBLIC_FIXED_RUN_ID
    if PUBLIC_FIXED_MODE
    else PUBLIC_SYNTHETIC_RUN_ID if PUBLIC_SYNTHETIC_MODE else PRODUCTION_RUN_ID
)
PURPOSE = (
    "public_synthetic_three_backend_engineering_seam"
    if PUBLIC_FIXED_MODE
    else "train_only_medium_profile"
)
SUPERVISOR_PURPOSE = (
    "public_synthetic_three_backend_engineering_seam"
    if PUBLIC_FIXED_MODE
    else "feasibility_evidence_generation"
)
PRODUCTION_EXACT_BASE = Path("/tmp/qm9-v31-medium-profile-v1")
PUBLIC_SYNTHETIC_BASE = Path("/tmp/qm9-v31-medium-profile-public-synthetic-v1")
PUBLIC_FIXED_BASE = Path("/tmp/qm9-v31-public-fixed-three-backend-cpu-v1")
EXACT_BASE = (
    PUBLIC_FIXED_BASE
    if PUBLIC_FIXED_MODE
    else PUBLIC_SYNTHETIC_BASE if PUBLIC_SYNTHETIC_MODE else PRODUCTION_EXACT_BASE
)
OUTER_ENVELOPE = Path("/tmp/qm9-v31-medium-profile-launch-envelope-v2")
PUBLIC_SYNTHETIC_OUTER_ENVELOPE = Path(
    "/tmp/qm9-v31-medium-profile-public-synthetic-launch-envelope-v1"
)
PUBLIC_FIXED_OUTER_ENVELOPE = Path(
    "/tmp/qm9-v31-public-fixed-three-backend-envelope-v1"
)
SEED = 20260713
SAMPLE_ROWS = 8192
TUNE_FRACTION = 0.10
RSS_HARD_BYTES = 4 * 1024**3
RSS_SUCCESS_BYTES = int(3.2 * 1024**3)
OUTER_SECONDS = 2400
OUTER_COMMAND_SECONDS = 2280
WORKLOAD_SECONDS = 2280
FINALIZE_SECONDS = 120
ARTIFACT_MAX_BYTES = 1024**3
DISK_FREE_MIN_BYTES = 5 * 1024**3
FAMILY_SOFT_SECONDS = {"ridge": 180, "xgboost": 1200, "mlp": 720}
THREADS = 1
SCHEMA_PREFIX = "qm9-v31-public-fixed" if PUBLIC_FIXED_MODE else "qm9-v31-medium-profile"
PREPARE_SCHEMA = f"{SCHEMA_PREFIX}-prepare-v1"
PREPARE_CLAIM_SCHEMA = f"{SCHEMA_PREFIX}-prepare-claim-v1"
PREPARE_FAILURE_SCHEMA = f"{SCHEMA_PREFIX}-prepare-failure-v1"
REVIEW_SCHEMA = f"{SCHEMA_PREFIX}-review-v1"
AUTH_SCHEMA = f"{SCHEMA_PREFIX}-authorization-v1"
TERMINAL_SCHEMA = f"{SCHEMA_PREFIX}-terminal-v1"
POSTRUN_SCHEMA = f"{SCHEMA_PREFIX}-postrun-v2"
EXTERNAL_REVIEW_SCHEMA = f"{SCHEMA_PREFIX}-external-review-v1"
EXTERNAL_AUTH_SCHEMA = f"{SCHEMA_PREFIX}-external-authorization-v1"
REVIEW_DECISION = (
    "APPROVE_PUBLIC_FIXED_THREE_BACKEND_ONLY"
    if PUBLIC_FIXED_MODE
    else "APPROVE_MEDIUM_PROFILE_ONLY"
)
AUTH_DECISION = (
    "AUTHORIZE_PUBLIC_FIXED_THREE_BACKEND_ONCE"
    if PUBLIC_FIXED_MODE
    else "AUTHORIZE_MEDIUM_PROFILE_ONLY"
)
_DECISION_BINDING_KEYS = {
    "nonce",
    "argv",
    "argv_sha256",
    "cwd",
    "output",
    "harness_sha256",
    "script_sha256",
    "code_sha256",
    "lock_sha256",
    "protocol_sha256",
    "config_sha256",
    "input_sha256",
    "sample_sha256",
    "plan_sha256",
    "device_sha256",
    "budget_sha256",
    "outer_wrapper",
    "outer_wrapper_sha256",
}
_EXTERNAL_REVIEW_KEYS = {
    "schema_version",
    "run_id",
    "decision",
    "reviewer_identity",
    "prepare_file_sha256",
    "prepare_semantic_sha256",
    "issued_at_utc",
    "expires_at_utc",
    "semantic_sha256",
} | _DECISION_BINDING_KEYS
_EXTERNAL_AUTH_KEYS = {
    "schema_version",
    "run_id",
    "decision",
    "authorizer_identity",
    "reviewer_identity",
    "review_file_sha256",
    "review_semantic_sha256",
    "issued_at_utc",
    "expires_at_utc",
    "semantic_sha256",
} | _DECISION_BINDING_KEYS
if PUBLIC_FIXED_MODE:
    _EXTERNAL_REVIEW_KEYS |= {"purpose", "reviewer_tool_identity"}
    _EXTERNAL_AUTH_KEYS |= {"purpose", "authorizer_tool_identity", "one_use"}


def _false_boundaries() -> dict[str, bool]:
    return {
        "outer_validation_targets_read": False,
        "test_targets_read": False,
        "private_evaluation_labels_read": False,
        "protected_labels_read": False,
        "ranking_allowed": False,
        "ensemble_allowed": False,
        "promotion_allowed": False,
        "w1_allowed": False,
        "general_training_allowed": False,
        "scientific_result": False,
        "publication_ready": False,
    }


def _label_reads(*, source: int = 0, derived: int = 0) -> dict[str, int]:
    return {
        "source_fixed_train_target_loads": source,
        "derived_train_target_artifact_reads": derived,
        "outer_validation": 0,
        "test": 0,
        "protected": 0,
        "private_evaluation": 0,
    }


def _successful_label_reads() -> dict[str, int]:
    if PUBLIC_FIXED_MODE:
        return {**_label_reads(), "synthetic_target_artifact_reads": 3}
    return _label_reads() if PUBLIC_SYNTHETIC_MODE else _label_reads(source=1, derived=3)


def run_public_synthetic_outer_workload(
    base: Path, _prepared: Mapping[str, object]
) -> Mapping[str, object]:
    """Public-data-free workload used only by the explicitly opt-in outer seam test."""

    if not PUBLIC_SYNTHETIC_MODE or base.resolve() != PUBLIC_SYNTHETIC_BASE.resolve():
        raise SupervisorError("public synthetic workload is forbidden outside its isolated seam")
    marker = {
        "schema_version": "qm9-v31-public-synthetic-workload-v1",
        "run_id": RUN_ID,
        "status": "COMPLETE",
        "label_reads": _label_reads(),
        **_false_boundaries(),
    }
    _write_immutable(base / "PUBLIC_SYNTHETIC_WORKLOAD.json", marker)
    return {
        "peak_rss_bytes": 0,
        "label_reads": _label_reads(),
        "trial_states": {"public_synthetic": "complete"},
    }


def frozen_config() -> dict[str, object]:
    return {
        "schema_version": "qm9-v31-medium-profile-config-v1",
        "run_id": RUN_ID,
        "purpose": PURPOSE,
        "sample": {
            "rows": SAMPLE_ROWS,
            "method": "stable-hash-whole-identity-groups-until-at-least-threshold",
            "inner_split": "identity-group-90-10",
            "seed": SEED,
        },
        "cpu": {"only": True, "threads": THREADS},
        "limits": {
            "rss_hard_bytes": RSS_HARD_BYTES,
            "rss_success_bytes": RSS_SUCCESS_BYTES,
            "outer_seconds": OUTER_SECONDS,
            "outer_command_seconds": OUTER_COMMAND_SECONDS,
            "workload_drain_seconds": WORKLOAD_SECONDS,
            "finalize_seconds": FINALIZE_SECONDS,
            "artifact_max_bytes": ARTIFACT_MAX_BYTES,
            "disk_free_min_bytes": DISK_FREE_MIN_BYTES,
            "family_soft_seconds": FAMILY_SOFT_SECONDS,
        },
        "candidates": candidates(),
        **_false_boundaries(),
    }


def candidates() -> list[dict[str, object]]:
    return [
        {
            "id": "ridge|descriptor_scaled_only|alpha-1",
            "family": "ridge",
            "recipe": "descriptor_scaled_only",
            "params": {"alpha": 1.0},
        },
        {
            "id": "xgboost|all_raw|shallow_fast-200",
            "family": "xgboost",
            "recipe": "all_raw",
            "params": {
                "max_depth": 4,
                "learning_rate": 0.08,
                "n_estimators": 200,
                "subsample": 0.90,
                "colsample_bytree": 0.90,
                "min_child_weight": 1.0,
            },
            "early_stopping_rounds": 35,
        },
        {
            "id": "mlp|all_raw|compact",
            "family": "mlp",
            "recipe": "all_raw",
            "params": {
                "hidden_dims": [384, 192],
                "dropout": 0.05,
                "learning_rate": 0.001,
                "weight_decay": 0.00001,
            },
            "training": {
                "batch_size": 512,
                "max_epochs": 20,
                "patience": 10,
                "min_delta": 0.0001,
            },
            "monitoring": {
                "validation_increase_mark_after": 2,
                "max_validation_mnmae_after_five_epochs": 0.5,
                "minimum_relative_train_improvement_by_epoch_ten": 0.01,
                "gradient_norm_hard_max": 100.0,
                "gradient_norm_hard_min": 0.00000001,
                "gradient_min_consecutive_batches": 3,
                "target_degradation_warning": 0.02,
            },
        },
    ]


def _require_exact_base(output: str | Path) -> Path:
    base = Path(output).resolve()
    if base != EXACT_BASE.resolve():
        raise SupervisorError(f"medium profile requires exact {EXACT_BASE}")
    return base


def _write_immutable(path: Path, payload: Mapping[str, object]) -> dict[str, object]:
    unsigned = dict(payload)
    artifact = {**unsigned, "semantic_sha256": canonical_sha256(unsigned)}
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    try:
        encoded = json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        os.write(descriptor, encoded)
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)
    fsync_parent(path)
    return artifact


def _load_artifact(path: Path, schema: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise SupervisorError("medium lifecycle artifact must be a regular file")
    artifact = json.loads(path.read_text())
    unsigned = dict(artifact)
    semantic = unsigned.pop("semantic_sha256", None)
    if artifact.get("schema_version") != schema or semantic != canonical_sha256(unsigned):
        raise SupervisorError("medium lifecycle artifact semantic binding differs")
    if any(artifact.get(key) is not False for key in _false_boundaries()):
        raise SupervisorError("medium lifecycle label/scientific boundary differs")
    return artifact


def _repository_provenance() -> dict[str, object]:
    try:
        root = Path(
            subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ).resolve()
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        listed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-co", "--exclude-standard", "-z"],
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
    except (OSError, subprocess.SubprocessError) as error:
        raise SupervisorError("medium repository provenance is unavailable") from error
    manifest = []
    for raw in sorted(row for row in listed if row):
        relative = raw.decode(errors="surrogateescape")
        path = root / relative
        if path.is_file() and not path.is_symlink():
            manifest.append(
                {"path": relative, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            )
    lock = root / "uv.lock"
    config_source = root / "configs/qm9_model_selection_v3.toml"
    if any(path.is_symlink() or not path.is_file() for path in (lock, config_source)):
        raise SupervisorError("medium dependency/config provenance is unavailable")
    unsigned = {
        "repository_root": str(root),
        "head": head,
        "manifest": manifest,
        "manifest_sha256": canonical_sha256(manifest),
        "uv_lock_sha256": file_sha256(lock),
        "source_config_sha256": file_sha256(config_source),
    }
    harness_code_state = {
        "repository_root": str(root),
        "head": head,
        "file_manifest_sha256": canonical_sha256(manifest),
        "file_count": len(manifest),
    }
    unsigned["harness_code_state_sha256"] = canonical_sha256(harness_code_state)
    return {**unsigned, "provenance_sha256": canonical_sha256(unsigned)}


def _tree_manifest(root: Path, *, excluded: set[str] | None = None) -> list[dict[str, object]]:
    rows = []
    exclusions = excluded or set()
    if not root.exists():
        return rows
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative in exclusions:
            continue
        if path.is_symlink():
            raise SupervisorError("medium artifact tree contains a symlink")
        if path.is_dir():
            rows.append({"path": relative, "type": "directory"})
        elif path.is_file():
            rows.append(
                {
                    "path": relative,
                    "type": "file",
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
        else:
            raise SupervisorError("medium artifact tree contains an unsupported entry")
    return rows


def _artifact_bytes(root: Path) -> int:
    return sum(
        int(row.get("bytes", 0))
        for row in _tree_manifest(root)
        if row.get("type") == "file"
    )


def _load_external_decision(
    path: Path,
    *,
    schema: str,
    expected_sha256: str,
    expected_decision: str,
) -> dict[str, object]:
    if path.is_symlink() or not path.is_file() or file_sha256(path) != expected_sha256:
        raise SupervisorError("external decision physical hash differs")
    artifact = json.loads(path.read_text())
    expected_keys = (
        _EXTERNAL_REVIEW_KEYS if schema == EXTERNAL_REVIEW_SCHEMA else _EXTERNAL_AUTH_KEYS
    )
    if not isinstance(artifact, dict) or set(artifact) != expected_keys:
        raise SupervisorError("external decision exact schema differs")
    unsigned = dict(artifact)
    semantic = unsigned.pop("semantic_sha256", None)
    if (
        artifact.get("schema_version") != schema
        or semantic != canonical_sha256(unsigned)
        or artifact.get("decision") != expected_decision
        or artifact.get("run_id") != RUN_ID
        or (PUBLIC_FIXED_MODE and artifact.get("purpose") != PURPOSE)
        or (
            PUBLIC_FIXED_MODE
            and schema == EXTERNAL_REVIEW_SCHEMA
            and not artifact.get("reviewer_tool_identity")
        )
        or (
            PUBLIC_FIXED_MODE
            and schema == EXTERNAL_AUTH_SCHEMA
            and (
                not artifact.get("authorizer_tool_identity")
                or artifact.get("one_use") is not True
            )
        )
    ):
        raise SupervisorError("external decision semantic contract differs")
    try:
        issued = datetime.fromisoformat(str(artifact.get("issued_at_utc")))
        expires = datetime.fromisoformat(str(artifact.get("expires_at_utc")))
    except ValueError as error:
        raise SupervisorError("external decision time contract differs") from error
    now = datetime.now(UTC)
    if (
        issued.tzinfo is None
        or expires.tzinfo is None
        or not issued <= now < expires
        or expires - issued > timedelta(hours=1)
    ):
        raise SupervisorError("external decision time contract differs")
    return artifact


def _decision_bindings(prepared: Mapping[str, object], *, nonce: str) -> dict[str, object]:
    contract = prepared["execution_contract"]
    provenance = prepared["repository_provenance"]
    if not isinstance(nonce, str) or len(nonce) != 64 or any(
        character not in "0123456789abcdef" for character in nonce
    ):
        raise SupervisorError("medium decision nonce must be a lowercase SHA-256 token")
    return {
        "nonce": nonce,
        "argv": contract["argv"],
        "argv_sha256": contract["argv_sha256"],
        "cwd": contract["cwd"],
        "output": contract["output"],
        "harness_sha256": contract["harness_sha256"],
        "script_sha256": contract["script_sha256"],
        "code_sha256": provenance["manifest_sha256"],
        "lock_sha256": contract["lock_sha256"],
        "protocol_sha256": contract["protocol_sha256"],
        "config_sha256": prepared["config_file_sha256"],
        "input_sha256": prepared["target_free_input_inventory_sha256"],
        "sample_sha256": prepared["sample_sha256"],
        "plan_sha256": prepared["plan_sha256"],
        "device_sha256": prepared["device_identity_sha256"],
        "budget_sha256": prepared["budget_contract_sha256"],
        "outer_wrapper": contract["outer_wrapper"],
        "outer_wrapper_sha256": contract["outer_wrapper_sha256"],
    }


def stable_group_sample(groups: Sequence[object], *, rows: int = SAMPLE_ROWS) -> np.ndarray:
    """Append complete hash-ordered identity groups until the threshold is first reached."""

    values = np.asarray(groups).astype(str)
    members: dict[str, list[int]] = {}
    for index, group in enumerate(values):
        members.setdefault(group, []).append(index)
    if len(values) < rows:
        raise SupervisorError("fixed train has too few rows for medium sample")
    ordered = sorted(
        members,
        key=lambda group: hashlib.sha256(str(SEED).encode() + group.encode()).hexdigest(),
    )
    selected: list[int] = []
    for group in ordered:
        selected.extend(members[group])
        if len(selected) >= rows:
            break
    result = np.asarray(selected, dtype=np.int64)
    selected_set = set(result)
    if len(result) < rows or any(
        set(members[group]) - selected_set for group in set(values[result])
    ):
        raise SupervisorError("medium sample split an identity group")
    return result


def medium_split(groups: Sequence[object]) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(groups)
    train, tune = group_safe_inner_split(values, tune_fraction=TUNE_FRACTION, seed=SEED)
    if set(values[train]) & set(values[tune]):
        raise SupervisorError("medium identity groups cross train/tune")
    return train, tune


def sample_evidence(
    source_indices: Sequence[int], groups: Sequence[object]
) -> dict[str, object]:
    indices = np.asarray(source_indices, dtype=np.int64)
    identities = np.asarray(groups).astype(str)
    if len(indices) != len(identities) or len(indices) < SAMPLE_ROWS:
        raise SupervisorError("medium sample evidence dimensions differ")
    train, tune = medium_split(identities)
    unsigned = {
        "threshold_rows": SAMPLE_ROWS,
        "actual_rows": len(indices),
        "actual_groups": len(set(identities)),
        "ordered_source_rows_sha256": canonical_sha256(indices.tolist()),
        "ordered_identity_groups_sha256": canonical_sha256(identities.tolist()),
        "train_positions_sha256": canonical_sha256(train.tolist()),
        "tune_positions_sha256": canonical_sha256(tune.tolist()),
        "actual_tune_row_fraction": len(tune) / len(indices),
        "seed": SEED,
    }
    return {**unsigned, "semantic_sha256": canonical_sha256(unsigned)}


def _save_selected_npz(
    path: Path,
    x_train: sparse.csr_matrix,
    y_train: np.ndarray,
    x_tune: sparse.csr_matrix,
    y_tune: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                x_train_data=x_train.data,
                x_train_indices=x_train.indices,
                x_train_indptr=x_train.indptr,
                x_train_shape=np.asarray(x_train.shape, dtype=np.int64),
                x_tune_data=x_tune.data,
                x_tune_indices=x_tune.indices,
                x_tune_indptr=x_tune.indptr,
                x_tune_shape=np.asarray(x_tune.shape, dtype=np.int64),
                y_train=y_train,
                y_tune=y_tune,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o444)
        fsync_parent(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sparse_sha256(value: sparse.csr_matrix) -> str:
    matrix = value.tocsr()
    return canonical_sha256(
        {
            "shape": list(matrix.shape),
            "data": hashlib.sha256(np.ascontiguousarray(matrix.data).tobytes()).hexdigest(),
            "indices": hashlib.sha256(
                np.ascontiguousarray(matrix.indices).tobytes()
            ).hexdigest(),
            "indptr": hashlib.sha256(
                np.ascontiguousarray(matrix.indptr).tobytes()
            ).hexdigest(),
        }
    )


def _target_free_contract(
    *, qm9_csv: Path, feature_matrix: Path, feature_manifest: Path, phase1_dir: Path
) -> tuple[dict[str, object], sparse.csr_matrix, object]:
    """Authenticate and freeze sample/split/plan without loading any target column."""

    repository = Path(_repository_provenance()["repository_root"])
    with (repository / "configs/qm9_28m.toml").open("rb") as handle:
        phase1_config = tomllib.load(handle)
    identities = load_qm9_identities(qm9_csv)
    evidence = verify_phase1_evidence(phase1_config, phase1_dir, qm9_csv)
    matrix, manifest, _ = authenticate_feature_artifact(
        feature_matrix,
        feature_manifest,
        expected_source_row_identity_sha256=identities.row_identity_sha256,
    )
    if (
        manifest["source_csv_sha256"] != file_sha256(qm9_csv)
        or manifest["source_smiles_sha256"] != identities.raw_smiles_sha256
    ):
        raise SupervisorError("medium feature/source identity differs")
    fixed_train = np.asarray(evidence.split.train, dtype=np.int64)
    all_groups = canonical_connectivity_groups(identities.source_smiles)
    selected_positions = stable_group_sample(all_groups[fixed_train])
    selected_source = fixed_train[selected_positions]
    selected_groups = all_groups[selected_source]
    train, tune = medium_split(selected_groups)
    selected_matrix = matrix[selected_source].tocsr()
    sample = sample_evidence(selected_source, selected_groups)
    plan = planned_trials(len(tune))
    unsigned = {
        "schema_version": "qm9-v31-medium-target-free-plan-v1",
        "run_id": RUN_ID,
        "selected_source_indices": selected_source.tolist(),
        "train_positions": train.tolist(),
        "tune_positions": tune.tolist(),
        "sample": sample,
        "selected_feature_sha256": _sparse_sha256(selected_matrix),
        "train_feature_sha256": _sparse_sha256(selected_matrix[train].tocsr()),
        "tune_feature_sha256": _sparse_sha256(selected_matrix[tune].tocsr()),
        "plan": [row.payload() for row in plan],
        "plan_sha256": canonical_sha256([row.payload() for row in plan]),
        "input_inventory_sha256": canonical_sha256(
            {
                "qm9_csv": file_sha256(qm9_csv),
                "feature_matrix": file_sha256(feature_matrix),
                "feature_manifest": file_sha256(feature_manifest),
                "phase1_tree": canonical_sha256(_tree_manifest(phase1_dir)),
            }
        ),
        "label_reads": _label_reads(),
        **_false_boundaries(),
    }
    return {**unsigned, "semantic_sha256": canonical_sha256(unsigned)}, selected_matrix, identities


def load_authenticated_selected_train(
    base: Path, prepared: Mapping[str, object]
) -> tuple[Path, dict[str, object]]:
    """Authenticate inputs, sample without labels, then read selected train labels once."""

    binding = prepared.get("input_binding")
    if not isinstance(binding, Mapping):
        raise SupervisorError("medium prepared input binding is missing")
    if PUBLIC_FIXED_MODE:
        from .model_selection_v31_public_fixed import validate_bundle

        bundle_dir = Path(str(binding["public_fixed_bundle"]["path"]))
        manifest = validate_bundle(bundle_dir)
        if (
            file_sha256(bundle_dir / "MANIFEST.json")
            != binding["public_fixed_bundle"]["manifest_sha256"]
            or manifest["bundle_file_sha256"]
            != binding["public_fixed_bundle"]["bundle_sha256"]
        ):
            raise SupervisorError("public fixed bundle changed after review")
        data_path = Path(str(binding["public_fixed_adapter"]["path"]))
        if (
            data_path.is_symlink()
            or not data_path.is_file()
            or file_sha256(data_path) != binding["public_fixed_adapter"]["sha256"]
        ):
            raise SupervisorError("public fixed worker adapter changed after review")
        label_record = {
            "schema_version": "qm9-v31-public-fixed-label-read-v1",
            "run_id": RUN_ID,
            "label_reads": _label_reads(),
            "synthetic_target_artifact_materializations_before_review": 1,
            **_false_boundaries(),
        }
        _write_immutable(base / "LABEL_READS.json", label_record)
        sample = dict(manifest["sample"])
        sample.update(
            {
                "data_file_sha256": file_sha256(data_path),
                "label_read_file_sha256": file_sha256(base / "LABEL_READS.json"),
            }
        )
        sample["semantic_sha256"] = canonical_sha256(
            {key: value for key, value in sample.items() if key != "semantic_sha256"}
        )
        _write_immutable(
            base / "SAMPLE.json",
            {
                "schema_version": "qm9-v31-medium-sample-v1",
                "run_id": RUN_ID,
                "sample": sample,
                **_false_boundaries(),
            },
        )
        return data_path, sample
    paths = {name: Path(str(binding[name]["path"])) for name in binding}
    for name in ("qm9_csv", "feature_matrix", "feature_manifest"):
        path = paths[name]
        if path.is_symlink() or not path.is_file() or file_sha256(path) != binding[name]["sha256"]:
            raise SupervisorError(f"medium authenticated input changed: {name}")
    phase1 = paths["phase1_dir"]
    if phase1.is_symlink() or not phase1.is_dir():
        raise SupervisorError("medium phase-1 input changed")
    observed_phase1 = _tree_manifest(phase1)
    if canonical_sha256(observed_phase1) != binding["phase1_dir"]["tree_sha256"]:
        raise SupervisorError("medium phase-1 tree changed")
    replayed, selected_matrix, identities = _target_free_contract(
        qm9_csv=paths["qm9_csv"],
        feature_matrix=paths["feature_matrix"],
        feature_manifest=paths["feature_manifest"],
        phase1_dir=phase1,
    )
    frozen_path = base / "TARGET_FREE_PLAN.json"
    frozen = _load_artifact(frozen_path, "qm9-v31-medium-target-free-plan-v1")
    if replayed != frozen:
        raise SupervisorError("medium target-free sample/split/plan replay differs")
    selected_source = np.asarray(frozen["selected_source_indices"], dtype=np.int64)
    train = np.asarray(frozen["train_positions"], dtype=np.int64)
    tune = np.asarray(frozen["tune_positions"], dtype=np.int64)
    selected_targets = load_targets_for_indices(
        paths["qm9_csv"], selected_source, identities
    )
    label_record = {
        "schema_version": "qm9-v31-medium-label-read-v1",
        "run_id": RUN_ID,
        "label_reads": {
            "source_fixed_train_target_loads": 1,
            "derived_train_target_artifact_reads": 0,
            "outer_validation": 0,
            "test": 0,
            "protected": 0,
            "private_evaluation": 0,
        },
        **_false_boundaries(),
    }
    _write_immutable(base / "LABEL_READS.json", label_record)
    data_path = base / "selected/fixed-train-selected.npz"
    _save_selected_npz(
        data_path,
        selected_matrix[train].tocsr(),
        selected_targets[train],
        selected_matrix[tune].tocsr(),
        selected_targets[tune],
    )
    sample = dict(frozen["sample"])
    sample.update(
        {
            "data_file_sha256": file_sha256(data_path),
            "train_rows": len(train),
            "tune_rows": len(tune),
            "label_read_file_sha256": file_sha256(base / "LABEL_READS.json"),
        }
    )
    sample["semantic_sha256"] = canonical_sha256(
        {key: value for key, value in sample.items() if key != "semantic_sha256"}
    )
    _write_immutable(base / "SAMPLE.json", {
        "schema_version": "qm9-v31-medium-sample-v1", "run_id": RUN_ID,
        "sample": sample, **_false_boundaries(),
    })
    return data_path, sample


def planned_trials(tune_rows: int) -> list[PlannedTrial]:
    plan = []
    for index, candidate in enumerate(candidates(), start=1):
        identity = TrialIdentity(
            run_id=RUN_ID,
            wave="MEDIUM_PROFILE",
            strategy="identity",
            candidate_id=str(candidate["id"]),
            family=str(candidate["family"]),
            fold=0,
            seed=SEED,
            trial_index=index,
            trial_total=3,
        )
        plan.append(
            PlannedTrial(
                identity=identity,
                expected_shape=(tune_rows, len(TARGET_COLUMNS)),
                target_order=tuple(TARGET_COLUMNS),
                prediction_order_sha256=canonical_sha256(list(range(tune_rows))),
                candidate_sha256=canonical_sha256(candidate),
            )
        )
    return plan


def _outer_envelope_for(base: Path) -> Path:
    resolved = base.resolve()
    if resolved == PUBLIC_SYNTHETIC_BASE.resolve():
        return PUBLIC_SYNTHETIC_OUTER_ENVELOPE
    if resolved == PUBLIC_FIXED_BASE.resolve():
        return PUBLIC_FIXED_OUTER_ENVELOPE
    if resolved == PRODUCTION_EXACT_BASE.resolve():
        return OUTER_ENVELOPE
    return resolved.parent / f"{resolved.name}-launch-envelope-v2"


def outer_wrapper_contract(
    base: Path,
    repository: Path,
    *,
    extra_bound_files: Sequence[Path] = (),
) -> dict[str, object]:
    """Freeze the complete outer harness invocation, with one explicit auth-hash slot."""

    script = repository / (
        "scripts/run_qm9_model_selection_v31_public_fixed.py"
        if PUBLIC_FIXED_MODE
        else "scripts/run_qm9_model_selection_v31_medium_profile.py"
    )
    harness = repository / "scripts/run_bounded_process.py"
    protocol = repository / "docs/qm9_model_selection_v31_protocol.md"
    lock = repository / "uv.lock"
    envelope = _outer_envelope_for(base)
    contained = [
        sys.executable,
        str(script.resolve()),
        "--stage",
        "execute-contained",
        "--output",
        str(base.resolve()),
    ]
    authorization_path = str((base / "AUTHORIZED.json").resolve())
    authorization_token = "{AUTHORIZED_FILE_SHA256}"
    argv_template = [
        sys.executable,
        str(harness.resolve()),
        "--profile",
        "public-fixed-v1" if PUBLIC_FIXED_MODE else "medium-v1",
        "--status-jsonl",
        str((envelope / "status.jsonl").resolve()),
        "--result-json",
        str((envelope / "result.json").resolve()),
        "--cwd",
        str(repository.resolve()),
        "--wall-seconds",
        str(OUTER_SECONDS),
        "--rss-bytes",
        str(RSS_HARD_BYTES),
        "--command-seconds",
        str(OUTER_COMMAND_SECONDS),
        "--term-grace-seconds",
        str(FINALIZE_SECONDS),
        "--expected-command-sha256",
        canonical_sha256(contained),
        "--expected-harness-sha256",
        file_sha256(harness),
        "--expected-code-state-sha256",
        "{CODE_STATE_SHA256}",
        "--expected-cwd",
        str(repository.resolve()),
        "--expected-profile",
        authorization_path,
        "--expected-profile-sha256",
        authorization_token,
        "--expected-bound-file",
        f"{authorization_path}={authorization_token}",
        "--expected-bound-file",
        f"{script.resolve()}={file_sha256(script)}",
        "--expected-bound-file",
        f"{protocol.resolve()}={file_sha256(protocol)}",
        "--expected-bound-file",
        f"{lock.resolve()}={file_sha256(lock)}",
    ]
    for path in extra_bound_files:
        resolved = path.resolve()
        argv_template.extend(
            ["--expected-bound-file", f"{resolved}={file_sha256(resolved)}"]
        )
    argv_template.extend(["--", *contained])
    parameters = {
        "profile": "public-fixed-v1" if PUBLIC_FIXED_MODE else "medium-v1",
        "wall_seconds": OUTER_SECONDS,
        "memory_max_bytes": RSS_HARD_BYTES,
        "command_seconds": OUTER_COMMAND_SECONDS,
        "term_grace_seconds": FINALIZE_SECONDS,
        "status_path": str((envelope / "status.jsonl").resolve()),
        "result_path": str((envelope / "result.json").resolve()),
        "envelope": str(envelope.resolve()),
    }
    return {
        "argv_template": argv_template,
        "argv_template_sha256": canonical_sha256(argv_template),
        "authorization_sha256_token": authorization_token,
        "code_state_sha256_token": "{CODE_STATE_SHA256}",
        "contained_argv": contained,
        "contained_argv_sha256": canonical_sha256(contained),
        "parameters": parameters,
        "parameters_sha256": canonical_sha256(parameters),
    }


def materialize_outer_wrapper_argv(
    contract: Mapping[str, object], *, authorization_sha256: str, code_state_sha256: str
) -> list[str]:
    if any(
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for value in (authorization_sha256, code_state_sha256)
    ):
        raise SupervisorError("medium outer materialization digest differs")
    return [
        str(value)
        .replace(str(contract["authorization_sha256_token"]), authorization_sha256)
        .replace(str(contract["code_state_sha256_token"]), code_state_sha256)
        for value in contract["argv_template"]
    ]


def execution_contract(
    base: Path,
    repository: Path,
    *,
    extra_bound_files: Sequence[Path] = (),
) -> dict[str, object]:
    script = repository / (
        "scripts/run_qm9_model_selection_v31_public_fixed.py"
        if PUBLIC_FIXED_MODE
        else "scripts/run_qm9_model_selection_v31_medium_profile.py"
    )
    harness = repository / "scripts/run_bounded_process.py"
    protocol = repository / "docs/qm9_model_selection_v31_protocol.md"
    lock = repository / "uv.lock"
    argv = [
        sys.executable,
        str(script.resolve()),
        "--stage",
        "execute-contained",
        "--output",
        str(base.resolve()),
    ]
    wrapper = outer_wrapper_contract(
        base, repository, extra_bound_files=extra_bound_files
    )
    return {
        "argv": argv,
        "argv_sha256": canonical_sha256(argv),
        "cwd": str(repository.resolve()),
        "output": str(base.resolve()),
        "harness_path": str(harness.resolve()),
        "harness_sha256": file_sha256(harness),
        "script_path": str(script.resolve()),
        "script_sha256": file_sha256(script),
        "protocol_path": str(protocol.resolve()),
        "protocol_sha256": file_sha256(protocol),
        "lock_path": str(lock.resolve()),
        "lock_sha256": file_sha256(lock),
        "outer_wrapper": wrapper,
        "outer_wrapper_sha256": canonical_sha256(wrapper),
        "extra_bound_files": [str(path.resolve()) for path in extra_bound_files],
    }


def validate_supervisor_authorization(
    binding: MediumProfileRunBinding, planned: Sequence[PlannedTrial]
) -> dict[str, object]:
    """Re-authenticate the narrow bootstrap authorization at every reservation."""

    binding.validate()
    authorization_path = Path(binding.authorization_path)
    authorization = _load_artifact(authorization_path, AUTH_SCHEMA)
    prepared = _load_artifact(authorization_path.parent / "PREPARED.json", PREPARE_SCHEMA)
    provenance = prepared["repository_provenance"]
    decision = authorization["decision_bindings"]
    if (
        file_sha256(authorization_path) != binding.authorization_file_sha256
        or authorization.get("semantic_sha256") != binding.authorization_semantic_sha256
        or datetime.now(UTC) >= datetime.fromisoformat(str(authorization["expires_at_utc"]))
        or canonical_sha256([row.payload() for row in planned]) != binding.plan_sha256
        or canonical_sha256(dict(binding.device_identity)) != binding.device_identity_sha256
        or prepared["config_file_sha256"] != binding.config_sha256
        or provenance["manifest_sha256"] != binding.code_sha256
        or provenance["uv_lock_sha256"] != binding.dependency_sha256
        or authorization.get("nonce") != binding.authorization_nonce
        or tuple(decision["argv"]) != binding.authorized_argv
        or decision["cwd"] != binding.authorized_cwd
        or decision["output"] != binding.authorized_output
        or decision["input_sha256"] != binding.authorized_input_sha256
        or decision["sample_sha256"] != binding.authorized_sample_sha256
        or decision["plan_sha256"] != binding.authorized_plan_sha256
        or decision["budget_sha256"] != binding.authorized_budget_sha256
        or prepared["sample_sha256"] != binding.authorized_sample_sha256
        or prepared["target_free_input_inventory_sha256"]
        != binding.authorized_input_sha256
        or prepared["plan_sha256"] != binding.authorized_plan_sha256
    ):
        raise SupervisorError("medium supervisor authorization binding differs")
    return authorization


def build_supervisor_binding(
    base: Path,
    *,
    data_path: Path,
    sample: Mapping[str, object],
    plan: Sequence[PlannedTrial],
) -> MediumProfileRunBinding:
    prepared = _load_artifact(base / "PREPARED.json", PREPARE_SCHEMA)
    authorization = _load_artifact(base / "AUTHORIZED.json", AUTH_SCHEMA)
    provenance = prepared["repository_provenance"]
    decision_bindings = authorization["decision_bindings"]
    current_provenance = _repository_provenance()
    current_device = detect_device_identity("cpu")
    if (
        current_provenance["provenance_sha256"] != provenance["provenance_sha256"]
        or current_device != prepared["device_identity"]
    ):
        raise SupervisorError("medium execution code/device provenance changed after review")
    return MediumProfileRunBinding(
        run_id=RUN_ID,
        run_purpose=SUPERVISOR_PURPOSE,
        plan_sha256=canonical_sha256([row.payload() for row in plan]),
        config_sha256=str(prepared["config_file_sha256"]),
        input_sha256=file_sha256(data_path),
        profile_sha256=str(sample["semantic_sha256"]),
        code_sha256=str(provenance["manifest_sha256"]),
        dependency_sha256=str(provenance["uv_lock_sha256"]),
        authorization_path=str((base / "AUTHORIZED.json").resolve()),
        authorization_file_sha256=file_sha256(base / "AUTHORIZED.json"),
        authorization_semantic_sha256=str(authorization["semantic_sha256"]),
        authorization_nonce=str(authorization["nonce"]),
        authorized_argv=tuple(decision_bindings["argv"]),
        authorized_cwd=str(decision_bindings["cwd"]),
        authorized_output=str(decision_bindings["output"]),
        authorized_input_sha256=str(decision_bindings["input_sha256"]),
        authorized_sample_sha256=str(decision_bindings["sample_sha256"]),
        authorized_plan_sha256=str(decision_bindings["plan_sha256"]),
        authorized_budget_sha256=str(decision_bindings["budget_sha256"]),
        device_identity=prepared["device_identity"],
        device_identity_sha256=str(prepared["device_identity_sha256"]),
    )


def run_real_train_only_workload(
    base: Path, prepared: Mapping[str, object]
) -> dict[str, object]:
    """Run the three fixed candidates through external workers and strict replay."""

    from .model_selection_v31_medium_worker import validate_profile, write_spec

    set_fixed_cpu_threads()
    data_path, sample = load_authenticated_selected_train(base, prepared)
    plan = planned_trials(int(sample["tune_rows"]))
    binding = build_supervisor_binding(base, data_path=data_path, sample=sample, plan=plan)
    supervisor = RunSupervisor.for_medium_profile(base / "run", binding, plan)
    started = time.monotonic()
    profiles: list[dict[str, object]] = []
    cgroup_peaks: list[int] = []
    try:
        worker_data_path = supervisor.root / "input/fixed-train-selected.npz"
        worker_data_path.parent.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(data_path, worker_data_path)
        worker_data_path.chmod(0o444)
        fsync_parent(worker_data_path)
        if file_sha256(worker_data_path) != binding.input_sha256:
            raise SupervisorError("medium worker input copy binding differs")
        for specification, candidate in zip(plan, candidates(), strict=True):
            elapsed = time.monotonic() - started
            family_budget = FAMILY_SOFT_SECONDS[str(candidate["family"])]
            if elapsed + family_budget > WORKLOAD_SECONDS:
                raise SupervisorError("medium family cannot start before the 2280-second drain")
            soft_remaining = min(float(family_budget), WORKLOAD_SECONDS - elapsed)
            hard_remaining = OUTER_SECONDS - elapsed
            if soft_remaining <= 1 or hard_remaining <= soft_remaining:
                raise SupervisorError("medium workload lacks safe drain budget")
            trial = specification.identity
            supervisor.reserve(trial)
            supervisor.start_trial(trial)
            completion = supervisor.completion_contract(trial)
            worker_root = supervisor.root
            spec_path = worker_root / f"specs/trial-{trial.trial_index}.json"
            result_path = worker_root / f"worker/trial-{trial.trial_index}.result.json"
            profile_path = worker_root / f"profiles/trial-{trial.trial_index}.json"
            telemetry_path = worker_root / f"telemetry/trial-{trial.trial_index}.jsonl"
            checkpoint_bindings = {
                "plan_sha256": binding.plan_sha256,
                "config_sha256": binding.config_sha256,
                "input_sha256": binding.input_sha256,
                "cache_sha256": binding.profile_sha256,
                "run_start_sha256": supervisor.run_start_sha256,
            }
            write_spec(
                spec_path,
                run_root=worker_root,
                data_path=worker_data_path,
                result_path=result_path,
                profile_path=profile_path,
                telemetry_path=telemetry_path,
                checkpoint_root=supervisor.checkpoints.root,
                candidate=candidate,
                trial=trial,
                completion=completion,
                checkpoint_bindings=checkpoint_bindings,
            )
            policy = SidecarPolicy(
                heartbeat_seconds=10,
                heartbeat_stale_after_seconds=20,
                model_progress_stale_after_seconds=60,
                model_progress_stale_overrides={
                    "*:data_load_start": 60,
                    "*:recipe_transform_start": 60,
                    "ridge:*": 170,
                    "xgboost:*": 600,
                    "mlp:*": 120,
                },
                soft_deadline_seconds=soft_remaining,
                hard_deadline_seconds=hard_remaining,
                rss_max_bytes=RSS_HARD_BYTES,
                disk_free_min_bytes=5 * 1024**3,
                gpu_allocated_max_bytes=1,
                gpu_reserved_max_bytes=1,
                gpu_temperature_max_c=1,
                require_rss_sensor=True,
                require_disk_sensor=True,
                require_gpu_sensors=False,
                poll_seconds=0.5,
            )
            worker_command = [
                sys.executable,
                "-m",
                "mist_transfer_benchmark.qm9.model_selection_v31_medium_worker",
                "--worker-spec",
                str(spec_path),
            ]
            repository = Path(str(prepared["repository_provenance"]["repository_root"]))
            harness = repository / "scripts/run_bounded_process.py"
            envelope_status = worker_root / f"envelope/trial-{trial.trial_index}.jsonl"
            envelope_result = worker_root / f"envelope/trial-{trial.trial_index}.result.json"
            command = worker_command
            if not PUBLIC_FIXED_MODE:
                command = [
                    sys.executable,
                    str(harness),
                    "--profile", "medium-v1",
                    "--status-jsonl", str(envelope_status),
                    "--result-json", str(envelope_result),
                    "--cwd", str(repository),
                    "--wall-seconds", str(OUTER_SECONDS),
                    "--rss-bytes", str(RSS_HARD_BYTES),
                    "--command-seconds", str(OUTER_COMMAND_SECONDS),
                    "--term-grace-seconds", str(FINALIZE_SECONDS),
                    "--expected-command-sha256", canonical_sha256(worker_command),
                    "--expected-harness-sha256", file_sha256(harness),
                    "--expected-cwd", str(repository),
                    "--expected-profile", str(base / "AUTHORIZED.json"),
                    "--expected-profile-sha256", binding.authorization_file_sha256,
                    "--expected-bound-file", f"{spec_path}={file_sha256(spec_path)}",
                    "--expected-bound-file", f"{worker_data_path}={file_sha256(worker_data_path)}",
                    "--",
                    *worker_command,
                ]
            code = ExternalWorkerSidecar(supervisor, policy).run(
                trial,
                command,
                heartbeat_path=worker_root
                / f"telemetry/trial-{trial.trial_index}.heartbeat",
                environment={name: str(THREADS) for name in (
                    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                )} | ({"QM9_V31_PUBLIC_FIXED_BACKENDS": "1"} if PUBLIC_FIXED_MODE else {}),
                prepared=True,
                result_path=result_path,
            )
            if code != 0:
                raise SupervisorError(f"medium external worker failed: {trial.family}")
            worker_result = json.loads(result_path.read_text())
            validate_profile(
                profile_path,
                trial=trial,
                expected_metadata_sha256=str(worker_result["metadata_sha256"]),
            )
            if not PUBLIC_FIXED_MODE:
                envelope = json.loads(envelope_result.read_text())
                cgroup_peak = envelope.get("cgroup", {}).get("memory_peak_bytes")
                if (
                    envelope.get("state") != "complete"
                    or envelope.get("cgroup", {}).get("limit_verified") is not True
                    or envelope.get("cgroup_empty") is not True
                    or not isinstance(cgroup_peak, int)
                ):
                    raise SupervisorError("medium cgroup completion evidence differs")
                cgroup_peaks.append(cgroup_peak)
            profiles.append(
                {
                    "trial": trial_payload(trial),
                    "profile_path": str(profile_path.relative_to(worker_root)),
                    "profile_file_sha256": file_sha256(profile_path),
                    "family_soft_budget_seconds": family_budget,
                    "cumulative_elapsed_seconds": time.monotonic() - started,
                }
            )
            if _artifact_bytes(base) > ARTIFACT_MAX_BYTES:
                raise SupervisorError("medium artifacts exceeded the 1-GiB cap")
        supervisor.close_ledger()
        final = supervisor.finalize_profile_success(
            {"profiles": profiles, "label_reads": _successful_label_reads()}
        )
        event_rows = validate_event_stream(supervisor.events.path)
        if _artifact_bytes(base) > ARTIFACT_MAX_BYTES:
            raise SupervisorError("medium finalized artifacts exceeded the 1-GiB cap")
        sidecar_peak = max(
            (int(row.get("resources", {}).get("rss_bytes") or 0) for row in event_rows), default=0
        )
        peak = max([sidecar_peak, *cgroup_peaks])
        return {
            "peak_rss_bytes": peak,
            "label_reads": _successful_label_reads(),
            "trial_states": {row.identity.family: "complete" for row in plan},
            "supervisor_final_sha256": file_sha256(final),
        }
    except BaseException as error:
        if not getattr(supervisor, "_terminal", False):
            supervisor.finalize_failure(
                {"reason": "MEDIUM_WORKLOAD_FAILURE", "error_type": type(error).__name__}
            )
        raise
    finally:
        supervisor.close()


def prepare(
    output: str | Path,
    *,
    runner_identity: str,
    qm9_csv: Path | None = None,
    feature_matrix: Path | None = None,
    feature_manifest: Path | None = None,
    phase1_dir: Path | None = None,
    public_fixture_plan: Path | None = None,
    public_fixed_bundle: Path | None = None,
) -> Path:
    """Bind paths and bytes only; preparation does not read any target arrays."""

    base = _require_exact_base(output)
    if not runner_identity:
        raise SupervisorError("runner identity must be nonempty")
    if shutil.disk_usage(base.parent).free < DISK_FREE_MIN_BYTES:
        raise SupervisorError("medium prepare requires at least 5 GiB free disk")
    base.mkdir(exist_ok=False)
    fsync_parent(base)
    claim_path = base / "PREPARE_CLAIM.json"
    try:
        claim = _write_immutable(
            claim_path,
            {
                "schema_version": PREPARE_CLAIM_SCHEMA,
                "run_id": RUN_ID,
                "purpose": PURPOSE,
                "runner_identity": runner_identity,
                "claimed_at_utc": datetime.now(UTC).isoformat(),
                "base": str(base),
                **_false_boundaries(),
            },
        )
        if PUBLIC_FIXED_MODE:
            if public_fixed_bundle is None or any(
                value is not None
                for value in (qm9_csv, feature_matrix, feature_manifest, phase1_dir)
            ):
                raise SupervisorError("public fixed prepare requires only its frozen bundle")
            from .model_selection_v31_public_fixed import validate_bundle

            public_manifest = validate_bundle(public_fixed_bundle)
        else:
            if any(
                path is None or path.is_symlink() or not path.is_file()
                for path in (qm9_csv, feature_matrix, feature_manifest)
            ):
                raise SupervisorError("medium profile input path is missing or symlinked")
            if phase1_dir is None or phase1_dir.is_symlink() or not phase1_dir.is_dir():
                raise SupervisorError("medium phase-1 input must be a regular directory")
        provenance = _repository_provenance()
        config_path = base / "config.json"
        atomic_json(config_path, frozen_config())
        if PUBLIC_FIXED_MODE:
            bundle_path = public_fixed_bundle / str(public_manifest["bundle_file"])
            with np.load(bundle_path, allow_pickle=False) as archive:
                matrix = sparse.csr_matrix(
                    (archive["data"], archive["indices"], archive["indptr"]),
                    shape=tuple(int(value) for value in archive["shape"]),
                )
                targets = archive["targets"]
                train = archive["train_positions"].astype(np.int64)
                tune = archive["tune_positions"].astype(np.int64)
            adapter_path = base / "selected/public-fixed-selected.npz"
            _save_selected_npz(
                adapter_path,
                matrix[train].tocsr(),
                targets[train],
                matrix[tune].tocsr(),
                targets[tune],
            )
            adapter_path.chmod(0o444)
            plan = planned_trials(len(tune))
            plan_payload = [row.payload() for row in plan]
            if (
                public_manifest["plan"] != plan_payload
                or public_manifest["plan_sha256"] != canonical_sha256(plan_payload)
            ):
                raise SupervisorError("public fixed bundle/runtime plan differs")
            target_free_unsigned = {
                "schema_version": "qm9-v31-medium-target-free-plan-v1",
                "run_id": RUN_ID,
                "selected_source_indices": list(range(int(public_manifest["dimensions"]["rows"]))),
                "train_positions": train.tolist(),
                "tune_positions": tune.tolist(),
                "sample": public_manifest["sample"],
                "selected_feature_sha256": public_manifest["feature_contract"]["matrix"],
                "train_feature_sha256": _sparse_sha256(matrix[train].tocsr()),
                "tune_feature_sha256": _sparse_sha256(matrix[tune].tocsr()),
                "plan": plan_payload,
                "plan_sha256": canonical_sha256(plan_payload),
                "input_inventory_sha256": canonical_sha256(
                    {
                        "manifest": file_sha256(public_fixed_bundle / "MANIFEST.json"),
                        "bundle": file_sha256(bundle_path),
                    }
                ),
                "label_reads": _label_reads(),
                "synthetic_targets_materialized_before_review": True,
                **_false_boundaries(),
            }
            target_free = {
                **target_free_unsigned,
                "semantic_sha256": canonical_sha256(target_free_unsigned),
            }
            phase1_tree = []
        else:
            assert feature_manifest is not None and phase1_dir is not None
            manifest = json.loads(feature_manifest.read_text())
            if manifest.get("schema_version") != "qm9-paper-feature-artifact-v1":
                raise SupervisorError("medium feature manifest schema differs")
            phase1_tree = _tree_manifest(phase1_dir)
        if public_fixture_plan is not None:
            if (
                not PUBLIC_SYNTHETIC_MODE
                or base.resolve() != PUBLIC_SYNTHETIC_BASE.resolve()
            ):
                raise SupervisorError("public fixture plan is forbidden outside public seam")
            target_free = _load_artifact(
                public_fixture_plan, "qm9-v31-medium-target-free-plan-v1"
            )
            if (
                target_free.get("run_id") != PUBLIC_SYNTHETIC_RUN_ID
                or target_free.get("label_reads") != _label_reads()
                or any(target_free.get(key) is not False for key in _false_boundaries())
            ):
                raise SupervisorError("public fixture target-free scope differs")
        elif not PUBLIC_FIXED_MODE:
            assert all(
                value is not None
                for value in (qm9_csv, feature_matrix, feature_manifest, phase1_dir)
            )
            target_free, _, _ = _target_free_contract(
                qm9_csv=qm9_csv,
                feature_matrix=feature_matrix,
                feature_manifest=feature_manifest,
                phase1_dir=phase1_dir,
            )
        target_free_unsigned = dict(target_free)
        target_free_unsigned.pop("semantic_sha256")
        target_free_path = base / "TARGET_FREE_PLAN.json"
        persisted_target_free = _write_immutable(target_free_path, target_free_unsigned)
        if PUBLIC_FIXED_MODE:
            input_binding = {
                "public_fixed_bundle": {
                    "path": str(public_fixed_bundle.resolve()),
                    "manifest_sha256": file_sha256(public_fixed_bundle / "MANIFEST.json"),
                    "bundle_sha256": public_manifest["bundle_file_sha256"],
                },
                "public_fixed_adapter": {
                    "path": str(adapter_path.resolve()),
                    "sha256": file_sha256(adapter_path),
                },
            }
        else:
            assert all(
                value is not None
                for value in (qm9_csv, feature_matrix, feature_manifest, phase1_dir)
            )
            input_binding = {
                "qm9_csv": {"path": str(qm9_csv.resolve()), "sha256": file_sha256(qm9_csv)},
                "feature_matrix": {
                    "path": str(feature_matrix.resolve()),
                    "sha256": file_sha256(feature_matrix),
                },
                "feature_manifest": {
                    "path": str(feature_manifest.resolve()),
                    "sha256": file_sha256(feature_manifest),
                },
                "phase1_dir": {
                    "path": str(phase1_dir.resolve()),
                    "tree_sha256": canonical_sha256(phase1_tree),
                    "tree": phase1_tree,
                },
            }
        if public_fixture_plan is not None:
            input_binding["public_fixture_plan"] = {
                "path": str(public_fixture_plan.resolve()),
                "sha256": file_sha256(public_fixture_plan),
            }
        device = detect_device_identity("cpu")
        extra_bound_files: tuple[Path, ...] = ()
        if PUBLIC_FIXED_MODE:
            repository_root = Path(provenance["repository_root"])
            extra_bound_files = (
                public_fixed_bundle / "MANIFEST.json",
                public_fixed_bundle / str(public_manifest["bundle_file"]),
                adapter_path,
                repository_root
                / "src/mist_transfer_benchmark/qm9/model_selection_v31_public_fixed.py",
                repository_root
                / "src/mist_transfer_benchmark/qm9/model_selection_v31_medium_worker.py",
                repository_root
                / "src/mist_transfer_benchmark/qm9/model_selection_v31_supervisor.py",
            )
        execute_contract = execution_contract(
            base,
            Path(provenance["repository_root"]),
            extra_bound_files=extra_bound_files,
        )
        preterminal_tree = _tree_manifest(base)
        _write_immutable(
            base / "PREPARED.json",
            {
                "schema_version": PREPARE_SCHEMA,
                "run_id": RUN_ID,
                "purpose": PURPOSE,
                "runner_identity": runner_identity,
                "created_at_utc": datetime.now(UTC).isoformat(),
                "prepare_claim_file_sha256": file_sha256(claim_path),
                "prepare_claim_semantic_sha256": claim["semantic_sha256"],
                "config_path": str(config_path),
                "config_file_sha256": file_sha256(config_path),
                "config_semantic_sha256": canonical_sha256(frozen_config()),
                "sample_contract_sha256": canonical_sha256(frozen_config()["sample"]),
                "budget_contract_sha256": canonical_sha256(frozen_config()["limits"]),
                "target_free_plan_path": str(target_free_path),
                "target_free_plan_file_sha256": file_sha256(target_free_path),
                "target_free_plan_semantic_sha256": persisted_target_free["semantic_sha256"],
                "sample_sha256": persisted_target_free["sample"]["semantic_sha256"],
                "plan_sha256": persisted_target_free["plan_sha256"],
                "target_free_input_inventory_sha256": persisted_target_free[
                    "input_inventory_sha256"
                ],
                "execution_contract": execute_contract,
                "execution_contract_sha256": canonical_sha256(execute_contract),
                "input_binding": input_binding,
                "input_binding_sha256": canonical_sha256(input_binding),
                "repository_provenance": provenance,
                "device_identity": device,
                "device_identity_sha256": canonical_sha256(device),
                "preterminal_tree": preterminal_tree,
                "preterminal_tree_sha256": canonical_sha256(preterminal_tree),
                "label_reads": _label_reads(),
                **_false_boundaries(),
            },
        )
        return base / "PREPARED.json"
    except BaseException as error:
        failure_path = base / "PREPARE_FAILED.json"
        if not failure_path.exists():
            partial = _tree_manifest(base)
            _write_immutable(
                failure_path,
                {
                    "schema_version": PREPARE_FAILURE_SCHEMA,
                    "run_id": RUN_ID,
                    "purpose": PURPOSE,
                    "status": "PREPARE_FAILED",
                    "error_type": type(error).__name__,
                    "partial_tree": partial,
                    "partial_tree_sha256": canonical_sha256(partial),
                    "label_reads": _label_reads(),
                    **_false_boundaries(),
                },
            )
        raise


def review(output: str | Path, *, approval_path: Path, expected_approval_sha256: str) -> Path:
    base = _require_exact_base(output)
    prepared_path = base / "PREPARED.json"
    prepared = _load_artifact(prepared_path, PREPARE_SCHEMA)
    approval = _load_external_decision(
        approval_path,
        schema=EXTERNAL_REVIEW_SCHEMA,
        expected_sha256=expected_approval_sha256,
        expected_decision=REVIEW_DECISION,
    )
    reviewer_identity = approval.get("reviewer_identity")
    expected_bindings = _decision_bindings(prepared, nonce=str(approval.get("nonce")))
    if (
        not isinstance(reviewer_identity, str)
        or reviewer_identity == prepared["runner_identity"]
        or approval.get("prepare_file_sha256") != file_sha256(prepared_path)
        or approval.get("prepare_semantic_sha256") != prepared["semantic_sha256"]
        or any(approval.get(key) != value for key, value in expected_bindings.items())
    ):
        raise SupervisorError("medium review must be independent")
    _write_immutable(
        base / "REVIEWED.json",
        {
            "schema_version": REVIEW_SCHEMA,
            "run_id": RUN_ID,
            "purpose": PURPOSE,
            "reviewer_identity": reviewer_identity,
            **(
                {"reviewer_tool_identity": approval["reviewer_tool_identity"]}
                if PUBLIC_FIXED_MODE
                else {}
            ),
            "prepare_file_sha256": file_sha256(prepared_path),
            "external_approval_path": str(approval_path.resolve()),
            "external_approval_file_sha256": expected_approval_sha256,
            "external_approval_semantic_sha256": approval["semantic_sha256"],
            "nonce": approval["nonce"],
            "decision_bindings": expected_bindings,
            "decision_bindings_sha256": canonical_sha256(expected_bindings),
            "decision": REVIEW_DECISION,
            "reviewed_at_utc": datetime.now(UTC).isoformat(),
            **_false_boundaries(),
        },
    )
    return base / "REVIEWED.json"


def authorize(
    output: str | Path,
    *,
    authorization_path: Path,
    expected_authorization_sha256: str,
) -> Path:
    base = _require_exact_base(output)
    review_path = base / "REVIEWED.json"
    reviewed = _load_artifact(review_path, REVIEW_SCHEMA)
    prepared = _load_artifact(base / "PREPARED.json", PREPARE_SCHEMA)
    external = _load_external_decision(
        authorization_path,
        schema=EXTERNAL_AUTH_SCHEMA,
        expected_sha256=expected_authorization_sha256,
        expected_decision=AUTH_DECISION,
    )
    authorizer_identity = external.get("authorizer_identity")
    expected_bindings = _decision_bindings(prepared, nonce=str(external.get("nonce")))
    if (
        not isinstance(authorizer_identity, str)
        or authorizer_identity in {reviewed["reviewer_identity"], prepared["runner_identity"]}
        or external.get("review_file_sha256") != file_sha256(review_path)
        or external.get("review_semantic_sha256") != reviewed["semantic_sha256"]
        or external.get("reviewer_identity") != reviewed["reviewer_identity"]
        or external.get("nonce") != reviewed["nonce"]
        or any(external.get(key) != value for key, value in expected_bindings.items())
        or reviewed.get("decision_bindings_sha256") != canonical_sha256(expected_bindings)
    ):
        raise SupervisorError("medium authorization identity must be independent")
    now = datetime.now(UTC)
    _write_immutable(
        base / "AUTHORIZED.json",
        {
            "schema_version": AUTH_SCHEMA,
            "run_id": RUN_ID,
            "purpose": PURPOSE,
            "authorizer_identity": authorizer_identity,
            **(
                {"authorizer_tool_identity": external["authorizer_tool_identity"]}
                if PUBLIC_FIXED_MODE
                else {}
            ),
            "review_file_sha256": file_sha256(review_path),
            "external_authorization_path": str(authorization_path.resolve()),
            "external_authorization_file_sha256": expected_authorization_sha256,
            "external_authorization_semantic_sha256": external["semantic_sha256"],
            "issued_at_utc": now.isoformat(),
            "expires_at_utc": (now + timedelta(minutes=30)).isoformat(),
            "one_use": True,
            "nonce": external["nonce"],
            "decision_bindings": expected_bindings,
            "decision_bindings_sha256": canonical_sha256(expected_bindings),
            **_false_boundaries(),
        },
    )
    return base / "AUTHORIZED.json"


def execute(
    output: str | Path,
    *,
    expected_authorization_sha256: str,
    workload: Callable[[Path, Mapping[str, object]], Mapping[str, object]],
) -> Path:
    """Consume one authorization and execute an injected governed workload.

    The production CLI supplies the private train-only workload. Tests inject a no-data fixture;
    there is deliberately no default that could execute merely by importing this module.
    """

    base = _require_exact_base(output)
    authorization_path = base / "AUTHORIZED.json"
    authorization = _load_artifact(authorization_path, AUTH_SCHEMA)
    prepared = _load_artifact(base / "PREPARED.json", PREPARE_SCHEMA)
    if file_sha256(authorization_path) != expected_authorization_sha256:
        raise SupervisorError("medium authorization out-of-band hash differs")
    if datetime.now(UTC) >= datetime.fromisoformat(str(authorization["expires_at_utc"])):
        raise SupervisorError("medium authorization expired")
    expected_bindings = _decision_bindings(prepared, nonce=str(authorization["nonce"]))
    repository = Path(str(prepared["repository_provenance"]["repository_root"]))
    if (
        authorization.get("decision_bindings") != expected_bindings
        or execution_contract(
            base,
            repository,
            extra_bound_files=tuple(
                Path(path) for path in prepared["execution_contract"]["extra_bound_files"]
            ),
        )
        != prepared["execution_contract"]
        or str(Path.cwd().resolve()) != expected_bindings["cwd"]
    ):
        raise SupervisorError("medium execution differs from exact reviewed binding")
    nonce_claim = base / f"NONCE-{authorization['nonce']}.claimed"
    _write_immutable(
        nonce_claim,
        {
            "schema_version": f"{SCHEMA_PREFIX}-nonce-claim-v1",
            "run_id": RUN_ID,
            "nonce": authorization["nonce"],
            "authorization_file_sha256": expected_authorization_sha256,
            "claimed_at_utc": datetime.now(UTC).isoformat(),
            **_false_boundaries(),
        },
    )
    consumed = base / "AUTHORIZATION_CONSUMED.json"
    _write_immutable(
        consumed,
        {
            "schema_version": f"{SCHEMA_PREFIX}-authorization-consumed-v1",
            "run_id": RUN_ID,
            "purpose": PURPOSE,
            "authorization_file_sha256": expected_authorization_sha256,
            "authorization_semantic_sha256": authorization["semantic_sha256"],
            "nonce": authorization["nonce"],
            "nonce_claim_file_sha256": file_sha256(nonce_claim),
            "decision_bindings": expected_bindings,
            "decision_bindings_sha256": canonical_sha256(expected_bindings),
            "consumed_at_utc": datetime.now(UTC).isoformat(),
            "label_reads": _label_reads(),
            **_false_boundaries(),
        },
    )
    started = time.monotonic()
    result: dict[str, object] = {}
    try:
        result = dict(workload(base, _load_artifact(base / "PREPARED.json", PREPARE_SCHEMA)))
        elapsed = time.monotonic() - started
        if elapsed > WORKLOAD_SECONDS:
            raise SupervisorError("medium workload exceeded 2280-second deadline")
        peak = int(result.get("peak_rss_bytes", RSS_HARD_BYTES + 1))
        if peak > RSS_SUCCESS_BYTES:
            raise SupervisorError("medium workload exceeded 3.2-GiB success ceiling")
        if result.get("label_reads") != _successful_label_reads():
            raise SupervisorError("medium workload label-read counters differ")
        tree = _tree_manifest(base)
        _write_immutable(
            base / "TERMINAL.json",
            {
                "schema_version": TERMINAL_SCHEMA,
                "run_id": RUN_ID,
                "purpose": PURPOSE,
                "status": "PROFILE_COMPLETE",
                "workload_seconds": elapsed,
                "peak_rss_bytes": peak,
                "authorization_consumed_sha256": file_sha256(consumed),
                "label_reads": result["label_reads"],
                "preterminal_tree": tree,
                "preterminal_tree_sha256": canonical_sha256(tree),
                "result": result,
                "terminal_at_utc": datetime.now(UTC).isoformat(),
                **_false_boundaries(),
            },
        )
        return base / "TERMINAL.json"
    except BaseException as error:
        terminal_path = base / "TERMINAL.json"
        if not terminal_path.exists():
            tree = _tree_manifest(base)
            observed_reads = _label_reads()
            label_path = base / "LABEL_READS.json"
            if label_path.is_file():
                try:
                    observed_reads = dict(json.loads(label_path.read_text())["label_reads"])
                except (OSError, KeyError, TypeError, json.JSONDecodeError):
                    pass
            derived = 0
            synthetic = 0
            for worker_result in (base / "run/worker").glob("trial-*.result.json"):
                try:
                    row = json.loads(worker_result.read_text())
                    if row.get("status") == "COMPLETE":
                        derived += int(
                            row.get("label_reads", {}).get(
                                "derived_train_target_artifact_reads", 0
                            )
                        )
                        synthetic += int(
                            row.get("label_reads", {}).get(
                                "synthetic_target_artifact_reads", 0
                            )
                        )
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    continue
            observed_reads["derived_train_target_artifact_reads"] = derived
            if PUBLIC_FIXED_MODE:
                observed_reads["synthetic_target_artifact_reads"] = synthetic
            _write_immutable(
                terminal_path,
                {
                    "schema_version": TERMINAL_SCHEMA,
                    "run_id": RUN_ID,
                    "purpose": PURPOSE,
                    "status": "PROFILE_FAILED",
                    "error_type": type(error).__name__,
                    "workload_seconds": time.monotonic() - started,
                    "authorization_consumed_sha256": file_sha256(consumed),
                    "label_reads": result.get("label_reads", observed_reads),
                    "preterminal_tree": tree,
                    "preterminal_tree_sha256": canonical_sha256(tree),
                    "terminal_at_utc": datetime.now(UTC).isoformat(),
                    **_false_boundaries(),
                },
            )
        raise


def _verify_public_fixed_workload(base: Path, terminal: Mapping[str, object]) -> dict[str, object]:
    """Independently replay all three public workers, ledgers, curves, and checkpoints."""

    if not PUBLIC_FIXED_MODE:
        raise SupervisorError("public fixed workload verifier is mode restricted")
    from .model_selection_v3 import monitor_curve
    from .model_selection_v31_medium_worker import validate_profile

    run_root = base / "run"
    run_start = json.loads((run_root / "run_start.json").read_text())
    final = json.loads((run_root / "final/manifest.json").read_text())
    ledger_path = run_root / "ledger/trials.jsonl"
    rows = TrialLedger.validate(ledger_path)
    complete_rows = [row for row in rows if row["event"] == "complete"]
    if (
        len(rows) != 9
        or [row["event"] for row in rows]
        != [event for _ in range(3) for event in ("reserve", "start", "complete")]
        or len(complete_rows) != 3
        or final.get("complete") is not True
        or final.get("purpose") != PURPOSE
        or final.get("private_data_allowed") is not False
        or final.get("ledger_sha256") != file_sha256(ledger_path)
        or terminal.get("result", {}).get("supervisor_final_sha256")
        != file_sha256(run_root / "final/manifest.json")
    ):
        raise SupervisorError("public fixed ledger/final contract differs")
    planned_payload = run_start.get("planned_trials")
    sample = json.loads((base / "SAMPLE.json").read_text())["sample"]
    plan = planned_trials(int(sample["tune_rows"]))
    if planned_payload != [row.payload() for row in plan]:
        raise SupervisorError("public fixed run-start plan replay differs")
    binding = run_start.get("binding")
    if not isinstance(binding, Mapping):
        raise SupervisorError("public fixed run-start binding is missing")
    store = CheckpointStore(
        run_root / "checkpoints",
        plan_sha256=str(binding["plan_sha256"]),
        config_sha256=str(binding["config_sha256"]),
        input_sha256=str(binding["input_sha256"]),
        cache_sha256=str(binding["profile_sha256"]),
        run_start_sha256=file_sha256(run_root / "run_start.json"),
    )
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
        planned_trials=tuple(trial_payload(row.identity) for row in plan),
        completed_entries=completed_entries,
        ledger_sha256=file_sha256(ledger_path),
        terminal_status=COMPLETION_STATUS,
    )
    replayed = store.load_completed(str(binding["plan_sha256"]), replay=replay)
    profile_rows = terminal.get("result", {}).get("profiles")
    if not isinstance(profile_rows, list) or len(profile_rows) != 3 or len(replayed) != 3:
        raise SupervisorError("public fixed profile/checkpoint count differs")
    profile_hashes: list[str] = []
    for planned, profile_row in zip(plan, profile_rows, strict=True):
        if profile_row.get("trial") != trial_payload(planned.identity):
            raise SupervisorError("public fixed profile order differs")
        profile_path = run_root / str(profile_row["profile_path"])
        if file_sha256(profile_path) != profile_row["profile_file_sha256"]:
            raise SupervisorError("public fixed profile physical hash differs")
        raw_profile = json.loads(profile_path.read_text())
        profile = validate_profile(
            profile_path,
            trial=planned.identity,
            expected_metadata_sha256=str(raw_profile["metadata_sha256"]),
        )
        profile_hashes.append(file_sha256(profile_path))
        event_path = run_root / f"telemetry/trial-{planned.identity.trial_index}.jsonl"
        events = validate_event_stream(event_path)
        if not events:
            raise SupervisorError("public fixed worker event stream is empty")
        if planned.identity.family == "mlp":
            metadata = profile["metadata"]
            curves = metadata["curves"]
            recomputed = monitor_curve(
                curves["train_standardized_mse"],
                curves["inner_tune_mnmae"],
                max_epochs=int(candidates()[2]["training"]["max_epochs"]),
                restored_hash=metadata["restored_checkpoint_sha256"],
                best_hash=metadata["best_checkpoint_sha256"],
                thresholds=candidates()[2]["monitoring"],
                per_target_nmae=curves["inner_tune_per_target_nmae"],
                gradient_norm=curves["gradient_norm"],
                parameter_norm=curves["parameter_norm"],
            )
            if (
                recomputed["warnings"] != metadata["curve"]["warnings"]
                or recomputed["anomalies"] != metadata["curve"]["anomalies"]
                or recomputed["status"] != metadata["curve"]["status"]
                or recomputed["maximum_consecutive_validation_increases"]
                != metadata["curve"]["maximum_consecutive_validation_increases"]
                or recomputed["anomalies"]
            ):
                raise SupervisorError("public fixed MLP raw-curve replay differs/abnormal")
    return {
        "ledger_rows": len(rows),
        "checkpoint_replay_count": len(replayed),
        "profile_file_sha256s": profile_hashes,
        "mlp_curve_recomputed": True,
    }


def _verify_inner_at(base: Path) -> dict[str, object]:
    terminal = _load_artifact(base / "TERMINAL.json", TERMINAL_SCHEMA)
    if terminal.get("status") != "PROFILE_COMPLETE":
        raise SupervisorError("medium terminal status differs")
    if terminal.get("label_reads") != _successful_label_reads():
        raise SupervisorError("medium terminal label counters differ")
    observed_tree = _tree_manifest(base, excluded={"TERMINAL.json", "POSTRUN.json"})
    if terminal.get("preterminal_tree") != observed_tree or terminal.get(
        "preterminal_tree_sha256"
    ) != canonical_sha256(observed_tree):
        raise SupervisorError("medium final exact tree differs")
    if PUBLIC_FIXED_MODE:
        _verify_public_fixed_workload(base, terminal)
    return terminal


def verify(output: str | Path) -> dict[str, object]:
    return _verify_inner_at(_require_exact_base(output))


def _parse_utc(value: object, *, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise SupervisorError(f"medium {name} timestamp differs") from error
    if parsed.tzinfo is None:
        raise SupervisorError(f"medium {name} timestamp differs")
    return parsed


def _replay_authorization_chain(
    base: Path,
    *,
    expected_authorization_sha256: str,
    external_review_source: Path | None = None,
    external_authorization_source: Path | None = None,
) -> dict[str, object]:
    """Replay the complete external decision through one-use terminal consumption."""

    false_keys = set(_false_boundaries())
    paths = {
        "PREPARED.json": base / "PREPARED.json",
        "REVIEWED.json": base / "REVIEWED.json",
        "AUTHORIZED.json": base / "AUTHORIZED.json",
        "AUTHORIZATION_CONSUMED.json": base / "AUTHORIZATION_CONSUMED.json",
        "TERMINAL.json": base / "TERMINAL.json",
    }
    prepared = _load_artifact(paths["PREPARED.json"], PREPARE_SCHEMA)
    reviewed = _load_artifact(paths["REVIEWED.json"], REVIEW_SCHEMA)
    authorized = _load_artifact(paths["AUTHORIZED.json"], AUTH_SCHEMA)
    consumed = _load_artifact(
        paths["AUTHORIZATION_CONSUMED.json"],
        f"{SCHEMA_PREFIX}-authorization-consumed-v1",
    )
    terminal = _load_artifact(paths["TERMINAL.json"], TERMINAL_SCHEMA)
    review_keys = {
        "schema_version", "run_id", "purpose", "reviewer_identity",
        "prepare_file_sha256", "external_approval_path",
        "external_approval_file_sha256", "external_approval_semantic_sha256",
        "nonce", "decision_bindings", "decision_bindings_sha256", "decision",
        "reviewed_at_utc", "semantic_sha256",
    } | false_keys
    auth_keys = {
        "schema_version", "run_id", "purpose", "authorizer_identity",
        "review_file_sha256", "external_authorization_path",
        "external_authorization_file_sha256",
        "external_authorization_semantic_sha256", "issued_at_utc", "expires_at_utc",
        "one_use", "nonce", "decision_bindings", "decision_bindings_sha256",
        "semantic_sha256",
    } | false_keys
    if PUBLIC_FIXED_MODE:
        review_keys.add("reviewer_tool_identity")
        auth_keys.add("authorizer_tool_identity")
    consumed_keys = {
        "schema_version", "run_id", "purpose", "authorization_file_sha256",
        "authorization_semantic_sha256", "nonce", "nonce_claim_file_sha256",
        "decision_bindings", "decision_bindings_sha256", "consumed_at_utc",
        "label_reads", "semantic_sha256",
    } | false_keys
    terminal_keys = {
        "schema_version", "run_id", "purpose", "status", "workload_seconds",
        "peak_rss_bytes", "authorization_consumed_sha256", "label_reads",
        "preterminal_tree", "preterminal_tree_sha256", "result", "terminal_at_utc",
        "semantic_sha256",
    } | false_keys
    if (
        set(reviewed) != review_keys
        or set(authorized) != auth_keys
        or set(consumed) != consumed_keys
        or set(terminal) != terminal_keys
    ):
        raise SupervisorError("medium authorization chain exact schema differs")
    if any(
        artifact.get("run_id") != RUN_ID
        or artifact.get("purpose") != PURPOSE
        or any(artifact.get(key) is not False for key in false_keys)
        for artifact in (reviewed, authorized, consumed, terminal)
    ):
        raise SupervisorError("medium authorization chain scope differs")

    recorded_external_review_path = Path(str(reviewed["external_approval_path"]))
    recorded_external_auth_path = Path(str(authorized["external_authorization_path"]))
    external_review_path = external_review_source or recorded_external_review_path
    external_auth_path = external_authorization_source or recorded_external_auth_path
    external_review = json.loads(external_review_path.read_text())
    external_auth = json.loads(external_auth_path.read_text())
    if (
        set(external_review) != _EXTERNAL_REVIEW_KEYS
        or set(external_auth) != _EXTERNAL_AUTH_KEYS
        or file_sha256(external_review_path) != reviewed["external_approval_file_sha256"]
        or file_sha256(external_auth_path)
        != authorized["external_authorization_file_sha256"]
        or external_review["semantic_sha256"]
        != reviewed["external_approval_semantic_sha256"]
        or external_auth["semantic_sha256"]
        != authorized["external_authorization_semantic_sha256"]
    ):
        raise SupervisorError("medium external decision chain binding differs")
    for artifact, schema, decision in (
        (external_review, EXTERNAL_REVIEW_SCHEMA, REVIEW_DECISION),
        (external_auth, EXTERNAL_AUTH_SCHEMA, AUTH_DECISION),
    ):
        unsigned = dict(artifact)
        semantic = unsigned.pop("semantic_sha256")
        if (
            artifact["schema_version"] != schema
            or artifact["run_id"] != RUN_ID
            or artifact["decision"] != decision
            or (PUBLIC_FIXED_MODE and artifact.get("purpose") != PURPOSE)
            or semantic != canonical_sha256(unsigned)
        ):
            raise SupervisorError("medium external decision semantics differ")

    nonce = str(authorized["nonce"])
    nonce_claims = list(base.glob("NONCE-*.claimed"))
    nonce_path = base / f"NONCE-{nonce}.claimed"
    if len(nonce_claims) != 1 or nonce_claims[0] != nonce_path:
        raise SupervisorError("medium authorization nonce was not consumed exactly once")
    nonce_claim = _load_artifact(
        nonce_path, f"{SCHEMA_PREFIX}-nonce-claim-v1"
    )
    nonce_keys = {
        "schema_version", "run_id", "nonce", "authorization_file_sha256",
        "claimed_at_utc", "semantic_sha256",
    } | false_keys
    if set(nonce_claim) != nonce_keys:
        raise SupervisorError("medium nonce claim exact schema differs")
    expected_bindings = _decision_bindings(prepared, nonce=nonce)
    if (
        file_sha256(paths["AUTHORIZED.json"]) != expected_authorization_sha256
        or reviewed["prepare_file_sha256"] != file_sha256(paths["PREPARED.json"])
        or external_review["prepare_file_sha256"] != file_sha256(paths["PREPARED.json"])
        or external_review["prepare_semantic_sha256"] != prepared["semantic_sha256"]
        or external_auth["review_file_sha256"] != file_sha256(paths["REVIEWED.json"])
        or external_auth["review_semantic_sha256"] != reviewed["semantic_sha256"]
        or authorized["review_file_sha256"] != file_sha256(paths["REVIEWED.json"])
        or any(external_review.get(key) != value for key, value in expected_bindings.items())
        or any(external_auth.get(key) != value for key, value in expected_bindings.items())
        or any(
            artifact.get("nonce") != nonce
            or artifact.get("decision_bindings") != expected_bindings
            or artifact.get("decision_bindings_sha256")
            != canonical_sha256(expected_bindings)
            for artifact in (reviewed, authorized, consumed)
        )
        or nonce_claim["authorization_file_sha256"] != expected_authorization_sha256
        or consumed["authorization_file_sha256"] != expected_authorization_sha256
        or consumed["authorization_semantic_sha256"] != authorized["semantic_sha256"]
        or consumed["nonce_claim_file_sha256"] != file_sha256(nonce_path)
        or terminal["authorization_consumed_sha256"]
        != file_sha256(paths["AUTHORIZATION_CONSUMED.json"])
        or terminal["status"] != "PROFILE_COMPLETE"
        or authorized["one_use"] is not True
        or consumed["label_reads"] != _label_reads()
        or terminal["label_reads"] != _successful_label_reads()
        or (
            PUBLIC_FIXED_MODE
            and (
                external_review.get("reviewer_tool_identity") is None
                or external_auth.get("authorizer_tool_identity") is None
                or external_auth.get("one_use") is not True
                or external_auth.get("reviewer_identity")
                != external_review.get("reviewer_identity")
                or reviewed.get("reviewer_identity")
                != external_review.get("reviewer_identity")
                or authorized.get("authorizer_identity")
                != external_auth.get("authorizer_identity")
                or len(
                    {
                        prepared.get("runner_identity"),
                        reviewed.get("reviewer_identity"),
                        authorized.get("authorizer_identity"),
                    }
                )
                != 3
            )
        )
    ):
        raise SupervisorError("medium authorization chain semantic replay differs")
    chronology = {
        "external_review_issued": _parse_utc(
            external_review["issued_at_utc"], name="external review issued"
        ),
        "internal_reviewed": _parse_utc(reviewed["reviewed_at_utc"], name="internal review"),
        "external_authorization_issued": _parse_utc(
            external_auth["issued_at_utc"], name="external authorization issued"
        ),
        "internal_authorized": _parse_utc(
            authorized["issued_at_utc"], name="internal authorization"
        ),
        "nonce_claimed": _parse_utc(nonce_claim["claimed_at_utc"], name="nonce claim"),
        "authorization_consumed": _parse_utc(
            consumed["consumed_at_utc"], name="authorization consumed"
        ),
        "terminal": _parse_utc(terminal["terminal_at_utc"], name="terminal"),
    }
    if not (
        chronology["external_review_issued"] <= chronology["internal_reviewed"]
        <= chronology["internal_authorized"] <= chronology["nonce_claimed"]
        <= chronology["authorization_consumed"] <= chronology["terminal"]
        and chronology["external_review_issued"]
        <= chronology["external_authorization_issued"]
        <= chronology["internal_authorized"]
    ):
        raise SupervisorError("medium authorization chain chronology differs")
    return {
        "files": {name: file_sha256(path) for name, path in paths.items()},
        "nonce_claim": {"path": nonce_path.name, "sha256": file_sha256(nonce_path)},
        "external_review": {
            "path": str(recorded_external_review_path.resolve()),
            "sha256": file_sha256(external_review_path),
            "semantic_sha256": external_review["semantic_sha256"],
        },
        "external_authorization": {
            "path": str(recorded_external_auth_path.resolve()),
            "sha256": file_sha256(external_auth_path),
            "semantic_sha256": external_auth["semantic_sha256"],
        },
        "chronology": {key: value.isoformat() for key, value in chronology.items()},
        "one_use": True,
    }


def _validate_outer_evidence(
    base: Path,
    result: Mapping[str, object],
    status_rows: Sequence[Mapping[str, object]],
    *,
    launch_status_path: Path,
    launch_result_path: Path,
    expected_authorization_sha256: str,
    observed_launch_status_path: Path | None = None,
) -> tuple[int, dict[str, bool]]:
    result_keys = {
        "schema_version", "provenance", "expected_provenance", "provenance_mismatches",
        "limits", "cgroup", "started_at_utc", "finished_at_utc", "elapsed_seconds", "pid",
        "pgid", "session_id", "leader_starttime", "peak_observed_rss_bytes",
        "child_exit_code", "reason", "state", "term_action_at_utc", "term_survivors",
        "kill_action_at_utc", "kill_survivors", "group_empty", "cgroup_empty",
        "action_evidence", "signal_received", "status_file_sha256", "status_append_error",
        "harness_private_data_accessed",
    }
    status_keys = {
        "timestamp", "elapsed_seconds", "pid", "pgid", "live_descendants",
        "live_descendant_count", "group_members", "aggregate_rss_bytes", "peak_rss_bytes",
        "state", "reason",
    }
    provenance_keys = {
        "command", "command_sha256", "wrapper_argv", "wrapper_argv_sha256",
        "harness_path", "harness_code_sha256", "execution_cwd", "repository",
        "bound_files", "launch_parameters",
    }
    repository_keys = {
        "repository_root", "head", "file_manifest_sha256", "file_count",
        "code_state_sha256",
    }
    expected_keys = {
        "command_sha256", "harness_code_sha256", "code_state_sha256", "cwd",
        "bound_files",
    }
    mismatch_keys = {"command", "harness", "code_state", "cwd", "bound_files"}
    limit_keys = {
        "absolute_wall_seconds", "memory_max_bytes", "poll_seconds",
        "term_grace_seconds", "command_runtime_max_seconds", "result_headroom_seconds",
    }
    cgroup_keys = {
        "unit", "properties", "capability", "memory_peak_bytes",
        "observed_memory_max_bytes", "control_group", "memory_events", "limit_verified",
        "result", "oom_killed", "empty",
    }
    if (
        set(result) != result_keys
        or result.get("schema_version") != "bounded-process-result-v2"
        or not status_rows
        or any(set(row) != status_keys for row in status_rows)
    ):
        raise SupervisorError("medium outer exact top-level schema differs")
    provenance = result["provenance"]
    expected = result["expected_provenance"]
    mismatches = result["provenance_mismatches"]
    limits = result["limits"]
    cgroup = result["cgroup"]
    if not all(
        isinstance(value, Mapping)
        for value in (provenance, expected, mismatches, limits, cgroup)
    ):
        raise SupervisorError("medium outer nested objects differ")
    if (
        set(provenance) != provenance_keys
        or set(provenance["repository"]) != repository_keys
        or set(expected) != expected_keys
        or set(mismatches) != mismatch_keys
        or set(limits) != limit_keys
        or set(cgroup) != cgroup_keys
        or set(cgroup["properties"]) != {"MemoryMax", "OOMPolicy", "RuntimeMaxSec"}
        or set(cgroup["capability"])
        != {"cgroup_filesystem", "memory_controller", "systemd_version"}
        or set(cgroup["memory_events"])
        != {"low", "high", "max", "oom", "oom_kill", "oom_group_kill"}
    ):
        raise SupervisorError("medium outer exact nested schema differs")
    prepared = _load_artifact(base / "PREPARED.json", PREPARE_SCHEMA)
    wrapper = prepared["execution_contract"]["outer_wrapper"]
    actual_wrapper = materialize_outer_wrapper_argv(
        wrapper,
        authorization_sha256=expected_authorization_sha256,
        code_state_sha256=prepared["repository_provenance"]["harness_code_state_sha256"],
    )
    contained = wrapper["contained_argv"]
    expected_bound_files: dict[str, str] = {}
    values = iter(actual_wrapper)
    for value in values:
        if value == "--expected-bound-file":
            bound = next(values)
            path, digest = bound.rsplit("=", 1)
            expected_bound_files[str(Path(path).resolve())] = digest
    bound_files = provenance["bound_files"]
    if set(bound_files) != set(expected_bound_files) or any(
        set(bound_files[path]) != {"expected_sha256", "observed_sha256"}
        or bound_files[path]["expected_sha256"] != digest
        or bound_files[path]["observed_sha256"] != digest
        for path, digest in expected_bound_files.items()
    ):
        raise SupervisorError("medium outer bound-file schema/value differs")
    expected_launch = {
        "profile": "public-fixed-v1" if PUBLIC_FIXED_MODE else "medium-v1",
        "wall_seconds": float(OUTER_SECONDS),
        "memory_max_bytes": RSS_HARD_BYTES,
        "poll_seconds": 0.5,
        "term_grace_seconds": float(FINALIZE_SECONDS),
        "command_seconds": float(OUTER_COMMAND_SECONDS),
        "status_path": str(launch_status_path.resolve()),
        "result_path": str(launch_result_path.resolve()),
    }
    expected_limits = {
        "absolute_wall_seconds": float(OUTER_SECONDS),
        "memory_max_bytes": RSS_HARD_BYTES,
        "poll_seconds": 0.5,
        "term_grace_seconds": float(FINALIZE_SECONDS),
        "command_runtime_max_seconds": float(OUTER_COMMAND_SECONDS),
        "result_headroom_seconds": float(FINALIZE_SECONDS),
    }
    repository = provenance["repository"]
    expected_provenance = {
        "command_sha256": wrapper["contained_argv_sha256"],
        "harness_code_sha256": prepared["execution_contract"]["harness_sha256"],
        "code_state_sha256": prepared["repository_provenance"][
            "harness_code_state_sha256"
        ],
        "cwd": prepared["execution_contract"]["cwd"],
        "bound_files": expected_bound_files,
    }
    if (
        provenance["command"] != contained
        or provenance["command_sha256"] != canonical_sha256(contained)
        or provenance["wrapper_argv"] != actual_wrapper
        or provenance["wrapper_argv_sha256"] != canonical_sha256(actual_wrapper)
        or provenance["harness_path"] != prepared["execution_contract"]["harness_path"]
        or provenance["harness_code_sha256"]
        != prepared["execution_contract"]["harness_sha256"]
        or provenance["execution_cwd"] != prepared["execution_contract"]["cwd"]
        or repository["repository_root"]
        != prepared["repository_provenance"]["repository_root"]
        or repository["head"] != prepared["repository_provenance"]["head"]
        or repository["file_manifest_sha256"]
        != prepared["repository_provenance"]["manifest_sha256"]
        or repository["file_count"] != len(prepared["repository_provenance"]["manifest"])
        or repository["code_state_sha256"]
        != prepared["repository_provenance"]["harness_code_state_sha256"]
        or wrapper["parameters"]["status_path"] != str(launch_status_path.resolve())
        or wrapper["parameters"]["result_path"] != str(launch_result_path.resolve())
        or provenance["launch_parameters"] != expected_launch
        or expected != expected_provenance
        or mismatches != {key: False for key in mismatch_keys}
        or limits != expected_limits
        or cgroup["properties"]
        != {"MemoryMax": RSS_HARD_BYTES, "OOMPolicy": "kill", "RuntimeMaxSec": 2280}
        or cgroup["capability"]["cgroup_filesystem"] != "cgroup2"
        or cgroup["capability"]["memory_controller"] is not True
        or not isinstance(cgroup["capability"]["systemd_version"], str)
        or not isinstance(cgroup["unit"], str)
        or not cgroup["unit"].endswith(".scope")
        or not isinstance(cgroup["control_group"], str)
        or type(cgroup["memory_peak_bytes"]) is not int
        or cgroup["memory_peak_bytes"] < 0
        or cgroup["result"] not in (None, "success")
        or cgroup["observed_memory_max_bytes"] != RSS_HARD_BYTES
    ):
        raise SupervisorError("medium outer provenance/limit values differ")
    elapsed = [row["elapsed_seconds"] for row in status_rows]
    terminal_rows = [row for row in status_rows if row["state"] in {"complete", "failed"}]
    status_peak = max(int(row["peak_rss_bytes"]) for row in status_rows)
    peak_values = [
        int(result["peak_observed_rss_bytes"]),
        int(cgroup["memory_peak_bytes"]),
        status_peak,
    ]
    aggregate_peak = max(peak_values)
    started = _parse_utc(result["started_at_utc"], name="outer started")
    finished = _parse_utc(result["finished_at_utc"], name="outer finished")
    status_times = [_parse_utc(row["timestamp"], name="outer status") for row in status_rows]
    if (
        any(type(value) not in (int, float) or value < 0 for value in elapsed)
        or elapsed != sorted(elapsed)
        or any(
            row["pid"] != result["pid"]
            or row["pgid"] != result["pgid"]
            or row["live_descendant_count"] != len(row["live_descendants"])
            or any(type(member) is not int for member in row["group_members"])
            or row["aggregate_rss_bytes"] < 0
            or row["peak_rss_bytes"] < row["aggregate_rss_bytes"]
            or row["state"] not in {"running", "finalizing", "complete", "failed"}
            or not isinstance(row["reason"], str)
            for row in status_rows
        )
        or len(terminal_rows) != 1
        or terminal_rows[0] != status_rows[-1]
        or terminal_rows[0]["state"] != "complete"
        or terminal_rows[0]["reason"] != "CHILD_EXIT"
        or status_peak != int(result["peak_observed_rss_bytes"])
        or not started <= status_times[0] <= status_times[-1] <= finished
    ):
        raise SupervisorError("medium outer status chronology/peak differs")
    gates = {
        "outer_complete": result["state"] == "complete" and result["reason"] == "CHILD_EXIT",
        "child_exit_zero": result["child_exit_code"] == 0,
        "cgroup_limit_verified": cgroup["limit_verified"] is True,
        "cgroup_empty": result["cgroup_empty"] is True and cgroup["empty"] is True,
        "group_empty": result["group_empty"] is True,
        "not_oom_killed": cgroup["oom_killed"] is False
        and all(
            int(value) == 0
            for key, value in cgroup["memory_events"].items()
            if key.startswith("oom")
        ),
        "aggregate_peak_accepted": aggregate_peak <= RSS_SUCCESS_BYTES,
        "hard_wall_accepted": float(result["elapsed_seconds"]) <= OUTER_SECONDS,
        "status_hash_bound": result["status_file_sha256"]
        == file_sha256(observed_launch_status_path or launch_status_path),
        "no_provenance_mismatch": not any(mismatches.values()),
        "private_data_absent": result["harness_private_data_accessed"] is False,
        "status_append_clean": result["status_append_error"] is None,
        "clean_success_no_action": result["term_action_at_utc"] is None
        and result["kill_action_at_utc"] is None
        and result["term_survivors"] == []
        and result["kill_survivors"] == []
        and result["action_evidence"] == []
        and result["signal_received"] is None,
    }
    return aggregate_peak, gates


def finalize_postrun(
    output: str | Path,
    *,
    launch_status_path: Path,
    launch_result_path: Path,
    expected_authorization_sha256: str,
) -> Path:
    """Freeze outer cgroup disposition; inner success alone is never acceptance."""

    base = _require_exact_base(output)
    reasons: list[str] = []
    try:
        result = json.loads(launch_result_path.read_text())
        status_rows = [json.loads(line) for line in launch_status_path.read_text().splitlines()]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SupervisorError("medium outer launch evidence cannot be loaded") from error
    aggregate_peak = RSS_HARD_BYTES + 1
    gates: dict[str, bool] = {}
    try:
        aggregate_peak, gates = _validate_outer_evidence(
            base,
            result,
            status_rows,
            launch_status_path=launch_status_path,
            launch_result_path=launch_result_path,
            expected_authorization_sha256=expected_authorization_sha256,
        )
    except BaseException as error:
        reasons.append(f"OUTER_EVIDENCE_{type(error).__name__.upper()}")
    reasons.extend(key.upper() for key, value in gates.items() if not value)
    authorization_chain: dict[str, object] = {}
    try:
        authorization_chain = _replay_authorization_chain(
            base, expected_authorization_sha256=expected_authorization_sha256
        )
    except BaseException as error:
        reasons.append(f"AUTHORIZATION_CHAIN_{type(error).__name__.upper()}")
    try:
        verify(base)
    except BaseException:
        reasons.append("INNER_TERMINAL_VERIFY")
    if _artifact_bytes(base) > ARTIFACT_MAX_BYTES:
        reasons.append("ARTIFACT_CAP")
    unsigned = {
        "schema_version": POSTRUN_SCHEMA,
        "run_id": RUN_ID,
        "status": "POSTRUN_ACCEPTED" if not reasons else "POSTRUN_REJECTED",
        "automatic_disposition_only": True,
        "independent_post_review_approved": False,
        "reasons": reasons,
        "launch_status_path": str(launch_status_path.resolve()),
        "launch_status_file_sha256": file_sha256(launch_status_path),
        "launch_status_semantic_sha256": canonical_sha256(status_rows),
        "launch_result_path": str(launch_result_path.resolve()),
        "launch_result_file_sha256": file_sha256(launch_result_path),
        "launch_result_semantic_sha256": canonical_sha256(result),
        "aggregate_cgroup_peak_bytes": aggregate_peak,
        "gates": gates,
        "authorization_chain": authorization_chain,
        "artifact_bytes_before_postrun": _artifact_bytes(base),
        "created_at_utc": datetime.now(UTC).isoformat(),
        **_false_boundaries(),
    }
    _write_immutable(base / "POSTRUN.json", unsigned)
    return base / "POSTRUN.json"


def _verify_postrun_at(
    base: Path,
    *,
    launch_status_source: Path | None = None,
    launch_result_source: Path | None = None,
    external_review_source: Path | None = None,
    external_authorization_source: Path | None = None,
) -> dict[str, object]:
    postrun = _load_artifact(base / "POSTRUN.json", POSTRUN_SCHEMA)
    expected_postrun_keys = {
        "schema_version", "run_id", "status", "automatic_disposition_only",
        "independent_post_review_approved", "reasons", "launch_status_path",
        "launch_status_file_sha256", "launch_status_semantic_sha256",
        "launch_result_path", "launch_result_file_sha256",
        "launch_result_semantic_sha256", "aggregate_cgroup_peak_bytes", "gates",
        "authorization_chain", "artifact_bytes_before_postrun", "created_at_utc",
        "semantic_sha256",
    } | set(_false_boundaries())
    if postrun.get("status") != "POSTRUN_ACCEPTED" or postrun.get("reasons") != []:
        raise SupervisorError("medium postrun disposition is not accepted")
    if (
        set(postrun) != expected_postrun_keys
        or postrun.get("run_id") != RUN_ID
        or postrun.get("automatic_disposition_only") is not True
        or postrun.get("independent_post_review_approved") is not False
        or any(value is not True for value in postrun.get("gates", {}).values())
        or any(postrun.get(key) is not False for key in _false_boundaries())
    ):
        raise SupervisorError("medium postrun acceptance schema differs")
    loaded_outer: dict[str, object] = {}
    for prefix in ("launch_status", "launch_result"):
        path = (
            launch_status_source
            if prefix == "launch_status" and launch_status_source is not None
            else launch_result_source
            if prefix == "launch_result" and launch_result_source is not None
            else Path(str(postrun[f"{prefix}_path"]))
        )
        if path.is_symlink() or not path.is_file() or file_sha256(path) != postrun[
            f"{prefix}_file_sha256"
        ]:
            raise SupervisorError("medium postrun outer physical binding differs")
        loaded = (
            [json.loads(line) for line in path.read_text().splitlines()]
            if prefix == "launch_status"
            else json.loads(path.read_text())
        )
        if canonical_sha256(loaded) != postrun[f"{prefix}_semantic_sha256"]:
            raise SupervisorError("medium postrun outer semantic binding differs")
        loaded_outer[prefix] = loaded
    authorization_sha = postrun["authorization_chain"]["files"]["AUTHORIZED.json"]
    peak, gates = _validate_outer_evidence(
        base,
        loaded_outer["launch_result"],
        loaded_outer["launch_status"],
        launch_status_path=Path(str(postrun["launch_status_path"])),
        launch_result_path=Path(str(postrun["launch_result_path"])),
        expected_authorization_sha256=authorization_sha,
        observed_launch_status_path=launch_status_source,
    )
    if (
        peak != postrun["aggregate_cgroup_peak_bytes"]
        or gates != postrun["gates"]
        or peak > RSS_SUCCESS_BYTES
    ):
        raise SupervisorError("medium postrun aggregate peak exceeds acceptance")
    replayed_chain = _replay_authorization_chain(
        base,
        expected_authorization_sha256=authorization_sha,
        external_review_source=external_review_source,
        external_authorization_source=external_authorization_source,
    )
    if replayed_chain != postrun["authorization_chain"]:
        raise SupervisorError("medium postrun authorization replay differs")
    _verify_inner_at(base)
    if _artifact_bytes(base) > ARTIFACT_MAX_BYTES:
        raise SupervisorError("medium postrun artifact cap differs")
    return postrun


def verify_postrun(output: str | Path) -> dict[str, object]:
    return _verify_postrun_at(_require_exact_base(output))


class WorkerTelemetryBridge:
    """Give a real backend live loss/metric telemetry plus independent heartbeat/guard."""

    def __init__(self, root: Path, trial: TrialIdentity, run_start_sha256: str) -> None:
        self.trial = trial
        self.sink = EventSink(
            root / f"telemetry/worker-{trial.trial_index}.jsonl",
            RUN_ID,
            run_start_sha256=run_start_sha256,
        )
        self.resources = _CpuResourceProvider(root)
        self.heartbeat = HeartbeatEmitter(
            self.sink,
            interval_seconds=10,
            trial_provider=lambda: self.trial,
            progress_provider=lambda: {"state": "training", "live_loss_stream": True},
            resource_provider=self.resources,
        )
        now = time.monotonic_ns()
        self.guard = RunGuard(
            GuardThresholds(
                deadline_monotonic_ns=now + OUTER_SECONDS * 1_000_000_000,
                soft_drain_monotonic_ns=now + WORKLOAD_SECONDS * 1_000_000_000,
                heartbeat_seconds=10,
                stale_after_seconds=20,
                sample_seconds=1,
                rss_max_bytes=RSS_HARD_BYTES,
                gpu_allocated_max_bytes=1,
                gpu_reserved_max_bytes=1,
                gpu_temperature_max_c=1,
                disk_free_min_bytes=1024**3,
            ),
            self.sink,
            self.resources,
            trial_provider=lambda: self.trial,
        )

    def __enter__(self) -> WorkerTelemetryBridge:
        self.heartbeat.start()
        self.guard.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.guard.close()
        self.heartbeat.close()
        self.sink.close()


class _CpuResourceProvider(SystemResourceProvider):
    def sample(self) -> Mapping[str, object]:
        resources = dict(super().sample())
        resources.update(
            gpu_allocated_bytes=None,
            gpu_reserved_bytes=None,
            gpu_temperature_c=None,
        )
        return resources


def set_fixed_cpu_threads() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = str(THREADS)


__all__ = [
    "AUTH_SCHEMA",
    "EXACT_BASE",
    "FINALIZE_SECONDS",
    "OUTER_SECONDS",
    "RSS_HARD_BYTES",
    "RSS_SUCCESS_BYTES",
    "RUN_ID",
    "SAMPLE_ROWS",
    "SEED",
    "TERMINAL_SCHEMA",
    "WORKLOAD_SECONDS",
    "WorkerTelemetryBridge",
    "authorize",
    "candidates",
    "execute",
    "frozen_config",
    "medium_split",
    "planned_trials",
    "prepare",
    "review",
    "sample_evidence",
    "set_fixed_cpu_threads",
    "stable_group_sample",
    "verify",
]
