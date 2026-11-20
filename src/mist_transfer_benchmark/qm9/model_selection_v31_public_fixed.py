"""Frozen public-synthetic input for the v3.1 three-backend engineering seam.

This module deliberately contains no training entrypoint.  It creates a target-bearing
*synthetic* bundle before external review so that the later Ridge/XGBoost/MLP run can
be authorized against immutable, real hashes without touching any QM9 target file.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import sparse

from .constants import TARGET_COLUMNS
from .model_selection_v31_medium_profile import candidates
from .model_selection_v31_observability import TrialIdentity
from .model_selection_v31_supervisor import (
    PlannedTrial,
    canonical_sha256,
    file_sha256,
)

RUN_ID = "qm9-v31-public-fixed-three-backend-cpu-v1"
PURPOSE = "public_synthetic_three_backend_engineering_seam"
EXACT_BASE = Path("/tmp/qm9-v31-public-fixed-three-backend-cpu-v1")
BUNDLE_BASE = Path("/tmp/qm9-v31-public-fixed-three-backend-bundle-v1")
OUTER_ENVELOPE = Path("/tmp/qm9-v31-public-fixed-three-backend-envelope-v1")
EVIDENCE_ARCHIVE = Path("/tmp/qm9-v31-public-fixed-three-backend-evidence-v1.tar.gz")
BUNDLE_SCHEMA = "qm9-v31-public-fixed-three-backend-bundle-v1"
MIN_ROWS = 8192
ECFP_BITS = 2048
DESCRIPTORS = 17
FEATURES = ECFP_BITS + DESCRIPTORS
SEED = 20260713
TUNE_FRACTION = 0.10


class PublicFixedBundleError(RuntimeError):
    """The fixed public-synthetic input is absent, changed, or semantically invalid."""


def _tree_identity(root: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise PublicFixedBundleError("evidence tree contains a symlink")
        if path.is_file():
            rows.append(
                {"path": relative, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
            )
    return rows


def _semantic_json(path: Path) -> dict[str, object]:
    artifact = json.loads(path.read_text())
    unsigned = dict(artifact)
    semantic = unsigned.pop("semantic_sha256", None)
    if semantic != canonical_sha256(unsigned):
        raise PublicFixedBundleError("review artifact semantic hash differs")
    return artifact


def _timestamp(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise PublicFixedBundleError("evidence timestamp lacks timezone")
    return parsed


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    identity = {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "bytes_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
    }
    return canonical_sha256(identity)


def _csr_identity(matrix: sparse.csr_matrix) -> dict[str, object]:
    value = matrix.tocsr()
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "nnz": int(value.nnz),
        "data_sha256": _array_sha256(value.data),
        "indices_sha256": _array_sha256(value.indices),
        "indptr_sha256": _array_sha256(value.indptr),
    }


def _groups() -> np.ndarray:
    """Select whole unequal groups until the fixed minimum row threshold is reached."""

    values: list[int] = []
    group = 0
    while len(values) < MIN_ROWS:
        size = 1 + ((group * 5 + 3) % 7)
        values.extend([group] * size)
        group += 1
    return np.asarray(values, dtype=np.int64)


def _group_split(groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(groups)
    rng = np.random.default_rng(SEED + 1)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    target = int(round(len(groups) * TUNE_FRACTION))
    tune_groups: list[int] = []
    rows = 0
    group_sizes = {int(group): int(np.count_nonzero(groups == group)) for group in unique}
    for group in shuffled:
        if rows >= target:
            break
        tune_groups.append(int(group))
        rows += group_sizes[int(group)]
    tune_mask = np.isin(groups, np.asarray(tune_groups, dtype=np.int64))
    train = np.flatnonzero(~tune_mask).astype(np.int64)
    tune = np.flatnonzero(tune_mask).astype(np.int64)
    if set(groups[train].tolist()) & set(groups[tune].tolist()):
        raise PublicFixedBundleError("synthetic group-safe split overlaps")
    return train, tune


def _features(groups: np.ndarray) -> sparse.csr_matrix:
    rng = np.random.default_rng(SEED)
    bits_per_row = 16
    row_count = len(groups)
    indices = np.empty(row_count * bits_per_row, dtype=np.int32)
    for row in range(row_count):
        indices[row * bits_per_row : (row + 1) * bits_per_row] = np.sort(
            rng.choice(ECFP_BITS, size=bits_per_row, replace=False)
        )
    indptr = np.arange(0, len(indices) + 1, bits_per_row, dtype=np.int32)
    bit_matrix = sparse.csr_matrix(
        (np.ones(len(indices), dtype=np.float32), indices, indptr),
        shape=(row_count, ECFP_BITS),
    )
    row = np.arange(row_count, dtype=np.float32)
    descriptor_columns = []
    group_float = groups.astype(np.float32)
    for column in range(DESCRIPTORS):
        frequency = np.float32(column + 1) / np.float32(37.0)
        signal = np.sin(row * frequency) + np.cos(group_float * frequency * 0.37)
        noise = rng.normal(0.0, 0.03 + 0.002 * column, size=row_count)
        descriptor_columns.append((signal + noise).astype(np.float32))
    descriptors = sparse.csr_matrix(np.column_stack(descriptor_columns), dtype=np.float32)
    return sparse.hstack([bit_matrix, descriptors], format="csr", dtype=np.float32)


def _targets(matrix: sparse.csr_matrix) -> np.ndarray:
    """Create finite feature-related targets with deliberately different units/scales."""

    rng = np.random.default_rng(SEED + 2)
    descriptors = matrix[:, ECFP_BITS:].toarray().astype(np.float64)
    selected_bits = matrix[:, np.arange(0, ECFP_BITS, 97)[:12]].toarray().astype(np.float64)
    scales = np.asarray(
        [0.02, 0.1, 0.5, 1.0, 2.5, 8.0, 25.0, 80.0, 250.0, 800.0, 2500.0, 8000.0],
        dtype=np.float64,
    )
    offsets = np.asarray(
        [-40.0, -8.0, -1.0, 0.0, 2.0, 12.0, 50.0, 120.0, 500.0, 1500.0, 5000.0, 15000.0],
        dtype=np.float64,
    )
    row_count = matrix.shape[0]
    targets = np.empty((row_count, len(TARGET_COLUMNS)), dtype=np.float64)
    for target in range(len(TARGET_COLUMNS)):
        first = target % DESCRIPTORS
        second = (target * 5 + 3) % DESCRIPTORS
        raw = (
            0.65 * descriptors[:, first]
            - 0.30 * descriptors[:, second]
            + 0.45 * selected_bits[:, target]
            + 0.12 * descriptors[:, first] * descriptors[:, second]
            + rng.normal(0.0, 0.08, size=row_count)
        )
        targets[:, target] = offsets[target] + scales[target] * raw
    return targets


def _plan(tune_rows: int) -> list[dict[str, object]]:
    plan = []
    for index, candidate in enumerate(candidates(), start=1):
        plan.append(
            PlannedTrial(
                identity=TrialIdentity(
                    run_id=RUN_ID,
                    wave="MEDIUM_PROFILE",
                    strategy="identity",
                    candidate_id=str(candidate["id"]),
                    family=str(candidate["family"]),
                    fold=0,
                    seed=SEED,
                    trial_index=index,
                    trial_total=3,
                ),
                expected_shape=(tune_rows, len(TARGET_COLUMNS)),
                target_order=tuple(TARGET_COLUMNS),
                prediction_order_sha256=canonical_sha256(list(range(tune_rows))),
                candidate_sha256=canonical_sha256(candidate),
            ).payload()
        )
    return plan


def create_bundle(directory: str | Path) -> Path:
    root = Path(directory).resolve()
    if root.exists():
        raise PublicFixedBundleError("public fixed bundle directory already exists")
    root.mkdir(parents=True, mode=0o700)
    groups = _groups()
    train, tune = _group_split(groups)
    matrix = _features(groups)
    targets = _targets(matrix)
    row_count = len(groups)
    rows = np.arange(row_count, dtype=np.int64)
    plan = _plan(len(tune))
    sample_unsigned = {
        "rows": row_count,
        "train_rows": int(len(train)),
        "tune_rows": int(len(tune)),
        "row_ids_sha256": _array_sha256(rows),
        "groups_sha256": _array_sha256(groups),
        "train_positions_sha256": _array_sha256(train),
        "tune_positions_sha256": _array_sha256(tune),
    }
    sample = {**sample_unsigned, "semantic_sha256": canonical_sha256(sample_unsigned)}
    bundle_path = root / "public-synthetic-three-backend.npz"
    np.savez_compressed(
        bundle_path,
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=np.asarray(matrix.shape, dtype=np.int64),
        targets=targets,
        row_ids=rows,
        groups=groups,
        train_positions=train,
        tune_positions=tune,
    )
    os.chmod(bundle_path, 0o444)
    unsigned = {
        "schema_version": BUNDLE_SCHEMA,
        "run_id": RUN_ID,
        "purpose": PURPOSE,
        "source": "deterministic_public_synthetic",
        "private_data_allowed": False,
        "scientific_result": False,
        "capacity_result": False,
        "ranking_allowed": False,
        "selection_allowed": False,
        "publication_allowed": False,
        "dimensions": {
            "rows": row_count,
            "minimum_rows": MIN_ROWS,
            "features": FEATURES,
            "targets": len(TARGET_COLUMNS),
        },
        "feature_contract": {
            "ecfp_like_binary_bits": ECFP_BITS,
            "dense_descriptors": DESCRIPTORS,
            "matrix": _csr_identity(matrix),
        },
        "target_contract": {
            "names": list(TARGET_COLUMNS),
            "values_sha256": _array_sha256(targets),
            "finite": bool(np.isfinite(targets).all()),
            "per_target_min": targets.min(axis=0).tolist(),
            "per_target_max": targets.max(axis=0).tolist(),
        },
        "row_ids_sha256": _array_sha256(rows),
        "groups_sha256": _array_sha256(groups),
        "group_count": int(len(np.unique(groups))),
        "group_sizes_unequal": bool(len(np.unique(np.bincount(groups))) > 1),
        "train_positions_sha256": _array_sha256(train),
        "tune_positions_sha256": _array_sha256(tune),
        "train_rows": int(len(train)),
        "tune_rows": int(len(tune)),
        "split_group_overlap": 0,
        "sample": sample,
        "candidate_contract": candidates(),
        "candidate_contract_sha256": canonical_sha256(candidates()),
        "plan": plan,
        "plan_sha256": canonical_sha256(plan),
        "bundle_file": bundle_path.name,
        "bundle_file_sha256": file_sha256(bundle_path),
        "label_reads": {
            "synthetic_target_generation": 1,
            "source_fixed_train_target_loads": 0,
            "outer_validation": 0,
            "test": 0,
            "protected": 0,
            "private_evaluation": 0,
        },
    }
    manifest = {**unsigned, "semantic_sha256": canonical_sha256(unsigned)}
    manifest_path = root / "MANIFEST.json"
    descriptor = os.open(manifest_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(descriptor, json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return manifest_path


def validate_bundle(directory: str | Path) -> dict[str, object]:
    root = Path(directory).resolve()
    manifest_path = root / "MANIFEST.json"
    if (
        root.is_symlink()
        or not root.is_dir()
        or {path.name for path in root.iterdir()}
        != {"MANIFEST.json", "public-synthetic-three-backend.npz"}
        or any(path.is_symlink() for path in root.iterdir())
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
    ):
        raise PublicFixedBundleError("public fixed manifest is absent or symlinked")
    manifest = json.loads(manifest_path.read_text())
    expected_manifest_keys = {
        "schema_version",
        "run_id",
        "purpose",
        "source",
        "private_data_allowed",
        "scientific_result",
        "capacity_result",
        "ranking_allowed",
        "selection_allowed",
        "publication_allowed",
        "dimensions",
        "feature_contract",
        "target_contract",
        "row_ids_sha256",
        "groups_sha256",
        "group_count",
        "group_sizes_unequal",
        "train_positions_sha256",
        "tune_positions_sha256",
        "train_rows",
        "tune_rows",
        "split_group_overlap",
        "sample",
        "candidate_contract",
        "candidate_contract_sha256",
        "plan",
        "plan_sha256",
        "bundle_file",
        "bundle_file_sha256",
        "label_reads",
        "semantic_sha256",
    }
    unsigned = {key: value for key, value in manifest.items() if key != "semantic_sha256"}
    bundle_path = root / str(manifest.get("bundle_file", ""))
    if (
        set(manifest) != expected_manifest_keys
        or manifest.get("schema_version") != BUNDLE_SCHEMA
        or manifest.get("run_id") != RUN_ID
        or manifest.get("purpose") != PURPOSE
        or manifest.get("private_data_allowed") is not False
        or manifest.get("scientific_result") is not False
        or manifest.get("semantic_sha256") != canonical_sha256(unsigned)
        or bundle_path.is_symlink()
        or not bundle_path.is_file()
        or manifest.get("bundle_file_sha256") != file_sha256(bundle_path)
        or manifest.get("candidate_contract") != candidates()
        or manifest.get("candidate_contract_sha256") != canonical_sha256(candidates())
        or manifest_path.stat().st_mode & 0o777 != 0o444
        or bundle_path.stat().st_mode & 0o777 != 0o444
    ):
        raise PublicFixedBundleError("public fixed manifest contract differs")
    with np.load(bundle_path, allow_pickle=False) as archive:
        required = {
            "data", "indices", "indptr", "shape", "targets", "row_ids", "groups",
            "train_positions", "tune_positions",
        }
        if set(archive.files) != required:
            raise PublicFixedBundleError("public fixed NPZ members differ")
        shape = tuple(int(value) for value in archive["shape"])
        matrix = sparse.csr_matrix(
            (archive["data"], archive["indices"], archive["indptr"]), shape=shape
        )
        targets = archive["targets"]
        rows = archive["row_ids"]
        groups = archive["groups"]
        train = archive["train_positions"]
        tune = archive["tune_positions"]
    expected_groups = _groups()
    expected_train, expected_tune = _group_split(expected_groups)
    expected_matrix = _features(expected_groups)
    expected_targets = _targets(expected_matrix)
    expected_plan = _plan(len(expected_tune))
    expected = {
        "dimensions": {
            "rows": matrix.shape[0],
            "minimum_rows": MIN_ROWS,
            "features": FEATURES,
            "targets": len(TARGET_COLUMNS),
        },
        "matrix": _csr_identity(matrix),
        "targets": _array_sha256(targets),
        "rows": _array_sha256(rows),
        "groups": _array_sha256(groups),
        "train": _array_sha256(train),
        "tune": _array_sha256(tune),
        "plan": expected_plan,
    }
    expected_sample_unsigned = {
        "rows": matrix.shape[0],
        "train_rows": len(expected_train),
        "tune_rows": len(expected_tune),
        "row_ids_sha256": _array_sha256(np.arange(matrix.shape[0], dtype=np.int64)),
        "groups_sha256": _array_sha256(expected_groups),
        "train_positions_sha256": _array_sha256(expected_train),
        "tune_positions_sha256": _array_sha256(expected_tune),
    }
    expected_sample = {
        **expected_sample_unsigned,
        "semantic_sha256": canonical_sha256(expected_sample_unsigned),
    }
    expected_label_reads = {
        "synthetic_target_generation": 1,
        "source_fixed_train_target_loads": 0,
        "outer_validation": 0,
        "test": 0,
        "protected": 0,
        "private_evaluation": 0,
    }
    if (
        manifest.get("dimensions") != expected["dimensions"]
        or manifest.get("feature_contract", {}).get("matrix") != expected["matrix"]
        or manifest.get("target_contract", {}).get("values_sha256") != expected["targets"]
        or manifest.get("row_ids_sha256") != expected["rows"]
        or manifest.get("groups_sha256") != expected["groups"]
        or manifest.get("train_positions_sha256") != expected["train"]
        or manifest.get("tune_positions_sha256") != expected["tune"]
        or manifest.get("plan") != expected["plan"]
        or manifest.get("plan_sha256") != canonical_sha256(expected["plan"])
        or manifest.get("source") != "deterministic_public_synthetic"
        or any(
            manifest.get(key) is not False
            for key in (
                "private_data_allowed",
                "scientific_result",
                "capacity_result",
                "ranking_allowed",
                "selection_allowed",
                "publication_allowed",
            )
        )
        or manifest.get("feature_contract", {}).get("ecfp_like_binary_bits") != ECFP_BITS
        or manifest.get("feature_contract", {}).get("dense_descriptors") != DESCRIPTORS
        or manifest.get("target_contract", {}).get("names") != list(TARGET_COLUMNS)
        or manifest.get("target_contract", {}).get("finite") is not True
        or manifest.get("target_contract", {}).get("per_target_min")
        != expected_targets.min(axis=0).tolist()
        or manifest.get("target_contract", {}).get("per_target_max")
        != expected_targets.max(axis=0).tolist()
        or manifest.get("group_count") != len(np.unique(expected_groups))
        or manifest.get("group_sizes_unequal") is not True
        or manifest.get("train_rows") != len(expected_train)
        or manifest.get("tune_rows") != len(expected_tune)
        or manifest.get("split_group_overlap") != 0
        or manifest.get("sample") != expected_sample
        or manifest.get("label_reads") != expected_label_reads
        or not np.array_equal(groups, expected_groups)
        or not np.array_equal(train, expected_train)
        or not np.array_equal(tune, expected_tune)
        or _csr_identity(matrix) != _csr_identity(expected_matrix)
        or _array_sha256(targets) != _array_sha256(expected_targets)
        or not np.isfinite(targets).all()
        or not np.all(np.isin(matrix[:, :ECFP_BITS].data, (0.0, 1.0)))
        or set(groups[train].tolist()) & set(groups[tune].tolist())
        or len(train) + len(tune) != matrix.shape[0]
        or set(train.tolist()) & set(tune.tolist())
        or sorted(np.concatenate([train, tune]).tolist()) != list(range(matrix.shape[0]))
    ):
        raise PublicFixedBundleError("public fixed bundle replay differs")
    return manifest


def validate_postreview(
    path: str | Path,
    *,
    base: str | Path,
    envelope: str | Path,
) -> dict[str, object]:
    review_path = Path(path).resolve()
    base_path = Path(base).resolve()
    envelope_path = Path(envelope).resolve()
    if review_path.is_symlink() or not review_path.is_file():
        raise PublicFixedBundleError("independent postreview is missing")
    review = _semantic_json(review_path)
    postrun_path = base_path / "POSTRUN.json"
    from . import model_selection_v31_medium_profile as medium

    if (
        not medium.PUBLIC_FIXED_MODE
        or base_path != medium.EXACT_BASE.resolve()
        or medium.verify_postrun(base_path).get("status") != "POSTRUN_ACCEPTED"
    ):
        raise PublicFixedBundleError("postreview is outside verified public fixed lineage")
    postrun = _semantic_json(postrun_path)
    prepared = _semantic_json(base_path / "PREPARED.json")
    reviewed = _semantic_json(base_path / "REVIEWED.json")
    authorized = _semantic_json(base_path / "AUTHORIZED.json")
    expected_keys = {
        "schema_version",
        "run_id",
        "purpose",
        "decision",
        "reviewer_identity",
        "reviewer_tool_identity",
        "postrun_file_sha256",
        "postrun_semantic_sha256",
        "base_tree_sha256",
        "envelope_tree_sha256",
        "issued_at_utc",
        "semantic_sha256",
    }
    if (
        set(review) != expected_keys
        or review.get("schema_version") != "qm9-v31-public-fixed-postreview-v1"
        or review.get("run_id") != RUN_ID
        or review.get("purpose") != PURPOSE
        or review.get("decision") != "ACCEPT_ENGINEERING_EVIDENCE_ONLY"
        or not review.get("reviewer_identity")
        or not review.get("reviewer_tool_identity")
        or review.get("reviewer_identity")
        in {
            prepared.get("runner_identity"),
            reviewed.get("reviewer_identity"),
            authorized.get("authorizer_identity"),
        }
        or _timestamp(review.get("issued_at_utc"))
        < _timestamp(postrun.get("created_at_utc"))
        or postrun.get("status") != "POSTRUN_ACCEPTED"
        or postrun.get("independent_post_review_approved") is not False
        or review.get("postrun_file_sha256") != file_sha256(postrun_path)
        or review.get("postrun_semantic_sha256") != postrun["semantic_sha256"]
        or review.get("base_tree_sha256") != canonical_sha256(_tree_identity(base_path))
        or review.get("envelope_tree_sha256")
        != canonical_sha256(_tree_identity(envelope_path))
    ):
        raise PublicFixedBundleError("independent postreview contract differs")
    return review


def build_evidence_archive(
    *,
    base: str | Path,
    envelope: str | Path,
    external_review: str | Path,
    external_authorization: str | Path,
    postreview: str | Path,
    output: str | Path = EVIDENCE_ARCHIVE,
) -> Path:
    base_path = Path(base).resolve()
    envelope_path = Path(envelope).resolve()
    review_path = Path(external_review).resolve()
    authorization_path = Path(external_authorization).resolve()
    postreview_path = Path(postreview).resolve()
    destination = Path(output).resolve()
    if destination.exists():
        raise PublicFixedBundleError("evidence archive output already exists")
    validate_postreview(postreview_path, base=base_path, envelope=envelope_path)
    reviewed = _semantic_json(base_path / "REVIEWED.json")
    authorized = _semantic_json(base_path / "AUTHORIZED.json")
    if (
        reviewed.get("external_approval_path") != str(review_path)
        or reviewed.get("external_approval_file_sha256") != file_sha256(review_path)
        or authorized.get("external_authorization_path") != str(authorization_path)
        or authorized.get("external_authorization_file_sha256")
        != file_sha256(authorization_path)
    ):
        raise PublicFixedBundleError("archive decisions differ from executed lineage")
    for decision in (review_path, authorization_path):
        _semantic_json(decision)
    with tempfile.TemporaryDirectory(prefix="qm9-v31-public-fixed-archive-") as temporary:
        root = Path(temporary) / "evidence"
        shutil.copytree(base_path, root / "run")
        shutil.copytree(envelope_path, root / "outer-envelope")
        decisions = root / "decisions"
        decisions.mkdir()
        shutil.copy2(review_path, decisions / "external-review.json")
        shutil.copy2(authorization_path, decisions / "external-authorization.json")
        shutil.copy2(postreview_path, decisions / "independent-postreview.json")
        inventory = _tree_identity(root)
        inventory_unsigned = {
            "schema_version": "qm9-v31-public-fixed-evidence-inventory-v1",
            "run_id": RUN_ID,
            "purpose": PURPOSE,
            "files": inventory,
            "files_sha256": canonical_sha256(inventory),
        }
        inventory_artifact = {
            **inventory_unsigned,
            "semantic_sha256": canonical_sha256(inventory_unsigned),
        }
        (root / "ARCHIVE_MANIFEST.json").write_text(
            json.dumps(inventory_artifact, sort_keys=True, separators=(",", ":"))
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o200)
        with os.fdopen(descriptor, "wb") as raw:
            import gzip

            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as archive:
                    for source in sorted(root.rglob("*")):
                        info = archive.gettarinfo(
                            str(source), arcname=f"evidence/{source.relative_to(root).as_posix()}"
                        )
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        info.mtime = 0
                        if source.is_file():
                            with source.open("rb") as handle:
                                archive.addfile(info, handle)
                        else:
                            archive.addfile(info)
        destination.chmod(0o444)
    validate_evidence_archive(destination)
    return destination


def validate_evidence_archive(path: str | Path) -> dict[str, object]:
    archive_path = Path(path).resolve()
    if archive_path.is_symlink() or not archive_path.is_file():
        raise PublicFixedBundleError("evidence archive is missing")
    with tempfile.TemporaryDirectory(prefix="qm9-v31-public-fixed-verify-") as temporary:
        target = Path(temporary)
        with tarfile.open(archive_path, mode="r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            normalized_names = [Path(name).as_posix() for name in names]
            if any(
                member.issym()
                or member.islnk()
                or not (member.isfile() or member.isdir())
                or not member.name.startswith("evidence/")
                or Path(member.name).is_absolute()
                or ".." in Path(member.name).parts
                for member in members
            ) or (
                len(members) > 20_000
                or sum(member.size for member in members) > 2 * 1024**3
                or names != normalized_names
                or len(normalized_names) != len(set(normalized_names))
            ):
                raise PublicFixedBundleError("evidence archive member is unsafe")
            archive.extractall(target, filter="data")
        root = target / "evidence"
        if (
            not root.is_dir()
            or {entry.name for entry in root.iterdir()}
            != {"ARCHIVE_MANIFEST.json", "run", "outer-envelope", "decisions"}
            or not (root / "run").is_dir()
            or not (root / "outer-envelope").is_dir()
            or not (root / "decisions").is_dir()
            or {entry.name for entry in (root / "decisions").iterdir()}
            != {
                "external-review.json",
                "external-authorization.json",
                "independent-postreview.json",
            }
        ):
            raise PublicFixedBundleError("evidence archive structure differs")
        manifest_path = root / "ARCHIVE_MANIFEST.json"
        manifest = _semantic_json(manifest_path)
        observed = _tree_identity(root)
        observed_without_manifest = [
            row for row in observed if row["path"] != "ARCHIVE_MANIFEST.json"
        ]
        if (
            set(manifest)
            != {
                "schema_version",
                "run_id",
                "purpose",
                "files",
                "files_sha256",
                "semantic_sha256",
            }
            or
            manifest.get("schema_version")
            != "qm9-v31-public-fixed-evidence-inventory-v1"
            or manifest.get("run_id") != RUN_ID
            or manifest.get("purpose") != PURPOSE
            or manifest.get("files") != observed_without_manifest
            or manifest.get("files_sha256") != canonical_sha256(observed_without_manifest)
        ):
            raise PublicFixedBundleError("evidence archive inventory replay differs")
        run = root / "run"
        envelope = root / "outer-envelope"
        decisions = root / "decisions"
        postreview = _semantic_json(decisions / "independent-postreview.json")
        postrun = _semantic_json(run / "POSTRUN.json")
        reviewed = _semantic_json(run / "REVIEWED.json")
        authorized = _semantic_json(run / "AUTHORIZED.json")
        prepared = _semantic_json(run / "PREPARED.json")
        external_review = decisions / "external-review.json"
        external_authorization = decisions / "external-authorization.json"
        _semantic_json(external_review)
        _semantic_json(external_authorization)
        expected_postreview_keys = {
            "schema_version",
            "run_id",
            "purpose",
            "decision",
            "reviewer_identity",
            "reviewer_tool_identity",
            "postrun_file_sha256",
            "postrun_semantic_sha256",
            "base_tree_sha256",
            "envelope_tree_sha256",
            "issued_at_utc",
            "semantic_sha256",
        }
        if (
            set(postreview) != expected_postreview_keys
            or postreview.get("schema_version")
            != "qm9-v31-public-fixed-postreview-v1"
            or postreview.get("run_id") != RUN_ID
            or postreview.get("purpose") != PURPOSE
            or postreview.get("decision") != "ACCEPT_ENGINEERING_EVIDENCE_ONLY"
            or not postreview.get("reviewer_identity")
            or not postreview.get("reviewer_tool_identity")
            or postreview.get("reviewer_identity")
            in {
                prepared.get("runner_identity"),
                reviewed.get("reviewer_identity"),
                authorized.get("authorizer_identity"),
            }
            or _timestamp(postreview.get("issued_at_utc"))
            < _timestamp(postrun.get("created_at_utc"))
            or postrun.get("status") != "POSTRUN_ACCEPTED"
            or postrun.get("independent_post_review_approved") is not False
            or postreview.get("postrun_file_sha256") != file_sha256(run / "POSTRUN.json")
            or postreview.get("postrun_semantic_sha256") != postrun["semantic_sha256"]
            or postreview.get("base_tree_sha256") != canonical_sha256(_tree_identity(run))
            or postreview.get("envelope_tree_sha256")
            != canonical_sha256(_tree_identity(envelope))
            or reviewed.get("external_approval_file_sha256")
            != file_sha256(external_review)
            or reviewed.get("external_approval_semantic_sha256")
            != _semantic_json(external_review)["semantic_sha256"]
            or authorized.get("external_authorization_file_sha256")
            != file_sha256(external_authorization)
            or authorized.get("external_authorization_semantic_sha256")
            != _semantic_json(external_authorization)["semantic_sha256"]
        ):
            raise PublicFixedBundleError("evidence archive lifecycle binding differs")
        status_source = envelope / Path(str(postrun.get("launch_status_path", ""))).name
        result_source = envelope / Path(str(postrun.get("launch_result_path", ""))).name
        replay_code = (
            "from pathlib import Path; "
            "import sys; "
            "from mist_transfer_benchmark.qm9 import model_selection_v31_medium_profile as m; "
            "m._verify_postrun_at(Path(sys.argv[1]), "
            "launch_status_source=Path(sys.argv[2]), launch_result_source=Path(sys.argv[3]), "
            "external_review_source=Path(sys.argv[4]), "
            "external_authorization_source=Path(sys.argv[5]))"
        )
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    replay_code,
                    str(run),
                    str(status_source),
                    str(result_source),
                    str(external_review),
                    str(external_authorization),
                ],
                env={**os.environ, "QM9_V31_PUBLIC_FIXED_BACKENDS": "1"},
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired as error:
            raise PublicFixedBundleError(
                "evidence archive deep lifecycle replay timed out"
            ) from error
        if completed.returncode != 0:
            raise PublicFixedBundleError(
                "evidence archive deep lifecycle replay differs: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
        return manifest


__all__ = [
    "BUNDLE_BASE",
    "BUNDLE_SCHEMA",
    "EVIDENCE_ARCHIVE",
    "EXACT_BASE",
    "OUTER_ENVELOPE",
    "PURPOSE",
    "RUN_ID",
    "build_evidence_archive",
    "create_bundle",
    "validate_evidence_archive",
    "validate_bundle",
    "validate_postreview",
]
