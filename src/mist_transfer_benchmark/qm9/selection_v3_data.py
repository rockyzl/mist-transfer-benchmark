"""Target-free data contracts for QM9 v3 feature selection.

This module accepts only molecular identities, feature matrices, and source-row
indices.  It deliberately has no target-loader or model-training interface.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from mist_transfer_benchmark.qm9.engineered_features import engineered_feature_schema

RECIPE_IDS = (
    "descriptor_scaled_only",
    "all_sparse_scaled",
    "all_raw",
    "log_count_descriptor_scaled",
    "drop_constant_descriptor_scaled",
)


class SelectionV3DataError(ValueError):
    """Raised when a target-free selection-data contract is violated."""


_FEATURE_ARTIFACT_TOP_LEVEL_KEYS = {
    "schema_version",
    "source_csv_sha256",
    "source_row_identity_sha256",
    "source_smiles_sha256",
    "rows",
    "feature_schema",
    "feature_matrix",
    "scaffold_groups",
}
_FEATURE_MATRIX_MANIFEST_KEYS = {"path", "file_sha256", "semantic_sha256", "shape", "nnz"}
_SCAFFOLD_GROUP_MANIFEST_KEYS = {"path", "file_sha256", "groups_sha256", "count"}


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SelectionV3DataError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sparse_semantic_sha256(matrix: sparse.spmatrix) -> str:
    """Match the semantic digest emitted by ``prepare_qm9_paper_features.py``."""

    values = sparse.csr_matrix(matrix)
    digest = hashlib.sha256()
    header = {"kind": "csr", "shape": list(values.shape), "dtype": str(values.dtype)}
    digest.update(json.dumps(header, sort_keys=True).encode())
    for array in (values.indptr, values.indices, values.data):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _decimal_lines_sha256(values: np.ndarray, *, sort: bool = False) -> str:
    array = np.asarray(values, dtype=np.int64)
    if sort:
        array = np.sort(array)
    digest = hashlib.sha256()
    for value in array:
        digest.update(str(int(value)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _string_lines_sha256(values: Sequence[str], *, sort: bool = False) -> str:
    items = [str(value) for value in values]
    if sort:
        items.sort()
    digest = hashlib.sha256()
    for value in items:
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_indices(
    values: np.ndarray | Sequence[int], rows: int, *, name: str, allow_empty: bool = False
) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise SelectionV3DataError(f"{name} must be a one-dimensional integer array")
    result = raw.astype(np.int64, copy=False)
    if not allow_empty and len(result) == 0:
        raise SelectionV3DataError(f"{name} must not be empty")
    if np.any(result < 0) or np.any(result >= rows):
        raise SelectionV3DataError(f"{name} contains an out-of-range source index")
    if len(np.unique(result)) != len(result):
        raise SelectionV3DataError(f"{name} contains duplicate source indices")
    return result.copy()


def _as_finite_csr(matrix: Any) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        result = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    else:
        values = np.asarray(matrix)
        if values.ndim != 2 or values.dtype.kind not in "biuf":
            raise SelectionV3DataError("feature matrix must be a two-dimensional numeric array")
        result = sparse.csr_matrix(values.astype(np.float64, copy=False))
    if result.ndim != 2:
        raise SelectionV3DataError("feature matrix must be two-dimensional")
    result.sum_duplicates()
    result.eliminate_zeros()
    result.sort_indices()
    if not np.all(np.isfinite(result.data)):
        raise SelectionV3DataError("feature matrix contains nonfinite values")
    return result


def _validate_fingerprint_columns(columns: int, fingerprint_columns: int) -> int:
    if type(fingerprint_columns) is not int or not 0 < fingerprint_columns <= columns:
        raise SelectionV3DataError("fingerprint_columns must be within the feature matrix")
    return fingerprint_columns


def _quantiles(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {}
    points = np.quantile(array, [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])
    return {
        name: float(value)
        for name, value in zip(
            ("min", "p01", "p10", "median", "p90", "p99", "max"),
            points,
            strict=True,
        )
    }


def _column_mean_variance(matrix: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    means = np.asarray(matrix.mean(axis=0)).ravel()
    second = np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel()
    variances = np.maximum(second - means * means, 0.0)
    return means, variances


def canonical_connectivity_groups(smiles: Sequence[str]) -> np.ndarray:
    """Return canonical non-isomeric SMILES identities in input-row order."""

    if isinstance(smiles, (str, bytes)):
        raise SelectionV3DataError("smiles must be a sequence of row values, not one string")
    groups: list[str] = []
    for row, value in enumerate(smiles):
        if not isinstance(value, str) or not value.strip():
            raise SelectionV3DataError(f"SMILES row {row} is not a nonempty string")
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise SelectionV3DataError(f"SMILES row {row} cannot be parsed")
        groups.append(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False))
    return np.asarray(groups, dtype=str)


def bemis_murcko_scaffold_groups(smiles: Sequence[str]) -> np.ndarray:
    """Return target-free Bemis-Murcko scaffold IDs in input-row order.

    RDKit returns an empty scaffold for acyclic molecules.  Those rows use their
    full canonical structure so unrelated acyclic molecules are not merged into
    one artificial validation group.
    """

    if isinstance(smiles, (str, bytes)):
        raise SelectionV3DataError("smiles must be a sequence of row values, not one string")
    groups: list[str] = []
    for row, value in enumerate(smiles):
        if not isinstance(value, str) or not value.strip():
            raise SelectionV3DataError(f"SMILES row {row} is not a nonempty string")
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise SelectionV3DataError(f"SMILES row {row} cannot be parsed")
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(
            mol=molecule, includeChirality=True
        )
        groups.append(
            scaffold or f"acyclic:{Chem.MolToSmiles(molecule, canonical=True)}"
        )
    return np.asarray(groups, dtype=str)


def authenticate_feature_artifact(
    matrix_path: str | Path,
    manifest_path: str | Path,
    expected_source_row_identity_sha256: str | None = None,
) -> tuple[sparse.csr_matrix, dict[str, object], dict[str, object]]:
    """Load a feature matrix only after authenticating every artifact contract.

    This is deliberately target-free.  It binds the physical NPZ file, sparse
    matrix semantics, row identity, and the exact 2065-column feature schema.
    """

    feature_path = Path(matrix_path)
    metadata_path = Path(manifest_path)
    if not feature_path.is_file():
        raise SelectionV3DataError(f"feature matrix does not exist: {feature_path}")
    if not metadata_path.is_file():
        raise SelectionV3DataError(f"feature manifest does not exist: {metadata_path}")
    try:
        manifest_raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SelectionV3DataError("feature manifest is not valid UTF-8 JSON") from error
    if not isinstance(manifest_raw, dict):
        raise SelectionV3DataError("feature manifest must be a JSON object")
    manifest: dict[str, object] = manifest_raw
    if set(manifest) != _FEATURE_ARTIFACT_TOP_LEVEL_KEYS:
        raise SelectionV3DataError("feature manifest top-level schema mismatch")
    if manifest["schema_version"] != "qm9-paper-feature-artifact-v1":
        raise SelectionV3DataError("feature manifest schema_version mismatch")

    for field in (
        "source_csv_sha256",
        "source_row_identity_sha256",
        "source_smiles_sha256",
    ):
        _require_sha256(manifest[field], name=field)
    source_row_digest = str(manifest["source_row_identity_sha256"])
    if expected_source_row_identity_sha256 is not None:
        expected = _require_sha256(
            expected_source_row_identity_sha256,
            name="expected_source_row_identity_sha256",
        )
        if source_row_digest != expected:
            raise SelectionV3DataError("source row identity SHA-256 mismatch")

    rows = manifest["rows"]
    if type(rows) is not int or rows <= 0:
        raise SelectionV3DataError("feature manifest rows must be a positive integer")
    expected_schema = engineered_feature_schema(fp_size=2048)
    if manifest["feature_schema"] != expected_schema:
        raise SelectionV3DataError("2065-column feature schema/order/representation mismatch")

    matrix_record = manifest["feature_matrix"]
    if not isinstance(matrix_record, dict) or set(matrix_record) != _FEATURE_MATRIX_MANIFEST_KEYS:
        raise SelectionV3DataError("feature_matrix manifest schema mismatch")
    scaffold_record = manifest["scaffold_groups"]
    if (
        not isinstance(scaffold_record, dict)
        or set(scaffold_record) != _SCAFFOLD_GROUP_MANIFEST_KEYS
    ):
        raise SelectionV3DataError("scaffold_groups manifest schema mismatch")
    for field in ("file_sha256", "semantic_sha256"):
        _require_sha256(matrix_record[field], name=f"feature_matrix.{field}")
    for field in ("file_sha256", "groups_sha256"):
        _require_sha256(scaffold_record[field], name=f"scaffold_groups.{field}")
    if type(scaffold_record["count"]) is not int or scaffold_record["count"] <= 0:
        raise SelectionV3DataError("scaffold_groups.count must be a positive integer")
    if not isinstance(scaffold_record["path"], str) or not scaffold_record["path"]:
        raise SelectionV3DataError("scaffold_groups.path must be a nonempty string")

    recorded_path = matrix_record["path"]
    if not isinstance(recorded_path, str) or not recorded_path:
        raise SelectionV3DataError("feature_matrix.path must be a nonempty string")
    bound_path = (metadata_path.parent / recorded_path).resolve()
    if bound_path != feature_path.resolve():
        raise SelectionV3DataError("feature matrix path is not bound to this manifest")
    physical_digest = _file_sha256(feature_path)
    if physical_digest != matrix_record["file_sha256"]:
        raise SelectionV3DataError("feature matrix file SHA-256 mismatch")

    try:
        loaded = sparse.load_npz(feature_path)
    except (OSError, ValueError) as error:
        raise SelectionV3DataError("feature matrix is not a valid sparse NPZ artifact") from error
    if not sparse.isspmatrix_csr(loaded):
        raise SelectionV3DataError("feature matrix storage must be CSR")
    matrix = loaded.copy()
    if matrix.dtype != np.dtype(np.float64):
        raise SelectionV3DataError("feature matrix dtype must be float64")
    if not np.all(np.isfinite(matrix.data)):
        raise SelectionV3DataError("feature matrix contains nonfinite values")
    expected_shape = [rows, 2065]
    if matrix_record["shape"] != expected_shape or list(matrix.shape) != expected_shape:
        raise SelectionV3DataError("feature matrix shape mismatch")
    recorded_nnz = matrix_record["nnz"]
    if type(recorded_nnz) is not int or recorded_nnz < 0 or matrix.nnz != recorded_nnz:
        raise SelectionV3DataError("feature matrix nnz mismatch")
    semantic_digest = _sparse_semantic_sha256(matrix)
    if semantic_digest != matrix_record["semantic_sha256"]:
        raise SelectionV3DataError("feature matrix semantic SHA-256 mismatch")

    provenance: dict[str, object] = {
        "schema_version": "qm9-selection-v3-authenticated-feature-artifact-v1",
        "targets_accessed": False,
        "artifact_schema_version": manifest["schema_version"],
        "source_row_identity_sha256": source_row_digest,
        "feature_matrix_file_sha256": physical_digest,
        "feature_matrix_semantic_sha256": semantic_digest,
        "feature_schema_sha256": _canonical_json_sha256(expected_schema),
        "shape": list(matrix.shape),
        "nnz": int(matrix.nnz),
        "finite": True,
    }
    return matrix, manifest, provenance


def build_group_folds(
    fixed_train_indices: np.ndarray | Sequence[int],
    group_ids: np.ndarray | Sequence[str],
    n_splits: int,
    seed: int,
    *,
    strategy: str = "identity",
) -> list[dict[str, object]]:
    """Build deterministic folds without splitting identity or scaffold groups."""

    if strategy not in {"identity", "scaffold"}:
        raise SelectionV3DataError("strategy must be 'identity' or 'scaffold'")

    groups = np.asarray(group_ids)
    if groups.ndim != 1 or groups.dtype.kind not in "USO":
        raise SelectionV3DataError("group_ids must be a one-dimensional string array")
    if len(groups) == 0:
        raise SelectionV3DataError("group_ids must not be empty")
    indices = _validate_indices(fixed_train_indices, len(groups), name="fixed_train_indices")
    selected_groups: list[str] = []
    for position, value in enumerate(groups[indices]):
        if not isinstance(value, (str, np.str_)) or not str(value):
            raise SelectionV3DataError(
                f"group_ids contains an invalid value at fixed-train position {position}"
            )
        selected_groups.append(str(value))
    selected = np.asarray(selected_groups, dtype=str)
    unique, inverse, counts = np.unique(selected, return_inverse=True, return_counts=True)
    if type(n_splits) is not int or n_splits < 2 or n_splits > len(unique):
        raise SelectionV3DataError("n_splits must be between 2 and the unique group count")
    if type(seed) is not int:
        raise SelectionV3DataError("seed must be an integer")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(unique))
    ordered = shuffled[np.argsort(-counts[shuffled], kind="stable")]
    fold_rows = np.zeros(n_splits, dtype=np.int64)
    assignments = np.empty(len(unique), dtype=np.int64)
    for group_position in ordered:
        fold = int(np.argmin(fold_rows))
        assignments[group_position] = fold
        fold_rows[fold] += int(counts[group_position])

    folds: list[dict[str, object]] = []
    tune_seen: list[np.ndarray] = []
    for fold in range(n_splits):
        tune_mask = assignments[inverse] == fold
        tune = indices[tune_mask]
        train = indices[~tune_mask]
        if len(tune) == 0 or len(train) == 0:
            raise SelectionV3DataError("group allocation produced an empty train or tune fold")
        train_groups = set(selected[~tune_mask])
        tune_groups = set(selected[tune_mask])
        if train_groups & tune_groups:
            raise SelectionV3DataError(f"a {strategy} group crosses a train/tune fold boundary")
        folds.append(
            {
                "fold": fold,
                "strategy": strategy,
                "train_source_indices": train,
                "tune_source_indices": tune,
                "train_ordered_sha256": _decimal_lines_sha256(train),
                "train_membership_sha256": _decimal_lines_sha256(train, sort=True),
                "tune_ordered_sha256": _decimal_lines_sha256(tune),
                "tune_membership_sha256": _decimal_lines_sha256(tune, sort=True),
                "train_group_count": len(train_groups),
                "tune_group_count": len(tune_groups),
                "tune_group_membership_sha256": _string_lines_sha256(tune_groups, sort=True),
            }
        )
        tune_seen.append(tune)

    tune_union = np.concatenate(tune_seen)
    if len(tune_union) != len(indices) or not np.array_equal(
        np.sort(tune_union), np.sort(indices)
    ):
        raise SelectionV3DataError("tune folds do not cover fixed_train_indices exactly once")
    return folds


def audit_feature_matrix(
    matrix: Any,
    train_indices: np.ndarray | Sequence[int],
    validation_indices: np.ndarray | Sequence[int],
    test_indices: np.ndarray | Sequence[int],
    fingerprint_columns: int = 2048,
) -> dict[str, object]:
    """Return a JSON-safe, target-free audit fitted only on train feature statistics."""

    values = _as_finite_csr(matrix)
    rows, columns = values.shape
    fp_columns = _validate_fingerprint_columns(columns, fingerprint_columns)
    partitions = {
        "train": _validate_indices(train_indices, rows, name="train_indices"),
        "validation": _validate_indices(validation_indices, rows, name="validation_indices"),
        "test": _validate_indices(test_indices, rows, name="test_indices"),
    }
    joined = np.concatenate(tuple(partitions.values()))
    if len(joined) != rows or not np.array_equal(np.sort(joined), np.arange(rows)):
        raise SelectionV3DataError("train/validation/test indices must be a disjoint row cover")

    fingerprint = values[:, :fp_columns]
    train_fp = fingerprint[partitions["train"]]
    train_all = values[partitions["train"]]
    train_df = np.asarray(train_fp.getnnz(axis=0), dtype=np.int64)
    fp_means, fp_variances = _column_mean_variance(train_fp)
    fp_scales = np.sqrt(fp_variances)
    safe_fp_scales = np.where(fp_scales > 0, fp_scales, 1.0)
    fp_max = np.asarray(train_fp.max(axis=0).toarray()).ravel()
    all_means, all_variances = _column_mean_variance(train_all)
    train_constant = np.flatnonzero(all_variances == 0)

    partition_records: dict[str, object] = {}
    for name, indices in partitions.items():
        block = values[indices]
        fp_block = fingerprint[indices]
        row_nnz = np.diff(fp_block.indptr)
        row_count_sum = np.asarray(fp_block.sum(axis=1)).ravel()
        partition_records[name] = {
            "rows": len(indices),
            "ordered_sha256": _decimal_lines_sha256(indices),
            "membership_sha256": _decimal_lines_sha256(indices, sort=True),
            "nnz": int(block.nnz),
            "density": float(block.nnz / (len(indices) * columns)),
            "fingerprint_active_columns_per_row": _quantiles(row_nnz),
            "fingerprint_count_sum_per_row": _quantiles(row_count_sum),
        }

    train_prevalence = train_df / len(partitions["train"])
    drift: dict[str, object] = {}
    train_descriptor_means = all_means[fp_columns:]
    descriptor_scales = np.sqrt(all_variances[fp_columns:])
    safe_descriptor_scales = np.where(descriptor_scales > 0, descriptor_scales, 1.0)
    for name in ("validation", "test"):
        indices = partitions[name]
        block_fp = fingerprint[indices]
        prevalence = np.asarray(block_fp.getnnz(axis=0), dtype=np.float64) / len(indices)
        prevalence_delta = np.abs(prevalence - train_prevalence)
        descriptor = values[indices, fp_columns:]
        descriptor_mean = np.asarray(descriptor.mean(axis=0)).ravel()
        standardized_mean_delta = np.abs(
            (descriptor_mean - train_descriptor_means) / safe_descriptor_scales
        )
        drift[name] = {
            "fingerprint_absolute_prevalence_delta": _quantiles(prevalence_delta),
            "fingerprint_mean_absolute_prevalence_delta": float(prevalence_delta.mean()),
            "fingerprint_train_unseen_active_columns": int(
                np.sum((train_df == 0) & (prevalence > 0))
            ),
            "descriptor_absolute_standardized_mean_delta": _quantiles(
                standardized_mean_delta
            ),
        }

    return {
        "schema_version": "qm9-selection-v3-feature-audit-v1",
        "targets_accessed": False,
        "shape": [rows, columns],
        "nnz": int(values.nnz),
        "density": float(values.nnz / (rows * columns)),
        "finite": True,
        "fingerprint_columns": fp_columns,
        "descriptor_columns": columns - fp_columns,
        "partitions": partition_records,
        "train_fit_statistics": {
            "fit_row_ordered_sha256": _decimal_lines_sha256(partitions["train"]),
            "fit_row_membership_sha256": _decimal_lines_sha256(
                partitions["train"], sort=True
            ),
            "constant_columns": train_constant.astype(int).tolist(),
            "constant_descriptor_columns": train_constant[
                train_constant >= fp_columns
            ].astype(int).tolist(),
            "fingerprint_document_frequency": train_df.astype(int).tolist(),
            "fingerprint_document_frequency_summary": _quantiles(train_df),
            "fingerprint_zero_document_frequency_columns": int(np.sum(train_df == 0)),
            "fingerprint_standard_scale": _quantiles(safe_fp_scales),
            "fingerprint_nonzero_amplification": _quantiles(1.0 / safe_fp_scales),
            "fingerprint_scaled_column_maximum": _quantiles(fp_max / safe_fp_scales),
            "descriptor_mean": train_descriptor_means.astype(float).tolist(),
            "descriptor_scale": safe_descriptor_scales.astype(float).tolist(),
            "descriptor_mean_over_scale": (
                train_descriptor_means / safe_descriptor_scales
            ).astype(float).tolist(),
            "fingerprint_mean_checksum_sha256": hashlib.sha256(
                np.ascontiguousarray(fp_means, dtype=np.float64).tobytes()
            ).hexdigest(),
        },
        "split_drift_from_train_features_only": drift,
    }


def _selected_column_provenance(columns: np.ndarray) -> tuple[list[int], str]:
    selected = np.asarray(columns, dtype=np.int64)
    return selected.astype(int).tolist(), _decimal_lines_sha256(selected)


def _scale_descriptors(
    fit: sparse.csr_matrix,
    apply: sparse.csr_matrix,
    fingerprint_columns: int,
    *,
    drop_constant: bool,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray, dict[str, object]]:
    fit_fp = fit[:, :fingerprint_columns]
    apply_fp = apply[:, :fingerprint_columns]
    fit_descriptor = fit[:, fingerprint_columns:].toarray()
    apply_descriptor = apply[:, fingerprint_columns:].toarray()
    if fit_descriptor.shape[1] == 0:
        selected_descriptor = np.empty(0, dtype=np.int64)
        return fit_fp, apply_fp, np.arange(fingerprint_columns), {
            "descriptor_mean": [],
            "descriptor_scale": [],
            "dropped_constant_descriptor_columns": [],
            "with_mean": True,
        }
    variances = fit_descriptor.var(axis=0)
    keep = variances > 0 if drop_constant else np.ones(len(variances), dtype=bool)
    selected_descriptor = np.flatnonzero(keep)
    dropped = np.flatnonzero(~keep) + fingerprint_columns
    if len(selected_descriptor) == 0:
        return fit_fp, apply_fp, np.arange(fingerprint_columns), {
            "descriptor_mean": [],
            "descriptor_scale": [],
            "dropped_constant_descriptor_columns": dropped.astype(int).tolist(),
            "with_mean": True,
        }
    scaler = StandardScaler(with_mean=True, with_std=True)
    fit_scaled = scaler.fit_transform(fit_descriptor[:, keep])
    apply_scaled = scaler.transform(apply_descriptor[:, keep])
    fit_result = sparse.hstack((fit_fp, sparse.csr_matrix(fit_scaled)), format="csr")
    apply_result = sparse.hstack((apply_fp, sparse.csr_matrix(apply_scaled)), format="csr")
    original_columns = np.concatenate(
        (np.arange(fingerprint_columns), fingerprint_columns + selected_descriptor)
    )
    return fit_result, apply_result, original_columns, {
        "descriptor_mean": scaler.mean_.astype(float).tolist(),
        "descriptor_scale": scaler.scale_.astype(float).tolist(),
        "dropped_constant_descriptor_columns": dropped.astype(int).tolist(),
        "with_mean": True,
    }


def fit_feature_recipe(
    recipe_id: str,
    matrix: Any,
    fit_indices: np.ndarray | Sequence[int],
    apply_indices: np.ndarray | Sequence[int],
    fingerprint_columns: int = 2048,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix, dict[str, object]]:
    """Fit a target-free feature recipe on fit rows and transform fit/apply rows."""

    if recipe_id not in RECIPE_IDS:
        raise SelectionV3DataError(f"unsupported feature recipe: {recipe_id}")
    values = _as_finite_csr(matrix)
    fp_columns = _validate_fingerprint_columns(values.shape[1], fingerprint_columns)
    fit_rows = _validate_indices(fit_indices, values.shape[0], name="fit_indices")
    apply_rows = _validate_indices(apply_indices, values.shape[0], name="apply_indices")
    if np.intersect1d(fit_rows, apply_rows).size:
        raise SelectionV3DataError("fit_indices and apply_indices must be disjoint")
    fit = values[fit_rows]
    apply = values[apply_rows]
    transform: dict[str, object]

    if recipe_id == "all_raw":
        fit_result, apply_result = fit.copy(), apply.copy()
        original_columns = np.arange(values.shape[1])
        transform = {"fingerprint": "raw", "descriptor": "raw"}
    elif recipe_id == "all_sparse_scaled":
        scaler = StandardScaler(with_mean=False, with_std=True).fit(fit)
        fit_result = sparse.csr_matrix(scaler.transform(fit))
        apply_result = sparse.csr_matrix(scaler.transform(apply))
        original_columns = np.arange(values.shape[1])
        transform = {
            "fingerprint": "standard-scale-without-centering",
            "descriptor": "standard-scale-without-centering",
            "mean": scaler.mean_.astype(float).tolist(),
            "scale": scaler.scale_.astype(float).tolist(),
            "with_mean": False,
        }
    else:
        source_fit, source_apply = fit, apply
        fingerprint_transform = "raw"
        if recipe_id == "log_count_descriptor_scaled":
            if np.any(source_fit[:, :fp_columns].data < 0) or np.any(
                source_apply[:, :fp_columns].data < 0
            ):
                raise SelectionV3DataError("log-count recipe requires nonnegative fingerprints")
            source_fit = source_fit.copy()
            source_apply = source_apply.copy()
            fit_fingerprint_values = source_fit.indices < fp_columns
            apply_fingerprint_values = source_apply.indices < fp_columns
            source_fit.data[fit_fingerprint_values] = np.log1p(
                source_fit.data[fit_fingerprint_values]
            )
            source_apply.data[apply_fingerprint_values] = np.log1p(
                source_apply.data[apply_fingerprint_values]
            )
            fingerprint_transform = "log1p-count"
        drop_constant = recipe_id == "drop_constant_descriptor_scaled"
        fit_result, apply_result, original_columns, descriptor_transform = _scale_descriptors(
            source_fit,
            source_apply,
            fp_columns,
            drop_constant=drop_constant,
        )
        transform = {
            "fingerprint": fingerprint_transform,
            "descriptor": "train-fit-centered-standard-scale",
            **descriptor_transform,
        }

    fit_result.sum_duplicates()
    fit_result.eliminate_zeros()
    fit_result.sort_indices()
    apply_result.sum_duplicates()
    apply_result.eliminate_zeros()
    apply_result.sort_indices()
    fit_finite = bool(np.all(np.isfinite(fit_result.data)))
    apply_finite = bool(np.all(np.isfinite(apply_result.data)))
    if not fit_finite or not apply_finite:
        raise SelectionV3DataError("feature recipe produced nonfinite values")
    selected_columns, selected_hash = _selected_column_provenance(original_columns)
    provenance = {
        "schema_version": "qm9-selection-v3-feature-recipe-v1",
        "recipe_id": recipe_id,
        "statistics_scope": "fit_indices_only",
        "input_shape": [int(values.shape[0]), int(values.shape[1])],
        "fit_rows": len(fit_rows),
        "apply_rows": len(apply_rows),
        "fit_row_ordered_sha256": _decimal_lines_sha256(fit_rows),
        "fit_row_membership_sha256": _decimal_lines_sha256(fit_rows, sort=True),
        "apply_row_ordered_sha256": _decimal_lines_sha256(apply_rows),
        "apply_row_membership_sha256": _decimal_lines_sha256(apply_rows, sort=True),
        "fingerprint_columns": fp_columns,
        "selected_original_columns": selected_columns,
        "selected_column_sha256": selected_hash,
        "output_columns": int(fit_result.shape[1]),
        "transform": transform,
        "finite": {"fit": fit_finite, "apply": apply_finite},
    }
    return fit_result, apply_result, provenance
