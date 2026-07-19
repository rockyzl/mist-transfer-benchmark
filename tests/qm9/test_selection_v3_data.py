from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from mist_transfer_benchmark.qm9.engineered_features import engineered_feature_schema
from mist_transfer_benchmark.qm9.paper_evaluation import _array_sha256
from mist_transfer_benchmark.qm9.selection_v3_data import (
    SelectionV3DataError,
    audit_feature_matrix,
    authenticate_feature_artifact,
    bemis_murcko_scaffold_groups,
    build_group_folds,
    canonical_connectivity_groups,
    fit_feature_recipe,
)


def _feature_matrix() -> sparse.csr_matrix:
    return sparse.csr_matrix(
        np.asarray(
            [
                [1, 0, 2, 10, 1, 7],
                [0, 1, 1, 12, 2, 7],
                [1, 1, 0, 14, 3, 7],
                [2, 0, 1, 16, 4, 7],
                [0, 2, 0, 18, 5, 7],
                [1, 0, 1, 100, 6, 7],
                [0, 1, 2, 120, 7, 7],
                [3, 0, 0, 140, 8, 7],
            ],
            dtype=np.float64,
        )
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_artifact(
    directory: Path,
    *,
    matrix: sparse.csr_matrix | None = None,
    source_row_identity_sha256: str = "a" * 64,
) -> tuple[Path, Path, dict[str, object]]:
    if matrix is None:
        matrix = sparse.csr_matrix(
            (
                np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
                (np.asarray([0, 0, 1, 2]), np.asarray([0, 2048, 1, 2064])),
            ),
            shape=(3, 2065),
            dtype=np.float64,
        )
    matrix_path = directory / "feature_matrix.npz"
    manifest_path = directory / "manifest.json"
    sparse.save_npz(matrix_path, matrix, compressed=True)
    manifest: dict[str, object] = {
        "schema_version": "qm9-paper-feature-artifact-v1",
        "source_csv_sha256": "c" * 64,
        "source_row_identity_sha256": source_row_identity_sha256,
        "source_smiles_sha256": "d" * 64,
        "rows": matrix.shape[0],
        "feature_schema": engineered_feature_schema(),
        "feature_matrix": {
            "path": matrix_path.name,
            "file_sha256": _file_sha256(matrix_path),
            "semantic_sha256": _array_sha256(matrix),
            "shape": list(matrix.shape),
            "nnz": int(matrix.nnz),
        },
        "scaffold_groups": {
            "path": "scaffold_group_ids.npy",
            "file_sha256": "e" * 64,
            "groups_sha256": "f" * 64,
            "count": matrix.shape[0],
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return matrix_path, manifest_path, manifest


def _rewrite_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_canonical_connectivity_groups_collapse_order_and_stereochemistry() -> None:
    groups = canonical_connectivity_groups(
        ["CCO", "OCC", "F[C@H](Cl)Br", "F[C@@H](Cl)Br", "CCN"]
    )
    assert groups.dtype.kind == "U"
    assert groups[0] == groups[1]
    assert groups[2] == groups[3]
    assert groups[0] != groups[4]
    with pytest.raises(SelectionV3DataError, match="cannot be parsed"):
        canonical_connectivity_groups(["not-a-smiles"])
    with pytest.raises(SelectionV3DataError, match="sequence"):
        canonical_connectivity_groups("CCO")


def test_bemis_murcko_scaffolds_group_ring_cores_but_not_acyclic_molecules() -> None:
    groups = bemis_murcko_scaffold_groups(
        ["c1ccccc1C", "Oc1ccccc1", "C1CCCCC1", "CCO", "OCC", "CCN"]
    )
    assert groups.dtype.kind == "U"
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]
    assert groups[3] == groups[4]
    assert groups[3].startswith("acyclic:")
    assert groups[3] != groups[5]
    with pytest.raises(SelectionV3DataError, match="cannot be parsed"):
        bemis_murcko_scaffold_groups(["not-a-smiles"])
    with pytest.raises(SelectionV3DataError, match="sequence"):
        bemis_murcko_scaffold_groups("CCO")


def test_authenticate_feature_artifact_accepts_exact_generator_contract(
    tmp_path: Path,
) -> None:
    matrix_path, manifest_path, expected_manifest = _write_artifact(tmp_path)
    matrix, manifest, provenance = authenticate_feature_artifact(
        matrix_path,
        manifest_path,
        expected_source_row_identity_sha256="a" * 64,
    )
    assert sparse.isspmatrix_csr(matrix)
    assert matrix.shape == (3, 2065)
    assert manifest == expected_manifest
    assert provenance["targets_accessed"] is False
    assert provenance["source_row_identity_sha256"] == "a" * 64
    json.dumps(provenance, allow_nan=False)


def test_authenticate_feature_artifact_rejects_file_hash_tampering(tmp_path: Path) -> None:
    matrix_path, manifest_path, manifest = _write_artifact(tmp_path)
    changed = deepcopy(manifest)
    changed["feature_matrix"]["file_sha256"] = "0" * 64
    _rewrite_manifest(manifest_path, changed)
    with pytest.raises(SelectionV3DataError, match="file SHA-256 mismatch"):
        authenticate_feature_artifact(matrix_path, manifest_path)


def test_authenticate_feature_artifact_rejects_row_reordering(tmp_path: Path) -> None:
    matrix_path, manifest_path, manifest = _write_artifact(tmp_path)
    matrix = sparse.load_npz(matrix_path)[[2, 0, 1]].tocsr()
    sparse.save_npz(matrix_path, matrix, compressed=True)
    changed = deepcopy(manifest)
    changed["source_row_identity_sha256"] = "b" * 64
    changed["feature_matrix"]["file_sha256"] = _file_sha256(matrix_path)
    changed["feature_matrix"]["semantic_sha256"] = _array_sha256(matrix)
    _rewrite_manifest(manifest_path, changed)
    with pytest.raises(SelectionV3DataError, match="source row identity"):
        authenticate_feature_artifact(
            matrix_path,
            manifest_path,
            expected_source_row_identity_sha256="a" * 64,
        )


def test_authenticate_feature_artifact_rejects_schema_order_and_nnz_mismatch(
    tmp_path: Path,
) -> None:
    matrix_path, manifest_path, manifest = _write_artifact(tmp_path)
    schema_changed = deepcopy(manifest)
    descriptors = schema_changed["feature_schema"]["global_descriptors"]
    descriptors[0], descriptors[1] = descriptors[1], descriptors[0]
    _rewrite_manifest(manifest_path, schema_changed)
    with pytest.raises(SelectionV3DataError, match="schema/order/representation"):
        authenticate_feature_artifact(matrix_path, manifest_path)

    nnz_changed = deepcopy(manifest)
    nnz_changed["feature_matrix"]["nnz"] += 1
    _rewrite_manifest(manifest_path, nnz_changed)
    with pytest.raises(SelectionV3DataError, match="nnz mismatch"):
        authenticate_feature_artifact(matrix_path, manifest_path)


def test_authenticate_feature_artifact_rejects_semantic_mismatch(tmp_path: Path) -> None:
    matrix_path, manifest_path, manifest = _write_artifact(tmp_path)
    changed = deepcopy(manifest)
    changed["feature_matrix"]["semantic_sha256"] = "1" * 64
    _rewrite_manifest(manifest_path, changed)
    with pytest.raises(SelectionV3DataError, match="semantic SHA-256 mismatch"):
        authenticate_feature_artifact(matrix_path, manifest_path)


def test_group_folds_are_deterministic_covering_and_group_isolated() -> None:
    fixed = np.asarray([9, 2, 8, 1, 7, 0, 6, 3], dtype=np.int64)
    groups = np.asarray(["a", "b", "b", "c", "unused", "unused", "d", "d", "e", "a"])
    first = build_group_folds(fixed, groups, n_splits=3, seed=41, strategy="scaffold")
    second = build_group_folds(fixed, groups, n_splits=3, seed=41, strategy="scaffold")
    assert len(first) == 3
    for left, right in zip(first, second, strict=True):
        assert left.keys() == right.keys()
        assert left["strategy"] == "scaffold"
        assert np.array_equal(left["train_source_indices"], right["train_source_indices"])
        assert np.array_equal(left["tune_source_indices"], right["tune_source_indices"])
        assert left["train_membership_sha256"] == right["train_membership_sha256"]
        train_groups = set(groups[left["train_source_indices"]])
        tune_groups = set(groups[left["tune_source_indices"]])
        assert not train_groups & tune_groups
        assert set(left["train_source_indices"]) | set(left["tune_source_indices"]) == set(
            fixed
        )
    tune = np.concatenate([fold["tune_source_indices"] for fold in first])
    assert sorted(tune.tolist()) == sorted(fixed.tolist())


def test_group_fold_contract_rejects_bad_indices_and_invalid_groups() -> None:
    groups = np.asarray(["a", "a", "b", "c"])
    with pytest.raises(SelectionV3DataError, match="duplicate"):
        build_group_folds([0, 0, 2], groups, 2, 1)
    with pytest.raises(SelectionV3DataError, match="out-of-range"):
        build_group_folds([0, 4], groups, 2, 1)
    with pytest.raises(SelectionV3DataError, match="unique group count"):
        build_group_folds([0, 1], groups, 2, 1)
    bad_groups = np.asarray(["a", None, "b"], dtype=object)
    with pytest.raises(SelectionV3DataError, match="invalid value"):
        build_group_folds([0, 1, 2], bad_groups, 2, 1)
    with pytest.raises(SelectionV3DataError, match="strategy"):
        build_group_folds([0, 2, 3], groups, 2, 1, strategy="random")


def test_feature_audit_is_json_safe_target_free_and_train_fitted() -> None:
    matrix = _feature_matrix()
    audit = audit_feature_matrix(matrix, [0, 1, 2, 3], [4, 5], [6, 7], 3)
    json.dumps(audit, allow_nan=False)
    assert audit["targets_accessed"] is False
    assert audit["shape"] == [8, 6]
    assert audit["fingerprint_columns"] == 3
    assert audit["descriptor_columns"] == 3
    assert audit["partitions"]["train"]["rows"] == 4
    assert audit["train_fit_statistics"]["fingerprint_document_frequency"] == [3, 2, 3]
    assert audit["train_fit_statistics"]["constant_descriptor_columns"] == [5]
    assert audit["split_drift_from_train_features_only"]["test"][
        "fingerprint_mean_absolute_prevalence_delta"
    ] >= 0


def test_feature_audit_rejects_nonfinite_and_partition_tampering() -> None:
    matrix = _feature_matrix()
    bad = matrix.copy()
    bad.data[0] = np.inf
    with pytest.raises(SelectionV3DataError, match="nonfinite"):
        audit_feature_matrix(bad, [0, 1, 2, 3], [4, 5], [6, 7], 3)
    with pytest.raises(SelectionV3DataError, match="disjoint row cover"):
        audit_feature_matrix(matrix, [0, 1, 2, 3], [3, 4], [5, 6, 7], 3)
    with pytest.raises(SelectionV3DataError, match="duplicate"):
        audit_feature_matrix(matrix, [0, 0, 1, 2], [3, 4], [5, 6, 7], 3)


def test_feature_recipes_only_change_the_declared_blocks() -> None:
    matrix = _feature_matrix()
    fit = np.arange(5)
    apply = np.arange(5, 8)
    raw_fit, raw_apply, raw_provenance = fit_feature_recipe(
        "all_raw", matrix, fit, apply, 3
    )
    descriptor_fit, descriptor_apply, descriptor_provenance = fit_feature_recipe(
        "descriptor_scaled_only", matrix, fit, apply, 3
    )
    log_fit, log_apply, log_provenance = fit_feature_recipe(
        "log_count_descriptor_scaled", matrix, fit, apply, 3
    )
    dropped_fit, dropped_apply, dropped_provenance = fit_feature_recipe(
        "drop_constant_descriptor_scaled", matrix, fit, apply, 3
    )
    sparse_fit, sparse_apply, sparse_provenance = fit_feature_recipe(
        "all_sparse_scaled", matrix, fit, apply, 3
    )

    dense = matrix.toarray()
    assert np.array_equal(raw_fit.toarray(), dense[fit])
    assert np.array_equal(raw_apply.toarray(), dense[apply])
    assert np.array_equal(descriptor_fit[:, :3].toarray(), dense[fit, :3])
    assert np.array_equal(descriptor_apply[:, :3].toarray(), dense[apply, :3])
    assert np.allclose(descriptor_fit[:, 3:].toarray().mean(axis=0), 0.0)
    assert np.allclose(descriptor_fit[:, 3:5].toarray().std(axis=0), 1.0)
    assert np.allclose(log_fit[:, :3].toarray(), np.log1p(dense[fit, :3]))
    assert np.allclose(log_apply[:, :3].toarray(), np.log1p(dense[apply, :3]))
    assert np.allclose(log_fit[:, 3:].toarray(), descriptor_fit[:, 3:].toarray())
    assert dropped_fit.shape == (5, 5)
    assert dropped_apply.shape == (3, 5)
    assert dropped_provenance["transform"]["dropped_constant_descriptor_columns"] == [5]
    expected_scaler = StandardScaler(with_mean=False).fit(matrix[fit])
    assert np.allclose(sparse_fit.toarray(), expected_scaler.transform(matrix[fit]).toarray())
    assert np.allclose(sparse_apply.toarray(), expected_scaler.transform(matrix[apply]).toarray())

    for provenance in (
        raw_provenance,
        descriptor_provenance,
        log_provenance,
        dropped_provenance,
        sparse_provenance,
    ):
        json.dumps(provenance, allow_nan=False)
        assert provenance["statistics_scope"] == "fit_indices_only"
        assert provenance["finite"] == {"fit": True, "apply": True}
        assert len(provenance["selected_column_sha256"]) == 64


def test_recipe_statistics_ignore_apply_row_tampering() -> None:
    matrix = _feature_matrix()
    tampered = matrix.copy().tolil()
    tampered[5:, 3:] = np.asarray([[1000, 600, 7], [1200, 700, 7], [1400, 800, 7]])
    tampered = tampered.tocsr()
    base_fit, base_apply, base = fit_feature_recipe(
        "descriptor_scaled_only", matrix, [0, 1, 2, 3, 4], [5, 6, 7], 3
    )
    changed_fit, changed_apply, changed = fit_feature_recipe(
        "descriptor_scaled_only", tampered, [0, 1, 2, 3, 4], [5, 6, 7], 3
    )
    assert np.array_equal(base_fit.toarray(), changed_fit.toarray())
    assert base["transform"] == changed["transform"]
    assert not np.array_equal(base_apply.toarray(), changed_apply.toarray())


def test_feature_recipe_rejects_leakage_bad_values_and_unknown_recipe() -> None:
    matrix = _feature_matrix()
    with pytest.raises(SelectionV3DataError, match="must be disjoint"):
        fit_feature_recipe("all_raw", matrix, [0, 1], [1, 2], 3)
    bad = matrix.copy()
    bad.data[0] = np.nan
    with pytest.raises(SelectionV3DataError, match="nonfinite"):
        fit_feature_recipe("all_raw", bad, [0, 1], [2, 3], 3)
    negative = matrix.copy().tolil()
    negative[0, 0] = -1
    with pytest.raises(SelectionV3DataError, match="nonnegative"):
        fit_feature_recipe(
            "log_count_descriptor_scaled", negative.tocsr(), [0, 1], [2, 3], 3
        )
    with pytest.raises(SelectionV3DataError, match="unsupported"):
        fit_feature_recipe("unknown", matrix, [0, 1], [2, 3], 3)


def test_drop_constant_recipe_allows_every_descriptor_to_be_constant() -> None:
    matrix = sparse.csr_matrix([[1, 0, 7], [0, 1, 7], [1, 1, 7], [2, 0, 7]])
    fit, apply, provenance = fit_feature_recipe(
        "drop_constant_descriptor_scaled", matrix, [0, 1, 2], [3], 2
    )
    assert fit.shape == (3, 2)
    assert apply.shape == (1, 2)
    assert provenance["selected_original_columns"] == [0, 1]
    assert provenance["transform"]["dropped_constant_descriptor_columns"] == [2]
