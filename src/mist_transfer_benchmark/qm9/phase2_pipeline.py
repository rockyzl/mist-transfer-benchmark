"""Staged Phase 2 classical execution for the authenticated QM9 reconstruction."""

from __future__ import annotations

import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import tomllib
from scipy import sparse

from .data import load_qm9_identities
from .download import assert_source_unchanged, copy_validated_source
from .io import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_hash,
    sha256_file,
)
from .phase2_contract import verify_phase1_evidence
from .phase2_features import (
    build_ecfp4_csr,
    contract_from_config,
    csr_canonical_sha256,
    feature_manifest,
    save_csr_atomic,
    validate_feature_matrix,
)
from .phase2_metrics import native_metrics
from .phase2_output import (
    OWNER_MARKER,
    OWNER_PAYLOAD,
    discard_phase2_workspace,
    finalize_phase2_workspace,
    prepare_phase2_workspace,
    write_phase2_owner,
)
from .phase2_scaler import fit_frozen_scaler
from .phase2_selection import (
    random_forest_candidates,
    ridge_candidates,
    select_first_minimum,
)
from .phase2_similarity import tanimoto_1nn_predict
from .phase2_targets import load_targets_for_indices
from .phase2_test_lock import (
    TestLabelGate,
    complete_test_reservation,
    freeze_selection,
    reserve_test_once,
)
from .provenance import assert_code_provenance_unchanged, capture_code_provenance


def run_phase2_feature_stage(
    *,
    config_path: str | Path,
    cache_dir: str | Path,
    phase1_dir: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
    progress=print,
) -> dict[str, object]:
    """Authenticate Phase 1 and persist the complete frozen ECFP4 CSR artifact."""

    started = time.monotonic()
    repo_root = Path(__file__).resolve().parents[3]
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    source_path = Path(cache_dir)
    if not source_path.is_absolute():
        source_path = repo_root / source_path
    source_path = source_path.resolve(strict=True) / "qm9.csv"
    evidence = verify_phase1_evidence(
        config, _resolve_repo_path(repo_root, phase1_dir), source_path
    )
    provenance = capture_code_provenance(repo_root, config_path)
    workspace = prepare_phase2_workspace(output_dir, repo_root, overwrite=overwrite)
    try:
        private_source = workspace.staging_dir / "source.snapshot.csv"
        private_snapshot = copy_validated_source(
            source_path,
            private_source,
            evidence.source_snapshot,
            expected_bytes=evidence.source_snapshot.bytes,
            expected_sha256=evidence.source_snapshot.sha256,
        )
        data = load_qm9_identities(private_source)
        contract = contract_from_config(config)
        matrix = build_ecfp4_csr(data.source_smiles, contract, progress=progress)
        matrix_record = save_csr_atomic(workspace.staging_dir / "feature_matrix.npz", matrix)
        manifest = feature_manifest(
            contract,
            matrix_record,
            source_row_identity_sha256=data.row_identity_sha256,
            source_smiles_sha256=data.raw_smiles_sha256,
        )
        manifest["runtime_seconds"] = time.monotonic() - started
        manifest["phase1_run_sha256"] = evidence.phase1_run_sha256
        manifest["phase1_code_provenance_aggregate_sha256"] = (
            evidence.code_provenance_aggregate_sha256
        )
        atomic_write_json(workspace.staging_dir / "feature_manifest.json", manifest, mode=0o600)
        atomic_write_json(
            workspace.staging_dir / "phase1_verification.json",
            {
                "schema_version": "qm9-phase1-verification-for-phase2-v1",
                "phase1_run_sha256": evidence.phase1_run_sha256,
                "source_sha256": evidence.source_snapshot.sha256,
                "source_bytes": evidence.source_snapshot.bytes,
                "split_counts": evidence.split.counts(),
                "split_ordered_sha256": evidence.split.ordered_hashes(),
                "duplicate_clean_test_rows": len(evidence.duplicate_clean_test),
                "phase1_code_provenance_aggregate_sha256": (
                    evidence.code_provenance_aggregate_sha256
                ),
            },
            mode=0o600,
        )
        atomic_write_json(
            workspace.staging_dir / "code_provenance.json", provenance, mode=0o600
        )
        assert_source_unchanged(
            private_source,
            private_snapshot,
            expected_bytes=evidence.source_snapshot.bytes,
            expected_sha256=evidence.source_snapshot.sha256,
        )
        assert_source_unchanged(
            source_path,
            evidence.source_snapshot,
            expected_bytes=evidence.source_snapshot.bytes,
            expected_sha256=evidence.source_snapshot.sha256,
        )
        assert_code_provenance_unchanged(provenance, repo_root, config_path)
        private_source.unlink()
        write_phase2_owner(workspace.staging_dir)
        finalize_phase2_workspace(workspace)
        manifest["feature_manifest_sha256"] = sha256_file(
            workspace.output_dir / "feature_manifest.json"
        )
        manifest["output_dir"] = str(workspace.output_dir)
        return manifest
    finally:
        discard_phase2_workspace(workspace)


def _rss_gib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024**3 if sys.platform == "darwin" else 1024**2
    return float(value) / divisor


def _resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _load_authenticated_features(
    feature_dir: Path,
    config: dict[str, object],
    *,
    phase1_run_sha256: str,
    source_row_identity_sha256: str,
    source_smiles_sha256: str,
) -> tuple[sparse.csr_matrix, dict[str, object]]:
    directory = feature_dir.resolve(strict=True)
    marker = directory / OWNER_MARKER
    if marker.is_symlink() or json.loads(marker.read_text(encoding="utf-8")) != OWNER_PAYLOAD:
        raise ValueError("feature-stage ownership marker is invalid")
    manifest_path = directory / "feature_manifest.json"
    matrix_path = directory / "feature_matrix.npz"
    if manifest_path.is_symlink() or matrix_path.is_symlink():
        raise ValueError("feature-stage artifacts must not be symlinks")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = contract_from_config(config)
    if manifest.get("generator_canonical_json_sha256") != canonical_hash(contract.manifest()):
        raise ValueError("feature generator contract differs from the frozen TOML")
    if manifest.get("phase1_run_sha256") != phase1_run_sha256:
        raise ValueError("feature artifact was not derived from the authenticated Phase 1 run")
    if manifest.get("source_row_identity_sha256") != source_row_identity_sha256:
        raise ValueError("feature row identity differs from the authenticated source")
    if manifest.get("source_smiles_sha256") != source_smiles_sha256:
        raise ValueError("feature SMILES sequence differs from the authenticated source")
    record = manifest.get("matrix", {})
    if sha256_file(matrix_path) != record.get("file_sha256"):
        raise ValueError("feature matrix file SHA-256 differs")
    matrix = sparse.load_npz(matrix_path)
    validate_feature_matrix(matrix, rows=133_885, columns=contract.fp_size)
    if csr_canonical_sha256(matrix) != record.get("canonical_csr_sha256"):
        raise ValueError("feature matrix canonical CSR SHA-256 differs")
    return matrix, manifest


def _phase1_verification_payload(evidence) -> dict[str, object]:
    return {
        "schema_version": "qm9-phase1-verification-for-phase2-v1",
        "phase1_run_sha256": evidence.phase1_run_sha256,
        "source_sha256": evidence.source_snapshot.sha256,
        "source_bytes": evidence.source_snapshot.bytes,
        "split_counts": evidence.split.counts(),
        "split_ordered_sha256": evidence.split.ordered_hashes(),
        "duplicate_clean_test_rows": len(evidence.duplicate_clean_test),
        "phase1_code_provenance_aggregate_sha256": (
            evidence.code_provenance_aggregate_sha256
        ),
    }


def _prediction_rows(
    data,
    test_indices: np.ndarray,
    y_test: np.ndarray,
    predictions: dict[str, np.ndarray],
    clean_indices: np.ndarray,
):
    clean = {int(index) for index in clean_indices}
    for method, values in predictions.items():
        for position, source_index in enumerate(test_indices):
            index = int(source_index)
            yield {
                "method": method,
                "source_row_index": index,
                "record_id": data.record_id(index),
                "duplicate_clean_test": index in clean,
                "target_order": [
                    "mu",
                    "alpha",
                    "homo",
                    "lumo",
                    "gap",
                    "r2",
                    "zpve",
                    "u0",
                    "u298",
                    "h298",
                    "g298",
                    "cv",
                ],
                "observed": y_test[position].tolist(),
                "predicted": values[position].tolist(),
            }


def run_phase2_classical(
    *,
    config_path: str | Path,
    cache_dir: str | Path,
    phase1_dir: str | Path,
    feature_dir: str | Path,
    output_dir: str | Path,
    overwrite: bool = False,
    run_random_forest: bool = False,
    progress=print,
) -> dict[str, object]:
    """Select on validation, durably lock, then evaluate test labels exactly once."""

    started = time.monotonic()
    repo_root = Path(__file__).resolve().parents[3]
    config_path = _resolve_repo_path(repo_root, config_path).resolve(strict=True)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    cache = _resolve_repo_path(repo_root, cache_dir).resolve(strict=True)
    source_path = cache / "qm9.csv"
    evidence = verify_phase1_evidence(
        config, _resolve_repo_path(repo_root, phase1_dir), source_path
    )
    provenance = capture_code_provenance(repo_root, config_path)
    workspace = prepare_phase2_workspace(output_dir, repo_root, overwrite=overwrite)
    reservation_path: Path | None = None
    try:
        protocol_snapshot_path = workspace.staging_dir / "protocol_config.snapshot.toml"
        atomic_write_bytes(protocol_snapshot_path, config_path.read_bytes(), mode=0o600)
        protocol_snapshot_sha256 = sha256_file(protocol_snapshot_path)
        private_source = workspace.staging_dir / "source.snapshot.csv"
        private_snapshot = copy_validated_source(
            source_path,
            private_source,
            evidence.source_snapshot,
            expected_bytes=evidence.source_snapshot.bytes,
            expected_sha256=evidence.source_snapshot.sha256,
        )
        data = load_qm9_identities(private_source)
        matrix, feature = _load_authenticated_features(
            _resolve_repo_path(repo_root, feature_dir),
            config,
            phase1_run_sha256=evidence.phase1_run_sha256,
            source_row_identity_sha256=data.row_identity_sha256,
            source_smiles_sha256=data.raw_smiles_sha256,
        )
        progress("loading train/validation labels only; test labels remain gated")
        y_train = load_targets_for_indices(private_source, evidence.split.train, data)
        y_validation = load_targets_for_indices(private_source, evidence.split.validation, data)
        scaler, scaler_artifact = fit_frozen_scaler(config, y_train)
        y_train_scaled = scaler.transform(y_train)
        x_train = matrix[evidence.split.train]
        x_validation = matrix[evidence.split.validation]

        mean_start = time.monotonic()
        train_mean = np.mean(y_train, axis=0)
        mean_prediction = np.broadcast_to(train_mean, y_validation.shape).copy()
        mean_metrics = native_metrics(y_validation, mean_prediction, scaler.scale_)
        mean_result = {
            "method": "training-target-means",
            "runtime_seconds": time.monotonic() - mean_start,
            "metrics": mean_metrics,
        }

        ridge_models = {}
        ridge_results: list[dict[str, object]] = []
        ridge_specs = ridge_candidates(config)
        for candidate in ridge_specs:
            progress(f"fitting Ridge validation candidate {candidate.candidate_id}")
            candidate_start = time.monotonic()
            candidate.estimator.fit(x_train, y_train_scaled)
            prediction = scaler.inverse_transform(candidate.estimator.predict(x_validation))
            metrics = native_metrics(y_validation, prediction, scaler.scale_)
            ridge_models[candidate.candidate_id] = candidate.estimator
            ridge_results.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "parameters": candidate.parameters,
                    "parameters_sha256": candidate.parameters_sha256,
                    "runtime_seconds": time.monotonic() - candidate_start,
                    "mean_normalized_mae_across_12_targets": metrics[
                        "mean_normalized_mae_across_12_targets"
                    ],
                    "metrics": metrics,
                }
            )
        ridge_order = [candidate.candidate_id for candidate in ridge_specs]
        selected_ridge = select_first_minimum(ridge_results, ridge_order)
        progress(f"selected Ridge candidate {selected_ridge['candidate_id']}")

        similarity_start = time.monotonic()
        try:
            similarity_prediction, _, _ = tanimoto_1nn_predict(
                matrix,
                evidence.split.train,
                y_train,
                evidence.split.validation,
                progress=progress,
            )
            similarity_result = {
                "method": "ecfp-tanimoto-1nn",
                "status": "complete",
                "runtime_seconds": time.monotonic() - similarity_start,
                "metrics": native_metrics(y_validation, similarity_prediction, scaler.scale_),
            }
        except Exception as error:
            similarity_result = {
                "method": "ecfp-tanimoto-1nn",
                "status": "stopped-after-bounded-attempt",
                "runtime_seconds": time.monotonic() - similarity_start,
                "failure": {"type": type(error).__name__, "message": str(error)},
            }

        rf_models = {}
        rf_results: list[dict[str, object]] = []
        rf_status = "not-attempted-by-request"
        rf_failure: dict[str, str] | None = None
        if run_random_forest:
            rf_status = "running"
            try:
                for candidate in random_forest_candidates(config):
                    progress(f"fitting random-forest validation candidate {candidate.candidate_id}")
                    candidate_start = time.monotonic()
                    candidate.estimator.fit(x_train, y_train_scaled)
                    prediction = scaler.inverse_transform(
                        candidate.estimator.predict(x_validation)
                    )
                    metrics = native_metrics(y_validation, prediction, scaler.scale_)
                    elapsed = time.monotonic() - candidate_start
                    rf_models[candidate.candidate_id] = candidate.estimator
                    rf_results.append(
                        {
                            "candidate_id": candidate.candidate_id,
                            "parameters": candidate.parameters,
                            "parameters_sha256": candidate.parameters_sha256,
                            "runtime_seconds": elapsed,
                            "mean_normalized_mae_across_12_targets": metrics[
                                "mean_normalized_mae_across_12_targets"
                            ],
                            "metrics": metrics,
                        }
                    )
                    if elapsed > (
                        config["resource_budget"][
                            "max_wall_clock_hours_per_validation_candidate"
                        ]
                        * 3600
                    ):
                        raise RuntimeError("random-forest candidate exceeded wall-clock ceiling")
                    if _rss_gib() > config["resource_budget"][
                        "max_parent_process_peak_rss_gib"
                    ]:
                        raise MemoryError("random-forest candidate exceeded peak-RSS ceiling")
                rf_status = "complete"
            except Exception as error:
                rf_status = "stopped-after-bounded-attempt"
                rf_failure = {"type": type(error).__name__, "message": str(error)}
        selected_rf = None
        if rf_status == "complete":
            rf_order = config["classical"]["random_forest"]["candidate_order"]
            selected_rf = select_first_minimum(rf_results, rf_order)

        validation = {
            "schema_version": "qm9-phase2-validation-v1",
            "test_labels_loaded": False,
            "selection_metric": config["classical"]["selection_metric"],
            "training_target_means": mean_result,
            "ecfp_tanimoto_1nn": similarity_result,
            "ridge": {
                "candidate_order": ridge_order,
                "candidates": ridge_results,
                "selected": selected_ridge,
            },
            "random_forest": {
                "status": rf_status,
                "candidates_completed": rf_results,
                "failure": rf_failure,
                "selected": selected_rf,
            },
        }
        validation_path = workspace.staging_dir / "validation_metrics.json"
        atomic_write_json(validation_path, validation, mode=0o600)
        atomic_write_json(
            workspace.staging_dir / "random_forest_attempt.json",
            validation["random_forest"],
            mode=0o600,
        )
        atomic_write_json(workspace.staging_dir / "scaler.json", scaler_artifact, mode=0o600)
        selection = freeze_selection(
            {
                "phase1_run_sha256": evidence.phase1_run_sha256,
                "protocol_config_sha256": protocol_snapshot_sha256,
                "code_provenance_aggregate_sha256": provenance["aggregate_sha256"],
                "feature_canonical_csr_sha256": feature["matrix"][
                    "canonical_csr_sha256"
                ],
                "scaler_fitted_state_sha256": scaler_artifact[
                    "fitted_state_canonical_json_sha256"
                ],
                "validation_scientific_basis": {
                    "training-target-means": mean_metrics[
                        "mean_normalized_mae_across_12_targets"
                    ],
                    "ecfp-tanimoto-1nn": {
                        "status": similarity_result["status"],
                        "mean_normalized_mae_across_12_targets": (
                            similarity_result.get("metrics", {}).get(
                                "mean_normalized_mae_across_12_targets"
                            )
                        ),
                    },
                    "ridge": [
                        {
                            "candidate_id": item["candidate_id"],
                            "parameters_sha256": item["parameters_sha256"],
                            "mean_normalized_mae_across_12_targets": item[
                                "mean_normalized_mae_across_12_targets"
                            ],
                        }
                        for item in ridge_results
                    ],
                    "random_forest": {
                        "status": rf_status,
                        "completed": [
                            {
                                "candidate_id": item["candidate_id"],
                                "parameters_sha256": item["parameters_sha256"],
                                "mean_normalized_mae_across_12_targets": item[
                                    "mean_normalized_mae_across_12_targets"
                                ],
                            }
                            for item in rf_results
                        ],
                    },
                },
                "selected": {
                    "training-target-means": "fixed-control",
                    "ecfp-tanimoto-1nn": (
                        "fixed-control"
                        if similarity_result["status"] == "complete"
                        else {"excluded_status": similarity_result["status"]}
                    ),
                    "ridge": {
                        "candidate_id": selected_ridge["candidate_id"],
                        "parameters_sha256": selected_ridge["parameters_sha256"],
                    },
                    "random_forest": (
                        {
                            "candidate_id": selected_rf["candidate_id"],
                            "parameters_sha256": selected_rf["parameters_sha256"],
                        }
                        if selected_rf is not None
                        else {"excluded_status": rf_status, "failure": rf_failure}
                    ),
                },
                "test_rows": len(evidence.split.test),
                "duplicate_clean_test_rows": len(evidence.duplicate_clean_test),
            }
        )
        selection_path = workspace.staging_dir / "selection_lock.json"
        atomic_write_json(selection_path, selection, mode=0o600)
        selection_sha = sha256_file(selection_path)
        atomic_write_bytes(
            workspace.staging_dir / "selection_lock.sha256",
            f"{selection_sha}  selection_lock.json\n".encode("ascii"),
            mode=0o600,
        )
        assert_code_provenance_unchanged(provenance, repo_root, config_path)
        reservation_path = reserve_test_once(cache / "test-locks", selection)
        gate = TestLabelGate()
        gate.authorize(reservation_path, selection)
        gate.require_authorized(selection)
        progress(f"selection locked as {selection['selection_fingerprint']}; loading test labels")
        y_test = load_targets_for_indices(private_source, evidence.split.test, data)

        predictions: dict[str, np.ndarray] = {
            "training-target-means": np.broadcast_to(train_mean, y_test.shape).copy(),
            "ridge": scaler.inverse_transform(
                ridge_models[str(selected_ridge["candidate_id"])].predict(
                    matrix[evidence.split.test]
                )
            ),
        }
        if similarity_result["status"] == "complete":
            test_similarity_prediction, _, _ = tanimoto_1nn_predict(
                matrix,
                evidence.split.train,
                y_train,
                evidence.split.test,
                progress=progress,
            )
            predictions["ecfp-tanimoto-1nn"] = test_similarity_prediction
        if selected_rf is not None:
            predictions["random_forest"] = scaler.inverse_transform(
                rf_models[str(selected_rf["candidate_id"])].predict(matrix[evidence.split.test])
            )
        clean_set = {int(index) for index in evidence.duplicate_clean_test}
        clean_mask = np.asarray(
            [int(index) in clean_set for index in evidence.split.test], dtype=bool
        )
        test_metrics = {
            "schema_version": "qm9-phase2-test-metrics-v1",
            "selection_fingerprint": selection["selection_fingerprint"],
            "test_rows": len(y_test),
            "duplicate_clean_test_rows": int(np.sum(clean_mask)),
            "methods": {
                method: {
                    "full_test": native_metrics(y_test, values, scaler.scale_),
                    "duplicate_clean_test": native_metrics(
                        y_test[clean_mask], values[clean_mask], scaler.scale_
                    ),
                }
                for method, values in predictions.items()
            },
        }
        metrics_path = workspace.staging_dir / "test_metrics.json"
        atomic_write_json(metrics_path, test_metrics, mode=0o600)
        predictions_path = workspace.staging_dir / "predictions.jsonl"
        prediction_rows = atomic_write_jsonl(
            predictions_path,
            _prediction_rows(
                data,
                evidence.split.test,
                y_test,
                predictions,
                evidence.duplicate_clean_test,
            ),
            mode=0o600,
        )
        feature_source = _resolve_repo_path(repo_root, feature_dir).resolve(strict=True)
        atomic_write_bytes(
            workspace.staging_dir / "feature_matrix.npz",
            (feature_source / "feature_matrix.npz").read_bytes(),
            mode=0o600,
        )
        atomic_write_bytes(
            workspace.staging_dir / "feature_manifest.json",
            (feature_source / "feature_manifest.json").read_bytes(),
            mode=0o600,
        )
        atomic_write_json(
            workspace.staging_dir / "phase1_verification.json",
            _phase1_verification_payload(evidence),
            mode=0o600,
        )
        atomic_write_json(
            workspace.staging_dir / "code_provenance.json", provenance, mode=0o600
        )
        assert_source_unchanged(
            private_source,
            private_snapshot,
            expected_bytes=evidence.source_snapshot.bytes,
            expected_sha256=evidence.source_snapshot.sha256,
        )
        assert_source_unchanged(
            source_path,
            evidence.source_snapshot,
            expected_bytes=evidence.source_snapshot.bytes,
            expected_sha256=evidence.source_snapshot.sha256,
        )
        assert_code_provenance_unchanged(provenance, repo_root, config_path)
        private_source.unlink()
        artifact_names = (
            "code_provenance.json",
            "feature_manifest.json",
            "feature_matrix.npz",
            "phase1_verification.json",
            "predictions.jsonl",
            "protocol_config.snapshot.toml",
            "random_forest_attempt.json",
            "scaler.json",
            "selection_lock.json",
            "selection_lock.sha256",
            "test_metrics.json",
            "validation_metrics.json",
        )
        artifact_manifest = {
            name: {
                "bytes": (workspace.staging_dir / name).stat().st_size,
                "sha256": sha256_file(workspace.staging_dir / name),
            }
            for name in artifact_names
        }
        run = {
            "schema_version": "qm9-phase2-classical-run-v1",
            "scientific_status": (
                "classical-phase2-complete"
                if selected_rf is not None
                else "partial-classical-result-random-forest-not-complete"
            ),
            "phase3_mist_started": False,
            "phase1_run_sha256": evidence.phase1_run_sha256,
            "source_sha256": evidence.source_snapshot.sha256,
            "feature_canonical_csr_sha256": feature["matrix"]["canonical_csr_sha256"],
            "selection_fingerprint": selection["selection_fingerprint"],
            "selection_lock_sha256": selection_sha,
            "test_reservation": str(reservation_path),
            "prediction_rows": prediction_rows,
            "runtime_seconds": time.monotonic() - started,
            "peak_rss_gib": _rss_gib(),
            "platform": platform.platform(),
            "artifacts": artifact_manifest,
        }
        run_path = workspace.staging_dir / "phase2_run.json"
        atomic_write_json(run_path, run, mode=0o600)
        run_sha = sha256_file(run_path)
        atomic_write_bytes(
            workspace.staging_dir / "phase2_run.sha256",
            f"{run_sha}  phase2_run.json\n".encode("ascii"),
            mode=0o600,
        )
        write_phase2_owner(workspace.staging_dir)
        finalize_phase2_workspace(workspace)
        complete_test_reservation(
            reservation_path,
            {
                "phase2_run_sha256": run_sha,
                "predictions_sha256": artifact_manifest["predictions.jsonl"]["sha256"],
                "test_metrics_sha256": artifact_manifest["test_metrics.json"]["sha256"],
            },
        )
        run["phase2_run_sha256"] = run_sha
        run["output_dir"] = str(workspace.output_dir)
        return run
    finally:
        discard_phase2_workspace(workspace)
