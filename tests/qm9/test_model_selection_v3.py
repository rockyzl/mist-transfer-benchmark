from __future__ import annotations

import json
import subprocess
import sys
import tomllib
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from mist_transfer_benchmark.qm9.model_selection_v3 import (
    BudgetLedger,
    ModelSelectionError,
    assert_paired_ablation,
    canonical_hash,
    fit_real_torch_mlp,
    fit_real_xgboost,
    group_safe_inner_split,
    monitor_curve,
    rank_candidates,
    run_real_selection_wave,
    run_smoke_selection,
    validate_config,
    validate_selection_lock,
)
from mist_transfer_benchmark.qm9.selection_v3_data import fit_feature_recipe


def config() -> dict:
    with Path("configs/qm9_model_selection_v3.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_inner_split_is_group_safe_and_train_only() -> None:
    groups = np.repeat(np.arange(20), 2)
    train, tune = group_safe_inner_split(groups, tune_fraction=0.2, seed=7)
    assert set(train).isdisjoint(tune)
    assert len(train) + len(tune) == len(groups)
    assert set(groups[train]).isdisjoint(groups[tune])


def test_ranking_requires_inner_tune() -> None:
    records = [
        {"id": "b", "mean_mnmae": 0.2, "selection_partition": "inner-tune-only"},
        {"id": "a", "mean_mnmae": 0.1, "selection_partition": "inner-tune-only"},
    ]
    assert rank_candidates(records)[0]["id"] == "a"
    records[0]["selection_partition"] = "outer-validation"
    with pytest.raises(ModelSelectionError, match="outside inner tune"):
        rank_candidates(records)


def test_budget_fails_closed() -> None:
    ledger = BudgetLedger(max_fits=1, max_seconds=10)
    ledger.reserve("ridge", "a", 1)
    ledger.complete(0.1)
    with pytest.raises(ModelSelectionError, match="budget exhausted"):
        ledger.reserve("ridge", "b", 1)


def test_ablation_changes_only_scaling() -> None:
    left = {"id": "descriptor_scaled_only", "scale_all_sparse": False, "lr": 0.1}
    right = {"id": "all_sparse_scaled", "scale_all_sparse": True, "lr": 0.1}
    assert_paired_ablation(left, right)
    right["lr"] = 0.2
    with pytest.raises(ModelSelectionError, match="more than one factor"):
        assert_paired_ablation(left, right)


def test_curve_anomaly_and_best_state_contract() -> None:
    result = monitor_curve(
        [1.0, 0.7, 0.5],
        [0.3, 0.31, 0.32],
        max_epochs=10,
        restored_hash="a" * 64,
        best_hash="b" * 64,
        thresholds={"validation_increase_mark_after": 2},
    )
    assert result["status"] == "abnormal"
    assert result["maximum_consecutive_validation_increases"] == 2


def test_monitor_marks_low_gradient_and_target_degradation() -> None:
    thresholds = config()["monitoring"]
    curves = np.full((3, 12), 0.1)
    curves[2, 0] = 0.2
    result = monitor_curve(
        [1.0, 0.9, 0.8],
        [0.3, 0.29, 0.28],
        max_epochs=3,
        restored_hash="a" * 64,
        best_hash="a" * 64,
        thresholds=thresholds,
        per_target_nmae=curves,
        gradient_norm=[1e-10, 1e-10, 1e-10],
    )
    assert result["status"] == "abnormal"
    assert "gradient-norm-below-hard-minimum-consecutively" in result["anomalies"]
    assert "per-target-nmae-degraded" in result["warnings"]


def test_config_is_fail_closed() -> None:
    mutations = []
    ridge = deepcopy(config())
    ridge["ridge"]["alphas"] = [1.0]
    mutations.append(ridge)
    learning_rate = deepcopy(config())
    learning_rate["xgboost"]["candidates"][0]["learning_rate"] = 999
    mutations.append(learning_rate)
    budget = deepcopy(config())
    budget["budget"]["max_candidate_fits"] = 999999
    mutations.append(budget)
    gradient = deepcopy(config())
    gradient["monitoring"]["gradient_norm_hard_max"] = -1
    mutations.append(gradient)
    for payload in mutations:
        with pytest.raises(ModelSelectionError, match="reviewed numeric contract hash"):
            validate_config(payload)


def test_feature_recipe_scaler_fits_fold_train_only() -> None:
    matrix = sparse.csr_matrix(
        np.column_stack((np.ones((6, 2048)), np.arange(6), np.arange(6) * 2.0))
    )
    train = np.asarray([0, 1, 2, 3])
    tune = np.asarray([4, 5])
    _, transformed, provenance = fit_feature_recipe("descriptor_scaled_only", matrix, train, tune)
    assert provenance["fit_rows"] == 4
    assert float(transformed[:, -2:].mean()) > 1.0


def test_tiny_real_xgboost_and_mlp_backends() -> None:
    pytest.importorskip("xgboost")
    pytest.importorskip("torch")
    rng = np.random.default_rng(9)
    x = sparse.csr_matrix(rng.normal(size=(36, 8)))
    y = rng.normal(size=(36, 12))
    xgb_prediction, rounds = fit_real_xgboost(
        x[:28],
        y[:28],
        x[28:],
        y[28:],
        {
            "id": "tiny",
            "max_depth": 2,
            "learning_rate": 0.1,
            "n_estimators": 5,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "min_child_weight": 1.0,
        },
        seed=3,
        early_stopping_rounds=2,
    )
    assert xgb_prediction.shape == (8, 12)
    assert len(rounds) == 12
    training = {"max_epochs": 3, "patience": 2, "min_delta": 0.0, "batch_size": 16}
    monitoring = dict(config()["monitoring"])
    monitoring["max_validation_mnmae_after_five_epochs"] = 1e6
    mlp_prediction, curve = fit_real_torch_mlp(
        x[:28],
        y[:28],
        x[28:],
        y[28:],
        {
            "id": "tiny",
            "hidden_dims": [8],
            "dropout": 0.0,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
        },
        training,
        monitoring,
        seed=3,
    )
    assert mlp_prediction.shape == (8, 12)
    assert curve["best_checkpoint_sha256"] == curve["restored_checkpoint_sha256"]


def test_smoke_lock_has_no_outer_or_test_reads_and_detects_tamper(tmp_path: Path) -> None:
    manifest = run_smoke_selection(config(), tmp_path)
    assert manifest["complete"] is True
    lock_path = tmp_path / "selection_lock.json"
    lock = validate_selection_lock(lock_path)
    assert lock["outer_validation_targets_read"] is False
    assert lock["test_targets_read"] is False
    payload = json.loads(lock_path.read_text())
    payload["selected"]["ridge"]["id"] = "tampered"
    lock_path.write_text(json.dumps(payload))
    with pytest.raises(ModelSelectionError, match="hash differs"):
        validate_selection_lock(lock_path)


def test_lock_detects_artifact_tamper_and_why_is_hashed(tmp_path: Path) -> None:
    run_smoke_selection(config(), tmp_path)
    why = json.loads((tmp_path / "recursive_why/index.json").read_text())
    assert why[0]["issue_id"] == "why-feature"
    assert len(why[0]["sha256"]) == 64
    artifact = tmp_path / "candidate_records/features.json"
    artifact.write_text("[]\n")
    with pytest.raises(ModelSelectionError, match="selection artifact changed"):
        validate_selection_lock(tmp_path / "selection_lock.json")


def test_runner_requires_explicit_smoke_or_authenticated_wave(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_qm9_model_selection_v3.py",
            "--output",
            str(tmp_path / "implicit"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "use explicit --smoke" in completed.stderr


def test_w1_cli_prior_lock_rejection_persists_governed_failure(tmp_path: Path) -> None:
    output = tmp_path / "w1-prior-rejected"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_qm9_model_selection_v3.py",
            "--wave",
            "scaler-ablation",
            "--output",
            str(output),
            "--qm9-csv",
            str(tmp_path / "qm9.csv"),
            "--feature-matrix",
            str(tmp_path / "features.npz"),
            "--feature-manifest",
            str(tmp_path / "manifest.json"),
            "--phase1-dir",
            str(tmp_path / "phase1"),
            "--prior-selection-lock",
            str(tmp_path / "forbidden-prior.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "does not accept --prior-selection-lock" in completed.stderr
    lock = validate_selection_lock(output / "selection_lock.json")
    assert lock["run_mode"] == "authenticated-train-only-failed"
    assert lock["outer_validation_targets_read"] is False
    assert lock["test_targets_read"] is False
    assert not (output / "budget_ledger.json").exists()
    assert (output / "recursive_why/index.json").is_file()
    assert (output / "selection_manifest.json").is_file()
    assert (output / "selection_lock.sha256").is_file()


def test_tiny_end_to_end_wave_chain_oof_ensemble_and_failure_why(tmp_path: Path) -> None:
    rng = np.random.default_rng(44)
    dense = np.zeros((30, 2065), dtype=np.float64)
    dense[:, :16] = rng.integers(0, 3, size=(30, 16))
    dense[:, 2048:] = rng.normal(size=(30, 17))
    matrix = sparse.csr_matrix(dense)
    targets = rng.normal(size=(30, 12))
    identity = np.asarray([f"identity-{row // 3}" for row in range(30)])
    scaffold = np.asarray([f"scaffold-{row % 10}" for row in range(30)])
    source_indices = np.arange(1000, 1030, dtype=np.int64)
    input_identity = {
        "fixture": "tiny-authenticated",
        "fixed_train_ordered_indices": source_indices.tolist(),
        "source_csv_sha256": "1" * 64,
        "source_row_identity_sha256": "2" * 64,
        "feature_matrix_file_sha256": "3" * 64,
        "feature_matrix_semantic_sha256": "4" * 64,
        "feature_manifest_sha256": "5" * 64,
        "scaffold_groups_file_sha256": "6" * 64,
        "phase1_run_sha256": "7" * 64,
        "fixed_split_ordered_sha256": {"train": "8" * 64},
        "fixed_split_membership_sha256": {"train": "9" * 64},
        "reviewed_config_sha256": canonical_hash(config()),
    }

    def tiny_backend(candidate, x_train, y_train, x_tune, y_tune, cfg, seed):
        del x_train, y_tune, cfg
        candidate_offset = sum(str(candidate["id"]).encode()) % 17 / 1000
        feature_signal = np.asarray(x_tune[:, 2048].toarray()).ravel()[:, None] * 0.001
        prediction = np.tile(y_train.mean(axis=0), (x_tune.shape[0], 1))
        return prediction + feature_signal + candidate_offset + seed % 7 / 10000, {
            "backend": "tiny-injected-real-path"
        }

    payload = config()
    scaler_dir = tmp_path / "scaler"
    scaler = run_real_selection_wave(
        payload,
        matrix,
        targets,
        identity,
        scaffold,
        scaler_dir,
        wave="scaler-ablation",
        input_identity=input_identity,
        fixed_train_source_indices=source_indices,
        backend_runner=tiny_backend,
    )
    assert scaler["scientific_result"] is False
    assert scaler["publication_ready"] is False
    scaler_lock = validate_selection_lock(scaler_dir / "selection_lock.json")
    assert len(scaler_lock["promoted"]) == 12
    scaler_ledger = json.loads((scaler_dir / "budget_ledger.json").read_text())
    assert scaler_ledger["completed_candidate_fits"] == 123

    mismatch_output = tmp_path / "input-mismatch"
    mismatched_identity = {**input_identity, "source_csv_sha256": "0" * 64}
    with pytest.raises(ModelSelectionError, match="input-mismatched"):
        run_real_selection_wave(
            payload,
            matrix,
            targets,
            identity,
            scaffold,
            mismatch_output,
            wave="feature-screen",
            input_identity=mismatched_identity,
            fixed_train_source_indices=source_indices,
            prior_selection_lock=scaler_dir / "selection_lock.json",
            backend_runner=tiny_backend,
        )
    mismatch_lock = validate_selection_lock(mismatch_output / "selection_lock.json")
    assert mismatch_lock["run_mode"] == "authenticated-train-only-failed"

    reorder_output = tmp_path / "source-reorder"
    with pytest.raises(ModelSelectionError, match="source index order"):
        run_real_selection_wave(
            payload,
            matrix,
            targets,
            identity,
            scaffold,
            reorder_output,
            wave="scaler-ablation",
            input_identity=input_identity,
            fixed_train_source_indices=source_indices[::-1],
            backend_runner=tiny_backend,
        )
    validate_selection_lock(reorder_output / "selection_lock.json")

    duplicate_indices = source_indices.copy()
    duplicate_indices[-1] = duplicate_indices[0]
    for label, invalid_indices, message in (
        ("duplicate", duplicate_indices, "contain duplicates"),
        ("missing", source_indices[:-1], "missing or misaligned"),
    ):
        invalid_output = tmp_path / f"source-{label}"
        with pytest.raises(ModelSelectionError, match=message):
            run_real_selection_wave(
                payload,
                matrix,
                targets,
                identity,
                scaffold,
                invalid_output,
                wave="scaler-ablation",
                input_identity=input_identity,
                fixed_train_source_indices=invalid_indices,
                backend_runner=tiny_backend,
            )
        validate_selection_lock(invalid_output / "selection_lock.json")

    invalid_config = deepcopy(payload)
    invalid_config["budget"]["max_candidate_fits"] = 999
    config_output = tmp_path / "invalid-config"
    with pytest.raises(ModelSelectionError, match="reviewed numeric contract"):
        run_real_selection_wave(
            invalid_config,
            matrix,
            targets,
            identity,
            scaffold,
            config_output,
            wave="scaler-ablation",
            input_identity=input_identity,
            fixed_train_source_indices=source_indices,
            backend_runner=tiny_backend,
        )
    assert json.loads((config_output / "selection_manifest.json").read_text())["complete"] is False

    for label in ("column", "transposed"):
        shape_output = tmp_path / f"bad-shape-{label}"

        def bad_backend(candidate, x_train, y_train, x_tune, y_tune, cfg, seed, shape_label=label):
            del candidate, x_train, y_train, y_tune, cfg, seed
            rows = x_tune.shape[0]
            shape = (rows, 1) if shape_label == "column" else (12, rows)
            return np.zeros(shape), {"backend": "bad-shape"}

        with pytest.raises(ModelSelectionError, match="prediction shape differs"):
            run_real_selection_wave(
                payload,
                matrix,
                targets,
                identity,
                scaffold,
                shape_output,
                wave="scaler-ablation",
                input_identity=input_identity,
                fixed_train_source_indices=source_indices,
                backend_runner=bad_backend,
            )
        validate_selection_lock(shape_output / "selection_lock.json")

    stale = json.loads((scaler_dir / "selection_lock.json").read_text())
    stale["config_sha256"] = "0" * 64
    stale.pop("lock_sha256")
    stale["lock_sha256"] = canonical_hash(stale)
    stale_path = scaler_dir / "stale.json"
    stale_path.write_text(json.dumps(stale))
    stale_output = tmp_path / "stale-output"
    with pytest.raises(ModelSelectionError, match="stale"):
        run_real_selection_wave(
            payload,
            matrix,
            targets,
            identity,
            scaffold,
            stale_output,
            wave="feature-screen",
            input_identity=input_identity,
            fixed_train_source_indices=source_indices,
            prior_selection_lock=stale_path,
            backend_runner=tiny_backend,
        )
    failure = json.loads((stale_output / "recursive_why/failure.json").read_text())
    assert failure["root_cause"]["status"] == "unknown"
    validate_selection_lock(stale_output / "selection_lock.json")

    feature_dir = tmp_path / "feature"
    feature = run_real_selection_wave(
        payload,
        matrix,
        targets,
        identity,
        scaffold,
        feature_dir,
        wave="feature-screen",
        input_identity=input_identity,
        fixed_train_source_indices=source_indices,
        prior_selection_lock=scaler_dir / "selection_lock.json",
        backend_runner=tiny_backend,
    )
    assert feature["scientific_result"] is False
    feature_lock = validate_selection_lock(feature_dir / "selection_lock.json")
    assert feature_lock["prior_selection_lock_sha256"] is not None
    assert len(feature_lock["promoted"]) == 3
    feature_ledger = json.loads((feature_dir / "budget_ledger.json").read_text())
    assert 60 <= feature_ledger["completed_candidate_fits"] <= 120
    provenance = json.loads((feature_dir / "candidate_records/identity.json").read_text())
    assert all(
        row["recipe_provenance"]["statistics_scope"] == "fit_indices_only" for row in provenance
    )
    assert all(min(row["tune_source_indices"]) >= 1000 for row in provenance)
    assert all(max(row["tune_local_indices"]) < 30 for row in provenance)

    one_finalist = json.loads((feature_dir / "selection_lock.json").read_text())
    one_finalist["promoted"] = one_finalist["promoted"][:1]
    one_finalist.pop("lock_sha256")
    one_finalist["lock_sha256"] = canonical_hash(one_finalist)
    one_finalist_path = feature_dir / "one-finalist.json"
    one_finalist_path.write_text(json.dumps(one_finalist))
    one_output = tmp_path / "one-finalist-output"
    with pytest.raises(ModelSelectionError, match="one finalist"):
        run_real_selection_wave(
            payload,
            matrix,
            targets,
            identity,
            scaffold,
            one_output,
            wave="full-selection",
            input_identity=input_identity,
            fixed_train_source_indices=source_indices,
            prior_selection_lock=one_finalist_path,
            backend_runner=tiny_backend,
        )
    one_failure = validate_selection_lock(one_output / "selection_lock.json")
    assert one_failure["run_mode"] == "authenticated-train-only-failed"

    full_dir = tmp_path / "full"
    full = run_real_selection_wave(
        payload,
        matrix,
        targets,
        identity,
        scaffold,
        full_dir,
        wave="full-selection",
        input_identity=input_identity,
        fixed_train_source_indices=source_indices,
        prior_selection_lock=feature_dir / "selection_lock.json",
        backend_runner=tiny_backend,
    )
    assert full["scientific_result"] is False
    assert full["publication_ready"] is False
    full_lock = validate_selection_lock(full_dir / "selection_lock.json")
    assert len(full_lock["selected"]["base_finalists"]) == 3
    ensemble = json.loads((full_dir / "ensemble.json").read_text())
    assert np.isclose(np.sum(ensemble["deployment_weights"]), 1.0)
    assert np.all(np.asarray(ensemble["deployment_weights"]) >= 0)
    with np.load(full_dir / "oof/finalists_and_ensemble.npz") as predictions:
        assert predictions["candidate_0_identity_oof"].shape == (5, 30, 12)
        assert predictions["candidate_0_identity_oof"].dtype == np.float64
        assert predictions["candidate_0_scaffold_oof"].shape == (5, 30, 12)
        assert predictions["ensemble_identity_crossfit_oof"].shape == (30, 12)
        assert np.array_equal(predictions["fixed_train_source_row_indices"], source_indices)
    full_ledger = json.loads((full_dir / "budget_ledger.json").read_text())
    assert full_ledger["completed_candidate_fits"] == 150
