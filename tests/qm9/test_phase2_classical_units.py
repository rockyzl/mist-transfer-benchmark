from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import tomllib
from scipy import sparse

from mist_transfer_benchmark.qm9.data import load_qm9_identities
from mist_transfer_benchmark.qm9.phase2_metrics import native_metrics
from mist_transfer_benchmark.qm9.phase2_scaler import ScalerContractError, fit_frozen_scaler
from mist_transfer_benchmark.qm9.phase2_selection import ridge_candidates, select_first_minimum
from mist_transfer_benchmark.qm9.phase2_similarity import tanimoto_1nn_predict
from mist_transfer_benchmark.qm9.phase2_targets import load_targets_for_indices
from mist_transfer_benchmark.qm9.phase2_test_lock import (
    TestLabelGate as LabelGate,
)
from mist_transfer_benchmark.qm9.phase2_test_lock import (
    TestLockError as LockError,
)
from mist_transfer_benchmark.qm9.phase2_test_lock import (
    freeze_selection,
    reserve_test_once,
)

from .conftest import write_qm9_csv

REPO_ROOT = Path(__file__).resolve().parents[2]


def _config():
    with (REPO_ROOT / "configs/qm9_28m.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_scaler_is_train_only_12_target_ddof_zero_and_hashed():
    y_train = np.arange(48, dtype=np.float64).reshape(4, 12)
    scaler, artifact = fit_frozen_scaler(_config(), y_train)
    assert np.allclose(scaler.var_, np.var(y_train, axis=0, ddof=0))
    assert artifact["fit_rows"] == 4
    assert len(artifact["fitted_state"]["mean_"]) == 12
    config = copy.deepcopy(_config())
    config["classical"]["target_standard_scaler"]["variance_ddof"] = 1
    with pytest.raises(ScalerContractError, match="variance_ddof"):
        fit_frozen_scaler(config, y_train)


def test_metrics_report_every_target_in_native_units():
    truth = np.tile(np.arange(12, dtype=float), (3, 1)) + np.arange(3)[:, None]
    prediction = truth + 1.0
    result = native_metrics(truth, prediction, np.ones(12))
    assert result["rows"] == 3
    assert len(result["per_target"]) == 12
    assert result["mean_normalized_mae_across_12_targets"] == pytest.approx(1.0)
    assert result["per_target"]["mu"]["mae"] == pytest.approx(1.0)


def test_ridge_order_and_exact_tie_choose_first():
    candidates = ridge_candidates(_config())
    order = [candidate.candidate_id for candidate in candidates]
    assert order == ["alpha-0.01", "alpha-0.1", "alpha-1", "alpha-10", "alpha-100"]
    results = [
        {"candidate_id": candidate_id, "mean_normalized_mae_across_12_targets": 1.0}
        for candidate_id in order
    ]
    assert select_first_minimum(results, order)["candidate_id"] == "alpha-0.01"


def test_target_loader_returns_only_requested_rows_in_requested_order(tmp_path):
    path = write_qm9_csv(tmp_path / "qm9.csv", ["CC", "O", "N"])
    data = load_qm9_identities(path, expected_rows=3)
    loaded = load_targets_for_indices(path, np.array([2, 0]), data)
    assert loaded.shape == (2, 12)
    assert loaded[0, 0] == pytest.approx(2.0)
    assert loaded[1, 0] == pytest.approx(0.0)


def test_tanimoto_tie_uses_lowest_training_source_index():
    matrix = sparse.csr_matrix(
        np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]], dtype=np.float64)
    )
    predictions, nearest, similarity = tanimoto_1nn_predict(
        matrix,
        np.array([1, 0]),
        np.array([[10.0] * 12, [20.0] * 12]),
        np.array([2]),
    )
    assert nearest.tolist() == [0]
    assert predictions[0, 0] == 20.0
    assert similarity.tolist() == [1.0]


def test_test_lock_reserves_selection_exactly_once(tmp_path):
    selection = freeze_selection({"selected": {"ridge": "alpha-1"}})
    gate = LabelGate()
    with pytest.raises(LockError, match="before selection reservation"):
        gate.require_authorized(selection)
    path = reserve_test_once(tmp_path / "locks", selection)
    gate.authorize(path, selection)
    gate.require_authorized(selection)
    assert path.is_file()
    with pytest.raises(LockError, match="already reserved"):
        reserve_test_once(tmp_path / "locks", selection)


def test_selection_fingerprint_ignores_execution_timing():
    first = freeze_selection(
        {"selected": {"ridge": "alpha-1"}, "runtime_seconds": 1.0}
    )
    second = freeze_selection(
        {"selected": {"ridge": "alpha-1"}, "runtime_seconds": 999.0}
    )
    assert first == second
