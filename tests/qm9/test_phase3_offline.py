from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from mist_transfer_benchmark.qm9.constants import TARGET_COLUMNS
from mist_transfer_benchmark.qm9.phase2_metrics import native_metrics
from mist_transfer_benchmark.qm9.phase3_adapter import (
    Phase3AdapterError,
    assert_no_manual_inverse_transform,
    batched_predict,
    stack_named_outputs,
)
from mist_transfer_benchmark.qm9.phase3_model import (
    EXPECTED_UNITS,
    Phase3ModelError,
    verify_snapshot,
)
from mist_transfer_benchmark.qm9.phase3_reservation import (
    Phase3ReservationError,
    authorize_test_inference,
    freeze_inference,
    reserve_inference_once,
)


def test_fake_adapter_preserves_batch_and_row_order():
    smiles = ["C", "CC", "CCC", "CCCC", "CCCCC"]

    def fake(batch: list[str]) -> np.ndarray:
        return np.asarray([[len(value) + offset for offset in range(12)] for value in batch])

    seen = []
    result = batched_predict(
        smiles,
        batch_size=2,
        predict_batch=fake,
        progress=lambda done, total: seen.append((done, total)),
    )
    assert result.shape == (5, 12)
    assert result[:, 0].tolist() == [1, 2, 3, 4, 5]
    assert seen == [(2, 5), (4, 5), (5, 5)]


def test_named_outputs_stack_by_frozen_order_not_mapping_order():
    output = {
        name: {"value": np.asarray([position, position + 100]), "unit": unit}
        for position, (name, unit) in reversed(
            list(enumerate(zip(TARGET_COLUMNS, EXPECTED_UNITS, strict=True)))
        )
    }
    stacked = stack_named_outputs(output, expected_rows=2)
    assert stacked[0].tolist() == list(range(12))
    assert stacked[1].tolist() == list(range(100, 112))


def test_named_outputs_reject_units_and_nonfinite_values():
    output = {
        name: {"value": np.ones(2), "unit": unit}
        for name, unit in zip(TARGET_COLUMNS, EXPECTED_UNITS, strict=True)
    }
    output["homo"]["unit"] = "eV"
    with pytest.raises(Phase3AdapterError, match="metadata"):
        stack_named_outputs(output, expected_rows=2)
    output["homo"]["unit"] = "hartree"
    output["gap"]["value"][0] = np.nan
    with pytest.raises(Phase3AdapterError, match="non-finite"):
        stack_named_outputs(output, expected_rows=2)


def test_manual_double_inverse_transform_fails_closed():
    assert_no_manual_inverse_transform(
        {
            "manual_inverse_transform_applied": False,
            "model_predict_returns_native_units": True,
        }
    )
    with pytest.raises(Phase3AdapterError, match="forbidden"):
        assert_no_manual_inverse_transform(
            {
                "manual_inverse_transform_applied": True,
                "model_predict_returns_native_units": True,
            }
        )


def test_inference_reservation_is_deterministic_and_exactly_once(tmp_path):
    selection = freeze_inference(
        {"model_revision": "a" * 40, "test_rows": 3, "batch_size": 128}
    )
    path = reserve_inference_once(tmp_path / "locks", selection)
    authorize_test_inference(path, selection)
    with pytest.raises(Phase3ReservationError, match="already reserved"):
        reserve_inference_once(tmp_path / "locks", selection)
    with pytest.raises(Phase3ReservationError, match="volatile"):
        freeze_inference({"runtime_seconds": 1.0})


def test_full_and_duplicate_clean_cohort_metrics_remain_aligned():
    truth = np.arange(60, dtype=np.float64).reshape(5, 12)
    prediction = truth + np.arange(5, dtype=np.float64)[:, None]
    clean_mask = np.asarray([True, False, True, True, False])
    full = native_metrics(truth, prediction, np.ones(12))
    clean = native_metrics(truth[clean_mask], prediction[clean_mask], np.ones(12))
    assert full["rows"] == 5
    assert clean["rows"] == 3
    assert full["per_target"]["homo"]["mae"] == pytest.approx(2.0)
    assert clean["per_target"]["homo"]["mae"] == pytest.approx(5 / 3)


def test_snapshot_allowlist_and_hash_guard(monkeypatch, tmp_path):
    import mist_transfer_benchmark.qm9.phase3_model as model_contract

    content = b"reviewed bytes"
    path = tmp_path / "file.txt"
    path.write_bytes(content)
    expected = {"file.txt": (len(content), hashlib.sha256(content).hexdigest())}
    monkeypatch.setattr(model_contract, "EXPECTED_FILES", expected)
    assert verify_snapshot(tmp_path)[0]["path"] == "file.txt"
    path.write_bytes(b"tampered")
    with pytest.raises(Phase3ModelError, match="bytes/hash"):
        verify_snapshot(tmp_path)
    Path(tmp_path / "extra.txt").write_text("unexpected")
    with pytest.raises(Phase3ModelError, match="allowlist"):
        verify_snapshot(tmp_path)
