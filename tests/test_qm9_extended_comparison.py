from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_extended_comparison_is_common_frozen_test_with_all_models():
    result = json.loads(
        (ROOT / "results/qm9-extended-comparison-v1/aggregate_metrics.json").read_text()
    )
    assert result["selection_uses_test_labels"] is False
    assert result["test_evaluations_after_freeze"] == 1
    assert result["rows"] == {
        "train": 107_108,
        "validation": 13_388,
        "full_test": 13_389,
        "duplicate_clean_test": 13_370,
    }
    assert set(result["methods"]) == {"ridge", "xgboost", "mlp", "mist", "ensemble"}
    assert result["selected_features"] == "count_ecfp_plus_globals"
    assert len(result["global_descriptor_names"]) == 17

    full = result["leaderboard"]["full_test"]
    assert [item["model"] for item in full] == [
        "ensemble",
        "xgboost",
        "mist",
        "mlp",
        "ridge",
    ]
    assert [item["rank"] for item in full] == [1, 2, 3, 4, 5]
    assert full[0]["mean_normalized_mae"] == pytest.approx(0.08115898007288892)


def test_ensemble_weights_were_validation_selected_and_sum_to_one():
    result = json.loads(
        (ROOT / "results/qm9-extended-comparison-v1/aggregate_metrics.json").read_text()
    )
    ensemble = result["selected"]["ensemble"]
    assert ensemble["optimizer_success"] is True
    assert ensemble["validation_mean_normalized_mae"] == pytest.approx(0.081181974081726)
    assert set(ensemble["weights"]) == {"ridge", "xgboost", "mlp", "mist"}
    assert sum(ensemble["weights"].values()) == pytest.approx(1.0)
    assert all(value >= 0 for value in ensemble["weights"].values())


def test_every_method_has_all_twelve_targets_in_both_cohorts():
    result = json.loads(
        (ROOT / "results/qm9-extended-comparison-v1/aggregate_metrics.json").read_text()
    )
    for method in result["methods"].values():
        for cohort in ("full_test", "duplicate_clean_test"):
            metrics = method[cohort]
            assert len(metrics["per_target"]) == 12
            assert len(metrics["target_order"]) == 12
            assert metrics["mean_normalized_mae_across_12_targets"] >= 0


def test_postspecified_traditional_ensemble_keeps_mist_out_of_primary_blend():
    result = json.loads(
        (
            ROOT
            / "results/qm9-traditional-ensemble-correction-v1/aggregate_metrics.json"
        ).read_text()
    )
    assert result["status"] == (
        "post-specified-conceptual-correction-after-initial-test-report"
    )
    assert result["selection_uses_test_labels"] is False
    assert result["weights_selected_from"] == "validation-only"
    selection = result["traditional_ensemble_selection"]
    assert set(selection["weights"]) == {"engineered_ridge", "xgboost", "mlp"}
    assert "mist" not in selection["weights"]
    assert sum(selection["weights"].values()) == pytest.approx(1.0)
    assert selection["validation_mean_normalized_mae"] == pytest.approx(
        0.08753979010627626
    )
    assert result["leaderboard"]["full_test"][0] == {
        "mean_normalized_mae": pytest.approx(0.08739987257697619),
        "model": "traditional_ensemble",
        "rank": 1,
    }
    assert result["second_layer_all_model_ensemble"]["metrics"]["full_test"][
        "mean_normalized_mae_across_12_targets"
    ] == pytest.approx(0.08115898007288892)
