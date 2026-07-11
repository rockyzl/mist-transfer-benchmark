from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_qm9_results import (
    DEFAULT_PHASE2_DIR,
    DEFAULT_PHASE3_DIR,
    DEFAULT_RF_DIR,
    HIGHLIGHTED_TARGETS,
    TARGET_ORDER,
    build_summary,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = REPO_ROOT / "site/qm9-results.json"
SIGNED_ARTIFACTS_AVAILABLE = all(
    path.is_dir() for path in (DEFAULT_PHASE2_DIR, DEFAULT_RF_DIR, DEFAULT_PHASE3_DIR)
)


def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


@pytest.mark.skipif(not SIGNED_ARTIFACTS_AVAILABLE, reason="signed local artifacts are ignored")
def test_tracked_summary_exactly_matches_authenticated_aggregate_artifacts():
    committed = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    assert committed == build_summary() == build_summary()


@pytest.mark.skipif(not SIGNED_ARTIFACTS_AVAILABLE, reason="signed local artifacts are ignored")
def test_generator_opens_aggregate_sources_only(monkeypatch):
    opened: list[Path] = []
    original_open = Path.open
    original_read_text = Path.read_text

    def tracked_open(path, *args, **kwargs):
        opened.append(path)
        return original_open(path, *args, **kwargs)

    def tracked_read_text(path, *args, **kwargs):
        opened.append(path)
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked_open)
    monkeypatch.setattr(Path, "read_text", tracked_read_text)
    build_summary()

    assert opened
    assert not any(path.suffix in {".csv", ".jsonl", ".npz", ".safetensors"} for path in opened)
    assert not any("prediction" in path.name for path in opened)


def test_summary_schema_math_and_all_target_coverage():
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    assert summary["schema_version"] == 1
    assert summary["scientific_status"] == "preliminary-local-point-estimates"
    assert summary["artifact_scope"] == "aggregate-only-no-row-level-data"
    assert summary["target_order"] == list(TARGET_ORDER)
    assert summary["highlighted_targets"] == list(HIGHLIGHTED_TARGETS)
    assert summary["dataset"]["full_test_rows"] == 13_389
    assert summary["dataset"]["duplicate_clean_test_rows"] == 13_370
    for cohort in ("full_test", "duplicate_clean_test"):
        record = summary["cohorts"][cohort]
        assert set(record["targets"]) == set(TARGET_ORDER)
        aggregate = record["aggregate"]
        expected = 100 * (aggregate["ridge"] - aggregate["mist"]) / aggregate["ridge"]
        assert aggregate["percent_reduction_vs_ridge"] == pytest.approx(expected)
        for target in TARGET_ORDER:
            metric = record["targets"][target]
            expected = 100 * (metric["ridge"]["mae"] - metric["mist"]["mae"]) / metric[
                "ridge"
            ]["mae"]
            assert metric["mae_percent_reduction_vs_ridge"] == pytest.approx(expected)


def test_summary_contains_no_row_level_or_private_fields():
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    forbidden = {
        "record_id",
        "source_row_index",
        "source_smiles",
        "smiles",
        "observed",
        "predicted",
        "predictions",
        "row_ids",
        "labels",
    }
    assert forbidden.isdisjoint(set(_walk_keys(summary)))
    assert "model.safetensors" not in SUMMARY_PATH.read_text(encoding="utf-8")
    assert summary["models"]["random_forest"]["test_evaluated"] is False
    assert summary["models"]["random_forest"]["scope"] == "validation-only"
    assert summary["provenance"]["test_inference_count"] == 1
    assert summary["provenance"]["test_inference_retries"] == 0
