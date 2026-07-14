from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from mist_transfer_benchmark.qm9.constants import TARGET_COLUMNS
from mist_transfer_benchmark.qm9.paper_evaluation import (
    ArrayTargetLoader,
    ExternalPredictionArtifact,
    PaperEvaluationError,
    bootstrap_confidence_intervals,
    canonical_hash,
    make_evaluation_split,
    merge_group_relations,
    molecular_identity_groups,
    run_protocol,
    scaffold_groups,
)


def _data(rows: int = 54):
    rng = np.random.default_rng(11)
    base = [
        "CC", "C(C)", "CCC", "CCO", "OCC", "CCN", "c1ccccc1", "C1CCCCC1",
        "CC(=O)O", "CC#N", "CO", "CN",
    ]
    smiles = (base * ((rows + len(base) - 1) // len(base)))[:rows]
    x = rng.normal(size=(rows, 7))
    y = x @ rng.normal(size=(7, 12)) + rng.normal(scale=0.1, size=(rows, 12))
    return x, y, smiles


def _protocol(*, external: bool = False):
    return {
        "schema_version": "qm9-paper-evaluation-v1",
        "protocol_id": "unit-smoke",
        "target_order": list(TARGET_COLUMNS),
        "seeds": [5],
        "splits": {"kinds": ["random", "scaffold"], "fractions": [0.7, 0.15, 0.15]},
        "similarity_analysis": {"edges": [0.0, 0.5, 0.8, 1.0000001]},
        "external_predictions": {
            "enabled": external,
            "released_qm9_mist_reuse_forbidden": True,
        },
        "bootstrap": {"samples": 12, "confidence": 0.95},
        "resource_instrumentation": {
            "enabled": True,
            "process_rss_method": "resource.getrusage-RUSAGE_SELF-ru_maxrss",
            "gpu_policy": "null-with-reason-unless-truthful-backend-telemetry",
        },
        "models": {
            "engineered_ridge": {
                "enabled": True,
                "candidates": [{"alpha": 0.1}, {"alpha": 1.0}],
            }
        },
    }


def _run(protocol, x, y, smiles, output, **kwargs):
    loader = ArrayTargetLoader(
        y,
        provenance={"fixture": "unit-v1"},
        full_target_identity=hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest(),
    )
    manifest = run_protocol(
        protocol,
        x,
        loader,
        smiles,
        output,
        feature_schema={"kind": "unit", "columns": x.shape[1]},
        test_similarity_resolver=lambda _kind, _seed, split: np.full(len(split.test), 0.75),
        **kwargs,
    )
    return manifest, loader


def test_grouped_random_keeps_exact_canonical_and_connectivity_duplicates_together():
    _, _, smiles = _data()
    identities = molecular_identity_groups(smiles)
    split = make_evaluation_split(
        len(smiles),
        kind="random",
        seed=3,
        random_group_ids=identities["connectivity_smiles"],
    )
    for values in identities.values():
        parts = [set(values[index]) for index in (split.train, split.validation, split.test)]
        assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
    with pytest.raises(PaperEvaluationError, match="grouped-random"):
        make_evaluation_split(len(smiles), kind="random", seed=3)


def test_bootstrap_returns_aggregate_and_twelve_target_intervals():
    rng = np.random.default_rng(4)
    truth = rng.normal(size=(40, 12))
    prediction = truth + rng.normal(scale=0.2, size=(40, 12))
    result = bootstrap_confidence_intervals(
        truth, prediction, np.ones(12), samples=50, seed=8
    )
    assert result["aggregate"]["lower"] <= result["aggregate"]["upper"]
    assert set(result["per_target"]) == set(TARGET_COLUMNS)


def test_global_freeze_gate_precedes_every_test_target_read(tmp_path):
    x, y, smiles = _data()
    manifest, loader = _run(_protocol(), x, y, smiles, tmp_path / "run")
    events = loader.access_log
    authorization = next(
        index for index, item in enumerate(events) if item["event"] == "authorize-test"
    )
    assert all(item["event"] == "selection" for item in events[:authorization])
    assert all(item["event"] == "test" for item in events[authorization + 1 :])
    assert all(item["test_authorized"] is False for item in events[:authorization])
    assert (tmp_path / "run/global_freeze_gate.json").is_file()
    event_names = [item["event"] for item in manifest["events"]]
    gate_position = event_names.index("global-freeze-gate-closed")
    assert all(name == "selection-frozen" for name in event_names[:gate_position])
    assert all(name != "test-cell-completed" for name in event_names[: gate_position + 1])


def test_manifest_resume_binds_inputs_and_reuses_similarity_cache(tmp_path):
    x, y, smiles = _data()
    calls = []

    def similarity(kind, seed, split):
        calls.append((kind, seed))
        return np.full(len(split.test), 0.6)

    loader = ArrayTargetLoader(
        y,
        provenance={"fixture": "unit-v1"},
        full_target_identity=hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest(),
    )
    first = run_protocol(
        _protocol(),
        x,
        loader,
        smiles,
        tmp_path / "run",
        feature_schema={"kind": "unit", "columns": 7},
        test_similarity_resolver=similarity,
    )
    assert first["complete"] is True
    assert len(calls) == 2
    second_loader = ArrayTargetLoader(
        y,
        provenance={"fixture": "unit-v1"},
        full_target_identity=hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest(),
    )
    second = run_protocol(
        _protocol(),
        x,
        second_loader,
        smiles,
        tmp_path / "run",
        feature_schema={"kind": "unit", "columns": 7},
        test_similarity_resolver=similarity,
    )
    assert second["completed_cells"] == first["completed_cells"]
    assert len(calls) == 2
    changed = x.copy()
    changed[0, 0] += 1
    with pytest.raises(PaperEvaluationError, match="inputs differ"):
        _run(_protocol(), changed, y, smiles, tmp_path / "run")


def test_traditional_ensemble_summary_and_paired_delta_are_written(tmp_path):
    x, y, smiles = _data()
    _run(_protocol(), x, y, smiles, tmp_path / "run")
    cell = json.loads((tmp_path / "run/cells/random-seed-5.json").read_text())
    assert "traditional_ensemble" in cell["models"]
    assert "all_model_ensemble" not in cell["models"]
    assert "engineered_ridge" in cell["paired_delta_traditional_ensemble_minus_model"]
    resource = cell["models"]["engineered_ridge"]["resource_observation"]
    assert resource["process_peak_rss"]["bytes"] > 0
    assert resource["gpu_memory"]["peak_allocated_bytes"] is None
    assert "misleading" in resource["gpu_memory"]["reason"]
    assert resource["test_inference"]["rows_per_second"] > 0
    assert resource["test_inference"]["milliseconds_per_row"] > 0
    assert resource["model_artifact"]["bytes"] is None
    summary = json.loads((tmp_path / "run/summary.json").read_text())
    assert set(summary["methods"]) == {"engineered_ridge", "traditional_ensemble"}
    assert len(summary["methods"]["traditional_ensemble"]["cells"]) == 2
    cost = summary["methods"]["engineered_ridge"]["cost_summary"]
    assert cost["training_or_refit_seconds_sum"] > 0
    assert cost["effective_test_rows_per_second"] > 0
    assert cost["peak_process_rss_bytes_max"] > 0


def test_external_mode_requires_every_cell_and_denies_released_qm9(tmp_path):
    x, y, smiles = _data()
    with pytest.raises(PaperEvaluationError, match="present exactly"):
        _run(_protocol(external=True), x, y, smiles, tmp_path / "missing")

    identities = molecular_identity_groups(smiles)
    scaffolds = scaffold_groups(smiles)

    def resolver(kind, seed):
        split = make_evaluation_split(
            len(smiles),
            kind=kind,
            seed=seed,
            fractions=(0.7, 0.15, 0.15),
            scaffold_group_ids=scaffolds,
            random_group_ids=identities["connectivity_smiles"],
        )
        path = tmp_path / f"{kind}.npz"
        np.savez(
            path,
            validation_predictions=np.zeros((len(split.validation), 12)),
            validation_source_row_index=split.validation,
            test_predictions=np.zeros((len(split.test), 12)),
            test_source_row_index=split.test,
        )
        return ExternalPredictionArtifact(
            path,
            {
                "model_id": "released-mist",
                "source_status": "released-qm9",
                "task_finetuned_on_qm9": True,
                "training_split_identity": "legacy",
            },
        )

    with pytest.raises(PaperEvaluationError, match="task-finetuned"):
        _run(
            _protocol(external=True),
            x,
            y,
            smiles,
            tmp_path / "denied",
            external_prediction_resolver=resolver,
        )


def test_eligible_external_predictions_join_all_model_layer_and_strata(tmp_path):
    x, y, smiles = _data()
    protocol = _protocol(external=True)
    protocol["external_predictions"]["expected_artifact_manifest_sha256"] = {}
    identities = molecular_identity_groups(smiles)
    scaffolds = scaffold_groups(smiles)
    merged_scaffolds = merge_group_relations(
        scaffolds, identities["connectivity_smiles"]
    )

    def resolver(kind, seed):
        split = make_evaluation_split(
            len(smiles),
            kind=kind,
            seed=seed,
            fractions=(0.7, 0.15, 0.15),
            scaffold_group_ids=merged_scaffolds,
            random_group_ids=identities["connectivity_smiles"],
        )
        path = tmp_path / f"eligible-{kind}.npz"
        np.savez(
            path,
            validation_predictions=np.zeros((len(split.validation), 12)),
            validation_source_row_index=split.validation,
            test_predictions=np.zeros((len(split.test), 12)),
            test_source_row_index=split.test,
        )
        provenance = {
            "model_id": "task-neutral-external-v1",
            "source_status": "independent-frozen",
            "task_finetuned_on_qm9": False,
            "training_split_identity": "none",
        }
        identity = {
            "prediction_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "prediction_bytes": path.stat().st_size,
            "provenance": provenance,
            "provenance_sha256": canonical_hash(provenance),
        }
        protocol["external_predictions"]["expected_artifact_manifest_sha256"][
            f"{kind}-seed-{seed}"
        ] = canonical_hash(identity)
        return ExternalPredictionArtifact(
            path,
            provenance,
        )

    _run(
        protocol,
        x,
        y,
        smiles,
        tmp_path / "eligible",
        external_prediction_resolver=resolver,
    )
    cell = json.loads((tmp_path / "eligible/cells/random-seed-5.json").read_text())
    assert "mist" in cell["models"]
    assert "all_model_ensemble" in cell["models"]
    assert cell["models"]["mist"]["test_similarity_strata"]
    assert cell["models"]["all_model_ensemble"]["test_similarity_strata"]


def test_similarity_cache_hash_change_is_rejected(tmp_path):
    x, y, smiles = _data()
    _run(_protocol(), x, y, smiles, tmp_path / "run")
    cache = tmp_path / "run/similarities/random-seed-5.npz"
    cache.write_bytes(b"changed")
    with pytest.raises(PaperEvaluationError, match="artifact identity|similarity cache identity"):
        _run(_protocol(), x, y, smiles, tmp_path / "run")


def test_test_only_target_mutation_is_rejected_by_full_artifact_identity(tmp_path):
    x, y, smiles = _data()
    protocol = _protocol()
    protocol["splits"]["kinds"] = ["random"]
    _run(protocol, x, y, smiles, tmp_path / "run")
    identities = molecular_identity_groups(smiles)
    split = make_evaluation_split(
        len(smiles),
        kind="random",
        seed=5,
        fractions=(0.7, 0.15, 0.15),
        random_group_ids=identities["connectivity_smiles"],
    )
    changed = y.copy()
    changed[split.test[0], 0] += 1.0
    with pytest.raises(PaperEvaluationError, match="inputs differ"):
        _run(protocol, x, changed, smiles, tmp_path / "run")


@pytest.mark.parametrize(
    "relative_path",
    [
        "protocol.snapshot.json",
        "input_identity.json",
        "global_freeze_gate.json",
        "predictions/random-seed-5-engineered_ridge.npy",
    ],
)
def test_complete_resume_rejects_persisted_artifact_tamper(tmp_path, relative_path):
    x, y, smiles = _data()
    _run(_protocol(), x, y, smiles, tmp_path / "run")
    artifact = tmp_path / "run" / relative_path
    artifact.write_bytes(b"fault-injection")
    with pytest.raises(PaperEvaluationError, match="persisted artifact identity"):
        _run(_protocol(), x, y, smiles, tmp_path / "run")
