#!/usr/bin/env python3
"""Run fixed-MIST-split repeated evaluation (public deterministic smoke by default)."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import numpy as np
from scipy import sparse

from mist_transfer_benchmark.qm9.constants import TARGET_COLUMNS
from mist_transfer_benchmark.qm9.data import load_qm9_identities
from mist_transfer_benchmark.qm9.fixed_split_evaluation import (
    FixedSplitEvaluationError,
    LazyTestTargetGate,
    critical_review_plan,
    file_sha256,
    run_fixed_split,
    run_smoke_protocol,
)
from mist_transfer_benchmark.qm9.phase2_contract import verify_phase1_evidence
from mist_transfer_benchmark.qm9.phase2_targets import load_targets_for_indices

EXPECTED_FEATURE_SHA256 = "ddaaff5608faa3428ee2720fca173e24dc4db90399852c1363db55268ea33810"
EXPECTED_MIST_TEST_SHA256 = "c3b7abf994f870f6066f0f890ea1c4d01ce10061b2ee0af115f920a28a5dcc6f"
EXPECTED_MODEL_ID = "mist-models/mist-26.9M-kkgx0omx-qm9"
EXPECTED_MODEL_REVISION = "65ceeed479609e9dcaef04e687556e2b39e25f23"


def _load_mist_test(path: Path, expected_indices: np.ndarray) -> np.ndarray:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    indices = np.asarray([row["source_row_index"] for row in rows], dtype=np.int64)
    if not np.array_equal(indices, expected_indices):
        raise FixedSplitEvaluationError("fixed MIST test predictions differ in source-row order")
    values = np.asarray([row["predicted"] for row in rows], dtype=np.float64)
    if values.shape != (len(expected_indices), 12) or not np.all(np.isfinite(values)):
        raise FixedSplitEvaluationError("fixed MIST test predictions have an invalid shape/value")
    return values


def _load_mist_validation(path: Path | None, expected_indices: np.ndarray) -> np.ndarray | None:
    if path is None:
        return None
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"source_row_index", "predictions", "model_id", "model_revision"}:
            raise FixedSplitEvaluationError("MIST validation NPZ schema is not exact")
        if str(payload["model_id"].item()) != EXPECTED_MODEL_ID:
            raise FixedSplitEvaluationError("MIST validation model ID differs")
        if str(payload["model_revision"].item()) != EXPECTED_MODEL_REVISION:
            raise FixedSplitEvaluationError("MIST validation model revision differs")
        indices = np.asarray(payload["source_row_index"], dtype=np.int64)
        values = np.asarray(payload["predictions"], dtype=np.float64)
    if (
        not np.array_equal(indices, expected_indices)
        or values.shape != (len(indices), 12)
        or not np.all(np.isfinite(values))
    ):
        raise FixedSplitEvaluationError("MIST validation predictions differ in fixed row order")
    return values


def _run_real(args: argparse.Namespace, config: dict) -> dict[str, object]:
    try:
        import torch
        import xgboost
    except ImportError as error:
        raise FixedSplitEvaluationError(
            "real fixed-split execution requires the paper dependencies; "
            "run `uv sync --extra paper --frozen`"
        ) from error
    runtime_dependencies = {
        "torch_version": torch.__version__,
        "xgboost_version": xgboost.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }
    with Path("configs/qm9_28m.toml").open("rb") as handle:
        phase1_config = tomllib.load(handle)
    identities = load_qm9_identities(args.qm9_csv)
    evidence = verify_phase1_evidence(phase1_config, args.phase1_dir, args.qm9_csv)
    feature_manifest = json.loads(args.feature_manifest.read_text(encoding="utf-8"))
    if feature_manifest.get("schema_version") != "qm9-paper-feature-artifact-v1":
        raise FixedSplitEvaluationError("feature manifest schema differs")
    if feature_manifest.get("source_csv_sha256") != file_sha256(args.qm9_csv):
        raise FixedSplitEvaluationError("feature manifest belongs to another QM9 CSV")
    if feature_manifest["feature_matrix"]["file_sha256"] != file_sha256(args.feature_matrix):
        raise FixedSplitEvaluationError("feature matrix hash differs from its manifest")
    if feature_manifest["feature_matrix"]["file_sha256"] != EXPECTED_FEATURE_SHA256:
        raise FixedSplitEvaluationError("feature matrix is not the frozen v2 artifact")
    schema = feature_manifest.get("feature_schema", {})
    if (
        feature_manifest.get("rows") != 133885
        or feature_manifest["feature_matrix"].get("shape") != [133885, 2065]
        or schema.get("schema_version") != "qm9-count-ecfp4-plus-globals-v1"
        or schema.get("representation") != "raw-count-ECFP4-plus-global-descriptors"
        or schema.get("storage") != "scipy.sparse.csr_matrix"
        or schema.get("columns") != 2065
    ):
        raise FixedSplitEvaluationError("feature schema/semantics differ from frozen v2")
    if feature_manifest.get("source_row_identity_sha256") != identities.row_identity_sha256:
        raise FixedSplitEvaluationError("feature source row identity differs")
    features = sparse.load_npz(args.feature_matrix).tocsr()
    if features.shape != tuple(feature_manifest["feature_matrix"]["shape"]):
        raise FixedSplitEvaluationError("feature matrix shape differs from its manifest")
    if not np.all(np.isfinite(features.data)):
        raise FixedSplitEvaluationError("feature matrix contains nonfinite sparse values")
    train = np.asarray(evidence.split.train, dtype=np.int64)
    validation = np.asarray(evidence.split.validation, dtype=np.int64)
    test = np.asarray(evidence.split.test, dtype=np.int64)
    y_train = load_targets_for_indices(args.qm9_csv, train, identities)
    y_validation = load_targets_for_indices(args.qm9_csv, validation, identities)
    mist_path = args.mist_dir / "mist_predictions.jsonl"
    if file_sha256(mist_path) != EXPECTED_MIST_TEST_SHA256:
        raise FixedSplitEvaluationError("fixed MIST test artifact SHA-256 differs")
    phase3 = json.loads((args.mist_dir / "phase3_run.json").read_text(encoding="utf-8"))
    audit = json.loads((args.mist_dir / "model_audit.json").read_text(encoding="utf-8"))
    if (
        phase3.get("model_id") != EXPECTED_MODEL_ID
        or phase3.get("model_revision") != EXPECTED_MODEL_REVISION
        or phase3.get("artifacts", {}).get("mist_predictions.jsonl", {}).get("sha256")
        != EXPECTED_MIST_TEST_SHA256
        or audit.get("model_id") != EXPECTED_MODEL_ID
        or audit.get("revision") != EXPECTED_MODEL_REVISION
        or audit.get("hard_gate_passed") is not True
        or audit.get("channel_order") != list(TARGET_COLUMNS)
    ):
        raise FixedSplitEvaluationError("MIST phase3/model audit provenance differs")
    mist_test = _load_mist_test(mist_path, test)
    mist_validation = _load_mist_validation(args.mist_validation_predictions, validation)
    artifact_dir = args.feature_manifest.parent.resolve()
    scaffold_path = (artifact_dir / feature_manifest["scaffold_groups"]["path"]).resolve()
    if scaffold_path.parent != artifact_dir:
        raise FixedSplitEvaluationError("scaffold cache path escapes artifact directory")
    if file_sha256(scaffold_path) != feature_manifest["scaffold_groups"]["file_sha256"]:
        raise FixedSplitEvaluationError("scaffold group cache hash differs from feature manifest")
    scaffold_groups = np.load(scaffold_path, allow_pickle=False).astype(str)
    train_scaffolds = set(scaffold_groups[train])
    novelty_labels = np.asarray(
        [
            "seen_scaffold" if value in train_scaffolds else "unseen_scaffold"
            for value in scaffold_groups[test]
        ]
    )
    gate = LazyTestTargetGate(lambda: load_targets_for_indices(args.qm9_csv, test, identities))
    identity = {
        "source_csv_sha256": file_sha256(args.qm9_csv),
        "source_row_identity_sha256": identities.row_identity_sha256,
        "feature_manifest_sha256": file_sha256(args.feature_manifest),
        "feature_matrix_sha256": file_sha256(args.feature_matrix),
        "scaffold_groups_sha256": file_sha256(scaffold_path),
        "phase1_run_sha256": file_sha256(args.phase1_dir / "phase1_run.json"),
        "fixed_mist_test_sha256": file_sha256(mist_path),
        "fixed_mist_validation_sha256": (
            file_sha256(args.mist_validation_predictions)
            if args.mist_validation_predictions is not None
            else None
        ),
        "fixed_mist_model_id": EXPECTED_MODEL_ID,
        "fixed_mist_model_revision": EXPECTED_MODEL_REVISION,
        "mist_model_audit_sha256": file_sha256(args.mist_dir / "model_audit.json"),
        "mist_phase3_run_sha256": file_sha256(args.mist_dir / "phase3_run.json"),
        "mist_historical_runtime": {
            "status": "available-from-phase3-artifact",
            "runtime_seconds": phase3.get("runtime_seconds"),
            "worker_runtime_seconds": phase3.get("worker_runtime_seconds"),
            "gpu_peak_memory_allocated_bytes": phase3.get("gpu_peak_memory_allocated_bytes"),
            "semantics": "historical-fixed-MIST-inference-not-current-run",
        },
        "split_membership_sha256": {
            **evidence.split.membership_hashes(),
        },
    }
    if args.preflight:
        args.output.mkdir(parents=True, exist_ok=True)
        result = {
            "schema_version": "qm9-fixed-mist-split-v2-preflight-v1",
            "status": "ready",
            "input_identity": identity,
            "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
            "features": {"shape": list(features.shape), "nnz": int(features.nnz)},
            "runtime_dependencies": runtime_dependencies,
            "structural_novelty": {
                label: int(np.sum(novelty_labels == label)) for label in sorted(set(novelty_labels))
            },
            "fixed_mist_validation": "available" if mist_validation is not None else "absent",
            "supplemental_all_model_ensemble": (
                "enabled"
                if mist_validation is not None
                else "omitted-no-fixed-mist-validation-predictions"
            ),
            "test_labels_read": False,
            "critical_reviews": critical_review_plan(
                {
                    "test_labels_read": False,
                    "source_csv_sha256": identity["source_csv_sha256"],
                    "split_membership_sha256": identity["split_membership_sha256"],
                }
            ),
            "next_required_review": "selection-freeze",
        }
        path = args.output / "preflight.json"
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"complete": False, "preflight": result}
    return run_fixed_split(
        config,
        features,
        y_train,
        y_validation,
        gate,
        train,
        validation,
        test,
        mist_validation,
        mist_test,
        args.output,
        input_identity=identity,
        smoke=False,
        structural_novelty_labels=novelty_labels,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/qm9_fixed_split_evaluation_v2.toml")
    )
    parser.add_argument("--output", type=Path, default=Path("results/qm9-fixed-split-v2-smoke"))
    parser.add_argument("--qm9-csv", type=Path)
    parser.add_argument("--feature-matrix", type=Path)
    parser.add_argument("--feature-manifest", type=Path)
    parser.add_argument("--phase1-dir", type=Path)
    parser.add_argument("--mist-dir", type=Path)
    parser.add_argument("--mist-validation-predictions", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    private = [
        args.qm9_csv,
        args.feature_matrix,
        args.feature_manifest,
        args.phase1_dir,
        args.mist_dir,
    ]
    with args.config.open("rb") as handle:
        config = tomllib.load(handle)
    if any(value is not None for value in private) and not all(
        value is not None for value in private
    ):
        parser.error("real mode requires all five private artifact arguments")
    manifest = _run_real(args, config) if all(private) else run_smoke_protocol(config, args.output)
    print(json.dumps({"output": str(args.output), "complete": manifest["complete"]}, indent=2))


if __name__ == "__main__":
    main()
