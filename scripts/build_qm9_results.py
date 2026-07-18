#!/usr/bin/env python3
"""Build the tracked aggregate-only QM9 result summary from signed local artifacts.

This generator deliberately reads no row-level prediction, label, identity, SMILES, or source-data
file. It accepts only aggregate Phase 2/3 JSON artifacts and the machine-readable protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path

from mist_transfer_benchmark.qm9.fixed_split_evaluation import require_publication_ready

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/qm9_28m.toml"
DEFAULT_PHASE2_DIR = REPO_ROOT / "results/qm9-phase2-classical-v1"
DEFAULT_RF_DIR = REPO_ROOT / "results/qm9-phase2-rf-attempt-v1"
DEFAULT_PHASE3_AUDIT_DIR = REPO_ROOT / "results/qm9-phase3-audit-v1"
DEFAULT_PHASE3_DIR = REPO_ROOT / "results/qm9-phase3-mist-v1"
DEFAULT_OUTPUT = REPO_ROOT / "site/qm9-results.json"

TARGET_ORDER = (
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
)
HIGHLIGHTED_TARGETS = ("homo", "lumo", "gap")


class SummarySourceError(ValueError):
    """Raised when aggregate artifacts do not authenticate one another."""


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_run_header(
    directory: Path, run_name: str, sidecar_name: str
) -> dict[str, object]:
    """Authenticate a run record without opening every artifact in its manifest."""

    run_path = directory / run_name
    run = _load_json(run_path)
    run_hash = _sha256(run_path)
    if (directory / sidecar_name).read_text(encoding="ascii") != (
        f"{run_hash}  {run_name}\n"
    ):
        raise SummarySourceError(f"invalid checksum sidecar: {run_name}")
    return run


def _verify_aggregate_artifact(
    directory: Path,
    run: dict[str, object],
    name: str,
) -> dict[str, object]:
    """Authenticate and load one declared aggregate JSON artifact only."""

    record = run["artifacts"][name]
    path = directory / record.get("file", name)
    if path.name != name or path.stat().st_size != record["bytes"]:
        raise SummarySourceError(f"aggregate artifact size/name differs: {name}")
    if _sha256(path) != record["sha256"]:
        raise SummarySourceError(f"aggregate artifact SHA-256 differs: {name}")
    return _load_json(path)


def _metric_record(source: dict[str, object], target: str) -> dict[str, float]:
    record = source["per_target"][target]
    return {
        "mae": record["mae"],
        "rmse": record["rmse"],
        "r2": record["r2"],
        "normalized_mae": record["mae_over_training_target_standard_deviation"],
    }


def _percent_reduction(reference: float, candidate: float) -> float:
    if reference <= 0:
        raise SummarySourceError("percentage reduction requires a positive reference metric")
    return 100.0 * (reference - candidate) / reference


def _cohort_summary(
    *,
    name: str,
    mist: dict[str, object],
    ridge: dict[str, object],
    units: dict[str, str],
) -> dict[str, object]:
    if mist["rows"] != ridge["rows"] or mist["target_order"] != list(TARGET_ORDER):
        raise SummarySourceError(f"MIST/Ridge cohort alignment differs: {name}")
    targets: dict[str, object] = {}
    for target in TARGET_ORDER:
        mist_record = _metric_record(mist, target)
        ridge_record = _metric_record(ridge, target)
        targets[target] = {
            "unit": units[target],
            "mist": mist_record,
            "ridge": ridge_record,
            "delta_mist_minus_ridge": {
                key: mist_record[key] - ridge_record[key]
                for key in ("mae", "rmse", "r2", "normalized_mae")
            },
            "mae_percent_reduction_vs_ridge": _percent_reduction(
                ridge_record["mae"], mist_record["mae"]
            ),
        }
    mist_aggregate = mist["mean_normalized_mae_across_12_targets"]
    ridge_aggregate = ridge["mean_normalized_mae_across_12_targets"]
    return {
        "rows": mist["rows"],
        "aggregate": {
            "metric": "mean-normalized-mae-across-12-targets",
            "lower_is_better": True,
            "mist": mist_aggregate,
            "ridge": ridge_aggregate,
            "mist_minus_ridge": mist_aggregate - ridge_aggregate,
            "percent_reduction_vs_ridge": _percent_reduction(
                ridge_aggregate, mist_aggregate
            ),
        },
        "targets": targets,
    }


def build_summary(
    *,
    config_path: Path = DEFAULT_CONFIG,
    phase2_dir: Path = DEFAULT_PHASE2_DIR,
    rf_dir: Path = DEFAULT_RF_DIR,
    phase3_audit_dir: Path = DEFAULT_PHASE3_AUDIT_DIR,
    phase3_dir: Path = DEFAULT_PHASE3_DIR,
) -> dict[str, object]:
    """Return a deterministic aggregate-only summary after authenticating all sources."""

    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    phase2_run = _verify_run_header(
        phase2_dir, "phase2_run.json", "phase2_run.sha256"
    )
    if _sha256(phase2_dir / "phase2_run.json") != config["phase_2_observation"][
        "locked_phase_2_run_sha256"
    ]:
        raise SummarySourceError("Phase 2 run differs from the protocol observation")
    phase3_audit_run = _verify_run_header(
        phase3_audit_dir, "phase3_audit_run.json", "phase3_audit_run.sha256"
    )
    phase3_run = _verify_run_header(
        phase3_dir, "phase3_run.json", "phase3_run.sha256"
    )
    observation = config["phase_3_observation"]
    if _sha256(phase3_audit_dir / "phase3_audit_run.json") != observation[
        "phase_3_audit_run_sha256"
    ]:
        raise SummarySourceError("Phase 3 audit run differs from TOML")
    if _sha256(phase3_dir / "phase3_run.json") != observation["phase_3_run_sha256"]:
        raise SummarySourceError("Phase 3 inference run differs from TOML")

    mist_metrics = _verify_aggregate_artifact(
        phase3_dir, phase3_run, "mist_metrics.json"
    )
    comparison = _verify_aggregate_artifact(
        phase3_dir, phase3_run, "comparison.json"
    )
    model_audit = _verify_aggregate_artifact(
        phase3_dir, phase3_run, "model_audit.json"
    )
    runtime = _verify_aggregate_artifact(
        phase3_dir, phase3_run, "runtime_environment.json"
    )
    phase1 = _verify_aggregate_artifact(
        phase3_dir, phase3_run, "phase1_verification.json"
    )
    phase2 = _verify_aggregate_artifact(
        phase3_dir, phase3_run, "phase2_verification.json"
    )
    phase2_metrics = _verify_aggregate_artifact(
        phase2_dir, phase2_run, "test_metrics.json"
    )
    validation = _verify_aggregate_artifact(
        phase2_dir, phase2_run, "validation_metrics.json"
    )

    rf_run = _verify_run_header(rf_dir, "phase2_run.json", "phase2_run.sha256")
    rf_attempt_path = rf_dir / "random_forest_attempt.json"
    rf_attempt_hash = _sha256(rf_attempt_path)
    rf_observation = config["phase_2_observation"][
        "random_forest_validation_supplement"
    ]
    if _sha256(rf_dir / "phase2_run.json") != rf_observation["supplement_run_sha256"]:
        raise SummarySourceError("RF supplement run differs from TOML")
    if (
        rf_attempt_hash != rf_run["random_forest_attempt_sha256"]
        or rf_attempt_hash != rf_observation["attempt_artifact_sha256"]
    ):
        raise SummarySourceError("RF validation artifact differs from its run/TOML")
    rf_attempt = _load_json(rf_attempt_path)

    expected_hashes = {
        "mist_metrics.json": observation["artifacts"]["metrics_sha256"],
        "comparison.json": observation["artifacts"]["comparison_sha256"],
        "model_audit.json": observation["artifacts"]["model_audit_sha256"],
        "runtime_environment.json": observation["artifacts"]["runtime_environment_sha256"],
    }
    for name, expected in expected_hashes.items():
        if _sha256(phase3_dir / name) != expected:
            raise SummarySourceError(f"Phase 3 source hash differs: {name}")
    if comparison["mist_metrics_sha256"] != expected_hashes["mist_metrics.json"]:
        raise SummarySourceError("comparison does not bind the MIST metrics")
    if phase3_audit_run["model_audit_sha256"] != expected_hashes["model_audit.json"]:
        raise SummarySourceError("Phase 3 audit does not bind the model audit")
    if comparison["locked_phase2_metrics_sha256"] != _sha256(
        phase2_dir / "test_metrics.json"
    ):
        raise SummarySourceError("comparison does not bind the locked Ridge metrics")
    if phase2["phase2_run_sha256"] != _sha256(phase2_dir / "phase2_run.json"):
        raise SummarySourceError("Phase 3 does not bind the locked Phase 2 run")
    if tuple(mist_metrics["target_order"]) != TARGET_ORDER:
        raise SummarySourceError("MIST target order differs")
    if set(phase2_metrics["methods"]) < {"ridge"}:
        raise SummarySourceError("locked Ridge metrics are missing")
    if rf_attempt["test_labels_loaded"] or rf_attempt["test_predictions_generated"]:
        raise SummarySourceError("RF supplement is not validation-only")
    if (
        rf_attempt["selected_on_validation_only"]["candidate_id"]
        != rf_observation["selected_candidate"]
        or [
            item["mean_normalized_mae_across_12_targets"]
            for item in rf_attempt["results"]
        ]
        != rf_observation["candidate_scores"]
    ):
        raise SummarySourceError("RF validation selection differs from TOML")

    units = dict(
        zip(
            config["targets"]["ordered_names"],
            config["targets"]["display_units"],
            strict=True,
        )
    )
    cohorts = {
        name: _cohort_summary(
            name=name,
            mist=mist_metrics[name],
            ridge=phase2_metrics["methods"]["ridge"][name],
            units=units,
        )
        for name in ("full_test", "duplicate_clean_test")
    }
    for name, record in cohorts.items():
        comparison_record = comparison["cohorts"][name]
        if record["aggregate"]["mist_minus_ridge"] != comparison_record[
            "mist_minus_locked_ridge_mean_normalized_mae"
        ]:
            raise SummarySourceError(f"comparison aggregate delta differs: {name}")

    selected_rf = rf_attempt["selected_on_validation_only"]
    selected_ridge = validation["ridge"]["selected"]
    return {
        "schema_version": 1,
        "scientific_status": "preliminary-local-point-estimates",
        "artifact_scope": "aggregate-only-no-row-level-data",
        "question": (
            "On the candidate QM9 split reconstructed from public MIST code, how does the "
            "released fine-tuned MIST-28M predictor compare with locked ECFP Ridge?"
        ),
        "target_order": list(TARGET_ORDER),
        "highlighted_targets": list(HIGHLIGHTED_TARGETS),
        "units": units,
        "dataset": {
            "id": config["dataset"]["id"],
            "label_origin": config["dataset"]["label_origin"],
            "source_rows": config["dataset"]["local_verification"]["parsed_rows"],
            "source_sha256": phase1["source_sha256"],
            "split_counts": phase1["split_counts"],
            "split_status": "candidate-reconstructed-not-publisher-certified",
            "full_test_rows": cohorts["full_test"]["rows"],
            "duplicate_clean_test_rows": cohorts["duplicate_clean_test"]["rows"],
        },
        "models": {
            "mist": {
                "label": "Released fine-tuned MIST-28M",
                "model_id": phase3_run["model_id"],
                "revision": phase3_run["model_revision"],
                "operation": "inference-only-no-retraining",
            },
            "ridge": {
                "label": "ECFP4 + multi-output Ridge",
                "selected_candidate": selected_ridge["candidate_id"],
                "parameters_sha256": selected_ridge["parameters_sha256"],
            },
            "random_forest": {
                "label": "ECFP4 + multi-output random forest",
                "scope": "validation-only",
                "selected_candidate": selected_rf["candidate_id"],
                "selected_validation_mean_normalized_mae": selected_rf[
                    "mean_normalized_mae_across_12_targets"
                ],
                "candidate_scores": [
                    {
                        "candidate_id": item["candidate_id"],
                        "mean_normalized_mae_across_12_targets": item[
                            "mean_normalized_mae_across_12_targets"
                        ],
                    }
                    for item in rf_attempt["results"]
                ],
                "test_evaluated": False,
            },
        },
        "cohorts": cohorts,
        "runtime_resources": {
            "phase2_locked_classical": {
                "runtime_seconds": phase2_run["runtime_seconds"],
                "peak_rss_gib": phase2_run["peak_rss_gib"],
            },
            "phase2_random_forest_validation_only": {
                "runtime_seconds": rf_observation["runtime_seconds"],
                "maximum_candidate_peak_rss_gib": rf_observation[
                    "maximum_candidate_peak_rss_gib"
                ],
            },
            "phase3_mist": {
                "device": phase3_run["device"],
                "device_name": runtime["torch_cuda"]["devices"][0]["name"],
                "batch_size": phase3_run["batch_size"],
                "worker_runtime_seconds": phase3_run["worker_runtime_seconds"],
                "total_runtime_seconds": phase3_run["runtime_seconds"],
                "worker_peak_rss_gib": phase3_run["worker_peak_rss_gib"],
                "orchestrator_peak_rss_gib": phase3_run["orchestrator_peak_rss_gib"],
                "gpu_peak_memory_allocated_bytes": phase3_run[
                    "gpu_peak_memory_allocated_bytes"
                ],
                "gpu_peak_memory_reserved_bytes": phase3_run[
                    "gpu_peak_memory_reserved_bytes"
                ],
            },
        },
        "provenance": {
            "phase1_run_sha256": phase1["phase1_run_sha256"],
            "phase2_run_sha256": phase2["phase2_run_sha256"],
            "phase3_audit_run_sha256": _sha256(
                phase3_audit_dir / "phase3_audit_run.json"
            ),
            "phase3_run_sha256": _sha256(phase3_dir / "phase3_run.json"),
            "rf_validation_run_sha256": _sha256(rf_dir / "phase2_run.json"),
            "rf_validation_attempt_sha256": rf_attempt_hash,
            "inference_fingerprint": phase3_run["inference_fingerprint"],
            "execution_protocol_snapshot_sha256": observation[
                "execution_protocol_snapshot_sha256"
            ],
            "model_audit_sha256": phase3_audit_run["model_audit_sha256"],
            "runtime_freeze_canonical_sha256": phase3_audit_run[
                "runtime_freeze_canonical_sha256"
            ],
            "locked_ridge_metrics_sha256": comparison[
                "locked_phase2_metrics_sha256"
            ],
            "mist_metrics_sha256": comparison["mist_metrics_sha256"],
            "comparison_sha256": _sha256(phase3_dir / "comparison.json"),
            "reservation_status": observation["reservation_status"],
            "test_inference_count": observation["test_inference_count"],
            "test_inference_retries": observation["test_inference_retries"],
        },
        "rights": {
            "policy": model_audit["license_boundary"]["conflict_policy"],
            "model_card_restrictions": model_audit["license_boundary"][
                "model_card_use_restrictions"
            ],
            "weights_redistributed": model_audit["license_boundary"][
                "weights_redistributed"
            ],
            "aggregate_only_publication_authorized_by_repository_owner": True,
            "publication_rights_review_open": False,
        },
        "caveats": [
            (
                "The split is a candidate reconstruction from public code, not a "
                "publisher-certified historical checkpoint split."
            ),
            (
                "QM9 labels are DFT-computed quantum-chemistry properties, not "
                "experimental measurements."
            ),
            (
                "These are preliminary point estimates without uncertainty intervals or "
                "repeated inference runs."
            ),
            (
                "The comparison tests a released task-specific predictor against locked "
                "Ridge; it does not isolate pretraining causally."
            ),
            "A lower error does not demonstrate causal or mechanistic chemical understanding.",
            "The random forest is validation-only and has no test metric.",
            (
                "Repository-owner authorization covers this aggregate-only presentation; "
                "raw data, row-level predictions, and weights are not published. "
                "Model-card restrictions remain in effect."
            ),
        ],
    }


def write_summary(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase2-dir", type=Path, default=DEFAULT_PHASE2_DIR)
    parser.add_argument("--rf-dir", type=Path, default=DEFAULT_RF_DIR)
    parser.add_argument("--phase3-audit-dir", type=Path, default=DEFAULT_PHASE3_AUDIT_DIR)
    parser.add_argument("--phase3-dir", type=Path, default=DEFAULT_PHASE3_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--v2-run",
        type=Path,
        help="optional fixed-split v2 result; rejected unless publication review is approved",
    )
    args = parser.parse_args()
    if args.v2_run is not None:
        require_publication_ready(args.v2_run)
    payload = build_summary(
        config_path=args.config,
        phase2_dir=args.phase2_dir,
        rf_dir=args.rf_dir,
        phase3_audit_dir=args.phase3_audit_dir,
        phase3_dir=args.phase3_dir,
    )
    write_summary(args.output, payload)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
