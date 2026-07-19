#!/usr/bin/env python3
"""Run authenticated, train-only QM9 v3 model-selection waves."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

import numpy as np

from mist_transfer_benchmark.qm9.data import load_qm9_identities
from mist_transfer_benchmark.qm9.io import sha256_file
from mist_transfer_benchmark.qm9.model_selection_v3 import (
    canonical_hash,
    persist_real_preflight_failure,
    run_real_selection_wave,
    run_smoke_selection,
)
from mist_transfer_benchmark.qm9.paper_evaluation import _array_sha256
from mist_transfer_benchmark.qm9.phase2_contract import verify_phase1_evidence
from mist_transfer_benchmark.qm9.phase2_targets import load_targets_for_indices
from mist_transfer_benchmark.qm9.selection_v3_data import (
    audit_feature_matrix,
    authenticate_feature_artifact,
    canonical_connectivity_groups,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/qm9_model_selection_v3.toml"))
    parser.add_argument("--output", type=Path, default=Path("results/qm9-model-selection-v3"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--wave",
        choices=("scaler-ablation", "feature-screen", "full-selection"),
    )
    parser.add_argument("--qm9-csv", type=Path)
    parser.add_argument("--feature-matrix", type=Path)
    parser.add_argument("--feature-manifest", type=Path)
    parser.add_argument("--phase1-dir", type=Path)
    parser.add_argument("--prior-selection-lock", type=Path)
    return parser


def _authenticated_inputs(args: argparse.Namespace, config: dict) -> dict[str, object]:
    """Authenticate features and load fixed-train labels only."""
    with Path("configs/qm9_28m.toml").open("rb") as handle:
        phase1_config = tomllib.load(handle)
    identities = load_qm9_identities(args.qm9_csv)
    evidence = verify_phase1_evidence(phase1_config, args.phase1_dir, args.qm9_csv)
    matrix, manifest, feature_provenance = authenticate_feature_artifact(
        args.feature_matrix,
        args.feature_manifest,
        expected_source_row_identity_sha256=identities.row_identity_sha256,
    )
    if manifest["source_csv_sha256"] != sha256_file(args.qm9_csv):
        raise ValueError("feature manifest is bound to a different QM9 CSV")
    if manifest["source_smiles_sha256"] != identities.raw_smiles_sha256:
        raise ValueError("feature manifest SMILES identity differs")

    artifact_dir = args.feature_manifest.parent.resolve()
    scaffold_path = (artifact_dir / manifest["scaffold_groups"]["path"]).resolve()
    if scaffold_path.parent != artifact_dir or not scaffold_path.is_file():
        raise ValueError("scaffold artifact path is missing or escapes its manifest directory")
    scaffold_record = manifest["scaffold_groups"]
    if sha256_file(scaffold_path) != scaffold_record["file_sha256"]:
        raise ValueError("scaffold artifact file hash differs")
    scaffold_groups = np.load(scaffold_path, allow_pickle=False).astype(str)
    if (
        scaffold_groups.shape != (identities.row_count,)
        or _array_sha256(scaffold_groups) != scaffold_record["groups_sha256"]
        or len(np.unique(scaffold_groups)) != scaffold_record["count"]
    ):
        raise ValueError("scaffold artifact content differs from its manifest")

    feature_audit = audit_feature_matrix(
        matrix,
        evidence.split.train,
        evidence.split.validation,
        evidence.split.test,
    )
    fixed_train = np.asarray(evidence.split.train, dtype=np.int64)
    # This is the only target read in the runner. Validation/test labels are not
    # accepted by the selection API and cannot influence candidate ranking.
    targets = load_targets_for_indices(args.qm9_csv, fixed_train, identities)
    identity_groups = canonical_connectivity_groups(identities.source_smiles)
    return {
        "matrix": matrix[fixed_train].tocsr(),
        "targets": targets,
        "identity_groups": identity_groups[fixed_train],
        "scaffold_groups": scaffold_groups[fixed_train],
        "fixed_train_source_indices": fixed_train,
        "input_identity": {
            **feature_provenance,
            "source_csv_sha256": sha256_file(args.qm9_csv),
            "source_row_identity_sha256": identities.row_identity_sha256,
            "feature_manifest_sha256": sha256_file(args.feature_manifest),
            "scaffold_groups_file_sha256": sha256_file(scaffold_path),
            "phase1_run_sha256": evidence.phase1_run_sha256,
            "fixed_train_rows": len(fixed_train),
            "fixed_train_ordered_indices": fixed_train.tolist(),
            "fixed_split_ordered_sha256": evidence.split.ordered_hashes(),
            "fixed_split_membership_sha256": evidence.split.membership_hashes(),
            "feature_audit": feature_audit,
            "reviewed_config_sha256": canonical_hash(config),
            "training_seeds": list(config["training_seeds"]),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        with args.config.open("rb") as handle:
            config = tomllib.load(handle)
    except BaseException as error:
        if not args.smoke and args.wave is not None:
            persist_real_preflight_failure(
                {},
                args.output,
                args.wave,
                error,
                {"config_path": str(args.config)},
                args.prior_selection_lock,
            )
        parser.error(f"cannot load config: {type(error).__name__}: {error}")
    private = [args.qm9_csv, args.feature_matrix, args.feature_manifest, args.phase1_dir]
    if any(private) and not all(private):
        parser.error("real selection requires all four authenticated artifact arguments")
    if args.smoke:
        if any(private) or args.wave is not None or args.prior_selection_lock is not None:
            parser.error(
                "--smoke is synthetic only and cannot be combined with private inputs/wave"
            )
        manifest = run_smoke_selection(config, args.output)
    else:
        if not all(private) or args.wave is None:
            parser.error("use explicit --smoke, or provide a --wave and all private artifacts")
        if args.wave == "scaler-ablation" and args.prior_selection_lock is not None:
            error = ValueError("scaler-ablation does not accept --prior-selection-lock")
            persist_real_preflight_failure(
                config,
                args.output,
                args.wave,
                error,
                {
                    "governed_cli_rejection": "W1-prior-lock-forbidden",
                    "qm9_csv": str(args.qm9_csv),
                    "feature_matrix": str(args.feature_matrix),
                    "feature_manifest": str(args.feature_manifest),
                    "phase1_dir": str(args.phase1_dir),
                },
                args.prior_selection_lock,
            )
            parser.error(str(error))
        if args.wave != "scaler-ablation" and args.prior_selection_lock is None:
            parser.error(f"{args.wave} requires --prior-selection-lock")
        try:
            inputs = _authenticated_inputs(args, config)
        except BaseException as error:
            persist_real_preflight_failure(
                config,
                args.output,
                args.wave,
                error,
                {
                    "qm9_csv": str(args.qm9_csv),
                    "feature_matrix": str(args.feature_matrix),
                    "feature_manifest": str(args.feature_manifest),
                    "phase1_dir": str(args.phase1_dir),
                },
                args.prior_selection_lock,
            )
            parser.error(f"authenticated preflight failed: {type(error).__name__}: {error}")
        manifest = run_real_selection_wave(
            config,
            inputs["matrix"],
            inputs["targets"],
            inputs["identity_groups"],
            inputs["scaffold_groups"],
            args.output,
            wave=args.wave,
            input_identity=inputs["input_identity"],
            fixed_train_source_indices=inputs["fixed_train_source_indices"],
            prior_selection_lock=args.prior_selection_lock,
        )
    print(json.dumps({"output": str(args.output), "complete": manifest["complete"]}, indent=2))


if __name__ == "__main__":
    main()
