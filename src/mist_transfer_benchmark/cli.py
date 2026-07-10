"""Command-line interface for validation and ECFP baselines."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from . import __version__
from .artifacts import sha256_file, source_control_metadata, stable_hash, write_run_artifacts
from .baseline import SUPPORTED_MODELS, run_ecfp_baselines
from .cohort import enforce_comparable_cohort
from .fingerprints import FingerprintConfig
from .schema import (
    DATA_CONTRACT_VERSION,
    DataContractError,
    load_validated_csv,
    read_redox_csv,
    validate_redox_dataframe,
)
from .splits import (
    SplitConfig,
    bemis_murcko_scaffold,
    make_split,
    split_counts,
    split_group_keys,
    split_overlap_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mist-transfer",
        description="Leakage-aware molecular property transfer benchmarks.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate a redox CSV data contract")
    validate.add_argument("csv", type=Path)
    validate.add_argument("--json", action="store_true", dest="as_json")

    run = commands.add_parser("run-baseline", help="run deterministic ECFP regressors")
    run.add_argument("csv", type=Path)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument(
        "--split",
        choices=("random", "scaffold", "group", "external"),
        default="scaffold",
    )
    run.add_argument("--group-column", default="chemical_family")
    run.add_argument("--external-column", default="external_set")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--train-fraction", type=float, default=0.7)
    run.add_argument("--validation-fraction", type=float, default=0.15)
    run.add_argument("--test-fraction", type=float, default=0.15)
    run.add_argument(
        "--models",
        default="dummy,tanimoto_1nn,ridge,random_forest",
        help=f"comma-separated model names; choices: {', '.join(SUPPORTED_MODELS)}",
    )
    run.add_argument("--radius", type=int, default=2)
    run.add_argument("--n-bits", type=int, default=2048)
    run.add_argument("--no-chirality", action="store_true")
    run.add_argument(
        "--unsafe-allow-condition-ignorant-mixing",
        action="store_true",
        help="UNSAFE: fit a molecular-only model across heterogeneous target conditions",
    )
    run.add_argument("--overwrite", action="store_true")
    return parser


def _print_validation(report, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return
    status = "VALID" if report.is_valid else "INVALID"
    print(
        f"{status}: {report.row_count} rows, {len(report.errors)} errors, "
        f"{len(report.warnings)} warnings (contract {report.contract_version})"
    )
    for issue in report.issues:
        location = ""
        if issue.row is not None:
            location += f" row={issue.row}"
        if issue.column is not None:
            location += f" column={issue.column}"
        print(f"- {issue.severity.upper()} {issue.code}{location}: {issue.message}")


def _validate_command(args: argparse.Namespace) -> int:
    frame = read_redox_csv(args.csv)
    report = validate_redox_dataframe(frame)
    _print_validation(report, args.as_json)
    return 0 if report.is_valid else 2


def _run_command(args: argparse.Namespace) -> int:
    frame, report = load_validated_csv(args.csv)
    for warning in report.warnings:
        print(f"warning [{warning.code}]: {warning.message}", file=sys.stderr)

    comparability = enforce_comparable_cohort(
        frame,
        unsafe_allow_condition_ignorant_mixing=args.unsafe_allow_condition_ignorant_mixing,
    )
    if comparability["unsafe_override_used"]:
        print(
            "WARNING: unsafe condition-ignorant mixing override is active; aggregate metrics "
            "must not be interpreted as a comparable property benchmark",
            file=sys.stderr,
        )

    split_config = SplitConfig(
        strategy=args.split,
        seed=args.seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        group_column=args.group_column,
        external_column=args.external_column,
    )
    fingerprint_config = FingerprintConfig(
        radius=args.radius,
        n_bits=args.n_bits,
        include_chirality=not args.no_chirality,
    )
    model_names = tuple(name.strip() for name in args.models.split(",") if name.strip())
    assignments = make_split(frame, split_config)
    results, predictions = run_ecfp_baselines(
        frame,
        assignments,
        model_names,
        fingerprint_config,
        args.seed,
    )

    assignment_table = frame[
        [
            "record_id",
            "canonical_smiles",
            "chemical_family",
            "group_id",
            "external_set",
            "source_id",
        ]
    ].copy()
    assignment_table["split"] = assignments.to_numpy()
    assignment_table["scaffold"] = frame["canonical_smiles"].map(bemis_murcko_scaffold)
    assignment_table["split_group_key"] = split_group_keys(frame, split_config).to_numpy()

    data_hash = sha256_file(args.csv)
    protocol = {
        "models": list(model_names),
        "split": asdict(split_config),
        "fingerprint": asdict(fingerprint_config),
        "seed": args.seed,
        "unsafe_allow_condition_ignorant_mixing": bool(
            args.unsafe_allow_condition_ignorant_mixing
        ),
        "comparability": comparability,
    }
    repo_root = Path(__file__).resolve().parents[2]
    source_control = source_control_metadata(repo_root)
    uv_lock_path = repo_root / "uv.lock"
    uv_lock_hash = sha256_file(uv_lock_path) if uv_lock_path.is_file() else None
    run_fingerprint = stable_hash(
        {
            "benchmark_code_version": __version__,
            "data_sha256": data_hash,
            "protocol": protocol,
            "split_rows": assignment_table[["record_id", "split"]].to_dict("records"),
            "source_control": source_control,
            "uv_lock_sha256": uv_lock_hash,
        }
    )
    source_types = sorted(frame["source_type"].unique().tolist())
    artifact: dict[str, object] = {
        "run_fingerprint": run_fingerprint,
        "scientific_status": (
            "software-smoke-test-only"
            if source_types == ["synthetic"]
            else "unreviewed-benchmark-run"
        ),
        "data_contract_version": DATA_CONTRACT_VERSION,
        "dataset": {
            "path_as_invoked": str(args.csv),
            "sha256": data_hash,
            "row_count": len(frame),
            "source_types": source_types,
        },
        "protocol": protocol,
        "split_counts": split_counts(assignments),
        "split_overlap": split_overlap_report(frame, assignments),
        "source_control": source_control,
        "uv_lock_sha256": uv_lock_hash,
        "claim_gate": {
            "ready_for_transfer_claim": False,
            "blocking_items": [
                "a pretrained MIST checkpoint with a downstream-trained regression head and "
                "optional adapter updates is not implemented",
                "learning curves, repeated-seed aggregation, bootstrap intervals, and "
                "uncertainty analysis remain planned",
                "an independently curated external dataset is required",
            ],
        },
        **results,
    }
    run_path = write_run_artifacts(
        args.output_dir,
        artifact,
        predictions,
        assignment_table,
        overwrite=args.overwrite,
    )
    print(f"wrote {run_path}")
    print(f"run fingerprint: {run_fingerprint}")
    if source_types == ["synthetic"]:
        print("scientific status: software smoke test only; synthetic values are not results")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            return _validate_command(args)
        return _run_command(args)
    except DataContractError as error:
        _print_validation(error.report, as_json=False)
        return 2
    except (FileExistsError, FileNotFoundError, ValueError, pd.errors.ParserError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
