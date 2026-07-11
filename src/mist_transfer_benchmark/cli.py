"""Command-line interface for redox benchmarks and the isolated QM9 Phase 1 audit."""

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

    repo_root = Path(__file__).resolve().parents[2]
    qm9 = commands.add_parser(
        "qm9-audit",
        help="download, validate, split, and duplicate-audit QM9 without running models",
    )
    qm9.add_argument("--config", type=Path, default=repo_root / "configs/qm9_28m.toml")
    qm9.add_argument("--cache-dir", type=Path, default=repo_root / "data/private/qm9")
    qm9.add_argument("--output-dir", type=Path, required=True)
    qm9.add_argument(
        "--datasets-python",
        type=Path,
        required=True,
        help="Python from an isolated datasets==3.2.0, numpy==2.5.1 environment",
    )
    qm9.add_argument("--force-download", action="store_true")
    qm9.add_argument("--overwrite", action="store_true")

    features = commands.add_parser(
        "qm9-features",
        help="authenticate Phase 1 and build the frozen full-row QM9 ECFP4 CSR artifact",
    )
    features.add_argument("--config", type=Path, default=repo_root / "configs/qm9_28m.toml")
    features.add_argument("--cache-dir", type=Path, default=repo_root / "data/private/qm9")
    features.add_argument("--phase1-dir", type=Path, required=True)
    features.add_argument("--output-dir", type=Path, required=True)
    features.add_argument("--overwrite", action="store_true")

    classical = commands.add_parser(
        "qm9-classical",
        help="run frozen Ridge/controls validation and exactly-once classical test evaluation",
    )
    classical.add_argument("--config", type=Path, default=repo_root / "configs/qm9_28m.toml")
    classical.add_argument("--cache-dir", type=Path, default=repo_root / "data/private/qm9")
    classical.add_argument("--phase1-dir", type=Path, required=True)
    classical.add_argument("--feature-dir", type=Path, required=True)
    classical.add_argument("--output-dir", type=Path, required=True)
    classical.add_argument("--overwrite", action="store_true")

    rf = commands.add_parser(
        "qm9-rf-supplement",
        help="run bounded random-forest validation only; never read or predict the test split",
    )
    rf.add_argument("--config", type=Path, default=repo_root / "configs/qm9_28m.toml")
    rf.add_argument("--cache-dir", type=Path, default=repo_root / "data/private/qm9")
    rf.add_argument("--phase1-dir", type=Path, required=True)
    rf.add_argument("--feature-dir", type=Path, required=True)
    rf.add_argument("--locked-run-dir", type=Path, required=True)
    rf.add_argument("--output-dir", type=Path, required=True)
    rf.add_argument("--overwrite", action="store_true")

    mist_audit = commands.add_parser(
        "qm9-mist-audit",
        help="verify the pinned local MIST snapshot and isolated runtime without executing it",
    )
    mist_audit.add_argument(
        "--config", type=Path, default=repo_root / "configs/qm9_28m.toml"
    )
    mist_audit.add_argument("--model-dir", type=Path, required=True)
    mist_audit.add_argument("--runtime-python", type=Path, required=True)
    mist_audit.add_argument("--output-dir", type=Path, required=True)
    mist_audit.add_argument("--overwrite", action="store_true")

    mist_acquire = commands.add_parser(
        "qm9-mist-acquire",
        help="atomically acquire or verify only the one pinned MIST model revision",
    )
    mist_acquire.add_argument(
        "--model-dir",
        type=Path,
        default=repo_root / "data/private/qm9/mist-phase3/model",
    )

    mist = commands.add_parser(
        "qm9-mist-infer",
        help="run guarded pinned MIST smoke and exactly-once candidate-test inference",
    )
    mist.add_argument("--config", type=Path, default=repo_root / "configs/qm9_28m.toml")
    mist.add_argument("--cache-dir", type=Path, default=repo_root / "data/private/qm9")
    mist.add_argument("--phase1-dir", type=Path, required=True)
    mist.add_argument("--phase2-dir", type=Path, required=True)
    mist.add_argument("--audit-dir", type=Path, required=True)
    mist.add_argument("--model-dir", type=Path, required=True)
    mist.add_argument("--runtime-python", type=Path, required=True)
    mist.add_argument("--output-dir", type=Path, required=True)
    mist.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    mist.add_argument("--initial-batch-size", type=int, default=128)
    mist.add_argument("--overwrite", action="store_true")
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


def _qm9_audit_command(args: argparse.Namespace) -> int:
    from .qm9.pipeline import run_phase1_audit

    run = run_phase1_audit(
        config_path=args.config,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        datasets_python=args.datasets_python,
        force_download=args.force_download,
        overwrite=args.overwrite,
        command=["mist-transfer", *sys.argv[1:]],
    )
    print(f"wrote {args.output_dir / 'phase1_run.json'}")
    print(f"source sha256: {run['source']['source_sha256']}")
    print(f"split counts: {json.dumps(run['split']['counts'], sort_keys=True)}")
    train_test = run["duplicates"]["train_test_overlap"]
    print(f"train-test duplicate identities: {train_test['identity_count']}")
    print("scientific status: Phase 1 data/split audit only; no model result")
    return 0


def _qm9_classical_command(args: argparse.Namespace) -> int:
    from .qm9.phase2_pipeline import run_phase2_classical

    run = run_phase2_classical(
        config_path=args.config,
        cache_dir=args.cache_dir,
        phase1_dir=args.phase1_dir,
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        run_random_forest=False,
    )
    print(f"wrote {args.output_dir / 'phase2_run.json'}")
    print(f"selection fingerprint: {run['selection_fingerprint']}")
    print(f"scientific status: {run['scientific_status']}")
    return 0


def _qm9_features_command(args: argparse.Namespace) -> int:
    from .qm9.phase2_pipeline import run_phase2_feature_stage

    result = run_phase2_feature_stage(
        config_path=args.config,
        cache_dir=args.cache_dir,
        phase1_dir=args.phase1_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(f"wrote {args.output_dir / 'feature_manifest.json'}")
    print(f"canonical CSR sha256: {result['matrix']['canonical_csr_sha256']}")
    return 0


def _qm9_rf_supplement_command(args: argparse.Namespace) -> int:
    from .qm9.phase2_rf_attempt import run_rf_validation_supplement

    result = run_rf_validation_supplement(
        config_path=args.config,
        cache_dir=args.cache_dir,
        phase1_dir=args.phase1_dir,
        feature_dir=args.feature_dir,
        locked_run_dir=args.locked_run_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(f"wrote {args.output_dir / 'random_forest_attempt.json'}")
    print(f"validation-only status: {result['status']}")
    print("test labels loaded: false")
    return 0


def _qm9_mist_audit_command(args: argparse.Namespace) -> int:
    from .qm9.phase3_pipeline import run_phase3_audit

    result = run_phase3_audit(
        config_path=args.config,
        model_dir=args.model_dir,
        runtime_python=args.runtime_python,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(f"wrote {args.output_dir / 'phase3_audit_run.json'}")
    print(f"model revision: {result['model_revision']}")
    print("remote code executed: false")
    return 0


def _qm9_mist_acquire_command(args: argparse.Namespace) -> int:
    from .qm9.phase3_acquire import acquire_snapshot

    repo_root = Path(__file__).resolve().parents[2]
    result = acquire_snapshot(args.model_dir, repo_root=repo_root)
    print(f"model revision: {result['revision']}")
    print(f"retrieval mode: {result['retrieval_mode']}")
    print(f"verified files: {len(result['files'])}")
    return 0


def _qm9_mist_infer_command(args: argparse.Namespace) -> int:
    from .qm9.phase3_pipeline import run_phase3_inference

    result = run_phase3_inference(
        config_path=args.config,
        cache_dir=args.cache_dir,
        phase1_dir=args.phase1_dir,
        phase2_dir=args.phase2_dir,
        audit_dir=args.audit_dir,
        model_dir=args.model_dir,
        runtime_python=args.runtime_python,
        output_dir=args.output_dir,
        device=args.device,
        initial_batch_size=args.initial_batch_size,
        overwrite=args.overwrite,
    )
    print(f"wrote {args.output_dir / 'phase3_run.json'}")
    print(f"inference fingerprint: {result['inference_fingerprint']}")
    print(f"scientific status: {result['scientific_status']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            return _validate_command(args)
        if args.command == "run-baseline":
            return _run_command(args)
        if args.command == "qm9-audit":
            return _qm9_audit_command(args)
        if args.command == "qm9-features":
            return _qm9_features_command(args)
        if args.command == "qm9-classical":
            return _qm9_classical_command(args)
        if args.command == "qm9-rf-supplement":
            return _qm9_rf_supplement_command(args)
        if args.command == "qm9-mist-audit":
            return _qm9_mist_audit_command(args)
        if args.command == "qm9-mist-acquire":
            return _qm9_mist_acquire_command(args)
        return _qm9_mist_infer_command(args)
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
