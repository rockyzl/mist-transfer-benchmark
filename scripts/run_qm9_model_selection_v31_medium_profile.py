#!/usr/bin/env python3
"""Lifecycle CLI for the fixed QM9 v3.1 train-only medium profiler."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from mist_transfer_benchmark.qm9 import model_selection_v31_medium_profile as medium


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("prepare", "review", "authorize", "execute", "execute-contained", "verify"),
        required=True,
    )
    parser.add_argument("--output", type=Path, default=medium.EXACT_BASE)
    parser.add_argument("--identity")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--decision-artifact", type=Path)
    parser.add_argument("--qm9-csv", type=Path)
    parser.add_argument("--feature-matrix", type=Path)
    parser.add_argument("--feature-manifest", type=Path)
    parser.add_argument("--phase1-dir", type=Path)
    parser.add_argument("--public-fixture-plan", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.stage == "prepare":
        required = (
            args.identity,
            args.qm9_csv,
            args.feature_matrix,
            args.feature_manifest,
            args.phase1_dir,
        )
        if not all(required):
            raise SystemExit("prepare requires identity and all four authenticated input paths")
        path = medium.prepare(
            args.output,
            runner_identity=args.identity,
            qm9_csv=args.qm9_csv,
            feature_matrix=args.feature_matrix,
            feature_manifest=args.feature_manifest,
            phase1_dir=args.phase1_dir,
            public_fixture_plan=args.public_fixture_plan,
        )
    elif args.stage == "review":
        if not args.decision_artifact or not args.expected_sha256:
            raise SystemExit("review requires an external decision artifact and its SHA-256")
        path = medium.review(
            args.output,
            approval_path=args.decision_artifact,
            expected_approval_sha256=args.expected_sha256,
        )
    elif args.stage == "authorize":
        if not args.decision_artifact or not args.expected_sha256:
            raise SystemExit("authorize requires an external decision artifact and its SHA-256")
        path = medium.authorize(
            args.output,
            authorization_path=args.decision_artifact,
            expected_authorization_sha256=args.expected_sha256,
        )
    elif args.stage == "execute":
        if not args.expected_sha256:
            raise SystemExit("execute requires the independently supplied authorization SHA-256")
        authorization = args.output / "AUTHORIZED.json"
        if medium.file_sha256(authorization) != args.expected_sha256:
            raise SystemExit("execute authorization SHA-256 differs")
        prepared = json.loads((args.output / "PREPARED.json").read_text())
        repository = Path(prepared["repository_provenance"]["repository_root"])
        envelope = medium._outer_envelope_for(args.output)
        if envelope.exists():
            raise SystemExit("medium launch envelope is not fresh")
        wrapper = prepared["execution_contract"]["outer_wrapper"]
        contained = wrapper["contained_argv"]
        if contained != prepared["execution_contract"]["argv"]:
            raise SystemExit("execute command differs from independently reviewed exact argv")
        command = medium.materialize_outer_wrapper_argv(
            wrapper,
            authorization_sha256=args.expected_sha256,
            code_state_sha256=prepared["repository_provenance"][
                "harness_code_state_sha256"
            ],
        )
        returncode = subprocess.run(command, cwd=repository, check=False).returncode
        try:
            postrun_path = medium.finalize_postrun(
                args.output,
                launch_status_path=envelope / "status.jsonl",
                launch_result_path=envelope / "result.json",
                expected_authorization_sha256=args.expected_sha256,
            )
        except BaseException as error:
            raise SystemExit(
                f"postrun evidence finalization failed: {type(error).__name__}"
            ) from error
        postrun = json.loads(postrun_path.read_text())
        return returncode if postrun.get("status") == "POSTRUN_ACCEPTED" else 1
    elif args.stage == "execute-contained":
        expected = os.environ.get("QM9_MEDIUM_OUTER_HARNESS_BINDING")
        if not expected:
            raise SystemExit("bare medium execution is forbidden; reviewed outer harness required")
        path = medium.execute(
            args.output,
            expected_authorization_sha256=expected,
            workload=(
                medium.run_public_synthetic_outer_workload
                if medium.PUBLIC_SYNTHETIC_MODE
                else medium.run_real_train_only_workload
            ),
        )
    else:
        result = medium.verify_postrun(args.output)
        print(json.dumps(result, sort_keys=True))
        return 0
    print(json.dumps({"stage": args.stage, "artifact": str(path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
