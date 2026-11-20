#!/usr/bin/env python3
"""Dedicated lifecycle CLI for the public-synthetic three-backend v3.1 seam."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

os.environ["QM9_V31_PUBLIC_FIXED_BACKENDS"] = "1"

from mist_transfer_benchmark.qm9 import model_selection_v31_medium_profile as medium  # noqa: E402
from mist_transfer_benchmark.qm9.model_selection_v31_public_fixed import (  # noqa: E402
    create_bundle,
    validate_bundle,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--stage",
        choices=(
            "bundle",
            "prepare",
            "review",
            "authorize",
            "execute",
            "execute-contained",
            "verify",
        ),
        required=True,
    )
    result.add_argument("--output", type=Path, default=medium.EXACT_BASE)
    result.add_argument("--bundle", type=Path)
    result.add_argument("--identity")
    result.add_argument("--decision-artifact", type=Path)
    result.add_argument("--expected-sha256")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.stage == "bundle":
        if args.bundle is None:
            raise SystemExit("bundle stage requires --bundle")
        path = create_bundle(args.bundle)
    elif args.stage == "prepare":
        if args.bundle is None or not args.identity:
            raise SystemExit("prepare requires --bundle and --identity")
        validate_bundle(args.bundle)
        path = medium.prepare(
            args.output,
            runner_identity=args.identity,
            public_fixed_bundle=args.bundle,
        )
    elif args.stage == "review":
        if args.decision_artifact is None or not args.expected_sha256:
            raise SystemExit("review requires an external artifact and out-of-band SHA-256")
        path = medium.review(
            args.output,
            approval_path=args.decision_artifact,
            expected_approval_sha256=args.expected_sha256,
        )
    elif args.stage == "authorize":
        if args.decision_artifact is None or not args.expected_sha256:
            raise SystemExit("authorize requires an external artifact and out-of-band SHA-256")
        path = medium.authorize(
            args.output,
            authorization_path=args.decision_artifact,
            expected_authorization_sha256=args.expected_sha256,
        )
    elif args.stage == "execute":
        if not args.expected_sha256:
            raise SystemExit("execute requires independently supplied authorization SHA-256")
        authorization = args.output / "AUTHORIZED.json"
        if medium.file_sha256(authorization) != args.expected_sha256:
            raise SystemExit("execute authorization SHA-256 differs")
        prepared = json.loads((args.output / "PREPARED.json").read_text())
        repository = Path(prepared["repository_provenance"]["repository_root"])
        envelope = medium._outer_envelope_for(args.output)
        if envelope.exists():
            raise SystemExit("public fixed launch envelope is not fresh")
        wrapper = prepared["execution_contract"]["outer_wrapper"]
        command = medium.materialize_outer_wrapper_argv(
            wrapper,
            authorization_sha256=args.expected_sha256,
            code_state_sha256=prepared["repository_provenance"]["harness_code_state_sha256"],
        )
        returncode = subprocess.run(command, cwd=repository, check=False).returncode
        path = medium.finalize_postrun(
            args.output,
            launch_status_path=envelope / "status.jsonl",
            launch_result_path=envelope / "result.json",
            expected_authorization_sha256=args.expected_sha256,
        )
        postrun = json.loads(path.read_text())
        return returncode if postrun.get("status") == "POSTRUN_ACCEPTED" else 1
    elif args.stage == "execute-contained":
        expected = os.environ.get("QM9_MEDIUM_OUTER_HARNESS_BINDING")
        if not expected:
            raise SystemExit("bare public fixed execution is forbidden")
        path = medium.execute(
            args.output,
            expected_authorization_sha256=expected,
            workload=medium.run_real_train_only_workload,
        )
    else:
        result = medium.verify_postrun(args.output)
        print(json.dumps(result, sort_keys=True))
        return 0
    print(json.dumps({"stage": args.stage, "artifact": str(path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
