#!/usr/bin/env python3
"""Create a human review approval bound to one immutable QM9 v2 run state."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from mist_transfer_benchmark.qm9.fixed_split_evaluation import (
    APPROVAL_SCHEMA,
    PUBLICATION_APPROVAL_SCHEMA,
    FixedSplitEvaluationError,
    file_sha256,
)


def _load_verified_manifest(run: Path) -> tuple[Path, dict[str, object]]:
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, expected in manifest.get("artifact_sha256", {}).items():
        artifact = run / relative
        if not artifact.is_file() or file_sha256(artifact) != expected:
            raise FixedSplitEvaluationError(f"review artifact changed: {relative}")
        if artifact.stat().st_size != manifest["artifact_bytes"].get(relative):
            raise FixedSplitEvaluationError(f"review artifact size changed: {relative}")
    return manifest_path, manifest


def create_approval(
    run: Path,
    stage: str,
    reviewer: str,
    notes: str,
    reviewed_at_utc: str,
) -> dict[str, object]:
    """Build an approval only after verifying the exact artifacts presented for review."""

    if not reviewer.strip() or not notes.strip():
        raise FixedSplitEvaluationError("reviewer and review notes must be nonempty")
    manifest_path, manifest = _load_verified_manifest(run)
    if stage == "selection":
        if manifest.get("stage") != "AWAITING_SELECTION_REVIEW":
            raise FixedSplitEvaluationError("run is not awaiting selection review")
        gate_path = run / "global-freeze-gate.json"
        gate_hash = file_sha256(gate_path)
        if gate_hash != manifest.get("global_freeze_sha256"):
            raise FixedSplitEvaluationError("global freeze differs from the manifest")
        return {
            "schema_version": APPROVAL_SCHEMA,
            "decision": "approve-test-unlock",
            "global_freeze_sha256": gate_hash,
            "reviewed_manifest_sha256": file_sha256(manifest_path),
            "reviewer": reviewer.strip(),
            "reviewed_at_utc": reviewed_at_utc,
            "notes": notes.strip(),
        }
    if stage == "publication":
        if manifest.get("stage") != "AWAITING_PUBLICATION_REVIEW":
            raise FixedSplitEvaluationError("run is not awaiting publication review")
        if manifest.get("publication_ready") is not False:
            raise FixedSplitEvaluationError("publication state is not fail-closed")
        for required in ("summary.json", "loss-monitor.html"):
            if required not in manifest["artifact_sha256"]:
                raise FixedSplitEvaluationError(f"publication review is missing {required}")
        return {
            "schema_version": PUBLICATION_APPROVAL_SCHEMA,
            "decision": "approve-publication",
            "reviewed_manifest_sha256": file_sha256(manifest_path),
            "summary_sha256": file_sha256(run / "summary.json"),
            "reviewer": reviewer.strip(),
            "reviewed_at_utc": reviewed_at_utc,
            "notes": notes.strip(),
        }
    raise FixedSplitEvaluationError(f"unknown review stage: {stage}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", choices=("selection", "publication"), required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--notes", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--approve",
        action="store_true",
        help="explicitly record that the independent review conclusion is approval",
    )
    args = parser.parse_args()
    if not args.approve:
        parser.error("approval artifact creation requires explicit --approve")
    reviewed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    payload = create_approval(
        args.run, args.stage, args.reviewer, args.notes, reviewed_at
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
