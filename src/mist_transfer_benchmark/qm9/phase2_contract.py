"""Authenticate Phase 1 evidence before any Phase 2 label or feature work."""

from __future__ import annotations

import json
import stat
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import EXPECTED_ROW_COUNT, EXPECTED_SOURCE_BYTES, EXPECTED_SOURCE_SHA256
from .download import SourceSnapshot, capture_source_snapshot
from .io import canonical_hash, iter_jsonl, sha256_file
from .output import owner_marker_payload
from .split import CandidateSplit, decimal_lines_sha256, reconstruct_candidate_split


class Phase2ContractError(ValueError):
    """Raised when Phase 1 evidence is missing, altered, or inconsistent."""


@dataclass(frozen=True)
class Phase1Evidence:
    run: dict[str, object]
    split: CandidateSplit
    duplicate_clean_test: np.ndarray
    source_snapshot: SourceSnapshot
    phase1_run_sha256: str
    code_provenance_aggregate_sha256: str


def _regular_file(path: Path) -> None:
    if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.lstat().st_mode):
        raise Phase2ContractError(f"required Phase 1 artifact is not a regular file: {path}")


def verify_phase1_evidence(
    config: dict[str, object],
    phase1_dir: str | Path,
    source_path: str | Path,
) -> Phase1Evidence:
    """Verify Phase 1 v2 contents, hashes, split, provenance, and current source bytes."""

    directory = Path(phase1_dir).resolve(strict=True)
    if directory.is_symlink() or not directory.is_dir():
        raise Phase2ContractError("Phase 1 evidence directory must be a real directory")
    marker = directory / ".qm9-phase1-owner.json"
    _regular_file(marker)
    if json.loads(marker.read_text(encoding="utf-8")) != owner_marker_payload():
        raise Phase2ContractError("Phase 1 ownership marker is invalid")

    run_path = directory / "phase1_run.json"
    sidecar_path = directory / "phase1_run.sha256"
    _regular_file(run_path)
    _regular_file(sidecar_path)
    run_sha256 = sha256_file(run_path)
    if sidecar_path.read_text(encoding="ascii") != f"{run_sha256}  phase1_run.json\n":
        raise Phase2ContractError("Phase 1 run checksum sidecar is invalid")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if run.get("schema_version") != "qm9-phase1-run-v2":
        raise Phase2ContractError("Phase 1 run is not the hardened v2 schema")
    if run.get("scientific_status") != "data-split-duplicate-audit-only-no-model-result":
        raise Phase2ContractError("Phase 1 scientific status is invalid")
    if run.get("next_phase_authorized") is not False:
        raise Phase2ContractError("Phase 1 authorization boundary is invalid")

    artifacts = run.get("artifacts")
    if not isinstance(artifacts, dict):
        raise Phase2ContractError("Phase 1 artifact manifest is missing")
    for name, record in artifacts.items():
        if not isinstance(record, dict) or set(record) != {"file", "bytes", "sha256"}:
            raise Phase2ContractError(f"Phase 1 artifact record is invalid: {name}")
        artifact_path = directory / str(record["file"])
        if artifact_path.parent != directory:
            raise Phase2ContractError("Phase 1 artifact path traversal is not allowed")
        _regular_file(artifact_path)
        if artifact_path.stat().st_size != record["bytes"]:
            raise Phase2ContractError(f"Phase 1 artifact size differs: {name}")
        if sha256_file(artifact_path) != record["sha256"]:
            raise Phase2ContractError(f"Phase 1 artifact SHA-256 differs: {name}")

    source_manifest = json.loads((directory / "source_manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("schema_version") != "qm9-source-manifest-v2":
        raise Phase2ContractError("Phase 1 source manifest schema is invalid")
    if source_manifest.get("source_bytes") != EXPECTED_SOURCE_BYTES:
        raise Phase2ContractError("Phase 1 source byte count differs")
    if source_manifest.get("source_sha256") != EXPECTED_SOURCE_SHA256:
        raise Phase2ContractError("Phase 1 source SHA-256 differs")
    if source_manifest.get("final_cache_and_private_snapshot_equality_verified") is not True:
        raise Phase2ContractError("Phase 1 final source equality evidence is missing")

    source_snapshot = capture_source_snapshot(
        source_path,
        expected_bytes=EXPECTED_SOURCE_BYTES,
        expected_sha256=EXPECTED_SOURCE_SHA256,
    )
    split = reconstruct_candidate_split(EXPECTED_ROW_COUNT)
    phase_split = run.get("split", {})
    if phase_split.get("counts") != split.counts():
        raise Phase2ContractError("Phase 1 split counts differ from reconstruction")
    if phase_split.get("ordered_index_sha256") != split.ordered_hashes():
        raise Phase2ContractError("Phase 1 ordered split hashes differ from reconstruction")
    if phase_split.get("membership_sha256") != split.membership_hashes():
        raise Phase2ContractError("Phase 1 membership hashes differ from reconstruction")
    configured_split = config["phase_1_observation"]["split"]
    expected_assignment_sha = configured_split["assignment_tsv_file_sha256"]
    if sha256_file(directory / "split_assignments.tsv") != expected_assignment_sha:
        raise Phase2ContractError("Phase 1 assignment TSV differs from the frozen config")

    provenance = json.loads((directory / "code_provenance.json").read_text(encoding="utf-8"))
    aggregate = provenance.get("aggregate_sha256")
    unsigned = {key: value for key, value in provenance.items() if key != "aggregate_sha256"}
    if not isinstance(aggregate, str) or canonical_hash(unsigned) != aggregate:
        raise Phase2ContractError("Phase 1 code provenance aggregate is internally invalid")
    if run.get("code_provenance", {}).get("aggregate_sha256") != aggregate:
        raise Phase2ContractError("Phase 1 run and provenance artifact disagree")

    clean_by_source = np.zeros(EXPECTED_ROW_COUNT, dtype=bool)
    observed_rows = 0
    split_names, split_positions = split.assignment_arrays(EXPECTED_ROW_COUNT)
    for expected_index, row in enumerate(iter_jsonl(directory / "row_manifest.jsonl")):
        if row.get("source_row_index") != expected_index:
            raise Phase2ContractError("Phase 1 row manifest is not source ordered")
        if row.get("split") != str(split_names[expected_index]):
            raise Phase2ContractError("Phase 1 row manifest split differs")
        if row.get("split_position") != int(split_positions[expected_index]):
            raise Phase2ContractError("Phase 1 row manifest split position differs")
        if type(row.get("duplicate_clean_test")) is not bool:
            raise Phase2ContractError("Phase 1 duplicate-clean flag has an invalid type")
        clean_by_source[expected_index] = row["duplicate_clean_test"]
        observed_rows += 1
    if observed_rows != EXPECTED_ROW_COUNT:
        raise Phase2ContractError("Phase 1 row manifest does not cover every source row")
    clean_test = np.asarray(
        [int(index) for index in split.test if clean_by_source[int(index)]], dtype=np.int64
    )
    duplicate = config["phase_1_observation"]["duplicates"]
    if len(clean_test) != duplicate["duplicate_clean_test_rows"]:
        raise Phase2ContractError("duplicate-clean test count differs from the frozen config")
    if decimal_lines_sha256(clean_test) != duplicate[
        "duplicate_clean_retained_ordered_index_sha256"
    ]:
        raise Phase2ContractError("duplicate-clean test hash differs from the frozen config")

    return Phase1Evidence(
        run=run,
        split=split,
        duplicate_clean_test=clean_test,
        source_snapshot=source_snapshot,
        phase1_run_sha256=run_sha256,
        code_provenance_aggregate_sha256=aggregate,
    )
