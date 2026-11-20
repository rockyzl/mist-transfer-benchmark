"""Explicitly opt-in public synthetic test of the complete outer CLI seam."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from mist_transfer_benchmark.qm9 import model_selection_v31_medium_profile as medium
from mist_transfer_benchmark.qm9.model_selection_v31_supervisor import file_sha256

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_QM9_MEDIUM_PUBLIC_OUTER_E2E") != "1",
        reason="explicit opt-in public synthetic full outer CLI seam",
    ),
]


def _external(path: Path, payload: dict[str, object]) -> Path:
    now = datetime.now(UTC)
    unsigned = {
        **payload,
        "issued_at_utc": (now - timedelta(seconds=1)).isoformat(),
        "expires_at_utc": (now + timedelta(minutes=30)).isoformat(),
    }
    artifact = {**unsigned, "semantic_sha256": medium.canonical_sha256(unsigned)}
    path.write_text(json.dumps(artifact, sort_keys=True))
    return path


def test_public_synthetic_full_outer_cli_e2e(
    tmp_path: Path,
) -> None:
    """No private inputs: public run ID/base, real CLI, harness, cgroup, and POSTRUN."""

    base = medium.PUBLIC_SYNTHETIC_BASE
    envelope = medium.PUBLIC_SYNTHETIC_OUTER_ENVELOPE
    assert not base.exists(), "public synthetic lineage must be fresh"
    assert not envelope.exists(), "public synthetic outer envelope must be fresh"
    selected = np.arange(medium.SAMPLE_ROWS, dtype=np.int64)
    groups = np.asarray([f"public-{index}" for index in selected])
    train, tune = medium.medium_split(groups)
    sample = medium.sample_evidence(selected, groups)
    plan = [
        replace(row, identity=replace(row.identity, run_id=medium.PUBLIC_SYNTHETIC_RUN_ID))
        for row in medium.planned_trials(len(tune))
    ]
    matrix = sparse.csr_matrix((len(selected), 1), dtype=np.float64)
    unsigned = {
        "schema_version": "qm9-v31-medium-target-free-plan-v1",
        "run_id": medium.PUBLIC_SYNTHETIC_RUN_ID,
        "selected_source_indices": selected.tolist(),
        "train_positions": train.tolist(),
        "tune_positions": tune.tolist(),
        "sample": sample,
        "selected_feature_sha256": medium._sparse_sha256(matrix),
        "train_feature_sha256": medium._sparse_sha256(matrix[train].tocsr()),
        "tune_feature_sha256": medium._sparse_sha256(matrix[tune].tocsr()),
        "plan": [row.payload() for row in plan],
        "plan_sha256": medium.canonical_sha256([row.payload() for row in plan]),
        "input_inventory_sha256": "4" * 64,
        "label_reads": medium._label_reads(),
        **medium._false_boundaries(),
    }
    target_free = {**unsigned, "semantic_sha256": medium.canonical_sha256(unsigned)}
    fixture_plan = tmp_path / "public-target-free-plan.json"
    fixture_plan.write_text(json.dumps(target_free, sort_keys=True))

    qm9 = tmp_path / "public-identities.csv"
    features = tmp_path / "public-features.npz"
    manifest = tmp_path / "public-feature-manifest.json"
    phase1 = tmp_path / "public-phase1"
    qm9.write_text("public synthetic identities only\n")
    features.write_bytes(b"public synthetic sparse feature fixture")
    manifest.write_text(json.dumps({"schema_version": "qm9-paper-feature-artifact-v1"}))
    phase1.mkdir()
    repository = Path(__file__).parents[2]
    script = repository / "scripts/run_qm9_model_selection_v31_medium_profile.py"
    environment = {**os.environ, "QM9_MEDIUM_PUBLIC_SYNTHETIC_SEAM": "1"}

    def run_stage(*arguments: str, timeout: float = 60) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            [sys.executable, str(script), *arguments],
            cwd=repository,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        assert completed.returncode == 0, completed.stderr
        return completed

    run_stage(
        "--stage", "prepare",
        "--output", str(base),
        "--identity", "public-synthetic-runner",
        "--qm9-csv", str(qm9),
        "--feature-matrix", str(features),
        "--feature-manifest", str(manifest),
        "--phase1-dir", str(phase1),
        "--public-fixture-plan", str(fixture_plan),
    )
    prepared_path = base / "PREPARED.json"
    prepared = json.loads(prepared_path.read_text())
    bindings = medium._decision_bindings(prepared, nonce="9" * 64)
    review = _external(
        tmp_path / "public-review.json",
        {
            "schema_version": medium.EXTERNAL_REVIEW_SCHEMA,
            "run_id": medium.PUBLIC_SYNTHETIC_RUN_ID,
            "decision": "APPROVE_MEDIUM_PROFILE_ONLY",
            "reviewer_identity": "public-seam-reviewer",
            "prepare_file_sha256": file_sha256(prepared_path),
            "prepare_semantic_sha256": prepared["semantic_sha256"],
            **bindings,
        },
    )
    run_stage(
        "--stage", "review", "--output", str(base),
        "--decision-artifact", str(review),
        "--expected-sha256", file_sha256(review),
    )
    reviewed_path = base / "REVIEWED.json"
    reviewed = json.loads(reviewed_path.read_text())
    external_auth = _external(
        tmp_path / "public-authorization.json",
        {
            "schema_version": medium.EXTERNAL_AUTH_SCHEMA,
            "run_id": medium.PUBLIC_SYNTHETIC_RUN_ID,
            "decision": "AUTHORIZE_MEDIUM_PROFILE_ONLY",
            "authorizer_identity": "public-seam-authorizer",
            "reviewer_identity": "public-seam-reviewer",
            "review_file_sha256": file_sha256(reviewed_path),
            "review_semantic_sha256": reviewed["semantic_sha256"],
            **bindings,
        },
    )
    run_stage(
        "--stage", "authorize", "--output", str(base),
        "--decision-artifact", str(external_auth),
        "--expected-sha256", file_sha256(external_auth),
    )
    authorized = base / "AUTHORIZED.json"
    run_stage(
        "--stage", "execute", "--output", str(base),
        "--expected-sha256", file_sha256(authorized),
        timeout=medium.OUTER_SECONDS + 30,
    )
    run_stage("--stage", "verify", "--output", str(base))
    postrun = json.loads((base / "POSTRUN.json").read_text())
    assert postrun["status"] == "POSTRUN_ACCEPTED"
    assert json.loads((base / "TERMINAL.json").read_text())["label_reads"] == (
        medium._label_reads()
    )
