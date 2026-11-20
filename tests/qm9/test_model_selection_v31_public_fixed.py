from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from mist_transfer_benchmark.qm9 import model_selection_v31_medium_profile as medium
from mist_transfer_benchmark.qm9 import model_selection_v31_public_fixed as public
from mist_transfer_benchmark.qm9.model_selection_v31_supervisor import file_sha256


def test_public_fixed_worker_uses_distinct_dynamic_schemas() -> None:
    code = (
        "from mist_transfer_benchmark.qm9 import model_selection_v31_medium_worker as w; "
        "print(w.SPEC_SCHEMA); print(w.PROFILE_SCHEMA)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "QM9_V31_PUBLIC_FIXED_BACKENDS": "1"},
    )
    assert completed.stdout.splitlines() == [
        "qm9-v31-public-fixed-worker-spec-v1",
        "qm9-v31-public-fixed-worker-profile-v1",
    ]


def _external_decision(path: Path, payload: dict[str, object]) -> Path:
    now = datetime.now(UTC)
    unsigned = {
        **payload,
        "issued_at_utc": (now - timedelta(seconds=1)).isoformat(),
        "expires_at_utc": (now + timedelta(minutes=20)).isoformat(),
    }
    artifact = {**unsigned, "semantic_sha256": medium.canonical_sha256(unsigned)}
    path.write_text(json.dumps(artifact, sort_keys=True))
    path.chmod(0o444)
    return path


def _signed(path: Path, payload: dict[str, object]) -> Path:
    artifact = {**payload, "semantic_sha256": medium.canonical_sha256(payload)}
    path.write_text(json.dumps(artifact, sort_keys=True))
    path.chmod(0o444)
    return path


def test_public_fixed_bundle_is_deterministic_group_safe_and_target_scaled(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    public.create_bundle(first)
    public.create_bundle(second)
    one = public.validate_bundle(first)
    two = public.validate_bundle(second)
    assert one["bundle_file_sha256"] == two["bundle_file_sha256"]
    assert one["semantic_sha256"] == two["semantic_sha256"]
    assert one["dimensions"]["rows"] >= 8192
    assert one["dimensions"]["minimum_rows"] == 8192
    assert one["dimensions"]["features"] == 2065
    assert one["dimensions"]["targets"] == 12
    assert one["feature_contract"]["ecfp_like_binary_bits"] == 2048
    assert one["feature_contract"]["dense_descriptors"] == 17
    assert one["group_sizes_unequal"] is True
    assert one["split_group_overlap"] == 0
    spans = [
        high - low
        for low, high in zip(
            one["target_contract"]["per_target_min"],
            one["target_contract"]["per_target_max"],
            strict=True,
        )
    ]
    assert max(spans) / min(spans) > 100_000
    assert one["private_data_allowed"] is False
    assert one["label_reads"]["test"] == 0
    assert one["label_reads"]["protected"] == 0
    assert one["label_reads"]["private_evaluation"] == 0


def test_public_fixed_bundle_rejects_manifest_tampering(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    manifest_path = public.create_bundle(root)
    manifest = json.loads(manifest_path.read_text())
    manifest["train_rows"] += 1
    manifest_path.chmod(0o600)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(public.PublicFixedBundleError, match="manifest contract differs"):
        public.validate_bundle(root)


def test_public_fixed_bundle_is_one_create_only(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    public.create_bundle(root)
    with pytest.raises(public.PublicFixedBundleError, match="already exists"):
        public.create_bundle(root)


def test_public_fixed_prepare_freezes_bundle_before_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle"
    public.create_bundle(bundle)
    base = tmp_path / "run"
    monkeypatch.setattr(medium, "PUBLIC_FIXED_MODE", True)
    monkeypatch.setattr(medium, "RUN_ID", public.RUN_ID)
    monkeypatch.setattr(medium, "PURPOSE", public.PURPOSE)
    monkeypatch.setattr(medium, "SUPERVISOR_PURPOSE", public.PURPOSE)
    monkeypatch.setattr(medium, "EXACT_BASE", base)
    monkeypatch.setattr(
        medium,
        "_EXTERNAL_REVIEW_KEYS",
        medium._EXTERNAL_REVIEW_KEYS | {"purpose", "reviewer_tool_identity"},
    )
    monkeypatch.setattr(
        medium,
        "_EXTERNAL_AUTH_KEYS",
        medium._EXTERNAL_AUTH_KEYS | {"purpose", "authorizer_tool_identity", "one_use"},
    )
    prepared_path = medium.prepare(
        base,
        runner_identity="public-fixed-test-runner",
        public_fixed_bundle=bundle,
    )
    prepared = json.loads(prepared_path.read_text())
    assert prepared["run_id"] == public.RUN_ID
    assert prepared["purpose"] == public.PURPOSE
    assert prepared["label_reads"]["test"] == 0
    assert prepared["label_reads"]["protected"] == 0
    assert prepared["label_reads"]["private_evaluation"] == 0
    assert prepared["input_binding"]["public_fixed_bundle"]["bundle_sha256"]
    assert (base / "selected/public-fixed-selected.npz").is_file()
    assert not (base / "REVIEWED.json").exists()
    assert not (base / "AUTHORIZED.json").exists()


def test_public_fixed_requires_two_external_decisions_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "bundle"
    public.create_bundle(bundle)
    base = tmp_path / "run"
    monkeypatch.setattr(medium, "PUBLIC_FIXED_MODE", True)
    monkeypatch.setattr(medium, "RUN_ID", public.RUN_ID)
    monkeypatch.setattr(medium, "PURPOSE", public.PURPOSE)
    monkeypatch.setattr(medium, "SUPERVISOR_PURPOSE", public.PURPOSE)
    monkeypatch.setattr(medium, "EXACT_BASE", base)
    monkeypatch.setattr(
        medium,
        "_EXTERNAL_REVIEW_KEYS",
        medium._EXTERNAL_REVIEW_KEYS | {"purpose", "reviewer_tool_identity"},
    )
    monkeypatch.setattr(
        medium,
        "_EXTERNAL_AUTH_KEYS",
        medium._EXTERNAL_AUTH_KEYS | {"purpose", "authorizer_tool_identity", "one_use"},
    )
    prepared_path = medium.prepare(
        base, runner_identity="runner", public_fixed_bundle=bundle
    )
    prepared = json.loads(prepared_path.read_text())
    bindings = medium._decision_bindings(prepared, nonce="7" * 64)
    review = _external_decision(
        tmp_path / "external-review.json",
        {
            "schema_version": medium.EXTERNAL_REVIEW_SCHEMA,
            "run_id": public.RUN_ID,
            "decision": medium.REVIEW_DECISION,
            "reviewer_identity": "independent-reviewer",
            "reviewer_tool_identity": "pytest-independent-review-tool",
            "purpose": public.PURPOSE,
            "prepare_file_sha256": file_sha256(prepared_path),
            "prepare_semantic_sha256": prepared["semantic_sha256"],
            **bindings,
        },
    )
    reviewed_path = medium.review(
        base, approval_path=review, expected_approval_sha256=file_sha256(review)
    )
    reviewed = json.loads(reviewed_path.read_text())
    authorization = _external_decision(
        tmp_path / "external-authorization.json",
        {
            "schema_version": medium.EXTERNAL_AUTH_SCHEMA,
            "run_id": public.RUN_ID,
            "decision": medium.AUTH_DECISION,
            "authorizer_identity": "independent-authorizer",
            "authorizer_tool_identity": "pytest-independent-authorization-tool",
            "reviewer_identity": "independent-reviewer",
            "purpose": public.PURPOSE,
            "one_use": True,
            "review_file_sha256": file_sha256(reviewed_path),
            "review_semantic_sha256": reviewed["semantic_sha256"],
            **bindings,
        },
    )
    authorized_path = medium.authorize(
        base,
        authorization_path=authorization,
        expected_authorization_sha256=file_sha256(authorization),
    )
    authorized = json.loads(authorized_path.read_text())
    assert authorized["one_use"] is True
    assert authorized["purpose"] == public.PURPOSE
    assert not (base / "AUTHORIZATION_CONSUMED.json").exists()
    assert not (base / "TERMINAL.json").exists()


def test_public_fixed_archive_requires_bound_independent_postreview(tmp_path: Path) -> None:
    base = tmp_path / "run"
    envelope = tmp_path / "envelope"
    base.mkdir()
    envelope.mkdir()
    postrun = _signed(
        base / "POSTRUN.json",
        {
            "status": "POSTRUN_ACCEPTED",
            "independent_post_review_approved": False,
        },
    )
    (envelope / "result.json").write_text("{}")
    external_review = _signed(tmp_path / "review.json", {"decision": "APPROVE"})
    external_authorization = _signed(
        tmp_path / "authorization.json", {"decision": "AUTHORIZE"}
    )
    postreview = _signed(
        tmp_path / "postreview.json",
        {
            "schema_version": "qm9-v31-public-fixed-postreview-v1",
            "run_id": public.RUN_ID,
            "purpose": public.PURPOSE,
            "decision": "ACCEPT_ENGINEERING_EVIDENCE_ONLY",
            "reviewer_identity": "postreviewer",
            "reviewer_tool_identity": "pytest-review-tool",
            "postrun_file_sha256": file_sha256(postrun),
            "postrun_semantic_sha256": json.loads(postrun.read_text())["semantic_sha256"],
            "base_tree_sha256": medium.canonical_sha256(public._tree_identity(base)),
            "envelope_tree_sha256": medium.canonical_sha256(
                public._tree_identity(envelope)
            ),
            "issued_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    # A synthetic POSTRUN is intentionally not sufficient for archive creation.
    # The archive builder must reject evidence that is not on the verified
    # public-fixed lineage (EXACT_BASE plus a replayable POSTRUN).
    with pytest.raises(public.PublicFixedBundleError, match="verified public fixed lineage"):
        public.build_evidence_archive(
            base=base,
            envelope=envelope,
            external_review=external_review,
            external_authorization=external_authorization,
            postreview=postreview,
            output=tmp_path / "evidence.tar.gz",
        )


def test_public_fixed_archive_rejects_self_consistent_missing_lifecycle(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "evil.txt").write_text("not lifecycle evidence")
    files = public._tree_identity(root)
    unsigned = {
        "schema_version": "qm9-v31-public-fixed-evidence-inventory-v1",
        "run_id": public.RUN_ID,
        "purpose": public.PURPOSE,
        "files": files,
        "files_sha256": medium.canonical_sha256(files),
    }
    (root / "ARCHIVE_MANIFEST.json").write_text(
        json.dumps(
            {**unsigned, "semantic_sha256": medium.canonical_sha256(unsigned)},
            sort_keys=True,
        )
    )
    archive_path = tmp_path / "fake-evidence.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for source in root.iterdir():
            archive.add(source, arcname=f"evidence/{source.name}")
    with pytest.raises(public.PublicFixedBundleError, match="structure differs"):
        public.validate_evidence_archive(archive_path)


def test_public_fixed_archive_rejects_duplicate_tar_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "duplicate-evidence.tar.gz"
    payload = b"duplicate"
    with tarfile.open(archive_path, "w:gz") as archive:
        for _ in range(2):
            member = tarfile.TarInfo("evidence/run/PREPARED.json")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    with pytest.raises(public.PublicFixedBundleError, match="member is unsafe"):
        public.validate_evidence_archive(archive_path)
