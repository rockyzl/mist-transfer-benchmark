from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mist_transfer_benchmark.qm9 import model_selection_v31_profile_smoke as smoke
from mist_transfer_benchmark.qm9.model_selection_v31_checkpoint import CompletionContract
from mist_transfer_benchmark.qm9.model_selection_v31_supervisor import SupervisorError


def _write_postrun_binding_fixture(base: Path) -> None:
    (base / "run/final").mkdir(parents=True)
    (base / "run/ledger").mkdir()
    (base / "post-profile").mkdir()
    smoke._write_immutable_json(
        base / "run/run_start.json",
        {"schema_version": smoke.RUN_START_SCHEMA, "fixture": True},
    )
    (base / "run/ledger/trials.jsonl").write_text("fixture\n")
    (base / "run/final/manifest.json").write_text("{}\n")
    summary = {
        "schema_version": smoke.SUMMARY_SCHEMA,
        "purpose": "engineering_smoke",
        "status": "complete",
        "run_id": smoke.RUN_ID,
        "trial_states": {"ridge": "complete", "xgboost": "complete", "mlp": "complete"},
        "event_count": 10,
        "checkpoint_count": 3,
        "replay_count": 3,
        "replay_sha256": "a" * 64,
        "ledger_sha256": "b" * 64,
        "duration_seconds": 1.0,
        "resource_peaks": {},
        "profiles": [],
        "mlp_cache": {},
        "protected_label_reads": 0,
        **smoke._false_boundaries(),
    }
    summary_artifact = smoke._write_immutable_json(base / "profile-summary.json", summary)
    proposal_artifact = smoke._write_immutable_json(
        base / "post-profile/proposal.json",
        {
            "schema_version": smoke.POST_PROFILE_SCHEMA,
            "measurement_status": "INSUFFICIENT_OBSERVATIONS",
            "decision": "NOT_APPROVED",
            "observations_per_family": 1,
            "families": ["ridge", "xgboost", "mlp"],
            "w1_feasibility_approved": False,
            "general_training_approved": False,
            "scientific_result": False,
            "publication_ready": False,
        },
    )
    inventory = smoke._lstat_tree(base)
    path_types = [
        {"relative_path": row["relative_path"], "entry_type": row["entry_type"]}
        for row in inventory
    ]
    run_start = json.loads((base / "run/run_start.json").read_text())
    smoke._exclusive_artifact(
        base / "POSTRUN_MANIFEST.json",
        smoke.POSTRUN_MANIFEST_SCHEMA,
        {
            "status": "PROFILE_COMPLETE",
            "run_id": smoke.RUN_ID,
            "purpose": smoke.PURPOSE,
            "created_at_utc": "2026-07-19T12:00:00Z",
            "success_lock": True,
            "streams_closed": {"ledger": True, "telemetry": True},
            "expected_final_path_types": path_types,
            "expected_final_path_types_sha256": smoke.canonical_sha256(path_types),
            "run_start_path": str((base / "run/run_start.json").resolve()),
            "run_start_file_sha256": smoke.file_sha256(base / "run/run_start.json"),
            "run_start_semantic_sha256": run_start["semantic_sha256"],
            "ledger_path": str((base / "run/ledger/trials.jsonl").resolve()),
            "ledger_file_sha256": smoke.file_sha256(base / "run/ledger/trials.jsonl"),
            "terminal_path": str((base / "run/final/manifest.json").resolve()),
            "terminal_file_sha256": smoke.file_sha256(base / "run/final/manifest.json"),
            "summary_path": str((base / "profile-summary.json").resolve()),
            "summary_file_sha256": smoke.file_sha256(base / "profile-summary.json"),
            "summary_semantic_sha256": summary_artifact["semantic_sha256"],
            "proposal_path": str((base / "post-profile/proposal.json").resolve()),
            "proposal_file_sha256": smoke.file_sha256(base / "post-profile/proposal.json"),
            "proposal_semantic_sha256": proposal_artifact["semantic_sha256"],
            "execution_repository_provenance_sha256": "c" * 64,
            "final_base_inventory": inventory,
            "final_base_inventory_sha256": smoke.canonical_sha256(inventory),
            **smoke._false_boundaries(),
        },
    )


def test_synthetic_fixture_is_exact_and_deterministic(tmp_path: Path) -> None:
    left = smoke.write_synthetic_fixture(tmp_path / "left.npz")
    right = smoke.write_synthetic_fixture(tmp_path / "right.npz")
    with np.load(left, allow_pickle=False) as first, np.load(right, allow_pickle=False) as second:
        assert set(first.files) == {"x_train", "y_train", "x_tune", "y_tune"}
        assert first["x_train"].shape == (72, 64)
        assert first["y_train"].shape == (72, 12)
        assert first["x_tune"].shape == (24, 64)
        assert first["y_tune"].shape == (24, 12)
        for name in first.files:
            np.testing.assert_array_equal(first[name], second[name])
            assert np.all(np.isfinite(first[name]))


def test_frozen_plan_is_three_cpu_profile_only_trials() -> None:
    trials = smoke.planned_trials()
    assert [row.identity.family for row in trials] == ["ridge", "xgboost", "mlp"]
    assert [row.identity.trial_index for row in trials] == [1, 2, 3]
    assert all(row.identity.wave == "PROFILE_ONLY" for row in trials)
    assert all(row.identity.seed == 20260719 for row in trials)
    assert all(row.expected_shape == (24, 12) for row in trials)
    for family in ("ridge", "xgboost", "mlp"):
        policy = smoke.sidecar_policy(family)
        assert policy.require_rss_sensor is True
        assert policy.require_disk_sensor is True
        assert policy.require_gpu_sensors is False
        assert policy.rss_max_bytes == 2 * 1024**3


def test_repository_root_discovers_actual_lock_location() -> None:
    root = smoke._repository_root()
    assert root == Path(smoke.__file__).resolve().parents[3]
    assert (root / "pyproject.toml").is_file()
    assert (root / "uv.lock").is_file()
    preflight = smoke._repository_preflight()
    assert set(preflight) == smoke._repository_provenance_keys()
    assert preflight["repository_root"] == str(root)
    assert preflight["git_top_level"] == str(root)
    assert preflight["project_name"] == smoke.EXPECTED_PROJECT_NAME
    assert preflight["pyproject_sha256"] == smoke.file_sha256(root / "pyproject.toml")
    assert preflight["dependency_sha256"] == smoke.file_sha256(root / "uv.lock")
    assert preflight["code_state_sha256"] == smoke._code_state_sha256(root)
    unsigned_identity = dict(preflight)
    identity_sha256 = unsigned_identity.pop("repository_identity_sha256")
    assert identity_sha256 == smoke.canonical_sha256(unsigned_identity)
    changed = {**unsigned_identity, "pyproject_sha256": "0" * 64}
    assert smoke.canonical_sha256(changed) != identity_sha256


def test_prepare_wrong_repository_root_refuses_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "bounded"
    monkeypatch.setattr(smoke, "EXACT_BASE", base)
    monkeypatch.setattr(
        smoke.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=str(tmp_path / "wrong-root")),
    )
    with pytest.raises(SupervisorError, match="git top-level differs"):
        smoke.prepare(base, runner_identity="pytest-runner")
    assert not base.exists()


@pytest.mark.parametrize("lock_kind", ["missing", "symlink"])
def test_prepare_invalid_lock_refuses_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lock_kind: str
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    if lock_kind == "symlink":
        target = tmp_path / "actual.lock"
        target.write_text("fixture")
        (repository / "uv.lock").symlink_to(target)
    base = tmp_path / "bounded"
    monkeypatch.setattr(smoke, "EXACT_BASE", base)
    monkeypatch.setattr(smoke, "_repository_root", lambda *args: repository)
    with pytest.raises(SupervisorError, match="repository prerequisite"):
        smoke.prepare(base, runner_identity="pytest-runner")
    assert not base.exists()


def test_repository_resolver_rejects_ambiguous_ancestors(tmp_path: Path) -> None:
    outer = tmp_path / "outer"
    inner = outer / "src/mist_transfer_benchmark/nested"
    source = inner / "src/mist_transfer_benchmark/qm9/runner.py"
    source.parent.mkdir(parents=True)
    source.write_text("# fixture\n")
    for candidate in (outer, inner):
        (candidate / "src/mist_transfer_benchmark").mkdir(parents=True, exist_ok=True)
        (candidate / "pyproject.toml").write_text("[project]\nname='mist-transfer-benchmark'\n")
        (candidate / "uv.lock").write_text("fixture\n")
    with pytest.raises(SupervisorError, match="missing or ambiguous"):
        smoke._repository_root(source)


def test_prepare_wrong_project_identity_refuses_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "pyproject.toml").write_text("[project]\nname='wrong-project'\n")
    (repository / "uv.lock").write_text("fixture\n")
    base = tmp_path / "bounded"
    monkeypatch.setattr(smoke, "EXACT_BASE", base)
    monkeypatch.setattr(smoke, "_repository_root", lambda *args: repository)
    with pytest.raises(SupervisorError, match="project identity differs"):
        smoke.prepare(base, runner_identity="pytest-runner")
    assert not base.exists()


def test_prepare_success_writes_claim_then_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "bounded"
    monkeypatch.setattr(smoke, "EXACT_BASE", base)
    original_validator = smoke._validate_semantic_artifact

    def fake_prepare(
        claimed_base: Path,
        **kwargs: object,
    ) -> Path:
        claim_path = claimed_base / "prepare-claim.json"
        claim = json.loads(claim_path.read_text())
        assert claim["schema_version"] == smoke.PREPARE_CLAIM_SCHEMA
        assert kwargs["progress"] == {"last_completed_checkpoint": "prepare-claim-written"}
        preflight = claimed_base / "preflight"
        preflight.mkdir()
        profile = preflight / "bootstrap-profile.json"
        profile.write_text("{}")
        return profile

    monkeypatch.setattr(smoke, "_prepare_claimed", fake_prepare)
    monkeypatch.setattr(
        smoke,
        "_validate_semantic_artifact",
        lambda *args, **kwargs: {"semantic_sha256": "a" * 64},
    )
    assert smoke.prepare(base, runner_identity="pytest-runner") == (
        base / "preflight/bootstrap-profile.json"
    )
    monkeypatch.setattr(smoke, "_validate_semantic_artifact", original_validator)
    claim = original_validator(base / "prepare-claim.json", smoke.PREPARE_CLAIM_SCHEMA)
    complete = original_validator(base / "PREPARE_COMPLETE.json", smoke.PREPARE_COMPLETE_SCHEMA)
    assert complete["prepare_attempt_id"] == claim["prepare_attempt_id"]
    assert complete["prepare_claim_file_sha256"] == smoke.file_sha256(base / "prepare-claim.json")
    assert complete["authorization_allowed"] is True
    assert complete["execution_allowed"] is True


def test_review_and_build_binding_refuse_missing_prepare_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "bounded"
    preflight = base / "preflight"
    preflight.mkdir(parents=True)
    monkeypatch.setattr(smoke, "EXACT_BASE", base)
    with pytest.raises(SupervisorError, match="regular non-symlink"):
        smoke.freeze_bounded_bootstrap_review(
            preflight / "review.json",
            profile_path=preflight / "bootstrap-profile.json",
            reviewer_identity="reviewer-b",
            review_tool_identity="pytest-review-tool",
            review_tool_code_sha256="d" * 64,
            runner_identity="runner-a",
        )
    with pytest.raises(SupervisorError, match="regular non-symlink"):
        smoke.build_binding(
            preflight,
            expected_review_file_sha256="a" * 64,
            expected_authorization_file_sha256="b" * 64,
            expected_reviewer_identity="reviewer-b",
        )


def test_prepare_state_refuses_failure_and_v1_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "bounded"
    base.mkdir()
    (base / "PREPARE_FAILED.json").write_text("failure\n")
    monkeypatch.setattr(smoke, "EXACT_BASE", base)
    with pytest.raises(SupervisorError, match="authorization and execution are forbidden"):
        smoke._validate_prepare_state(base)
    with pytest.raises(SupervisorError, match="V1_INCIDENT_INCOMPLETE"):
        smoke._require_v5_base(smoke.V1_INCIDENT_BASE)
    with pytest.raises(SupervisorError, match="V2_EVIDENCE_READ_ONLY"):
        smoke._require_v5_base(smoke.V2_COMPLETED_BASE)
    with pytest.raises(SupervisorError, match="V3_EVIDENCE_READ_ONLY"):
        smoke._require_v5_base(smoke.V3_REJECTED_BASE)
    with pytest.raises(SupervisorError, match="V4_EVIDENCE_READ_ONLY"):
        smoke._require_v5_base(smoke.V4_READ_ONLY_BASE)


def test_v5_exact_base_accepts_prepare_fixture_and_v4_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    v4 = tmp_path / "v4"
    v5 = tmp_path / "v5"
    monkeypatch.setattr(smoke, "V4_READ_ONLY_BASE", v4)
    monkeypatch.setattr(smoke, "EXACT_BASE", v5)
    with pytest.raises(SupervisorError, match="V4_EVIDENCE_READ_ONLY"):
        smoke.prepare(v4, runner_identity=smoke.DEFAULT_RUNNER_IDENTITY)

    def fixture_prepare(claimed_base: Path, **_kwargs: object) -> Path:
        preflight = claimed_base / "preflight"
        preflight.mkdir()
        profile = preflight / "bootstrap-profile.json"
        profile.write_text("{}")
        return profile

    monkeypatch.setattr(smoke, "_prepare_claimed", fixture_prepare)
    monkeypatch.setattr(
        smoke,
        "_validate_semantic_artifact",
        lambda *_args, **_kwargs: {"semantic_sha256": "a" * 64},
    )
    prepared = smoke.prepare(v5, runner_identity=smoke.DEFAULT_RUNNER_IDENTITY)
    assert prepared == v5 / "preflight/bootstrap-profile.json"
    assert json.loads((v5 / "prepare-claim.json").read_text())["run_id"] == smoke.RUN_ID
    assert not v4.exists()


def test_v1_v4_refusal_preserves_existing_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    v1 = tmp_path / "v1"
    v2 = tmp_path / "v2"
    v3 = tmp_path / "v3"
    v4 = tmp_path / "v4"
    historical = (
        (v1, b"v1-evidence"),
        (v2, b"v2-evidence"),
        (v3, b"v3-evidence"),
        (v4, b"v4-evidence"),
    )
    for path, content in historical:
        path.mkdir()
        (path / "sentinel.bin").write_bytes(content)
    monkeypatch.setattr(smoke, "V1_INCIDENT_BASE", v1)
    monkeypatch.setattr(smoke, "V2_COMPLETED_BASE", v2)
    monkeypatch.setattr(smoke, "V3_REJECTED_BASE", v3)
    monkeypatch.setattr(smoke, "V4_READ_ONLY_BASE", v4)
    monkeypatch.setattr(smoke, "EXACT_BASE", tmp_path / "v5")
    before = {
        path: (path.lstat().st_mode, smoke.file_sha256(path / "sentinel.bin"))
        for path, _content in historical
    }
    for path, message in (
        (v1, "V1_INCIDENT_INCOMPLETE"),
        (v2, "V2_EVIDENCE_READ_ONLY"),
        (v3, "V3_EVIDENCE_READ_ONLY"),
        (v4, "V4_EVIDENCE_READ_ONLY"),
    ):
        with pytest.raises(SupervisorError, match=message):
            smoke._require_v5_base(path)
    after = {
        path: (path.lstat().st_mode, smoke.file_sha256(path / "sentinel.bin"))
        for path, _content in historical
    }
    assert after == before


def test_v5_build_binding_validates_and_terminal_schema_is_v5(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "v5"
    preflight = base / "preflight"
    preflight.mkdir(parents=True)
    paths = {
        "profile": preflight / "bootstrap-profile.json",
        "review": preflight / "independent-review.json",
        "authorization": preflight / "bootstrap-authorization.json",
        "claim": base / "prepare-claim.json",
        "complete": base / "PREPARE_COMPLETE.json",
    }
    for path in paths.values():
        path.write_text("fixture\n")
    digest = "a" * 64
    provenance = {"head": "fixture"}
    device = smoke.detect_device_identity("cpu")
    claim = {"prepare_attempt_id": digest, "semantic_sha256": "b" * 64}
    complete = {"semantic_sha256": "c" * 64}
    complete_fields = {
        "prepare_complete_file_sha256": smoke.file_sha256(paths["complete"]),
        "prepare_complete_path": str(paths["complete"].resolve()),
        "prepare_complete_semantic_sha256": complete["semantic_sha256"],
    }
    profile = {
        "plan_sha256": digest,
        "config_sha256": digest,
        "input_file_sha256": digest,
        "cache_choice_file_sha256": digest,
        "code_sha256": digest,
        "dependency_sha256": digest,
        "repository_provenance": provenance,
        "repository_provenance_sha256": smoke.canonical_sha256(provenance),
        "device_identity": device,
        "device_identity_sha256": smoke.canonical_sha256(device),
        "semantic_sha256": "d" * 64,
    }
    review = {
        "reviewer_identity": "reviewer-v5",
        "semantic_sha256": "e" * 64,
        **complete_fields,
    }
    authorization = {
        "reviewer_identity": "reviewer-v5",
        "semantic_sha256": "f" * 64,
        **complete_fields,
    }
    artifacts = {
        smoke.PROFILE_SCHEMA: profile,
        smoke.REVIEW_SCHEMA: review,
        smoke.AUTH_SCHEMA: authorization,
    }
    original_validator = smoke._validate_semantic_artifact
    monkeypatch.setattr(smoke, "EXACT_BASE", base)
    monkeypatch.setattr(
        smoke, "_validate_prepare_state", lambda *_args, **_kwargs: (claim, complete)
    )
    monkeypatch.setattr(
        smoke,
        "_validate_semantic_artifact",
        lambda _path, schema, **_kwargs: artifacts[schema],
    )
    binding = smoke.build_binding(
        preflight,
        expected_review_file_sha256=smoke.file_sha256(paths["review"]),
        expected_authorization_file_sha256=smoke.file_sha256(paths["authorization"]),
        expected_reviewer_identity="reviewer-v5",
    )
    binding.validate()
    assert binding.run_id == smoke.RUN_ID == "qm9-v31-bounded-smoke-cpu-v5"
    assert smoke.BOUNDED_TERMINAL_SCHEMA == "qm9-v31-bounded-profile-terminal-v5"

    monkeypatch.setattr(smoke, "_validate_semantic_artifact", original_validator)
    terminal = smoke._write_immutable_json(
        base / "terminal-v5.json",
        {"schema_version": smoke.BOUNDED_TERMINAL_SCHEMA, "run_id": smoke.RUN_ID},
    )
    assert (
        smoke._validate_semantic_artifact(base / "terminal-v5.json", smoke.BOUNDED_TERMINAL_SCHEMA)
        == terminal
    )
    with pytest.raises(SupervisorError, match="semantic hash differs"):
        smoke._validate_semantic_artifact(
            base / "terminal-v5.json", "qm9-v31-bounded-profile-terminal-v4"
        )


@pytest.mark.parametrize("entry_kind", ["file", "symlink"])
def test_complete_base_inventory_rejects_unexpected_base_entry(
    tmp_path: Path, entry_kind: str
) -> None:
    base = tmp_path / "bounded"
    (base / "preflight").mkdir(parents=True)
    (base / "prepare-claim.json").write_text("claim\n")
    inventory = smoke._lstat_tree(base)
    unexpected = base / "unexpected"
    if entry_kind == "file":
        unexpected.write_text("unexpected\n")
    else:
        unexpected.symlink_to(base / "missing-target")
    with pytest.raises(SupervisorError, match="unexpected entries"):
        smoke._validate_lstat_tree(base, inventory)


def test_postrun_binding_fixture_verifies_after_completion(tmp_path: Path) -> None:
    base = tmp_path / "bounded"
    _write_postrun_binding_fixture(base)
    postrun, summary, proposal = smoke._validate_postrun_manifest_bindings(base)
    assert postrun["status"] == "PROFILE_COMPLETE"
    assert summary["status"] == "complete"
    assert proposal["decision"] == "NOT_APPROVED"
    assert postrun["success_lock"] is True
    assert postrun["streams_closed"] == {"ledger": True, "telemetry": True}
    assert all(
        row["relative_path"] != "POSTRUN_MANIFEST.json" for row in postrun["final_base_inventory"]
    )


def test_postrun_success_lock_is_last_and_read_only_validation_does_not_mutate(
    tmp_path: Path,
) -> None:
    base = tmp_path / "bounded"
    _write_postrun_binding_fixture(base)
    before = smoke._lstat_tree(base)
    smoke._validate_postrun_manifest_bindings(base)
    after = smoke._lstat_tree(base)
    assert after == before
    assert {row["relative_path"] for row in after} == {
        row["relative_path"]
        for row in json.loads((base / "POSTRUN_MANIFEST.json").read_text())["final_base_inventory"]
    } | {"POSTRUN_MANIFEST.json"}


def test_frozen_final_path_contract_rejects_rogue_file_before_manifest(tmp_path: Path) -> None:
    base = tmp_path / "bounded"
    (base / "expected").mkdir(parents=True)
    (base / "expected/result.json").write_text("{}\n")
    contract = [
        {"relative_path": "expected", "entry_type": "directory"},
        {"relative_path": "expected/result.json", "entry_type": "file"},
    ]
    smoke._validate_path_type_contract(base, contract)
    (base / "rogue.txt").write_text("rogue\n")
    with pytest.raises(SupervisorError, match="path/type contract differs"):
        smoke._validate_path_type_contract(base, contract)


def test_phase_path_contracts_match_preterminal_terminal_and_final_lock(tmp_path: Path) -> None:
    base = tmp_path / "bounded"
    (base / "run/ledger").mkdir(parents=True)
    (base / "run/ledger/trials.jsonl").write_text("ledger\n")
    terminal_contract = [
        {"relative_path": "run", "entry_type": "directory"},
        {"relative_path": "run/final", "entry_type": "directory"},
        {"relative_path": "run/final/manifest.json", "entry_type": "file"},
        {"relative_path": "run/ledger", "entry_type": "directory"},
        {"relative_path": "run/ledger/trials.jsonl", "entry_type": "file"},
    ]
    preterminal = smoke._path_type_contract_for_phase(terminal_contract, "preterminal")
    assert {row["relative_path"] for row in preterminal} == {
        "run",
        "run/ledger",
        "run/ledger/trials.jsonl",
    }
    smoke._validate_path_type_contract(base, preterminal)

    (base / "run/final").mkdir()
    (base / "run/final/manifest.json").write_text("{}\n")
    smoke._validate_path_type_contract(
        base, smoke._path_type_contract_for_phase(terminal_contract, "terminal")
    )

    (base / "POSTRUN_MANIFEST.json").write_text("{}\n")
    smoke._validate_path_type_contract(
        base, smoke._path_type_contract_for_phase(terminal_contract, "postrun")
    )


def test_final_contract_uses_production_cache_identity_hash_domain() -> None:
    identity = {
        "feature_artifact_sha256": "a" * 64,
        "recipe_provenance_sha256": "b" * 64,
        "strategy": "identity",
        "fold": 0,
        "train_ordered_sha256": "c" * 64,
        "tune_ordered_sha256": "d" * 64,
    }
    assert smoke.cache_identity_sha256(identity) != smoke.canonical_sha256(identity)
    assert smoke._run_cache_manifest_relative_path(identity) == (
        f"run/cache/features/identity-{smoke.cache_identity_sha256(identity)}.json"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("event_count", 3),
        ("event_head_sha256", "c" * 64),
        ("expected_run_complete_sequence", 4),
    ],
)
def test_terminal_event_chain_evidence_rejects_tampering(field: str, value: object) -> None:
    events = [
        {"sequence": 1, "event": "run_start", "event_sha256": "a" * 64},
        {"sequence": 2, "event": "trial_complete", "event_sha256": "b" * 64},
        {"sequence": 3, "event": "run_complete", "event_sha256": "d" * 64},
    ]
    evidence = {
        "event_count": 2,
        "event_head_sha256": "b" * 64,
        "expected_run_complete_sequence": 3,
    }
    smoke._validate_terminal_event_chain_evidence(events, evidence)
    evidence[field] = value
    with pytest.raises(SupervisorError, match="event-chain evidence differs"):
        smoke._validate_terminal_event_chain_evidence(events, evidence)


@pytest.mark.parametrize(
    "stage",
    [
        "preterminal_contract",
        "terminal_write",
        "run_complete_append",
        "terminal_tree",
        "ledger_close",
        "events_close",
        "postrun_write",
        "public_verify",
    ],
)
def test_claimed_execute_fault_seams_write_nonresumable_incident(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    base = tmp_path / "v5"
    monkeypatch.setattr(smoke, "EXACT_BASE", base)

    class SeamFailure(SupervisorError):
        pass

    def fail_after_claim(*_args: object, **_kwargs: object) -> Path:
        claim = base / "attempt/nonce-claim.json"
        claim.parent.mkdir(parents=True)
        claim.write_text("claimed\n")
        error = SeamFailure(stage)
        error.failure_stage = stage
        raise error

    monkeypatch.setattr(smoke, "_execute_v5_body", fail_after_claim)
    with pytest.raises(SeamFailure, match=stage):
        smoke.execute(
            base,
            expected_review_file_sha256="a" * 64,
            expected_authorization_file_sha256="b" * 64,
            expected_reviewer_identity="reviewer",
        )
    incident = smoke._validate_semantic_artifact(
        base / "EXECUTION_INCIDENT.json", smoke.EXECUTION_INCIDENT_SCHEMA
    )
    assert incident["failure_stage"] == stage
    assert incident["resumable"] is False
    assert incident["execution_allowed"] is False
    assert incident["ranking_allowed"] is False


def test_valid_postrun_success_lock_is_never_mutated_by_failure_handler(tmp_path: Path) -> None:
    base = tmp_path / "v5"
    (base / "attempt").mkdir(parents=True)
    (base / "attempt/nonce-claim.json").write_text("claimed\n")
    _write_postrun_binding_fixture(base)
    before = smoke._lstat_tree(base)
    assert smoke._record_v5_execution_incident(base, RuntimeError("late verifier")) is None
    assert smoke._lstat_tree(base) == before
    assert not (base / "EXECUTION_INCIDENT.json").exists()


def test_complete_state_domains_keep_terminal_trial_hashes_and_summary_families_separate() -> None:
    specifications = smoke.planned_trials()
    terminal, summary = smoke._complete_state_domains(specifications)
    assert terminal == {
        smoke.canonical_sha256(asdict(row.identity)): "complete" for row in specifications
    }
    assert summary == {"ridge": "complete", "xgboost": "complete", "mlp": "complete"}
    assert set(terminal).isdisjoint(summary)


def test_worker_metadata_prediction_hash_is_independent_from_checkpoint_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specification = smoke.planned_trials()[0]
    base = tmp_path
    result_path = base / "run/worker-results/ridge.json"
    result_path.parent.mkdir(parents=True)
    checkpoint = {"prediction_sha256": "a" * 64, "checkpoint": "canonical-domain"}
    result_path.write_text(
        json.dumps(
            {
                "status": smoke.COMPLETION_STATUS,
                "checkpoint_evidence": checkpoint,
                "metadata_sha256": "c" * 64,
            }
        )
    )
    profile = {
        "trial": asdict(specification.identity),
        "family": "ridge",
        "metadata": {
            "checkpoint_evidence": checkpoint,
            "prediction_sha256": "b" * 64,
            "stage_seconds": {"fit": 1.0},
        },
    }
    monkeypatch.setattr(smoke, "load_real_backend_profile", lambda *_args, **_kwargs: profile)
    observed = smoke._validate_completed_worker_evidence(base, specification, checkpoint)
    assert observed["metadata"]["prediction_sha256"] != checkpoint["prediction_sha256"]


@pytest.mark.parametrize(
    "tamper",
    ["extra-file", "missing-summary", "summary-bytes", "proposal-schema", "terminal-hash"],
)
def test_postrun_binding_rejects_missing_tampered_or_extra_files(
    tmp_path: Path, tamper: str
) -> None:
    base = tmp_path / "bounded"
    _write_postrun_binding_fixture(base)
    if tamper == "extra-file":
        (base / "unexpected.txt").write_text("unexpected\n")
    elif tamper == "missing-summary":
        (base / "profile-summary.json").unlink()
    elif tamper == "summary-bytes":
        (base / "profile-summary.json").write_text("tampered\n")
    elif tamper == "proposal-schema":
        proposal = base / "post-profile/proposal.json"
        proposal.unlink()
        smoke._write_immutable_json(proposal, {"schema_version": "wrong-schema"})
    else:
        (base / "run/final/manifest.json").write_text('{"tampered":true}\n')
    with pytest.raises(SupervisorError):
        smoke._validate_postrun_manifest_bindings(base)


def test_postrun_checkpoint_replay_uses_store_canonical_path_order(tmp_path: Path) -> None:
    specifications = smoke.planned_trials()
    binding = SimpleNamespace(
        plan_sha256=smoke.canonical_sha256([row.payload() for row in specifications]),
        config_sha256="1" * 64,
        input_sha256="2" * 64,
        profile_sha256="3" * 64,
    )
    run_start_sha256 = "4" * 64
    ledger_sha256 = "5" * 64
    store = smoke.CheckpointStore(
        tmp_path / "checkpoints",
        plan_sha256=binding.plan_sha256,
        config_sha256=binding.config_sha256,
        input_sha256=binding.input_sha256,
        cache_sha256=binding.profile_sha256,
        run_start_sha256=run_start_sha256,
    )
    entries: dict[str, dict[str, object]] = {}
    for index in (2, 0, 1):
        specification = specifications[index]
        completion = CompletionContract(
            expected_shape=specification.expected_shape,
            target_order=specification.target_order,
            prediction_order_sha256=specification.prediction_order_sha256,
            start_ledger_entry_sha256=f"{index + 10:064x}",
        )
        path = store.mark_trial_complete(
            specification.identity,
            record={
                "status": smoke.COMPLETION_STATUS,
                "candidate_id": specification.identity.candidate_id,
                "family": specification.identity.family,
                "seed": specification.identity.seed,
                "metadata_sha256": f"{index + 20:064x}",
            },
            prediction=np.full(specification.expected_shape, float(index)),
            completion=completion,
        )
        evidence = store.validate_completed_checkpoint(
            path, trial=specification.identity, completion=completion
        )
        key = smoke.canonical_sha256(asdict(specification.identity))
        entries[key] = {
            "status": smoke.COMPLETION_STATUS,
            "complete_ledger_entry_sha256": f"{index + 30:064x}",
            "start_ledger_entry_sha256": completion.start_ledger_entry_sha256,
            "checkpoint": evidence,
        }
    replayed = smoke._strict_postrun_checkpoint_replay(
        tmp_path / "checkpoints",
        binding,
        specifications,
        entries,
        run_start_sha256=run_start_sha256,
        ledger_sha256=ledger_sha256,
    )
    paths = [str(row["path"]) for row in replayed]
    assert paths == sorted(paths)
    first_key = smoke.canonical_sha256(asdict(specifications[0].identity))
    latest_path = tmp_path / "checkpoints" / first_key / "latest.json"
    latest = json.loads(latest_path.read_text())
    latest["file_sha256"] = "0" * 64
    latest_path.write_text(json.dumps(latest))
    with pytest.raises(Exception, match="latest checkpoint artifact differs"):
        smoke._strict_postrun_checkpoint_replay(
            tmp_path / "checkpoints",
            binding,
            specifications,
            entries,
            run_start_sha256=run_start_sha256,
            ledger_sha256=ledger_sha256,
        )


def test_dangling_prepare_failure_symlink_blocks_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "bounded"
    base.mkdir()
    (base / "PREPARE_FAILED.json").symlink_to(base / "missing-failure-target")
    monkeypatch.setattr(smoke, "EXACT_BASE", base)
    with pytest.raises(SupervisorError, match="authorization and execution are forbidden"):
        smoke._validate_prepare_state(base)


def test_claim_write_failure_records_durable_terminal_and_burns_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "bounded"
    monkeypatch.setattr(smoke, "EXACT_BASE", base)
    original = smoke._exclusive_artifact

    def fail_claim(path: Path, schema: str, payload: dict[str, object]) -> dict[str, object]:
        if schema == smoke.PREPARE_CLAIM_SCHEMA:
            raise OSError("injected-claim-write-failure")
        return original(path, schema, payload)

    monkeypatch.setattr(smoke, "_exclusive_artifact", fail_claim)
    with pytest.raises(OSError, match="claim-write"):
        smoke.prepare(base, runner_identity="pytest-runner")
    marker = smoke._validate_semantic_artifact(
        base / "PREPARE_CLAIM_FAILED.json", smoke.PREPARE_CLAIM_FAILURE_SCHEMA
    )
    assert marker["failure_code"] == "PREPARE_CLAIM_WRITE_FAILURE"
    assert marker["resumable"] is False
    assert marker["authorization_allowed"] is False
    assert marker["execution_allowed"] is False
    with pytest.raises(SupervisorError, match="fresh"):
        smoke.prepare(base, runner_identity="pytest-runner")


def test_failure_marker_write_failure_still_burns_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "bounded"
    monkeypatch.setattr(smoke, "EXACT_BASE", base)
    original = smoke._exclusive_artifact

    def fail_prepare(*args: object, **kwargs: object) -> Path:
        raise RuntimeError("injected-prepare-failure")

    def fail_marker(path: Path, schema: str, payload: dict[str, object]) -> dict[str, object]:
        if schema == smoke.PREPARE_FAILURE_SCHEMA:
            raise OSError("injected-marker-write-failure")
        return original(path, schema, payload)

    monkeypatch.setattr(smoke, "_prepare_claimed", fail_prepare)
    monkeypatch.setattr(smoke, "_exclusive_artifact", fail_marker)
    with pytest.raises(SupervisorError, match="permanently burned"):
        smoke.prepare(base, runner_identity="pytest-runner")
    assert base.is_dir()
    with pytest.raises(SupervisorError, match="fresh"):
        smoke.prepare(base, runner_identity="pytest-runner")


def test_prepare_state_rejects_repository_provenance_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "bounded"
    preflight = base / "preflight"
    preflight.mkdir(parents=True)
    claim_path = base / "prepare-claim.json"
    complete_path = base / "PREPARE_COMPLETE.json"
    profile_path = preflight / "bootstrap-profile.json"
    for path in (claim_path, complete_path, profile_path):
        path.write_text("fixture\n")
    monkeypatch.setattr(smoke, "EXACT_BASE", base)
    provenance = smoke._repository_preflight()
    tampered = {**provenance, "pyproject_sha256": "0" * 64}
    claim = {
        "prepare_attempt_id": "a" * 64,
        "base_path": str(base.resolve()),
        "run_id": smoke.RUN_ID,
        "purpose": smoke.PURPOSE,
        "prerequisite_provenance": provenance,
        "semantic_sha256": "b" * 64,
    }
    complete = {
        "prepare_attempt_id": "a" * 64,
        "prepare_claim_path": str(claim_path.resolve()),
        "prepare_claim_file_sha256": "f" * 64,
        "prepare_claim_semantic_sha256": "b" * 64,
        "profile_path": str(profile_path.resolve()),
        "profile_file_sha256": "f" * 64,
        "profile_semantic_sha256": "c" * 64,
        "prerequisite_provenance": provenance,
        "preflight_inventory": [],
    }
    profile = {
        "semantic_sha256": "c" * 64,
        "repository_provenance": tampered,
        "repository_provenance_sha256": smoke.canonical_sha256(tampered),
        "prepare_attempt_id": "a" * 64,
        "prepare_claim_file_sha256": "f" * 64,
        "prepare_claim_semantic_sha256": "b" * 64,
    }

    def artifact(path: str | Path, schema: str, **kwargs: object) -> dict[str, object]:
        return {
            smoke.PREPARE_CLAIM_SCHEMA: claim,
            smoke.PREPARE_COMPLETE_SCHEMA: complete,
            smoke.PROFILE_SCHEMA: profile,
        }[schema]

    monkeypatch.setattr(smoke, "_validate_semantic_artifact", artifact)
    monkeypatch.setattr(smoke, "_validate_lstat_tree", lambda *args, **kwargs: None)
    monkeypatch.setattr(smoke, "file_sha256", lambda *args, **kwargs: "f" * 64)
    with pytest.raises(SupervisorError, match="provenance chain differs"):
        smoke._validate_prepare_state(base)


def test_prepare_failure_is_durable_and_output_cannot_be_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "bounded"
    monkeypatch.setattr(smoke, "EXACT_BASE", base)
    prerequisites = smoke._repository_preflight()

    def fail_prepare(*args: object, **kwargs: object) -> Path:
        raise RuntimeError("injected-safe-prepare-failure")

    monkeypatch.setattr(smoke, "_prepare_claimed", fail_prepare)
    with pytest.raises(RuntimeError, match="injected-safe"):
        smoke.prepare(base, runner_identity="pytest-runner")
    failure_path = base / "PREPARE_FAILED.json"
    failure = smoke._validate_semantic_artifact(failure_path, smoke.PREPARE_FAILURE_SCHEMA)
    assert failure["status"] == "PREPARE_FAILED"
    assert failure["error_type"] == "RuntimeError"
    assert failure["error_message"] == "injected-safe-prepare-failure"
    assert failure["failure_code"] == "PREPARE_INTERNAL_FAILURE"
    assert failure["prerequisite_provenance"] == prerequisites
    assert failure["prepare_attempt_id"]
    assert failure["prepare_claim_file_sha256"] == smoke.file_sha256(base / "prepare-claim.json")
    assert failure["last_completed_checkpoint"] == "prepare-claim-written"
    assert failure["partial_tree_inventory_sha256"] == smoke.canonical_sha256(
        failure["partial_tree_inventory"]
    )
    assert failure["resumable"] is False
    assert failure["authorization_allowed"] is False
    assert failure["execution_allowed"] is False
    assert failure["scientific_result"] is False
    failure_hash = smoke.file_sha256(failure_path)
    with pytest.raises(SupervisorError, match="fresh"):
        smoke.prepare(base, runner_identity="pytest-runner")
    assert smoke.file_sha256(failure_path) == failure_hash


def test_dry_run_refuses_nonfresh_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "bounded"
    monkeypatch.setattr(smoke, "EXACT_BASE", output)
    monkeypatch.setattr(
        "sys.argv",
        ["profile-smoke", "--output", str(output), "--stage", "dry-run"],
    )
    assert smoke._main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["families"] == ["ridge", "xgboost", "mlp"]
    assert payload["ranking_allowed"] is False
    output.mkdir()
    with pytest.raises(SystemExit):
        smoke._main()


def test_verify_cli_is_read_only_and_uses_postrun_assertion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    base = tmp_path / "bounded"
    base.mkdir()
    monkeypatch.setattr(smoke, "EXACT_BASE", base)
    monkeypatch.setattr(
        smoke,
        "assert_profile_complete_postrun",
        lambda *args, **kwargs: {
            "evidence_reauthenticated": True,
            "execution_identity_matches_current_code": False,
        },
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "profile-smoke",
            "--output",
            str(base),
            "--stage",
            "verify",
            "--expected-review-sha256",
            "a" * 64,
            "--expected-authorization-sha256",
            "b" * 64,
            "--expected-reviewer-identity",
            "reviewer",
        ],
    )
    assert smoke._main() == 0
    assert json.loads(capsys.readouterr().out)["evidence_reauthenticated"] is True


def test_external_review_must_be_independent_and_binds_proposal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = {
        "schema_version": smoke.PROFILE_SCHEMA,
        "runner_identity": "runner-a",
        "shape": {"rows": 96},
        "candidate_sha256": ["a" * 64, "b" * 64, "c" * 64],
        "policies": {},
    }
    proposal = smoke._write_immutable_json(tmp_path / "proposal.json", profile)
    with pytest.raises(SupervisorError, match="independent"):
        smoke.freeze_bounded_bootstrap_review(
            tmp_path / "bad-review.json",
            profile_path=tmp_path / "proposal.json",
            reviewer_identity="runner-a",
            review_tool_identity="pytest-review-tool",
            review_tool_code_sha256="d" * 64,
            runner_identity="runner-a",
        )
    monkeypatch.setattr(
        smoke,
        "_validate_semantic_artifact",
        lambda *args, **kwargs: proposal,
    )
    monkeypatch.setattr(
        smoke,
        "_validate_prepare_state",
        lambda *args, **kwargs: ({}, {"semantic_sha256": "e" * 64}),
    )
    monkeypatch.setattr(smoke, "file_sha256", lambda *args, **kwargs: "f" * 64)
    review = smoke.freeze_bounded_bootstrap_review(
        tmp_path / "review.json",
        profile_path=tmp_path / "proposal.json",
        reviewer_identity="reviewer-b",
        review_tool_identity="pytest-review-tool",
        review_tool_code_sha256="d" * 64,
        runner_identity="runner-a",
    )
    assert review["profile_semantic_sha256"] == proposal["semantic_sha256"]


def test_lifecycle_schema_rejects_extra_keys(tmp_path: Path) -> None:
    path = tmp_path / "claim.json"
    smoke._exclusive_artifact(
        path,
        "qm9-v31-bounded-attempt-claim-v5",
        {
            "nonce": "a" * 64,
            "authorization_file_sha256": "b" * 64,
            "authorization_semantic_sha256": "c" * 64,
            "runner_identity": "runner",
            "run_id": smoke.RUN_ID,
            "purpose": smoke.PURPOSE,
            "claimed_at_utc": "2026-07-19T12:00:00Z",
            "unexpected": True,
            **smoke._false_boundaries(),
        },
    )
    with pytest.raises(SupervisorError, match="exact key schema"):
        smoke._validate_semantic_artifact(path, "qm9-v31-bounded-attempt-claim-v5")


def test_stored_resource_inequalities_are_recomputed() -> None:
    artifact = {
        "static_required_with_margin_bytes": smoke.RSS_LIMIT,
        "available_ram_bytes": smoke.RSS_LIMIT,
        "current_rss_bytes": 1,
        "free_disk_bytes": smoke.DISK_FLOOR + 256 * 1024**2,
        "inequalities": {
            "static_within_available_ram": True,
            "static_within_rss_fraction": True,
            "rss_plus_static_within_limit": True,
            "disk_within_floor_and_allowance": True,
        },
        "approved": True,
    }
    with pytest.raises(SupervisorError, match="inequalities"):
        smoke._validate_stored_resource_observation(artifact)


def test_initial_observation_rejects_runner_value_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(smoke, "EXACT_BASE", tmp_path)
    authorization = {
        "nonce": "a" * 64,
        "runner_identity": "expected-runner",
    }
    observation = smoke._bound_resource_observation(
        tmp_path,
        1,
        authorization,
    )
    observation["runner_identity"] = "tampered-runner"
    monkeypatch.setattr(
        smoke,
        "_validate_semantic_artifact",
        lambda path, schema, **kwargs: (
            {"memory_worksheet": {"capacity": {"cpu_required_with_margin_bytes": 1}}}
            if schema == smoke.PROFILE_SCHEMA
            else observation
        ),
    )
    binding = SimpleNamespace(
        device_identity=smoke.detect_device_identity("cpu"),
        device_identity_sha256=smoke.canonical_sha256(smoke.detect_device_identity("cpu")),
        bootstrap_profile_path=str(tmp_path / "profile.json"),
        bootstrap_profile_file_sha256="a" * 64,
    )
    claim = {"claimed_at_utc": observation["observed_at"]}
    with pytest.raises(SupervisorError, match="observation values"):
        smoke._validate_initial_observation_values(
            binding,
            authorization,
            claim,
        )


@pytest.mark.parametrize("run_start_tamper", ["physical-hash", "extra-schema-key"])
def test_receipt_rejects_run_start_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_start_tamper: str
) -> None:
    monkeypatch.setattr(smoke, "EXACT_BASE", tmp_path)
    (tmp_path / "attempt").mkdir()
    (tmp_path / "run").mkdir()
    (tmp_path / "attempt/nonce-claim.json").write_text("claim")
    (tmp_path / "attempt/resource-observation.json").write_text("observation")
    run_start_payload = {
        "schema_version": smoke.RUN_START_SCHEMA,
        "binding": {},
        "planned_trials": [],
        "planned_trial_keys": [],
        "bootstrap_authorization": {},
        "bootstrap_lifecycle": {},
        "run_purpose": smoke.PURPOSE,
        "scientific_result": False,
        "publication_ready": False,
        "ranking_allowed": False,
        "promotion_allowed": False,
        "ensemble_allowed": False,
        "outer_validation_targets_read": False,
        "test_targets_read": False,
    }
    if run_start_tamper == "extra-schema-key":
        run_start_payload["unexpected"] = True
    run_start = smoke._write_immutable_json(tmp_path / "run/run_start.json", run_start_payload)
    actual_hash = smoke.file_sha256(tmp_path / "run/run_start.json")
    now = smoke._iso(smoke._now())
    authorization = {"nonce": "a" * 64, "runner_identity": "runner"}
    receipt = {
        "authorization_semantic_sha256": "b" * 64,
        "authorization_file_sha256": "c" * 64,
        "nonce": authorization["nonce"],
        "run_start_sha256": actual_hash,
        "run_start_file_sha256": actual_hash,
        "run_start_semantic_sha256": run_start["semantic_sha256"],
        "claim_file_sha256": smoke.file_sha256(tmp_path / "attempt/nonce-claim.json"),
        "observation_file_sha256": smoke.file_sha256(
            tmp_path / "attempt/resource-observation.json"
        ),
        "runner_identity": "runner",
        "run_id": smoke.RUN_ID,
        "purpose": smoke.PURPOSE,
        "consumed_at_utc": now,
        **smoke._false_boundaries(),
    }
    monkeypatch.setattr(
        smoke,
        "_validate_semantic_artifact",
        lambda *args, **kwargs: receipt,
    )
    binding = SimpleNamespace(
        bootstrap_authorization_semantic_sha256="b" * 64,
        bootstrap_authorization_file_sha256="c" * 64,
    )
    expected = "f" * 64 if run_start_tamper == "physical-hash" else actual_hash
    with pytest.raises(SupervisorError, match="receipt values"):
        smoke._validate_receipt_values(
            binding,
            authorization,
            {"claimed_at_utc": now},
            {"observed_at": now},
            expected_run_start_sha256=expected,
        )


def test_initial_observation_rejects_reviewed_memory_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(smoke, "EXACT_BASE", tmp_path)
    authorization = {"nonce": "a" * 64, "runner_identity": "runner"}
    observation = smoke._bound_resource_observation(tmp_path, 2, authorization)
    profile = {"memory_worksheet": {"capacity": {"cpu_required_with_margin_bytes": 1}}}
    monkeypatch.setattr(
        smoke,
        "_validate_semantic_artifact",
        lambda path, schema, **kwargs: profile if schema == smoke.PROFILE_SCHEMA else observation,
    )
    device = smoke.detect_device_identity("cpu")
    binding = SimpleNamespace(
        device_identity=device,
        device_identity_sha256=smoke.canonical_sha256(device),
        bootstrap_profile_path=str(tmp_path / "profile.json"),
        bootstrap_profile_file_sha256="b" * 64,
    )
    with pytest.raises(SupervisorError, match="observation values"):
        smoke._validate_initial_observation_values(
            binding,
            authorization,
            {"claimed_at_utc": observation["observed_at"]},
        )


def test_trial_observation_rejects_reviewed_memory_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(smoke, "EXACT_BASE", tmp_path)
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    trial = smoke.planned_trials()[0].identity
    path = attempt / f"resource-observation-{trial.trial_index}-{trial.family}.json"
    path.write_text("trial-observation")
    (attempt / "authorization-consumed.json").write_text("receipt")
    authorization = {"nonce": "a" * 64, "runner_identity": "runner"}
    artifact = {
        **smoke._bound_resource_observation(tmp_path, 2, authorization),
        "trial": asdict(trial),
        "trial_key": smoke.canonical_sha256(asdict(trial)),
        "policy_sha256": smoke.canonical_sha256(asdict(smoke.sidecar_policy(trial.family))),
        "run_start_sha256": "d" * 64,
        "receipt_file_sha256": smoke.file_sha256(attempt / "authorization-consumed.json"),
        "previous_ledger_entry_sha256": None,
        "semantic_sha256": "e" * 64,
    }
    profile = {"memory_worksheet": {"capacity": {"cpu_required_with_margin_bytes": 1}}}
    monkeypatch.setattr(
        smoke,
        "_validate_semantic_artifact",
        lambda source, schema, **kwargs: (
            profile
            if schema == smoke.PROFILE_SCHEMA
            else authorization
            if schema == smoke.AUTH_SCHEMA
            else artifact
        ),
    )
    device = smoke.detect_device_identity("cpu")
    binding = SimpleNamespace(
        device_identity=device,
        device_identity_sha256=smoke.canonical_sha256(device),
        bootstrap_profile_path=str(tmp_path / "profile.json"),
        bootstrap_profile_file_sha256="b" * 64,
        bootstrap_authorization_path=str(tmp_path / "authorization.json"),
        bootstrap_authorization_file_sha256="c" * 64,
    )
    with pytest.raises(SupervisorError, match="observation differs"):
        smoke.validate_bounded_trial_observation(
            binding,
            path=path,
            trial=trial,
            run_start_sha256="d" * 64,
            previous_ledger_entry_sha256=None,
        )


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("QM9_V31_RUN_BOUNDED_INTEGRATION") != "1",
    reason="requires explicit QM9_V31_RUN_BOUNDED_INTEGRATION=1 approval",
)
def test_prepare_execute_real_subprocess_all_families_is_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Seam test for the reviewed path; never invoke as the standalone fixed smoke."""

    base = tmp_path / "bounded"
    monkeypatch.setattr(smoke, "EXACT_BASE", base)
    smoke.prepare(base, runner_identity="pytest-runner")
    preflight = base / "preflight"
    assert not (base / "run").exists()
    smoke.freeze_bounded_bootstrap_review(
        preflight / "independent-review.json",
        profile_path=preflight / "bootstrap-profile.json",
        reviewer_identity="pytest-independent-reviewer",
        review_tool_identity="pytest-review-tool",
        review_tool_code_sha256="d" * 64,
        runner_identity="pytest-runner",
    )
    smoke.freeze_bounded_bootstrap_authorization(
        preflight / "bootstrap-authorization.json",
        profile_path=preflight / "bootstrap-profile.json",
        review_path=preflight / "independent-review.json",
    )
    review_sha256 = smoke.file_sha256(preflight / "independent-review.json")
    authorization_sha256 = smoke.file_sha256(preflight / "bootstrap-authorization.json")
    summary_path = smoke.execute(
        base,
        expected_review_file_sha256=review_sha256,
        expected_authorization_file_sha256=authorization_sha256,
        expected_reviewer_identity="pytest-independent-reviewer",
    )
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "complete"
    assert summary["trial_states"] == {
        "ridge": "complete",
        "xgboost": "complete",
        "mlp": "complete",
    }
    for field in (
        "ranking_allowed",
        "promotion_allowed",
        "ensemble_allowed",
        "scientific_result",
        "publication_ready",
    ):
        assert summary[field] is False
    post = json.loads((base / "post-profile/proposal.json").read_text())
    assert post["measurement_status"] == "INSUFFICIENT_OBSERVATIONS"
    assert post["decision"] == "NOT_APPROVED"
    with pytest.raises(SupervisorError):
        smoke.execute(
            base,
            expected_review_file_sha256=review_sha256,
            expected_authorization_file_sha256=authorization_sha256,
            expected_reviewer_identity="pytest-independent-reviewer",
        )
