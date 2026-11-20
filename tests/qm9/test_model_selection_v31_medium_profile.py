from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from mist_transfer_benchmark.qm9 import model_selection_v31_medium_profile as medium
from mist_transfer_benchmark.qm9.model_selection_v31_medium_worker import (
    PROFILE_SCHEMA,
    MediumWorkerError,
    validate_profile,
)
from mist_transfer_benchmark.qm9.model_selection_v31_observability import validate_event_stream
from mist_transfer_benchmark.qm9.model_selection_v31_supervisor import SupervisorError, file_sha256

REAL_TARGET_FREE_CONTRACT = medium._target_free_contract


@pytest.fixture(autouse=True)
def _target_free_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    selected = np.arange(8192, dtype=np.int64)
    groups = np.asarray([f"fixture-group-{index}" for index in selected])
    train, tune = medium.medium_split(groups)
    sample = medium.sample_evidence(selected, groups)
    plan = medium.planned_trials(len(tune))
    unsigned = {
        "schema_version": "qm9-v31-medium-target-free-plan-v1",
        "run_id": medium.RUN_ID,
        "selected_source_indices": selected.tolist(),
        "train_positions": train.tolist(),
        "tune_positions": tune.tolist(),
        "sample": sample,
        "selected_feature_sha256": "1" * 64,
        "train_feature_sha256": "2" * 64,
        "tune_feature_sha256": "3" * 64,
        "plan": [row.payload() for row in plan],
        "plan_sha256": medium.canonical_sha256([row.payload() for row in plan]),
        "input_inventory_sha256": "4" * 64,
        "label_reads": medium._label_reads(),
        **medium._false_boundaries(),
    }
    artifact = {**unsigned, "semantic_sha256": medium.canonical_sha256(unsigned)}
    monkeypatch.setattr(
        medium,
        "_target_free_contract",
        lambda **_kwargs: (artifact, sparse.csr_matrix((8192, 1)), None),
    )


def _inputs(tmp_path: Path) -> dict[str, Path]:
    source = tmp_path / "qm9.csv"
    matrix = tmp_path / "feature_matrix.npz"
    manifest = tmp_path / "manifest.json"
    phase1 = tmp_path / "phase1"
    source.write_text("metadata-only-fixture\n")
    matrix.write_bytes(b"feature-fixture")
    manifest.write_text(json.dumps({"schema_version": "qm9-paper-feature-artifact-v1"}))
    phase1.mkdir(exist_ok=True)
    return {
        "qm9_csv": source,
        "feature_matrix": matrix,
        "feature_manifest": manifest,
        "phase1_dir": phase1,
    }


def test_real_target_free_entry_replays_with_zero_label_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the production target-free function; never replace that function itself."""

    inputs = _inputs(tmp_path)
    rows = 9000
    smiles = np.asarray([f"C-{index}" for index in range(rows)])
    groups = np.asarray([f"group-{index}" for index in range(rows)])
    identities = SimpleNamespace(
        row_identity_sha256="1" * 64,
        raw_smiles_sha256="2" * 64,
        source_smiles=smiles,
    )
    evidence = SimpleNamespace(split=SimpleNamespace(train=np.arange(rows, dtype=np.int64)))
    matrix = sparse.csr_matrix((rows, 2065), dtype=np.float64)
    manifest = {
        "source_csv_sha256": file_sha256(inputs["qm9_csv"]),
        "source_smiles_sha256": identities.raw_smiles_sha256,
    }
    label_calls = 0

    def forbidden_label_loader(*_args: object, **_kwargs: object) -> None:
        nonlocal label_calls
        label_calls += 1
        raise AssertionError("target-free entry attempted a label read")

    monkeypatch.setattr(medium, "load_qm9_identities", lambda _path: identities)
    monkeypatch.setattr(medium, "verify_phase1_evidence", lambda *_args: evidence)
    monkeypatch.setattr(
        medium,
        "authenticate_feature_artifact",
        lambda *_args, **_kwargs: (matrix, manifest, {"targets_accessed": False}),
    )
    monkeypatch.setattr(medium, "canonical_connectivity_groups", lambda _values: groups)
    monkeypatch.setattr(medium, "load_targets_for_indices", forbidden_label_loader)

    first, first_matrix, _ = REAL_TARGET_FREE_CONTRACT(**inputs)
    second, second_matrix, _ = REAL_TARGET_FREE_CONTRACT(**inputs)
    assert first == second
    assert medium._sparse_sha256(first_matrix) == medium._sparse_sha256(second_matrix)
    assert first["label_reads"] == medium._label_reads()
    assert label_calls == 0


def _external_decision(path: Path, payload: dict[str, object]) -> Path:
    now = datetime.now(UTC)
    unsigned = {
        **payload,
        "issued_at_utc": (now - timedelta(seconds=1)).isoformat(),
        "expires_at_utc": (now + timedelta(minutes=30)).isoformat(),
    }
    artifact = {**unsigned, "semantic_sha256": medium.canonical_sha256(unsigned)}
    path.write_text(json.dumps(artifact, sort_keys=True))
    return path


def _authorized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    base = tmp_path / "medium"
    monkeypatch.setattr(medium, "EXACT_BASE", base)
    prepared = medium.prepare(base, runner_identity="runner", **_inputs(tmp_path))
    prepared_artifact = json.loads(prepared.read_text())
    bindings = medium._decision_bindings(prepared_artifact, nonce="1" * 64)
    approval = _external_decision(
        tmp_path / "external-review.json",
        {
            "schema_version": medium.EXTERNAL_REVIEW_SCHEMA,
            "run_id": medium.RUN_ID,
            "decision": "APPROVE_MEDIUM_PROFILE_ONLY",
            "reviewer_identity": "reviewer",
            "prepare_file_sha256": file_sha256(prepared),
            "prepare_semantic_sha256": prepared_artifact["semantic_sha256"],
            **bindings,
        },
    )
    reviewed = medium.review(
        base,
        approval_path=approval,
        expected_approval_sha256=file_sha256(approval),
    )
    reviewed_artifact = json.loads(reviewed.read_text())
    external_authorization = _external_decision(
        tmp_path / "external-authorization.json",
        {
            "schema_version": medium.EXTERNAL_AUTH_SCHEMA,
            "run_id": medium.RUN_ID,
            "decision": "AUTHORIZE_MEDIUM_PROFILE_ONLY",
            "authorizer_identity": "authorizer",
            "reviewer_identity": "reviewer",
            "review_file_sha256": file_sha256(reviewed),
            "review_semantic_sha256": reviewed_artifact["semantic_sha256"],
            **bindings,
        },
    )
    return medium.authorize(
        base,
        authorization_path=external_authorization,
        expected_authorization_sha256=file_sha256(external_authorization),
    )


def _complete_fixture_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    authorization = _authorized(tmp_path, monkeypatch)
    medium.execute(
        medium.EXACT_BASE,
        expected_authorization_sha256=file_sha256(authorization),
        workload=lambda *_args: {
            "peak_rss_bytes": 1024**3,
            "label_reads": medium._label_reads(source=1, derived=3),
            "trial_states": {"ridge": "complete", "xgboost": "complete", "mlp": "complete"},
        },
    )
    return authorization


def _outer_evidence(
    tmp_path: Path,
    authorization: Path,
    *,
    peak: int = 1024**3,
    state: str = "complete",
) -> tuple[Path, Path]:
    prepared = json.loads((medium.EXACT_BASE / "PREPARED.json").read_text())
    contract = prepared["execution_contract"]
    wrapper = contract["outer_wrapper"]
    status_path = Path(wrapper["parameters"]["status_path"])
    result_path = Path(wrapper["parameters"]["result_path"])
    status_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    status_row = {
        "timestamp": now.isoformat(),
        "elapsed_seconds": 1.0,
        "pid": 123,
        "pgid": 123,
        "live_descendants": [],
        "live_descendant_count": 0,
        "group_members": [],
        "aggregate_rss_bytes": 0,
        "peak_rss_bytes": peak,
        "state": state,
        "reason": "CHILD_EXIT" if state == "complete" else "WALL_LIMIT",
    }
    status_path.write_text(json.dumps(status_row) + "\n")
    authorization_sha = file_sha256(authorization)
    actual_wrapper = medium.materialize_outer_wrapper_argv(
        wrapper,
        authorization_sha256=authorization_sha,
        code_state_sha256=prepared["repository_provenance"]["harness_code_state_sha256"],
    )
    bound_files: dict[str, dict[str, str]] = {}
    iterator = iter(actual_wrapper)
    for value in iterator:
        if value == "--expected-bound-file":
            binding = next(iterator)
            path, digest = binding.rsplit("=", 1)
            bound_files[str(Path(path).resolve())] = {
                "expected_sha256": digest,
                "observed_sha256": digest,
            }
    expected_bound = {
        path: row["expected_sha256"] for path, row in bound_files.items()
    }
    result = {
        "schema_version": "bounded-process-result-v2",
        "provenance": {
            "command": wrapper["contained_argv"],
            "command_sha256": wrapper["contained_argv_sha256"],
            "wrapper_argv": actual_wrapper,
            "wrapper_argv_sha256": medium.canonical_sha256(actual_wrapper),
            "harness_path": contract["harness_path"],
            "harness_code_sha256": contract["harness_sha256"],
            "execution_cwd": contract["cwd"],
            "repository": {
                "repository_root": prepared["repository_provenance"]["repository_root"],
                "head": prepared["repository_provenance"]["head"],
                "file_manifest_sha256": prepared["repository_provenance"]["manifest_sha256"],
                "file_count": len(prepared["repository_provenance"]["manifest"]),
                "code_state_sha256": prepared["repository_provenance"][
                    "harness_code_state_sha256"
                ],
            },
            "bound_files": bound_files,
            "launch_parameters": {
                "profile": "medium-v1",
                "wall_seconds": 2400.0,
                "memory_max_bytes": 4 * 1024**3,
                "poll_seconds": 0.5,
                "term_grace_seconds": 120.0,
                "command_seconds": 2280.0,
                "status_path": str(status_path.resolve()),
                "result_path": str(result_path.resolve()),
            },
        },
        "expected_provenance": {
            "command_sha256": wrapper["contained_argv_sha256"],
            "harness_code_sha256": contract["harness_sha256"],
            "code_state_sha256": prepared["repository_provenance"][
                "harness_code_state_sha256"
            ],
            "cwd": contract["cwd"],
            "bound_files": expected_bound,
        },
        "provenance_mismatches": {
            key: False for key in ("command", "harness", "code_state", "cwd", "bound_files")
        },
        "limits": {
            "absolute_wall_seconds": 2400.0,
            "memory_max_bytes": 4 * 1024**3,
            "poll_seconds": 0.5,
            "term_grace_seconds": 120.0,
            "command_runtime_max_seconds": 2280.0,
            "result_headroom_seconds": 120.0,
        },
        "cgroup": {
            "unit": "synthetic.scope",
            "properties": {"MemoryMax": 4 * 1024**3, "OOMPolicy": "kill", "RuntimeMaxSec": 2280},
            "capability": {
                "cgroup_filesystem": "cgroup2",
                "memory_controller": True,
                "systemd_version": "systemd synthetic",
            },
            "memory_peak_bytes": peak,
            "observed_memory_max_bytes": 4 * 1024**3,
            "control_group": "/synthetic",
            "memory_events": {
                "low": 0, "high": 0, "max": 0, "oom": 0,
                "oom_kill": 0, "oom_group_kill": 0,
            },
            "limit_verified": True,
            "result": "success",
            "empty": True,
            "oom_killed": False,
        },
        "started_at_utc": (now - timedelta(seconds=1)).isoformat(),
        "finished_at_utc": (now + timedelta(seconds=1)).isoformat(),
        "elapsed_seconds": 1.0,
        "pid": 123,
        "pgid": 123,
        "session_id": 123,
        "leader_starttime": 1,
        "peak_observed_rss_bytes": peak,
        "child_exit_code": 0 if state == "complete" else 1,
        "reason": "CHILD_EXIT" if state == "complete" else "WALL_LIMIT",
        "state": state,
        "term_action_at_utc": None,
        "term_survivors": [],
        "kill_action_at_utc": None,
        "kill_survivors": [],
        "group_empty": True,
        "cgroup_empty": True,
        "action_evidence": [],
        "signal_received": None,
        "status_file_sha256": file_sha256(status_path),
        "status_append_error": None,
        "harness_private_data_accessed": False,
    }
    result_path.write_text(json.dumps(result))
    return status_path, result_path


def test_frozen_medium_contract_is_exact() -> None:
    config = medium.frozen_config()
    assert medium.RUN_ID == "qm9-v31-medium-profile-cpu-v1"
    assert config["sample"] == {
        "rows": 8192,
        "method": "stable-hash-whole-identity-groups-until-at-least-threshold",
        "inner_split": "identity-group-90-10",
        "seed": 20260713,
    }
    assert config["limits"] == {
        "rss_hard_bytes": 4 * 1024**3,
        "rss_success_bytes": int(3.2 * 1024**3),
        "outer_seconds": 2400,
        "outer_command_seconds": 2280,
        "workload_drain_seconds": 2280,
        "finalize_seconds": 120,
        "artifact_max_bytes": 1024**3,
        "disk_free_min_bytes": 5 * 1024**3,
        "family_soft_seconds": {"ridge": 180, "xgboost": 1200, "mlp": 720},
    }
    ridge, xgboost, mlp = medium.candidates()
    assert ridge == {
        "id": "ridge|descriptor_scaled_only|alpha-1",
        "family": "ridge",
        "recipe": "descriptor_scaled_only",
        "params": {"alpha": 1.0},
    }
    assert xgboost["params"]["n_estimators"] == 200
    assert xgboost["early_stopping_rounds"] == 35
    assert mlp["params"]["hidden_dims"] == [384, 192]
    assert mlp["training"] == {
        "batch_size": 512,
        "max_epochs": 20,
        "patience": 10,
        "min_delta": 0.0001,
    }
    assert all(config[key] is False for key in medium._false_boundaries())


def test_hash_sample_and_inner_split_are_deterministic_and_group_safe() -> None:
    groups = np.asarray([f"group-{index}" for index in range(9000)])
    first = medium.stable_group_sample(groups)
    second = medium.stable_group_sample(groups)
    assert np.array_equal(first, second)
    assert len(first) == len(set(groups[first])) == 8192
    train, tune = medium.medium_split(groups[first])
    assert len(train) + len(tune) == 8192
    assert not set(groups[first][train]) & set(groups[first][tune])
    assert abs(len(tune) / 8192 - 0.10) < 0.01
    evidence = medium.sample_evidence(first, groups[first])
    assert evidence["threshold_rows"] == 8192
    assert evidence["actual_rows"] == len(first)
    assert evidence["actual_groups"] == len(set(groups[first]))


def test_sample_refuses_too_few_identity_groups() -> None:
    with pytest.raises(SupervisorError, match="too few rows"):
        medium.stable_group_sample(np.repeat(np.arange(100), 10))


def test_sample_keeps_boundary_group_whole() -> None:
    groups = np.repeat(np.asarray([f"group-{index}" for index in range(3000)]), 3)
    selected = medium.stable_group_sample(groups)
    assert len(selected) >= 8192
    selected_set = set(selected)
    for group in set(groups[selected]):
        assert set(np.flatnonzero(groups == group)) <= selected_set


def test_exact_base_and_independent_stage_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exact = tmp_path / "exact"
    monkeypatch.setattr(medium, "EXACT_BASE", exact)
    with pytest.raises(SupervisorError, match="requires exact"):
        medium.prepare(tmp_path / "other", runner_identity="runner", **_inputs(tmp_path))
    prepared = medium.prepare(exact, runner_identity="runner", **_inputs(tmp_path))
    assert prepared.stat().st_mode & 0o777 == 0o444
    assert not (exact / "REVIEWED.json").exists()
    prepared_artifact = json.loads(prepared.read_text())
    bindings = medium._decision_bindings(prepared_artifact, nonce="2" * 64)
    self_review = _external_decision(
        tmp_path / "self-review.json",
        {
            "schema_version": medium.EXTERNAL_REVIEW_SCHEMA,
            "run_id": medium.RUN_ID,
            "decision": "APPROVE_MEDIUM_PROFILE_ONLY",
            "reviewer_identity": "runner",
            "prepare_file_sha256": file_sha256(prepared),
            "prepare_semantic_sha256": prepared_artifact["semantic_sha256"],
            **bindings,
        },
    )
    with pytest.raises(SupervisorError, match="independent"):
        medium.review(
            exact,
            approval_path=self_review,
            expected_approval_sha256=file_sha256(self_review),
        )
    with pytest.raises(SupervisorError, match="physical hash"):
        medium.review(
            exact,
            approval_path=self_review,
            expected_approval_sha256="0" * 64,
        )


def test_prepare_failure_after_claim_is_durable_and_burns_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "medium"
    monkeypatch.setattr(medium, "EXACT_BASE", base)
    inputs = _inputs(tmp_path)
    inputs["feature_matrix"].unlink()
    with pytest.raises(SupervisorError, match="missing"):
        medium.prepare(base, runner_identity="runner", **inputs)
    claim = base / "PREPARE_CLAIM.json"
    failure = base / "PREPARE_FAILED.json"
    assert claim.is_file() and failure.is_file()
    assert claim.stat().st_mode & 0o777 == 0o444
    assert failure.stat().st_mode & 0o777 == 0o444
    assert json.loads(failure.read_text())["status"] == "PREPARE_FAILED"
    assert not (base / "PREPARED.json").exists()
    with pytest.raises(FileExistsError):
        medium.prepare(base, runner_identity="runner", **inputs)


def test_authorization_is_hard_stop_and_has_no_label_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization = _authorized(tmp_path, monkeypatch)
    artifact = json.loads(authorization.read_text())
    assert artifact["one_use"] is True
    assert authorization.stat().st_mode & 0o777 == 0o444
    assert not (medium.EXACT_BASE / "AUTHORIZATION_CONSUMED.json").exists()
    assert not (medium.EXACT_BASE / "TERMINAL.json").exists()
    prepared = json.loads((medium.EXACT_BASE / "PREPARED.json").read_text())
    assert prepared["label_reads"] == medium._label_reads()


def test_injected_fixture_execute_and_verify_preserve_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization = _authorized(tmp_path, monkeypatch)

    def workload(_base: Path, prepared: dict[str, object]) -> dict[str, object]:
        assert prepared["label_reads"] == medium._label_reads()
        return {
            "peak_rss_bytes": 1024**3,
            "label_reads": medium._label_reads(source=1, derived=3),
            "trial_states": {"ridge": "complete", "xgboost": "complete", "mlp": "complete"},
        }

    terminal_path = medium.execute(
        medium.EXACT_BASE,
        expected_authorization_sha256=file_sha256(authorization),
        workload=workload,
    )
    terminal = medium.verify(medium.EXACT_BASE)
    assert terminal_path.stat().st_mode & 0o777 == 0o444
    assert terminal["status"] == "PROFILE_COMPLETE"
    assert all(terminal[key] is False for key in medium._false_boundaries())
    (medium.EXACT_BASE / "rogue.txt").write_text("rogue")
    with pytest.raises(SupervisorError, match="exact tree"):
        medium.verify(medium.EXACT_BASE)


@pytest.mark.parametrize(
    "bad_result",
    [
        {
            "peak_rss_bytes": int(3.2 * 1024**3) + 1,
            "label_reads": medium._label_reads(source=1, derived=3),
        },
        {
            "peak_rss_bytes": 1,
            "label_reads": {
                **medium._label_reads(source=1, derived=3),
                "outer_validation": 1,
            },
        },
    ],
)
def test_execute_rejects_resource_or_label_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bad_result: dict[str, object],
) -> None:
    authorization = _authorized(tmp_path, monkeypatch)
    with pytest.raises(SupervisorError):
        medium.execute(
            medium.EXACT_BASE,
            expected_authorization_sha256=file_sha256(authorization),
            workload=lambda *_args: bad_result,
        )
    terminal = json.loads((medium.EXACT_BASE / "TERMINAL.json").read_text())
    assert terminal["status"] == "PROFILE_FAILED"


def test_postrun_accepts_only_bound_outer_cgroup_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization = _complete_fixture_run(tmp_path, monkeypatch)
    status_path, result_path = _outer_evidence(tmp_path, authorization)
    postrun_path = medium.finalize_postrun(
        medium.EXACT_BASE,
        launch_status_path=status_path,
        launch_result_path=result_path,
        expected_authorization_sha256=file_sha256(authorization),
    )
    assert json.loads(postrun_path.read_text())["status"] == "POSTRUN_ACCEPTED"
    assert medium.verify_postrun(medium.EXACT_BASE)["automatic_disposition_only"] is True
    status_path.write_text(status_path.read_text() + "{}\n")
    with pytest.raises(SupervisorError, match="physical binding"):
        medium.verify_postrun(medium.EXACT_BASE)


def test_postrun_rejects_outer_aggregate_peak_above_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization = _complete_fixture_run(tmp_path, monkeypatch)
    status_path, result_path = _outer_evidence(
        tmp_path, authorization, peak=medium.RSS_SUCCESS_BYTES + 1
    )
    postrun_path = medium.finalize_postrun(
        medium.EXACT_BASE,
        launch_status_path=status_path,
        launch_result_path=result_path,
        expected_authorization_sha256=file_sha256(authorization),
    )
    postrun = json.loads(postrun_path.read_text())
    assert postrun["status"] == "POSTRUN_REJECTED"
    assert "AGGREGATE_PEAK_ACCEPTED" in postrun["reasons"]
    with pytest.raises(SupervisorError, match="not accepted"):
        medium.verify_postrun(medium.EXACT_BASE)


def test_postrun_replay_rejects_external_authorization_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization = _complete_fixture_run(tmp_path, monkeypatch)
    status_path, result_path = _outer_evidence(tmp_path, authorization)
    medium.finalize_postrun(
        medium.EXACT_BASE,
        launch_status_path=status_path,
        launch_result_path=result_path,
        expected_authorization_sha256=file_sha256(authorization),
    )
    internal = json.loads(authorization.read_text())
    external_path = Path(internal["external_authorization_path"])
    external_path.write_text(external_path.read_text() + "\n")
    with pytest.raises(SupervisorError, match="decision chain binding|authorization replay"):
        medium.verify_postrun(medium.EXACT_BASE)


def test_postrun_rejects_outer_nested_limit_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization = _complete_fixture_run(tmp_path, monkeypatch)
    status_path, result_path = _outer_evidence(tmp_path, authorization)
    result = json.loads(result_path.read_text())
    result["limits"]["result_headroom_seconds"] = 119.0
    result_path.write_text(json.dumps(result))
    postrun = medium.finalize_postrun(
        medium.EXACT_BASE,
        launch_status_path=status_path,
        launch_result_path=result_path,
        expected_authorization_sha256=file_sha256(authorization),
    )
    assert json.loads(postrun.read_text())["status"] == "POSTRUN_REJECTED"


def test_execute_cli_returns_nonzero_for_postrun_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authorization = _authorized(tmp_path, monkeypatch)
    script_path = (
        Path(__file__).parents[2] / "scripts/run_qm9_model_selection_v31_medium_profile.py"
    )
    spec = importlib.util.spec_from_file_location("medium_profile_cli_test", script_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    rejected = tmp_path / "POSTRUN_REJECTED.json"
    rejected.write_text(json.dumps({"status": "POSTRUN_REJECTED"}))
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(cli.medium, "finalize_postrun", lambda *_args, **_kwargs: rejected)
    assert cli.main(
        [
            "--stage", "execute", "--output", str(medium.EXACT_BASE),
            "--expected-sha256", file_sha256(authorization),
        ]
    ) == 1


@pytest.mark.parametrize("family_index", [0, 1, 2])
def test_worker_profile_rejects_monitoring_or_timing_inconsistency(
    tmp_path: Path, family_index: int
) -> None:
    trial = medium.planned_trials(10)[family_index].identity
    base_metadata: dict[str, object] = {
        "checkpoint_evidence": {},
        "prediction_sha256": "a" * 64,
    }
    if trial.family == "ridge":
        base_metadata.update(
            backend="sklearn-ridge",
            stage_seconds={"forward": 0.1, "predict": 0.1},
            profile_monitoring={
                "curve_type": "non-iterative", "curves": {}, "warnings": [],
                "anomalies": [], "status": "normal",
            },
        )
    elif trial.family == "xgboost":
        rounds = [10] * 12
        base_metadata.update(
            backend="xgboost",
            per_target_rounds=rounds,
            stage_seconds={"forward": 0.1, "predict": 0.1},
            profile_monitoring={
                "curve_type": "per-target-best-rounds",
                "curves": {"per_target_rounds": rounds},
                "warnings": [], "anomalies": [], "status": "normal",
            },
        )
    else:
        curves = {
            "train_standardized_mse": [1.0], "inner_tune_mnmae": [0.5],
            "inner_tune_per_target_nmae": [[0.5] * 12], "gradient_norm": [1.0],
            "parameter_norm": [1.0],
        }
        curve = {
            **curves, "gpu_memory_bytes": [], "process_rss_bytes": [],
            "best_epoch": 1, "restored_epoch": 1,
            "restored_checkpoint_sha256": "b" * 64,
            "best_checkpoint_sha256": "b" * 64,
            "stop_reason": "max-epochs-reached",
            "maximum_consecutive_validation_increases": 0,
            "warnings": [], "anomalies": [], "status": "normal",
        }
        base_metadata.update(
            backend="torch", curves=curves, curve=curve, best_epoch=1,
            best_checkpoint_sha256="b" * 64, restored_checkpoint_sha256="b" * 64,
            stop_reason="max-epochs-reached",
            maximum_consecutive_validation_increases=0,
            stage_seconds={
                "feature_transform": 0.1, "forward": 0.1, "loss": 0.1,
                "backward": 0.1, "optimizer": 0.1, "validation": 0.1,
            },
            profile_monitoring={
                "curve_type": "epoch", "curves": curves, "warnings": [],
                "anomalies": [], "status": "normal",
            },
        )
    metadata_sha = medium.canonical_sha256(base_metadata)
    unsigned = {
        "schema_version": PROFILE_SCHEMA, "run_id": medium.RUN_ID,
        "trial": asdict(trial), "metadata": base_metadata,
        "metadata_sha256": metadata_sha,
        "label_reads": medium._label_reads(derived=1),
        "stage_timing": {
            "data_load": {"wall_seconds": 0.1, "cpu_seconds": 0.1},
            "recipe_transform": {"wall_seconds": 0.1, "cpu_seconds": 0.1},
        },
        **medium._false_boundaries(),
    }
    path = tmp_path / f"{trial.family}.profile.json"
    path.write_text(json.dumps({**unsigned, "semantic_sha256": medium.canonical_sha256(unsigned)}))
    validate_profile(path, trial=trial, expected_metadata_sha256=metadata_sha)
    base_metadata["profile_monitoring"]["status"] = "warning"
    metadata_sha = medium.canonical_sha256(base_metadata)
    unsigned["metadata_sha256"] = metadata_sha
    unsigned["metadata"] = base_metadata
    path.write_text(json.dumps({**unsigned, "semantic_sha256": medium.canonical_sha256(unsigned)}))
    with pytest.raises(MediumWorkerError, match="monitoring contract"):
        validate_profile(path, trial=trial, expected_metadata_sha256=metadata_sha)


def test_worker_telemetry_bridge_emits_live_heartbeat(tmp_path: Path) -> None:
    trial = medium.planned_trials(820)[0].identity
    run_start_sha256 = "a" * 64
    with medium.WorkerTelemetryBridge(tmp_path, trial, run_start_sha256) as bridge:
        bridge.sink.emit(
            "epoch_end",
            trial=trial,
            progress={"current": 1, "total": 20},
            metrics={
                "train_standardized_mse": 1.0,
                "inner_tune_mnmae": 0.5,
                "per_target_nmae": [0.5] * 12,
            },
        )
    rows = validate_event_stream(tmp_path / "telemetry/worker-1.jsonl")
    assert [row["event"] for row in rows] == ["heartbeat", "epoch_end"]
    assert rows[-1]["metrics"]["inner_tune_mnmae"] == 0.5


def test_cpu_thread_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        monkeypatch.delenv(name, raising=False)
    medium.set_fixed_cpu_threads()
    assert os.environ["OMP_NUM_THREADS"] == "1"


def test_bare_execute_contained_cli_is_rejected(tmp_path: Path) -> None:
    script = Path(__file__).parents[2] / "scripts/run_qm9_model_selection_v31_medium_profile.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--stage",
            "execute-contained",
            "--output",
            str(tmp_path),
            "--expected-sha256",
            "a" * 64,
        ],
        capture_output=True,
        text=True,
        env={
            key: value
            for key, value in os.environ.items()
            if key != "QM9_MEDIUM_OUTER_HARNESS_BINDING"
        },
    )
    assert result.returncode != 0
    assert "bare medium execution is forbidden" in result.stderr
