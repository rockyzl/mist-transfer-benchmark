from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from mist_transfer_benchmark.qm9 import model_selection_v31_medium_profile as medium
from mist_transfer_benchmark.qm9.model_selection_v31_medium_worker import write_spec
from mist_transfer_benchmark.qm9.model_selection_v31_supervisor import (
    ExternalWorkerSidecar,
    RunSupervisor,
    SidecarPolicy,
    file_sha256,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_QM9_MEDIUM_INTEGRATION") != "1",
        reason="explicit opt-in real three-backend medium fixture",
    ),
]


def _decision(path: Path, payload: dict[str, object]) -> Path:
    now = datetime.now(UTC)
    unsigned = {
        **payload,
        "issued_at_utc": (now - timedelta(seconds=1)).isoformat(),
        "expires_at_utc": (now + timedelta(minutes=30)).isoformat(),
    }
    path.write_text(json.dumps({**unsigned, "semantic_sha256": medium.canonical_sha256(unsigned)}))
    return path


def test_real_three_backend_medium_vertical_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opt-in only: 8192+ unequal groups, real workers, sidecar, checkpoints and replay."""

    base = tmp_path / "medium"
    monkeypatch.setattr(medium, "EXACT_BASE", base)
    sizes = np.resize(np.arange(1, 6), 4000)
    groups = np.concatenate(
        [np.repeat(f"g-{index}", size) for index, size in enumerate(sizes)]
    )
    selected = medium.stable_group_sample(groups)
    selected_groups = groups[selected]
    train, tune = medium.medium_split(selected_groups)
    rng = np.random.default_rng(medium.SEED)
    raw = sparse.random(
        len(selected),
        2065,
        density=0.003,
        random_state=rng,
        format="csr",
        dtype=np.float64,
    )
    sample = medium.sample_evidence(selected, selected_groups)
    plan = medium.planned_trials(len(tune))
    target_free_unsigned = {
        "schema_version": "qm9-v31-medium-target-free-plan-v1",
        "run_id": medium.RUN_ID,
        "selected_source_indices": selected.tolist(),
        "train_positions": train.tolist(),
        "tune_positions": tune.tolist(),
        "sample": sample,
        "selected_feature_sha256": medium._sparse_sha256(raw),
        "train_feature_sha256": medium._sparse_sha256(raw[train].tocsr()),
        "tune_feature_sha256": medium._sparse_sha256(raw[tune].tocsr()),
        "plan": [row.payload() for row in plan],
        "plan_sha256": medium.canonical_sha256([row.payload() for row in plan]),
        "input_inventory_sha256": "4" * 64,
        "label_reads": medium._label_reads(),
        **medium._false_boundaries(),
    }
    target_free = {
        **target_free_unsigned,
        "semantic_sha256": medium.canonical_sha256(target_free_unsigned),
    }
    monkeypatch.setattr(
        medium,
        "_target_free_contract",
        lambda **_kwargs: (target_free, raw, None),
    )
    qm9 = tmp_path / "qm9.csv"
    matrix = tmp_path / "features.npz"
    manifest = tmp_path / "manifest.json"
    phase1 = tmp_path / "phase1"
    qm9.write_text("fixture\n")
    matrix.write_bytes(b"fixture")
    manifest.write_text(json.dumps({"schema_version": "qm9-paper-feature-artifact-v1"}))
    phase1.mkdir()
    prepared_path = medium.prepare(
        base, runner_identity="runner", qm9_csv=qm9, feature_matrix=matrix,
        feature_manifest=manifest, phase1_dir=phase1,
    )
    prepared = json.loads(prepared_path.read_text())
    bindings = medium._decision_bindings(prepared, nonce="5" * 64)
    review = _decision(tmp_path / "review.json", {
        "schema_version": medium.EXTERNAL_REVIEW_SCHEMA, "run_id": medium.RUN_ID,
        "decision": "APPROVE_MEDIUM_PROFILE_ONLY", "reviewer_identity": "reviewer",
        "prepare_file_sha256": file_sha256(prepared_path),
        "prepare_semantic_sha256": prepared["semantic_sha256"],
        **bindings,
    })
    reviewed_path = medium.review(
        base, approval_path=review, expected_approval_sha256=file_sha256(review)
    )
    reviewed = json.loads(reviewed_path.read_text())
    authorization = _decision(tmp_path / "authorization.json", {
        "schema_version": medium.EXTERNAL_AUTH_SCHEMA, "run_id": medium.RUN_ID,
        "decision": "AUTHORIZE_MEDIUM_PROFILE_ONLY", "authorizer_identity": "authorizer",
        "reviewer_identity": "reviewer", "review_file_sha256": file_sha256(reviewed_path),
        "review_semantic_sha256": reviewed["semantic_sha256"],
        **bindings,
    })
    medium.authorize(
        base, authorization_path=authorization,
        expected_authorization_sha256=file_sha256(authorization),
    )

    targets = rng.normal(size=(len(selected), 12))
    data = base / "fixture-selected.npz"
    medium._save_selected_npz(
        data, raw[train].tocsr(), targets[train], raw[tune].tocsr(), targets[tune]
    )
    binding = medium.build_supervisor_binding(base, data_path=data, sample=sample, plan=plan)
    supervisor = RunSupervisor.for_medium_profile(base / "run", binding, plan)
    worker_data = supervisor.root / "input/fixture.npz"
    worker_data.parent.mkdir()
    shutil.copyfile(data, worker_data)
    worker_data.chmod(0o444)
    try:
        for trial_spec, candidate in zip(plan, medium.candidates(), strict=True):
            trial = trial_spec.identity
            supervisor.reserve(trial)
            supervisor.start_trial(trial)
            result = supervisor.root / f"worker/{trial.trial_index}.json"
            spec = supervisor.root / f"spec/{trial.trial_index}.json"
            write_spec(
                spec, run_root=supervisor.root, data_path=worker_data, result_path=result,
                profile_path=supervisor.root / f"profile/{trial.trial_index}.json",
                telemetry_path=supervisor.root / f"telemetry/worker-{trial.trial_index}.jsonl",
                checkpoint_root=supervisor.checkpoints.root, candidate=candidate, trial=trial,
                completion=supervisor.completion_contract(trial),
                checkpoint_bindings={
                    "plan_sha256": binding.plan_sha256, "config_sha256": binding.config_sha256,
                    "input_sha256": binding.input_sha256, "cache_sha256": binding.profile_sha256,
                    "run_start_sha256": supervisor.run_start_sha256,
                },
            )
            policy = SidecarPolicy(
                heartbeat_seconds=10,
                heartbeat_stale_after_seconds=20,
                model_progress_stale_after_seconds=20,
                soft_deadline_seconds=2280,
                hard_deadline_seconds=2400, rss_max_bytes=medium.RSS_HARD_BYTES,
                disk_free_min_bytes=5 * 1024**3, gpu_allocated_max_bytes=1,
                gpu_reserved_max_bytes=1, gpu_temperature_max_c=1, require_rss_sensor=True,
                require_disk_sensor=True, require_gpu_sensors=False, poll_seconds=0.5,
            )
            code = ExternalWorkerSidecar(supervisor, policy).run(
                trial,
                [
                    sys.executable,
                    "-m",
                    "mist_transfer_benchmark.qm9.model_selection_v31_medium_worker",
                    "--worker-spec",
                    str(spec),
                ],
                heartbeat_path=supervisor.root / f"telemetry/worker-{trial.trial_index}.heartbeat",
                prepared=True, result_path=result,
            )
            assert code == 0
        supervisor.close_ledger()
        final = supervisor.finalize_profile_success({"fixture": True})
        assert final.is_file()
        supervisor.assert_profile_complete()
    finally:
        supervisor.close()
