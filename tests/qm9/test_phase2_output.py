from __future__ import annotations

import json

import pytest

from mist_transfer_benchmark.qm9.phase2_output import (
    OWNER_MARKER,
    OWNER_PAYLOAD,
    Phase2OutputError,
    finalize_phase2_workspace,
    prepare_phase2_workspace,
    write_phase2_owner,
)


def test_phase2_output_will_not_overwrite_unowned_content(tmp_path):
    repo = tmp_path / "repo"
    output = repo / "results" / "important"
    output.mkdir(parents=True)
    (output / "keep.txt").write_text("keep")
    with pytest.raises(Phase2OutputError, match="ownership"):
        prepare_phase2_workspace(output, repo, overwrite=True)
    assert (output / "keep.txt").read_text() == "keep"


def test_owned_phase2_output_can_be_atomically_replaced(tmp_path):
    repo = tmp_path / "repo"
    output = repo / "results" / "phase2"
    output.mkdir(parents=True)
    (output / OWNER_MARKER).write_text(json.dumps(OWNER_PAYLOAD))
    (output / "feature_manifest.json").write_text("old")
    workspace = prepare_phase2_workspace(output, repo, overwrite=True)
    (workspace.staging_dir / "feature_manifest.json").write_text("new")
    write_phase2_owner(workspace.staging_dir)
    finalize_phase2_workspace(workspace)
    assert (output / "feature_manifest.json").read_text() == "new"


def test_phase2_output_rejects_paths_outside_results(tmp_path):
    repo = tmp_path / "repo"
    (repo / "results").mkdir(parents=True)
    with pytest.raises(Phase2OutputError):
        prepare_phase2_workspace(repo / "outside", repo, overwrite=False)
