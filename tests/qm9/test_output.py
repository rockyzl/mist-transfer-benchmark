from __future__ import annotations

import json

import pytest

from mist_transfer_benchmark.qm9.output import (
    OutputSafetyError,
    discard_staging_workspace,
    finalize_output_workspace,
    owner_marker_payload,
    prepare_output_workspace,
    write_owner_marker,
)


def _repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "results").mkdir(parents=True)
    return repo


def test_overwrite_never_deletes_unowned_directory(tmp_path):
    repo = _repo(tmp_path)
    output = repo / "results" / "important"
    output.mkdir()
    sentinel = output / "do-not-delete.txt"
    sentinel.write_text("user data")

    with pytest.raises(OutputSafetyError, match="ownership marker"):
        prepare_output_workspace(output, repo, overwrite=True)

    assert sentinel.read_text() == "user data"


def test_owned_output_can_be_replaced_without_recursive_arbitrary_delete(tmp_path):
    repo = _repo(tmp_path)
    output = repo / "results" / "qm9-phase1"
    output.mkdir()
    (output / ".qm9-phase1-owner.json").write_text(
        json.dumps(owner_marker_payload()) + "\n"
    )
    (output / "phase1_run.json").write_text("old")
    workspace = prepare_output_workspace(output, repo, overwrite=True)
    (workspace.staging_dir / "phase1_run.json").write_text("new")
    write_owner_marker(workspace.staging_dir)

    finalize_output_workspace(workspace)

    assert (output / "phase1_run.json").read_text() == "new"
    assert json.loads((output / ".qm9-phase1-owner.json").read_text()) == owner_marker_payload()


def test_dangerous_or_symlink_output_is_rejected(tmp_path):
    repo = _repo(tmp_path)
    with pytest.raises(OutputSafetyError):
        prepare_output_workspace(repo, repo, overwrite=True)
    real = repo / "results" / "real"
    real.mkdir()
    link = repo / "results" / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(OutputSafetyError, match="symlink"):
        prepare_output_workspace(link, repo, overwrite=True)


def test_unfinished_staging_can_only_be_discarded_inside_results(tmp_path):
    repo = _repo(tmp_path)
    workspace = prepare_output_workspace(repo / "results" / "qm9", repo, overwrite=False)
    (workspace.staging_dir / "partial").write_text("partial")
    discard_staging_workspace(workspace)
    assert not workspace.staging_dir.exists()
