from __future__ import annotations

import subprocess

from mist_transfer_benchmark.qm9.provenance import capture_code_provenance


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _repo(tmp_path):
    repo = tmp_path / "repo"
    source = repo / "src" / "mist_transfer_benchmark"
    source.mkdir(parents=True)
    (source / "core.py").write_text("VALUE = 1\n")
    (repo / "configs").mkdir()
    config = repo / "configs" / "qm9.toml"
    config.write_text('status = "test"\n')
    (repo / "pyproject.toml").write_text('[project]\nname = "test"\nversion = "0"\n')
    (repo / "uv.lock").write_text("version = 1\n")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "initial")
    return repo, config


def test_provenance_covers_untracked_source_and_is_order_stable(tmp_path):
    repo, config = _repo(tmp_path)
    first = capture_code_provenance(repo, config)
    second = capture_code_provenance(repo, config)
    assert first == second
    untracked = repo / "src" / "mist_transfer_benchmark" / "new.py"
    untracked.write_text("NEW = True\n")

    changed = capture_code_provenance(repo, config)

    assert changed["aggregate_sha256"] != first["aggregate_sha256"]
    assert changed["git_dirty_in_scope"] is True
    assert any(item["path"].endswith("new.py") for item in changed["files"])


def test_provenance_changes_with_mode_or_content(tmp_path):
    repo, config = _repo(tmp_path)
    first = capture_code_provenance(repo, config)
    source = repo / "src" / "mist_transfer_benchmark" / "core.py"
    source.chmod(0o755)
    mode_changed = capture_code_provenance(repo, config)
    assert mode_changed["aggregate_sha256"] != first["aggregate_sha256"]
    source.write_text("VALUE = 2\n")
    content_changed = capture_code_provenance(repo, config)
    assert content_changed["aggregate_sha256"] != mode_changed["aggregate_sha256"]
