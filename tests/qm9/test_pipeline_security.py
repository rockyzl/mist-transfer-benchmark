from __future__ import annotations

import copy
import shlex
import tomllib
from pathlib import Path

import pytest

from mist_transfer_benchmark.qm9.pipeline import (
    QM9AuditError,
    _enforce_resource_limits,
    _run_reference,
    _validate_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _config():
    with (REPO_ROOT / "configs/qm9_28m.toml").open("rb") as handle:
        return tomllib.load(handle)


def _set_nested(value, path, replacement):
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("dataset", "url"), "https://example.test/wrong.csv"),
        (("dataset", "local_verification", "content_sha256"), "0" * 64),
        (("dataset", "local_verification", "observed_header"), ["wrong"]),
        (("dataset", "upstream_declaration", "required_target_columns"), ["mu"]),
        (("dataset", "protocol_expectation", "identity_columns"), ["smiles"]),
        (("split", "seed_first"), 7),
        (("split", "first_test_size"), 0.25),
        (("split", "derived_test_rows_if_upstream_row_count_matches"), 1),
        (("identity_audit", "audit_train_test_overlap"), False),
        (("cohorts", "secondary", "within_test_keep"), "highest-source-row-index"),
        (("reconstruction_environment", "numpy"), "==1.0.0"),
        (("resource_budget", "reference_timeout_seconds"), 0),
    ],
)
def test_config_preflight_rejects_single_field_drift(path, replacement):
    config = copy.deepcopy(_config())
    _set_nested(config, path, replacement)
    with pytest.raises(QM9AuditError):
        _validate_config(config)


def test_current_config_and_runtime_pass_preflight():
    _validate_config(_config())


def test_parent_and_reference_child_resource_limits_are_separate():
    config = _config()
    _enforce_resource_limits(config, parent_peak_rss_gib=1.0, child_peak_rss_gib=2.0)
    with pytest.raises(QM9AuditError, match="parent"):
        _enforce_resource_limits(config, parent_peak_rss_gib=65.0, child_peak_rss_gib=2.0)
    with pytest.raises(QM9AuditError, match="reference"):
        _enforce_resource_limits(config, parent_peak_rss_gib=1.0, child_peak_rss_gib=65.0)


def test_reference_subprocess_timeout_terminates_process_group(tmp_path):
    executable = tmp_path / "slow-python"
    executable.write_text("#!/bin/sh\nsleep 30\n")
    executable.chmod(0o755)
    with pytest.raises(QM9AuditError, match="exceeded 1s"):
        _run_reference(executable, tmp_path / "out.json", REPO_ROOT, 20, timeout_seconds=1)


def test_reference_subprocess_does_not_inherit_pythonpath(tmp_path, monkeypatch):
    evidence = tmp_path / "environment.txt"
    executable = tmp_path / "inspect-python"
    executable.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n%s\\n' \"$PYTHONPATH\" \"$PWD\" > {shlex.quote(str(evidence))}\n"
        "exit 3\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PYTHONPATH", "/tmp/malicious-pythonpath")
    with pytest.raises(QM9AuditError, match="isolated Datasets reference failed"):
        _run_reference(executable, tmp_path / "out.json", REPO_ROOT, 20, timeout_seconds=5)
    pythonpath, cwd = evidence.read_text().splitlines()
    assert pythonpath == str(REPO_ROOT / "src")
    assert cwd == str(REPO_ROOT)
