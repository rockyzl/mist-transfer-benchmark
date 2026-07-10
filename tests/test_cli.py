import json

import pandas as pd

import mist_transfer_benchmark.cli as cli
from mist_transfer_benchmark.cli import main

from .conftest import FIXTURE_CSV, INTERNAL_FIXTURE_CSV


def test_validate_command(capsys):
    exit_code = main(["validate", str(FIXTURE_CSV), "--json"])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["valid"] is True
    assert output["row_count"] == 24


def test_run_command_writes_auditable_artifacts(tmp_path, monkeypatch):
    output_dir = tmp_path / "run"
    monkeypatch.setattr(
        cli,
        "source_control_metadata",
        lambda _root: {"revision": "test-revision", "dirty": False, "git_available": True},
    )

    exit_code = main(
        [
            "run-baseline",
            str(INTERNAL_FIXTURE_CSV),
            "--output-dir",
            str(output_dir),
            "--split",
            "scaffold",
            "--models",
            "dummy,ridge",
            "--n-bits",
            "128",
            "--seed",
            "42",
        ]
    )

    assert exit_code == 0
    assert (output_dir / "predictions.csv").is_file()
    assert (output_dir / "split_assignments.csv").is_file()
    run = json.loads((output_dir / "run.json").read_text())
    assert run["scientific_status"] == "software-smoke-test-only"
    assert run["dataset"]["row_count"] == 12
    assert run["source_control"]["revision"] == "test-revision"
    assert run["source_control"]["dirty"] is False
    assert len(run["uv_lock_sha256"]) == 64
    assert set(run["metrics"]) == {"dummy", "ridge"}
    assert len(run["run_fingerprint"]) == 64
    assert "split_group_key" in (output_dir / "split_assignments.csv").read_text().splitlines()[0]
    assert "scaffold" in (output_dir / "split_assignments.csv").read_text().splitlines()[0]


def test_run_refuses_nonempty_output_without_overwrite(tmp_path):
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("keep")

    exit_code = main(
        [
            "run-baseline",
            str(INTERNAL_FIXTURE_CSV),
            "--output-dir",
            str(output_dir),
            "--models",
            "dummy",
        ]
    )

    assert exit_code == 2
    assert (output_dir / "keep.txt").read_text() == "keep"


def test_cli_comparability_gate_and_unsafe_override(tmp_path):
    frame = pd.read_csv(INTERNAL_FIXTURE_CSV, dtype=str, keep_default_na=False)
    frame.loc[frame.index[-1], "measurement_method"] = "different_protocol"
    mixed_csv = tmp_path / "mixed.csv"
    frame.to_csv(mixed_csv, index=False)

    rejected = main(
        [
            "run-baseline",
            str(mixed_csv),
            "--output-dir",
            str(tmp_path / "rejected"),
            "--models",
            "dummy",
        ]
    )
    accepted = main(
        [
            "run-baseline",
            str(mixed_csv),
            "--output-dir",
            str(tmp_path / "accepted"),
            "--models",
            "dummy",
            "--unsafe-allow-condition-ignorant-mixing",
        ]
    )

    assert rejected == 2
    assert accepted == 0
    run = json.loads((tmp_path / "accepted" / "run.json").read_text())
    assert run["protocol"]["unsafe_allow_condition_ignorant_mixing"] is True
    assert run["protocol"]["comparability"]["unsafe_override_used"] is True
