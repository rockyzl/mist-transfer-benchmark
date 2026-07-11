"""Auditable result artifact creation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from . import __version__


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment_metadata() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: _version(name)
            for name in ("mist-transfer-benchmark", "numpy", "pandas", "rdkit", "scikit-learn")
        },
    }


def git_revision(repo_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def source_control_metadata(repo_root: Path) -> dict[str, object]:
    """Capture revision plus dirty state without assuming a commit exists."""

    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "revision": git_revision(repo_root),
        "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
        "git_available": status.returncode == 0,
    }


def write_run_artifacts(
    output_dir: Path,
    artifact: dict[str, object],
    predictions: pd.DataFrame,
    assignments: pd.DataFrame,
    overwrite: bool = False,
) -> Path:
    """Write a JSON run record and auditable CSV tables."""

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "predictions.csv"
    assignments_path = output_dir / "split_assignments.csv"
    predictions.to_csv(predictions_path, index=False, float_format="%.12g")
    assignments.to_csv(assignments_path, index=False)

    complete = dict(artifact)
    complete["artifact_schema_version"] = "1.0"
    complete["benchmark_code_version"] = __version__
    complete["created_at_utc"] = datetime.now(UTC).isoformat()
    complete["environment"] = environment_metadata()
    complete["artifacts"] = {
        "predictions": predictions_path.name,
        "predictions_sha256": sha256_file(predictions_path),
        "split_assignments": assignments_path.name,
        "split_assignments_sha256": sha256_file(assignments_path),
    }
    run_path = output_dir / "run.json"
    run_path.write_text(
        json.dumps(complete, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return run_path
