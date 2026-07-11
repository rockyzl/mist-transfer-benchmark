"""Path-safe atomic workspace management for ignored Phase 2 artifacts."""

from __future__ import annotations

import json
import os
import shutil
import stat
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from .io import atomic_write_json

OWNER_MARKER = ".qm9-phase2-owner.json"
OWNER_PAYLOAD = {"kind": "mist-transfer-benchmark-qm9-phase2-output", "schema_version": "1"}
ALLOWED_FILES = {
    OWNER_MARKER,
    "code_provenance.json",
    "feature_manifest.json",
    "feature_matrix.npz",
    "phase1_verification.json",
    "phase2_run.json",
    "phase2_run.sha256",
    "predictions.jsonl",
    "protocol_config.snapshot.toml",
    "random_forest_attempt.json",
    "scaler.json",
    "selection_lock.json",
    "selection_lock.sha256",
    "test_metrics.json",
    "validation_metrics.json",
}


class Phase2OutputError(ValueError):
    """Raised before an unsafe Phase 2 output mutation."""


@dataclass(frozen=True)
class Phase2Workspace:
    results_root: Path
    output_dir: Path
    staging_dir: Path


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_owned(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise Phase2OutputError("Phase 2 output must be a real directory")
    marker = path / OWNER_MARKER
    if marker.is_symlink() or not marker.is_file():
        raise Phase2OutputError("nonempty Phase 2 output lacks its ownership marker")
    if json.loads(marker.read_text(encoding="utf-8")) != OWNER_PAYLOAD:
        raise Phase2OutputError("Phase 2 ownership marker is invalid")
    for entry in path.iterdir():
        if entry.name not in ALLOWED_FILES:
            raise Phase2OutputError(f"unexpected Phase 2 output entry: {entry.name}")
        if entry.is_symlink() or not stat.S_ISREG(entry.lstat().st_mode):
            raise Phase2OutputError(f"Phase 2 output entry is not a regular file: {entry}")


def prepare_phase2_workspace(
    output_dir: str | Path, repo_root: str | Path, *, overwrite: bool
) -> Phase2Workspace:
    root = Path(repo_root).resolve(strict=True)
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    if results.is_symlink():
        raise Phase2OutputError("results root must not be a symlink")
    results = results.resolve(strict=True)
    raw = Path(output_dir)
    if ".." in raw.parts:
        raise Phase2OutputError("Phase 2 output path traversal is not allowed")
    requested = raw if raw.is_absolute() else root / raw
    if requested.is_symlink():
        raise Phase2OutputError("Phase 2 output must not be a symlink")
    output = requested.resolve(strict=False)
    if output.parent != results or output == results:
        raise Phase2OutputError("Phase 2 output must be a direct child of results/")
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is not empty: {output}")
        _validate_owned(output)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=results))
    return Phase2Workspace(results, output, staging)


def write_phase2_owner(path: str | Path) -> None:
    atomic_write_json(Path(path) / OWNER_MARKER, OWNER_PAYLOAD, mode=0o600)


def _remove_owned(path: Path) -> None:
    _validate_owned(path)
    for entry in path.iterdir():
        entry.unlink()
    path.rmdir()


def finalize_phase2_workspace(workspace: Phase2Workspace) -> None:
    _validate_owned(workspace.staging_dir)
    backup: Path | None = None
    if workspace.output_dir.exists():
        if any(workspace.output_dir.iterdir()):
            _validate_owned(workspace.output_dir)
            backup = workspace.results_root / (
                f".{workspace.output_dir.name}.backup-{uuid.uuid4().hex}"
            )
            os.replace(workspace.output_dir, backup)
        else:
            workspace.output_dir.rmdir()
    try:
        os.replace(workspace.staging_dir, workspace.output_dir)
        _fsync_directory(workspace.results_root)
    except BaseException:
        if backup is not None and not workspace.output_dir.exists():
            os.replace(backup, workspace.output_dir)
        raise
    if backup is not None:
        _remove_owned(backup)
        _fsync_directory(workspace.results_root)


def discard_phase2_workspace(workspace: Phase2Workspace) -> None:
    staging = workspace.staging_dir
    if not staging.exists():
        return
    if (
        staging.is_symlink()
        or staging.parent != workspace.results_root
        or not staging.name.startswith(f".{workspace.output_dir.name}.staging-")
    ):
        raise Phase2OutputError("refusing to discard an unrecognized Phase 2 staging path")
    shutil.rmtree(staging)
