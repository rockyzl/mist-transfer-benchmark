"""Path-safe atomic workspaces for ignored QM9 Phase 3 artifacts."""

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

OWNER_MARKER = ".qm9-phase3-owner.json"
OWNER_PAYLOAD = {"kind": "mist-transfer-benchmark-qm9-phase3-output", "schema_version": "1"}
ALLOWED_FILES = {
    OWNER_MARKER,
    "code_provenance.json",
    "comparison.json",
    "failure_log.json",
    "mist_metrics.json",
    "mist_predictions.jsonl",
    "model_audit.json",
    "phase1_verification.json",
    "phase2_verification.json",
    "phase3_audit_run.json",
    "phase3_audit_run.sha256",
    "phase3_run.json",
    "phase3_run.sha256",
    "protocol_config.snapshot.toml",
    "runtime_environment.freeze.txt",
    "runtime_environment.json",
    "smoke.json",
    "worker_report.json",
}


class Phase3OutputError(ValueError):
    """Raised before an unsafe Phase 3 output mutation."""


@dataclass(frozen=True)
class Phase3Workspace:
    results_root: Path
    output_dir: Path
    staging_dir: Path


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_owned(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise Phase3OutputError("Phase 3 output must be a real directory")
    marker = path / OWNER_MARKER
    if marker.is_symlink() or not marker.is_file():
        raise Phase3OutputError("Phase 3 output lacks its ownership marker")
    if json.loads(marker.read_text(encoding="utf-8")) != OWNER_PAYLOAD:
        raise Phase3OutputError("Phase 3 ownership marker is invalid")
    for entry in path.iterdir():
        observed = entry.lstat()
        if entry.name not in ALLOWED_FILES or not stat.S_ISREG(observed.st_mode):
            raise Phase3OutputError(f"unexpected/non-regular Phase 3 output: {entry.name}")


def prepare_phase3_workspace(
    output_dir: str | Path, repo_root: str | Path, *, overwrite: bool
) -> Phase3Workspace:
    root = Path(repo_root).resolve(strict=True)
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    if results.is_symlink():
        raise Phase3OutputError("results root must not be a symlink")
    results = results.resolve(strict=True)
    raw = Path(output_dir)
    if ".." in raw.parts:
        raise Phase3OutputError("Phase 3 output path traversal is not allowed")
    requested = raw if raw.is_absolute() else root / raw
    if requested.is_symlink():
        raise Phase3OutputError("Phase 3 output target must not be a symlink")
    output = requested.resolve(strict=False)
    if output == results or output.parent != results:
        raise Phase3OutputError("Phase 3 output must be a direct child of results/")
    if output.exists():
        if not output.is_dir():
            raise Phase3OutputError("Phase 3 output target is not a directory")
        if any(output.iterdir()):
            if not overwrite:
                raise FileExistsError(f"output directory is not empty: {output}")
            _validate_owned(output)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=results))
    return Phase3Workspace(results, output, staging)


def write_owner(staging: str | Path) -> None:
    atomic_write_json(Path(staging) / OWNER_MARKER, OWNER_PAYLOAD, mode=0o600)


def _remove_owned(path: Path) -> None:
    _validate_owned(path)
    for entry in path.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise Phase3OutputError(f"refusing to unlink non-regular output: {entry}")
        entry.unlink()
    path.rmdir()


def finalize_workspace(workspace: Phase3Workspace) -> None:
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
        _fsync_dir(workspace.results_root)
    except BaseException:
        if backup is not None and not workspace.output_dir.exists():
            os.replace(backup, workspace.output_dir)
        raise
    if backup is not None:
        _remove_owned(backup)
        _fsync_dir(workspace.results_root)


def discard_workspace(workspace: Phase3Workspace) -> None:
    staging = workspace.staging_dir
    if not staging.exists():
        return
    if (
        staging.is_symlink()
        or staging.parent != workspace.results_root
        or not staging.name.startswith(f".{workspace.output_dir.name}.staging-")
    ):
        raise Phase3OutputError("refusing to discard an unrecognized Phase 3 staging path")
    shutil.rmtree(staging)
