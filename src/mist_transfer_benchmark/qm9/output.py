"""Path-safe, tool-owned Phase 1 output directory handling."""

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

OWNER_MARKER = ".qm9-phase1-owner.json"
OWNER_KIND = "mist-transfer-benchmark-qm9-phase1-output"
OWNER_SCHEMA_VERSION = "1"
ALLOWED_OUTPUT_FILES = {
    OWNER_MARKER,
    "code_provenance.json",
    "datasets_environment.freeze.txt",
    "datasets_reference.json",
    "duplicate_events.jsonl",
    "duplicate_summary.json",
    "phase1_run.json",
    "phase1_run.sha256",
    "row_manifest.jsonl",
    "source_manifest.json",
    "split_assignments.tsv",
}


class OutputSafetyError(ValueError):
    """Raised before an unrelated path can be removed or replaced."""


@dataclass(frozen=True)
class OutputWorkspace:
    results_root: Path
    output_dir: Path
    staging_dir: Path


def owner_marker_payload() -> dict[str, str]:
    return {"kind": OWNER_KIND, "schema_version": OWNER_SCHEMA_VERSION}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _resolved_output(output_dir: Path, repo_root: Path) -> tuple[Path, Path]:
    if repo_root.is_symlink():
        raise OutputSafetyError("repository root must not be a symlink")
    results_root = repo_root / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    if results_root.is_symlink():
        raise OutputSafetyError("results root must not be a symlink")
    results_root = results_root.resolve(strict=True)
    requested = output_dir if output_dir.is_absolute() else repo_root / output_dir
    if ".." in output_dir.parts:
        raise OutputSafetyError("output path traversal is not allowed")
    if requested.is_symlink():
        raise OutputSafetyError("output directory must not be a symlink")
    resolved = requested.resolve(strict=False)
    if resolved == results_root or resolved.parent != results_root:
        raise OutputSafetyError("QM9 output must be a direct child of the repository results/")
    if resolved in {Path("/").resolve(), Path.home().resolve(), repo_root.resolve()}:
        raise OutputSafetyError("refusing a dangerous output target")
    return results_root, resolved


def _validate_owned_directory(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise OutputSafetyError(f"owned output must be a real directory: {path}")
    marker = path / OWNER_MARKER
    if marker.is_symlink() or not marker.is_file():
        raise OutputSafetyError(f"nonempty output lacks a regular ownership marker: {path}")
    if json.loads(marker.read_text(encoding="utf-8")) != owner_marker_payload():
        raise OutputSafetyError(f"output ownership marker is invalid: {path}")
    entries = {entry.name for entry in path.iterdir()}
    unexpected = entries - ALLOWED_OUTPUT_FILES
    if unexpected:
        raise OutputSafetyError(f"owned output contains unexpected entries: {sorted(unexpected)}")
    for entry in path.iterdir():
        observed = entry.lstat()
        if not stat.S_ISREG(observed.st_mode):
            raise OutputSafetyError(f"owned output contains a non-regular entry: {entry}")


def prepare_output_workspace(
    output_dir: str | Path,
    repo_root: str | Path,
    *,
    overwrite: bool,
) -> OutputWorkspace:
    root = Path(repo_root).resolve(strict=True)
    results_root, resolved = _resolved_output(Path(output_dir), root)
    if resolved.exists():
        if resolved.is_symlink() or not resolved.is_dir():
            raise OutputSafetyError(f"output target is not a real directory: {resolved}")
        if any(resolved.iterdir()):
            if not overwrite:
                raise FileExistsError(f"output directory is not empty: {resolved}")
            _validate_owned_directory(resolved)
    staging = Path(tempfile.mkdtemp(prefix=f".{resolved.name}.staging-", dir=results_root))
    return OutputWorkspace(results_root=results_root, output_dir=resolved, staging_dir=staging)


def write_owner_marker(staging_dir: str | Path) -> None:
    staging = Path(staging_dir)
    atomic_write_json(staging / OWNER_MARKER, owner_marker_payload(), mode=0o600)


def _remove_owned_directory(path: Path) -> None:
    _validate_owned_directory(path)
    for entry in path.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise OutputSafetyError(f"refusing to unlink non-regular output entry: {entry}")
        entry.unlink()
    path.rmdir()


def finalize_output_workspace(workspace: OutputWorkspace) -> None:
    staging = workspace.staging_dir
    target = workspace.output_dir
    _validate_owned_directory(staging)
    backup: Path | None = None
    if target.exists():
        if any(target.iterdir()):
            _validate_owned_directory(target)
            backup = workspace.results_root / f".{target.name}.backup-{uuid.uuid4().hex}"
            os.replace(target, backup)
        else:
            target.rmdir()
    try:
        os.replace(staging, target)
        _fsync_directory(workspace.results_root)
    except BaseException:
        if backup is not None and not target.exists():
            os.replace(backup, target)
        raise
    if backup is not None:
        _remove_owned_directory(backup)
        _fsync_directory(workspace.results_root)


def discard_staging_workspace(workspace: OutputWorkspace) -> None:
    staging = workspace.staging_dir
    if not staging.exists():
        return
    if (
        staging.is_symlink()
        or staging.parent != workspace.results_root
        or not staging.name.startswith(f".{workspace.output_dir.name}.staging-")
    ):
        raise OutputSafetyError(f"refusing to discard unrecognized staging path: {staging}")
    shutil.rmtree(staging)
