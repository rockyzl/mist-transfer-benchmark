"""Reproducible provenance for dirty or clean Phase 1 execution code."""

from __future__ import annotations

import hashlib
import stat
import subprocess
from pathlib import Path

from .io import canonical_hash, sha256_file


class ProvenanceError(ValueError):
    """Raised if execution-relevant source cannot be identified exactly."""


def _git(repo_root: Path, arguments: list[str]) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ProvenanceError(
            f"git {' '.join(arguments)} failed: {completed.stderr.decode(errors='replace')}"
        )
    return completed.stdout


def _relevant_paths(repo_root: Path, config_path: Path) -> list[Path]:
    source_root = repo_root / "src" / "mist_transfer_benchmark"
    paths = list(source_root.rglob("*.py"))
    paths.extend((config_path, repo_root / "pyproject.toml", repo_root / "uv.lock"))
    unique = {path.resolve(strict=True) for path in paths}
    return sorted(unique, key=lambda path: path.relative_to(repo_root).as_posix())


def capture_code_provenance(repo_root: str | Path, config_path: str | Path) -> dict[str, object]:
    root = Path(repo_root).resolve(strict=True)
    config = Path(config_path)
    if not config.is_absolute():
        config = root / config
    paths = _relevant_paths(root, config)
    files: list[dict[str, object]] = []
    relative_paths: list[str] = []
    for path in paths:
        if path.is_symlink():
            raise ProvenanceError(f"execution source must not be a symlink: {path}")
        observed = path.lstat()
        if not stat.S_ISREG(observed.st_mode):
            raise ProvenanceError(f"execution source is not a regular file: {path}")
        relative = path.relative_to(root).as_posix()
        relative_paths.append(relative)
        files.append(
            {
                "path": relative,
                "mode": f"{stat.S_IMODE(observed.st_mode):04o}",
                "bytes": observed.st_size,
                "sha256": sha256_file(path),
            }
        )
    head = _git(root, ["rev-parse", "HEAD"]).decode().strip()
    scope = ["--", *relative_paths]
    status = _git(root, ["status", "--porcelain=v1", "-z", "--untracked-files=all", *scope])
    diff = _git(root, ["diff", "--binary", "--no-ext-diff", "HEAD", *scope])
    status_entries = [item.decode(errors="surrogateescape") for item in status.split(b"\0") if item]
    file_manifest_sha256 = canonical_hash(files)
    result: dict[str, object] = {
        "schema_version": "qm9-code-provenance-v1",
        "git_head": head,
        "git_dirty_in_scope": bool(status),
        "git_status_porcelain_v1_z_sha256": hashlib.sha256(status).hexdigest(),
        "git_status_entries": status_entries,
        "git_tracked_binary_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "git_tracked_binary_diff_bytes": len(diff),
        "scope": relative_paths,
        "files": files,
        "file_manifest_canonical_json_sha256": file_manifest_sha256,
    }
    result["aggregate_sha256"] = canonical_hash(result)
    return result


def assert_code_provenance_unchanged(
    expected: dict[str, object], repo_root: str | Path, config_path: str | Path
) -> None:
    observed = capture_code_provenance(repo_root, config_path)
    if observed != expected:
        raise ProvenanceError("execution-relevant source changed during the Phase 1 audit")
