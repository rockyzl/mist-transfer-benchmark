"""Atomic acquisition of the one allowlisted pinned MIST snapshot."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

from .phase3_model import EXPECTED_FILES, MODEL_ID, MODEL_REVISION, verify_snapshot


class Phase3AcquireError(ValueError):
    """Raised before partial or altered model bytes can be installed."""


def acquire_snapshot(
    destination: str | Path, *, repo_root: str | Path, timeout_seconds: int = 300
) -> dict[str, object]:
    """Download only the fixed revision, or verify and reuse the complete cache."""

    root = Path(repo_root).resolve(strict=True)
    expected_parent = (root / "data/private/qm9/mist-phase3").resolve(strict=True)
    target = Path(destination)
    target = target if target.is_absolute() else root / target
    if target.is_symlink() or target.resolve(strict=False).parent != expected_parent:
        raise Phase3AcquireError(
            "model cache must be a direct child of data/private/qm9/mist-phase3"
        )
    target = target.resolve(strict=False)
    if target.name != "model":
        raise Phase3AcquireError("pinned model cache directory must be named model")
    if target.exists():
        manifest = verify_snapshot(target)
        return {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "retrieval_mode": "existing-verified-local-cache",
            "files": manifest,
        }
    staging = Path(tempfile.mkdtemp(prefix=".model.download-", dir=expected_parent))
    try:
        for name, (expected_bytes, expected_hash) in EXPECTED_FILES.items():
            url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}"
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "mist-transfer-benchmark-phase3/0.1.1"},
                method="GET",
            )
            path = staging / name
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            digest = hashlib.sha256()
            count = 0
            try:
                with os.fdopen(descriptor, "wb") as output, urllib.request.urlopen(
                    request, timeout=timeout_seconds
                ) as response:
                    if int(response.status) != 200:
                        raise Phase3AcquireError(f"unexpected HTTP status for {name}")
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        count += len(chunk)
                        if count > expected_bytes:
                            raise Phase3AcquireError(f"download exceeded byte limit: {name}")
                        digest.update(chunk)
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
            except BaseException:
                path.unlink(missing_ok=True)
                raise
            if count != expected_bytes or digest.hexdigest() != expected_hash:
                raise Phase3AcquireError(f"downloaded model artifact differs: {name}")
        manifest = verify_snapshot(staging)
        os.replace(staging, target)
        return {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "retrieval_mode": "atomic-fixed-revision-http-get",
            "files": manifest,
        }
    except BaseException:
        if staging.exists() and staging.parent == expected_parent:
            shutil.rmtree(staging)
        raise
