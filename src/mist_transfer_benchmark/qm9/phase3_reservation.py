"""Durable exactly-once reservation for pinned MIST candidate-test inference."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .io import atomic_write_json, canonical_hash


class Phase3ReservationError(ValueError):
    """Raised if candidate-test inference is unreserved or already attempted."""


def freeze_inference(payload: dict[str, object]) -> dict[str, object]:
    forbidden = {"runtime_seconds", "peak_rss_gib", "gpu_peak_memory_bytes"}
    if forbidden & set(payload):
        raise Phase3ReservationError("inference fingerprint contains volatile execution fields")
    unsigned = {"schema_version": "qm9-mist-inference-lock-v1", **payload}
    return {**unsigned, "inference_fingerprint": canonical_hash(unsigned)}


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def reserve_inference_once(lock_dir: str | Path, selection: dict[str, object]) -> Path:
    fingerprint = selection.get("inference_fingerprint")
    unsigned = {key: value for key, value in selection.items() if key != "inference_fingerprint"}
    if not isinstance(fingerprint, str) or canonical_hash(unsigned) != fingerprint:
        raise Phase3ReservationError("inference fingerprint is invalid")
    if re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
        raise Phase3ReservationError("inference fingerprint is not SHA-256 hex")
    directory = Path(lock_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise Phase3ReservationError("inference reservation directory is unsafe")
    path = directory / f"{fingerprint}.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise Phase3ReservationError(
            f"candidate-test inference was already reserved: {fingerprint}"
        ) from error
    record = {
        "schema_version": "qm9-mist-inference-reservation-v1",
        "status": "reserved-before-candidate-test-inference",
        "inference_fingerprint": fingerprint,
        "selection": selection,
    }
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(record, indent=2, sort_keys=True).encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_dir(directory)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def authorize_test_inference(path: str | Path, selection: dict[str, object]) -> None:
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("status") != "reserved-before-candidate-test-inference":
        raise Phase3ReservationError("inference reservation is not in its pre-test state")
    if record.get("selection") != selection:
        raise Phase3ReservationError("inference reservation does not authenticate this run")


def complete_inference(path: str | Path, result_hashes: dict[str, str]) -> None:
    reservation = Path(path)
    record = json.loads(reservation.read_text(encoding="utf-8"))
    if record.get("status") != "reserved-before-candidate-test-inference":
        raise Phase3ReservationError("inference reservation cannot be completed")
    record["status"] = "completed"
    record["result_hashes"] = result_hashes
    atomic_write_json(reservation, record, mode=0o600)
    _fsync_dir(reservation.parent)
