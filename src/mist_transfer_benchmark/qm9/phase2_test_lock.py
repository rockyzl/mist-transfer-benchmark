"""Durable exactly-once reservation for the locked Phase 2 test evaluation."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .io import atomic_write_json, canonical_hash


class TestLockError(ValueError):
    """Raised when a frozen selection is absent or its test was already reserved."""


@dataclass
class TestLabelGate:
    """In-process guard proving a durable reservation precedes test-label loading."""

    authorized_selection_fingerprint: str | None = None

    def authorize(self, reservation_path: str | Path, selection: dict[str, object]) -> None:
        record = json.loads(Path(reservation_path).read_text(encoding="utf-8"))
        fingerprint = selection.get("selection_fingerprint")
        if (
            record.get("status") != "reserved-before-reading-test-labels"
            or record.get("selection_fingerprint") != fingerprint
            or record.get("selection") != selection
        ):
            raise TestLockError("durable reservation does not authenticate this selection")
        self.authorized_selection_fingerprint = str(fingerprint)

    def require_authorized(self, selection: dict[str, object]) -> None:
        if self.authorized_selection_fingerprint != selection.get("selection_fingerprint"):
            raise TestLockError("test labels cannot be loaded before selection reservation")


def freeze_selection(payload: dict[str, object]) -> dict[str, object]:
    def scientific_basis(value):
        if isinstance(value, dict):
            return {
                key: scientific_basis(item)
                for key, item in value.items()
                if key
                not in {
                    "runtime_seconds",
                    "started_at_utc",
                    "completed_at_utc",
                    "peak_rss_gib",
                }
            }
        if isinstance(value, list):
            return [scientific_basis(item) for item in value]
        return value

    unsigned = {
        "schema_version": "qm9-phase2-selection-lock-v1",
        **scientific_basis(payload),
    }
    return {**unsigned, "selection_fingerprint": canonical_hash(unsigned)}


def reserve_test_once(lock_dir: str | Path, selection: dict[str, object]) -> Path:
    fingerprint = selection.get("selection_fingerprint")
    unsigned = {key: value for key, value in selection.items() if key != "selection_fingerprint"}
    if not isinstance(fingerprint, str) or canonical_hash(unsigned) != fingerprint:
        raise TestLockError("selection fingerprint is missing or invalid")
    if re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
        raise TestLockError("selection fingerprint is not a SHA-256 hex digest")
    directory = Path(lock_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink():
        raise TestLockError("test-lock directory must not be a symlink")
    path = directory / f"{fingerprint}.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        raise TestLockError(f"test evaluation was already reserved: {fingerprint}") from error
    record = {
        "schema_version": "qm9-phase2-test-reservation-v1",
        "status": "reserved-before-reading-test-labels",
        "selection_fingerprint": fingerprint,
        "selection": selection,
    }
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(record, indent=2, sort_keys=True).encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def complete_test_reservation(path: str | Path, result_hashes: dict[str, str]) -> None:
    reservation = Path(path)
    record = json.loads(reservation.read_text(encoding="utf-8"))
    if record.get("status") != "reserved-before-reading-test-labels":
        raise TestLockError("test reservation is not in its pre-label reserved state")
    record["status"] = "completed"
    record["result_hashes"] = result_hashes
    atomic_write_json(reservation, record, mode=0o600)
