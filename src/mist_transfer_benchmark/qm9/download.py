"""Safe atomic retrieval and immutable snapshots of the DeepChem QM9 CSV."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


class DownloadError(ValueError):
    """Raised before an untrusted response can replace a verified local source."""


@dataclass(frozen=True)
class SourceSnapshot:
    bytes: int
    sha256: str
    device: int
    inode: int
    mode: int
    mtime_ns: int
    ctime_ns: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DownloadRecord:
    requested_url: str
    final_url: str | None
    retrieval_mode: str
    http_status: int | None
    response_headers: dict[str, str]
    bytes: int
    sha256: str
    started_at_utc: str
    completed_at_utc: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_regular_readonly(path: Path) -> tuple[int, os.stat_result]:
    if path.is_symlink():
        raise DownloadError(f"source path must not be a symlink: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    observed = os.fstat(descriptor)
    if not stat.S_ISREG(observed.st_mode):
        os.close(descriptor)
        raise DownloadError(f"source path is not a regular file: {path}")
    return descriptor, observed


def _validate_snapshot(
    snapshot: SourceSnapshot,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    if snapshot.bytes != expected_bytes:
        raise DownloadError(f"source has {snapshot.bytes} bytes; expected {expected_bytes}")
    if snapshot.sha256 != expected_sha256:
        raise DownloadError(
            f"source SHA-256 {snapshot.sha256} does not match {expected_sha256}"
        )


def capture_source_snapshot(
    path: str | Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> SourceSnapshot:
    """Hash one regular-file descriptor and require its stat identity to remain unchanged."""

    source = Path(path)
    descriptor, before = _open_regular_readonly(source)
    digest = hashlib.sha256()
    with os.fdopen(descriptor, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        after = os.fstat(handle.fileno())
    if _stat_identity(before) != _stat_identity(after):
        raise DownloadError(f"source changed while it was being hashed: {source}")
    snapshot = SourceSnapshot(
        bytes=after.st_size,
        sha256=digest.hexdigest(),
        device=after.st_dev,
        inode=after.st_ino,
        mode=stat.S_IMODE(after.st_mode),
        mtime_ns=after.st_mtime_ns,
        ctime_ns=after.st_ctime_ns,
    )
    _validate_snapshot(
        snapshot, expected_bytes=expected_bytes, expected_sha256=expected_sha256
    )
    return snapshot


def assert_source_unchanged(
    path: str | Path,
    expected: SourceSnapshot,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> None:
    observed = capture_source_snapshot(
        path, expected_bytes=expected_bytes, expected_sha256=expected_sha256
    )
    if observed != expected:
        raise DownloadError(f"source snapshot changed during the audit: {path}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def copy_validated_source(
    source: str | Path,
    destination: str | Path,
    expected: SourceSnapshot,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> SourceSnapshot:
    """Copy the verified cache through one fd into a private, verified audit snapshot."""

    source_path = Path(source)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    source_descriptor, source_before = _open_regular_readonly(source_path)
    if (
        source_before.st_dev != expected.device
        or source_before.st_ino != expected.inode
        or source_before.st_size != expected.bytes
        or source_before.st_mtime_ns != expected.mtime_ns
        or source_before.st_ctime_ns != expected.ctime_ns
    ):
        os.close(source_descriptor)
        raise DownloadError("cache changed before the private snapshot copy began")
    output_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.", suffix=".tmp", dir=destination_path.parent
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with os.fdopen(source_descriptor, "rb") as input_handle, os.fdopen(
            output_descriptor, "wb"
        ) as output_handle:
            for chunk in iter(lambda: input_handle.read(1024 * 1024), b""):
                byte_count += len(chunk)
                if byte_count > expected_bytes:
                    raise DownloadError("source exceeded its expected size during snapshot copy")
                output_handle.write(chunk)
                digest.update(chunk)
            source_after = os.fstat(input_handle.fileno())
            output_handle.flush()
            os.fsync(output_handle.fileno())
        if _stat_identity(source_before) != _stat_identity(source_after):
            raise DownloadError("cache changed while the private snapshot was copied")
        if byte_count != expected_bytes or digest.hexdigest() != expected_sha256:
            raise DownloadError("private source snapshot differs from the verified cache")
        os.chmod(temporary, 0o600)
        os.replace(temporary, destination_path)
        _fsync_directory(destination_path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return capture_source_snapshot(
        destination_path,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
    )


def _validated_headers(
    response,
    *,
    requested_url: str,
    expected_bytes: int,
    expected_etag: str,
    expected_last_modified: str,
    expected_content_type: str,
) -> tuple[int, str, dict[str, str]]:
    status = int(response.status)
    final_url = response.geturl()
    selected = {
        key.lower(): value
        for key, value in response.headers.items()
        if key.lower()
        in {"content-length", "content-type", "etag", "last-modified", "accept-ranges"}
    }
    if status != 200:
        raise DownloadError(f"unexpected HTTP status {status} for {requested_url}")
    if final_url != requested_url:
        raise DownloadError(f"unexpected final URL {final_url!r}; expected {requested_url!r}")
    if "content-length" not in selected:
        raise DownloadError("response is missing Content-Length")
    try:
        declared_bytes = int(selected["content-length"])
    except ValueError as error:
        raise DownloadError("response Content-Length is not an integer") from error
    if declared_bytes != expected_bytes:
        raise DownloadError(
            f"response declares {declared_bytes} bytes; expected {expected_bytes}"
        )
    if selected.get("etag", "").strip('"') != expected_etag.strip('"'):
        raise DownloadError("response ETag differs from the verified source contract")
    if selected.get("last-modified") != expected_last_modified:
        raise DownloadError("response Last-Modified differs from the verified source contract")
    media_type = selected.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type != expected_content_type.lower():
        raise DownloadError("response Content-Type differs from the verified source contract")
    return status, final_url, selected


def download_atomic(
    url: str,
    destination: str | Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    expected_etag: str,
    expected_last_modified: str,
    expected_content_type: str = "text/csv",
    force: bool = False,
    timeout_seconds: int = 120,
) -> DownloadRecord:
    """Validate all response metadata and bytes before replacing a verified cache."""

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = _now()
    if path.exists() and not force:
        snapshot = capture_source_snapshot(
            path, expected_bytes=expected_bytes, expected_sha256=expected_sha256
        )
        return DownloadRecord(
            requested_url=url,
            final_url=None,
            retrieval_mode="existing-local-cache",
            http_status=None,
            response_headers={},
            bytes=snapshot.bytes,
            sha256=snapshot.sha256,
            started_at_utc=started,
            completed_at_utc=_now(),
        )
    if path.is_symlink():
        raise DownloadError(f"destination must not be a symlink: {path}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".part", dir=path.parent
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    byte_count = 0
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "mist-transfer-benchmark-qm9-audit/0.1.1"},
        method="GET",
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                status, final_url, selected_headers = _validated_headers(
                    response,
                    requested_url=url,
                    expected_bytes=expected_bytes,
                    expected_etag=expected_etag,
                    expected_last_modified=expected_last_modified,
                    expected_content_type=expected_content_type,
                )
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    byte_count += len(chunk)
                    if byte_count > expected_bytes:
                        raise DownloadError("response exceeded the hard byte limit")
                    output.write(chunk)
                    digest.update(chunk)
                output.flush()
                os.fsync(output.fileno())
        observed_sha256 = digest.hexdigest()
        if byte_count != expected_bytes:
            raise DownloadError(f"downloaded {byte_count} bytes; expected {expected_bytes}")
        if observed_sha256 != expected_sha256:
            raise DownloadError(
                f"downloaded SHA-256 {observed_sha256} does not match {expected_sha256}"
            )
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        installed = capture_source_snapshot(
            path, expected_bytes=expected_bytes, expected_sha256=expected_sha256
        )
        return DownloadRecord(
            requested_url=url,
            final_url=final_url,
            retrieval_mode="atomic-http-get",
            http_status=status,
            response_headers=selected_headers,
            bytes=installed.bytes,
            sha256=installed.sha256,
            started_at_utc=started,
            completed_at_utc=_now(),
        )
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
