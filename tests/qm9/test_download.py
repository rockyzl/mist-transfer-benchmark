from __future__ import annotations

import hashlib
import io

import pytest

from mist_transfer_benchmark.qm9.download import (
    DownloadError,
    assert_source_unchanged,
    capture_source_snapshot,
    download_atomic,
)


class _Response(io.BytesIO):
    status = 200

    def __init__(
        self,
        content: bytes,
        *,
        declared_bytes: int | None = None,
        final_url: str = "https://example.test/qm9.csv",
        include_length: bool = True,
    ):
        super().__init__(content)
        self._final_url = final_url
        self.headers = {
            "ETag": '"fixture-etag"',
            "Last-Modified": "fixture-date",
            "Content-Type": "text/csv",
        }
        if include_length:
            self.headers["Content-Length"] = str(
                len(content) if declared_bytes is None else declared_bytes
            )

    def geturl(self):
        return self._final_url


def _download(destination, content: bytes, *, force=False):
    return download_atomic(
        "https://example.test/qm9.csv",
        destination,
        expected_bytes=len(content),
        expected_sha256=hashlib.sha256(content).hexdigest(),
        expected_etag='"fixture-etag"',
        expected_last_modified="fixture-date",
        force=force,
    )


def test_atomic_download_authenticates_before_install(tmp_path, monkeypatch):
    content = b"header\nvalue\n"
    monkeypatch.setattr(
        "mist_transfer_benchmark.qm9.download.urllib.request.urlopen",
        lambda *_args, **_kwargs: _Response(content),
    )
    destination = tmp_path / "private" / "qm9.csv"

    record = _download(destination, content)

    assert destination.read_bytes() == content
    assert record.http_status == 200
    assert record.sha256 == hashlib.sha256(content).hexdigest()
    assert record.response_headers["etag"] == '"fixture-etag"'
    assert not list(destination.parent.glob(".*.part"))


@pytest.mark.parametrize(
    ("response", "match"),
    [
        (_Response(b"evil", declared_bytes=4), "SHA-256"),
        (_Response(b"evil!", declared_bytes=4), "hard byte limit"),
        (_Response(b"evil", include_length=False), "Content-Length"),
        (_Response(b"evil", final_url="https://evil.test/qm9.csv"), "final URL"),
    ],
)
def test_bad_forced_response_never_replaces_verified_cache(
    tmp_path, monkeypatch, response, match
):
    destination = tmp_path / "qm9.csv"
    good = b"good"
    destination.write_bytes(good)
    monkeypatch.setattr(
        "mist_transfer_benchmark.qm9.download.urllib.request.urlopen",
        lambda *_args, **_kwargs: response,
    )

    with pytest.raises(DownloadError, match=match):
        _download(destination, good, force=True)

    assert destination.read_bytes() == good


def test_source_mutation_is_a_hard_stop(tmp_path):
    path = tmp_path / "qm9.csv"
    original = b"abcd"
    path.write_bytes(original)
    snapshot = capture_source_snapshot(
        path,
        expected_bytes=len(original),
        expected_sha256=hashlib.sha256(original).hexdigest(),
    )
    path.write_bytes(b"wxyz")

    with pytest.raises(DownloadError):
        assert_source_unchanged(
            path,
            snapshot,
            expected_bytes=len(original),
            expected_sha256=hashlib.sha256(original).hexdigest(),
        )


def test_symlink_source_is_rejected(tmp_path):
    target = tmp_path / "target.csv"
    target.write_bytes(b"good")
    link = tmp_path / "qm9.csv"
    link.symlink_to(target)
    with pytest.raises(DownloadError, match="symlink"):
        capture_source_snapshot(
            link,
            expected_bytes=4,
            expected_sha256=hashlib.sha256(b"good").hexdigest(),
        )
