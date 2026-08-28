#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared CI artifact download validation."""

import io
import struct
import urllib.error
import zipfile

import download_and_verify
import pytest


def corrupt_deflate_zip_bytes():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("bin/protoc", b"placeholder executable")

    payload = bytearray(buffer.getvalue())
    name_length, extra_length = struct.unpack_from("<HH", payload, 26)
    compressed_offset = 30 + name_length + extra_length
    # Deflate block type 3 is reserved, so reading this member raises zlib.error.
    payload[compressed_offset] = (payload[compressed_offset] & 0xF8) | 0x07
    return bytes(payload)


def test_validate_artifact_accepts_zip_archive(tmp_path):
    artifact = tmp_path / "protoc.zip"
    with zipfile.ZipFile(str(artifact), "w") as archive:
        archive.writestr("bin/protoc", "placeholder")

    assert download_and_verify.validate_artifact(artifact, "zip") is None


def test_validate_artifact_rejects_non_zip_payload(tmp_path):
    artifact = tmp_path / "protoc.zip"
    artifact.write_text("upstream error page")

    error = download_and_verify.validate_artifact(artifact, "zip")

    assert error is not None
    assert "Expected ZIP archive" in error


def test_zip_download_retries_connection_reset(tmp_path, monkeypatch):
    artifact = tmp_path / "protoc.zip"
    download_attempts = []

    def fake_download(url, destination, attempts, timeout_seconds):
        del url, attempts, timeout_seconds
        download_attempts.append(True)
        if len(download_attempts) == 1:
            raise OSError("connection reset by peer")
        with zipfile.ZipFile(str(destination), "w") as archive:
            archive.writestr("bin/protoc", "placeholder")

    monkeypatch.setattr(
        download_and_verify,
        "download_with_retry",
        fake_download,
    )
    monkeypatch.setattr(download_and_verify.time, "sleep", lambda _delay: None)

    download_and_verify.download_and_verify_with_retry(
        "zip",
        "https://example.invalid/protoc.zip",
        artifact,
        attempts=2,
        timeout_seconds=1,
    )

    assert len(download_attempts) == 2


def test_zip_download_retries_corrupt_deflate_stream(tmp_path, monkeypatch):
    artifact = tmp_path / "protoc.zip"
    download_attempts = []

    def fake_download(url, destination, attempts, timeout_seconds):
        del url, attempts, timeout_seconds
        download_attempts.append(True)
        if len(download_attempts) == 1:
            destination.write_bytes(corrupt_deflate_zip_bytes())
            return
        with zipfile.ZipFile(str(destination), "w") as archive:
            archive.writestr("bin/protoc", "placeholder")

    monkeypatch.setattr(
        download_and_verify,
        "download_with_retry",
        fake_download,
    )
    monkeypatch.setattr(download_and_verify.time, "sleep", lambda _delay: None)

    download_and_verify.download_and_verify_with_retry(
        "zip",
        "https://example.invalid/protoc.zip",
        artifact,
        attempts=2,
        timeout_seconds=1,
    )

    assert len(download_attempts) == 2


class FakeResponse:
    def __init__(self, payload, reset_after_payload=False):
        self.payload = payload
        self.reset_after_payload = reset_after_payload
        self.read_count = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback
        return False

    def read(self, size=-1):
        del size
        if self.read_count == 0:
            self.read_count += 1
            return self.payload
        if self.reset_after_payload:
            raise OSError("OpenSSL SSL_connect: Connection reset by peer")
        return b""


def test_elf_download_retries_tls_reset_cleans_partial_temp_and_rejects_non_elf(
    tmp_path, monkeypatch
):
    artifact = tmp_path / "slang"
    download_attempts = []
    first_temp_path = []

    def fake_urlopen(request, timeout):
        del request, timeout
        download_attempts.append(True)
        if len(download_attempts) == 1:
            first_temp_path.extend(tmp_path.glob("slang.*.tmp"))
            assert len(first_temp_path) == 1
            return FakeResponse(b"\x7fEL", reset_after_payload=True)
        if len(download_attempts) == 2:
            assert not artifact.exists()
            assert not first_temp_path[0].exists()
            assert len(list(tmp_path.glob("slang.*.tmp"))) == 1
            return FakeResponse(b"\x7fEL")
        return FakeResponse(download_and_verify.ELF_MAGIC + b"slang")

    monkeypatch.setattr(
        download_and_verify.urllib.request,
        "urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr(download_and_verify.time, "sleep", lambda _delay: None)

    download_and_verify.download_and_verify_with_retry(
        "elf",
        "https://example.invalid/slang",
        artifact,
        attempts=3,
        timeout_seconds=1,
    )

    assert len(download_attempts) == 3
    assert artifact.read_bytes().startswith(download_and_verify.ELF_MAGIC)


def test_elf_download_stops_after_bounded_tls_reset_retries(tmp_path, monkeypatch):
    artifact = tmp_path / "slang"
    download_attempts = []

    def fake_urlopen(request, timeout):
        del request, timeout
        download_attempts.append(True)
        raise urllib.error.URLError("connection reset by peer")

    monkeypatch.setattr(
        download_and_verify.urllib.request,
        "urlopen",
        fake_urlopen,
    )
    monkeypatch.setattr(download_and_verify.time, "sleep", lambda _delay: None)

    with pytest.raises(urllib.error.URLError):
        download_and_verify.download_and_verify_with_retry(
            "elf",
            "https://example.invalid/slang",
            artifact,
            attempts=3,
            timeout_seconds=1,
        )

    assert len(download_attempts) == 3
    assert not artifact.exists()
    assert not list(tmp_path.glob("slang.*.tmp"))
