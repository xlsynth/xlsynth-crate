#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared CI artifact download validation."""

import io
import struct
import zipfile

import download_and_verify_shared_lib


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

    assert download_and_verify_shared_lib.validate_artifact(artifact, "zip") is None


def test_validate_artifact_rejects_non_zip_payload(tmp_path):
    artifact = tmp_path / "protoc.zip"
    artifact.write_text("upstream error page")

    error = download_and_verify_shared_lib.validate_artifact(artifact, "zip")

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
        download_and_verify_shared_lib,
        "download_with_retry",
        fake_download,
    )
    monkeypatch.setattr(
        download_and_verify_shared_lib.time, "sleep", lambda _delay: None
    )

    download_and_verify_shared_lib.download_and_verify_with_retry(
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
        download_and_verify_shared_lib,
        "download_with_retry",
        fake_download,
    )
    monkeypatch.setattr(
        download_and_verify_shared_lib.time, "sleep", lambda _delay: None
    )

    download_and_verify_shared_lib.download_and_verify_with_retry(
        "zip",
        "https://example.invalid/protoc.zip",
        artifact,
        attempts=2,
        timeout_seconds=1,
    )

    assert len(download_attempts) == 2
