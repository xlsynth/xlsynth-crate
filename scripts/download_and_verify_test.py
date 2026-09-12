#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared CI artifact download validation."""

import io
import hashlib
from email.message import Message
from email.utils import formatdate
from pathlib import Path
import re
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


def test_literal_sha256_accepts_valid_elf(tmp_path, monkeypatch):
    artifact = tmp_path / "slang"
    payload = download_and_verify.ELF_MAGIC + b"pinned-slang"
    expected = hashlib.sha256(payload).hexdigest()

    def fake_download(url, destination, attempts, timeout_seconds):
        del url, attempts, timeout_seconds
        destination.write_bytes(payload)

    monkeypatch.setattr(download_and_verify, "download_with_retry", fake_download)

    download_and_verify.download_and_verify_with_retry(
        "elf",
        "https://example.invalid/slang",
        artifact,
        attempts=1,
        timeout_seconds=1,
        sha256=expected,
    )

    assert artifact.read_bytes() == payload


def test_literal_sha256_rejects_valid_elf_with_wrong_hash(tmp_path, monkeypatch):
    artifact = tmp_path / "slang"
    payload = download_and_verify.ELF_MAGIC + b"wrong-pinned-slang"

    def fake_download(url, destination, attempts, timeout_seconds):
        del url, attempts, timeout_seconds
        destination.write_bytes(payload)

    monkeypatch.setattr(download_and_verify, "download_with_retry", fake_download)

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        download_and_verify.download_and_verify_with_retry(
            "elf",
            "https://example.invalid/slang",
            artifact,
            attempts=1,
            timeout_seconds=1,
            sha256="0" * 64,
        )

    assert not artifact.exists()


def test_malformed_literal_sha256_fails_before_destination_mutation(tmp_path):
    artifact = tmp_path / "missing-parent" / "slang"

    with pytest.raises(ValueError, match="64 hexadecimal"):
        download_and_verify.download_and_verify_with_retry(
            "elf",
            "https://example.invalid/slang",
            artifact,
            attempts=1,
            timeout_seconds=1,
            sha256="not-a-digest",
        )

    assert not artifact.parent.exists()


def test_cli_rejects_malformed_literal_sha256_and_checksum_conflict(monkeypatch):
    common_args = ["download_and_verify.py", "elf", "slang", "https://example.invalid"]
    monkeypatch.setattr(
        download_and_verify.sys,
        "argv",
        common_args + ["--sha256", "not-a-digest"],
    )
    with pytest.raises(SystemExit) as malformed:
        download_and_verify.parse_args()
    assert malformed.value.code == 2

    monkeypatch.setattr(
        download_and_verify.sys,
        "argv",
        common_args
        + [
            "--sha256",
            "0" * 64,
            "--sha256-url",
            "https://example.invalid/slang.sha256",
        ],
    )
    with pytest.raises(SystemExit) as conflict:
        download_and_verify.parse_args()
    assert conflict.value.code == 2


def test_sha256_url_callers_remain_supported(tmp_path, monkeypatch):
    artifact = tmp_path / "slang"
    payload = download_and_verify.ELF_MAGIC + b"url-checksum-slang"
    expected = hashlib.sha256(payload).hexdigest().encode("ascii")

    def fake_download(url, destination, attempts, timeout_seconds):
        del url, attempts, timeout_seconds
        destination.write_bytes(payload)

    monkeypatch.setattr(download_and_verify, "download_with_retry", fake_download)
    monkeypatch.setattr(
        download_and_verify.urllib.request,
        "urlopen",
        lambda request, timeout: FakeResponse(expected),
    )

    download_and_verify.download_and_verify_with_retry(
        "elf",
        "https://example.invalid/slang",
        artifact,
        attempts=1,
        timeout_seconds=1,
        sha256_url="https://example.invalid/slang.sha256",
    )

    assert artifact.read_bytes() == payload


def test_slang_callers_share_asset_id_and_checked_in_digest():
    repo_root = Path(__file__).resolve().parent.parent
    digest_text = (repo_root / "scripts/slang_rocky8.sha256").read_text()
    assert re.fullmatch(r"[0-9a-f]{64}\n", digest_text)

    asset_url = (
        "https://api.github.com/repos/xlsynth/slang-rs/releases/assets/220397578"
    )
    mutable_url = (
        "https://github.com/xlsynth/slang-rs/releases/download/ci/slang-rocky8"
    )
    workflow_text = (repo_root / ".github/workflows/ci.yml").read_text()
    install_text = (repo_root / "docker/install_tools.sh").read_text()
    dockerfile_text = (repo_root / "docker/Dockerfile").read_text()

    assert workflow_text.count(asset_url) == 4
    assert workflow_text.count("scripts/slang_rocky8.sha256") == 4
    assert workflow_text.count('--sha256 "${slang_sha256}"') == 4
    assert asset_url in install_text
    assert "scripts/slang_rocky8.sha256" in install_text
    assert '--sha256 "${slang_sha256}"' in install_text
    assert mutable_url not in workflow_text
    assert mutable_url not in install_text
    assert "scripts/slang_rocky8.sha256 scripts/" in dockerfile_text
    assert dockerfile_text.index("scripts/slang_rocky8.sha256") < dockerfile_text.index(
        "RUN bash docker/install_tools.sh"
    )


def test_github_asset_requests_accept_binary_redirects(monkeypatch):
    monkeypatch.delenv("GH_PAT", raising=False)
    request = download_and_verify.build_request(
        "https://api.github.com/repos/xlsynth/slang-rs/releases/assets/220397578"
    )

    assert request.get_header("Accept") == "application/octet-stream"
    assert request.get_header("Authorization") is None


@pytest.fixture(params=["download_with_retry", "download_and_verify_with_retry"])
def artifact_download(request):
    """Exercises the same retry contract through either public download API."""
    download = getattr(download_and_verify, request.param)

    def run(destination, attempts, **kwargs):
        args = ["https://example.invalid/slang", destination]
        if request.param == "download_and_verify_with_retry":
            args.insert(0, "elf")
        return download(*args, attempts=attempts, timeout_seconds=1, **kwargs)

    return run


@pytest.fixture
def retry_clock(monkeypatch):
    clock = {"now": 1700000000.0, "sleeps": [], "jitter_bounds": []}

    def sleep(delay):
        clock["sleeps"].append(delay)
        clock["now"] += delay

    def uniform(lower, upper):
        clock["jitter_bounds"].append((lower, upper))
        return upper

    monkeypatch.setattr(download_and_verify.time, "time", lambda: clock["now"])
    monkeypatch.setattr(download_and_verify.time, "sleep", sleep)
    monkeypatch.setattr(download_and_verify.random, "uniform", uniform)
    return clock


def http_error(code, headers=None, reason="rate limited", url=None):
    message = Message()
    for name, value in (headers or {}).items():
        message[name] = value
    return urllib.error.HTTPError(
        url or "https://example.invalid/slang", code, reason, message, None
    )


def mock_http_outcomes(monkeypatch, outcomes):
    pending = iter(outcomes)
    requests = []

    def urlopen(request, timeout):
        del timeout
        requests.append(request.full_url)
        outcome = next(pending)
        if isinstance(outcome, Exception):
            raise outcome
        return FakeResponse(outcome)

    monkeypatch.setattr(download_and_verify.urllib.request, "urlopen", urlopen)
    return requests


@pytest.mark.parametrize(
    "code,headers,expected_wait",
    [
        pytest.param(429, {"Retry-After": "7"}, 7.0, id="retry-after-seconds"),
        pytest.param(
            429,
            {"Retry-After": formatdate(1700000010, usegmt=True)},
            10.0,
            id="retry-after-http-date",
        ),
        pytest.param(
            403,
            {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1700000012"},
            12.0,
            id="github-primary-rate-limit",
        ),
        pytest.param(
            429,
            {
                "Retry-After": "7",
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": "1700000012",
            },
            12.0,
            id="reset-later-than-retry-after",
        ),
        pytest.param(
            429,
            {
                "Retry-After": "15",
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": "1700000012",
            },
            15.0,
            id="retry-after-later-than-reset",
        ),
        pytest.param(429, {"Retry-After": "1"}, 2.5, id="jittered-backoff-is-longer"),
        pytest.param(429, {"Retry-After": "0"}, 2.5, id="zero-retry-after"),
        pytest.param(
            429,
            {"Retry-After": formatdate(1699999990, usegmt=True)},
            2.5,
            id="past-http-date",
        ),
        pytest.param(
            403,
            {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1699999990"},
            2.5,
            id="past-reset",
        ),
        pytest.param(
            429, {"Retry-After": "not a date"}, 75.0, id="malformed-retry-after"
        ),
        pytest.param(429, {"Retry-After": "NaN"}, 75.0, id="nan-retry-after"),
        pytest.param(429, {"Retry-After": "inf"}, 75.0, id="infinite-retry-after"),
        pytest.param(429, {"Retry-After": "-1"}, 75.0, id="negative-retry-after"),
        pytest.param(
            403,
            {"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "invalid"},
            75.0,
            id="malformed-reset",
        ),
    ],
)
def test_http_retry_respects_server_deadlines_and_jitter(
    artifact_download, tmp_path, monkeypatch, retry_clock, code, headers, expected_wait
):
    artifact = tmp_path / "slang"
    payload = download_and_verify.ELF_MAGIC + b"rate-limited-slang"
    requests = mock_http_outcomes(monkeypatch, [http_error(code, headers), payload])

    artifact_download(artifact, attempts=2)

    assert retry_clock["sleeps"] == [expected_wait]
    assert len(requests) == 2
    assert artifact.read_bytes() == payload
    assert not list(tmp_path.glob("slang.*.tmp"))


@pytest.mark.parametrize(
    "code,headers,reason,expected_waits",
    [
        (429, {}, "Too Many Requests", [75.0, 150.0]),
        (403, {"X-RateLimit-Remaining": "0"}, "Forbidden", [75.0, 150.0]),
        (403, {}, "API rate limit exceeded", [75.0, 150.0]),
        (403, {}, "Forbidden", [2.5, 5.0]),
    ],
)
def test_http_retry_without_deadline_distinguishes_rate_limits(
    artifact_download,
    tmp_path,
    monkeypatch,
    retry_clock,
    code,
    headers,
    reason,
    expected_waits,
):
    payload = download_and_verify.ELF_MAGIC + b"slang"
    requests = mock_http_outcomes(
        monkeypatch,
        [http_error(code, headers, reason), http_error(code, headers, reason), payload],
    )

    artifact_download(tmp_path / "slang", attempts=3)

    assert retry_clock["sleeps"] == expected_waits
    assert len(requests) == 3


def test_network_retry_doubles_and_caps_base_before_adding_jitter(
    artifact_download, tmp_path, monkeypatch, retry_clock
):
    failures = [urllib.error.URLError("connection reset") for _ in range(8)]
    payload = download_and_verify.ELF_MAGIC + b"slang"
    requests = mock_http_outcomes(monkeypatch, failures + [payload])

    artifact_download(tmp_path / "slang", attempts=9, max_retry_wait_seconds=400)

    assert retry_clock["sleeps"] == [2.5, 5.0, 10.0, 20.0, 40.0, 75.0, 75.0, 75.0]
    assert retry_clock["jitter_bounds"] == [
        (0.0, upper) for upper in [0.5, 1.0, 2.0, 4.0, 8.0, 15.0, 15.0, 15.0]
    ]
    assert len(requests) == 9


def test_default_retry_wait_budget_is_cumulative_and_stops_at_300_seconds(
    artifact_download, tmp_path, monkeypatch, retry_clock
):
    artifact = tmp_path / "slang"
    final_error = http_error(429, {"Retry-After": "1"})
    requests = mock_http_outcomes(
        monkeypatch,
        [
            http_error(429, {"Retry-After": "100"}),
            http_error(429, {"Retry-After": "200"}),
            final_error,
        ],
    )

    with pytest.raises(RuntimeError, match="retry wait budget") as exc_info:
        artifact_download(artifact, attempts=4)

    assert exc_info.value.__cause__ is final_error
    assert retry_clock["sleeps"] == [100.0, 200.0]
    assert len(requests) == 3
    assert not artifact.exists()
    assert not list(tmp_path.glob("slang.*.tmp"))


@pytest.mark.parametrize(
    "retry_after_values,budget,expected_waits",
    [
        (["6"], 5.0, []),
        (["3", "4"], 5.0, [3.0]),
        (["0"], 0.0, []),
    ],
)
def test_custom_retry_wait_budget_never_retries_early(
    artifact_download,
    tmp_path,
    monkeypatch,
    retry_clock,
    retry_after_values,
    budget,
    expected_waits,
):
    artifact = tmp_path / "slang"
    failures = [http_error(429, {"Retry-After": value}) for value in retry_after_values]
    requests = mock_http_outcomes(monkeypatch, failures)

    with pytest.raises(RuntimeError, match="retry wait budget") as exc_info:
        artifact_download(artifact, attempts=5, max_retry_wait_seconds=budget)

    assert exc_info.value.__cause__ is failures[-1]
    assert retry_clock["sleeps"] == expected_waits
    assert len(requests) == len(expected_waits) + 1
    assert not artifact.exists()
    assert not list(tmp_path.glob("slang.*.tmp"))


@pytest.mark.parametrize("budget", [-1, float("nan"), float("inf")])
def test_invalid_retry_wait_budget_fails_before_destination_mutation(
    artifact_download, tmp_path, monkeypatch, budget
):
    artifact = tmp_path / "missing-parent" / "slang"
    requests = mock_http_outcomes(monkeypatch, [])

    with pytest.raises(ValueError, match="finite and nonnegative"):
        artifact_download(artifact, attempts=2, max_retry_wait_seconds=budget)

    assert requests == []
    assert not artifact.parent.exists()


def test_final_attempt_propagates_http_error_without_another_sleep(
    artifact_download, tmp_path, monkeypatch, retry_clock
):
    final_error = http_error(429, {"Retry-After": "10000"})
    requests = mock_http_outcomes(
        monkeypatch, [http_error(429, {"Retry-After": "7"}), final_error]
    )

    with pytest.raises(urllib.error.HTTPError) as exc_info:
        artifact_download(tmp_path / "slang", attempts=2)

    assert exc_info.value is final_error
    assert retry_clock["sleeps"] == [7.0]
    assert len(requests) == 2
    assert not list(tmp_path.iterdir())


def test_checksum_http_error_respects_retry_after_and_removes_unverified_artifact(
    tmp_path, monkeypatch, retry_clock
):
    artifact = tmp_path / "slang"
    url = "https://example.invalid/slang"
    sha256_url = url + ".sha256"
    payload = download_and_verify.ELF_MAGIC + b"checksum-rate-limited-slang"
    digest = hashlib.sha256(payload).hexdigest().encode("ascii")
    requests = []

    def urlopen(request, timeout):
        del timeout
        requests.append(request.full_url)
        if request.full_url == url:
            assert not artifact.exists()
            return FakeResponse(payload)
        assert request.full_url == sha256_url
        assert artifact.read_bytes() == payload
        if len(requests) == 2:
            raise http_error(429, {"Retry-After": "9"}, url=sha256_url)
        return FakeResponse(digest)

    def sleep(delay):
        assert not artifact.exists()
        retry_clock["sleeps"].append(delay)
        retry_clock["now"] += delay

    monkeypatch.setattr(download_and_verify.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(download_and_verify.time, "sleep", sleep)

    download_and_verify.download_and_verify_with_retry(
        "elf", url, artifact, attempts=2, timeout_seconds=1, sha256_url=sha256_url
    )

    assert requests == [url, sha256_url, url, sha256_url]
    assert retry_clock["sleeps"] == [9.0]
    assert artifact.read_bytes() == payload
    assert not list(tmp_path.glob("slang.*.tmp"))


@pytest.mark.parametrize(
    "attempts,expected_exception", [(1, urllib.error.HTTPError), (2, RuntimeError)]
)
def test_checksum_http_failure_cleans_output_and_preserves_error_headers(
    tmp_path, monkeypatch, retry_clock, attempts, expected_exception
):
    artifact = tmp_path / "slang"
    url = "https://example.invalid/slang"
    sha256_url = url + ".sha256"
    failure = http_error(429, {"Retry-After": "301"}, url=sha256_url)
    requests = mock_http_outcomes(
        monkeypatch, [download_and_verify.ELF_MAGIC + b"unverified-slang", failure]
    )

    with pytest.raises(expected_exception) as exc_info:
        download_and_verify.download_and_verify_with_retry(
            "elf",
            url,
            artifact,
            attempts=attempts,
            timeout_seconds=1,
            sha256_url=sha256_url,
        )

    if attempts == 1:
        assert exc_info.value is failure
    else:
        assert exc_info.value.__cause__ is failure
    assert failure.headers["Retry-After"] == "301"
    assert requests == [url, sha256_url]
    assert retry_clock["sleeps"] == []
    assert not artifact.exists()
    assert not list(tmp_path.glob("slang.*.tmp"))
