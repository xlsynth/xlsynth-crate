#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import subprocess
from unittest import mock

import pytest

from publish_crates import (
    MAX_PUBLISH_ATTEMPTS,
    PublishFailureKind,
    _classify_publish_failure,
    publish_crate_with_retries,
)


CRATE_NAME = "xlsynth-mcmc-pir"
VERSION = "0.52.0"
PACKAGE_DIR = "xlsynth-mcmc-pir"


def cargo_result(returncode=0, stderr=""):
    return subprocess.CompletedProcess(
        ["cargo", "publish"],
        returncode,
        stdout="",
        stderr=stderr,
    )


def publish_crate():
    publish_crate_with_retries(CRATE_NAME, VERSION, PACKAGE_DIR)


def test_publish_retries_transient_failure():
    failure = cargo_result(101, "failed to get a 200 OK response, got 503")
    with (
        mock.patch(
            "publish_crates.check_crate_version",
            side_effect=[False, False, False],
        ),
        mock.patch(
            "publish_crates._run_cargo_publish", side_effect=[failure, cargo_result()]
        ) as run_publish,
        mock.patch(
            "publish_crates.wait_until_version_seen", side_effect=[False, True]
        ) as wait_until_seen,
        mock.patch("publish_crates.time.sleep") as sleep,
    ):
        publish_crate()

    assert run_publish.call_count == 2
    assert wait_until_seen.call_args_list == [
        mock.call(CRATE_NAME, VERSION),
        mock.call(CRATE_NAME, VERSION),
    ]
    sleep.assert_called_once()


def test_publish_stops_retrying_when_failed_upload_becomes_visible():
    failure = cargo_result(101, "failed to get a 200 OK response, got 503")
    with (
        mock.patch("publish_crates.check_crate_version", side_effect=[False, True]),
        mock.patch(
            "publish_crates._run_cargo_publish", return_value=failure
        ) as run_publish,
        mock.patch("publish_crates.wait_until_version_seen") as wait_until_seen,
    ):
        publish_crate()

    run_publish.assert_called_once_with(PACKAGE_DIR)
    wait_until_seen.assert_not_called()


def test_publish_raises_after_retry_limit():
    failure = cargo_result(101, "failed to get a 200 OK response, got 503")
    with (
        mock.patch(
            "publish_crates.check_crate_version",
            side_effect=[False] * (MAX_PUBLISH_ATTEMPTS * 2),
        ),
        mock.patch(
            "publish_crates._run_cargo_publish", return_value=failure
        ) as run_publish,
        mock.patch(
            "publish_crates.wait_until_version_seen", return_value=False
        ) as wait_until_seen,
        mock.patch("publish_crates.time.sleep") as sleep,
    ):
        with pytest.raises(subprocess.CalledProcessError):
            publish_crate()

    assert run_publish.call_count == MAX_PUBLISH_ATTEMPTS
    assert wait_until_seen.call_count == MAX_PUBLISH_ATTEMPTS
    assert sleep.call_count == MAX_PUBLISH_ATTEMPTS - 1


def test_publish_accepts_final_failed_upload_that_becomes_visible():
    failure = cargo_result(101, "failed to get a 200 OK response, got 503")
    with (
        mock.patch(
            "publish_crates.check_crate_version",
            side_effect=[False] * (MAX_PUBLISH_ATTEMPTS * 2),
        ),
        mock.patch(
            "publish_crates._run_cargo_publish", return_value=failure
        ) as run_publish,
        mock.patch(
            "publish_crates.wait_until_version_seen",
            side_effect=[False, False, True],
        ) as wait_until_seen,
        mock.patch("publish_crates.time.sleep"),
    ):
        publish_crate()

    assert run_publish.call_count == MAX_PUBLISH_ATTEMPTS
    assert wait_until_seen.call_count == MAX_PUBLISH_ATTEMPTS


def test_publish_skips_version_already_on_crates_io():
    with (
        mock.patch("publish_crates.check_crate_version", return_value=True),
        mock.patch("publish_crates._run_cargo_publish") as run_publish,
        mock.patch("publish_crates.wait_until_version_seen") as wait_until_seen,
    ):
        publish_crate()

    run_publish.assert_not_called()
    wait_until_seen.assert_not_called()


def test_publish_fails_when_successful_upload_never_becomes_visible():
    with (
        mock.patch("publish_crates.check_crate_version", return_value=False),
        mock.patch(
            "publish_crates._run_cargo_publish", return_value=cargo_result()
        ) as run_publish,
        mock.patch("publish_crates.wait_until_version_seen", return_value=False),
    ):
        with pytest.raises(RuntimeError):
            publish_crate()

    run_publish.assert_called_once_with(PACKAGE_DIR)


def test_publish_fails_fast_for_fatal_error():
    failure = cargo_result(101, "authentication failed")
    with (
        mock.patch("publish_crates.check_crate_version", side_effect=[False, False]),
        mock.patch("publish_crates._run_cargo_publish", return_value=failure),
        mock.patch("publish_crates.wait_until_version_seen") as wait_until_seen,
        mock.patch("publish_crates.time.sleep") as sleep,
    ):
        with pytest.raises(subprocess.CalledProcessError):
            publish_crate()

    wait_until_seen.assert_not_called()
    sleep.assert_not_called()


def test_publish_accepts_duplicate_failure_that_becomes_visible():
    failure = cargo_result(101, "crate version is already uploaded")
    with (
        mock.patch("publish_crates.check_crate_version", side_effect=[False, False]),
        mock.patch("publish_crates._run_cargo_publish", return_value=failure),
        mock.patch(
            "publish_crates.wait_until_version_seen", return_value=True
        ) as wait_until_seen,
    ):
        publish_crate()

    wait_until_seen.assert_called_once_with(CRATE_NAME, VERSION)


def test_publish_rejects_duplicate_failure_that_remains_absent():
    failure = cargo_result(101, "crate version is already uploaded")
    with (
        mock.patch("publish_crates.check_crate_version", side_effect=[False, False]),
        mock.patch("publish_crates._run_cargo_publish", return_value=failure),
        mock.patch("publish_crates.wait_until_version_seen", return_value=False),
    ):
        with pytest.raises(RuntimeError):
            publish_crate()


@pytest.mark.parametrize(
    ("stderr", "expected"),
    [
        ("failed to get a 200 OK response, got 503", PublishFailureKind.TRANSIENT),
        ("failed to get a 200 OK response, got 429", PublishFailureKind.TRANSIENT),
        ("spurious network error: connection reset", PublishFailureKind.TRANSIENT),
        ("crate version is already uploaded", PublishFailureKind.DUPLICATE),
        ("authentication failed", PublishFailureKind.FATAL),
    ],
)
def test_publish_failure_classification(stderr, expected):
    assert _classify_publish_failure(cargo_result(101, stderr)) == expected
