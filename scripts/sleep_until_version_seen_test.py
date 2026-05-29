#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import urllib.error
from unittest import mock

import pytest

from sleep_until_version_seen import check_crate_version_for_polling


def test_polling_retries_transient_server_error(capsys):
    error = urllib.error.HTTPError("url", 503, "unavailable", {}, None)
    with mock.patch("sleep_until_version_seen.check_crate_version", side_effect=error):
        assert not check_crate_version_for_polling("xlsynth-sys", "0.51.1")

    assert "[Sparse index HTTP error: 503]" in capsys.readouterr().out


def test_polling_retries_transient_request_error(capsys):
    error = urllib.error.URLError("temporarily unavailable")
    with mock.patch("sleep_until_version_seen.check_crate_version", side_effect=error):
        assert not check_crate_version_for_polling("xlsynth-sys", "0.51.1")

    assert "[Sparse index request error:" in capsys.readouterr().out


def test_polling_does_not_retry_policy_error():
    error = urllib.error.HTTPError("url", 403, "forbidden", {}, None)
    with mock.patch("sleep_until_version_seen.check_crate_version", side_effect=error):
        with pytest.raises(urllib.error.HTTPError):
            check_crate_version_for_polling("xlsynth-sys", "0.51.1")
