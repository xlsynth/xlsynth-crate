#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import urllib.error
from unittest import mock

import pytest

from crates_io_index import USER_AGENT, crate_version_is_published, sparse_index_path


class FakeResponse:
    def __init__(self, body: str):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self) -> bytes:
        return self.body.encode("utf-8")


def test_sparse_index_path_uses_cargo_index_layout():
    assert sparse_index_path("a") == "1/a"
    assert sparse_index_path("ab") == "2/ab"
    assert sparse_index_path("abc") == "3/a/abc"
    assert sparse_index_path("xlsynth-sys") == "xl/sy/xlsynth-sys"
    assert sparse_index_path("XLSynth_Aot") == "xl/sy/xlsynth_aot"


def test_published_version_is_found_through_sparse_index():
    response = FakeResponse(
        '{"name":"xlsynth-sys","vers":"0.51.0","yanked":false}\n'
        '{"name":"xlsynth-sys","vers":"0.51.1","yanked":true}\n'
    )
    with mock.patch(
        "crates_io_index.urllib.request.urlopen", return_value=response
    ) as open_url:
        assert crate_version_is_published("xlsynth-sys", "0.51.1")

    request = open_url.call_args.args[0]
    assert request.full_url == "https://index.crates.io/xl/sy/xlsynth-sys"
    assert request.get_header("User-agent") == USER_AGENT
    assert open_url.call_args.kwargs["timeout"] == 15


def test_missing_crate_is_not_published():
    error = urllib.error.HTTPError("url", 404, "not found", {}, None)
    with mock.patch("crates_io_index.urllib.request.urlopen", side_effect=error):
        assert not crate_version_is_published("first-release", "0.1.0")


def test_registry_access_failure_is_not_treated_as_absence():
    error = urllib.error.HTTPError("url", 403, "forbidden", {}, None)
    with mock.patch("crates_io_index.urllib.request.urlopen", side_effect=error):
        with pytest.raises(urllib.error.HTTPError):
            crate_version_is_published("xlsynth-sys", "0.51.1")


def test_malformed_sparse_index_metadata_is_rejected():
    with mock.patch(
        "crates_io_index.urllib.request.urlopen",
        return_value=FakeResponse('{"name":"xlsynth-sys"}\n'),
    ):
        with pytest.raises(ValueError):
            crate_version_is_published("xlsynth-sys", "0.51.1")
