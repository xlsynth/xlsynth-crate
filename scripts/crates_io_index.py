#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Read crate publication state from the crates.io sparse index."""

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List


SPARSE_INDEX_ROOT = "https://index.crates.io"
USER_AGENT = "xlsynth-crate/release-tools (https://github.com/xlsynth/xlsynth-crate)"


def sparse_index_path(crate_name: str) -> str:
    """Return the sparse-index metadata path for a crates.io crate name."""
    normalized_name = crate_name.lower()
    if len(normalized_name) == 1:
        return "1/{}".format(normalized_name)
    if len(normalized_name) == 2:
        return "2/{}".format(normalized_name)
    if len(normalized_name) == 3:
        return "3/{}/{}".format(normalized_name[0], normalized_name)
    return "{}/{}/{}".format(
        normalized_name[0:2], normalized_name[2:4], normalized_name
    )


def get_published_versions(crate_name: str) -> List[str]:
    """Return all published versions in the crate's sparse-index record.

    A missing record means the crate has never been published. Any other
    lookup or parse failure is surfaced to callers so publication does not
    continue without reliable registry state.
    """
    url = "{}/{}".format(SPARSE_INDEX_ROOT, sparse_index_path(crate_name))
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            index_text = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return []
        raise

    versions: List[str] = []
    for line_number, line in enumerate(index_text.splitlines(), start=1):
        if not line:
            continue
        try:
            entry: Dict[str, Any] = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                "Invalid sparse-index metadata for {} at line {}".format(
                    crate_name, line_number
                )
            ) from error
        version = entry.get("vers")
        if not isinstance(version, str):
            raise ValueError(
                "Missing version in sparse-index metadata for {} at line {}".format(
                    crate_name, line_number
                )
            )
        versions.append(version)
    return versions


def crate_version_is_published(crate_name: str, version_wanted: str) -> bool:
    """Return whether a crate version already exists in the sparse index."""
    # A yanked release still occupies its version and cannot be republished.
    return version_wanted in get_published_versions(crate_name)
