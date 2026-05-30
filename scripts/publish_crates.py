#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Publish crates from publish_order.toml in dependency order."""

import argparse
import enum
import json
import os
import re
import subprocess
import sys
import time

from sleep_until_version_seen import (
    check_crate_version,
    check_crate_version_for_polling,
    wait_until_version_seen,
)


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER_PATH = os.path.join(REPO_ROOT, "publish_order.toml")
MAX_PUBLISH_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 15
_TRANSIENT_ERROR_PATTERNS = (
    r"\b(?:got|status(?: code)?)\s+(?:429|5\d\d)\b",
    r"\b(?:timed out|timeout|connection refused|connection reset|failed to connect|network failure|spurious network error)\b",
)
_DUPLICATE_ERROR_PATTERNS = (
    r"\balready (?:exists|uploaded|published)\b",
    r"\bversion .* is already uploaded\b",
)


class PublishFailureKind(enum.Enum):
    TRANSIENT = "transient"
    DUPLICATE = "duplicate"
    FATAL = "fatal"


def load_publish_order():
    with open(ORDER_PATH, "r", encoding="utf-8") as f:
        contents = f.read()
    match = re.search(r"crates\s*=\s*\[(.*?)\]", contents, re.DOTALL)
    if match is None:
        raise ValueError("publish_order.toml must contain a crates array")
    crates = re.findall(r'"([^"]+)"', match.group(1))
    if not crates:
        raise ValueError("publish_order.toml must contain at least one crate")
    return crates


def load_workspace_packages():
    output = subprocess.check_output(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=REPO_ROOT,
        universal_newlines=True,
    )
    metadata = json.loads(output)
    workspace_member_ids = set(metadata["workspace_members"])
    return {
        package["name"]: package
        for package in metadata["packages"]
        if package["id"] in workspace_member_ids
    }


def crate_dir(package):
    manifest_path = package["manifest_path"]
    return os.path.dirname(os.path.relpath(manifest_path, REPO_ROOT))


def _run_cargo_publish(package_dir):
    result = subprocess.run(
        ["cargo", "publish"],
        cwd=os.path.join(REPO_ROOT, package_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)
    return result


def _classify_publish_failure(result):
    """Classify cargo's unstructured publish output at the subprocess boundary."""
    output = "{}\n{}".format(result.stdout, result.stderr)
    if any(
        re.search(pattern, output, re.IGNORECASE)
        for pattern in _DUPLICATE_ERROR_PATTERNS
    ):
        return PublishFailureKind.DUPLICATE
    if any(
        re.search(pattern, output, re.IGNORECASE)
        for pattern in _TRANSIENT_ERROR_PATTERNS
    ):
        return PublishFailureKind.TRANSIENT
    return PublishFailureKind.FATAL


def _publish_error(result):
    return subprocess.CalledProcessError(
        result.returncode,
        ["cargo", "publish"],
        output=result.stdout,
        stderr=result.stderr,
    )


def publish_crate_with_retries(crate_name, version, package_dir):
    """Publish one crate, retrying failed uploads that remain absent from crates.io."""
    for attempt in range(1, MAX_PUBLISH_ATTEMPTS + 1):
        if check_crate_version(crate_name, version):
            print(
                "{} {} is already on crates.io; skipping.".format(crate_name, version),
                flush=True,
            )
            return
        print(
            "Publishing {} (attempt {}/{})...".format(
                crate_name, attempt, MAX_PUBLISH_ATTEMPTS
            ),
            flush=True,
        )
        result = _run_cargo_publish(package_dir)
        if result.returncode == 0:
            if not wait_until_version_seen(crate_name, version):
                raise RuntimeError(
                    "{} {} did not appear in the crates.io sparse index".format(
                        crate_name, version
                    )
                )
            return

        failure_kind = _classify_publish_failure(result)
        if check_crate_version_for_polling(crate_name, version):
            return
        if failure_kind in (PublishFailureKind.TRANSIENT, PublishFailureKind.DUPLICATE):
            if wait_until_version_seen(crate_name, version):
                return
        if failure_kind == PublishFailureKind.TRANSIENT:
            if attempt == MAX_PUBLISH_ATTEMPTS:
                raise _publish_error(result)
            print(
                "cargo publish failed for {}; retrying.".format(crate_name),
                flush=True,
            )
            time.sleep(RETRY_BACKOFF_SECONDS)
        elif failure_kind == PublishFailureKind.DUPLICATE:
            raise RuntimeError(
                "cargo publish reported that {} {} already exists, but the version did not appear in the crates.io sparse index".format(
                    crate_name, version
                )
            )
        else:
            raise _publish_error(result)


def publish_crates(version):
    # Validate configuration early. Cargo reads the token from the environment,
    # which avoids exposing it in the process argument list and exceptions.
    os.environ["CARGO_REGISTRY_TOKEN"]
    packages = load_workspace_packages()
    for crate_name in load_publish_order():
        package = packages[crate_name]
        package_dir = crate_dir(package)
        publish_crate_with_retries(crate_name, version, package_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    args = parser.parse_args()
    publish_crates(args.version)


if __name__ == "__main__":
    main()
