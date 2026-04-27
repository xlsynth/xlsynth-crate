#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Publish crates from publish_order.toml in dependency order."""

import argparse
import json
import os
import re
import subprocess
import sys

from sleep_until_version_seen import check_crate_version


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER_PATH = os.path.join(REPO_ROOT, "publish_order.toml")


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


def publish_crates(version):
    cargo_registry_token = os.environ["CARGO_REGISTRY_TOKEN"]
    packages = load_workspace_packages()
    for crate_name in load_publish_order():
        package = packages[crate_name]
        package_dir = crate_dir(package)
        if check_crate_version(crate_name, version):
            print(
                "{} {} is already on crates.io; skipping.".format(crate_name, version),
                flush=True,
            )
            continue
        print("Publishing {}...".format(crate_name), flush=True)
        subprocess.check_call(
            ["cargo", "publish", "--token", cargo_registry_token],
            cwd=os.path.join(REPO_ROOT, package_dir),
        )
        subprocess.check_call(
            [
                sys.executable,
                os.path.join(REPO_ROOT, "scripts", "sleep_until_version_seen.py"),
                crate_name,
                version,
            ],
            cwd=REPO_ROOT,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    args = parser.parse_args()
    publish_crates(args.version)


if __name__ == "__main__":
    main()
