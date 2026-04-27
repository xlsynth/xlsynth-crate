#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Print '<crate-name> <crate-dir>' lines from publish_order.toml."""

import json
import os
import re
import subprocess


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER_PATH = os.path.join(REPO_ROOT, "publish_order.toml")


def load_publish_order():
    with open(ORDER_PATH, "r", encoding="utf-8") as f:
        contents = f.read()
    match = re.search(r"crates\s*=\s*\[(.*?)\]", contents, re.DOTALL)
    if match is None:
        raise ValueError("publish_order.toml must contain a crates array")
    return re.findall(r'"([^"]+)"', match.group(1))


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


def main():
    packages = load_workspace_packages()
    for crate_name in load_publish_order():
        manifest_path = packages[crate_name]["manifest_path"]
        crate_dir = os.path.dirname(os.path.relpath(manifest_path, REPO_ROOT))
        print("{} {}".format(crate_name, crate_dir))


if __name__ == "__main__":
    main()
