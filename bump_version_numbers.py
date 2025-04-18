#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import os
import re
import sys


def find_cargo_toml_files():
    cargo_files = []
    for root, _, files in os.walk("."):
        if "Cargo.toml" in files:
            cargo_files.append(os.path.join(root, "Cargo.toml"))
    return cargo_files


def gather_workspace_info():
    """Collect package info from all Cargo.toml files in the workspace.
    Returns a dict mapping package names to a tuple of (file_path, old_version, new_version).
    """
    workspace_info = {}
    for file in find_cargo_toml_files():
        with open(file, "r") as f:
            lines = f.readlines()
        in_package = False
        pkg_name = None
        version_line_index = None
        old_version = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "[package]":
                in_package = True
                continue
            if in_package and stripped.startswith("[") and stripped != "[package]":
                break
            if in_package:
                m = re.match(r'\s*name\s*=\s*"([^"]+)"', line)
                if m:
                    pkg_name = m.group(1)
                m = re.match(r'\s*version\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"', line)
                if m:
                    if version_line_index is not None:
                        print(
                            f"Multiple version lines found in [package] section of {file}"
                        )
                        sys.exit(1)
                    version_line_index = i
                    old_version = m.group(1)
        if pkg_name and old_version:
            parts = old_version.split(".")
            if len(parts) != 3:
                print(f"Invalid version format in {file}: {old_version}")
                sys.exit(1)
            try:
                new_patch = int(parts[2]) + 1
            except ValueError:
                print(f"Invalid patch number in version in {file}: {old_version}")
                sys.exit(1)
            new_version = f"{parts[0]}.{parts[1]}.{new_patch}"
            workspace_info[pkg_name] = (file, old_version, new_version)
    return workspace_info


def bump_package_version(file_path, do_bump):
    """Update the [package] section's version in a Cargo.toml file."""
    with open(file_path, "r") as f:
        lines = f.readlines()
    in_package = False
    version_line_index = None
    old_version = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "[package]":
            in_package = True
            continue
        if in_package and stripped.startswith("[") and stripped != "[package]":
            break
        if in_package:
            m = re.match(r'\s*version\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+)"', line)
            if m:
                if version_line_index is not None:
                    print(
                        f"Multiple version lines found in [package] section of {file_path}"
                    )
                    sys.exit(1)
                version_line_index = i
                old_version = m.group(1)
    if version_line_index is None:
        print(f"No version found in [package] section of {file_path}.")
        return (False, None, None)
    parts = old_version.split(".")
    new_patch = int(parts[2]) + 1
    new_version = f"{parts[0]}.{parts[1]}.{new_patch}"
    new_line = lines[version_line_index].replace(old_version, new_version, 1)
    if do_bump:
        lines[version_line_index] = new_line
        with open(file_path, "w") as f:
            f.writelines(lines)
        print(
            f"Updated {file_path}: bumped [package] version from {old_version} to {new_version}."
        )
    else:
        print(
            f"Would update {file_path}: bump [package] version from {old_version} to {new_version}."
        )
    return (True, old_version, new_version)


def update_dependency_versions(file_path, workspace_info, do_bump):
    """Update dependency version numbers in dependency declarations for local packages.

    For each dependency in the file that has a local path and whose package name is in workspace_info,
    ensure its version matches the expected version. In bump mode, update the version; in check mode, error if mismatch.
    """
    with open(file_path, "r") as f:
        content = f.read()
    updated = False
    # For each package in workspace_info, update dependency declarations
    for pkg, (_, old_version, new_version) in workspace_info.items():
        expected_version = new_version if do_bump else old_version
        # Regex to match a dependency declaration for the package in a single line
        # Example: xlsynth = { path = "../xlsynth", version = "0.0.107" }
        pattern = re.compile(
            r"(^\s*"
            + re.escape(pkg)
            + r"\s*=\s*\{[^}]*version\s*=\s*\")([0-9]+\.[0-9]+\.[0-9]+)(\")",
            re.MULTILINE,
        )

        def dep_repl(m):
            current_version = m.group(2)
            if current_version != expected_version:
                if do_bump:
                    nonlocal updated
                    updated = True
                    return f"{m.group(1)}{expected_version}{m.group(3)}"
                else:
                    print(
                        f"In file {file_path}, dependency {pkg} has version {current_version} but expected {expected_version}"
                    )
                    sys.exit(1)
            else:
                return m.group(0)

        content, sub_count = pattern.subn(dep_repl, content)
        if sub_count > 0:
            updated = True
    if updated and do_bump:
        with open(file_path, "w") as f:
            f.write(content)
        print(f"Updated dependency versions in {file_path}.")
    elif updated and not do_bump:
        print(f"Dependencies in {file_path} would be updated.")
    return updated


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("check", "bump"):
        print("Usage: bump_version_numbers.py [check|bump]")
        sys.exit(1)
    do_bump = sys.argv[1] == "bump"
    mode = "Bumping" if do_bump else "Checking"
    print(f"{mode} package versions in Cargo.toml files...")
    cargo_files = find_cargo_toml_files()
    workspace_info = gather_workspace_info()
    if not workspace_info:
        print("No workspace package info found.")
        sys.exit(1)
    bumped_info = {}
    # First update package versions
    for file in cargo_files:
        bumped, old_version, new_version = bump_package_version(file, do_bump)
        if bumped:
            bumped_info[file] = (old_version, new_version)
    # Then update dependency versions
    for file in cargo_files:
        update_dependency_versions(file, workspace_info, do_bump)
    if bumped_info:
        old_versions = {info[0] for info in bumped_info.values()}
        new_versions = {info[1] for info in bumped_info.values()}
        if len(old_versions) > 1 or len(new_versions) > 1:
            print(f"Inconsistent package version bump across files: {bumped_info}")
            sys.exit(1)
        else:
            common_old = old_versions.pop()
            common_new = new_versions.pop()
            if do_bump:
                print(
                    f"All packages bumped from {common_old} to {common_new}. Files: {list(bumped_info.keys())}"
                )
            else:
                print(
                    f"All packages would be updated from {common_old} to {common_new} if bump is performed. Files: {list(bumped_info.keys())}"
                )
    else:
        print("No Cargo.toml files modified.")


if __name__ == "__main__":
    main()
