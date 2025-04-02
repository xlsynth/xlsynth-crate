#!/usr/bin/env python3
import os
import re
import sys


def bump_version_in_file(file_path, do_bump):
    # Regex to match a version line e.g. version = "1.2.3"
    pattern = re.compile(r'^(version\s*=\s*")([0-9]+)\.([0-9]+)\.([0-9]+)(")', re.MULTILINE)
    with open(file_path, 'r') as f:
        content = f.read()

    match = pattern.search(content)
    if not match:
        print(f"No version found in {file_path}.")
        return (False, None, None)

    old_version = f"{match.group(2)}.{match.group(3)}.{match.group(4)}"
    new_patch = int(match.group(4)) + 1
    new_version = f"{match.group(2)}.{match.group(3)}.{new_patch}"

    def repl(m):
        return f'{m.group(1)}{new_version}{m.group(5)}'

    new_content, count = pattern.subn(repl, content)
    if count > 0:
        if do_bump:
            with open(file_path, 'w') as f:
                f.write(new_content)
            print(f"Updated {file_path}: bumped from {old_version} to {new_version}.")
        else:
            print(f"Would update {file_path}: bump from {old_version} to {new_version}.")
        return (True, old_version, new_version)
    else:
        print(f"No version bumped in {file_path}.")
        return (False, None, None)


def find_cargo_toml_files():
    cargo_files = []
    for root, _, files in os.walk('.'):
        if 'Cargo.toml' in files:
            cargo_files.append(os.path.join(root, 'Cargo.toml'))
    return cargo_files


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("check", "bump"):
        print("Usage: bump_version_numbers.py [check|bump]")
        sys.exit(1)
    command = sys.argv[1]
    do_bump = (command == "bump")
    if do_bump:
        print("Bumping version numbers in Cargo.toml files...")
    else:
        print("Checking version numbers in Cargo.toml files (no changes will be made)...")

    cargo_files = find_cargo_toml_files()
    bumped_info = {}
    for file in cargo_files:
        bumped, old_version, new_version = bump_version_in_file(file, do_bump)
        if bumped:
            bumped_info[file] = (old_version, new_version)
    if bumped_info:
        old_versions = set(info[0] for info in bumped_info.values())
        new_versions = set(info[1] for info in bumped_info.values())
        if len(old_versions) > 1 or len(new_versions) > 1:
            print(f"Inconsistent version bump across files: {bumped_info}")
            sys.exit(1)
        else:
            common_old = old_versions.pop()
            common_new = new_versions.pop()
            if do_bump:
                print(f"All bumped files updated from {common_old} to {common_new}. Files: {list(bumped_info.keys())}")
            else:
                print(f"All files would be updated from {common_old} to {common_new} if bump is performed. Files: {list(bumped_info.keys())}")
    else:
        print("No Cargo.toml files modified.")


if __name__ == "__main__":
    main() 