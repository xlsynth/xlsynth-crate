#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import sys
import os

if len(sys.argv) < 2:
    print("Usage: check_version_is.py <version>")
    sys.exit(1)

# The argument might be e.g. "v0.0.57" or "0.0.57"
arg_version = sys.argv[1]
print(f"Argument version: {arg_version!r}")

# If you expect the script to always receive e.g. "v0.0.57",
# you can strip the leading 'v' here:
tag_version = arg_version.lstrip("v")
print(f"Tag version:       {tag_version!r}")


def find_cargo_toml_files():
    cargo_files = []
    for root, _, files in os.walk("."):
        if "Cargo.toml" in files:
            if "target" in root.split(os.sep):  # Exclude target directory
                continue
            # Exclude Cargo.toml in fuzz directories for now, as they might not have a package section or version
            if "fuzz" in root.split(os.sep) and "Cargo.toml" in files:
                # A bit coarse, but good enough for now.
                if os.path.join(root, "Cargo.toml").count("fuzz") > 0:
                    print(
                        f"Skipping fuzzer Cargo.toml: {os.path.join(root, 'Cargo.toml')}"
                    )
                    continue
            cargo_files.append(os.path.join(root, "Cargo.toml"))
    return cargo_files


all_versions_match = True
cargo_files_to_check = find_cargo_toml_files()

if not cargo_files_to_check:
    print("Error: No Cargo.toml files found to check.")
    sys.exit(1)

print(f"Checking Cargo.toml files: {cargo_files_to_check}")

for toml_path in cargo_files_to_check:
    with open(toml_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    in_package_section = False
    found_version_in_file = False
    # print(f"Processing {toml_path}...") # Debug: Indicate file processing start
    for line_num, line in enumerate(lines):
        stripped_line = line.strip()
        # print(f"  Line {line_num+1}: '{stripped_line}'") # Debug: Show each stripped line

        if stripped_line == "[package]":
            in_package_section = True
            # print(f"    Found [package] section at line {line_num+1}") # Debug
            continue

        if in_package_section:
            # print(f"    In package section, evaluating: '{stripped_line}'") # Debug
            if stripped_line.startswith("["):
                if not found_version_in_file:
                    print(
                        f"Warning: No 'version =' line found in [package] section of {toml_path} before next section '{stripped_line}'. File will be skipped."
                    )
                break

            # Attempt to parse version line without regex as a fallback/primary
            if stripped_line.startswith("version") and "=" in stripped_line:
                try:
                    # Basic parsing: version = "x.y.z"
                    parts = stripped_line.split("=", 1)  # Split on the first equals
                    value_part = parts[1].strip()  # Get the part after '='
                    if value_part.startswith('"') and value_part.endswith('"'):
                        cargo_version = value_part[1:-1]  # Remove quotes
                        print(
                            f"Found version '{cargo_version}' in {toml_path}"
                        )  # Keep this informative print
                        found_version_in_file = True
                        if cargo_version != tag_version:
                            print(
                                f"Error: version mismatch in {toml_path}. "
                                f"Tag is {tag_version!r}, but {toml_path} is {cargo_version!r}."
                            )
                            all_versions_match = False
                        break  # Found version, exit loop for this file
                    else:
                        # print(f"      Line '{stripped_line}' looked like version but quote format was wrong.") # Remove debug
                        pass  # No need to print if format is wrong, just won't match

                except Exception:
                    # print(f"      Error parsing line '{stripped_line}' as version: {e}") # Remove debug
                    pass  # If parsing fails, it's not the version line we are looking for
            else:
                # print(f"      Line '{stripped_line}' did not start with 'version' or contain '='.")
                pass  # Ensure block is not empty if print is commented

    if not found_version_in_file and in_package_section:
        print(
            f"Warning: Reached end of {toml_path} or [package] section without finding a 'version =' line."
        )
    elif not in_package_section and not found_version_in_file:
        is_workspace_root_toml = False
        # Efficiently check for [workspace] without reading the whole file again if possible
        # For simplicity here, just re-opening. Could optimize if this script was slow.
        temp_content = "".join(lines)  # Reconstruct content from lines read
        if "[workspace]" in temp_content and "[package]" not in temp_content:
            print(
                f"Info: {toml_path} appears to be a workspace root Cargo.toml (contains [workspace] but no [package]). Skipping version check for it."
            )
        elif "[package]" not in temp_content:
            print(
                f"Warning: No [package] section found in {toml_path}. This might be a virtual manifest. Skipping."
            )

if not all_versions_match:
    print("Version check failed for one or more crates.")
    sys.exit(1)

print(f"Success: Tag version {tag_version!r} matches all checked Cargo.toml versions.")
