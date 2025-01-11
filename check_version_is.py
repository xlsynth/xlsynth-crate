#!/usr/bin/env python3

import sys
import re
import os

if len(sys.argv) < 2:
    print("Usage: check_version_is.py <version>")
    sys.exit(1)

# The argument might be e.g. "v0.0.57" or "0.0.57"
arg_version = sys.argv[1]

# If you expect the script to always receive e.g. "v0.0.57",
# you can strip the leading 'v' here:
tag_version = arg_version.lstrip('v')

# Read the version from xlsynth-sys/Cargo.toml
toml_path = os.path.join("xlsynth-sys", "Cargo.toml")
with open(toml_path, "r", encoding="utf-8") as f:
    cargo_toml = f.read()

# Use a regex to extract the version from a line like: version = "0.0.57"
match = re.search(r'^version\s*=\s*"([^"]+)"', cargo_toml, re.MULTILINE)
if not match:
    print("Error: Could not find a valid `version = \"...\"` line in xlsynth-sys/Cargo.toml.")
    sys.exit(1)

cargo_version = match.group(1)

print(f"Tag version:       {tag_version}")
print(f"Cargo.toml version: {cargo_version}")

# Compare the extracted version with the tag version
if cargo_version != tag_version:
    print(
        f"Error: version mismatch. "
        f"Tag is {tag_version}, but xlsynth-sys/Cargo.toml is {cargo_version}."
    )
    sys.exit(1)

print("Success: Tag version matches Cargo.toml version.")