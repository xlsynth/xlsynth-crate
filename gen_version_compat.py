# SPDX-License-Identifier: Apache-2.0

"""
Generates version compatibility table in markdown for our documentation.

We do this by scanning the repository history of `xlsynth-sys/build.rs` and extracting the
release library version tag that it pulls in.

For each release, the left-hand column is the xlsynth‑crate release version (determined by the git tag)
and the right-hand column is the RELEASE_LIB_VERSION_TAG value from `xlsynth-sys/build.rs` at that tag.

The table format is:

| xlsynth crate version | xlsynth release version |
|-----------------------|-------------------------|
| X.Y.Z                 | A.B.C                   |

The cells link to the corresponding release:
  - Left cell: "https://crates.io/crates/xlsynth/{version}"
  - Right cell: "https://github.com/xlsynth/xlsynth/releases/tag/v{version}"

We place this result into `docs/version_metadata.md`.
"""

import re
import os
import subprocess
import sys
import urllib.request
import json
from dataclasses import dataclass
from typing import Optional, Dict, List


@dataclass
class VersionMapping:
    crate_version: str
    lib_version: str


def get_file_content_at_commit(commit: str, file_path: str) -> Optional[str]:
    """Return content of file at a specific commit. Returns None if an error occurs."""
    try:
        result = subprocess.run([
            'git', 'show', f'{commit}:{file_path}'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError:
        return None


def extract_version_from_content(content: str) -> Optional[str]:
    """
    Extract the first occurrence of a version in the form vX.Y.Z
    from the file content and return the version without the leading 'v'.
    """
    match = re.search(r'v(\d+\.\d+\.\d+)', content)
    if match:
        return match.group(1)
    return None


def generate_markdown_table(mapping: Dict[str, str]) -> str:
    """
    Generates a Markdown table from a mapping of:
      xlsynth‑crate version -> library release version

    Each cell is a markdown link, and this function dynamically pads each column
    based on the maximum cell width.
    """
    # Link to the xlsynth crate on crates.io for the left column.
    crate_base_url = "https://crates.io/crates/xlsynth/{}"
    # Right column still links to the GitHub release for the lib.
    lib_base_url = "https://github.com/xlsynth/xlsynth/releases/tag/v{}"
    
    def version_key(v: str):
        return tuple(int(x) for x in v.split('.'))
    
    sorted_crate_versions = sorted(mapping.keys(), key=version_key, reverse=True)
    
    left_header = "xlsynth crate version"
    right_header = "xlsynth release version"
    
    rows = []
    for crate_ver in sorted_crate_versions:
        lib_ver = mapping[crate_ver]
        left_cell = f"[{crate_ver}]({crate_base_url.format(crate_ver)})"
        right_cell = f"[{lib_ver}]({lib_base_url.format(lib_ver)})"
        rows.append((left_cell, right_cell))
    
    left_width = max(len(left_header), *(len(row[0]) for row in rows))
    right_width = max(len(right_header), *(len(row[1]) for row in rows))
    
    header_row = f"| {left_header.ljust(left_width)} | {right_header.ljust(right_width)} |"
    separator_row = f"| {'-' * left_width} | {'-' * right_width} |"
    data_rows = [f"| {left.ljust(left_width)} | {right.ljust(right_width)} |" for left, right in rows]
    
    return "\n".join([header_row, separator_row] + data_rows) + "\n"


def get_all_tags() -> List[str]:
    """Return a list of tags matching the pattern vX.Y.Z."""
    try:
        result = subprocess.run([
            'git', 'tag', '--list', 'v[0-9]*.[0-9]*.[0-9]*'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        tags = result.stdout.strip().splitlines()
        return tags
    except subprocess.CalledProcessError as e:
        print(f"Error listing tags: {e.stderr}", file=sys.stderr)
        return []


def crate_published(crate_version: str) -> bool:
    """Check if the given crate version is published on crates.io for xlsynth."""
    url = "https://crates.io/api/v1/crates/xlsynth/versions"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = resp.read().decode("utf-8")
            jdata = json.loads(data)
            versions = [v["num"] for v in jdata.get("versions", [])]
            return crate_version in versions
    except Exception as e:
        print(f"Error checking crates.io for version {crate_version}: {e}", file=sys.stderr)
        return False


def get_version_mapping() -> Dict[str, str]:
    file_path = "xlsynth-sys/build.rs"
    all_tags = get_all_tags()
    print(f"Found {len(all_tags)} tags. Processing...", flush=True)
    mappings: Dict[str, str] = {}
    skipped_unpublished = 0
    for tag in all_tags:
        print(f"Processing tag {tag}...", flush=True)
        content = get_file_content_at_commit(tag, file_path)
        if not content:
            print(f"  Skipped tag {tag}: no file content found.", flush=True)
            continue
        lib_version = extract_version_from_content(content)
        if not lib_version:
            print(f"  Skipped tag {tag}: no lib version extracted.", flush=True)
            continue
        crate_version = tag.lstrip("v")
        if not crate_published(crate_version):
            print(f"  Skipped tag {tag}: crate version {crate_version} not published on crates.io.", flush=True)
            skipped_unpublished += 1
            continue
        mappings[crate_version] = lib_version
        print(f"  Mapped crate version {crate_version} -> lib version {lib_version}", flush=True)
    print(f"Skipped {skipped_unpublished} tag(s) due to unpublished crate versions.", flush=True)
    return mappings


def get_last_tag_time() -> str:
    """Return the creatordate (ISO8601) of the most recent tag matching v* or 'Unknown' if not available."""
    try:
        result = subprocess.run([
            'git', 'for-each-ref', '--sort=-creatordate', '--format=%(creatordate:iso8601)', 'refs/tags/v*'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        tags = result.stdout.strip().splitlines()
        if tags:
            return tags[0]
        else:
            return "Unknown"
    except subprocess.CalledProcessError:
        return "Unknown"


def get_single_version_mapping(crate_version: str) -> Optional[VersionMapping]:
    """
    Return a VersionMapping for a single crate version by processing tag v{crate_version}.
    Returns a VersionMapping instance if successful, otherwise None.
    """
    file_path = "xlsynth-sys/build.rs"
    tag = f"v{crate_version}"
    print(f"Processing tag {tag} (single)...", flush=True)
    content = get_file_content_at_commit(tag, file_path)
    if not content:
        print(f"  Skipped tag {tag}: no file content found.", flush=True)
        return None
    lib_version = extract_version_from_content(content)
    if not lib_version:
        print(f"  Skipped tag {tag}: no lib version extracted.", flush=True)
        return None
    if not crate_published(crate_version):
        print(f"  Skipped tag {tag}: crate version {crate_version} not published on crates.io.", flush=True)
        return None
    return VersionMapping(crate_version=crate_version, lib_version=lib_version)


def main() -> None:
    mappings = get_version_mapping()
    if not mappings:
        print("No release tag mappings found in commit history", file=sys.stderr)
        sys.exit(1)
    table = generate_markdown_table(mappings)
    last_tag_time = get_last_tag_time()
    full_content = f"# Version Map\n\nUpdated for tags as of: {last_tag_time}\n\n{table}"
    output_path = os.path.join("docs", "version_metadata.md")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(full_content)
    print(f"Version compatibility table written to {output_path}")


def test_mapping_for_v0_0_101() -> None:
    mapping = get_single_version_mapping("0.0.101")
    assert mapping is not None, "Mapping for xlsynth-crate v0.0.101 not found"
    assert mapping.crate_version == "0.0.101", f"Expected crate version '0.0.101', got {mapping.crate_version}"
    assert mapping.lib_version == "0.0.173", f"Expected lib version '0.0.173', got {mapping.lib_version}"


def test_mapping_for_v0_0_100() -> None:
    mapping = get_single_version_mapping("0.0.100")
    assert mapping is not None, "Mapping for xlsynth-crate v0.0.100 not found"
    assert mapping.crate_version == "0.0.100", f"Expected crate version '0.0.100', got {mapping.crate_version}"
    assert mapping.lib_version == "0.0.173", f"Expected lib version '0.0.173', got {mapping.lib_version}"


if __name__ == "__main__":
    main()