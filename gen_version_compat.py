# SPDX-License-Identifier: Apache-2.0

"""
Generates version compatibility table in markdown for our documentation.

We do this by scanning repo history of `xlsynth-sys/build.rs` and extracting the
release library version tag that it pulls in.

For each release, the left-hand column is the xlsynth-crate release version (determined
by the git tag on the commit) and the right-hand column is the RELEASE_LIB_VERSION_TAG
value from `xlsynth-sys/build.rs` at that commit.

The table format is:

| xlsynth crate version | xlsynth release version |
|-----------------------|-------------------------|
| X.Y.Z                 | A.B.C                   |

The cells link to the corresponding release tag on GitHub:
  - Left cell: "https://github.com/xlsynth/xlsynth-crate/releases/tag/vX.Y.Z"
  - Right cell: "https://github.com/xlsynth/xlsynth/releases/tag/vA.B.C"

We place this result into `docs/version_metadata.md`.
"""

import re
import os
import subprocess
import sys


def get_git_commits(file_path):
    """Return list of commit hashes that touched file_path."""
    try:
        result = subprocess.run([
            'git', 'log', '--format=%H', '--', file_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        commits = result.stdout.strip().splitlines()
        return commits
    except subprocess.CalledProcessError as e:
        print(f"Error getting git commits for {file_path}: {e.stderr}", file=sys.stderr)
        return []


def get_git_tags_at_commit(commit):
    """Return list of tags that point at the given commit."""
    try:
        result = subprocess.run([
            'git', 'tag', '--points-at', commit
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        tags = result.stdout.strip().splitlines()
        return tags
    except subprocess.CalledProcessError as e:
        print(f"Error getting git tags for commit {commit}: {e.stderr}", file=sys.stderr)
        return []


def get_file_content_at_commit(commit, file_path):
    """Return content of file at a specific commit. Returns None if an error occurs."""
    try:
        result = subprocess.run([
            'git', 'show', f'{commit}:{file_path}'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError:
        return None


def extract_version_from_content(content):
    """
    Extract the first occurrence of a version in the form vX.Y.Z
    from the file content and return the version without the leading 'v'.
    """
    match = re.search(r'v(\d+\.\d+\.\d+)', content)
    if match:
        return match.group(1)  # e.g. returns "0.0.173"
    return None


def generate_markdown_table(mapping):
    """
    Generates a Markdown table from a mapping of:
      xlsynth-crate version  -> library release version

    The table header is as follows:
      | xlsynth crate version | xlsynth release version |
      |-----------------------|-------------------------|

    Each cell is a markdown link, and this function dynamically pads each column based on the maximum cell width.
    """
    crate_base_url = "https://github.com/xlsynth/xlsynth-crate/releases/tag/v{}"
    lib_base_url = "https://github.com/xlsynth/xlsynth/releases/tag/v{}"
    
    # local helper for sorting versions
    def version_key(v):
        return tuple(int(x) for x in v.split('.'))
    
    sorted_crate_versions = sorted(mapping.keys(), key=version_key, reverse=True)
    
    left_header = "xlsynth crate version"
    right_header = "xlsynth release version"
    
    # Prepare rows: each row is a tuple of (left_cell, right_cell)
    rows = []
    for crate_ver in sorted_crate_versions:
        lib_ver = mapping[crate_ver]
        left_cell = f"[{crate_ver}]({crate_base_url.format(crate_ver)})"
        right_cell = f"[{lib_ver}]({lib_base_url.format(lib_ver)})"
        rows.append((left_cell, right_cell))
    
    # Calculate the maximum width for each column (comparing header and each cell's length)
    left_width = max(len(left_header), *(len(row[0]) for row in rows))
    right_width = max(len(right_header), *(len(row[1]) for row in rows))
    
    # Construct the header row with padding
    header_row = f"| {left_header.ljust(left_width)} | {right_header.ljust(right_width)} |"
    # Construct the separator row
    separator_row = f"| {'-' * left_width} | {'-' * right_width} |"
    
    # Construct each data row
    data_rows = [f"| {left.ljust(left_width)} | {right.ljust(right_width)} |" for left, right in rows]
    
    return "\n".join([header_row, separator_row] + data_rows) + "\n"


# New helper function to get the mapping data without rendering

def get_version_mapping():
    file_path = "xlsynth-sys/build.rs"
    commits = get_git_commits(file_path)
    if not commits:
        return {}
    
    mappings = {}
    for commit in commits:
        tags = get_git_tags_at_commit(commit)
        # Consider only tags that match the release format: vX.Y.Z
        release_tags = [tag for tag in tags if re.fullmatch(r'v\d+\.\d+\.\d+', tag)]
        if not release_tags:
            continue
        
        content = get_file_content_at_commit(commit, file_path)
        if not content:
            continue
        
        lib_version = extract_version_from_content(content)
        if not lib_version:
            continue
        
        for tag in release_tags:
            crate_version = tag.lstrip("v")
            if crate_version not in mappings:
                mappings[crate_version] = lib_version
    
    return mappings


def get_last_tag_time():
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


def main():
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


def test_mapping_for_v0_0_101():
    mapping = get_version_mapping()
    assert "0.0.101" in mapping, "Mapping for xlsynth-crate v0.0.101 not found in mapping"
    assert mapping["0.0.101"] == "0.0.173", f"Expected mapping for v0.0.101 to be v0.0.173, got {mapping['0.0.101']}"


if __name__ == "__main__":
    main()