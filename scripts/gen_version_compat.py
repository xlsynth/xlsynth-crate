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

Additionally, we emit a machine-readable mapping at repo root:
`generated_version_compat.json`, which is an object keyed by crate version
with values that contain the corresponding xlsynth release version.
"""

import argparse
import re
import os
import subprocess
import sys
import json
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, List, Dict, FrozenSet, Tuple

from datetime import datetime, timezone, tzinfo

from crates_io_index import get_published_versions


PUBLISH_ORDER_PATH = "publish_order.toml"

# Prefer stdlib zoneinfo (Python ≥3.9) for accurate TZ handling; fall back to a fixed offset if unavailable.
try:
    from zoneinfo import ZoneInfo  # type: ignore
except ImportError:  # pragma: no cover –<3.9 fallback
    ZoneInfo = None  # type: ignore


@dataclass
class VersionMapping:
    crate_version: str
    lib_version: str
    # Human-readable commit/tag datetime in America/Los_Angeles.
    crate_release_datetime: str


def get_file_content_at_commit(commit: str, file_path: str) -> Optional[str]:
    """Return content of file at a specific commit. Returns None if an error occurs."""
    try:
        result = subprocess.run(
            ["git", "show", f"{commit}:{file_path}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return None


def extract_version_from_content(content: str) -> Optional[str]:
    """
    Extract the first occurrence of a version in the form vX.Y.Z
    from the file content and return the version without the leading 'v'.
    """
    match = re.search(r"v(\d+\.\d+\.\d+)", content)
    if match:
        return match.group(1)
    return None


# -- Git helpers


def _to_los_angeles(iso_dt: str) -> str:
    """Convert an ISO-8601 datetime string to America/Los_Angeles formatted output."""
    # Git may emit a 'Z' suffix for UTC – replace with +00:00 for fromisoformat.
    iso_dt = iso_dt.replace("Z", "+00:00")
    dt = datetime.fromisoformat(iso_dt)

    la_tz: tzinfo
    if ZoneInfo is not None:
        la_tz = ZoneInfo("America/Los_Angeles")
    else:
        # Fallback: fixed ‑08:00 offset (DST ignored). Better than nothing.
        from datetime import timedelta

        la_tz = timezone(timedelta(hours=-8))

    return dt.astimezone(la_tz).strftime("%Y-%m-%d %H:%M:%S %Z")


def get_tag_datetime(tag: str) -> Optional[str]:
    """Return the committer date of a tag as a Los_Angeles-formatted string."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%cI", tag],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        iso_dt = result.stdout.strip()
        if not iso_dt:
            return None
        return _to_los_angeles(iso_dt)
    except subprocess.CalledProcessError:
        return None


def generate_markdown_table(mappings: List[VersionMapping]) -> str:
    """
    Generates a Markdown table showing the relationship:
      xlsynth-crate version → library release version → tag datetime (Los_Angeles)

    Each cell is a markdown link where appropriate. Column widths are padded
    dynamically based on the maximum rendered cell width.
    """
    # Link to the xlsynth crate on crates.io for the left column.
    crate_base_url = "https://crates.io/crates/xlsynth/{}"
    # Right column still links to the GitHub release for the lib.
    lib_base_url = "https://github.com/xlsynth/xlsynth/releases/tag/v{}"

    def version_key(v: VersionMapping):
        return tuple(int(x) for x in v.crate_version.split("."))

    sorted_mappings = sorted(mappings, key=version_key, reverse=True)

    left_header = "xlsynth crate version"
    mid_header = "xlsynth release version"
    right_header = "crate release datetime (Los_Angeles)"

    rows = []
    for vm in sorted_mappings:
        left_cell = f"[{vm.crate_version}]({crate_base_url.format(vm.crate_version)})"
        mid_cell = f"[{vm.lib_version}]({lib_base_url.format(vm.lib_version)})"
        right_cell = vm.crate_release_datetime
        rows.append((left_cell, mid_cell, right_cell))

    left_width = max(len(left_header), *(len(r[0]) for r in rows))
    mid_width = max(len(mid_header), *(len(r[1]) for r in rows))
    right_width = max(len(right_header), *(len(r[2]) for r in rows))

    header_row = f"| {left_header.ljust(left_width)} | {mid_header.ljust(mid_width)} | {right_header.ljust(right_width)} |"
    separator_row = f"| {'-' * left_width} | {'-' * mid_width} | {'-' * right_width} |"
    data_rows = [
        f"| {left.ljust(left_width)} | {mid.ljust(mid_width)} | {right.ljust(right_width)} |"
        for left, mid, right in rows
    ]

    return "\n".join([header_row, separator_row] + data_rows) + "\n"


def get_all_tags() -> List[str]:
    """Return a list of tags matching the pattern vX.Y.Z."""
    try:
        result = subprocess.run(
            ["git", "tag", "--list", "v[0-9]*.[0-9]*.[0-9]*"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        tags = result.stdout.strip().splitlines()
        return tags
    except subprocess.CalledProcessError as e:
        print(f"Error listing tags: {e.stderr}", file=sys.stderr)
        return []


@lru_cache(maxsize=None)
def _published_versions(crate_name: str) -> FrozenSet[str]:
    """Return the crate's sparse-index versions, caching one lookup per generator run."""
    return frozenset(get_published_versions(crate_name))


def _get_publish_order_at_tag(tag: str) -> Optional[List[str]]:
    """Return the crate publication order declared by tag, if that tag has one."""
    content = get_file_content_at_commit(tag, PUBLISH_ORDER_PATH)
    if content is None:
        return None
    data = tomllib.loads(content)
    crates = data.get("crates")
    if (
        not isinstance(crates, list)
        or not crates
        or not all(isinstance(crate, str) for crate in crates)
    ):
        raise ValueError(
            f"{tag}:{PUBLISH_ORDER_PATH} must contain a non-empty crates array"
        )
    return crates


def _get_release_ineligibility_reason(tag: str, crate_version: str) -> Optional[str]:
    """Return why a release cannot enter generated compatibility metadata."""
    publish_order = _get_publish_order_at_tag(tag)
    if publish_order is None:
        return f"{tag} does not contain {PUBLISH_ORDER_PATH}"

    missing = [
        crate
        for crate in publish_order
        if crate_version not in _published_versions(crate)
    ]
    if missing:
        return f"missing crate publications: {', '.join(missing)}"
    return None


def _load_existing_mappings(json_path: str) -> Dict[str, VersionMapping]:
    """Load existing mappings from generated_version_compat.json if present."""
    if not os.path.exists(json_path):
        return {}
    try:
        with open(json_path, "r") as jf:
            data = json.load(jf)
    except Exception as e:
        print(f"Error reading existing mapping file {json_path}: {e}", file=sys.stderr)
        return {}
    mappings: Dict[str, VersionMapping] = {}
    for crate_version, entry in data.items():
        lib_version = entry.get("xlsynth_release_version")
        release_dt = entry.get("crate_release_datetime")
        if not lib_version or not release_dt:
            continue
        mappings[crate_version] = VersionMapping(
            crate_version=crate_version,
            lib_version=lib_version,
            crate_release_datetime=release_dt,
        )
    return mappings


def get_version_mapping(recompute_all_entries: bool) -> List[VersionMapping]:
    file_path = "xlsynth-sys/build.rs"
    all_tags = get_all_tags()
    print(f"Found {len(all_tags)} tags. Processing...", flush=True)
    mappings: List[VersionMapping] = []
    skipped_unpublished = 0
    existing_path = "generated_version_compat.json"
    existing = {} if recompute_all_entries else _load_existing_mappings(existing_path)
    for tag in all_tags:
        print(f"Processing tag {tag}...", flush=True)
        crate_version = tag.lstrip("v")
        if not recompute_all_entries and crate_version in existing:
            mappings.append(existing[crate_version])
            print(
                f"  Reused existing mapping for crate version {crate_version}",
                flush=True,
            )
            continue
        content = get_file_content_at_commit(tag, file_path)
        if not content:
            print(f"  Skipped tag {tag}: no file content found.", flush=True)
            continue
        lib_version = extract_version_from_content(content)
        if not lib_version:
            print(f"  Skipped tag {tag}: no lib version extracted.", flush=True)
            continue
        release_dt = get_tag_datetime(tag) or "Unknown"
        ineligibility_reason = _get_release_ineligibility_reason(tag, crate_version)
        if ineligibility_reason is not None:
            print(
                f"  Skipped tag {tag}: {ineligibility_reason}.",
                flush=True,
            )
            skipped_unpublished += 1
            continue
        mappings.append(
            VersionMapping(
                crate_version=crate_version,
                lib_version=lib_version,
                crate_release_datetime=release_dt,
            )
        )
        print(
            f"  Mapped crate version {crate_version} -> lib version {lib_version}",
            flush=True,
        )
    print(
        f"Skipped {skipped_unpublished} tag(s) due to incomplete crate publications.",
        flush=True,
    )
    return mappings


def get_last_tag_time() -> str:
    """Return the creatordate (ISO8601) of the most recent tag matching v* or 'Unknown' if not available."""
    try:
        result = subprocess.run(
            [
                "git",
                "for-each-ref",
                "--sort=-creatordate",
                "--format=%(creatordate:iso8601)",
                "refs/tags/v*",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
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
    ineligibility_reason = _get_release_ineligibility_reason(tag, crate_version)
    if ineligibility_reason is not None:
        print(
            f"  Skipped tag {tag}: {ineligibility_reason}.",
            flush=True,
        )
        return None
    release_dt = get_tag_datetime(tag) or "Unknown"
    return VersionMapping(
        crate_version=crate_version,
        lib_version=lib_version,
        crate_release_datetime=release_dt,
    )


# -- Entry point


def mappings_to_json_object(
    mappings: List[VersionMapping],
) -> Dict[str, Dict[str, str]]:
    """Convert VersionMapping list to the JSON object we persist.

    Shape:
      {
        "<crate_version>": {
          "xlsynth_release_version": "<lib_version>",
          "crate_release_datetime": "<crate_release_datetime>"
        },
        ...
      }

    Entries are ordered by descending semantic version of the crate version
    (e.g. 0.0.172 before 0.0.101 before 0.0.99). Python dicts preserve
    insertion order, and we rely on that when writing JSON to keep a
    human-friendly ordering.
    """

    def version_key(vm: VersionMapping) -> Tuple[int, int, int]:
        return tuple(int(x) for x in vm.crate_version.split("."))  # type: ignore[return-value]

    result: Dict[str, Dict[str, str]] = {}
    for vm in sorted(mappings, key=version_key, reverse=True):
        # Later entries for the same crate version would overwrite earlier ones.
        result[vm.crate_version] = {
            "xlsynth_release_version": vm.lib_version,
            "crate_release_datetime": vm.crate_release_datetime,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate version compatibility table and JSON mapping."
    )
    parser.add_argument(
        "--recompute-all-entries",
        action="store_true",
        help="Recompute every mapping instead of reusing existing entries.",
    )
    args = parser.parse_args()

    mappings = get_version_mapping(args.recompute_all_entries)
    if not mappings:
        print("No release tag mappings found in commit history", file=sys.stderr)
        sys.exit(1)
    table = generate_markdown_table(mappings)
    last_tag_time = get_last_tag_time()
    full_content = (
        f"# Version Map\n\nUpdated for tags as of: {last_tag_time}\n\n{table}"
    )
    output_path = os.path.join("docs", "version_metadata.md")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(full_content)
    print(f"Version compatibility table written to {output_path}")

    # Also emit a machine-readable JSON mapping at repo root.
    json_obj = mappings_to_json_object(mappings)
    json_output_path = "generated_version_compat.json"
    with open(json_output_path, "w") as jf:
        # Preserve insertion order to keep semantic sorting in the file.
        json.dump(json_obj, jf, indent=2)
        jf.write("\n")
    print(f"JSON mapping written to {json_output_path}")


def test_mapping_for_v0_49_0() -> None:
    mapping = get_single_version_mapping("0.49.0")
    assert mapping is not None, "Mapping for xlsynth-crate v0.49.0 not found"
    assert (
        mapping.crate_version == "0.49.0"
    ), f"Expected crate version '0.49.0', got {mapping.crate_version}"
    assert (
        mapping.lib_version == "0.45.0"
    ), f"Expected lib version '0.45.0', got {mapping.lib_version}"
    expected_dt = get_tag_datetime("v0.49.0")
    assert expected_dt is not None, "Could not determine expected datetime for v0.49.0"
    assert (
        mapping.crate_release_datetime == expected_dt
    ), f"Expected datetime '{expected_dt}', got {mapping.crate_release_datetime}"


def test_mapping_for_v0_48_0() -> None:
    mapping = get_single_version_mapping("0.48.0")
    assert mapping is not None, "Mapping for xlsynth-crate v0.48.0 not found"
    assert (
        mapping.crate_version == "0.48.0"
    ), f"Expected crate version '0.48.0', got {mapping.crate_version}"
    assert (
        mapping.lib_version == "0.45.0"
    ), f"Expected lib version '0.45.0', got {mapping.lib_version}"
    expected_dt = get_tag_datetime("v0.48.0")
    assert expected_dt is not None, "Could not determine expected datetime for v0.48.0"
    assert (
        mapping.crate_release_datetime == expected_dt
    ), f"Expected datetime '{expected_dt}', got {mapping.crate_release_datetime}"


def test_mappings_to_json_object() -> None:
    mappings = [
        VersionMapping(
            crate_version="0.0.1",
            lib_version="0.0.10",
            crate_release_datetime="2024-01-01 00:00:00 PST",
        ),
        VersionMapping(
            crate_version="0.0.2",
            lib_version="0.0.11",
            crate_release_datetime="2024-01-02 00:00:00 PST",
        ),
    ]
    want = {
        "0.0.1": {
            "xlsynth_release_version": "0.0.10",
            "crate_release_datetime": "2024-01-01 00:00:00 PST",
        },
        "0.0.2": {
            "xlsynth_release_version": "0.0.11",
            "crate_release_datetime": "2024-01-02 00:00:00 PST",
        },
    }
    got = mappings_to_json_object(mappings)
    assert got == want, f"JSON mapping mismatch.\nwant={want}\n got={got}"


def test_mappings_to_json_object_ordering() -> None:
    mappings = [
        VersionMapping(
            crate_version="0.0.99",
            lib_version="0.0.170",
            crate_release_datetime="2024-01-03 00:00:00 PST",
        ),
        VersionMapping(
            crate_version="0.0.101",
            lib_version="0.0.173",
            crate_release_datetime="2024-01-05 00:00:00 PST",
        ),
        VersionMapping(
            crate_version="0.0.100",
            lib_version="0.0.173",
            crate_release_datetime="2024-01-04 00:00:00 PST",
        ),
    ]
    obj = mappings_to_json_object(mappings)
    got_keys = list(obj.keys())
    want_keys = ["0.0.101", "0.0.100", "0.0.99"]
    assert (
        got_keys == want_keys
    ), f"Ordering mismatch.\nwant={want_keys}\n got={got_keys}"


if __name__ == "__main__":
    main()
