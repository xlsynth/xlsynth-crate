# SPDX-License-Identifier: Apache-2.0

"""
Emits the XLS release tag required by this repo, as defined in
`xlsynth-sys/build.rs` (RELEASE_LIB_VERSION_TAG).

Usage:
  python3 get_required_xls_release_tag.py

Prints the tag string (e.g., "v0.0.224") to stdout.
"""

import os
import re
import sys
from typing import Optional


def extract_release_tag_from_build_rs(text: str) -> Optional[str]:
    m = re.search(
        r'RELEASE_LIB_VERSION_TAG:\s*&str\s*=\s*"(v\d+\.\d+\.\d+(?:-\d+)?)"', text
    )
    return m.group(1) if m else None


def main() -> None:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    build_rs_path = os.path.join(repo_root, "xlsynth-sys", "build.rs")
    try:
        with open(build_rs_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: could not find {build_rs_path}", file=sys.stderr)
        sys.exit(2)

    tag = extract_release_tag_from_build_rs(content)
    if not tag:
        print(
            "Error: could not extract RELEASE_LIB_VERSION_TAG from build.rs",
            file=sys.stderr,
        )
        sys.exit(3)

    print(tag)


if __name__ == "__main__":
    main()
