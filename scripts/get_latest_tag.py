#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Print the latest git tag matching vX.Y.Z.

This script refreshes tags (git pull --tags) and then selects the highest
semantic version among tags matching ^v<major>.<minor>.<patch>$.

Usage:
  python3 scripts/get_latest_tag.py
"""

import os
import re
import subprocess
import sys
from typing import List, Optional, Tuple


def _repo_root() -> str:
    # scripts/ is one level below the repo root.
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _run_git(args: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=_repo_root(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def _parse_v_tag(tag: str) -> Optional[Tuple[int, int, int]]:
    m = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", tag)
    if not m:
        return None
    try:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    except ValueError:
        return None


def _select_latest(tags: List[str]) -> Optional[str]:
    best: Optional[Tuple[int, int, int]] = None
    best_tag: Optional[str] = None
    for t in tags:
        key = _parse_v_tag(t.strip())
        if key is None:
            continue
        if best is None or key > best:
            best = key
            best_tag = t.strip()
    return best_tag


def main() -> int:
    # Refresh tags; ignore failure (offline or no remote), continue with local tags.
    _run_git(["pull", "--tags"])  # best-effort

    # Prefer listing only vX.Y.Z tags directly.
    cp = _run_git(["tag", "--list", "v[0-9]*.[0-9]*.[0-9]*"])  # glob pattern
    if cp.returncode != 0:
        print(cp.stderr or "git tag --list failed", file=sys.stderr)
        return 2

    tags = [line for line in cp.stdout.splitlines() if line.strip()]
    latest = _select_latest(tags)
    if not latest:
        print("No vX.Y.Z tags found.", file=sys.stderr)
        return 1

    print(latest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
