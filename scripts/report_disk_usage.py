# SPDX-License-Identifier: Apache-2.0

"""
Reports a compact disk-usage snapshot for CI diagnostics.

This is intended for GitHub Actions jobs that need lightweight, repeatable
instrumentation around package installs and Cargo builds/tests.
"""

import argparse
import os
import subprocess
import sys
from typing import Iterable, List, Optional, Tuple


def humanize_kib(kib: int) -> str:
    units = ["KiB", "MiB", "GiB", "TiB"]
    value = float(kib)
    unit = units[0]
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            break
        value /= 1024.0
    if unit == "KiB":
        return "{}{}".format(int(value), unit)
    return "{:.1f}{}".format(value, unit)


def run_command(argv: List[str]) -> int:
    print("$ {}".format(" ".join(argv)))
    completed = subprocess.run(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.stdout:
        sys.stdout.write(completed.stdout)
        if not completed.stdout.endswith("\n"):
            sys.stdout.write("\n")
    if completed.returncode != 0:
        print("[command exited with status {}]".format(completed.returncode))
    return completed.returncode


def du_kib(path: str) -> Tuple[Optional[int], Optional[str]]:
    completed = subprocess.run(
        ["du", "-sk", path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stdout.strip() or "du failed"
        return None, message
    first_field = completed.stdout.split(None, 1)[0]
    return int(first_field), None


def existing_paths(paths: Iterable[str]) -> List[str]:
    return [path for path in paths if os.path.exists(path)]


def print_size_table(title: str, paths: Iterable[str]) -> None:
    print("== {} ==".format(title))
    rows = []
    for path in paths:
        if not os.path.exists(path):
            print("missing  {}".format(path))
            continue
        size_kib, error = du_kib(path)
        if error is not None:
            print("error    {}  {}".format(error, path))
            continue
        rows.append((size_kib, path))

    if not rows:
        print("(no existing paths)")
        print()
        return

    rows.sort(reverse=True)
    for size_kib, path in rows:
        print("{:>8}  {}".format(humanize_kib(size_kib), path))
    print()


def list_children(path: str) -> List[str]:
    try:
        with os.scandir(path) as it:
            return [entry.path for entry in it]
    except OSError as e:
        print("warning: could not scan {}: {}".format(path, e))
        return []


def print_expanded_children(path: str, max_children: int) -> None:
    title = "Largest children under {}".format(path)
    print("== {} ==".format(title))
    if not os.path.exists(path):
        print("missing  {}".format(path))
        print()
        return
    children = list_children(path)
    if not children:
        print("(no children)")
        print()
        return

    rows = []
    for child in children:
        size_kib, error = du_kib(child)
        if error is not None:
            print("error    {}  {}".format(error, child))
            continue
        rows.append((size_kib, child))
    rows.sort(reverse=True)
    for size_kib, child in rows[:max_children]:
        print("{:>8}  {}".format(humanize_kib(size_kib), child))
    if len(rows) > max_children:
        print("... {} more entries omitted".format(len(rows) - max_children))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Human-readable snapshot label")
    parser.add_argument(
        "--workspace-root",
        help="Workspace root whose common build/output subpaths should be included",
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Path to measure with du -sk; may be repeated",
    )
    parser.add_argument(
        "--expand",
        action="append",
        default=[],
        help="Path whose immediate children should be ranked by size; may be repeated",
    )
    parser.add_argument(
        "--max-children",
        type=int,
        default=8,
        help="Maximum number of expanded child entries to print per path",
    )
    args = parser.parse_args()

    selected_paths = list(args.path)
    expanded_paths = list(args.expand)
    if args.workspace_root:
        root = args.workspace_root
        inferred_paths = [
            root,
            os.path.join(root, "target"),
            os.path.join(root, "target", "debug"),
            os.path.join(root, "target", "debug", "deps"),
            os.path.join(root, "target", "debug", "incremental"),
            os.path.join(root, "target", "debug", "build"),
            os.path.join(root, "xlsynth_tools"),
            os.path.join(root, "slang"),
            "/github/home/.cargo",
            "/github/home/.rustup",
            "/var/cache/dnf",
        ]
        selected_paths.extend(inferred_paths)
        expanded_paths.extend(
            [
                root,
                os.path.join(root, "target"),
                "/github/home/.cargo",
                "/github/home/.rustup",
                "/var/cache/dnf",
            ]
        )

    print("========== Disk Usage Snapshot: {} ==========".format(args.label))
    print()
    run_command(["df", "-h"])
    print()
    run_command(["df", "-ih"])
    print()
    print_size_table("Selected paths", selected_paths)
    for path in expanded_paths:
        print_expanded_children(path, args.max_children)


if __name__ == "__main__":
    main()
