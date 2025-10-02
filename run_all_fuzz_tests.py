#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run all fuzz targets beneath xlsynth-crate.

Builds all fuzz targets and runs each for a short period of time.

Usage:
  python3 run_all_fuzz_tests.py

  # With custom args:
  #   cargo fuzz run --features=foo,bar <target> -- -max_total_time=10
  python3 run_all_fuzz_tests.py --fuzz-run-args=--features=foo,bar --fuzz-bin-args=-max_total_time=10
"""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

DEFAULT_FUZZ_BIN_ARGS: str = "-max_total_time=5"

# Targets which are known to fail.
SKIP_TARGETS: list[str] = ["fuzz_bulk_replace", "fuzz_gate_transform_arbitrary"]

def find_fuzz_dirs(repo_root: Path) -> list[Path]:
    """Return paths to top-level <crate>/fuzz/ directories."""
    fuzz_dirs: list[Path] = []
    for child in repo_root.iterdir():
        if not child.is_dir():
            continue
        fuzz_dir = child / "fuzz"
        if fuzz_dir.exists():
            fuzz_dirs.append(child)
    return fuzz_dirs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fuzz-run-args",
        default="",
        help="Arguments string passed to 'cargo fuzz run'. Example: \"--features=with-z3-system\"",
    )
    parser.add_argument(
        "--fuzz-bin-args",
        default=DEFAULT_FUZZ_BIN_ARGS,
        help="Arguments string passed to the fuzz binary. Example: \"-max_total_time=10\"",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    fuzz_dirs = find_fuzz_dirs(repo_root)

    if not fuzz_dirs:
        print("No fuzz projects found.", file=sys.stderr)
        sys.exit(1)


    fuzz_run_args_list: list[str] = shlex.split(args.fuzz_run_args) if args.fuzz_run_args else []
    fuzz_bin_args_list: list[str] = shlex.split(args.fuzz_bin_args) if args.fuzz_bin_args else []

    # Find the list of fuzz targets in each fuzz directory.
    fuzz_targets: list[tuple[str, list[str]]] = []
    for fuzz_dir in fuzz_dirs:
        list_output = subprocess.check_output(
            [
                "cargo",
                "fuzz",
                "list",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            cwd=fuzz_dir,
        )
        targets = [line.strip() for line in list_output.splitlines() if line.strip()]
        if not targets:
            continue
        fuzz_targets.append((fuzz_dir, targets))

    # First build all targets. This surfaces build errors quickly.
    for fuzz_dir, _ in fuzz_targets:
        print(f"\n=== Building fuzz targets in {fuzz_dir} ===")
        subprocess.check_call(
                [
                    "cargo",
                    "fuzz",
                    "build",
                ],
                cwd=fuzz_dir,
            )
    for fuzz_dir, targets in fuzz_targets:
        print(f"\n=== Running fuzz targets in {fuzz_dir} ===")
        for target in targets:
            if target in SKIP_TARGETS:
                print(f"Skipping {target} in {fuzz_dir}.")
                continue

            print(f"\n--- Running {target} in {fuzz_dir} ---")
            subprocess.check_call(
                [
                    "cargo",
                    "fuzz",
                    "run",
                    *fuzz_run_args_list,
                    target,
                    "--",
                    *fuzz_bin_args_list,
                ],
                cwd=fuzz_dir,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())


