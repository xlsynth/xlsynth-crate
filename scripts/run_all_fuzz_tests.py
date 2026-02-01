#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run all fuzz targets beneath xlsynth-crate.

Builds all fuzz targets and runs each for a short period of time.

Usage:
  python3 scripts/run_all_fuzz_tests.py

  # With custom args:
  #   cargo fuzz run --release --features=foo,bar <target> -- -max_total_time=10
  python3 scripts/run_all_fuzz_tests.py --fuzz-run-args=--release --features=foo,bar --fuzz-bin-args=-max_total_time=10
"""

import argparse
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

DEFAULT_FEATURES: list[str] = [
    "with-z3-binary-test",
    "with-boolector-system",
    "with-bitwuzla-system",
    "with-z3-system",
]
DEFAULT_FUZZ_RUN_ARGS: str = ""
DEFAULT_FUZZ_BIN_ARGS: str = "-max_total_time=5"

# Targets which are known to fail.
SKIP_TARGETS: list[str] = [
    "fuzz_bulk_replace",
    "fuzz_gate_transform_arbitrary",
    "fuzz_ir_opt_equiv",
    "fuzz_ir_outline_equiv",
]


def find_fuzz_dirs(repo_root: Path) -> list[Path]:
    """Return paths to top-level <crate>/fuzz/ directories."""
    fuzz_dirs: list[Path] = []
    for child in repo_root.iterdir():
        if not child.is_dir():
            continue
        fuzz_dir = child / "fuzz"
        if fuzz_dir.exists():
            fuzz_dirs.append(fuzz_dir)
    return fuzz_dirs


def run_cmd(cmd: list[str]) -> None:
    """Print the command to be run, then execute it.

    The command is echoed in a shell-safe, quoted form for easy copy/paste.
    """
    print("  => " + " ".join(shlex.quote(part) for part in cmd), file=sys.stderr)
    subprocess.check_call(cmd)


def get_crate_features(crate_path: Path) -> list[str]:
    """Return the list of features defined in <crate>/Cargo.toml."""
    cargo_toml = crate_path / "Cargo.toml"
    with open(cargo_toml, "rb") as f:
        cargo_data = tomllib.load(f)
    features = cargo_data.get("features")
    if not features:
        return []
    return sorted(features.keys())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fuzz-run-args",
        default=DEFAULT_FUZZ_RUN_ARGS,
        help="Arguments string passed to 'cargo fuzz run'. Example: \"--features=with-z3-system\"",
    )
    parser.add_argument(
        "--fuzz-bin-args",
        default=DEFAULT_FUZZ_BIN_ARGS,
        help='Arguments string passed to the fuzz binary. Example: "-max_total_time=10"',
    )
    parser.add_argument(
        "--features",
        default=DEFAULT_FEATURES,
        help='Features to pass to the fuzz targets. Example: "with-z3-system,with-foo"',
    )
    parser.add_argument(
        "--sanitizer",
        default="none",
        help='Sanitizer to enable via RUSTFLAGS, e.g. "address", "thread", or "none".',
    )
    args = parser.parse_args()
    sanitizer_args = ["--sanitizer", args.sanitizer]

    # scripts/ is one level below the repo root.
    repo_root = Path(__file__).resolve().parent.parent
    fuzz_dirs = find_fuzz_dirs(repo_root)

    if not fuzz_dirs:
        print("No fuzz projects found.", file=sys.stderr)
        sys.exit(1)

    fuzz_run_args_list: list[str] = (
        shlex.split(args.fuzz_run_args) if args.fuzz_run_args else []
    )
    fuzz_bin_args_list: list[str] = (
        shlex.split(args.fuzz_bin_args) if args.fuzz_bin_args else []
    )

    # Find the list of fuzz targets in each fuzz directory.
    fuzz_targets: list[tuple[str, list[str]]] = []
    for fuzz_dir in fuzz_dirs:
        list_output = subprocess.check_output(
            [
                "cargo",
                "fuzz",
                "list",
                "--fuzz-dir",
                fuzz_dir.as_posix(),
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
        targets = [line.strip() for line in list_output.splitlines() if line.strip()]
        if not targets:
            continue
        fuzz_targets.append((fuzz_dir, targets))

    # First build all targets. This surfaces build errors quickly.
    for fuzz_dir, _ in fuzz_targets:
        print(f"\n=== Building fuzz targets in {fuzz_dir} ===", file=sys.stderr)
        features = get_crate_features(fuzz_dir)
        supported_features = [f for f in features if f in args.features]
        features_args = (
            ["--features", ",".join(supported_features)] if supported_features else []
        )
        run_cmd(
            [
                "cargo",
                "fuzz",
                "build",
                "--fuzz-dir",
                fuzz_dir.as_posix(),
                *sanitizer_args,
                *features_args,
                *fuzz_run_args_list,
            ]
        )
    for fuzz_dir, targets in fuzz_targets:
        print(f"\n=== Running fuzz targets in {fuzz_dir} ===", file=sys.stderr)
        features = get_crate_features(fuzz_dir)
        supported_features = [f for f in features if f in args.features]
        features_args = (
            ["--features", ",".join(supported_features)] if supported_features else []
        )
        for target in targets:
            if target in SKIP_TARGETS:
                print(f"Skipping {target} in {fuzz_dir}.", file=sys.stderr)
                continue

            print(f"\n--- Running {target} in {fuzz_dir} ---", file=sys.stderr)
            run_cmd(
                [
                    "cargo",
                    "fuzz",
                    "run",
                    "--fuzz-dir",
                    fuzz_dir.as_posix(),
                    *sanitizer_args,
                    *features_args,
                    *fuzz_run_args_list,
                    target,
                    "--",
                    *fuzz_bin_args_list,
                ]
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
