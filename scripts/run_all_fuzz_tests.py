#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run all fuzz targets beneath xlsynth-crate.

Builds all fuzz targets and runs each for a short period of time.

Usage:
  python3 scripts/run_all_fuzz_tests.py

  # With custom args:
  #   cargo fuzz run --release --features=foo,bar <target> -- -max_total_time=10 -timeout=5
  python3 scripts/run_all_fuzz_tests.py --fuzz-run-args=--release --features=foo,bar --fuzz-bin-args="-max_total_time=10 -timeout=5"
"""

import argparse
import concurrent.futures
import re
import shlex
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

# Optional solver comparison backends can be requested with --features.
DEFAULT_FEATURES: list[str] = [
    "with-bitwuzla-system",
]
DEFAULT_FUZZ_RUN_ARGS: str = ""
# Formal fuzz targets apply a 10-second per-query solver limit internally. Some
# samples issue multiple queries, so leave headroom for the outer libFuzzer
# watchdog to catch genuinely stuck executions without misclassifying expected
# solver timeouts.
DEFAULT_FUZZ_BIN_ARGS: str = "-max_total_time=5 -timeout=60"
DEFAULT_THREADS: int = 4
DONE_RUNS_RE = re.compile(r"^Done ([0-9]+) runs in ([0-9]+) second\(s\)$")


def find_fuzz_dirs(repo_root: Path) -> list[Path]:
    """Return paths to top-level <crate>/fuzz/ directories."""
    fuzz_dirs: list[Path] = []
    for child in repo_root.iterdir():
        if not child.is_dir():
            continue
        fuzz_dir = child / "fuzz"
        if (fuzz_dir / "Cargo.toml").is_file():
            fuzz_dirs.append(fuzz_dir)
    return fuzz_dirs


def run_cmd(cmd: list[str]) -> None:
    """Print the command to be run, then execute it.

    The command is echoed in a shell-safe, quoted form for easy copy/paste.
    """
    print("  => " + " ".join(shlex.quote(part) for part in cmd), file=sys.stderr)
    subprocess.check_call(cmd)


def run_cmd_captured(cmd: list[str], log_dir: Path) -> tuple[int, Path]:
    """Execute `cmd` and spool combined stdout/stderr for deterministic replay."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=log_dir,
        prefix="fuzz_target_",
        suffix=".log",
        delete=False,
    ) as log_file:
        completed = subprocess.run(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return completed.returncode, Path(log_file.name)


def replay_log(log_path: Path) -> None:
    """Stream a captured log file to stdout without reading it all into memory."""
    with open(log_path, encoding="utf-8", errors="replace") as f:
        while chunk := f.read(1024 * 1024):
            sys.stdout.write(chunk)


def read_run_summary(log_path: Path) -> tuple[int, int] | None:
    """Return the last libFuzzer sample-count summary in a captured log."""
    summary: tuple[int, int] | None = None
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            match = DONE_RUNS_RE.match(line.rstrip("\n"))
            if match:
                summary = (int(match.group(1)), int(match.group(2)))
    return summary


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
        help='Arguments string passed to the fuzz binary. Example: "-max_total_time=10 -timeout=5"',
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
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=f"Maximum number of fuzz targets to run in parallel. Defaults to {DEFAULT_THREADS}.",
    )
    args = parser.parse_args()
    num_workers = args.threads
    if num_workers <= 0:
        parser.error("--threads must be positive")
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
    run_jobs: list[tuple[Path, str, list[str]]] = []
    for fuzz_dir, targets in fuzz_targets:
        features = get_crate_features(fuzz_dir)
        supported_features = [f for f in features if f in args.features]
        features_args = (
            ["--features", ",".join(supported_features)] if supported_features else []
        )
        for target in targets:
            run_jobs.append(
                (
                    fuzz_dir,
                    target,
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
                    ],
                )
            )

    print(
        f"\n=== Running {len(run_jobs)} fuzz targets with {num_workers} worker threads ===",
        file=sys.stderr,
    )
    with tempfile.TemporaryDirectory(prefix="run_all_fuzz_tests_logs_") as log_dir_text:
        log_dir = Path(log_dir_text)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_job = {}
            for fuzz_dir, target, cmd in run_jobs:
                print(f"\n--- Starting {target} in {fuzz_dir} ---", file=sys.stderr)
                print(
                    "  => " + " ".join(shlex.quote(part) for part in cmd),
                    file=sys.stderr,
                )
                future = executor.submit(run_cmd_captured, cmd, log_dir)
                future_to_job[future] = (fuzz_dir, target)

            failed_targets: list[tuple[Path, str]] = []
            run_summaries: list[tuple[int, int, Path, str]] = []
            for future in concurrent.futures.as_completed(future_to_job):
                fuzz_dir, target = future_to_job[future]
                returncode, log_path = future.result()
                try:
                    summary = read_run_summary(log_path)
                    if summary is not None:
                        runs, seconds = summary
                        run_summaries.append((runs, seconds, fuzz_dir, target))
                    if returncode == 0:
                        print(
                            f"\n--- Passed {target} in {fuzz_dir} ---", file=sys.stderr
                        )
                    else:
                        print(
                            f"\n--- Failed {target} in {fuzz_dir} ---", file=sys.stderr
                        )
                        replay_log(log_path)
                        failed_targets.append((fuzz_dir, target))
                finally:
                    log_path.unlink(missing_ok=True)
            if run_summaries:
                print("\n=== Fuzz target sample counts ===", file=sys.stderr)
                for runs, seconds, fuzz_dir, target in sorted(run_summaries):
                    rate = runs / seconds if seconds else 0.0
                    print(
                        f"  {runs:>12} runs  {rate:>10.1f} runs/s  {fuzz_dir}: {target}",
                        file=sys.stderr,
                    )
            if failed_targets:
                print("\n=== Failed fuzz targets ===", file=sys.stderr)
                for fuzz_dir, target in failed_targets:
                    print(f"  {fuzz_dir}: {target}", file=sys.stderr)
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
