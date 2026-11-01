#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Build an isolated DSLX DSO whose native counters feed Rust libFuzzer."""

import argparse
import json
from pathlib import Path
import re
import subprocess


TARGET = "//xls/dslx/semantics_native:libdslx_fuzz_native.so"


def run(command, cwd, **kwargs):
    """Run a checked native build or metadata query."""
    return subprocess.run(command, cwd=cwd, check=True, text=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xls-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("jobs must be positive")
    xls_root = args.xls_root.resolve()
    output = args.output.resolve()
    here = Path(__file__).resolve().parent
    package = output / "overlay/xls/dslx/semantics_native"
    package.mkdir(parents=True, exist_ok=True)
    for source, target in [
        ("BUILD", "BUILD.bazel"),
        ("dslx_fuzz_native.cc", "dslx_fuzz_native.cc"),
        ("exports.lds", "exports.lds"),
    ]:
        contents = (here / source).read_bytes()
        if source == "BUILD":
            # XLS renamed its root Abseil repository alias. Use the selected
            # checkout's label without changing that checkout's BUILD files.
            label = re.search(
                r'"(@[^"\s]+)//absl/status"',
                (xls_root / "xls/dslx/BUILD").read_text(),
            )
            if label is None:
                raise RuntimeError("cannot identify the checkout's Abseil label")
            contents = contents.replace(b"@com_google_absl", label[1].encode())
        if (
            not (package / target).exists()
            or (package / target).read_bytes() != contents
        ):
            (package / target).write_bytes(contents)
    bazel = ["bazel", "--output_base=" + str(output / "bazel")]
    flags = [
        "-c",
        "opt",
        "--package_path=" + str(output / "overlay") + ":%workspace%",
        "--symlink_prefix=" + str(output / "bazel-"),
        "--jobs=" + str(args.jobs),
        "--per_file_copt=xls/dslx/.*,-xls/dslx/semantics_native.*@-fsanitize-coverage=inline-8bit-counters,-fsanitize-coverage=pc-table,-fsanitize-coverage=trace-cmp",
        "--strip=never",
    ]
    forbidden = run(
        bazel
        + [
            "query",
            "--package_path=" + str(output / "overlay") + ":%workspace%",
            'filter("//xls/jit[:/]|//xls/interpreter[:/]|//xls/dslx/ir_convert:ir_converter", deps('
            + TARGET
            + "))",
        ],
        xls_root,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if forbidden:
        raise RuntimeError(
            "native checker unexpectedly links execution backends: " + forbidden
        )
    run(bazel + ["build"] + flags + [TARGET], xls_root)
    paths = run(
        bazel + ["cquery"] + flags + ["--output=files", TARGET],
        xls_root,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    libraries = [path for path in paths if path.endswith("/libdslx_fuzz_native.so")]
    if len(libraries) != 1:
        raise RuntimeError("cannot identify the instrumented DSLX shared library")
    execution_root = run(
        bazel + ["info", "execution_root"], xls_root, stdout=subprocess.PIPE
    ).stdout.strip()
    library = Path(execution_root) / libraries[0]
    symbols = run(
        ["nm", "-D", "--defined-only", str(library)],
        xls_root,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    exports = {line.split()[-1] for line in symbols if line.strip()}
    if exports != {"dslx_fuzz_native_check", "dslx_fuzz_native_counters"}:
        raise RuntimeError(
            "native checker exports unexpected symbols: " + repr(sorted(exports))
        )
    config = {"library": str(library), "stdlib": str(xls_root / "xls/dslx/stdlib")}
    (output / "native.json").write_text(json.dumps(config, indent=2) + "\n")
    print("Native configuration: " + str(output / "native.json"))


if __name__ == "__main__":
    main()
