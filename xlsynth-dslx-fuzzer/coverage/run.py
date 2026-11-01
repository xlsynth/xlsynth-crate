#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Build and measure the DSLX-only C++ interpreter against generated samples."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time


TARGET = "//xls/dslx/semantics_coverage:dslx_coverage_main"


def execute(command, cwd, **kwargs):
    """Run one checked build or profiling command."""
    return subprocess.run(command, cwd=cwd, check=True, text=True, **kwargs)


def relative_dslx_path(path):
    """Select DSLX source paths while excluding the measurement harness."""
    normalized = path.replace("\\", "/")
    if normalized.startswith("xls/dslx/"):
        relative = normalized[len("xls/dslx/") :]
    elif "/xls/dslx/" in normalized:
        relative = normalized.split("/xls/dslx/", 1)[1]
    else:
        return None
    return None if relative.startswith("semantics_coverage/") else relative


def summarize(coverage):
    """Aggregate coverage by subsystem and retain uncalled native functions."""
    groups = {}
    files = []
    functions = []
    for unit in coverage["data"]:
        for source in unit["files"]:
            relative = relative_dslx_path(source["filename"])
            if relative is None:
                continue
            group = relative.split("/", 1)[0] if "/" in relative else "dslx_support"
            summary = source["summary"]
            totals = groups.setdefault(group, {})
            for metric in ["lines", "branches", "functions", "regions"]:
                total = totals.setdefault(metric, {"count": 0, "covered": 0})
                for key in ["count", "covered"]:
                    total[key] += summary[metric][key]
            files.append({"file": relative, "summary": summary})
        for function in unit["functions"]:
            if function["count"] != 0:
                continue
            # A function's first filename is the owning source. Macro expansions
            # may additionally refer to headers, including third-party headers.
            relative = relative_dslx_path(function["filenames"][0])
            if relative is None:
                continue
            regions = function["regions"]
            line = regions[0][0] if regions else 0
            functions.append({"file": relative, "line": line, "name": function["name"]})
    functions.sort(key=lambda item: (item["file"], item["line"], item["name"]))
    if functions:
        demangled = subprocess.run(
            ["c++filt"],
            input="\n".join(item["name"] for item in functions) + "\n",
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.splitlines()
        for function, name in zip(functions, demangled):
            function["name"] = name
    return {"groups": groups, "files": files, "uncovered_functions": functions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xls-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--constexpr-only", action="store_true")
    args = parser.parse_args()
    if args.samples < 0 or args.jobs < 1:
        parser.error("samples must be nonnegative and jobs must be positive")
    xls_root = args.xls_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).resolve().parent
    crate_root = here.parent
    overlay = output / "overlay"
    package = overlay / "xls/dslx/semantics_coverage"
    package.mkdir(parents=True, exist_ok=True)
    staged_names = {
        "BUILD": "BUILD.bazel",
        "dslx_coverage_main.cc": "dslx_coverage_main.cc",
    }
    for source, target in staged_names.items():
        contents = (here / source).read_bytes()
        if source == "BUILD":
            # Keep overlays compatible with both root Abseil repository names.
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
        "--package_path=" + str(overlay) + ":%workspace%",
        "--symlink_prefix=" + str(output / "bazel-"),
        "--jobs=" + str(args.jobs),
        "--per_file_copt=xls/dslx/.*@-fprofile-instr-generate,-fcoverage-mapping,-gline-tables-only",
        "--linkopt=-fprofile-instr-generate",
        "--strip=never",
    ]
    if not args.skip_build:
        execute(bazel + ["build"] + flags + [TARGET], xls_root)
    paths = execute(
        bazel + ["cquery"] + flags + ["--output=files", TARGET],
        xls_root,
        stdout=subprocess.PIPE,
    ).stdout.splitlines()
    binary_paths = [path for path in paths if path.endswith("/dslx_coverage_main")]
    if len(binary_paths) != 1:
        raise RuntimeError("cannot identify the native coverage executable")
    execution_root = execute(
        bazel + ["info", "execution_root"],
        xls_root,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    binary = Path(execution_root) / binary_paths[0]
    candidates = sorted((output / "bazel/external").glob("*/bin/llvm-cov"))
    if len(candidates) != 1:
        raise RuntimeError(
            "cannot uniquely identify coverage tools matching the Bazel LLVM toolchain: "
            + repr(candidates)
        )
    llvm_cov = candidates[0]
    llvm_profdata = llvm_cov.with_name("llvm-profdata")
    sample_directory = output / ("samples-" + str(time.time_ns()))
    export = [
        "cargo",
        "run",
        "--manifest-path",
        str(crate_root / "Cargo.toml"),
        "--example",
        "coverage_corpus",
        "--",
        str(sample_directory),
        str(args.samples),
    ]
    if args.corpus is not None:
        export.append(str(args.corpus.resolve()))
    if args.constexpr_only:
        export.append("--typecheck-only")
    execute(export, crate_root.parent)
    profile_directory = output / ("profiles-" + str(time.time_ns()))
    profile_directory.mkdir()
    environment = dict(
        os.environ, LLVM_PROFILE_FILE=str(profile_directory / "%p.profraw")
    )
    replay = [
        str(binary),
        str(xls_root / "xls/dslx/stdlib"),
        str(sample_directory / "manifest.txt"),
    ]
    if args.constexpr_only:
        replay.append("--typecheck-only")
    execute(
        replay,
        xls_root,
        env=environment,
    )
    profiles = sorted(profile_directory.glob("*.profraw"))
    if not profiles:
        raise RuntimeError("native run produced no LLVM coverage profile")
    profile = output / "coverage.profdata"
    execute(
        [str(llvm_profdata), "merge", "-sparse"]
        + [str(path) for path in profiles]
        + ["-o", str(profile)],
        xls_root,
    )
    common = [str(binary), "-instr-profile=" + str(profile)]
    with (output / "coverage.json").open("w") as stream:
        execute([str(llvm_cov), "export"] + common, xls_root, stdout=stream)
    with (output / "coverage.json").open() as stream:
        coverage = json.load(stream)
    summary = summarize(coverage)
    if not summary["files"]:
        raise RuntimeError("profile contains no DSLX source coverage")
    sources = [
        source["filename"]
        for unit in coverage["data"]
        for source in unit["files"]
        if relative_dslx_path(source["filename"]) is not None
    ]
    with (output / "report.txt").open("w") as stream:
        execute(
            [str(llvm_cov), "report"] + common + ["-sources"] + sources,
            xls_root,
            stdout=stream,
        )
    execute(
        [str(llvm_cov), "show"]
        + common
        + [
            "-format=html",
            "-output-dir=" + str(output / "html"),
            "-show-branches=count",
            "-show-line-counts-or-regions",
            "-sources",
        ]
        + sources,
        xls_root,
    )
    summary["binary"] = str(binary)
    summary["mode"] = "constexpr_only" if args.constexpr_only else "interpret"
    summary["sample_manifest"] = str(sample_directory / "manifest.txt")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "uncovered-functions.tsv").open("w") as stream:
        stream.write("file\tline\tfunction\n")
        for function in summary["uncovered_functions"]:
            stream.write("{file}\t{line}\t{name}\n".format(**function))
    for group, metrics in sorted(summary["groups"].items()):
        percentages = []
        for name in ["lines", "branches", "functions"]:
            metric = metrics[name]
            percent = (
                100 * metric["covered"] / metric["count"] if metric["count"] else 0
            )
            percentages.append("{} {:.2f}%".format(name, percent))
        print(group + ": " + ", ".join(percentages))
    print("Coverage HTML: " + str(output / "html/index.html"))


if __name__ == "__main__":
    main()
