#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Freeze frontend outputs, then compare native and directly invoked ABC backends.

No stage invokes Yosys. Configuration and corpus paths are command-line inputs;
private libraries, designs, and experiment results are never repository fixtures.
"""

import argparse
from collections import Counter
import concurrent.futures
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def invoke(command, directory, label, timeout, stdout_path=None):
    """Capture every subprocess and reject accidental invocation of Yosys."""
    command = [str(item) for item in command]
    if Path(command[0]).name.lower() in ("yosys", "yosys.exe"):
        raise ValueError("this experiment deliberately does not permit Yosys")
    started = time.monotonic()
    write_json(directory / (label + ".command.json"), command)
    try:
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   timeout=timeout, check=False)
    except subprocess.TimeoutExpired as error:
        (directory / (label + ".log")).write_bytes(
            (error.stdout or b"") + (error.stderr or b"") + b"\nStage timed out.\n")
        raise
    (directory / (label + ".log")).write_bytes(completed.stdout + completed.stderr)
    if completed.returncode:
        raise RuntimeError("{} failed: {}".format(label, completed.stderr.decode(errors="replace")[-1200:]))
    if stdout_path:
        Path(stdout_path).write_bytes(completed.stdout)
    return time.monotonic() - started


def parallel_cases(cases, worker, workers, output):
    """Persist partial results without letting one timeout lose the sweep."""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(worker, case): case for case in cases}
        for future in concurrent.futures.as_completed(futures):
            case = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = dict(case, status="timeout" if isinstance(error, subprocess.TimeoutExpired) else "error",
                              error=str(error))
            results.append(result)
            results.sort(key=lambda row: (row["corpus"], row["design"]))
            write_json(output / "results.json", results)
            print("{}/{} {}/{} {}".format(len(results), len(cases), case["corpus"], case["design"], result["status"]), flush=True)
    return results


def freeze(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.selection:
        cases = json.loads(args.selection.read_text())
    else:
        cases = []
        for value in args.corpus:
            name, directory = value.split("=", 1)
            for path in sorted(Path(directory).glob("*.ir")):
                cases.append({"corpus": name, "design": path.stem.removesuffix(".opt"), "input": str(path.resolve())})
    write_json(output / "manifest.json", {
        "stage": "frozen-prebackend", "driver": str(args.driver.resolve()),
        "command": sys.argv, "python_version": sys.version,
        "driver_sha256": digest(args.driver), "utility_driver_sha256": digest(args.utility_driver),
        "workers": args.workers, "timeout": args.timeout, "cases": cases,
        "register_policy": "uninitialized-dont-care", "flop_inputs": True, "flop_outputs": True,
    })

    def worker(case):
        directory = output / case["corpus"] / case["design"]
        directory.mkdir(parents=True)
        core = directory / "core.g8rbin"
        pipeline = directory / "pipeline.g8rbin"
        cleaned = directory / "cleaned.g8rbin"
        aig = directory / "transition.aig"
        started = time.monotonic()
        invoke([args.driver, "ir2g8r", case["input"], "--unsafe-gatify-gate-operation=true",
                "--enable-formal-array-alias-analysis=false",
                "--cut-db-rewrite-mode=delay", "--compute_graph_logical_effort=false",
                "--bin-out", core, "--aiger-out", directory / "core.aig"],
               directory, "lower", args.timeout, directory / "core.g8r")
        invoke([args.driver, "g8r-stitch-pipeline", core, "--output_design_name", "top",
                "--clock_name", "clk", "--flop_inputs", "true", "--flop_outputs", "true",
                "--bin-out", pipeline], directory, "register", args.timeout)
        invoke([args.utility_driver, "g8r-cleanup-registers", pipeline,
                "--initialization-policy", "uninitialized-dont-care", "--quiet", "true",
                "--bin-out", cleaned, "--transition-aiger-out", aig], directory, "cleanup", args.timeout)
        return dict(case, status="ok", input_sha256=digest(case["input"]),
                    core=str(core), core_sha256=digest(core), design_path=str(cleaned),
                    design_sha256=digest(cleaned), aig=str(aig), aig_sha256=digest(aig),
                    frontend_seconds=time.monotonic() - started)

    parallel_cases(cases, worker, args.workers, output)


def abc_quote(path):
    value = str(path)
    if any(char in value for char in ('"', '\n', '\r', ';')):
        raise ValueError("unsafe character in ABC path")
    return '"' + value + '"'


def abc_program(args, source, destination, mapping):
    lines = ["# SPDX-License-Identifier: Apache-2.0", "read_aiger " + abc_quote(source)]
    for index, library in enumerate(args.liberty):
        lines.append("read_lib{} -w {}".format(" -m" if index else "", abc_quote(library)))
    lines += ["read_constr " + abc_quote(args.constraints), "&get -n", "&st", "&dch", "&nf"]
    for index in range(args.rounds):
        lines += ["&st", args.syn_command, "&if -g -K 6", "&synch2"]
        if index + 1 < args.rounds:
            lines.append("&nf")
    if mapping:
        lines += ["&nf", "&put", "buffer -c", "topo", "stime -c", "upsize -c", "dnsize -c",
                  "write_verilog " + abc_quote(destination)]
    else:
        lines += ["&dfs -c", "&write -s " + abc_quote(destination)]
    return "\n".join(lines) + "\n"


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cases = [case for case in json.loads(args.inputs.read_text()) if case["status"] == "ok"]
    if args.selection:
        selected = {(r["corpus"], r["design"]) for r in json.loads(args.selection.read_text())}
        cases = [r for r in cases if (r["corpus"], r["design"]) in selected]
    if not cases:
        raise ValueError("no successfully frozen inputs selected")
    write_json(output / "manifest.json", {
        "stage": "post-techmap", "backend": args.backend, "name": args.name,
        "command": sys.argv, "python_version": sys.version,
        "inputs_sha256": digest(args.inputs), "driver_sha256": digest(args.driver),
        "evaluator_sha256": digest(args.evaluator), "adapter_sha256": digest(args.adapter),
        "abc_sha256": digest(args.abc), "liberties": {str(p): digest(p) for p in args.liberty},
        "proto_sha256": digest(args.proto), "constraints_sha256": digest(args.constraints),
        "fixed_cell": args.cell, "workers": args.workers, "timeout": args.timeout,
        "rounds": args.rounds, "resize_rounds": args.resize_rounds,
        "syn_command": args.syn_command,
        "resize_iterations": args.resize_iterations, "area_iterations": args.area_iterations,
        "max_fanout": args.max_fanout, "verify": args.verify, "yosys_invocations": 0,
        "input_status_counts": dict(Counter(r["status"] for r in json.loads(args.inputs.read_text()))),
        "primary_input_transition": 0.01, "module_output_load": 2.308,
        "timeout_scope": "each subprocess", "verification_seed": 0,
    })
    mapping_proto = args.proto
    if args.backend == "native" and args.cell:
        mapping_proto = output / "fixed-flop.proto"
        invoke([args.adapter, "filter-library", "--design", cases[0]["design_path"],
                "--liberty-proto", args.proto, "--cell", args.cell, "--output", mapping_proto],
               output, "filter-library", args.timeout)

    def worker(case):
        directory = output / case["corpus"] / case["design"]
        directory.mkdir(parents=True)
        for field in ("design", "aig", "core"):
            path = case["design_path" if field == "design" else field]
            if digest(path) != case[field + "_sha256"]:
                raise ValueError("frozen input changed: " + path)
        started = time.monotonic()
        source = Path(case["aig"])
        netlist = directory / "mapped.gv"
        header = source.read_bytes().split(b"\n", 1)[0].split()
        if int(header[4]) == 0:
            return dict(case, status="degenerate", reason="no transition outputs")
        if args.backend == "abc":
            if not args.cell:
                raise ValueError("the ABC register shell requires an explicit --cell")
            source = directory / "physical-transition.aig"
            invoke([args.adapter, "prepare", "--design", case["design_path"],
                    "--liberty-proto", args.proto, "--cell", args.cell, "--output", source],
                   directory, "prepare", args.timeout)
        mapped_transition = directory / ("transition.gv" if args.backend == "abc" else "choices.aig")
        script = directory / "flow.abc"
        script.write_text(abc_program(args, source, mapped_transition, args.backend == "abc"))
        invoke([args.abc, "-f", script], directory, "abc", args.timeout)
        if not mapped_transition.is_file():
            raise RuntimeError("ABC exited without producing the requested artifact; inspect abc.log")
        if args.backend == "abc":
            invoke([args.adapter, "restore", "--design", case["design_path"],
                    "--liberty-proto", args.proto, "--cell", args.cell,
                    "--netlist", mapped_transition, "--output", netlist], directory, "restore", args.timeout)
        else:
            invoke([args.driver, "choice-aig-tech-map", mapped_transition,
                    "--sequential-design", case["design_path"], "--liberty_proto", mapping_proto,
                    "--module_name", "top", "--timing-model", "nf-liberty",
                    "--buffer", "true", "--max-fanout", args.max_fanout,
                    "--resize", "true", "--resize-rounds", args.resize_rounds,
                    "--resize-iterations", args.resize_iterations,
                    "--resize-area-iterations", args.area_iterations, "--resize-max-evaluations", "64",
                    "--primary-input-transition", "0.01", "--module-output-load", "2.308",
                    "--netlist_out", netlist], directory, "native", args.timeout)
        synthesis_seconds = time.monotonic() - started
        stats_path = directory / "stats.json"
        invoke([args.evaluator, "gv-stats", "--netlist", netlist, "--liberty_proto", args.proto,
                "--module_name", "top", "--primary_input_transition", "0.01",
                "--module_output_load", "2.308", "--json_out", stats_path], directory, "stats", args.timeout)
        stats = json.loads(stats_path.read_text())
        result = dict(case, status="ok" if stats.get("max_register_to_register_delay") else "degenerate",
                      area=stats["cell_area"], sequential_area=stats["sequential_cell_area"],
                      combinational_area=stats["cell_area"] - stats["sequential_cell_area"],
                      delay_ps=stats.get("max_register_to_register_delay"),
                      synthesis_seconds=synthesis_seconds, netlist=str(netlist), stats=str(stats_path))
        if args.verify and result["status"] == "ok":
            metadata = directory / "interface.json"
            invoke([args.adapter, "inspect", "--design", case["design_path"], "--liberty-proto", args.proto,
                    "--cell", args.cell or args.verify_cell, "--output", metadata], directory, "inspect", args.timeout)
            ports = json.loads(metadata.read_text())["inputs"]
            rng = random.Random(0)
            vectors = directory / "inputs.irvals"
            vectors.write_text("\n".join("{" + ", ".join(
                "{}: bits[{}]:{}".format(p["name"], p["width"], rng.getrandbits(p["width"]))
                for p in ports) + "}" for _ in range(args.verify + 2)) + "\n")
            expected = directory / "expected.txt"
            actual = directory / "actual.txt"
            invoke([args.evaluator, "g8r-eval", "--input-irvals", vectors, "--initial-state-all-zeros",
                    case["design_path"]], directory, "simulate-aig", args.timeout, expected)
            invoke([args.evaluator, "gv-eval", "--sequential", "--clock-port-name", "clk",
                    "--initial-state-all-zeros", "--netlist", netlist, "--liberty_proto", args.proto,
                    "--input-irvals", vectors], directory, "simulate-netlist", args.timeout, actual)
            left, right = expected.read_text().splitlines(), actual.read_text().splitlines()
            if len(left) != args.verify + 2 or len(right) != len(left) or left[2:] != right[2:]:
                raise RuntimeError("random sequential equivalence mismatch")
            result["verified_vectors"] = args.verify
        return result

    results = parallel_cases(cases, worker, args.workers, output)
    fields = ["corpus", "design", "status", "area", "sequential_area", "combinational_area", "delay_ps", "synthesis_seconds", "error", "netlist"]
    with (output / "results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def summarize(args):
    reference = {(r["corpus"], r["design"]): r for r in json.loads(args.reference.read_text())}
    summaries = []
    for path in args.candidate:
        candidate = {(r["corpus"], r["design"]): r for r in json.loads(path.read_text())}
        paired = [(r, candidate[key]) for key, r in reference.items() if key in candidate
                  and r["status"] == candidate[key]["status"] == "ok"
                  and all(math.isfinite(float(x.get(k) or 0)) and float(x.get(k) or 0) > 0
                          for x in (r, candidate[key]) for k in ("area", "delay_ps"))]
        corpuses = {r["corpus"] for r in list(reference.values()) + list(candidate.values())}
        for corpus in sorted(corpuses) + ["combined"]:
            rows = [(r, c) for r, c in paired if corpus == "combined" or r["corpus"] == corpus]
            result = {"candidate": str(path), "corpus": corpus, "designs": len(rows)}
            for label, source in (("reference", reference), ("candidate", candidate)):
                result[label + "_status_counts"] = dict(Counter(
                    r["status"] for r in source.values()
                    if corpus == "combined" or r["corpus"] == corpus))
            result["paired_designs"] = [r["design"] for r, _ in rows]
            for metric in ("area", "combinational_area", "delay_ps"):
                valid = [(r[metric], c[metric]) for r, c in rows if r[metric] > 0 and c[metric] > 0]
                result[metric + "_geomean_delta_percent"] = 100 * (math.exp(sum(math.log(c / r) for r, c in valid) / len(valid)) - 1) if valid else None
                result[metric + "_sum_delta_percent"] = 100 * (sum(c[metric] for r, c in rows) / sum(r[metric] for r, c in rows) - 1) if sum(r[metric] for r, c in rows) else None
            result["reference_area_sum"] = sum(r["area"] for r, _ in rows)
            result["candidate_area_sum"] = sum(c["area"] for _, c in rows)
            result["reference_register_area_sum"] = sum(r["sequential_area"] for r, _ in rows)
            result["candidate_register_area_sum"] = sum(c["sequential_area"] for _, c in rows)
            summaries.append(result)
    write_json(args.output, summaries)
    print(json.dumps(summaries, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freezing = commands.add_parser("freeze")
    freezing.add_argument("--driver", type=Path, required=True)
    freezing.add_argument("--utility-driver", type=Path, required=True)
    freezing.add_argument("--corpus", action="append", default=[])
    freezing.add_argument("--selection", type=Path)
    mapping = commands.add_parser("run")
    mapping.add_argument("--inputs", type=Path, required=True)
    mapping.add_argument("--selection", type=Path)
    mapping.add_argument("--backend", choices=["abc", "native"], required=True)
    mapping.add_argument("--name", required=True)
    mapping.add_argument("--syn-command", choices=["&syn2", "&resyn3"], default="&syn2")
    for option in ("driver", "evaluator", "adapter", "abc", "proto", "constraints"):
        mapping.add_argument("--" + option, type=Path, required=True)
    mapping.add_argument("--liberty", type=Path, action="append", required=True)
    mapping.add_argument("--cell")
    mapping.add_argument("--verify-cell", default="DFF")
    for option, default in (("rounds", 5), ("resize-rounds", 3), ("resize-iterations", 16),
                            ("area-iterations", 32), ("max-fanout", 12), ("verify", 0)):
        mapping.add_argument("--" + option, type=int, default=default)
    for command in (freezing, mapping):
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--workers", type=int, default=8)
        command.add_argument("--timeout", type=int, default=600)
    summary = commands.add_parser("summarize")
    summary.add_argument("--reference", type=Path, required=True)
    summary.add_argument("--candidate", type=Path, action="append", required=True)
    summary.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"freeze": freeze, "run": run, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
