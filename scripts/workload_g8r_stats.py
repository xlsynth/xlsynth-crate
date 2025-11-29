#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generate g8r stats bundle for a selected BF16 workload.

Outputs into --out-dir:
- <workload>.x (DSLX)
- <workload>.ir, <workload>.opt.ir
- <workload>.stripped.opt.ir (optimized IR without position data)
- <workload>.g8r (GateFn text), <workload>.g8r.bin (bincode)
- <workload>.sv (netlist)
- report.txt (human-readable ir2gates report incl. repeated structures)
- stats.json (core stats only; no toggle data)
- run.json (run metadata)

Environment:
- Requires DSLX_STDLIB_PATH to point to the DSLX stdlib directory
  (path ending with xls/dslx/stdlib/). Optionally, DSLX_PATH can provide
  additional search paths (semicolon-separated).

Usage:
  python scripts/workload_g8r_stats.py --workload bf16_add --out-dir /tmp/bf16stats
  python scripts/workload_g8r_stats.py --workload bf16_mul --out-dir /tmp/bf16stats
  python scripts/workload_g8r_stats.py --workload clzt_10  --out-dir /tmp/clztstats
  # If --workload is omitted, prints a summary table for all known workloads:
  #   workload nodes depth
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_xlsynth_driver(repo_root: Path) -> Path:
    # Prefer a locally built release driver; fall back to debug if present.
    candidates = [
        repo_root / "target" / "release" / "xlsynth-driver",
        # repo_root / "target" / "debug" / "xlsynth-driver",
    ]
    for cand in candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand
    raise FileNotFoundError(
        "xlsynth-driver not found. Expected at:\n"
        f"  {candidates[0]}\n"
        "Please build it first, e.g.:\n"
        "  cargo build -p xlsynth-driver --release"
    )


def _run_cmd(
    args: list[str],
    *,
    cwd: Optional[Path] = None,
    capture_stdout: bool = True,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd) if cwd is not None else None,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture_stdout else None,
        stderr=subprocess.PIPE,
    )


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _bf16_dslx(op: str) -> str:
    # Keep formatting minimal and stable.
    return (
        "import bfloat16;\n"
        "\n"
        "fn main(x: bfloat16::BF16, y: bfloat16::BF16) -> bfloat16::BF16 {\n"
        f"    bfloat16::{op}(x, y)\n"
        "}\n"
    )


def _clzt10_dslx() -> str:
    return "import std;\n" "\n" "fn main(x: u10) -> u4 {\n" "    std::clzt(x)\n" "}\n"


def _collect_git_info(cwd: Path) -> Dict[str, object]:
    info: Dict[str, object] = {}
    try:
        sha = _run_cmd(["git", "rev-parse", "HEAD"], cwd=cwd).stdout.strip()
        branch = _run_cmd(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd
        ).stdout.strip()
        dirty_out = _run_cmd(["git", "status", "--porcelain"], cwd=cwd).stdout
        info.update(
            {"head": sha, "branch": branch, "dirty": len(dirty_out.strip()) > 0}
        )
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def _collect_driver_version(cwd: Path, driver_path: Path) -> Dict[str, object]:
    try:
        out = _run_cmd([str(driver_path), "version"], cwd=cwd).stdout.strip()
        return {"version": out, "exe": str(driver_path)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _collect_cargo_metadata(cwd: Path) -> Dict[str, object]:
    try:
        out = _run_cmd(
            ["cargo", "metadata", "--format-version", "1", "--no-deps"],
            cwd=cwd,
        ).stdout
        return json.loads(out)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _collect_env() -> Dict[str, Optional[str]]:
    keys = ["XLS_DSO_PATH", "DSLX_STDLIB_PATH"]
    return {k: os.environ.get(k) for k in keys}


KNOWN_WORKLOADS = ("bf16_add", "bf16_mul", "clzt_10")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate g8r stats bundle for BF16 workloads."
    )
    parser.add_argument(
        "--workload",
        required=False,
        choices=list(KNOWN_WORKLOADS),
        help="Workload to generate (bf16_add|bf16_mul|clzt_10). If omitted, prints a summary table for all.",
    )
    parser.add_argument(
        "--out-dir",
        required=False,
        help="Output directory to populate with artifacts.",
    )
    args = parser.parse_args()

    repo = _repo_root()

    # Resolve and require a locally built xlsynth-driver binary.
    try:
        driver = _resolve_xlsynth_driver(repo)
    except FileNotFoundError as e:
        sys.stderr.write(f"error: {e}\n")
        return 1

    # Ensure DSLX stdlib is discoverable.
    dslx_stdlib_path = os.environ.get("DSLX_STDLIB_PATH")
    if not dslx_stdlib_path:
        sys.stderr.write(
            "error: DSLX_STDLIB_PATH is not set.\n"
            "Please set DSLX_STDLIB_PATH to the DSLX stdlib directory "
            "(a path ending with xls/dslx/stdlib/), then re-run.\n"
        )
        return 1

    # File paths
    base = args.workload
    if base is None:
        # Summary mode: print table of nodes/depth for all known workloads.
        print("workload nodes depth equiv")
        dslx_path_env = os.environ.get("DSLX_PATH")
        for wl in KNOWN_WORKLOADS:
            with tempfile.TemporaryDirectory(prefix=f"g8r_{wl}_") as tmpd:
                tmp = Path(tmpd)
                tmp_dslx = tmp / f"{wl}.x"
                if wl in ("bf16_add", "bf16_mul"):
                    op = "add" if wl == "bf16_add" else "mul"
                    _write_text(tmp_dslx, _bf16_dslx(op))
                else:
                    _write_text(tmp_dslx, _clzt10_dslx())
                # Get optimized IR via dslx2ir --opt
                try:
                    dslx2ir_opt_args = [
                        str(driver),
                        "dslx2ir",
                        "--dslx_input_file",
                        str(tmp_dslx),
                        "--dslx_top",
                        "main",
                        "--opt",
                        "true",
                    ]
                    if dslx_stdlib_path:
                        dslx2ir_opt_args += ["--dslx_stdlib_path", dslx_stdlib_path]
                    if dslx_path_env:
                        dslx2ir_opt_args += ["--dslx_path", dslx_path_env]
                    res = _run_cmd(dslx2ir_opt_args, cwd=repo)
                    tmp_opt_ir = tmp / f"{wl}.opt.ir"
                    _write_text(tmp_opt_ir, res.stdout)
                except subprocess.CalledProcessError as e:
                    sys.stderr.write(e.stderr or "")
                    print(f"{wl} - -")
                    continue
                # Compute stats via ir2g8r --stats-out and emit GateFn text/bin to temp.
                tmp_stats = tmp / "stats.json"
                tmp_g8r_txt = tmp / f"{wl}.g8r"
                tmp_g8r_bin = tmp / f"{wl}.g8rbin"
                try:
                    res_ir2g8r = _run_cmd(
                        [
                            str(driver),
                            "ir2g8r",
                            str(tmp_opt_ir),
                            "--stats-out",
                            str(tmp_stats),
                            "--bin-out",
                            str(tmp_g8r_bin),
                        ],
                        cwd=repo,
                        capture_stdout=True,
                    )
                    # Capture GateFn text to a temp file.
                    _write_text(tmp_g8r_txt, res_ir2g8r.stdout)
                    stats = json.loads(tmp_stats.read_text(encoding="utf-8"))
                    nodes = stats.get("live_nodes", "-")
                    depth = stats.get("deepest_path", "-")
                    # Prove equivalence between text and binary encodings via g8r-equiv.
                    equiv_status = "error"
                    try:
                        res_equiv = _run_cmd(
                            [
                                str(driver),
                                "g8r-equiv",
                                str(tmp_g8r_txt),
                                str(tmp_g8r_bin),
                            ],
                            cwd=repo,
                            capture_stdout=True,
                            check=False,
                        )
                        if res_equiv.returncode == 0:
                            equiv_status = "true"
                        elif res_equiv.returncode == 1:
                            equiv_status = "false"
                        else:
                            equiv_status = "error"
                    except Exception:
                        equiv_status = "error"
                    print(f"{wl} {nodes} {depth} {equiv_status}")
                except subprocess.CalledProcessError as e:
                    sys.stderr.write(e.stderr or "")
                    print(f"{wl} - - error")
        return 0
    # Require and prepare output directory when a specific workload is requested.
    if not args.out_dir:
        sys.stderr.write("error: --out-dir is required when --workload is provided.\n")
        return 1
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dslx_path = out_dir / f"{base}.x"
    ir_path = out_dir / f"{base}.ir"
    opt_ir_path = out_dir / f"{base}.opt.ir"
    opt_ir_stripped_path = out_dir / f"{base}.stripped.opt.ir"
    g8r_txt_path = out_dir / f"{base}.g8r"
    g8r_bin_path = out_dir / f"{base}.g8rbin"
    sv_path = out_dir / f"{base}.sv"
    report_txt_path = out_dir / "report.txt"
    stats_json_path = out_dir / "stats.json"
    run_json_path = out_dir / "run.json"

    # 1) Emit DSLX
    if args.workload in ("bf16_add", "bf16_mul"):
        op = "add" if args.workload == "bf16_add" else "mul"
        _write_text(dslx_path, _bf16_dslx(op))
    elif args.workload == "clzt_10":
        _write_text(dslx_path, _clzt10_dslx())
    else:
        sys.stderr.write(f"error: unsupported workload: {args.workload}\n")
        return 2

    # 2) dslx2ir
    try:
        dslx2ir_args = [
            str(driver),
            "dslx2ir",
            "--dslx_input_file",
            str(dslx_path),
            "--dslx_top",
            "main",
        ]
        # If env provides stdlib or search path, pass them through as flags.
        if dslx_stdlib_path:
            dslx2ir_args += ["--dslx_stdlib_path", dslx_stdlib_path]
        dslx_path_env = os.environ.get("DSLX_PATH")
        if dslx_path_env:
            dslx2ir_args += ["--dslx_path", dslx_path_env]
        res = _run_cmd(dslx2ir_args, cwd=repo)
        _write_text(ir_path, res.stdout)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(e.stderr or "")
        sys.stderr.write("error: dslx2ir failed\n")
        return 2

    # 3) optimized IR via dslx2ir --opt (avoids needing the IR-level top symbol)
    try:
        dslx2ir_opt_args = [
            str(driver),
            "dslx2ir",
            "--dslx_input_file",
            str(dslx_path),
            "--dslx_top",
            "main",
            "--opt",
            "true",
        ]
        if dslx_stdlib_path:
            dslx2ir_opt_args += ["--dslx_stdlib_path", dslx_stdlib_path]
        dslx_path_env = os.environ.get("DSLX_PATH")
        if dslx_path_env:
            dslx2ir_opt_args += ["--dslx_path", dslx_path_env]
        res = _run_cmd(dslx2ir_opt_args, cwd=repo)
        _write_text(opt_ir_path, res.stdout)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(e.stderr or "")
        sys.stderr.write("error: dslx2ir --opt failed\n")
        return 3

    # 3a) Strip position data from optimized IR
    try:
        res = _run_cmd(
            [
                str(driver),
                "ir-strip-pos-data",
                str(opt_ir_path),
            ],
            cwd=repo,
        )
        _write_text(opt_ir_stripped_path, res.stdout)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(e.stderr or "")
        sys.stderr.write("error: ir-strip-pos-data failed\n")
        return 3

    # 3b) ir2gates (human-readable report to stdout; capture to report.txt)
    try:
        res = _run_cmd(
            [
                str(driver),
                "ir2gates",
                str(opt_ir_path),
            ],
            cwd=repo,
        )
        _write_text(report_txt_path, res.stdout)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(e.stderr or "")
        sys.stderr.write("error: ir2gates (report) failed\n")
        return 3

    # 4) ir2g8r (stats/netlist/bin; GateFn text goes to stdout)
    try:
        res = _run_cmd(
            [
                str(driver),
                "ir2g8r",
                str(opt_ir_path),
                "--stats-out",
                str(stats_json_path),
                "--netlist-out",
                str(sv_path),
                "--bin-out",
                str(g8r_bin_path),
            ],
            cwd=repo,
        )
        _write_text(g8r_txt_path, res.stdout)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(e.stderr or "")
        sys.stderr.write("error: ir2g8r failed\n")
        return 4

    # 4a) Prove GateFn equivalence across engines and capture JSON report.
    # We compare the text-vs-binary serializations using the driver's g8r-equiv.
    # Note: This runs multiple engines (including an IR-based checker when configured)
    # and writes a JSON report to proof.json; we don't fail the overall script if the
    # prover exits non-zero, but we do capture its stdout/stderr for postmortem.
    proof_json_path = out_dir / "proof.json"
    try:
        res = _run_cmd(
            [
                str(driver),
                "g8r-equiv",
                str(g8r_txt_path),
                str(g8r_bin_path),
            ],
            cwd=repo,
            check=False,  # Do not make the entire script fail if any engine disagrees.
        )
        _write_text(proof_json_path, res.stdout or "")
        if res.returncode != 0 and (res.stderr or "").strip():
            sys.stderr.write(res.stderr)
    except Exception as e:
        sys.stderr.write(f"warning: failed to run g8r-equiv: {type(e).__name__}: {e}\n")
        # Still continue; proof.json may be empty or missing.

    # 5) Metadata
    run_meta = {
        "workload": args.workload,
        "out_dir": str(out_dir),
        "files": {
            "dslx": str(dslx_path),
            "ir": str(ir_path),
            "opt_ir": str(opt_ir_path),
            "opt_ir_stripped": str(opt_ir_stripped_path),
            "g8r_txt": str(g8r_txt_path),
            "g8r_bin": str(g8r_bin_path),
            "netlist_sv": str(sv_path),
            "stats_json": str(stats_json_path),
            "proof_json": str(proof_json_path),
        },
        "git": _collect_git_info(repo),
        "xlsynth_driver": _collect_driver_version(repo, driver),
        "cargo_metadata": _collect_cargo_metadata(repo),
        "env": _collect_env(),
    }
    run_json_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
