# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""
Generate g8r stats bundle for a selected BF16 workload.

Outputs into --out-dir:
- <workload>.x (DSLX)
- <workload>.ir, <workload>.opt.ir
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
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


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


def _collect_driver_version(cwd: Path) -> Dict[str, object]:
    try:
        out = _run_cmd(["xlsynth-driver", "version"], cwd=cwd).stdout.strip()
        return {"version": out}
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate g8r stats bundle for BF16 workloads."
    )
    parser.add_argument(
        "--workload",
        required=True,
        choices=["bf16_add", "bf16_mul", "clzt_10"],
        help="Workload to generate (bf16_add|bf16_mul|clzt_10).",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory to populate with artifacts.",
    )
    args = parser.parse_args()

    repo = _repo_root()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

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
    dslx_path = out_dir / f"{base}.x"
    ir_path = out_dir / f"{base}.ir"
    opt_ir_path = out_dir / f"{base}.opt.ir"
    g8r_txt_path = out_dir / f"{base}.g8r"
    g8r_bin_path = out_dir / f"{base}.g8r.bin"
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
            "xlsynth-driver",
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
            "xlsynth-driver",
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

    # 3b) ir2gates (human-readable report to stdout; capture to report.txt)
    try:
        res = _run_cmd(
            [
                "xlsynth-driver",
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
                "xlsynth-driver",
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

    # 5) Metadata
    run_meta = {
        "workload": args.workload,
        "out_dir": str(out_dir),
        "files": {
            "dslx": str(dslx_path),
            "ir": str(ir_path),
            "opt_ir": str(opt_ir_path),
            "g8r_txt": str(g8r_txt_path),
            "g8r_bin": str(g8r_bin_path),
            "netlist_sv": str(sv_path),
            "stats_json": str(stats_json_path),
        },
        "git": _collect_git_info(repo),
        "xlsynth_driver": _collect_driver_version(repo),
        "cargo_metadata": _collect_cargo_metadata(repo),
        "env": _collect_env(),
    }
    run_json_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
