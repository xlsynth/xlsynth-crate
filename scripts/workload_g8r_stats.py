#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Generate g8r stats bundle for a selected BF16 workload.

Outputs into --out-dir:
- <workload>.x (DSLX)
- <workload>.ir, <workload>.opt.ir
- <workload>.stripped.opt.ir (optimized IR without position data)
- <workload>.g8r (GateFn text), <workload>.g8r.bin (bincode)
- <workload>.aig (binary AIGER emitted from GateFn; suitable for Berkeley ABC `read_aiger`)
- <workload>.g8r.ir (XLS IR package reconstructed from the GateFn)
- <workload>.gv (gate-level netlist emitted from GateFn)
- <workload>.combo.v (IR-level combinational SystemVerilog codegen via ir2combo)
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
  python scripts/workload_g8r_stats.py --workload clz_10   --out-dir /tmp/clzstats
  python scripts/workload_g8r_stats.py --workload clzt_10  --out-dir /tmp/clztstats
  python scripts/workload_g8r_stats.py --workload popcount_32 --out-dir /tmp/popcountstats
  # If the driver is not at target/release/xlsynth-driver (e.g. macOS target triple),
  # pass an explicit path:
  python scripts/workload_g8r_stats.py --bin target/aarch64-apple-darwin/release/xlsynth-driver --workload bf16_add --out-dir /tmp/bf16stats
  # If --workload is omitted, prints a summary table for all known workloads:
  #   workload nodes depth

Cut DB:
- The `g8r` pipeline embeds the in-tree cut database by default; this script does
  not need to pass any cut-db path.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_xlsynth_driver(repo_root: Path) -> Path:
    # Prefer a locally built release driver; fall back to debug if present.
    #
    # On macOS, cargo often places artifacts under target/<triple>/release/.
    target_dir = repo_root / "target"
    candidates = [target_dir / "release" / "xlsynth-driver"]
    if target_dir.is_dir():
        candidates.extend(sorted(target_dir.glob("*/release/xlsynth-driver")))
        # candidates.extend(sorted(target_dir.glob("*/debug/xlsynth-driver")))
    # candidates.append(target_dir / "debug" / "xlsynth-driver")
    for cand in candidates:
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand
    raise FileNotFoundError(
        "xlsynth-driver not found. Expected at:\n"
        f"  {candidates[0]}\n"
        "Or under a target triple directory, e.g.:\n"
        f"  {repo_root / 'target' / '<triple>' / 'release' / 'xlsynth-driver'}\n"
        "Or pass an explicit path via --bin.\n"
        "Please build it first, e.g.:\n"
        "  cargo build -p xlsynth-driver --release"
    )


def _resolve_xlsynth_driver_from_arg(bin_arg: str) -> Path:
    p = Path(bin_arg).expanduser()
    if not p.is_absolute():
        p = (_repo_root() / p).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"--bin does not exist or is not a file: {p}")
    if not os.access(p, os.X_OK):
        raise FileNotFoundError(f"--bin is not executable: {p}")
    return p


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


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(s: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", s)


_ABC_PRINT_STATS_RE = re.compile(r"\band\s*=\s*(\d+)\s+lev\s*=\s*(\d+)\b")


def _run_abc_cmd_and_get_stats(abc_exe: str, abc_cmd: str) -> Optional[Dict[str, int]]:
    # If abc isn't available we just skip (no-op).
    if not abc_exe:
        return None
    try:
        # ABC writes an `abc.history` file in its working directory. Run it in a
        # temporary directory so the repo doesn't get dirtied (and pre-commit
        # / SPDX checks don't fail).
        with tempfile.TemporaryDirectory(prefix="xlsynth-abc-") as tmpdir:
            res = _run_cmd(
                [abc_exe, "-c", abc_cmd],
                cwd=Path(tmpdir),
                capture_stdout=True,
                check=False,
            )
    except Exception:
        return None
    text = _strip_ansi((res.stdout or "") + "\n" + (res.stderr or ""))
    # Example line:
    #   /tmp/foo : i/o = 32/16 lat = 0 and = 1059 lev = 86
    m = _ABC_PRINT_STATS_RE.search(text)
    if not m:
        return None
    return {"and": int(m.group(1)), "lev": int(m.group(2))}


def _run_abc_baseline_and_get_stats(
    abc_exe: str, aig_path: Path
) -> Optional[Dict[str, int]]:
    # Baseline: read + strash + stats (keeps it consistent with the opt flow).
    abc_cmd = f"read_aiger {aig_path}; strash; print_stats"
    return _run_abc_cmd_and_get_stats(abc_exe, abc_cmd)


def _run_abc_opt_and_get_stats(
    abc_exe: str, aig_path: Path
) -> Optional[Dict[str, int]]:
    # Avoid relying on user-provided aliases (e.g. "resyn2") by using an
    # explicit command sequence.
    #
    # Note: ABC's `print_stats` reports `and` (AND-gate count) and `lev`
    # (logic depth / levels).
    abc_cmd = (
        f"read_aiger {aig_path}; "
        "strash; "
        "balance; rewrite; refactor; rewrite; balance; "
        "print_stats"
    )
    return _run_abc_cmd_and_get_stats(abc_exe, abc_cmd)


def _format_signed_int(v: int) -> str:
    return f"{v:+d}"


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


def _clz10_dslx() -> str:
    return (
        "fn main(x: u10) -> u4 {\n"
        "    // 'clz' is the built-in count-leading-zeros (not the std::clzt helper).\n"
        "    // Cast to u4 so the IO shape matches the clzt_10 workload.\n"
        "    (clz(x) as u4)\n"
        "}\n"
    )


def _popcount_32_dslx() -> str:
    return (
        "import std;\n"
        "\n"
        "fn main(x: u32) -> u6 {\n"
        "    // std::popcount returns bits[N]; cast down to the needed count width.\n"
        "    (std::popcount(x) as u6)\n"
        "}\n"
    )


def _abs_diff_8_dslx() -> str:
    return (
        "import abs_diff;\n"
        "\n"
        "fn main(x: u8, y: u8) -> abs_diff::AbsDiffResult<8> {\n"
        "    abs_diff::abs_diff(x, y)\n"
        "}\n"
    )


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


KNOWN_WORKLOADS = (
    "bf16_add",
    "bf16_mul",
    "clz_10",
    "clzt_10",
    "abs_diff_8",
    "popcount_32",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate g8r stats bundle for BF16 workloads."
    )
    parser.add_argument(
        "--bin",
        required=False,
        help="Path to the xlsynth-driver executable to use (overrides auto-discovery).",
    )
    parser.add_argument(
        "--workload",
        required=False,
        choices=list(KNOWN_WORKLOADS),
        help="Workload to generate (bf16_add|bf16_mul|clz_10|clzt_10|abs_diff_8|popcount_32). If omitted, prints a summary table for all.",
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
        if args.bin:
            driver = _resolve_xlsynth_driver_from_arg(args.bin)
        else:
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

    xlsynth_tools_env = os.environ.get("XLSYNTH_TOOLS")

    # File paths
    base = args.workload
    if base is None:
        # Summary mode: print table of nodes/depth for all known workloads.
        abc_exe = shutil.which("abc") or ""
        print("workload nodes depth equiv abc_and abc_lev delta_and delta_lev")
        dslx_path_env = os.environ.get("DSLX_PATH")
        for wl in KNOWN_WORKLOADS:
            with tempfile.TemporaryDirectory(prefix=f"g8r_{wl}_") as tmpd:
                tmp = Path(tmpd)
                tmp_dslx = tmp / f"{wl}.x"
                if wl in ("bf16_add", "bf16_mul"):
                    op = "add" if wl == "bf16_add" else "mul"
                    _write_text(tmp_dslx, _bf16_dslx(op))
                elif wl == "clz_10":
                    _write_text(tmp_dslx, _clz10_dslx())
                elif wl == "clzt_10":
                    _write_text(tmp_dslx, _clzt10_dslx())
                elif wl == "popcount_32":
                    _write_text(tmp_dslx, _popcount_32_dslx())
                elif wl == "abs_diff_8":
                    _write_text(tmp_dslx, _abs_diff_8_dslx())
                else:
                    raise ValueError(f"unhandled workload {wl}")
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
                tmp_aig = tmp / f"{wl}.aig"
                try:
                    ir2g8r_args = [
                        str(driver),
                        "ir2g8r",
                        str(tmp_opt_ir),
                        "--stats-out",
                        str(tmp_stats),
                        "--aiger-out",
                        str(tmp_aig),
                        "--bin-out",
                        str(tmp_g8r_bin),
                    ]
                    res_ir2g8r = _run_cmd(
                        ir2g8r_args,
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
                    abc_base = _run_abc_baseline_and_get_stats(abc_exe, tmp_aig)
                    abc_opt = _run_abc_opt_and_get_stats(abc_exe, tmp_aig)

                    if abc_base is not None and abc_opt is not None:
                        abc_and = abc_opt["and"]
                        abc_lev = abc_opt["lev"]
                        delta_and = _format_signed_int(abc_opt["and"] - abc_base["and"])
                        delta_lev = _format_signed_int(abc_opt["lev"] - abc_base["lev"])
                    else:
                        abc_and = "-"
                        abc_lev = "-"
                        delta_and = "-"
                        delta_lev = "-"

                    print(
                        f"{wl} {nodes} {depth} {equiv_status} {abc_and} {abc_lev} {delta_and} {delta_lev}"
                    )
                except subprocess.CalledProcessError as e:
                    sys.stderr.write(e.stderr or "")
                    print(f"{wl} - - error - - - -")
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
    aiger_aig_path = out_dir / f"{base}.aig"
    g8r_ir_path = out_dir / f"{base}.g8r.ir"
    gv_path = out_dir / f"{base}.gv"
    codegen_combo_path = out_dir / f"{base}.combo.v"
    report_txt_path = out_dir / "report.txt"
    stats_json_path = out_dir / "stats.json"
    run_json_path = out_dir / "run.json"
    netlist_flags: list[str] = ["--netlist-out", str(gv_path)]
    toolchain_flag: list[str] = []
    toolchain_meta: Optional[str] = None
    if xlsynth_tools_env:
        toolchain_path = out_dir / "xlsynth-toolchain.toml"
        tool_path_value = Path(xlsynth_tools_env).expanduser().resolve()
        toolchain_content = (
            "[toolchain]\n" f"tool_path = {json.dumps(str(tool_path_value))}\n"
        )
        _write_text(toolchain_path, toolchain_content)
        toolchain_flag = ["--toolchain", str(toolchain_path)]
        toolchain_meta = str(toolchain_path)
    combo_generated = False

    # 1) Emit DSLX
    if args.workload in ("bf16_add", "bf16_mul"):
        op = "add" if args.workload == "bf16_add" else "mul"
        _write_text(dslx_path, _bf16_dslx(op))
    elif args.workload == "clz_10":
        _write_text(dslx_path, _clz10_dslx())
    elif args.workload == "clzt_10":
        _write_text(dslx_path, _clzt10_dslx())
    elif args.workload == "popcount_32":
        _write_text(dslx_path, _popcount_32_dslx())
    elif args.workload == "abs_diff_8":
        _write_text(dslx_path, _abs_diff_8_dslx())
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
        ir2gates_args = [str(driver), "ir2gates", str(opt_ir_path)]
        res = _run_cmd(ir2gates_args, cwd=repo)
        _write_text(report_txt_path, res.stdout)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(e.stderr or "")
        sys.stderr.write("error: ir2gates (report) failed\n")
        return 3

    # 4) ir2g8r (stats/netlist/bin; GateFn text goes to stdout)
    try:
        ir2g8r_args = [
            str(driver),
            "ir2g8r",
            str(opt_ir_path),
            "--stats-out",
            str(stats_json_path),
            "--aiger-out",
            str(aiger_aig_path),
            *netlist_flags,
            "--bin-out",
            str(g8r_bin_path),
        ]
        res = _run_cmd(ir2g8r_args, cwd=repo)
        _write_text(g8r_txt_path, res.stdout)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(e.stderr or "")
        sys.stderr.write("error: ir2g8r failed\n")
        return 4

    # 4b) g8r2ir (reconstruct XLS IR package corresponding to the GateFn)
    try:
        res = _run_cmd(
            [
                str(driver),
                "g8r2ir",
                str(g8r_txt_path),
            ],
            cwd=repo,
        )
        _write_text(g8r_ir_path, res.stdout)
    except subprocess.CalledProcessError as e:
        sys.stderr.write(e.stderr or "")
        sys.stderr.write("error: g8r2ir failed\n")
        return 4

    combo_generated = False
    if toolchain_flag:
        try:
            res = _run_cmd(
                [
                    str(driver),
                    *toolchain_flag,
                    "ir2combo",
                    str(opt_ir_path),
                    "--delay_model",
                    "unit",
                    "--use_system_verilog",
                    "true",
                ],
                cwd=repo,
            )
            _write_text(codegen_combo_path, res.stdout)
            combo_generated = True
        except subprocess.CalledProcessError as e:
            sys.stderr.write(e.stderr or "")
            sys.stderr.write("error: ir2combo failed\n")
            return 3
    else:
        sys.stderr.write(
            "info: skipping ir2combo because no toolchain configuration is available.\n"
        )

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
            "aiger_aig": str(aiger_aig_path),
            "g8r_ir": str(g8r_ir_path),
            "netlist_gv": str(gv_path),
            "codegen_combo_v": str(codegen_combo_path) if combo_generated else None,
            "stats_json": str(stats_json_path),
            "proof_json": str(proof_json_path),
        },
        "git": _collect_git_info(repo),
        "xlsynth_driver": _collect_driver_version(repo, driver),
        "cargo_metadata": _collect_cargo_metadata(repo),
        "env": _collect_env(),
        "toolchain": toolchain_meta,
    }
    run_json_path.write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
