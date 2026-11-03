#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check the caller's mapping script and staged inputs before emitting an inverter."""

import sys
from pathlib import Path

if sys.argv[1:] == ["-s", "-c", "version"]:
    sys.exit(0)
if sys.argv[1:] != ["-s", "-f", "map.abc"]:
    sys.exit(2)

script = Path("map.abc")
golden = Path(__file__).parent.parent / "custom_script.golden.txt"
try:
    matches_golden = script.read_bytes() == golden.read_bytes()
except OSError as error:
    print(error, file=sys.stderr)
    sys.exit(3)
if not matches_golden:
    print(f"{script} differs from {golden.name}", file=sys.stderr)
    sys.exit(3)

aig = Path("transition.aig")
if not aig.is_file() or aig.stat().st_size == 0:
    sys.exit(4)

constraints = Path("timing.constr")
expected = b"set_driving_cell BUF\nset_load 4"
if not constraints.is_file() or constraints.read_bytes().rstrip(b"\n") != expected:
    sys.exit(5)

Path("mapped.v").write_text(
    "module mapped(i0,o0); input i0; output o0; INV c(.A(i0),.Y(o0)); endmodule\n"
)
