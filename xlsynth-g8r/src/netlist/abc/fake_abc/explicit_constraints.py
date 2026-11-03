#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Map an inverter only when the caller's explicit load reaches ABC."""

import sys
from pathlib import Path

if sys.argv[1:] == ["-s", "-c", "version"]:
    sys.exit(0)

constraints = Path("timing.constr")
expected = b"set_driving_cell BUF\nset_load 20"
if not constraints.is_file() or constraints.read_bytes().rstrip(b"\n") != expected:
    sys.exit(3)

Path("mapped.v").write_text(
    "module mapped(i0,o0); input i0; output o0; INV c(.A(i0),.Y(o0)); endmodule\n"
)
