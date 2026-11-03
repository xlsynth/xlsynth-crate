#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Mimic ABC warning about a missing driving cell while still exiting successfully."""

import sys
from pathlib import Path

if sys.argv[1:] == ["-s", "-c", "version"]:
    sys.exit(0)

constraints = Path("timing.constr")
expected = b"set_driving_cell BUF\nset_load 4"
if not constraints.is_file() or constraints.read_bytes().rstrip(b"\n") != expected:
    sys.exit(3)

print("Cannot find the default PI driving cell (BUF) in the library.")
Path("mapped.v").write_text("module mapped(); endmodule\n")
