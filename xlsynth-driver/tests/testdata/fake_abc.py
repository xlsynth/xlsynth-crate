#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Emit a scalar Verilog interface with each output tied to zero for g8r2gv tests."""

import os
import sys
from pathlib import Path


def main():
    args = sys.argv[1:]
    if args == ["-s", "-c", "version"]:
        print("fake ABC")
        return 0
    if args != ["-s", "-f", "map.abc"]:
        return 2

    aig = Path("transition.aig")
    for path in (aig, Path("timing.constr")):
        if not path.is_file() or path.stat().st_size == 0:
            return 3
    if not Path("library_0.lib").is_file():
        return 3

    # Only the AIGER header (aig M I L O A) is guaranteed to be ASCII.
    with aig.open("rb") as stream:
        header = stream.readline().split()
    inputs = [f"i{i}" for i in range(int(header[2]))]
    outputs = [f"o{i}" for i in range(int(header[4]))]

    sys.stdout.buffer.write(os.environb.get(b"XLSYNTH_TEST_ABC_STDOUT", b""))
    sys.stderr.buffer.write(os.environb.get(b"XLSYNTH_TEST_ABC_STDERR", b""))
    with open("mapped.v", "w", encoding="ascii") as verilog:
        print(f"module mapped({','.join(inputs + outputs)});", file=verilog)
        for port in inputs:
            print(f"input {port};", file=verilog)
        for port in outputs:
            print(f"output {port}; assign {port}=1'b0;", file=verilog)
        print("endmodule", file=verilog)
    return 0


if __name__ == "__main__":
    sys.exit(main())
