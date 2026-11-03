#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Emit ABC diagnostics and the invalid netlist selected by the test."""

import shutil
import sys
from pathlib import Path

if sys.argv[1:] == ["-s", "-c", "version"]:
    print("test ABC")
    sys.exit(0)

print("mapping stdout")
print("mapping stderr", file=sys.stderr)
# Each case puts its netlist beside the symlink used to invoke this fixture.
shutil.copyfile(Path(sys.argv[0]).with_name("output.v"), "mapped.v")
