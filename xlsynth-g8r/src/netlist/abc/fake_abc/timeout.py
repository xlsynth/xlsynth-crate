#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Answer the version probe, then outlast the mapping timeout."""

import sys
import time

if sys.argv[1:] == ["-s", "-c", "version"]:
    sys.exit(0)

time.sleep(30)
