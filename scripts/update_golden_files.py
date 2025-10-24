#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import subprocess
import os
import sys

# Path to the xlsynth-driver executable (anchored at repo root)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
XLSYNTH_DRIVER_EXE = os.path.join(REPO_ROOT, "target", "debug", "xlsynth-driver")

# Environment variable to trigger golden file updates in Rust tests
UPDATE_ENV_VAR = "XLSYNTH_UPDATE_GOLDEN"


def main():
    # Check if driver executable exists (needed by tests)
    if not os.path.exists(XLSYNTH_DRIVER_EXE):
        print(f"ERROR: '{XLSYNTH_DRIVER_EXE}' not found.", file=sys.stderr)
        print("Please build the driver first (e.g., 'cargo build').", file=sys.stderr)
        sys.exit(1)
    print(f"Found {XLSYNTH_DRIVER_EXE}.")

    # Set environment variable to trigger update mode in Rust tests
    os.environ[UPDATE_ENV_VAR] = "1"

    success = False
    try:
        # Run only the invoke_test integration tests within the xlsynth-driver package
        test_command = [
            "cargo",
            "test",
            "-p",
            "xlsynth-driver",
            "--test",
            "invoke_test",
        ]
        print(f"Running command: {' '.join(test_command)}")
        result = subprocess.run(
            test_command, check=False
        )  # Don't check=True, handle failure below

        if result.returncode == 0:
            print("\nTest run successful, golden files updated.")
            success = True
        else:
            print(
                "\nERROR: Test run failed. See output above for details.",
                file=sys.stderr,
            )
            success = False

    except Exception as e:
        print(f"ERROR: Failed to run cargo test: {e}", file=sys.stderr)
        success = False
    finally:
        # Ensure environment variable is removed regardless of success/failure
        os.environ.pop(UPDATE_ENV_VAR, None)
        print(f"Cleaned up environment variable {UPDATE_ENV_VAR}.")

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
