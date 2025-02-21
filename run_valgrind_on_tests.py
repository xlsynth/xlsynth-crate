#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import glob
import sys
import re
import json

import termcolor

def get_target_to_cwd_mapping():
    """
    Returns a dict mapping a test target name to its package root directory.
    We use 'cargo metadata' to get information on each package.
    """
    p = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        capture_output=True,
        check=True
    )
    metadata = json.loads(p.stdout.decode("utf-8"))
    mapping = {}
    for package in metadata["packages"]:
        # The package root is the directory containing Cargo.toml.
        package_root = os.path.dirname(package["manifest_path"])
        for target in package["targets"]:
            # We assume test targets have kind "test" or "integration-test".
            if any(kind in target["kind"] for kind in ("test", "integration-test")):
                mapping[target["name"]] = package_root
    return mapping

def run_xlsynth_test(exe, cwd):
    """
    For a given test executable, run a subset of tests (skipping the ones 
    with known issues) under valgrind. The binary is run from the provided cwd.
    """
    # Get the list of tests from the executable.
    output = subprocess.check_output([exe, "--test", "--list"], cwd=cwd).decode("utf-8")
    lines = output.splitlines()
    to_run = []
    # Skip the header line.
    for line in lines[1:]:
        if not line.strip():
            continue
        if 'benchmark' in line:
            continue
        lhs, rhs = line.rsplit(': ', 1)
        assert rhs == 'test', line
        if "bridge_builder" in lhs:
            continue
        to_run.append(lhs)
    
    test_str = " ".join(to_run)
    termcolor.cprint(f"Running valgrind on subset of {exe} ...", "yellow")
    subprocess.run(
        ["valgrind", "--suppressions=valgrind.supp", "--leak-check=full", exe, test_str],
        check=True,
        cwd=cwd
    )

def main():
    # Compile all tests in the workspace.
    print("Compiling tests for package 'xlsynth' with cargo test --no-run ...")
    p = subprocess.run(
        ["cargo", "test", "--no-run", "--workspace"],
        check=True,
        stderr=subprocess.PIPE
    )
    output = p.stderr.decode("utf-8")

    # Get the test binary paths from the cargo output.
    test_binaries = []
    for line in output.splitlines():
        if "Executable" in line and "(" in line:
            # Skip unwanted binaries.
            if 'spdx' in line:
                continue
            if any(x in line for x in ['readme_test', 'version_test', 'sample_usage', 'ir_interpret_test', 'sv_bridge_test']):
                continue
            test_binaries.append(line.split("(")[1].rstrip(")"))

    if not test_binaries:
        print("No test executables found")
        sys.exit(1)

    print('Test binaries:')
    for exe in test_binaries:
        print(f"  {exe}")

    # Build a mapping from test target names to package roots.
    target_to_cwd = get_target_to_cwd_mapping()

    for exe in test_binaries:
        # Extract base name.
        basename = os.path.basename(exe)
        # Extract the test target name using the pattern: <target>-<hash>.
        m = re.match(r'^(?P<target>.+?)-[0-9a-f]+$', basename)
        if m:
            target_name = m.group("target")
        else:
            target_name = basename

        exe = os.path.realpath(exe)

        # Determine the correct cwd; if not found, default to the current directory.
        cwd = target_to_cwd.get(target_name, os.getcwd())
        termcolor.cprint(f"Running valgrind on {exe} with cwd {cwd} ...", "yellow")

        # If the binary requires special handling (in this case matching a pattern),
        # call run_xlsynth_test passing in the proper cwd.
        if re.match(r'.*/xlsynth-[0-9a-f]{16}$', exe):
            run_xlsynth_test(exe, cwd)
        else:
            # Otherwise, simply run valgrind.
            valgrind_command = [
                "valgrind",
                "--error-exitcode=1",
                f"--suppressions={os.getcwd()}/valgrind.supp",
                "--leak-check=full",
                exe
            ]
            termcolor.cprint(f'Running command: {subprocess.list2cmdline(valgrind_command)}', 'cyan')
            subprocess.run(valgrind_command, check=True, cwd=cwd, env=dict(XLSYNTH_TOOLS=os.environ["XLSYNTH_TOOLS"]))


if __name__ == "__main__":
    main()
