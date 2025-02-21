# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import optparse
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
        check=True,
        env=os.environ
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

def run_subset_of_tests(exe, cwd, filter_out, all_filtered_ok=False):
    """
    Runs a subset of tests from the given test executable, filtering out tests that contain any of the substrings specified in filter_out.
    The binary is run from the provided cwd.
    """
    # Get the list of tests from the executable.
    output = subprocess.check_output([exe, "--test", "--list"], cwd=cwd).decode("utf-8")
    lines = output.splitlines()
    to_run = []
    to_skip = []
    # Skip the header line.
    for line in lines[1:]:
        if not line.strip():
            continue
        if 'benchmark' in line:
            continue
        lhs, rhs = line.rsplit(': ', 1)
        assert rhs == 'test', line
        if any(f in lhs for f in filter_out):
            to_skip.append(lhs)
        else:
            to_run.append(lhs)
    
    termcolor.cprint(f'Discovered {len(to_run)} tests to run:', 'green')
    for test in to_run:
        termcolor.cprint(f'  {test}', 'green')
    if to_skip:
        termcolor.cprint('Skipping tests:', 'red')
        for test in to_skip:
                termcolor.cprint(f'  {test}', 'red')
    else:
        termcolor.cprint('  No tests to skip', 'green')

    if not to_run:
        termcolor.cprint('No tests to run', 'red')
        if all_filtered_ok:
            termcolor.cprint('All filtered tests are ok for this binary', 'green')
            return
        else:
            raise ValueError(f'No tests to run for binary {exe} with filter_out {filter_out}')

    test_str = " ".join(to_run)
    run_valgrind(exe, cwd, test_str)

def run_valgrind(exe, cwd, test_str=None):
    """Runs valgrind with a standardized set of options on the given executable in the provided working directory."""
    valgrind_command = [
        "valgrind",
        "--error-exitcode=1",
        f"--suppressions={os.getcwd()}/valgrind.supp",
        "--leak-check=full",
        exe
    ]
    if test_str:
        valgrind_command.append(test_str)
    termcolor.cprint(f'Running command: {subprocess.list2cmdline(valgrind_command)}', 'cyan')
    subprocess.run(valgrind_command, check=True, cwd=cwd, env=os.environ)

def main():
    parser = optparse.OptionParser()
    parser.add_option('-k', "--filter-to-run", type="string", default="", help="Filter to run only the given executable substring")
    (opts, args) = parser.parse_args()

    # Compile all tests in the workspace.
    print("Compiling tests for package 'xlsynth' with cargo test --no-run ...")
    p = subprocess.run(
        ["cargo", "test", "--no-run", "--workspace"],
        check=True,
        stderr=subprocess.PIPE,
        env=os.environ
    )
    output = p.stderr.decode("utf-8")

    # Get the test binary paths from the cargo output.
    test_binaries = []
    for line in output.splitlines():
        if "Executable" in line and "(" in line:
            if opts.filter_to_run and opts.filter_to_run not in line:
                print(f"Skipping {line} because it does not contain {opts.filter_to_run}")
                continue
            # Skip unwanted binaries.
            if 'spdx' in line:
                continue
            if any(x in line for x in ['readme_test', 'version_test']):
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
        # call run_subset_of_tests passing in the proper cwd and filter_out.
        #
        # Note the binary names are test target names with a hash suffix.
        test_binary_name = re.match(r'.*/(.*)-[0-9a-f]{16}$', exe).group(1)
        if test_binary_name == 'xlsynth':
            run_subset_of_tests(exe, cwd, filter_out=['bridge_builder'])
        elif test_binary_name == 'ir_interpret_test':
            run_subset_of_tests(exe, cwd, filter_out=['ir_interpret_array_values'], all_filtered_ok=True)
        elif test_binary_name == 'sample_usage':
            run_subset_of_tests(exe, cwd, filter_out=['test_validate_fail'])
        elif test_binary_name == 'sv_bridge_test':
            run_subset_of_tests(exe, cwd, filter_out=['test_sv_bridge_structure_zoo'], all_filtered_ok=True)
        else:
            run_valgrind(exe, cwd)


if __name__ == "__main__":
    main()
