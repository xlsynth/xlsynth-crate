# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import optparse
import sys
import re
import json
import time
from collections import defaultdict
import concurrent.futures
from typing import List, Dict, Optional, DefaultDict, Union, Any, Tuple, TypedDict

import termcolor

# Type alias for test duration dictionary
TestDurations = DefaultDict[str, float]

# Type alias for the result of run_single_test_binary or run_single_test_case
# Success is Tuple[TestDurations, parsed_count, expected_count]
WorkerResult = Union[Tuple[TestDurations, int, int], Exception]


class TestBinaryConfig(TypedDict, total=False):
    """
    Attributes:
        filter_out: Tests to exclude from the run.
        all_filtered_ok: Is it ok if all tests are filtered out?
        shard_test_cases: Shard individual #[test] cases across workers?
    """

    filter_out: List[str]
    all_filtered_ok: bool
    shard_test_cases: bool


TEST_BINARY_CONFIGS: Dict[str, TestBinaryConfig] = {
    "xlsynth": {"filter_out": ["bridge_builder"]},
    "ir_interpret_test": {
        "filter_out": ["ir_interpret_array_values"],
        "all_filtered_ok": True,
    },
    "sample_usage": {"filter_out": ["test_validate_fail"]},
    "sv_bridge_test": {
        "filter_out": ["test_sv_bridge_structure_zoo"],
        "all_filtered_ok": True,
    },
    "gatify_tests": {
        "filter_out": [
            # Exclude tests known to be very slow under valgrind (>100s)
            "test_gatify_bf16_mul::opt_yes_expects",
            "test_gatify_bf16_mul::opt_no_expects",
            "test_gatify_bf16_add::opt_yes_expects",
            "test_encode_ir_to_gates::bit_count_4_fold_true",
            "test_gatify_bf16_add::opt_no_expects",
            "test_eq_ir_to_gates::bit_count_1_fold_false",
            "test_encode_ir_to_gates::bit_count_4_fold_false",
            "test_eqz_ir_to_gates::bit_count_3_fold_false",
            "test_eqz_ir_to_gates::bit_count_2_fold_true",
            "test_eqz_ir_to_gates::bit_count_4_fold_false",
            "test_eq_all_zeros_all_ones_to_gates::bit_count_1_fold_false",
            "test_eqz_ir_to_gates::bit_count_1_fold_true",
            "test_eqz_ir_to_gates::bit_count_4_fold_true",
        ],
    },
    # Added filter for slow tests (~100s)
    "gate_sim_vs_ir_interp": {
        "filter_out": [
            "test_bf16_mul_g8r_stats",
            "test_bf16_mul_zero_zero",
            "test_bf16_mul_random",
        ],
        "all_filtered_ok": True,
    },
    # Added filter for slow tests (~100s)
    "xlsynth_g8r": {
        "filter_out": [
            "validate_equiv::tests::test_validate_equiv_bf16_mul",
            "validate_equiv::tests::test_validate_equiv_bf16_add",
        ]
    },
}


def get_target_to_cwd_mapping() -> Dict[str, str]:
    """
    Returns a dict mapping a test target name to its package root directory.
    We use 'cargo metadata' to get information on each package.
    """
    p: subprocess.CompletedProcess[bytes] = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        capture_output=True,
        check=True,
        env=os.environ,
    )
    metadata: Dict[str, Any] = json.loads(p.stdout.decode("utf-8"))
    mapping: Dict[str, str] = {}
    for package in metadata["packages"]:
        # The package root is the directory containing Cargo.toml.
        package_root = os.path.dirname(package["manifest_path"])
        for target in package["targets"]:
            # We assume test targets have kind "test" or "integration-test".
            if any(kind in target["kind"] for kind in ("test", "integration-test")):
                mapping[target["name"]] = package_root
    return mapping


def _sanity_check_test_executable(exe: str) -> None:
    """Raises an error if the provided executable path is invalid."""
    if not os.path.isabs(exe):
        termcolor.cprint(
            f"Error: Path passed to run_valgrind is not absolute: {exe}",
            "red",
            file=sys.stderr,
        )
        raise ValueError(f"Non-absolute path provided: {exe}")
    if not os.path.isfile(exe):
        termcolor.cprint(
            f"Error: Executable path does not exist or is not a file: {exe}",
            "red",
            file=sys.stderr,
        )
        raise FileNotFoundError(f"Executable not found: {exe}")
    if not os.access(exe, os.X_OK):
        termcolor.cprint(
            f"Error: Executable path is not executable: {exe}", "red", file=sys.stderr
        )
        try:
            stat_result = os.stat(exe)
            termcolor.cprint(
                f"  Permissions: {oct(stat_result.st_mode)}", "red", file=sys.stderr
            )
        except Exception as stat_e:
            termcolor.cprint(f"  Could not stat file: {stat_e}", "red", file=sys.stderr)
        raise PermissionError(f"Executable not runnable: {exe}")


def _insert_test_filter(command: List[str], test_filters: List[str]) -> None:
    """Inserts the test filter strings into the command list immediately after the executable path.

    Raises ValueError if the executable path cannot be determined (heuristically).
    """
    exe_index = -1
    for i, arg in enumerate(command):
        # Heuristic: Find the first arg that is likely the executable path
        if (
            i > 0
            and not arg.startswith("-")
            and os.path.isabs(arg)
            and os.path.isfile(arg)
        ):
            exe_index = i
            break

    if exe_index == -1:
        raise ValueError(
            f"Could not reliably determine executable path index in command to insert test filter: {command}"
        )

    # Insert each test filter string individually after the executable path
    command[exe_index + 1 : exe_index + 1] = test_filters  # Splice the list in


def run_subset_of_tests(
    exe: str,
    cwd: str,
    filter_out: List[str],
    suppression_path: str,
    all_filtered_ok: bool = False,
) -> Tuple[TestDurations, int, int]:
    """
    Runs a subset of tests from the given test executable, filtering out tests that contain any of the substrings specified in filter_out.
    The binary is run from the provided cwd.
    Returns a tuple: (dictionary of test durations, count of tests run, count of expected tests).
    """
    # Get the list of tests from the executable.
    output: str = subprocess.check_output([exe, "--test", "--list"], cwd=cwd).decode(
        "utf-8"
    )
    lines: List[str] = output.splitlines()
    to_run: List[str] = []
    to_skip: List[str] = []
    # Skip the header line.
    for line in lines[1:]:
        if not line.strip():
            continue
        if "benchmark" in line:
            continue
        lhs, rhs = line.rsplit(": ", 1)
        assert rhs == "test", line
        if any(f in lhs for f in filter_out):
            to_skip.append(lhs)
        else:
            to_run.append(lhs)

    termcolor.cprint(f"Discovered {len(to_run)} tests to run:", "green")
    for test in to_run:
        termcolor.cprint(f"  {test}", "green")
    if to_skip:
        termcolor.cprint("Skipping tests:", "red")
        for test in to_skip:
            termcolor.cprint(f"  {test}", "red")
    else:
        termcolor.cprint("  No tests to skip", "green")

    expected_count = len(to_run)
    if not to_run:
        termcolor.cprint("No tests to run", "red")
        if all_filtered_ok:
            termcolor.cprint("All filtered tests are ok for this binary", "green")
            # Return success with 0 parsed, 0 expected
            return defaultdict(float), 0, 0
        else:
            raise ValueError(
                f"No tests to run for binary {exe} with filter_out {filter_out}"
            )

    # Pass the list of tests directly to run_valgrind
    durations, parsed_count = run_valgrind(
        exe, cwd, suppression_path, test_filters=to_run
    )
    # Return the result along with the expected count
    return durations, parsed_count, expected_count


def run_valgrind(
    exe: str,
    cwd: str,
    suppression_path: str,
    test_filters: Optional[List[str]] = None,
    expect_tests: bool = True,
) -> Tuple[TestDurations, int]:
    """Runs valgrind with JSON test output and parses durations.

    Args:
        exe: The executable path.
        cwd: The working directory to run the executable in.
        suppression_path: The path to the valgrind suppression file.
        test_filters: An optional list of specific test names to run.
        expect_tests: Whether test events are expected in the output.

    Returns tuple: (dict of test durations, count of tests parsed).
    """
    _sanity_check_test_executable(exe)
    test_durations: TestDurations = defaultdict(float)

    # Base command
    valgrind_command: List[str] = [
        "valgrind",
        "--error-exitcode=1",
        f"--suppressions={suppression_path}",
        "--leak-check=full",
        exe,
    ]

    if test_filters:
        # Insert test filter strings right after the executable
        _insert_test_filter(valgrind_command, test_filters)

    # Append test harness options AFTER the filter strings (if present)
    valgrind_command.extend(
        [
            "-Z",
            "unstable-options",
            "--report-time",
            "--format",
            "json",
        ]
    )

    start_time: float = time.time()
    try:
        result: subprocess.CompletedProcess[str] = subprocess.run(
            valgrind_command,
            check=True,
            cwd=cwd,
            env=os.environ,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except subprocess.TimeoutExpired as e:
        termcolor.cprint(
            f"Timeout expired running valgrind on {os.path.basename(exe)}", "red"
        )
        raise subprocess.CalledProcessError(
            999, e.cmd, output=e.stdout, stderr=e.stderr
        ) from e
    except subprocess.CalledProcessError as e:
        termcolor.cprint(
            f"Error running valgrind on {os.path.basename(exe)}: {e}",
            "red",
            file=sys.stderr,
        )
        if e.stdout:
            termcolor.cprint("--- stdout ---", "red", file=sys.stderr)
            print(e.stdout, file=sys.stderr)
        if e.stderr:
            termcolor.cprint("--- stderr ---", "red", file=sys.stderr)
            print(e.stderr, file=sys.stderr)
        raise
    end_time: float = time.time()

    parsed_specific_test_count: int = 0
    json_parsing_errors: int = 0
    suite_exec_time: Optional[float] = None  # Store suite exec_time if found

    for line in result.stdout.splitlines():
        try:
            data = json.loads(line.strip())  # Parse each line as JSON

            # Check if it's a successful test event
            if data.get("type") == "test" and data.get("event") == "ok":
                # Check only for the presence of a valid name
                name = data.get("name")
                exec_time = data.get("exec_time")
                if (
                    name is not None
                    and isinstance(name, str)
                    and exec_time is not None
                    and isinstance(exec_time, (int, float))
                ):
                    try:
                        test_name: str = f"{os.path.basename(exe)}::{name}"
                        duration: float = float(exec_time)
                        test_durations[test_name] = duration
                        parsed_specific_test_count += 1
                    except (ValueError, TypeError) as conversion_err:
                        json_parsing_errors += 1
                        termcolor.cprint(
                            f"Warning: Could not convert exec_time '{exec_time}' for test {name}. Error: {conversion_err}",
                            "yellow",
                            file=sys.stderr,
                        )
                else:
                    # Missing name or exec_time
                    json_parsing_errors += 1
                    termcolor.cprint(
                        f"Warning: JSON test event missing name or exec_time. Data: {data}",
                        "yellow",
                        file=sys.stderr,
                    )

            # Check if it's the final suite summary event
            elif data.get("type") == "suite" and data.get("event") == "ok":
                exec_time = data.get("exec_time")
                if exec_time is not None and isinstance(exec_time, (int, float)):
                    try:
                        suite_exec_time = float(exec_time)
                    except (ValueError, TypeError):
                        pass  # Ignore if conversion fails

        except json.JSONDecodeError:
            json_parsing_errors += 1
            pass  # Ignore lines that are not valid JSON
        except Exception as e:
            json_parsing_errors += 1
            termcolor.cprint(
                f"Warning: Error processing line: '{line.strip()}' - {e}",
                "yellow",
                file=sys.stderr,
            )

    # Fallback logic: Record total time only if no individual tests passed.
    if parsed_specific_test_count == 0:
        fallback_name: str = f"{os.path.basename(exe)}::execution"
        # Prefer suite exec_time if available, otherwise use wall clock time
        fallback_duration = (
            suite_exec_time if suite_exec_time is not None else (end_time - start_time)
        )
        test_durations[fallback_name] = fallback_duration

        # Print warning only if tests were expected
        if expect_tests:
            warning_message = f"Warning: Recorded fallback time ({fallback_duration:.3f}s) for {os.path.basename(exe)} (tests expected)."
            if json_parsing_errors > 0:
                warning_message += (
                    f" Encountered {json_parsing_errors} JSON parsing/schema issues."
                )
            else:
                warning_message += (
                    " No valid 'test ok' JSON events found (or missing exec_time)."
                )
            termcolor.cprint(warning_message, "yellow", file=sys.stderr)
            termcolor.cprint(
                f"--- start stdout for {os.path.basename(exe)} --- ",
                "magenta",
                file=sys.stderr,
            )
            print(result.stdout, file=sys.stderr)
            termcolor.cprint(
                f"--- end stdout for {os.path.basename(exe)} --- ",
                "magenta",
                file=sys.stderr,
            )

    return test_durations, parsed_specific_test_count


def print_test_durations(test_durations: TestDurations) -> None:
    """Print test durations sorted by duration in descending order, excluding 0.00s entries."""
    if not test_durations:
        return

    termcolor.cprint(
        "\nTest Durations (sorted by duration, descending, >0.00s):", "yellow"
    )
    termcolor.cprint("=" * 80, "yellow")

    # Sort by duration in descending order
    sorted_tests: List[Tuple[str, float]] = sorted(
        test_durations.items(), key=lambda x: x[1], reverse=True
    )

    printed_any = False
    for test_name, duration in sorted_tests:
        # Check if duration rounded to 2 decimal places is greater than 0
        if round(duration, 2) > 0.0:
            termcolor.cprint(f"{test_name:<60} {duration:>8.2f}s", "cyan")
            printed_any = True

    if not printed_any:
        termcolor.cprint("  (No tests took > 0.00s)", "cyan")


def run_single_test_case(
    exe: str, test_name: str, cwd: str, suppression_path: str
) -> WorkerResult:
    """Runs valgrind for a single test case. Returns tuple (durations, parsed_count, expected_count) or Exception."""
    basename = os.path.basename(exe)
    termcolor.cprint(f"Starting test: {test_name} in {basename}", "blue")
    expected_count = 1  # Always expect 1 test when running a single case
    try:
        durations, parsed_count = run_valgrind(
            exe, cwd, suppression_path, test_filters=[test_name], expect_tests=True
        )
        termcolor.cprint(f"Finished test: {test_name} in {basename}", "green")
        # Return durations, parsed count, and expected count (1)
        return durations, parsed_count, expected_count
    except (subprocess.CalledProcessError, ValueError, subprocess.TimeoutExpired) as e:
        # Error likely already printed by run_valgrind
        termcolor.cprint(f"Failed test:   {test_name} in {basename}", "red")
        return e  # Return the exception


def run_single_test_binary(
    exe: str, target_to_cwd: Dict[str, str], script_cwd: str
) -> WorkerResult:
    """Determines how to run valgrind for a single binary and executes it."""
    basename: str = os.path.basename(exe)
    m: Optional[re.Match[str]] = re.match(r"^(?P<target>.+?)-[0-9a-f]+$", basename)
    target_name: str = m.group("target") if m else basename

    # Determine CWD and suppression path
    cwd: str = os.path.realpath(target_to_cwd.get(target_name, script_cwd))
    suppression_path: str = os.path.join(script_cwd, "valgrind.supp")

    termcolor.cprint(f"Starting: {basename}", "blue")

    test_binary_name_match: Optional[re.Match[str]] = re.match(
        r".*/(.*)-[0-9a-f]{16}$", exe
    )
    test_binary_name: Optional[str] = None
    if test_binary_name_match:
        test_binary_name = test_binary_name_match.group(1)
    else:
        termcolor.cprint(
            f"Warning: Could not extract test binary name from {exe}",
            "yellow",
            file=sys.stderr,
        )

    result: WorkerResult
    try:
        # Look up configuration for this binary
        config = TEST_BINARY_CONFIGS.get(test_binary_name) if test_binary_name else None

        if config:
            # Found specific config, use run_subset_of_tests
            termcolor.cprint(
                f"Applying special config for {test_binary_name}", "magenta"
            )
            result = run_subset_of_tests(
                exe,
                cwd,
                config.get("filter_out", []),  # Get filter_out list or default to empty
                suppression_path,
                config.get("all_filtered_ok", False),  # Get flag or default to False
            )
        else:
            # No specific config found, run valgrind directly
            # Check if binary has tests before running valgrind and count them
            expected_count = 0
            try:
                list_output = subprocess.check_output(
                    [exe, "--test", "--list"], cwd=cwd, stderr=subprocess.PIPE
                ).decode("utf-8")
                for list_line in list_output.splitlines():
                    if list_line.strip().endswith(": test"):
                        expected_count += 1
            except (subprocess.CalledProcessError, FileNotFoundError) as list_err:
                termcolor.cprint(
                    f"Warning: Failed to list tests for {basename}: {list_err}",
                    "yellow",
                    file=sys.stderr,
                )
                # We failed to list, assume tests are expected, but expected count is unknown (use -1?)
                # Let's stick with 0 for now, run_valgrind's warning will trigger if needed.
                expected_count = 0

            # Pass expect_tests flag to run_valgrind
            durations, parsed_count = run_valgrind(
                exe, cwd, suppression_path, expect_tests=(expected_count > 0)
            )
            # Assign the result tuple including the calculated expected count
            result = (durations, parsed_count, expected_count)

        termcolor.cprint(f"Finished: {basename}", "green")
        return result

    except (subprocess.CalledProcessError, ValueError, subprocess.TimeoutExpired) as e:
        termcolor.cprint(f"Failed:   {basename}", "red")
        return e


def submit_valgrind_tasks(
    executor: concurrent.futures.ProcessPoolExecutor,
    exe_path: str,
    target_to_cwd: Dict[str, str],
    script_cwd: str,
) -> Dict[concurrent.futures.Future[WorkerResult], str]:
    """Lists tests, applies filters, and submits tasks for a single binary.

    Returns a dictionary mapping submitted Future objects to their task names.
    Task names are either the binary basename or 'binary::test_case'.
    """
    tasks: Dict[concurrent.futures.Future[WorkerResult], str] = {}
    basename = os.path.basename(exe_path)
    m = re.match(r"^(?P<target>.+?)-[0-9a-f]+$", basename)
    target_name = m.group("target") if m else basename

    # Determine CWD and suppression path for this binary
    cwd = os.path.realpath(target_to_cwd.get(target_name, script_cwd))
    suppression_path = os.path.join(script_cwd, "valgrind.supp")

    # Check configuration for sharding
    config = TEST_BINARY_CONFIGS.get(target_name)
    shard_this_binary = config.get("shard_test_cases", False) if config else False
    filter_out = config.get("filter_out", []) if config else []

    if shard_this_binary:
        termcolor.cprint(f"Sharding tests for {basename}...", "magenta")
        try:
            list_output = subprocess.check_output(
                [exe_path, "--test", "--list"], cwd=cwd, stderr=subprocess.PIPE
            ).decode("utf-8")

            individual_tests_to_run: List[str] = []
            tests_skipped: List[str] = []
            for list_line in list_output.splitlines():
                if list_line.strip().endswith(": test"):
                    test_case_name = list_line.split(": test")[0].strip()
                    if any(f in test_case_name for f in filter_out):
                        tests_skipped.append(test_case_name)
                    else:
                        individual_tests_to_run.append(test_case_name)

            if tests_skipped:
                termcolor.cprint(
                    f"  Skipping {len(tests_skipped)} tests from {basename} due to config.",
                    "red",
                )

            if not individual_tests_to_run:
                termcolor.cprint(
                    f"  No tests left to run in {basename} after filtering.", "yellow"
                )
                return tasks  # Return empty dict if no tests to run

            termcolor.cprint(
                f"  Submitting {len(individual_tests_to_run)} individual test tasks for {basename}.",
                "magenta",
            )
            for test_case in individual_tests_to_run:
                future = executor.submit(
                    run_single_test_case, exe_path, test_case, cwd, suppression_path
                )
                task_name = f"{basename}::{test_case}"
                tasks[future] = task_name

        except (subprocess.CalledProcessError, FileNotFoundError) as list_err:
            termcolor.cprint(
                f"Error listing tests for sharding {basename}: {list_err}. Running binary as whole.",
                "red",
                file=sys.stderr,
            )
            # Fallback: submit the whole binary
            future = executor.submit(
                run_single_test_binary, exe_path, target_to_cwd, script_cwd
            )
            tasks[future] = basename
    else:
        # No sharding: submit task for the whole binary
        future = executor.submit(
            run_single_test_binary, exe_path, target_to_cwd, script_cwd
        )
        tasks[future] = basename

    return tasks


def process_completed_task(
    future: concurrent.futures.Future[WorkerResult],
    task_name: str,
    task_results: Dict[str, WorkerResult],
) -> None:
    """Processes the result of a completed future, storing the raw result."""
    try:
        result: WorkerResult = future.result()
        task_results[task_name] = result
    except Exception as e:
        termcolor.cprint(f"Task '{task_name}' future raised exception: {e}", "red")
        task_results[task_name] = e  # Store the exception from the future itself


def main() -> None:
    script_start_time = time.time()

    # Check for nightly toolchain since we're going to use nightly unstable flags/options.
    try:
        version_result = subprocess.run(
            ["cargo", "--version"], capture_output=True, text=True, check=True
        )
        if "nightly" not in version_result.stdout:
            termcolor.cprint(
                "Error: This script requires the nightly Rust toolchain to use JSON test output.",
                "red",
            )
            termcolor.cprint(
                f"Detected version: {version_result.stdout.strip()}", "red"
            )
            termcolor.cprint(
                "Please run `rustup default nightly` or use `cargo +nightly ...`", "red"
            )
            sys.exit(1)
        else:
            termcolor.cprint(
                f"Using nightly toolchain: {version_result.stdout.strip()}", "green"
            )
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        termcolor.cprint(f"Error checking cargo version: {e}", "red")
        termcolor.cprint("Is cargo installed and in PATH?", "red")
        sys.exit(1)

    parser = optparse.OptionParser()
    parser.add_option(
        "-k",
        "--filter-to-run",
        type="string",
        default="",
        help="Filter to run only the given executable substring",
    )
    parser.add_option(
        "--release",
        action="store_true",
        default=False,
        help="Compile tests in release mode (slower build, faster tests)",
    )
    (opts, args) = parser.parse_args()

    # Check if XLSYNTH_TOOLS environment variable is set.
    xlsynth_tools_path: Optional[str] = os.environ.get("XLSYNTH_TOOLS")
    if not xlsynth_tools_path:
        termcolor.cprint(
            "Error: The XLSYNTH_TOOLS environment variable must be set.", "red"
        )
        termcolor.cprint(
            "Please set it to the directory containing the XLS tools (e.g., dslx_parser, ir_converter_main).",
            "red",
        )
        sys.exit(1)
    else:
        termcolor.cprint(f"Using XLSYNTH_TOOLS path: {xlsynth_tools_path}", "green")

    # Determine build mode
    build_mode = "release" if opts.release else "debug (fast build)"
    termcolor.cprint(
        f"Compiling tests in {build_mode} mode with unstable options...", "yellow"
    )
    cargo_command = [
        "cargo",
        "test",
        "--no-run",
        "--workspace",
    ]
    if opts.release:
        cargo_command.append("--release")

    # Add unstable options flag (passed to test binary compiler)
    cargo_command.extend(["--", "-Z", "unstable-options"])
    # Note: passing flags after '--' to cargo test passes them to the test binary compiler *invocation*
    # We might just need the -Z flag when RUNNING the binary. Let's keep this for now.

    p: subprocess.CompletedProcess[bytes] = subprocess.run(
        cargo_command, check=True, stderr=subprocess.PIPE, env=os.environ
    )
    output: str = p.stderr.decode("utf-8")

    test_binaries: List[str] = []

    # Adjust executable path based on build mode
    target_dir = "target/release/deps" if opts.release else "target/debug/deps"

    for line in output.splitlines():
        # Check for the correct target directory
        if "Executable" in line and "(" in line and target_dir in line:
            if opts.filter_to_run and opts.filter_to_run not in line:
                continue
            if "spdx" in line or any(
                x in line for x in ["readme_test", "version_test"]
            ):
                continue

            # Extract the path within the parentheses
            path_match = re.search(r"\(([^)]+)\)", line)
            if path_match:
                relative_path: str = path_match.group(1)  # It's typically relative
                absolute_path: str = os.path.abspath(relative_path)
                test_binaries.append(absolute_path)
            else:
                termcolor.cprint(
                    f"Warning: Could not parse executable path from line: {line}",
                    "yellow",
                    file=sys.stderr,
                )

    if not test_binaries:
        print("No test executables found in release build matching filter.")
        sys.exit(0)

    print(f"Found {len(test_binaries)} test binaries:")

    target_to_cwd: Dict[str, str] = {
        name: os.path.realpath(path)
        for name, path in get_target_to_cwd_mapping().items()
    }
    script_cwd: str = os.getcwd()

    num_workers: Optional[int] = os.cpu_count()
    if num_workers is None:
        termcolor.cprint(
            "Could not determine CPU count, defaulting to 1 worker.", "yellow"
        )
        num_workers = 1
    termcolor.cprint(
        f"\nRunning valgrind on {len(test_binaries)} binaries using up to {num_workers} workers...",
        "yellow",
    )

    # Dictionary to store results: task_name -> (TestDurations, count) or Exception
    task_results: Dict[str, WorkerResult] = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        all_futures_map: Dict[concurrent.futures.Future[WorkerResult], str] = {}
        for exe_path in test_binaries:
            submitted_tasks = submit_valgrind_tasks(
                executor, exe_path, target_to_cwd, script_cwd
            )
            all_futures_map.update(submitted_tasks)

        # Process results as they complete
        for future in concurrent.futures.as_completed(all_futures_map):
            task_name = all_futures_map[future]
            process_completed_task(
                future,
                task_name,
                task_results,
            )

    combined_test_durations: TestDurations = defaultdict(float)
    failed_binaries: List[str] = []
    failed_tests: List[str] = []
    parsed_counts: Dict[str, int] = {}
    task_statuses: Dict[str, Tuple[str, str]] = {}

    # Sort tasks by name for consistent summary output
    sorted_task_names = sorted(task_results.keys())

    for task_name in sorted_task_names:
        result = task_results[task_name]
        is_single_test = "::" in task_name

        if isinstance(result, Exception):
            status = "Failed (Exception)"
            termcolor.cprint(
                f"Task '{task_name}' failed: {result}", "red", file=sys.stderr
            )
            if is_single_test:
                if task_name not in failed_tests:
                    failed_tests.append(task_name)
            else:
                if task_name not in failed_binaries:
                    failed_binaries.append(task_name)
        elif (
            isinstance(result, tuple)
            and len(result) == 3
            and isinstance(result[0], defaultdict)
        ):
            durations, parsed_count, expected_count = result
            # Determine status and color based on parsed vs expected
            if expected_count > 0:
                status = f"Success (Parsed {parsed_count}/{expected_count} tests)"
                color = "yellow" if parsed_count != expected_count else "green"
            else:
                status = "Success (No tests expected/found)"
                color = "green"
                # Sanity check: if no tests expected, parsed should be 0
                if parsed_count != 0:
                    termcolor.cprint(
                        f"Warning: Task '{task_name}' expected 0 tests but parsed {parsed_count}!",
                        "yellow",
                        file=sys.stderr,
                    )
                    color = "yellow"  # Mark as yellow due to inconsistency
                    status = f"Success (Parsed {parsed_count}/{expected_count} tests - unexpected!)"  # Update status

            parsed_counts[task_name] = parsed_count  # Store parsed count anyway
            task_statuses[task_name] = (status, color)  # Store status and color

            for test_name, duration in durations.items():
                combined_test_durations[test_name] = duration
        else:
            status = "Failed (Unexpected Result Type)"
            termcolor.cprint(
                f"Error: Worker for '{task_name}' returned unexpected type: {type(result)}",
                "red",
                file=sys.stderr,
            )
            if is_single_test:
                if task_name not in failed_tests:
                    failed_tests.append(task_name)
            else:
                if task_name not in failed_binaries:
                    failed_binaries.append(task_name)

        task_statuses[task_name] = task_statuses[task_name]

    print_test_durations(combined_test_durations)

    termcolor.cprint("\nTask Summary:", "yellow")
    termcolor.cprint("=" * 80, "yellow")
    max_name_len = (
        max(len(name) for name in task_statuses.keys()) if task_statuses else 0
    )
    for task_name in sorted_task_names:
        status, color = task_statuses[task_name]
        termcolor.cprint(f"{task_name:<{max_name_len}} : {status}", color)

    script_end_time = time.time()
    total_duration = script_end_time - script_start_time
    termcolor.cprint(
        f"\nTotal script execution time: {total_duration:.2f} seconds", "blue"
    )

    exit_code = 0
    if failed_binaries:
        termcolor.cprint(
            f"\nSummary: Valgrind runs failed for {len(failed_binaries)} binaries:",
            "red",
        )
        for name in failed_binaries:
            termcolor.cprint(f"  {name}", "red")
        exit_code = 1
    if failed_tests:
        termcolor.cprint(
            f"\nSummary: Valgrind runs failed for {len(failed_tests)} individual test cases:",
            "red",
        )
        for name in failed_tests:
            termcolor.cprint(f"  {name}", "red")
        exit_code = 1

    if exit_code == 0:
        termcolor.cprint(
            "\nSummary: All Valgrind runs completed successfully.", "green"
        )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
