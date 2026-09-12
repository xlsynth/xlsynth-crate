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
from typing import (
    List,
    Dict,
    Optional,
    DefaultDict,
    Union,
    Tuple,
    TypedDict,
    NamedTuple,
)

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
    "xlsynth_codegen": {"filter_out": ["bridge_builder"]},
    "ir_interpret_test": {
        "shard_test_cases": True,
        "filter_out": ["test_ir_interpret_array_values"],
    },
    "sample_usage": {
        "filter_out": [
            "test_validate_fail",
            # Added based on moderate run time (>5s)
            "tests::test_validate_use",
            "tests::test_validate_use_popcount",
        ]
    },
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
    "invoke_test": {
        "filter_out": [
            # Added based on moderate run time (>20s)
            "test_ir2gates_determinism",
        ]
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
            "dslx_stitch_pipeline",
            # Added based on >100s runtime
            "gate_builder::tests::test_tree_reduce_eq_linear_reduce::_5_expects",
            "gate_builder::tests::test_tree_reduce_eq_linear_reduce::_2_expects",
            "bulk_replace::tests::test_replace_redundant_node_equivalence",
            "gate2ir::tests::test_gate_fn_to_ir_nand",
            "bulk_replace::tests::test_replace_multiple_redundant_nodes",
            "gate_builder::tests::test_tree_reduce_eq_linear_reduce::_4_expects",
            "gate2ir::tests::test_gate_fn_to_ir_one_and_gate",
            "gate_builder::tests::test_tree_reduce_eq_linear_reduce::_1_expects",
            "gate_builder::tests::test_tree_reduce_eq_linear_reduce::_7_expects",
            "gate_builder::tests::test_tree_reduce_eq_linear_reduce::_3_expects",
            "gate_builder::tests::test_tree_reduce_eq_linear_reduce::_8_expects",
            "fraig::tests::test_fraig_optimize_bf16_add",
            "ir2gate_utils::tests::test_gatify_add_carry_select::_3_bit_2_partition",
            "gate2ir::tests::test_gate_fn_to_ir_inverter",
            "ir2gate_utils::tests::test_gatify_add_carry_select::_4_bit_3_partition",
            "ir2gate_utils::tests::test_gatify_add_carry_select::_2_bit_2_partition",
            "ir2gate_utils::tests::test_gatify_add_carry_select::_1_bit_1_partition",
            "gate_builder::tests::test_tree_reduce_eq_linear_reduce::_6_expects",
            "ir2gate_utils::tests::test_gatify_add_carry_select::_8_bit_2_partition",
            "ir2gate_utils::tests::test_gatify_add_carry_select::_4_bit_2_partition",
            "fraig::tests::test_fraig_optimize_bf16_mul",
            # Added based on very long runs (>100s)
            "fraig::tests::test_equiv_class_with_equal_depth_and_opposite_polarity_canonicalizes",
            "ir2gate_utils::tests::test_gatify_ule::_4_expects",
            "get_summary_stats::tests::test_get_summary_stats_bf16_mul",
            "ir2gate_utils::tests::test_gatify_add_carry_select::_8_bit_3_partition",
            "ir2gate_utils::tests::test_gatify_ule::_6_expects",
            "ir2gate_utils::tests::test_gatify_ule::_7_expects",
            "ir2gate_utils::tests::test_gatify_ule::_8_expects",
            "count_toggles::integration_tests::test_bf16_adder_toggle_counting",
            "ir2gate_utils::tests::test_gatify_ule::_3_expects",
            "ir2gate_utils::tests::test_gatify_ule::_1_expects",
            "ir2gate_utils::tests::test_gatify_ule::_2_expects",
            "ir2gate_utils::tests::test_gatify_ule::_5_expects",
            "bulk_replace::tests::test_replace_node_with_constant",
            "get_summary_stats::tests::test_get_summary_stats_bf16_add",
            "count_toggles::integration_tests::test_bf16_mul_toggle_counting",
            # Exclude additional slow tests as requested
            "logical_effort::tests::test_compute_logical_effort_min_delay_bf16_mul",
            "logical_effort::tests::test_compute_logical_effort_min_delay_bf16_add",
            "propose_equiv::tests::test_propose_equiv_graph_with_redundancies",
            "graph_logical_effort::tests::test_graph_logical_effort_bf16_add",
            "graph_logical_effort::tests::test_graph_logical_effort_bf16_mul",
            "propose_equiv::tests::test_propose_equiv_simple_graph",
            # Newly observed >300s cases
            "ir2gate_utils::tests::test_gatify_add_kogge_stone_1_to_16",
            "ir2gate_utils::tests::test_gatify_add_brent_kung_1_to_16",
            "prove_gate_fn_equiv_sat::tests::test_validate_equiv_bf16_mul",
            "prove_gate_fn_equiv_sat::tests::test_validate_equiv_bf16_add",
        ],
        "shard_test_cases": True,
    },
    "test_gate_transform_arbitrary": {
        "filter_out": [
            # Already existing list
        ]
    },
    # Newly added filters to reduce very long (>100s) test runs
    "emit_netlist_integration_test": {
        "filter_out": [
            "test_emit_bf16_mul_with_flops",
            "test_emit_bf16_add_with_flops",
        ],
        "all_filtered_ok": True,
    },
    "test_gatefn_serdes": {
        "filter_out": [
            "test_gatefn_bincode_roundtrip",
        ],
        "all_filtered_ok": True,
    },
    "test_adder_depths": {
        "filter_out": [
            "adder_depths",
        ],
        "all_filtered_ok": True,
    },
}


class TestExecutable(NamedTuple):
    """A Cargo test artifact and the package directory it must run from."""

    path: str
    target_name: str
    cwd: str


def parse_cargo_test_executables(output: str, build_cwd: str) -> List[TestExecutable]:
    """Reads Cargo artifacts without assuming a target directory or filename layout."""
    executables: Dict[str, TestExecutable] = {}
    for line in output.splitlines():
        if not line.lstrip().startswith("{"):
            # Procedural macros and other tools can emit non-JSON output.
            continue
        message = json.loads(line)
        if message.get("reason") == "compiler-message":
            rendered = message.get("message", {}).get("rendered")
            if rendered:
                print(rendered, file=sys.stderr, end="")
        if message.get("reason") != "compiler-artifact":
            continue
        if message.get("profile", {}).get("test") is not True:
            continue
        executable = message.get("executable")
        if not executable:
            continue
        target_name = message.get("target", {}).get("name")
        manifest_path = message.get("manifest_path")
        if not all(
            isinstance(value, str) and value
            for value in (executable, target_name, manifest_path)
        ):
            raise ValueError(
                "Cargo test artifact is missing executable, target name, or manifest path"
            )
        path = os.path.abspath(os.path.join(build_cwd, executable))
        cwd = os.path.realpath(os.path.dirname(os.path.join(build_cwd, manifest_path)))
        artifact = TestExecutable(path, target_name.replace("-", "_"), cwd)
        if path in executables and executables[path] != artifact:
            raise ValueError(
                "Conflicting Cargo metadata for executable: {}".format(path)
            )
        # Cargo reports fresh artifacts too; they are just as important to run.
        executables[path] = artifact
    return list(executables.values())


def compile_test_executables(release: bool, build_cwd: str) -> List[TestExecutable]:
    """Builds test harnesses and obtains their paths from Cargo's JSON interface."""
    command = ["cargo", "test", "--no-run", "--workspace", "--message-format=json"]
    if release:
        command.append("--release")
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        universal_newlines=True,
        cwd=build_cwd,
        env=os.environ,
    )
    # Parse before checking the exit code so rustc diagnostics remain visible.
    executables = parse_cargo_test_executables(result.stdout, build_cwd)
    result.check_returncode()
    return executables


def select_test_executables(
    executables: List[TestExecutable], filter_substrings: List[str]
) -> List[TestExecutable]:
    """Applies executable filters and rejects empty or partially unmatched requests."""
    selected = []
    matched_filters = set()
    for executable in executables:
        if any(
            name in executable.target_name
            for name in ("spdx", "readme_test", "version_test")
        ):
            continue
        matches = [
            substring
            for substring in filter_substrings
            if substring in executable.path
            or substring.replace("-", "_") in executable.target_name
        ]
        if filter_substrings and not matches:
            continue
        matched_filters.update(matches)
        selected.append(executable)
    unmatched = sorted(set(filter_substrings) - matched_filters)
    if not selected or unmatched:
        available = (
            ", ".join(sorted(set(exe.target_name for exe in executables))) or "<none>"
        )
        detail = (
            "Unmatched executable filters: {}. ".format(", ".join(unmatched))
            if unmatched
            else ""
        )
        raise ValueError(
            "{}Selected {} test executables. Available Cargo test targets: {}".format(
                detail, len(selected), available
            )
        )
    return selected


def parse_test_case_names(output: str) -> List[str]:
    """Reads libtest's list output, including headerless lists and trailing summaries."""
    names = []
    for line in output.splitlines():
        match = re.fullmatch(r"(.+): test", line.strip())
        if match:
            names.append(match.group(1))
    return names


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


def run_subset_of_tests(
    exe: str,
    cwd: str,
    filter_out: List[str],
    suppression_path: str,
    all_filtered_ok: bool = False,
    demangle: bool = True,
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
    to_run: List[str] = []
    to_skip: List[str] = []
    for test_name in parse_test_case_names(output):
        if any(f in test_name for f in filter_out):
            to_skip.append(test_name)
        else:
            to_run.append(test_name)

    termcolor.cprint(f"Discovered {len(to_run)} tests to run:", "green")
    for test in to_run:
        termcolor.cprint(f"  {test}", "green")
    if to_skip:
        termcolor.cprint("Skipping tests:", "blue")
        for test in to_skip:
            termcolor.cprint(f"  {test}", "blue")
    else:
        termcolor.cprint("  No tests to skip", "green")

    expected_count = len(to_run)
    if not to_run:
        termcolor.cprint("No tests to run", "blue")
        if all_filtered_ok:
            termcolor.cprint("All filtered tests are ok for this binary", "blue")
            # Return success with 0 parsed, 0 expected
            return defaultdict(float), 0, 0
        else:
            raise ValueError(
                f"No tests to run for binary {exe} with filter_out {filter_out}"
            )

    # Pass the list of tests directly to run_valgrind
    durations, parsed_count = run_valgrind(
        exe, cwd, suppression_path, test_filters=to_run, demangle=demangle
    )
    # Return the result along with the expected count
    return durations, parsed_count, expected_count


def run_valgrind(
    exe: str,
    cwd: str,
    suppression_path: str,
    test_filters: Optional[List[str]] = None,
    expect_tests: bool = True,
    demangle: bool = True,
) -> Tuple[TestDurations, int]:
    """Runs valgrind with JSON test output and parses durations.

    Args:
        exe: The executable path.
        cwd: The working directory to run the executable in.
        suppression_path: The path to the valgrind suppression file.
        test_filters: An optional list of specific test names to run.
        expect_tests: Whether test events are expected in the output.
        demangle: Whether to pass --demangle=no to valgrind.

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
        "--track-origins=yes",
        "--sym-offsets=yes",
        # --demangle=no is added conditionally below
        exe,
    ]

    # Conditionally add demangle flag
    if not demangle:
        valgrind_command.insert(-1, "--demangle=no")  # Insert before exe path

    if test_filters:
        # These names came from libtest's list, so match complete names only.
        valgrind_command.extend(test_filters)
        valgrind_command.append("--exact")

    # Append test harness options AFTER the filter strings (if present)
    valgrind_command.extend(
        [
            "-Z",
            "unstable-options",
            "--report-time",
            # Parallelism is managed by the outer runner; libtest runs serially.
            "--test-threads=1",
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
        termcolor.cprint(f"Command: {' '.join(valgrind_command)}", "red")
        raise subprocess.CalledProcessError(
            999, e.cmd, output=e.stdout, stderr=e.stderr
        ) from e
    except subprocess.CalledProcessError as e:
        termcolor.cprint(
            f"Error running valgrind on {os.path.basename(exe)}: {e}",
            "red",
            file=sys.stderr,
        )
        termcolor.cprint(
            f"Command: {' '.join(valgrind_command)}", "red", file=sys.stderr
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
            raise ValueError(
                "No successful test events were recorded for {} although tests were expected".format(
                    exe
                )
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
    exe: str, test_name: str, cwd: str, suppression_path: str, demangle: bool = True
) -> WorkerResult:
    """Runs valgrind for a single test case. Returns tuple (durations, parsed_count, expected_count) or Exception."""
    basename = os.path.basename(exe)
    termcolor.cprint(f"Starting test: {test_name} in {basename}", "blue")
    expected_count = 1  # Always expect 1 test when running a single case
    try:
        durations, parsed_count = run_valgrind(
            exe,
            cwd,
            suppression_path,
            test_filters=[test_name],
            expect_tests=True,
            demangle=demangle,
        )
        termcolor.cprint(f"Finished test: {test_name} in {basename}", "green")
        # Return durations, parsed count, and expected count (1)
        return durations, parsed_count, expected_count
    except (subprocess.CalledProcessError, ValueError, subprocess.TimeoutExpired) as e:
        # Error likely already printed by run_valgrind
        termcolor.cprint(f"Failed test:   {test_name} in {basename}", "red")
        return e  # Return the exception


def run_single_test_binary(
    executable: TestExecutable,
    script_cwd: str,
    demangle: bool = True,
) -> WorkerResult:
    """Determines how to run valgrind for a single binary and executes it."""
    exe, target_name, cwd = executable
    basename: str = os.path.basename(exe)

    # Determine CWD and suppression path
    suppression_path: str = os.path.join(script_cwd, "valgrind.supp")

    termcolor.cprint(f"Starting: {basename}", "blue")

    result: WorkerResult
    try:
        # Look up configuration for this binary
        config = TEST_BINARY_CONFIGS.get(target_name)

        if config:
            # Found specific config, use run_subset_of_tests
            termcolor.cprint(f"Applying special config for {target_name}", "magenta")
            result = run_subset_of_tests(
                exe,
                cwd,
                config.get("filter_out", []),  # Get filter_out list or default to empty
                suppression_path,
                config.get("all_filtered_ok", False),  # Get flag or default to False
                demangle=demangle,
            )
        else:
            # No specific config found, run valgrind directly
            # Check if binary has tests before running valgrind and count them
            list_output = subprocess.check_output(
                [exe, "--test", "--list"], cwd=cwd, stderr=subprocess.PIPE
            ).decode("utf-8")
            expected_count = len(parse_test_case_names(list_output))

            # Pass expect_tests flag to run_valgrind
            try:
                durations, parsed_count = run_valgrind(
                    exe,
                    cwd,
                    suppression_path,
                    expect_tests=(expected_count > 0),
                    demangle=demangle,
                )
                # Assign the result tuple including the calculated expected count
                result = (durations, parsed_count, expected_count)
            except Exception:
                termcolor.cprint(
                    f"\n[ERROR] Exception while running valgrind on {basename}",
                    "red",
                    file=sys.stderr,
                )
                termcolor.cprint(f"Command: {exe}", "red", file=sys.stderr)
                import traceback

                traceback.print_exc()
                raise

        termcolor.cprint(f"Finished: {basename}", "green")
        return result

    except (subprocess.CalledProcessError, ValueError, subprocess.TimeoutExpired) as e:
        termcolor.cprint(f"Failed:   {basename}", "red")
        termcolor.cprint(f"[ERROR] Exception details: {e}", "red", file=sys.stderr)
        if hasattr(e, "stdout") and e.stdout:
            termcolor.cprint("--- stdout ---", "red", file=sys.stderr)
            print(e.stdout, file=sys.stderr)
        if hasattr(e, "stderr") and e.stderr:
            termcolor.cprint("--- stderr ---", "red", file=sys.stderr)
            print(e.stderr, file=sys.stderr)
        import traceback

        traceback.print_exc()
        return e


def submit_valgrind_tasks(
    executor: concurrent.futures.ProcessPoolExecutor,
    executable: TestExecutable,
    script_cwd: str,
    demangle: bool = True,
) -> Dict[concurrent.futures.Future[WorkerResult], str]:
    """Lists tests, applies filters, and submits tasks for a single binary.

    Returns a dictionary mapping submitted Future objects to their task names.
    Full executable paths keep task identities distinct across packages.
    """
    tasks: Dict[concurrent.futures.Future[WorkerResult], str] = {}
    exe_path, target_name, cwd = executable
    basename = os.path.basename(exe_path)

    # Determine CWD and suppression path for this binary
    suppression_path = os.path.join(script_cwd, "valgrind.supp")

    # Check configuration for sharding
    config = TEST_BINARY_CONFIGS.get(target_name)
    shard_this_binary = config.get("shard_test_cases", False) if config else False
    filter_out = config.get("filter_out", []) if config else []

    if shard_this_binary:
        termcolor.cprint(f"Sharding tests for {basename}...", "magenta")
        list_output = subprocess.check_output(
            [exe_path, "--test", "--list"], cwd=cwd, stderr=subprocess.PIPE
        ).decode("utf-8")

        individual_tests_to_run: List[str] = []
        tests_skipped: List[str] = []
        for test_case_name in parse_test_case_names(list_output):
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
            if config and config.get("all_filtered_ok", False):
                return tasks
            raise ValueError(
                "No tests left to run in {} after filtering".format(exe_path)
            )

        termcolor.cprint(
            f"  Submitting {len(individual_tests_to_run)} individual test tasks for {basename}.",
            "magenta",
        )
        for test_case in individual_tests_to_run:
            future = executor.submit(
                run_single_test_case,
                exe_path,
                test_case,
                cwd,
                suppression_path,
                demangle=demangle,
            )
            tasks[future] = f"{exe_path}::{test_case}"
    else:
        # No sharding: submit task for the whole binary
        future = executor.submit(
            run_single_test_binary,
            executable,
            script_cwd,
            demangle=demangle,
        )
        tasks[future] = exe_path

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
        help="Comma-separated substrings matching executable paths or Cargo target names; every filter must match",
    )
    parser.add_option(
        "--release",
        action="store_true",
        default=False,
        help="Compile tests in release mode (slower build, faster tests)",
    )
    parser.add_option(
        "--no-demangle",
        action="store_true",
        default=False,
        help="Pass --demangle=no to valgrind to disable symbol demangling.",
    )
    (opts, args) = parser.parse_args()

    # Normalize the --filter-to-run option into a list of substrings.
    # Empty entries (e.g., consecutive commas) are ignored.
    filter_substrings: List[str] = [
        s.strip() for s in opts.filter_to_run.split(",") if s.strip()
    ]

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
    termcolor.cprint(f"Compiling tests in {build_mode} mode...", "yellow")
    script_cwd = os.path.realpath(os.getcwd())
    try:
        test_binaries = select_test_executables(
            compile_test_executables(opts.release, script_cwd), filter_substrings
        )
    except (subprocess.CalledProcessError, ValueError, OSError) as error:
        termcolor.cprint(
            "Error discovering test executables: {}".format(error),
            "red",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Found {len(test_binaries)} test binaries:")
    for executable in test_binaries:
        print(
            "  {}: {} (cwd: {})".format(
                executable.target_name, executable.path, executable.cwd
            )
        )

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
        for executable in test_binaries:
            try:
                submitted_tasks = submit_valgrind_tasks(
                    executor,
                    executable,
                    script_cwd,
                    demangle=(not opts.no_demangle),
                )
                all_futures_map.update(submitted_tasks)
            except (subprocess.CalledProcessError, ValueError, OSError) as error:
                task_results[executable.path] = error

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
            color = "red"
            termcolor.cprint(
                f"Task '{task_name}' failed: {result}", "red", file=sys.stderr
            )
            if is_single_test:
                if task_name not in failed_tests:
                    failed_tests.append(task_name)
            else:
                if task_name not in failed_binaries:
                    failed_binaries.append(task_name)
            # Assign failure status
            task_statuses[task_name] = (status, color)

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
            color = "red"
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
            # Assign failure status
            task_statuses[task_name] = (status, color)

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
    executed_tests = sum(parsed_counts.values())
    if executed_tests == 0:
        termcolor.cprint(
            "\nError: No test cases completed under Valgrind.", "red", file=sys.stderr
        )
        exit_code = 1
    else:
        termcolor.cprint(
            "\nCompleted {} test cases under Valgrind.".format(executed_tests), "green"
        )
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
