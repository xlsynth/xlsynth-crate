# SPDX-License-Identifier: Apache-2.0

"""Exercise Valgrind orchestration without compiling Rust or invoking Valgrind."""

import concurrent.futures
import io
import json
import os
import subprocess
import unittest
from collections import defaultdict
from unittest import mock

import run_valgrind_on_tests as runner


class ImmediateExecutor:
    """Runs mocked workers synchronously while retaining the Future interface."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def submit(self, function, *args, **kwargs):
        future = concurrent.futures.Future()
        try:
            future.set_result(function(*args, **kwargs))
        except Exception as error:
            future.set_exception(error)
        return future


class ValgrindRunnerTest(unittest.TestCase):
    def setUp(self):
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        for name, stream in [("stdout", self.stdout), ("stderr", self.stderr)]:
            patch = mock.patch.object(runner.sys, name, stream)
            patch.start()
            self.addCleanup(patch.stop)

    def artifact(
        self, name="ir_interpret_test", path=None, package="xlsynth", **updates
    ):
        message = {
            "reason": "compiler-artifact",
            "package_id": "path+file:///workspace/{}#0.70.0".format(package),
            "manifest_path": "/workspace/{}/Cargo.toml".format(package),
            "target": {"name": name, "kind": ["test"]},
            "profile": {"test": True},
            "executable": path
            or "/cache/custom target/build/{}/abcd/out/{}".format(package, name),
            "fresh": True,
        }
        message.update(updates)
        return message

    def parse(self, *messages):
        return runner.parse_cargo_test_executables(
            "macro diagnostic\n"
            + "\n".join(json.dumps(message) for message in messages),
            "/workspace",
        )

    def test_discovers_modern_traditional_and_custom_paths_including_fresh(self):
        messages = [
            self.artifact(path="target/debug/deps/ir_interpret_test-1234", fresh=False),
            self.artifact(
                path="/cache/target/release/build/xlsynth/abcd/out/ir_interpret_test"
            ),
            self.artifact(
                path="/cache/with spaces (release)/aarch64-unknown-linux-gnu/debug/ir_interpret_test"
            ),
        ]
        found = self.parse(*messages)
        self.assertEqual(
            [item.path for item in found],
            [
                "/workspace/target/debug/deps/ir_interpret_test-1234",
                messages[1]["executable"],
                messages[2]["executable"],
            ],
        )
        self.assertTrue(all(item.cwd == "/workspace/xlsynth" for item in found))

    def test_ignores_non_test_artifacts_and_deduplicates_executable(self):
        artifact = self.artifact()
        found = self.parse(
            self.artifact(profile={"test": False}),
            self.artifact(executable=None),
            self.artifact(reason="build-script-executed"),
            {"reason": "build-finished", "success": True},
            artifact,
            artifact,
        )
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].path, artifact["executable"])

    def test_metadata_retains_package_cwd_for_duplicate_target_names_and_libs(self):
        first = self.artifact(name="shared-name", package="one")
        second = self.artifact(
            name="shared-name",
            package="two",
            target={"name": "shared-name", "kind": ["lib"]},
        )
        found = self.parse(first, second)
        self.assertEqual(
            [item.cwd for item in found], ["/workspace/one", "/workspace/two"]
        )
        self.assertEqual(
            [item.target_name for item in found], ["shared_name", "shared_name"]
        )

    def test_rejects_incomplete_or_conflicting_cargo_metadata(self):
        with self.assertRaisesRegex(ValueError, "manifest path"):
            self.parse(self.artifact(manifest_path=None))
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.parse(
                self.artifact(), self.artifact(manifest_path="/different/Cargo.toml")
            )

    def test_filters_match_normalized_targets_and_explicit_paths(self):
        found = self.parse(self.artifact(name="sample-usage"), self.artifact())
        selected = runner.select_test_executables(
            found, ["sample_usage", "/abcd/out/ir_"]
        )
        self.assertEqual(selected, found)
        self.assertEqual(
            runner.select_test_executables(found, ["sample-usage"]), found[:1]
        )

    def test_rejects_empty_discovery_and_each_unmatched_filter(self):
        with self.assertRaisesRegex(ValueError, "Selected 0 test executables"):
            runner.select_test_executables([], [])
        found = self.parse(self.artifact())
        with self.assertRaisesRegex(
            ValueError, "Unmatched executable filters: missing"
        ):
            runner.select_test_executables(found, ["ir_interpret_test", "missing"])
        self.assertIn("ir_interpret_test", str(found))

    def test_explicitly_excluded_targets_cannot_satisfy_a_filter(self):
        found = self.parse(self.artifact(name="version_test"))
        with self.assertRaisesRegex(
            ValueError, "Unmatched executable filters: version_test"
        ):
            runner.select_test_executables(found, ["version_test"])

    def test_compilation_preserves_workspace_custom_target_dir_and_profile(self):
        result = subprocess.CompletedProcess([], 0, stdout=json.dumps(self.artifact()))
        for release in (False, True):
            with mock.patch.dict(
                os.environ, {"CARGO_TARGET_DIR": "/cache/target (tests)"}
            ):
                with mock.patch.object(
                    runner.subprocess, "run", return_value=result
                ) as run:
                    found = runner.compile_test_executables(release, "/workspace")
                    command = run.call_args[0][0]
                    self.assertEqual(
                        command,
                        [
                            "cargo",
                            "test",
                            "--no-run",
                            "--workspace",
                            "--message-format=json",
                        ]
                        + (["--release"] if release else []),
                    )
                    self.assertEqual(
                        run.call_args[1]["env"]["CARGO_TARGET_DIR"],
                        "/cache/target (tests)",
                    )
                    self.assertEqual(run.call_args[1]["cwd"], "/workspace")
                    self.assertEqual(len(found), 1)

    def test_compile_failure_preserves_rustc_diagnostics_and_failure(self):
        message = {
            "reason": "compiler-message",
            "message": {"rendered": "a real compiler error\n"},
        }
        result = subprocess.CompletedProcess(["cargo"], 42, stdout=json.dumps(message))
        with mock.patch.object(runner.subprocess, "run", return_value=result):
            with self.assertRaises(subprocess.CalledProcessError) as raised:
                runner.compile_test_executables(False, "/workspace")
        self.assertEqual(raised.exception.returncode, 42)
        self.assertIn("a real compiler error", self.stderr.getvalue())

    def test_subset_keeps_first_test_and_ignores_summary_and_benchmarks(self):
        output = b"first: test\ntest_validate_fail: test\nbenchmark_named_test: test\nbench: benchmark\n\n3 tests, 1 benchmark\n"
        with mock.patch.object(runner.subprocess, "check_output", return_value=output):
            with mock.patch.object(
                runner, "run_valgrind", return_value=(defaultdict(float), 2)
            ) as run:
                result = runner.run_subset_of_tests(
                    "/exe", "/package", ["validate_fail"], "/supp"
                )
        self.assertEqual(result[1:], (2, 2))
        self.assertEqual(
            run.call_args[1]["test_filters"], ["first", "benchmark_named_test"]
        )
        self.assertEqual(run.call_args[0], ("/exe", "/package", "/supp"))

    def test_subset_config_uses_cargo_target_for_hashed_and_unhashed_binaries(self):
        for path in ("/out/anything-1234567890abcdef", "/out/unhashed-name"):
            executable = self.parse(self.artifact(name="sample-usage", path=path))[0]
            with mock.patch.object(
                runner, "run_subset_of_tests", return_value=(defaultdict(float), 3, 3)
            ) as run:
                result = runner.run_single_test_binary(executable, "/workspace")
            self.assertEqual(result[1:], (3, 3))
            self.assertEqual(run.call_args[0][1], "/workspace/xlsynth")
            self.assertIn("test_validate_fail", run.call_args[0][2])

    def test_sharding_retains_cwd_filters_first_test_and_distinct_task_ids(self):
        artifacts = self.parse(
            self.artifact(package="one"), self.artifact(package="two")
        )
        output = b"test_force_assert_fn: test\ntest_ir_interpret_array_values: test\n\n2 tests, 0 benchmarks\n"
        tasks = {}
        with mock.patch.object(runner.subprocess, "check_output", return_value=output):
            with mock.patch.object(
                runner, "run_single_test_case", return_value=(defaultdict(float), 1, 1)
            ) as run:
                for artifact in artifacts:
                    tasks.update(
                        runner.submit_valgrind_tasks(
                            ImmediateExecutor(), artifact, "/workspace"
                        )
                    )
        self.assertEqual(len(tasks), 2)
        self.assertEqual(len(set(tasks.values())), 2)
        self.assertEqual(
            [call[0][2] for call in run.call_args_list],
            ["/workspace/one", "/workspace/two"],
        )
        self.assertTrue(
            all(call[0][1] == "test_force_assert_fn" for call in run.call_args_list)
        )

    def test_empty_shard_selection_and_failed_listing_are_errors(self):
        executable = self.parse(self.artifact())[0]
        with mock.patch.object(
            runner.subprocess,
            "check_output",
            return_value=b"test_ir_interpret_array_values: test\n1 test, 0 benchmarks\n",
        ):
            with self.assertRaisesRegex(ValueError, "No tests left"):
                runner.submit_valgrind_tasks(
                    ImmediateExecutor(), executable, "/workspace"
                )
        with mock.patch.object(
            runner.subprocess,
            "check_output",
            side_effect=subprocess.CalledProcessError(7, ["list"]),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                runner.submit_valgrind_tasks(
                    ImmediateExecutor(), executable, "/workspace"
                )

    def test_valgrind_preserves_paths_and_uses_exact_case_names(self):
        output = json.dumps(
            {"type": "test", "event": "ok", "name": "a::first", "exec_time": 0.1}
        )
        result = subprocess.CompletedProcess([], 0, stdout=output, stderr="")
        with mock.patch.object(runner, "_sanity_check_test_executable"):
            with mock.patch.object(
                runner.subprocess, "run", return_value=result
            ) as run:
                _, count = runner.run_valgrind(
                    "/path with (spaces)/test", "/package", "/supp file", ["a::first"]
                )
        self.assertEqual(count, 1)
        command = run.call_args[0][0]
        index = command.index("/path with (spaces)/test")
        self.assertEqual(
            command[index : index + 3],
            ["/path with (spaces)/test", "a::first", "--exact"],
        )
        self.assertIn("--test-threads=1", command)
        self.assertEqual(run.call_args[1]["cwd"], "/package")

    def test_valgrind_zero_test_events_and_process_failure_cannot_succeed(self):
        empty = subprocess.CompletedProcess(
            [], 0, stdout='{"type":"suite","event":"ok","passed":0}', stderr=""
        )
        with mock.patch.object(runner, "_sanity_check_test_executable"):
            with mock.patch.object(runner.subprocess, "run", return_value=empty):
                with self.assertRaisesRegex(ValueError, "No successful test events"):
                    runner.run_valgrind("/test", "/cwd", "/supp", expect_tests=True)
            with mock.patch.object(
                runner.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(9, ["valgrind"]),
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    runner.run_valgrind("/test", "/cwd", "/supp")

    def invoke_main(self, artifacts, worker_result=None, filters="sample_usage"):
        version = subprocess.CompletedProcess([], 0, stdout="cargo 1.94.0-nightly\n")
        patches = [
            mock.patch.object(
                runner.sys, "argv", ["runner", "--filter-to-run=" + filters]
            ),
            mock.patch.dict(os.environ, {"XLSYNTH_TOOLS": "/tools"}),
            mock.patch.object(runner.subprocess, "run", return_value=version),
            mock.patch.object(
                runner, "compile_test_executables", return_value=artifacts
            ),
            mock.patch.object(
                runner.concurrent.futures,
                "ProcessPoolExecutor",
                return_value=ImmediateExecutor(),
            ),
            mock.patch.object(
                runner,
                "run_single_test_binary",
                return_value=worker_result or (defaultdict(float), 0, 0),
            ),
        ]
        for patch in patches:
            patch.start()
        try:
            with self.assertRaises(SystemExit) as raised:
                runner.main()
            return raised.exception.code
        finally:
            for patch in reversed(patches):
                patch.stop()

    def test_main_fails_for_no_executables_and_zero_total_test_cases(self):
        self.assertEqual(self.invoke_main([]), 1)
        artifacts = self.parse(self.artifact(name="sample-usage"))
        self.assertEqual(self.invoke_main(artifacts), 1)
        self.assertIn("No test cases completed", self.stderr.getvalue())

    def test_main_reports_actual_execution_and_propagates_worker_failure(self):
        artifacts = self.parse(self.artifact(name="sample-usage"))
        self.assertEqual(self.invoke_main(artifacts, (defaultdict(float), 3, 3)), 0)
        self.assertIn("Completed 3 test cases", self.stdout.getvalue())
        self.assertEqual(self.invoke_main(artifacts, ValueError("worker failed")), 1)


if __name__ == "__main__":
    unittest.main()
