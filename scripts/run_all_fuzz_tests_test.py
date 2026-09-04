# SPDX-License-Identifier: Apache-2.0

"""Test fuzz orchestration without compiling or executing real fuzz targets.

Like the runner, these tests require Python 3.11+. CI runs them in the fuzz job.
"""

import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import run_all_fuzz_tests as runner


class FuzzRunnerTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="xlsynth-fuzz-runner-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.projects = {}
        self.commands = []
        self.logs = []
        self.failed_target = None
        self.build_failure = False

    def project(self, name, targets, features=()):
        path = self.root / name / "fuzz"
        path.mkdir(parents=True)
        (path / "Cargo.toml").write_text(
            "[features]\n" + "".join(f"{feature} = []\n" for feature in features),
            encoding="utf-8",
        )
        self.projects[path] = targets
        return path

    def list_targets(self, cmd, **kwargs):
        self.assertEqual(cmd[:4], ["cargo", "fuzz", "list", "--fuzz-dir"])
        return "\n".join(self.projects[Path(cmd[4])]) + "\n"

    def build(self, cmd):
        self.commands.append(cmd)
        if self.build_failure:
            raise subprocess.CalledProcessError(42, cmd)

    def execute(self, cmd, log_dir):
        self.commands.append(cmd)
        target = cmd[cmd.index("--") - 1]
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=log_dir, delete=False
        ) as stream:
            stream.write("Done 17 runs in 5 second(s)\n")
            path = Path(stream.name)
        self.logs.append(path)
        return (1 if target == self.failed_target else 0), path

    def run_suite(self, *args):
        with (
            mock.patch.object(runner.sys, "argv", ["runner", "--threads", "1", *args]),
            mock.patch.object(runner.sys, "stdout", new_callable=io.StringIO),
            mock.patch.object(runner.sys, "stderr", new_callable=io.StringIO),
            mock.patch.object(
                runner, "find_fuzz_dirs", return_value=list(self.projects)
            ),
            mock.patch.object(runner.subprocess, "check_output", self.list_targets),
            mock.patch.object(runner, "run_cmd", self.build),
            mock.patch.object(runner, "run_cmd_captured", self.execute),
        ):
            return runner.main()

    def test_default_keeps_workspace_outputs_and_smoke_settings(self):
        path = self.project("alpha", ["first"], ["with-bitwuzla-system"])
        self.assertEqual(self.run_suite(), 0)
        common = [
            "--fuzz-dir",
            str(path),
            "--sanitizer",
            "none",
            "--features",
            "with-bitwuzla-system",
        ]
        self.assertEqual(
            self.commands,
            [
                ["cargo", "fuzz", "build", *common],
                [
                    "cargo",
                    "fuzz",
                    "run",
                    *common,
                    "first",
                    "--",
                    "-max_total_time=5",
                    "-timeout=60",
                ],
            ],
        )
        self.assertTrue(all(not path.exists() for path in self.logs))

    def test_shared_relative_path_and_flags_match_for_build_and_run(self):
        first = self.project("alpha", ["first", "second"], ["foo", "foobar"])
        second = self.project("beta", ["third"], ["foo", "bar"])
        target_dir = "target/fuzz output"
        self.assertEqual(
            self.run_suite(
                "--target-dir",
                target_dir,
                "--features",
                " foobar, bar ",
                "--sanitizer",
                "address",
                "--fuzz-run-args=--dev",
                "--fuzz-bin-args=-runs=1 -timeout=60",
            ),
            0,
        )
        expected_args = {
            path: [
                "--fuzz-dir",
                str(path),
                "--sanitizer",
                "address",
                "--features",
                feature,
                "--target-dir",
                str(Path(target_dir).resolve()),
                "--dev",
            ]
            for path, feature in [(first, "foobar"), (second, "bar")]
        }
        # Every prebuild precedes execution, and flags match exactly. In
        # particular, requesting "foobar" must not also enable "foo".
        self.assertEqual(
            self.commands,
            [
                ["cargo", "fuzz", "build", *expected_args[first]],
                ["cargo", "fuzz", "build", *expected_args[second]],
                *[
                    [
                        "cargo",
                        "fuzz",
                        "run",
                        *expected_args[path],
                        target,
                        "--",
                        "-runs=1",
                        "-timeout=60",
                    ]
                    for path, target in [
                        (first, "first"),
                        (first, "second"),
                        (second, "third"),
                    ]
                ],
            ],
        )

    def test_shared_directory_rejects_colliding_names_before_building(self):
        self.project("alpha", ["same"])
        self.project("beta", ["same"])
        with self.assertRaises(SystemExit) as caught:
            self.run_suite("--target-dir", str(self.root / "shared"))
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(self.commands, [])

    def test_separate_directories_allow_colliding_names(self):
        self.project("alpha", ["same"])
        self.project("beta", ["same"])
        self.assertEqual(self.run_suite(), 0)
        self.assertEqual(len(self.commands), 4)

    def test_default_exclusions_are_built_but_not_run(self):
        excluded = sorted(runner.DEFAULT_SKIPPED_FUZZ_TARGETS)[0]
        path = self.project("alpha", ["first", excluded])
        self.assertEqual(self.run_suite("--features", ""), 0)
        self.assertEqual(
            self.commands[0],
            ["cargo", "fuzz", "build", "--fuzz-dir", str(path), "--sanitizer", "none"],
        )
        self.assertEqual(len(self.commands), 2)
        self.assertEqual(self.commands[1][self.commands[1].index("--") - 1], "first")

    def test_excluded_target_collisions_are_also_rejected(self):
        excluded = sorted(runner.DEFAULT_SKIPPED_FUZZ_TARGETS)[0]
        self.project("alpha", ["first", excluded])
        self.project("beta", ["second", excluded])
        with self.assertRaises(SystemExit):
            self.run_suite("--target-dir", str(self.root / "shared"))
        self.assertEqual(self.commands, [])

    def test_empty_requested_features_disable_optional_features(self):
        self.project("alpha", ["first"], ["with-bitwuzla-system"])
        self.assertEqual(self.run_suite("--features", ""), 0)
        self.assertTrue(all("--features" not in cmd for cmd in self.commands))

    def test_target_dir_cannot_bypass_validation_in_extra_args(self):
        self.project("alpha", ["first"])
        for extra in ["--target-dir=target/shared", "--target-dir target/shared"]:
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.run_suite("--fuzz-run-args=" + extra)
        self.assertEqual(self.commands, [])

    def test_build_failure_prevents_execution(self):
        self.project("alpha", ["first"])
        self.build_failure = True
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_suite()
        self.assertEqual(len(self.commands), 1)
        self.assertEqual(self.logs, [])

    def test_target_failure_fails_suite_and_cleans_logs(self):
        self.project("alpha", ["first", "second"])
        self.failed_target = "first"
        self.assertEqual(self.run_suite(), 1)
        self.assertEqual(len(self.commands), 3)
        self.assertTrue(all(not path.exists() for path in self.logs))

    def test_find_fuzz_dirs_is_sorted(self):
        second = self.project("beta", ["second"])
        first = self.project("alpha", ["first"])
        (self.root / "not_a_crate").mkdir()
        self.assertEqual(runner.find_fuzz_dirs(self.root), [first, second])


if __name__ == "__main__":
    unittest.main()
