# SPDX-License-Identifier: Apache-2.0

"""Test fuzz orchestration without compiling or executing real fuzz targets.

Like the runner, these tests require Python 3.11+. CI runs them in the fuzz job.
"""

import io
import json
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
        self.root = Path(temporary.name).resolve()
        self.output = self.root / "target/fuzz-ci"
        self.projects = {}
        self.commands = []
        self.logs = []
        self.failed_target = None
        self.build_failure = False

    def project(self, name, targets, features=()):
        path = self.root / name / "fuzz"
        path.mkdir(parents=True, exist_ok=True)
        if not isinstance(features, dict):
            features = {name: [] for name in features}
        lines = ["[features]"]
        lines.extend(
            f"{name} = {json.dumps(values)}" for name, values in features.items()
        )
        for target in targets:
            target = {"name": target} if isinstance(target, str) else target
            lines.append("[[bin]]")
            lines.extend(
                f"{name} = {json.dumps(value)}" for name, value in target.items()
            )
        (path / "Cargo.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.projects[path] = targets
        return path

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
            mock.patch.object(runner, "__file__", str(self.root / "scripts/runner.py")),
            mock.patch.object(runner, "run_cmd", self.build),
            mock.patch.object(runner, "run_cmd_captured", self.execute),
        ):
            return runner.main()

    def test_default_shares_build_outputs_and_preserves_smoke_settings(self):
        path = self.project("alpha", ["first"], ["with-bitwuzla-system"])
        self.assertEqual(self.run_suite(), 0)
        common = [
            "--fuzz-dir",
            str(path),
            "--sanitizer",
            "none",
            "--features",
            "with-bitwuzla-system",
            "--target-dir",
            str(self.output),
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

    def test_default_directory_also_rejects_colliding_names(self):
        self.project("alpha", ["same"])
        self.project("beta", ["same"])
        with self.assertRaises(SystemExit):
            self.run_suite()
        self.assertEqual(self.commands, [])

    def test_default_exclusions_are_built_but_not_run(self):
        excluded = sorted(runner.DEFAULT_SKIPPED_FUZZ_TARGETS)[0]
        path = self.project("alpha", ["first", excluded])
        self.assertEqual(self.run_suite("--features", ""), 0)
        self.assertEqual(
            self.commands[0],
            [
                "cargo",
                "fuzz",
                "build",
                "--fuzz-dir",
                str(path),
                "--sanitizer",
                "none",
                "--target-dir",
                str(self.output),
            ],
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

    def test_shared_configuration_cannot_bypass_validation_in_extra_args(self):
        self.project("alpha", ["first"])
        for extra in [
            "--target-dir=target/shared",
            "--target-dir target/shared",
            "--features=x",
            "-Fx",
            "--all-features",
        ]:
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

    def test_unknown_features_fail_before_building(self):
        self.project("alpha", ["first"])
        with self.assertRaises(SystemExit):
            self.run_suite("--features", "typo")
        self.assertEqual(self.commands, [])

    def test_required_features_expand_transitively_and_follow_defaults(self):
        features = {"default": ["a"], "a": ["b"], "b": ["a"]}
        self.project(
            "alpha", ["plain", {"name": "gated", "required-features": ["b"]}], features
        )
        self.assertEqual(self.run_suite("--features", ""), 0)
        self.assertEqual(len(self.commands), 3)
        self.commands.clear()
        self.assertEqual(
            self.run_suite("--features", "", "--fuzz-run-args=--no-default-features"), 0
        )
        self.assertEqual(len(self.commands), 2)
        self.assertEqual(self.commands[-1][self.commands[-1].index("--") - 1], "plain")

    def test_only_packages_with_selected_targets_are_built(self):
        self.project(
            "alpha",
            [{"name": "formal", "required-features": ["solver"]}],
            {"solver": []},
        )
        self.project("beta", ["plain"])
        self.assertEqual(self.run_suite("--features", ""), 0)
        self.assertEqual(len(self.commands), 2)

    def test_warm_cache_cannot_resurrect_removed_or_disabled_targets(self):
        features = {"solver": []}
        gated = {"name": "formal", "required-features": ["solver"]}
        self.project("alpha", ["kept", "removed", gated], features)
        self.assertEqual(self.run_suite("--features", "solver"), 0)
        self.assertEqual(len(self.commands), 4)
        self.output.mkdir(parents=True)
        for name in ["removed", "formal", "Cargo.toml", "Cargo.lock"]:
            (self.output / name).write_text("stale cached state")
        path = self.project("alpha", ["kept", gated], features)
        before = (path / "Cargo.toml").read_bytes()
        self.commands.clear()
        self.assertEqual(self.run_suite("--features", ""), 0)
        self.assertEqual(len(self.commands), 2)
        self.assertEqual(self.commands[-1][self.commands[-1].index("--") - 1], "kept")
        self.assertTrue(all("--features" not in cmd for cmd in self.commands))
        self.assertEqual((path / "Cargo.toml").read_bytes(), before)
        self.assertFalse((self.root / "target/fuzz-smoke-workspace").exists())


if __name__ == "__main__":
    unittest.main()
