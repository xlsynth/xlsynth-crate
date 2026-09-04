# SPDX-License-Identifier: Apache-2.0

"""Test shared fuzz orchestration without compiling or executing fuzz targets.

Like the runner, these tests require Python 3.11+. CI runs them in the fuzz job.
"""

import io
import os
from pathlib import Path
import subprocess
import tempfile
import tomllib
import unittest
from unittest import mock

from fuzz_smoke_workspace import (
    prepare_smoke_workspace,
    smoke_cache_outputs,
    toml_value,
)
import run_all_fuzz_tests as runner


class FuzzRunnerTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="xlsynth-fuzz-runner-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.output = self.root / "target/fuzz-ci"
        self.workspace = self.root / "target/fuzz-smoke-workspace"
        self.projects = []
        self.commands = []
        self.logs = []
        self.failed_target = None
        self.build_failure = False

    def project(self, name, targets, features=None, dependencies=None, library=False):
        path = self.root / name / "fuzz"
        path.mkdir(parents=True, exist_ok=True)
        manifest = {
            "package": {"name": name + "-fuzz", "version": "0.0.0", "edition": "2024"},
            "features": features
            if features is not None
            else {"with-bitwuzla-system": []},
            "dependencies": dependencies or {},
        }
        if library:
            manifest["lib"] = {"name": name + "_helpers", "path": "src/lib.rs"}
        lines = [f"{key} = {toml_value(value)}" for key, value in manifest.items()]
        for target in targets:
            target = {"name": target} if isinstance(target, str) else dict(target)
            target.update(
                path=f"fuzz_targets/{target['name']}.rs", test=False, doc=False
            )
            lines.append("[[bin]]")
            lines.extend(
                f"{key} = {toml_value(value)}" for key, value in target.items()
            )
        (path / "Cargo.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")
        if path not in self.projects:
            self.projects.append(path)
        return path

    def build(self, cmd):
        self.commands.append(cmd)
        if self.build_failure:
            raise subprocess.CalledProcessError(42, cmd)

    def execute(self, cmd, log_dir):
        self.commands.append(cmd)
        target = cmd[cmd.index("--") - 2]
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
            mock.patch.object(runner, "__file__", str(self.root / "scripts/runner.py")),
            mock.patch.object(runner, "find_fuzz_dirs", return_value=self.projects),
            mock.patch.object(runner, "run_cmd", self.build),
            mock.patch.object(runner, "run_cmd_captured", self.execute),
        ):
            return runner.main()

    def generated(self):
        return tomllib.loads((self.workspace / "Cargo.toml").read_text())

    def test_one_build_and_identical_features_for_every_run(self):
        first = self.project("alpha", ["first"])
        second = self.project("beta", ["second"], features={})
        self.assertEqual(self.run_suite(), 0)
        common = [
            "--fuzz-dir",
            str(self.workspace),
            "--sanitizer",
            "none",
            "--target-dir",
            str(self.output),
            "--features",
            "with-bitwuzla-system",
        ]
        self.assertEqual(
            self.commands,
            [
                ["cargo", "fuzz", "build", *common],
                *[
                    [
                        "cargo",
                        "fuzz",
                        "run",
                        *common,
                        target,
                        str(path / "corpus" / target),
                        "--",
                        "-max_total_time=5",
                        "-timeout=60",
                    ]
                    for path, target in [(first, "first"), (second, "second")]
                ],
            ],
        )
        self.assertTrue(all(not path.exists() for path in self.logs))

    def test_relative_path_and_extra_build_flags_are_shared(self):
        self.project("alpha", ["first"], {"foo": [], "foobar": []})
        self.project("beta", ["second"], {"bar": []})
        # A relative path containing '..' is normalized once for every command.
        relative = Path(os.path.relpath(self.root / "fuzz output", Path.cwd()))
        self.assertEqual(
            self.run_suite(
                "--target-dir",
                str(relative),
                "--features",
                " foobar, bar ",
                "--sanitizer",
                "address",
                "--fuzz-run-args=--dev",
                "--fuzz-bin-args=-runs=1",
            ),
            0,
        )
        for cmd in self.commands:
            self.assertEqual(cmd[cmd.index("--features") + 1], "bar,foobar")
            self.assertEqual(
                cmd[cmd.index("--target-dir") + 1], str(self.root / "fuzz output")
            )
            self.assertIn("--dev", cmd)
            self.assertEqual(cmd[cmd.index("--sanitizer") + 1], "address")

    def test_colliding_names_fail_before_build_including_exclusions(self):
        for name in ["same", sorted(runner.DEFAULT_SKIPPED_FUZZ_TARGETS)[0]]:
            with self.subTest(name=name):
                self.projects.clear()
                self.project("alpha-" + name, [name])
                self.project("beta-" + name, [name])
                with self.assertRaises(SystemExit):
                    self.run_suite()
                self.assertEqual(self.commands, [])

    def test_default_exclusions_stay_registered_but_are_not_run(self):
        excluded = sorted(runner.DEFAULT_SKIPPED_FUZZ_TARGETS)[0]
        self.project("alpha", ["first", excluded])
        self.assertEqual(self.run_suite(), 0)
        self.assertEqual(len(self.commands), 2)
        self.assertEqual(
            {b["name"] for b in self.generated()["bin"]}, {"first", excluded}
        )

    def test_required_features_follow_transitive_enabling(self):
        self.project(
            "alpha",
            ["first", {"name": "formal", "required-features": ["has-solver"]}],
            {"with-bitwuzla-system": ["has-solver"], "has-solver": []},
        )
        self.assertEqual(self.run_suite(), 0)
        self.assertEqual(len(self.commands), 3)
        self.commands.clear()
        self.assertEqual(self.run_suite("--features", ""), 0)
        self.assertEqual(len(self.commands), 2)
        self.assertTrue(all("--features" not in cmd for cmd in self.commands))

    def test_unknown_features_fail_before_build(self):
        self.project("alpha", ["first"])
        with self.assertRaises(SystemExit):
            self.run_suite("--features", "typo")
        self.assertEqual(self.commands, [])

    def test_build_configuration_cannot_bypass_validation(self):
        self.project("alpha", ["first"])
        for extra in [
            "--target-dir=x",
            "--target-dir x",
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
        self.assertTrue(all(not path.exists() for path in self.logs))

    def test_find_fuzz_dirs_is_sorted(self):
        second = self.project("beta", ["second"])
        first = self.project("alpha", ["first"])
        self.assertEqual(runner.find_fuzz_dirs(self.root), [first, second])

    def test_dependency_features_and_version_requirements_are_combined(self):
        self.project(
            "alpha",
            ["first"],
            dependencies={
                "serde": {
                    "version": "1",
                    "features": ["derive"],
                    "default-features": False,
                },
            },
        )
        self.project(
            "beta",
            ["second"],
            dependencies={
                "serde": {"version": "1.0", "features": ["alloc"]},
            },
        )
        self.run_suite()
        self.assertEqual(
            self.generated()["dependencies"]["serde"],
            {
                "version": "1, 1.0",
                "features": ["alloc", "derive"],
                "default-features": True,
            },
        )

    def test_library_features_are_forwarded_and_sources_are_not_copied(self):
        original = self.project(
            "alpha", ["first"], {"with-bitwuzla-system": []}, library=True
        )
        self.run_suite()
        manifest = self.generated()
        self.assertNotIn("version", manifest["package"])
        self.assertFalse(manifest["dependencies"]["alpha-fuzz"]["default-features"])
        self.assertEqual(manifest["dependencies"]["alpha-fuzz"]["path"], str(original))
        self.assertEqual(
            manifest["features"]["with-bitwuzla-system"],
            ["alpha-fuzz/with-bitwuzla-system"],
        )
        self.assertEqual(
            manifest["bin"][0]["path"], str(original / "fuzz_targets/first.rs")
        )

    def test_conflicting_dependency_sources_fail(self):
        self.project("alpha", ["first"], dependencies={"shared": {"path": "../one"}})
        self.project("beta", ["second"], dependencies={"shared": {"path": "../two"}})
        with self.assertRaisesRegex(ValueError, "conflicting path"):
            prepare_smoke_workspace(self.projects, self.output)

    def test_explicit_default_package_name_matches_implicit_name(self):
        self.project("alpha", ["first"], dependencies={"shared": "1"})
        self.project(
            "beta",
            ["second"],
            dependencies={"shared": {"version": "1", "package": "shared"}},
        )
        prepare_smoke_workspace(self.projects, self.output)

    def test_unknown_dependency_configuration_is_not_silently_lost(self):
        self.project(
            "alpha",
            ["first"],
            dependencies={"shared": {"version": "1", "optional": True}},
        )
        with self.assertRaisesRegex(ValueError, "unsupported smoke manifest"):
            prepare_smoke_workspace(self.projects, self.output)

    def test_manifest_is_deterministic_and_not_rewritten_when_unchanged(self):
        self.project("beta", ["second"])
        self.project("alpha", ["first"])
        prepare_smoke_workspace(self.projects, self.output)
        path = self.output / "Cargo.toml"
        before = path.stat().st_mtime_ns
        prepare_smoke_workspace(list(reversed(self.projects)), self.output)
        self.assertEqual(path.stat().st_mtime_ns, before)

    def test_build_hook_is_preserved_and_conflicts_fail(self):
        first = self.project("alpha", ["first"])
        second = self.project("beta", ["second"])
        content = 'fn main() { println!("cargo:rerun-if-changed=build.rs"); }\n'
        for project in self.projects:
            (project / "build.rs").write_text(content)
        prepare_smoke_workspace(self.projects, self.output)
        self.assertEqual((self.output / "build.rs").read_text(), content)
        (first / "build.rs").write_text("fn main() {}\n")
        with self.assertRaisesRegex(ValueError, "build scripts must match"):
            prepare_smoke_workspace(self.projects, self.output)
        for project in (first, second):
            (project / "build.rs").unlink()
        prepare_smoke_workspace(self.projects, self.output)
        self.assertFalse((self.output / "build.rs").exists())

    def test_feature_defaults_and_cycles(self):
        self.project("alpha", ["first"], {"default": ["a"], "a": ["b"], "b": ["a"]})
        workspace = prepare_smoke_workspace(self.projects, self.output)
        self.assertEqual(workspace.enabled_features([], True), {"default", "a", "b"})
        self.assertEqual(workspace.enabled_features([], False), set())

    def test_workspace_and_cache_directories_must_not_overlap(self):
        self.project("alpha", ["first"])
        for workspace in [self.output, self.output / "nested", self.output.parent]:
            with self.subTest(workspace=workspace), self.assertRaises(SystemExit):
                self.run_suite("--workspace-dir", str(workspace))
        self.assertEqual(self.commands, [])

    def test_non_generated_workspace_is_not_overwritten(self):
        original = self.project("alpha", ["first"])
        before = (original / "Cargo.toml").read_text()
        with self.assertRaises(SystemExit):
            self.run_suite("--workspace-dir", str(original))
        self.assertEqual((original / "Cargo.toml").read_text(), before)

    def test_prepare_only_resolves_current_lockfile_without_building(self):
        self.project("alpha", ["first"])
        external = self.root / "runner-temp/smoke"
        outputs = self.root / "github-output"
        with mock.patch.object(
            runner,
            "smoke_cache_outputs",
            return_value={
                "cache-workspace": "workspace -> ../build",
                "cache-key": "current",
            },
        ) as cache:
            self.assertEqual(
                self.run_suite(
                    "--workspace-dir",
                    str(external),
                    "--prepare-only",
                    "--github-output",
                    str(outputs),
                ),
                0,
            )
        self.assertEqual(
            self.commands,
            [
                [
                    "cargo",
                    "generate-lockfile",
                    "--manifest-path",
                    str(external / "Cargo.toml"),
                ],
            ],
        )
        self.assertEqual(cache.call_args.args[:3], (self.root, external, self.output))
        self.assertEqual(
            outputs.read_text(),
            "cache-workspace=workspace -> ../build\ncache-key=current\n",
        )
        self.assertFalse(self.logs)

    def test_warm_outputs_do_not_resurrect_removed_or_feature_gated_targets(self):
        self.project(
            "alpha",
            [
                "kept",
                "removed",
                {
                    "name": "formal",
                    "required-features": ["solver"],
                },
            ],
            features={"solver": []},
        )
        self.assertEqual(self.run_suite("--features", "solver"), 0)
        # Model stale files restored into the build directory. None can become
        # input to current target selection, feature selection or lock resolution.
        self.output.mkdir(parents=True)
        for name in ["removed", "formal", "Cargo.lock", "Cargo.toml"]:
            (self.output / name).write_text("stale build-cache state")
        self.project(
            "alpha",
            [
                "kept",
                {
                    "name": "formal",
                    "required-features": ["solver"],
                },
            ],
            features={"solver": []},
        )
        self.commands.clear()
        self.assertEqual(self.run_suite("--features", ""), 0)
        self.assertEqual(len(self.commands), 2)
        self.assertEqual(self.commands[-1][self.commands[-1].index("--") - 2], "kept")
        self.assertTrue(all("--features" not in cmd for cmd in self.commands))
        self.assertEqual(
            {b["name"] for b in self.generated()["bin"]}, {"kept", "formal"}
        )
        self.assertFalse((self.workspace / "Cargo.lock").exists())

    def test_cache_mapping_and_key_cover_manifests_lockfile_and_flags(self):
        original = self.project("alpha", ["first"])
        prepare_smoke_workspace(self.projects, self.workspace)
        (self.workspace / "Cargo.lock").write_text("current resolution")
        paths = b"alpha/fuzz/Cargo.toml\0"

        def outputs(flags):
            with mock.patch(
                "fuzz_smoke_workspace.subprocess.check_output", return_value=paths
            ):
                return smoke_cache_outputs(
                    self.root, self.workspace, self.output, flags
                )

        first = outputs(["--features", "solver"])
        root, target = first["cache-workspace"].split(" -> ")
        self.assertEqual((Path(root) / target).resolve(), self.output)
        self.assertEqual(outputs(["--features", "solver"]), first)
        self.assertNotEqual(outputs([])["cache-key"], first["cache-key"])
        (self.workspace / "Cargo.lock").write_text("new resolution")
        self.assertNotEqual(
            outputs(["--features", "solver"])["cache-key"], first["cache-key"]
        )
        (self.workspace / "Cargo.lock").write_text("current resolution")
        with (original / "Cargo.toml").open("a") as stream:
            stream.write("\n# dependency configuration changed\n")
        self.assertNotEqual(
            outputs(["--features", "solver"])["cache-key"], first["cache-key"]
        )


if __name__ == "__main__":
    unittest.main()
