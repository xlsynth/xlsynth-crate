# SPDX-License-Identifier: Apache-2.0

"""Exercises Docker workspace preparation with fake tools and no system writes."""

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parent.parent

FAKE_TOOL = r"""#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path
import sys

program = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["DOCKER_TEST_LOG"], "a") as stream:
    stream.write(json.dumps([program, args, os.environ.get("CARGO_NET_OFFLINE")]) + "\n")
if program == "cargo" and args[0] == "fetch" and os.environ.get("DOCKER_TEST_FAIL_FETCH"):
    sys.exit(42)
"""

FAKE_DOWNLOAD = r"""# SPDX-License-Identifier: Apache-2.0
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("-p")
parser.add_argument("-o")
parser.add_argument("-v")
args = parser.parse_args()
tools = Path(args.o)
(tools / "xls/dslx/stdlib").mkdir(parents=True, exist_ok=True)
(tools / ("libxls-" + args.p + ".so")).touch()
"""


class DockerPrepareTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="xlsynth-docker-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.repo = self.root / "repo with spaces"
        self.test_home = self.root / "home with spaces"
        self.test_home.mkdir()
        (self.repo / "docker").mkdir(parents=True)
        (self.repo / "scripts").mkdir()
        (self.repo / "xlsynth-sys").mkdir()
        shutil.copy2(REPO_ROOT / "docker/prepare_workspace.sh", self.repo / "docker")
        shutil.copy2(REPO_ROOT / "docker/check_offline.sh", self.repo / "docker")
        shutil.copy2(
            REPO_ROOT / "scripts/get_required_libxls_dso_and_tools_release_tag.py",
            self.repo / "scripts",
        )
        (self.repo / "scripts/download_release.py").write_text(FAKE_DOWNLOAD)
        (self.repo / "xlsynth-sys/build.rs").write_text(
            "// SPDX-License-Identifier: Apache-2.0\n"
            'const RELEASE_LIB_VERSION_TAG: &str = "v0.0.1";\n'
        )
        fake_bin = self.root / "bin"
        fake_bin.mkdir()
        for program in ("cargo", "pre-commit"):
            path = fake_bin / program
            path.write_text(FAKE_TOOL)
            path.chmod(0o755)
        self.log = self.root / "commands.jsonl"
        self.env = dict(os.environ)
        self.env.pop("CARGO_NET_OFFLINE", None)
        self.env.pop("DOCKER_TEST_FAIL_FETCH", None)
        self.env.update(
            HOME=str(self.test_home),
            PATH=str(fake_bin) + os.pathsep + self.env["PATH"],
            LD_LIBRARY_PATH="/previous/library/path",
            DOCKER_TEST_LOG=str(self.log),
            CARGO_BUILD_JOBS="2",
        )

    def run_prepare(self):
        return subprocess.run(
            ["bash", str(self.repo / "docker/prepare_workspace.sh")],
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )

    def commands(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def assert_preparation_order(self):
        expected = [
            ["cargo", ["fetch", "--manifest-path", manifest, "--quiet"], None]
            for manifest in (
                "Cargo.toml",
                "xlsynth-g8r/fuzz/Cargo.toml",
                "xlsynth-vastly/fuzz/Cargo.toml",
            )
        ]
        expected.extend(
            [
                ["pre-commit", ["install"], None],
                ["pre-commit", ["install-hooks"], None],
                ["pre-commit", ["run", "--all-files"], "true"],
                [
                    "cargo",
                    [
                        "fuzz",
                        "build",
                        "--fuzz-dir",
                        "xlsynth-g8r/fuzz",
                        "--sanitizer",
                        "none",
                        "fuzz_gate_fn_roundtrip",
                    ],
                    "true",
                ],
                [
                    "cargo",
                    [
                        "build",
                        "--workspace",
                        "--all-targets",
                        "--locked",
                        "--features",
                        "with-bitwuzla-system",
                    ],
                    "true",
                ],
            ]
        )
        self.assertEqual(self.commands(), expected)

    def test_fresh_cache_prefetches_before_offline_hooks(self):
        result = self.run_prepare()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_preparation_order()

    def test_inherited_offline_environment_can_refresh(self):
        self.env["CARGO_NET_OFFLINE"] = "true"
        result = self.run_prepare()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assert_preparation_order()

    def test_fuzz_smoke_target_is_registered_without_required_features(self):
        result = self.run_prepare()
        self.assertEqual(result.returncode, 0, result.stderr)
        fuzz_args = next(
            args
            for program, args, _ in self.commands()
            if program == "cargo" and args[:2] == ["fuzz", "build"]
        )
        target = fuzz_args[-1]
        manifest = (REPO_ROOT / "xlsynth-g8r/fuzz/Cargo.toml").read_text()
        target_sections = [
            section
            for section in manifest.split("[[bin]]")[1:]
            if re.search(r'^name\s*=\s*"' + re.escape(target) + r'"\s*$', section, re.M)
        ]
        self.assertEqual(len(target_sections), 1)
        self.assertIsNone(
            re.search(r"^required-features\s*=", target_sections[0], re.M)
        )

    def test_fetch_failure_stops_before_hooks(self):
        self.env["DOCKER_TEST_FAIL_FETCH"] = "1"
        result = self.run_prepare()
        self.assertEqual(result.returncode, 42, result.stderr)
        self.assertEqual(
            self.commands(),
            [["cargo", ["fetch", "--manifest-path", "Cargo.toml", "--quiet"], None]],
        )

    def test_offline_checks_keep_the_prepared_solver_feature_and_resolution(self):
        prepared = self.run_prepare()
        self.assertEqual(prepared.returncode, 0, prepared.stderr)
        self.log.unlink()
        (self.test_home / ".cargo").mkdir()
        (self.test_home / ".cargo/env").touch()
        result = subprocess.run(
            ["bash", str(self.repo / "docker/check_offline.sh")],
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        features = ["--features", "with-bitwuzla-system"]
        self.assertEqual(
            self.commands(),
            [
                [
                    "cargo",
                    ["check", "--workspace", "--all-targets", "--frozen"] + features,
                    "true",
                ],
                [
                    "cargo",
                    ["nextest", "run", "--workspace", "--frozen"]
                    + features
                    + ["--no-fail-fast", "--test-threads", "2"],
                    "true",
                ],
                [
                    "cargo",
                    ["test", "--workspace", "--doc", "--frozen"] + features,
                    "true",
                ],
                ["pre-commit", ["run", "--all-files"], "true"],
            ],
        )

    def test_persisted_environment_quotes_paths_and_keeps_loader_path(self):
        result = self.run_prepare()
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.run_prepare()
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in (".bashrc", ".bash_profile", ".profile"):
            self.assertEqual(len((self.test_home / name).read_text().splitlines()), 1)
        (self.test_home / ".cargo").mkdir()
        (self.test_home / ".cargo/env").write_text(
            "export DOCKER_TEST_RUST_ENV_LOADED=1\n"
        )
        probe = subprocess.run(
            [
                "bash",
                "-c",
                'source "$1"; printf "%s\\n" "$XLSYNTH_TOOLS" "$XLS_DSO_PATH" '
                '"$DSLX_STDLIB_PATH" "$LD_LIBRARY_PATH" "$CARGO_NET_OFFLINE" '
                '"${DOCKER_TEST_RUST_ENV_LOADED:-missing}"',
                "probe",
                str(self.test_home / ".bash_profile"),
            ],
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(probe.returncode, 0, probe.stderr)
        tools, dso, stdlib, loader_path, offline, rust_env = probe.stdout.splitlines()
        self.assertEqual(Path(tools).parent, self.repo / "xlsynth_tools/v0.0.1")
        self.assertEqual(
            Path(dso), Path(tools) / ("libxls-" + Path(tools).name + ".so")
        )
        self.assertEqual(Path(stdlib), Path(tools) / "xls/dslx/stdlib")
        self.assertEqual(loader_path, tools + ":/previous/library/path")
        self.assertEqual(offline, "true")
        self.assertEqual(rust_env, "1")

    def test_install_tools_rejects_unknown_arguments_before_installing(self):
        result = subprocess.run(
            [
                "bash",
                str(REPO_ROOT / "docker/install_tools.sh"),
                "--unknown",
            ],
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse(self.log.exists())


if __name__ == "__main__":
    unittest.main()
