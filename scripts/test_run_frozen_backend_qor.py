#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the direct-ABC experiment harness; no EDA tools required."""

import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_frozen_backend_qor as qor


class FrozenBackendTest(unittest.TestCase):
    def test_abc_path_quoting(self):
        self.assertEqual(qor.abc_quote(Path("input with spaces.aig")), '"input with spaces.aig"')
        for invalid in ['a"b', "a;b", "a\nb", "a\rb"]:
            with self.assertRaises(ValueError):
                qor.abc_quote(invalid)

    def test_abc_script_golden(self):
        args = SimpleNamespace(liberty=[Path("gates.lib"), Path("registers.lib")],
                               constraints=Path("timing.constr"), rounds=1, syn_command="&syn2")
        expected = Path(__file__).with_name("testdata") / "frozen_backend_flow.golden.txt"
        self.assertEqual(qor.abc_program(args, "input.aig", "output.gv", True), expected.read_text())

    def test_yosys_invocation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError), patch.object(qor.subprocess, "run") as run:
                qor.invoke(["/path/to/yosys", "-V"], Path(directory), "test", 1)
            run.assert_not_called()

    def test_timeout_output_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            failure = subprocess.TimeoutExpired(["abc"], 1, output=b"partial\n", stderr=b"diagnostic\n")
            with self.assertRaises(subprocess.TimeoutExpired), patch.object(qor.subprocess, "run", side_effect=failure):
                qor.invoke(["abc"], root, "abc", 1)
            self.assertEqual((root / "abc.log").read_bytes(), b"partial\ndiagnostic\n\nStage timed out.\n")

    def test_summary_uses_paired_successes_and_reports_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = [dict(corpus="sample", design="adder", status="ok", area=10,
                              sequential_area=2, combinational_area=8, delay_ps=100),
                         dict(corpus="sample", design="slow", status="ok", area=100,
                              sequential_area=20, combinational_area=80, delay_ps=1000)]
            candidate = [dict(reference[0], area=9, combinational_area=7, delay_ps=80),
                         dict(corpus="sample", design="slow", status="timeout")]
            qor.write_json(root / "reference.json", reference)
            qor.write_json(root / "candidate.json", candidate)
            with contextlib.redirect_stdout(io.StringIO()):
                qor.summarize(SimpleNamespace(reference=root / "reference.json",
                    candidate=[root / "candidate.json"], output=root / "summary.json"))
            result = json.loads((root / "summary.json").read_text())[-1]
            self.assertEqual(result["paired_designs"], ["adder"])
            self.assertEqual(result["candidate_status_counts"], {"ok": 1, "timeout": 1})
            self.assertAlmostEqual(result["area_geomean_delta_percent"], -10)
            self.assertAlmostEqual(result["delay_ps_geomean_delta_percent"], -20)
            self.assertEqual(result["reference_area_sum"], 10)
            self.assertEqual(result["candidate_register_area_sum"], 2)


if __name__ == "__main__":
    unittest.main()
