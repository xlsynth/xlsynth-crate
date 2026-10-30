// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use anyhow::{Result, bail};

use super::ResolvedStitchPipeline;
use crate::netlist::gv_eval::project_sequential_module;
use crate::netlist::io::{load_liberty_from_path, read_gv_for_proof_from_path, select_module};
use crate::process_ir_path::CanonicalG8rOptions;
use crate::prove_gate_fn_equiv_sat::GateFormalOptions;
#[cfg(feature = "has-bitwuzla")]
use crate::stitch_equiv::BehavioralPipeline;
use crate::stitch_equiv::{
    ProofReference, StitchEquivResult, StitchProofOptions, StitchProofReport, StitchProofSolver,
    prove_with_stage_options,
};

impl ResolvedStitchPipeline {
    /// Checks the free-running pipeline's outputs against a Liberty-backed
    /// netlist after fill.
    pub fn prove_equivalent_to_gv(
        &self,
        netlist_path: &Path,
        liberty_path: &Path,
        module_name: Option<&str>,
        options: GateFormalOptions,
    ) -> Result<StitchEquivResult> {
        let options = StitchProofOptions {
            solver: StitchProofSolver::Cadical(options),
            ..StitchProofOptions::default()
        };
        Ok(self
            .prove_equivalent_to_gv_with_options(netlist_path, liberty_path, module_name, &options)?
            .result)
    }

    /// Checks the mapped circuit using the selected stage scripts and worker
    /// budget, and reports how the external equivalence result was obtained.
    pub fn prove_equivalent_to_gv_with_options(
        &self,
        netlist_path: &Path,
        liberty_path: &Path,
        module_name: Option<&str>,
        options: &StitchProofOptions,
    ) -> Result<StitchProofReport> {
        let controls = [
            (
                "input_valid_signal",
                self.boundary.input_valid_signal.as_deref(),
            ),
            (
                "output_valid_signal",
                self.boundary.output_valid_signal.as_deref(),
            ),
            ("reset", self.boundary.reset_signal.as_deref()),
        ];
        let configured = controls
            .iter()
            .filter_map(|(field, value)| value.map(|name| format!("{field} = {name:?}")))
            .collect::<Vec<_>>();
        if !configured.is_empty() {
            bail!(
                "unsupported stitch controls for '{}': {}. stitch-equiv checks free-running pipelines and compares data on every cycle after fill. Valid-qualified observations and reset sequences are not yet supported",
                self.name(),
                configured.join(", ")
            );
        }
        // A proof must preserve the source ABI and literal values. The strict
        // GV reader rejects lossy constructs such as unsized signed literals
        // that would become zero-extended in the transition graph.
        let parsed = read_gv_for_proof_from_path(netlist_path)?;
        let module = select_module(&parsed, Some(module_name.unwrap_or(self.name())))?;
        // A zero-layer RTL wrapper still declares clk. Native combinational g8r
        // omits it. If present, treat it as a clock even if mapping removed all
        // FFs.
        let clock = self.boundary.clock.name.as_str();
        let clock_hint = module
            .ports
            .iter()
            .any(|port| parsed.interner.resolve(port.name) == Some(clock))
            .then_some(clock);
        let liberty = load_liberty_from_path(liberty_path)?;
        let mapped = project_sequential_module(&parsed, module, &liberty, clock_hint)?;
        if let Some(actual) = &mapped.clock
            && actual.port_name != clock
        {
            bail!(
                "expected stitch clock '{clock}', got '{}'",
                actual.port_name
            );
        }
        // Give gate! its DSLX value semantics. Keep optional mapping rewrites
        // out of the reference so a native implementation using them is
        // checked against a separately lowered graph.
        let lower_gates = || {
            self.lower_to_g8r_with_layout(&CanonicalG8rOptions {
                unsafe_gatify_gate_operation: true,
                enable_formal_array_alias_analysis: false,
                enable_rewrite_carry_out: false,
                enable_rewrite_prio_encode: false,
                enable_rewrite_nary_add: false,
                enable_rewrite_mask_low: false,
                enable_rewrite_normalize_left: false,
                fraig: false,
                reassociation: false,
                cut_db_rewrite: false,
                compute_graph_logical_effort: false,
                ..CanonicalG8rOptions::default()
            })
        };
        let stage_names = self
            .stages
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<Vec<_>>();
        let check = |layout, model| {
            prove_with_stage_options(
                layout,
                &mapped,
                self.latency(),
                &stage_names,
                options,
                model,
            )
            .map_err(anyhow::Error::msg)
        };
        match &options.solver {
            StitchProofSolver::Cadical(formal) => {
                let reference = lower_gates()?;
                check(&reference, ProofReference::Gates(formal))
            }
            #[cfg(feature = "has-bitwuzla")]
            StitchProofSolver::Bitwuzla { timeout } => {
                let stages = BehavioralPipeline::from_xls(
                    &self.ir,
                    &self.stages,
                    self.boundary.flop_inputs,
                    self.boundary.flop_outputs,
                )
                .map_err(anyhow::Error::msg)?;
                let layout = stages
                    .interface_layout(&self.boundary)
                    .map_err(anyhow::Error::msg)?;
                let lower_for_pairing = || {
                    lower_gates().map_err(|e| {
                        format!("could not gatify reference for register pairing: {e}")
                    })
                };
                check(
                    &layout,
                    ProofReference::Behavioral {
                        stages: &stages,
                        timeout: *timeout,
                        lower_for_pairing: &lower_for_pairing,
                    },
                )
            }
            #[cfg(not(feature = "has-bitwuzla"))]
            StitchProofSolver::Bitwuzla { .. } => bail!(
                "Bitwuzla behavioral stitch proofs require a build with with-bitwuzla-system or with-bitwuzla-built. Use --solver cadical for a gate-level proof"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::dslx_stitch_pipeline::StitchPipelineSpec;
    use crate::prove_gate_fn_equiv_sat::GateFormalOptions;
    use std::fs;

    #[test]
    fn proof_frontend_rejects_mismatched_modules_clocks_and_lossy_gv() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = dir.path().join("pipe.toml");
        fs::write(
            &manifest,
            r#"dslx_input_file = "pipe.x"
dslx_top = "pipe"
clock_name = "tick"
flop_inputs = false"#,
        )
        .unwrap();
        fs::write(
            dir.path().join("pipe.x"),
            "fn pipe_cycle0(x: bool) -> bool { x }",
        )
        .unwrap();
        let pipe = StitchPipelineSpec::load(&manifest)
            .unwrap()
            .resolve()
            .unwrap();
        let netlist = dir.path().join("pipe.gv");
        let liberty = dir.path().join("cells.textproto");
        fs::write(
            &liberty,
            r#"
format_magic: 5496997758228255311
cells: {
  name: "DFF"
  pins: { name_string_id: 1 direction: INPUT }
  pins: { name_string_id: 2 direction: INPUT is_clocking_pin: true }
  pins: { name_string_id: 3 direction: OUTPUT function_string_id: 4 }
  sequential: { state_var: "IQ" next_state: "D" clock_expr: "CLK" kind: SEQUENTIAL_KIND_FF }
}
cells: {
  name: "BUF"
  pins: { name_string_id: 5 direction: INPUT }
  pins: { name_string_id: 6 direction: OUTPUT function_string_id: 5 }
}
interned_strings: ["D", "CLK", "Q", "IQ", "A", "Y"]
"#,
        )
        .unwrap();
        for (source, expected) in [
            (
                "module other(input x, output out); assign out = x; endmodule",
                "module 'pipe' not found in netlist; available modules: [other]",
            ),
            (
                "module pipe(input x, output out); assign out = x; endmodule",
                "mapped module 'pipe' is missing the stitch clock input 'tick'",
            ),
            (
                "module pipe(input x, input clk, output out); DFF ff (.CLK(clk), .D(x), .Q(out)); endmodule",
                "expected stitch clock 'tick', got 'clk'",
            ),
            (
                "module pipe(input x, input clk, input tick, output out); DFF ff (.CLK(clk), .D(x), .Q(out)); endmodule",
                "clock-port hint 'tick' disagrees with inferred clock port 'clk'",
            ),
            (
                "module pipe(input x, input tick, output out); DFF ff (.D(x), .Q(out)); endmodule",
                "clock pin 'CLK' on instance 'ff' is unconnected",
            ),
            (
                "module pipe(input x, input tick, input clk, output out); wire mid; DFF ff0 (.CLK(tick), .D(x), .Q(mid)); DFF ff1 (.CLK(clk), .D(mid), .Q(out)); endmodule",
                "multiple clock nets detected: 'tick' vs 'clk'",
            ),
            (
                "module pipe(input x, input tick, output out); BUF g (.A(x), .Y(tick)); DFF ff (.CLK(tick), .D(x), .Q(out)); endmodule",
                "cell 'BUF' instance 'g' output pin 'Y' drives clock net 'tick'. Clock wiring must be driven only by top-level input 'tick'",
            ),
            (
                "module pipe(input x, input tick, output out); wire clk_wire; assign clk_wire = tick; BUF g (.A(x), .Y(clk_wire)); DFF ff (.CLK(clk_wire), .D(x), .Q(out)); endmodule",
                "cell 'BUF' instance 'g' output pin 'Y' drives clock net 'clk_wire'. Clock wiring must be driven only by top-level input 'tick'",
            ),
            (
                "module pipe(input x, input tick, output out); assign tick = x; assign out = x; endmodule",
                "top-level clock wiring 'tick' is also driven by a continuous assignment",
            ),
            (
                "module pipe(input x, input tick, output out); wire via; assign via = tick; DFF ff (.CLK(tick), .D(via), .Q(out)); endmodule",
                "top-level clock 'tick' feeds data logic. Sequential netlist projection supports clock wiring only to FF clock pins",
            ),
            (
                "module pipe(input x, output out); BLACK_BOX u_missing (.A(x), .Y(out)); endmodule",
                "cell 'BLACK_BOX' used by instance 'u_missing' was not found in Liberty data",
            ),
            (
                "module pipe(input x, input tick, output [63:0] out); assign out = 4294967295; endmodule",
                "GV reader: unsized decimal with its sign bit set cannot be preserved; specify an explicit literal width",
            ),
            (
                "module pipe(input x, input tick, output out); wire [1:0] w, b; assign w = {x,x}; assign b = w[0:1]; DFF ff (.CLK(tick), .D(b[0]), .Q(out)); endmodule",
                "GV reader: part-select 'w[0:1]' does not follow its declared packed range [1:0]",
            ),
            (
                "module pipe(input x, input tick, output out); wire [1:0] a, b; assign a = {x,x}; tran(a,b); DFF ff (.CLK(tick), .D(b[0]), .Q(out)); endmodule",
                "GV reader: plain tran terminal 'a' has width 2. Use one tran primitive per bit",
            ),
        ] {
            fs::write(&netlist, source).unwrap();
            let error = pipe
                .prove_equivalent_to_gv(&netlist, &liberty, None, GateFormalOptions::default())
                .unwrap_err()
                .to_string();
            // A GV syntax error continues with a file-specific location and
            // the source line. Import and ABI errors have no such suffix.
            assert_eq!(error.split(" @ ").next().unwrap(), expected);
        }
        fs::write(
            &netlist,
            "module pipe(input x, input tick, output out); DFF ff (.CLK(tick), .D(x), .Q(out)); endmodule",
        )
        .unwrap();
        let ff = fs::read_to_string(&liberty).unwrap();
        for (bad_ff, expected) in [
            (
                ff.replace(r#"clock_expr: "CLK""#, ""),
                "instance 'ff': cell 'DFF' identifies clock pin 'CLK' but its Liberty FF entry has no clock_expr. Specify the active clock edge to project this FF",
            ),
            (
                ff.replace(
                    r#"state_var: "IQ""#,
                    r#"state_var: "IQ" complementary_state_var: "IQ""#,
                ),
                "instance 'ff': cell 'DFF' FF state_var and complementary_state_var both name 'IQ'. Q and its complement require distinct names",
            ),
            (
                ff.replace(r#"state_var: "IQ""#, r#"state_var: "D""#)
                    .replace("function_string_id: 4", "function_string_id: 1"),
                "instance 'ff': cell 'DFF' uses 'D' as both an FF state variable and a physical input pin",
            ),
            (
                ff.replace(
                    r#"state_var: "IQ""#,
                    r#"state_var: "IQ" complementary_state_var: "D""#,
                ),
                "instance 'ff': cell 'DFF' uses 'D' as both an FF state variable and a physical input pin",
            ),
            (
                ff.replace(
                    r#"state_var: "IQ""#,
                    r#"state_var: "IQ" complementary_state_var: "CLK""#,
                ),
                "instance 'ff': cell 'DFF' uses 'CLK' as both an FF state variable and a physical input pin",
            ),
        ] {
            fs::write(&liberty, bad_ff).unwrap();
            let error = pipe
                .prove_equivalent_to_gv(&netlist, &liberty, None, GateFormalOptions::default())
                .unwrap_err();
            assert_eq!(error.to_string(), expected);
        }
        // With no output function, an output named for Q directly exposes
        // that state bit. It contributes no binding to the next-state formula.
        let implicit_q = ff
            .replace(r#"state_var: "IQ""#, r#"state_var: "Q""#)
            .replace("function_string_id: 4", "");
        fs::write(&liberty, implicit_q).unwrap();
        assert_eq!(
            pipe.prove_equivalent_to_gv(&netlist, &liberty, None, GateFormalOptions::default())
                .unwrap(),
            crate::stitch_equiv::StitchEquivResult::Proved,
        );
    }
    #[test]
    fn proof_rejects_competing_cell_or_pin_definitions() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = dir.path().join("pipe.toml");
        fs::write(
            &manifest,
            r#"dslx_input_file = "pipe.x"
dslx_top = "pipe"
flop_inputs = false
flop_outputs = false
"#,
        )
        .unwrap();
        fs::write(
            dir.path().join("pipe.x"),
            "fn pipe_cycle0(x: bool) -> bool { !x }",
        )
        .unwrap();
        let pipe = StitchPipelineSpec::load(&manifest)
            .unwrap()
            .resolve()
            .unwrap();
        let netlist = dir.path().join("pipe.gv");
        fs::write(
            &netlist,
            "module pipe(input x, output out); BUF g(.A(x), .Y(out)); endmodule",
        )
        .unwrap();
        let cell = r#"cells: { name: "BUF"
          pins: { name_string_id: 1 direction: INPUT }
          pins: { name_string_id: 2 direction: OUTPUT function_string_id: 1 }
        }"#;
        let alternate_pin = "pins: { name_string_id: 2 direction: OUTPUT function_string_id: 3 }";
        let liberty = dir.path().join("cells.textproto");
        for (definitions, expected) in [
            (
                format!(
                    "{cell}\n{}",
                    cell.replace("function_string_id: 1", "function_string_id: 3")
                ),
                "Liberty contains duplicate cell 'BUF'",
            ),
            (
                format!("{} {alternate_pin} }}", cell.trim_end_matches('}')),
                "Liberty cell 'BUF' contains duplicate pin 'Y'",
            ),
        ] {
            fs::write(&liberty, format!("format_magic: 5496997758228255311\ninterned_strings: [\"A\", \"Y\", \"!A\"]\n{definitions}")).unwrap();
            let error = pipe
                .prove_equivalent_to_gv(&netlist, &liberty, None, GateFormalOptions::default())
                .unwrap_err();
            assert_eq!(error.to_string(), expected);
        }
    }
}
