// SPDX-License-Identifier: Apache-2.0

use tempfile::NamedTempFile;
use xlsynth_pir::IrBits;

use xlsynth_g8r::aig_sim::gate_sim::PreparedGateSim;
use xlsynth_g8r::aig_sim::sequential::SequentialState;
use xlsynth_g8r::liberty::parser::parse_liberty_files;
use xlsynth_g8r::liberty_model::Library;
use xlsynth_g8r::netlist::emit::emit_netlist_as_text;
use xlsynth_g8r::netlist::gatefn_from_netlist::project_labeled_sequential_netlist_aig;
use xlsynth_g8r::netlist::gv_eval::{
    GvEvalOptions, LabeledSequentialNetlistAig, SequentialAigSignal,
    load_labeled_sequential_netlist_aig_with_liberty,
};
use xlsynth_g8r::netlist::io::{read_gv_for_proof_from_path, read_gv_from_str};
use xlsynth_g8r::netlist::power::GvSequentialDynamicPowerOptions;
use xlsynth_g8r::prove_gate_fn_equiv_sat::{
    EquivResult, GateFormalBackend, prove_gate_fn_equiv_with_backend,
};

const FEEDBACK: &str = r#"module bank(input clk, input en, input data, output [1:0] q);
  wire ready, load, d0, d1;
  DFF ready_ff (.CLK(clk), .D(en), .Q(ready));
  AND2 predicate (.A(en), .B(ready), .Y(load));
  MUX2 mux0 (.S(load), .A(q[0]), .B(data), .Y(d0));
  MUX2 mux1 (.S(load), .A(q[1]), .B(1'b0), .Y(d1));
  DFF bit0 (.CLK(clk), .D(d0), .Q(q[0]));
  DFF bit1 (.CLK(clk), .D(d1), .Q(q[1]));
endmodule"#;

const GATED: &str = r#"module bank(input clk, input en, input data, output [1:0] q);
  wire ready, load, gclk;
  DFF ready_ff (.CLK(clk), .D(en), .Q(ready));
  AND2 predicate (.A(en), .B(ready), .Y(load));
  ICG gate (.CLK(clk), .ENA(load), .SE(1'b0), .GCLK(gclk));
  DFF bit0 (.CLK(gclk), .D(data), .Q(q[0]));
  DFF bit1 (.CLK(gclk), .D(1'b0), .Q(q[1]));
endmodule"#;

#[test]
fn icg_preserves_outputs_and_every_next_state_for_arbitrary_prior_state() -> anyhow::Result<()> {
    let library = parse_clock_gate_library(CLOCK_GATE_LIB)?;
    let project_emitted = |gv: &str| -> anyhow::Result<_> {
        let emitted = emit_netlist_as_text(&read_gv_from_str(gv)?)?;
        Ok(project_clock_gate_for_proof(&emitted, &library)?.sequential_gate_fn)
    };
    let reference = project_emitted(FEEDBACK)?;
    let gated = project_emitted(GATED)?;
    assert_eq!(reference.registers.len(), 3);
    assert_eq!(reference.registers, gated.registers);
    assert_eq!(reference.clock, gated.clock);
    // The miter pairs inputs by position, so check the external and Q inputs
    // have the same identities before quantifying over all of them.
    assert_eq!(
        reference.transition.get_signature(),
        gated.transition.get_signature()
    );
    assert_eq!(reference.outputs, gated.outputs);
    assert_eq!(
        prove_gate_fn_equiv_with_backend(
            &reference.transition,
            &gated.transition,
            GateFormalBackend::Cadical
        )?,
        EquivResult::Proved
    );

    // Both faults must be observable in the next-state relation even though
    // the module output at the current edge still denotes the same prior Q.
    for (fault, bypassed_bit) in [
        (
            GATED
                .replace(
                    "ICG gate",
                    "wire hold; INV inv (.A(load), .Y(hold)); ICG gate",
                )
                .replace(".ENA(load)", ".ENA(hold)"),
            false,
        ),
        (GATED.replace("bit1 (.CLK(gclk)", "bit1 (.CLK(clk)"), true),
    ] {
        let faulty = project_emitted(&fault)?;
        assert_eq!(reference.registers, faulty.registers);
        let EquivResult::Disproved(witness) = prove_gate_fn_equiv_with_backend(
            &reference.transition,
            &faulty.transition,
            GateFormalBackend::Cadical,
        )?
        else {
            panic!("clock-gate wiring error should produce a counterexample");
        };
        let expected = PreparedGateSim::new(&reference.transition).eval_outputs(&witness);
        let observed = PreparedGateSim::new(&faulty.transition).eval_outputs(&witness);
        assert_ne!(expected, observed);
        assert_eq!(
            expected[reference.outputs[0].index()],
            observed[faulty.outputs[0].index()]
        );
        if bypassed_bit {
            // The constant-zero bit can only differ if it previously held a
            // one.
            let bit1 = reference
                .registers
                .iter()
                .find(|r| r.name == "bit1__IQ")
                .unwrap();
            assert!(witness[bit1.q.index()].get_bit(0)?);
            assert!(expected[bit1.d.index()].get_bit(0)?);
            assert!(!observed[bit1.d.index()].get_bit(0)?);
        }
    }
    Ok(())
}

/// The two assignments exercise clock wiring on both sides of the ICG.
const CLOCK_GATE_WIRING: &str = r#"module bank(input clk, input en, input d, output q);
  wire root_alias, gclk, gated_alias;
  assign root_alias = clk;
  assign gated_alias = gclk;
  ICG gate (.CLK(root_alias), .ENA(en), .SE(1'b0), .GCLK(gclk));
  DFF bit0 (.CLK(gated_alias), .D(d), .Q(q));
endmodule"#;

const CLOCK_GATE_LIB: &str = include_str!("data/clock_gate.lib");

/// Imports the bank's Liberty fixture, including variants with a different FF.
fn parse_clock_gate_library(source: &str) -> anyhow::Result<Library> {
    let file = NamedTempFile::new()?;
    std::fs::write(file.path(), source)?;
    parse_liberty_files(&[file.path()]).map_err(anyhow::Error::msg)
}

/// Projects the bank through the strict GV reader used by equivalence checks.
fn project_clock_gate_for_proof(
    gv: &str,
    library: &Library,
) -> anyhow::Result<LabeledSequentialNetlistAig> {
    let file = NamedTempFile::new()?;
    std::fs::write(file.path(), gv)?;
    let parsed = read_gv_for_proof_from_path(file.path())?;
    project_labeled_sequential_netlist_aig(
        &parsed.modules[0],
        &parsed.nets,
        &parsed.interner,
        library,
        None,
    )
    .map_err(anyhow::Error::msg)
}

#[test]
fn icg_projection_labels_clocks_and_holds_physical_state_for_q_or_qn() -> anyhow::Result<()> {
    let path = NamedTempFile::new()?;
    std::fs::write(path.path(), CLOCK_GATE_WIRING)?;
    for (output, inverted) in [("IQ", false), ("IQN", true)] {
        let library = parse_clock_gate_library(
            &CLOCK_GATE_LIB.replace(r#"function: "IQ""#, &format!(r#"function: "{output}""#)),
        )?;
        let model = load_labeled_sequential_netlist_aig_with_liberty(
            path.path(),
            &library,
            &GvEvalOptions::default(),
        )?;
        let pin_signal = |instance: &str, pin: &str| {
            model
                .instances
                .iter()
                .find(|i| i.instance_name == instance)
                .and_then(|i| i.pins.iter().find(|p| p.pin_name == pin))
                .map(|p| p.signal)
        };
        assert_eq!(pin_signal("gate", "CLK"), Some(SequentialAigSignal::Clock));
        let Some(SequentialAigSignal::Operand(enable)) = pin_signal("gate", "ENA") else {
            panic!("ICG enable must bind to a pre-edge operand");
        };
        let gated = Some(SequentialAigSignal::GatedClock(enable));
        assert_eq!(pin_signal("gate", "GCLK"), gated);
        assert_eq!(pin_signal("bit0", "CLK"), gated);
        assert_eq!(model.sequential_gate_fn.registers.len(), 1);
        let trace = model
            .simulate_bits(
                &[
                    vec![IrBits::bool(false), IrBits::bool(true)],
                    vec![IrBits::bool(true), IrBits::bool(true)],
                    vec![IrBits::bool(false), IrBits::bool(false)],
                ],
                SequentialState::all_zeros(&model.sequential_gate_fn),
            )
            .map_err(anyhow::Error::msg)?;
        assert_eq!(
            trace.external_outputs(),
            &[
                vec![IrBits::bool(inverted)],
                vec![IrBits::bool(inverted)],
                vec![IrBits::bool(!inverted)],
            ],
        );
        assert_eq!(trace.final_state().values(), [IrBits::bool(true)]);
        assert_eq!(
            model.count_toggle_activity(&trace).unwrap_err(),
            "sequential toggle reporting does not support gated clocks"
        );
        assert_eq!(
            model
                .analyze_dynamic_power(&library, &trace, GvSequentialDynamicPowerOptions::default())
                .unwrap_err(),
            "sequential dynamic-power analysis does not support gated clocks"
        );
    }
    Ok(())
}

#[test]
fn icg_projection_rejects_an_unresolved_driver_of_a_live_enable() -> anyhow::Result<()> {
    let library = parse_clock_gate_library(CLOCK_GATE_LIB)?;
    let dead_cycle = CLOCK_GATE_WIRING
        .replace(".ENA(en)", ".ENA(enable_alias)")
        .replace(
            "  ICG gate",
            r#"  wire capture, enable_alias, a, b;
  assign capture = en;
  tran (capture, enable_alias);
  assign a = b;
  assign b = a;
  ICG gate"#,
        );
    project_clock_gate_for_proof(&dead_cycle, &library)?;

    // The first driver supplies a Boolean value, but the second driver of
    // the same enable is cyclic and has no unique value to use in a proof.
    let live_cycle = dead_cycle.replace("assign b = a;", "assign b = a; assign capture = a;");
    let error = project_clock_gate_for_proof(&live_cycle, &library).unwrap_err();
    assert_eq!(
        error.to_string(),
        "Could not resolve sequential netlist combinational dependencies: 0 instances and 3 required continuous assign bits remain"
    );
    Ok(())
}

#[test]
fn icg_projection_rejects_unmodeled_test_and_clock_wiring() -> anyhow::Result<()> {
    let library = parse_clock_gate_library(CLOCK_GATE_LIB)?;
    let active_test = "clock-gate instance 'gate' test pin 'SE' must be tied inactive";
    let output_is_data = "clock-gate instance 'gate' output feeds data logic or a module output";
    let competing_driver = "clock-gate output 'gclk' has a competing driver";
    for (from, to, expected) in [
        (
            "output q);",
            "output q, input extra_clock); tran (clk, extra_clock);",
            "clock net has multiple top-level input drivers 'clk' and 'extra_clock'",
        ),
        (".SE(1'b0)", ".SE(en)", active_test),
        (".SE(1'b0)", ".SE(1'b1)", active_test),
        (
            ".ENA(en)",
            ".ENA()",
            "clock-gate instance 'gate' pin 'ENA' must connect exactly one known bit",
        ),
        (".D(d)", ".D(gated_alias)", output_is_data),
        (
            ".CLK(gated_alias), .D(d)",
            ".CLK(gclk), .D(gated_alias)",
            output_is_data,
        ),
        (
            "output q);\n  wire root_alias, gclk, gated_alias;",
            "output q, output gclk);\n  wire root_alias, gated_alias;",
            output_is_data,
        ),
        (
            "DFF bit0",
            "wire nested; ICG other (.CLK(gated_alias), .ENA(en), .SE(1'b0), .GCLK(nested)); DFF bit0",
            "derived clock 'gclk' is not a top-level input port",
        ),
        ("DFF bit0", "assign gclk = clk; DFF bit0", competing_driver),
        (
            "DFF bit0",
            "ICG other (.CLK(clk), .ENA(en), .SE(1'b0), .GCLK(gclk)); DFF bit0",
            competing_driver,
        ),
        (
            "DFF bit0",
            "AND2 other (.A(en), .B(d), .Y(gated_alias)); DFF bit0",
            "cell 'AND2' instance 'other' output pin 'Y' also drives gated clock net 'gated_alias'",
        ),
    ] {
        let gv = CLOCK_GATE_WIRING.replace(from, to);
        let error = project_clock_gate_for_proof(&gv, &library).unwrap_err();
        assert_eq!(error.to_string(), expected, "wiring change: {to}");
    }
    Ok(())
}

#[test]
fn icg_projection_rejects_falling_edge_capture() -> anyhow::Result<()> {
    let library = parse_clock_gate_library(
        &CLOCK_GATE_LIB.replace(r#"clocked_on: "CLK""#, r#"clocked_on: "!CLK""#),
    )?;
    let error = project_clock_gate_for_proof(CLOCK_GATE_WIRING, &library).unwrap_err();
    assert_eq!(
        error.to_string(),
        "clock-gate output drives falling-edge FF 'bit0'; rising-edge FF required"
    );
    Ok(())
}
