// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use tempfile::NamedTempFile;
use xlsynth_pir::IrBits;

use xlsynth_g8r::aig_sim::gate_sim::PreparedGateSim;
use xlsynth_g8r::aig_sim::sequential::SequentialState;
use xlsynth_g8r::liberty::parser::parse_liberty_files;
use xlsynth_g8r::liberty_model::Library;
use xlsynth_g8r::netlist::emit::emit_netlist_as_text;
use xlsynth_g8r::netlist::gatefn_from_netlist::project_labeled_sequential_netlist_aig;
use xlsynth_g8r::netlist::gv_eval::{
    GvEvalOptions, GvSequentialSignalToggleActivity, LabeledSequentialNetlistAig,
    SequentialAigSignal, load_labeled_sequential_netlist_aig_with_liberty,
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
        let activity = model
            .count_toggle_activity(&trace)
            .map_err(anyhow::Error::msg)?;
        let clock_pins = activity
            .instances
            .iter()
            .flat_map(|instance| {
                instance.pins.iter().filter_map(|pin| match pin.activity {
                    GvSequentialSignalToggleActivity::Clock { toggle_count } => Some((
                        (instance.instance_name.as_str(), pin.pin_name.as_str()),
                        toggle_count,
                    )),
                    GvSequentialSignalToggleActivity::Sampled { .. } => None,
                })
            })
            .collect::<BTreeMap<_, _>>();
        assert_eq!(
            clock_pins,
            BTreeMap::from([
                (("gate", "CLK"), 6),
                (("gate", "GCLK"), 2),
                (("bit0", "CLK"), 2)
            ])
        );
        let power = model
            .analyze_dynamic_power(&library, &trace, GvSequentialDynamicPowerOptions::default())
            .map_err(anyhow::Error::msg)?;
        let q = &power
            .instances
            .iter()
            .find(|instance| instance.instance_name == "bit0")
            .unwrap()
            .outputs[0];
        assert_eq!(
            (q.rise_count, q.fall_count),
            (usize::from(!inverted), usize::from(inverted))
        );
        assert_eq!(power.total_dynamic_energy, 61.0);
    }
    Ok(())
}

#[test]
fn icg_dynamic_power_counts_passed_pulses_and_blocked_clock_energy() -> anyhow::Result<()> {
    let library = parse_clock_gate_library(CLOCK_GATE_LIB)?;
    let path = NamedTempFile::new()?;
    std::fs::write(path.path(), CLOCK_GATE_WIRING)?;
    let model = load_labeled_sequential_netlist_aig_with_liberty(
        path.path(),
        &library,
        &GvEvalOptions::default(),
    )?;
    // D and the initial Q are zero, so only the enable and the two clocks
    // switch.
    let cycles = [false, true, false, true].map(|en| vec![IrBits::bool(en), IrBits::bool(false)]);
    let trace = model
        .simulate_bits(
            &cycles,
            SequentialState::all_zeros(&model.sequential_gate_fn),
        )
        .map_err(anyhow::Error::msg)?;
    let report = model
        .analyze_dynamic_power(&library, &trace, GvSequentialDynamicPowerOptions::default())
        .map_err(anyhow::Error::msg)?;

    assert_eq!(report.active_edge_transition_count, 4);
    assert_eq!(report.inactive_edge_transition_count, 4);
    assert_eq!(report.clock_transition_count, 8);
    let mut output_edges = BTreeMap::new();
    let mut pin_energy = BTreeMap::new();
    for instance in &report.instances {
        for output in &instance.outputs {
            output_edges.insert(
                (instance.instance_name.as_str(), output.pin_name.as_str()),
                (output.rise_count, output.fall_count),
            );
        }
        for pin in &instance.pin_internal_energy {
            pin_energy.insert(
                (instance.instance_name.as_str(), pin.pin_name.as_str()),
                pin.internal_energy,
            );
        }
    }
    assert_eq!(
        output_edges,
        BTreeMap::from([(("gate", "GCLK"), (2, 2)), (("bit0", "Q"), (0, 0))])
    );
    assert_eq!(
        pin_energy,
        BTreeMap::from([
            (("gate", "CLK"), 12.0),  // 4 blocked edges × 1 + 4 passed edges × 2
            (("gate", "ENA"), 21.0),  // 3 input transitions × 7
            (("gate", "GCLK"), 12.0), // 4 gated edges × 3
            (("bit0", "CLK"), 20.0),  // 4 gated edges × 5; stable Q consumes no output energy
        ])
    );
    // At 1 V each edge costs half its net capacitance in pJ.
    assert_eq!(report.clock_switching_energy, 8.0); // 8 root edges × 2 pF / 2
    assert_eq!(report.primary_input_switching_energy, 3.0); // 3 ENA edges × 2 pF / 2
    assert_eq!(report.cell_output_switching_energy, 8.0); // 4 GCLK edges × 4 pF / 2
    assert_eq!(report.cell_internal_energy, 65.0);
    assert_eq!(report.total_dynamic_energy, 84.0);
    Ok(())
}

#[test]
fn icg_latches_the_pre_edge_enable_across_packed_batches() -> anyhow::Result<()> {
    let netlist = r#"module bank(input clk, input en, output q);
      wire ready, gclk;
      DFF enable_ff (.CLK(clk), .D(en), .Q(ready));
      ICG gate (.CLK(clk), .ENA(ready), .SE(1'b0), .GCLK(gclk));
      DFF sink (.CLK(gclk), .D(1'b1), .Q(q));
    endmodule"#;
    let library = parse_clock_gate_library(
        &CLOCK_GATE_LIB
            .replace("!ENA", "!IQ")
            .replace(r#"when: "ENA""#, r#"when: "IQ""#),
    )?;
    let model = project_clock_gate_for_proof(netlist, &library)?;
    // Cycle 85's rise starts a new 256-phase SIMD chunk, and cycle 170's
    // fall starts the next. The final root edge sets ready for a future pulse.
    let passed = [85, 170, 171, 255];
    let cycles = (0..257)
        .map(|i| vec![IrBits::bool(passed.contains(&(i + 1)) || i == 256)])
        .collect::<Vec<_>>();
    let trace = model
        .simulate_bits(
            &cycles,
            SequentialState::all_zeros(&model.sequential_gate_fn),
        )
        .map_err(anyhow::Error::msg)?;
    let activity = model
        .count_toggle_activity(&trace)
        .map_err(anyhow::Error::msg)?;
    let gate = activity
        .instances
        .iter()
        .find(|i| i.instance_name == "gate")
        .unwrap();
    let clock = &gate
        .pins
        .iter()
        .find(|p| p.pin_name == "GCLK")
        .unwrap()
        .activity;
    assert_eq!(
        *clock,
        GvSequentialSignalToggleActivity::Clock { toggle_count: 8 }
    );
    let power = model
        .analyze_dynamic_power(&library, &trace, GvSequentialDynamicPowerOptions::default())
        .map_err(anyhow::Error::msg)?;
    let gate = power
        .instances
        .iter()
        .find(|i| i.instance_name == "gate")
        .unwrap();
    assert_eq!(
        (gate.outputs[0].rise_count, gate.outputs[0].fall_count),
        (4, 4)
    );
    let clock_energy = gate
        .pin_internal_energy
        .iter()
        .find(|p| p.pin_name == "CLK")
        .unwrap();
    // Both root edges see the pre-edge latch state. 506 blocked edges cost
    // 1 pJ each, and the eight edges of four full pulses cost 2 pJ each.
    assert_eq!(clock_energy.internal_energy, 522.0);
    Ok(())
}

#[test]
fn icgs_with_one_enable_keep_separate_loads_and_propagate_local_clock_slew() -> anyhow::Result<()> {
    let netlist = r#"module gated_bit(input gclk, input d, output q);
      DFF bit0 (.CLK(gclk), .D(d), .Q(q));
    endmodule
    module bank(input clk, input en, input d, output [2:0] q);
      wire g0, g1, ck0, ck1, d_alias, d_copy;
      assign ck0 = g0;
      assign ck1 = g1;
      assign d_alias = d;
      BUF copy (.A(d), .Y(d_copy));
      gated_bit lane (.gclk(ck0), .d(d_alias), .q(q[0]));
      DFF bit1 (.CLK(ck1), .D(d_copy), .Q(q[1]));
      DFF bit2 (.CLK(ck1), .D(d_copy), .Q(q[2]));
      ICG light (.CLK(clk), .ENA(en), .SE(1'b0), .GCLK(g0));
      ICG heavy (.CLK(clk), .ENA(en), .SE(1'b0), .GCLK(g1));
    endmodule"#;
    let library = parse_clock_gate_library(CLOCK_GATE_LIB)?;
    let file = NamedTempFile::new()?;
    std::fs::write(file.path(), netlist)?;
    let model = load_labeled_sequential_netlist_aig_with_liberty(
        file.path(),
        &library,
        &GvEvalOptions {
            module_name: Some("bank".into()),
            ..Default::default()
        },
    )?;
    let trace = model
        .simulate_bits(
            &[vec![IrBits::bool(true), IrBits::bool(true)]],
            SequentialState::all_zeros(&model.sequential_gate_fn),
        )
        .map_err(anyhow::Error::msg)?;
    let activity = model
        .count_toggle_activity(&trace)
        .map_err(anyhow::Error::msg)?;
    let child = activity
        .module_boundaries
        .iter()
        .find(|b| b.instance_path == "lane")
        .unwrap();
    let child_clock = child.ports.iter().find(|p| p.port_name == "gclk").unwrap();
    assert_eq!(
        child_clock.bits_lsb_to_msb[0].activity,
        GvSequentialSignalToggleActivity::Clock { toggle_count: 2 }
    );

    let options = GvSequentialDynamicPowerOptions {
        clock_transition: 1.0,
        primary_input_transition: 1.0,
        ..Default::default()
    };
    let power = model
        .analyze_dynamic_power(&library, &trace, options)
        .map_err(anyhow::Error::msg)?;
    let outputs = power
        .instances
        .iter()
        .filter_map(|i| i.outputs.first().map(|o| (i.instance_name.as_str(), o)))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(outputs["light"].rise_load, 4.0);
    assert_eq!(outputs["light"].switching_energy, 4.0);
    assert_eq!(outputs["heavy"].rise_load, 8.0);
    assert_eq!(outputs["heavy"].switching_energy, 8.0);
    // Root slew is 1 ns, both GCLK outputs have 8 ns slew, and that input
    // slew selects a 6 ns Q transition on every sink, including the first
    // declared cell in the child hierarchy.
    let q_bucket = power
        .slew_buckets
        .iter()
        .position(|b| b.lower <= 6.0 && 6.0 < b.upper)
        .unwrap();
    for name in ["lane/bit0", "bit1", "bit2"] {
        assert_eq!(outputs[name].slew_histogram.rise[q_bucket], 1.0, "{name}");
    }
    assert_eq!(power.diagnostics.unattributed_output_transition_count, 0);
    Ok(())
}

#[test]
fn icg_power_requires_compatible_roles_in_the_characterized_library() -> anyhow::Result<()> {
    let library = parse_clock_gate_library(CLOCK_GATE_LIB)?;
    let model = project_clock_gate_for_proof(CLOCK_GATE_WIRING, &library)?;
    let trace = model
        .simulate_bits(
            &[vec![IrBits::bool(true), IrBits::bool(false)]],
            SequentialState::all_zeros(&model.sequential_gate_fn),
        )
        .map_err(anyhow::Error::msg)?;
    // Both libraries certify a low-transparent gate. Swapping ENA and SE in
    // the characterization changes which physical pin must be tied low.
    let mut characterized = library;
    let gate = characterized
        .cells
        .iter_mut()
        .find(|c| c.name == "ICG")
        .unwrap()
        .clock_gate
        .as_mut()
        .unwrap();
    std::mem::swap(&mut gate.enable_pins, &mut gate.test_pins);
    let error = model
        .analyze_dynamic_power(
            &characterized,
            &trace,
            GvSequentialDynamicPowerOptions::default(),
        )
        .unwrap_err();
    assert_eq!(
        error,
        "ICG 'gate' has incompatible power-library pin role for 'SE'"
    );
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
