// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::Path;

use xlsynth_g8r::aig::SequentialGateFn;
use xlsynth_g8r::aig_sim::gate_sim::PreparedGateSim;
use xlsynth_g8r::liberty::parser::parse_liberty_files;
use xlsynth_g8r::liberty_model::Library;
use xlsynth_g8r::netlist::emit::emit_netlist_as_text;
use xlsynth_g8r::netlist::gatefn_from_netlist::project_labeled_sequential_netlist_aig;
use xlsynth_g8r::netlist::io::{read_gv_for_proof_from_path, read_gv_from_str};
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

/// Emits the physical netlist and projects the resulting GV through the public
/// reader.
fn read_emitted_design(
    path: &Path,
    gv: &str,
    library: &Library,
) -> anyhow::Result<SequentialGateFn> {
    let text = emit_netlist_as_text(&read_gv_from_str(gv)?)?;
    fs::write(path, text)?;
    let parsed = read_gv_for_proof_from_path(path)?;
    let model = project_labeled_sequential_netlist_aig(
        &parsed.modules[0],
        &parsed.nets,
        &parsed.interner,
        library,
        None,
    )
    .map_err(anyhow::Error::msg)?;
    Ok(model.sequential_gate_fn)
}

#[test]
fn icg_preserves_outputs_and_every_next_state_for_arbitrary_prior_state() -> anyhow::Result<()> {
    let temp = tempfile::tempdir()?;
    let path = temp.path().join("bank.gv");
    let library = parse_liberty_files(&[
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/clock_gate.lib")
    ])
    .map_err(anyhow::Error::msg)?;
    let reference = read_emitted_design(&path, FEEDBACK, &library)?;
    let gated = read_emitted_design(&path, GATED, &library)?;
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
        let faulty = read_emitted_design(&path, &fault, &library)?;
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
