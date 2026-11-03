// SPDX-License-Identifier: Apache-2.0

use std::error::Error;
use std::fs;
use std::process::Command;
use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig_serdes::g8r::decode_g8r_binary;

/// Lowers the given observation of a modular sum and reads its serialized
/// graph.
fn lower_wrap(observation: &str, flags: &[&str]) -> Result<GateFn, Box<dyn Error>> {
    let dir = tempfile::tempdir()?;
    let ir = dir.path().join("wrap.ir");
    let binary = dir.path().join("wrap.g8rbin");
    fs::write(
        &ir,
        format!(
            r#"package wrap
top fn f(x: bits[2] id=1, c: bits[1] id=2) -> bits[1] {{
  inc: bits[2] = zero_ext(c, new_bit_count=2, id=3)
  sum: bits[2] = add(x, inc, id=4)
  {observation}
}}"#
        ),
    )?;
    let result = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .arg("ir2g8r")
        .arg(ir)
        .arg("--bin-out")
        .arg(&binary)
        .args([
            "--fold=true",
            "--hash=true",
            "--enable-rewrite-nary-add=false",
            "--enable-formal-array-alias-analysis=false",
            "--fraig=false",
            "--reassociation=false",
            "--cut-db-rewrite=false",
        ])
        .args(flags)
        .output()?;
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    Ok(decode_g8r_binary(&fs::read(binary)?)?.try_into_gate_fn()?)
}

#[test]
fn auto_uses_the_endpoint_test_and_brent_kung_keeps_the_adder() -> Result<(), Box<dyn Error>> {
    let observation = r#"zero: bits[2] = literal(value=0, id=5)
  ret p: bits[1] = eq(sum, zero, id=6)"#;
    let default = lower_wrap(observation, &[])?;
    let auto = lower_wrap(observation, &["--adder-mapping=auto"])?;
    let brent_kung = lower_wrap(observation, &["--adder-mapping=brent-kung"])?;
    assert_eq!(get_aig_stats(&default).and_nodes, 5);
    assert_eq!(get_aig_stats(&auto).and_nodes, 5);
    assert_eq!(get_aig_stats(&brent_kung).and_nodes, 8);
    Ok(())
}

#[test]
fn replacement_output_keeps_the_reduction_and_returned_not_ids() -> Result<(), Box<dyn Error>> {
    let observation = r#"p: bits[1] = or_reduce(sum, id=5)
  ret result: bits[1] = not(p, id=6)"#;
    let stock = lower_wrap(
        observation,
        &["--track-pir-node-ids=true", "--adder-mapping=brent-kung"],
    )?;
    let graft = lower_wrap(observation, &["--track-pir-node-ids=true"])?;
    assert_eq!(get_aig_stats(&stock).and_nodes, 8);
    assert_eq!(get_aig_stats(&graft).and_nodes, 5);
    let stock_ids = stock
        .get(stock.outputs[0].bit_vector.get_lsb(0).node)
        .get_pir_node_ids();
    let graft_ids = graft
        .get(graft.outputs[0].bit_vector.get_lsb(0).node)
        .get_pir_node_ids();
    assert_eq!(stock_ids, &[5, 6]);
    assert_eq!(graft_ids, stock_ids);
    Ok(())
}
