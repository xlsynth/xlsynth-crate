// SPDX-License-Identifier: Apache-2.0

//! Run with `cargo run -p xlsynth-pir-compiler --example clock_block`.

use xlsynth_pir::{IrValue, block_eval::RegisterInit, ir_parser::Parser};
use xlsynth_pir_compiler::block::CompiledBlock;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let package = Parser::new(
        r#"package demo
top block delay(clk: clock, x: bits[8], y: bits[8]) {
  reg q(bits[8])
  x: bits[8] = input_port(name=x, id=1)
  q: bits[8] = register_read(register=q, id=2)
  w: () = register_write(x, register=q, id=3)
  y: () = output_port(q, name=y, id=4)
}"#,
    )
    .parse_package()
    .expect("valid delay block");
    let block = CompiledBlock::compile(&package, "delay")?;
    let input = [IrValue::make_ubits(8, 17)?];

    let state = block.initial_state(RegisterInit::AllZeros)?;
    let cycle = block.step(&state, &input)?;
    assert_eq!(cycle.outputs, [IrValue::make_ubits(8, 0)?]);
    assert_eq!(cycle.next_state.values(), &input);

    let mut instance = block.prepare_instance(RegisterInit::AllZeros)?;
    instance.restore(&cycle.next_state)?;
    let y = block.bind_output("y")?;
    instance.tick(&[IrValue::make_ubits(8, 42)?])?;
    println!("sampled y = {}", instance.read_output(&y)?);
    Ok(())
}
