// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::IrValue;
use xlsynth_pir::block_eval::{BlockEvalError, BlockValueKind, RegisterInit};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir_compiler::CompilerError;
use xlsynth_pir_compiler::block::{CompiledBlock, CompiledBlockError};

// Writing q succeeds even on an edge where writes to r conflict. The runtime
// must discard both staged registers and the newly sampled outputs on failure.
const TWO_REGISTERS: &str = r#"package design
top block dut(clk: clock, data: bits[8], x: bits[1], y: bits[1], out: bits[8]) {
  reg q(bits[8])
  reg r(bits[8])
  data: bits[8] = input_port(name=data, id=1)
  x: bits[1] = input_port(name=x, id=2)
  y: bits[1] = input_port(name=y, id=3)
  old_q: bits[8] = register_read(register=q, id=4)
  old_r: bits[8] = register_read(register=r, id=9)
  out: () = output_port(old_q, name=out, id=5)
  wq: () = register_write(data, register=q, id=6)
  wx: () = register_write(data, register=r, load_enable=x, id=7)
  wy: () = register_write(old_q, register=r, load_enable=y, id=8)
}"#;

#[test]
fn port_bindings_check_names_owners_and_value_shapes() {
    let package = Parser::new(TWO_REGISTERS)
        .parse_and_validate_package()
        .unwrap();
    let code = CompiledBlock::compile(&package, "dut").unwrap();
    for (names, expected) in [
        (vec!["oops"], "unknown block input 'oops'"),
        (vec!["data", "data", "y"], "duplicate block input 'data'"),
        (vec!["data", "x"], "missing block input 'y'"),
    ] {
        assert!(
            matches!(code.bind_inputs(&names), Err(CompiledBlockError::Compiler(
            CompilerError::InvalidArgument(message))) if message == expected)
        );
    }
    assert!(
        matches!(code.bind_output("unknown"), Err(CompiledBlockError::Compiler(
        CompilerError::InvalidArgument(message))) if message == "unknown block output 'unknown'")
    );

    let other = CompiledBlock::compile(&package, "dut").unwrap();
    let other_binding = other.bind_inputs(&["y", "data", "x"]).unwrap();
    let other_port = other.bind_output("out").unwrap();
    let port = code.bind_output("out").unwrap();
    let binding = code.bind_inputs(&["y", "data", "x"]).unwrap();
    let mut instance = code.prepare_instance(RegisterInit::AllZeros).unwrap();
    let values = [
        IrValue::bool(false),
        IrValue::make_ubits(8, 42).unwrap(),
        IrValue::bool(true),
    ];
    let mut peer = code
        .prepare_instance(RegisterInit::Supplied(vec![values[1].clone(); 2]))
        .unwrap();
    assert!(instance.read_output(&port).is_err());
    assert!(instance.tick_bound(&other_binding, &values).is_err());
    assert!(matches!(
        instance.tick_bound(&binding, &values[..2]),
        Err(CompiledBlockError::Evaluation(BlockEvalError::ValueCount {
            kind: BlockValueKind::Input,
            expected: 3,
            actual: 2
        }))
    ));
    let wrong = [
        values[0].clone(),
        IrValue::make_tuple(&[values[1].clone()]),
        values[2].clone(),
    ];
    assert!(
        matches!(instance.tick_bound(&binding, &wrong), Err(CompiledBlockError::Evaluation(
        BlockEvalError::ValueType { name, .. })) if name == "data")
    );

    instance.tick_bound(&binding, &values).unwrap();
    peer.tick_bound(&binding, &values).unwrap();
    assert!(instance.read_output(&other_port).is_err());
    assert_eq!(
        instance.read_output(&port).unwrap(),
        IrValue::make_ubits(8, 0).unwrap()
    );
    assert_eq!(peer.read_output(&port).unwrap(), values[1]);
    assert_eq!(
        instance.snapshot().unwrap().values(),
        [values[1].clone(), values[1].clone()]
    );
}

#[test]
fn output_and_register_commit_is_atomic_and_restore_discards_old_outputs() {
    let package = Parser::new(TWO_REGISTERS)
        .parse_and_validate_package()
        .unwrap();
    let code = CompiledBlock::compile(&package, "dut").unwrap();
    let other = CompiledBlock::compile(&package, "dut").unwrap();
    let foreign = other.initial_state(RegisterInit::AllZeros).unwrap();
    let mut instance = code.prepare_instance(RegisterInit::AllZeros).unwrap();
    let zero = IrValue::make_ubits(8, 0).unwrap();
    let a = IrValue::make_ubits(8, 13).unwrap();
    let b = IrValue::make_ubits(8, 24).unwrap();
    instance
        .tick(&[a.clone(), IrValue::bool(true), IrValue::bool(false)])
        .unwrap();
    let saved = instance.snapshot().unwrap();
    assert_eq!(saved.values(), [a.clone(), a.clone()]);
    assert!(matches!(
        instance.tick(&[b.clone(), IrValue::bool(true), IrValue::bool(true)]),
        Err(CompiledBlockError::Evaluation(BlockEvalError::ConflictingWrites {
            register, first: 7, second: 8
        })) if register == "r"
    ));
    assert!(matches!(
        instance.restore(&foreign),
        Err(CompiledBlockError::Evaluation(BlockEvalError::ForeignState))
    ));
    assert_eq!(instance.output(0).unwrap(), zero);
    assert_eq!(instance.snapshot().unwrap().values(), saved.values());

    let valid = [b.clone(), IrValue::bool(false), IrValue::bool(true)];
    instance.tick(&valid).unwrap();
    let owned_output = instance.output(0).unwrap();
    assert_eq!(owned_output, a);
    assert_eq!(
        instance.snapshot().unwrap().values(),
        [b.clone(), a.clone()]
    );
    instance.restore(&saved).unwrap();
    assert!(instance.output(0).is_err());
    assert_eq!(owned_output, a);
    instance.tick(&valid).unwrap();
    let expected = code.step(&saved, &valid).unwrap();
    assert_eq!(instance.output(0).unwrap(), expected.outputs[0]);
    assert_eq!(
        instance.snapshot().unwrap().values(),
        expected.next_state.values()
    );
}
