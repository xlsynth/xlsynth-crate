// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::ir::{NodePayload, Package, Type};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_value_utils::zero_ir_value_for_type;
use xlsynth_pir::stateful_fn_to_block::stateful_fn_to_block;

/// Exercise package parsing, conversion, and the emitted IR consumed by other
/// tools. Both PIR and upstream XLS must be able to read the result.
fn convert(source: &str) -> Result<Package, String> {
    let package = Parser::new(source)
        .parse_and_verify_package()
        .map_err(|e| e.to_string())?;
    let converted = stateful_fn_to_block(package, "step")?;
    let text = converted.to_string();
    xlsynth::IrPackage::parse_ir(&text, None)
        .and_then(|package| package.verify())
        .map_err(|e| format!("{e}\n{text}"))?;
    Parser::new(&text)
        .parse_and_verify_package()
        .map_err(|e| e.to_string())
}

#[test]
fn preserves_other_members_and_wires_outputs_and_next_state_in_order() {
    let source = r#"package transition
file_number 1 "step.x"
top fn invert(x: bits[1] id=10) -> bits[1] {
  ret not.11: bits[1] = not(x, id=11)
}
fn step(data: bits[3][2] id=1, __reg_arr: bits[3][2] id=2, __reg_valid: bits[1] id=3) -> ((bits[3][2], bits[1]), bits[1], bits[3][2], bits[1]) {
  out0: (bits[3][2], bits[1]) = tuple(__reg_arr, __reg_valid, id=4)
  flip: bits[1] = invoke(__reg_valid, to_apply=invert, id=5, pos=[(1,2,3)])
  ret tuple.6: ((bits[3][2], bits[1]), bits[1], bits[3][2], bits[1]) = tuple(out0, __reg_valid, data, flip, id=6)
}
block other(x: bits[1]) {
  x: bits[1] = input_port(name=x, id=100)
}
"#;
    let original = Parser::new(source).parse_and_verify_package().unwrap();
    let package = convert(source).unwrap();
    assert_eq!(package.name, original.name);
    assert_eq!(
        package.file_table.id_to_path,
        original.file_table.id_to_path
    );
    assert_eq!(
        package.get_fn("invert").unwrap().to_string(),
        original.get_fn("invert").unwrap().to_string()
    );
    assert_eq!(
        package.get_block("other").unwrap().to_string(),
        original.get_block("other").unwrap().to_string()
    );
    let block = package.get_top_block().unwrap();
    assert_eq!(block.name, "step");
    assert_eq!(block.clock_port_name(), Some("clk"));
    assert_eq!(
        block
            .input_ports()
            .map(|p| block.port_name(p))
            .collect::<Vec<_>>(),
        ["rst", "data"]
    );
    assert_eq!(
        block
            .output_ports()
            .map(|p| block.port_name(p))
            .collect::<Vec<_>>(),
        ["out0", "out1"]
    );
    let reset = block.reset.as_ref().unwrap();
    assert_eq!(reset.port, block.get_input_port("rst").unwrap());
    assert!(!reset.asynchronous && !reset.active_low);
    assert_eq!(block.port_type(reset.port), &Type::Bits(1));
    assert!(block.get_node(reset.port).text_id > 100);
    assert_eq!(
        block
            .registers
            .iter()
            .map(|reg| reg.name.as_str())
            .collect::<Vec<_>>(),
        ["arr", "valid"]
    );
    for reg in &block.registers {
        assert_eq!(reg.reset_value, Some(zero_ir_value_for_type(&reg.ty)));
    }
    let first = block.output_value(block.get_output_port("out0").unwrap());
    let NodePayload::Tuple(args) = &block.get_node(first).payload else {
        panic!("aggregate port was flattened")
    };
    assert!(
        matches!(&block.get_node(args[0]).payload, NodePayload::RegisterRead {register} if register == "arr")
    );
    assert!(
        matches!(&block.get_node(args[1]).payload, NodePayload::RegisterRead {register} if register == "valid")
    );
    assert_eq!(
        block.output_value(block.get_output_port("out1").unwrap()),
        args[1]
    );
    let writes: Vec<_> = block
        .nodes
        .iter()
        .filter_map(|node| match &node.payload {
            NodePayload::RegisterWrite {
                register,
                arg,
                load_enable,
                reset: operand,
            } => {
                assert_eq!(*operand, Some(reset.port));
                assert!(load_enable.is_none());
                Some((register.as_str(), *arg))
            }
            _ => None,
        })
        .collect();
    assert_eq!(writes.len(), 2);
    assert_eq!(writes[0], ("arr", block.get_input_port("data").unwrap()));
    assert_eq!(writes[1].0, "valid");
    let flip = block.get_node(writes[1].1);
    assert!(
        matches!(&flip.payload, NodePayload::Invoke {to_apply, operands} if to_apply == "invert" && operands == &[args[1]])
    );
    assert_eq!(
        flip.pos,
        original
            .get_fn("step")
            .unwrap()
            .nodes
            .iter()
            .find(|n| n.text_id == 5)
            .unwrap()
            .pos
    );
}

#[test]
fn splits_a_singleton_returned_parameter_even_without_registers() {
    let package = convert(
        r#"package p
fn step(pair: ((bits[1], bits[3])) id=1) -> ((bits[1], bits[3])) {
  ret pair: ((bits[1], bits[3])) = param(name=pair, id=1)
}"#,
    )
    .unwrap();
    let block = package.get_top_block().unwrap();
    assert!(block.registers.is_empty());
    let port = block.get_output_port("out0").unwrap();
    assert_eq!(
        block.port_type(port),
        &Type::Tuple(vec![Box::new(Type::Bits(1)), Box::new(Type::Bits(3))])
    );
    assert!(
        matches!(&block.get_node(block.output_value(port)).payload, NodePayload::TupleIndex {tuple, index: 0} if *tuple == block.get_input_port("pair").unwrap())
    );
}

#[test]
fn state_only_function_reads_both_old_values_for_swap() {
    let package = convert(
        r#"package p
fn step(__reg_a: bits[5] id=1, __reg_b: bits[5] id=2) -> (bits[5], bits[5]) {
  ret tuple.3: (bits[5], bits[5]) = tuple(__reg_b, __reg_a, id=3)
}"#,
    )
    .unwrap();
    let block = package.get_top_block().unwrap();
    assert_eq!(block.output_ports().count(), 0);
    let writes: Vec<_> = block
        .nodes
        .iter()
        .filter_map(|node| {
            if let NodePayload::RegisterWrite { register, arg, .. } = &node.payload {
                let NodePayload::RegisterRead { register: from } = &block.get_node(*arg).payload
                else {
                    panic!("swap must read old state")
                };
                Some((register.as_str(), from.as_str()))
            } else {
                None
            }
        })
        .collect();
    assert_eq!(writes, [("a", "b"), ("b", "a")]);
}

#[test]
fn empty_transition_still_has_clock_and_reset() {
    let package = convert("package p\nfn step() -> () { ret tuple.1: () = tuple(id=1) }").unwrap();
    let block = package.get_top_block().unwrap();
    assert_eq!(block.clock_port_name(), Some("clk"));
    assert_eq!(block.input_ports().count(), 1);
    assert_eq!(
        block.reset.as_ref().unwrap().port,
        block.get_input_port("rst").unwrap()
    );
    assert_eq!(block.output_ports().count(), 0);
    assert!(block.registers.is_empty());
}

#[test]
fn rejects_invalid_interfaces_and_dangling_callers() {
    for (function, reason) in [
        (
            "fn step(x: bits[1] id=1) -> bits[1] { ret x: bits[1] = param(name=x, id=1) }",
            "return a tuple",
        ),
        (
            "fn step(__reg_x: bits[1] id=1, y: bits[1] id=2) -> (bits[1], bits[1]) { ret tuple.3: (bits[1], bits[1]) = tuple(y, __reg_x, id=3) }",
            "contiguous suffix",
        ),
        (
            "fn step(__reg_x: bits[1] id=1) -> () { ret tuple.2: () = tuple(id=2) }",
            "0 elements for 1 registers",
        ),
        (
            "fn step(__reg_x: bits[1] id=1, __reg_y: bits[2] id=2) -> (bits[2], bits[1]) { ret tuple.3: (bits[2], bits[1]) = tuple(__reg_y, __reg_x, id=3) }",
            "next value has type bits[2]",
        ),
        (
            "fn step(__reg_: bits[1] id=1) -> (bits[1]) { ret tuple.2: (bits[1]) = tuple(__reg_, id=2) }",
            "invalid register name",
        ),
        (
            "fn step(rst: bits[1] id=1) -> () { ret tuple.2: () = tuple(id=2) }",
            "generated block port",
        ),
        (
            "fn step(out0: bits[1] id=1) -> (bits[1]) { ret tuple.2: (bits[1]) = tuple(out0, id=2) }",
            "generated block port",
        ),
        (
            "fn step() -> () { ret tuple.1: () = tuple(id=1) }\nfn caller() -> () { ret invoke.2: () = invoke(to_apply=step, id=2) }",
            "step",
        ),
    ] {
        let source = format!("package p\n{function}");
        let original = Parser::new(&source).parse_and_verify_package().unwrap();
        let error = stateful_fn_to_block(original, "step").unwrap_err();
        assert!(error.contains(reason), "{error}");
    }
}
