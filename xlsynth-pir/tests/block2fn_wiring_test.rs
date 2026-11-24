// SPDX-License-Identifier: Apache-2.0

//! Function conversion automatically legalizes acyclic structural feedback.

use xlsynth_pir::block_legalize::legalize_block_hierarchy;
use xlsynth_pir::block2fn::{
    Block2FnOptions, block_ir_to_fn, block_package_to_fn, block_package_to_fn_package,
};
use xlsynth_pir::ir::{Package, Type};
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::{BlockBuilder, FnBuilder, IrValue};

const FEEDBACK: &str = r#"package feedback
block leaf(d: bits[4], q: bits[4]) {
  d: bits[4] = input_port(name=d, id=1)
  q: () = output_port(d, name=q, id=2)
}
top block root(seed: bits[1], result: bits[4]) {
  instantiation u(block=leaf, kind=block)
  seed: bits[1] = input_port(name=seed, id=3)
  q: bits[4] = instantiation_output(instantiation=u, port_name=q, id=4)
  low: bits[3] = bit_slice(q, start=0, width=3, id=5)
  driver: bits[4] = concat(low, seed, id=6)
  drive: () = instantiation_input(driver, instantiation=u, port_name=d, id=7)
  result: () = output_port(q, name=result, id=8)
}
"#;

/// Parses a valid package containing implicit hierarchy edges.
fn parse(source: &str) -> Package {
    Parser::new(source).parse_and_verify_package().unwrap()
}

/// Evaluates one successfully projected function.
fn evaluate(package: &Package, args: &[IrValue], options: &Block2FnOptions) -> IrValue {
    let function = block_package_to_fn(package, options).unwrap().function;
    match eval_fn(&function, args) {
        FnEvalResult::Success(result) => result.value,
        FnEvalResult::Failure(error) => panic!("function evaluation failed: {error:?}"),
    }
}

#[test]
fn automatically_legalizes_feedback_without_modifying_the_source() {
    let package = parse(FEEDBACK);
    let before = package.to_string();
    let explicit = legalize_block_hierarchy(&package, "root").unwrap();
    for seed in 0..=1 {
        let arg = IrValue::make_ubits(1, seed).unwrap();
        let expected = IrValue::make_ubits(4, seed * 15).unwrap();
        assert_eq!(
            evaluate(&package, &[arg.clone()], &Default::default()),
            expected
        );
        assert_eq!(evaluate(&explicit, &[arg], &Default::default()), expected);
    }
    assert_eq!(package.to_string(), before);
}

#[test]
fn input_tie_offs_are_applied_after_automatic_legalization() {
    let package = parse(FEEDBACK);
    let mut options = Block2FnOptions::default();
    options.tie_input_ports.insert(
        "seed".to_string(),
        IrValue::make_ubits(1, 1).unwrap().to_bits().unwrap(),
    );
    assert_eq!(
        evaluate(&package, &[], &options),
        IrValue::make_ubits(4, 15).unwrap()
    );
}

#[test]
fn accepts_an_acyclic_hierarchy_through_the_ordinary_path() {
    let mut leaf = BlockBuilder::new("leaf");
    let d = leaf.input_port("d", Type::Bits(8)).unwrap();
    let value = leaf.not(d).unwrap();
    leaf.output_port("q", value).unwrap();
    let mut package = leaf.build_package("plain").unwrap();
    let mut root = BlockBuilder::new("root");
    let x = root.input_port("x", Type::Bits(8)).unwrap();
    let u = root
        .instantiate("u", package.get_block("leaf").unwrap())
        .unwrap();
    root.instantiation_input(u, "d", x).unwrap();
    let q = root.instantiation_output(u, "q").unwrap();
    root.output_port("result", q).unwrap();
    root.build_into_package(&mut package).unwrap();
    package.set_top_block("root").unwrap();
    assert_eq!(
        evaluate(
            &package,
            &[IrValue::make_ubits(8, 0xa5).unwrap()],
            &Default::default()
        ),
        IrValue::make_ubits(8, 0x5a).unwrap()
    );
}

#[test]
fn zero_width_alias_feedback_converts_to_the_unique_empty_value() {
    let mut leaf = BlockBuilder::new("leaf");
    let d = leaf.input_port("d", Type::Bits(0)).unwrap();
    leaf.output_port("q", d).unwrap();
    let mut package = leaf.build_package("empty_feedback").unwrap();
    let mut root = BlockBuilder::new("root");
    let u = root
        .instantiate("u", package.get_block("leaf").unwrap())
        .unwrap();
    let q = root.instantiation_output(u, "q").unwrap();
    root.instantiation_input(u, "d", q).unwrap();
    root.output_port("result", q).unwrap();
    root.build_structural_into_package(&mut package).unwrap();
    package.set_top_block("root").unwrap();
    assert_eq!(
        evaluate(&package, &[], &Default::default()),
        IrValue::make_ubits(0, 0).unwrap()
    );
}

#[test]
fn rejects_a_real_feedback_loop() {
    let loop_source = FEEDBACK.replace("instantiation_input(driver,", "instantiation_input(q,");
    let error = block_package_to_fn(&parse(&loop_source), &Default::default()).unwrap_err();
    assert!(
        error.to_lowercase().contains("cycle"),
        "unexpected error: {error}"
    );
}

#[test]
fn external_boxes_remain_explicitly_unsupported_for_function_conversion() {
    let mut foreign = FnBuilder::new("foreign");
    let arg = foreign.param("arg", Type::Bits(8)).unwrap();
    let mut package = foreign.build_package(arg, "foreign_box").unwrap();
    let mut root = BlockBuilder::new("root");
    let input = root.input_port("input", Type::Bits(8)).unwrap();
    let u = root
        .instantiate_extern("external", package.get_fn("foreign").unwrap())
        .unwrap();
    root.instantiation_input(u, "arg", input).unwrap();
    let q = root.instantiation_output(u, "return").unwrap();
    root.output_port("result", q).unwrap();
    root.build_into_package(&mut package).unwrap();
    package.set_top_block("root").unwrap();
    let error = block_package_to_fn(&package, &Default::default()).unwrap_err();
    assert!(error.contains("extern"), "unexpected error: {error}");
}

/// Tying a control may turn feedback across a child boundary into static
/// wiring.
#[test]
fn tied_input_controls_feedback_during_function_projection() {
    let source = r#"package tied
block shift(d: bits[3], amount: bits[3], q: bits[3]) {
  d: bits[3] = input_port(name=d, id=1)
  amount: bits[3] = input_port(name=amount, id=2)
  shifted: bits[3] = shll(d, amount, id=3)
  q: () = output_port(shifted, name=q, id=4)
}
top block wrapper(amount: bits[3], result: bits[3]) {
  instantiation u(block=shift, kind=block)
  amount: bits[3] = input_port(name=amount, id=5)
  q: bits[3] = instantiation_output(instantiation=u, port_name=q, id=6)
  data: () = instantiation_input(q, instantiation=u, port_name=d, id=7)
  control: () = instantiation_input(amount, instantiation=u, port_name=amount, id=8)
  result: () = output_port(q, name=result, id=9)
}"#;
    for amount in [None, Some(0), Some(1)] {
        let mut options = Block2FnOptions::default();
        if let Some(amount) = amount {
            options.tie_input_ports.insert(
                "amount".into(),
                IrValue::make_ubits(3, amount).unwrap().to_bits().unwrap(),
            );
        }
        let projected = block_ir_to_fn(source, &options);
        if amount == Some(1) {
            let function = projected.unwrap().function;
            assert!(function.params.is_empty());
            match eval_fn(&function, &[]) {
                FnEvalResult::Success(result) => {
                    assert_eq!(result.value, IrValue::make_ubits(3, 0).unwrap())
                }
                FnEvalResult::Failure(error) => panic!("function evaluation failed: {error:?}"),
            }
        } else {
            assert!(projected.unwrap_err().contains("cycle"));
        }
    }
}

/// Mixed aggregates do not hide a tied control from feedback simplification.
#[test]
fn mixed_aggregate_control_can_break_feedback() {
    let source = r#"package aggregate_tied
block gate(d: bits[1], spare: bits[1], control: bits[1], q: bits[1]) {
  d: bits[1] = input_port(name=d, id=1)
  spare: bits[1] = input_port(name=spare, id=2)
  control: bits[1] = input_port(name=control, id=3)
  pair: (bits[1], bits[1]) = tuple(spare, control, id=4)
  select_control: bits[1] = tuple_index(pair, index=1, id=5)
  choices: bits[1][2] = array(spare, select_control, id=6)
  offset: bits[2] = literal(value=3, id=7)
  clamped_control: bits[1] = array_index(choices, indices=[offset], assumed_in_bounds=false, id=8)
  value: bits[1] = and(d, clamped_control, id=9)
  q: () = output_port(value, name=q, id=10)
}
top block wrapper(spare: bits[1], control: bits[1], result: bits[1]) {
  instantiation u(block=gate, kind=block)
  spare: bits[1] = input_port(name=spare, id=11)
  control: bits[1] = input_port(name=control, id=12)
  q: bits[1] = instantiation_output(instantiation=u, port_name=q, id=13)
  data: () = instantiation_input(q, instantiation=u, port_name=d, id=14)
  spare_connection: () = instantiation_input(spare, instantiation=u, port_name=spare, id=15)
  control_connection: () = instantiation_input(control, instantiation=u, port_name=control, id=16)
  result: () = output_port(q, name=result, id=17)
}"#;
    let mut options = Block2FnOptions::default();
    options.tie_input_ports.insert(
        "control".into(),
        IrValue::make_ubits(1, 0).unwrap().to_bits().unwrap(),
    );
    let function = block_ir_to_fn(source, &options).unwrap().function;
    for spare in 0..=1 {
        match eval_fn(&function, &[IrValue::make_ubits(1, spare).unwrap()]) {
            FnEvalResult::Success(result) => assert_eq!(result.value, IrValue::bool(false)),
            FnEvalResult::Failure(error) => panic!("function evaluation failed: {error:?}"),
        }
    }
}

/// Register collapse can introduce a word-level cycle in otherwise acyclic
/// bit wiring, independently of hierarchy inlining.
#[test]
fn legalizes_feedback_introduced_by_register_collapse() {
    let source = r#"package collapsed_feedback
top block root(clk: clock, seed: bits[1], result: bits[4]) {
  reg saved(bits[4])
  seed: bits[1] = input_port(name=seed, id=1)
  q: bits[4] = register_read(register=saved, id=2)
  low: bits[3] = bit_slice(q, start=0, width=3, id=3)
  driver: bits[4] = concat(low, seed, id=4)
  write: () = register_write(driver, register=saved, id=5)
  result: () = output_port(q, name=result, id=6)
}"#;
    let package = parse(source);
    for seed in 0..=1 {
        assert_eq!(
            evaluate(
                &package,
                &[IrValue::make_ubits(1, seed).unwrap()],
                &Default::default(),
            ),
            IrValue::make_ubits(4, 15 * seed).unwrap(),
        );
    }
    let cyclic = parse(&source.replace("start=0", "start=1"));
    let error = block_package_to_fn(&cyclic, &Default::default()).unwrap_err();
    assert!(error.contains("cycle"), "unexpected error: {error}");
}

/// Tying a reset input before feedback legalization keeps the private block's
/// reset declaration, registers, and write operands mutually consistent.
#[test]
fn reset_ties_remain_valid_during_feedback_legalization() {
    let source = r#"package reset_tie
block leaf(d: bits[4], q: bits[4]) {
  d: bits[4] = input_port(name=d, id=1)
  q: () = output_port(d, name=q, id=2)
}
top block root(clk: clock, rst: bits[1], seed: bits[1], result: bits[4]) {
  #![reset(port="rst", asynchronous=false, active_low=false)]
  reg saved(bits[1], reset_value=0)
  instantiation u(block=leaf, kind=block)
  rst: bits[1] = input_port(name=rst, id=3)
  seed: bits[1] = input_port(name=seed, id=4)
  old: bits[1] = register_read(register=saved, id=5)
  write: () = register_write(seed, register=saved, reset=rst, id=6)
  q: bits[4] = instantiation_output(instantiation=u, port_name=q, id=7)
  low: bits[3] = bit_slice(q, start=0, width=3, id=8)
  driver: bits[4] = concat(low, seed, id=9)
  drive: () = instantiation_input(driver, instantiation=u, port_name=d, id=10)
  result: () = output_port(q, name=result, id=11)
}"#;
    let package = parse(source);
    let original = package.to_string();
    for reset in 0..=1 {
        let mut options = Block2FnOptions::default();
        options.tie_input_ports.insert(
            "rst".into(),
            IrValue::make_ubits(1, reset).unwrap().to_bits().unwrap(),
        );
        for seed in 0..=1 {
            assert_eq!(
                evaluate(&package, &[IrValue::make_ubits(1, seed).unwrap()], &options,),
                IrValue::make_ubits(4, 15 * seed).unwrap(),
            );
        }
    }
    assert_eq!(package.to_string(), original);
}

/// A native function's implementation becomes visible to tie-driven
/// simplification on a private feedback graph.
#[test]
fn tied_control_simplifies_feedback_through_native_invoke() {
    let source = r#"package tied_invoke
fn gate(d: bits[1] id=1, c: bits[1] id=2) -> bits[1] {
  ret result: bits[1] = and(d, c, id=3)
}
block leaf(d: bits[1], c: bits[1], q: bits[1]) {
  d: bits[1] = input_port(name=d, id=4)
  c: bits[1] = input_port(name=c, id=5)
  masked: bits[1] = invoke(d, c, to_apply=gate, id=6)
  q: () = output_port(masked, name=q, id=7)
}
top block root(control: bits[1], result: bits[1]) {
  instantiation u(block=leaf, kind=block)
  control: bits[1] = input_port(name=control, id=8)
  q: bits[1] = instantiation_output(instantiation=u, port_name=q, id=9)
  dd: () = instantiation_input(q, instantiation=u, port_name=d, id=10)
  cc: () = instantiation_input(control, instantiation=u, port_name=c, id=11)
  result: () = output_port(q, name=result, id=12)
}"#;
    let package = parse(source);
    let original = package.to_string();
    for control in 0..=1 {
        let mut options = Block2FnOptions::default();
        options.tie_input_ports.insert(
            "control".into(),
            IrValue::make_ubits(1, control).unwrap().to_bits().unwrap(),
        );
        if control == 0 {
            assert_eq!(evaluate(&package, &[], &options), IrValue::bool(false));
        } else {
            let error = block_package_to_fn(&package, &options).unwrap_err();
            assert!(error.contains("cycle"), "unexpected error: {error}");
        }
    }
    assert_eq!(package.to_string(), original);

    // Expansion can itself remove the apparent feedback when a native
    // function ignores the argument that participates in the loop.
    let independent = parse(&source.replace("and(d, c, id=3)", "identity(c, id=3)"));
    let mut options = Block2FnOptions::default();
    options.tie_input_ports.insert(
        "control".into(),
        IrValue::make_ubits(1, 1).unwrap().to_bits().unwrap(),
    );
    assert_eq!(evaluate(&independent, &[], &options), IrValue::bool(true));
}

/// An unmarked package uses its first block as the implicit top. Unrelated
/// definitions do not need to be flattened or cycle-checked for projection.
#[test]
fn unmarked_package_projects_only_its_implicit_top() {
    let source = r#"package implicit_top
block first(x: bits[1], result: bits[1]) {
  x: bits[1] = input_port(name=x, id=1)
  result: () = output_port(x, name=result, id=2)
}
block unrelated(result: bits[1]) {
  instantiation u(block=first, kind=block)
  q: bits[1] = instantiation_output(instantiation=u, port_name=result, id=3)
  feed: () = instantiation_input(q, instantiation=u, port_name=x, id=4)
  result: () = output_port(q, name=result, id=5)
}"#;
    let package = parse(source);
    assert_eq!(
        evaluate(
            &package,
            &[IrValue::bool(true)],
            &Block2FnOptions::default(),
        ),
        IrValue::bool(true),
    );
}

/// Const propagation must clear a canonical name when it changes the opcode.
#[test]
fn const_propagation_preserves_valid_programmatic_node_names() {
    let source = r#"package canonical_name
top block root(x: bits[1], result: bits[1]) {
  x: bits[1] = input_port(name=x, id=1)
  zero: bits[1] = literal(value=0, id=2)
  value: bits[1] = and(x, zero, id=3)
  result: () = output_port(value, name=result, id=4)
}"#;
    let mut package = parse(source);
    let node = package
        .get_top_block_mut()
        .unwrap()
        .nodes
        .iter_mut()
        .find(|node| node.text_id == 3)
        .unwrap();
    node.name = Some("and.3".into());
    xlsynth_pir::ir_verify::verify_package(&package).unwrap();

    let result = block_package_to_fn_package(&package, &Default::default()).unwrap();
    xlsynth_pir::ir_verify::verify_package(&result).unwrap();
    match eval_fn(result.get_top_fn().unwrap(), &[IrValue::bool(true)]) {
        FnEvalResult::Success(value) => assert_eq!(value.value, IrValue::bool(false)),
        FnEvalResult::Failure(error) => panic!("function evaluation failed: {error:?}"),
    }
}
