// SPDX-License-Identifier: Apache-2.0

//! Structural feedback is checked against the block bodies used for emission.

#[cfg(feature = "iverilog-tests")]
mod support;

use xlsynth_codegen::{BlockCodegenOptions, emit_system_verilog};
use xlsynth_pir::block_simplify::simplify_block_package;
use xlsynth_pir::ir_builder::{verify_block_combinational_cycles, verify_block_wiring_cycles};
use xlsynth_pir::ir_parser::Parser;

/// Builds a structural feedback loop through a three-bit child.
fn feedback(operation: &str) -> String {
    format!(
        r#"package feedback
block leaf(d: bits[3], q: bits[3]) {{
  d: bits[3] = input_port(name=d, id=1)
  one: bits[2] = literal(value=1, id=2)
  value: bits[3] = {operation}
  q: () = output_port(value, name=q, id=4)
}}
top block root(result: bits[3]) {{
  instantiation u(block=leaf, kind=block)
  q: bits[3] = instantiation_output(instantiation=u, port_name=q, id=5)
  drive: () = instantiation_input(q, instantiation=u, port_name=d, id=6)
  result: () = output_port(q, name=result, id=7)
}}"#
    )
}

#[test]
fn simplifying_a_fixed_shift_changes_the_emitted_circuit_and_preserves_hierarchy() {
    let source = feedback("shll(d, one, id=3)");
    let original = Parser::new(&source).parse_and_verify_package().unwrap();
    let options = BlockCodegenOptions::default();
    assert!(
        verify_block_combinational_cycles(original.get_top_block().unwrap(), &original).is_err()
    );
    assert!(emit_system_verilog(&original, &options).is_err());

    let simplified = simplify_block_package(&original).unwrap();
    assert_eq!(simplified.members.len(), original.members.len());
    let root = simplified.get_top_block().unwrap();
    let old = original.get_top_block().unwrap();
    assert_eq!(root.instantiations.len(), old.instantiations.len());
    for (new, old) in root.instantiations.iter().zip(&old.instantiations) {
        assert_eq!(
            (&new.name, &new.block, &new.kind),
            (&old.name, &old.block, &old.kind)
        );
    }
    assert!(verify_block_combinational_cycles(root, &simplified).is_err());
    verify_block_wiring_cycles(root, &simplified).unwrap();
    let sv = emit_system_verilog(&simplified, &options)
        .unwrap()
        .system_verilog;
    assert!(sv.contains("module leaf(") && sv.contains("leaf u ("));
    assert!(!sv.contains(" << "));

    // The unsimplified package is still available to the caller.
    assert!(
        original
            .get_block("leaf")
            .unwrap()
            .nodes
            .iter()
            .any(|n| matches!(
                n.payload,
                xlsynth_pir::ir::NodePayload::Binop(xlsynth_pir::ir::Binop::Shll, ..)
            ))
    );

    #[cfg(feature = "iverilog-tests")]
    {
        use std::collections::BTreeMap;
        use xlsynth_test_helpers::rtl_sim::LogicValue;
        let rtl = support::TestRtl::emit(&simplified, &options);
        assert_eq!(
            rtl.evaluate(&BTreeMap::new()).unwrap()["result"],
            LogicValue::from_u64(3, 0)
        );
    }
}

#[test]
fn a_true_wiring_cycle_remains_an_error_after_simplification() {
    let source = feedback("identity(d, id=3)");
    let package = Parser::new(&source).parse_and_verify_package().unwrap();
    let simplified = simplify_block_package(&package).unwrap();
    assert!(
        emit_system_verilog(&simplified, &Default::default())
            .unwrap_err()
            .to_string()
            .contains("combinational cycle")
    );
}

#[test]
fn a_selector_tied_off_in_the_parent_does_not_specialize_the_child() {
    let source = r#"package mux
block leaf(s: bits[1], a: bits[2], b: bits[2], q: bits[2]) {
  s: bits[1] = input_port(name=s, id=1)
  a: bits[2] = input_port(name=a, id=2)
  b: bits[2] = input_port(name=b, id=3)
  v: bits[2] = sel(s, cases=[a, b], id=4)
  q: () = output_port(v, name=q, id=5)
}
top block root(seed: bits[2], result: bits[2]) {
  instantiation u(block=leaf, kind=block)
  seed: bits[2] = input_port(name=seed, id=6)
  zero: bits[1] = literal(value=0, id=7)
  q: bits[2] = instantiation_output(instantiation=u, port_name=q, id=8)
  ds: () = instantiation_input(zero, instantiation=u, port_name=s, id=9)
  da: () = instantiation_input(seed, instantiation=u, port_name=a, id=10)
  db: () = instantiation_input(q, instantiation=u, port_name=b, id=11)
  result: () = output_port(q, name=result, id=12)
}"#;
    let package = Parser::new(source).parse_and_verify_package().unwrap();
    let simplified = simplify_block_package(&package).unwrap();
    assert_eq!(simplified.members.len(), package.members.len());
    assert!(
        simplified
            .get_block("leaf")
            .unwrap()
            .nodes
            .iter()
            .any(|n| matches!(n.payload, xlsynth_pir::ir::NodePayload::Sel { .. }))
    );
    assert!(emit_system_verilog(&simplified, &Default::default()).is_err());
}

/// The strict builder remains suitable for evaluation; structural construction
/// is explicit.
#[test]
fn builder_finalization_selects_the_consumer_cycle_model() {
    use xlsynth_pir::{BlockBuilder, ir::Type};
    let mut leaf = BlockBuilder::new("leaf");
    let d = leaf.input_port("d", Type::Bits(3)).unwrap();
    leaf.output_port("q", d).unwrap();
    let mut package = leaf.build_package("wiring").unwrap();
    let make_root = |package: &xlsynth_pir::ir::Package| {
        let mut parent = BlockBuilder::new("root");
        let seed = parent.input_port("seed", Type::Bits(1)).unwrap();
        let instance = parent
            .instantiate("u", package.get_block("leaf").unwrap())
            .unwrap();
        let q = parent.instantiation_output(instance, "q").unwrap();
        let lower = parent.bit_slice(q, 0, 2).unwrap();
        let d = parent.concat(&[lower, seed]).unwrap();
        parent.instantiation_input(instance, "d", d).unwrap();
        parent.output_port("result", q).unwrap();
        parent
    };
    assert!(make_root(&package).build_in_package(&package).is_err());
    make_root(&package)
        .build_structural_into_package(&mut package)
        .unwrap();
    package.set_top_block("root").unwrap();
    let sv = emit_system_verilog(&package, &Default::default())
        .unwrap()
        .system_verilog;
    assert!(sv.contains("leaf u ("));
}

/// Disabling bounds checks must validate the wiring actually emitted.
#[test]
fn unchecked_array_index_cannot_hide_a_real_feedback_cycle() {
    let source = r#"package p
block leaf(d: bits[2], q: bits[1]) {
  d: bits[2] = input_port(name=d, id=1)
  low: bits[1] = bit_slice(d, start=0, width=1, id=2)
  high: bits[1] = bit_slice(d, start=1, width=1, id=3)
  a: bits[1][2] = array(low, high, id=4)
  i: bits[2] = literal(value=2, id=5)
  selected: bits[1] = array_index(a, indices=[i], id=6)
  q: () = output_port(selected, name=q, id=7)
}
top block top(seed: bits[1], result: bits[1]) {
  instantiation u(block=leaf, kind=block)
  seed: bits[1] = input_port(name=seed, id=8)
  q: bits[1] = instantiation_output(instantiation=u, port_name=q, id=9)
  d: bits[2] = concat(seed, q, id=10)
  drive: () = instantiation_input(d, instantiation=u, port_name=d, id=11)
  result: () = output_port(q, name=result, id=12)
}
"#;
    let package = Parser::new(source).parse_and_verify_package().unwrap();
    emit_system_verilog(&package, &Default::default()).unwrap();
    let options = BlockCodegenOptions {
        array_index_bounds_checking: false,
        ..Default::default()
    };
    assert!(emit_system_verilog(&package, &options).is_err());
}

/// Cross-checks the original circuit, legalized circuit, and proof function.
#[cfg(feature = "iverilog-tests")]
#[test]
fn legalized_feedback_matches_hierarchical_rtl_and_function_for_all_inputs() {
    use std::collections::BTreeMap;
    use xlsynth_pir::block_legalize::legalize_block_hierarchy;
    use xlsynth_pir::block2fn::{Block2FnOptions, block_package_to_fn};
    use xlsynth_pir::ir::Type;
    use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
    use xlsynth_pir::{BlockBuilder, IrValue};
    use xlsynth_test_helpers::rtl_sim::LogicValue;

    let mut leaf = BlockBuilder::new("leaf");
    let data = leaf.input_port("data", Type::Bits(4)).unwrap();
    leaf.output_port("result", data).unwrap();
    let mut package = leaf.build_package("feedback_equivalence").unwrap();
    let mut root = BlockBuilder::new("root");
    let seed = root.input_port("seed", Type::Bits(2)).unwrap();
    let instance = root
        .instantiate("u", package.get_block("leaf").unwrap())
        .unwrap();
    let result = root.instantiation_output(instance, "result").unwrap();
    let lower = root.bit_slice(result, 0, 2).unwrap();
    let drive = root.concat(&[lower, seed]).unwrap();
    root.instantiation_input(instance, "data", drive).unwrap();
    root.output_port("result", result).unwrap();
    root.build_structural_into_package(&mut package).unwrap();
    package.set_top_block("root").unwrap();

    let legalized = legalize_block_hierarchy(&package, "root").unwrap();
    let function = block_package_to_fn(&legalized, &Block2FnOptions::default())
        .unwrap()
        .function;
    let options = BlockCodegenOptions::default();
    let hierarchical_rtl = support::TestRtl::emit(&package, &options);
    let flat_rtl = support::TestRtl::emit(&legalized, &options);
    for seed in 0..4 {
        let inputs = BTreeMap::from([("seed".to_string(), LogicValue::from_u64(2, seed))]);
        let hierarchical = hierarchical_rtl.evaluate(&inputs).unwrap();
        let flat = flat_rtl.evaluate(&inputs).unwrap();
        let value = match eval_fn(&function, &[IrValue::make_ubits(2, seed).unwrap()]) {
            FnEvalResult::Success(result) => result.value.to_bits().unwrap(),
            failure => panic!("function evaluation failed: {failure:?}"),
        };
        assert_eq!(hierarchical["result"], flat["result"], "seed={seed}");
        assert_eq!(
            hierarchical["result"].to_bits().unwrap(),
            value,
            "seed={seed}"
        );
    }
}
