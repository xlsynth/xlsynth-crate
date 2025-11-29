// SPDX-License-Identifier: Apache-2.0

//! Tests our gate mapped representation for an adder.

use test_case::test_case;

use xlsynth_g8r::aig_serdes::gate2ir;
use xlsynth_g8r::aig_serdes::ir2gate;
use xlsynth_g8r::check_equivalence;
use xlsynth_pir::ir_parser;

#[test_case(1)]
fn test_n_bit_adder(n: usize) {
    let _ = env_logger::try_init();
    let original_ir = format!(
        "package adder
top fn add_{n}_bits(a: bits[{n}] id=1, b: bits[{n}] id=2) -> bits[{n}] {{
    ret add.3: bits[{n}] = add(a, b, id=3)
}}"
    );
    let mut parser = ir_parser::Parser::new(&original_ir);
    let orig_package = parser.parse_and_validate_package().unwrap();
    let orig_package_ir_text = orig_package.to_string();
    let orig_ir_fn = orig_package.get_top_fn().unwrap();
    let gatify_output = ir2gate::gatify(
        &orig_ir_fn,
        ir2gate::GatifyOptions {
            fold: true,
            check_equivalence: true,
            hash: true,
            adder_mapping: xlsynth_g8r::ir2gate_utils::AdderMapping::default(),
        },
    )
    .unwrap();

    // Now we convert the gate_fn back to IR and check their equivalence.
    let gate_package =
        gate2ir::gate_fn_to_xlsynth_ir(&gatify_output.gate_fn, "adder", &orig_ir_fn.get_type())
            .unwrap();
    let gate_package_ir_text = gate_package.to_string();

    let result = check_equivalence::check_equivalence(&orig_package_ir_text, &gate_package_ir_text);
    assert!(result.is_ok(), "{}", result.unwrap_err());
}
