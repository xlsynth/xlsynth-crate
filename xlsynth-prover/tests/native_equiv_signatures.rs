// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "has-bitwuzla")]

use test_case::test_case;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_prover::ir_equiv::{IrEquivRequest, IrModule, run_ir_equiv};
use xlsynth_prover::prover::translate::{get_fn_inputs, ir_to_smt_data};
use xlsynth_prover::prover::types::{EquivParallelism, EquivResult, ProverFn};
use xlsynth_prover::solver::bitwuzla::{Bitwuzla, BitwuzlaOptions};
use xlsynth_prover::solver::{Response, Solver};

const STRATEGIES: [EquivParallelism; 3] = [
    EquivParallelism::SingleThreaded,
    EquivParallelism::OutputBits,
    EquivParallelism::InputBitSplit,
];

#[test]
fn indexing_a_multi_index_result_selects_the_expected_bit() {
    let ir = r#"package p
top fn main(a: bits[1][2][2][2], i: bits[1], j: bits[1], k: bits[1]) -> bits[1] {
  row: bits[1][2] = array_index(a, indices=[i, j], id=5)
  ret out: bits[1] = array_index(row, indices=[k], id=6)
}"#;
    let package = Parser::new(ir).parse_package().unwrap();
    let mut solver = Bitwuzla::new(&BitwuzlaOptions::new()).unwrap();
    let function = ProverFn::new(package.get_top_fn().unwrap(), Some(&package));
    let inputs = get_fn_inputs(&mut solver, function, None).unwrap();
    let output = ir_to_smt_data(&mut solver, &inputs);

    // Element zero is at the LSB in each dimension, so the offset into
    // the packed eight-bit input is 4*i + 2*j + k.
    let offset = solver.concat_many(
        ["i", "j", "k"]
            .map(|name| &inputs.inputs[name].bitvec)
            .to_vec(),
    );
    let shifted = solver.xls_shrl(&inputs.inputs["a"].bitvec, &offset);
    let expected = solver.slice(&shifted, 0, 1);
    let mismatch = solver.ne(&output, &expected);
    solver.assert(&mismatch).unwrap();
    assert_eq!(solver.check().unwrap(), Response::Unsat);
}

#[test_case("identity", true; "identical_callees")]
#[test_case("not", false; "different_callees_with_the_same_name")]
fn output_partitions_resolve_callees_in_their_own_package(rhs_op: &str, equivalent: bool) {
    let package = |op: &str| {
        format!(
            r#"package p
fn helper(x: bits[2]) -> bits[2] {{
  ret value: bits[2] = {op}(x, id=2)
}}
top fn main(x: bits[2]) -> bits[2] {{
  ret result: bits[2] = invoke(x, to_apply=helper, id=4)
}}"#
        )
    };
    let lhs = package("identity");
    let rhs = package(rhs_op);
    let request = IrEquivRequest::new(IrModule::new(&lhs), IrModule::new(&rhs))
        .with_parallelism(EquivParallelism::OutputBits);
    let result = run_ir_equiv(&request).unwrap().result;
    assert!(
        matches!(
            (equivalent, &result),
            (true, EquivResult::Proved) | (false, EquivResult::Disproved { .. })
        ),
        "{result:?}"
    );
}

fn constant_function(params: &str, width: usize) -> String {
    format!(
        r#"package p
top fn main({params}) -> bits[{width}] {{
  ret result: bits[{width}] = literal(value=1, id=100)
}}
"#
    )
}

fn expect_error(request: IrEquivRequest<'_>, message: &str) {
    for strategy in STRATEGIES {
        let result = run_ir_equiv(&request.clone().with_parallelism(strategy));
        assert_eq!(result.unwrap_err(), message, "{strategy}");
    }
}

#[test_case("x: bits[8]", "x: bits[16]", false, "Input type mismatch for x vs x: bits[8] vs bits[16]"; "u8 versus u16")]
#[test_case("x: bits[8]", "x: bits[16]", true, "Free input widths differ: LHS has 8 bits, RHS has 16 bits"; "unequal flattened widths")]
#[test_case("x: bits[8]", "a: bits[4], b: bits[4]", false, "Free input counts differ: LHS has 1, RHS has 2"; "unequal arity")]
#[test_case("x: bits[8]", "a: (bits[4], bits[4])", false, "Input type mismatch for x vs a: bits[8] vs (bits[4], bits[4])"; "equal widths with different types")]
fn incompatible_input_signatures(lhs_params: &str, rhs_params: &str, flatten: bool, message: &str) {
    let lhs = constant_function(lhs_params, 8);
    let rhs = constant_function(rhs_params, 8);
    let request = IrEquivRequest::new(IrModule::new(&lhs), IrModule::new(&rhs));
    expect_error(request.with_flatten_aggregates(flatten), message);
}

#[test]
fn incompatible_result_widths() {
    let lhs = constant_function("x: bits[8]", 4);
    let rhs = constant_function("x: bits[8]", 8);
    expect_error(
        IrEquivRequest::new(IrModule::new(&lhs), IrModule::new(&rhs)),
        "Return widths differ: LHS has 4 bits, RHS has 8 bits",
    );
}

#[test_case(""; "missing token")]
#[test_case("x: bits[1], y: bits[1]"; "wrong token type")]
#[test_case("t: token"; "missing activation")]
#[test_case("t: token, a: bits[2]"; "wrong activation type")]
fn invalid_fixed_activation_prefix(params: &str) {
    let ir = constant_function(params, 1);
    expect_error(
        IrEquivRequest::new(
            IrModule::new(&ir),
            IrModule::new(&ir).with_fixed_implicit_activation(true),
        ),
        "Fixed implicit activation on RHS requires leading token and bits[1] parameters",
    );
}

#[test]
fn flattening_allows_different_parameter_counts() {
    let lhs = r#"package p
top fn main(x: bits[8]) -> bits[8] {
  ret result: bits[8] = identity(x, id=100)
}"#;
    let rhs = r#"package q
top fn main(lo: bits[4], hi: bits[4]) -> bits[8] {
  ret result: bits[8] = concat(hi, lo, id=100)
}"#;
    for strategy in STRATEGIES {
        let request = IrEquivRequest::new(IrModule::new(lhs), IrModule::new(rhs))
            .with_flatten_aggregates(true)
            .with_parallelism(strategy);
        assert!(
            matches!(run_ir_equiv(&request).unwrap().result, EquivResult::Proved),
            "{strategy}"
        );
    }
}

#[test]
fn fixed_activation_without_free_inputs_can_match_a_bare_result() {
    let lhs = r#"package p
top fn main(tok: token, active: bits[1]) -> (token, bits[1]) {
  ret result: (token, bits[1]) = tuple(tok, active, id=100)
}"#;
    let rhs = constant_function("", 1);
    for strategy in STRATEGIES {
        let request = IrEquivRequest::new(
            IrModule::new(lhs).with_fixed_implicit_activation(true),
            IrModule::new(&rhs),
        )
        .with_parallelism(strategy);
        assert!(
            matches!(run_ir_equiv(&request).unwrap().result, EquivResult::Proved),
            "{strategy}"
        );
    }
}

#[test_case("identity(x, id=100)", true; "both_bit_values_match")]
#[test_case("literal(value=0, id=100)", false; "counterexample_when_x_is_one")]
fn automatic_split_uses_a_free_input_after_fixed_activation(rhs_expr: &str, equivalent: bool) {
    let lhs = r#"package p
top fn main(tok: token, active: bits[1], x: bits[1]) -> (token, bits[1]) {
  ret result: (token, bits[1]) = tuple(tok, x, id=100)
}"#;
    let rhs = format!(
        r#"package q
top fn main(x: bits[1]) -> bits[1] {{
  ret result: bits[1] = {rhs_expr}
}}"#
    );
    let request = IrEquivRequest::new(
        IrModule::new(lhs).with_fixed_implicit_activation(true),
        IrModule::new(&rhs),
    )
    .with_parallelism(EquivParallelism::InputBitSplit);
    let result = run_ir_equiv(&request).unwrap().result;
    assert!(
        matches!(
            (equivalent, &result),
            (true, EquivResult::Proved) | (false, EquivResult::Disproved { .. })
        ),
        "{result:?}"
    );
}
