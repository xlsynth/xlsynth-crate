// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::IrValue;
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn_in_package};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir_compiler::PirFunctionCompiler;

/// Checks storage-view results against both explicit values and the
/// interpreter.
fn assert_result(ir: &str, expected: IrValue) {
    let package = Parser::new(ir).parse_and_validate_package().unwrap();
    let function = package.get_top_fn().unwrap();
    let FnEvalResult::Success(interpreted) = eval_fn_in_package(&package, function, &[]) else {
        panic!("storage-view example must evaluate successfully");
    };
    assert_eq!(interpreted.value, expected);
    let compiler = PirFunctionCompiler::compile_package(&package).unwrap();
    assert_eq!(compiler.run_ir_values(&[]).unwrap(), expected, "{ir}");
}

#[test]
fn zero_trip_loop_preserves_shared_scalar_wide_and_aggregate_storage() {
    for (ty, first, second) in [
        ("bits[8]", "17", "29"),
        ("bits[97]", "17", "29"),
        ("bits[8][1]", "[17]", "[29]"),
        ("(bits[8], bits[97])", "(17, 19)", "(29, 31)"),
    ] {
        let ir = format!(
            r#"package test
fn body(i: bits[1] id=1, carry: {ty} id=2) -> {ty} {{
  ret carry: {ty} = param(name=carry, id=2)
}}
top fn f() -> ({ty}, {ty}[2]) {{
  data: {ty}[2] = literal(value=[{first}, {second}], id=3)
  index: bits[1] = literal(value=0, id=4)
  item: {ty} = array_index(data, indices=[index], id=5)
  result: {ty} = counted_for(item, trip_count=0, stride=1, body=body, id=6)
  replacement: {ty} = literal(value={second}, id=7)
  updated: {ty}[2] = array_update(data, replacement, indices=[index], id=8)
  ret output: ({ty}, {ty}[2]) = tuple(result, updated, id=9)
}}
"#
        );
        let data = Parser::new(&ir).parse_and_validate_package().unwrap();
        let function = data.get_top_fn().unwrap();
        let values: Vec<_> = function
            .nodes
            .iter()
            .filter_map(|node| {
                if let xlsynth_pir::ir::NodePayload::Literal(value) = &node.payload {
                    Some(value.clone())
                } else {
                    None
                }
            })
            .collect();
        let elements = values[0].get_elements().unwrap();
        let expected = IrValue::make_tuple(&[
            elements[0].clone(),
            IrValue::make_array(&[elements[1].clone(), elements[1].clone()]).unwrap(),
        ]);
        assert_result(&ir, expected);
    }
}

#[test]
fn deferred_array_index_keeps_its_wide_index_storage_alive() {
    assert_result(
        r#"package test
top fn f() -> (bits[8], bits[97]) {
  data: bits[8][2] = literal(value=[17, 29], id=1)
  zero: bits[8] = literal(value=0, id=2)
  index: bits[97] = zero_ext(zero, new_bit_count=97, id=3)
  item: bits[8] = array_index(data, indices=[index], id=4)
  replacement: bits[97] = literal(value=1, id=5)
  ret result: (bits[8], bits[97]) = tuple(item, replacement, id=6)
}
"#,
        IrValue::parse_typed("(bits[8]:17, bits[97]:1)").unwrap(),
    );
}
