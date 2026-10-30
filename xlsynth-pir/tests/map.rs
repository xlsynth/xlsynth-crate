// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::IrValue;
use xlsynth_pir::ir::Type;
use xlsynth_pir::ir_eval::{
    CoverCount, FnEvalResult, eval_fn_in_package, eval_fn_in_package_value,
};
use xlsynth_pir::ir_parser::Parser;

const MAP: &str = r#"package example

fn widen(x: bits[4] id=1) -> bits[8] {
  zero: bits[4] = literal(value=0, id=2)
  negative: bits[1] = slt(x, zero, id=3)
  observed: () = cover(negative, label="negative", id=4)
  ret wide: bits[8] = sign_ext(x, new_bit_count=8, id=5)
}

top fn apply(a: bits[4][3] id=6) -> bits[8][3] {
  ret mapped: bits[8][3] = map(a, to_apply=widen, id=7)
}
"#;

const EMPTY_MAP: &str = r#"package empty

fn widen(x: bits[4] id=1) -> bits[8] {
  ret wide: bits[8] = sign_ext(x, new_bit_count=8, id=2)
}

top fn apply() -> bits[8][0] {
  empty: bits[4][0] = array(id=3)
  ret mapped: bits[8][0] = map(empty, to_apply=widen, id=4)
}
"#;

#[test]
fn round_trips_and_evaluates_each_element_in_order() {
    let pkg = Parser::new(MAP).parse_and_validate_package().unwrap();
    let printed = pkg.to_string();
    let pkg = Parser::new(&printed).parse_and_validate_package().unwrap();
    assert_eq!(pkg.to_string(), printed);
    let f = pkg.get_top_fn().unwrap();
    let args = [IrValue::parse_typed("[bits[4]:8, bits[4]:7, bits[4]:8]").unwrap()];
    let expected = IrValue::parse_typed("[bits[8]:248, bits[8]:7, bits[8]:248]").unwrap();
    let FnEvalResult::Success(result) = eval_fn_in_package(&pkg, f, &args) else {
        panic!("map evaluation failed")
    };
    assert_eq!(result.value, expected);
    assert_eq!(
        result.cover_counts,
        [CoverCount {
            node_text_id: 4,
            label: "negative".into(),
            count: 2
        }]
    );
    assert_eq!(eval_fn_in_package_value(&pkg, f, &args), expected);
}

#[test]
fn empty_map_retains_the_callees_result_type() {
    let pkg = Parser::new(EMPTY_MAP).parse_and_validate_package().unwrap();
    let result = eval_fn_in_package_value(&pkg, pkg.get_top_fn().unwrap(), &[]);
    assert_eq!(
        result,
        IrValue::make_array_typed(Type::Bits(8), &[]).unwrap()
    );
}

#[test]
fn rejects_invalid_map_operands_callees_and_result_types() {
    for invalid in [
        MAP.replace("map(a,", "map(a, a,"),
        MAP.replace("a: bits[4][3]", "a: bits[4]"),
        MAP.replace("a: bits[4][3]", "a: bits[3][3]"),
        MAP.replace("bits[8][3]", "bits[8][2]"),
        MAP.replace("bits[8][3]", "bits[7][3]"),
        MAP.replace("to_apply=widen", "to_apply=missing"),
        MAP.replace("x: bits[4] id=1)", "x: bits[4] id=1, y: bits[4] id=8)"),
        EMPTY_MAP.replace("empty: bits[4][0]", "empty: bits[4][1]"),
        EMPTY_MAP.replace("empty: bits[4][0]", "empty: bits[4]"),
    ] {
        assert!(
            Parser::new(&invalid).parse_and_validate_package().is_err(),
            "{invalid}"
        );
    }
}
