// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::desugar_extensions::{ExtensionEmitMode, emit_package_with_extension_mode};
use xlsynth_pir::ir_parser::{ParseOrValidateError, Parser};

fn assert_wrapped_text_round_trips_via_pir(
    ir_text: &str,
    caller_name: &str,
    helper_name: &str,
) -> xlsynth_pir::ir::Package {
    let pkg = {
        let mut parser = Parser::new(ir_text);
        parser
            .parse_and_validate_package()
            .expect("parse/validate verilogffi-form PIR")
    };
    assert_eq!(pkg.members.len(), 1, "expected helper to be lifted away");
    assert!(
        pkg.get_fn(helper_name).is_none(),
        "wrapped ffi helper should not remain as a PIR function after parse"
    );
    assert!(
        pkg.get_fn(caller_name).is_some(),
        "expected caller function '{}' to remain after parse",
        caller_name
    );
    let wrapped_text =
        emit_package_with_extension_mode(&pkg, ExtensionEmitMode::AsFfiFunction).unwrap();
    assert_eq!(wrapped_text, ir_text);
    pkg
}

fn assert_wrapped_text_parse_error_contains(ir_text: &str, expected_substr: &str) {
    let err = {
        let mut parser = Parser::new(ir_text);
        parser
            .parse_package()
            .expect_err("expected parse error for malformed wrapped PIR")
    };
    let msg = format!("{}", err);
    assert!(
        msg.contains(expected_substr),
        "unexpected parse error: {}",
        msg
    );

    let err = {
        let mut parser = Parser::new(ir_text);
        parser
            .parse_and_validate_package()
            .expect_err("expected parse error for malformed wrapped PIR")
    };
    match err {
        ParseOrValidateError::Parse(parse_err) => {
            let msg = format!("{}", parse_err);
            assert!(
                msg.contains(expected_substr),
                "unexpected parse error: {}",
                msg
            );
        }
        other => panic!("expected parse error, got {:?}", other),
    }
}

#[test]
fn verilogffi_wrapped_ext_carry_out_round_trips_verbatim() {
    let ir_text = r#"package carry_out_wrapped

#[ffi_proto("""code_template: "pir_ext_carry_out {fn} (.lhs({lhs}), .rhs({rhs}), .c_in({c_in}), .out({return})); /* xlsynth_pir_ext=ext_carry_out;width=8 */"
""")]
fn __pir_ext__ext_carry_out__w8(lhs: bits[8] id=5, rhs: bits[8] id=6, c_in: bits[1] id=7) -> bits[1] {
  zero_ext.8: bits[9] = zero_ext(lhs, new_bit_count=9, id=8)
  zero_ext.9: bits[9] = zero_ext(rhs, new_bit_count=9, id=9)
  add.10: bits[9] = add(zero_ext.8, zero_ext.9, id=10)
  zero_ext.11: bits[9] = zero_ext(c_in, new_bit_count=9, id=11)
  add.12: bits[9] = add(add.10, zero_ext.11, id=12)
  bit_slice.13: bits[1] = bit_slice(add.12, start=8, width=1, id=13)
  ret identity.14: bits[1] = identity(bit_slice.13, id=14)
}

fn f(lhs: bits[8] id=1, rhs: bits[8] id=2, c_in: bits[1] id=3) -> bits[1] {
  ret r: bits[1] = invoke(lhs, rhs, c_in, to_apply=__pir_ext__ext_carry_out__w8, id=4)
}
"#;

    let _pkg =
        assert_wrapped_text_round_trips_via_pir(ir_text, "f", "__pir_ext__ext_carry_out__w8");
}

#[test]
fn verilogffi_wrapped_ext_prio_encode_msb_round_trips_verbatim() {
    let ir_text = r#"package prio_encode_msb_wrapped

#[ffi_proto("""code_template: "pir_ext_prio_encode {fn} (.arg({arg}), .out({return})); /* xlsynth_pir_ext=ext_prio_encode;width=4;lsb_prio=false */"
""")]
fn __pir_ext__ext_prio_encode__w4__lsb0(arg: bits[4] id=3) -> bits[3] {
  one_hot.4: bits[5] = one_hot(arg, lsb_prio=false, id=4)
  encode.5: bits[3] = encode(one_hot.4, id=5)
  ret identity.6: bits[3] = identity(encode.5, id=6)
}

fn f(arg: bits[4] id=1) -> bits[3] {
  ret r: bits[3] = invoke(arg, to_apply=__pir_ext__ext_prio_encode__w4__lsb0, id=2)
}
"#;

    let _pkg = assert_wrapped_text_round_trips_via_pir(
        ir_text,
        "f",
        "__pir_ext__ext_prio_encode__w4__lsb0",
    );
}

#[test]
fn verilogffi_wrapped_ext_prio_encode_lsb_round_trips_verbatim() {
    let ir_text = r#"package prio_encode_lsb_wrapped

#[ffi_proto("""code_template: "pir_ext_prio_encode {fn} (.arg({arg}), .out({return})); /* xlsynth_pir_ext=ext_prio_encode;width=4;lsb_prio=true */"
""")]
fn __pir_ext__ext_prio_encode__w4__lsb1(arg: bits[4] id=3) -> bits[3] {
  one_hot.4: bits[5] = one_hot(arg, lsb_prio=true, id=4)
  encode.5: bits[3] = encode(one_hot.4, id=5)
  ret identity.6: bits[3] = identity(encode.5, id=6)
}

fn f(arg: bits[4] id=1) -> bits[3] {
  ret r: bits[3] = invoke(arg, to_apply=__pir_ext__ext_prio_encode__w4__lsb1, id=2)
}
"#;

    let _pkg = assert_wrapped_text_round_trips_via_pir(
        ir_text,
        "f",
        "__pir_ext__ext_prio_encode__w4__lsb1",
    );
}

#[test]
fn verilogffi_wrapped_ext_nary_add_round_trips_verbatim() {
    let ir_text = r#"package nary_add_wrapped

#[ffi_proto("""code_template: "pir_ext_nary_add {fn} (.op0({op0}), .op1({op1}), .op2({op2}), .out({return})); /* xlsynth_pir_ext=ext_nary_add;out_width=6;operand_widths=3,5,9 */"
""")]
fn __pir_ext__ext_nary_add__outw6__ops3_5_9(op0: bits[3] id=5, op1: bits[5] id=6, op2: bits[9] id=7) -> bits[6] {
  zero_ext.8: bits[6] = zero_ext(op0, new_bit_count=6, id=8)
  zero_ext.9: bits[6] = zero_ext(op1, new_bit_count=6, id=9)
  bit_slice.10: bits[6] = bit_slice(op2, start=0, width=6, id=10)
  add.11: bits[6] = add(zero_ext.8, zero_ext.9, id=11)
  add.12: bits[6] = add(add.11, bit_slice.10, id=12)
  ret identity.13: bits[6] = identity(add.12, id=13)
}

fn f(a: bits[3] id=1, b: bits[5] id=2, c: bits[9] id=3) -> bits[6] {
  ret r: bits[6] = invoke(a, b, c, to_apply=__pir_ext__ext_nary_add__outw6__ops3_5_9, id=4)
}
"#;

    let _pkg = assert_wrapped_text_round_trips_via_pir(
        ir_text,
        "f",
        "__pir_ext__ext_nary_add__outw6__ops3_5_9",
    );
}

#[test]
fn verilogffi_wrapped_ext_carry_out_operand_width_mismatch_is_rejected() {
    let ir_text = r#"package carry_out_wrapped_bad_invoke

#[ffi_proto("""code_template: "pir_ext_carry_out {fn} (.lhs({lhs}), .rhs({rhs}), .c_in({c_in}), .out({return})); /* xlsynth_pir_ext=ext_carry_out;width=8 */"
""")]
fn __pir_ext__ext_carry_out__w8(lhs: bits[8] id=5, rhs: bits[8] id=6, c_in: bits[1] id=7) -> bits[1] {
  literal.8: bits[1] = literal(value=0, id=8)
  ret identity.9: bits[1] = identity(literal.8, id=9)
}

fn f(lhs: bits[7] id=1, rhs: bits[8] id=2, c_in: bits[1] id=3) -> bits[1] {
  ret r: bits[1] = invoke(lhs, rhs, c_in, to_apply=__pir_ext__ext_carry_out__w8, id=4)
}
"#;

    assert_wrapped_text_parse_error_contains(
        ir_text,
        "invoke of wrapped ext_carry_out helper '__pir_ext__ext_carry_out__w8' operand 0 had type bits[7], expected bits[8]",
    );
}

#[test]
fn verilogffi_wrapped_ext_prio_encode_operand_width_mismatch_is_rejected() {
    let ir_text = r#"package prio_encode_wrapped_bad_invoke

#[ffi_proto("""code_template: "pir_ext_prio_encode {fn} (.arg({arg}), .out({return})); /* xlsynth_pir_ext=ext_prio_encode;width=4;lsb_prio=false */"
""")]
fn __pir_ext__ext_prio_encode__w4__lsb0(arg: bits[4] id=3) -> bits[3] {
  literal.4: bits[3] = literal(value=0, id=4)
  ret identity.5: bits[3] = identity(literal.4, id=5)
}

fn f(arg: bits[5] id=1) -> bits[3] {
  ret r: bits[3] = invoke(arg, to_apply=__pir_ext__ext_prio_encode__w4__lsb0, id=2)
}
"#;

    assert_wrapped_text_parse_error_contains(
        ir_text,
        "invoke of wrapped ext_prio_encode helper '__pir_ext__ext_prio_encode__w4__lsb0' operand 0 had type bits[5], expected bits[4]",
    );
}

#[test]
fn verilogffi_wrapped_ext_nary_add_operand_width_mismatch_is_rejected() {
    let ir_text = r#"package nary_add_wrapped_bad_invoke

#[ffi_proto("""code_template: "pir_ext_nary_add {fn} (.op0({op0}), .op1({op1}), .op2({op2}), .out({return})); /* xlsynth_pir_ext=ext_nary_add;out_width=6;operand_widths=3,5,9 */"
""")]
fn __pir_ext__ext_nary_add__outw6__ops3_5_9(op0: bits[3] id=5, op1: bits[5] id=6, op2: bits[9] id=7) -> bits[6] {
  literal.8: bits[6] = literal(value=0, id=8)
  ret identity.9: bits[6] = identity(literal.8, id=9)
}

fn f(a: bits[3] id=1, b: bits[4] id=2, c: bits[9] id=3) -> bits[6] {
  ret r: bits[6] = invoke(a, b, c, to_apply=__pir_ext__ext_nary_add__outw6__ops3_5_9, id=4)
}
"#;

    assert_wrapped_text_parse_error_contains(
        ir_text,
        "invoke of wrapped ext_nary_add helper '__pir_ext__ext_nary_add__outw6__ops3_5_9' operand 1 had type bits[4], expected bits[5]",
    );
}

#[test]
fn verilogffi_wrapped_helper_referenced_from_counted_for_is_rejected() {
    let ir_text = r#"package carry_out_wrapped_counted_for

#[ffi_proto("""code_template: "pir_ext_carry_out {fn} (.lhs({lhs}), .rhs({rhs}), .c_in({c_in}), .out({return})); /* xlsynth_pir_ext=ext_carry_out;width=8 */"
""")]
fn __pir_ext__ext_carry_out__w8(lhs: bits[8] id=5, rhs: bits[8] id=6, c_in: bits[1] id=7) -> bits[1] {
  literal.8: bits[1] = literal(value=0, id=8)
  ret identity.9: bits[1] = identity(literal.8, id=9)
}

fn f(init: bits[1] id=1) -> bits[1] {
  ret loop: bits[1] = counted_for(init, trip_count=2, stride=1, body=__pir_ext__ext_carry_out__w8, id=2)
}
"#;

    assert_wrapped_text_parse_error_contains(
        ir_text,
        "only invoke nodes may reference wrapped extension helpers",
    );
}
