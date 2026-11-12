// SPDX-License-Identifier: Apache-2.0

use std::error::Error;

use super::directed_expr_eval::{
    assert_expr_matches_dslx, assert_expr_rejected_by_both, compile_dslx, compile_dslx_module,
};
use super::expr_eval::{BlockEvaluator, ExpressionEvaluator, bx_source, dslx_source};
use xlsynth_bx::{
    ImportOptions, ResetBehavior, check_module, compile, empty_package, parse_module,
};
use xlsynth_codegen::{BlockCodegenOptions, emit_system_verilog};
use xlsynth_pir::IrValue;
use xlsynth_pir::ir::Type;
use xlsynth_vastly::{FixtureRunner, compile_pipeline_module};

#[test]
fn index_zero_is_the_most_significant_chunk_for_every_signedness() -> Result<(), Box<dyn Error>> {
    let packed = IrValue::make_ubits(8, 0x83)?;
    let chunks = IrValue::parse_typed("[bits[4]:8, bits[4]:3]")?;
    for scalar in ["u8", "s8"] {
        for array in ["u4[2]", "s4[2]"] {
            assert_expr_matches_dslx(
                &format!("x: {scalar}"),
                array,
                &format!("x as {array}"),
                [(vec![packed.clone()], chunks.clone())],
            );
            assert_expr_matches_dslx(
                &format!("xs: {array}"),
                scalar,
                &format!("xs as {scalar}"),
                [(vec![chunks.clone()], packed.clone())],
            );
        }
    }
    let body = "let chunks = x as s4[2]; (chunks[u32:0] as s8, (chunks as s8) < s8:0)";
    let evaluator = ExpressionEvaluator::compile("x: u8", "(s8, bool)", body, compile_dslx)?;
    let args = [packed];
    let expected = evaluator.compare(&args)?;
    assert_eq!(expected, IrValue::parse_typed("(bits[8]:-8, bits[1]:1)")?);
    evaluator.compare_reset(
        &args,
        "(s8, bool)",
        "let chunks = x as s4[2];",
        "(chunks[u32:0] as s8, (chunks as s8) < s8:0)",
        &expected,
    )?;
    Ok(())
}

#[test]
fn bool_singletons_and_wide_chunks_keep_their_bit_patterns() -> Result<(), Box<dyn Error>> {
    let flags = IrValue::parse_typed("[bits[1]:1, bits[1]:0, bits[1]:1, bits[1]:1]")?;
    let packed = IrValue::make_ubits(4, 0xb)?;
    assert_expr_matches_dslx(
        "x: u4",
        "bool[4]",
        "x as bool[4]",
        [(vec![packed.clone()], flags.clone())],
    );
    assert_expr_matches_dslx("xs: bool[4]", "u4", "xs as u4", [(vec![flags], packed)]);
    assert_expr_matches_dslx(
        "x: bool",
        "bool",
        "(x as bool[1]) as bool",
        [(vec![IrValue::bool(true)], IrValue::bool(true))],
    );

    let scalar = IrValue::parse_typed("bits[136]:0x80123456789abcdef1fedcba9876543210")?;
    let chunks =
        IrValue::parse_typed("[bits[68]:0x80123456789abcdef, bits[68]:0x1fedcba9876543210]")?;
    assert_expr_matches_dslx(
        "x: uN[136]",
        "sN[68][2]",
        "x as sN[68][2]",
        [(vec![scalar.clone()], chunks.clone())],
    );
    assert_expr_matches_dslx(
        "xs: sN[68][2]",
        "uN[136]",
        "xs as uN[136]",
        [(vec![chunks], scalar)],
    );
    Ok(())
}

#[test]
fn zero_total_width_preserves_array_shape() -> Result<(), Box<dyn Error>> {
    let zero = IrValue::make_ubits(0, 0)?;
    for (array_type, element_width) in [("u4[0]", 4), ("uN[0][0]", 0)] {
        let body = format!("let xs = x as {array_type}; (xs, xs as uN[0])");
        let evaluator = ExpressionEvaluator::compile(
            "x: uN[0]",
            &format!("({array_type}, uN[0])"),
            &body,
            compile_dslx,
        )?;
        // The pinned DSLX converter accepts this program. Its IR interpreter
        // cannot construct empty array values, so check the PIR result
        // directly.
        assert_eq!(
            evaluator.bx.evaluate(std::slice::from_ref(&zero))?,
            IrValue::make_tuple(&[
                IrValue::make_array_typed(Type::Bits(element_width), &[])?,
                zero.clone(),
            ])
        );
    }
    let chunks = IrValue::make_array(&[zero.clone(), zero.clone(), zero.clone()])?;
    assert_expr_matches_dslx(
        "x: uN[0]",
        "(uN[0][3], uN[0])",
        "let xs = x as uN[0][3]; (xs, xs as uN[0])",
        [(vec![zero.clone()], IrValue::make_tuple(&[chunks, zero]))],
    );
    Ok(())
}

#[test]
fn constants_containing_empty_cast_results_lower_to_valid_ir() -> Result<(), Box<dyn Error>> {
    let source = r#"
const EMPTY = bits[0]:0 as u4[0];
const NESTED = [(EMPTY, u3:1), (EMPTY, u3:2)];
#[allow(combinational_block)]
block sample(clk) -> (u4[0], (u4[0], u3)[2]) { (EMPTY, NESTED) }
"#;
    let checked = check_module(&parse_module(source)?)?;
    let evaluator = BlockEvaluator::lower(&checked, source)?;
    let empty = IrValue::make_array_typed(Type::Bits(4), &[])?;
    let nested = IrValue::make_array(&[
        IrValue::make_tuple(&[empty.clone(), IrValue::make_ubits(3, 1)?]),
        IrValue::make_tuple(&[empty.clone(), IrValue::make_ubits(3, 2)?]),
    ])?;
    assert_eq!(
        evaluator.evaluate(&[])?,
        IrValue::make_tuple(&[empty, nested])
    );
    Ok(())
}

#[test]
fn empty_array_cast_as_register_reset_lowers_to_valid_ir() -> Result<(), Box<dyn Error>> {
    let source = r#"
const E = bits[0]:0 as u8[0];
block test(clk, data_in: u8[0]) -> u8[0] {
    reg r: u8[0] = E;
    r := data_in;
    r
}
"#;
    let package = compile(source, empty_package("design"), ResetBehavior::default())?;
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    assert_eq!(
        package.get_top_block().unwrap().registers[0].reset_value,
        None
    );
    Ok(())
}

#[test]
fn mixed_empty_array_reset_preserves_registered_payload_and_enable() -> Result<(), Box<dyn Error>> {
    let source = r#"
const E = bits[0]:0 as u8[0];
block mixed(clk, en: bool, x: u8, tag: u2) -> u10 {
    reg q: (u2, (u8[0], u4)[2]) = (u2:1, [(E, u4:10), (E, u4:11)]);
    if en { q := (tag, [(E, (x >> u32:4) as u4), (E, x as u4)]); }
    q.0 ++ q.1[u32:0].1 ++ q.1[u32:1].1
}
"#;
    let package = compile(source, empty_package("design"), ResetBehavior::default())?;
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    let regs = &package.get_top_block().unwrap().registers;
    assert_eq!(regs.len(), 1);
    assert_eq!(
        regs[0].reset_value.as_ref(),
        Some(&IrValue::parse_typed(
            "(bits[2]:1, [(bits[0]:0, bits[4]:10), (bits[0]:0, bits[4]:11)])"
        )?)
    );
    let rtl = emit_system_verilog(&package, &BlockCodegenOptions::default())?;
    let pipeline = compile_pipeline_module(&rtl.system_verilog).unwrap();
    let mut runner = FixtureRunner::new(&pipeline).unwrap();
    let ports = runner.port_bindings();
    let reset = ports.input("rst").unwrap();
    let enable = ports.input("en").unwrap();
    let data = ports.input("x").unwrap();
    let tag = ports.input("tag").unwrap();
    for (rst, en, x, t, expected) in [
        (1, 1, 0x55, 3, 0x1ab),
        (0, 1, 0x37, 2, 0x237),
        (0, 0, 0x18, 0, 0x237),
        (1, 0, 0x00, 0, 0x1ab),
    ] {
        let edge = runner
            .step_cycle_with_drive(&mut [], |drive| {
                drive.set_u64(&reset, rst)?;
                drive.set_u64(&enable, en)?;
                drive.set_u64(&data, x)?;
                drive.set_u64(&tag, t)
            })
            .unwrap();
        assert_eq!(edge.outputs["out"].to_u64_if_known(), Some(expected));
    }
    Ok(())
}

#[test]
fn cast_operand_width_is_inferred_independently() -> Result<(), Box<dyn Error>> {
    for (body, result, expected) in [
        ("[1, 2] as u4", "u4", "bits[4]:6"),
        ("[u4:1, 2] as u8", "u8", "bits[8]:0x12"),
        ("0xff as u4[2]", "u4[2]", "[bits[4]:15, bits[4]:15]"),
    ] {
        assert_expr_matches_dslx(
            "",
            result,
            body,
            [(vec![], IrValue::parse_typed(expected)?)],
        );
    }
    assert_expr_rejected_by_both("", "u8", "[1, 2] as u8");
    assert_expr_rejected_by_both("", "u1[4]", "1 as u1[4]");
    Ok(())
}

#[test]
fn casts_in_local_functions_and_module_constants_use_resolved_dimensions()
-> Result<(), Box<dyn Error>> {
    let declarations = r#"
const WIDTH = u32:3;
const COUNT = u32:3;
fn unpack(x: u9) -> uN[WIDTH][COUNT] { x as uN[WIDTH][COUNT] }
fn pack(xs: uN[WIDTH][COUNT]) -> u9 { xs as u9 }
const SEED = unpack(u9:0x15e);
const PACKED = pack([u3:4, u3:2, u3:7]);
"#;
    let result = "(u3[3], u9, u3[3], u9)";
    let params = "x: u9, xs: u3[3]";
    let body = "(SEED, PACKED, unpack(x), pack(xs))";
    let evaluator = ExpressionEvaluator::compile_block(
        &format!("{declarations}\n{}", bx_source(params, result, body)),
        &ImportOptions::default(),
        compile_dslx_module(
            &format!("{declarations}\n{}", dslx_source(params, result, body)),
            &[],
        ),
    )?;
    let input = IrValue::parse_typed("[bits[3]:3, bits[3]:4, bits[3]:5]")?;
    let actual = evaluator.compare(&[IrValue::make_ubits(9, 0x71)?, input])?;
    assert_eq!(
        actual,
        IrValue::parse_typed(
            "([bits[3]:5, bits[3]:3, bits[3]:6], bits[9]:0x117, [bits[3]:1, bits[3]:6, bits[3]:1], bits[9]:0xe5)"
        )?
    );
    Ok(())
}

#[test]
fn mismatched_widths_and_non_bits_shapes_are_rejected() -> Result<(), Box<dyn Error>> {
    let declarations = "enum E : u4 { A = 0 } struct S { data: u4 }";
    for (a, b) in [
        ("u7", "u4[2]"),
        ("u9", "u4[2]"),
        ("u1", "u4[0]"),
        ("u8", "u2[2][2]"),
        ("u8", "(u2, u2)[2]"),
        ("u8", "S[2]"),
        ("u8", "E[2]"),
        ("E", "u2[2]"),
        ("u2[4]", "u4[2]"),
    ] {
        for (from, to) in [(a, b), (b, a)] {
            let source = format!("{declarations}\nfn sample(x: {from}) -> {to} {{ x as {to} }}");
            assert!(
                check_module(&parse_module(&source)?).is_err(),
                "BX accepted {source}"
            );
            assert!(
                compile_dslx_module(&source, &[]).is_err(),
                "DSLX accepted {source}"
            );
        }
    }
    Ok(())
}
