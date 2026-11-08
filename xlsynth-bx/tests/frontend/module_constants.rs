// SPDX-License-Identifier: Apache-2.0

use super::directed_expr_eval::compile_dslx_module;
use super::expr_eval::{ExpressionEvaluator, bx_source, dslx_source};
use std::error::Error;
use xlsynth_bx::ast::Type;
use xlsynth_bx::{
    ImportOptions, ResetBehavior, check_module, compile, empty_package, parse_module,
};
use xlsynth_codegen::{BlockCodegenOptions, emit_system_verilog};
use xlsynth_pir::IrValue;
use xlsynth_vastly::{FixtureRunner, compile_pipeline_module};

#[test]
fn computed_constants_preserve_signed_and_aggregate_values() -> Result<(), Box<dyn Error>> {
    let declarations = r#"
struct Payload { words: s8[2], tag: uN[97] }
const DELTA: s8 = -3;
fn shift(x: s8) -> s8 { x + DELTA }
fn twice(x: s8) -> s8 { shift(shift(x)) }
pub const SEED: Payload = Payload {
    words: [twice(s8:2), shift(s8:-126)],
    tag: uN[97]:0x1000000000000000000000000,
};
const WORDS = SEED.words;
const NEGATIVE = WORDS[u32:0] < s8:0;
const METADATA = (DELTA, NEGATIVE);
fn sample_word(DELTA: s8) -> s8 { DELTA + WORDS[u32:0] }
"#;
    let checked = check_module(&parse_module(declarations)?)?;
    let constants = checked.constants();
    assert_eq!(
        constants
            .iter()
            .map(|c| (c.name.as_str(), c.public))
            .collect::<Vec<_>>(),
        [
            ("DELTA", false),
            ("SEED", true),
            ("WORDS", false),
            ("NEGATIVE", false),
            ("METADATA", false)
        ]
    );
    let expected_seed =
        IrValue::parse_typed("([bits[8]:252,bits[8]:127],bits[97]:0x1000000000000000000000000)")?;
    assert_eq!(constants[1].ty, Type::Named("Payload".into()));
    assert_eq!(constants[1].value, expected_seed);
    let params = "x: s8";
    let result = "(s8, Payload, (s8, bool))";
    let body = "(sample_word(x), SEED, METADATA)";
    let bx = format!("{declarations}\n{}", bx_source(params, result, body));
    let dslx = format!("{declarations}\n{}", dslx_source(params, result, body));
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&dslx, &[]),
    )?;
    assert_eq!(
        evaluator.compare(&[IrValue::make_sbits(8, 12)?])?,
        IrValue::make_tuple(&[
            IrValue::make_sbits(8, 8)?,
            expected_seed,
            IrValue::parse_typed("(bits[8]:253,bits[1]:1)")?,
        ])
    );
    Ok(())
}

#[test]
fn constant_drives_register_reset_and_block_datapath() -> xlsynth_vastly::Result<()> {
    let source = r#"
const BIAS = u8:7;
fn adjust(x: u8) -> u8 { x + BIAS }
const INITIAL = adjust(u8:250);
pub block delay(clk, rst, data: u8) -> u8 {
    reg q: u8 = INITIAL;
    let const = data;
    if const!=BIAS { q := const + BIAS; }
    q
}
"#;
    let package = compile(source, empty_package("design"), ResetBehavior::default()).unwrap();
    assert_eq!(
        package.get_top_block().unwrap().registers[0].reset_value,
        Some(IrValue::make_ubits(8, 1).unwrap())
    );
    let rtl = emit_system_verilog(&package, &BlockCodegenOptions::default()).unwrap();
    let pipeline = compile_pipeline_module(&rtl.system_verilog)?;
    let mut runner = FixtureRunner::new(&pipeline)?;
    let ports = runner.port_bindings();
    let reset = ports.input("rst")?;
    let input = ports.input("data")?;
    for (rst, x, expected) in [(1, 42, 1), (0, 9, 16), (0, 7, 16), (1, 9, 1)] {
        let cycle = runner.step_cycle_with_drive(&mut [], |drive| {
            drive.set_u64(&reset, rst)?;
            drive.set_u64(&input, x)
        })?;
        assert_eq!(cycle.outputs["out"].to_u64_if_known(), Some(expected));
    }
    Ok(())
}

#[test]
fn lambdas_resolve_globals_and_shadow_them_with_their_own_parameters() -> Result<(), Box<dyn Error>>
{
    let declarations = r#"
const OFFSET = u8:7;
fn lanes(xs: u8[2]) -> (u8[2], u8[2]) {
    (map(xs, |item| item + OFFSET), map(xs, |OFFSET| OFFSET + u8:1))
}
"#;
    let bx = format!("{declarations} block sample(clk, xs:u8[2])->(u8[2],u8[2]) {{ lanes(xs) }}");
    let dslx = format!("{declarations} fn sample(xs:u8[2])->(u8[2],u8[2]) {{ lanes(xs) }}");
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&dslx, &[]),
    )?;
    assert_eq!(
        evaluator.compare(&[IrValue::parse_typed("[bits[8]:4,bits[8]:255]")?])?,
        IrValue::parse_typed("([bits[8]:11,bits[8]:6],[bits[8]:5,bits[8]:0])")?
    );

    // The enclosing let hides the global even in a callback that disallows
    // captures.
    for head in ["fn read(xs:u8[2])", "block read(clk, xs:u8[2])"] {
        let source = format!(
            "const OFFSET = u8:7; {head}->u8[2] {{ let OFFSET = xs[0]; map(xs, |item| item + OFFSET) }}"
        );
        let error = check_module(&parse_module(&source)?).unwrap_err();
        assert!(
            error.contains("cannot capture enclosing value 'OFFSET'"),
            "{source}\n{error}"
        );
    }
    Ok(())
}

#[test]
fn named_scalar_patterns_match_their_evaluated_bit_values() -> Result<(), Box<dyn Error>> {
    let declarations = "const NEG: s8 = -4; const ZERO = NEG + s8:4;";
    let body = "match x { NEG | ZERO => true, _ => false }";
    let bx = format!("{declarations}\n{}", bx_source("x: s8", "bool", body));
    let dslx = format!("{declarations}\n{}", dslx_source("x: s8", "bool", body));
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&dslx, &[]),
    )?;
    for (x, expected) in [(-4, true), (0, true), (-1, false), (1, false)] {
        assert_eq!(
            evaluator.compare(&[IrValue::make_sbits(8, x)?])?,
            IrValue::bool(expected)
        );
    }
    for (prefix, patterns) in [
        ("", "UNKNOWN"),
        ("let NEG = x;", "NEG"),
        ("", "NEG | s8:-4"),
        ("", "WRONG_SIGN"),
        ("", "ARRAY"),
    ] {
        let source = format!(
            "{declarations} const WRONG_SIGN = u8:0; const ARRAY = [s8:0]; \
             fn f(x:s8)->bool {{ {prefix} match x {{ {patterns} => true, _ => false }} }}"
        );
        assert!(check_module(&parse_module(&source)?).is_err(), "{source}");
    }
    Ok(())
}

#[test]
fn invalid_constants_are_rejected_even_when_unused() -> Result<(), String> {
    for source in [
        "const A = A;",
        "const A = B; const B = u8:1;",
        "const A = late(u8:0); fn late(x:u8)->u8 { x }",
        "fn early()->u8 { A } const A = u8:1;",
        "block early(clk)->u8 { A } const A = u8:1;",
        "const A = u8:1; const A = u8:2;",
        "fn A()->u8 { u8:1 } const A = u8:1;",
        "const A = u8:1; struct A { value: u8 }",
        "const bit_count = u8:1;",
        "const IOResult = u8:1;",
        "extern fn linked(x:u8)->u8; const A = linked(u8:1);",
        "const A = 1 + 2;",
        "const A: u8[2] = [u8:0];",
        "const A: u8 = s8:-1;",
    ] {
        assert!(check_module(&parse_module(source)?).is_err(), "{source}");
    }
    Ok(())
}
