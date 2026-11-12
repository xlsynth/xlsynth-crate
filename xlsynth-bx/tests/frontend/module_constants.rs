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
pub block delay(clk, data: u8) -> u8 {
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

    let body = "let OFFSET = xs[u32:0]; map(xs, |item| item + OFFSET)";
    let native = format!(
        "const OFFSET = u8:7; {}",
        dslx_source("xs:u8[2]", "u8[2]", body)
    );
    for bx in [
        format!(
            "const OFFSET = u8:7; {}",
            bx_source("xs:u8[2]", "u8[2]", body)
        ),
        format!(
            "const OFFSET = u8:7; fn read(xs:u8[2])->u8[2] {{ {body} }} {}",
            bx_source("xs:u8[2]", "u8[2]", "read(xs)")
        ),
    ] {
        let evaluator = ExpressionEvaluator::compile_block(
            &bx,
            &ImportOptions::default(),
            compile_dslx_module(&native, &[]),
        )?;
        assert_eq!(
            evaluator.compare(&[IrValue::parse_typed("[bits[8]:4, bits[8]:255]")?])?,
            IrValue::parse_typed("[bits[8]:8, bits[8]:3]")?
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

#[test]
fn named_dimensions_shape_a_nested_datapath_and_its_counted_traversal() -> Result<(), Box<dyn Error>>
{
    let declarations = r#"
fn next_width(x: u32) -> u32 { x + u32:1 }
const W = next_width(u32:6);
const N = u32:3;
const SIGNED = u1:1;
enum Mask : uN[W] { High = 3 }
struct Frame { rows: xN[SIGNED][W][N][2], tag: bits[W] }
const MASKS: uN[W][N] = [1, 2, Mask::High as uN[W]];
fn sum(row: sN[W][N]) -> sN[W] {
    for (i, acc): (u32, sN[W]) in u32:0..N { acc + row[i] }(sN[W]:0)
}
"#;
    let params = "frame: Frame";
    let result = "(sN[W], s16, bits[W])";
    let body = r#"
    let row: xN[SIGNED][W][N] = frame.rows[u32:1];
    (sum(row), (row[u32:0] as xN[SIGNED][W]) as s16, frame.tag ^ MASKS[u32:2])
"#;
    let bx = format!("{declarations}{}", bx_source(params, result, body));
    let native = format!("{declarations}{}", dslx_source(params, result, body));
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&native, &[]),
    )?;
    let frame = IrValue::parse_typed(
        "([[bits[7]:124, bits[7]:2, bits[7]:7], [bits[7]:65, bits[7]:5, bits[7]:12]], bits[7]:91)",
    )?;
    assert_eq!(
        evaluator.compare(&[frame])?,
        IrValue::parse_typed("(bits[7]:82, bits[16]:65473, bits[7]:88)")?,
    );
    Ok(())
}

#[test]
fn named_dimensions_in_registers_preserve_the_concrete_reset_shape() -> Result<(), Box<dyn Error>> {
    let source = r#"
const WIDTH = u32:5;
const LANES = u32:2;
const RESET: uN[WIDTH][LANES] = [31, 7];
pub block delay(clk, data: uN[WIDTH][LANES]) -> uN[WIDTH][LANES] {
    reg q: uN[WIDTH][LANES] = RESET;
    q := data;
    q
}
"#;
    let package = compile(source, empty_package("design"), ResetBehavior::default())?;
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    assert_eq!(
        package.get_top_block().unwrap().registers[0].reset_value,
        Some(IrValue::parse_typed("[bits[5]:31, bits[5]:7]")?),
    );
    Ok(())
}

#[test]
fn body_dimensions_follow_lexical_shadowing() -> Result<(), Box<dyn Error>> {
    let declaration = "const W = u32:3;";
    let body = "let W = u32:4; let narrow: uN[W] = x as uN[W]; narrow as u8";
    let bx = format!("{declaration} {}", bx_source("x:u8", "u8", body));
    let dslx = format!("{declaration} {}", dslx_source("x:u8", "u8", body));
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&dslx, &[]),
    )?;
    assert_eq!(
        evaluator.compare(&[IrValue::make_ubits(8, 0xff)?])?,
        IrValue::make_ubits(8, 15)?
    );
    Ok(())
}

#[test]
fn named_zero_dimensions_lower_to_empty_array_and_bit_types() -> Result<(), String> {
    let declarations = "const Z = u32:0;";
    let params = "xs: u8[Z], y: bits[Z]";
    let result = "(u8[Z], uN[Z])";
    let bx = format!("{declarations} {}", bx_source(params, result, "(xs, y)"));
    let native = format!("{declarations} {}", dslx_source(params, result, "(xs, y)"));
    ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&native, &[]),
    )?;
    Ok(())
}

#[test]
fn concrete_dimensions_require_visible_constants_with_the_dslx_type() {
    for source in [
        "const _=u32:3; struct S { x:uN[_] } fn sample(x:S)->u8 { x.x as u8 }",
        "fn sample(x:uN[N])->uN[N] { x } const N=u32:5;",
        "fn sample(x:u8[MISSING])->u8 { x[u32:0] }",
        "const N=u8:3; fn sample(x:uN[N])->u3 { x }",
        "const N=s32:3; fn sample(x:u8[N])->u8 { x[u32:0] }",
        "const N=u64:3; struct P { data:u8[N] } fn sample()->u8 { u8:0 }",
        "const N=(u32:3,u32:4); fn sample(x:bits[N])->u3 { x }",
        "enum Length:u32 { A=3 } const N=Length::A; fn sample(x:uN[N])->u3 { x }",
        "const N=true; fn sample(x:bits[N])->u1 { x }",
        "const S=s1:-1; fn sample(x:xN[S][8])->s8 { x }",
        "const S=u2:1; fn sample(x:xN[S][8])->s8 { x }",
        "const N=u32:3; fn sample(N:u32,x:uN[N])->uN[N] { x }",
        "const N=u32:3; fn sample(N:u32)->u8 { let v:uN[N]=1; v as u8 }",
    ] {
        let bx = parse_module(source).and_then(|m| check_module(&m));
        assert!(bx.is_err(), "BX accepted {source}");
        assert!(
            compile_dslx_module(source, &[]).is_err(),
            "DSLX accepted {source}"
        );
    }
}

#[test]
fn overwritten_literal_dimensions_resolve_names_in_their_source_scope() -> Result<(), Box<dyn Error>>
{
    for source in [
        "fn sample() -> u8 { u8:(uN[MISSING]:1) }",
        "struct S { x: u8 } fn sample() -> u8 { u8:(uN[S]:1) }",
        "enum E: u8 { A = 0 } fn sample() -> u8 { u8:(uN[E]:1) }",
        "fn sample() -> u8 { u8:(uN[Source]:1) }",
        "fn sample() -> u8 { u8:(uN[IOResult]:1) }",
        "const C = u8:(uN[LATER]:1); const LATER = u32:4; fn sample() -> u8 { C }",
        "fn sample() -> u8 { let a = { let gone = u8:3; gone }; u8:(uN[gone]:1) }",
    ] {
        assert!(parse_module(source).is_err(), "{source}");
        assert!(compile_dslx_module(source, &[]).is_err(), "{source}");
    }
    let declarations = r#"
const W = s8:-1;
enum Code: u8 { A = 0 }
struct Input { x: u8 }
fn qualified() -> u8 { u8:(uN[Source::Member]:1) }
fn shadow_builtin(Source: u8) -> u8 { u8:(uN[Source]:1) }
fn pick(Input: u8) -> u8 {
    let local = Input;
    u8:(uN[W]:(uN[local]:(uN[Input]:(uN[Code::A::NoSuchMember]:129))))
}
"#;
    let bx = format!("{declarations} {}", bx_source("x:u8", "u8", "pick(x)"));
    let native = format!("{declarations} {}", dslx_source("x:u8", "u8", "pick(x)"));
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&native, &[]),
    )?;
    assert_eq!(
        evaluator.compare(&[IrValue::make_ubits(8, 4)?])?,
        IrValue::make_ubits(8, 129)?
    );
    let declarations = r#"
fn lane<N: uN[u32:(uN[N]:32)]>(x: uN[N]) -> u8 {
    let mapped = map([x as u8], |v: uN[u32:(uN[v]:8)]| -> uN[u32:(uN[v]:8)] { v });
    mapped[u32:0] ^ u8:(uN[match x { y => u8:(uN[y]:2) }]:0)
}
"#;
    let bx = format!("{declarations} {}", bx_source("x:u8", "u8", "lane(x)"));
    let native = format!("{declarations} {}", dslx_source("x:u8", "u8", "lane(x)"));
    let evaluator = ExpressionEvaluator::compile_block(
        &bx,
        &ImportOptions::default(),
        compile_dslx_module(&native, &[]),
    )?;
    let input = IrValue::make_ubits(8, 72)?;
    assert_eq!(evaluator.compare(std::slice::from_ref(&input))?, input);
    Ok(())
}

#[test]
fn overwritten_loop_dimensions_resolve_binders_in_the_annotation() -> Result<(), Box<dyn Error>> {
    for spelling in ["for", "unroll_for!"] {
        let declarations = format!(
            r#"
fn pick<N: u32>(x: u8) -> u8 {{
    {spelling} (i, a): (uN[u32:(uN[i]:32)], uN[u32:(uN[a]:8)]) in u32:0..u32:2 {{ a }}(x)
}}
"#
        );
        let call = "pick<u32:1>(x)";
        let bx = format!("{declarations} {}", bx_source("x:u8", "u8", call));
        let native = format!("{declarations} {}", dslx_source("x:u8", "u8", call));
        let evaluator = ExpressionEvaluator::compile_block(
            &bx,
            &ImportOptions::default(),
            compile_dslx_module(&native, &[]),
        )?;
        let input = IrValue::make_ubits(8, 39)?;
        assert_eq!(evaluator.compare(std::slice::from_ref(&input))?, input);
        for (old, replacement) in [
            ("..u32:2", "..u32:(uN[i]:2)"),
            ("{ a }(x)", "{ a }(u8:(uN[a]:1))"),
        ] {
            assert!(
                parse_module(&bx.replace(old, replacement)).is_err(),
                "{spelling} {replacement}"
            );
            assert!(
                compile_dslx_module(&native.replace(old, replacement), &[]).is_err(),
                "{spelling} {replacement}"
            );
        }
    }
    Ok(())
}

#[test]
fn a_forward_struct_layout_waits_for_its_constant_and_enum_dependencies() -> Result<(), String> {
    let source = r#"
struct Frame { lane: Lane }
const N = u32:2;
const W = u32:3;
enum Code: uN[W] { A = 0, B = 3 }
fn sample(f: Frame) -> uN[W] { f.lane.codes[u32:1] as uN[W] }
struct Lane { codes: Code[N] }
"#;
    let checked = check_module(&parse_module(source)?)?;
    let lane = checked.structs().iter().find(|s| s.name == "Lane").unwrap();
    let Type::Array { element, size } = &lane.fields[0].ty else {
        panic!("expected an array of codes");
    };
    assert_eq!(*size, 2);
    assert!(matches!(element.as_ref(), Type::Enum { name, width: 3, .. } if name == "Code"));
    Ok(())
}
