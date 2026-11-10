// SPDX-License-Identifier: Apache-2.0

use std::{error::Error, fs};

use tempfile::tempdir;
use xlsynth_bx::{ImportOptions, check_module, check_source_with_options, parse_module};
use xlsynth_pir::IrValue;

use super::expr_eval::{BlockEvaluator, bx_source, compile_bundled_module, dslx_source};

#[test]
fn nested_struct_layouts_cross_function_and_block_signatures() -> Result<(), Box<dyn Error>> {
    let declarations = r#"
struct Chunk<W: u32> { data: uN[W] }
struct Frame<W: u32, N: u32 = {W + u32:1}> { chunks: Chunk<W>[N] }
fn second<W: u32>(x: Frame<W>) -> Chunk<W> { x.chunks[u32:1] }
fn relay(x: Frame<2>) -> Chunk<2> { second<2>(x) }
"#;
    let body = "relay(Frame<2> { chunks: x.chunks })";
    let native = format!(
        "{declarations}\n{}",
        dslx_source("x: Frame<2>", "Chunk<2>", body)
    );
    let dslx = compile_bundled_module(&native, &[])?;
    let input = IrValue::parse_typed("([(bits[2]:0), (bits[2]:3), (bits[2]:1)])")?;
    for (params, result, body) in [
        ("x: Frame<2>", "Chunk<2>", body),
        (
            "x: Frame::<2>",
            "Chunk::<2>",
            "second::<2>(Frame::<2> { chunks: x.chunks })",
        ),
    ] {
        let bx = format!("{declarations}\n{}", bx_source(params, result, body));
        let checked = check_module(&parse_module(&bx)?)?;
        let evaluator = BlockEvaluator::lower(&checked, &bx)?;
        assert_eq!(
            evaluator.compare_ir(&dslx, std::slice::from_ref(&input))?,
            IrValue::parse_typed("(bits[2]:3)")?
        );
    }
    Ok(())
}

#[test]
fn tuple_fields_infer_missing_widths_and_use_known_widths() -> Result<(), Box<dyn Error>> {
    let declarations = r#"
struct Record<N: u32, M: u32> { data: uN[N], tail: (uN[N], uN[M]) }
fn read(x: u8) -> (u8, u1) {
    let record = Record { data: x, tail: (1, 1) };
    record.tail
}
"#;
    let bx = format!(
        "{declarations}\n{}",
        bx_source("x: u8", "(u8, u1)", "read(x)")
    );
    let dslx = format!(
        "{declarations}\n{}",
        dslx_source("x: u8", "(u8, u1)", "read(x)")
    );
    let checked = check_module(&parse_module(&bx)?)?;
    let dslx = compile_bundled_module(&dslx, &[])?;
    assert_eq!(
        BlockEvaluator::lower(&checked, &bx)?.compare_ir(&dslx, &[IrValue::make_ubits(8, 7)?])?,
        IrValue::parse_typed("(bits[8]:1, bits[1]:1)")?
    );
    Ok(())
}

#[test]
fn concrete_struct_types_specialize_compile_time_helpers() -> Result<(), Box<dyn Error>> {
    let declarations = r#"
fn width<N: u32>() -> u32 { N }
fn plain_width() -> u32 { u32:6 }
struct Word<N: u32 = {width<7>()}> { data: uN[N] }
struct Bundle {
    computed: Word<{width<9>()}>,
    defaulted: Word<>,
    plain: Word<{plain_width()}>,
}
fn read(word: Word<{width<8>()}>, bundle: Bundle) -> (u8, u9, u7, u6) {
    (word.data, bundle.computed.data, bundle.defaulted.data, bundle.plain.data)
}
"#;
    let params = "word: Word<{width<8>()}>, bundle: Bundle";
    let result = "(u8, u9, u7, u6)";
    let body = "read(word, bundle)";
    let bx = format!("{declarations}\n{}", bx_source(params, result, body));
    let dslx = format!("{declarations}\n{}", dslx_source(params, result, body));
    let checked = check_module(&parse_module(&bx)?)?;
    let dslx = compile_bundled_module(&dslx, &[])?;
    let inputs = [
        IrValue::parse_typed("(bits[8]:17)")?,
        IrValue::parse_typed("((bits[9]:300), (bits[7]:65), (bits[6]:33))")?,
    ];
    assert_eq!(
        BlockEvaluator::lower(&checked, &bx)?.compare_ir(&dslx, &inputs)?,
        IrValue::parse_typed("(bits[8]:17, bits[9]:300, bits[7]:65, bits[6]:33)")?
    );
    Ok(())
}

#[test]
fn siblings_supply_inferred_struct_arguments() -> Result<(), Box<dyn Error>> {
    let declarations = r#"
struct Word<N: u32> { data: uN[N] }
fn choose(c: bool, x: u8) -> (u8, u8, u8, u8) {
    let selected = if c { Word { data: 1 } } else { Word<8> { data: x } };
    let words = [Word { data: 1 }, Word<8> { data: x }];
    let matched = match c {
        true => Word { data: 1 },
        false => Word<8> { data: x },
    };
    let nested = if c { Word { data: 1 } } else {
        if c { Word { data: 2 } } else { Word<8> { data: x } }
    };
    (selected.data, words[u32:0].data, matched.data, nested.data)
}
"#;
    let params = "c: bool, x: u8";
    let result = "(u8, u8, u8, u8)";
    let body = "choose(c, x)";
    // The later constructor can supply the binding through an explicit
    // argument or through its typed data field.
    for declarations in [
        declarations.to_owned(),
        declarations.replace("Word<8>", "Word"),
    ] {
        let bx = format!("{declarations}\n{}", bx_source(params, result, body));
        let dslx = format!("{declarations}\n{}", dslx_source(params, result, body));
        let checked = check_module(&parse_module(&bx)?)?;
        let dslx = compile_bundled_module(&dslx, &[])?;
        assert_eq!(
            BlockEvaluator::lower(&checked, &bx)?
                .compare_ir(&dslx, &[IrValue::bool(false), IrValue::make_ubits(8, 7)?])?,
            IrValue::parse_typed("(bits[8]:7, bits[8]:1, bits[8]:7, bits[8]:7)")?
        );
    }
    Ok(())
}

#[test]
fn siblings_merge_partial_struct_arguments_before_inferring_fields() -> Result<(), Box<dyn Error>> {
    for (tag_parametric, tag) in [("M: u32", "u9:1"), ("M: u32 = {N + u32:1}", "1")] {
        let declarations = format!(
            r#"
struct Word<N: u32, {tag_parametric}> {{ data: uN[N], tag: uN[M] }}
fn choose(c: bool, x: u8, y: u9) -> (u8, u8, u8) {{
    let selected = if c {{ Word<8> {{ data: x, tag: y }} }} else {{ Word {{ data: 1, tag: {tag} }} }};
    let words = [Word<{{u32:4 + u32:4}}> {{ data: x, tag: y }}, Word {{ data: 1, tag: {tag} }}];
    let matched = match c {{
        true => Word<8> {{ data: x, tag: y }},
        false => Word {{ data: 1, tag: {tag} }},
    }};
    (selected.data, words[u32:0].data, matched.data)
}}
"#
        );
        let params = "c: bool, x: u8, y: u9";
        let result = "(u8, u8, u8)";
        let body = "choose(c, x, y)";
        let bx = format!("{declarations}\n{}", bx_source(params, result, body));
        let dslx = format!("{declarations}\n{}", dslx_source(params, result, body));
        let checked = check_module(&parse_module(&bx)?)?;
        let dslx = compile_bundled_module(&dslx, &[])?;
        assert_eq!(
            BlockEvaluator::lower(&checked, &bx)?.compare_ir(
                &dslx,
                &[
                    IrValue::bool(false),
                    IrValue::make_ubits(8, 7)?,
                    IrValue::make_ubits(9, 9)?
                ],
            )?,
            IrValue::parse_typed("(bits[8]:1, bits[8]:7, bits[8]:1)")?
        );
    }
    Ok(())
}

#[test]
fn struct_constructors_use_typed_fields_and_expected_nominal_types() -> Result<(), Box<dyn Error>> {
    let source = r#"
struct Payload<W: u32, N: u32> { word: uN[W], samples: uN[W][N] }
struct Tag<N: u32> { data: u8 }
fn sum(x: Payload<5,2>) -> u5 { x.word + x.samples[u32:1] }
fn read_tag(x: Tag<5>) -> u8 { x.data }
fn tag_code<N: u32>(x: Tag<N>) -> u32 { (x.data as u32) + N }
#[allow(combinational_block)]
block sample(clk, x: u5) -> (u5, u5, u8, u32) {
    let inferred = Payload { word: x, samples: [u5:2, u5:3] };
    let annotated: Payload<5,2> = Payload { word: 6, samples: [1, 2] };
    (sum(inferred), sum(annotated), read_tag(Tag { data: 7 }), tag_code(Tag<9> { data: 4 }))
}
"#;
    let checked = check_module(&parse_module(source)?)?;
    assert_eq!(
        BlockEvaluator::lower(&checked, source)?.evaluate(&[IrValue::make_ubits(5, 11)?])?,
        IrValue::parse_typed("(bits[5]:14, bits[5]:8, bits[8]:7, bits[32]:13)")?
    );
    Ok(())
}

#[test]
fn struct_parametrics_distinguish_identical_field_layouts() -> Result<(), Box<dyn Error>> {
    let source = r#"
struct Tag<N: u32> { data: u8 }
fn read(x: Tag<2>) -> u8 { x.data }
block sample(clk, tag: Tag<3>) -> u8 { read(tag) }
"#;
    let error = check_module(&parse_module(source)?).unwrap_err();
    assert!(
        error.contains("Tag<2>") && error.contains("Tag<3>"),
        "{error}"
    );
    Ok(())
}

#[test]
fn type_arguments_and_defaults_respect_shadowing() -> Result<(), Box<dyn Error>> {
    let cases = [
        r#"
const N = u32:7;
struct Word<W: u32> { data: uN[W] }
fn read(N: u32, x: Word<{N + u32:1}>) -> u8 { x.data }
"#,
        r#"
const N = u32:7;
struct Word<N: u32 = {N}> { data: uN[N] }
fn read(x: Word<>) -> u7 { x.data }
"#,
    ];
    for source in cases {
        let error = check_module(&parse_module(source)?).unwrap_err();
        assert!(error.contains("unknown value 'N'"), "{error}");
    }
    Ok(())
}

#[test]
fn qualified_type_arguments_can_use_module_constants() -> Result<(), Box<dyn Error>> {
    let source = r#"
use apfloat;
const WIDTH = u32:4;
struct Holder { value: apfloat::APFloat<{WIDTH + u32:1}, 10> }
fn exponent(x: Holder) -> u5 { x.value.bexp }
"#;
    check_module(&parse_module(source)?)?;
    Ok(())
}

#[test]
fn struct_templates_must_precede_referencing_structs() -> Result<(), Box<dyn Error>> {
    let source = r#"
struct Outer { inner: Inner<8> }
struct Inner<W: u32> { data: uN[W] }
"#;
    let error = check_module(&parse_module(source)?).unwrap_err();
    assert_eq!(
        error,
        "struct template 'Inner' is declared after struct 'Outer'"
    );
    Ok(())
}

#[test]
fn unused_templates_check_field_and_default_names() -> Result<(), Box<dyn Error>> {
    for source in [
        "struct Bad<N: u32> { data: uN[TYPO] }",
        "struct Bad<N: u32 = {TYPO}> { data: uN[N] }",
    ] {
        let error = check_module(&parse_module(source)?).unwrap_err();
        assert!(error.contains("TYPO"), "{error}");
    }
    Ok(())
}

#[test]
fn imported_native_templates_evaluate_arguments_and_defaults_in_their_scopes()
-> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    fs::write(
        directory.path().join("wire.bx"),
        r#"
const BASE = u32:2;
struct Data<W: u32> { value: uN[W] }
pub struct Packet<N: u32 = {BASE}, W: u32 = {N + BASE}> { data: Data<W> }
pub fn default_packet(x: u4) -> Packet<> { Packet<> { data: Data<4> { value: x } } }
pub fn widened(x: u5) -> Packet<3> { Packet<3> { data: Data<5> { value: x } } }
pub fn read(p: Packet<3>) -> u5 { p.data.value }
"#,
    )?;
    let options = ImportOptions {
        additional_search_paths: vec![directory.path()],
        ..Default::default()
    };
    let source = r#"
use wire;
use wire::Packet;
const BASE = u32:3;
fn convert(p: Packet<BASE>) -> u5 { wire::read(p) }
#[allow(combinational_block)]
block sample(clk, x: u4, y: u5) -> (u4, u5) {
    let fallback: wire::Packet<> = wire::default_packet(x);
    (fallback.data.value, convert(wire::widened(y)))
}
"#;
    let checked = check_source_with_options(source, &options)?;
    let input = [IrValue::make_ubits(4, 13)?, IrValue::make_ubits(5, 27)?];
    assert_eq!(
        BlockEvaluator::lower(&checked, source)?.evaluate(&input)?,
        IrValue::parse_typed("(bits[4]:13, bits[5]:27)")?
    );
    Ok(())
}
