// SPDX-License-Identifier: Apache-2.0

use super::directed_expr_eval::{
    assert_expr_matches_dslx, assert_expr_rejected_by_both, compile_dslx_module,
};
use super::expr_eval::BlockEvaluator;
use std::{error::Error, fs};
use tempfile::tempdir;
use xlsynth_bx::ast::Expr;
use xlsynth_bx::{
    ImportOptions, ResetBehavior, check_module, check_module_with_options,
    check_source_with_options, compile, compile_with_options, empty_package, lower_module,
    parse_module,
};
use xlsynth_pir::IrValue;
use xlsynth_pir::block2fn::combinational_block_to_fn;
use xlsynth_pir::ir::NodePayload;
use xlsynth_pir::ir_eval::eval_fn_in_package_value;
use xlsynth_pir::ir_parser::Parser;

#[test]
fn named_map_specializes_unary_dslx_builtins() -> Result<(), Box<dyn Error>> {
    assert_expr_matches_dslx(
        "xs: u8[3]",
        "(u8[3], u8[3], u8[3], bool[3], bool[3], bool[3], u3[3])",
        "(map(xs, rev), map(xs, clz), map(xs, ctz), map(xs, and_reduce), map(xs, or_reduce), map(xs, xor_reduce), map(xs, encode))",
        [(
            vec![IrValue::parse_typed(
                "[bits[8]:1, bits[8]:128, bits[8]:255]",
            )?],
            IrValue::parse_typed(
                "([bits[8]:128, bits[8]:1, bits[8]:255], [bits[8]:7, bits[8]:0, bits[8]:0], [bits[8]:0, bits[8]:7, bits[8]:0], [bits[1]:0, bits[1]:0, bits[1]:1], [bits[1]:1, bits[1]:1, bits[1]:1], [bits[1]:1, bits[1]:1, bits[1]:0], [bits[3]:0, bits[3]:7, bits[3]:7])",
            )?,
        )],
    );
    Ok(())
}

#[test]
fn named_map_evaluates_ceil_log2_and_array_builtins_in_constants() -> Result<(), Box<dyn Error>> {
    let constants = r#"
const VALUES = [u32:0, u32:1, u32:8, u32:9];
const LOGS = map(VALUES, ceillog2);
const ROWS = [[u8:1, u8:2], [u8:3, u8:4]];
const REVERSED = map(ROWS, array_rev);
const SIZES = map(ROWS, array_size);
"#;
    let body = "-> (u32[4], u8[2][2], u32[2]) { (LOGS, REVERSED, SIZES) }";
    let bx = format!("{constants} block sample(clk) {body}");
    let native = format!("{constants} fn sample() {body}");
    let evaluator = BlockEvaluator::lower(&check_module(&parse_module(&bx)?)?, &bx)?;
    let reference = compile_dslx_module(&native, &[])?;
    assert_eq!(
        evaluator.compare_ir(&reference, &[])?,
        IrValue::parse_typed(
            "([bits[32]:0,bits[32]:0,bits[32]:3,bits[32]:4], [[bits[8]:2,bits[8]:1],[bits[8]:4,bits[8]:3]], [bits[32]:2,bits[32]:2])"
        )?,
    );
    Ok(())
}

#[test]
fn named_map_ceil_log2_preserves_wide_element_type() -> Result<(), Box<dyn Error>> {
    // The bundled interpreter evaluates the mapped values as u32. The
    // builtin's declared result type has the same width as its input.
    let bx = r#"
block sample(clk) -> uN[96][2] {
    let values = [uN[96]:0x1_0000_0000_0000_0000, uN[96]:0x1_0000_0000_0000_0001];
    map(values, ceillog2)
}"#;
    let evaluator = BlockEvaluator::lower(&check_module(&parse_module(bx)?)?, bx)?;
    assert_eq!(
        evaluator.evaluate(&[])?,
        IrValue::parse_typed("[bits[96]:64, bits[96]:65]")?,
    );
    Ok(())
}

#[test]
fn named_map_ceil_log2_requires_compile_time_unsigned_inputs() {
    assert_expr_rejected_by_both("xs: u32[2]", "u32[2]", "map(xs, ceillog2)");
    assert_expr_rejected_by_both("", "s32[2]", "map([s32:1, s32:8], ceillog2)");
}

#[test]
fn named_map_builtin_yields_to_a_visible_binding() -> Result<(), Box<dyn Error>> {
    let helpers = r#"
fn before(xs: u8[2]) -> u8[2] { map(xs, rev) }
fn rev(x: u8) -> u8 { x + u8:1 }
"#;
    let body = "-> (u8[2], u8[2]) { (before(xs), map(xs, rev)) }";
    let bx = format!("{helpers} block sample(clk, xs: u8[2]) {body}");
    let native = format!("{helpers} fn sample(xs: u8[2]) {body}");
    let evaluator = BlockEvaluator::lower(&check_module(&parse_module(&bx)?)?, &bx)?;
    let reference = compile_dslx_module(&native, &[])?;
    assert_eq!(
        evaluator.compare_ir(
            &reference,
            &[IrValue::parse_typed("[bits[8]:1, bits[8]:128]")?],
        )?,
        IrValue::parse_typed("([bits[8]:128, bits[8]:1], [bits[8]:2, bits[8]:129])")?
    );
    assert_expr_rejected_by_both("xs: u8[2]", "u8[2]", "let rev = u8:0; map(xs, rev)");
    Ok(())
}

#[test]
fn named_map_builtin_yields_to_a_visible_type() {
    for declaration in ["struct rev { field: u8 }", "enum rev : u1 { A = 0 }"] {
        let bx = format!("{declaration}\nblock sample(clk, xs: u8[2]) -> u8[2] {{ map(xs, rev) }}");
        let native = format!("{declaration}\nfn sample(xs: u8[2]) -> u8[2] {{ map(xs, rev) }}");
        assert!(parse_module(&bx).and_then(|m| check_module(&m)).is_err());
        assert!(compile_dslx_module(&native, &[]).is_err());
    }
}

#[test]
fn parametric_struct_shadows_map_from_its_declaration() -> Result<(), Box<dyn Error>> {
    let declaration = "struct map<N: u32> { data: uN[N] }";
    let callback = "fn flip(x: u8) -> u8 { rev(x) }";
    let sample = "fn sample(xs: u8[2]) -> u8[2] { map(xs, flip) }";
    let source = format!("{callback}\n{sample}\n{declaration}");
    check_module(&parse_module(&source)?)?;
    compile_dslx_module(&source, &[])?;

    let block = r#"
block sample(clk, xs: u8[2]) -> (u8[2], u8[2]) { (map(xs, flip), map(xs, |x| x)) }
"#;
    let source = format!("{callback}\n{block}\n{declaration}");
    let plain = lower_module(
        &check_module(&parse_module(&source)?)?,
        empty_package("test"),
        ResetBehavior::default(),
    )?;
    let mut module = parse_module(&source)?;
    module.require_define_before_use = true;
    let strict = lower_module(
        &check_module(&module)?,
        empty_package("test"),
        ResetBehavior::default(),
    )?;
    assert_eq!(strict.to_string(), plain.to_string());

    // The block's map still calls a later local function named rev.
    let source = format!(
        "#![require_define_before_use]\nblock sample(clk, xs: u8[2]) -> u8[2] {{ map(xs, rev) }}\nfn rev(x: u8) -> u8 {{ x }}\n{declaration}"
    );
    assert_eq!(
        check_module(&parse_module(&source)?).unwrap_err(),
        "in 'sample': 'rev' is declared later in the module",
    );

    let source = format!("{callback}\n{declaration}\n{sample}");
    assert!(
        check_module(&parse_module(&source)?).is_err(),
        "BX accepted {source}"
    );
    assert!(
        compile_dslx_module(&source, &[]).is_err(),
        "DSLX accepted {source}"
    );

    let lambda =
        format!("{declaration}\nblock sample(clk, xs: u8[2]) -> u8[2] {{ map(xs, |x| x) }}");
    let error = check_module(&parse_module(&lambda)?).unwrap_err();
    assert!(
        error.contains("an inline lambda argument requires the built-in map"),
        "{error}"
    );
    Ok(())
}

#[test]
fn declaration_order_checks_the_selected_map_arguments() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    fs::write(dir.path().join("enums.x"), "pub enum map : u8 { A = 0 }")?;
    let options = ImportOptions {
        additional_search_paths: vec![dir.path()],
        ..Default::default()
    };
    let mut module = parse_module(
        r#"
block sample(clk, xs: u8[2]) -> u8[2] { map(xs, rev) }
fn later(xs: u8[2]) -> u8[2] { xs }
use enums::map;
"#,
    )?;
    // AST callers can retain distinct ordinary and builtin interpretations.
    // The later enum selects the builtin, whose operands have no forward use.
    module.blocks[0].output = Expr::MapBeforeUse {
        name: "map".into(),
        args: vec![
            Expr::Call {
                name: "later".into(),
                args: vec![Expr::Name("xs".into())],
            },
            Expr::Name("rev".into()),
        ],
        builtin_args: Some(vec![Expr::Name("xs".into()), Expr::Name("rev".into())]),
    };
    let plain = lower_module(
        &check_module_with_options(&module, &options)?,
        empty_package("test"),
        ResetBehavior::default(),
    )?;
    module.require_define_before_use = true;
    let strict = lower_module(
        &check_module_with_options(&module, &options)?,
        empty_package("test"),
        ResetBehavior::default(),
    )?;
    assert_eq!(strict.to_string(), plain.to_string());
    let Expr::MapBeforeUse {
        args,
        builtin_args: Some(builtin_args),
        ..
    } = &mut module.blocks[0].output
    else {
        unreachable!("constructed an ambiguous map call");
    };
    std::mem::swap(args, builtin_args);
    assert_eq!(
        check_module_with_options(&module, &options).unwrap_err(),
        "in 'sample': 'later' is declared later in the module",
    );
    Ok(())
}

#[test]
fn bx_library_keeps_its_named_map_builtin_binding() -> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    fs::write(
        root.path().join("lanes.bx"),
        "pub fn flip(xs: u8[2]) -> u8[2] { map(xs, rev) }",
    )?;
    let bx = r#"
use lanes;
fn rev(x: u8) -> u8 { x + u8:1 }
#[allow(combinational_block)]
block sample(clk, xs: u8[2]) -> (u8[2], u8[2]) {
    (lanes::flip(xs), map(xs, rev))
}
"#;
    let options = ImportOptions {
        additional_search_paths: vec![root.path()],
        ..Default::default()
    };
    let package = compile_with_options(
        bx,
        empty_package("test"),
        ResetBehavior::default(),
        &options,
    )?;
    let sample = combinational_block_to_fn(package.get_block("sample").ok_or("missing sample")?)?;
    assert_eq!(
        eval_fn_in_package_value(
            &package,
            &sample,
            &[IrValue::parse_typed("[bits[8]:1, bits[8]:128]")?],
        ),
        IrValue::parse_typed("([bits[8]:128, bits[8]:1], [bits[8]:2, bits[8]:129])")?
    );
    Ok(())
}

#[test]
fn direct_and_named_map_builtins_precede_a_later_imported_type() -> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    let declaration = "pub enum rev : u1 { ZERO = 0 }";
    fs::write(root.path().join("types.bx"), declaration)?;
    fs::write(root.path().join("enums.x"), declaration)?;
    fs::write(
        root.path().join("structs.x"),
        "pub struct rev { value: u8 }",
    )?;
    fs::write(root.path().join("aliases.x"), "pub type rev = u8;")?;
    fs::write(
        root.path().join("functions.x"),
        "pub fn rev(x: u8) -> u8 { x + u8:1 }",
    )?;
    let block = r#"
#[allow(combinational_block)]
block sample(clk, xs: u8[2]) -> (u8, u8[2]) { (rev(xs[u32:0]), map(xs, rev)) }
"#;
    let options = ImportOptions {
        additional_search_paths: vec![root.path()],
        ..Default::default()
    };
    for (module, expected) in [
        ("types", "(bits[8]:128, [bits[8]:128, bits[8]:1])"),
        ("enums", "(bits[8]:128, [bits[8]:128, bits[8]:1])"),
        ("structs", "(bits[8]:128, [bits[8]:128, bits[8]:1])"),
        ("aliases", "(bits[8]:128, [bits[8]:128, bits[8]:1])"),
        ("functions", "(bits[8]:2, [bits[8]:2, bits[8]:129])"),
    ] {
        let bx = format!("{block}\nuse {module}::{{rev}};");
        let package = compile_with_options(
            &bx,
            empty_package("test"),
            ResetBehavior::default(),
            &options,
        )?;
        let sample =
            combinational_block_to_fn(package.get_block("sample").ok_or("missing sample")?)?;
        let args = [IrValue::parse_typed("[bits[8]:1, bits[8]:128]")?];
        assert_eq!(
            eval_fn_in_package_value(&package, &sample, &args),
            IrValue::parse_typed(expected)?
        );
    }
    for module in ["enums", "structs", "aliases"] {
        for body in ["rev(xs[u32:0])", "map(xs, rev)[u32:0]"] {
            let bx =
                format!("use {module}::{{rev}};\nblock sample(clk, xs: u8[2]) -> u8 {{ {body} }}");
            assert!(check_source_with_options(&bx, &options).is_err());
            let native = format!(
                "#![feature(use_syntax)]\nuse {module}::{{rev}};\nfn sample(xs: u8[2]) -> u8 {{ {body} }}"
            );
            assert!(compile_dslx_module(&native, &[root.path()]).is_err());
        }
        let native = format!(
            "#![feature(use_syntax)]\nfn sample(xs: u8[2]) -> (u8, u8[2]) {{ (rev(xs[u32:0]), map(xs, rev)) }}\nuse {module}::{{rev}};"
        );
        compile_dslx_module(&native, &[root.path()])?;
    }
    Ok(())
}

#[test]
fn named_map_uses_its_callee_for_runtime_and_constant_values() -> Result<(), Box<dyn Error>> {
    let bx = r#"
fn widen(x: s8) -> (s16, bool) { (x as s16, x < s8:0) }
fn mapped(xs: s8[3]) -> (s16, bool)[3] { map(xs, widen) }
const INITIAL = map(s8:-1..=s8:1, widen);
block sample(clk, rst, xs: s8[3]) -> (s16, bool)[3] {
    reg q: (s16, bool)[3] = INITIAL;
    q := mapped(xs);
    q
}"#;
    let package = compile(bx, empty_package("test"), ResetBehavior::default())?;
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    let mapped = package.get_fn("mapped").ok_or("missing mapped")?;
    let targets: Vec<_> = mapped
        .nodes
        .iter()
        .filter_map(|node| match &node.payload {
            NodePayload::Map { to_apply, .. } => Some(to_apply.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(targets, ["widen"]);
    let expected = IrValue::parse_typed(
        "[(bits[16]:65535,bits[1]:1),(bits[16]:0,bits[1]:0),(bits[16]:1,bits[1]:0)]",
    )?;
    let input = IrValue::parse_typed("[bits[8]:255,bits[8]:0,bits[8]:1]")?;
    assert_eq!(
        eval_fn_in_package_value(&package, mapped, &[input]),
        expected
    );
    let block = package.get_block("sample").ok_or("missing sample block")?;
    assert_eq!(block.registers[0].reset_value.as_ref(), Some(&expected));
    Ok(())
}

#[test]
fn named_map_infers_signedness_and_width_for_each_element_type() -> Result<(), Box<dyn Error>> {
    let helper = r#"
fn widen<S: bool, N: u32, W: u32 = {N + 1}>(x: xN[S][N]) -> xN[S][W] {
    x as xN[S][W]
}
"#;
    let params = "a: u8[2], b: s8[2]";
    let result = "-> (u9[2], s9[2]) { (map(a, widen), map(b, widen)) }";
    let bx = format!("{helper} block sample(clk, {params}) {result}");
    let native = format!("{helper} fn sample({params}) {result}");
    let evaluator = BlockEvaluator::lower(&check_module(&parse_module(&bx)?)?, &bx)?;
    let reference = compile_dslx_module(&native, &[])?;
    let lanes = IrValue::parse_typed("[bits[8]:128, bits[8]:255]")?;
    assert_eq!(
        evaluator.compare_ir(&reference, &[lanes.clone(), lanes])?,
        IrValue::parse_typed("([bits[9]:128,bits[9]:255],[bits[9]:384,bits[9]:511])")?
    );
    Ok(())
}

#[test]
fn named_map_binds_explicit_values_before_inferring_the_element_width() -> Result<(), Box<dyn Error>>
{
    let helpers = r#"
fn offset<B: u8, N: u32>(x: uN[N]) -> uN[N] { x + (B as uN[N]) }
fn shifted(xs: u4[2]) -> u4[2] {
    let bias = u8:1 + u8:1;
    map(xs, offset<bias>)
}
"#;
    let result = "-> (u4[2], u8[2]) { (shifted(xs), map(u8:4..u8:6, offset<3>)) }";
    let native = format!("{helpers} fn sample(xs: u4[2]) {result}");
    let reference = compile_dslx_module(&native, &[])?;
    let expected = IrValue::parse_typed("([bits[4]:3,bits[4]:1],[bits[8]:7,bits[8]:8])")?;
    // The pinned DSLX IR converter crashes on arithmetic in a map callback's
    // explicit argument. Its equivalent local constant provides the oracle.
    let inline = helpers
        .replace("let bias = u8:1 + u8:1;", "let bias = u8:1;")
        .replace("offset<bias>", "offset<{bias + u8:1}>");
    for helpers in [helpers, inline.as_str()] {
        let bx = format!("{helpers} block sample(clk, xs: u4[2]) {result}");
        let evaluator = BlockEvaluator::lower(&check_module(&parse_module(&bx)?)?, &bx)?;
        assert_eq!(
            evaluator.compare_ir(
                &reference,
                &[IrValue::parse_typed("[bits[4]:1, bits[4]:15]")?]
            )?,
            expected
        );
    }
    Ok(())
}

#[test]
fn an_unspecialized_value_template_cannot_supply_an_explicit_callable_argument()
-> Result<(), Box<dyn Error>> {
    let bx = r#"
fn offset<K: u8 = {1}>(x: u8) -> u8 { x + K }
fn apply<F: fn(u8) -> u8>(x: u8) -> u8 { F(x) }
block sample(clk, x: u8) -> u8 { apply<offset>(x) }
"#;
    let error = check_module(&parse_module(bx)?).unwrap_err();
    assert!(
        error.contains("must name a concrete BX function"),
        "{error}"
    );
    Ok(())
}

#[test]
fn function_and_block_formals_map_to_the_selected_callback() -> Result<(), Box<dyn Error>> {
    let bx = r#"
fn add1(x:u8) -> u16 { (x as u16) + u16:1 }
fn double(x:u8) -> u16 { (x as u16) * u16:2 }
fn mapped<F:fn(u8)->u16>(xs:u8[2])->u16[2] { map(xs, F) }
block stage<In, Out, F:fn(In)->Out>(clk, xs:In[2])->Out[2] { map(xs, F) }
block sample(clk, xs:u8[2])->(u16[2], u16[2]) {
    let y = stage<double>(xs);
    (mapped<add1>(xs), y)
}"#;
    let native = r#"
fn add1(x:u8) -> u16 { (x as u16) + u16:1 }
fn double(x:u8) -> u16 { (x as u16) * u16:2 }
fn sample(xs:u8[2])->(u16[2], u16[2]) { (map(xs, add1), map(xs, double)) }
"#;
    let checked = check_module(&parse_module(bx)?)?;
    let evaluator = BlockEvaluator::lower(&checked, bx)?;
    let reference = compile_dslx_module(native, &[])?;
    let value = evaluator.compare_ir(
        &reference,
        &[IrValue::parse_typed("[bits[8]:7,bits[8]:255]")?],
    )?;
    assert_eq!(
        value,
        IrValue::parse_typed("([bits[16]:8,bits[16]:256],[bits[16]:14,bits[16]:510])")?
    );
    Ok(())
}

#[test]
fn bx_library_maps_to_bx_and_inferred_dslx_functions() -> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    fs::write(
        root.path().join("math.x"),
        r#"
pub fn inc<N:u32>(x:uN[N])->uN[N] { x + uN[N]:1 }
"#,
    )?;
    fs::write(
        root.path().join("lanes.bx"),
        r#"
use math;
pub fn high(x:u8)->bool { x >= u8:128 }
pub fn bumps(xs:u8[2])->u8[2] { map(xs, math::inc) }
"#,
    )?;
    let bx = r#"
use lanes;
use lanes::bumps;
use math::inc;
#[allow(combinational_block)]
block sample(clk, a:u8[2], b:u16[2])->(bool[2], u8[2], u16[2]) {
    (map(a, lanes::high), bumps(a), map(b, inc))
}"#;
    let opts = ImportOptions {
        additional_search_paths: vec![root.path()],
        ..Default::default()
    };
    let package = compile_with_options(bx, empty_package("test"), ResetBehavior::default(), &opts)?;
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    let sample = combinational_block_to_fn(package.get_block("sample").ok_or("missing sample")?)?;
    let args = [
        IrValue::parse_typed("[bits[8]:127,bits[8]:255]")?,
        IrValue::parse_typed("[bits[16]:32768,bits[16]:65535]")?,
    ];
    assert_eq!(
        eval_fn_in_package_value(&package, &sample, &args),
        IrValue::parse_typed(
            "([bits[1]:0,bits[1]:1],[bits[8]:128,bits[8]:0],[bits[16]:32769,bits[16]:0])"
        )?
    );
    Ok(())
}

#[test]
fn named_extern_map_checks_and_uses_the_linked_ir_signature() -> Result<(), Box<dyn Error>> {
    let bx = r#"
extern fn ext::widen(x:s8)->s16 = "sext";
#[allow(combinational_block)]
block sample(clk, ext:bool, xs:s8[1])->(s16[1], s16[1], s16[1]) {
    (map(xs, ext::widen), map(xs, |x| ext::widen(x)), map(s8:-128..=s8:-128, ext::widen))
}
"#;
    let linked = Parser::new(
        r#"package test
fn sext(x: bits[8] id=1) -> bits[16] {
  ret sign_ext.2: bits[16] = sign_ext(x, new_bit_count=16, id=2)
}
"#,
    )
    .parse_and_verify_package()?;
    // A BX library triggers source qualification before type checking.
    let root = tempdir()?;
    fs::write(
        root.path().join("helpers.bx"),
        "pub fn identity(x:bool)->bool { x }",
    )?;
    let opts = ImportOptions {
        additional_search_paths: vec![root.path()],
        ..Default::default()
    };
    for source in [bx.to_owned(), format!("use helpers;\n{bx}")] {
        let package =
            compile_with_options(&source, linked.clone(), ResetBehavior::default(), &opts)?;
        let sample =
            combinational_block_to_fn(package.get_block("sample").ok_or("missing sample")?)?;
        let targets: Vec<_> = sample
            .nodes
            .iter()
            .filter_map(|node| match &node.payload {
                NodePayload::Map { to_apply, .. } => Some(to_apply.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(targets, ["sext"; 3]);
        let expected = IrValue::parse_typed("[bits[16]:65408]")?;
        assert_eq!(
            eval_fn_in_package_value(
                &package,
                &sample,
                &[IrValue::bool(false), IrValue::parse_typed("[bits[8]:128]")?]
            ),
            IrValue::make_tuple(&[expected.clone(), expected.clone(), expected])
        );
    }
    let before_map = r#"
use helpers;
extern fn ext::widen(x:s8)->s16 = "sext";
fn before(ext:bool, xs:s8[1])->s16[1] { map(xs, ext::widen) }
fn map(xs:s16[1], index:u1)->s16 { xs[index] }
#[allow(combinational_block)]
block sample(clk, xs:s8[1])->s16 { map(before(false, xs), u1:0) }
"#;
    let package =
        compile_with_options(before_map, linked.clone(), ResetBehavior::default(), &opts)?;
    let sample = combinational_block_to_fn(package.get_block("sample").ok_or("missing sample")?)?;
    assert_eq!(
        eval_fn_in_package_value(&package, &sample, &[IrValue::parse_typed("[bits[8]:128]")?]),
        IrValue::make_ubits(16, 65408)?
    );
    assert!(compile(&bx.replace("s16", "s17"), linked, ResetBehavior::default()).is_err());
    Ok(())
}

#[test]
fn ordinary_functions_named_map_keep_their_call_semantics() -> Result<(), Box<dyn Error>> {
    let bx = r#"
fn inc(x:u8)->u8 { x+u8:1 }
fn before(xs:u8[2])->u8[2] { map(xs, inc) }
fn map(xs:u8[2], index:u1)->u8 { xs[index] }
block sample(clk, xs:u8[2], index:u1)->u8 { map(before(xs), index) }
"#;
    let evaluator = BlockEvaluator::lower(&check_module(&parse_module(bx)?)?, bx)?;
    assert_eq!(
        evaluator.evaluate(&[
            IrValue::parse_typed("[bits[8]:3,bits[8]:5]")?,
            IrValue::make_ubits(1, 1)?
        ])?,
        IrValue::make_ubits(8, 6)?
    );
    Ok(())
}

#[test]
fn ordinary_block_calls_named_map_reject_builtin_operand_syntax() -> Result<(), Box<dyn Error>> {
    for (input_ty, operand) in [("u8", "{ u8:0 }"), ("()", "if true { () }")] {
        let source = format!(
            r#"
fn map(x: {input_ty}, data: u8) -> u8 {{ data }}
#[allow(combinational_block)]
block sample(clk, data: u8) -> u8 {{ map({operand}, data) }}
"#
        );
        assert!(
            compile(&source, empty_package("test"), ResetBehavior::default()).is_err(),
            "{source}"
        );
    }

    let ordinary = r#"
fn map(a: u8, data: u8) -> u8 { a + data }
block sample(clk, x: u8) -> u8 {
    map(unroll_for! (i, a) in u8:0..u8:1 { let v = a + i; v }(x) + u8:2, x)
}
"#;
    let evaluator = BlockEvaluator::lower(&check_module(&parse_module(ordinary)?)?, ordinary)?;
    assert_eq!(
        evaluator.evaluate(&[IrValue::make_ubits(8, 3)?])?,
        IrValue::make_ubits(8, 8)?
    );

    let builtin = r#"
fn inc(x: u8) -> u8 { x + u8:1 }
block sample(clk, x: u8) -> (u8[1], u8[1], u8[1]) {
    let xs = [x];
    (map({ xs }, inc), map((if true { () }, xs).1, inc),
     map(unroll_for! (i, a) in u8:0..u8:1 { (); let (b, _) = (a, i); b }(xs), inc))
}
"#;
    let evaluator = BlockEvaluator::lower(&check_module(&parse_module(builtin)?)?, builtin)?;
    assert_eq!(
        evaluator.evaluate(&[IrValue::make_ubits(8, 3)?])?,
        IrValue::parse_typed("([bits[8]:4], [bits[8]:4], [bits[8]:4])")?
    );
    Ok(())
}

#[test]
fn named_callbacks_respect_scope_even_inside_an_empty_unroll() -> Result<(), Box<dyn Error>> {
    // A call hidden in a skipped iteration still uses ordinary BX argument
    // grammar.
    for binding in ["map", "other"] {
        let operand = "unroll_for! (i,b) in u8:0..u8:1 { b }(a) + u8:1";
        let bx = format!(
            r#"
block sample(clk, {binding}:u8, x:u8)->u8 {{
    unroll_for! (_,a) in u8:0..u8:0 {{ {binding}({operand}, a) }}(x)
}}"#
        );
        check_module(&parse_module(&bx)?)?;
        assert!(parse_module(&bx.replace(operand, "{u8:1}")).is_err());
    }
    for trips in [0, 1] {
        for (callback, expected) in [
            ("unknown", "callback 'unknown'"),
            ("inc", "'inc' names a local value"),
        ] {
            let bx = format!(
                r#"
fn inc(x:u8)->u8 {{ x+u8:1 }}
block sample(clk, xs:u8[2], inc:u8)->u8[2] {{
    unroll_for! (_, a) in u8:0..u8:{trips} {{ map(a, {callback}) }}(xs)
}}
"#
            );
            let error = check_module(&parse_module(&bx)?).unwrap_err();
            assert!(error.contains(expected), "{error}");
        }
    }
    Ok(())
}

#[test]
fn named_map_arity_is_checked_when_an_unrolled_iteration_is_elaborated()
-> Result<(), Box<dyn Error>> {
    for call in ["map(a)", "map(a, inc, u8:0)"] {
        let functions = format!(
            r#"
fn inc(x:u8)->u8 {{ x+u8:1 }}
fn folded(xs:u8[2])->u8[2] {{ unroll_for! (_, a) in u8:0..u8:0 {{ {call} }}(xs) }}
"#
        );
        let bx = format!("{functions}\nblock sample(clk, xs:u8[2])->u8[2] {{ folded(xs) }}");
        let native = format!("{functions}\nfn sample(xs:u8[2])->u8[2] {{ folded(xs) }}");
        let evaluator = BlockEvaluator::lower(&check_module(&parse_module(&bx)?)?, &bx)?;
        let xs = IrValue::parse_typed("[bits[8]:3,bits[8]:5]")?;
        assert_eq!(
            evaluator.compare_ir(
                &compile_dslx_module(&native, &[])?,
                std::slice::from_ref(&xs)
            )?,
            xs
        );
        let error =
            check_module(&parse_module(&bx.replace("u8:0..u8:0", "u8:0..u8:1"))?).unwrap_err();
        assert!(
            error.contains("map expects an array and a callback"),
            "{error}"
        );
    }
    Ok(())
}

#[test]
fn named_map_rejects_an_import_requiring_an_implicit_token() -> Result<(), Box<dyn Error>> {
    let root = tempdir()?;
    fs::write(
        root.path().join("effects.x"),
        r#"
pub fn checked(x:u8)->u8 { assert!(x != u8:0, "zero"); x }
"#,
    )?;
    let bx = r#"
use effects;
#[allow(combinational_block)]
block sample(clk, xs:u8[2])->u8[2] { map(xs, effects::checked) }
"#;
    let opts = ImportOptions {
        additional_search_paths: vec![root.path()],
        ..Default::default()
    };
    let error = compile_with_options(bx, empty_package("test"), ResetBehavior::default(), &opts)
        .unwrap_err();
    assert!(
        error.contains("implicit token") && error.contains("assert!"),
        "{error}"
    );
    Ok(())
}
