// SPDX-License-Identifier: Apache-2.0

use super::directed_expr_eval::compile_dslx_module;
use super::expr_eval::BlockEvaluator;
use std::{error::Error, fs};
use tempfile::tempdir;
use xlsynth_bx::{
    ImportOptions, ResetBehavior, check_module, compile, compile_with_options, empty_package,
    parse_module,
};
use xlsynth_pir::IrValue;
use xlsynth_pir::block2fn::combinational_block_to_fn;
use xlsynth_pir::ir::NodePayload;
use xlsynth_pir::ir_eval::eval_fn_in_package_value;
use xlsynth_pir::ir_parser::Parser;

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
