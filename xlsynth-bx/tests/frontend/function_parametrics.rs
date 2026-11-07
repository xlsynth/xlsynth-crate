// SPDX-License-Identifier: Apache-2.0

use super::directed_expr_eval::compile_dslx_module;
use super::expr_eval::{BlockEvaluator, bx_source};
use std::{error::Error, fs};
use tempfile::tempdir;
use xlsynth_bx::{
    ImportOptions, ResetBehavior, check_module, check_module_with_options, empty_package,
    lower_module, parse_module,
};
use xlsynth_pir::IrValue;
use xlsynth_pir::block2fn::combinational_block_to_fn;
use xlsynth_pir::ir::NodePayload;
use xlsynth_pir::ir_eval::eval_fn_in_package_value;
use xlsynth_pir::ir_parser::Parser;

/// Inspect the concrete call targets without depending on their generated
/// names.
fn invoke_targets(evaluator: &BlockEvaluator) -> Vec<&str> {
    evaluator
        .function
        .nodes
        .iter()
        .filter_map(|node| match &node.payload {
            NodePayload::Invoke { to_apply, .. } => Some(to_apply.as_str()),
            _ => None,
        })
        .collect()
}

#[test]
fn counted_fold_calls_the_selected_helper_at_runtime() -> Result<(), Box<dyn Error>> {
    let bx = r#"
fn fold4<F: fn(u16, u16) -> u16>(xs: u16[4], seed: u16) -> u16 {
    for (x, acc) in xs { F(acc, x) }(seed)
}
fn sub(a: u16, b: u16) -> u16 { a - b }
fn step(a: u16, b: u16) -> u16 { b - a }
fn with_sub(xs: u16[4], seed: u16) -> u16 { fold4<sub>(xs, seed) }
block sample(clk, xs: u16[4], seed: u16) -> (u16,u16) {
    (with_sub(xs, seed), fold4<step>(xs, seed))
}"#;
    let native = r#"
fn sample(xs: u16[4], seed: u16) -> (u16,u16) {
    (for (x,a) in xs { a-x }(seed), for (x,a) in xs { x-a }(seed))
}"#;
    let checked = check_module(&parse_module(bx)?)?;
    let evaluator = BlockEvaluator::lower(&checked, bx)?;
    let ir = compile_dslx_module(native, &[])?;
    for (seed, expected) in [
        (0, "(bits[16]:65522,bits[16]:6)"),
        (20, "(bits[16]:6,bits[16]:26)"),
    ] {
        assert_eq!(
            evaluator.compare_ir(
                &ir,
                &[
                    IrValue::parse_typed("[bits[16]:1,bits[16]:3,bits[16]:3,bits[16]:7]")?,
                    IrValue::make_ubits(16, seed)?,
                ]
            )?,
            IrValue::parse_typed(expected)?
        );
    }
    Ok(())
}

#[test]
fn ordered_callback_identities_distinguish_and_reuse_specializations() -> Result<(), Box<dyn Error>>
{
    let bx = r#"
fn compose<F: fn(u8) -> u8, G: fn(u8) -> u8>(x: u8) -> u8 { F(G(x)) }
fn inc(x: u8) -> u8 { x + u8:1 }
fn twice(x: u8) -> u8 { x + x }
block sample(clk, x:u8) -> (u8,u8,u8) {
    (compose<inc,twice>(x), compose<twice,inc>(x), compose<inc,twice>(x + u8:1))
}"#;
    let native = r#"
fn inc(x: u8) -> u8 { x + u8:1 }
fn twice(x: u8) -> u8 { x + x }
fn sample(x:u8) -> (u8,u8,u8) { (inc(twice(x)), twice(inc(x)), inc(twice(x+u8:1))) }"#;
    let checked = check_module(&parse_module(bx)?)?;
    let evaluator = BlockEvaluator::lower(&checked, bx)?;
    let targets = invoke_targets(&evaluator);
    assert_eq!(targets.len(), 3);
    assert_ne!(targets[0], targets[1]);
    assert_eq!(targets[0], targets[2]);
    let ir = compile_dslx_module(native, &[])?;
    assert_eq!(
        evaluator.compare_ir(&ir, &[IrValue::make_ubits(8, 128)?])?,
        IrValue::parse_typed("(bits[8]:1,bits[8]:2,bits[8]:3)")?
    );
    Ok(())
}

#[test]
fn different_templates_can_bind_the_same_nullary_helper() -> Result<(), Box<dyn Error>> {
    let bx = r#"
fn one<F: fn() -> u8>() -> u8 { F() }
fn two<F: fn() -> u8>() -> u8 { F() + F() }
fn three() -> u8 { u8:3 }
block sample(clk) -> (u8, u8) { (one<three>(), two<three>()) }
"#;
    let checked = check_module(&parse_module(bx)?)?;
    let evaluator = BlockEvaluator::lower(&checked, bx)?;
    let targets = invoke_targets(&evaluator);
    assert_eq!(targets.len(), 2);
    assert_ne!(targets[0], targets[1]);
    assert_eq!(
        evaluator.evaluate(&[])?,
        IrValue::parse_typed("(bits[8]:3,bits[8]:6)")?
    );
    Ok(())
}

#[test]
fn specialization_calls_survive_occupied_destination_symbols() -> Result<(), Box<dyn Error>> {
    let bx = r#"
fn apply<F: fn(u8) -> u8>(x:u8) -> u8 { F(x) }
fn step(x:u8) -> u8 { x+u8:1 }
block sample(clk, x:u8) -> u8 { apply<step>(x) }
"#;
    let checked = check_module(&parse_module(bx)?)?;
    let spec = checked
        .functions()
        .iter()
        .find(|f| f.name != "step")
        .ok_or("missing specialization")?;
    let occupied = format!(
        "package linked\nfn {}(x: bits[8] id=1) -> bits[8] {{ ret literal.2: bits[8] = literal(value=0, id=2) }}\n",
        spec.name
    );
    let linked = Parser::new(&occupied).parse_and_verify_package()?;
    let package = lower_module(&checked, linked, ResetBehavior::default())?;
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    let sample =
        combinational_block_to_fn(package.get_block("sample").ok_or("missing sample block")?)?;
    let args = [IrValue::make_ubits(8, 5)?];
    assert_eq!(
        eval_fn_in_package_value(&package, &sample, &args),
        IrValue::make_ubits(8, 6)?
    );
    assert_eq!(
        eval_fn_in_package_value(
            &package,
            package
                .get_fn(&spec.name)
                .ok_or("missing prelinked function")?,
            &args
        ),
        IrValue::make_ubits(8, 0)?
    );
    Ok(())
}

#[test]
fn specialized_callbacks_are_static_inside_map_and_generic_blocks() -> Result<(), Box<dyn Error>> {
    let bx = r#"
fn transform<F: fn((s8, bool)) -> s16>(xs:(s8,bool)[2]) -> s16[2] {
    map(xs, |x| F(x))
}
fn choose(p:(s8,bool)) -> s16 { if p.1 { p.0 as s16 } else { s16:0 } }
block stage<T>(clk, xs:T[2]) -> s16[2] { transform<choose>(xs) }
block sample(clk, xs:(s8,bool)[2]) -> s16[2] { let s = stage(xs); s }
"#;
    let checked = check_module(&parse_module(bx)?)?;
    let package = lower_module(&checked, empty_package("generic"), ResetBehavior::default())?;
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    let stage = checked
        .blocks()
        .iter()
        .find(|b| b.name != "sample")
        .ok_or("missing generic block")?;
    let stage_ir = package
        .get_block(&stage.name)
        .ok_or("missing lowered generic block")?;
    let projected = combinational_block_to_fn(stage_ir)?;
    let args = [IrValue::parse_typed(
        "[(bits[8]:255,bits[1]:1),(bits[8]:3,bits[1]:0)]",
    )?];
    assert_eq!(
        eval_fn_in_package_value(&package, &projected, &args),
        IrValue::parse_typed("[bits[16]:65535,bits[16]:0]")?
    );
    Ok(())
}

#[test]
fn imported_callbacks_use_canonical_module_and_concrete_dslx_types() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    fs::write(
        dir.path().join("math.x"),
        r#"
pub fn combine<N:u32>(x:uN[N], y:uN[N]) -> uN[N] { x-y }
fn private(x:u8, y:u8) -> u8 { x+y }
pub fn signed_result(x:u8, y:u8) -> s8 { (x+y) as s8 }
"#,
    )?;
    fs::write(
        dir.path().join("other.x"),
        "pub fn combine(x:u8, y:u8) -> u8 { x+y }",
    )?;
    let options = ImportOptions {
        additional_search_paths: vec![dir.path()],
        ..ImportOptions::default()
    };
    let bx = r#"
fn apply<F: fn(u8,u8) -> u8>(x:u8, y:u8) -> u8 { F(x,y) }
use math;
use math::combine;
use other;
block sample(clk, x:u8, y:u8) -> (u8,u8,u8) {
    (apply<math::combine>(x,y), apply<combine>(x,y), apply<other::combine>(x,y))
}"#;
    let native = r#"
import math;
import other;
fn sample(x:u8, y:u8) -> (u8,u8,u8) { (math::combine(x,y), math::combine(x,y), other::combine(x,y)) }
"#;
    let checked = check_module_with_options(&parse_module(bx)?, &options)?;
    let evaluator = BlockEvaluator::lower(&checked, bx)?;
    let targets = invoke_targets(&evaluator);
    assert_eq!(targets.len(), 3);
    assert_eq!(targets[0], targets[1]);
    assert_ne!(targets[0], targets[2]);
    let ir = compile_dslx_module(native, &[dir.path()])?;
    assert_eq!(
        evaluator.compare_ir(
            &ir,
            &[IrValue::make_ubits(8, 2)?, IrValue::make_ubits(8, 7)?]
        )?,
        IrValue::parse_typed("(bits[8]:251,bits[8]:251,bits[8]:9)")?
    );
    for actual in ["math::private", "math::missing", "math::signed_result"] {
        let invalid = bx.replace("apply<math::combine>", &format!("apply<{actual}>"));
        assert!(
            check_module_with_options(&parse_module(&invalid)?, &options).is_err(),
            "{invalid}"
        );
    }
    Ok(())
}

#[test]
fn imported_callbacks_require_function_declarations_even_in_empty_loops()
-> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    fs::write(
        dir.path().join("provider.x"),
        r#"
pub fn identity(x: u8) -> u8 { x }
pub const ALIAS = identity;
pub struct Payload { x: u8 }
pub fn sink(_x: u8) { () }
pub fn checked(x: u8) -> u8 { if x == u8:0 { fail!("zero", x) } else { x } }
"#,
    )?;
    let options = ImportOptions {
        additional_search_paths: vec![dir.path()],
        ..ImportOptions::default()
    };
    for trips in [0, 1] {
        for member in ["identity", "ALIAS", "Payload"] {
            let source = format!(
                r#"
use provider;
fn apply<F: fn(u8) -> u8>(x: u8) -> u8 {{ F(x) }}
block sample(clk, x: u8) -> u8 {{
    unroll_for! (_, acc) in u32:0..u32:{trips} {{ apply<provider::{member}>(acc) }}(x)
}}
"#
            );
            let checked = check_module_with_options(&parse_module(&source)?, &options);
            if member == "identity" {
                checked?;
            } else {
                let error = checked.unwrap_err();
                assert!(
                    error.contains(&format!(
                        "callback 'provider::{member}' requires a public DSLX function declaration"
                    )),
                    "{error}"
                );
            }
        }
    }
    let unit = r#"
use provider::sink;
fn consume<F: fn(u8)>(x: u8) -> u8 { F(x); x }
block sample(clk, x: u8) -> u8 { consume<sink>(x) }
"#;
    check_module_with_options(&parse_module(unit)?, &options)?;
    let effectful = r#"
use provider;
fn apply<F: fn(u8) -> u8>(x: u8) -> u8 { F(x) }
block sample(clk, x: u8) -> u8 { apply<provider::checked>(x) }
"#;
    let error = check_module_with_options(&parse_module(effectful)?, &options).unwrap_err();
    assert!(
        error.contains("implicit token") || error.contains("effectful"),
        "{error}"
    );
    Ok(())
}

#[test]
fn actual_and_template_names_resolve_in_their_own_lexical_scopes() -> Result<(), Box<dyn Error>> {
    let bx = r#"
fn adjust(x:u8) -> u8 { x + u8:2 }
fn apply<F: fn(u8) -> u8>(x:u8) -> u8 { F(adjust(x)) }
fn callback(x:u8) -> u8 { x*u8:3 }
fn caller(adjust:u8) -> u8 { apply<callback>(adjust) }
block sample(clk, x:u8) -> u8 { caller(x) }"#;
    let checked = check_module(&parse_module(bx)?)?;
    let evaluator = BlockEvaluator::lower(&checked, bx)?;
    assert_eq!(
        evaluator.evaluate(&[IrValue::make_ubits(8, 4)?])?,
        IrValue::make_ubits(8, 18)?
    );
    for invalid in [
        bx.replace("fn callback(x:u8) -> u8 { x*u8:3 }", "")
            + "\nfn callback(x:u8) -> u8 { x*u8:3 }",
        bx.replace("fn adjust(x:u8) -> u8 { x + u8:2 }", "")
            + "\nfn adjust(x:u8) -> u8 { x + u8:2 }",
        bx.replace(
            "apply<callback>(adjust)",
            "{ let callback = adjust; apply<callback>(adjust) }",
        ),
        bx.replace("apply<callback>(adjust)", "apply<adjust>(adjust)"),
    ] {
        assert!(check_module(&parse_module(&invalid)?).is_err(), "{invalid}");
    }
    let scoped = bx.replace(
        "apply<callback>(adjust)",
        "match adjust { u8:0 => { let callback=u8:1; apply<callback>(adjust) }, _ => adjust }",
    );
    let error = check_module(&parse_module(&scoped)?).unwrap_err();
    assert!(error.contains("'callback'"), "{error}");
    Ok(())
}

#[test]
fn callback_formals_require_named_targets_and_matching_signatures() {
    let defs = r#"
fn apply<F: fn(u8) -> u8>(x:u8) -> u8 { F(x) }
fn add(x:u8,y:u8) -> u8 { x+y }
fn wide(x:u8) -> u16 { x as u16 }
fn signed(x:s8) -> u8 { x as u8 }
fn inc(x:u8) -> u8 { x+u8:1 }
block child(clk, x:u8) -> u8 { x }"#;
    for expression in [
        "apply<add>(x)",
        "apply<wide>(x)",
        "apply<signed>(x)",
        "apply<child>(x)",
        "apply<apply>(x)",
        "apply<missing>(x)",
        "apply<x>(x)",
        "apply<inc,inc>(x)",
        "apply(x)",
        "inc<inc>(x)",
        "apply<|v| v + u8:1>(x)",
        "apply<(inc)>(x)",
        "let alias = inc; apply<alias>(x)",
    ] {
        let bx = format!("{defs}\n{}", bx_source("x:u8", "u8", expression));
        assert!(
            parse_module(&bx).and_then(|m| check_module(&m)).is_err(),
            "BX accepted {bx}"
        );
    }
    for bad in ["F", "let alias = F; alias(x)", "apply<F>(x)", "F(x,x)"] {
        let bx = format!(
            "{defs}\nfn unused<F: fn(u8) -> u8>(x:u8) -> u8 {{ {bad} }}\n{}",
            bx_source("x:u8", "u8", "x")
        );
        assert!(
            parse_module(&bx).and_then(|m| check_module(&m)).is_err(),
            "BX accepted {bx}"
        );
    }
}

#[test]
fn repeated_calls_reuse_wide_callback_signature_checks() -> Result<(), Box<dyn Error>> {
    // Comparing the same 1,024-field signature on every iteration would
    // consume the expansion budget even though the fold carries only a u8.
    let fields = vec!["u1"; 1024].join(",");
    let source = format!(
        r#"
fn head(x: ({fields})) -> u1 {{ x.0 }}
fn keep<F: fn(({fields})) -> u1>(x: u8) -> u8 {{ x }}
block sample(clk, x: u8) -> u8 {{
    unroll_for! (_, acc) in u32:0..u32:1024 {{ keep<head>(acc) }}(x)
}}
"#
    );
    let checked = check_module(&parse_module(&source)?)?;
    assert_eq!(checked.functions().len(), 2);
    Ok(())
}

#[test]
fn cached_callbacks_still_check_each_formal_and_caller_scope() -> Result<(), Box<dyn Error>> {
    let declarations = r#"
fn inc(x: u8) -> u8 { x + u8:1 }
fn apply<F: fn(u8) -> u8>(x: u8) -> u8 { F(x) }
fn signed<F: fn(s8) -> u8>(x: u8) -> u8 { F(x as s8) }
fn pair<F: fn(u8) -> u8, G: fn(s8) -> u8>(x: u8) -> u8 { F(x) + G(x as s8) }
"#;
    for (call, reason) in [
        ("signed<inc>(x)", "expected (s8)"),
        ("pair<inc,inc>(x)", "expected (s8)"),
        (
            "{ let inc = x; apply<inc>(x) }",
            "'inc' names a local value",
        ),
    ] {
        let source =
            format!("{declarations}\nfn caller(x:u8)->u8 {{ let a = apply<inc>(x); {call} }}");
        let error = check_module(&parse_module(&source)?).unwrap_err();
        assert!(error.contains(reason), "{error}");
    }
    Ok(())
}

#[test]
fn unroll_wildcards_leave_named_callbacks_visible() -> Result<(), Box<dyn Error>> {
    for trips in [0, 1] {
        let bx = format!(
            r#"
fn _<F:fn(u8)->u8, G:fn(u8)->u8>(x:u8)->u8 {{ F(G(x)) }}
fn inc(x:u8)->u8 {{ x + u8:1 }}
block sample(clk, x:u8)->u8 {{
    unroll_for! (_, _) in u32:0..u32:{trips} {{
        let _ = x;
        _<inc,inc>(x)
    }}(x)
}}"#
        );
        let checked = check_module(&parse_module(&bx)?)?;
        let evaluator = BlockEvaluator::lower(&checked, &bx)?;
        assert_eq!(
            evaluator.evaluate(&[IrValue::make_ubits(8, 3)?])?,
            IrValue::make_ubits(8, 3 + trips * 2)?
        );
    }
    Ok(())
}

#[test]
fn map_and_counted_wildcards_leave_named_callback_actuals_visible() -> Result<(), Box<dyn Error>> {
    let bx = r#"
fn _(x:u8)->u8 { x + u8:1 }
fn apply<F:fn(u8)->u8>(x:u8)->u8 { F(x) }
block sample(clk, xs:u8[2])->u8[2] {
    map(xs, |_| for ((_, _), _) in [(u8:1, u8:2)] { apply<_>(u8:1) }(u8:0))
}"#;
    let checked = check_module(&parse_module(bx)?)?;
    let evaluator = BlockEvaluator::lower(&checked, bx)?;
    assert_eq!(
        evaluator.evaluate(&[IrValue::parse_typed("[bits[8]:9,bits[8]:4]")?])?,
        IrValue::parse_typed("[bits[8]:2,bits[8]:2]")?
    );
    Ok(())
}

#[test]
fn block_type_parameters_shadow_templates_and_callback_actuals() -> Result<(), Box<dyn Error>> {
    for name in ["apply", "inc"] {
        let bx = format!(
            r#"
fn apply<F:fn(u8)->u8>(x:u8)->u8 {{ F(x) }}
fn inc(x:u8)->u8 {{ x+u8:1 }}
block wrapper<{name}>(clk, payload:{name}, x:u8)->u8 {{ apply<inc>(x) }}
block sample(clk, x:u8)->u8 {{ let y=wrapper(x,x); y }}
"#
        );
        let error = check_module(&parse_module(&bx)?).unwrap_err();
        assert!(
            error.contains(&format!("'{name}' names a type parameter")),
            "{error}"
        );
    }
    Ok(())
}

#[test]
fn recursion_through_a_bound_function_is_rejected() -> Result<(), Box<dyn Error>> {
    let bx = r#"
fn apply<F: fn(u8) -> u8>(x:u8) -> u8 { F(x) }
fn callback(x:u8) -> u8 { apply<callback>(x) }
block sample(clk, x:u8) -> u8 { callback(x) }"#;
    let error = check_module(&parse_module(bx)?).unwrap_err();
    assert!(
        error.contains("recurs") || error.contains("cycle"),
        "{error}"
    );
    Ok(())
}
