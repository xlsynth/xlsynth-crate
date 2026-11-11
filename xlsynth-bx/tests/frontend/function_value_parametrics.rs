// SPDX-License-Identifier: Apache-2.0

use super::support::search_paths;
use std::{error::Error, fs};

use super::directed_expr_eval::compile_dslx_module;
use super::expr_eval::{BlockEvaluator, bx_source, dslx_source};
use tempfile::tempdir;
use xlsynth::dslx::{ImportData, parse_and_typecheck};
use xlsynth_bx::{check_module, check_source_with_options, parse_module};
use xlsynth_pir::{IrValue, ir::NodePayload};

#[test]
fn scalar_and_enum_match_patterns_compare_against_the_bound_parametric()
-> Result<(), Box<dyn Error>> {
    let declarations = r#"
enum Phase:u8 { Cold=2, Hot=7 }
fn number<N:u8>(x:u8) -> bool {
    let _ = { let N = x; N + u8:1 };
    match x { N => true, _ => false }
}
fn phase<P:Phase>(x:Phase) -> bool { match x { P => true, _ => false } }
"#;
    let body = "(number<2>(x), phase<{Phase::Hot}>(x as Phase))";
    let bx = format!("{declarations}\n{}", bx_source("x:u8", "(bool,bool)", body));
    let native = format!(
        "{declarations}\n{}",
        dslx_source("x:u8", "(bool,bool)", body)
    );
    let reference = compile_dslx_module(&native, &[])?;
    let checked = check_module(&parse_module(&bx)?)?;
    let evaluator = BlockEvaluator::lower(&checked, &bx)?;
    for (x, expected) in [(2, "(bits[1]:1,bits[1]:0)"), (7, "(bits[1]:0,bits[1]:1)")] {
        assert_eq!(
            evaluator.compare_ir(&reference, &[IrValue::make_ubits(8, x)?])?,
            IrValue::parse_typed(expected)?
        );
    }
    let shadowed = bx.replace("let _ = { let N = x; N + u8:1 };", "let N = x;");
    assert!(check_module(&parse_module(&shadowed)?).is_err());
    Ok(())
}

#[test]
fn a_parametric_binding_can_name_a_concrete_imported_dslx_struct() -> Result<(), Box<dyn Error>> {
    let directory = tempdir()?;
    fs::write(
        directory.path().join("cfg.x"),
        "pub struct Config<N:u32> { gain:uN[N] }",
    )?;
    let options = search_paths(&[directory.path()]);
    let declaration = "fn gain<C:cfg::Config<8> = {cfg::Config<8> {gain:u8:3}}>() -> u8 { C.gain }";
    let result = "(u8,u8)";
    let body = "(gain(), gain<{cfg::Config<8> {gain:u8:9}}>())";
    let bx = format!("use cfg;\n{declaration}\n{}", bx_source("", result, body));
    let native = format!(
        "import cfg;\n{declaration}\n{}",
        dslx_source("", result, body)
    );
    let checked = check_source_with_options(&bx, &options)?;
    let reference = compile_dslx_module(&native, &[directory.path()])?;
    assert_eq!(
        BlockEvaluator::lower(&checked, &bx)?.compare_ir(&reference, &[])?,
        IrValue::parse_typed("(bits[8]:3, bits[8]:9)")?
    );
    Ok(())
}

#[test]
fn nominal_parametrics_retain_enum_and_struct_identity_across_bx_imports()
-> Result<(), Box<dyn Error>> {
    let declarations = r#"
pub enum Phase:u8 { Warm=1, Hot=2 }
pub struct Config { gain:u8 }
pub fn code<P:Phase={Phase::Warm}>() -> u8 { P as u8 }
pub fn gain<C:Config={Config {gain:u8:3}}>() -> u8 { C.gain }
"#;
    let result = "(u8,u8,u8,u8)";
    let body = "(code(), code<{Phase::Hot}>(), gain(), gain<{Config {gain:u8:9}}>())";
    let native = format!("{declarations}\n{}", dslx_source("", result, body));
    let reference = compile_dslx_module(&native, &[])?;
    let expected = IrValue::parse_typed("(bits[8]:1, bits[8]:2, bits[8]:3, bits[8]:9)")?;
    let bx = format!("{declarations}\n{}", bx_source("", result, body));
    let checked = check_module(&parse_module(&bx)?)?;
    assert_eq!(
        BlockEvaluator::lower(&checked, &bx)?.compare_ir(&reference, &[])?,
        expected
    );

    let directory = tempdir()?;
    fs::write(directory.path().join("settings.bx"), declarations)?;
    let options = search_paths(&[directory.path()]);
    let imported = format!(
        "use settings::{{code,gain,Config,Phase}};\n{}",
        bx_source("", result, body)
    );
    let checked = check_source_with_options(&imported, &options)?;
    assert_eq!(
        BlockEvaluator::lower(&checked, &imported)?.compare_ir(&reference, &[])?,
        expected
    );

    let other_types = "enum OtherPhase:u8 { Hot=2 } struct OtherConfig { gain:u8 }";
    for wrong_actual in [
        "code<{OtherPhase::Hot}>()",
        "gain<{OtherConfig {gain:u8:9}}>()",
    ] {
        let bx = format!(
            "use settings::{{code,gain}};\n{other_types}\n{}",
            bx_source("", "u8", wrong_actual)
        );
        assert!(
            check_source_with_options(&bx, &options).is_err(),
            "{wrong_actual}"
        );
        let native = format!(
            "{declarations}\n{other_types}\n{}",
            dslx_source("", "u8", wrong_actual)
        );
        assert!(compile_dslx_module(&native, &[]).is_err(), "{wrong_actual}");
    }
    Ok(())
}

#[test]
fn identical_values_reuse_a_function_with_a_literal_body() -> Result<(), Box<dyn Error>> {
    let declaration = "fn tag<K:u8>() -> u8 { K }";
    let body = "(tag<3>(), tag<5>(), tag<u8:3>())";
    let result = "(u8, u8, u8)";
    let bx = format!("{declaration}\n{}", bx_source("", result, body));
    let native = format!("{declaration}\n{}", dslx_source("", result, body));
    let checked = check_module(&parse_module(&bx)?)?;
    assert_eq!(checked.functions().len(), 2);
    assert!(checked.functions().iter().all(|f| f.params.is_empty()));
    let evaluator = BlockEvaluator::lower(&checked, &bx)?;
    let targets = evaluator
        .function
        .nodes
        .iter()
        .filter_map(|n| match &n.payload {
            NodePayload::Invoke { to_apply, .. } => Some(to_apply),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(targets.len(), 3);
    assert_ne!(targets[0], targets[1]);
    assert_eq!(targets[0], targets[2]);
    assert_eq!(
        evaluator.compare_ir(&compile_dslx_module(&native, &[])?, &[])?,
        IrValue::parse_typed("(bits[8]:3, bits[8]:5, bits[8]:3)")?
    );
    Ok(())
}

#[test]
fn computed_actual_uses_caller_scope_while_default_and_body_use_definition_scope()
-> Result<(), Box<dyn Error>> {
    let declarations = r#"
const STEP = u8:2;
fn bump(x:u8) -> u8 { x + u8:1 }
fn combine<A:u8, B:u8={STEP}>(x:u8) -> (u8,u8,u8) { (A, B, x + STEP) }
fn wrapper(x:u8) -> (u8,u8,u8) {
    let STEP = u8:7;
    combine<{bump(STEP)}>(x)
}
"#;
    let bx = format!(
        "{declarations}\n{}",
        bx_source("x:u8", "(u8,u8,u8)", "wrapper(x)")
    );
    let native = format!(
        "{declarations}\n{}",
        dslx_source("x:u8", "(u8,u8,u8)", "wrapper(x)")
    );
    let checked = check_module(&parse_module(&bx)?)?;
    let evaluator = BlockEvaluator::lower(&checked, &bx)?;
    assert_eq!(
        evaluator.compare_ir(
            &compile_dslx_module(&native, &[])?,
            &[IrValue::make_ubits(8, 10)?]
        )?,
        IrValue::parse_typed("(bits[8]:8, bits[8]:2, bits[8]:12)")?
    );
    Ok(())
}

#[test]
fn a_module_constant_can_call_nested_value_specializations() -> Result<(), Box<dyn Error>> {
    let declarations = r#"
fn add<K:u8>(x:u8) -> u8 { x + K }
fn adjust<K:u8>(x:u8) -> u8 { add<K>(add<1>(x)) }
const ANSWER = adjust<2>(u8:39);
"#;
    let bx = format!("{declarations}\n{}", bx_source("", "u8", "ANSWER"));
    let native = format!("{declarations}\n{}", dslx_source("", "u8", "ANSWER"));
    let checked = check_module(&parse_module(&bx)?)?;
    let expected = IrValue::make_ubits(8, 42)?;
    assert_eq!(checked.constants()[0].value, expected);
    let evaluator = BlockEvaluator::lower(&checked, &bx)?;
    assert_eq!(
        evaluator.compare_ir(&compile_dslx_module(&native, &[])?, &[])?,
        expected
    );
    Ok(())
}

#[test]
fn an_explicit_value_typechecks_the_default_with_the_binding_in_scope() -> Result<(), Box<dyn Error>>
{
    let bad = "fn tag<K:u8={u16:1}>() -> u8 { K } fn sample() -> u8 { tag<7>() }";
    assert!(check_module(&parse_module(bad)?).is_err());
    assert!(compile_dslx_module(bad, &[]).is_err());

    let declaration = "fn tag<K:u8={K}>() -> u8 { K }";
    let bx = format!("{declaration}\n{}", bx_source("", "u8", "tag<7>()"));
    let native = format!("{declaration}\n{}", dslx_source("", "u8", "tag<7>()"));
    let checked = check_module(&parse_module(&bx)?)?;
    let evaluator = BlockEvaluator::lower(&checked, &bx)?;
    assert_eq!(
        evaluator.compare_ir(&compile_dslx_module(&native, &[])?, &[])?,
        IrValue::make_ubits(8, 7)?
    );
    Ok(())
}

#[test]
fn array_literal_inference_uses_evidence_from_earlier_arguments() -> Result<(), Box<dyn Error>> {
    let declarations = r#"
fn size<M:u32>(xs:u8[M])->u32 { M }
fn leading<N:u32>(x:uN[N], xs:uN[N][1])->u32 { N }
fn trailing<N:u32>(xs:uN[N][1], x:uN[N])->u32 { N }
"#;
    let empty = format!(
        "{declarations}\nfn sample()->(u32,u32) {{ let e:u8[0]=[]; (size<0>([]), size(e)) }}"
    );
    check_module(&parse_module(&empty)?)?;
    parse_and_typecheck(&empty, "sample.x", "sample", &mut ImportData::default())?;

    let body = "leading(u8:3,[1])";
    let bx = format!("{declarations}\n{}", bx_source("", "u32", body));
    let native = format!("{declarations}\n{}", dslx_source("", "u32", body));
    let checked = check_module(&parse_module(&bx)?)?;
    assert_eq!(
        BlockEvaluator::lower(&checked, &bx)?
            .compare_ir(&compile_dslx_module(&native, &[])?, &[])?,
        IrValue::make_ubits(32, 8)?
    );

    for call in ["size([])", "trailing([1],u8:3)"] {
        let source = format!("{declarations}\nfn sample()->u32 {{ {call} }}");
        assert!(check_module(&parse_module(&source)?).is_err(), "{source}");
        assert!(
            parse_and_typecheck(&source, "sample.x", "sample", &mut ImportData::default()).is_err(),
            "{source}"
        );
    }
    Ok(())
}

#[test]
fn helper_calls_can_compute_slice_widths_and_loop_bounds() -> Result<(), Box<dyn Error>> {
    let declarations = r#"
fn successor(n:u32)->u32 { n + u32:1 }
fn sample_width<N:u32>(x:uN[N])->(uN[N],u32) {
    let slice = x[0+:uN[successor(N)-u32:1]];
    let sum = for (i,acc) in u32:0..successor(N) { acc+i }(u32:0);
    (slice,sum)
}
"#;
    let bx = format!(
        "{declarations}\n{}",
        bx_source("x:u8", "(u8,u32)", "sample_width(x)")
    );
    let native = format!(
        "{declarations}\n{}",
        dslx_source("x:u8", "(u8,u32)", "sample_width(x)")
    );
    let checked = check_module(&parse_module(&bx)?)?;
    assert_eq!(
        BlockEvaluator::lower(&checked, &bx)?.compare_ir(
            &compile_dslx_module(&native, &[])?,
            &[IrValue::make_ubits(8, 127)?]
        )?,
        IrValue::parse_typed("(bits[8]:127, bits[32]:36)")?
    );
    Ok(())
}

#[test]
fn a_callback_template_can_compute_an_actual_with_a_value_parametric_helper()
-> Result<(), Box<dyn Error>> {
    let helpers = r#"
fn value<K:u8>() -> u8 { K }
fn add<K:u8>(x:u8) -> u8 { x+K }
fn identity(x:u8) -> u8 { x }
"#;
    let bx = format!(
        "{helpers}\nfn wrap<F:fn(u8)->u8>(x:u8)->u8 {{ add<{{value<3>()}}>(F(x)) }}\n{}",
        bx_source("x:u8", "u8", "wrap<identity>(x)")
    );
    let native = format!(
        "{helpers}\n{}",
        dslx_source("x:u8", "u8", "add<{value<3>()}>(identity(x))")
    );
    let checked = check_module(&parse_module(&bx)?)?;
    assert_eq!(
        BlockEvaluator::lower(&checked, &bx)?.compare_ir(
            &compile_dslx_module(&native, &[])?,
            &[IrValue::make_ubits(8, 39)?]
        )?,
        IrValue::make_ubits(8, 42)?
    );
    Ok(())
}

#[test]
fn parametric_matches_require_constant_inputs_in_every_arm() -> Result<(), Box<dyn Error>> {
    let declaration = "fn tag<N:u32>()->u32 { N }";
    for (body, accepted) in [
        ("tag<{match true { true=>u32:2, _=>x }}>()", false),
        (
            "let k=match true { true=>u32:2, _=>x }; let alias=k; tag<alias>()",
            false,
        ),
        (
            "tag<{match true { true=>u32:2, _=>if true {u32:2} else {x} }}>()",
            false,
        ),
        (
            "tag<{if true {u32:2} else {match true { true=>u32:2, _=>x }}}>()",
            true,
        ),
        (
            "let k=if true {u32:2} else {x}; tag<{match true { true=>u32:2, _=>k }}>()",
            true,
        ),
    ] {
        let source = format!("{declaration} fn sample(x:u32)->u32 {{ {body} }}");
        let native = parse_and_typecheck(&source, "sample.x", "sample", &mut ImportData::default());
        assert_eq!(native.is_ok(), accepted, "{source}");
        let checked = check_module(&parse_module(&source)?);
        if accepted {
            checked?;
        } else {
            assert!(
                checked
                    .unwrap_err()
                    .contains("requires a compile-time constant"),
                "{source}"
            );
        }
    }

    // Ordinary BX folding can still supply a reset from the selected match arm.
    let bx = format!(
        "{declaration} block b(clk,rst,x:u32)->u32 {{ let k=match true {{ true=>u32:2, _=>x }}; reg q:u32=k; q:=x; q }}"
    );
    let checked = check_module(&parse_module(&bx)?)?;
    let expected = IrValue::make_ubits(32, 2)?;
    assert!(checked.blocks()[0].statements.iter().any(|stmt| {
        matches!(stmt, xlsynth_bx::check::CheckedStmt::Reg { reset: Some(value), .. } if value == &expected)
    }));
    Ok(())
}

#[test]
fn a_bound_conditional_can_supply_a_constexpr_helper_result() -> Result<(), Box<dyn Error>> {
    let declarations = r#"
fn tag<N:u32>()->u32 { N }
fn successor(x:u32)->u32 { x+u32:1 }
fn bound(x:u32)->u32 {
    let a = if true { successor(u32:1) } else { x };
    let b = a;
    tag<b>()
}
"#;
    let bx = format!("{declarations}\n{}", bx_source("x:u32", "u32", "bound(x)"));
    let native = format!(
        "{declarations}\n{}",
        dslx_source("x:u32", "u32", "bound(x)")
    );
    let checked = check_module(&parse_module(&bx)?)?;
    assert_eq!(
        BlockEvaluator::lower(&checked, &bx)?.compare_ir(
            &compile_dslx_module(&native, &[])?,
            &[IrValue::make_ubits(32, 19)?],
        )?,
        IrValue::make_ubits(32, 2)?,
    );
    Ok(())
}

#[test]
fn a_constexpr_index_must_be_in_bounds_when_used_as_a_parametric() -> Result<(), Box<dyn Error>> {
    let declarations = "fn tag<N:u32>()->u32 { N }";
    for (index, accepted) in [(1, true), (2, false)] {
        let body =
            format!("let a=if true {{u32:{index}}} else {{x}}; let k=[u32:3,u32:4][a]; tag<k>()");
        let source = format!("{declarations}\nfn sample(x:u32)->u32 {{ {body} }}");
        let bx = check_module(&parse_module(&source)?);
        let native = parse_and_typecheck(&source, "sample.x", "sample", &mut ImportData::default());
        assert_eq!(bx.is_ok(), accepted, "{source}");
        assert_eq!(native.is_ok(), accepted, "{source}");
    }

    let body = "[u32:3,u32:4][x]";
    let bx = format!("{declarations}\n{}", bx_source("x:u32", "u32", body));
    let native = format!("{declarations}\n{}", dslx_source("x:u32", "u32", body));
    let checked = check_module(&parse_module(&bx)?)?;
    assert_eq!(
        BlockEvaluator::lower(&checked, &bx)?.compare_ir(
            &compile_dslx_module(&native, &[])?,
            &[IrValue::make_ubits(32, 2)?],
        )?,
        IrValue::make_ubits(32, 4)?,
    );
    Ok(())
}
