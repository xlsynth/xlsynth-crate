// SPDX-License-Identifier: Apache-2.0

use super::directed_expr_eval::compile_dslx_module;
use super::expr_eval::{BlockEvaluator, bx_source, dslx_source};
use super::function_parametrics::invoke_targets;
use super::support::search_paths;
use std::{error::Error, fs};
use tempfile::tempdir;
use xlsynth_bx::ast::Type;
use xlsynth_bx::{
    ResetBehavior, check_module, check_source_with_options, lower_module, parse_module,
};
use xlsynth_pir::IrValue;
use xlsynth_pir::block2fn::combinational_block_to_fn;
use xlsynth_pir::ir::NodePayload;
use xlsynth_pir::ir_eval::eval_fn_in_package_value;
use xlsynth_pir::ir_parser::Parser;

const ZIP_WITH: &str = r#"
pub fn zip_with<A, B, R, F: fn(A, B) -> R, N: u32>(xs: A[N], ys: B[N]) -> R[N] {
    for (i, acc) in u32:0..N {
        update(acc, i, F(xs[i], ys[i]))
    }(zero!<R[N]>())
}
"#;

#[test]
fn library_zip_with_infers_record_inputs_result_and_length() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    fs::write(dir.path().join("arrays.bx"), ZIP_WITH)?;
    let records = r#"
struct Reading { valid: bool, value: s8 }
struct Bias { amount: u4 }
struct Adjusted { value: s16, enabled: bool }
fn adjust(x: Reading, b: Bias) -> Adjusted {
    Adjusted { value: (x.value as s16) + (b.amount as s16), enabled: x.valid }
}
"#;
    let params = "xs: Reading[3], ys: Bias[3]";
    let result = "(Adjusted[3], Adjusted[3])";
    let bx = format!(
        "use arrays; use arrays::zip_with;\n{records}\n{}",
        bx_source(
            params,
            result,
            "(arrays::zip_with<adjust>(xs,ys), zip_with<adjust,3>(xs,ys))"
        )
    );
    let options = search_paths(&[dir.path()]);
    let checked = check_source_with_options(&bx, &options)?;
    let evaluator = BlockEvaluator::lower(&checked, &bx)?;
    let targets = invoke_targets(&evaluator);
    assert_eq!(targets.len(), 2);
    assert_eq!(targets[0], targets[1]);
    let native = format!(
        "{records}\n{}",
        dslx_source(
            params,
            result,
            "let adjusted = [adjust(xs[0],ys[0]), adjust(xs[1],ys[1]), adjust(xs[2],ys[2])]; (adjusted,adjusted)"
        )
    );
    let xs = IrValue::parse_typed(
        "[(bits[1]:1,bits[8]:-4),(bits[1]:0,bits[8]:127),(bits[1]:1,bits[8]:-128)]",
    )?;
    let ys = IrValue::parse_typed("[(bits[4]:3),(bits[4]:15),(bits[4]:0)]")?;
    let out = IrValue::parse_typed(
        "[(bits[16]:-1,bits[1]:1),(bits[16]:142,bits[1]:0),(bits[16]:-128,bits[1]:1)]",
    )?;
    assert_eq!(
        evaluator.compare_ir(&compile_dslx_module(&native, &[])?, &[xs, ys])?,
        IrValue::make_tuple(&[out.clone(), out])
    );
    Ok(())
}

#[test]
fn inferred_identity_preserves_nominal_types_in_direct_calls_and_maps() -> Result<(), Box<dyn Error>>
{
    let records = "struct Left { v:u8 } struct Right { v:u8 }";
    let params = "x:Left, y:Right, z:s8";
    let result = "(Left, Right, Left, s8, u8, Right[2])";
    let body = "(id(x), id(y), id(x), id(z), id(x.v), map([y,y],id))";
    let bx = format!(
        "{records} fn id<T>(x:T)->T {{ x }} {}",
        bx_source(params, result, body)
    );
    let native = format!(
        "#![feature(generics)]\n{records} fn id<T:type>(x:T)->T {{ x }} {}",
        dslx_source(params, result, body)
    );
    let checked = check_module(&parse_module(&bx)?)?;
    let evaluator = BlockEvaluator::lower(&checked, &bx)?;
    let targets = invoke_targets(&evaluator);
    assert_eq!(targets.len(), 5);
    assert_ne!(targets[0], targets[1], "same-layout nominal types");
    assert_eq!(targets[0], targets[2]);
    assert_ne!(targets[3], targets[4], "signed and unsigned byte");
    let mapped = evaluator
        .function
        .nodes
        .iter()
        .find_map(|node| match &node.payload {
            NodePayload::Map { to_apply, .. } => Some(to_apply.as_str()),
            _ => None,
        });
    assert_eq!(mapped, Some(targets[1]));
    let args = [
        IrValue::parse_typed("(bits[8]:17)")?,
        IrValue::parse_typed("(bits[8]:51)")?,
        IrValue::parse_typed("bits[8]:-3")?,
    ];
    evaluator.compare_ir(&compile_dslx_module(&native, &[])?, &args)?;
    Ok(())
}

#[test]
fn nested_tuple_and_array_patterns_bind_types_for_body_annotations() -> Result<(), Box<dyn Error>> {
    let generic = r#"
fn swap_first<A:type,B:type,N:u32>(x:(A,B[N]))->(B,A) {
    let head:B = x.1[u32:0];
    (head,x.0)
}
"#;
    let params = "x: (s16, (u8,bool)[2])";
    let result = "((u8,bool),s16)";
    let bx = format!("{generic}{}", bx_source(params, result, "swap_first(x)"));
    let native = format!(
        "#![feature(generics)]\n{generic}{}",
        dslx_source(params, result, "swap_first(x)")
    );
    let checked = check_module(&parse_module(&bx)?)?;
    let evaluator = BlockEvaluator::lower(&checked, &bx)?;
    let args = [IrValue::parse_typed(
        "(bits[16]:-9, [(bits[8]:255,bits[1]:0),(bits[8]:1,bits[1]:1)])",
    )?];
    assert_eq!(
        evaluator.compare_ir(&compile_dslx_module(&native, &[])?, &args)?,
        IrValue::parse_typed("((bits[8]:255,bits[1]:0),bits[16]:-9)")?
    );
    Ok(())
}

#[test]
fn let_bound_aggregate_literals_use_the_preceding_type_witness() -> Result<(), Box<dyn Error>> {
    let params = "xs:s8[1], pair:(s8[1],u8), matrix:s8[1][1], cond:bool";
    let result = "(s8[1],(s8[1],u8),s8[1][1],(s8[1],u8),s8[1],s8[1],(s8[1],u8))";
    let body = r#"
let ys = [0x80];
let other = ([0x80],u8:1);
(pick(xs,ys), pick(pair,other), pick(matrix,[ys]), pick(pair,(ys,u8:1)),
 pick(xs,if cond { let a = ys; a } else { let b = ys; b }),
 pick(xs,match cond {true => ys, _ => ys}),
 pick(pair,({ let a = [0x80]; a },u8:1)))
"#;
    let bx = format!(
        "fn pick<T>(a:T,b:T)->T {{ b }} {}",
        bx_source(params, result, body)
    );
    let native = format!(
        "#![feature(generics)]\nfn pick<T:type>(a:T,b:T)->T {{ b }} {}",
        dslx_source(params, result, body)
    );
    let checked = check_module(&parse_module(&bx)?)?;
    let evaluator = BlockEvaluator::lower(&checked, &bx)?;
    let inputs = [
        IrValue::parse_typed("[bits[8]:7]")?,
        IrValue::parse_typed("([bits[8]:9],bits[8]:2)")?,
        IrValue::parse_typed("[[bits[8]:5]]")?,
        IrValue::parse_typed("bits[1]:1")?,
    ];
    assert_eq!(
        evaluator.compare_ir(&compile_dslx_module(&native, &[])?, &inputs)?,
        IrValue::parse_typed(concat!(
            "([bits[8]:128],([bits[8]:128],bits[8]:1),[[bits[8]:128]],",
            "([bits[8]:128],bits[8]:1),[bits[8]:128],[bits[8]:128],([bits[8]:128],bits[8]:1))"
        ))?
    );

    for rejected in [
        bx.replace("let ys =", "let ys: u8[1] ="),
        bx.replace("pick(xs,ys)", "pick(ys,xs)"),
    ] {
        assert!(
            check_module(&parse_module(&rejected)?).is_err(),
            "{rejected}"
        );
    }
    Ok(())
}

#[test]
fn tuple_fields_infer_independent_types_and_dimensions() -> Result<(), Box<dyn Error>> {
    let helpers = r#"
fn pick<T,N:u32>(a:T,b:(T,uN[N]))->(T,u32) { (b.0,N) }
fn second<A,B>(a:A,pair:(A,B))->B { pair.1 }
"#;
    let params = "xs:s8[1], pair:(s8[1],u8), matrix:s8[1][1], cond:bool";
    let result = "(((s8[1],u8),u32),(s8[1][1],u32),(s8[1],u32),(s8[1],u32),u16,u16)";
    let body = r#"
let ys = [0x80];
let parts = (ys,u16:5);
(pick(pair,(({let a=[0x80]; a},u8:1),u16:99)),
 pick(matrix,([ys],u16:99)),
 pick(xs,(if cond {let a=ys; a} else {let b=ys; b},u16:99)),
 pick(xs,if cond {let a=ys; (a,u16:9)} else {let b=ys; (b,u16:10)}),
 second(xs,if cond {let a=parts; a} else {let b=parts; b}),
 second(xs,match cond {true => parts, _ => parts}))
"#;
    let bx = format!("{helpers}{}", bx_source(params, result, body));
    let native = format!(
        "#![feature(generics)]\n{}{}",
        helpers
            .replace("<T,N", "<T:type,N")
            .replace("<A,B>", "<A:type,B:type>"),
        dslx_source(params, result, body)
    );
    let checked = check_module(&parse_module(&bx)?)?;
    let evaluator = BlockEvaluator::lower(&checked, &bx)?;
    let inputs = [
        IrValue::parse_typed("[bits[8]:7]")?,
        IrValue::parse_typed("([bits[8]:9],bits[8]:2)")?,
        IrValue::parse_typed("[[bits[8]:9]]")?,
        IrValue::parse_typed("bits[1]:1")?,
    ];
    assert_eq!(
        evaluator.compare_ir(&compile_dslx_module(&native, &[])?, &inputs)?,
        IrValue::parse_typed(
            "((([bits[8]:128],bits[8]:1),bits[32]:16),([[bits[8]:128]],bits[32]:16),([bits[8]:128],bits[32]:16),([bits[8]:128],bits[32]:16),bits[16]:5,bits[16]:5)"
        )?
    );
    Ok(())
}

#[test]
fn type_parameters_cannot_be_supplied_positionally() {
    let source = "fn id<T:type>(x:T)->T { x } fn sample(x:u8)->u8 { id<u8>(x) }";
    let result = parse_module(source).and_then(|module| check_module(&module));
    assert!(result.is_err(), "{source}");
}

#[test]
fn resolved_enum_in_callback_result_is_independent_of_same_named_type_parameter()
-> Result<(), Box<dyn Error>> {
    let mut module = parse_module(
        r#"
enum T: u8 { A = 0 }
fn ret() -> T { T::A }
fn apply<T, F: fn() -> T>(x: T) -> T { let _ = F(); x }
const RESULT = apply<ret>(u16:0x1234);
"#,
    )?;
    // AST callers can refer directly to the resolved enum inside this scope.
    module.functions[1].callback_params[0].result = Type::Enum {
        name: "T".into(),
        signed: false,
        width: 8,
    };
    let checked = check_module(&module)?;
    assert_eq!(
        checked.constants()[0].value,
        IrValue::make_ubits(16, 0x1234)?
    );
    Ok(())
}

#[test]
fn extern_and_imported_callback_results_bind_successive_types() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    fs::write(
        dir.path().join("words.x"),
        "pub fn describe<N:u32>(x:uN[N])->(uN[N],u32) { (!x,N) }",
    )?;
    let options = search_paths(&[dir.path()]);
    let bx = r#"
use words;
struct Packet { valid:bool, payload:u8 }
extern fn extract(p:Packet)->u8 = "extract_payload";
fn chain<A,B,C,F:fn(A)->B,G:fn(B)->C>(x:A)->C { G(F(x)) }
block sample(clk, p:Packet)->(u8,u32) { chain<extract,words::describe>(p) }
"#;
    let checked = check_source_with_options(bx, &options)?;
    let projection = |field, width| {
        format!(
            r#"package wires
fn extract_payload(p:(bits[1],bits[8]) id=1)->bits[{width}] {{
    ret tuple_index.2: bits[{width}] = tuple_index(p, index={field}, id=2)
}}
"#
        )
    };
    let package = lower_module(
        &checked,
        Parser::new(&projection(1, 8)).parse_and_verify_package()?,
        ResetBehavior::default(),
    )?;
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    let sample = combinational_block_to_fn(package.get_block("sample").ok_or("missing sample")?)?;
    assert_eq!(
        eval_fn_in_package_value(
            &package,
            &sample,
            &[IrValue::parse_typed("(bits[1]:0,bits[8]:51)")?]
        ),
        IrValue::parse_typed("(bits[8]:204,bits[32]:8)")?
    );
    let incompatible = Parser::new(&projection(0, 1)).parse_and_verify_package()?;
    assert_eq!(
        lower_module(&checked, incompatible, ResetBehavior::default()).unwrap_err(),
        "extern fn 'extract' expects ((bits[1], bits[8])) -> bits[8], but linked IR function 'extract_payload' has ((bits[1], bits[8])) -> bits[1]"
    );
    Ok(())
}

#[test]
fn conflicting_type_and_length_evidence_is_rejected() -> Result<(), Box<dyn Error>> {
    let decls = r#"
struct A { value: u8 }
struct B { value: u8 }
fn pair<T>(a:T, b:T)->(T,T) { (a,b) }
fn combine(a:A,b:B)->u8 { a.value + b.value }
fn wrong(a:B,b:A)->u8 { a.value + b.value }
"#;
    let source = format!(
        "{decls}{ZIP_WITH} fn probe(a:A,b:B)->A {{ let _ = pair(a,a); let _ = zip_with<combine>([a],[b]); a }}"
    );
    check_module(&parse_module(&source)?)?;
    for call in [
        "pair(a,b)",
        "pair([a],[b])",
        "pair((a,u8:1),(a,s8:1))",
        "zip_with<wrong>([a],[b])",
        "zip_with<combine>([a,a],[b])",
        "zip_with<combine,2>([a],[b])",
    ] {
        let source = format!("{decls}{ZIP_WITH} fn probe(a:A,b:B)->A {{ let _ = {call}; a }}");
        let parsed = parse_module(&source)?;
        assert!(check_module(&parsed).is_err(), "{source}");
    }
    Ok(())
}
