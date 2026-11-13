// SPDX-License-Identifier: Apache-2.0

use super::PIPE_REG;
use super::support::search_paths;
use std::{error::Error, fs};
use tempfile::tempdir;
use xlsynth_bx::{
    ImportOptions, ResetBehavior, check_module, check_module_with_options, compile,
    compile_with_options, empty_package, lower_module, parse_module,
};
use xlsynth_codegen::{BlockCodegenOptions, emit_system_verilog};
use xlsynth_pir::IrValue;
use xlsynth_pir::block_inline::inline_all_blocks_in_package;
use xlsynth_pir::block_invoke_inline::inline_block_invokes_in_package;
use xlsynth_pir::block2fn::combinational_block_to_fn;
use xlsynth_pir::ir::{NodePayload, Package};
use xlsynth_pir::ir_eval::eval_fn_in_package_value;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_vastly::{FixtureRunner, compile_pipeline_module};

const TRANSFORM: &str = r#"
block transform<In, Out, F: fn(In) -> Out>(clk, x: In) -> Out { F(x) }
"#;

const ADVANCE: &str = r#"
block advance<In, Out, F: fn(In) -> Out>(clk, prev: (bool, In)) -> (bool, Out) {
    let next = pipe_reg(prev.0, F(prev.1));
    next
}
"#;

/// Simulate the two registered stages after inlining their BX hierarchy.
fn chain_outputs(
    source: &str,
    options: &ImportOptions<'_>,
    inputs: &[(bool, bool, u64)],
) -> Result<Vec<[Option<u64>; 2]>, Box<dyn Error>> {
    let mut package = compile_with_options(
        source,
        empty_package("pipeline"),
        ResetBehavior::default(),
        options,
    )?;
    xlsynth::IrPackage::parse_ir(&package.to_string(), None)?;
    inline_all_blocks_in_package(&mut package)?;
    let package = inline_block_invokes_in_package(package)?;
    let rtl = emit_system_verilog(&package, &BlockCodegenOptions::default())?;
    simulate_chain(&rtl.system_verilog, inputs).map_err(|e| format!("{e:?}").into())
}

/// Collect the valid bit and payload immediately after each rising clock edge.
fn simulate_chain(
    rtl: &str,
    inputs: &[(bool, bool, u64)],
) -> xlsynth_vastly::Result<Vec<[Option<u64>; 2]>> {
    let module = compile_pipeline_module(rtl)?;
    let mut runner = FixtureRunner::new(&module)?;
    let ports = runner.port_bindings();
    let rst = ports.input("rst")?;
    let valid = ports.input("valid")?;
    let data = ports.input("data")?;
    let mut outputs = Vec::new();
    for &(reset, v, x) in inputs {
        let edge = runner.step_cycle_with_drive(&mut [], |drive| {
            drive.set_bool(&rst, reset)?;
            drive.set_bool(&valid, v)?;
            drive.set_u64(&data, x)
        })?;
        outputs.push(["out_0", "out_1"].map(|name| edge.outputs[name].to_u64_if_known()));
    }
    Ok(outputs)
}

#[test]
fn imported_widen_and_square_match_a_hand_specialized_pipeline() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    fs::write(
        dir.path().join("math.x"),
        r#"
pub fn widen(x:u8) -> u16 { x as u16 }
pub fn square(x:u16) -> u32 { let wide = x as u32; wide * wide }
"#,
    )?;
    let options = search_paths(&[dir.path()]);
    let generic = r#"
pub block chain(clk, valid: bool, data: u8) -> (bool, u32) {
    let widened = advance<math::widen>((valid, data));
    let squared = advance<math::square>(widened);
    squared
}
"#;
    let concrete = r#"
block widen_stage(clk, prev: (bool, u8)) -> (bool, u16) {
    let wide = math::widen(prev.1);
    let pipe_stage = pipe_reg(prev.0, wide);
    pipe_stage
}
block square_stage(clk, prev: (bool, u16)) -> (bool, u32) {
    let square = math::square(prev.1);
    let pipe_stage = pipe_reg(prev.0, square);
    pipe_stage
}
pub block chain(clk, valid: bool, data: u8) -> (bool, u32) {
    let widened = widen_stage((valid, data));
    let squared = square_stage(widened);
    squared
}
"#;
    let inputs = [
        (true, false, 0),
        (false, true, 3),
        (false, true, 5),
        (false, false, 99),
        (false, true, 7),
        (true, true, 9),
        (false, true, 255),
        (false, false, 18),
        (false, false, 17),
    ];
    let actual = chain_outputs(
        &format!("use math; {PIPE_REG} {ADVANCE} {generic}"),
        &options,
        &inputs,
    )?;
    let reference = chain_outputs(
        &format!("use math; {PIPE_REG} {concrete}"),
        &options,
        &inputs,
    )?;
    assert_eq!(actual, reference);
    assert_eq!(actual[0][0], Some(0));
    assert_eq!(actual[1][0], Some(0));
    assert_eq!(
        actual[2..],
        [
            (1, 9),
            (1, 25),
            (0, 25),
            (0, 49),
            (0, 49),
            (1, 65025),
            (0, 65025)
        ]
        .map(|(valid, data)| [Some(valid), Some(data)])
    );
    Ok(())
}

#[test]
fn identical_callback_stages_keep_independent_registers() -> Result<(), Box<dyn Error>> {
    let chain = r#"
fn inc(x:u8)->u8 { x + u8:1 }
pub block chain(clk, valid:bool, data:u8)->(bool,u8) {
    let left = advance<inc>((valid,data));
    let right = advance<inc>(left);
    right
}
"#;
    let inputs = [
        (true, false, 0),
        (false, true, 7),
        (false, true, 13),
        (false, false, 80),
        (false, false, 99),
    ];
    let actual = chain_outputs(
        &format!("{PIPE_REG} {ADVANCE} {chain}"),
        &ImportOptions::default(),
        &inputs,
    )?;
    assert_eq!(actual[0][0], Some(0));
    assert_eq!(actual[1][0], Some(0));
    assert_eq!(
        actual[2..],
        [[Some(1), Some(9)], [Some(1), Some(15)], [Some(0), Some(15)]]
    );
    Ok(())
}

/// Evaluate an already linked combinational hierarchy at its top interface.
fn evaluate_top(mut package: Package, args: &[IrValue]) -> Result<IrValue, String> {
    inline_all_blocks_in_package(&mut package)?;
    let top = package.get_top_block().ok_or("missing top block")?;
    let function = combinational_block_to_fn(top)?;
    Ok(eval_fn_in_package_value(&package, &function, args))
}

#[test]
fn cached_blocks_distinguish_callback_identity_and_concrete_dslx_width()
-> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    fs::write(
        dir.path().join("math.x"),
        "pub fn inv<N:u32>(x:uN[N])->uN[N] { !x }",
    )?;
    let source = format!(
        r#"
use math;
use math::inv;
fn bump(x:u8)->u8 {{ x + u8:1 }}
{TRANSFORM}
block sample(clk, a:u8, b:u16)->(u8,u8,u8,u16) {{
    let left = transform<math::inv>(a);
    let right = transform<inv>(a + u8:1);
    let increment = transform<bump>(a);
    let wide = transform<math::inv>(b);
    (left, right, increment, wide)
}}
"#
    );
    let package = compile_with_options(
        &source,
        empty_package("bindings"),
        ResetBehavior::default(),
        &search_paths(&[dir.path()]),
    )?;
    let [left, right, increment, wide] = package.get_top_block().unwrap().instantiations.as_slice()
    else {
        panic!("expected four transform instances");
    };
    assert_ne!(left.name, right.name);
    assert_eq!(left.block, right.block);
    assert_ne!(left.block, increment.block);
    assert_ne!(left.block, wide.block);
    assert_eq!(
        evaluate_top(
            package,
            &[IrValue::make_ubits(8, 37)?, IrValue::make_ubits(16, 1)?]
        )?,
        IrValue::parse_typed("(bits[8]:218,bits[8]:217,bits[8]:38,bits[16]:65534)")?
    );
    Ok(())
}

#[test]
fn imported_callback_results_retain_nominal_identity() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    fs::write(
        dir.path().join("payload.x"),
        r#"
pub struct A { value:u8 }
pub struct B { value:u8 }
pub fn wrap(x:u8)->A { A { value:x } }
pub fn unpack(x:A)->u8 { x.value }
pub fn other(x:B)->u8 { x.value }
"#,
    )?;
    let source = format!(
        r#"
use payload;
{TRANSFORM}
block sample(clk,x:u8)->u8 {{
    let packet=transform<payload::wrap>(x);
    let value=transform<payload::unpack>(packet);
    value
}}
"#
    );
    let options = search_paths(&[dir.path()]);
    let package = compile_with_options(
        &source,
        empty_package("nominal"),
        ResetBehavior::default(),
        &options,
    )?;
    assert_eq!(
        evaluate_top(package, &[IrValue::make_ubits(8, 17)?])?,
        IrValue::make_ubits(8, 17)?
    );
    let different = source.replace("payload::unpack", "payload::other");
    assert!(check_module_with_options(&parse_module(&different)?, &options).is_err());
    for stage in [
        "block stage<payload,F:fn(payload::A)->u8>(clk,x:payload)->u8 { u8:0 }",
        "block stage<payload,F:fn(payload)->u8>(clk,x:payload)->u8 { let p=payload::A{value:u8:0}; F(x) }",
        "block stage<payload,F:fn(payload)->u8>(clk,x:payload)->u8 { unroll_for! (_,a) in u32:0..u32:0 { let p=payload::A{value:a}; a }(F(x)) }",
    ] {
        let shadowed = format!(
            "use payload; fn id(x:u8)->u8 {{ x }} {stage} block sample(clk,x:u8)->u8 {{ let v=stage<id>(x); v }}"
        );
        let error = check_module_with_options(&parse_module(&shadowed)?, &options).unwrap_err();
        assert!(
            error.contains("'payload' names a type parameter here"),
            "{error}"
        );
    }
    Ok(())
}

#[test]
fn callback_results_supply_later_input_types_and_nested_result_members()
-> Result<(), Box<dyn Error>> {
    let source = r#"
fn split(x:u8)->(u16,bool[1]) { (x as u16,[x == u8:0]) }
fn pack(lo:u16,hi:u16)->u4[2] { [lo as u4, hi as u4] }
block chain<I,M,O,F:fn(I)->(M,bool[1]),G:fn(M,M)->O[2]>(clk,x:I)->O[2] {
    let word = F(x).0;
    G(word,word >> u32:4)
}
block sample(clk,x:u8)->u4[2] { let result = chain<split,pack>(x); result }
"#;
    let package = compile(source, empty_package("ordered"), ResetBehavior::default())?;
    assert_eq!(
        evaluate_top(package, &[IrValue::make_ubits(8, 0xa7)?])?,
        IrValue::parse_typed("[bits[4]:7,bits[4]:10]")?
    );
    Ok(())
}

#[test]
fn callbacks_reject_unresolved_inputs_conflicts_and_dynamic_targets() -> Result<(), Box<dyn Error>>
{
    let declarations = format!(
        r#"
fn widen(x:u8)->u16 {{ x as u16 }}
fn signed(x:s8)->u16 {{ x as u16 }}
{TRANSFORM}
block fixed<I,O,F:fn(I)->O>(clk,x:I,expected:O)->O {{ F(x) }}
block sample(clk,x:u8)->u16 {{ BODY }}
"#
    );
    for body in [
        "let ok = fixed<widen>(x, u16:0); let bad = fixed<widen>(x, u32:0); bad as u16",
        "let v = transform<signed>(x); v",
        "let v = transform<widen>(1); v",
        "let v = transform(x); v",
        "let widen=x; let v=transform<widen>(x); v",
        "let v = transform<|x| x as u16>(x); v",
        "let v = if true { transform<widen>(x) } else { u16:0 }; v",
    ] {
        let source = declarations.replace("BODY", body);
        assert!(
            parse_module(&source)
                .and_then(|m| check_module(&m))
                .is_err(),
            "{source}"
        );
    }
    let contextual = declarations.replace("BODY", "let v = fixed<widen>(x, 1); v");
    let pkg = compile(
        &contextual,
        empty_package("contextual"),
        ResetBehavior::default(),
    )?;
    assert_eq!(
        evaluate_top(pkg, &[IrValue::make_ubits(8, 0xff)?])?,
        IrValue::make_ubits(16, 0xff)?
    );
    let unanchored = r#"
fn widen(x:u8)->u16 { x as u16 }
block missing<I,O,F:fn(I)->O>(clk)->O { F(zero!<I>()) }
block sample(clk)->u16 { let v=missing<widen>(); v }
"#;
    assert!(check_module(&parse_module(unanchored)?).is_err());
    let forwarded = format!(
        r#"
fn widen(x:u8)->u16 {{ x as u16 }}
{TRANSFORM}
block wrapper<I,O,F:fn(I)->O>(clk,x:I)->O {{ let child=transform<F>(x); child }}
block sample(clk,x:u8)->u16 {{ let child=wrapper<widen>(x); child }}
"#
    );
    assert!(check_module(&parse_module(&forwarded)?).is_err());
    let shadowed = r#"
fn widen(x:u8)->u16 { x as u16 }
block H(clk,x:u8)->u8 { x }
block wrapper<I,O,H:fn(I)->O>(clk,x:I)->O {
    let h: u8 = H(x);
    H(x)
}
block sample(clk,x:u8)->u16 { let child=wrapper<widen>(x); child }
"#;
    assert!(check_module(&parse_module(shadowed)?).is_err());
    let dir = tempdir()?;
    fs::write(
        dir.path().join("math.x"),
        "fn private_fn(x:u8)->u16 { x as u16 }",
    )?;
    let private = format!(
        "use math; {TRANSFORM} block sample(clk,x:u8)->u16 {{ let child=transform<math::private_fn>(x); child }}"
    );
    let options = search_paths(&[dir.path()]);
    assert!(check_module_with_options(&parse_module(&private)?, &options).is_err());
    Ok(())
}

#[test]
fn unrolled_child_blocks_accept_callbacks_and_validate_empty_bodies() -> Result<(), Box<dyn Error>>
{
    let source = format!(
        r#"
fn widen(x:u8)->u16 {{ x as u16 }}
{TRANSFORM}
block sample(clk,x:u8)->u16 {{
    unroll_for! (i,acc) in u32:0..u32:2 {{
        let child = transform<widen>(x); acc + child
    }}(u16:0)
}}
"#
    );
    let package = compile(&source, empty_package("unrolled"), ResetBehavior::default())?;
    assert_eq!(
        evaluate_top(package, &[IrValue::make_ubits(8, 7)?])?,
        IrValue::make_ubits(16, 14)?
    );
    let empty = source.replace("u32:0..u32:2", "u32:0..u32:0");
    check_module(&parse_module(&empty)?)?;
    let unknown = empty.replace("transform<widen>", "transform<unknown>");
    assert!(check_module(&parse_module(&unknown)?).is_err());
    let missing = empty.replace("transform<widen>", "transform");
    assert!(check_module(&parse_module(&missing)?).is_err());
    Ok(())
}

#[test]
fn extern_callbacks_link_and_validate_for_blocks_and_functions() -> Result<(), Box<dyn Error>> {
    let bx = format!(
        r#"
extern fn linked(x:s8)->s16 = "sign";
fn apply<F:fn(s8)->s16>(x:s8)->s16 {{ F(x) }}
{TRANSFORM}
block sample(clk,x:s8)->(s16,s16) {{
    let child=transform<linked>(x);
    (child,apply<linked>(x))
}}
"#
    );
    let ir = r#"package prelinked
fn sign(x: bits[8] id=1) -> bits[16] {
  ret sign_ext.2: bits[16] = sign_ext(x, new_bit_count=16, id=2)
}
"#;
    let checked = check_module(&parse_module(&bx)?)?;
    let package = lower_module(
        &checked,
        Parser::new(ir).parse_and_validate_package()?,
        ResetBehavior::default(),
    )?;
    let child = &package.get_top_block().unwrap().instantiations[0].block;
    let calls: Vec<_> = package
        .get_block(child)
        .unwrap()
        .nodes
        .iter()
        .filter_map(|node| match &node.payload {
            NodePayload::Invoke { to_apply, .. } => Some(to_apply.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(calls, ["sign"]);
    assert_eq!(
        evaluate_top(package, &[IrValue::make_ubits(8, 255)?])?,
        IrValue::parse_typed("(bits[16]:65535,bits[16]:65535)")?
    );
    let missing =
        lower_module(&checked, empty_package("missing"), ResetBehavior::default()).unwrap_err();
    assert!(missing.contains("no linked IR function"), "{missing}");
    let wrong_ir = ir.replace("bits[8]", "bits[7]");
    assert!(
        lower_module(
            &checked,
            Parser::new(&wrong_ir).parse_and_validate_package()?,
            ResetBehavior::default()
        )
        .is_err()
    );
    let wrong_signedness = bx.replace("linked(x:s8)", "linked(x:u8)");
    assert!(check_module(&parse_module(&wrong_signedness)?).is_err());

    let effectful = r#"package prelinked
fn observed(x: bits[8] id=1) -> bits[16] {
  bit_slice.2: bits[1] = bit_slice(x, start=0, width=1, id=2)
  event: () = cover(bit_slice.2, label="seen", id=3)
  ret sign_ext.4: bits[16] = sign_ext(x, new_bit_count=16, id=4)
}
fn sign(x: bits[8] id=5) -> bits[16] {
  ret invoke.6: bits[16] = invoke(x, to_apply=observed, id=6)
}
"#;
    let err = lower_module(
        &checked,
        Parser::new(effectful).parse_and_validate_package()?,
        ResetBehavior::default(),
    )
    .unwrap_err();
    assert!(err.contains("effectful IR function 'observed'"), "{err}");
    Ok(())
}
