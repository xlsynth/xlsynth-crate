// SPDX-License-Identifier: Apache-2.0

use xlsynth_bx::{check_module, parse_module};

fn check(source: &str) -> Result<(), String> {
    check_module(&parse_module(source)?)?;
    Ok(())
}

#[test]
fn checks_different_stages_for_different_opcodes() -> Result<(), String> {
    let source = r#"
enum Op : u1 { Fast = 0, Slow = 1 }
fn stage(op: Op) -> u32 {
    match op { Op::Fast => u32:1, Op::Slow => u32:2 }
}
block pulse(clk, op: Op) -> bool @stage(op) {
    reg fast_q: bool;
    reg slow_q: bool;
    reg slow_qq: bool;
    fast_q := op == Op::Fast;
    slow_q := op == Op::Slow;
    slow_qq := slow_q;
    fast_q | slow_qq
}
"#;
    check(source)?;
    assert_eq!(
        check(&source.replace("fast_q | slow_qq", "fast_q | slow_q")).unwrap_err(),
        "block 'pulse': case (op=Op::Slow): expected cycle 2, got cycle 1"
    );
    assert!(check(&source.replace("reg fast_q: bool;", "reg fast_q: bool = false;")).is_err());
    Ok(())
}

#[test]
fn follows_multiple_registered_controls_into_a_tuple_match() -> Result<(), String> {
    check(
        r#"
enum Op : u1 { Fast = 0, Slow = 1 }
fn stage(valid: bool, op: Op) -> u32 {
    match (valid, op) { (true, Op::Fast) => u32:1, _ => u32:1 }
}
block route(clk, valid: bool, op: Op, data: u8) -> u8 @stage(valid, op) {
    reg valid_q: bool;
    reg op_q: Op;
    reg data_q: u8;
    valid_q := valid;
    op_q := op;
    data_q := data;
    match (valid_q, op_q) { (true, Op::Fast) => data_q, _ => u8:0 }
}
"#,
    )
}

#[test]
fn checks_opcode_alignment_at_the_mux() -> Result<(), String> {
    let source = r#"
enum Op : u1 { Fast = 0, Slow = 1 }
fn stage(op: Op) -> u32 { u32:1 }
block route(clk, op: Op, data: u8) -> u8 @stage(op) {
    reg op_q: Op;
    reg data_q: u8;
    op_q := op;
    data_q := data;
    match op_q { Op::Fast => data_q, Op::Slow => !data_q }
}

"#;
    check(source)?;
    assert_eq!(
        check(&source.replace("match op_q", "match op")).unwrap_err(),
        "block 'route': case (op=Op::Fast): match expression has condition at cycle 0 and data at cycle 1"
    );
    Ok(())
}

#[test]
fn a_disabled_register_still_has_its_stage() -> Result<(), String> {
    check(
        r#"
fn stage(en: bool) -> u32 { u32:1 }
block held(clk, en: bool, x: u8) -> u8 @stage(en) {
    reg q: u8;
    if en { q := x; }
    q
}
"#,
    )
}

#[test]
fn selecting_the_same_register_on_both_arms_does_not_need_an_aligned_select() -> Result<(), String>
{
    check(
        r#"
fn stage(en: bool) -> u32 { u32:1 }
block held(clk, en: bool, x: u8) -> u8 @stage(en) {
    reg q: u8;
    if en { q := x; }
    match en { true => q, false => q }
}
"#,
    )
}

#[test]
fn fixed_output_stage_also_checks_match_arms() -> Result<(), String> {
    check(
        r#"
block route(op: bool, data: u8) -> u8 @0 {
    match op { true => data, false => !data }
}
"#,
    )
}

#[test]
fn rejects_unbounded_control_domains() {
    assert_eq!(
        check("fn stage(op:u8)->u32 { u32:0 } block b(op:u8)->u8 @stage(op) { op }").unwrap_err(),
        "block 'b': stage input domain exceeds 64 cases"
    );
}
