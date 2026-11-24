// SPDX-License-Identifier: Apache-2.0

//! Interpreted and compiled clocks legalize structural feedback without losing
//! hierarchical state or diagnostic identities.

use xlsynth_pir::IrValue;
use xlsynth_pir::block_eval::{BlockEvaluator, RegisterInit};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir_compiler::block::CompiledBlock;

const FEEDBACK: &str = r#"package feedback
block leaf(clk: clock, d: bits[4], q: bits[4], old: bits[4]) {
  reg saved(bits[4])
  d: bits[4] = input_port(name=d, id=1)
  previous: bits[4] = register_read(register=saved, id=2)
  write: () = register_write(d, register=saved, id=3)
  tap: bits[1] = bit_slice(d, start=0, width=1, id=4)
  hit: () = cover(tap, label="set", id=5)
  q: () = output_port(d, name=q, id=6)
  old: () = output_port(previous, name=old, id=7)
}
top block root(clk: clock, seed: bits[1], result: bits[4], old: bits[4]) {
  instantiation u(block=leaf, kind=block)
  seed: bits[1] = input_port(name=seed, id=11)
  q: bits[4] = instantiation_output(instantiation=u, port_name=q, id=12)
  lower: bits[3] = bit_slice(q, start=0, width=3, id=13)
  combined: bits[4] = concat(lower, seed, id=14)
  drive: () = instantiation_input(combined, instantiation=u, port_name=d, id=15)
  previous: bits[4] = instantiation_output(instantiation=u, port_name=old, id=16)
  result: () = output_port(q, name=result, id=17)
  old: () = output_port(previous, name=old, id=18)
}"#;

/// Interpreted, compiled, and reusable compiled clocks retain source-facing
/// register and cover identities across a feedback legalization.
#[test]
fn clock_feedback_preserves_hierarchical_state_and_events() {
    for source in [
        FEEDBACK.to_string(),
        FEEDBACK.replace(
            "instantiation u(block=leaf, kind=block)",
            "instantiation u(block=leaf, module_name=external_leaf, kind=behavioral_extern_block)",
        ),
    ] {
        let package = Parser::new(&source).parse_package().unwrap();
        let interpreter = BlockEvaluator::prepare(&package, "root").unwrap();
        let compiled = CompiledBlock::compile(&package, "root").unwrap();
        assert_eq!(interpreter.registers()[0].name, "u.saved");
        assert_eq!(compiled.registers(), interpreter.registers());
        let mut oracle_state = interpreter.initial_state(RegisterInit::AllZeros).unwrap();
        let mut compiled_state = compiled.initial_state(RegisterInit::AllZeros).unwrap();
        let mut reusable = compiled.prepare_instance(RegisterInit::AllZeros).unwrap();
        let mut old = 0;
        for seed in [false, true, false] {
            let inputs = [IrValue::bool(seed)];
            let expected = interpreter.step(&oracle_state, &inputs).unwrap();
            let actual = compiled.step(&compiled_state, &inputs).unwrap();
            let events = reusable.tick(&inputs).unwrap();
            let value = if seed { 15 } else { 0 };
            assert_eq!(
                expected.outputs,
                [
                    IrValue::make_ubits(4, value).unwrap(),
                    IrValue::make_ubits(4, old).unwrap()
                ]
            );
            assert_eq!(actual.outputs, expected.outputs);
            assert_eq!(actual.cover_counts, expected.cover_counts);
            assert_eq!(events.cover_counts, expected.cover_counts);
            assert_eq!(expected.cover_counts.len(), 1);
            assert_eq!(expected.cover_counts[0].site.instance, "u");
            assert_eq!(expected.cover_counts[0].site.node_text_id, 5);
            assert_eq!(expected.cover_counts[0].count, u64::from(seed));
            assert_eq!(actual.next_state.values(), expected.next_state.values());
            assert_eq!(
                reusable.snapshot().unwrap().values(),
                expected.next_state.values()
            );
            oracle_state = expected.next_state;
            compiled_state = actual.next_state;
            old = value;
        }
    }
}

/// Real combinational feedback is rejected consistently by both clocks.
#[test]
fn actual_clock_feedback_remains_an_error() {
    let source = FEEDBACK.replace(
        "instantiation_input(combined, instantiation=u",
        "instantiation_input(q, instantiation=u",
    );
    let package = Parser::new(&source).parse_package().unwrap();
    let expected = BlockEvaluator::prepare(&package, "root")
        .err()
        .unwrap()
        .to_string();
    let actual = CompiledBlock::compile(&package, "root")
        .err()
        .unwrap()
        .to_string();
    assert!(expected.contains("cycle"), "{expected}");
    assert_eq!(actual, expected);
}

/// Reusing one model does not merge instance state or source event identities
/// when both instances require structural feedback legalization.
#[test]
fn reused_modeled_feedback_keeps_instance_order_and_distinct_event_paths() {
    let leaf = FEEDBACK.split("top block root(").next().unwrap();
    let source = format!(
        r#"{leaf}top block root(clk: clock, a: bits[1], b: bits[1], qa: bits[4], qb: bits[4], olda: bits[4], oldb: bits[4]) {{
  instantiation u(block=leaf, module_name=external_leaf, kind=behavioral_extern_block)
  instantiation v(block=leaf, module_name=external_leaf, kind=behavioral_extern_block)
  a: bits[1] = input_port(name=a, id=11)
  b: bits[1] = input_port(name=b, id=12)
  u_q: bits[4] = instantiation_output(instantiation=u, port_name=q, id=13)
  u_low: bits[3] = bit_slice(u_q, start=0, width=3, id=14)
  u_d: bits[4] = concat(u_low, a, id=15)
  u_bind: () = instantiation_input(u_d, instantiation=u, port_name=d, id=16)
  v_q: bits[4] = instantiation_output(instantiation=v, port_name=q, id=17)
  v_low: bits[3] = bit_slice(v_q, start=0, width=3, id=18)
  v_d: bits[4] = concat(v_low, b, id=19)
  v_bind: () = instantiation_input(v_d, instantiation=v, port_name=d, id=20)
  u_old: bits[4] = instantiation_output(instantiation=u, port_name=old, id=21)
  v_old: bits[4] = instantiation_output(instantiation=v, port_name=old, id=22)
  qa: () = output_port(u_q, name=qa, id=23)
  qb: () = output_port(v_q, name=qb, id=24)
  olda: () = output_port(u_old, name=olda, id=25)
  oldb: () = output_port(v_old, name=oldb, id=26)
}}"#
    );
    let package = Parser::new(&source).parse_package().unwrap();
    let interpreter = BlockEvaluator::prepare(&package, "root").unwrap();
    let compiled = CompiledBlock::compile(&package, "root").unwrap();
    assert_eq!(
        interpreter
            .registers()
            .iter()
            .map(|field| field.name.as_str())
            .collect::<Vec<_>>(),
        ["u.saved", "v.saved"]
    );
    assert_eq!(compiled.registers(), interpreter.registers());
    let initial = RegisterInit::Supplied(vec![
        IrValue::make_ubits(4, 7).unwrap(),
        IrValue::make_ubits(4, 9).unwrap(),
    ]);
    let mut oracle = interpreter.initial_state(initial.clone()).unwrap();
    let mut native = compiled.initial_state(initial.clone()).unwrap();
    let mut reusable = compiled.prepare_instance(initial).unwrap();
    let mut old = [7, 9];
    for [a, b] in [[true, false], [false, true]] {
        let inputs = [IrValue::bool(a), IrValue::bool(b)];
        let expected = interpreter.step(&oracle, &inputs).unwrap();
        let actual = compiled.step(&native, &inputs).unwrap();
        let events = reusable.tick(&inputs).unwrap();
        let values = [if a { 15 } else { 0 }, if b { 15 } else { 0 }];
        assert_eq!(
            expected.outputs,
            [values[0], values[1], old[0], old[1]]
                .map(|value| IrValue::make_ubits(4, value).unwrap())
        );
        assert_eq!(actual.outputs, expected.outputs);
        assert_eq!(actual.cover_counts, expected.cover_counts);
        assert_eq!(events.cover_counts, expected.cover_counts);
        let mut sites = expected
            .cover_counts
            .iter()
            .map(|count| {
                (
                    count.site.instance.as_str(),
                    count.site.node_text_id,
                    count.count,
                )
            })
            .collect::<Vec<_>>();
        sites.sort_unstable();
        assert_eq!(sites, [("u", 5, u64::from(a)), ("v", 5, u64::from(b))]);
        assert_eq!(actual.next_state.values(), expected.next_state.values());
        assert_eq!(
            reusable.snapshot().unwrap().values(),
            expected.next_state.values()
        );
        oracle = expected.next_state;
        native = actual.next_state;
        old = values;
    }
}
