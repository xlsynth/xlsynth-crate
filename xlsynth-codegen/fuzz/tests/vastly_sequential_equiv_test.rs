// SPDX-License-Identifier: Apache-2.0

use xlsynth_codegen::BlockCodegenOptions;
use xlsynth_codegen_fuzz::semantics::{Sample, Trace};
use xlsynth_codegen_fuzz::vastly::check_rtl_trace;
use xlsynth_codegen_fuzz::{emit, parse_reference};
use xlsynth_test_helpers::rtl_sim::{Bindings, LogicValue};

const BLOCK: &str = r#"package p

top block mixed(clk: clock, rst: bits[1], en: bits[1], data: bits[8], wide: bits[65], out: bits[8]) {
  #![reset(port="rst", asynchronous=false, active_low=false)]
  reg visible(bits[8], reset_value=7)
  reg hidden(bits[65][2])
  rst: bits[1] = input_port(name=rst, id=1)
  en: bits[1] = input_port(name=en, id=2)
  data: bits[8] = input_port(name=data, id=3)
  wide: bits[65] = input_port(name=wide, id=4)
  old_q: bits[8] = register_read(register=visible, id=5)
  old_pair: bits[65][2] = register_read(register=hidden, id=6)
  zero: bits[1] = literal(value=0, id=7)
  earlier: bits[65] = array_index(old_pair, indices=[zero], id=8)
  pair: bits[65][2] = array(wide, earlier, id=9)
  wr_q: () = register_write(data, register=visible, load_enable=en, reset=rst, id=10)
  wr_pair: () = register_write(pair, register=hidden, id=11)
  out: () = output_port(old_q, name=out, id=12)
}
"#;

/// Packs two 65-bit lanes in RTL order, with array index zero at the LSB.
fn hidden_value(newest: u128, older: u128) -> Result<LogicValue, String> {
    LogicValue::parse_binary(&format!("{older:065b}{newest:065b}"), 130)
}

#[test]
fn checks_pre_edge_outputs_and_unobserved_registers_from_supplied_state() -> Result<(), String> {
    let package = parse_reference(BLOCK);
    let rtl = emit(&package, &BlockCodegenOptions::default());
    let a = (1u128 << 64) | 1;
    let b = 2;
    let c = (1u128 << 63) | 4;
    let mut trace = Trace {
        initial_state: Bindings::from([
            ("visible".into(), LogicValue::from_u64(8, 19)),
            ("hidden".into(), hidden_value(a, b)?),
        ]),
        samples: Vec::new(),
        observed_live_behaviors: Default::default(),
    };
    // The hidden pipeline shifts on all three edges, including during reset.
    for (rst, en, wide, preceding_wide, pre_q, post_q) in [
        (1, 0, c, a, 19, 7),
        (0, 1, a, c, 7, 55),
        (0, 0, b, a, 55, 55),
    ] {
        trace.samples.push(Sample {
            inputs: Bindings::from([
                ("rst".into(), LogicValue::from_u64(1, rst)),
                ("en".into(), LogicValue::from_u64(1, en)),
                ("data".into(), LogicValue::from_u64(8, 55)),
                (
                    "wide".into(),
                    LogicValue::parse_binary(&format!("{wide:065b}"), 65)?,
                ),
            ]),
            outputs: Bindings::from([("out".into(), LogicValue::from_u64(8, pre_q))]),
            next_state: Some(Bindings::from([
                ("visible".into(), LogicValue::from_u64(8, post_q)),
                ("hidden".into(), hidden_value(wide, preceding_wide)?),
            ])),
        });
    }
    check_rtl_trace(&rtl, &trace)?;

    // A post-edge output mistakenly sampled during the reset cycle must fail.
    let out = trace.samples[0].outputs.get_mut("out").unwrap();
    let before_reset = std::mem::replace(out, LogicValue::from_u64(8, 7));
    assert!(check_rtl_trace(&rtl, &trace).is_err());
    trace.samples[0].outputs.insert("out".into(), before_reset);

    // Corrupt only bit 64 of a hidden lane. Every output still matches.
    let regs = trace.samples[1].next_state.as_mut().unwrap();
    regs.insert("hidden".into(), hidden_value(a ^ (1u128 << 64), c)?);
    assert!(check_rtl_trace(&rtl, &trace).is_err());
    Ok(())
}
