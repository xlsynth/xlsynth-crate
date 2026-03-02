// SPDX-License-Identifier: Apache-2.0

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use xlsynth_g8r::gatify::ir2gate::{self, GatifyOptions};
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_pir::ir_parser::Parser;

#[derive(Debug, Clone, Arbitrary)]
struct MulConstSample {
    width: u8,
    constant: u16,
    literal_on_lhs: bool,
}

fn build_ir_text(sample: &MulConstSample) -> String {
    let width = usize::from(sample.width).clamp(1, 16);
    let modulus = if width == 16 { 1u64 << 16 } else { 1u64 << width };
    let constant = u64::from(sample.constant) % modulus;
    let umul = if sample.literal_on_lhs {
        "umul(c, x, id=3)"
    } else {
        "umul(x, c, id=3)"
    };
    format!(
        "package sample

top fn mul_const(x: bits[{width}] id=1) -> bits[{width}] {{
  c: bits[{width}] = literal(value={constant}, id=2)
  ret p: bits[{width}] = {umul}
}}
"
    )
}

fuzz_target!(|sample: MulConstSample| {
    let _ = env_logger::builder().is_test(true).try_init();

    let ir_text = build_ir_text(&sample);
    let mut parser = Parser::new(&ir_text);
    let pkg = parser
        .parse_and_validate_package()
        .expect("constructed mul-by-const IR should parse");
    let pir_fn = pkg.get_top_fn().expect("top fn");

    let _gatify = ir2gate::gatify(
        pir_fn,
        GatifyOptions {
            fold: true,
            hash: true,
            check_equivalence: true,
            adder_mapping: AdderMapping::default(),
            mul_adder_mapping: None,
            range_info: None,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
        },
    )
    .expect("gatify with built-in mul-by-const lowering");

});
