// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gatify::ir2gate::{self, GatifyOptions};
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_pir::ir_parser;

#[derive(Clone, Copy, Debug)]
enum LiteralSide {
    Lhs,
    Rhs,
}

#[derive(Clone)]
struct ConstantCase {
    name: &'static str,
    value: u64,
}

fn bit_mask(width: usize) -> u64 {
    assert!(width > 0 && width <= 64);
    if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

fn alternating_pattern(width: usize, odd_bits_set: bool) -> u64 {
    let mut value = 0u64;
    for i in 0..width {
        let should_set = if odd_bits_set { i % 2 == 1 } else { i % 2 == 0 };
        if should_set {
            value |= 1u64 << i;
        }
    }
    value
}

fn dense_mid_run(width: usize) -> u64 {
    let mut value = 0u64;
    let start = width / 4;
    let end = (3 * width) / 4;
    for i in start..end {
        value |= 1u64 << i;
    }
    value
}

fn interesting_constants(width: usize) -> Vec<ConstantCase> {
    let mask = bit_mask(width);
    let mut cases: Vec<ConstantCase> = vec![
        ConstantCase {
            name: "zero",
            value: 0,
        },
        ConstantCase {
            name: "one",
            value: 1,
        },
        ConstantCase {
            name: "pow2_lsb1",
            value: 1u64 << 1,
        },
        ConstantCase {
            name: "pow2_mid",
            value: 1u64 << (width / 2),
        },
        ConstantCase {
            name: "pow2_msb",
            value: 1u64 << (width - 1),
        },
        ConstantCase {
            name: "pow2_minus1_small",
            value: (1u64 << 3) - 1,
        },
        ConstantCase {
            name: "pow2_minus1_mid",
            value: (1u64 << (width / 2)) - 1,
        },
        ConstantCase {
            name: "all_ones",
            value: mask,
        },
        ConstantCase {
            name: "alternating_10",
            value: alternating_pattern(width, /* odd_bits_set= */ true),
        },
        ConstantCase {
            name: "alternating_01",
            value: alternating_pattern(width, /* odd_bits_set= */ false),
        },
        ConstantCase {
            name: "sparse_ends",
            value: (1u64 << (width - 1)) | 1u64,
        },
        ConstantCase {
            name: "dense_mid_run",
            value: dense_mid_run(width),
        },
        ConstantCase {
            name: "mixed_irregular",
            value: (0x9d37_5a5a_u64 & mask),
        },
    ];
    for case in &mut cases {
        case.value &= mask;
    }
    cases
}

fn make_umul_const_ir(width: usize, constant: u64, literal_side: LiteralSide) -> String {
    let mul_expr = match literal_side {
        LiteralSide::Lhs => "umul(c, x, id=3)",
        LiteralSide::Rhs => "umul(x, c, id=3)",
    };
    format!(
        "package sample

top fn mul_const(x: bits[{width}] id=1) -> bits[{width}] {{
  c: bits[{width}] = literal(value={constant}, id=2)
  ret p: bits[{width}] = {mul_expr}
}}
"
    )
}

fn prove_case(width: usize, case: &ConstantCase, literal_side: LiteralSide) {
    let ir_text = make_umul_const_ir(width, case.value, literal_side);
    let mut parser = ir_parser::Parser::new(&ir_text);
    let pkg = parser.parse_and_validate_package().expect("parse/validate");
    let pir_fn = pkg.get_top_fn().expect("top fn");
    let output = ir2gate::gatify(
        pir_fn,
        GatifyOptions {
            fold: true,
            hash: true,
            check_equivalence: false,
            adder_mapping: AdderMapping::default(),
            mul_adder_mapping: None,
            range_info: None,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
            enable_rewrite_nary_add: false,
            array_index_lowering_strategy: Default::default(),
        },
    )
    .unwrap_or_else(|e| {
        panic!(
            "gatify failed for width={} case={} constant={} side={:?}: {}",
            width, case.name, case.value, literal_side, e
        )
    });
    check_equivalence::validate_same_fn(pir_fn, &output.gate_fn).unwrap_or_else(|e| {
        panic!(
            "equivalence proof failed for width={} case={} constant={} side={:?}: {}",
            width, case.name, case.value, literal_side, e
        )
    });
}

#[test]
fn prove_umul_const_lowering_equivalence_for_interesting_constants_width8() {
    let width = 8usize;
    for case in interesting_constants(width) {
        prove_case(width, &case, LiteralSide::Rhs);
        prove_case(width, &case, LiteralSide::Lhs);
    }
}

#[test]
fn prove_umul_const_lowering_equivalence_for_interesting_constants_width16() {
    let width = 16usize;
    for case in interesting_constants(width) {
        prove_case(width, &case, LiteralSide::Rhs);
        prove_case(width, &case, LiteralSide::Lhs);
    }
}
