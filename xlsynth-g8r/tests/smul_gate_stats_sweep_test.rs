// SPDX-License-Identifier: Apache-2.0

use xlsynth::IrBits;
use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::aig::get_summary_stats::{SummaryStats, get_summary_stats};
use xlsynth_g8r::aig_sim::gate_sim::{self, Collect};
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify};
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_pir::ir_parser;

#[derive(Clone, Debug, PartialEq, Eq)]
struct SmulWideningRow {
    operand_width: usize,
    output_width: usize,
    sign_extend_live_nodes: usize,
    sign_extend_deepest_path: usize,
    native_signed_live_nodes: usize,
    native_signed_deepest_path: usize,
}

fn build_smul_ir_text(operand_width: usize, output_width: usize) -> String {
    format!(
        r#"package sample

top fn smul_w{operand_width}_o{output_width}(lhs: bits[{operand_width}] id=1, rhs: bits[{operand_width}] id=2) -> bits[{output_width}] {{
  ret p: bits[{output_width}] = smul(lhs, rhs, id=3)
}}
"#
    )
}

fn build_sign_extend_then_umul_ir_text(operand_width: usize, output_width: usize) -> String {
    format!(
        r#"package sample

top fn smul_sign_extend_w{operand_width}_o{output_width}(lhs: bits[{operand_width}] id=1, rhs: bits[{operand_width}] id=2) -> bits[{output_width}] {{
  lhs_ext: bits[{output_width}] = sign_ext(lhs, new_bit_count={output_width}, id=3)
  rhs_ext: bits[{output_width}] = sign_ext(rhs, new_bit_count={output_width}, id=4)
  ret p: bits[{output_width}] = umul(lhs_ext, rhs_ext, id=5)
}}
"#
    )
}

fn gatify_options() -> GatifyOptions {
    GatifyOptions {
        fold: true,
        hash: true,
        check_equivalence: false,
        adder_mapping: AdderMapping::RippleCarry,
        mul_adder_mapping: None,
        range_info: None,
        enable_rewrite_carry_out: false,
        enable_rewrite_prio_encode: false,
        enable_rewrite_nary_add: false,
        enable_rewrite_mask_low: false,
        array_index_lowering_strategy: Default::default(),
        unsafe_gatify_gate_operation: false,
    }
}

fn gate_fn_for_ir_text(ir_text: &str) -> GateFn {
    let mut parser = ir_parser::Parser::new(ir_text);
    let ir_package = parser.parse_and_validate_package().unwrap();
    let ir_fn = ir_package.get_top_fn().unwrap();
    gatify(ir_fn, gatify_options()).unwrap().gate_fn
}

fn stats_for_gate_fn(gate_fn: &GateFn) -> SummaryStats {
    get_summary_stats(gate_fn)
}

fn bit_mask(width: usize) -> u64 {
    assert!(width > 0 && width < 64);
    (1u64 << width) - 1
}

fn signed_value(raw: u64, width: usize) -> i128 {
    let sign_bit = 1u64 << (width - 1);
    let masked = raw & bit_mask(width);
    if masked & sign_bit == 0 {
        i128::from(masked)
    } else {
        i128::from(masked) - (1i128 << width)
    }
}

fn expected_smul_bits(
    lhs_raw: u64,
    rhs_raw: u64,
    operand_width: usize,
    output_width: usize,
) -> IrBits {
    let lhs = signed_value(lhs_raw, operand_width);
    let rhs = signed_value(rhs_raw, operand_width);
    let modulus = 1i128 << output_width;
    let product = (lhs * rhs).rem_euclid(modulus) as u64;
    IrBits::make_ubits(output_width, product).unwrap()
}

fn interesting_values(width: usize) -> Vec<u64> {
    let mask = bit_mask(width);
    let mut values = vec![
        0,
        1,
        2,
        mask,
        mask - 1,
        1u64 << (width - 1),
        (1u64 << (width - 1)) - 1,
        0x55aa_33cc_u64 & mask,
        0xaa55_cc33_u64 & mask,
    ];
    for i in 0..width {
        values.push(1u64 << i);
        values.push((!((1u64 << i) - 1)) & mask);
    }
    values.sort_unstable();
    values.dedup();
    values
}

fn validate_smul_by_simulation(gate_fn: &GateFn, operand_width: usize, output_width: usize) {
    let values = if operand_width <= 5 {
        (0..(1u64 << operand_width)).collect()
    } else {
        interesting_values(operand_width)
    };

    for &lhs in &values {
        for &rhs in &values {
            let got = gate_sim::eval(
                gate_fn,
                &[
                    IrBits::make_ubits(operand_width, lhs).unwrap(),
                    IrBits::make_ubits(operand_width, rhs).unwrap(),
                ],
                Collect::None,
            );
            assert_eq!(got.outputs.len(), 1);
            let want = expected_smul_bits(lhs, rhs, operand_width, output_width);
            assert_eq!(
                got.outputs[0], want,
                "smul mismatch for operand_width={operand_width} output_width={output_width} \
                 lhs=0x{lhs:x} rhs=0x{rhs:x}"
            );
        }
    }
}

fn gather_smul_widening_rows() -> Vec<SmulWideningRow> {
    let mut got = Vec::new();
    for (operand_width, output_width) in [
        (2usize, 3usize),
        (2, 4),
        (3, 4),
        (3, 5),
        (3, 6),
        (4, 5),
        (4, 6),
        (4, 8),
        (5, 6),
        (5, 8),
        (5, 10),
        (8, 10),
        (8, 12),
        (8, 14),
        (8, 16),
        (12, 18),
        (12, 24),
    ] {
        let sign_extend = gate_fn_for_ir_text(&build_sign_extend_then_umul_ir_text(
            operand_width,
            output_width,
        ));
        let native_signed = gate_fn_for_ir_text(&build_smul_ir_text(operand_width, output_width));

        validate_smul_by_simulation(&native_signed, operand_width, output_width);
        if operand_width <= 3 {
            check_equivalence::prove_same_gate_fn_via_ir(&sign_extend, &native_signed)
                .expect("sign-extend and native signed multiply lowerings should be equivalent");
        }

        let sign_extend_stats = stats_for_gate_fn(&sign_extend);
        let native_signed_stats = stats_for_gate_fn(&native_signed);
        got.push(SmulWideningRow {
            operand_width,
            output_width,
            sign_extend_live_nodes: sign_extend_stats.live_nodes,
            sign_extend_deepest_path: sign_extend_stats.deepest_path,
            native_signed_live_nodes: native_signed_stats.live_nodes,
            native_signed_deepest_path: native_signed_stats.deepest_path,
        });
    }
    got
}

#[test]
fn test_smul_widening_qor_and_equivalence_sweep() {
    let _ = env_logger::builder().is_test(true).try_init();

    let got = gather_smul_widening_rows();

    for row in &got {
        assert!(
            row.native_signed_live_nodes <= row.sign_extend_live_nodes,
            "expected native signed multiply to reduce live nodes: {:?}",
            row
        );
        assert!(
            row.native_signed_live_nodes * row.native_signed_deepest_path
                < row.sign_extend_live_nodes * row.sign_extend_deepest_path,
            "expected native signed multiply to reduce live_nodes*depth product: {:?}",
            row
        );
    }

    #[rustfmt::skip]
    let want: &[SmulWideningRow] = &[
        SmulWideningRow { operand_width: 2, output_width: 3, sign_extend_live_nodes: 21, sign_extend_deepest_path: 8, native_signed_live_nodes: 15, native_signed_deepest_path: 5 },
        SmulWideningRow { operand_width: 2, output_width: 4, sign_extend_live_nodes: 38, sign_extend_deepest_path: 12, native_signed_live_nodes: 19, native_signed_deepest_path: 6 },
        SmulWideningRow { operand_width: 3, output_width: 4, sign_extend_live_nodes: 48, sign_extend_deepest_path: 12, native_signed_live_nodes: 42, native_signed_deepest_path: 9 },
        SmulWideningRow { operand_width: 3, output_width: 5, sign_extend_live_nodes: 78, sign_extend_deepest_path: 16, native_signed_live_nodes: 55, native_signed_deepest_path: 10 },
        SmulWideningRow { operand_width: 3, output_width: 6, sign_extend_live_nodes: 114, sign_extend_deepest_path: 18, native_signed_live_nodes: 62, native_signed_deepest_path: 12 },
        SmulWideningRow { operand_width: 4, output_width: 5, sign_extend_live_nodes: 87, sign_extend_deepest_path: 16, native_signed_live_nodes: 81, native_signed_deepest_path: 13 },
        SmulWideningRow { operand_width: 4, output_width: 6, sign_extend_live_nodes: 129, sign_extend_deepest_path: 18, native_signed_live_nodes: 106, native_signed_deepest_path: 14 },
        SmulWideningRow { operand_width: 4, output_width: 8, sign_extend_live_nodes: 231, sign_extend_deepest_path: 22, native_signed_live_nodes: 132, native_signed_deepest_path: 18 },
        SmulWideningRow { operand_width: 5, output_width: 6, sign_extend_live_nodes: 138, sign_extend_deepest_path: 18, native_signed_live_nodes: 132, native_signed_deepest_path: 17 },
        SmulWideningRow { operand_width: 5, output_width: 8, sign_extend_live_nodes: 258, sign_extend_deepest_path: 22, native_signed_live_nodes: 202, native_signed_deepest_path: 21 },
        SmulWideningRow { operand_width: 5, output_width: 10, sign_extend_live_nodes: 400, sign_extend_deepest_path: 28, native_signed_live_nodes: 228, native_signed_deepest_path: 25 },
        SmulWideningRow { operand_width: 8, output_width: 10, sign_extend_live_nodes: 473, sign_extend_deepest_path: 28, native_signed_live_nodes: 440, native_signed_deepest_path: 26 },
        SmulWideningRow { operand_width: 8, output_width: 12, sign_extend_live_nodes: 671, sign_extend_deepest_path: 32, native_signed_live_nodes: 562, native_signed_deepest_path: 30 },
        SmulWideningRow { operand_width: 8, output_width: 14, sign_extend_live_nodes: 906, sign_extend_deepest_path: 36, native_signed_live_nodes: 640, native_signed_deepest_path: 34 },
        SmulWideningRow { operand_width: 8, output_width: 16, sign_extend_live_nodes: 1104, sign_extend_deepest_path: 40, native_signed_live_nodes: 674, native_signed_deepest_path: 38 },
        SmulWideningRow { operand_width: 12, output_width: 18, sign_extend_live_nodes: 1614, sign_extend_deepest_path: 44, native_signed_live_nodes: 1366, native_signed_deepest_path: 44 },
        SmulWideningRow { operand_width: 12, output_width: 24, sign_extend_live_nodes: 2508, sign_extend_deepest_path: 56, native_signed_live_nodes: 1624, native_signed_deepest_path: 56 },
    ];
    assert_eq!(got.as_slice(), want);
}
