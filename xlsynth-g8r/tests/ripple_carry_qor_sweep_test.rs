// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::get_summary_stats::{AigStats, get_aig_stats};
use xlsynth_g8r::aig::{AigBitVector, GateFn};
use xlsynth_g8r::check_equivalence;
use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions, ReductionKind};
use xlsynth_g8r::ir2gate_utils::gatify_add_ripple_carry;

#[derive(Debug, PartialEq, Eq)]
struct RippleCarryQorRow {
    bit_count: usize,
    old_and_nodes: usize,
    old_depth: usize,
    public_and_nodes: usize,
    public_depth: usize,
}

fn gatify_add_ripple_carry_old_maj3(
    gb: &mut GateBuilder,
    lhs: &AigBitVector,
    rhs: &AigBitVector,
    mut c_in: xlsynth_g8r::aig::AigOperand,
) -> (xlsynth_g8r::aig::AigOperand, AigBitVector) {
    assert_eq!(lhs.get_bit_count(), rhs.get_bit_count());
    let mut sum_bits = Vec::new();
    for i in 0..lhs.get_bit_count() {
        let lhs_i = lhs.get_lsb(i);
        let rhs_i = rhs.get_lsb(i);
        let sum = gb.add_xor_nary(&[*lhs_i, *rhs_i, c_in], ReductionKind::Linear);
        let c_out_0 = gb.add_and_binary(*lhs_i, *rhs_i);
        let c_out_1 = gb.add_and_binary(*rhs_i, c_in);
        let c_out_2 = gb.add_and_binary(*lhs_i, c_in);
        c_in = gb.add_or_nary(&[c_out_0, c_out_1, c_out_2], ReductionKind::Linear);
        sum_bits.push(sum);
    }
    (c_in, AigBitVector::from_lsb_is_index_0(&sum_bits))
}

fn make_ripple_carry_old(bit_count: usize) -> GateFn {
    let mut gb = GateBuilder::new(
        format!("ripple_carry_old_{bit_count}b"),
        GateBuilderOptions::opt(),
    );
    let lhs = gb.add_input("lhs".to_string(), bit_count);
    let rhs = gb.add_input("rhs".to_string(), bit_count);
    let c_in = *gb.add_input("c_in".to_string(), 1).get_lsb(0);
    let (c_out, sum) = gatify_add_ripple_carry_old_maj3(&mut gb, &lhs, &rhs, c_in);
    gb.add_output("c_out".to_string(), AigBitVector::from_bit(c_out));
    gb.add_output("sum".to_string(), sum);
    gb.build()
}

fn make_ripple_carry_public(bit_count: usize) -> GateFn {
    let mut gb = GateBuilder::new(
        format!("ripple_carry_public_{bit_count}b"),
        GateBuilderOptions::opt(),
    );
    let lhs = gb.add_input("lhs".to_string(), bit_count);
    let rhs = gb.add_input("rhs".to_string(), bit_count);
    let c_in = *gb.add_input("c_in".to_string(), 1).get_lsb(0);
    let (c_out, sum) = gatify_add_ripple_carry(&lhs, &rhs, c_in, Some("ripple_carry"), &mut gb);
    gb.add_output("c_out".to_string(), AigBitVector::from_bit(c_out));
    gb.add_output("sum".to_string(), sum);
    gb.build()
}

fn stats_for_gate_fn(gate_fn: &GateFn) -> AigStats {
    get_aig_stats(gate_fn)
}

fn gather_ripple_carry_qor_rows() -> Vec<RippleCarryQorRow> {
    let mut got = Vec::new();
    for bit_count in [1usize, 2, 3, 4, 5, 8, 16, 32, 64] {
        let old = make_ripple_carry_old(bit_count);
        let public = make_ripple_carry_public(bit_count);
        check_equivalence::prove_same_gate_fn_via_ir(&old, &public)
            .expect("old and public ripple-carry lowerings should be equivalent");
        let old_stats = stats_for_gate_fn(&old);
        let public_stats = stats_for_gate_fn(&public);
        got.push(RippleCarryQorRow {
            bit_count,
            old_and_nodes: old_stats.and_nodes,
            old_depth: old_stats.max_depth,
            public_and_nodes: public_stats.and_nodes,
            public_depth: public_stats.max_depth,
        });
    }
    got
}

#[test]
fn test_ripple_carry_propagate_generate_qor_and_equivalence_sweep() {
    let _ = env_logger::builder().is_test(true).try_init();

    let got = gather_ripple_carry_qor_rows();

    for row in &got {
        assert!(
            row.public_and_nodes < row.old_and_nodes,
            "expected propagate/generate ripple carry to reduce AND nodes: {:?}",
            row
        );
        assert!(
            row.public_depth <= row.old_depth,
            "expected propagate/generate ripple carry not to increase depth: {:?}",
            row
        );
    }

    #[rustfmt::skip]
    let want: &[RippleCarryQorRow] = &[
        RippleCarryQorRow { bit_count: 1, old_and_nodes: 11, old_depth: 4, public_and_nodes: 9, public_depth: 4 },
        RippleCarryQorRow { bit_count: 2, old_and_nodes: 22, old_depth: 6, public_and_nodes: 18, public_depth: 6 },
        RippleCarryQorRow { bit_count: 3, old_and_nodes: 33, old_depth: 9, public_and_nodes: 27, public_depth: 8 },
        RippleCarryQorRow { bit_count: 4, old_and_nodes: 44, old_depth: 12, public_and_nodes: 36, public_depth: 10 },
        RippleCarryQorRow { bit_count: 5, old_and_nodes: 55, old_depth: 15, public_and_nodes: 45, public_depth: 12 },
        RippleCarryQorRow { bit_count: 8, old_and_nodes: 88, old_depth: 24, public_and_nodes: 72, public_depth: 18 },
        RippleCarryQorRow { bit_count: 16, old_and_nodes: 176, old_depth: 48, public_and_nodes: 144, public_depth: 34 },
        RippleCarryQorRow { bit_count: 32, old_and_nodes: 352, old_depth: 96, public_and_nodes: 288, public_depth: 66 },
        RippleCarryQorRow { bit_count: 64, old_and_nodes: 704, old_depth: 192, public_and_nodes: 576, public_depth: 130 },
    ];
    assert_eq!(got, want);
}
