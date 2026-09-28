// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use rand::{SeedableRng, rngs::StdRng};
use xlsynth_g8r::aig::GateFn;
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig_sim::gate_sim::{self, Collect};
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify_prepared_fn};
use xlsynth_pir::ir::{self, NodePayload, Type};
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::ir_range_info::IrRangeInfo;
use xlsynth_pir::ir_value_utils::flatten_ir_value_to_lsb0_bits_for_type;
use xlsynth_pir::random_inputs::generate_mixed_argument_sets_with_rng;
use xlsynth_pir::{FnBuilder, IrBits, IrValue, known_bits};

fn lower(f: &ir::Fn, facts: IrRangeInfo) -> GateFn {
    gatify_prepared_fn(
        f,
        GatifyOptions {
            range_info: Some(Arc::new(facts)),
            ..GatifyOptions::all_opts_disabled()
        },
    )
    .unwrap()
    .gate_fn
}

fn flatten(value: &IrValue, ty: &Type) -> IrBits {
    let mut bits = Vec::new();
    flatten_ir_value_to_lsb0_bits_for_type(value, ty, &mut bits).unwrap();
    IrBits::from_lsb_is_0(&bits)
}

/// Exhausts the small control input while mixing uniform and special data.
fn check_concrete(f: &ir::Fn, gates: &GateFn) {
    let mut rng = StdRng::seed_from_u64(0x5e1ec7);
    let width = f.get_node_ty(f.params[0]).bit_count();
    for (i, mut args) in generate_mixed_argument_sets_with_rng(f, &mut rng, 128)
        .into_iter()
        .enumerate()
    {
        args[0] = IrValue::make_ubits(width, (i % (1 << width)) as u64).unwrap();
        let expected = match eval_fn(f, &args) {
            FnEvalResult::Success(result) => flatten(&result.value, &f.ret_ty),
            failure => panic!("unexpected interpretation failure: {failure:?}"),
        };
        let inputs = args
            .iter()
            .zip(&f.params)
            .map(|(v, &n)| flatten(v, f.get_node_ty(n)))
            .collect::<Vec<_>>();
        assert_eq!(
            gate_sim::eval(gates, &inputs, Collect::None).outputs,
            vec![expected]
        );
    }
}

/// A truncated decoder permits zero; a full decoder produces exactly one.
fn decoded_one_hot(exactly_one: bool, lsb_priority: bool) -> ir::Fn {
    let mut b = FnBuilder::new("decoded_one_hot");
    let input = b
        .param("input", Type::Bits(if exactly_one { 2 } else { 3 }))
        .unwrap();
    let decoded = b.decode(input, Some(4)).unwrap();
    let result = b.one_hot(decoded, lsb_priority).unwrap();
    b.build(result).unwrap()
}

#[test]
fn one_hot_population_shortcut_preserves_zero_flag_and_priority() {
    for exactly_one in [false, true] {
        for lsb_priority in [false, true] {
            let f = decoded_one_hot(exactly_one, lsb_priority);
            // Population alone must suffice, even without range information
            // or a single fixed one bit in the operand.
            let known = known_bits::analyze_fn(&f).unwrap();
            let specialized = lower(&f, IrRangeInfo::from_known_bits(&known));
            check_concrete(&f, &specialized);
            let baseline = gatify_prepared_fn(&f, GatifyOptions::all_opts_disabled()).unwrap();
            assert!(
                get_aig_stats(&specialized).and_nodes < get_aig_stats(&baseline.gate_fn).and_nodes
            );
        }
    }
    for width in [0, 1, 4] {
        for lsb_priority in [false, true] {
            let mut b = FnBuilder::new("unconstrained_one_hot");
            let input = b.param("input", Type::Bits(width)).unwrap();
            let result = b.one_hot(input, lsb_priority).unwrap();
            let f = b.build(result).unwrap();
            check_concrete(
                &f,
                &lower(
                    &f,
                    IrRangeInfo::from_known_bits(&known_bits::analyze_fn(&f).unwrap()),
                ),
            );
        }
    }
}

#[cfg(feature = "has-bitwuzla")]
#[test]
fn one_hot_population_lowering_is_formally_equivalent() {
    for exactly_one in [false, true] {
        for lsb_priority in [false, true] {
            let f = decoded_one_hot(exactly_one, lsb_priority);
            xlsynth_g8r::check_equivalence::validate_same_fn(
                &f,
                &lower(
                    &f,
                    IrRangeInfo::from_known_bits(&known_bits::analyze_fn(&f).unwrap()),
                ),
            )
            .unwrap();
        }
    }
}

#[test]
fn xls_adapter_queries_population_for_one_hot_operands() {
    for exactly_one in [false, true] {
        let f = decoded_one_hot(exactly_one, false);
        let package =
            xlsynth::IrPackage::parse_ir(&format!("package test\n\ntop {f}"), None).unwrap();
        let analysis = package.create_ir_analysis().unwrap();
        let facts = IrRangeInfo::build_from_analysis(&analysis, &f).unwrap();
        let NodePayload::OneHot { arg, .. } = f.get_node(f.ret_node_ref.unwrap()).payload else {
            unreachable!();
        };
        let known = facts
            .get(f.get_node(arg).text_id)
            .unwrap()
            .known_bits
            .as_ref()
            .unwrap();
        assert_eq!(known.known_bit_count(), 0);
        assert_eq!(
            (known.min_ones(), known.max_ones()),
            (usize::from(exactly_one), 1)
        );
        let specialized = gatify_prepared_fn(
            &f,
            GatifyOptions {
                range_info: Some(facts),
                ..GatifyOptions::all_opts_disabled()
            },
        )
        .unwrap();
        check_concrete(&f, &specialized.gate_fn);
    }
}
