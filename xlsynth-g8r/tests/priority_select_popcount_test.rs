// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;

use rand::{SeedableRng, rngs::StdRng};
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::aig_sim::gate_sim::{self, Collect};
use xlsynth_g8r::gatify::ir2gate::{GatifyOptions, gatify_prepared_fn};
use xlsynth_pir::ir::{self, NodePayload, Type};
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::ir_range_info::{IrRangeInfo, NodeRangeInfo};
use xlsynth_pir::ir_value_utils::flatten_ir_value_to_lsb0_bits_for_type;
use xlsynth_pir::known_bits;
use xlsynth_pir::random_inputs::generate_mixed_argument_sets_with_rng;
use xlsynth_pir::{FnBuilder, IrBits, IrValue};

#[derive(Clone, Copy, Debug)]
enum Selector {
    AtMostOne,
    ExactlyOne,
    Unconstrained,
    AtLeastOne,
}

/// Keeps the data/default symbolic while choosing a selector population.
fn make_function(selector_kind: Selector, ty: Type) -> ir::Fn {
    let mut builder = FnBuilder::new("priority");
    let input_width = match selector_kind {
        Selector::AtMostOne | Selector::AtLeastOne => 3,
        Selector::ExactlyOne => 2,
        Selector::Unconstrained => 4,
    };
    let input = builder.param("control", Type::Bits(input_width)).unwrap();
    let cases = (0..4)
        .map(|i| builder.param(&format!("case_{i}"), ty.clone()).unwrap())
        .collect::<Vec<_>>();
    let default = builder.param("fallback", ty).unwrap();
    let selector = match selector_kind {
        Selector::AtMostOne | Selector::ExactlyOne => builder.decode(input, Some(4)).unwrap(),
        Selector::AtLeastOne => {
            let one = builder
                .literal(IrValue::from_bits(&IrBits::bool(true)))
                .unwrap();
            builder.concat(&[input, one]).unwrap()
        }
        Selector::Unconstrained => input,
    };
    let result = builder.priority_select(selector, &cases, default).unwrap();
    builder.build(result).unwrap()
}

/// Matches the gate lowering's tuple/array packing, including zero-width
/// leaves.
fn flattened(value: &IrValue, ty: &Type) -> IrBits {
    let mut bits = Vec::new();
    flatten_ir_value_to_lsb0_bits_for_type(value, ty, &mut bits).unwrap();
    IrBits::from_lsb_is_0(&bits)
}

#[test]
fn priority_select_population_lowering_matches_concrete_evaluation() {
    let mut rng = StdRng::seed_from_u64(0xface_1024);
    for kind in [
        Selector::AtMostOne,
        Selector::ExactlyOne,
        Selector::Unconstrained,
        Selector::AtLeastOne,
    ] {
        for ty in [
            Type::Bits(0),
            Type::Bits(1),
            Type::Bits(2),
            Type::Bits(8),
            Type::Bits(129),
            Type::Tuple(vec![Box::new(Type::Bits(2)), Box::new(Type::Bits(3))]),
            Type::new_array(Type::Bits(3), 2),
        ] {
            let f = make_function(kind, ty);
            let facts = known_bits::analyze_fn(&f).unwrap();
            let NodePayload::PrioritySel { selector, .. } =
                f.get_node(f.ret_node_ref.unwrap()).payload
            else {
                unreachable!();
            };
            let population = facts.bits(selector).unwrap();
            match kind {
                Selector::AtMostOne => {
                    assert_eq!((population.min_ones(), population.max_ones()), (0, 1))
                }
                Selector::ExactlyOne => {
                    assert_eq!((population.min_ones(), population.max_ones()), (1, 1))
                }
                Selector::AtLeastOne => {
                    assert_eq!((population.min_ones(), population.max_ones()), (1, 4))
                }
                Selector::Unconstrained => {
                    assert_eq!((population.min_ones(), population.max_ones()), (0, 4))
                }
            }
            let gates = gatify_prepared_fn(
                &f,
                GatifyOptions {
                    range_info: Some(Arc::new(IrRangeInfo::from_known_bits(&facts))),
                    ..GatifyOptions::all_opts_disabled()
                },
            )
            .unwrap()
            .gate_fn;
            // Cover all selector inputs independently of the shared mixed data
            // sampling, including multi-hot inputs when priority is required.
            for (i, mut args) in generate_mixed_argument_sets_with_rng(&f, &mut rng, 128)
                .into_iter()
                .enumerate()
            {
                let width = f.get_node_ty(f.params[0]).bit_count();
                args[0] = IrValue::make_ubits(width, (i % (1 << width)) as u64).unwrap();
                let expected = match eval_fn(&f, &args) {
                    FnEvalResult::Success(result) => flattened(&result.value, &f.ret_ty),
                    failure => panic!("unexpected interpretation failure: {failure:?}"),
                };
                let inputs = args
                    .iter()
                    .zip(&f.params)
                    .map(|(arg, &node)| flattened(arg, f.get_node_ty(node)))
                    .collect::<Vec<_>>();
                assert_eq!(
                    gate_sim::eval(&gates, &inputs, Collect::None).outputs,
                    vec![expected],
                    "selector={kind:?}, args={args:?}"
                );
            }
        }
    }
}

#[test]
fn population_facts_remove_priority_logic_without_constant_selector_bits() {
    for kind in [Selector::AtMostOne, Selector::ExactlyOne] {
        for width in [1, 2, 8, 129] {
            // Isolate the lowering cost from the structure that proves the
            // population bound; arbitrary selectors are constrained here.
            let f = make_function(Selector::Unconstrained, Type::Bits(width));
            let NodePayload::PrioritySel { selector, .. } =
                f.get_node(f.ret_node_ref.unwrap()).payload
            else {
                unreachable!();
            };
            let min = usize::from(matches!(kind, Selector::ExactlyOne));
            let known = known_bits::KnownBits::unknown(4)
                .with_popcount_bounds(min, 1)
                .unwrap();
            assert_eq!(known.known_bit_count(), 0);
            let info = IrRangeInfo::from_node_range_info(BTreeMap::from([(
                f.get_node(selector).text_id,
                NodeRangeInfo {
                    known_bits: Some(known),
                    intervals: None,
                    unsigned_min: None,
                    unsigned_max: None,
                },
            )]));
            let baseline = gatify_prepared_fn(&f, GatifyOptions::all_opts_disabled()).unwrap();
            let specialized = gatify_prepared_fn(
                &f,
                GatifyOptions {
                    range_info: Some(Arc::new(info)),
                    ..GatifyOptions::all_opts_disabled()
                },
            )
            .unwrap();
            let before = get_aig_stats(&baseline.gate_fn);
            let after = get_aig_stats(&specialized.gate_fn);
            // At width one, preserving the zero/default case ties the mux
            // chain's gate count; the other configurations remove gates.
            if matches!(kind, Selector::AtMostOne) && width == 1 {
                assert_eq!(after.and_nodes, before.and_nodes);
            } else {
                assert!(
                    after.and_nodes < before.and_nodes,
                    "selector={kind:?}, width={width}, before={}, after={}",
                    before.and_nodes,
                    after.and_nodes
                );
            }
        }
    }
}

#[cfg(feature = "has-bitwuzla")]
#[test]
fn population_lowering_is_formally_equivalent() {
    for kind in [
        Selector::AtMostOne,
        Selector::ExactlyOne,
        Selector::AtLeastOne,
    ] {
        for ty in [
            Type::Bits(1),
            Type::Bits(8),
            Type::Tuple(vec![Box::new(Type::Bits(3)), Box::new(Type::Bits(5))]),
        ] {
            let f = make_function(kind, ty);
            let facts = known_bits::analyze_fn(&f).unwrap();
            let specialized = gatify_prepared_fn(
                &f,
                GatifyOptions {
                    range_info: Some(Arc::new(IrRangeInfo::from_known_bits(&facts))),
                    ..GatifyOptions::all_opts_disabled()
                },
            )
            .unwrap();
            xlsynth_g8r::check_equivalence::validate_same_fn(&f, &specialized.gate_fn).unwrap();
        }
    }
}

#[test]
fn xls_analysis_adapter_preserves_selector_population() {
    for kind in [Selector::AtMostOne, Selector::ExactlyOne] {
        let f = make_function(kind, Type::Bits(8));
        let package_text = format!("package test\n\ntop {f}");
        let package = xlsynth::IrPackage::parse_ir(&package_text, None).unwrap();
        let analysis = package.create_ir_analysis().unwrap();
        let info = IrRangeInfo::build_from_analysis(&analysis, &f).unwrap();
        let NodePayload::PrioritySel { selector, .. } = f.get_node(f.ret_node_ref.unwrap()).payload
        else {
            unreachable!();
        };
        let known = info
            .get(f.get_node(selector).text_id)
            .unwrap()
            .known_bits
            .as_ref()
            .unwrap();
        assert_eq!(known.known_bit_count(), 0);
        assert_eq!(known.max_ones(), 1);
        assert_eq!(
            known.min_ones(),
            usize::from(matches!(kind, Selector::ExactlyOne))
        );
    }
}
