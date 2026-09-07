// SPDX-License-Identifier: Apache-2.0

use xlsynth_pir::ir::{self, Binop, Block, NodeGraph, NodeRef, Type};
use xlsynth_pir::ir_eval::{EvalObserver, FnEvalResult, SelectEvent, eval_fn_with_observer};
use xlsynth_pir::known_bits::{KnownBitsAnalysis, KnownValue, analyze_block, analyze_fn};
use xlsynth_pir::{BValue, FnBuilder, IrBits, IrValue};

fn bits(width: usize, value: u64) -> IrValue {
    IrValue::make_ubits(width, value).unwrap()
}

fn bit_value(width: usize, bit: impl Fn(usize) -> bool) -> IrValue {
    IrValue::from_bits(&IrBits::from_lsb_is_0(
        &(0..width).map(bit).collect::<Vec<_>>(),
    ))
}

fn named(graph: &NodeGraph, name: &str) -> NodeRef {
    NodeRef {
        index: graph
            .nodes
            .iter()
            .position(|node| node.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("missing node {name}")),
    }
}

/// Accumulates exact per-bit concrete hulls without using operation transfers.
struct ConcreteHullObserver<'a, 'ir> {
    analysis: &'a KnownBitsAnalysis<'ir>,
    hulls: Vec<Option<KnownValue>>,
}

impl EvalObserver for ConcreteHullObserver<'_, '_> {
    fn on_select(&mut self, _event: SelectEvent) {
        // The node callback already observes the selected value.
    }

    fn on_node_value(&mut self, node_ref: NodeRef, node_text_id: usize, value: &IrValue) {
        let facts = self.analysis.get(node_ref).unwrap();
        assert!(
            facts.contains(value),
            "{} node id={node_text_id} excludes a concrete value",
            self.analysis.graph().name,
        );
        let concrete = KnownValue::constant(value);
        self.hulls[node_ref.index] = Some(match self.hulls[node_ref.index].take() {
            Some(previous) => previous.join(&concrete).unwrap(),
            None => concrete,
        });
    }
}

/// Checks every node, requiring exact hulls only for exhaustive input sets.
fn check_claims(function: &ir::Fn, arguments: &[Vec<IrValue>], exact: bool) {
    let analysis = analyze_fn(function).unwrap();
    let block = Block::from_function(function.clone(), None).unwrap();
    let block_analysis = analyze_block(&block).unwrap();
    for (reference, facts) in analysis.iter() {
        assert_eq!(block_analysis.get(reference), Some(facts));
    }
    let mut observer = ConcreteHullObserver {
        analysis: &analysis,
        hulls: vec![None; function.nodes.len()],
    };
    for args in arguments {
        assert_eq!(args.len(), function.params.len());
        for (&param, value) in function.params.iter().zip(args) {
            observer.on_node_value(param, function.get_node(param).text_id, value);
        }
        match eval_fn_with_observer(function, args, Some(&mut observer)) {
            FnEvalResult::Success(_) => {
                // The observer checked every computed value, including the
                // return.
            }
            FnEvalResult::Failure(failure) => {
                panic!("unexpected interpreter failure: {failure:?}");
            }
        }
    }
    for (reference, facts) in analysis.iter() {
        let observed = observer.hulls[reference.index].as_ref().unwrap();
        if exact {
            assert_eq!(
                facts,
                observed,
                "{} node id={} is not the exact concrete hull",
                function.name,
                function.get_node(reference).text_id,
            );
        }
    }
}

fn shift(builder: &mut FnBuilder, op: Binop, arg: BValue, amount: BValue) -> BValue {
    match op {
        Binop::Shll => builder.shll(arg, amount),
        Binop::Shrl => builder.shrl(arg, amount),
        Binop::Shra => builder.shra(arg, amount),
        _ => unreachable!("only shift operations are used in these fixtures"),
    }
    .unwrap()
}

#[test]
fn large_constant_aggregates_bypass_only_routing_work_limits() {
    let width = 32_769;
    let a_value = bit_value(width, |i| i % 7 == 1);
    let b_value = bit_value(width, |i| i % 7 != 1);
    let pair_value = IrValue::make_array(&[a_value.clone(), b_value.clone()]).unwrap();
    let mut builder = FnBuilder::new("large_constant_routes");
    let aggregate = builder
        .literal(IrValue::make_tuple(&[
            pair_value,
            IrValue::Token,
            bits(0, 0),
            IrValue::make_tuple(&[]),
        ]))
        .unwrap();
    builder.set_name(aggregate, "aggregate").unwrap();
    let identity = builder.identity(aggregate).unwrap();
    builder.set_name(identity, "identity").unwrap();
    let extracted = builder.tuple_index(identity, 0).unwrap();
    builder.set_name(extracted, "extracted").unwrap();
    let a = builder.literal(a_value).unwrap();
    let b = builder.literal(b_value).unwrap();
    let array = builder.array(Type::Bits(width), &[a, b]).unwrap();
    builder.set_name(array, "array").unwrap();
    let reversed = builder.array(Type::Bits(width), &[b, a]).unwrap();
    let empty = builder.array(Type::Bits(width), &[]).unwrap();
    let concat = builder.array_concat(&[array, empty, extracted]).unwrap();
    builder.set_name(concat, "concat").unwrap();
    let nested = builder
        .array(Type::new_array(Type::Bits(width), 2), &[array, reversed])
        .unwrap();
    let index = builder.literal(bits(2, 1)).unwrap();
    let out_of_bounds = builder.literal(bit_value(129, |i| i == 128)).unwrap();
    let indexed = builder.array_index(nested, index).unwrap();
    builder.set_name(indexed, "indexed").unwrap();
    let clamped = builder.array_index(nested, out_of_bounds).unwrap();
    builder.set_name(clamped, "clamped").unwrap();
    let no_indices = builder.array_index_multi(nested, &[]).unwrap();
    builder.set_name(no_indices, "no_indices").unwrap();
    let updated = builder.array_update(nested, array, &[index]).unwrap();
    builder.set_name(updated, "updated").unwrap();
    let unchanged = builder
        .array_update(nested, array, &[out_of_bounds])
        .unwrap();
    builder.set_name(unchanged, "unchanged").unwrap();
    let slice = builder.array_slice(nested, index, 3).unwrap();
    builder.set_name(slice, "slice").unwrap();
    let clamped_slice = builder.array_slice(nested, out_of_bounds, 2).unwrap();
    builder.set_name(clamped_slice, "clamped_slice").unwrap();
    let result = builder
        .tuple(&[
            aggregate,
            identity,
            extracted,
            array,
            concat,
            indexed,
            clamped,
            no_indices,
            updated,
            unchanged,
            slice,
            clamped_slice,
        ])
        .unwrap();
    builder.set_name(result, "result").unwrap();
    let function = builder.build(result).unwrap();
    for name in [
        "aggregate",
        "identity",
        "extracted",
        "array",
        "concat",
        "indexed",
        "clamped",
        "no_indices",
        "updated",
        "unchanged",
        "slice",
        "clamped_slice",
        "result",
    ] {
        assert!(
            function
                .get_node_ty(named(&function, name))
                .checked_bit_count()
                .unwrap()
                > 65_536,
            "{name} must exercise the aggregate work limit"
        );
    }
    check_claims(&function, &[vec![]], true);
}

#[test]
fn large_aggregate_routes_preserve_partial_facts_and_empty_leaves() {
    let width = 32_769;
    let mut builder = FnBuilder::new("large_partial_routes");
    let input = builder.param("input", Type::Bits(2)).unwrap();
    let widened = builder.zero_extend(input, width).unwrap();
    let ones = builder
        .literal(IrValue::from_bits(&IrBits::all_ones(width)))
        .unwrap();
    let array = builder.array(Type::Bits(width), &[widened, ones]).unwrap();
    let token = builder.literal(IrValue::Token).unwrap();
    let empty_bits = builder.literal(bits(0, 0)).unwrap();
    let empty_array = builder.array(Type::Bits(width), &[]).unwrap();
    let tuple = builder
        .tuple(&[array, token, empty_bits, empty_array])
        .unwrap();
    let identity = builder.identity(tuple).unwrap();
    let extracted = builder.tuple_index(identity, 0).unwrap();
    let result = builder
        .array_concat(&[array, extracted, empty_array])
        .unwrap();
    let function = builder.build(result).unwrap();
    let arguments = (0..4).map(|input| vec![bits(2, input)]).collect::<Vec<_>>();
    check_claims(&function, &arguments, true);
}

#[test]
fn wide_fixed_shifts_preserve_exact_partial_facts_across_word_boundaries() {
    let width = 513;
    let mut builder = FnBuilder::new("wide_fixed_shifts");
    let low = builder.param("low", Type::Bits(3)).unwrap();
    let sign = builder.param("sign", Type::Bits(1)).unwrap();
    let middle = builder.literal(bits(width - 4, 0)).unwrap();
    let input = builder.concat(&[sign, middle, low]).unwrap();
    let mut results = Vec::new();
    for amount in [0, 1, 63, 64, 65, 256, 512, 513, 1024] {
        let amount = builder.literal(bits(16, amount)).unwrap();
        for op in [Binop::Shll, Binop::Shrl, Binop::Shra] {
            results.push(shift(&mut builder, op, input, amount));
        }
    }
    let enormous = builder.literal(bit_value(129, |i| i == 128)).unwrap();
    for op in [Binop::Shll, Binop::Shrl, Binop::Shra] {
        results.push(shift(&mut builder, op, input, enormous));
    }
    let result = builder.tuple(&results).unwrap();
    let function = builder.build(result).unwrap();
    let arguments = (0..8)
        .flat_map(|low| (0..2).map(move |sign| vec![bits(3, low), bits(1, sign)]))
        .collect::<Vec<_>>();
    check_claims(&function, &arguments, true);
}

#[test]
fn wide_saturated_shift_domains_preserve_unknown_arithmetic_sign() {
    let width = 513;
    let mut builder = FnBuilder::new("wide_saturated_shifts");
    let sign = builder.param("sign", Type::Bits(1)).unwrap();
    let amount = builder.param("amount", Type::Bits(2)).unwrap();
    let data = builder
        .literal(bit_value(width - 1, |i| i % 3 == 1))
        .unwrap();
    let input = builder.concat(&[sign, data]).unwrap();
    let narrow_amount = builder.zero_extend(amount, 16).unwrap();
    let narrow_high = builder.literal(bits(16, 1024)).unwrap();
    let narrow_amount = builder.or(narrow_amount, narrow_high).unwrap();
    let wide_amount = builder.zero_extend(amount, 129).unwrap();
    let wide_high = builder.literal(bit_value(129, |i| i == 128)).unwrap();
    let wide_amount = builder.or(wide_amount, wide_high).unwrap();
    let mut results = Vec::new();
    for amount in [narrow_amount, wide_amount] {
        for op in [Binop::Shll, Binop::Shrl, Binop::Shra] {
            results.push(shift(&mut builder, op, input, amount));
        }
    }
    let result = builder.tuple(&results).unwrap();
    let function = builder.build(result).unwrap();
    let arguments = (0..2)
        .flat_map(|sign| (0..4).map(move |amount| vec![bits(1, sign), bits(2, amount)]))
        .collect::<Vec<_>>();
    check_claims(&function, &arguments, true);
}

#[test]
fn wide_zero_and_arithmetic_all_ones_ignore_dynamic_shift_amounts() {
    let width = 513;
    let mut builder = FnBuilder::new("wide_shift_identities");
    let amount = builder.param("amount", Type::Bits(2)).unwrap();
    let zero = builder.literal(bits(width, 0)).unwrap();
    let ones = builder
        .literal(IrValue::from_bits(&IrBits::all_ones(width)))
        .unwrap();
    let mut results = Vec::new();
    for op in [Binop::Shll, Binop::Shrl, Binop::Shra] {
        results.push(shift(&mut builder, op, zero, amount));
    }
    results.push(builder.shra(ones, amount).unwrap());
    let result = builder.tuple(&results).unwrap();
    let function = builder.build(result).unwrap();
    let arguments = (0..4)
        .map(|amount| vec![bits(2, amount)])
        .collect::<Vec<_>>();
    check_claims(&function, &arguments, true);
}

#[test]
fn wide_slice_updates_handle_clipping_out_of_bounds_and_empty_updates() {
    let width = 513;
    let mut builder = FnBuilder::new("wide_slice_updates");
    let input = builder.param("input", Type::Bits(3)).unwrap();
    let update = builder.param("update", Type::Bits(2)).unwrap();
    let dynamic_start = builder.param("dynamic_start", Type::Bits(2)).unwrap();
    let input = builder.zero_extend(input, width).unwrap();
    let upper = builder
        .literal(IrValue::from_bits(&IrBits::all_ones(255)))
        .unwrap();
    let update = builder.concat(&[upper, update]).unwrap();
    let mut results = Vec::new();
    for start in [0, 1, 63, 64, 65, 256, 510, 512, 513] {
        let start = builder.literal(bits(16, start)).unwrap();
        results.push(builder.bit_slice_update(input, start, update).unwrap());
    }
    let huge = builder.literal(bit_value(129, |i| i == 128)).unwrap();
    results.push(builder.bit_slice_update(input, huge, update).unwrap());
    let unknown_low = builder.zero_extend(dynamic_start, 129).unwrap();
    let certainly_out_of_bounds = builder.or(unknown_low, huge).unwrap();
    results.push(
        builder
            .bit_slice_update(input, certainly_out_of_bounds, update)
            .unwrap(),
    );
    let empty = builder.literal(bits(0, 0)).unwrap();
    results.push(
        builder
            .bit_slice_update(input, dynamic_start, empty)
            .unwrap(),
    );
    results.push(
        builder
            .bit_slice_update(empty, dynamic_start, update)
            .unwrap(),
    );
    let result = builder.tuple(&results).unwrap();
    let function = builder.build(result).unwrap();
    let arguments = (0..8)
        .flat_map(|input| {
            (0..4).flat_map(move |update| {
                (0..4).map(move |start| vec![bits(3, input), bits(2, update), bits(2, start)])
            })
        })
        .collect::<Vec<_>>();
    check_claims(&function, &arguments, true);
}

#[test]
fn expensive_dynamic_cases_keep_conservative_unknown_fallbacks() {
    let mut builder = FnBuilder::new("dynamic_work_limits");
    let choice = builder.param("choice", Type::Bits(1)).unwrap();
    let value = builder.literal(bit_value(513, |i| i % 7 == 2)).unwrap();
    let update = builder.literal(bits(4, 9)).unwrap();
    let mut results = Vec::new();
    let mut fallback_names = Vec::new();
    for (op, name) in [
        (Binop::Shll, "left"),
        (Binop::Shrl, "right"),
        (Binop::Shra, "arithmetic"),
    ] {
        let shifted = shift(&mut builder, op, value, choice);
        builder.set_name(shifted, name).unwrap();
        results.push(shifted);
        fallback_names.push(name);
    }
    let updated = builder.bit_slice_update(value, choice, update).unwrap();
    builder.set_name(updated, "updated").unwrap();
    results.push(updated);
    fallback_names.push("updated");
    let result = builder.tuple(&results).unwrap();
    let function = builder.build(result).unwrap();
    let analysis = analyze_fn(&function).unwrap();
    for name in fallback_names {
        assert_eq!(
            analysis
                .bits(named(&function, name))
                .unwrap()
                .known_bit_count(),
            0,
            "{name} should retain the expensive dynamic fallback"
        );
    }
    check_claims(&function, &[vec![bits(1, 0)], vec![bits(1, 1)]], false);
}
