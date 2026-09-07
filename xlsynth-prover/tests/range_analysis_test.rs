// SPDX-License-Identifier: Apache-2.0

//! Optional, bounded formal checks using the existing Bitwuzla feature.

#![cfg(feature = "has-bitwuzla")]

use std::collections::HashMap;

use xlsynth_pir::ir::{self, NodeRef, Type};
use xlsynth_pir::ir_eval::{EvalObserver, FnEvalResult, SelectEvent, eval_fn_with_observer};
use xlsynth_pir::ir_random::{
    EntropySource, OperationSet, RandomFnOptions, RandomOperation, StopPolicy, generate_fn,
};
use xlsynth_pir::range_analysis::{IntervalSet, RangeValue, analyze_fn};
use xlsynth_pir::{FnBuilder, IrBits, IrValue};
use xlsynth_prover::prover::translate::{get_fn_inputs, ir_to_smt_with_node_terms};
use xlsynth_prover::prover::types::{ProverFn, UfRegistry};
use xlsynth_prover::solver::bitwuzla::{Bitwuzla, BitwuzlaOptions};
use xlsynth_prover::solver::{BitVec, Response, Solver};

/// Deterministic SplitMix64 entropy without another test-only dependency.
struct SeededEntropy(u64);

impl EntropySource for SeededEntropy {
    fn is_depleted(&self) -> bool {
        false
    }

    fn take_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}

#[derive(Clone, Debug)]
struct Claim {
    node_ref: NodeRef,
    facts: RangeValue,
}

#[derive(Debug)]
struct Violation {
    text_id: usize,
    value: IrValue,
}

#[derive(Debug)]
enum ProofResult {
    Proved,
    Unknown,
    Disproved {
        arguments: Vec<IrValue>,
        violations: Vec<Violation>,
    },
}

/// Creates a width-exact SMT constant without host-integer conversion.
fn constant<S: Solver>(solver: &mut S, value: &IrBits) -> BitVec<S::Term> {
    if value.get_bit_count() == 0 {
        return solver.zero_width();
    }
    let mut text = String::with_capacity(value.get_bit_count() + 2);
    text.push_str("#b");
    for bit in (0..value.get_bit_count()).rev() {
        text.push(if value.get_bit(bit).unwrap() {
            '1'
        } else {
            '0'
        });
    }
    solver.from_raw_str(value.get_bit_count(), &text)
}

/// Encodes exact membership, preserving interval holes and aggregate layout.
fn membership<S: Solver>(
    solver: &mut S,
    value: &BitVec<S::Term>,
    ty: &Type,
    facts: &RangeValue,
) -> BitVec<S::Term> {
    assert_eq!(value.get_width(), ty.bit_count());
    match (ty, facts) {
        (Type::Bits(width), RangeValue::Bits(intervals)) => {
            assert_eq!(*width, intervals.width());
            if intervals.is_full() {
                return solver.one(1);
            }
            let mut result = solver.zero(1);
            // The only nonempty zero-width interval set is full, handled above.
            for interval in intervals.intervals() {
                let lower = constant(solver, interval.lower());
                let upper = constant(solver, interval.upper());
                let at_least = solver.uge(value, &lower);
                let at_most = solver.ule(value, &upper);
                let inside = solver.and(&at_least, &at_most);
                result = solver.or(&result, &inside);
            }
            result
        }
        (Type::Tuple(types), RangeValue::Tuple(elements)) => {
            assert_eq!(types.len(), elements.len());
            let mut result = solver.one(1);
            // Prover tuples place the first element at the most-significant
            // end.
            let mut offset = value.get_width();
            for (ty, facts) in types.iter().zip(elements) {
                let width = ty.bit_count();
                offset -= width;
                let element = solver.slice(value, offset, width);
                let valid = membership(solver, &element, ty, facts);
                result = solver.and(&result, &valid);
            }
            assert_eq!(offset, 0);
            result
        }
        (Type::Array(array), RangeValue::Array(elements)) => {
            assert_eq!(array.element_count, elements.len());
            let width = array.element_type.bit_count();
            let mut result = solver.one(1);
            // Prover arrays place the first element at the least-significant
            // end.
            for (index, facts) in elements.iter().enumerate() {
                let element = solver.slice(value, index * width, width);
                let valid = membership(solver, &element, &array.element_type, facts);
                result = solver.and(&result, &valid);
            }
            result
        }
        (Type::Token, RangeValue::Token) => solver.one(1),
        _ => panic!("range/type shape mismatch: {ty} versus {facts:?}"),
    }
}

struct ReplayObserver<'a> {
    claims: &'a [Claim],
    seen: Vec<bool>,
    violations: Vec<Violation>,
}

impl EvalObserver for ReplayObserver<'_> {
    fn on_select(&mut self, _event: SelectEvent) {
        // Node-value callbacks already observe the concrete selected value.
    }

    fn on_node_value(&mut self, node_ref: NodeRef, text_id: usize, value: &IrValue) {
        self.seen[node_ref.index] = true;
        if let Some(claim) = self.claims.iter().find(|claim| claim.node_ref == node_ref)
            && !claim.facts.contains(value)
        {
            self.violations.push(Violation {
                text_id,
                value: value.clone(),
            });
        }
    }
}

/// Asks whether any claimed node/leaf can fall outside its interval union.
fn prove_claims(function: &ir::Fn, claims: &[Claim]) -> ProofResult {
    let mut options = BitwuzlaOptions::new();
    options.set_time_limit_per(250);
    options.set_seed(0);
    let mut solver = Bitwuzla::new(&options).unwrap();
    let inputs = get_fn_inputs(&mut solver, ProverFn::new(function, None), None);
    let smt = ir_to_smt_with_node_terms(
        &mut solver,
        &inputs,
        &HashMap::new(),
        &UfRegistry {
            ufs: HashMap::new(),
        },
    );
    assert!(smt.smt_fn.assertions.is_empty());
    let mut any_violation = solver.zero(1);
    for claim in claims {
        let node = function.get_node(claim.node_ref);
        let term = smt.node_terms.get(&node.text_id).unwrap();
        let inside = membership(&mut solver, &term.bitvec, &node.ty, &claim.facts);
        let outside = solver.not(&inside);
        any_violation = solver.or(&any_violation, &outside);
    }
    solver.assert(&any_violation).unwrap();
    match solver.check().unwrap() {
        Response::Unsat => ProofResult::Proved,
        Response::Unknown => ProofResult::Unknown,
        Response::Sat => {
            let arguments: Vec<_> = smt
                .smt_fn
                .inputs
                .iter()
                .map(|input| solver.get_value(&input.bitvec, input.ir_type).unwrap())
                .collect();
            let mut observer = ReplayObserver {
                claims,
                seen: vec![false; function.nodes.len()],
                violations: Vec::new(),
            };
            for (&node_ref, argument) in function.params.iter().zip(&arguments) {
                observer.on_node_value(node_ref, function.get_node(node_ref).text_id, argument);
            }
            let evaluated = eval_fn_with_observer(function, &arguments, Some(&mut observer));
            assert!(
                matches!(evaluated, FnEvalResult::Success(_)),
                "counterexample replay failed: {evaluated:?}; arguments={arguments:?}\n{function}"
            );
            for claim in claims {
                assert!(
                    observer.seen[claim.node_ref.index],
                    "counterexample replay missed node {}\n{function}",
                    function.get_node(claim.node_ref).text_id,
                );
            }
            assert!(
                !observer.violations.is_empty(),
                "SMT counterexample did not replay; arguments={arguments:?}; claims={claims:?}\n{function}"
            );
            ProofResult::Disproved {
                arguments,
                violations: observer.violations,
            }
        }
    }
}

fn analysis_claims(function: &ir::Fn) -> Vec<Claim> {
    analyze_fn(function)
        .unwrap()
        .iter()
        .map(|(node_ref, facts)| Claim {
            node_ref,
            facts: facts.clone(),
        })
        .collect()
}

#[test]
fn random_graph_range_claims_are_formally_sound() {
    let options = RandomFnOptions {
        max_params: 3,
        max_nodes: 20,
        max_bit_width: 8,
        max_aggregate_leaves: 8,
        max_array_length: 3,
        max_tuple_length: 3,
        // Zero-width arithmetic and empty-case selects are not generally
        // supported by the SMT translator. Directed tests below still check
        // zero-width values and aggregates in the membership encoder.
        allow_zero_width_bits: false,
        allow_arbitrary_width_multiply: true,
        allow_empty_case_sel: false,
        allow_gate: true,
        allow_extension_ops: true,
        // The SMT translator has no partial-product-pair implementation;
        // concrete range fuzzing covers those operations independently.
        enabled_operations: OperationSet::new(
            OperationSet::all_supported()
                .iter()
                .filter(|op| !matches!(op, RandomOperation::Umulp | RandomOperation::Smulp)),
        ),
        ..Default::default()
    };
    let mut proved = 0;
    let mut unknown = 0;
    for sample in 0..256 {
        let seed = 0x7261_6e67_6500_0000 | sample;
        let function = generate_fn(
            &mut SeededEntropy(seed),
            &options,
            StopPolicy::ExactBodyNodes(16),
        )
        .unwrap()
        .function;
        let claims = analysis_claims(&function);
        match prove_claims(&function, &claims) {
            ProofResult::Proved => proved += 1,
            ProofResult::Unknown => {
                // A bounded solver query may time out; this is inconclusive,
                // not evidence that the analysis or the sample is unsound.
                unknown += 1;
                eprintln!("range proof inconclusive for seed {seed:#x}\n{function}");
            }
            ProofResult::Disproved {
                arguments,
                violations,
            } => panic!(
                "range analysis is unsound for seed {seed:#x}: arguments={arguments:?}, violations={violations:?}\n{function}"
            ),
        }
    }
    assert!(proved > 0, "all bounded range proofs were inconclusive");
    eprintln!("range formal checks: {proved} proved, {unknown} inconclusive");
}

#[test]
fn formal_checker_observes_dead_nodes_and_replays_counterexamples() {
    let mut builder = FnBuilder::new("dead_node_claim");
    let input = builder.param("x", Type::Bits(4)).unwrap();
    let mask = builder.literal(IrValue::make_ubits(4, 9).unwrap()).unwrap();
    let dead = builder.and_all(&[input, mask]).unwrap();
    builder.set_name(dead, "dead").unwrap();
    let function = builder.build(input).unwrap();
    let mut claims = analysis_claims(&function);
    assert!(matches!(
        prove_claims(&function, &claims),
        ProofResult::Proved
    ));
    let node_ref = NodeRef {
        index: function
            .nodes
            .iter()
            .position(|node| node.name.as_deref() == Some("dead"))
            .unwrap(),
    };
    claims
        .iter_mut()
        .find(|claim| claim.node_ref == node_ref)
        .unwrap()
        .facts = RangeValue::Bits(IntervalSet::singleton(IrBits::zero(4)));
    let ProofResult::Disproved { violations, .. } = prove_claims(&function, &claims) else {
        panic!("the deliberately false dead-node claim should be disproved");
    };
    assert_eq!(violations.len(), 1);
    assert_eq!(violations[0].text_id, function.get_node(node_ref).text_id);
    assert!(!violations[0].value.as_bits().unwrap().is_zero());
}

#[test]
fn formal_checker_preserves_tuple_array_and_zero_width_layout() {
    let value = IrValue::make_tuple(&[
        IrValue::make_ubits(3, 5).unwrap(),
        IrValue::make_array(&[
            IrValue::make_ubits(2, 1).unwrap(),
            IrValue::make_ubits(2, 2).unwrap(),
        ])
        .unwrap(),
        IrValue::make_tuple(&[
            IrValue::from_bits(&IrBits::zero(0)),
            IrValue::make_tuple(&[]),
            IrValue::Token,
        ]),
    ]);
    let mut builder = FnBuilder::new("aggregate_claim");
    let literal = builder.literal(value).unwrap();
    let function = builder.build(literal).unwrap();
    let mut claims = analysis_claims(&function);
    assert!(matches!(
        prove_claims(&function, &claims),
        ProofResult::Proved
    ));
    let claim = claims
        .iter_mut()
        .find(|claim| claim.node_ref == function.ret_node_ref.unwrap())
        .unwrap();
    let RangeValue::Tuple(tuple) = &mut claim.facts else {
        panic!("tuple facts");
    };
    let RangeValue::Array(array) = &mut tuple[1] else {
        panic!("array facts");
    };
    array[1] = RangeValue::Bits(IntervalSet::singleton(IrBits::make_ubits(2, 1).unwrap()));
    assert!(matches!(
        prove_claims(&function, &claims),
        ProofResult::Disproved { .. }
    ));
}

#[test]
fn formal_checker_checks_interval_holes_and_zero_width_bottom() {
    let mut builder = FnBuilder::new("interval_holes");
    let input = builder.param("x", Type::Bits(4)).unwrap();
    let function = builder.build(input).unwrap();
    let claims = vec![Claim {
        node_ref: function.params[0],
        facts: RangeValue::Bits(
            IntervalSet::from_intervals(
                4,
                [
                    (IrBits::zero(4), IrBits::zero(4)),
                    (IrBits::all_ones(4), IrBits::all_ones(4)),
                ],
            )
            .unwrap(),
        ),
    }];
    assert!(matches!(
        prove_claims(&function, &claims),
        ProofResult::Disproved { .. }
    ));

    let mut builder = FnBuilder::new("zero_width_bottom");
    let input = builder.param("x", Type::Bits(0)).unwrap();
    let function = builder.build(input).unwrap();
    let claims = vec![Claim {
        node_ref: function.params[0],
        facts: RangeValue::Bits(IntervalSet::empty(0)),
    }];
    assert!(matches!(
        prove_claims(&function, &claims),
        ProofResult::Disproved { .. }
    ));
}
