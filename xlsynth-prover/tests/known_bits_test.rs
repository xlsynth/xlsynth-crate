// SPDX-License-Identifier: Apache-2.0

//! Bounded formal checks of known masks and population bounds.

#![cfg(feature = "has-bitwuzla")]

use std::collections::HashMap;

use xlsynth_pir::ir::{self, NodeRef, Type};
use xlsynth_pir::ir_eval::{EvalObserver, FnEvalResult, SelectEvent, eval_fn_with_observer};
use xlsynth_pir::ir_random::{
    EntropySource, OperationSet, RandomFnOptions, RandomOperation, StopPolicy, generate_fn,
};
use xlsynth_pir::known_bits::{KnownBits, KnownValue, analyze_fn};
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
    facts: KnownValue,
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

/// Counts every input bit in a vector wide enough to represent its full width.
fn population_count<S: Solver>(solver: &mut S, value: &BitVec<S::Term>) -> BitVec<S::Term> {
    let count_width = (usize::BITS - value.get_width().leading_zeros()).max(1) as usize;
    let mut count = solver.zero(count_width);
    for bit in 0..value.get_width() {
        let term = solver.slice(value, bit, 1);
        let term = solver.zero_extend_to(&term, count_width);
        count = solver.add(&count, &term);
    }
    count
}

/// Encodes a structural count without truncating it to a narrower host word.
fn count_constant<S: Solver>(solver: &mut S, count: usize, width: usize) -> BitVec<S::Term> {
    let text = format!("#b{count:0width$b}");
    solver.from_raw_str(width, &text)
}

/// Encodes both mask agreement and population bounds in the prover's layout.
fn membership<S: Solver>(
    solver: &mut S,
    value: &BitVec<S::Term>,
    ty: &Type,
    facts: &KnownValue,
) -> BitVec<S::Term> {
    assert_eq!(value.get_width(), ty.bit_count());
    match (ty, facts) {
        (Type::Bits(width), KnownValue::Bits(facts)) => {
            assert_eq!(*width, facts.bit_count());
            if *width == 0 {
                assert_eq!((facts.min_ones(), facts.max_ones()), (0, 0));
                return solver.one(1);
            }
            let mask = constant(solver, facts.mask());
            let expected = constant(solver, facts.value());
            let masked = solver.and(value, &mask);
            let matches_mask = solver.eq(&masked, &expected);
            let count = population_count(solver, value);
            let lower = count_constant(solver, facts.min_ones(), count.get_width());
            let upper = count_constant(solver, facts.max_ones(), count.get_width());
            let at_least = solver.uge(&count, &lower);
            let at_most = solver.ule(&count, &upper);
            let in_bounds = solver.and(&at_least, &at_most);
            solver.and(&matches_mask, &in_bounds)
        }
        (Type::Tuple(types), KnownValue::Tuple(elements)) => {
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
        (Type::Array(array), KnownValue::Array(elements)) => {
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
        (Type::Token, KnownValue::Token) => solver.one(1),
        _ => panic!("known-bits/type shape mismatch: {ty} versus {facts:?}"),
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

/// Asks whether any node or aggregate leaf can violate a mask or count claim.
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
fn random_graph_known_bit_and_population_claims_are_formally_sound() {
    let options = RandomFnOptions {
        max_params: 3,
        max_nodes: 20,
        max_bit_width: 8,
        max_aggregate_leaves: 8,
        max_array_length: 3,
        max_tuple_length: 3,
        // General zero-width arithmetic and partial-product pairs are not
        // supported by the SMT translator; concrete fuzzing still covers them.
        allow_zero_width_bits: false,
        allow_arbitrary_width_multiply: true,
        allow_empty_case_sel: false,
        allow_gate: true,
        allow_extension_ops: true,
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
        let seed = 0x6b6e_6f77_6e00_0000 | sample;
        let function = generate_fn(
            &mut SeededEntropy(seed),
            &options,
            StopPolicy::ExactBodyNodes(16),
        )
        .unwrap()
        .function;
        match prove_claims(&function, &analysis_claims(&function)) {
            ProofResult::Proved => proved += 1,
            ProofResult::Unknown => {
                // Solver timeout is inconclusive, not a soundness failure.
                unknown += 1;
                eprintln!("known-bits proof inconclusive for seed {seed:#x}\n{function}");
            }
            ProofResult::Disproved {
                arguments,
                violations,
            } => panic!(
                "known-bits analysis is unsound for seed {seed:#x}: arguments={arguments:?}, violations={violations:?}\n{function}"
            ),
        }
    }
    assert!(
        proved > 0,
        "all bounded known-bits proofs were inconclusive"
    );
    eprintln!("known-bits formal checks: {proved} proved, {unknown} inconclusive");
}

#[test]
fn population_claim_counterexample_replays_even_without_known_mask_bits() {
    let mut builder = FnBuilder::new("false_population");
    let x = builder.param("x", Type::Bits(4)).unwrap();
    let function = builder.build(x).unwrap();
    let claims = vec![Claim {
        node_ref: function.params[0],
        facts: KnownValue::Bits(KnownBits::unknown(4).with_popcount_bounds(1, 1).unwrap()),
    }];
    assert!(claims[0].facts.as_bits().unwrap().mask().is_zero());
    let ProofResult::Disproved {
        arguments,
        violations,
    } = prove_claims(&function, &claims)
    else {
        panic!("an arbitrary four-bit input does not always have population one");
    };
    assert_eq!(violations.len(), 1);
    assert_eq!(violations[0].text_id, function.get_param(0).text_id);
    assert_eq!(violations[0].value, arguments[0]);
    let value = arguments[0].as_bits().unwrap();
    let count = (0..value.get_bit_count())
        .filter(|&bit| value.get_bit(bit).unwrap())
        .count();
    assert_ne!(count, 1);
}

#[test]
fn population_encoder_retains_counts_above_one_word_and_zero_width_values() {
    for width in [0, 1, 64, 65, 129, 257] {
        let mut builder = FnBuilder::new("wide_population");
        let literal = builder
            .literal(IrValue::from_bits(&IrBits::all_ones(width)))
            .unwrap();
        let function = builder.build(literal).unwrap();
        assert!(matches!(
            prove_claims(&function, &analysis_claims(&function)),
            ProofResult::Proved
        ));
        if width > 1 {
            let claims = vec![Claim {
                node_ref: function.ret_node_ref.unwrap(),
                // A valid abstract fact that is false for the actual literal;
                // at width 257 this requires a nine-bit population counter.
                facts: KnownValue::Bits(
                    KnownBits::unknown(width)
                        .with_popcount_bounds(0, width - 1)
                        .unwrap(),
                ),
            }];
            assert!(matches!(
                prove_claims(&function, &claims),
                ProofResult::Disproved { .. }
            ));
        }
    }
}

#[test]
fn formal_checker_observes_dead_masks_and_aggregate_count_layout() {
    let mut builder = FnBuilder::new("dead_mask");
    let x = builder.param("x", Type::Bits(4)).unwrap();
    let mask = builder.literal(IrValue::make_ubits(4, 9).unwrap()).unwrap();
    let dead = builder.and(x, mask).unwrap();
    builder.set_name(dead, "dead").unwrap();
    let function = builder.build(x).unwrap();
    let mut claims = analysis_claims(&function);
    assert!(matches!(
        prove_claims(&function, &claims),
        ProofResult::Proved
    ));
    let claim = claims
        .iter_mut()
        .find(|claim| function.get_node(claim.node_ref).name.as_deref() == Some("dead"))
        .unwrap();
    claim.facts = KnownValue::Bits(KnownBits::constant(&IrBits::zero(4)));
    assert!(matches!(
        prove_claims(&function, &claims),
        ProofResult::Disproved { .. }
    ));

    let value = IrValue::make_tuple(&[
        IrValue::make_ubits(3, 0).unwrap(),
        IrValue::make_array(&[
            IrValue::make_ubits(2, 1).unwrap(),
            IrValue::make_ubits(2, 3).unwrap(),
        ])
        .unwrap(),
        IrValue::make_tuple(&[
            IrValue::from_bits(&IrBits::zero(0)),
            IrValue::make_tuple(&[]),
            IrValue::Token,
        ]),
    ]);
    let mut builder = FnBuilder::new("aggregate_population");
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
    let KnownValue::Tuple(tuple) = &mut claim.facts else {
        panic!("tuple facts")
    };
    let KnownValue::Array(array) = &mut tuple[1] else {
        panic!("array facts")
    };
    array[1] = KnownValue::Bits(KnownBits::unknown(2).with_popcount_bounds(1, 1).unwrap());
    assert!(matches!(
        prove_claims(&function, &claims),
        ProofResult::Disproved { .. }
    ));
}
