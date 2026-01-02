// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use xlsynth::IrValue;

use super::types::{
    Assertion, AssertionSemantics, AssertionViolation, EquivResult, FnInput, FnInputs, FnOutput,
    IrTypedBitVec, ParamDomains, ProverFn, SmtFn, UfRegistry, UfSignature,
};
use super::{
    translate::{get_fn_inputs, ir_to_smt, ir_value_to_bv},
    uf::infer_merged_uf_signatures,
};
use crate::solver::{BitVec, Response, Solver};
use regex::Regex;
use xlsynth_pir::ir;
pub struct AlignedFnInputs<'a, R> {
    pub lhs: FnInputs<'a, R>,
    pub rhs: FnInputs<'a, R>,
    pub flattened: BitVec<R>,
}

/// Given the bit-vector representation of the two functions' inputs, this
/// function generates a flattened bit-vector, and use slices of the flattened
/// bit-vector for inputs to both functions.
///
/// We need this is to enhance term sharing. For example, when two functions
/// shares the same prelude, they will be compiled into exactly the same SMT
/// terms with the aligned inputs. The solver will then be able to just reason
/// about the behavior of the remaining parts of the functions.
pub fn align_fn_inputs<'a, S: Solver>(
    solver: &mut S,
    lhs_inputs: &FnInputs<'a, S::Term>,
    rhs_inputs: &FnInputs<'a, S::Term>,
    allow_flatten: bool,
) -> AlignedFnInputs<'a, S::Term> {
    let lhs_inputs_total_width = lhs_inputs.total_free_width();
    let rhs_inputs_total_width = rhs_inputs.total_free_width();
    assert_eq!(
        lhs_inputs_total_width, rhs_inputs_total_width,
        "LHS and RHS must have the same number of bits"
    );
    if !allow_flatten {
        assert_eq!(
            lhs_inputs.free_params_len(),
            rhs_inputs.free_params_len(),
            "LHS and RHS must have the same number of inputs"
        );
        for (l, r) in lhs_inputs
            .free_params()
            .iter()
            .zip(rhs_inputs.free_params().iter())
        {
            assert_eq!(
                l.ty, r.ty,
                "Input type mismatch for {} vs {}: {:?} vs {:?}",
                l.name, r.name, l.ty, r.ty
            );
        }
    }
    if lhs_inputs_total_width == 0 {
        return AlignedFnInputs {
            lhs: lhs_inputs.clone(),
            rhs: rhs_inputs.clone(),
            flattened: BitVec::ZeroWidth,
        };
    }
    let params_name = format!(
        "__flattened_params__{}__{}",
        lhs_inputs.name(),
        rhs_inputs.name()
    );
    let flattened = solver
        .declare_fresh(&params_name, lhs_inputs_total_width)
        .unwrap();
    // Split into individual param symbols
    let mut split_map =
        |inputs: &FnInputs<'a, S::Term>| -> HashMap<String, IrTypedBitVec<'a, S::Term>> {
            let mut m = HashMap::new();
            let mut params_iter = inputs.params().iter();

            if inputs.fixed_implicit_activation() {
                let itok = params_iter.next().unwrap();
                assert_eq!(itok.ty, ir::Type::Token);
                m.insert(
                    itok.name.clone(),
                    IrTypedBitVec {
                        ir_type: &itok.ty,
                        bitvec: solver.zero_width(),
                    },
                );
                let iact = params_iter.next().unwrap();
                assert_eq!(iact.ty, ir::Type::Bits(1));
                m.insert(
                    iact.name.clone(),
                    IrTypedBitVec {
                        ir_type: &iact.ty,
                        bitvec: solver.one(1),
                    },
                );
            }

            let mut offset = 0;
            for n in params_iter {
                let existing_bitvec = inputs.inputs.get(&n.name).unwrap();
                let new_bitvec = {
                    let w = existing_bitvec.bitvec.get_width();
                    let h = offset as i32 + (w as i32) - 1;
                    let l = offset as i32;
                    offset += w;
                    let extracted = solver.extract(&flattened, h, l);
                    let eq_expr = solver.eq(&extracted, &existing_bitvec.bitvec);
                    solver.assert(&eq_expr).unwrap();
                    extracted
                };
                m.insert(
                    n.name.clone(),
                    IrTypedBitVec {
                        ir_type: &n.ty,
                        bitvec: new_bitvec,
                    },
                );
            }
            m
        };
    let lhs_params = split_map(&lhs_inputs);
    let rhs_params = split_map(&rhs_inputs);

    AlignedFnInputs {
        lhs: FnInputs {
            prover_fn: lhs_inputs.prover_fn.clone(),
            inputs: lhs_params,
        },
        rhs: FnInputs {
            prover_fn: rhs_inputs.prover_fn.clone(),
            inputs: rhs_params,
        },
        flattened,
    }
}

fn check_aligned_fn_equiv_internal<'a, S: Solver>(
    solver: &mut S,
    lhs: &SmtFn<'a, S::Term>,
    rhs: &SmtFn<'a, S::Term>,
    assertion_semantics: AssertionSemantics,
    assert_label_include: Option<&Regex>,
) -> EquivResult {
    // --------------------------------------------
    // Helper: build a 1-bit "failed" flag for each
    // side indicating whether **any** assertion is
    // violated (active bit == 0).
    // --------------------------------------------

    // Optionally filter assertions by label before applying semantics.
    let lhs_asserts =
        super::assertion_filter::filter_assertions(&lhs.assertions, assert_label_include);
    let rhs_asserts =
        super::assertion_filter::filter_assertions(&rhs.assertions, assert_label_include);

    // Build a flag that is true iff ALL assertions are active (i.e., no violation).
    let mk_success_flag = |solver: &mut S, asserts: &[&Assertion<'_, S::Term>]| {
        if asserts.is_empty() {
            // No assertions → always succeed (true)
            solver.numerical(1, 1)
        } else {
            let mut acc: Option<BitVec<S::Term>> = None;
            for a in asserts.iter() {
                acc = Some(match acc {
                    None => a.active.clone(),
                    Some(prev) => solver.and(&prev, &a.active),
                });
            }
            acc.expect("acc populated")
        }
    };

    let lhs_pass = mk_success_flag(solver, &lhs_asserts);
    let rhs_pass = mk_success_flag(solver, &rhs_asserts);

    let lhs_failed = solver.not(&lhs_pass);
    let rhs_failed = solver.not(&rhs_pass);

    // diff of outputs
    let outputs_diff = solver.ne(&lhs.output.bitvec, &rhs.output.bitvec);

    // Build the overall condition to assert based on semantics
    let condition = match assertion_semantics {
        AssertionSemantics::Ignore => outputs_diff.clone(),
        AssertionSemantics::Never => {
            let any_failed = solver.or(&lhs_failed, &rhs_failed);
            solver.or(&outputs_diff, &any_failed)
        }
        AssertionSemantics::Same => {
            // fail status differs OR (both pass AND outputs differ)
            let fail_status_diff = solver.ne(&lhs_failed, &rhs_failed);
            let both_pass = solver.and(&lhs_pass, &rhs_pass);
            let diff_when_pass = solver.and(&outputs_diff, &both_pass);
            solver.or(&fail_status_diff, &diff_when_pass)
        }
        AssertionSemantics::Assume => {
            let both_pass = solver.and(&lhs_pass, &rhs_pass);
            solver.and(&both_pass, &outputs_diff)
        }
        AssertionSemantics::Implies => {
            // lhs passes AND (rhs fails OR outputs differ)
            let rhs_fail_or_diff = solver.or(&rhs_failed, &outputs_diff);
            solver.and(&lhs_pass, &rhs_fail_or_diff)
        }
    };

    solver.assert(&condition).unwrap();

    match solver.check().unwrap() {
        Response::Sat => {
            // Helper to fetch first violated assertion (if any)
            let get_assertion =
                |solver: &mut S, asserts: &[&Assertion<'_, S::Term>]| -> Option<(String, String)> {
                    for a in asserts.iter() {
                        let val = solver.get_value(&a.active, &ir::Type::Bits(1)).unwrap();
                        let bits = val.to_bits().unwrap();
                        let ok = bits.get_bit(0).unwrap();
                        if !ok {
                            return Some((a.message.to_string(), a.label.to_string()));
                        }
                    }
                    None
                };

            let (lhs_violation, rhs_violation) = match assertion_semantics {
                AssertionSemantics::Ignore => (None, None),
                _ => (
                    get_assertion(solver, &lhs_asserts),
                    get_assertion(solver, &rhs_asserts),
                ),
            };

            // Now we can safely create a helper closure that borrows `solver` again
            let get_value = |solver: &mut S, i: &IrTypedBitVec<'a, S::Term>| -> IrValue {
                solver.get_value(&i.bitvec, &i.ir_type).unwrap()
            };

            // Helper to produce FnOutput given optional violation info.
            let get_output =
                |solver: &mut S,
                 vio: Option<(String, String)>,
                 tbv: &IrTypedBitVec<'a, S::Term>| FnOutput {
                    value: get_value(solver, tbv),
                    assertion_violation: vio.map(|(msg, lbl)| AssertionViolation {
                        message: msg,
                        label: lbl,
                    }),
                };

            let build_inputs = |solver: &mut S, smt_fn: &SmtFn<'a, S::Term>| {
                smt_fn
                    .fn_ref
                    .params
                    .iter()
                    .zip(smt_fn.inputs.iter())
                    .map(|(p, i)| FnInput {
                        name: p.name.clone(),
                        value: get_value(solver, i),
                    })
                    .collect()
            };

            let lhs_inputs = build_inputs(solver, lhs);
            let rhs_inputs = build_inputs(solver, rhs);

            let lhs_output = get_output(solver, lhs_violation, &lhs.output);
            let rhs_output = get_output(solver, rhs_violation, &rhs.output);

            EquivResult::Disproved {
                lhs_inputs,
                rhs_inputs,
                lhs_output,
                rhs_output,
            }
        }
        Response::Unsat => EquivResult::Proved,
        Response::Unknown => panic!("Solver returned unknown result"),
    }
}

/// Prove equivalence between two IR functions, applying any provided domain
/// restrictions and UF mappings before delegating to the solver.
pub fn prove_ir_fn_equiv<'a, S: Solver>(
    solver_config: &S::Config,
    lhs: &ProverFn<'a>,
    rhs: &ProverFn<'a>,
    assertion_semantics: AssertionSemantics,
    assert_label_include: Option<&Regex>,
    allow_flatten: bool,
) -> EquivResult {
    let uf_signatures: HashMap<String, UfSignature> =
        if lhs.uf_map.is_empty() && rhs.uf_map.is_empty() {
            HashMap::new()
        } else {
            let lhs_pkg = match lhs.pkg_ref {
                Some(pkg) => pkg,
                None => {
                    return EquivResult::Error(
                        "UF map provided for LHS but no package reference available".to_string(),
                    );
                }
            };
            let rhs_pkg = match rhs.pkg_ref {
                Some(pkg) => pkg,
                None => {
                    return EquivResult::Error(
                        "UF map provided for RHS but no package reference available".to_string(),
                    );
                }
            };
            match infer_merged_uf_signatures(lhs_pkg, &lhs.uf_map, rhs_pkg, &rhs.uf_map) {
                Ok(sigs) => sigs,
                Err(e) => return EquivResult::Error(e),
            }
        };
    let mut solver = S::new(solver_config).unwrap();
    let fn_inputs_lhs = get_fn_inputs(&mut solver, lhs.clone(), Some("lhs"));
    let fn_inputs_rhs = get_fn_inputs(&mut solver, rhs.clone(), Some("rhs"));

    let mut assert_domains = |inputs: &FnInputs<'_, S::Term>, domains: &Option<ParamDomains>| {
        if let Some(dom) = domains {
            for p in inputs.params().iter() {
                if let Some(allowed) = dom.get(&p.name) {
                    if let Some(sym) = inputs.inputs.get(&p.name) {
                        let mut or_chain: Option<BitVec<S::Term>> = None;
                        for v in allowed {
                            let bv = ir_value_to_bv(&mut solver, v, &p.ty).bitvec;
                            let eq = solver.eq(&sym.bitvec, &bv);
                            or_chain = Some(match or_chain {
                                None => eq,
                                Some(prev) => solver.or(&prev, &eq),
                            });
                        }
                        if let Some(expr) = or_chain {
                            solver.assert(&expr).unwrap();
                        }
                    }
                }
            }
        }
    };

    assert_domains(&fn_inputs_lhs, &lhs.domains);
    assert_domains(&fn_inputs_rhs, &rhs.domains);

    let uf_registry = UfRegistry::from_uf_signatures(&mut solver, &uf_signatures);

    let aligned = align_fn_inputs(&mut solver, &fn_inputs_lhs, &fn_inputs_rhs, allow_flatten);
    let smt_lhs = ir_to_smt(&mut solver, &aligned.lhs, &lhs.uf_map, &uf_registry);
    let smt_rhs = ir_to_smt(&mut solver, &aligned.rhs, &rhs.uf_map, &uf_registry);
    check_aligned_fn_equiv_internal(
        &mut solver,
        &smt_lhs,
        &smt_rhs,
        assertion_semantics,
        assert_label_include,
    )
}

// Add parallel equivalence-checking strategies that were previously only
// implemented in the Boolector-specific backend.  These generic versions work
// with any `Solver` implementation by accepting a factory closure that can
// create fresh solver instances on demand.

/// Helper: create a variant of `f` that returns only the single output bit at
/// position `bit`.
fn make_bit_fn(f: &ir::Fn, bit: usize) -> ir::Fn {
    use ir::{Node, NodePayload, NodeRef, Type};

    let mut nf = f.clone();
    let ret_ref = nf.ret_node_ref.expect("Function must have a return node");
    let slice_ref = NodeRef {
        index: nf.nodes.len(),
    };
    nf.nodes.push(Node {
        text_id: nf.nodes.len(),
        name: None,
        ty: Type::Bits(1),
        payload: NodePayload::BitSlice {
            arg: ret_ref,
            start: bit,
            width: 1,
        },
        pos: None,
    });
    nf.ret_node_ref = Some(slice_ref);
    nf.ret_ty = Type::Bits(1);
    nf
}

/// Prove equivalence by splitting on each output bit in parallel.
/// `solver_factory` must be able to create an independent solver instance for
/// each spawned thread.
pub fn prove_ir_fn_equiv_output_bits_parallel<'a, S: Solver>(
    solver_config: &S::Config,
    lhs: &ProverFn<'a>,
    rhs: &ProverFn<'a>,
    assertion_semantics: AssertionSemantics,
    assert_label_include: Option<&Regex>,
    allow_flatten: bool,
) -> EquivResult {
    let width = lhs.fn_ref.ret_ty.bit_count();
    assert_eq!(
        width,
        rhs.fn_ref.ret_ty.bit_count(),
        "Return widths must match"
    );
    if width == 0 {
        // Zero-width values – fall back to the standard equivalence prover because
        // there is no bit to split on.
        return prove_ir_fn_equiv::<S>(
            solver_config,
            lhs,
            rhs,
            assertion_semantics,
            assert_label_include,
            allow_flatten,
        );
    };

    let found = Arc::new(AtomicBool::new(false));
    let counterexample: Arc<Mutex<Option<EquivResult>>> = Arc::new(Mutex::new(None));
    let next_bit = Arc::new(AtomicUsize::new(0));

    let thread_cnt = std::cmp::min(width, num_cpus::get());

    std::thread::scope(|scope| {
        for _ in 0..thread_cnt {
            let lhs_cl = lhs.clone();
            let rhs_cl = rhs.clone();
            let found_cl = found.clone();
            let cex_cl = counterexample.clone();
            let next_cl = next_bit.clone();

            scope.spawn(move || {
                loop {
                    if found_cl.load(Ordering::SeqCst) {
                        break;
                    }
                    let idx = next_cl.fetch_add(1, Ordering::SeqCst);
                    if idx >= width {
                        break;
                    }

                    let lf_ir = make_bit_fn(&lhs_cl.fn_ref, idx);
                    let rf_ir = make_bit_fn(&rhs_cl.fn_ref, idx);
                    let lf = ProverFn::new(&lf_ir, None)
                        .with_fixed_implicit_activation(lhs_cl.fixed_implicit_activation)
                        .with_domains(lhs_cl.domains.clone());
                    let rf = ProverFn::new(&rf_ir, None)
                        .with_fixed_implicit_activation(rhs_cl.fixed_implicit_activation)
                        .with_domains(rhs_cl.domains.clone());
                    let res = prove_ir_fn_equiv::<S>(
                        solver_config,
                        &lf,
                        &rf,
                        assertion_semantics.clone(),
                        assert_label_include,
                        allow_flatten,
                    );
                    if let EquivResult::Disproved { .. } = &res {
                        let mut guard = cex_cl.lock().unwrap();
                        *guard = Some(res.clone());
                        found_cl.store(true, Ordering::SeqCst);
                        break;
                    }
                }
            });
        }
    });

    let maybe_cex = {
        let mut guard = counterexample.lock().unwrap();
        guard.take()
    };
    if let Some(res) = maybe_cex {
        res
    } else {
        EquivResult::Proved
    }
}

/// Prove equivalence by case-splitting on a single input bit value (0 / 1).
/// `split_input_index` selects the parameter to split on and
/// `split_input_bit_index` selects the bit inside that parameter.
pub fn prove_ir_fn_equiv_split_input_bit<'a, S: Solver>(
    solver_config: &S::Config,
    lhs: &ProverFn<'a>,
    rhs: &ProverFn<'a>,
    split_input_index: usize,
    split_input_bit_index: usize,
    assertion_semantics: AssertionSemantics,
    assert_label_include: Option<&Regex>,
    allow_flatten: bool,
) -> EquivResult {
    if lhs.fn_ref.params.is_empty() || rhs.fn_ref.params.is_empty() {
        return prove_ir_fn_equiv::<S>(
            solver_config,
            lhs,
            rhs,
            assertion_semantics,
            assert_label_include,
            allow_flatten,
        );
    }

    assert_eq!(
        lhs.fn_ref.params.len(),
        rhs.fn_ref.params.len(),
        "Parameter count mismatch"
    );
    assert!(
        split_input_index < lhs.fn_ref.params.len(),
        "split_input_index out of bounds"
    );
    assert!(
        split_input_bit_index < lhs.fn_ref.params[split_input_index].ty.bit_count(),
        "split_input_bit_index out of bounds"
    );

    for bit_val in 0..=1u64 {
        let mut solver = S::new(solver_config).unwrap();
        // Build aligned SMT representations first so we can assert the bit-constraint.
        let fn_inputs_lhs = get_fn_inputs(&mut solver, lhs.clone(), Some("lhs"));
        let fn_inputs_rhs = get_fn_inputs(&mut solver, rhs.clone(), Some("rhs"));
        let aligned = align_fn_inputs(&mut solver, &fn_inputs_lhs, &fn_inputs_rhs, allow_flatten);
        let empty_map: HashMap<String, String> = HashMap::new();
        let empty_registry = UfRegistry {
            ufs: HashMap::new(),
        };
        let smt_lhs = ir_to_smt(&mut solver, &aligned.lhs, &empty_map, &empty_registry);
        let smt_rhs = ir_to_smt(&mut solver, &aligned.rhs, &empty_map, &empty_registry);

        // Locate the chosen parameter on the LHS side.
        let param_name = &lhs.fn_ref.params[split_input_index].name;

        let param_bv = aligned
            .lhs
            .inputs
            .get(param_name)
            .expect("Parameter BV missing")
            .bitvec
            .clone();
        let bit_bv = solver.slice(&param_bv, split_input_bit_index, 1);
        let val_bv = solver.numerical(1, bit_val);
        let eq_bv = solver.eq(&bit_bv, &val_bv);
        solver.assert(&eq_bv).unwrap();

        check_aligned_fn_equiv_internal(
            &mut solver,
            &smt_lhs,
            &smt_rhs,
            assertion_semantics,
            assert_label_include,
        );
    }

    EquivResult::Proved
}

#[cfg(test)]
pub mod test_utils {

    use std::collections::HashMap;

    use xlsynth::IrValue;

    use super::{align_fn_inputs, get_fn_inputs, ir_to_smt, ir_value_to_bv, prove_ir_fn_equiv};
    use crate::prover::types::{AssertionSemantics, EquivResult, FnInputs, ParamDomains, ProverFn};
    use crate::solver::{BitVec, Solver, test_utils::assert_solver_eq};
    use xlsynth_pir::{ir, ir_parser};

    pub fn test_invoke_basic<S: Solver>(solver_config: &S::Config) {
        // Package with a callee that doubles its input and two wrappers:
        // one using invoke, the other inlining the computation.
        let ir_pkg_text = r#"
            package p

            fn g(x: bits[8]) -> bits[8] {
              ret add.1: bits[8] = add(x, x, id=1)
            }

            fn call(x: bits[8]) -> bits[8] {
              ret r: bits[8] = invoke(x, to_apply=g, id=1)
            }

            fn inline(x: bits[8]) -> bits[8] {
              ret add.2: bits[8] = add(x, x, id=2)
            }
        "#;

        let pkg = ir_parser::Parser::new(ir_pkg_text)
            .parse_package()
            .expect("Failed to parse IR package");
        let call_fn = pkg.get_fn("call").expect("call not found");
        let inline_fn = pkg.get_fn("inline").expect("inline not found");

        let call_pf = ProverFn::new(call_fn, Some(&pkg));
        let inline_pf = ProverFn::new(inline_fn, Some(&pkg));
        let res = super::prove_ir_fn_equiv::<S>(
            solver_config,
            &call_pf,
            &inline_pf,
            AssertionSemantics::Same,
            None,
            false,
        );
        assert!(matches!(res, super::EquivResult::Proved));
    }

    /// Uninterpreted-function (UF) handling tests
    ///
    /// Construct a package with two different callees `g` and `h` and two
    /// wrappers that invoke them. Without UF mapping they are not
    /// equivalent; with UF mapping to the same UF symbol and signature they
    /// become equivalent.
    pub fn test_uf_basic_equiv<S: Solver>(solver_config: &S::Config) {
        let ir_pkg_text = r#"
            package p_uf

            fn g(x: bits[8]) -> bits[8] {
              ret add.1: bits[8] = add(x, x, id=1)
            }

            fn h(x: bits[8]) -> bits[8] {
              ret sub.2: bits[8] = sub(x, x, id=2)
            }

            fn call_g(x: bits[8]) -> bits[8] {
              ret r: bits[8] = invoke(x, to_apply=g, id=3)
            }

            fn call_h(x: bits[8]) -> bits[8] {
              ret r: bits[8] = invoke(x, to_apply=h, id=4)
            }
        "#;

        let pkg = ir_parser::Parser::new(ir_pkg_text)
            .parse_package()
            .expect("Failed to parse IR package");
        let call_g = pkg.get_fn("call_g").expect("call_g not found");
        let call_h = pkg.get_fn("call_h").expect("call_h not found");

        // 1) Without UF mapping: should be inequivalent (add(x,x) vs 0)
        let lhs_pf = ProverFn::new(call_g, Some(&pkg));
        let rhs_pf = ProverFn::new(call_h, Some(&pkg));

        let res_no_uf = super::prove_ir_fn_equiv::<S>(
            solver_config,
            &lhs_pf,
            &rhs_pf,
            super::AssertionSemantics::Same,
            None,
            false,
        );
        assert!(matches!(res_no_uf, super::EquivResult::Disproved { .. }));

        // 2) With UF mapping: map g (LHS) and h (RHS) to the same UF symbol "F".
        let mut lhs_uf_map: HashMap<String, String> = HashMap::new();
        lhs_uf_map.insert("g".to_string(), "F".to_string());
        let mut rhs_uf_map: HashMap<String, String> = HashMap::new();
        rhs_uf_map.insert("h".to_string(), "F".to_string());
        let res_uf = super::prove_ir_fn_equiv::<S>(
            solver_config,
            &ProverFn::new(call_g, Some(&pkg)).with_uf_map(lhs_uf_map),
            &ProverFn::new(call_h, Some(&pkg)).with_uf_map(rhs_uf_map),
            super::AssertionSemantics::Same,
            None,
            false,
        );
        assert!(matches!(res_uf, super::EquivResult::Proved));
    }

    /// Nested invoke case: inner callees differ, but both sides map inner to
    /// the same UF.
    pub fn test_uf_nested_invoke_equiv<S: Solver>(solver_config: &S::Config) {
        let ir_pkg_text = r#"
            package p_uf_nested

            fn inner_g(x: bits[4]) -> bits[4] {
              ret add.1: bits[4] = add(x, x, id=1)
            }

            fn inner_h(x: bits[4]) -> bits[4] {
              ret literal.2: bits[4] = literal(value=7, id=2)
            }

            fn mid_g(x: bits[4]) -> bits[4] {
              ret r: bits[4] = invoke(x, to_apply=inner_g, id=3)
            }

            fn mid_h(x: bits[4]) -> bits[4] {
              ret r: bits[4] = invoke(x, to_apply=inner_h, id=4)
            }

            fn top_g(x: bits[4]) -> bits[4] {
              ret r: bits[4] = invoke(x, to_apply=mid_g, id=5)
            }

            fn top_h(x: bits[4]) -> bits[4] {
              ret r: bits[4] = invoke(x, to_apply=mid_h, id=6)
            }
        "#;

        let pkg = ir_parser::Parser::new(ir_pkg_text)
            .parse_package()
            .expect("Failed to parse IR package");
        let top_g = pkg.get_fn("top_g").expect("top_g not found");
        let top_h = pkg.get_fn("top_h").expect("top_h not found");

        // With UF mapping on inner functions, equality should hold at the top.
        let mut lhs_uf_map: HashMap<String, String> = HashMap::new();
        lhs_uf_map.insert("inner_g".to_string(), "F".to_string());
        let mut rhs_uf_map: HashMap<String, String> = HashMap::new();
        rhs_uf_map.insert("inner_h".to_string(), "F".to_string());
        let lhs_pf = ProverFn::new(top_g, Some(&pkg)).with_uf_map(lhs_uf_map);
        let rhs_pf = ProverFn::new(top_h, Some(&pkg)).with_uf_map(rhs_uf_map);
        let res = super::prove_ir_fn_equiv::<S>(
            solver_config,
            &lhs_pf,
            &rhs_pf,
            super::AssertionSemantics::Same,
            None,
            false,
        );
        assert!(matches!(res, super::EquivResult::Proved));
    }

    /// UF + implicit-token with two data args to ensure we slice args after the
    /// first two params.
    pub fn test_uf_implicit_token_two_args_equiv<S: Solver>(solver_config: &S::Config) {
        let ir_pkg_text = r#"
            package p_uf_itok2

            fn __itokg(__token: token, __activate: bits[1], a: bits[4], b: bits[4]) -> (token, bits[8]) {
              s: bits[8] = umul(a, b, id=1)
              ret t: (token, bits[8]) = tuple(__token, s, id=2)
            }

            fn h(a: bits[4], b: bits[4]) -> bits[8] {
              x: bits[4] = xor(a, b, id=3)
              ret z: bits[8] = zero_ext(x, new_bit_count=8, id=4)
            }

            fn call_g(tok: token, act: bits[1], a: bits[4], b: bits[4]) -> (token, bits[8]) {
              ret r: (token, bits[8]) = invoke(tok, act, a, b, to_apply=__itokg, id=6)
            }

            fn call_h(tok: token, act: bits[1], a: bits[4], b: bits[4]) -> bits[8] {
              ret r: bits[8] = invoke(a, b, to_apply=h, id=7)
            }
        "#;

        let pkg = ir_parser::Parser::new(ir_pkg_text)
            .parse_package()
            .expect("Failed to parse IR package");
        let call_g = pkg.get_fn("call_g").expect("call_g not found");
        let call_h = pkg.get_fn("call_h").expect("call_h not found");

        let call_g_pf = ProverFn::new(call_g, Some(&pkg));
        let call_h_pf = ProverFn::new(call_h, Some(&pkg));
        let res_no_uf = super::prove_ir_fn_equiv::<S>(
            solver_config,
            &call_g_pf,
            &call_h_pf,
            super::AssertionSemantics::Same,
            None,
            false,
        );
        assert!(matches!(res_no_uf, super::EquivResult::Disproved { .. }));

        let mut lhs_uf_map: HashMap<String, String> = HashMap::new();
        lhs_uf_map.insert("g".to_string(), "F".to_string());
        let mut rhs_uf_map: HashMap<String, String> = HashMap::new();
        rhs_uf_map.insert("h".to_string(), "F".to_string());
        // Two 4-bit data args, 8-bit result.
        let call_g_pf_with_uf = ProverFn::new(call_g, Some(&pkg)).with_uf_map(lhs_uf_map);
        let call_h_pf_with_uf = ProverFn::new(call_h, Some(&pkg)).with_uf_map(rhs_uf_map);
        let res_uf = super::prove_ir_fn_equiv::<S>(
            solver_config,
            &call_g_pf_with_uf,
            &call_h_pf_with_uf,
            super::AssertionSemantics::Same,
            None,
            false,
        );
        assert!(matches!(res_uf, super::EquivResult::Proved));
    }

    /// Assertion-label include filter: without filter, inequivalent due to LHS
    /// 'red' assertion that can fail while RHS only has a trivially-true
    /// 'blue' assertion; including only 'blue' labels should prove
    /// equivalence under AssertionSemantics::Same.
    pub fn test_assert_label_filter_equiv<S: Solver>(solver_config: &S::Config) {
        let ir_pkg_text = r#"
            package p_label_filter

            fn lhs(__token: token, a: bits[1]) -> bits[1] {
              assert.1: token = assert(__token, a, message="rf", label="red", id=1)
              ret lit1: bits[1] = literal(value=1, id=2)
            }

            fn rhs(__token: token, a: bits[1]) -> bits[1] {
              t: bits[1] = literal(value=1, id=1)
              assert.2: token = assert(__token, t, message="bf", label="blue", id=2)
              ret lit1: bits[1] = literal(value=1, id=3)
            }
        "#;

        let pkg = ir_parser::Parser::new(ir_pkg_text)
            .parse_package()
            .expect("Failed to parse IR package");
        let lhs = pkg.get_fn("lhs").expect("lhs not found");
        let rhs = pkg.get_fn("rhs").expect("rhs not found");

        let lhs_pf = ProverFn::new(lhs, Some(&pkg));
        let rhs_pf = ProverFn::new(rhs, Some(&pkg));
        let res_no_filter = super::prove_ir_fn_equiv::<S>(
            solver_config,
            &lhs_pf,
            &rhs_pf,
            super::AssertionSemantics::Same,
            None,
            false,
        );
        assert!(matches!(
            res_no_filter,
            super::EquivResult::Disproved { .. }
        ));

        let include = regex::Regex::new(r"^(?:blue)$").unwrap();
        let res_filtered = super::prove_ir_fn_equiv::<S>(
            solver_config,
            &lhs_pf,
            &rhs_pf,
            super::AssertionSemantics::Same,
            Some(&include),
            false,
        );
        assert!(matches!(res_filtered, super::EquivResult::Proved));
    }

    pub fn test_invoke_two_args<S: Solver>(solver_config: &S::Config) {
        // Callee adds two 4-bit args. Compare invoke wrapper vs inline.
        let ir_pkg_text = r#"
            package p2

            fn g(a: bits[4], b: bits[4]) -> bits[4] {
              ret add.1: bits[4] = add(a, b, id=1)
            }

            fn call(a: bits[4], b: bits[4]) -> bits[4] {
              ret r: bits[4] = invoke(a, b, to_apply=g, id=2)
            }

            fn inline(a: bits[4], b: bits[4]) -> bits[4] {
              ret add.3: bits[4] = add(a, b, id=3)
            }
        "#;

        let pkg = ir_parser::Parser::new(ir_pkg_text)
            .parse_package()
            .expect("Failed to parse IR package");
        let call_fn = pkg.get_fn("call").expect("call not found");
        let inline_fn = pkg.get_fn("inline").expect("inline not found");

        let call_pf = ProverFn::new(call_fn, Some(&pkg));
        let inline_pf = ProverFn::new(inline_fn, Some(&pkg));
        let res = super::prove_ir_fn_equiv::<S>(
            solver_config,
            &call_pf,
            &inline_pf,
            AssertionSemantics::Same,
            None,
            false,
        );
        assert!(matches!(res, super::EquivResult::Proved));
    }

    pub fn test_counted_for_basic<S: Solver>(solver_config: &S::Config) {
        // Package with a simple body that accumulates the induction variable into acc.
        // Loop it three times; compare to a manually unrolled implementation.
        let ir_pkg_text = r#"
            package p_cf

            fn body(i: bits[8], acc: bits[8]) -> bits[8] {
              ret add.1: bits[8] = add(acc, i, id=1)
            }

            fn looped(init: bits[8]) -> bits[8] {
              ret cf: bits[8] = counted_for(init, trip_count=3, stride=1, body=body, id=2)
            }

            fn inline(init: bits[8]) -> bits[8] {
              z: bits[8] = literal(value=0, id=3)
              a0: bits[8] = add(init, z, id=4)
              o1: bits[8] = literal(value=1, id=5)
              a1: bits[8] = add(a0, o1, id=6)
              o2: bits[8] = literal(value=2, id=7)
              ret a2: bits[8] = add(a1, o2, id=8)
            }
        "#;

        let pkg = ir_parser::Parser::new(ir_pkg_text)
            .parse_package()
            .expect("Failed to parse IR package");
        let looped = pkg.get_fn("looped").expect("looped not found");
        let inline = pkg.get_fn("inline").expect("inline not found");

        let looped_pf = ProverFn::new(looped, Some(&pkg));
        let inline_pf = ProverFn::new(inline, Some(&pkg));
        let res = super::prove_ir_fn_equiv::<S>(
            solver_config,
            &looped_pf,
            &inline_pf,
            AssertionSemantics::Same,
            None,
            false,
        );
        assert!(matches!(res, super::EquivResult::Proved));
    }

    pub fn test_counted_for_stride<S: Solver>(solver_config: &S::Config) {
        // Stride=2 for 3 trips: add 0,2,4
        let ir_pkg_text = r#"
            package p_cf2

            fn body(i: bits[8], acc: bits[8]) -> bits[8] {
              ret add.1: bits[8] = add(acc, i, id=1)
            }

            fn looped(init: bits[8]) -> bits[8] {
              ret cf: bits[8] = counted_for(init, trip_count=3, stride=2, body=body, id=2)
            }

            fn inline(init: bits[8]) -> bits[8] {
              i0: bits[8] = literal(value=0, id=3)
              a0: bits[8] = add(init, i0, id=4)
              i1: bits[8] = literal(value=2, id=5)
              a1: bits[8] = add(a0, i1, id=6)
              i2: bits[8] = literal(value=4, id=7)
              ret a2: bits[8] = add(a1, i2, id=8)
            }
        "#;

        let pkg = ir_parser::Parser::new(ir_pkg_text)
            .parse_package()
            .expect("Failed to parse IR package");
        let looped = pkg.get_fn("looped").expect("looped not found");
        let inline = pkg.get_fn("inline").expect("inline not found");

        let looped_pf = ProverFn::new(looped, Some(&pkg));
        let inline_pf = ProverFn::new(inline, Some(&pkg));
        let res = super::prove_ir_fn_equiv::<S>(
            solver_config,
            &looped_pf,
            &inline_pf,
            AssertionSemantics::Same,
            None,
            false,
        );
        assert!(matches!(res, super::EquivResult::Proved));
    }

    pub fn test_counted_for_zero_trip<S: Solver>(solver_config: &S::Config) {
        // Zero trips should produce the init unchanged.
        let ir_pkg_text = r#"
            package p_cf3

            fn body(i: bits[8], acc: bits[8]) -> bits[8] {
              ret add.1: bits[8] = add(acc, i, id=1)
            }

            fn looped(init: bits[8]) -> bits[8] {
              ret cf: bits[8] = counted_for(init, trip_count=0, stride=1, body=body, id=2)
            }

            fn inline(init: bits[8]) -> bits[8] {
              ret identity.1: bits[8] = identity(init, id=1)
            }
        "#;

        let pkg = ir_parser::Parser::new(ir_pkg_text)
            .parse_package()
            .expect("Failed to parse IR package");
        let looped = pkg.get_fn("looped").expect("looped not found");
        let inline = pkg.get_fn("inline").expect("inline not found");

        let looped_pf = ProverFn::new(looped, Some(&pkg));
        let inline_pf = ProverFn::new(inline, Some(&pkg));
        let res = super::prove_ir_fn_equiv::<S>(
            solver_config,
            &looped_pf,
            &inline_pf,
            AssertionSemantics::Same,
            None,
            false,
        );
        assert!(matches!(res, super::EquivResult::Proved));
    }

    pub fn test_counted_for_invariant_args<S: Solver>(solver_config: &S::Config) {
        // Body uses an invariant offset 'k' added to the induction variable each
        // iteration. Compare counted_for vs inline unrolling.
        let ir_pkg_text = r#"
            package p_cf_inv

            fn body(i: bits[8], acc: bits[8], k: bits[8]) -> bits[8] {
              t0: bits[8] = add(i, k, id=1)
              ret a: bits[8] = add(acc, t0, id=2)
            }

            fn looped(init: bits[8], k: bits[8]) -> bits[8] {
              ret cf: bits[8] = counted_for(init, trip_count=3, stride=1, body=body, invariant_args=[k], id=3)
            }

            fn inline(init: bits[8], k: bits[8]) -> bits[8] {
              z: bits[8] = literal(value=0, id=4)
              s0: bits[8] = add(z, k, id=5)
              a0: bits[8] = add(init, s0, id=6)
              one: bits[8] = literal(value=1, id=7)
              s1: bits[8] = add(one, k, id=8)
              a1: bits[8] = add(a0, s1, id=9)
              two: bits[8] = literal(value=2, id=10)
              s2: bits[8] = add(two, k, id=11)
              ret a2: bits[8] = add(a1, s2, id=12)
            }
        "#;

        let pkg = ir_parser::Parser::new(ir_pkg_text)
            .parse_package()
            .expect("Failed to parse IR package");
        let looped = pkg.get_fn("looped").expect("looped not found");
        let inline = pkg.get_fn("inline").expect("inline not found");

        let looped_pf = ProverFn::new(looped, Some(&pkg));
        let inline_pf = ProverFn::new(inline, Some(&pkg));
        let res = super::prove_ir_fn_equiv::<S>(
            solver_config,
            &looped_pf,
            &inline_pf,
            AssertionSemantics::Same,
            None,
            false,
        );
        assert!(matches!(res, super::EquivResult::Proved));
    }
    pub fn align_zero_width_fn_inputs<S: Solver>(solver_config: &S::Config) {
        let ir_text = r#"
            fn lhs(tok: token) -> token {
                ret t: token = param(name=tok, id=1)
            }
        "#;

        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();
        let mut solver = S::new(solver_config).unwrap();
        let prover_fn = ProverFn::new(&f, None);
        let fn_inputs = get_fn_inputs(&mut solver, prover_fn.clone(), None);
        // Must not panic.
        let _ = align_fn_inputs(&mut solver, &fn_inputs, &fn_inputs, false);
    }

    pub fn align_non_zero_width_fn_inputs_first_token<S: Solver>(solver_config: &S::Config) {
        let ir_text = r#"
            fn lhs(tok: token, x: bits[4]) -> token {
                ret t: token = param(name=tok, id=1)
            }
        "#;

        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();
        let mut solver = S::new(solver_config).unwrap();
        let prover_fn = ProverFn::new(&f, None);
        let fn_inputs = get_fn_inputs(&mut solver, prover_fn.clone(), None);
        // Must not panic.
        let _ = align_fn_inputs(&mut solver, &fn_inputs, &fn_inputs, false);
    }

    pub fn assert_smt_fn_eq<S: Solver>(
        solver_config: &S::Config,
        ir_text: &str,
        expected: impl Fn(&mut S, &FnInputs<S::Term>) -> BitVec<S::Term>,
    ) {
        let mut parser = ir_parser::Parser::new(ir_text);
        let f = parser.parse_fn().unwrap();
        let mut solver = S::new(solver_config).unwrap();
        let prover_fn = ProverFn::new(&f, None);
        let fn_inputs = get_fn_inputs(&mut solver, prover_fn.clone(), None);
        let empty_map: HashMap<String, String> = HashMap::new();
        let empty_registry = super::UfRegistry {
            ufs: HashMap::new(),
        };
        let smt_fn = ir_to_smt(&mut solver, &fn_inputs, &empty_map, &empty_registry);
        let expected_result = expected(&mut solver, &fn_inputs);
        assert_solver_eq(&mut solver, &smt_fn.output.bitvec, &expected_result);
    }

    pub fn assert_ir_value_to_bv_eq<S: Solver>(
        solver_config: &S::Config,
        ir_value: &IrValue,
        ir_type: &ir::Type,
        expected: impl Fn(&mut S) -> BitVec<S::Term>,
    ) {
        let mut solver = S::new(solver_config).unwrap();
        let bv = ir_value_to_bv(&mut solver, &ir_value, &ir_type);
        assert_eq!(bv.ir_type, ir_type);
        let expected_value = expected(&mut solver);
        crate::solver::test_utils::assert_solver_eq(&mut solver, &bv.bitvec, &expected_value);
    }

    fn assert_ir_fn_equiv_base_with_implicit_token_policy<S: Solver>(
        solver_config: &S::Config,
        lhs_text: &str,
        rhs_text: &str,
        lhs_fixed_implicit_activation: bool,
        rhs_fixed_implicit_activation: bool,
        allow_flatten: bool,
        assertion_semantics: AssertionSemantics,
        expected_proven: bool,
    ) {
        let mut parser = ir_parser::Parser::new(lhs_text);
        let lhs_ir_fn = parser.parse_fn().unwrap();
        let mut parser = ir_parser::Parser::new(rhs_text);
        let rhs_ir_fn = parser.parse_fn().unwrap();
        let actual = prove_ir_fn_equiv::<S>(
            solver_config,
            &ProverFn::new(&lhs_ir_fn, None)
                .with_fixed_implicit_activation(lhs_fixed_implicit_activation),
            &ProverFn::new(&rhs_ir_fn, None)
                .with_fixed_implicit_activation(rhs_fixed_implicit_activation),
            assertion_semantics,
            None,
            allow_flatten,
        );
        if expected_proven {
            assert!(matches!(actual, EquivResult::Proved));
        } else {
            assert!(matches!(actual, EquivResult::Disproved { .. }));
        }
    }

    fn assert_ir_fn_equiv_base<S: Solver>(
        solver_config: &S::Config,
        ir_text_1: &str,
        ir_text_2: &str,
        allow_flatten: bool,
        assertion_semantics: AssertionSemantics,
        expected_proven: bool,
    ) {
        assert_ir_fn_equiv_base_with_implicit_token_policy::<S>(
            solver_config,
            ir_text_1,
            ir_text_2,
            false,
            false,
            allow_flatten,
            assertion_semantics,
            expected_proven,
        );
    }

    pub fn assert_ir_fn_equiv<S: Solver>(
        solver_config: &S::Config,
        ir_text_1: &str,
        ir_text_2: &str,
        allow_flatten: bool,
    ) {
        assert_ir_fn_equiv_base::<S>(
            solver_config,
            ir_text_1,
            ir_text_2,
            allow_flatten,
            AssertionSemantics::Same,
            true,
        );
    }

    pub fn assert_ir_fn_inequiv<S: Solver>(
        solver_config: &S::Config,
        ir_text_1: &str,
        ir_text_2: &str,
        allow_flatten: bool,
    ) {
        assert_ir_fn_equiv_base::<S>(
            solver_config,
            ir_text_1,
            ir_text_2,
            allow_flatten,
            AssertionSemantics::Same,
            false,
        );
    }

    fn test_ir_fn_equiv_to_self<S: Solver>(solver_config: &S::Config, ir_text: &str) {
        assert_ir_fn_equiv::<S>(solver_config, ir_text, ir_text, false);
    }

    pub fn test_ir_value_bits<S: Solver>(solver_config: &S::Config) {
        assert_ir_value_to_bv_eq::<S>(
            solver_config,
            &IrValue::u32(0x12345678),
            &ir::Type::Bits(32),
            |solver: &mut S| solver.from_raw_str(32, "#x12345678"),
        );
    }

    pub fn test_ir_bits<S: Solver>(solver_config: &S::Config) {
        assert_smt_fn_eq::<S>(
            solver_config,
            r#"fn f() -> bits[32] {
                ret literal.1: bits[32] = literal(value=0x12345678, id=1)
            }"#,
            |solver: &mut S, _: &FnInputs<S::Term>| solver.from_raw_str(32, "#x12345678"),
        );
    }

    pub fn test_ir_value_array<S: Solver>(solver_config: &S::Config) {
        crate::prover::ir_equiv::test_utils::assert_ir_value_to_bv_eq::<S>(
            solver_config,
            &IrValue::make_array(&[
                IrValue::make_array(&[
                    IrValue::make_ubits(8, 0x12).unwrap(),
                    IrValue::make_ubits(8, 0x34).unwrap(),
                ])
                .unwrap(),
                IrValue::make_array(&[
                    IrValue::make_ubits(8, 0x56).unwrap(),
                    IrValue::make_ubits(8, 0x78).unwrap(),
                ])
                .unwrap(),
                IrValue::make_array(&[
                    IrValue::make_ubits(8, 0x9a).unwrap(),
                    IrValue::make_ubits(8, 0xbc).unwrap(),
                ])
                .unwrap(),
            ])
            .unwrap(),
            &ir::Type::Array(ir::ArrayTypeData {
                element_type: Box::new(ir::Type::Array(ir::ArrayTypeData {
                    element_type: Box::new(ir::Type::Bits(8)),
                    element_count: 2,
                })),
                element_count: 3,
            }),
            |solver: &mut S| solver.from_raw_str(48, "#xbc9a78563412"),
        );
    }

    pub fn test_ir_array<S: Solver>(solver_config: &S::Config) {
        assert_smt_fn_eq::<S>(
            solver_config,
            r#"fn f() -> bits[8][2][3] {
                ret literal.1: bits[8][2][3] = literal(value=[[0x12, 0x34], [0x56, 0x78], [0x9a, 0xbc]], id=1)
            }"#,
            |solver: &mut S, _: &FnInputs<S::Term>| solver.from_raw_str(48, "#xbc9a78563412"),
        );
    }

    /// array_slice(input, start, width=3) on bits[8][4]
    pub fn test_ir_array_slice_basic<S: Solver>(solver_config: &S::Config) {
        assert_smt_fn_eq::<S>(
            solver_config,
            r#"fn f(input: bits[8][4] id=1, start: bits[3] id=2) -> bits[8][3] {
                ret s: bits[8][3] = array_slice(input, start, width=3, id=3)
            }"#,
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                // Piecewise expected for width=3 on N=4:
                // (1) in-bounds start<=1, (2) cross-bound start==2, (3) OOB start>=3.
                let input_bv = inputs.inputs.get("input").unwrap().bitvec.clone();
                let start_bv = inputs.inputs.get("start").unwrap().bitvec.clone();
                let width = start_bv.get_width();

                // Precompute element slices A0..A3 (LSB chunk is index 0)
                let a0 = solver.extract(&input_bv, 7, 0);
                let a1 = solver.extract(&input_bv, 15, 8);
                let a2 = solver.extract(&input_bv, 23, 16);
                let a3 = solver.extract(&input_bv, 31, 24);

                // In-bounds result for start in {0,1}
                let s1 = solver.numerical(width, 1);
                let is1 = solver.eq(&start_bv, &s1);
                let r0 = solver.concat(&a2, &a1);
                let r0 = solver.concat(&r0, &a0); // [A0,A1,A2]
                let r1 = solver.concat(&a3, &a2);
                let r1 = solver.concat(&r1, &a1); // [A1,A2,A3]
                let r_in = solver.ite(&is1, &r1, &r0);

                // Cross-bound for start==2 → [A2,A3,A3]
                let s2 = solver.numerical(width, 2);
                let eq2 = solver.eq(&start_bv, &s2);
                let r_cross = solver.concat(&a3, &a3);
                let r_cross = solver.concat(&r_cross, &a2);

                // Totally OOB start>=3 → [A3,A3,A3]
                let rr = solver.concat(&a3, &a3);
                let rr = solver.concat(&rr, &a3);

                // Conditions
                let le1 = solver.ule(&start_bv, &s1);
                let tmp = solver.ite(&eq2, &r_cross, &rr);
                solver.ite(&le1, &r_in, &tmp)
            },
        );
    }

    pub fn test_ir_value_tuple<S: Solver>(solver_config: &S::Config) {
        assert_ir_value_to_bv_eq::<S>(
            solver_config,
            &IrValue::make_tuple(&[
                IrValue::make_ubits(8, 0x12).unwrap(),
                IrValue::make_ubits(4, 0x4).unwrap(),
            ]),
            &ir::Type::Tuple(vec![
                Box::new(ir::Type::Bits(8)),
                Box::new(ir::Type::Bits(4)),
            ]),
            |solver: &mut S| solver.from_raw_str(12, "#x124"),
        );
    }

    pub fn test_ir_tuple<S: Solver>(solver_config: &S::Config) {
        assert_smt_fn_eq::<S>(
            solver_config,
            r#"fn f() -> (bits[8], bits[4]) {
                ret literal.1: (bits[8], bits[4]) = literal(value=(0x12, 0x4), id=1)
            }"#,
            |solver: &mut S, _: &FnInputs<S::Term>| solver.from_raw_str(12, "#x124"),
        );
    }

    pub fn test_ir_value_token<S: Solver>(solver_config: &S::Config) {
        assert_ir_value_to_bv_eq::<S>(
            solver_config,
            &IrValue::make_ubits(0, 0).unwrap(),
            &ir::Type::Token,
            |_: &mut S| BitVec::ZeroWidth,
        );
    }

    pub fn test_unop_base<S: Solver>(
        solver_config: &S::Config,
        unop_xls_name: &str,
        unop: impl Fn(&mut S, &BitVec<S::Term>) -> BitVec<S::Term>,
    ) {
        assert_smt_fn_eq::<S>(
            solver_config,
            &(format!(
                r#"fn f(x: bits[8]) -> bits[8] {{
                ret {op}.1: bits[8] = {op}(x, id=1)
            }}"#,
                op = unop_xls_name
            )),
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                unop(solver, &inputs.inputs.get("x").unwrap().bitvec)
            },
        );
    }

    pub fn test_binop_base<S: Solver>(
        solver_config: &S::Config,
        binop_xls_name: &str,
        binop: impl Fn(&mut S, &BitVec<S::Term>, &BitVec<S::Term>) -> BitVec<S::Term>,
        lhs_width: usize,
        rhs_width: usize,
        result_width: usize,
    ) {
        let lhs_width_str = lhs_width.to_string();
        let rhs_width_str = rhs_width.to_string();
        let result_width_str = result_width.to_string();
        let ir_text = format!(
            r#"fn f(x: bits[{lw}], y: bits[{rw}]) -> bits[{rw2}] {{
                ret {op}.1: bits[{rw2}] = {op}(x, y, id=1)
            }}"#,
            lw = lhs_width_str,
            rw = rhs_width_str,
            rw2 = result_width_str,
            op = binop_xls_name
        );
        let mut parser = ir_parser::Parser::new(&ir_text);
        let f = parser.parse_fn().unwrap();
        let mut solver = S::new(solver_config).unwrap();
        let prover_fn = ProverFn::new(&f, None);
        let fn_inputs = get_fn_inputs(&mut solver, prover_fn.clone(), None);
        let empty_map: HashMap<String, String> = HashMap::new();
        let empty_registry = super::UfRegistry {
            ufs: HashMap::new(),
        };
        let smt_fn = ir_to_smt(&mut solver, &fn_inputs, &empty_map, &empty_registry);
        let x = fn_inputs.inputs.get("x").unwrap().bitvec.clone();
        let y = fn_inputs.inputs.get("y").unwrap().bitvec.clone();
        let expected = binop(&mut solver, &x, &y);
        crate::solver::test_utils::assert_solver_eq(&mut solver, &smt_fn.output.bitvec, &expected);
    }

    pub fn test_binop_8_bit<S: Solver>(
        solver_config: &S::Config,
        binop_xls_name: &str,
        binop: impl Fn(&mut S, &BitVec<S::Term>, &BitVec<S::Term>) -> BitVec<S::Term>,
    ) {
        test_binop_base::<S>(solver_config, binop_xls_name, binop, 8, 8, 8);
    }

    pub fn test_binop_bool<S: Solver>(
        solver_config: &S::Config,
        binop_xls_name: &str,
        binop: impl Fn(&mut S, &BitVec<S::Term>, &BitVec<S::Term>) -> BitVec<S::Term>,
    ) {
        test_binop_base::<S>(solver_config, binop_xls_name, binop, 8, 8, 1);
    }

    pub fn test_ir_tuple_index<S: Solver>(solver_config: &S::Config) {
        assert_smt_fn_eq::<S>(
            solver_config,
            r#"fn f(input: (bits[8], bits[4])) -> bits[8] {
                ret tuple_index.1: bits[8] = tuple_index(input, index=0, id=1)
            }"#,
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let tuple = inputs.inputs.get("input").unwrap().bitvec.clone();
                solver.extract(&tuple, 11, 4)
            },
        );
    }

    pub fn test_ir_tuple_index_literal<S: Solver>(solver_config: &S::Config) {
        use crate::prover::ir_equiv::test_utils::assert_smt_fn_eq;
        assert_smt_fn_eq::<S>(
            solver_config,
            &(r#"fn f() -> bits[8] {
                    literal.1: (bits[8], bits[4]) = literal(value=(0x12, 0x4), id=1)
                    ret tuple_index.1: bits[8] = tuple_index(literal.1, index=0, id=1)
                    }"#),
            |solver: &mut S, _: &FnInputs<S::Term>| solver.from_raw_str(8, "#x12"),
        );
    }

    pub fn test_ir_tuple_reverse<S: Solver>(solver_config: &S::Config) {
        use crate::prover::ir_equiv::test_utils::assert_smt_fn_eq;
        assert_smt_fn_eq::<S>(
            solver_config,
            r#"fn f(a: (bits[8], bits[4])) -> (bits[4], bits[8]) {
                    tuple_index.3: bits[4] = tuple_index(a, index=1, id=3)
                    tuple_index.5: bits[8] = tuple_index(a, index=0, id=5)
                    ret tuple.6: (bits[4], bits[8]) = tuple(tuple_index.3, tuple_index.5, id=6)
                }"#,
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let tuple = inputs.inputs.get("a").unwrap().bitvec.clone();
                let tuple_0 = solver.extract(&tuple, 11, 4);
                let tuple_1 = solver.extract(&tuple, 3, 0);
                solver.concat(&tuple_1, &tuple_0)
            },
        );
    }

    pub fn test_ir_tuple_flattened<S: Solver>(solver_config: &S::Config) {
        assert_ir_fn_equiv::<S>(
            solver_config,
            r#"fn f() -> (bits[8], bits[4]) {
            ret literal.1: (bits[8], bits[4]) = literal(value=(0x12, 0x4), id=1)
        }"#,
            r#"fn g() -> bits[12] {
            ret literal.1: bits[12] = literal(value=0x124, id=1)
        }"#,
            true,
        );
    }

    pub fn test_tuple_literal_vs_constructed<S: Solver>(solver_config: &S::Config) {
        assert_ir_fn_equiv::<S>(
            solver_config,
            r#"fn lhs() -> (bits[8], bits[4]) {
            ret lit_tuple: (bits[8], bits[4]) = literal(value=(0x12, 0x4), id=1)
        }"#,
            r#"fn rhs() -> (bits[8], bits[4]) {
            lit0: bits[8] = literal(value=0x12, id=1)
            lit1: bits[4] = literal(value=0x4, id=2)
            ret tup: (bits[8], bits[4]) = tuple(lit0, lit1, id=3)
        }"#,
            false,
        );
    }

    pub fn test_tuple_index_on_literal<S: Solver>(solver_config: &S::Config) {
        assert_ir_fn_equiv::<S>(
            solver_config,
            r#"fn f() -> bits[8] {
            lit_tuple: (bits[8], bits[4]) = literal(value=(0x12, 0x4), id=1)
            ret idx0: bits[8] = tuple_index(lit_tuple, index=0, id=2)
        }"#,
            r#"fn g() -> bits[8] {
            ret lit: bits[8] = literal(value=0x12, id=1)
        }"#,
            false,
        );
    }

    pub fn test_ir_array_index_base<S: Solver>(
        solver_config: &S::Config,
        index: &str,
        expected_low: i32,
    ) {
        use crate::prover::ir_equiv::test_utils::assert_smt_fn_eq;
        assert_smt_fn_eq::<S>(
            solver_config,
            &(r#"fn f(input: bits[8][4] id=1) -> bits[8] {
                    literal.4: bits[3] = literal(value="#
                .to_string()
                + index
                + r#", id=4)
                    ret array_index.5: bits[8] = array_index(input, indices=[literal.4], assumed_in_bounds=true, id=5)
                }"#),
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let array = inputs.inputs.get("input").unwrap().bitvec.clone();
                solver.extract(&array, expected_low + 7, expected_low)
            },
        );
    }

    pub fn test_ir_array_index_multi_level<S: Solver>(solver_config: &S::Config) {
        use crate::prover::ir_equiv::test_utils::assert_smt_fn_eq;
        assert_smt_fn_eq::<S>(
            solver_config,
            r#"fn f(input: bits[8][4][2] id=1) -> bits[8] {
                literal.4: bits[2] = literal(value=1, id=4)
                ret array_index.6: bits[8] = array_index(input, indices=[literal.4], assumed_in_bounds=true, id=6)
            }"#,
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let array = inputs.inputs.get("input").unwrap().bitvec.clone();
                solver.extract(&array, 63, 32)
            },
        );
    }

    pub fn test_ir_array_index_deep_multi_level<S: Solver>(solver_config: &S::Config) {
        use crate::prover::ir_equiv::test_utils::assert_smt_fn_eq;
        assert_smt_fn_eq::<S>(
            solver_config,
            r#"fn f(input: bits[8][4][2] id=1) -> bits[8] {
                literal.4: bits[2] = literal(value=1, id=4)
                literal.5: bits[2] = literal(value=0, id=5)
                ret array_index.6: bits[8] = array_index(input, indices=[literal.4, literal.5], assumed_in_bounds=true, id=6)
            }"#,
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let array = inputs.inputs.get("input").unwrap().bitvec.clone();
                solver.extract(&array, 39, 32)
            },
        );
    }

    pub fn test_array_update_inbound_value<S: Solver>(solver_config: &S::Config) {
        use crate::prover::ir_equiv::test_utils::assert_smt_fn_eq;
        assert_smt_fn_eq::<S>(
            solver_config,
            r#"fn f(input: bits[4][4] id=1, val: bits[4] id=2) -> bits[4][4] {
                lit: bits[2] = literal(value=1, id=3)
                ret upd: bits[4][4] = array_update(input, val, indices=[lit], id=4)
            }"#,
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let array = inputs.inputs.get("input").unwrap().bitvec.clone();
                let val = inputs.inputs.get("val").unwrap().bitvec.clone();
                let pre = solver.extract(&array, 3, 0);
                let with_mid = solver.concat(&val, &pre);
                let post = solver.extract(&array, 15, 8);
                solver.concat(&post, &with_mid)
            },
        );
    }

    pub fn test_array_update_nested<S: Solver>(solver_config: &S::Config) {
        use crate::prover::ir_equiv::test_utils::assert_smt_fn_eq;
        assert_smt_fn_eq::<S>(
            solver_config,
            r#"fn f(input: bits[8][4][4] id=1, val: bits[8][4] id=2) -> bits[8][4][4] {
                idx0: bits[2] = literal(value=1, id=3)
                ret upd: bits[8][4][4] = array_update(input, val, indices=[idx0], id=5)
            }"#,
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let array = inputs.inputs.get("input").unwrap().bitvec.clone();
                let val = inputs.inputs.get("val").unwrap().bitvec.clone();
                let pre = solver.extract(&array, 31, 0);
                let with_mid = solver.concat(&val, &pre);
                let post = solver.extract(&array, 127, 64);
                solver.concat(&post, &with_mid)
            },
        );
    }

    pub fn test_array_update_deep_nested<S: Solver>(solver_config: &S::Config) {
        use crate::prover::ir_equiv::test_utils::assert_smt_fn_eq;
        assert_smt_fn_eq::<S>(
            solver_config,
            r#"fn f(input: bits[8][2][2] id=1, val: bits[8] id=2) -> bits[8][2][2] {
                idx0: bits[2] = literal(value=1, id=3)
                idx1: bits[2] = literal(value=0, id=4)
                ret upd: bits[8][2][2] = array_update(input, val, indices=[idx0, idx1], id=5)
            }"#,
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let array = inputs.inputs.get("input").unwrap().bitvec.clone();
                let val = inputs.inputs.get("val").unwrap().bitvec.clone();
                let pre = solver.extract(&array, 15, 0);
                let with_mid = solver.concat(&val, &pre);
                let post = solver.extract(&array, 31, 24);
                solver.concat(&post, &with_mid)
            },
        );
    }

    pub fn test_extend_base<S: Solver>(
        solver_config: &S::Config,
        extend_width: usize,
        signed: bool,
    ) {
        use crate::prover::ir_equiv::test_utils::assert_smt_fn_eq;
        assert_smt_fn_eq::<S>(
            solver_config,
            &(format!(
                r#"fn f(x: bits[8]) -> bits[{}] {{
                ret {}.1: bits[{}] = {}(x, new_bit_count={}, id=1)
            }}"#,
                extend_width,
                if signed { "sign_ext" } else { "zero_ext" },
                extend_width,
                if signed { "sign_ext" } else { "zero_ext" },
                extend_width
            )),
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let x = inputs.inputs.get("x").unwrap().bitvec.clone();
                solver.extend_to(&x, extend_width, signed)
            },
        );
    }

    pub fn test_dynamic_bit_slice_base<S: Solver>(
        solver_config: &S::Config,
        start: usize,
        width: usize,
    ) {
        use crate::prover::ir_equiv::test_utils::assert_smt_fn_eq;
        assert_smt_fn_eq::<S>(
            solver_config,
            &(format!(
                r#"fn f(input: bits[8]) -> bits[{}] {{
                start: bits[4] = literal(value={}, id=2)
                ret dynamic_bit_slice.1: bits[{}] = dynamic_bit_slice(input, start, width={}, id=1)
            }}"#,
                width, start, width, width
            )),
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let input = inputs.inputs.get("input").unwrap().bitvec.clone();
                let needed_width = start + width;
                let input_ext = if needed_width > input.get_width() {
                    solver.extend_to(&input, needed_width, false)
                } else {
                    input
                };
                solver.slice(&input_ext, start, width)
            },
        );
    }

    pub fn test_bit_slice_base<S: Solver>(solver_config: &S::Config, start: usize, width: usize) {
        assert_ir_fn_equiv::<S>(
            solver_config,
            &format!(
                r#"fn f(input: bits[8]) -> bits[{}] {{
                ret bit_slice.1: bits[{}] = bit_slice(input, start={}, width={}, id=1)
            }}"#,
                width, width, start, width
            ),
            &format!(
                r#"fn f(input: bits[8]) -> bits[{}] {{
                start: bits[4] = literal(value={}, id=2)
                ret dynamic_bit_slice.1: bits[{}] = dynamic_bit_slice(input, start, width={}, id=1)
            }}"#,
                width, start, width, width
            ),
            false,
        );
    }

    pub fn test_bit_slice_update_zero<S: Solver>(solver_config: &S::Config) {
        use crate::prover::ir_equiv::test_utils::assert_smt_fn_eq;
        assert_smt_fn_eq::<S>(
            solver_config,
            r#"fn f(input: bits[8], slice: bits[4]) -> bits[8] {
                    start: bits[4] = literal(value=0, id=2)
                    ret bit_slice_update.1: bits[8] = bit_slice_update(input, start, slice, id=1)
                }"#,
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let input = inputs.inputs.get("input").unwrap().bitvec.clone();
                let slice = inputs.inputs.get("slice").unwrap().bitvec.clone();
                let input_upper = solver.slice(&input, 4, 4);
                let updated = solver.concat(&input_upper, &slice);
                updated
            },
        );
    }

    pub fn test_bit_slice_update_middle<S: Solver>(solver_config: &S::Config) {
        use crate::prover::ir_equiv::test_utils::assert_smt_fn_eq;
        assert_smt_fn_eq::<S>(
            solver_config,
            r#"fn f(input: bits[8], slice: bits[4]) -> bits[8] {
                    start: bits[4] = literal(value=1, id=2)
                    ret bit_slice_update.1: bits[8] = bit_slice_update(input, start, slice, id=1)
                }"#,
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let input = inputs.inputs.get("input").unwrap().bitvec.clone();
                let slice = inputs.inputs.get("slice").unwrap().bitvec.clone();
                let input_lower = solver.slice(&input, 0, 1);
                let input_upper = solver.slice(&input, 5, 3);
                let updated = solver.concat(&slice, &input_lower);
                let updated = solver.concat(&input_upper, &updated);
                updated
            },
        );
    }

    pub fn test_bit_slice_update_end<S: Solver>(solver_config: &S::Config) {
        use crate::prover::ir_equiv::test_utils::assert_smt_fn_eq;
        assert_smt_fn_eq::<S>(
            solver_config,
            r#"fn f(input: bits[8], slice: bits[4]) -> bits[8] {
                    start: bits[4] = literal(value=4, id=2)
                    ret bit_slice_update.1: bits[8] = bit_slice_update(input, start, slice, id=1)
                }"#,
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let input = inputs.inputs.get("input").unwrap().bitvec.clone();
                let slice = inputs.inputs.get("slice").unwrap().bitvec.clone();
                let input_lower = solver.slice(&input, 0, 4);
                let updated = solver.concat(&slice, &input_lower);
                updated
            },
        );
    }

    pub fn test_bit_slice_update_beyond_end<S: Solver>(solver_config: &S::Config) {
        use crate::prover::ir_equiv::test_utils::assert_smt_fn_eq;
        assert_smt_fn_eq::<S>(
            solver_config,
            r#"fn f(input: bits[8], slice: bits[10]) -> bits[8] {
                    start: bits[4] = literal(value=4, id=2)
                    ret bit_slice_update.1: bits[8] = bit_slice_update(input, start, slice, id=1)
                }"#,
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let input = inputs.inputs.get("input").unwrap().bitvec.clone();
                let slice = inputs.inputs.get("slice").unwrap().bitvec.clone();
                let input_lower = solver.slice(&input, 0, 4);
                let slice_extracted = solver.slice(&slice, 0, 4);
                let updated = solver.concat(&slice_extracted, &input_lower);
                updated
            },
        );
    }

    pub fn test_bit_slice_update_wider_update_value<S: Solver>(solver_config: &S::Config) {
        assert_ir_fn_equiv::<S>(
            solver_config,
            r#"fn f(input: bits[7] id=1) -> bits[5] {
                    dynamic_bit_slice.2: bits[5] = dynamic_bit_slice(input, input, width=5, id=2)
                    ret bit_slice_update.3: bits[5] = bit_slice_update(dynamic_bit_slice.2, input, input, id=3)
                }"#,
            r#"fn f(input: bits[7] id=1) -> bits[5] {
                    dynamic_bit_slice.2: bits[5] = dynamic_bit_slice(input, input, width=5, id=2)
                    ret bit_slice_update.3: bits[5] = bit_slice_update(dynamic_bit_slice.2, input, input, id=3)
                }"#,
            false,
        );
    }

    pub fn test_bit_slice_update_large_update_value<S: Solver>(solver_config: &S::Config) {
        assert_smt_fn_eq::<S>(
            solver_config,
            r#"fn f() -> bits[32] {
                    operand: bits[32] = literal(value=0xABCD1234, id=1)
                    start: bits[5] = literal(value=4, id=2)
                    upd_val: bits[80] = literal(value=0xFFFFFFFFFFFFFFFFFFF, id=3)
                    ret r: bits[32] = bit_slice_update(operand, start, upd_val, id=4)
                }"#,
            |solver: &mut S, _: &FnInputs<S::Term>| solver.from_raw_str(32, "#xFFFFFFF4"),
        );
    }

    pub fn test_fuzz_dynamic_bit_slice_shrink_panics<S: Solver>(solver_config: &S::Config) {
        let ir = r#"fn bad(input: bits[32]) -> bits[16] {
                    start: bits[4] = literal(value=4, id=2)
                    ret r: bits[16] = dynamic_bit_slice(input, start, width=16, id=1)
                }"#;
        let f = ir_parser::Parser::new(ir).parse_fn().unwrap();
        let mut solver = S::new(solver_config).unwrap();
        let prover_fn = ProverFn::new(&f, None);
        let inputs = get_fn_inputs(&mut solver, prover_fn.clone(), None);
        // This call should not panic
        let empty_map: HashMap<String, String> = HashMap::new();
        let empty_registry = super::UfRegistry {
            ufs: HashMap::new(),
        };
        let _ = ir_to_smt(&mut solver, &inputs, &empty_map, &empty_registry);
    }

    pub fn test_fuzz_ir_opt_equiv_regression_bit_slice_update_oob<S: Solver>(
        solver_config: &S::Config,
    ) {
        assert_ir_fn_equiv::<S>(
            solver_config,
            r#"fn fuzz_test(input: bits[8] id=1) -> bits[8] {
                    literal_255: bits[8] = literal(value=255, id=3)
                    bsu1: bits[8] = bit_slice_update(input, input, input, id=2)
                    ret bsu2: bits[8] = bit_slice_update(literal_255, literal_255, bsu1, id=4)
                }"#,
            r#"fn fuzz_test(input: bits[8] id=1) -> bits[8] {
                    ret literal_255: bits[8] = literal(value=255, id=3)
                }"#,
            false,
        );
    }

    pub fn test_prove_fn_equiv<S: Solver>(solver_config: &S::Config) {
        test_ir_fn_equiv_to_self::<S>(
            solver_config,
            r#"fn f(x: bits[8], y: bits[8]) -> bits[8] {
                ret identity.1: bits[8] = identity(x, id=1)
            }"#,
        );
    }

    pub fn test_prove_fn_inequiv<S: Solver>(solver_config: &S::Config) {
        assert_ir_fn_inequiv::<S>(
            solver_config,
            r#"fn f(x: bits[8]) -> bits[8] {
                    ret identity.1: bits[8] = identity(x, id=1)
                }"#,
            r#"fn g(x: bits[8]) -> bits[8] {
                    ret not.1: bits[8] = not(x, id=1)
                }"#,
            false,
        );
    }

    // --------------------------------------------------------------
    // Nary operation test helpers (Concat/And/Or/Xor/Nor/Nand)
    // --------------------------------------------------------------
    pub fn test_nary_base<S: Solver>(
        solver_config: &S::Config,
        op_name: &str,
        builder: impl Fn(
            &mut S,
            &BitVec<S::Term>,
            &BitVec<S::Term>,
            &BitVec<S::Term>,
        ) -> BitVec<S::Term>,
    ) {
        assert_smt_fn_eq::<S>(
            solver_config,
            &format!(
                r#"fn f(a: bits[4], b: bits[4], c: bits[4]) -> bits[4] {{
                    ret {op_name}.1: bits[4] = {op_name}(a, b, c, id=1)
                }}"#,
            ),
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let a = inputs.inputs.get("a").unwrap().bitvec.clone();
                let b = inputs.inputs.get("b").unwrap().bitvec.clone();
                let c = inputs.inputs.get("c").unwrap().bitvec.clone();
                builder(solver, &a, &b, &c)
            },
        );
    }

    pub fn test_ir_decode_base<S: Solver>(
        solver_config: &S::Config,
        in_width: usize,
        out_width: usize,
    ) {
        assert_smt_fn_eq::<S>(
            solver_config,
            &format!(
                r#"fn f(x: bits[{iw}]) -> bits[{ow}] {{
    ret decode.1: bits[{ow}] = decode(x, width={ow}, id=1)
}}"#,
                iw = in_width,
                ow = out_width
            ),
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let x = inputs.inputs.get("x").unwrap().bitvec.clone();
                solver.xls_decode(&x, out_width)
            },
        );
    }

    pub fn test_ir_encode_base<S: Solver>(
        solver_config: &S::Config,
        in_width: usize,
        out_width: usize,
    ) {
        assert_smt_fn_eq::<S>(
            solver_config,
            &format!(
                r#"fn f(x: bits[{iw}]) -> bits[{ow}] {{
                        ret encode.1: bits[{ow}] = encode(x, id=1)
                    }}"#,
                iw = in_width,
                ow = out_width
            ),
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let x = inputs.inputs.get("x").unwrap().bitvec.clone();
                solver.xls_encode(&x)
            },
        );
    }

    pub fn test_ir_one_hot_base<S: Solver>(solver_config: &S::Config, lsb_prio: bool) {
        assert_smt_fn_eq::<S>(
            solver_config,
            if lsb_prio {
                r#"fn f(x: bits[16]) -> bits[16] {
                        ret one_hot.1: bits[16] = one_hot(x, lsb_prio=true, id=1)
                    }"#
            } else {
                r#"fn f(x: bits[16]) -> bits[16] {
                        ret one_hot.1: bits[16] = one_hot(x, lsb_prio=false, id=1)
                    }"#
            },
            |solver: &mut S, inputs: &FnInputs<S::Term>| {
                let x = inputs.inputs.get("x").unwrap().bitvec.clone();
                if lsb_prio {
                    solver.xls_one_hot_lsb_prio(&x)
                } else {
                    solver.xls_one_hot_msb_prio(&x)
                }
            },
        );
    }

    // ----------------------------
    // Sel / OneHotSel / PrioritySel tests
    // ----------------------------

    // sel basic: selector in range (bits[2]=1) -> second case
    pub fn test_sel_basic<S: Solver>(solver_config: &S::Config) {
        assert_ir_fn_equiv::<S>(
            solver_config,
            r#"fn f() -> bits[4] {
                    selidx: bits[2] = literal(value=1, id=10)
                    a: bits[4] = literal(value=10, id=11)
                    b: bits[4] = literal(value=11, id=12)
                    c: bits[4] = literal(value=12, id=13)
                    d: bits[4] = literal(value=13, id=14)
                    ret s: bits[4] = sel(selidx, cases=[a, b, c, d], id=15)
                }"#,
            r#"fn g() -> bits[4] {
                    ret k: bits[4] = literal(value=11, id=1)
                }"#,
            false,
        );
    }

    // sel default path: selector out of range chooses default
    pub fn test_sel_default<S: Solver>(solver_config: &S::Config) {
        assert_ir_fn_equiv::<S>(
            solver_config,
            r#"fn f() -> bits[4] {
                    selidx: bits[2] = literal(value=3, id=10)
                    a: bits[4] = literal(value=10, id=11)
                    b: bits[4] = literal(value=11, id=12)
                    c: bits[4] = literal(value=12, id=13)
                    def: bits[4] = literal(value=15, id=14)
                    ret s: bits[4] = sel(selidx, cases=[a, b, c], default=def, id=15)
                }"#,
            r#"fn g() -> bits[4] {
                    ret k: bits[4] = literal(value=15, id=1)
                }"#,
            false,
        );
    }

    pub fn test_sel_missing_default_panics<S: Solver>(solver_config: &S::Config) {
        let ir = r#"fn bad() -> bits[4] {
                    idx: bits[2] = literal(value=3, id=1)
                    a: bits[4] = literal(value=1, id=2)
                    b: bits[4] = literal(value=2, id=3)
                    c: bits[4] = literal(value=3, id=4)
                    ret s: bits[4] = sel(idx, cases=[a, b, c], id=5)
                }"#;
        let f = ir_parser::Parser::new(ir).parse_fn().unwrap();
        let mut solver = S::new(solver_config).unwrap();
        let prover_fn = ProverFn::new(&f, None);
        let inputs = get_fn_inputs(&mut solver, prover_fn.clone(), None);
        // Should panic during conversion due to missing default
        let empty_map: HashMap<String, String> = HashMap::new();
        let empty_registry = super::UfRegistry {
            ufs: HashMap::new(),
        };
        let _ = ir_to_smt(&mut solver, &inputs, &empty_map, &empty_registry);
    }

    pub fn test_sel_unexpected_default_panics<S: Solver>(solver_config: &S::Config) {
        let ir = r#"fn bad() -> bits[4] {
                    idx: bits[2] = literal(value=1, id=1)
                    a: bits[4] = literal(value=1, id=2)
                    b: bits[4] = literal(value=2, id=3)
                    c: bits[4] = literal(value=3, id=4)
                    d: bits[4] = literal(value=4, id=5)
                    def: bits[4] = literal(value=5, id=6)
                    ret s: bits[4] = sel(idx, cases=[a,b,c,d], default=def, id=7)
                }"#;
        let f = ir_parser::Parser::new(ir).parse_fn().unwrap();
        let mut solver = S::new(solver_config).unwrap();
        let prover_fn = ProverFn::new(&f, None);
        let inputs = get_fn_inputs(&mut solver, prover_fn.clone(), None);
        let empty_map: HashMap<String, String> = HashMap::new();
        let empty_registry = super::UfRegistry {
            ufs: HashMap::new(),
        };
        let _ = ir_to_smt(&mut solver, &inputs, &empty_map, &empty_registry);
    }

    // one_hot_sel: multiple bits
    pub fn test_one_hot_sel_multi<S: Solver>(solver_config: &S::Config) {
        assert_ir_fn_equiv::<S>(
            solver_config,
            r#"fn f() -> bits[4] {
                    sel: bits[3] = literal(value=3, id=1)
                    c0: bits[4] = literal(value=1, id=2)
                    c1: bits[4] = literal(value=2, id=3)
                    c2: bits[4] = literal(value=4, id=4)
                    ret o: bits[4] = one_hot_sel(sel, cases=[c0, c1, c2], id=5)
                }"#,
            r#"fn g() -> bits[4] {
                    ret lit: bits[4] = literal(value=3, id=1)
                }"#,
            false,
        );
    }

    // priority_sel: multiple bits -> lowest index wins
    pub fn test_priority_sel_multi<S: Solver>(solver_config: &S::Config) {
        assert_ir_fn_equiv::<S>(
            solver_config,
            r#"fn f() -> bits[4] {
                    sel: bits[3] = literal(value=5, id=1)
                    c0: bits[4] = literal(value=1, id=2)
                    c1: bits[4] = literal(value=2, id=3)
                    c2: bits[4] = literal(value=4, id=4)
                    def: bits[4] = literal(value=8, id=5)
                    ret o: bits[4] = priority_sel(sel, cases=[c0, c1, c2], default=def, id=6)
                }"#,
            r#"fn g() -> bits[4] {
                    ret lit: bits[4] = literal(value=1, id=1)
                }"#,
            false,
        );
    }

    // priority_sel: no bits set selects default
    pub fn test_priority_sel_default<S: Solver>(solver_config: &S::Config) {
        assert_ir_fn_equiv::<S>(
            solver_config,
            r#"fn f() -> bits[4] {
                    sel: bits[3] = literal(value=0, id=1)
                    c0: bits[4] = literal(value=1, id=2)
                    c1: bits[4] = literal(value=2, id=3)
                    c2: bits[4] = literal(value=4, id=4)
                    def: bits[4] = literal(value=8, id=5)
                    ret o: bits[4] = priority_sel(sel, cases=[c0, c1, c2], default=def, id=6)
                }"#,
            r#"fn g() -> bits[4] {
                    ret lit: bits[4] = literal(value=8, id=1)
                }"#,
            false,
        );
    }

    fn assert_ir_text(is_lt: bool, assert_value: u32, ret_value: u32) -> String {
        format!(
            r#"fn f(__token: token, a: bits[4]) -> (token, bits[4]) {{
            literal.1: bits[4] = literal(value={assert_value}, id=1)
            {op}.2: bits[1] = {op}(a, literal.1, id=2)
            assert.3: token = assert(__token, {op}.2, message="Assertion failure!", label="a", id=3)
            literal.4: bits[4] = literal(value={ret_value}, id=4)
            ret tuple.4: (token, bits[4]) = tuple(assert.3, literal.4, id=4)
        }}"#,
            op = if is_lt { "ult" } else { "uge" },
            ret_value = ret_value,
        )
    }

    // Helper builders leveraging `assert_ir_text` from above.
    fn lt_ir(threshold: u32, ret_val: u32) -> String {
        // Assertion passes when a < threshold
        assert_ir_text(true, threshold, ret_val)
    }

    fn ge_ir(threshold: u32, ret_val: u32) -> String {
        // Assertion passes when a >= threshold
        assert_ir_text(false, threshold, ret_val)
    }

    fn vacuous_ir(ret_val: u32) -> String {
        assert_ir_text(true, 0, ret_val)
    }

    fn no_assertion_ir(ret_val: u32) -> String {
        assert_ir_text(false, 0, ret_val)
    }

    // ----- Same -----
    pub fn test_assert_semantics_same<S: Solver>(solver_config: &S::Config) {
        assert_ir_fn_equiv_base::<S>(
            solver_config,
            &lt_ir(1, 2),
            &lt_ir(1, 2),
            false,
            AssertionSemantics::Same,
            true,
        );
    }

    pub fn test_assert_semantics_same_different_assertion<S: Solver>(solver_config: &S::Config) {
        assert_ir_fn_equiv_base::<S>(
            solver_config,
            &lt_ir(1, 2),
            &lt_ir(2, 2),
            false,
            AssertionSemantics::Same,
            false,
        );
    }

    pub fn test_assert_semantics_same_different_result<S: Solver>(solver_config: &S::Config) {
        assert_ir_fn_equiv_base::<S>(
            solver_config,
            &lt_ir(1, 2),
            &lt_ir(1, 3),
            false,
            AssertionSemantics::Same,
            false,
        );
    }

    // ----- Ignore -----
    pub fn test_assert_semantics_ignore_proved<S: Solver>(solver_config: &S::Config) {
        let lhs = lt_ir(5, 1);
        let rhs = ge_ir(10, 1);
        assert_ir_fn_equiv_base::<S>(
            solver_config,
            &lhs,
            &rhs,
            false,
            AssertionSemantics::Ignore,
            true,
        );
    }

    pub fn test_assert_semantics_ignore_different_result<S: Solver>(solver_config: &S::Config) {
        let lhs = lt_ir(8, 1);
        let rhs = lt_ir(8, 2);
        assert_ir_fn_equiv_base::<S>(
            solver_config,
            &lhs,
            &rhs,
            false,
            AssertionSemantics::Ignore,
            false,
        );
    }

    /// Under Ignore semantics, counterexamples must not report assertion
    /// violations.
    pub fn test_assert_semantics_ignore_disproved_no_violation<S: Solver>(
        solver_config: &S::Config,
    ) {
        let lhs = lt_ir(8, 1);
        let rhs = lt_ir(8, 2);

        let mut parser = ir_parser::Parser::new(&lhs);
        let lhs_ir_fn = parser.parse_fn().unwrap();
        let mut parser = ir_parser::Parser::new(&rhs);
        let rhs_ir_fn = parser.parse_fn().unwrap();
        let lhs_pf = ProverFn::new(&lhs_ir_fn, None);
        let rhs_pf = ProverFn::new(&rhs_ir_fn, None);

        let res = super::prove_ir_fn_equiv::<S>(
            solver_config,
            &lhs_pf,
            &rhs_pf,
            AssertionSemantics::Ignore,
            None,
            false,
        );

        match res {
            super::EquivResult::Disproved {
                lhs_output,
                rhs_output,
                ..
            } => {
                assert!(lhs_output.assertion_violation.is_none());
                assert!(rhs_output.assertion_violation.is_none());
            }
            other => panic!("Expected Disproved, got {:?}", other),
        }
    }

    // ----- Never -----
    pub fn test_assert_semantics_never_proved<S: Solver>(solver_config: &S::Config) {
        let lhs = no_assertion_ir(1);
        let rhs = no_assertion_ir(1);
        assert_ir_fn_equiv_base::<S>(
            solver_config,
            &lhs,
            &rhs,
            false,
            AssertionSemantics::Never,
            true,
        );
    }

    pub fn test_assert_semantics_never_any_fail<S: Solver>(solver_config: &S::Config) {
        let lhs = lt_ir(4, 1);
        let rhs = lt_ir(4, 1);
        assert_ir_fn_equiv_base::<S>(
            solver_config,
            &lhs,
            &rhs,
            false,
            AssertionSemantics::Never,
            false,
        );
    }

    // ----- Assume -----
    pub fn test_assert_semantics_assume_proved_vacuous<S: Solver>(solver_config: &S::Config) {
        let lhs = vacuous_ir(1);
        let rhs = no_assertion_ir(2);
        assert_ir_fn_equiv_base::<S>(
            solver_config,
            &lhs,
            &rhs,
            false,
            AssertionSemantics::Assume,
            true,
        );
    }

    pub fn test_assert_semantics_assume_disproved<S: Solver>(solver_config: &S::Config) {
        let lhs = lt_ir(8, 1);
        let rhs = lt_ir(8, 2); // both succeed for same range a<8
        assert_ir_fn_equiv_base::<S>(
            solver_config,
            &lhs,
            &rhs,
            false,
            AssertionSemantics::Assume,
            false,
        );
    }

    // ----- Implies -----
    pub fn test_assert_semantics_implies_proved_lhs_fails<S: Solver>(solver_config: &S::Config) {
        let lhs = lt_ir(5, 1); // passes a<5
        let rhs = lt_ir(8, 1); // passes a<8
        assert_ir_fn_equiv_base::<S>(
            solver_config,
            &lhs,
            &rhs,
            false,
            AssertionSemantics::Implies,
            true,
        );
    }

    pub fn test_assert_semantics_implies_disproved_rhs_fails<S: Solver>(solver_config: &S::Config) {
        let lhs = lt_ir(8, 1); // passes a<8
        let rhs = lt_ir(5, 1); // passes a<5 (subset) - rhs may fail when lhs passes
        assert_ir_fn_equiv_base::<S>(
            solver_config,
            &lhs,
            &rhs,
            false,
            AssertionSemantics::Implies,
            false,
        );
    }

    fn fail_if_deactivated() -> &'static str {
        r#"fn f(__token: token, __activate: bits[1], tok: token, a: bits[4]) -> (token, bits[4]) {
                assert.3: token = assert(__token, __activate, message="Assertion failure!", label="a", id=3)
                literal.4: bits[4] = literal(value=1, id=4)
                ret tuple.4: (token, bits[4]) = tuple(assert.3, literal.4, id=4)
            }"#
    }

    fn fail_if_activated() -> &'static str {
        r#"fn f(__token: token, __activate: bits[1], tok: token, a: bits[4]) -> (token, bits[4]) {
                not.2: bits[1] = not(__activate, id=2)
                assert.3: token = assert(__token, not.2, message="Assertion failure!", label="a", id=3)
                literal.4: bits[4] = literal(value=1, id=4)
                ret tuple.4: (token, bits[4]) = tuple(assert.3, literal.4, id=4)
            }"#
    }

    fn plain_success() -> &'static str {
        r#"fn f(tok: token, a: bits[4]) -> (token, bits[4]) {
                literal.1: bits[4] = literal(value=1, id=1)
                ret tuple.2: (token, bits[4]) = tuple(tok, literal.1, id=2)
            }"#
    }

    fn plain_failure() -> &'static str {
        r#"fn f(tok: token, a: bits[4]) -> (token, bits[4]) {
                literal.1: bits[1] = literal(value=0, id=1)
                assert.2: token = assert(tok, literal.1, message="Assertion failure!", label="a", id=2)
                literal.3: bits[4] = literal(value=1, id=3)
                ret tuple.4: (token, bits[4]) = tuple(tok, literal.3, id=4)
            }"#
    }

    pub fn test_both_implicit_token_no_fixed_implicit_activation<S: Solver>(
        solver_config: &S::Config,
    ) {
        assert_ir_fn_equiv_base_with_implicit_token_policy::<S>(
            solver_config,
            &fail_if_deactivated(),
            &fail_if_deactivated(),
            false,
            false,
            false,
            AssertionSemantics::Same,
            true,
        );
    }
    pub fn test_implicit_token_policy_fixed_implicit_activation<S: Solver>(
        solver_config: &S::Config,
    ) {
        assert_ir_fn_equiv_base_with_implicit_token_policy::<S>(
            solver_config,
            &fail_if_deactivated(),
            &fail_if_deactivated(),
            true,
            true,
            false,
            AssertionSemantics::Same,
            true,
        );
    }
    pub fn test_single_fixed_implicit_activation_success<S: Solver>(solver_config: &S::Config) {
        assert_ir_fn_equiv_base_with_implicit_token_policy::<S>(
            solver_config,
            &fail_if_deactivated(),
            &plain_success(),
            true,
            false,
            false,
            AssertionSemantics::Same,
            true,
        );
    }
    pub fn test_single_fixed_implicit_activation_failure<S: Solver>(solver_config: &S::Config) {
        assert_ir_fn_equiv_base_with_implicit_token_policy::<S>(
            solver_config,
            &fail_if_activated(),
            &plain_failure(),
            true,
            false,
            false,
            AssertionSemantics::Same,
            true,
        );
    }

    pub fn test_counterexample_input_order<S: Solver>(solver_config: &S::Config) {
        // IR pair intentionally inequivalent: returns different parameters to force a
        // counterexample.
        let lhs_ir = r#"fn lhs(a: bits[8], b: bits[8]) -> bits[8] {
                ret identity.1: bits[8] = identity(a, id=1)
            }"#;
        let rhs_ir = r#"fn rhs(a: bits[8], b: bits[8]) -> bits[8] {
                ret identity.1: bits[8] = identity(b, id=1)
            }"#;
        // Parse the IR text into functions.
        let mut parser = ir_parser::Parser::new(lhs_ir);
        let lhs_fn_ir = parser.parse_fn().unwrap();
        let mut parser = ir_parser::Parser::new(rhs_ir);
        let rhs_fn_ir = parser.parse_fn().unwrap();
        let lhs_pf = ProverFn::new(&lhs_fn_ir, None);
        let rhs_pf = ProverFn::new(&rhs_fn_ir, None);

        // Run equivalence prover – expect a counter-example (Disproved).
        let res = super::prove_ir_fn_equiv::<S>(
            solver_config,
            &lhs_pf,
            &rhs_pf,
            AssertionSemantics::Same,
            None,
            false,
        );

        match res {
            EquivResult::Disproved {
                lhs_inputs,
                rhs_inputs,
                ..
            } => {
                // Verify LHS input ordering & naming.
                assert_eq!(lhs_inputs.len(), lhs_fn_ir.params.len());
                for (idx, param) in lhs_fn_ir.params.iter().enumerate() {
                    assert_eq!(lhs_inputs[idx].name, param.name);
                    assert_eq!(
                        lhs_inputs[idx].value.bit_count().unwrap(),
                        param.ty.bit_count()
                    );
                }
                // Verify RHS input ordering & naming.
                assert_eq!(rhs_inputs.len(), rhs_fn_ir.params.len());
                for (idx, param) in rhs_fn_ir.params.iter().enumerate() {
                    assert_eq!(rhs_inputs[idx].name, param.name);
                    assert_eq!(
                        rhs_inputs[idx].value.bit_count().unwrap(),
                        param.ty.bit_count()
                    );
                }
            }
            _ => panic!("Expected inequivalence with counter-example, but proof succeeded"),
        }
    }

    // New: shared test that exercises prove_ir_fn_equiv.
    pub fn test_param_domains_equiv<S: Solver>(solver_config: &S::Config) {
        let lhs_ir = r#"
            fn f(x: bits[2]) -> bits[2] {
                ret identity.1: bits[2] = identity(x, id=1)
            }
        "#;
        let rhs_ir = r#"
            fn g(x: bits[2]) -> bits[2] {
                one: bits[2] = literal(value=1, id=1)
                ret and.2: bits[2] = and(x, one, id=2)
            }
        "#;
        let mut parser = ir_parser::Parser::new(lhs_ir);
        let lhs_fn_ir = parser.parse_fn().unwrap();
        let mut parser = ir_parser::Parser::new(rhs_ir);
        let rhs_fn_ir = parser.parse_fn().unwrap();

        let lhs_pf = ProverFn::new(&lhs_fn_ir, None);
        let rhs_pf = ProverFn::new(&rhs_fn_ir, None);

        let res = super::prove_ir_fn_equiv::<S>(
            solver_config,
            &lhs_pf,
            &rhs_pf,
            AssertionSemantics::Same,
            None,
            false,
        );
        assert!(matches!(res, super::EquivResult::Disproved { .. }));

        let mut doms: ParamDomains = ParamDomains::new();
        doms.insert(
            "x".to_string(),
            vec![
                xlsynth::IrValue::make_ubits(2, 0).unwrap(),
                xlsynth::IrValue::make_ubits(2, 1).unwrap(),
            ],
        );

        let res2 = prove_ir_fn_equiv::<S>(
            solver_config,
            &lhs_pf.clone().with_domains(Some(doms.clone())),
            &rhs_pf.clone().with_domains(Some(doms)),
            AssertionSemantics::Same,
            None,
            false,
        );
        assert!(matches!(res2, super::EquivResult::Proved));
    }
}

#[cfg(all(
    test,
    any(
        feature = "with-bitwuzla-binary-test",
        feature = "with-boolector-binary-test",
        feature = "with-z3-binary-test",
        feature = "with-bitwuzla-built"
    )
))]
macro_rules! test_with_solver {
    ($mod_ident:ident, $solver_type:ty, $solver_config:expr) => {
        #[cfg(test)]
        mod $mod_ident {
            use super::*;
            use crate::prover::ir_equiv::test_utils;

            #[test]
            fn test_align_zero_width_fn_inputs() {
                test_utils::align_zero_width_fn_inputs::<$solver_type>($solver_config);
            }
            #[test]
            fn test_align_non_zero_width_fn_inputs_first_token() {
                test_utils::align_non_zero_width_fn_inputs_first_token::<$solver_type>(
                    $solver_config,
                );
            }
            #[test]
            fn test_param_domains_equiv() {
                test_utils::test_param_domains_equiv::<$solver_type>($solver_config);
            }
            #[test]
            fn test_ir_value_bits() {
                test_utils::test_ir_value_bits::<$solver_type>($solver_config);
            }
            #[test]
            fn test_invoke_basic_equiv() {
                test_utils::test_invoke_basic::<$solver_type>($solver_config);
            }
            #[test]
            fn test_invoke_two_args_equiv() {
                test_utils::test_invoke_two_args::<$solver_type>($solver_config);
            }
            #[test]
            fn test_counted_for_basic() {
                test_utils::test_counted_for_basic::<$solver_type>($solver_config);
            }
            #[test]
            fn test_counted_for_stride() {
                test_utils::test_counted_for_stride::<$solver_type>($solver_config);
            }
            #[test]
            fn test_counted_for_zero_trip() {
                test_utils::test_counted_for_zero_trip::<$solver_type>($solver_config);
            }
            #[test]
            fn test_counted_for_invariant_args() {
                test_utils::test_counted_for_invariant_args::<$solver_type>($solver_config);
            }
            #[test]
            fn test_ir_bits() {
                test_utils::test_ir_bits::<$solver_type>($solver_config);
            }
            #[test]
            fn test_ir_value_array() {
                test_utils::test_ir_value_array::<$solver_type>($solver_config);
            }
            #[test]
            fn test_ir_array() {
                test_utils::test_ir_array::<$solver_type>($solver_config);
            }
            #[test]
            fn test_ir_array_slice_basic() {
                test_utils::test_ir_array_slice_basic::<$solver_type>($solver_config);
            }
            #[test]
            fn test_ir_value_tuple() {
                test_utils::test_ir_value_tuple::<$solver_type>($solver_config);
            }
            #[test]
            fn test_ir_tuple() {
                test_utils::test_ir_tuple::<$solver_type>($solver_config);
            }
            #[test]
            fn test_ir_value_token() {
                test_utils::test_ir_value_token::<$solver_type>($solver_config);
            }
            #[test]
            fn test_not() {
                test_utils::test_unop_base::<$solver_type>($solver_config, "not", Solver::not);
            }
            #[test]
            fn test_neg() {
                test_utils::test_unop_base::<$solver_type>($solver_config, "neg", Solver::neg);
            }
            #[test]
            fn test_bit_slice() {
                test_utils::test_unop_base::<$solver_type>(
                    $solver_config,
                    "or_reduce",
                    Solver::or_reduce,
                );
            }
            #[test]
            fn test_and_reduce() {
                test_utils::test_unop_base::<$solver_type>(
                    $solver_config,
                    "and_reduce",
                    Solver::and_reduce,
                );
            }
            #[test]
            fn test_xor_reduce() {
                test_utils::test_unop_base::<$solver_type>(
                    $solver_config,
                    "xor_reduce",
                    Solver::xor_reduce,
                );
            }
            #[test]
            fn test_identity() {
                test_utils::test_unop_base::<$solver_type>(
                    $solver_config,
                    "identity",
                    |_solver, x| x.clone(),
                );
            }
            #[test]
            fn test_reverse() {
                test_utils::test_unop_base::<$solver_type>(
                    $solver_config,
                    "reverse",
                    Solver::reverse,
                );
            }
            // crate::test_all_binops!($solver_type, $solver_config);
            #[test]
            fn test_add() {
                test_utils::test_binop_8_bit::<$solver_type>($solver_config, "add", Solver::add);
            }
            #[test]
            fn test_sub() {
                test_utils::test_binop_8_bit::<$solver_type>($solver_config, "sub", Solver::sub);
            }
            #[test]
            fn test_eq() {
                test_utils::test_binop_bool::<$solver_type>($solver_config, "eq", Solver::eq);
            }
            #[test]
            fn test_ne() {
                test_utils::test_binop_bool::<$solver_type>($solver_config, "ne", Solver::ne);
            }
            #[test]
            fn test_uge() {
                test_utils::test_binop_bool::<$solver_type>($solver_config, "uge", Solver::uge);
            }
            #[test]
            fn test_ugt() {
                test_utils::test_binop_bool::<$solver_type>($solver_config, "ugt", Solver::ugt);
            }
            #[test]
            fn test_ule() {
                test_utils::test_binop_bool::<$solver_type>($solver_config, "ule", Solver::ule);
            }
            #[test]
            fn test_ult() {
                test_utils::test_binop_bool::<$solver_type>($solver_config, "ult", Solver::ult);
            }
            #[test]
            fn test_slt() {
                test_utils::test_binop_bool::<$solver_type>($solver_config, "slt", Solver::slt);
            }
            #[test]
            fn test_sle() {
                test_utils::test_binop_bool::<$solver_type>($solver_config, "sle", Solver::sle);
            }
            #[test]
            fn test_sgt() {
                test_utils::test_binop_bool::<$solver_type>($solver_config, "sgt", Solver::sgt);
            }
            #[test]
            fn test_sge() {
                test_utils::test_binop_bool::<$solver_type>($solver_config, "sge", Solver::sge);
            }

            #[test]
            fn test_xls_shll() {
                test_utils::test_binop_base::<$solver_type>(
                    $solver_config,
                    "shll",
                    Solver::xls_shll,
                    8,
                    4,
                    8,
                );
            }
            #[test]
            fn test_xls_shrl() {
                test_utils::test_binop_base::<$solver_type>(
                    $solver_config,
                    "shrl",
                    Solver::xls_shrl,
                    8,
                    4,
                    8,
                );
            }
            #[test]
            fn test_xls_shra() {
                test_utils::test_binop_base::<$solver_type>(
                    $solver_config,
                    "shra",
                    Solver::xls_shra,
                    8,
                    4,
                    8,
                );
            }

            #[test]
            fn test_xls_umul() {
                test_utils::test_binop_base::<$solver_type>(
                    $solver_config,
                    "umul",
                    |solver, x, y| Solver::xls_arbitrary_width_umul(solver, x, y, 16),
                    8,
                    12,
                    16,
                );
            }
            #[test]
            fn test_xls_smul() {
                test_utils::test_binop_base::<$solver_type>(
                    $solver_config,
                    "smul",
                    |solver, x, y| Solver::xls_arbitrary_width_smul(solver, x, y, 16),
                    8,
                    12,
                    16,
                );
            }
            #[test]
            fn test_xls_udiv() {
                test_utils::test_binop_8_bit::<$solver_type>(
                    $solver_config,
                    "udiv",
                    Solver::xls_udiv,
                );
            }

            #[test]
            fn test_xls_sdiv() {
                test_utils::test_binop_8_bit::<$solver_type>(
                    $solver_config,
                    "sdiv",
                    Solver::xls_sdiv,
                );
            }
            #[test]
            fn test_xls_umod() {
                test_utils::test_binop_8_bit::<$solver_type>(
                    $solver_config,
                    "umod",
                    Solver::xls_umod,
                );
            }
            #[test]
            fn test_xls_smod() {
                test_utils::test_binop_8_bit::<$solver_type>(
                    $solver_config,
                    "smod",
                    Solver::xls_smod,
                );
            }

            #[test]
            fn test_ir_tuple_index() {
                test_utils::test_ir_tuple_index::<$solver_type>($solver_config);
            }

            #[test]
            fn test_ir_tuple_index_literal() {
                test_utils::test_ir_tuple_index_literal::<$solver_type>($solver_config);
            }
            #[test]
            fn test_ir_tuple_reverse() {
                test_utils::test_ir_tuple_reverse::<$solver_type>($solver_config);
            }
            #[test]
            fn test_ir_tuple_flattened() {
                test_utils::test_ir_tuple_flattened::<$solver_type>($solver_config);
            }
            #[test]
            fn test_tuple_literal_vs_constructed() {
                test_utils::test_tuple_literal_vs_constructed::<$solver_type>($solver_config);
            }
            #[test]
            fn test_tuple_index_on_literal() {
                test_utils::test_tuple_index_on_literal::<$solver_type>($solver_config);
            }
            #[test]
            fn test_ir_array_index_0() {
                test_utils::test_ir_array_index_base::<$solver_type>($solver_config, "0", 0);
            }
            #[test]
            fn test_ir_array_index_1() {
                test_utils::test_ir_array_index_base::<$solver_type>($solver_config, "1", 8);
            }
            #[test]
            fn test_ir_array_index_2() {
                test_utils::test_ir_array_index_base::<$solver_type>($solver_config, "2", 16);
            }
            #[test]
            fn test_ir_array_index_3() {
                test_utils::test_ir_array_index_base::<$solver_type>($solver_config, "3", 24);
            }
            #[test]
            fn test_ir_array_index_4() {
                test_utils::test_ir_array_index_base::<$solver_type>($solver_config, "4", 24);
            }
            #[test]
            fn test_ir_array_index_multi_level() {
                test_utils::test_ir_array_index_multi_level::<$solver_type>($solver_config);
            }
            #[test]
            fn test_ir_array_index_deep_multi_level() {
                test_utils::test_ir_array_index_deep_multi_level::<$solver_type>($solver_config);
            }
            #[test]
            fn test_array_update_inbound_value() {
                test_utils::test_array_update_inbound_value::<$solver_type>($solver_config);
            }
            #[test]
            fn test_array_update_nested() {
                test_utils::test_array_update_nested::<$solver_type>($solver_config);
            }
            #[test]
            fn test_array_update_deep_nested() {
                test_utils::test_array_update_deep_nested::<$solver_type>($solver_config);
            }
            #[test]
            fn test_prove_fn_equiv() {
                test_utils::test_prove_fn_equiv::<$solver_type>($solver_config);
            }
            #[test]
            fn test_prove_fn_inequiv() {
                test_utils::test_prove_fn_inequiv::<$solver_type>($solver_config);
            }
            #[test]
            fn test_extend_zero() {
                test_utils::test_extend_base::<$solver_type>($solver_config, 16, false);
            }
            #[test]
            fn test_extend_sign() {
                test_utils::test_extend_base::<$solver_type>($solver_config, 16, true);
            }
            #[test]
            fn test_dynamic_bit_slice_0_4() {
                test_utils::test_dynamic_bit_slice_base::<$solver_type>($solver_config, 0, 4);
            }
            #[test]
            fn test_dynamic_bit_slice_5_4() {
                test_utils::test_dynamic_bit_slice_base::<$solver_type>($solver_config, 5, 4);
            }
            #[test]
            fn test_dynamic_bit_slice_0_8() {
                test_utils::test_dynamic_bit_slice_base::<$solver_type>($solver_config, 0, 8);
            }
            #[test]
            fn test_dynamic_bit_slice_5_8() {
                test_utils::test_dynamic_bit_slice_base::<$solver_type>($solver_config, 5, 8);
            }
            #[test]
            fn test_dynamic_bit_slice_10_4() {
                test_utils::test_dynamic_bit_slice_base::<$solver_type>($solver_config, 10, 4);
            }
            #[test]
            fn test_dynamic_bit_slice_10_8() {
                test_utils::test_dynamic_bit_slice_base::<$solver_type>($solver_config, 10, 8);
            }
            // crate::test_bit_slice!($solver_type, $solver_config);
            #[test]
            fn test_bit_slice_0_4() {
                test_utils::test_bit_slice_base::<$solver_type>($solver_config, 0, 4);
            }
            #[test]
            fn test_bit_slice_5_3() {
                test_utils::test_bit_slice_base::<$solver_type>($solver_config, 5, 3);
            }
            #[test]
            fn test_bit_slice_0_8() {
                test_utils::test_bit_slice_base::<$solver_type>($solver_config, 0, 8);
            }
            #[test]
            fn test_bit_slice_update_zero() {
                test_utils::test_bit_slice_update_zero::<$solver_type>($solver_config);
            }
            #[test]
            fn test_bit_slice_update_middle() {
                test_utils::test_bit_slice_update_middle::<$solver_type>($solver_config);
            }
            #[test]
            fn test_bit_slice_update_end() {
                test_utils::test_bit_slice_update_end::<$solver_type>($solver_config);
            }
            #[test]
            fn test_bit_slice_update_beyond_end() {
                test_utils::test_bit_slice_update_beyond_end::<$solver_type>($solver_config);
            }
            #[test]
            fn test_bit_slice_update_wider_update_value() {
                test_utils::test_bit_slice_update_wider_update_value::<$solver_type>(
                    $solver_config,
                );
            }
            #[test]
            fn test_bit_slice_update_large_update_value() {
                test_utils::test_bit_slice_update_large_update_value::<$solver_type>(
                    $solver_config,
                );
            }
            #[test]
            fn test_fuzz_dynamic_bit_slice_shrink_panics() {
                test_utils::test_fuzz_dynamic_bit_slice_shrink_panics::<$solver_type>(
                    $solver_config,
                );
            }
            #[test]
            fn test_fuzz_ir_opt_equiv_regression_bit_slice_update_oob() {
                test_utils::test_fuzz_ir_opt_equiv_regression_bit_slice_update_oob::<$solver_type>(
                    $solver_config,
                );
            }
            #[test]
            fn test_nary_concat() {
                test_utils::test_nary_base::<$solver_type>(
                    $solver_config,
                    "concat",
                    |solver, a, b, c| solver.concat_many([a, b, c].to_vec()),
                );
            }
            #[test]
            fn test_nary_and() {
                test_utils::test_nary_base::<$solver_type>(
                    $solver_config,
                    "and",
                    |solver, a, b, c| solver.and_many([a, b, c].to_vec()),
                );
            }
            #[test]
            fn test_nary_or() {
                test_utils::test_nary_base::<$solver_type>(
                    $solver_config,
                    "or",
                    |solver, a, b, c| solver.or_many([a, b, c].to_vec()),
                );
            }
            #[test]
            fn test_nary_xor() {
                test_utils::test_nary_base::<$solver_type>(
                    $solver_config,
                    "xor",
                    |solver, a, b, c| solver.xor_many([a, b, c].to_vec()),
                );
            }
            #[test]
            fn test_nary_nor() {
                test_utils::test_nary_base::<$solver_type>(
                    $solver_config,
                    "nor",
                    |solver, a, b, c| solver.nor_many([a, b, c].to_vec()),
                );
            }
            #[test]
            fn test_nary_nand() {
                test_utils::test_nary_base::<$solver_type>(
                    $solver_config,
                    "nand",
                    |solver, a, b, c| solver.nand_many([a, b, c].to_vec()),
                );
            }

            #[test]
            fn test_ir_decode_0() {
                test_utils::test_ir_decode_base::<$solver_type>($solver_config, 8, 8);
            }
            #[test]
            fn test_ir_decode_1() {
                test_utils::test_ir_decode_base::<$solver_type>($solver_config, 8, 16);
            }
            #[test]
            fn test_ir_encode_0() {
                test_utils::test_ir_encode_base::<$solver_type>($solver_config, 8, 3);
            }
            #[test]
            fn test_ir_encode_1() {
                test_utils::test_ir_encode_base::<$solver_type>($solver_config, 16, 4);
            }
            #[test]
            fn test_ir_one_hot_true() {
                test_utils::test_ir_one_hot_base::<$solver_type>($solver_config, true);
            }
            #[test]
            fn test_ir_one_hot_false() {
                test_utils::test_ir_one_hot_base::<$solver_type>($solver_config, false);
            }
            #[test]
            fn test_sel_basic() {
                test_utils::test_sel_basic::<$solver_type>($solver_config);
            }
            #[test]
            fn test_sel_default() {
                test_utils::test_sel_default::<$solver_type>($solver_config);
            }
            #[should_panic]
            #[test]
            fn test_sel_missing_default_panics() {
                test_utils::test_sel_missing_default_panics::<$solver_type>($solver_config);
            }
            #[should_panic]
            #[test]
            fn test_sel_unexpected_default_panics() {
                test_utils::test_sel_unexpected_default_panics::<$solver_type>($solver_config);
            }
            #[test]
            fn test_one_hot_sel_multi() {
                test_utils::test_one_hot_sel_multi::<$solver_type>($solver_config);
            }
            #[test]
            fn test_priority_sel_multi() {
                test_utils::test_priority_sel_multi::<$solver_type>($solver_config);
            }
            #[test]
            fn test_priority_sel_default() {
                test_utils::test_priority_sel_default::<$solver_type>($solver_config);
            }
            #[test]
            fn test_assert_semantics_same() {
                test_utils::test_assert_semantics_same::<$solver_type>($solver_config);
            }
            #[test]
            fn test_assert_semantics_same_different_assertion() {
                test_utils::test_assert_semantics_same_different_assertion::<$solver_type>(
                    $solver_config,
                );
            }
            #[test]
            fn test_assert_semantics_same_different_result() {
                test_utils::test_assert_semantics_same_different_result::<$solver_type>(
                    $solver_config,
                );
            }
            #[test]
            fn test_assert_semantics_ignore_proved() {
                test_utils::test_assert_semantics_ignore_proved::<$solver_type>($solver_config);
            }
            #[test]
            fn test_assert_semantics_ignore_different_result() {
                test_utils::test_assert_semantics_ignore_different_result::<$solver_type>(
                    $solver_config,
                );
            }
            #[test]
            fn test_assert_semantics_ignore_disproved_no_violation() {
                test_utils::test_assert_semantics_ignore_disproved_no_violation::<$solver_type>(
                    $solver_config,
                );
            }
            #[test]
            fn test_assert_semantics_never_proved() {
                test_utils::test_assert_semantics_never_proved::<$solver_type>($solver_config);
            }
            #[test]
            fn test_assert_semantics_never_any_fail() {
                test_utils::test_assert_semantics_never_any_fail::<$solver_type>($solver_config);
            }
            #[test]
            fn test_assert_semantics_assume_proved_vacuous() {
                test_utils::test_assert_semantics_assume_proved_vacuous::<$solver_type>(
                    $solver_config,
                );
            }
            #[test]
            fn test_assert_semantics_assume_disproved() {
                test_utils::test_assert_semantics_assume_disproved::<$solver_type>($solver_config);
            }
            #[test]
            fn test_assert_semantics_implies_proved_lhs_fails() {
                test_utils::test_assert_semantics_implies_proved_lhs_fails::<$solver_type>(
                    $solver_config,
                );
            }
            #[test]
            fn test_assert_semantics_implies_disproved_rhs_fails() {
                test_utils::test_assert_semantics_implies_disproved_rhs_fails::<$solver_type>(
                    $solver_config,
                );
            }
            #[test]
            fn test_both_implicit_token_no_fixed_implicit_activation() {
                test_utils::test_both_implicit_token_no_fixed_implicit_activation::<$solver_type>(
                    $solver_config,
                );
            }
            #[test]
            fn test_implicit_token_policy_fixed_implicit_activation() {
                test_utils::test_implicit_token_policy_fixed_implicit_activation::<$solver_type>(
                    $solver_config,
                );
            }
            #[test]
            fn test_single_fixed_implicit_activation_success() {
                test_utils::test_single_fixed_implicit_activation_success::<$solver_type>(
                    $solver_config,
                );
            }
            #[test]
            fn test_single_fixed_implicit_activation_failure() {
                test_utils::test_single_fixed_implicit_activation_failure::<$solver_type>(
                    $solver_config,
                );
            }
            #[test]
            fn test_counterexample_input_order() {
                test_utils::test_counterexample_input_order::<$solver_type>($solver_config);
            }
            #[test]
            fn test_uf_basic_equiv() {
                test_utils::test_uf_basic_equiv::<$solver_type>($solver_config);
            }
            #[test]
            fn test_uf_nested_invoke_equiv() {
                test_utils::test_uf_nested_invoke_equiv::<$solver_type>($solver_config);
            }
            #[test]
            fn test_uf_implicit_token_two_args_equiv() {
                test_utils::test_uf_implicit_token_two_args_equiv::<$solver_type>($solver_config);
            }
            #[test]
            fn test_assert_label_filter_equiv() {
                test_utils::test_assert_label_filter_equiv::<$solver_type>($solver_config);
            }
        }
    };
}

#[cfg(test)]
#[cfg(feature = "with-bitwuzla-binary-test")]
test_with_solver!(
    bitwuzla_tests,
    crate::solver::easy_smt::EasySmtSolver,
    &crate::solver::easy_smt::EasySmtConfig::bitwuzla()
);

#[cfg(test)]
#[cfg(feature = "with-boolector-binary-test")]
test_with_solver!(
    boolector_tests,
    crate::solver::easy_smt::EasySmtSolver,
    &crate::solver::easy_smt::EasySmtConfig::boolector()
);

#[cfg(test)]
#[cfg(feature = "with-z3-binary-test")]
test_with_solver!(
    z3_tests,
    crate::solver::easy_smt::EasySmtSolver,
    &crate::solver::easy_smt::EasySmtConfig::z3()
);

#[cfg(test)]
#[cfg(feature = "with-bitwuzla-built")]
test_with_solver!(
    bitwuzla_built_tests,
    crate::solver::bitwuzla::Bitwuzla,
    &crate::solver::bitwuzla::BitwuzlaOptions::new()
);
