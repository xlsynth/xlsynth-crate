// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::marker::PhantomData;

use xlsynth::IrValue;
use xlsynth_pir::corners::CornerKind;
use xlsynth_pir::ir;

use crate::solver::{BitVec, Response, Solver};

use super::translate::{get_fn_inputs, ir_to_smt_with_node_terms};
use super::types::{Assertion, IrTypedBitVec, ProverFn, SmtFnWithNodeTerms, UfRegistry};

#[derive(Debug, Clone, Copy)]
pub struct CornerProverOptions {
    pub require_pass: bool,
}

impl Default for CornerProverOptions {
    fn default() -> Self {
        Self { require_pass: true }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CornerQuery {
    pub node_text_id: usize,
    pub kind: CornerKind,
    pub tag: u8,
}

impl PartialOrd for CornerQuery {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CornerQuery {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.node_text_id, self.kind as u8, self.tag).cmp(&(
            other.node_text_id,
            other.kind as u8,
            other.tag,
        ))
    }
}

#[derive(Debug, Clone)]
pub enum CornerOrResult {
    Unsat,
    Sat { witness: IrValue },
    Unknown { message: String },
}

pub struct CornerProverSession<'a, S: Solver> {
    solver: S,
    f: &'a ir::Fn,
    node_ref_by_text_id: BTreeMap<usize, ir::NodeRef>,
    smt: SmtFnWithNodeTerms<'a, S::Term>,
}

fn mk_pass_flag<S: Solver>(
    solver: &mut S,
    assertions: &[Assertion<'_, S::Term>],
) -> BitVec<S::Term> {
    if assertions.is_empty() {
        return solver.one(1);
    }
    let mut acc_opt: Option<BitVec<S::Term>> = None;
    for a in assertions {
        acc_opt = Some(match acc_opt {
            None => a.active.clone(),
            Some(prev) => solver.and(&prev, &a.active),
        });
    }
    acc_opt.expect("acc populated")
}

impl<'a, S: Solver> CornerProverSession<'a, S> {
    fn new(
        cfg: &S::Config,
        pkg: &'a ir::Package,
        f: &'a ir::Fn,
        options: CornerProverOptions,
    ) -> Result<Self, String> {
        let mut solver = S::new(cfg).map_err(|e| e.to_string())?;
        let prover_fn = ProverFn::new(f, Some(pkg));
        let fn_inputs = get_fn_inputs(&mut solver, prover_fn, None);
        let empty_map: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        let empty_registry = UfRegistry {
            ufs: std::collections::HashMap::new(),
        };
        let smt = ir_to_smt_with_node_terms(&mut solver, &fn_inputs, &empty_map, &empty_registry);

        let pass_flag = mk_pass_flag(&mut solver, &smt.smt_fn.assertions);
        if options.require_pass {
            solver.assert(&pass_flag).map_err(|e| e.to_string())?;
        }

        let mut node_ref_by_text_id: BTreeMap<usize, ir::NodeRef> = BTreeMap::new();
        for (idx, n) in f.nodes.iter().enumerate() {
            node_ref_by_text_id.insert(n.text_id, ir::NodeRef { index: idx });
        }

        Ok(Self {
            solver,
            f,
            node_ref_by_text_id,
            smt,
        })
    }

    fn get_node_ref(&self, node_text_id: usize) -> Result<ir::NodeRef, String> {
        self.node_ref_by_text_id
            .get(&node_text_id)
            .copied()
            .ok_or_else(|| format!("node_text_id not found: {}", node_text_id))
    }

    fn term_for_ref(&self, r: ir::NodeRef) -> Result<&IrTypedBitVec<'a, S::Term>, String> {
        let text_id = self.f.nodes[r.index].text_id;
        self.smt
            .node_terms
            .get(&text_id)
            .ok_or_else(|| format!("no SMT term for node_text_id={}", text_id))
    }

    fn build_predicate(&mut self, q: CornerQuery) -> Result<BitVec<S::Term>, String> {
        let nr = self.get_node_ref(q.node_text_id)?;
        let n = &self.f.nodes[nr.index];
        match q.kind {
            CornerKind::Add => {
                let ir::NodePayload::Binop(ir::Binop::Add, lhs, rhs) = &n.payload else {
                    return Err(format!("node {} is not add", q.node_text_id));
                };
                match q.tag {
                    0 => {
                        let lhs_bv = self.term_for_ref(*lhs)?.bitvec.clone();
                        Ok(self.solver.is_zero(&lhs_bv))
                    }
                    1 => {
                        let rhs_bv = self.term_for_ref(*rhs)?.bitvec.clone();
                        Ok(self.solver.is_zero(&rhs_bv))
                    }
                    _ => Err(format!("Add: unknown tag {}", q.tag)),
                }
            }
            CornerKind::Neg => {
                let ir::NodePayload::Unop(ir::Unop::Neg, operand) = &n.payload else {
                    return Err(format!("node {} is not neg", q.node_text_id));
                };
                let op_bv = self.term_for_ref(*operand)?.bitvec.clone();
                match q.tag {
                    0 => Ok(self.solver.is_signed_min_value(&op_bv)),
                    1 => {
                        let sign = self.solver.sign_bit(&op_bv);
                        let one = self.solver.one(1);
                        Ok(self.solver.eq(&sign, &one))
                    }
                    _ => Err(format!("Neg: unknown tag {}", q.tag)),
                }
            }
            CornerKind::Shift => {
                let ir::NodePayload::Binop(binop, lhs, rhs) = &n.payload else {
                    return Err(format!("node {} is not binop", q.node_text_id));
                };
                if !matches!(binop, ir::Binop::Shll | ir::Binop::Shrl) {
                    return Err(format!("node {} is not shll/shrl", q.node_text_id));
                }
                let lhs_bv = self.term_for_ref(*lhs)?.bitvec.clone();
                let rhs_bv = self.term_for_ref(*rhs)?.bitvec.clone();
                let lhs_w = lhs_bv.get_width();
                let rhs_w = rhs_bv.get_width();
                let width_const: BitVec<S::Term> = self.solver.numerical_u128(rhs_w, lhs_w as u128);
                match q.tag {
                    0 => Ok(self.solver.is_zero(&rhs_bv)),
                    1 => Ok(self.solver.ult(&rhs_bv, &width_const)),
                    2 => Ok(self.solver.uge(&rhs_bv, &width_const)),
                    _ => Err(format!("Shift: unknown tag {}", q.tag)),
                }
            }
            _ => Err(format!("unsupported corner kind in MVP: {:?}", q.kind)),
        }
    }

    fn mk_or_predicate(&mut self, qs: &[CornerQuery]) -> Result<BitVec<S::Term>, String> {
        if qs.is_empty() {
            return Ok(self.solver.false_bv());
        }
        let mut acc = self.solver.false_bv();
        for &q in qs {
            let p = self.build_predicate(q)?;
            acc = self.solver.or(&acc, &p);
        }
        Ok(acc)
    }

    fn extract_witness_tuple(&mut self) -> Result<IrValue, String> {
        let mut elems: Vec<IrValue> = Vec::new();
        for inp in self.smt.smt_fn.inputs.iter() {
            let v = self
                .solver
                .get_value(&inp.bitvec, inp.ir_type)
                .map_err(|e| e.to_string())?;
            elems.push(v);
        }
        Ok(IrValue::make_tuple(&elems))
    }

    fn solve_any(&mut self, qs: &[CornerQuery]) -> Result<CornerOrResult, String> {
        self.solver.push().map_err(|e| e.to_string())?;
        let pred = self.mk_or_predicate(qs)?;
        self.solver.assert(&pred).map_err(|e| e.to_string())?;
        let r = self.solver.check().map_err(|e| e.to_string())?;
        let out = match r {
            Response::Unsat => CornerOrResult::Unsat,
            Response::Unknown => CornerOrResult::Unknown {
                message: "solver returned unknown".to_string(),
            },
            Response::Sat => {
                let witness = self.extract_witness_tuple()?;
                CornerOrResult::Sat { witness }
            }
        };
        self.solver.pop().map_err(|e| e.to_string())?;
        Ok(out)
    }
}

pub enum CornerProver<'a> {
    #[cfg(feature = "has-bitwuzla")]
    Bitwuzla(CornerProverSession<'a, crate::solver::bitwuzla::Bitwuzla>),
    #[cfg(feature = "has-boolector")]
    Boolector(CornerProverSession<'a, crate::solver::boolector::Boolector>),
    #[cfg(feature = "has-easy-smt")]
    EasySmt(CornerProverSession<'a, crate::solver::easy_smt::EasySmtSolver>),
    /// No SMT backend is enabled in this build.
    NoSolver(PhantomData<&'a ()>),
}

impl<'a> CornerProver<'a> {
    pub fn new_auto(
        pkg: &'a ir::Package,
        f: &'a ir::Fn,
        options: CornerProverOptions,
    ) -> Result<Self, String> {
        #[cfg(feature = "has-bitwuzla")]
        {
            let cfg = crate::solver::bitwuzla::BitwuzlaOptions::new();
            let s = CornerProverSession::new(&cfg, pkg, f, options)?;
            return Ok(CornerProver::Bitwuzla(s));
        }
        #[cfg(all(feature = "has-boolector", not(feature = "has-bitwuzla")))]
        {
            let cfg = crate::solver::boolector::BoolectorConfig::new();
            let s = CornerProverSession::new(&cfg, pkg, f, options)?;
            return Ok(CornerProver::Boolector(s));
        }
        #[cfg(all(
            feature = "has-easy-smt",
            not(feature = "has-bitwuzla"),
            not(feature = "has-boolector")
        ))]
        {
            use crate::solver::easy_smt::EasySmtConfig;

            // Try a few common binaries.
            let candidates = [
                EasySmtConfig::z3(),
                EasySmtConfig::boolector(),
                EasySmtConfig::bitwuzla(),
            ];
            for cfg in candidates.iter() {
                if crate::solver::easy_smt::EasySmtSolver::new(cfg).is_ok() {
                    let s = CornerProverSession::new(cfg, pkg, f, options)?;
                    return Ok(CornerProver::EasySmt(s));
                }
            }
            return Err("no usable easy-smt backend found (z3/boolector/bitwuzla)".to_string());
        }
        #[cfg(all(
            not(feature = "has-bitwuzla"),
            not(feature = "has-boolector"),
            not(feature = "has-easy-smt")
        ))]
        {
            let _ = pkg;
            let _ = f;
            let _ = options;
            Err(
                "no in-process solver enabled (enable xlsynth-prover with bitwuzla/boolector/easy-smt)"
                    .to_string(),
            )
        }
    }

    pub fn solve_any(&mut self, qs: &[CornerQuery]) -> Result<CornerOrResult, String> {
        match self {
            #[cfg(feature = "has-bitwuzla")]
            CornerProver::Bitwuzla(s) => s.solve_any(qs),
            #[cfg(feature = "has-boolector")]
            CornerProver::Boolector(s) => s.solve_any(qs),
            #[cfg(feature = "has-easy-smt")]
            CornerProver::EasySmt(s) => s.solve_any(qs),
            CornerProver::NoSolver(_) => {
                let _ = qs;
                Err(
                "no in-process solver enabled (enable xlsynth-prover with bitwuzla/boolector/easy-smt)"
                    .to_string(),
                )
            }
        }
    }
}

#[cfg(all(test, feature = "has-bitwuzla"))]
mod tests {
    use super::*;
    use xlsynth_pir::corners::CornerEvent;
    use xlsynth_pir::ir_eval::{EvalObserver, FnEvalResult, SelectEvent};
    use xlsynth_pir::ir_parser::Parser;

    #[derive(Default)]
    struct RecordingCornerObserver {
        events: Vec<(usize, CornerKind, u8)>,
    }

    impl EvalObserver for RecordingCornerObserver {
        fn on_select(&mut self, _ev: SelectEvent) {}

        fn on_corner_event(&mut self, ev: CornerEvent) {
            self.events.push((ev.node_text_id, ev.kind, ev.tag));
        }
    }

    #[test]
    fn or_prover_finds_witness_and_witness_hits_a_corner() {
        let ir_text = r#"package test

fn f(x: bits[8] id=1, y: bits[8] id=2, shamt: bits[8] id=3) -> bits[8] {
  add0: bits[8] = add(x, y, id=10)
  ret sh0: bits[8] = shll(add0, shamt, id=11)
}
"#;
        let mut parser = Parser::new(ir_text);
        let pkg = parser.parse_and_validate_package().unwrap();
        let f = pkg.get_fn("f").unwrap();

        let mut prover = CornerProver::new_auto(&pkg, f, CornerProverOptions::default()).unwrap();
        let qs = vec![
            CornerQuery {
                node_text_id: 10,
                kind: CornerKind::Add,
                tag: 0, // lhs is zero
            },
            CornerQuery {
                node_text_id: 11,
                kind: CornerKind::Shift,
                tag: 0, // amount is zero
            },
        ];
        let r = prover.solve_any(&qs).unwrap();
        let witness = match r {
            CornerOrResult::Sat { witness } => witness,
            other => panic!("expected sat, got {:?}", other),
        };

        let args = witness.get_elements().unwrap();
        let mut obs = RecordingCornerObserver::default();
        let eval = xlsynth_pir::ir_eval::eval_fn_with_observer(f, &args, Some(&mut obs));
        assert!(matches!(eval, FnEvalResult::Success(_)));
        assert!(
            obs.events.contains(&(10, CornerKind::Add, 0))
                || obs.events.contains(&(11, CornerKind::Shift, 0)),
            "expected witness to hit at least one queried corner; got events={:?} witness={}",
            obs.events,
            witness
        );
    }

    #[test]
    fn or_prover_unsat_when_corner_is_impossible() {
        // shift amount is a literal 1, so the corner amount==0 is unreachable.
        let ir_text = r#"package test

fn f(x: bits[8] id=1) -> bits[8] {
  one: bits[8] = literal(value=1, id=9)
  ret sh0: bits[8] = shll(x, one, id=10)
}
"#;
        let mut parser = Parser::new(ir_text);
        let pkg = parser.parse_and_validate_package().unwrap();
        let f = pkg.get_fn("f").unwrap();

        let mut prover = CornerProver::new_auto(&pkg, f, CornerProverOptions::default()).unwrap();
        let qs = vec![CornerQuery {
            node_text_id: 10,
            kind: CornerKind::Shift,
            tag: 0,
        }];
        let r = prover.solve_any(&qs).unwrap();
        assert!(matches!(r, CornerOrResult::Unsat));
    }
}
