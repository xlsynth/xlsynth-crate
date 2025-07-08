// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "has-easy-smt")]

use std::{
    collections::{HashMap, HashSet},
    io,
    path::PathBuf,
    sync::{Arc, Mutex, MutexGuard},
};

use easy_smt::{Context, ContextBuilder, SExpr};

use crate::{
    equiv::solver_interface::{BitVec, Solver},
    ir_value_utils::{ir_bits_from_lsb_is_0, ir_value_from_bits_with_type},
    test_solver,
    xls_ir::ir,
};

#[derive(Debug, Clone)]
pub struct SolverFn {
    pub push_fn: fn(&mut Context) -> io::Result<()>,
    pub pop_fn: fn(&mut Context) -> io::Result<()>,
    pub check_fn: fn(&mut Context) -> io::Result<easy_smt::Response>,
    pub assert_fn: fn(&mut Context, SExpr) -> io::Result<()>,
}

impl SolverFn {
    pub fn default() -> Self {
        Self {
            push_fn: Context::push,
            pop_fn: Context::pop,
            check_fn: Context::check,
            assert_fn: Context::assert,
        }
    }
}

#[derive(Clone)]
pub struct EasySMTConfig {
    pub solver_path: PathBuf,
    pub solver_args: Vec<String>,
    pub replay_file: Option<PathBuf>,
    pub solver_fn: SolverFn,
}

impl EasySMTConfig {
    pub fn bitwuzla() -> Self {
        Self {
            solver_path: PathBuf::from("bitwuzla"),
            solver_args: ["--produce-models".to_string()]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            replay_file: None,
            solver_fn: SolverFn {
                push_fn: |ctx| ctx.push_many(1),
                pop_fn: |ctx| ctx.pop_many(1),
                ..SolverFn::default()
            },
        }
    }

    pub fn boolector() -> Self {
        Self {
            solver_path: PathBuf::from("boolector"),
            solver_args: [
                "--smt2",
                "-m",
                "--output-format=smt2",
                "--no-exit-codes",
                "--incremental",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            replay_file: None,
            solver_fn: SolverFn {
                push_fn: |ctx| ctx.push_many(1),
                pop_fn: |ctx| ctx.pop_many(1),
                ..SolverFn::default()
            },
        }
    }

    pub fn z3() -> Self {
        Self {
            solver_path: PathBuf::from("z3"),
            solver_args: ["-nw", "-smt2", "-in"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            replay_file: None,
            solver_fn: SolverFn::default(),
        }
    }
}

pub struct EasySMTSolver {
    context: Arc<Mutex<Context>>,
    solver_fn: SolverFn,
    next_name_index: usize,
    term_cache: HashMap<String, SExpr>,
    reverse_cache: HashMap<SExpr, String>,
}

impl EasySMTSolver {
    fn get_next_available_name(&mut self) -> String {
        let mut name = format!("t{}", self.next_name_index);
        while self.term_cache.contains_key(&name) {
            self.next_name_index += 1;
            name = format!("t{}", self.next_name_index);
        }
        name
    }
    fn cache_term(
        &mut self,
        bit_vec: BitVec<SExpr>,
        context: &mut MutexGuard<Context>,
    ) -> BitVec<SExpr> {
        let rep = match bit_vec {
            BitVec::BitVec { rep, .. } => rep.clone(),
            BitVec::ZeroWidth => {
                return bit_vec;
            }
        };
        if rep.is_atom() {
            return bit_vec;
        }
        if self.reverse_cache.contains_key(&rep) {
            let name = self.reverse_cache.get(&rep).unwrap();
            return BitVec::BitVec {
                width: bit_vec.get_width(),
                rep: context.atom(name),
            };
        }
        let name = self.get_next_available_name();
        self.term_cache.insert(name.clone(), rep.clone());
        let bv_sort = context.bit_vec_sort(context.numeral(bit_vec.get_width()));
        let new_expr = context
            .define_const(name.clone(), bv_sort, rep.clone())
            .unwrap();
        self.reverse_cache.insert(rep.clone(), name.clone());
        self.reverse_cache.insert(new_expr.clone(), name.clone());
        BitVec::BitVec {
            width: bit_vec.get_width(),
            rep: new_expr,
        }
    }
    fn bool_to(&mut self, context: &mut MutexGuard<Context>, value: SExpr) -> BitVec<SExpr> {
        self.cache_term(
            BitVec::BitVec {
                width: 1,
                rep: context.ite(value, context.binary(1, 1), context.binary(1, 0)),
            },
            context,
        )
    }

    fn bv_to_bool(&mut self, context: &Context, value: &BitVec<SExpr>) -> SExpr {
        match value {
            BitVec::BitVec { rep, width: 1 } => context.eq(rep.clone(), context.binary(1, 1)),
            _ => panic!(
                "Invalid bitvector width for boolean: {:?}",
                value.get_width()
            ),
        }
    }

    fn unary_op<F>(&mut self, bit_vec: &BitVec<SExpr>, op: F) -> BitVec<SExpr>
    where
        F: Fn(&Context, SExpr) -> SExpr,
    {
        let shared = Arc::clone(&self.context);
        let mut context = shared.lock().unwrap();
        let bv = match bit_vec {
            BitVec::BitVec { width, rep } => BitVec::BitVec {
                width: *width,
                rep: op(&context, rep.clone()),
            },
            BitVec::ZeroWidth => BitVec::ZeroWidth,
        };
        self.cache_term(bv, &mut context)
    }

    fn bin_bool_op<F>(
        &mut self,
        lhs: &BitVec<SExpr>,
        rhs: &BitVec<SExpr>,
        op: F,
        zero_width_result: BitVec<SExpr>,
    ) -> BitVec<SExpr>
    where
        F: Fn(&Context, SExpr, SExpr) -> SExpr,
    {
        let shared = Arc::clone(&self.context);
        let mut context = shared.lock().unwrap();
        match (lhs, rhs) {
            (BitVec::BitVec { width: w1, rep: r1 }, BitVec::BitVec { width: w2, rep: r2 }) => {
                assert_eq!(w1, w2, "Bitvector width mismatch");
                let res = op(&context, r1.clone(), r2.clone());
                self.bool_to(&mut context, res)
            }
            (BitVec::ZeroWidth, BitVec::ZeroWidth) => {
                self.cache_term(zero_width_result, &mut context)
            }
            _ => panic!("Bitvector width mismatch"),
        }
    }

    fn bin_op<F>(&mut self, lhs: &BitVec<SExpr>, rhs: &BitVec<SExpr>, op: F) -> BitVec<SExpr>
    where
        F: Fn(&Context, SExpr, SExpr) -> SExpr,
    {
        let shared = Arc::clone(&self.context);
        let mut context = shared.lock().unwrap();
        match (lhs, rhs) {
            (BitVec::BitVec { width: w1, rep: r1 }, BitVec::BitVec { width: w2, rep: r2 }) => {
                assert_eq!(w1, w2, "Bitvector width mismatch");
                self.cache_term(
                    BitVec::BitVec {
                        width: *w1,
                        rep: op(&context, r1.clone(), r2.clone()),
                    },
                    &mut context,
                )
            }
            (BitVec::ZeroWidth, BitVec::ZeroWidth) => BitVec::ZeroWidth,
            _ => panic!("Bitvector width mismatch"),
        }
    }

    fn reduce<F>(&mut self, bit_vec: &BitVec<SExpr>, op: F) -> BitVec<SExpr>
    where
        F: Fn(&Context, SExpr, SExpr) -> SExpr,
    {
        let shared = Arc::clone(&self.context);
        let mut context = shared.lock().unwrap();
        match bit_vec {
            BitVec::BitVec { width, rep } => {
                let mut res = context.extract(0, 0, rep.clone());
                for i in 1..*width {
                    let bit = context.extract(i as i32, i as i32, rep.clone());
                    res = op(&context, res.clone(), bit);
                }
                self.cache_term(BitVec::BitVec { width: 1, rep: res }, &mut context)
            }
            BitVec::ZeroWidth => panic!("Cannot reduce zero-width bitvector"),
        }
    }
}

impl Solver for EasySMTSolver {
    type Rep = SExpr;
    type Config = EasySMTConfig;
    fn new(config: &EasySMTConfig) -> io::Result<EasySMTSolver> {
        let mut builder = ContextBuilder::new();
        if let Some(ref replay_file) = config.replay_file {
            builder.replay_file(Some(std::fs::File::create(replay_file)?));
        }
        builder.solver(&config.solver_path);
        builder.solver_args(&config.solver_args);
        let context = builder.build()?;
        Ok(EasySMTSolver {
            context: Arc::new(Mutex::new(context)),
            solver_fn: config.solver_fn.clone(),
            next_name_index: 0,
            term_cache: HashMap::new(),
            reverse_cache: HashMap::new(),
        })
    }

    fn declare(&mut self, name: &str, width: usize) -> io::Result<BitVec<SExpr>> {
        if width == 0 {
            return Ok(BitVec::ZeroWidth);
        }
        let shared = Arc::clone(&self.context);
        let mut context = shared.lock().unwrap();
        let width_numeral = context.numeral(width as i64);
        let sort = context.bit_vec_sort(width_numeral);
        let rep = context.declare_const(name, sort)?;
        Ok(self.cache_term(BitVec::BitVec { width, rep }, &mut context))
    }

    fn numerical(&mut self, width: usize, mut value: u64) -> BitVec<SExpr> {
        assert!(width > 0, "Width must be positive");
        let shared = Arc::clone(&self.context);
        let mut context = shared.lock().unwrap();
        if width < 64 {
            value &= (1 << width) - 1;
        }
        let rep = context.binary(width, value);
        self.cache_term(BitVec::BitVec { width, rep }, &mut context)
    }

    fn from_raw_str(&mut self, width: usize, value: &str) -> BitVec<SExpr> {
        assert!(width > 0, "Width must be positive");
        let shared = Arc::clone(&self.context);
        let mut context = shared.lock().unwrap();
        self.cache_term(
            BitVec::BitVec {
                width,
                rep: context.atom(value),
            },
            &mut context,
        )
    }

    fn get_value(
        &mut self,
        bit_vec: &BitVec<SExpr>,
        ty: &ir::Type,
    ) -> io::Result<xlsynth::IrValue> {
        let shared = Arc::clone(&self.context);
        let mut context = shared.lock().unwrap();
        match bit_vec {
            BitVec::BitVec { rep, .. } => {
                let value = context.get_value(vec![rep.clone()])?[0].1;
                let atom = context.get_atom(value).expect("model value must be atom");
                let bitstr = if let Some(rest) = atom.strip_prefix("#b") {
                    rest.to_string()
                } else if let Some(rest) = atom.strip_prefix("#x") {
                    rest.chars()
                        .map(|c| match c {
                            '0'..='9' => c.to_digit(16).unwrap(),
                            'a'..='f' => c.to_digit(16).unwrap(),
                            _ => panic!("Invalid hex character: {}", c),
                        })
                        .map(|d| format!("{:04b}", d))
                        .collect::<Vec<_>>()
                        .join("")
                } else {
                    panic!("Invalid atom: {}", atom);
                };
                let bits: Vec<bool> = bitstr.chars().rev().map(|c| c == '1').collect();
                Ok(ir_value_from_bits_with_type(
                    &ir_bits_from_lsb_is_0(&bits),
                    ty,
                ))
            }
            BitVec::ZeroWidth => panic!("Cannot get value of zero-width bitvector"),
        }
    }

    fn extract(&mut self, bit_vec: &BitVec<SExpr>, high: i32, low: i32) -> BitVec<SExpr> {
        if high == low - 1 {
            return BitVec::ZeroWidth;
        }
        let shared = Arc::clone(&self.context);
        let mut context = shared.lock().unwrap();
        match bit_vec {
            BitVec::BitVec { width, rep } => {
                assert!(
                    high >= low,
                    "Invalid bit slice: high = {}, low = {}",
                    high,
                    low
                );
                assert!(
                    high < *width as i32,
                    "Invalid bit slice: high = {}, width = {}",
                    high,
                    width
                );
                assert!(low >= 0, "Invalid bit slice: low = {}", low);
                let new_width = high - low + 1;
                self.cache_term(
                    BitVec::BitVec {
                        width: new_width as usize,
                        rep: context.extract(high, low, rep.clone()),
                    },
                    &mut context,
                )
            }
            BitVec::ZeroWidth => panic!("Cannot extract from zero-width bitvector"),
        }
    }

    fn not(&mut self, bit_vec: &BitVec<SExpr>) -> BitVec<SExpr> {
        self.unary_op(bit_vec, Context::bvnot)
    }

    fn neg(&mut self, bit_vec: &BitVec<SExpr>) -> BitVec<SExpr> {
        self.unary_op(bit_vec, Context::bvneg)
    }

    fn reverse(&mut self, bit_vec: &BitVec<SExpr>) -> BitVec<SExpr> {
        let shared = Arc::clone(&self.context);
        let mut context = shared.lock().unwrap();
        match bit_vec {
            BitVec::BitVec { width, rep } => {
                let mut res = context.extract(0, 0, rep.clone());
                for i in 1..*width {
                    let bit = context.extract(i as i32, i as i32, rep.clone());
                    res = context.concat(res, bit);
                }
                self.cache_term(
                    BitVec::BitVec {
                        width: *width,
                        rep: res,
                    },
                    &mut context,
                )
            }
            BitVec::ZeroWidth => BitVec::ZeroWidth,
        }
    }

    fn or_reduce(&mut self, bit_vec: &BitVec<SExpr>) -> BitVec<SExpr> {
        self.reduce(bit_vec, Context::bvor)
    }

    fn and_reduce(&mut self, bit_vec: &BitVec<SExpr>) -> BitVec<SExpr> {
        self.reduce(bit_vec, Context::bvand)
    }

    fn xor_reduce(&mut self, bit_vec: &BitVec<SExpr>) -> BitVec<SExpr> {
        self.reduce(bit_vec, Context::bvxor)
    }

    fn add(&mut self, lhs: &BitVec<SExpr>, rhs: &BitVec<SExpr>) -> BitVec<SExpr> {
        self.bin_op(lhs, rhs, Context::bvadd)
    }

    fn sub(&mut self, lhs: &BitVec<SExpr>, rhs: &BitVec<SExpr>) -> BitVec<SExpr> {
        self.bin_op(lhs, rhs, Context::bvsub)
    }

    fn mul(&mut self, lhs: &BitVec<SExpr>, rhs: &BitVec<SExpr>) -> BitVec<SExpr> {
        self.bin_op(lhs, rhs, Context::bvmul)
    }

    fn udiv(&mut self, lhs: &BitVec<SExpr>, rhs: &BitVec<SExpr>) -> BitVec<SExpr> {
        self.bin_op(lhs, rhs, Context::bvudiv)
    }

    fn urem(&mut self, lhs: &BitVec<SExpr>, rhs: &BitVec<SExpr>) -> BitVec<SExpr> {
        self.bin_op(lhs, rhs, Context::bvurem)
    }

    fn srem(&mut self, lhs: &BitVec<SExpr>, rhs: &BitVec<SExpr>) -> BitVec<SExpr> {
        self.bin_op(lhs, rhs, Context::bvsrem)
    }

    fn sdiv(&mut self, lhs: &BitVec<SExpr>, rhs: &BitVec<SExpr>) -> BitVec<SExpr> {
        self.bin_op(lhs, rhs, Context::bvsdiv)
    }

    fn shl(&mut self, lhs: &BitVec<SExpr>, rhs: &BitVec<SExpr>) -> BitVec<SExpr> {
        self.bin_op(lhs, rhs, Context::bvshl)
    }

    fn lshr(&mut self, lhs: &BitVec<SExpr>, rhs: &BitVec<SExpr>) -> BitVec<SExpr> {
        self.bin_op(lhs, rhs, Context::bvlshr)
    }

    fn ashr(&mut self, lhs: &BitVec<SExpr>, rhs: &BitVec<SExpr>) -> BitVec<SExpr> {
        self.bin_op(lhs, rhs, Context::bvashr)
    }

    fn concat(&mut self, lhs: &BitVec<SExpr>, rhs: &BitVec<SExpr>) -> BitVec<SExpr> {
        let shared = Arc::clone(&self.context);
        let mut context = shared.lock().unwrap();
        match (&lhs, &rhs) {
            (BitVec::BitVec { width: w1, rep: r1 }, BitVec::BitVec { width: w2, rep: r2 }) => self
                .cache_term(
                    BitVec::BitVec {
                        width: w1 + w2,
                        rep: context.concat(r1.clone(), r2.clone()),
                    },
                    &mut context,
                ),
            (BitVec::ZeroWidth, _) => rhs.clone(),
            (_, BitVec::ZeroWidth) => lhs.clone(),
        }
    }

    fn or(&mut self, lhs: &BitVec<SExpr>, rhs: &BitVec<SExpr>) -> BitVec<SExpr> {
        self.bin_op(lhs, rhs, Context::bvor)
    }

    fn and(&mut self, lhs: &BitVec<SExpr>, rhs: &BitVec<SExpr>) -> BitVec<SExpr> {
        self.bin_op(lhs, rhs, Context::bvand)
    }

    fn xor(&mut self, lhs: &BitVec<SExpr>, rhs: &BitVec<SExpr>) -> BitVec<SExpr> {
        self.bin_op(lhs, rhs, Context::bvxor)
    }

    fn nand(&mut self, lhs: &BitVec<SExpr>, rhs: &BitVec<SExpr>) -> BitVec<SExpr> {
        self.bin_op(lhs, rhs, |ctx, r1, r2| ctx.bvnot(ctx.bvand(r1, r2)))
    }

    fn nor(&mut self, lhs: &BitVec<SExpr>, rhs: &BitVec<SExpr>) -> BitVec<SExpr> {
        self.bin_op(lhs, rhs, |ctx, r1, r2| ctx.bvnot(ctx.bvor(r1, r2)))
    }

    fn extend(
        &mut self,
        bit_vec: &BitVec<SExpr>,
        extend_width: usize,
        signed: bool,
    ) -> BitVec<SExpr> {
        let shared = Arc::clone(&self.context);
        let mut context = shared.lock().unwrap();
        match bit_vec {
            BitVec::BitVec { width, rep } => {
                if extend_width == 0 {
                    return bit_vec.clone();
                }
                let extend_operator = if signed {
                    context.atom("sign_extend")
                } else {
                    context.atom("zero_extend")
                };
                let extend_rep = context.list(vec![
                    context.list(vec![
                        context.atoms().und,
                        extend_operator,
                        context.numeral(extend_width),
                    ]),
                    rep.clone(),
                ]);
                self.cache_term(
                    BitVec::BitVec {
                        width: width + extend_width,
                        rep: extend_rep,
                    },
                    &mut context,
                )
            }
            BitVec::ZeroWidth => panic!("Cannot extend zero-width bitvector"),
        }
    }

    fn ite(
        &mut self,
        lhs: &BitVec<SExpr>,
        then: &BitVec<SExpr>,
        else_: &BitVec<SExpr>,
    ) -> BitVec<SExpr> {
        let shared = Arc::clone(&self.context);
        let mut context = shared.lock().unwrap();
        match (lhs.clone(), then, else_) {
            (
                BitVec::BitVec { width: w1, rep: _ },
                BitVec::BitVec { width: w2, rep: r2 },
                BitVec::BitVec { width: w3, rep: r3 },
            ) => {
                assert_eq!(w1, 1, "Condition must be 1-bit");
                assert_eq!(w2, w3, "Then and else must have the same width");
                let condition = self.bv_to_bool(&context, lhs);
                self.cache_term(
                    BitVec::BitVec {
                        width: *w2,
                        rep: context.ite(condition, r2.clone(), r3.clone()),
                    },
                    &mut context,
                )
            }
            (BitVec::BitVec { width: w1, rep: _ }, BitVec::ZeroWidth, BitVec::ZeroWidth) => {
                assert_eq!(w1, 1, "Condition must be 1-bit");
                BitVec::ZeroWidth
            }
            _ => panic!("Bitvector width mismatch"),
        }
    }

    fn eq(&mut self, lhs: &BitVec<SExpr>, rhs: &BitVec<SExpr>) -> BitVec<SExpr> {
        let result = self.true_bv();
        self.bin_bool_op(lhs, rhs, Context::eq, result)
    }

    fn ne(&mut self, lhs: &BitVec<SExpr>, rhs: &BitVec<SExpr>) -> BitVec<SExpr> {
        let result = self.false_bv();
        self.bin_bool_op(lhs, rhs, |ctx, r1, r2| ctx.not(ctx.eq(r1, r2)), result)
    }

    fn slt(&mut self, lhs: &BitVec<SExpr>, rhs: &BitVec<SExpr>) -> BitVec<SExpr> {
        let result = self.false_bv();
        self.bin_bool_op(lhs, rhs, Context::bvslt, result)
    }

    fn sgt(&mut self, lhs: &BitVec<SExpr>, rhs: &BitVec<SExpr>) -> BitVec<SExpr> {
        let result = self.false_bv();
        self.bin_bool_op(lhs, rhs, Context::bvsgt, result)
    }

    fn sle(&mut self, lhs: &BitVec<SExpr>, rhs: &BitVec<SExpr>) -> BitVec<SExpr> {
        let result = self.true_bv();
        self.bin_bool_op(lhs, rhs, Context::bvsle, result)
    }

    fn sge(&mut self, lhs: &BitVec<SExpr>, rhs: &BitVec<SExpr>) -> BitVec<SExpr> {
        let result = self.true_bv();
        self.bin_bool_op(lhs, rhs, Context::bvsge, result)
    }

    fn ult(&mut self, lhs: &BitVec<SExpr>, rhs: &BitVec<SExpr>) -> BitVec<SExpr> {
        let result = self.false_bv();
        self.bin_bool_op(lhs, rhs, Context::bvult, result)
    }

    fn ugt(&mut self, lhs: &BitVec<SExpr>, rhs: &BitVec<SExpr>) -> BitVec<SExpr> {
        let result = self.false_bv();
        self.bin_bool_op(lhs, rhs, Context::bvugt, result)
    }

    fn ule(&mut self, lhs: &BitVec<SExpr>, rhs: &BitVec<SExpr>) -> BitVec<SExpr> {
        let result = self.true_bv();
        self.bin_bool_op(lhs, rhs, Context::bvule, result)
    }

    fn uge(&mut self, lhs: &BitVec<SExpr>, rhs: &BitVec<SExpr>) -> BitVec<SExpr> {
        let result = self.true_bv();
        self.bin_bool_op(lhs, rhs, Context::bvuge, result)
    }

    fn push(&mut self) -> io::Result<()> {
        let shared = Arc::clone(&self.context);
        let mut context = shared.lock().unwrap();
        (self.solver_fn.push_fn)(&mut context)
    }

    fn pop(&mut self) -> io::Result<()> {
        let shared = Arc::clone(&self.context);
        let mut context = shared.lock().unwrap();
        (self.solver_fn.pop_fn)(&mut context)
    }

    fn check(&mut self) -> io::Result<super::solver_interface::Response> {
        let shared = Arc::clone(&self.context);
        let mut context = shared.lock().unwrap();
        match (self.solver_fn.check_fn)(&mut context) {
            Ok(easy_smt::Response::Sat) => Ok(super::solver_interface::Response::Sat),
            Ok(easy_smt::Response::Unsat) => Ok(super::solver_interface::Response::Unsat),
            Ok(easy_smt::Response::Unknown) => Ok(super::solver_interface::Response::Unknown),
            Err(e) => Err(io::Error::new(io::ErrorKind::Other, e)),
        }
    }

    fn assert(&mut self, bit_vec: &BitVec<SExpr>) -> io::Result<()> {
        let shared = Arc::clone(&self.context);
        let mut context = shared.lock().unwrap();
        let bool_expr = self.bv_to_bool(&context, bit_vec);
        (self.solver_fn.assert_fn)(&mut context, bool_expr)
    }

    fn render(&mut self, bit_vec: &BitVec<SExpr>) -> String {
        let shared = Arc::clone(&self.context);
        let context = shared.lock().unwrap();

        // Handle zero-width values first.
        let rep = match bit_vec.get_rep() {
            None => return "<zero-width>".to_string(),
            Some(r) => r.clone(),
        };

        // We can only build the pretty `let` representation if the final
        // expression is an atom that refers to an entry in our term cache.
        let final_name_opt = context.get_atom(rep);
        let final_name = match final_name_opt {
            Some(name) if self.term_cache.contains_key(name) => name.to_string(),
            // Fallback: display the raw S-expr as before.
            _ => return context.display(rep).to_string(),
        };

        // ------------------------------------------------------------------
        // Collect all term-cache names that the final expression transitively
        // depends on.  We perform a DFS and record the names in post-order so
        // that we obtain a topological ordering (dependencies before uses).
        // ------------------------------------------------------------------

        // Small helper to recursively gather atom names referenced in an SExpr.
        fn gather_names(
            expr: SExpr,
            ctx: &Context,
            term_cache: &HashMap<String, SExpr>,
            out: &mut Vec<String>,
        ) {
            if expr.is_atom() {
                if let Some(atom) = ctx.get_atom(expr) {
                    if term_cache.contains_key(atom) {
                        out.push(atom.to_string());
                    }
                }
            } else if expr.is_list() {
                if let Some(children) = ctx.get_list(expr) {
                    for &child in children {
                        gather_names(child, ctx, term_cache, out);
                    }
                }
            }
        }

        // Recursive DFS for topological sort.
        fn dfs(
            name: &str,
            ctx: &Context,
            term_cache: &HashMap<String, SExpr>,
            visited: &mut HashSet<String>,
            order: &mut Vec<String>,
        ) {
            if !visited.insert(name.to_string()) {
                return; // already processed
            }

            let expr = match term_cache.get(name) {
                Some(e) => *e,
                None => return,
            };

            let mut deps = Vec::new();
            gather_names(expr, ctx, term_cache, &mut deps);
            for dep in deps {
                dfs(&dep, ctx, term_cache, visited, order);
            }

            order.push(name.to_string());
        }

        let mut visited = HashSet::new();
        let mut order = Vec::new();
        dfs(
            &final_name,
            &context,
            &self.term_cache,
            &mut visited,
            &mut order,
        );

        // Ensure the root term itself is included in the bindings (it will
        // also be referenced as the final expression, which is harmless and
        // matches typical SMT-LIB pretty-printing patterns).

        // ------------------------------------------------------------------
        // Build the `(let …)` s-expression string.
        // ------------------------------------------------------------------
        let mut result = String::from("(let\n");

        for name in &order {
            if let Some(expr) = self.term_cache.get(name) {
                let expr_str = context.display(*expr).to_string();
                result.push_str("  ((");
                result.push_str(name);
                result.push(' ');
                result.push_str(&expr_str);
                result.push_str("))\n");
            }
        }

        // Final expression (the name of the root term)
        result.push_str("  ");
        result.push_str(&final_name);
        result.push_str("\n)");

        result
    }
}

#[cfg(feature = "with-bitwuzla-binary-test")]
test_solver!(
    bitwuzla_tests,
    super::EasySMTSolver::new(&super::EasySMTConfig::bitwuzla()).unwrap()
);

#[cfg(feature = "with-boolector-binary-test")]
test_solver!(
    boolector_tests,
    super::EasySMTSolver::new(&super::EasySMTConfig::boolector()).unwrap()
);

#[cfg(feature = "with-z3-binary-test")]
test_solver!(
    z3_tests,
    super::EasySMTSolver::new(&super::EasySMTConfig::z3()).unwrap()
);
