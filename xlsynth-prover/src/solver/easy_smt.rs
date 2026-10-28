// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "has-easy-smt")]

use std::{
    collections::{HashMap, HashSet},
    fmt::Write as _,
    io,
    path::PathBuf,
    sync::{Arc, Mutex, MutexGuard},
};

use easy_smt::{Context, ContextBuilder, SExpr};

use super::{BitVec, Response, Solver, SolverConfig, Uf};
use xlsynth_pir::{ir, ir_value_utils::ir_value_from_lsb0_bits_with_layout};

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

/// Configuration for executable solver backends using the `easy-smt` crate.
///
/// When using `easy-smt`, a persistent solver process is launched, and
/// communication occurs via stdin/stdout. This introduces serialization
/// overhead, so if an in-memory backend is available for a specific solver,
/// it is generally preferable to use that instead.
///
/// The main advantage of using an executable solver backend is that it requires
/// no additional FFI code. Any SMT-compatible solver can be used out of the
/// box.
#[derive(Clone)]
pub struct EasySmtConfig {
    pub solver_path: PathBuf,
    pub solver_args: Vec<String>,
    pub replay_file: Option<PathBuf>,
    pub solver_fn: SolverFn,
}

impl EasySmtConfig {
    pub fn bitwuzla() -> Self {
        Self {
            solver_path: PathBuf::from("bitwuzla"),
            solver_args: ["--produce-models"].iter().map(|s| s.to_string()).collect(),
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
                "--smt2",               // Use SMT2
                "-m",                   // Produce models
                "--output-format=smt2", // Output in SMT2 format
                "--no-exit-codes",      // Don't exit with error code on failure
                "--incremental",        // Use incremental solving
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
            solver_args: [
                "-nw",   // No warnings
                "-smt2", // Use SMT2
                "-in",   // Read from stdin
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            replay_file: None,
            solver_fn: SolverFn::default(),
        }
    }
}

pub struct EasySmtSolver {
    context: Arc<Mutex<Context>>,
    solver_fn: SolverFn,
    next_name_index: usize,
    term_cache: HashMap<String, SExpr>,
    reverse_cache: HashMap<SExpr, String>,
    fresh_counter: u64,
    // Term constructors are infallible in the Solver API. Preserve the first
    // rejected definition so no later command can use an undefined term.
    definition_error: Option<io::Error>,
}

impl SolverConfig for EasySmtConfig {
    type Solver = EasySmtSolver;
}

/// Encodes a named declaration as one SMT atom, disjoint from cached terms.
/// SMT-LIB punctuation in a caller's name must never reach the command stream.
fn declaration_symbol(prefix: &str, name: &str) -> String {
    let mut symbol = String::from(prefix);
    for byte in name.bytes() {
        write!(symbol, "{byte:02x}").expect("writing to String cannot fail");
    }
    symbol
}

impl EasySmtSolver {
    fn get_next_available_name(&mut self) -> String {
        let mut name = format!("t{}", self.next_name_index);
        while self.term_cache.contains_key(&name) {
            self.next_name_index += 1;
            name = format!("t{}", self.next_name_index);
        }
        name
    }

    fn check_definitions(&self) -> io::Result<()> {
        match &self.definition_error {
            Some(error) => Err(io::Error::new(error.kind(), error.to_string())),
            None => Ok(()),
        }
    }

    /// Names a compound term once and records any error defining it.
    fn cache_term(
        &mut self,
        bit_vec: BitVec<SExpr>,
        context: &mut MutexGuard<Context>,
    ) -> BitVec<SExpr> {
        let BitVec::BitVec { width, rep } = bit_vec else {
            return bit_vec;
        };
        if rep.is_atom() {
            return bit_vec;
        }
        if let Some(name) = self.reverse_cache.get(&rep) {
            return BitVec::BitVec {
                width,
                rep: context.atom(name),
            };
        }
        let name = self.get_next_available_name();
        let symbol = context.atom(&name);
        if self.definition_error.is_none() {
            let sort = context.bit_vec_sort(context.numeral(width));
            self.definition_error = context.define_const(&name, sort, rep).err();
        }
        self.term_cache.insert(name.clone(), rep);
        self.reverse_cache.insert(rep, name.clone());
        self.reverse_cache.insert(symbol, name);
        BitVec::BitVec { width, rep: symbol }
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

impl Solver for EasySmtSolver {
    type Term = SExpr;
    type Config = EasySmtConfig;
    fn new(config: &EasySmtConfig) -> io::Result<EasySmtSolver> {
        let mut builder = ContextBuilder::new();
        if let Some(ref replay_file) = config.replay_file {
            builder.replay_file(Some(std::fs::File::create(replay_file)?));
        }
        builder.solver(&config.solver_path);
        builder.solver_args(&config.solver_args);
        let context = builder.build()?;
        // Trick for boolector solver.
        // Boolector would render 1-bit bit-vector results as `true` or `false`
        // instead of `#b1` or `#b0`.
        //
        // We need to ensure that easy-smt already knows about `true` and
        // `false` so that it can parse them correctly.
        context.atom("true");
        context.atom("false");
        Ok(EasySmtSolver {
            context: Arc::new(Mutex::new(context)),
            solver_fn: config.solver_fn.clone(),
            next_name_index: 0,
            term_cache: HashMap::new(),
            reverse_cache: HashMap::new(),
            fresh_counter: 0,
            definition_error: None,
        })
    }

    fn fresh_symbol(&mut self, _name: &str) -> io::Result<String> {
        let symbol = format!("fresh_{}", self.fresh_counter);
        self.fresh_counter += 1;
        Ok(symbol)
    }

    fn declare(&mut self, name: &str, width: usize) -> io::Result<BitVec<SExpr>> {
        self.check_definitions()?;
        if width == 0 {
            return Ok(BitVec::ZeroWidth);
        }
        let shared = Arc::clone(&self.context);
        let mut context = shared.lock().unwrap();
        let width_numeral = context.numeral(width as i64);
        let sort = context.bit_vec_sort(width_numeral);
        let rep = context.declare_const(declaration_symbol("v_", name), sort)?;
        Ok(self.cache_term(BitVec::BitVec { width, rep }, &mut context))
    }

    fn declare_uf(
        &mut self,
        name: &str,
        arg_widths: &[usize],
        result_width: usize,
    ) -> io::Result<Uf<SExpr>> {
        self.check_definitions()?;
        if result_width == 0 {
            return Ok(Uf::UfZeroWidth {
                arg_widths: arg_widths.to_vec(),
            });
        } else {
            let shared = Arc::clone(&self.context);
            let mut context = shared.lock().unwrap();
            let width_numerals: Vec<SExpr> = arg_widths
                .iter()
                .copied()
                .filter(|w| *w > 0)
                .map(|w| context.numeral(w as i64))
                .collect();
            let arg_sorts = width_numerals
                .iter()
                .map(|w| context.bit_vec_sort(w.clone()))
                .collect();
            let result_sort = context.bit_vec_sort(context.numeral(result_width as i64));
            let rep =
                context.declare_fun(declaration_symbol("uf_", name), arg_sorts, result_sort)?;
            Ok(Uf::Uf {
                rep,
                arg_widths: arg_widths.to_vec(),
                result_width,
            })
        }
    }

    fn apply_uf(&mut self, uf: &Uf<SExpr>, args: &[&BitVec<SExpr>]) -> BitVec<SExpr> {
        uf.check_arg_widths(&args);
        match uf {
            Uf::UfZeroWidth { .. } => BitVec::ZeroWidth,
            Uf::Uf {
                rep,
                arg_widths,
                result_width,
            } => {
                // Build the actual argument list containing only non-zero-width
                // bitvectors, since the EasySMT UF was declared
                // skipping zero-width arguments.
                let shared = Arc::clone(&self.context);
                let mut context = shared.lock().unwrap();

                let mut sexprs: Vec<SExpr> = Vec::new();
                for (arg_bv, w) in args.iter().zip(arg_widths.iter()) {
                    if *w == 0 {
                        continue;
                    }
                    match arg_bv {
                        BitVec::BitVec { rep, .. } => sexprs.push(rep.clone()),
                        BitVec::ZeroWidth => {}
                    }
                }

                let app = if sexprs.is_empty() {
                    rep.clone()
                } else {
                    let mut items = Vec::with_capacity(1 + sexprs.len());
                    items.push(rep.clone());
                    items.extend(sexprs);
                    context.list(items)
                };

                self.cache_term(
                    BitVec::BitVec {
                        width: *result_width,
                        rep: app,
                    },
                    &mut context,
                )
            }
        }
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
    ) -> io::Result<xlsynth_pir::IrValue> {
        self.check_definitions()?;
        let shared = Arc::clone(&self.context);
        let mut context = shared.lock().unwrap();
        match bit_vec {
            BitVec::BitVec { rep, width } => {
                // easy-smt's get_value assumes every list it receives is a
                // list of pairs. Solvers can instead return (error "...").
                let query = context.list(vec![context.atoms().get_value, context.list(vec![*rep])]);
                context.raw_send(query)?;
                let reply = context.raw_recv()?;
                let invalid = || {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "Invalid get-value response for {}: {}",
                            context.display(*rep),
                            context.display(reply)
                        ),
                    )
                };
                let Some([binding]) = context.get_list(reply) else {
                    return Err(invalid());
                };
                let Some([term, value]) = context.get_list(*binding) else {
                    return Err(invalid());
                };
                if term != rep {
                    return Err(invalid());
                }
                let atom = context.get_atom(*value).ok_or_else(invalid)?;
                let mut bits = Vec::with_capacity(*width);
                if let Some(binary) = atom.strip_prefix("#b") {
                    for digit in binary.bytes().rev() {
                        bits.push(match digit {
                            b'0' => false,
                            b'1' => true,
                            _ => return Err(invalid()),
                        });
                    }
                } else if let Some(hex) = atom.strip_prefix("#x") {
                    for digit in hex.chars().rev() {
                        let nibble = digit.to_digit(16).ok_or_else(invalid)?;
                        bits.extend((0..4).map(|bit| nibble & (1 << bit) != 0));
                    }
                } else if atom == "true" || atom == "false" {
                    // Boolector prints one-bit values as booleans.
                    bits.push(atom == "true");
                } else {
                    return Err(invalid());
                }
                if bits.len() != *width {
                    return Err(invalid());
                }
                ir_value_from_lsb0_bits_with_layout(ty, &bits)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
            }
            BitVec::ZeroWidth => ir_value_from_lsb0_bits_with_layout(ty, &[])
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e)),
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
        self.check_definitions()?;
        let shared = Arc::clone(&self.context);
        let mut context = shared.lock().unwrap();
        (self.solver_fn.push_fn)(&mut context)
    }

    fn pop(&mut self) -> io::Result<()> {
        self.check_definitions()?;
        let shared = Arc::clone(&self.context);
        let mut context = shared.lock().unwrap();
        (self.solver_fn.pop_fn)(&mut context)
    }

    fn check(&mut self) -> io::Result<Response> {
        self.check_definitions()?;
        let shared = Arc::clone(&self.context);
        let mut context = shared.lock().unwrap();
        match (self.solver_fn.check_fn)(&mut context) {
            Ok(easy_smt::Response::Sat) => Ok(Response::Sat),
            Ok(easy_smt::Response::Unsat) => Ok(Response::Unsat),
            Ok(easy_smt::Response::Unknown) => Ok(Response::Unknown),
            Err(e) => Err(io::Error::new(io::ErrorKind::Other, e)),
        }
    }

    fn assert(&mut self, bit_vec: &BitVec<SExpr>) -> io::Result<()> {
        self.check_definitions()?;
        let shared = Arc::clone(&self.context);
        let mut context = shared.lock().unwrap();
        let bool_expr = self.bv_to_bool(&context, bit_vec);
        (self.solver_fn.assert_fn)(&mut context, bool_expr)
    }

    fn render(&mut self, bit_vec: &BitVec<SExpr>) -> String {
        let shared = Arc::clone(&self.context);
        let context = shared.lock().unwrap();

        // Handle zero-width values first.
        let rep = match bit_vec.get_term() {
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
        // depends on using an *iterative* depth-first search.  This avoids the
        // risk of overflowing the call-stack when rendering extremely deep
        // expression graphs.
        // ------------------------------------------------------------------

        // Small helper to recursively gather atom names referenced in an SExpr
        // (the depth of a single S-expression is typically shallow, so the
        // recursion here is not expected to cause stack issues).
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

        // ------------------------------------------------------------------
        // Iterative DFS (post-order)
        // ------------------------------------------------------------------

        let mut visited: HashSet<String> = HashSet::new();
        let mut order: Vec<String> = Vec::new();

        // Stack holds (name, processed_children_flag).  The second boolean
        // indicates whether all dependencies of `name` have already been
        // pushed to the stack (true ⇒ time to emit `name`).
        let mut stack: Vec<(String, bool)> = Vec::new();
        stack.push((final_name.clone(), false));

        while let Some((name, processed)) = stack.pop() {
            if processed {
                // All dependencies processed – record in post-order list.
                order.push(name);
                continue;
            }

            if !visited.contains(&name) {
                visited.insert(name.clone());

                // First encounter: push a marker so that we revisit the node
                // *after* its children have been handled.
                stack.push((name.clone(), true));

                // Push all dependencies onto the stack (if any).
                if let Some(&expr) = self.term_cache.get(&name) {
                    let mut deps = Vec::new();
                    gather_names(expr, &context, &self.term_cache, &mut deps);
                    for dep in deps {
                        if !visited.contains(&dep) {
                            stack.push((dep, false));
                        }
                    }
                }
            }
        }
        // Ensure the root term itself is included in the bindings (it will
        // also be referenced as the final expression, which is harmless and
        // matches typical SMT-LIB pretty-printing patterns).

        // ------------------------------------------------------------------
        // Build the `(let …)` s-expression string.
        // ------------------------------------------------------------------
        let mut result = String::from("(let\n");

        let mut first = true;
        for name in &order {
            if let Some(expr) = self.term_cache.get(name) {
                let expr_str = context.display(*expr).to_string();
                if first {
                    first = false;
                    result.push_str("  ((");
                } else {
                    result.push_str("   (");
                }
                result.push_str(name);
                result.push(' ');
                result.push_str(&expr_str);
                result.push_str(")\n");
            }
        }
        result.push_str("  )\n");

        // Final expression (the name of the root term)
        result.push_str("  ");
        result.push_str(&final_name);
        result.push_str("\n)");

        result
    }
}

#[cfg(test)]
use crate::test_solver;

#[cfg(test)]
#[cfg(feature = "with-bitwuzla-binary-test")]
test_solver!(
    bitwuzla_tests,
    super::EasySmtSolver::new(&super::EasySmtConfig::bitwuzla()).unwrap()
);

#[cfg(test)]
#[cfg(feature = "with-boolector-binary-test")]
test_solver!(
    boolector_tests,
    super::EasySmtSolver::new(&super::EasySmtConfig::boolector()).unwrap()
);

#[cfg(test)]
#[cfg(feature = "with-z3-binary-test")]
test_solver!(
    z3_tests,
    super::EasySmtSolver::new(&super::EasySmtConfig::z3()).unwrap()
);
