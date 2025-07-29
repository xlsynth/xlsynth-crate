// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "has-boolector")]

use std::{
    collections::HashMap,
    ffi::{CStr, CString},
    io,
    sync::Arc,
};

use boolector_sys::{
    BTOR_OPT_INCREMENTAL, BTOR_OPT_MODEL_GEN, BoolectorNode, Btor, BtorOption, boolector_add,
    boolector_and, boolector_assert, boolector_bitvec_sort, boolector_bv_assignment,
    boolector_concat, boolector_cond, boolector_const, boolector_consth, boolector_delete,
    boolector_eq, boolector_free_bv_assignment, boolector_get_refs, boolector_mul, boolector_nand,
    boolector_ne, boolector_neg, boolector_new, boolector_nor, boolector_not, boolector_one,
    boolector_or, boolector_pop, boolector_push, boolector_release_all, boolector_sat,
    boolector_sdiv, boolector_set_opt, boolector_sext, boolector_sgt, boolector_sgte,
    boolector_slice, boolector_sll, boolector_slt, boolector_slte, boolector_sra, boolector_srem,
    boolector_srl, boolector_sub, boolector_udiv, boolector_uext, boolector_ugt, boolector_ugte,
    boolector_ult, boolector_ulte, boolector_unsigned_int, boolector_urem, boolector_var,
    boolector_xor, boolector_zero,
};

use crate::{
    equiv::solver_interface::{BitVec, Response, Solver, angelic_div_mod_common},
    ir_value_utils::{ir_bits_from_lsb_is_0, ir_value_from_bits_with_type},
    xls_ir::ir,
};

// Low-level wrapper for the Boolector solver context.
struct RawBtor {
    raw: *mut Btor,
}

impl RawBtor {
    pub fn new() -> Self {
        let raw = unsafe { boolector_new() };
        RawBtor { raw }
    }
}

impl Drop for RawBtor {
    fn drop(&mut self) {
        unsafe {
            // Boolector requires us to call `boolector_release_all` before
            // calling `boolector_delete`.
            boolector_release_all(self.raw);
            assert_eq!(boolector_get_refs(self.raw) as i32, 0);
            boolector_delete(self.raw)
        };
    }
}

/// Boolector solver implementing the `Solver` trait via `boolector-sys`.
#[derive(Clone)]
pub struct Boolector {
    btor: Arc<RawBtor>,
    fresh_counter: u64,
}

impl Boolector {
    fn raw_btor(&self) -> *mut Btor {
        self.btor.raw
    }

    fn bool_to_bv(&mut self, cond: BoolectorTerm) -> BitVec<BoolectorTerm> {
        let sort1 = unsafe { boolector_bitvec_sort(self.raw_btor(), 1) };
        let one = unsafe { boolector_one(self.raw_btor(), sort1) };
        let zero = unsafe { boolector_zero(self.raw_btor(), sort1) };
        let rep = unsafe { boolector_cond(self.raw_btor(), cond.raw, one, zero) };
        BitVec::BitVec {
            width: 1,
            rep: BoolectorTerm { raw: rep },
        }
    }

    fn bv_to_bool(&mut self, bv: &BitVec<BoolectorTerm>) -> BoolectorTerm {
        match bv {
            BitVec::BitVec { width: 1, rep } => {
                let sort1 = unsafe { boolector_bitvec_sort(self.raw_btor(), 1) };
                let one = unsafe { boolector_one(self.raw_btor(), sort1) };
                let cond = unsafe { boolector_eq(self.raw_btor(), rep.raw, one) };
                BoolectorTerm { raw: cond }
            }
            _ => panic!("Invalid bitvector width for boolean: {:?}", bv.get_width()),
        }
    }

    fn unary_op<F>(&mut self, bv: &BitVec<BoolectorTerm>, op: F) -> BitVec<BoolectorTerm>
    where
        F: Fn(*mut Btor, *mut BoolectorNode) -> *mut BoolectorNode,
    {
        match bv {
            BitVec::ZeroWidth => BitVec::ZeroWidth,
            BitVec::BitVec { width, rep } => BitVec::BitVec {
                width: *width,
                rep: BoolectorTerm {
                    raw: op(self.raw_btor(), rep.raw),
                },
            },
        }
    }

    fn bin_op<F>(
        &mut self,
        lhs: &BitVec<BoolectorTerm>,
        rhs: &BitVec<BoolectorTerm>,
        op: F,
    ) -> BitVec<BoolectorTerm>
    where
        F: Fn(*mut Btor, *mut BoolectorNode, *mut BoolectorNode) -> *mut BoolectorNode,
    {
        match (lhs, rhs) {
            (BitVec::BitVec { width: w1, rep: r1 }, BitVec::BitVec { width: w2, rep: r2 }) => {
                assert_eq!(w1, w2, "Bitvector width mismatch");
                BitVec::BitVec {
                    width: *w1,
                    rep: BoolectorTerm {
                        raw: op(self.raw_btor(), r1.raw, r2.raw),
                    },
                }
            }
            (BitVec::ZeroWidth, BitVec::ZeroWidth) => BitVec::ZeroWidth,
            _ => panic!("Bitvector width mismatch"),
        }
    }

    fn bin_bool_op<F>(
        &mut self,
        lhs: &BitVec<BoolectorTerm>,
        rhs: &BitVec<BoolectorTerm>,
        op: F,
        zero_width_result: BitVec<BoolectorTerm>,
    ) -> BitVec<BoolectorTerm>
    where
        F: Fn(*mut Btor, *mut BoolectorNode, *mut BoolectorNode) -> *mut BoolectorNode,
    {
        match (lhs, rhs) {
            (BitVec::BitVec { width: w1, rep: r1 }, BitVec::BitVec { width: w2, rep: r2 }) => {
                assert_eq!(w1, w2, "Bitvector width mismatch");
                self.bool_to_bv(BoolectorTerm {
                    raw: op(self.raw_btor(), r1.raw, r2.raw),
                })
            }
            (BitVec::ZeroWidth, BitVec::ZeroWidth) => zero_width_result,
            _ => panic!("Bitvector width mismatch"),
        }
    }

    fn reduce_op<F>(&mut self, bv: &BitVec<BoolectorTerm>, op: F) -> BitVec<BoolectorTerm>
    where
        F: Fn(*mut Btor, *mut BoolectorNode, *mut BoolectorNode) -> *mut BoolectorNode,
    {
        match bv {
            BitVec::BitVec { width, rep } => {
                let mut acc = unsafe { boolector_slice(self.raw_btor(), rep.raw, 0, 0) };
                for i in 1..*width {
                    let bit =
                        unsafe { boolector_slice(self.raw_btor(), rep.raw, i as u32, i as u32) };
                    acc = op(self.raw_btor(), acc, bit);
                }
                BitVec::BitVec {
                    width: 1,
                    rep: BoolectorTerm { raw: acc },
                }
            }
            BitVec::ZeroWidth => panic!("Cannot reduce zero-width bitvector"),
        }
    }
}

#[derive(Clone)]
pub struct BoolectorTerm {
    raw: *mut BoolectorNode,
}

/// Configuration for creating a Boolector solver.
pub struct BoolectorConfig {
    options: HashMap<BtorOption, u32>,
}

unsafe impl Send for BoolectorConfig {}
unsafe impl Sync for BoolectorConfig {}

impl BoolectorConfig {
    pub fn new() -> Self {
        BoolectorConfig {
            options: HashMap::from_iter([(BTOR_OPT_MODEL_GEN, 2), (BTOR_OPT_INCREMENTAL, 1)]),
        }
    }

    pub fn set_option(&mut self, option: BtorOption, value: u32) -> &mut Self {
        self.options.insert(option, value);
        self
    }
}

impl Solver for Boolector {
    type Term = BoolectorTerm;
    type Config = BoolectorConfig;

    fn new(config: &Self::Config) -> io::Result<Self> {
        let btor = RawBtor::new();
        for (opt, val) in config.options.iter() {
            unsafe { boolector_set_opt(btor.raw, *opt, *val) };
        }
        Ok(Boolector {
            btor: Arc::new(btor),
            fresh_counter: 0,
        })
    }

    fn declare(&mut self, name: &str, width: usize) -> io::Result<BitVec<Self::Term>> {
        if width == 0 {
            return Ok(BitVec::ZeroWidth);
        }
        let sort = unsafe { boolector_bitvec_sort(self.raw_btor(), width as u32) };
        let c_name = CString::new(name).unwrap();
        let n = unsafe { boolector_var(self.raw_btor(), sort, c_name.as_ptr()) };
        Ok(BitVec::BitVec {
            width,
            rep: BoolectorTerm { raw: n },
        })
    }

    fn fresh_symbol(&mut self, name: &str) -> io::Result<String> {
        let symbol = format!("__boolector_fresh_{}_{}", name, self.fresh_counter);
        self.fresh_counter += 1;
        Ok(symbol)
    }

    fn numerical(&mut self, width: usize, mut value: u64) -> BitVec<Self::Term> {
        assert!(width > 0, "Width must be positive");
        if width < 64 {
            let mask = (1u64 << width) - 1;
            value &= mask;
        }
        let sort = unsafe { boolector_bitvec_sort(self.raw_btor(), width as u32) };
        let rep = if width <= 32 {
            unsafe { boolector_unsigned_int(self.raw_btor(), value as u32, sort) }
        } else {
            let bitstr = format!("{:0width$b}", value, width = width);
            let c = CString::new(bitstr).unwrap();
            unsafe { boolector_const(self.raw_btor(), c.as_ptr()) }
        };
        BitVec::BitVec {
            width,
            rep: BoolectorTerm { raw: rep },
        }
    }

    fn from_raw_str(&mut self, width: usize, value: &str) -> BitVec<Self::Term> {
        assert!(width > 0, "Width must be positive");
        let sort = unsafe { boolector_bitvec_sort(self.raw_btor(), width as u32) };
        let rep = if let Some(stripped) = value.strip_prefix("#b") {
            let c = CString::new(stripped).unwrap();
            unsafe { boolector_const(self.raw_btor(), c.as_ptr()) }
        } else if let Some(stripped) = value.strip_prefix("#x") {
            let c = CString::new(stripped).unwrap();
            unsafe { boolector_consth(self.raw_btor(), sort, c.as_ptr()) }
        } else {
            panic!("Invalid atom: {}", value);
        };
        BitVec::BitVec {
            width,
            rep: BoolectorTerm { raw: rep },
        }
    }

    fn get_value(
        &mut self,
        bit_vec: &BitVec<Self::Term>,
        ty: &ir::Type,
    ) -> io::Result<xlsynth::IrValue> {
        match bit_vec {
            BitVec::BitVec { rep, .. } => unsafe {
                let s = boolector_bv_assignment(self.raw_btor(), rep.raw);
                let bitstr = CStr::from_ptr(s).to_str().unwrap().to_string();
                boolector_free_bv_assignment(self.raw_btor(), s);
                let bits: Vec<bool> = bitstr.chars().rev().map(|c| c == '1').collect();
                Ok(ir_value_from_bits_with_type(
                    &ir_bits_from_lsb_is_0(&bits),
                    ty,
                ))
            },
            BitVec::ZeroWidth => {
                if matches!(ty, ir::Type::Token) {
                    Ok(xlsynth::IrValue::make_token())
                } else {
                    panic!("Cannot get value of zero-width bitvector for type {:?}", ty)
                }
            }
        }
    }

    fn extract(&mut self, bit_vec: &BitVec<Self::Term>, high: i32, low: i32) -> BitVec<Self::Term> {
        if high < low {
            return BitVec::ZeroWidth;
        }
        match bit_vec {
            BitVec::ZeroWidth => BitVec::ZeroWidth,
            BitVec::BitVec { width, rep } => {
                assert!(high < *width as i32, "High index out of bounds");
                assert!(low >= 0, "Low index out of bounds");
                let rep2 =
                    unsafe { boolector_slice(self.raw_btor(), rep.raw, high as u32, low as u32) };
                BitVec::BitVec {
                    width: (high - low + 1) as usize,
                    rep: BoolectorTerm { raw: rep2 },
                }
            }
        }
    }

    fn not(&mut self, bv: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.unary_op(bv, |b, n| unsafe { boolector_not(b, n) })
    }

    fn neg(&mut self, bv: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.unary_op(bv, |b, n| unsafe { boolector_neg(b, n) })
    }

    fn reverse(&mut self, bv: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        match bv {
            BitVec::ZeroWidth => BitVec::ZeroWidth,
            BitVec::BitVec { width, rep } => {
                let mut acc = unsafe { boolector_slice(self.raw_btor(), rep.raw, 0, 0) };
                for i in 1..*width {
                    let bit =
                        unsafe { boolector_slice(self.raw_btor(), rep.raw, i as u32, i as u32) };
                    acc = unsafe { boolector_concat(self.raw_btor(), acc, bit) };
                }
                BitVec::BitVec {
                    width: *width,
                    rep: BoolectorTerm { raw: acc },
                }
            }
        }
    }

    fn or_reduce(&mut self, bv: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.reduce_op(bv, |b, x, y| unsafe { boolector_or(b, x, y) })
    }

    fn and_reduce(&mut self, bv: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.reduce_op(bv, |b, x, y| unsafe { boolector_and(b, x, y) })
    }

    fn xor_reduce(&mut self, bv: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.reduce_op(bv, |b, x, y| unsafe { boolector_xor(b, x, y) })
    }

    fn add(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.bin_op(x, y, |b, x, y| unsafe { boolector_add(b, x, y) })
    }

    fn sub(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.bin_op(x, y, |b, x, y| unsafe { boolector_sub(b, x, y) })
    }

    fn mul(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.bin_op(x, y, |b, x, y| unsafe { boolector_mul(b, x, y) })
    }

    fn udiv(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.bin_op(x, y, |b, x, y| unsafe { boolector_udiv(b, x, y) })
    }

    fn urem(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.bin_op(x, y, |b, x, y| unsafe { boolector_urem(b, x, y) })
    }

    fn angelic_udiv(
        &mut self,
        x: &BitVec<Self::Term>,
        y: &BitVec<Self::Term>,
    ) -> BitVec<Self::Term> {
        angelic_div_mod_common(self, x, y, false).0
    }

    fn angelic_urem(
        &mut self,
        x: &BitVec<Self::Term>,
        y: &BitVec<Self::Term>,
    ) -> BitVec<Self::Term> {
        angelic_div_mod_common(self, x, y, false).1
    }

    fn angelic_sdiv(
        &mut self,
        x: &BitVec<Self::Term>,
        y: &BitVec<Self::Term>,
    ) -> BitVec<Self::Term> {
        angelic_div_mod_common(self, x, y, true).0
    }

    fn angelic_srem(
        &mut self,
        x: &BitVec<Self::Term>,
        y: &BitVec<Self::Term>,
    ) -> BitVec<Self::Term> {
        angelic_div_mod_common(self, x, y, true).1
    }

    fn srem(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.bin_op(x, y, |b, x, y| unsafe { boolector_srem(b, x, y) })
    }

    fn sdiv(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.bin_op(x, y, |b, x, y| unsafe { boolector_sdiv(b, x, y) })
    }

    fn shl(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.bin_op(x, y, |b, x, y| unsafe { boolector_sll(b, x, y) })
    }

    fn lshr(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.bin_op(x, y, |b, x, y| unsafe { boolector_srl(b, x, y) })
    }

    fn ashr(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.bin_op(x, y, |b, x, y| unsafe { boolector_sra(b, x, y) })
    }

    fn concat(&mut self, a: &BitVec<Self::Term>, b: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        match (a, b) {
            (BitVec::BitVec { width: w1, rep: r1 }, BitVec::BitVec { width: w2, rep: r2 }) => {
                let rep2 = unsafe { boolector_concat(self.raw_btor(), r1.raw, r2.raw) };
                BitVec::BitVec {
                    width: w1 + w2,
                    rep: BoolectorTerm { raw: rep2 },
                }
            }
            (BitVec::ZeroWidth, other) => other.clone(),
            (other, BitVec::ZeroWidth) => other.clone(),
        }
    }

    fn or(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.bin_op(x, y, |b, x, y| unsafe { boolector_or(b, x, y) })
    }

    fn and(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.bin_op(x, y, |b, x, y| unsafe { boolector_and(b, x, y) })
    }

    fn xor(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.bin_op(x, y, |b, x, y| unsafe { boolector_xor(b, x, y) })
    }

    fn nor(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.bin_op(x, y, |b, x, y| unsafe { boolector_nor(b, x, y) })
    }

    fn nand(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.bin_op(x, y, |b, x, y| unsafe { boolector_nand(b, x, y) })
    }

    fn extend(&mut self, bv: &BitVec<Self::Term>, ext: usize, signed: bool) -> BitVec<Self::Term> {
        match bv {
            BitVec::ZeroWidth => panic!("Cannot extend zero-width bitvector"),
            BitVec::BitVec { width, rep } => {
                if ext == 0 {
                    return BitVec::BitVec {
                        width: *width,
                        rep: BoolectorTerm { raw: rep.raw },
                    };
                }
                let rep2 = if signed {
                    unsafe { boolector_sext(self.raw_btor(), rep.raw, ext as u32) }
                } else {
                    unsafe { boolector_uext(self.raw_btor(), rep.raw, ext as u32) }
                };
                BitVec::BitVec {
                    width: *width + ext,
                    rep: BoolectorTerm { raw: rep2 },
                }
            }
        }
    }

    fn ite(
        &mut self,
        c: &BitVec<Self::Term>,
        t: &BitVec<Self::Term>,
        e: &BitVec<Self::Term>,
    ) -> BitVec<Self::Term> {
        match (c.clone(), t, e) {
            (
                BitVec::BitVec { width: wc, rep: rc },
                BitVec::BitVec { width: wt, rep: rt },
                BitVec::BitVec { width: we, rep: re },
            ) => {
                assert_eq!(wc, 1, "Condition must be 1-bit");
                assert_eq!(wt, we, "Then and else must have same width");
                let cond = self.bv_to_bool(&BitVec::BitVec {
                    width: 1,
                    rep: BoolectorTerm { raw: rc.raw },
                });
                let rep2 = unsafe { boolector_cond(self.raw_btor(), cond.raw, rt.raw, re.raw) };
                BitVec::BitVec {
                    width: *wt,
                    rep: BoolectorTerm { raw: rep2 },
                }
            }
            (BitVec::BitVec { width: wc, .. }, BitVec::ZeroWidth, BitVec::ZeroWidth) => {
                assert_eq!(wc, 1, "Condition must be 1-bit");
                BitVec::ZeroWidth
            }
            _ => panic!("Bitvector width mismatch in ite"),
        }
    }

    fn eq(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        let result = self.numerical(1, 1);
        self.bin_bool_op(x, y, |b, x, y| unsafe { boolector_eq(b, x, y) }, result)
    }

    fn ne(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        match (x, y) {
            (BitVec::BitVec { width: w1, rep: r1 }, BitVec::BitVec { width: w2, rep: r2 }) => {
                assert_eq!(w1, w2, "Bitvector width mismatch");
                let neq = unsafe { boolector_ne(self.raw_btor(), r1.raw, r2.raw) };
                self.bool_to_bv(BoolectorTerm { raw: neq })
            }
            (BitVec::ZeroWidth, BitVec::ZeroWidth) => self.numerical(1, 0),
            _ => panic!("Bitvector width mismatch"),
        }
    }

    fn slt(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        let result = self.numerical(1, 0);
        self.bin_bool_op(x, y, |b, x, y| unsafe { boolector_slt(b, x, y) }, result)
    }

    fn sgt(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        let result = self.numerical(1, 0);
        self.bin_bool_op(x, y, |b, x, y| unsafe { boolector_sgt(b, x, y) }, result)
    }

    fn sle(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        let result = self.numerical(1, 1);
        self.bin_bool_op(x, y, |b, x, y| unsafe { boolector_slte(b, x, y) }, result)
    }

    fn sge(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        let result = self.numerical(1, 1);
        self.bin_bool_op(x, y, |b, x, y| unsafe { boolector_sgte(b, x, y) }, result)
    }

    fn ult(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        let result = self.numerical(1, 0);
        self.bin_bool_op(x, y, |b, x, y| unsafe { boolector_ult(b, x, y) }, result)
    }

    fn ugt(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        let result = self.numerical(1, 0);
        self.bin_bool_op(x, y, |b, x, y| unsafe { boolector_ugt(b, x, y) }, result)
    }

    fn ule(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        let result = self.numerical(1, 1);
        self.bin_bool_op(x, y, |b, x, y| unsafe { boolector_ulte(b, x, y) }, result)
    }

    fn uge(&mut self, x: &BitVec<Self::Term>, y: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        let result = self.numerical(1, 1);
        self.bin_bool_op(x, y, |b, x, y| unsafe { boolector_ugte(b, x, y) }, result)
    }

    fn push(&mut self) -> io::Result<()> {
        unsafe { boolector_push(self.raw_btor(), 1) };
        Ok(())
    }

    fn pop(&mut self) -> io::Result<()> {
        unsafe { boolector_pop(self.raw_btor(), 1) };
        Ok(())
    }

    fn check(&mut self) -> io::Result<Response> {
        let r = unsafe { boolector_sat(self.raw_btor()) };
        match r {
            10 => Ok(Response::Sat),   // BTOR_RESULT_SAT
            20 => Ok(Response::Unsat), // BTOR_RESULT_UNSAT
            0 => Ok(Response::Unknown),
            _ => Err(io::Error::new(io::ErrorKind::Other, "boolector_sat failed")),
        }
    }

    fn assert(&mut self, bv: &BitVec<Self::Term>) -> io::Result<()> {
        let cond = self.bv_to_bool(bv);
        unsafe { boolector_assert(self.raw_btor(), cond.raw) };
        Ok(())
    }

    fn render(&mut self, bv: &BitVec<Self::Term>) -> String {
        match bv {
            BitVec::ZeroWidth => "<zero-width>".to_string(),
            BitVec::BitVec { rep, .. } => format!("<node {:?}>", rep.raw),
        }
    }
}

#[cfg(test)]
use crate::test_solver;

#[cfg(test)]
test_solver!(
    boolector_tests,
    super::Boolector::new(&super::BoolectorConfig::new()).unwrap()
);
