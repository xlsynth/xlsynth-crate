// SPDX-License-Identifier: Apache-2.0

use std::io;

use xlsynth::IrValue;

use crate::xls_ir::ir;

#[derive(Debug, PartialEq)]
pub enum Response {
    Sat,
    Unsat,
    Unknown,
}

// Generic helper: left-fold a vector with a binary operator implemented by the
// solver.
fn reduce_many<S: Solver, F>(
    solver: &mut S,
    bvs: Vec<&BitVec<S::Term>>,
    mut op: F,
) -> BitVec<S::Term>
where
    F: FnMut(&mut S, &BitVec<S::Term>, &BitVec<S::Term>) -> BitVec<S::Term>,
{
    let mut iter = bvs.into_iter();
    let mut acc = iter
        .next()
        .expect("*_many called with empty vector")
        .clone();
    for bv in iter {
        acc = op(solver, &acc, bv);
    }
    acc
}

pub trait Solver: Sized {
    type Term: Clone;
    type Config: Send + Sync;
    fn new(config: &Self::Config) -> io::Result<Self>;
    fn declare(&mut self, name: &str, width: usize) -> io::Result<BitVec<Self::Term>>;
    fn fresh_symbol(&mut self, name: &str) -> io::Result<String>;
    fn declare_fresh(&mut self, name: &str, width: usize) -> io::Result<BitVec<Self::Term>> {
        let symbol = self.fresh_symbol(name)?;
        self.declare(&symbol, width)
    }
    fn numerical(&mut self, width: usize, value: u64) -> BitVec<Self::Term>;
    /// Create a numerical bit-vector constant from a u128 value at an arbitrary width.
    /// Always produces exactly `width` bits (zero-extended or truncated as needed).
    fn numerical_u128(&mut self, width: usize, value: u128) -> BitVec<Self::Term> {
        if width == 0 {
            return BitVec::ZeroWidth;
        }
        // Build a binary string of exactly `width` bits (MSB-first) and delegate to from_raw_str.
        let mut s = String::with_capacity(2 + width);
        s.push_str("#b");
        for bit in (0..width).rev() {
            let b = ((value >> bit) & 1) != 0;
            s.push(if b { '1' } else { '0' });
        }
        self.from_raw_str(width, &s)
    }
    fn zero(&mut self, width: usize) -> BitVec<Self::Term> {
        self.numerical(width, 0)
    }
    fn one(&mut self, width: usize) -> BitVec<Self::Term> {
        self.numerical(width, 1)
    }
    fn all_ones(&mut self, width: usize) -> BitVec<Self::Term> {
        if width == 0 {
            return BitVec::ZeroWidth;
        }
        let mut str = "#b".to_string();
        str.push_str(&"1".repeat(width));
        self.from_raw_str(width, &str)
    }
    fn signed_max_value(&mut self, width: usize) -> BitVec<Self::Term> {
        if width == 0 {
            return BitVec::ZeroWidth;
        }
        if width == 1 {
            return self.zero(1);
        }
        let mut str = "#b0".to_string();
        str.push_str(&"1".repeat(width - 1));
        self.from_raw_str(width, &str)
    }
    fn signed_min_value(&mut self, width: usize) -> BitVec<Self::Term> {
        if width == 0 {
            return BitVec::ZeroWidth;
        }
        if width == 1 {
            return self.one(1);
        }
        let mut str = "#b1".to_string();
        str.push_str(&"0".repeat(width - 1));
        self.from_raw_str(width, &str)
    }
    fn sign_bit(&mut self, bit_vec: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        match bit_vec.clone() {
            BitVec::BitVec { width, .. } => {
                if width == 0 {
                    return BitVec::ZeroWidth;
                }
                self.slice(bit_vec, width - 1, 1)
            }
            BitVec::ZeroWidth => panic!("sign_bit: bit_vec is zero-width"),
        }
    }
    fn true_bv(&mut self) -> BitVec<Self::Term> {
        self.numerical(1, 1)
    }
    fn false_bv(&mut self) -> BitVec<Self::Term> {
        self.numerical(1, 0)
    }
    fn from_raw_str(&mut self, width: usize, value: &str) -> BitVec<Self::Term>;
    fn zero_width(&mut self) -> BitVec<Self::Term> {
        BitVec::ZeroWidth
    }
    fn get_value(&mut self, bit_vec: &BitVec<Self::Term>, ty: &ir::Type) -> io::Result<IrValue>;
    fn extract(&mut self, bit_vec: &BitVec<Self::Term>, high: i32, low: i32) -> BitVec<Self::Term>;
    fn slice(
        &mut self,
        bit_vec: &BitVec<Self::Term>,
        start: usize,
        width: usize,
    ) -> BitVec<Self::Term> {
        self.extract(bit_vec, (start + width - 1) as i32, start as i32)
    }
    fn not(&mut self, bit_vec: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn neg(&mut self, bit_vec: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn reverse(&mut self, bit_vec: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn or_reduce(&mut self, bit_vec: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn and_reduce(&mut self, bit_vec: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn xor_reduce(&mut self, bit_vec: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn add(&mut self, lhs: &BitVec<Self::Term>, rhs: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn sub(&mut self, lhs: &BitVec<Self::Term>, rhs: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn mul(&mut self, lhs: &BitVec<Self::Term>, rhs: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn udiv(&mut self, lhs: &BitVec<Self::Term>, rhs: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn urem(&mut self, lhs: &BitVec<Self::Term>, rhs: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn srem(&mut self, lhs: &BitVec<Self::Term>, rhs: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn sdiv(&mut self, lhs: &BitVec<Self::Term>, rhs: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn shl(&mut self, lhs: &BitVec<Self::Term>, rhs: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn lshr(&mut self, lhs: &BitVec<Self::Term>, rhs: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn ashr(&mut self, lhs: &BitVec<Self::Term>, rhs: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn concat(&mut self, lhs: &BitVec<Self::Term>, rhs: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn or(&mut self, lhs: &BitVec<Self::Term>, rhs: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn and(&mut self, lhs: &BitVec<Self::Term>, rhs: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn xor(&mut self, lhs: &BitVec<Self::Term>, rhs: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn nor(&mut self, lhs: &BitVec<Self::Term>, rhs: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn nand(&mut self, lhs: &BitVec<Self::Term>, rhs: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn extend(
        &mut self,
        bit_vec: &BitVec<Self::Term>,
        extend_width: usize,
        signed: bool,
    ) -> BitVec<Self::Term>;
    fn extend_to(
        &mut self,
        bit_vec: &BitVec<Self::Term>,
        new_width: usize,
        signed: bool,
    ) -> BitVec<Self::Term> {
        let width = bit_vec.get_width();
        assert!(new_width >= width, "Cannot extend to smaller width");
        self.extend(bit_vec, new_width - width, signed)
    }
    fn zero_extend(
        &mut self,
        bit_vec: &BitVec<Self::Term>,
        extend_width: usize,
    ) -> BitVec<Self::Term> {
        self.extend(bit_vec, extend_width, false)
    }
    fn sign_extend(
        &mut self,
        bit_vec: &BitVec<Self::Term>,
        extend_width: usize,
    ) -> BitVec<Self::Term> {
        self.extend(bit_vec, extend_width, true)
    }
    fn zero_extend_to(
        &mut self,
        bit_vec: &BitVec<Self::Term>,
        new_width: usize,
    ) -> BitVec<Self::Term> {
        self.extend_to(bit_vec, new_width, false)
    }
    fn sign_extend_to(
        &mut self,
        bit_vec: &BitVec<Self::Term>,
        new_width: usize,
    ) -> BitVec<Self::Term> {
        self.extend_to(bit_vec, new_width, true)
    }
    fn ite(
        &mut self,
        lhs: &BitVec<Self::Term>,
        then: &BitVec<Self::Term>,
        else_: &BitVec<Self::Term>,
    ) -> BitVec<Self::Term>;
    fn eq(&mut self, lhs: &BitVec<Self::Term>, rhs: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn ne(&mut self, lhs: &BitVec<Self::Term>, rhs: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn slt(&mut self, lhs: &BitVec<Self::Term>, rhs: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn sgt(&mut self, lhs: &BitVec<Self::Term>, rhs: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn sle(&mut self, lhs: &BitVec<Self::Term>, rhs: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn sge(&mut self, lhs: &BitVec<Self::Term>, rhs: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn ult(&mut self, lhs: &BitVec<Self::Term>, rhs: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn ugt(&mut self, lhs: &BitVec<Self::Term>, rhs: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn ule(&mut self, lhs: &BitVec<Self::Term>, rhs: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn uge(&mut self, lhs: &BitVec<Self::Term>, rhs: &BitVec<Self::Term>) -> BitVec<Self::Term>;
    fn push(&mut self) -> io::Result<()>;
    fn pop(&mut self) -> io::Result<()>;
    fn check(&mut self) -> io::Result<Response>;
    fn assert(&mut self, bit_vec: &BitVec<Self::Term>) -> io::Result<()>;
    fn render(&mut self, bit_vec: &BitVec<Self::Term>) -> String;
    fn smax(&mut self, lhs: &BitVec<Self::Term>, rhs: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        let cond = self.slt(lhs, rhs);
        self.ite(&cond, rhs, lhs)
    }
    fn umax(&mut self, lhs: &BitVec<Self::Term>, rhs: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        let cond = self.ult(lhs, rhs);
        self.ite(&cond, rhs, lhs)
    }
    fn smin(&mut self, lhs: &BitVec<Self::Term>, rhs: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        let cond = self.slt(lhs, rhs);
        self.ite(&cond, lhs, rhs)
    }
    fn umin(&mut self, lhs: &BitVec<Self::Term>, rhs: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        let cond = self.ult(lhs, rhs);
        self.ite(&cond, lhs, rhs)
    }
    fn is_zero(&mut self, bit_vec: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        let zero = self.zero(bit_vec.get_width());
        self.eq(bit_vec, &zero)
    }
    fn is_one(&mut self, bit_vec: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        let one = self.one(bit_vec.get_width());
        self.eq(bit_vec, &one)
    }
    fn is_all_ones(&mut self, bit_vec: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        let all_ones = self.all_ones(bit_vec.get_width());
        self.eq(bit_vec, &all_ones)
    }
    fn is_signed_max_value(&mut self, bit_vec: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        let signed_max_value = self.signed_max_value(bit_vec.get_width());
        self.eq(bit_vec, &signed_max_value)
    }
    fn is_signed_min_value(&mut self, bit_vec: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        let signed_min_value = self.signed_min_value(bit_vec.get_width());
        self.eq(bit_vec, &signed_min_value)
    }
    fn concat_many(&mut self, bvs: Vec<&BitVec<Self::Term>>) -> BitVec<Self::Term> {
        reduce_many(self, bvs, Self::concat)
    }
    fn and_many(&mut self, bvs: Vec<&BitVec<Self::Term>>) -> BitVec<Self::Term> {
        reduce_many(self, bvs, Self::and)
    }
    fn xor_many(&mut self, bvs: Vec<&BitVec<Self::Term>>) -> BitVec<Self::Term> {
        reduce_many(self, bvs, Self::xor)
    }
    fn nor_many(&mut self, bvs: Vec<&BitVec<Self::Term>>) -> BitVec<Self::Term> {
        let or_result = reduce_many(self, bvs, Self::or);
        self.not(&or_result)
    }
    fn or_many(&mut self, bvs: Vec<&BitVec<Self::Term>>) -> BitVec<Self::Term> {
        reduce_many(self, bvs, Self::or)
    }
    fn nand_many(&mut self, bvs: Vec<&BitVec<Self::Term>>) -> BitVec<Self::Term> {
        let and_result = reduce_many(self, bvs, Self::and);
        self.not(&and_result)
    }
    fn xls_div(
        &mut self,
        lhs: &BitVec<Self::Term>,
        rhs: &BitVec<Self::Term>,
        signed: bool,
    ) -> BitVec<Self::Term> {
        // Implements XLS semantics for unsigned division with division-by-zero
        // handling. When the divisor (b) is zero, XLS specifies that the result
        // should be all ones. For zero-width values the result is also zero-width.

        let width = lhs.get_width();
        assert_eq!(
            width,
            rhs.get_width(),
            "LHS and RHS must have the same width"
        );

        // Zero-width values carry through.
        if width == 0 {
            return BitVec::ZeroWidth;
        }

        // For signed division the XLS semantics specify:
        //   if divisor == 0 then result is:
        //      max_positive  if dividend >= 0
        //      max_negative  if dividend < 0
        // For unsigned division the result is all-ones.
        // Pre-compute these fallback values.  Note that for
        // width == 1 the "max positive" and "max negative"
        // are both 1-bit wide and simply 0 and 1 respectively.

        let div_res = if signed {
            self.sdiv(lhs, rhs)
        } else {
            self.udiv(lhs, rhs)
        };
        let rhs_is_zero = self.is_zero(rhs);
        let all_ones = self.all_ones(width);
        let max_positive = self.signed_max_value(width);
        let max_negative = self.signed_min_value(width);
        if signed {
            let sign_bit = self.sign_bit(lhs);
            let fallback = self.ite(&sign_bit, &max_negative, &max_positive);
            self.ite(&rhs_is_zero, &fallback, &div_res)
        } else {
            self.ite(&rhs_is_zero, &all_ones, &div_res)
        }
    }

    fn xls_sdiv(
        &mut self,
        lhs: &BitVec<Self::Term>,
        rhs: &BitVec<Self::Term>,
    ) -> BitVec<Self::Term> {
        self.xls_div(lhs, rhs, true)
    }
    fn xls_udiv(
        &mut self,
        lhs: &BitVec<Self::Term>,
        rhs: &BitVec<Self::Term>,
    ) -> BitVec<Self::Term> {
        self.xls_div(lhs, rhs, false)
    }

    fn xls_mod(
        &mut self,
        lhs: &BitVec<Self::Term>,
        rhs: &BitVec<Self::Term>,
        signed: bool,
    ) -> BitVec<Self::Term> {
        let width = lhs.get_width();
        if width == 0 {
            return BitVec::ZeroWidth;
        }
        assert_eq!(
            width,
            rhs.get_width(),
            "LHS and RHS must have the same width"
        );
        let rhs_is_zero = self.is_zero(rhs);
        let mod_res = if signed {
            self.srem(lhs, rhs)
        } else {
            self.urem(lhs, rhs)
        };
        let zero = self.zero(width);
        self.ite(&rhs_is_zero, &zero, &mod_res)
    }

    fn xls_smod(
        &mut self,
        lhs: &BitVec<Self::Term>,
        rhs: &BitVec<Self::Term>,
    ) -> BitVec<Self::Term> {
        self.xls_mod(lhs, rhs, true)
    }
    fn xls_umod(
        &mut self,
        lhs: &BitVec<Self::Term>,
        rhs: &BitVec<Self::Term>,
    ) -> BitVec<Self::Term> {
        self.xls_mod(lhs, rhs, false)
    }

    fn fit_width(
        &mut self,
        bv: &BitVec<Self::Term>,
        output_width: usize,
        signed: bool,
    ) -> BitVec<Self::Term> {
        if bv.get_width() < output_width {
            if signed {
                self.sign_extend_to(bv, output_width)
            } else {
                self.zero_extend_to(&bv, output_width)
            }
        } else if bv.get_width() > output_width {
            self.slice(&bv, 0, output_width)
        } else {
            bv.clone()
        }
    }

    fn signed_fit_width(
        &mut self,
        bv: &BitVec<Self::Term>,
        output_width: usize,
    ) -> BitVec<Self::Term> {
        self.fit_width(bv, output_width, true)
    }

    fn unsigned_fit_width(
        &mut self,
        bv: &BitVec<Self::Term>,
        output_width: usize,
    ) -> BitVec<Self::Term> {
        self.fit_width(bv, output_width, false)
    }

    fn xls_arbitrary_width_mul(
        &mut self,
        lhs: &BitVec<Self::Term>,
        rhs: &BitVec<Self::Term>,
        signed: bool,
        output_width: usize,
    ) -> BitVec<Self::Term> {
        let lhs_fit = self.fit_width(lhs, output_width, signed);
        let rhs_fit = self.fit_width(rhs, output_width, signed);
        self.mul(&lhs_fit, &rhs_fit)
    }

    fn xls_arbitrary_width_smul(
        &mut self,
        lhs: &BitVec<Self::Term>,
        rhs: &BitVec<Self::Term>,
        output_width: usize,
    ) -> BitVec<Self::Term> {
        self.xls_arbitrary_width_mul(lhs, rhs, true, output_width)
    }

    fn xls_arbitrary_width_umul(
        &mut self,
        lhs: &BitVec<Self::Term>,
        rhs: &BitVec<Self::Term>,
        output_width: usize,
    ) -> BitVec<Self::Term> {
        self.xls_arbitrary_width_mul(lhs, rhs, false, output_width)
    }

    fn xls_shift(
        &mut self,
        operand: &BitVec<Self::Term>,
        shift_amount: &BitVec<Self::Term>,
        left_shift: bool,
        arithmetic_shift: bool,
    ) -> BitVec<Self::Term> {
        let width = operand.get_width();
        if width == 0 {
            return BitVec::ZeroWidth;
        }

        // -------- choose common width ----------------------------------------
        let target_width = std::cmp::max(width, shift_amount.get_width().max(1));

        // -------- extend operand --------------------------------------------
        let op_ext = if target_width == width {
            operand.clone()
        } else if arithmetic_shift && !left_shift {
            // sign-extend for arithmetic right shift so the sign bit is in MSB.
            self.sign_extend_to(operand, target_width)
        } else {
            self.zero_extend_to(operand, target_width)
        };

        // -------- adjust shift amount to target_width -----------------------
        let shamt_full = if shift_amount.get_width() == target_width {
            shift_amount.clone()
        } else {
            self.zero_extend_to(shift_amount, target_width)
        };

        // -------- perform shift ---------------------------------------------
        let shifted = if left_shift {
            self.shl(&op_ext, &shamt_full)
        } else if arithmetic_shift {
            self.ashr(&op_ext, &shamt_full)
        } else {
            self.lshr(&op_ext, &shamt_full)
        };

        // Slice back to original width and return.
        self.slice(&shifted, 0, width)
    }

    fn xls_shll(
        &mut self,
        operand: &BitVec<Self::Term>,
        shift_amount: &BitVec<Self::Term>,
    ) -> BitVec<Self::Term> {
        self.xls_shift(operand, shift_amount, true, false)
    }

    fn xls_shra(
        &mut self,
        operand: &BitVec<Self::Term>,
        shift_amount: &BitVec<Self::Term>,
    ) -> BitVec<Self::Term> {
        self.xls_shift(operand, shift_amount, false, true)
    }

    fn xls_shrl(
        &mut self,
        operand: &BitVec<Self::Term>,
        shift_amount: &BitVec<Self::Term>,
    ) -> BitVec<Self::Term> {
        self.xls_shift(operand, shift_amount, false, false)
    }

    fn xls_one_hot(&mut self, arg: &BitVec<Self::Term>, lsb_prio: bool) -> BitVec<Self::Term> {
        if lsb_prio {
            let one = self.one(1);
            let arg_concated = self.concat(&one, arg);
            let neg = self.neg(&arg_concated);
            let one_hot = self.and(&arg_concated, &neg);
            one_hot
        } else {
            let zero = self.zero(1);
            let arg_concated = self.concat(&zero, arg);
            let width = arg_concated.get_width();
            let mut shift_amount_bits = 1;
            let mut shift_amount = 1;
            let mut result = arg_concated.clone();
            while shift_amount < width {
                let shift_amount_bv = self.numerical(shift_amount_bits, shift_amount as u64);
                let shifted_result = self.xls_shrl(&result, &shift_amount_bv);
                result = self.or(&result, &shifted_result);
                shift_amount *= 2;
                shift_amount_bits += 1;
            }
            // Isolate the MSB bit:  msb = result & ~ (result >> 1)
            let one = self.one(1);
            let shifted_result = self.xls_shrl(&result, &one);
            let flipped_shifted_result = self.not(&shifted_result);
            let non_zero_result = self.and(&flipped_shifted_result, &result);

            // Build a sentinel vector that has the top bit set iff the input was zero.
            // This avoids the use of an `ite`, generating a simpler SMT formula.
            let is_zero = self.is_zero(&arg_concated); // 1-bit
            let zeros = self.zero(width - 1); // width-1 zeros
            let sentinel_result = self.concat(&is_zero, &zeros); // width bits

            // The sentinel and non-zero results are mutually exclusive, so a bit-wise OR
            // yields the desired final value.
            self.or(&non_zero_result, &sentinel_result)
        }
    }

    fn xls_one_hot_lsb_prio(&mut self, arg: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.xls_one_hot(arg, true)
    }

    fn xls_one_hot_msb_prio(&mut self, arg: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        self.xls_one_hot(arg, false)
    }

    // --------------------------------------------------------------------
    // XLS decode/encode helpers
    // --------------------------------------------------------------------
    fn xls_decode(
        &mut self,
        operand: &BitVec<Self::Term>,
        result_width: usize,
    ) -> BitVec<Self::Term> {
        // Zero-width result: propagate.
        if result_width == 0 {
            return BitVec::ZeroWidth;
        }

        // A one-hot decode is simply `1 << operand` with XLS semantics: if the
        // shift amount is ≥ result_width the helper already returns 0, matching
        // the required saturation behaviour.

        let one = self.one(result_width);
        self.xls_shll(&one, operand)
    }

    fn xls_encode(&mut self, operand: &BitVec<Self::Term>) -> BitVec<Self::Term> {
        let n_bits = operand.get_width();
        // Compute ceil(log2(n_bits)).  For n_bits <= 1 the result is zero-width.
        let result_width = if n_bits <= 1 {
            0
        } else {
            (usize::BITS as usize) - ((n_bits - 1).leading_zeros() as usize)
        };

        if result_width == 0 {
            return BitVec::ZeroWidth;
        }

        let zero_vec = self.zero(result_width);
        let mut parts: Vec<BitVec<Self::Term>> = Vec::with_capacity(n_bits.max(1));

        for idx in 0..n_bits {
            // Extract the bit at position idx (LSB == 0).
            let bit_i = self.slice(operand, idx, 1); // 1-bit condition
            let idx_const = self.numerical(result_width, idx as u64);
            let selected = self.ite(&bit_i, &idx_const, &zero_vec);
            parts.push(selected);
        }

        self.or_many(parts.iter().collect())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BitVec<Term> {
    BitVec { width: usize, rep: Term },
    ZeroWidth,
}

impl<Term> BitVec<Term> {
    pub fn get_width(&self) -> usize {
        match self {
            BitVec::BitVec { width, .. } => *width,
            BitVec::ZeroWidth => 0,
        }
    }

    pub fn get_term(&self) -> Option<&Term> {
        match self {
            BitVec::BitVec { rep, .. } => Some(rep),
            BitVec::ZeroWidth => None,
        }
    }
}

#[cfg(test)]
pub mod test_utils {
    use super::*;
    pub fn test_bitvec_equiv_basic<S: Solver>(solver: &mut S) {
        let a = solver.declare("a", 16).unwrap();
        let eq_constraint = solver.eq(&a, &a);
        solver.assert(&eq_constraint).unwrap();
        assert_eq!(solver.check().unwrap(), Response::Sat);
    }

    pub fn test_numerical_inbound<S: Solver>(solver: &mut S) {
        let a = solver.declare("a", 16).unwrap();
        let b = solver.numerical(16, 14);
        let eq_constraint = solver.eq(&a, &b);
        solver.assert(&eq_constraint).unwrap();
        assert_eq!(solver.check().unwrap(), Response::Sat);
        assert_eq!(
            solver.get_value(&a, &ir::Type::Bits(16)).unwrap(),
            IrValue::make_ubits(16, 14).unwrap()
        );
    }

    pub fn test_numerical_outbound<S: Solver>(solver: &mut S) {
        let a = solver.declare("a", 16).unwrap();
        let b = solver.numerical(16, 0xfedca);
        let eq_constraint = solver.eq(&a, &b);
        solver.assert(&eq_constraint).unwrap();
        assert_eq!(solver.check().unwrap(), Response::Sat);
        assert_eq!(
            solver.get_value(&a, &ir::Type::Bits(16)).unwrap(),
            IrValue::make_ubits(16, 0xedca).unwrap()
        );
    }

    pub fn test_numerical_long<S: Solver>(solver: &mut S) {
        let a = solver.declare("a", 128).unwrap();
        let b = solver.numerical(128, 0xfedca);
        let eq_constraint = solver.eq(&a, &b);
        solver.assert(&eq_constraint).unwrap();
        assert_eq!(solver.check().unwrap(), Response::Sat);
        assert_eq!(
            solver.get_value(&a, &ir::Type::Bits(128)).unwrap(),
            IrValue::make_ubits(128, 0xfedca).unwrap()
        );
    }

    pub fn test_constants<S: Solver>(solver: &mut S) {
        let zero = solver.zero(128);
        let one = solver.one(128);
        let all_ones = solver.all_ones(128);
        let minus_one = solver.sub(&zero, &one);
        let signed_max_value = solver.signed_max_value(128);
        let signed_min_value = solver.signed_min_value(128);
        let two = solver.numerical(128, 2);
        let minus_two = solver.mul(&signed_max_value, &two);
        let zero_alt = solver.add(&minus_two, &two);
        let minus_one_alt = solver.add(&signed_max_value, &signed_min_value);
        assert_solver_eq(solver, &all_ones, &minus_one);
        assert_solver_eq(solver, &zero, &zero_alt);
        assert_solver_eq(solver, &minus_one, &minus_one_alt);

        let is_zero = solver.is_zero(&zero);
        let is_one = solver.is_one(&one);
        let is_all_ones = solver.is_all_ones(&all_ones);
        let is_signed_max_value = solver.is_signed_max_value(&signed_max_value);
        let is_signed_min_value = solver.is_signed_min_value(&signed_min_value);
        let true_bv = solver.true_bv();
        assert_solver_eq(solver, &is_zero, &true_bv);
        assert_solver_eq(solver, &is_one, &true_bv);
        assert_solver_eq(solver, &is_all_ones, &true_bv);
        assert_solver_eq(solver, &is_signed_max_value, &true_bv);
        assert_solver_eq(solver, &is_signed_min_value, &true_bv);
    }

    pub fn test_sign_bit<S: Solver>(solver: &mut S) {
        let neg = solver.numerical(16, 0x8000);
        let zero = solver.zero(16);
        let pos = solver.numerical(16, 0x7fff);
        let neg_sign_bit = solver.sign_bit(&neg);
        let zero_sign_bit = solver.sign_bit(&zero);
        let pos_sign_bit = solver.sign_bit(&pos);
        let true_bv = solver.true_bv();
        let false_bv = solver.false_bv();
        assert_solver_eq(solver, &neg_sign_bit, &true_bv);
        assert_solver_eq(solver, &zero_sign_bit, &false_bv);
        assert_solver_eq(solver, &pos_sign_bit, &false_bv);
    }

    pub fn test_extract<S: Solver>(solver: &mut S) {
        let a = solver.declare("a", 8).unwrap();
        let b = solver.numerical(16, 0x89ab);
        let c = solver.extract(&b, 12, 5);
        let eq_constraint = solver.eq(&a, &c);
        solver.assert(&eq_constraint).unwrap();
        assert_eq!(solver.check().unwrap(), Response::Sat);
        assert_eq!(
            solver.get_value(&a, &ir::Type::Bits(8)).unwrap(),
            IrValue::make_ubits(8, 0x4d).unwrap()
        );
    }

    pub fn test_slice<S: Solver>(solver: &mut S) {
        let a = solver.declare("a", 8).unwrap();
        let b = solver.numerical(16, 0x89ab);
        let c = solver.slice(&b, 5, 8);
        let eq_constraint = solver.eq(&a, &c);
        solver.assert(&eq_constraint).unwrap();
        assert_eq!(solver.check().unwrap(), Response::Sat);
        assert_eq!(
            solver.get_value(&a, &ir::Type::Bits(8)).unwrap(),
            IrValue::make_ubits(8, 0x4d).unwrap()
        );
    }

    pub fn assert_solver_eq<S: Solver>(
        solver: &mut S,
        lhs: &BitVec<S::Term>,
        rhs: &BitVec<S::Term>,
    ) {
        solver.push().unwrap();
        let eq_constraint = solver.eq(&lhs, &rhs);
        solver.assert(&eq_constraint).unwrap();
        assert_eq!(solver.check().unwrap(), Response::Sat);
        solver.pop().unwrap();
        solver.push().unwrap();
        let ne_constraint = solver.ne(&lhs, &rhs);
        solver.assert(&ne_constraint).unwrap();
        assert_eq!(solver.check().unwrap(), Response::Unsat);
        solver.pop().unwrap();
    }

    #[macro_export]
    macro_rules! test_get_value {
        ($fn_name:ident,
         $solver:expr,
         $in_w:expr,  $in_val:expr,
         $out_ty:expr, $out_val:expr $(,)?
        ) => {
            #[test]
            fn $fn_name() {
                let mut solver = $solver;
                let operand = solver.from_raw_str($in_w, $in_val);
                solver.check().unwrap();
                let actual = solver.get_value(&operand, &$out_ty).unwrap();
                let expected = $out_val;
                assert_eq!(actual, expected);
            }
        };
    }

    // -----------------------------------------------------------------------
    // Generic "one-operand → one result" test helper for Solver.
    //
    //  • `$fn_name`     – the Rust test-function identifier that will be
    //                     generated by the macro.
    //  • `$op`          – the `Solver` method to call (e.g. `not`, `neg`,
    //                     `reverse`, `or_reduce`, …).
    //  • `$in_w`        – bit-width of the operand fed to the solver.
    //  • `$in_val`      – literal value of that operand.
    //  • `$out_w`       – bit-width expected from the operation
    //                     (use 1 for reductions).
    //  • `$out_val`     – literal value expected from the operation.
    // -----------------------------------------------------------------------
    #[macro_export]
    macro_rules! test_solver_unary {
        ($fn_name:ident,
         $solver:expr,
         $op:ident,
         $in_w:expr,  $in_val:expr,
         $out_w:expr, $out_val:expr $(,)?
        ) => {
            #[test]
            fn $fn_name() {
                // Arrange
                let mut solver = $solver;
                let operand = solver.numerical($in_w, $in_val);

                // Act
                let actual = solver.$op(&operand);
                let expected = solver.numerical($out_w, $out_val);
                // Assert (via the common helper already in the file)
                crate::equiv::solver_interface::test_utils::assert_solver_eq(
                    &mut solver,
                    &actual,
                    &expected,
                );
            }
        };
    }

    // -----------------------------------------------------------------------
    // Generic "two-operand → one result" test helper for Solver.
    //
    //  • `$fn_name`     – the Rust test-function identifier that will be
    //                     generated by the macro.
    //  • `$op`          – the `Solver` method to call (e.g. `add`, `sub`,
    //                     `mul`, `udiv`, …).
    //  • `$in_w`        – bit-width of the operands fed to the solver.
    //  • `$in_val`      – literal value of that operand.
    //  • `$out_w`       – bit-width expected from the operation
    //                     (use 1 for reductions).
    //  • `$out_val`     – literal value expected from the operation.
    // -----------------------------------------------------------------------
    #[macro_export]
    macro_rules! test_solver_binary_arbitrary_width {
        ($fn_name:ident,
         $solver:expr,
         $op:ident,
         $in_w1:expr,  $in_val1:expr, $in_w2:expr, $in_val2:expr,
         $out_w:expr, $out_val:expr $(,)?
        ) => {
            #[test]
            fn $fn_name() {
                // Arrange
                let mut solver = $solver;
                let operand1 = solver.numerical($in_w1, $in_val1);
                let operand2 = solver.numerical($in_w2, $in_val2);

                // Act
                let actual = solver.$op(&operand1, &operand2);
                let expected = solver.numerical($out_w, $out_val);

                // Assert (via the common helper already in the file)
                crate::equiv::solver_interface::test_utils::assert_solver_eq(
                    &mut solver,
                    &actual,
                    &expected,
                );
            }
        };
    }

    #[macro_export]
    macro_rules! test_solver_binary {
        ($fn_name:ident,
         $solver:expr,
         $op:ident,
         $in_w:expr,  $in_val1:expr, $in_val2:expr,
         $out_w:expr, $out_val:expr $(,)?
        ) => {
            crate::test_solver_binary_arbitrary_width!(
                $fn_name, $solver, $op, $in_w, $in_val1, $in_w, $in_val2, $out_w, $out_val
            );
        };
    }

    // -----------------------------------------------------------------------
    // Generic "extend" test helper (both zero_extend/sign_extend variants).
    // The operator must have the signature (BitVec, usize) -> BitVec.
    // -----------------------------------------------------------------------
    #[macro_export]
    macro_rules! test_solver_extend {
        ($fn_name:ident,
         $solver:expr,
         $op:ident,
         $in_w:expr,  $in_val:expr,
         $ext_arg:expr,
         $out_w:expr, $out_val:expr $(,)?
        ) => {
            #[test]
            fn $fn_name() {
                let mut solver = $solver;
                let operand = solver.numerical($in_w, $in_val);
                let actual = solver.$op(&operand, $ext_arg);
                let expected = solver.numerical($out_w, $out_val);
                crate::equiv::solver_interface::test_utils::assert_solver_eq(
                    &mut solver,
                    &actual,
                    &expected,
                );
            }
        };
    }

    // -------------------------------------------------------------------
    // Macro for ITE tests
    // -------------------------------------------------------------------
    #[macro_export]
    macro_rules! test_solver_ite {
        ($fn_name:ident,
         $solver:expr,
         $cond_val:expr, $then_val:expr, $else_val:expr, $expected_val:expr $(,)?) => {
            #[test]
            fn $fn_name() {
                let mut solver = $solver;
                let cond = solver.numerical(1, $cond_val);
                let then_val = solver.numerical(8, $then_val);
                let else_val = solver.numerical(8, $else_val);
                let actual = solver.ite(&cond, &then_val, &else_val);
                let expected = solver.numerical(8, $expected_val);
                crate::equiv::solver_interface::test_utils::assert_solver_eq(
                    &mut solver,
                    &actual,
                    &expected,
                );
            }
        };
    }

    // -------------------------------------------------------------------
    // Comparison operator tests (eq, ne, slt … uge)
    // -------------------------------------------------------------------
    #[macro_export]
    macro_rules! test_solver_cmp {
        ($fn_name:ident,
         $solver:expr,
         $op:ident,
         $w:expr, $lhs:expr, $rhs:expr, $expected:expr $(,)?) => {
            #[test]
            fn $fn_name() {
                let mut solver = $solver;
                let lhs = solver.numerical($w, $lhs);
                let rhs = solver.numerical($w, $rhs);
                let actual = solver.$op(&lhs, &rhs);
                let expected = solver.numerical(1, $expected);
                crate::equiv::solver_interface::test_utils::assert_solver_eq(
                    &mut solver,
                    &actual,
                    &expected,
                );
            }
        };
    }

    #[macro_export]
    macro_rules! test_solver_cmp_zero_width {
        ($fn_name:ident,
         $solver:expr,
         $op:ident,
         $expected:expr) => {
            #[test]
            fn $fn_name() {
                let mut solver = $solver;
                let lhs = solver.zero_width();
                let rhs = solver.zero_width();
                let actual = solver.$op(&lhs, &rhs);
                let expected = $expected(&mut solver);
                crate::equiv::solver_interface::test_utils::assert_solver_eq(
                    &mut solver,
                    &actual,
                    &expected,
                );
            }
        };
    }

    #[macro_export]
    macro_rules! test_solver_binary_arbitrary_width_mul {
        ($fn_name:ident,
         $solver:expr,
         $op:ident,
         $in_w1:expr,  $in_val1:expr, $in_w2:expr, $in_val2:expr,
         $out_w:expr, $out_val:expr
        ) => {
            #[test]
            fn $fn_name() {
                // Arrange
                let mut solver = $solver;
                let operand1 = solver.numerical($in_w1, $in_val1);
                let operand2 = solver.numerical($in_w2, $in_val2);

                // Act
                let actual = solver.$op(&operand1, &operand2, $out_w);
                let expected = solver.numerical($out_w, $out_val);

                // Assert (via the common helper already in the file)
                crate::equiv::solver_interface::test_utils::assert_solver_eq(
                    &mut solver,
                    &actual,
                    &expected,
                );
            }
        };
    }
    // -------------------------------------------------------------------
    // Variadic helper ( *_many ) test macro
    // -------------------------------------------------------------------
    #[macro_export]
    macro_rules! test_solver_many {
    ($fn_name:ident,
     $solver:expr,
     $method:ident,
     $w:expr,
     [$($val:expr),+ $(,)?],
     $wr:expr,
     $expected:expr $(,)?
    ) => {
        #[test]
        fn $fn_name() {
            let mut solver = $solver;
            let vec_vals: Vec<_> = vec![ $( solver.numerical($w, $val) ),+ ];
            let vec_refs: Vec<_> = vec_vals.iter().map(|bv| bv).collect();
            let actual = solver.$method(vec_refs);
            let expected = solver.numerical($wr, $expected);
            crate::equiv::solver_interface::test_utils::assert_solver_eq(&mut solver, &actual, &expected);
        }
    };
}
}

#[cfg(test)]
#[macro_export]
macro_rules! test_solver {
    ($mod_ident:ident, $solver:expr) => {
        #[cfg(test)]
        mod $mod_ident {
            use crate::equiv::solver_interface::Solver;
            use crate::equiv::solver_interface::test_utils;

            #[test]
            fn test_bitvec_equiv_basic() {
                let mut solver = $solver;

                test_utils::test_bitvec_equiv_basic(&mut solver);
            }

            #[test]
            fn test_numerical_inbound() {
                let mut solver = $solver;
                test_utils::test_numerical_inbound(&mut solver);
            }

            #[test]
            fn test_numerical_outbound() {
                let mut solver = $solver;
                test_utils::test_numerical_outbound(&mut solver);
            }

            #[test]
            fn test_numerical_long() {
                let mut solver = $solver;
                test_utils::test_numerical_long(&mut solver);
            }

            crate::test_get_value!(
                test_get_value_hex,
                $solver,
                8,
                "#x53",
                crate::xls_ir::ir::Type::Tuple(vec![
                    Box::new(crate::xls_ir::ir::Type::Bits(2)),
                    Box::new(crate::xls_ir::ir::Type::Bits(1)),
                    Box::new(crate::xls_ir::ir::Type::Bits(5))
                ]),
                xlsynth::IrValue::make_tuple(&[
                    xlsynth::IrValue::make_ubits(2, 0b11).unwrap(),
                    xlsynth::IrValue::make_ubits(1, 0b0).unwrap(),
                    xlsynth::IrValue::make_ubits(5, 0b01010).unwrap(),
                ])
            );

            crate::test_get_value!(
                test_get_value_bin,
                $solver,
                8,
                "#b01010011",
                crate::xls_ir::ir::Type::Tuple(vec![
                    Box::new(crate::xls_ir::ir::Type::Bits(2)),
                    Box::new(crate::xls_ir::ir::Type::Bits(1)),
                    Box::new(crate::xls_ir::ir::Type::Bits(5))
                ]),
                xlsynth::IrValue::make_tuple(&[
                    xlsynth::IrValue::make_ubits(2, 0b11).unwrap(),
                    xlsynth::IrValue::make_ubits(1, 0b0).unwrap(),
                    xlsynth::IrValue::make_ubits(5, 0b01010).unwrap(),
                ])
            );

            #[test]
            fn test_constants() {
                let mut solver = $solver;
                test_utils::test_constants(&mut solver);
            }

            #[test]
            fn test_sign_bit() {
                let mut solver = $solver;
                test_utils::test_sign_bit(&mut solver);
            }

            #[test]
            fn test_extract() {
                let mut solver = $solver;
                test_utils::test_extract(&mut solver);
            }

            #[test]
            fn test_slice() {
                let mut solver = $solver;
                test_utils::test_slice(&mut solver);
            }

            // New tests for declare_fresh symbol generation behavior
            #[test]
            fn test_declare_fresh_generates_unique_symbols() {
                let mut solver = $solver;
                let f1 = solver.declare_fresh("a", 8).unwrap();
                let f2 = solver.declare_fresh("a", 8).unwrap();
                solver.push().unwrap();
                let ne = solver.ne(&f1, &f2);
                solver.assert(&ne).unwrap();
                assert_eq!(
                    solver.check().unwrap(),
                    crate::equiv::solver_interface::Response::Sat
                );
                solver.pop().unwrap();
            }

            #[test]
            fn test_declare_fresh_differs_from_declare() {
                let mut solver = $solver;
                let d = solver.declare("a", 8).unwrap();
                let f = solver.declare_fresh("a", 8).unwrap();
                solver.push().unwrap();
                let ne = solver.ne(&d, &f);
                solver.assert(&ne).unwrap();
                assert_eq!(
                    solver.check().unwrap(),
                    crate::equiv::solver_interface::Response::Sat
                );
                solver.pop().unwrap();
            }

            crate::test_solver_unary!(test_not, $solver, not, 8, 0x89, 8, 0x76);
            crate::test_solver_unary!(test_neg, $solver, neg, 8, 0x05, 8, 0xFB);

            // Bit-reverse (16-bit example)
            crate::test_solver_unary!(test_reverse, $solver, reverse, 16, 0x1234, 16, 0x2c48);

            // OR-reduce, AND-reduce, XOR-reduce examples
            crate::test_solver_unary!(test_or_reduce_some, $solver, or_reduce, 4, 0b1010, 1, 1);
            crate::test_solver_unary!(
                test_or_reduce_all_zeros,
                $solver,
                or_reduce,
                4,
                0b0000,
                1,
                0
            );
            crate::test_solver_unary!(test_and_reduce_some, $solver, and_reduce, 4, 0b1010, 1, 0);
            crate::test_solver_unary!(
                test_and_reduce_all_ones,
                $solver,
                and_reduce,
                4,
                0b1111,
                1,
                1
            );
            crate::test_solver_unary!(
                test_xor_reduce_odd_ones,
                $solver,
                xor_reduce,
                4,
                0b1011,
                1,
                1
            );
            crate::test_solver_unary!(
                test_xor_reduce_even_ones,
                $solver,
                xor_reduce,
                4,
                0b1010,
                1,
                0
            );

            crate::test_solver_binary!(test_add, $solver, add, 8, 0x15, 0x93, 8, 0xa8);
            crate::test_solver_binary!(test_add_overflow, $solver, add, 8, 0x75, 0x9c, 8, 0x11);
            crate::test_solver_binary!(test_sub, $solver, sub, 8, 0x93, 0x15, 8, 0x7e);
            crate::test_solver_binary!(test_sub_overflow, $solver, sub, 8, 0x15, 0x93, 8, 0x82);
            crate::test_solver_binary!(test_mul, $solver, mul, 8, 0x15, 0x03, 8, 0x3f);
            crate::test_solver_binary!(test_mul_overflow, $solver, mul, 8, 0x53, 0x42, 8, 0x66);
            crate::test_solver_binary!(test_udiv, $solver, udiv, 8, 0x05, 0x03, 8, 0x01);
            crate::test_solver_binary!(test_urem, $solver, urem, 8, 0x05, 0x03, 8, 0x02);
            crate::test_solver_binary!(test_srem, $solver, srem, 8, 0x05, 0x03, 8, 0x02);
            crate::test_solver_binary!(test_sdiv, $solver, sdiv, 8, 0x05, 0x03, 8, 0x01);

            // -----------------------------
            // Division / remainder edge-case tests (SMT-LIB semantics)
            // -----------------------------

            // Unsigned division/rem: divisor = 0 → udiv = all 1s, urem = dividend
            crate::test_solver_binary!(test_udiv_by_zero, $solver, udiv, 8, 0x05, 0x00, 8, 0xFF);
            crate::test_solver_binary!(test_urem_by_zero, $solver, urem, 8, 0x05, 0x00, 8, 0x05);
            crate::test_solver_binary!(
                test_udiv_zero_by_zero,
                $solver,
                udiv,
                8,
                0x00,
                0x00,
                8,
                0xFF
            );
            crate::test_solver_binary!(
                test_urem_zero_by_zero,
                $solver,
                urem,
                8,
                0x00,
                0x00,
                8,
                0x00
            );

            // Signed division/rem: divisor = 0 → sdiv = all 1s or 1, srem = dividend
            // (SMT-LIB spec)
            crate::test_solver_binary!(
                test_sdiv_pos_by_zero,
                $solver,
                sdiv,
                8,
                0x05,
                0x00,
                8,
                0xFF
            );
            crate::test_solver_binary!(
                test_srem_pos_by_zero,
                $solver,
                srem,
                8,
                0x05,
                0x00,
                8,
                0x05
            );
            crate::test_solver_binary!(
                test_sdiv_zero_by_zero,
                $solver,
                sdiv,
                8,
                0x00,
                0x00,
                8,
                0xFF
            );
            crate::test_solver_binary!(
                test_srem_zero_by_zero,
                $solver,
                srem,
                8,
                0x00,
                0x00,
                8,
                0x00
            );
            crate::test_solver_binary!(
                test_sdiv_neg_by_zero,
                $solver,
                sdiv,
                8,
                0xFB,
                0x00,
                8,
                0x01
            );
            crate::test_solver_binary!(
                test_srem_neg_by_zero,
                $solver,
                srem,
                8,
                0xFB,
                0x00,
                8,
                0xFB
            );

            // Signed INT_MIN / -1 → result = INT_MIN, remainder = 0
            crate::test_solver_binary!(
                test_sdiv_intmin_neg_one,
                $solver,
                sdiv,
                8,
                0x80,
                0xFF,
                8,
                0x80
            );
            crate::test_solver_binary!(
                test_srem_intmin_neg_one,
                $solver,
                srem,
                8,
                0x80,
                0xFF,
                8,
                0x00
            );

            // Various sign combinations for signed division/remainder
            // -5 / 3  → -1, rem = -2
            crate::test_solver_binary!(test_sdiv_neg_pos, $solver, sdiv, 8, 0xFB, 0x03, 8, 0xFF);
            crate::test_solver_binary!(test_srem_neg_pos, $solver, srem, 8, 0xFB, 0x03, 8, 0xFE);
            // 5 / -3  → -1, rem = 2
            crate::test_solver_binary!(test_sdiv_pos_neg, $solver, sdiv, 8, 0x05, 0xFD, 8, 0xFF);
            crate::test_solver_binary!(test_srem_pos_neg, $solver, srem, 8, 0x05, 0xFD, 8, 0x02);
            // -5 / -3 → 1,  rem = -2
            crate::test_solver_binary!(test_sdiv_neg_neg, $solver, sdiv, 8, 0xFB, 0xFD, 8, 0x01);
            crate::test_solver_binary!(test_srem_neg_neg, $solver, srem, 8, 0xFB, 0xFD, 8, 0xFE);

            // Shift operations -------------------------------------------------------
            crate::test_solver_binary!(test_shl_1, $solver, shl, 8, 0x96, 0x01, 8, 0x2C);
            crate::test_solver_binary!(test_lshr_1, $solver, lshr, 8, 0x96, 0x01, 8, 0x4B);
            crate::test_solver_binary!(test_ashr_1, $solver, ashr, 8, 0x96, 0x01, 8, 0xCB);

            crate::test_solver_binary!(test_shl_by_zero, $solver, shl, 8, 0xAA, 0x00, 8, 0xAA);
            crate::test_solver_binary!(test_lshr_by_zero, $solver, lshr, 8, 0xAA, 0x00, 8, 0xAA);
            crate::test_solver_binary!(test_ashr_by_zero, $solver, ashr, 8, 0xAA, 0x00, 8, 0xAA);

            // Shift by width (8) should zero-fill for shl/lshr; ashr depends on sign
            crate::test_solver_binary!(test_shl_by_width, $solver, shl, 8, 0xAA, 0x08, 8, 0x00);
            crate::test_solver_binary!(test_lshr_by_width, $solver, lshr, 8, 0xAA, 0x08, 8, 0x00);
            crate::test_solver_binary!(
                test_ashr_pos_by_width,
                $solver,
                ashr,
                8,
                0x21,
                0x08,
                8,
                0x00
            );
            crate::test_solver_binary!(
                test_ashr_neg_by_width,
                $solver,
                ashr,
                8,
                0x80,
                0x08,
                8,
                0xFF
            );

            crate::test_solver_binary!(test_or, $solver, or, 4, 0b1010, 0b1100, 4, 0b1110);
            crate::test_solver_binary!(test_and, $solver, and, 4, 0b1010, 0b1100, 4, 0b1000);
            crate::test_solver_binary!(test_xor, $solver, xor, 4, 0b1010, 0b1100, 4, 0b0110);
            crate::test_solver_binary!(test_nor, $solver, nor, 4, 0b1010, 0b1100, 4, 0b0001);
            crate::test_solver_binary!(test_nand, $solver, nand, 4, 0b1010, 0b1100, 4, 0b0111);

            crate::test_solver_binary!(test_umin, $solver, umin, 4, 0b1010, 0b1100, 4, 0b1010);
            crate::test_solver_binary!(test_umax, $solver, umax, 4, 0b1010, 0b1100, 4, 0b1100);
            crate::test_solver_binary!(test_smax_pos, $solver, smax, 4, 0b0010, 0b0100, 4, 0b0100);
            crate::test_solver_binary!(test_smin_pos, $solver, smin, 4, 0b0010, 0b0100, 4, 0b0010);
            crate::test_solver_binary!(test_smax_neg, $solver, smax, 4, 0b1110, 0b1000, 4, 0b1110);
            crate::test_solver_binary!(test_smin_neg, $solver, smin, 4, 0b1110, 0b1000, 4, 0b1000);
            crate::test_solver_binary!(
                test_smax_mixed,
                $solver,
                smax,
                4,
                0b0110,
                0b1000,
                4,
                0b0110
            );
            crate::test_solver_binary!(
                test_smin_mixed,
                $solver,
                smin,
                4,
                0b0110,
                0b1000,
                4,
                0b1000
            );

            crate::test_solver_extend!(
                test_zero_extend_zero,
                $solver,
                zero_extend,
                4,
                0b1010,
                0,
                4,
                0xA
            );
            crate::test_solver_extend!(
                test_sign_extend_zero,
                $solver,
                sign_extend,
                4,
                0b1010,
                0,
                4,
                0xA
            );
            crate::test_solver_extend!(
                test_zero_extend,
                $solver,
                zero_extend,
                4,
                0b1010,
                4,
                8,
                0x0A
            );
            crate::test_solver_extend!(
                test_sign_extend_positive,
                $solver,
                sign_extend,
                4,
                0b0010,
                4,
                8,
                0x02
            );
            crate::test_solver_extend!(
                test_sign_extend_negative,
                $solver,
                sign_extend,
                4,
                0b1010,
                4,
                8,
                0xFA
            );

            crate::test_solver_extend!(
                test_zero_extend_to,
                $solver,
                zero_extend_to,
                4,
                0b1010,
                8,
                8,
                0x0A
            );
            crate::test_solver_extend!(
                test_sign_extend_to_positive,
                $solver,
                sign_extend_to,
                4,
                0b0010,
                8,
                8,
                0x02
            );
            crate::test_solver_extend!(
                test_sign_extend_to_negative,
                $solver,
                sign_extend_to,
                4,
                0b1010,
                8,
                8,
                0xFA
            );
            crate::test_solver_ite!(test_ite_true, $solver, 1, 0x55, 0xAA, 0x55);
            crate::test_solver_ite!(test_ite_false, $solver, 0, 0x55, 0xAA, 0xAA);

            // eq / ne
            crate::test_solver_cmp!(test_eq_true, $solver, eq, 8, 0x5A, 0x5A, 1);
            crate::test_solver_cmp!(test_eq_false, $solver, eq, 8, 0x5A, 0x5B, 0);
            crate::test_solver_cmp!(test_ne_true, $solver, ne, 8, 0x5A, 0x5B, 1);
            crate::test_solver_cmp!(test_ne_false, $solver, ne, 8, 0x5A, 0x5A, 0);

            // Signed comparisons: -3 (0xFD) vs 2 (0x02)
            crate::test_solver_cmp!(test_slt_true, $solver, slt, 8, 0xFD, 0x02, 1);
            crate::test_solver_cmp!(test_slt_false_eq, $solver, slt, 8, 0x02, 0x02, 0);
            crate::test_solver_cmp!(test_slt_false_lt, $solver, slt, 8, 0x02, 0xFD, 0);
            crate::test_solver_cmp!(test_sgt_true, $solver, sgt, 8, 0x02, 0xFD, 1);
            crate::test_solver_cmp!(test_sgt_false_eq, $solver, sgt, 8, 0x02, 0x02, 0);
            crate::test_solver_cmp!(test_sgt_false_lt, $solver, sgt, 8, 0xFD, 0x02, 0);
            crate::test_solver_cmp!(test_sle_true_eq, $solver, sle, 8, 0x02, 0x02, 1);
            crate::test_solver_cmp!(test_sle_true_lt, $solver, sle, 8, 0xFD, 0x02, 1);
            crate::test_solver_cmp!(test_sle_false, $solver, sle, 8, 0x02, 0xFD, 0);
            crate::test_solver_cmp!(test_sge_true_eq, $solver, sge, 8, 0x02, 0x02, 1);
            crate::test_solver_cmp!(test_sge_true_gt, $solver, sge, 8, 0x02, 0xFD, 1);
            crate::test_solver_cmp!(test_sge_false, $solver, sge, 8, 0xFD, 0x02, 0);

            // Unsigned comparisons: 0x02 vs 0xFD (2 vs 253)
            crate::test_solver_cmp!(test_ult_true, $solver, ult, 8, 0x02, 0xFD, 1);
            crate::test_solver_cmp!(test_ult_false_eq, $solver, ult, 8, 0x02, 0x02, 0);
            crate::test_solver_cmp!(test_ult_false_gt, $solver, ult, 8, 0xFD, 0x02, 0);
            crate::test_solver_cmp!(test_ugt_true, $solver, ugt, 8, 0xFD, 0x02, 1);
            crate::test_solver_cmp!(test_ugt_false_eq, $solver, ugt, 8, 0xAB, 0xAB, 0);
            crate::test_solver_cmp!(test_ugt_false_gt, $solver, ugt, 8, 0x02, 0xFD, 0);
            crate::test_solver_cmp!(test_ule_true_eq, $solver, ule, 8, 0x02, 0x02, 1);
            crate::test_solver_cmp!(test_ule_true_lt, $solver, ule, 8, 0x02, 0xFD, 1);
            crate::test_solver_cmp!(test_ule_false, $solver, ule, 8, 0xFD, 0x02, 0);
            crate::test_solver_cmp!(test_uge_true_eq, $solver, uge, 8, 0xFD, 0xFD, 1);
            crate::test_solver_cmp!(test_uge_true_gt, $solver, uge, 8, 0xFD, 0x02, 1);
            crate::test_solver_cmp!(test_uge_false, $solver, uge, 8, 0x02, 0xFD, 0);

            // Zero-width comparisons
            crate::test_solver_cmp_zero_width!(test_eq_zero_width, $solver, eq, Solver::true_bv);
            crate::test_solver_cmp_zero_width!(test_sle_zero_width, $solver, sle, Solver::true_bv);
            crate::test_solver_cmp_zero_width!(test_sge_zero_width, $solver, sge, Solver::true_bv);
            crate::test_solver_cmp_zero_width!(test_ule_zero_width, $solver, ule, Solver::true_bv);
            crate::test_solver_cmp_zero_width!(test_uge_zero_width, $solver, uge, Solver::true_bv);
            crate::test_solver_cmp_zero_width!(test_ne_zero_width, $solver, ne, Solver::false_bv);
            crate::test_solver_cmp_zero_width!(test_slt_zero_width, $solver, slt, Solver::false_bv);
            crate::test_solver_cmp_zero_width!(test_sgt_zero_width, $solver, sgt, Solver::false_bv);
            crate::test_solver_cmp_zero_width!(test_ult_zero_width, $solver, ult, Solver::false_bv);
            crate::test_solver_cmp_zero_width!(test_ugt_zero_width, $solver, ugt, Solver::false_bv);

            // xls_sdiv
            crate::test_solver_binary!(
                test_xls_sdiv_signed_pos_pos,
                $solver,
                xls_sdiv,
                8,
                0x5A,
                0x07,
                8,
                0x0C
            );
            crate::test_solver_binary!(
                test_xls_sdiv_signed_neg_pos,
                $solver,
                xls_sdiv,
                8,
                0xBC,
                0x07,
                8,
                0xF7
            );
            crate::test_solver_binary!(
                test_xls_sdiv_signed_pos_neg,
                $solver,
                xls_sdiv,
                8,
                0x5A,
                0xF9,
                8,
                0xF4
            );
            crate::test_solver_binary!(
                test_xls_sdiv_signed_neg_neg,
                $solver,
                xls_sdiv,
                8,
                0xBC,
                0xF9,
                8,
                0x09
            );
            crate::test_solver_binary!(
                test_xls_sdiv_signed_pos_zero,
                $solver,
                xls_sdiv,
                8,
                0x5A,
                0x00,
                8,
                0x7F
            );
            crate::test_solver_binary!(
                test_xls_sdiv_signed_neg_zero,
                $solver,
                xls_sdiv,
                8,
                0xBC,
                0x00,
                8,
                0x80
            );
            crate::test_solver_binary!(
                test_xls_sdiv_signed_zero_zero,
                $solver,
                xls_sdiv,
                8,
                0x00,
                0x00,
                8,
                0x7F
            );
            // xls_udiv
            crate::test_solver_binary!(
                test_xls_udiv_non_zero,
                $solver,
                xls_udiv,
                8,
                0xBC,
                0x07,
                8,
                0x1A
            );
            crate::test_solver_binary!(
                test_xls_udiv_zero,
                $solver,
                xls_udiv,
                8,
                0xBC,
                0x00,
                8,
                0xFF
            );
            // xls_smod
            crate::test_solver_binary!(
                test_xls_smod_signed_pos_pos,
                $solver,
                xls_smod,
                8,
                0x5A,
                0x07,
                8,
                0x06
            );
            crate::test_solver_binary!(
                test_xls_smod_signed_neg_pos,
                $solver,
                xls_smod,
                8,
                0xBC,
                0x07,
                8,
                0xFB
            );
            crate::test_solver_binary!(
                test_xls_smod_signed_pos_neg,
                $solver,
                xls_smod,
                8,
                0x5A,
                0xF9,
                8,
                0x6
            );
            crate::test_solver_binary!(
                test_xls_smod_signed_neg_neg,
                $solver,
                xls_smod,
                8,
                0xBC,
                0xF9,
                8,
                0xFB
            );
            crate::test_solver_binary!(
                test_xls_smod_signed_pos_zero,
                $solver,
                xls_smod,
                8,
                0x5A,
                0x00,
                8,
                0x00
            );
            crate::test_solver_binary!(
                test_xls_smod_signed_neg_zero,
                $solver,
                xls_smod,
                8,
                0xBC,
                0x00,
                8,
                0x00
            );
            crate::test_solver_binary!(
                test_xls_smod_signed_zero_zero,
                $solver,
                xls_smod,
                8,
                0x00,
                0x00,
                8,
                0x00
            );
            // xls_umod
            crate::test_solver_binary!(
                test_xls_umod_non_zero,
                $solver,
                xls_umod,
                8,
                0xBC,
                0x07,
                8,
                0x06
            );
            crate::test_solver_binary!(
                test_xls_umod_zero,
                $solver,
                xls_umod,
                8,
                0xBC,
                0x00,
                8,
                0x00
            );

            // xls_arbitrary_width_mul
            crate::test_solver_binary_arbitrary_width_mul!(
                test_xls_arbitrary_width_smul,
                $solver,
                xls_arbitrary_width_smul,
                4,
                0x7,
                8,
                0xBA,
                12,
                0xE16
            );
            crate::test_solver_binary_arbitrary_width_mul!(
                test_xls_arbitrary_width_umul,
                $solver,
                xls_arbitrary_width_umul,
                4,
                0x7,
                8,
                0xBA,
                12,
                0x516
            );

            // xls_shll
            crate::test_solver_binary_arbitrary_width!(
                test_xls_shll_inbound,
                $solver,
                xls_shll,
                16,
                0x9abc,
                8,
                4,
                16,
                0xabc0
            );
            crate::test_solver_binary_arbitrary_width!(
                test_xls_shll_exact_width,
                $solver,
                xls_shll,
                16,
                0x9abf,
                8,
                16,
                16,
                0x0000
            );

            crate::test_solver_binary_arbitrary_width!(
                test_xls_shll_outbound,
                $solver,
                xls_shll,
                16,
                0x9abf,
                8,
                17,
                16,
                0x0000
            );

            crate::test_solver_binary_arbitrary_width!(
                test_xls_shll_small_shift_bit_width,
                $solver,
                xls_shll,
                16,
                0x9abf,
                4,
                4,
                16,
                0xabf0
            );

            // xls_shrl
            crate::test_solver_binary_arbitrary_width!(
                test_xls_shrl_inbound,
                $solver,
                xls_shrl,
                16,
                0x9abc,
                8,
                4,
                16,
                0x09ab
            );
            crate::test_solver_binary_arbitrary_width!(
                test_xls_shrl_exact_width,
                $solver,
                xls_shrl,
                16,
                0x9abf,
                8,
                16,
                16,
                0x0000
            );

            crate::test_solver_binary_arbitrary_width!(
                test_xls_shrl_outbound,
                $solver,
                xls_shrl,
                16,
                0x9abf,
                8,
                17,
                16,
                0x0000
            );

            crate::test_solver_binary_arbitrary_width!(
                test_xls_shrl_small_shift_bit_width,
                $solver,
                xls_shrl,
                16,
                0x9abf,
                4,
                4,
                16,
                0x09ab
            );
            // xls_shra
            crate::test_solver_binary_arbitrary_width!(
                test_xls_shra_negative_inbound,
                $solver,
                xls_shra,
                16,
                0x9abc,
                8,
                4,
                16,
                0xf9ab
            );
            crate::test_solver_binary_arbitrary_width!(
                test_xls_shra_negative_exact_width,
                $solver,
                xls_shra,
                16,
                0x9abf,
                8,
                16,
                16,
                0xffff
            );

            crate::test_solver_binary_arbitrary_width!(
                test_xls_shra_negative_outbound,
                $solver,
                xls_shra,
                16,
                0x9abf,
                8,
                17,
                16,
                0xffff
            );

            crate::test_solver_binary_arbitrary_width!(
                test_xls_shra_negative_small_shift_bit_width,
                $solver,
                xls_shra,
                16,
                0x9abf,
                4,
                4,
                16,
                0xf9ab
            );

            crate::test_solver_binary_arbitrary_width!(
                test_xls_shra_positive_inbound,
                $solver,
                xls_shra,
                16,
                0x7abf,
                8,
                4,
                16,
                0x07ab
            );
            crate::test_solver_binary_arbitrary_width!(
                test_xls_shra_positive_exact_width,
                $solver,
                xls_shra,
                16,
                0x7abf,
                8,
                16,
                16,
                0x0000
            );

            crate::test_solver_binary_arbitrary_width!(
                test_xls_shra_positive_outbound,
                $solver,
                xls_shra,
                16,
                0x7abf,
                8,
                17,
                16,
                0x0000
            );

            crate::test_solver_binary_arbitrary_width!(
                test_xls_shra_positive_small_shift_bit_width,
                $solver,
                xls_shra,
                16,
                0x7abf,
                4,
                4,
                16,
                0x07ab
            );

            // ----- *_many helpers -----
            crate::test_solver_many!(
                test_concat_many,
                $solver,
                concat_many,
                4,
                [0x1, 0x2, 0x3],
                12,
                0x123
            );
            crate::test_solver_many!(
                test_and_many,
                $solver,
                and_many,
                8,
                [0x0F, 0x33, 0x55],
                8,
                0x01
            );
            crate::test_solver_many!(
                test_or_many,
                $solver,
                or_many,
                8,
                [0x0F, 0x33, 0x55],
                8,
                0x7F
            );
            crate::test_solver_many!(
                test_xor_many,
                $solver,
                xor_many,
                8,
                [0x0F, 0x33, 0x55],
                8,
                0x69
            );
            crate::test_solver_many!(
                test_nor_many,
                $solver,
                nor_many,
                8,
                [0x0F, 0x33, 0x55],
                8,
                0x80
            );
            crate::test_solver_many!(
                test_nand_many,
                $solver,
                nand_many,
                8,
                [0x0F, 0x33, 0x55],
                8,
                0xFE
            );

            crate::test_solver_unary!(
                test_xls_one_hot_lsb,
                $solver,
                xls_one_hot_lsb_prio,
                16,
                0x0560,
                17,
                0x0020
            );
            crate::test_solver_unary!(
                test_xls_one_hot_lsb_lsb_set,
                $solver,
                xls_one_hot_lsb_prio,
                16,
                0x0001,
                17,
                0x0001
            );
            crate::test_solver_unary!(
                test_xls_one_hot_lsb_msb_set,
                $solver,
                xls_one_hot_lsb_prio,
                16,
                0x8000,
                17,
                0x8000
            );
            crate::test_solver_unary!(
                test_xls_one_hot_lsb_zero,
                $solver,
                xls_one_hot_lsb_prio,
                16,
                0x0000,
                17,
                0x10000
            );
            crate::test_solver_unary!(
                test_xls_one_hot_lsb_all_set,
                $solver,
                xls_one_hot_lsb_prio,
                16,
                0xFFFF,
                17,
                0x0001
            );

            crate::test_solver_unary!(
                test_xls_one_hot_msb,
                $solver,
                xls_one_hot_msb_prio,
                16,
                0x0560,
                17,
                0x0400
            );
            crate::test_solver_unary!(
                test_xls_one_hot_msb_lsb_set,
                $solver,
                xls_one_hot_msb_prio,
                16,
                0x0001,
                17,
                0x0001
            );
            crate::test_solver_unary!(
                test_xls_one_hot_msb_msb_set,
                $solver,
                xls_one_hot_msb_prio,
                16,
                0x8000,
                17,
                0x8000
            );
            crate::test_solver_unary!(
                test_xls_one_hot_msb_zero,
                $solver,
                xls_one_hot_msb_prio,
                16,
                0x0000,
                17,
                0x10000
            );
            crate::test_solver_unary!(
                test_xls_one_hot_msb_all_set,
                $solver,
                xls_one_hot_msb_prio,
                16,
                0xFFFF,
                17,
                0x8000
            );

            // ------------------------------------------------------------------
            // xls_decode tests
            // ------------------------------------------------------------------
            crate::test_solver_decode!(test_xls_decode_basic, $solver, 8, 0x05, 16, 0x0020);
            crate::test_solver_decode!(test_xls_decode_zero, $solver, 8, 0x00, 16, 0x0001);
            crate::test_solver_decode!(test_xls_decode_overflow, $solver, 8, 0x0A, 8, 0x00);

            // ------------------------------------------------------------------
            // xls_encode tests
            // ------------------------------------------------------------------
            crate::test_solver_encode!(test_xls_encode_single_hot, $solver, 8, 0x20, 3, 0x05);
            crate::test_solver_encode!(test_xls_encode_multiple_hot, $solver, 16, 0x0028, 4, 0x07);
            crate::test_solver_encode!(test_xls_encode_zero, $solver, 16, 0x0000, 4, 0x00);

            use crate::xls_ir::ir;
            use xlsynth::IrValue;

            /// Tests that we can handle single-bit values converting to/from the solver
            /// as this have a different syntax with Boolector vs. other solvers.
            #[test]
            fn get_value_1_bit() {
                let mut solver = $solver;
                let a = solver.declare("a", 1).unwrap();
                let b1 = solver.numerical(1, 1);
                let b0 = solver.numerical(1, 0);
                let eq1 = solver.eq(&a, &b1);
                let eq0 = solver.eq(&a, &b0);
                solver.push().unwrap();
                solver.assert(&eq1).unwrap();
                solver.check().unwrap();
                let a_value = solver.get_value(&a, &ir::Type::Bits(1)).unwrap();
                assert_eq!(a_value, IrValue::make_ubits(1, 1).unwrap());
                solver.pop().unwrap();
                solver.push().unwrap();
                solver.assert(&eq0).unwrap();
                solver.check().unwrap();
                let a_value = solver.get_value(&a, &ir::Type::Bits(1)).unwrap();
                assert_eq!(a_value, IrValue::make_ubits(1, 0).unwrap());
                solver.pop().unwrap();
            }
        }
    };
}

// -------------------------------------------------------------------
// Decode / Encode test macros
// -------------------------------------------------------------------
#[macro_export]
macro_rules! test_solver_decode {
    ($fn_name:ident,
     $solver:expr,
     $in_w:expr, $in_val:expr,
     $out_w:expr, $out_val:expr $(,)?) => {
        #[test]
        fn $fn_name() {
            let mut solver = $solver;
            let operand = solver.numerical($in_w, $in_val);
            let actual = solver.xls_decode(&operand, $out_w);
            let expected = solver.numerical($out_w, $out_val);
            crate::equiv::solver_interface::test_utils::assert_solver_eq(
                &mut solver,
                &actual,
                &expected,
            );
        }
    };
}

#[macro_export]
macro_rules! test_solver_encode {
    ($fn_name:ident,
     $solver:expr,
     $in_w:expr, $in_val:expr,
     $out_w:expr, $out_val:expr $(,)?) => {
        #[test]
        fn $fn_name() {
            let mut solver = $solver;
            let operand = solver.numerical($in_w, $in_val);
            let actual = solver.xls_encode(&operand);
            let expected = solver.numerical($out_w, $out_val);
            crate::equiv::solver_interface::test_utils::assert_solver_eq(
                &mut solver,
                &actual,
                &expected,
            );
        }
    };
}
