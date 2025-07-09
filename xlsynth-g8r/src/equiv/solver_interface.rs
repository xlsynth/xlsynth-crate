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

pub trait Solver: Sized {
    type Rep: Clone;
    type Config: Send + Sync;
    fn new(config: &Self::Config) -> io::Result<Self>;
    fn declare(&mut self, name: &str, width: usize) -> io::Result<BitVec<Self::Rep>>;
    fn numerical(&mut self, width: usize, value: u64) -> BitVec<Self::Rep>;
    fn true_bv(&mut self) -> BitVec<Self::Rep> {
        self.numerical(1, 1)
    }
    fn false_bv(&mut self) -> BitVec<Self::Rep> {
        self.numerical(1, 0)
    }
    fn from_raw_str(&mut self, width: usize, value: &str) -> BitVec<Self::Rep>;
    fn zero_width(&mut self) -> BitVec<Self::Rep> {
        BitVec::ZeroWidth
    }
    fn get_value(&mut self, bit_vec: &BitVec<Self::Rep>, ty: &ir::Type) -> io::Result<IrValue>;
    fn extract(&mut self, bit_vec: &BitVec<Self::Rep>, high: i32, low: i32) -> BitVec<Self::Rep>;
    fn slice(
        &mut self,
        bit_vec: &BitVec<Self::Rep>,
        start: usize,
        width: usize,
    ) -> BitVec<Self::Rep> {
        self.extract(bit_vec, (start + width - 1) as i32, start as i32)
    }
    fn not(&mut self, bit_vec: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn neg(&mut self, bit_vec: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn reverse(&mut self, bit_vec: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn or_reduce(&mut self, bit_vec: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn and_reduce(&mut self, bit_vec: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn xor_reduce(&mut self, bit_vec: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn add(&mut self, lhs: &BitVec<Self::Rep>, rhs: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn sub(&mut self, lhs: &BitVec<Self::Rep>, rhs: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn mul(&mut self, lhs: &BitVec<Self::Rep>, rhs: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn udiv(&mut self, lhs: &BitVec<Self::Rep>, rhs: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn urem(&mut self, lhs: &BitVec<Self::Rep>, rhs: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn srem(&mut self, lhs: &BitVec<Self::Rep>, rhs: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn sdiv(&mut self, lhs: &BitVec<Self::Rep>, rhs: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn shl(&mut self, lhs: &BitVec<Self::Rep>, rhs: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn lshr(&mut self, lhs: &BitVec<Self::Rep>, rhs: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn ashr(&mut self, lhs: &BitVec<Self::Rep>, rhs: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn concat(&mut self, lhs: &BitVec<Self::Rep>, rhs: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn or(&mut self, lhs: &BitVec<Self::Rep>, rhs: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn and(&mut self, lhs: &BitVec<Self::Rep>, rhs: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn xor(&mut self, lhs: &BitVec<Self::Rep>, rhs: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn nor(&mut self, lhs: &BitVec<Self::Rep>, rhs: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn nand(&mut self, lhs: &BitVec<Self::Rep>, rhs: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn extend(
        &mut self,
        bit_vec: &BitVec<Self::Rep>,
        extend_width: usize,
        signed: bool,
    ) -> BitVec<Self::Rep>;
    fn extend_to(
        &mut self,
        bit_vec: &BitVec<Self::Rep>,
        new_width: usize,
        signed: bool,
    ) -> BitVec<Self::Rep> {
        let width = bit_vec.get_width();
        assert!(new_width >= width, "Cannot extend to smaller width");
        self.extend(bit_vec, new_width - width, signed)
    }
    fn zero_extend(
        &mut self,
        bit_vec: &BitVec<Self::Rep>,
        extend_width: usize,
    ) -> BitVec<Self::Rep> {
        self.extend(bit_vec, extend_width, false)
    }
    fn sign_extend(
        &mut self,
        bit_vec: &BitVec<Self::Rep>,
        extend_width: usize,
    ) -> BitVec<Self::Rep> {
        self.extend(bit_vec, extend_width, true)
    }
    fn zero_extend_to(
        &mut self,
        bit_vec: &BitVec<Self::Rep>,
        new_width: usize,
    ) -> BitVec<Self::Rep> {
        self.extend_to(bit_vec, new_width, false)
    }
    fn sign_extend_to(
        &mut self,
        bit_vec: &BitVec<Self::Rep>,
        new_width: usize,
    ) -> BitVec<Self::Rep> {
        self.extend_to(bit_vec, new_width, true)
    }
    fn ite(
        &mut self,
        lhs: &BitVec<Self::Rep>,
        then: &BitVec<Self::Rep>,
        else_: &BitVec<Self::Rep>,
    ) -> BitVec<Self::Rep>;
    fn eq(&mut self, lhs: &BitVec<Self::Rep>, rhs: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn ne(&mut self, lhs: &BitVec<Self::Rep>, rhs: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn slt(&mut self, lhs: &BitVec<Self::Rep>, rhs: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn sgt(&mut self, lhs: &BitVec<Self::Rep>, rhs: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn sle(&mut self, lhs: &BitVec<Self::Rep>, rhs: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn sge(&mut self, lhs: &BitVec<Self::Rep>, rhs: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn ult(&mut self, lhs: &BitVec<Self::Rep>, rhs: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn ugt(&mut self, lhs: &BitVec<Self::Rep>, rhs: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn ule(&mut self, lhs: &BitVec<Self::Rep>, rhs: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn uge(&mut self, lhs: &BitVec<Self::Rep>, rhs: &BitVec<Self::Rep>) -> BitVec<Self::Rep>;
    fn push(&mut self) -> io::Result<()>;
    fn pop(&mut self) -> io::Result<()>;
    fn check(&mut self) -> io::Result<Response>;
    fn assert(&mut self, bit_vec: &BitVec<Self::Rep>) -> io::Result<()>;
    fn render(&mut self, bit_vec: &BitVec<Self::Rep>) -> String;
    fn smax(&mut self, lhs: &BitVec<Self::Rep>, rhs: &BitVec<Self::Rep>) -> BitVec<Self::Rep> {
        let cond = self.slt(lhs, rhs);
        self.ite(&cond, rhs, lhs)
    }
    fn umax(&mut self, lhs: &BitVec<Self::Rep>, rhs: &BitVec<Self::Rep>) -> BitVec<Self::Rep> {
        let cond = self.ult(lhs, rhs);
        self.ite(&cond, rhs, lhs)
    }
    fn smin(&mut self, lhs: &BitVec<Self::Rep>, rhs: &BitVec<Self::Rep>) -> BitVec<Self::Rep> {
        let cond = self.slt(lhs, rhs);
        self.ite(&cond, lhs, rhs)
    }
    fn umin(&mut self, lhs: &BitVec<Self::Rep>, rhs: &BitVec<Self::Rep>) -> BitVec<Self::Rep> {
        let cond = self.ult(lhs, rhs);
        self.ite(&cond, lhs, rhs)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BitVec<Rep> {
    BitVec { width: usize, rep: Rep },
    ZeroWidth,
}

impl<Rep> BitVec<Rep> {
    pub fn get_width(&self) -> usize {
        match self {
            BitVec::BitVec { width, .. } => *width,
            BitVec::ZeroWidth => 0,
        }
    }

    pub fn get_rep(&self) -> Option<&Rep> {
        match self {
            BitVec::BitVec { rep, .. } => Some(rep),
            BitVec::ZeroWidth => None,
        }
    }
}

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

    pub fn assert_solver_eq<S: Solver>(solver: &mut S, lhs: &BitVec<S::Rep>, rhs: &BitVec<S::Rep>) {
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
    macro_rules! test_solver_binary {
        ($fn_name:ident,
         $solver:expr,
         $op:ident,
         $in_w:expr,  $in_val1:expr, $in_val2:expr,
         $out_w:expr, $out_val:expr $(,)?
        ) => {
            #[test]
            fn $fn_name() {
                // Arrange
                let mut solver = $solver;
                let operand1 = solver.numerical($in_w, $in_val1);
                let operand2 = solver.numerical($in_w, $in_val2);

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
}

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
            fn test_extract() {
                let mut solver = $solver;
                test_utils::test_extract(&mut solver);
            }

            #[test]
            fn test_slice() {
                let mut solver = $solver;
                test_utils::test_slice(&mut solver);
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
        }
    };
}
