// SPDX-License-Identifier: Apache-2.0

use crate::ir_package::IrPackage;
use crate::ir_package::IrPackagePtr;
use crate::ir_package::IrType;
use crate::lib_support;
use crate::lib_support::{BValuePtr, IrFnBuilderPtr};
use crate::IrFunction;
use crate::IrValue;
use crate::XlsynthError;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::RwLockReadGuard;
use std::sync::RwLockWriteGuard;

#[derive(Clone)]
pub struct BValue {
    ptr: Arc<RwLock<BValuePtr>>,
}

pub struct FnBuilder {
    fn_builder: Arc<RwLock<IrFnBuilderPtr>>,
    package: Arc<RwLock<IrPackagePtr>>,
}

impl FnBuilder {
    pub fn new(package: &mut IrPackage, name: &str, should_verify: bool) -> Self {
        let package_guard = package.ptr.write().unwrap();
        Self {
            fn_builder: lib_support::xls_function_builder_new(
                package_guard.mut_c_ptr(),
                name,
                should_verify,
            ),
            package: package.ptr.clone(),
        }
    }

    pub fn build_with_return_value(
        &self,
        return_value: &BValue,
    ) -> Result<IrFunction, XlsynthError> {
        let package_guard: RwLockWriteGuard<IrPackagePtr> = self.package.write().unwrap();
        let builder_guard: RwLockWriteGuard<IrFnBuilderPtr> = self.fn_builder.write().unwrap();
        let return_value_guard: RwLockReadGuard<BValuePtr> = return_value.ptr.read().unwrap();
        lib_support::xls_function_builder_build_with_return_value(
            &self.package,
            package_guard,
            builder_guard,
            return_value_guard,
        )
    }

    pub fn param(&mut self, name: &str, type_: &IrType) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let bvalue_ptr =
            lib_support::xls_function_builder_add_parameter(fn_builder_guard, name, type_);
        BValue { ptr: bvalue_ptr }
    }

    pub fn literal(&mut self, value: &IrValue, name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let bvalue_ptr =
            lib_support::xls_function_builder_add_literal(fn_builder_guard, value, name);
        BValue { ptr: bvalue_ptr }
    }

    pub fn add(&mut self, a: &BValue, b: &BValue, name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_add(
            fn_builder_guard,
            a.ptr.read().unwrap(),
            b.ptr.read().unwrap(),
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn sub(&mut self, a: &BValue, b: &BValue, name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_sub(
            fn_builder_guard,
            a.ptr.read().unwrap(),
            b.ptr.read().unwrap(),
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn and(&mut self, a: &BValue, b: &BValue, name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_and(
            fn_builder_guard,
            a.ptr.read().unwrap(),
            b.ptr.read().unwrap(),
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn nand(&mut self, a: &BValue, b: &BValue, name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_nand(
            fn_builder_guard,
            a.ptr.read().unwrap(),
            b.ptr.read().unwrap(),
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn or(&mut self, a: &BValue, b: &BValue, name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_or(
            fn_builder_guard,
            a.ptr.read().unwrap(),
            b.ptr.read().unwrap(),
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn xor(&mut self, a: &BValue, b: &BValue, name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_xor(
            fn_builder_guard,
            a.ptr.read().unwrap(),
            b.ptr.read().unwrap(),
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn eq(&mut self, a: &BValue, b: &BValue, name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_eq(
            fn_builder_guard,
            a.ptr.read().unwrap(),
            b.ptr.read().unwrap(),
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn ne(&mut self, a: &BValue, b: &BValue, name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_ne(
            fn_builder_guard,
            a.ptr.read().unwrap(),
            b.ptr.read().unwrap(),
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn not(&mut self, a: &BValue, name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_not(
            fn_builder_guard,
            a.ptr.read().unwrap(),
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn neg(&mut self, a: &BValue, name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_negate(
            fn_builder_guard,
            a.ptr.read().unwrap(),
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn rev(&mut self, a: &BValue, name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_reverse(
            fn_builder_guard,
            a.ptr.read().unwrap(),
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn or_reduce(&mut self, a: &BValue, name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_or_reduce(
            fn_builder_guard,
            a.ptr.read().unwrap(),
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn and_reduce(&mut self, a: &BValue, name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_and_reduce(
            fn_builder_guard,
            a.ptr.read().unwrap(),
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn xor_reduce(&mut self, a: &BValue, name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_xor_reduce(
            fn_builder_guard,
            a.ptr.read().unwrap(),
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn bit_slice(
        &mut self,
        value: &BValue,
        start: u64,
        width: u64,
        name: Option<&str>,
    ) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let value_guard: RwLockReadGuard<BValuePtr> = value.ptr.read().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_bit_slice(
            fn_builder_guard,
            value_guard,
            start,
            width,
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn concat(&mut self, args: &[&BValue], name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let args_locks: Vec<RwLockReadGuard<BValuePtr>> =
            args.iter().map(|v| v.ptr.read().unwrap()).collect();
        let bvalue_ptr =
            lib_support::xls_function_builder_add_concat(fn_builder_guard, &args_locks, name);
        BValue { ptr: bvalue_ptr }
    }

    pub fn tuple(&mut self, elements: &[&BValue], name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let elements_locks: Vec<RwLockReadGuard<BValuePtr>> =
            elements.iter().map(|v| v.ptr.read().unwrap()).collect();
        let bvalue_ptr =
            lib_support::xls_function_builder_add_tuple(fn_builder_guard, &elements_locks, name);
        BValue { ptr: bvalue_ptr }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IrValue;

    #[test]
    fn test_ir_builder() {
        let mut package = IrPackage::new("sample_package").unwrap();
        let mut builder = FnBuilder::new(&mut package, "f", true);
        let param = builder.param("x", &package.get_bits_type(32));
        let f = builder.build_with_return_value(&param).unwrap();
        assert_eq!(f.get_name(), "f");
        let package_text = package.to_string();
        assert_eq!(
            package_text,
            "package sample_package

fn f(x: bits[32] id=1) -> bits[32] {
  ret x: bits[32] = param(name=x, id=1)
}
"
        );
    }

    /// Sample that builds an AOI21 function which is the formula:
    /// fn aoi21(a: bool, b: bool, c: bool) -> bool {
    ///     not(a & b | c)
    /// }
    #[test]
    fn test_ir_builder_aoi21() {
        let mut package = IrPackage::new("sample_package").unwrap();
        let mut builder = FnBuilder::new(&mut package, "aoi21", true);
        let u1 = package.get_bits_type(1);
        let a = builder.param("a", &u1);
        let b = builder.param("b", &u1);
        let c = builder.param("c", &u1);
        let a_and_b = builder.and(&a, &b, None);
        let a_and_b_or_c = builder.or(&a_and_b, &c, None);
        let not_a_and_b_or_c = builder.not(&a_and_b_or_c, None);
        let f = builder.build_with_return_value(&not_a_and_b_or_c).unwrap();
        assert_eq!(f.get_name(), "aoi21");
        let package_text = package.to_string();
        assert_eq!(
            package_text,
            "package sample_package

fn aoi21(a: bits[1] id=1, b: bits[1] id=2, c: bits[1] id=3) -> bits[1] {
  and.4: bits[1] = and(a, b, id=4)
  or.5: bits[1] = or(and.4, c, id=5)
  ret not.6: bits[1] = not(or.5, id=6)
}
"
        );
    }

    #[test]
    fn test_ir_builder_concat() {
        let mut package = IrPackage::new("sample_package").unwrap();
        let mut builder = FnBuilder::new(&mut package, "concat", true);
        let u2 = package.get_bits_type(2);
        let a = builder.param("a", &u2);
        let b = builder.param("b", &u2);
        let concat = builder.concat(&[&a, &b], None);
        let f = builder.build_with_return_value(&concat).unwrap();

        let result = f
            .interpret(&[
                IrValue::make_ubits(2, 0b10).unwrap(),
                IrValue::make_ubits(2, 0b00).unwrap(),
            ])
            .unwrap();
        assert_eq!(result, IrValue::make_ubits(4, 0b1000).unwrap());
    }

    #[test]
    fn test_ir_builder_concat_and_slice() {
        let mut package = IrPackage::new("sample_package").unwrap();
        let mut builder = FnBuilder::new(&mut package, "concat_and_slice", true);
        let u2 = package.get_bits_type(2);
        let a = builder.param("a", &u2);
        let b = builder.param("b", &u2);
        let concat = builder.concat(&[&a, &b], None);
        let slice = builder.bit_slice(&concat, 1, 2, None);
        let f = builder.build_with_return_value(&slice).unwrap();
        assert_eq!(f.get_name(), "concat_and_slice");

        let result = f
            .interpret(&[
                IrValue::make_ubits(2, 0b01).unwrap(),
                IrValue::make_ubits(2, 0b10).unwrap(),
            ])
            .unwrap();
        assert_eq!(result, IrValue::make_ubits(2, 0b11).unwrap());

        let result = f
            .interpret(&[
                IrValue::make_ubits(2, 0b00).unwrap(),
                IrValue::make_ubits(2, 0b10).unwrap(),
            ])
            .unwrap();
        assert_eq!(result, IrValue::make_ubits(2, 0b01).unwrap());
    }

    #[test]
    fn test_ir_builder_literal() {
        let mut package = IrPackage::new("sample_package").unwrap();
        let mut builder = FnBuilder::new(&mut package, "literal", true);
        let literal = builder.literal(&IrValue::make_ubits(2, 0b10).unwrap(), None);
        let f = builder.build_with_return_value(&literal).unwrap();

        let package_text = package.to_string();
        assert_eq!(
            package_text,
            "package sample_package

fn literal() -> bits[2] {
  ret literal.1: bits[2] = literal(value=2, id=1)
}
"
        );

        let result = f.interpret(&[]).unwrap();
        assert_eq!(result, IrValue::make_ubits(2, 0b10).unwrap());
    }

    #[test]
    fn test_ir_builder_tuple() {
        let mut package = IrPackage::new("sample_package").unwrap();
        let mut builder = FnBuilder::new(&mut package, "tuple_and_then_index", true);
        let u2 = package.get_bits_type(2);
        let a = builder.param("a", &u2);
        let b = builder.param("b", &u2);
        let tuple = builder.tuple(&[&a, &b], None);
        let f = builder.build_with_return_value(&tuple).unwrap();

        let package_text = package.to_string();
        assert_eq!(
            package_text,
            "package sample_package

fn tuple_and_then_index(a: bits[2] id=1, b: bits[2] id=2) -> (bits[2], bits[2]) {
  ret tuple.3: (bits[2], bits[2]) = tuple(a, b, id=3)
}
"
        );

        let a_value = IrValue::make_ubits(2, 0b01).unwrap();
        let b_value = IrValue::make_ubits(2, 0b10).unwrap();
        let result = f.interpret(&[a_value.clone(), b_value.clone()]).unwrap();
        assert_eq!(result, IrValue::make_tuple(&[a_value, b_value]));
    }

    #[test]
    fn test_ir_builder_bvalue_clone() {
        let mut package = IrPackage::new("sample_package").unwrap();
        let mut builder = FnBuilder::new(&mut package, "make_tuple_literal", true);
        let a_value = IrValue::make_ubits(2, 0b01).unwrap();
        let b_value = IrValue::make_ubits(2, 0b10).unwrap();
        let a = builder.literal(&a_value, None);
        let b = builder.literal(&b_value, None);
        let tuple = builder.tuple(&[&a, &b], None);
        let tuple2 = tuple.clone();
        let f = builder.build_with_return_value(&tuple2).unwrap();
        let result = f.interpret(&[]).unwrap();
        assert_eq!(result, IrValue::make_tuple(&[a_value, b_value]));
    }

    #[test]
    fn test_ir_builder_bvalue_reuse() {
        let mut package = IrPackage::new("sample_package").unwrap();
        let mut builder = FnBuilder::new(&mut package, "f", true);
        let x = builder.param("x", &package.get_bits_type(32));
        let y = builder.param("y", &package.get_bits_type(32));
        let x_mul_y = builder.and(&x, &y, Some("x_mul_y"));
        let x_plus_y = builder.or(&x, &y, Some("x_plus_y"));
        let a = builder.or(&x, &x_mul_y, Some("a"));
        let b = builder.or(&y, &x_plus_y, Some("b"));
        let result = builder.tuple(&[&a, &b], Some("result"));
        let _f = builder.build_with_return_value(&result).unwrap();

        assert_eq!(
            package.to_string(),
            "package sample_package

fn f(x: bits[32] id=1, y: bits[32] id=2) -> (bits[32], bits[32]) {
  x_mul_y: bits[32] = and(x, y, id=3)
  x_plus_y: bits[32] = or(x, y, id=4)
  a: bits[32] = or(x, x_mul_y, id=5)
  b: bits[32] = or(y, x_plus_y, id=6)
  ret result: (bits[32], bits[32]) = tuple(a, b, id=7)
}
"
        );
    }

    #[test]
    fn test_ir_builder_add() {
        let mut package = IrPackage::new("sample_package").unwrap();
        let mut builder = FnBuilder::new(&mut package, "add", true);
        let a = builder.param("a", &package.get_bits_type(32));
        let b = builder.param("b", &package.get_bits_type(32));
        let result = builder.add(&a, &b, None);
        let f = builder.build_with_return_value(&result).unwrap();

        let result = f
            .interpret(&[
                IrValue::make_ubits(32, 1).unwrap(),
                IrValue::make_ubits(32, 2).unwrap(),
            ])
            .unwrap();
        assert_eq!(result, IrValue::make_ubits(32, 3).unwrap());
    }

    #[test]
    fn test_ir_builder_bitwise_reductions() {
        let mut package = IrPackage::new("sample_package").unwrap();
        let mut builder = FnBuilder::new(&mut package, "f", true);
        let x = builder.param("x", &package.get_bits_type(3));
        let and_reduce = builder.and_reduce(&x, None);
        let or_reduce = builder.or_reduce(&x, None);
        let xor_reduce = builder.xor_reduce(&x, None);
        let t = builder.tuple(&[&and_reduce, &or_reduce, &xor_reduce], None);
        let f = builder.build_with_return_value(&t).unwrap();
        assert_eq!(f.get_name(), "f");

        let result = f
            .interpret(&[IrValue::make_ubits(3, 0b110).unwrap()])
            .unwrap();
        assert_eq!(
            result,
            IrValue::make_tuple(&[
                IrValue::make_ubits(1, 0).unwrap(), // and_reduce
                IrValue::make_ubits(1, 1).unwrap(), // or_reduce
                IrValue::make_ubits(1, 0).unwrap(), // xor_reduce
            ])
        );

        let result = f
            .interpret(&[IrValue::make_ubits(3, 0b111).unwrap()])
            .unwrap();
        assert_eq!(
            result,
            IrValue::make_tuple(&[
                IrValue::make_ubits(1, 1).unwrap(), // and_reduce
                IrValue::make_ubits(1, 1).unwrap(), // or_reduce
                IrValue::make_ubits(1, 1).unwrap(), // xor_reduce
            ])
        );
    }
}
