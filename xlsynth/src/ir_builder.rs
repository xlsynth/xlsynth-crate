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

    pub fn bit_slice_update(
        &mut self,
        value: &BValue,
        start: &BValue,
        update: &BValue,
        name: Option<&str>,
    ) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let value_guard: RwLockReadGuard<BValuePtr> = value.ptr.read().unwrap();
        let start_guard: RwLockReadGuard<BValuePtr> = start.ptr.read().unwrap();
        let update_guard: RwLockReadGuard<BValuePtr> = update.ptr.read().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_bit_slice_update(
            fn_builder_guard,
            value_guard,
            start_guard,
            update_guard,
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

    pub fn tuple_index(&mut self, tuple: &BValue, index: u64, name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let tuple_guard: RwLockReadGuard<BValuePtr> = tuple.ptr.read().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_tuple_index(
            fn_builder_guard,
            tuple_guard,
            index,
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn umul(&mut self, a: &BValue, b: &BValue, name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_umul(
            fn_builder_guard,
            a.ptr.read().unwrap(),
            b.ptr.read().unwrap(),
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn smul(&mut self, a: &BValue, b: &BValue, name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_smul(
            fn_builder_guard,
            a.ptr.read().unwrap(),
            b.ptr.read().unwrap(),
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn array(
        &mut self,
        element_type: &IrType,
        elements: &[&BValue],
        name: Option<&str>,
    ) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let elements_locks: Vec<RwLockReadGuard<BValuePtr>> =
            elements.iter().map(|v| v.ptr.read().unwrap()).collect();
        let bvalue_ptr = lib_support::xls_function_builder_add_array(
            fn_builder_guard,
            element_type,
            &elements_locks,
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn array_index(&mut self, array: &BValue, index: &BValue, name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let array_guard: RwLockReadGuard<BValuePtr> = array.ptr.read().unwrap();
        let index_guard: RwLockReadGuard<BValuePtr> = index.ptr.read().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_array_index_multi(
            fn_builder_guard,
            array_guard,
            &[index_guard],
            false,
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn array_concat(&mut self, arrays: &[&BValue], name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let arrays_guards: Vec<RwLockReadGuard<BValuePtr>> =
            arrays.iter().map(|v| v.ptr.read().unwrap()).collect();
        let bvalue_ptr = lib_support::xls_function_builder_add_array_concat(
            fn_builder_guard,
            &arrays_guards,
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn array_slice(
        &mut self,
        array: &BValue,
        start: &BValue,
        width: u64,
        name: Option<&str>,
    ) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let array_guard: RwLockReadGuard<BValuePtr> = array.ptr.read().unwrap();
        let start_guard: RwLockReadGuard<BValuePtr> = start.ptr.read().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_array_slice(
            fn_builder_guard,
            array_guard,
            start_guard,
            width as i64,
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn array_update(
        &mut self,
        array: &BValue,
        update_value: &BValue,
        index: &BValue,
        name: Option<&str>,
    ) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let array_guard: RwLockReadGuard<BValuePtr> = array.ptr.read().unwrap();
        let update_value_guard: RwLockReadGuard<BValuePtr> = update_value.ptr.read().unwrap();
        let index_guard: RwLockReadGuard<BValuePtr> = index.ptr.read().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_array_update(
            fn_builder_guard,
            array_guard,
            update_value_guard,
            &[index_guard],
            false,
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn shra(&mut self, a: &BValue, b: &BValue, name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_shra(
            fn_builder_guard,
            a.ptr.read().unwrap(),
            b.ptr.read().unwrap(),
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn shrl(&mut self, a: &BValue, b: &BValue, name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_shrl(
            fn_builder_guard,
            a.ptr.read().unwrap(),
            b.ptr.read().unwrap(),
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn shll(&mut self, a: &BValue, b: &BValue, name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_shll(
            fn_builder_guard,
            a.ptr.read().unwrap(),
            b.ptr.read().unwrap(),
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn nor(&mut self, a: &BValue, b: &BValue, name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_nor(
            fn_builder_guard,
            a.ptr.read().unwrap(),
            b.ptr.read().unwrap(),
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn clz(&mut self, a: &BValue, name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_clz(
            fn_builder_guard,
            a.ptr.read().unwrap(),
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn ctz(&mut self, a: &BValue, name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_ctz(
            fn_builder_guard,
            a.ptr.read().unwrap(),
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn encode(&mut self, a: &BValue, name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_encode(
            fn_builder_guard,
            a.ptr.read().unwrap(),
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn decode(&mut self, a: &BValue, name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_decode(
            fn_builder_guard,
            a.ptr.read().unwrap(),
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn identity(&mut self, a: &BValue, name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_identity(
            fn_builder_guard,
            a.ptr.read().unwrap(),
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn select(
        &mut self,
        selector: &BValue,
        cases: &[&BValue],
        default_value: &BValue,
        name: Option<&str>,
    ) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let selector_guard: RwLockReadGuard<BValuePtr> = selector.ptr.read().unwrap();
        let cases_guards: Vec<RwLockReadGuard<BValuePtr>> =
            cases.iter().map(|v| v.ptr.read().unwrap()).collect();
        let default_value_guard: RwLockReadGuard<BValuePtr> = default_value.ptr.read().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_select(
            fn_builder_guard,
            selector_guard,
            &cases_guards,
            default_value_guard,
            name,
        );
        BValue { ptr: bvalue_ptr }
    }

    pub fn last_value(&self) -> Result<BValue, XlsynthError> {
        let fn_builder_guard = self.fn_builder.read().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_last_value(fn_builder_guard)?;
        Ok(BValue { ptr: bvalue_ptr })
    }

    pub fn get_type(&self, value: &BValue) -> Option<IrType> {
        let fn_builder_guard = self.fn_builder.read().unwrap();
        lib_support::xls_function_builder_get_type(fn_builder_guard, value.ptr.read().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ir_value::IrFormatPreference, IrValue};
    use pretty_assertions::assert_eq;

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

    #[test]
    fn test_ir_builder_nand() {
        let mut package = IrPackage::new("sample_package").unwrap();
        let mut builder = FnBuilder::new(&mut package, "nand", true);
        let a = builder.param("a", &package.get_bits_type(1));
        let b = builder.param("b", &package.get_bits_type(1));
        let nand = builder.nand(&a, &b, None);
        let f = builder.build_with_return_value(&nand).unwrap();
        assert_eq!(f.get_name(), "nand");

        let truth_table = vec![(0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0)];

        for (a, b, expected) in truth_table {
            let result = f
                .interpret(&[
                    IrValue::make_ubits(1, a).unwrap(),
                    IrValue::make_ubits(1, b).unwrap(),
                ])
                .unwrap();
            assert_eq!(result, IrValue::make_ubits(1, expected).unwrap());
        }
    }

    #[test]
    fn test_ir_builder_tuple_index() {
        let mut package = IrPackage::new("sample_package").unwrap();
        let mut builder = FnBuilder::new(&mut package, "tuple_index", true);
        let tuple_type =
            package.get_tuple_type(&[package.get_bits_type(2), package.get_bits_type(4)]);
        let x = builder.param("x", &tuple_type);
        let index = builder.tuple_index(&x, 1, None);
        let f = builder.build_with_return_value(&index).unwrap();

        let result = f
            .interpret(&[IrValue::make_tuple(&[
                IrValue::make_ubits(2, 1).unwrap(),
                IrValue::make_ubits(4, 2).unwrap(),
            ])])
            .unwrap();
        assert_eq!(result, IrValue::make_ubits(4, 2).unwrap());
    }

    // Note: because the current signature takes N bit operands and produces an N
    // bit result, we cannot distinguish between umul and smul.
    #[test]
    fn test_ir_builder_umul_vs_smul() {
        let mut package = IrPackage::new("sample_package").unwrap();
        let mut builder = FnBuilder::new(&mut package, "umul_vs_smul", true);
        let a = builder.param("a", &package.get_bits_type(3));
        let b = builder.param("b", &package.get_bits_type(3));
        let umul = builder.umul(&a, &b, None);
        let smul = builder.smul(&a, &b, None);
        let result = builder.tuple(&[&umul, &smul], None);
        let f = builder.build_with_return_value(&result).unwrap();

        let result = f
            .interpret(&[
                IrValue::make_ubits(3, 2).unwrap(),
                IrValue::make_ubits(3, 0b110).unwrap(),
            ])
            .unwrap();
        assert_eq!(
            result,
            IrValue::make_tuple(&[
                IrValue::make_ubits(3, 0b100).unwrap(),
                IrValue::make_ubits(3, 0b100).unwrap(),
            ])
        );
    }

    #[test]
    fn test_ir_builder_make_array_and_index() {
        let mut package = IrPackage::new("sample_package").unwrap();
        let mut builder = FnBuilder::new(&mut package, "make_array_and_index", true);
        let u2 = package.get_bits_type(2);
        let u1 = package.get_bits_type(1);
        let x = builder.param("x", &u2);
        let y = builder.param("y", &u2);
        let i = builder.param("i", &u1);
        let array = builder.array(&u2, &[&x, &y], None);
        let index = builder.array_index(&array, &i, None);
        let f = builder.build_with_return_value(&index).unwrap();

        assert_eq!(
            package.to_string(),
            "package sample_package

fn make_array_and_index(x: bits[2] id=1, y: bits[2] id=2, i: bits[1] id=3) -> bits[2] {
  array.4: bits[2][2] = array(x, y, id=4)
  ret array_index.5: bits[2] = array_index(array.4, indices=[i], id=5)
}
"
        );

        let x_value = IrValue::make_ubits(2, 0b01).unwrap();
        let y_value = IrValue::make_ubits(2, 0b10).unwrap();
        let result = f
            .interpret(&[
                x_value.clone(),
                y_value.clone(),
                IrValue::make_ubits(1, 0).unwrap(),
            ])
            .unwrap();
        assert_eq!(result, IrValue::make_ubits(2, 0b01).unwrap());

        let result = f
            .interpret(&[
                x_value.clone(),
                y_value.clone(),
                IrValue::make_ubits(1, 1).unwrap(),
            ])
            .unwrap();
        assert_eq!(result, IrValue::make_ubits(2, 0b10).unwrap());
    }

    #[test]
    fn test_ir_builder_make_bit_slice_update() {
        let mut package = IrPackage::new("sample_package").unwrap();
        let mut builder = FnBuilder::new(&mut package, "make_bit_slice_update", true);
        let u4 = package.get_bits_type(4);
        let u2 = package.get_bits_type(2);
        let x = builder.param("x", &u4);
        let y = builder.param("y", &u2);
        let start = builder.param("start", &u2);
        let updated = builder.bit_slice_update(&x, &start, &y, None);
        let f = builder.build_with_return_value(&updated).unwrap();

        let got = f
            .interpret(&[
                IrValue::make_ubits(4, 0b1001).unwrap(),
                IrValue::make_ubits(2, 0b11).unwrap(),
                IrValue::make_ubits(2, 1).unwrap(),
            ])
            .unwrap();
        let want = IrValue::make_ubits(4, 0b1111).unwrap();
        assert_eq!(
            got.to_string_fmt(IrFormatPreference::Binary).unwrap(),
            want.to_string_fmt(IrFormatPreference::Binary).unwrap()
        );
    }

    #[test]
    fn test_ir_builder_shift_ops() {
        let mut package = IrPackage::new("sample_package").unwrap();
        let mut builder = FnBuilder::new(&mut package, "shift_ops", true);
        let x = builder.param("x", &package.get_bits_type(4));
        let amount = builder.param("amount", &package.get_bits_type(2));
        let shra = builder.shra(&x, &amount, None);
        let shrl = builder.shrl(&x, &amount, None);
        let shll = builder.shll(&x, &amount, None);
        let result = builder.tuple(&[&shra, &shrl, &shll], None);
        let f = builder.build_with_return_value(&result).unwrap();

        let got = f
            .interpret(&[
                IrValue::make_ubits(4, 0b1010).unwrap(),
                IrValue::make_ubits(2, 2).unwrap(),
            ])
            .unwrap();
        let want = IrValue::make_tuple(&[
            IrValue::make_ubits(4, 0b1110).unwrap(), // shra
            IrValue::make_ubits(4, 0b0010).unwrap(), // shrl
            IrValue::make_ubits(4, 0b1000).unwrap(), // shll
        ]);
        assert_eq!(
            got.to_string_fmt(IrFormatPreference::Binary).unwrap(),
            want.to_string_fmt(IrFormatPreference::Binary).unwrap()
        );
    }

    #[test]
    fn test_ir_builder_count_zeros() {
        let mut package = IrPackage::new("sample_package").unwrap();
        let mut builder = FnBuilder::new(&mut package, "count_zeros", true);
        let x = builder.param("x", &package.get_bits_type(4));
        let clz = builder.clz(&x, None);
        let ctz = builder.ctz(&x, None);
        let t = builder.tuple(&[&clz, &ctz], None);
        let f = builder.build_with_return_value(&t).unwrap();

        let got = f
            .interpret(&[IrValue::make_ubits(4, 0b0010).unwrap()])
            .unwrap();
        let want = IrValue::make_tuple(&[
            IrValue::make_ubits(4, 2).unwrap(), // clz
            IrValue::make_ubits(4, 1).unwrap(), // ctz
        ]);
        assert_eq!(got, want);

        let got = f
            .interpret(&[IrValue::make_ubits(4, 0b0000).unwrap()])
            .unwrap();
        let want = IrValue::make_tuple(&[
            IrValue::make_ubits(4, 4).unwrap(), // clz
            IrValue::make_ubits(4, 4).unwrap(), // ctz
        ]);
        assert_eq!(got, want);
    }

    #[test]
    fn test_ir_builder_encode() {
        let mut package = IrPackage::new("sample_package").unwrap();
        let mut builder = FnBuilder::new(&mut package, "do_encode", true);
        let x = builder.param("x", &package.get_bits_type(4));
        let encoded = builder.encode(&x, None);
        let f = builder.build_with_return_value(&encoded).unwrap();

        let result = f
            .interpret(&[IrValue::make_ubits(4, 0b1000).unwrap()])
            .unwrap();
        assert_eq!(result, IrValue::make_ubits(2, 3).unwrap());
    }

    #[test]
    fn test_ir_builder_decode() {
        let mut package = IrPackage::new("sample_package").unwrap();
        let mut builder = FnBuilder::new(&mut package, "do_decode", true);
        let x = builder.param("x", &package.get_bits_type(2));
        let decoded = builder.decode(&x, None);
        let f = builder.build_with_return_value(&decoded).unwrap();

        let result = f.interpret(&[IrValue::make_ubits(2, 3).unwrap()]).unwrap();
        assert_eq!(result, IrValue::make_ubits(4, 0b1000).unwrap());
    }

    #[test]
    fn test_ir_builder_bitops() {
        let mut package = IrPackage::new("sample_package").unwrap();
        let mut builder = FnBuilder::new(&mut package, "bitops", true);
        let u4 = package.get_bits_type(4);
        let a = builder.param("a", &u4);
        let b = builder.param("b", &u4);
        let nor = builder.nor(&a, &b, None);
        let xor = builder.xor(&a, &b, None);
        let and = builder.and(&a, &b, None);
        let or = builder.or(&a, &b, None);
        let nand = builder.nand(&a, &b, None);
        let result = builder.tuple(&[&xor, &and, &nand, &or, &nor], None);
        let f = builder.build_with_return_value(&result).unwrap();

        let got = f
            .interpret(&[
                IrValue::make_ubits(4, 0b0011).unwrap(),
                IrValue::make_ubits(4, 0b0101).unwrap(),
            ])
            .unwrap();
        let want = IrValue::make_tuple(&[
            IrValue::make_ubits(4, 0b0110).unwrap(), // xor
            IrValue::make_ubits(4, 0b0001).unwrap(), // and
            IrValue::make_ubits(4, 0b1110).unwrap(), // nand
            IrValue::make_ubits(4, 0b0111).unwrap(), // or
            IrValue::make_ubits(4, 0b1000).unwrap(), // nor
        ]);
        assert_eq!(got, want);
    }

    #[test]
    fn test_ir_builder_array_concat_and_slice() {
        let mut package = IrPackage::new("sample_package").unwrap();
        let mut fb = FnBuilder::new(&mut package, "array_concat", true);
        let u4 = package.get_bits_type(4);
        let lhs_array_type = package.get_array_type(&u4, 2);
        let rhs_array_type = package.get_array_type(&u4, 3);
        let lhs_array = fb.param("lhs_array", &lhs_array_type);
        let rhs_array = fb.param("rhs_array", &rhs_array_type);
        let concat = fb.array_concat(&[&lhs_array, &rhs_array], None);
        let start = fb.literal(&IrValue::make_ubits(4, 2).unwrap(), None);
        let slice = fb.array_slice(&concat, &start, 2, None);
        let f = fb.build_with_return_value(&slice).unwrap();

        let lhs_array_value = IrValue::make_array(&[
            IrValue::make_ubits(4, 0b0000).unwrap(),
            IrValue::make_ubits(4, 0b0001).unwrap(),
        ])
        .unwrap();
        let rhs_array_value = IrValue::make_array(&[
            IrValue::make_ubits(4, 0b0010).unwrap(),
            IrValue::make_ubits(4, 0b0011).unwrap(),
            IrValue::make_ubits(4, 0b0100).unwrap(),
        ])
        .unwrap();
        let result = f.interpret(&[lhs_array_value, rhs_array_value]).unwrap();
        assert_eq!(
            result,
            IrValue::make_array(&[
                IrValue::make_ubits(4, 0b0010).unwrap(),
                IrValue::make_ubits(4, 0b0011).unwrap(),
            ])
            .unwrap()
        );
    }

    #[test]
    fn test_ir_builder_array_update() {
        let mut package = IrPackage::new("sample_package").unwrap();
        let mut fb = FnBuilder::new(&mut package, "array_update", true);
        let u4 = package.get_bits_type(4);
        let array = fb.param("array", &package.get_array_type(&u4, 2));
        let index = fb.param("index", &package.get_bits_type(1));
        let update_value = fb.param("update_value", &u4);
        let updated = fb.array_update(&array, &update_value, &index, None);
        let f = fb.build_with_return_value(&updated).unwrap();

        let array_value = IrValue::make_array(&[
            IrValue::make_ubits(4, 0b0000).unwrap(),
            IrValue::make_ubits(4, 0b0001).unwrap(),
        ])
        .unwrap();
        let index_value = IrValue::make_ubits(1, 1).unwrap();
        let update_value = IrValue::make_ubits(4, 0b1111).unwrap();

        let got = f
            .interpret(&[array_value, index_value, update_value])
            .unwrap();

        let want = IrValue::make_array(&[
            IrValue::make_ubits(4, 0b0000).unwrap(),
            IrValue::make_ubits(4, 0b1111).unwrap(),
        ])
        .unwrap();
        assert_eq!(got, want);
    }

    #[test]
    fn test_ir_builder_get_type_of_last_value_identity_function() {
        let mut package = IrPackage::new("sample_package").unwrap();
        let mut fb = FnBuilder::new(&mut package, "f", true);
        let u4 = package.get_bits_type(4);
        let x = fb.param("x", &u4);
        let _identity = fb.identity(&x, None);
        let last_value = fb.last_value().unwrap();
        let last_value_type = fb.get_type(&last_value).unwrap();
        assert!(package.types_eq(&last_value_type, &u4).unwrap());

        let f = fb.build_with_return_value(&last_value).unwrap();
        let result = f
            .interpret(&[IrValue::make_ubits(4, 0b1010).unwrap()])
            .unwrap();
        assert_eq!(result, IrValue::make_ubits(4, 0b1010).unwrap());
    }

    #[test]
    fn test_ir_builder_get_last_value_after_construction_error() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut package = IrPackage::new("sample_package").unwrap();
        let mut fb = FnBuilder::new(&mut package, "f", true);
        let u4 = package.get_bits_type(4);
        let x = fb.param("x", &u4);
        let u5 = package.get_bits_type(5);
        let y = fb.param("y", &u5);
        fb.add(&x, &y, None);
        let last_value = fb.last_value();
        assert!(last_value.is_err());
        let error_str = last_value.err().unwrap().to_string();
        log::info!("error_str: {}", error_str);
        assert!(error_str.contains("bits[4], has type bits[5]"));
    }
}
