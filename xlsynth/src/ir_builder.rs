// SPDX-License-Identifier: Apache-2.0

use crate::ir_package::IrPackage;
use crate::ir_package::IrPackagePtr;
use crate::ir_package::IrType;
use crate::lib_support;
use crate::lib_support::{BValuePtr, IrFnBuilderPtr};
use crate::IrFunction;
use crate::XlsynthError;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::RwLockReadGuard;
use std::sync::RwLockWriteGuard;

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
        BValue {
            ptr: Arc::new(RwLock::new(bvalue_ptr)),
        }
    }

    pub fn and(&mut self, a: &BValue, b: &BValue, name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_and(
            fn_builder_guard,
            a.ptr.read().unwrap(),
            b.ptr.read().unwrap(),
            name,
        );
        BValue {
            ptr: Arc::new(RwLock::new(bvalue_ptr)),
        }
    }

    pub fn or(&mut self, a: &BValue, b: &BValue, name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_or(
            fn_builder_guard,
            a.ptr.read().unwrap(),
            b.ptr.read().unwrap(),
            name,
        );
        BValue {
            ptr: Arc::new(RwLock::new(bvalue_ptr)),
        }
    }

    pub fn not(&mut self, a: &BValue, name: Option<&str>) -> BValue {
        let fn_builder_guard = self.fn_builder.write().unwrap();
        let bvalue_ptr = lib_support::xls_function_builder_add_not(
            fn_builder_guard,
            a.ptr.read().unwrap(),
            name,
        );
        BValue {
            ptr: Arc::new(RwLock::new(bvalue_ptr)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
