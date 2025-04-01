// SPDX-License-Identifier: Apache-2.0

//! APIs that wrap the Verilog AST building facilities inside of XLS.

#![allow(unused)]

use xlsynth_sys::{self as sys};

use std::{
    ffi::CString,
    os::raw::c_char,
    sync::{Arc, Mutex},
};

use crate::{
    c_str_to_rust, ir_value::IrFormatPreference, lib_support::xls_format_preference_from_string,
    xls_parse_typed_value, XlsynthError,
};

struct VastFilePtr(pub *mut sys::CVastFile);

enum VastOperatorKind {
    // unary operators
    Negate = 0,
    BitwiseNot = 1,
    LogicalNot = 2,
    AndReduce = 3,
    OrReduce = 4,
    XorReduce = 5,

    // binary operators
    Add = 6,
    LogicalAnd = 7,
    BitwiseAnd = 8,
    Ne = 9,
    CaseNe = 10,
    Eq = 11,
    CaseEq = 12,
    Ge = 13,
    Gt = 14,
    Le = 15,
    Lt = 16,
    Div = 17,
    Mod = 18,
    Mul = 19,
    Power = 20,
    BitwiseOr = 21,
    LogicalOr = 22,
    BitwiseXor = 23,
    Shll = 24,
    Shra = 25,
    Shrl = 26,
    Sub = 27,
    NeX = 28,
    EqX = 29,
}

impl Drop for VastFilePtr {
    fn drop(&mut self) {
        unsafe { sys::xls_vast_verilog_file_free(self.0) }
    }
}

pub struct VastDataType {
    inner: *mut sys::CVastDataType,
    parent: Arc<Mutex<VastFilePtr>>,
}

#[derive(Clone)]
pub struct LogicRef {
    inner: *mut sys::CVastLogicRef,
    parent: Arc<Mutex<VastFilePtr>>,
}

impl LogicRef {
    pub fn to_expr(&self) -> Expr {
        let locked = self.parent.lock().unwrap();
        let inner = unsafe { sys::xls_vast_logic_ref_as_expression(self.inner) };
        Expr {
            inner,
            parent: self.parent.clone(),
        }
    }
    pub fn to_indexable_expr(&self) -> IndexableExpr {
        let locked = self.parent.lock().unwrap();
        let inner = unsafe { sys::xls_vast_logic_ref_as_indexable_expression(self.inner) };
        IndexableExpr {
            inner,
            parent: self.parent.clone(),
        }
    }
}

pub struct Expr {
    inner: *mut sys::CVastExpression,
    parent: Arc<Mutex<VastFilePtr>>,
}

pub struct IndexableExpr {
    inner: *mut sys::CVastIndexableExpression,
    parent: Arc<Mutex<VastFilePtr>>,
}

pub struct VastModule {
    inner: *mut sys::CVastModule,
    parent: Arc<Mutex<VastFilePtr>>,
}

pub struct Slice {
    inner: *mut sys::CVastSlice,
    parent: Arc<Mutex<VastFilePtr>>,
}

impl Slice {
    pub fn to_expr(&self) -> Expr {
        let locked = self.parent.lock().unwrap();
        let inner = unsafe { sys::xls_vast_slice_as_expression(self.inner) };
        Expr {
            inner,
            parent: self.parent.clone(),
        }
    }
}

pub struct Index {
    inner: *mut sys::CVastIndex,
    parent: Arc<Mutex<VastFilePtr>>,
}

impl Index {
    pub fn to_expr(&self) -> Expr {
        let locked = self.parent.lock().unwrap();
        let inner = unsafe { sys::xls_vast_index_as_expression(self.inner) };
        Expr {
            inner,
            parent: self.parent.clone(),
        }
    }

    pub fn to_indexable_expr(&self) -> IndexableExpr {
        let locked = self.parent.lock().unwrap();
        let inner = unsafe { sys::xls_vast_index_as_indexable_expression(self.inner) };
        IndexableExpr {
            inner,
            parent: self.parent.clone(),
        }
    }
}

pub struct Instantiation {
    inner: *mut sys::CVastInstantiation,
    parent: Arc<Mutex<VastFilePtr>>,
}

pub struct ContinuousAssignment {
    inner: *mut sys::CVastContinuousAssignment,
    parent: Arc<Mutex<VastFilePtr>>,
}

impl VastModule {
    pub fn add_input(&mut self, name: &str, data_type: &VastDataType) -> LogicRef {
        let c_name = CString::new(name).unwrap();
        let _locked = self.parent.lock().unwrap();
        let c_logic_ref = unsafe {
            sys::xls_vast_verilog_module_add_input(self.inner, c_name.as_ptr(), data_type.inner)
        };
        LogicRef {
            inner: c_logic_ref,
            parent: self.parent.clone(),
        }
    }

    pub fn add_output(&mut self, name: &str, data_type: &VastDataType) -> LogicRef {
        let c_name = CString::new(name).unwrap();
        let _locked = self.parent.lock().unwrap();
        let c_logic_ref = unsafe {
            sys::xls_vast_verilog_module_add_output(self.inner, c_name.as_ptr(), data_type.inner)
        };
        LogicRef {
            inner: c_logic_ref,
            parent: self.parent.clone(),
        }
    }

    pub fn add_wire(&mut self, name: &str, data_type: &VastDataType) -> LogicRef {
        let c_name = CString::new(name).unwrap();
        let _locked = self.parent.lock().unwrap();
        let c_logic_ref = unsafe {
            sys::xls_vast_verilog_module_add_wire(self.inner, c_name.as_ptr(), data_type.inner)
        };
        LogicRef {
            inner: c_logic_ref,
            parent: self.parent.clone(),
        }
    }

    pub fn add_member_instantiation(&mut self, instantiation: Instantiation) {
        let _locked = self.parent.lock().unwrap();
        unsafe {
            sys::xls_vast_verilog_module_add_member_instantiation(self.inner, instantiation.inner)
        }
    }

    pub fn add_member_continuous_assignment(&mut self, assignment: ContinuousAssignment) {
        let _locked = self.parent.lock().unwrap();
        unsafe {
            sys::xls_vast_verilog_module_add_member_continuous_assignment(
                self.inner,
                assignment.inner,
            )
        }
    }
}

pub enum VastFileType {
    Verilog,
    SystemVerilog,
}

pub struct VastFile {
    ptr: Arc<Mutex<VastFilePtr>>,
}

impl VastFile {
    /// Create a new VAST file.
    pub fn new(file_type: VastFileType) -> Self {
        let c_file_type = match file_type {
            VastFileType::Verilog => 0,
            VastFileType::SystemVerilog => 1,
        };
        Self {
            ptr: Arc::new(Mutex::new(VastFilePtr(unsafe {
                sys::xls_vast_make_verilog_file(c_file_type)
            }))),
        }
    }

    /// Adds a tick-include to the file.
    pub fn add_include(&mut self, include: &str) {
        let c_include = CString::new(include).unwrap();
        let locked = self.ptr.lock().unwrap();
        unsafe { sys::xls_vast_verilog_file_add_include(locked.0, c_include.as_ptr()) }
    }

    pub fn add_module(&mut self, name: &str) -> VastModule {
        let c_name = CString::new(name).unwrap();
        let locked = self.ptr.lock().unwrap();
        let module = unsafe { sys::xls_vast_verilog_file_add_module(locked.0, c_name.as_ptr()) };
        VastModule {
            inner: module,
            parent: self.ptr.clone(),
        }
    }

    pub fn make_instantiation(
        &mut self,
        module_name: &str,
        instance_name: &str,
        parameter_port_names: &[&str],
        parameter_expressions: &[&Expr],
        connection_port_names: &[&str],
        connection_expressions: &[Option<&Expr>],
    ) -> Instantiation {
        let c_module_name = CString::new(module_name).unwrap();
        let c_instance_name = CString::new(instance_name).unwrap();

        // Even though we only need char pointers to call the C++ API, we need to return
        // a Vec of CStrings in addition to the char pointers, to prevent the strings
        // from being dropped before the pointers are used.
        fn to_cstrings_and_ptrs(strings: &[&str]) -> (Vec<CString>, Vec<*const c_char>) {
            let cstrings: Vec<CString> =
                strings.iter().map(|&s| CString::new(s).unwrap()).collect();
            let ptrs = cstrings.iter().map(|s| s.as_ptr()).collect();
            (cstrings, ptrs)
        }

        let (c_param_names, c_param_name_ptrs) = to_cstrings_and_ptrs(parameter_port_names);
        let (c_conn_names, c_conn_name_ptrs) = to_cstrings_and_ptrs(connection_port_names);

        fn to_expr_ptrs(exprs: &[&Expr]) -> Vec<*const sys::CVastExpression> {
            exprs
                .iter()
                .map(|expr| expr.inner as *const sys::CVastExpression)
                .collect()
        }

        fn to_opt_expr_ptrs(exprs: &[Option<&Expr>]) -> Vec<*const sys::CVastExpression> {
            exprs
                .iter()
                .map(|expr| {
                    if let Some(expr) = expr {
                        expr.inner as *const sys::CVastExpression
                    } else {
                        std::ptr::null()
                    }
                })
                .collect()
        }

        let c_param_expr_ptrs = to_expr_ptrs(parameter_expressions);
        let c_conn_expr_ptrs = to_opt_expr_ptrs(connection_expressions);

        let locked = self.ptr.lock().unwrap();

        let instantiation = unsafe {
            sys::xls_vast_verilog_file_make_instantiation(
                locked.0,
                c_module_name.as_ptr(),
                c_instance_name.as_ptr(),
                c_param_name_ptrs.as_ptr(),
                c_param_expr_ptrs.as_ptr(),
                c_param_expr_ptrs.len(),
                c_conn_name_ptrs.as_ptr(),
                c_conn_expr_ptrs.as_ptr(),
                c_conn_expr_ptrs.len(),
            )
        };

        Instantiation {
            inner: instantiation,
            parent: self.ptr.clone(),
        }
    }

    /// Makes a literal expression from a string, `s`, using the given format,
    /// `fmt`. `s` must be in the form `bits[N]:value`, where `N` is the bit
    /// width and `value` is the value of the literal, expressed in decimal,
    /// hex, or binary. For example, `s` might be `bits[16]:42` or
    /// `bits[39]:0xABCD`. `fmt` indicates how the literal should be formatted
    /// in the output Verilog.
    pub fn make_literal(
        &mut self,
        s: &str,
        fmt: &IrFormatPreference,
    ) -> Result<Expr, XlsynthError> {
        let v = xls_parse_typed_value(s).unwrap();
        let mut fmt = xls_format_preference_from_string(fmt.to_string()).unwrap();

        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut literal_out: *mut sys::CVastLiteral = std::ptr::null_mut();

        unsafe {
            let success = sys::xls_vast_verilog_file_make_literal(
                self.ptr.lock().unwrap().0,
                v.to_bits().unwrap().ptr,
                fmt,
                true,
                &mut error_out,
                &mut literal_out,
            );

            if success {
                Ok(Expr {
                    inner: sys::xls_vast_literal_as_expression(literal_out),
                    parent: self.ptr.clone(),
                })
            } else {
                Err(XlsynthError(c_str_to_rust(error_out)))
            }
        }
    }

    pub fn make_scalar_type(&mut self) -> VastDataType {
        let locked = self.ptr.lock().unwrap();
        let data_type = unsafe { sys::xls_vast_verilog_file_make_scalar_type(locked.0) };
        VastDataType {
            inner: data_type,
            parent: self.ptr.clone(),
        }
    }

    pub fn make_bit_vector_type(&mut self, bit_count: i64, is_signed: bool) -> VastDataType {
        let locked = self.ptr.lock().unwrap();
        let data_type = unsafe {
            sys::xls_vast_verilog_file_make_bit_vector_type(locked.0, bit_count, is_signed)
        };
        VastDataType {
            inner: data_type,
            parent: self.ptr.clone(),
        }
    }

    pub fn make_extern_package_type(
        &mut self,
        package_name: &str,
        type_name: &str,
    ) -> VastDataType {
        let locked = self.ptr.lock().unwrap();
        let c_package_name = CString::new(package_name).unwrap();
        let c_type_name = CString::new(type_name).unwrap();
        let data_type = unsafe {
            sys::xls_vast_verilog_file_make_extern_package_type(
                locked.0,
                c_package_name.as_ptr(),
                c_type_name.as_ptr(),
            )
        };
        VastDataType {
            inner: data_type,
            parent: self.ptr.clone(),
        }
    }

    pub fn make_packed_array_type(
        &mut self,
        element_type: VastDataType,
        dimensions: &[i64],
    ) -> VastDataType {
        let locked = self.ptr.lock().unwrap();
        let data_type = unsafe {
            sys::xls_vast_verilog_file_make_packed_array_type(
                locked.0,
                element_type.inner,
                dimensions.as_ptr(),
                dimensions.len(),
            )
        };
        VastDataType {
            inner: data_type,
            parent: self.ptr.clone(),
        }
    }

    pub fn make_slice(&mut self, indexable: &IndexableExpr, hi: i64, lo: i64) -> Slice {
        let locked = self.ptr.lock().unwrap();
        let inner =
            unsafe { sys::xls_vast_verilog_file_make_slice_i64(locked.0, indexable.inner, hi, lo) };
        Slice {
            inner,
            parent: self.ptr.clone(),
        }
    }

    pub fn make_index(&mut self, indexable: &IndexableExpr, index: i64) -> Index {
        let locked = self.ptr.lock().unwrap();
        let inner =
            unsafe { sys::xls_vast_verilog_file_make_index_i64(locked.0, indexable.inner, index) };
        Index {
            inner,
            parent: self.ptr.clone(),
        }
    }

    pub fn make_concat(&mut self, exprs: &[&Expr]) -> Expr {
        let locked = self.ptr.lock().unwrap();
        let mut expr_ptrs: Vec<*mut sys::CVastExpression> =
            exprs.iter().map(|expr| expr.inner).collect();
        let inner = unsafe {
            sys::xls_vast_verilog_file_make_concat(locked.0, expr_ptrs.as_mut_ptr(), exprs.len())
        };
        Expr {
            inner,
            parent: self.ptr.clone(),
        }
    }

    fn make_unary(&mut self, op: VastOperatorKind, expr: &Expr) -> Expr {
        let locked = self.ptr.lock().unwrap();
        let op_i32 = op as i32;
        let inner = unsafe { sys::xls_vast_verilog_file_make_unary(locked.0, expr.inner, op_i32) };
        Expr {
            inner,
            parent: self.ptr.clone(),
        }
    }

    pub fn make_not(&mut self, expr: &Expr) -> Expr {
        self.make_unary(VastOperatorKind::BitwiseNot, expr)
    }

    pub fn make_negate(&mut self, expr: &Expr) -> Expr {
        self.make_unary(VastOperatorKind::Negate, expr)
    }

    pub fn make_logical_not(&mut self, expr: &Expr) -> Expr {
        self.make_unary(VastOperatorKind::LogicalNot, expr)
    }

    pub fn make_and_reduce(&mut self, expr: &Expr) -> Expr {
        self.make_unary(VastOperatorKind::AndReduce, expr)
    }

    pub fn make_or_reduce(&mut self, expr: &Expr) -> Expr {
        self.make_unary(VastOperatorKind::OrReduce, expr)
    }

    pub fn make_xor_reduce(&mut self, expr: &Expr) -> Expr {
        self.make_unary(VastOperatorKind::XorReduce, expr)
    }

    // -- binary ops

    fn make_binary(&mut self, op: VastOperatorKind, lhs: &Expr, rhs: &Expr) -> Expr {
        let locked = self.ptr.lock().unwrap();
        let op_i32 = op as i32;
        let inner = unsafe {
            sys::xls_vast_verilog_file_make_binary(locked.0, lhs.inner, rhs.inner, op_i32)
        };
        Expr {
            inner,
            parent: self.ptr.clone(),
        }
    }

    pub fn make_add(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(VastOperatorKind::Add, lhs, rhs)
    }

    pub fn make_sub(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(VastOperatorKind::Sub, lhs, rhs)
    }

    pub fn make_mul(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(VastOperatorKind::Mul, lhs, rhs)
    }

    pub fn make_div(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(VastOperatorKind::Div, lhs, rhs)
    }

    pub fn make_mod(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(VastOperatorKind::Mod, lhs, rhs)
    }

    pub fn make_power(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(VastOperatorKind::Power, lhs, rhs)
    }

    pub fn make_bitwise_and(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(VastOperatorKind::BitwiseAnd, lhs, rhs)
    }

    pub fn make_bitwise_or(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(VastOperatorKind::BitwiseOr, lhs, rhs)
    }

    pub fn make_bitwise_xor(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(VastOperatorKind::BitwiseXor, lhs, rhs)
    }

    pub fn make_bitwise_not(&mut self, expr: &Expr) -> Expr {
        self.make_unary(VastOperatorKind::BitwiseNot, expr)
    }

    pub fn make_shll(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(VastOperatorKind::Shll, lhs, rhs)
    }

    pub fn make_shra(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(VastOperatorKind::Shra, lhs, rhs)
    }

    pub fn make_shrl(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(VastOperatorKind::Shrl, lhs, rhs)
    }

    pub fn make_ne(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(VastOperatorKind::Ne, lhs, rhs)
    }

    pub fn make_case_ne(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(VastOperatorKind::CaseNe, lhs, rhs)
    }

    pub fn make_eq(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(VastOperatorKind::Eq, lhs, rhs)
    }

    pub fn make_case_eq(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(VastOperatorKind::CaseEq, lhs, rhs)
    }

    pub fn make_ge(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(VastOperatorKind::Ge, lhs, rhs)
    }

    pub fn make_gt(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(VastOperatorKind::Gt, lhs, rhs)
    }

    pub fn make_le(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(VastOperatorKind::Le, lhs, rhs)
    }

    pub fn make_lt(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(VastOperatorKind::Lt, lhs, rhs)
    }

    pub fn make_logical_and(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(VastOperatorKind::LogicalAnd, lhs, rhs)
    }

    pub fn make_logical_or(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(VastOperatorKind::LogicalOr, lhs, rhs)
    }

    pub fn make_ne_x(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(VastOperatorKind::NeX, lhs, rhs)
    }

    pub fn make_eq_x(&mut self, lhs: &Expr, rhs: &Expr) -> Expr {
        self.make_binary(VastOperatorKind::EqX, lhs, rhs)
    }

    // Creates a ternary operator that selects between `then_expr` and `else_expr`
    // based on the value of `cond`.
    pub fn make_ternary(&mut self, cond: &Expr, then_expr: &Expr, else_expr: &Expr) -> Expr {
        let locked = self.ptr.lock().unwrap();
        let inner = unsafe {
            sys::xls_vast_verilog_file_make_ternary(
                locked.0,
                cond.inner,
                then_expr.inner,
                else_expr.inner,
            )
        };
        Expr {
            inner,
            parent: self.ptr.clone(),
        }
    }

    // Creates an `assign` statement that drives `lhs` with the `rhs` expression
    // given.
    pub fn make_continuous_assignment(&mut self, lhs: &Expr, rhs: &Expr) -> ContinuousAssignment {
        let locked = self.ptr.lock().unwrap();
        let inner = unsafe {
            sys::xls_vast_verilog_file_make_continuous_assignment(locked.0, lhs.inner, rhs.inner)
        };
        ContinuousAssignment {
            inner,
            parent: self.ptr.clone(),
        }
    }

    pub fn emit(&self) -> String {
        let locked = self.ptr.lock().unwrap();
        let c_str = unsafe { sys::xls_vast_verilog_file_emit(locked.0) };
        unsafe { c_str_to_rust(c_str) }
    }
}
