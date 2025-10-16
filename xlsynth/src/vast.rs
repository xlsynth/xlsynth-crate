// SPDX-License-Identifier: Apache-2.0

//! APIs that wrap the Verilog AST building facilities inside of XLS.

#![allow(unused)]
#![allow(clippy::arc_with_non_send_sync)]

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

// No additional imports needed.

// Represents the direction of a module port.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModulePortDirection {
    Input,
    Output,
}

pub struct ModulePort {
    inner: *mut sys::CVastModulePort,
    parent: Arc<Mutex<VastFilePtr>>, // Keep the VAST file alive while this port exists.
}

impl ModulePort {
    pub fn direction(&self) -> ModulePortDirection {
        let _locked = self.parent.lock().unwrap();
        let dir = unsafe { sys::xls_vast_verilog_module_port_get_direction(self.inner) };
        match dir {
            x if x == sys::XLS_VAST_MODULE_PORT_DIRECTION_INPUT => ModulePortDirection::Input,
            x if x == sys::XLS_VAST_MODULE_PORT_DIRECTION_OUTPUT => ModulePortDirection::Output,
            _ => panic!("Invalid port direction: {}", dir),
        }
    }

    pub fn name(&self) -> String {
        let _locked = self.parent.lock().unwrap();
        let def_ptr = unsafe { sys::xls_vast_verilog_module_port_get_def(self.inner) };
        let c_str = unsafe { sys::xls_vast_def_get_name(def_ptr) };
        unsafe { c_str_to_rust(c_str) }
    }

    pub fn data_type(&self) -> VastDataType {
        let _locked = self.parent.lock().unwrap();
        let def_ptr = unsafe { sys::xls_vast_verilog_module_port_get_def(self.inner) };
        let dt_ptr = unsafe { sys::xls_vast_def_get_data_type(def_ptr) };
        VastDataType {
            inner: dt_ptr,
            parent: self.parent.clone(),
        }
    }

    /// Returns the width (bit count) of this port if determinable; otherwise 0.
    pub fn width(&self) -> i64 {
        self.data_type()
            .flat_bit_count_as_int64()
            .unwrap_or_default()
    }
}

impl VastDataType {
    /// Returns the declared width for bit-vector types as an i64.
    pub fn width_as_int64(&self) -> Result<i64, XlsynthError> {
        let _locked = self.parent.lock().unwrap();
        let mut width_out: i64 = 0;
        let mut error_out: *mut c_char = std::ptr::null_mut();
        let success = unsafe {
            sys::xls_vast_data_type_width_as_int64(self.inner, &mut width_out, &mut error_out)
        };
        if success {
            Ok(width_out)
        } else {
            Err(XlsynthError(unsafe { c_str_to_rust(error_out) }))
        }
    }

    /// Returns the total flat bit count for composite types as an i64.
    pub fn flat_bit_count_as_int64(&self) -> Result<i64, XlsynthError> {
        let _locked = self.parent.lock().unwrap();
        let mut count_out: i64 = 0;
        let mut error_out: *mut c_char = std::ptr::null_mut();
        let success = unsafe {
            sys::xls_vast_data_type_flat_bit_count_as_int64(
                self.inner,
                &mut count_out,
                &mut error_out,
            )
        };
        if success {
            Ok(count_out)
        } else {
            Err(XlsynthError(unsafe { c_str_to_rust(error_out) }))
        }
    }

    /// Returns an expression representing the width, if any.
    pub fn width_expr(&self) -> Option<Expr> {
        let _locked = self.parent.lock().unwrap();
        let expr_ptr = unsafe { sys::xls_vast_data_type_width(self.inner) };
        if expr_ptr.is_null() {
            None
        } else {
            Some(Expr {
                inner: expr_ptr,
                parent: self.parent.clone(),
            })
        }
    }

    /// Returns true if the type is signed.
    pub fn is_signed(&self) -> bool {
        let _locked = self.parent.lock().unwrap();
        unsafe { sys::xls_vast_data_type_is_signed(self.inner) }
    }
}

// Extend VastModule with querying capabilities.
impl VastModule {
    /// Returns all ports (both inputs and outputs) on this module.
    pub fn ports(&self) -> Vec<ModulePort> {
        let _locked = self.parent.lock().unwrap();
        let mut count: usize = 0;
        let ports_ptr = unsafe { sys::xls_vast_verilog_module_get_ports(self.inner, &mut count) };
        if ports_ptr.is_null() || count == 0 {
            return Vec::new();
        }
        let slice = unsafe { std::slice::from_raw_parts(ports_ptr, count) };
        let result = slice
            .iter()
            .map(|&ptr| ModulePort {
                inner: ptr,
                parent: self.parent.clone(),
            })
            .collect::<Vec<_>>();
        // Free the array returned by the C API (but not the individual port objects).
        unsafe { sys::xls_vast_verilog_module_free_ports(ports_ptr, count) };
        result
    }

    /// Returns only the input ports for this module.
    pub fn input_ports(&self) -> Vec<ModulePort> {
        self.ports()
            .into_iter()
            .filter(|p| p.direction() == ModulePortDirection::Input)
            .collect()
    }

    /// Returns only the output ports for this module.
    pub fn output_ports(&self) -> Vec<ModulePort> {
        self.ports()
            .into_iter()
            .filter(|p| p.direction() == ModulePortDirection::Output)
            .collect()
    }
}

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

#[derive(Clone)]
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
    pub fn name(&self) -> String {
        let locked = self.parent.lock().unwrap();
        let inner = unsafe { sys::xls_vast_logic_ref_get_name(self.inner) };
        unsafe { c_str_to_rust(inner) }
    }

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

pub struct VastAlwaysBase {
    inner: *mut sys::CVastAlwaysBase,
    parent: Arc<Mutex<VastFilePtr>>,
}

pub struct VastStatementBlock {
    inner: *mut sys::CVastStatementBlock,
    parent: Arc<Mutex<VastFilePtr>>,
}

pub struct VastStatement {
    inner: *mut sys::CVastStatement,
    parent: Arc<Mutex<VastFilePtr>>,
}

impl VastModule {
    pub fn name(&self) -> String {
        let locked = self.parent.lock().unwrap();
        let inner = unsafe { sys::xls_vast_verilog_module_get_name(self.inner) };
        unsafe { c_str_to_rust(inner) }
    }

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

    /// Adds a member to this module which is an instantiation of another module
    /// -- this is described by the given `Instantiation` -- see
    /// `VastFile::make_instantiation`.
    pub fn add_member_instantiation(&mut self, instantiation: Instantiation) {
        let _locked = self.parent.lock().unwrap();
        unsafe {
            sys::xls_vast_verilog_module_add_member_instantiation(self.inner, instantiation.inner)
        }
    }

    /// Adds a "continuous assignment" member to this module; i.e. a statement
    /// of the form: `assign <lhs> = <rhs>;`
    ///
    /// Create a `ContinuousAssignment` structure that describes the assignment
    /// via `VastFile::make_continuous_assignment`.
    pub fn add_member_continuous_assignment(&mut self, assignment: ContinuousAssignment) {
        let _locked = self.parent.lock().unwrap();
        unsafe {
            sys::xls_vast_verilog_module_add_member_continuous_assignment(
                self.inner,
                assignment.inner,
            )
        }
    }

    pub fn add_reg(
        &mut self,
        name: &str,
        data_type: &VastDataType,
    ) -> Result<LogicRef, XlsynthError> {
        let c_name = CString::new(name).unwrap();
        let _locked = self.parent.lock().unwrap();
        let mut reg_ref_out: *mut sys::CVastLogicRef = std::ptr::null_mut();
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let success = unsafe {
            sys::xls_vast_verilog_module_add_reg(
                self.inner,
                c_name.as_ptr(),
                data_type.inner,
                &mut reg_ref_out,
                &mut error_out,
            )
        };
        if success {
            Ok(LogicRef {
                inner: reg_ref_out,
                parent: self.parent.clone(),
            })
        } else {
            Err(XlsynthError(unsafe { c_str_to_rust(error_out) }))
        }
    }

    fn add_always_block(
        &mut self,
        sensitivity_list: &[&Expr],
        is_ff: bool,
    ) -> Result<VastAlwaysBase, XlsynthError> {
        let _locked = self.parent.lock().unwrap();
        let mut expr_ptrs: Vec<*mut sys::CVastExpression> =
            sensitivity_list.iter().map(|expr| expr.inner).collect();
        let mut always_base_out: *mut sys::CVastAlwaysBase = std::ptr::null_mut();
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let success = unsafe {
            if is_ff {
                sys::xls_vast_verilog_module_add_always_ff(
                    self.inner,
                    expr_ptrs.as_mut_ptr(),
                    expr_ptrs.len(),
                    &mut always_base_out,
                    &mut error_out,
                )
            } else {
                sys::xls_vast_verilog_module_add_always_at(
                    self.inner,
                    expr_ptrs.as_mut_ptr(),
                    expr_ptrs.len(),
                    &mut always_base_out,
                    &mut error_out,
                )
            }
        };
        if success {
            Ok(VastAlwaysBase {
                inner: always_base_out,
                parent: self.parent.clone(),
            })
        } else {
            Err(XlsynthError(unsafe { c_str_to_rust(error_out) }))
        }
    }

    /// Note: this does not warn or error if you add it to a Verilog-based file.
    pub fn add_always_ff(
        &mut self,
        sensitivity_list: &[&Expr],
    ) -> Result<VastAlwaysBase, XlsynthError> {
        self.add_always_block(sensitivity_list, true)
    }

    pub fn add_always_at(
        &mut self,
        sensitivity_list: &[&Expr],
    ) -> Result<VastAlwaysBase, XlsynthError> {
        self.add_always_block(sensitivity_list, false)
    }
}

impl VastAlwaysBase {
    pub fn get_statement_block(&self) -> VastStatementBlock {
        let _locked = self.parent.lock().unwrap();
        let inner = unsafe { sys::xls_vast_always_base_get_statement_block(self.inner) };
        VastStatementBlock {
            inner,
            parent: self.parent.clone(),
        }
    }
}

impl VastStatementBlock {
    pub fn add_nonblocking_assignment(&mut self, lhs: &Expr, rhs: &Expr) -> VastStatement {
        let _locked = self.parent.lock().unwrap();
        let inner = unsafe {
            sys::xls_vast_statement_block_add_nonblocking_assignment(
                self.inner, lhs.inner, rhs.inner,
            )
        };
        VastStatement {
            inner,
            parent: self.parent.clone(),
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

    /// Creates a structure that describes an instantiation of a module that we
    /// want to create (as a member within some module, see
    /// `VastModule::add_member_instantiation`).
    ///
    /// Args:
    /// - `module_name`: The name of the module to instantiate.
    /// - `instance_name`: The name of the instance of the module to create.
    /// - `parameter_port_names`: The names of the `parameter`s of the module to
    ///   instantiate.
    /// - `parameter_expressions`: The expressions to use in instantiating the
    ///   parameters of the module.
    /// - `connection_port_names`: The names of the ports of the module to
    ///   instantiate.
    /// - `connection_expressions`: The expressions to use in instantiating the
    ///   ports of the module.
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

    pub fn make_slice_expr(&mut self, indexable: &IndexableExpr, hi: &Expr, lo: &Expr) -> Slice {
        let locked = self.ptr.lock().unwrap();
        let inner = unsafe {
            sys::xls_vast_verilog_file_make_slice(locked.0, indexable.inner, hi.inner, lo.inner)
        };
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

    pub fn make_index_expr(&mut self, indexable: &IndexableExpr, index: &Expr) -> Index {
        let locked = self.ptr.lock().unwrap();
        let inner = unsafe {
            sys::xls_vast_verilog_file_make_index(locked.0, indexable.inner, index.inner)
        };
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

    /// Internal helper for binary operators, users should prefer the
    /// `VastFile::make_*` methods below, such as `VastFile::make_add`.
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

    pub fn make_pos_edge(&mut self, expr: &Expr) -> Expr {
        let locked = self.ptr.lock().unwrap();
        let inner = unsafe { sys::xls_vast_verilog_file_make_pos_edge(locked.0, expr.inner) };
        Expr {
            inner,
            parent: self.ptr.clone(),
        }
    }

    pub fn make_nonblocking_assignment(&mut self, lhs: &Expr, rhs: &Expr) -> VastStatement {
        let locked = self.ptr.lock().unwrap();
        let inner = unsafe {
            sys::xls_vast_verilog_file_make_nonblocking_assignment(locked.0, lhs.inner, rhs.inner)
        };
        VastStatement {
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
