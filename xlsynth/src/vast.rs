// SPDX-License-Identifier: Apache-2.0

//! APIs that wrap the Verilog AST building facilities inside of XLS.

#![allow(unused)]

use xlsynth_sys as sys;

use std::{
    ffi::CString,
    os::raw::c_char,
    sync::{Arc, Mutex},
};

use crate::{c_str_to_rust, ir_value::IrFormatPreference, XlsynthError};

pub(crate) struct VastFilePtr(pub *mut sys::CVastFile);

impl Drop for VastFilePtr {
    fn drop(&mut self) {
        unsafe { sys::xls_vast_verilog_file_free(self.0) }
    }
}

pub struct VastDataType {
    pub(crate) inner: *mut sys::CVastDataType,
    pub(crate) parent: Arc<Mutex<VastFilePtr>>,
}

pub struct LogicRef {
    pub(crate) inner: *mut sys::CVastLogicRef,
    pub(crate) parent: Arc<Mutex<VastFilePtr>>,
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
    pub(crate) inner: *mut sys::CVastExpression,
    pub(crate) parent: Arc<Mutex<VastFilePtr>>,
}

pub struct IndexableExpr {
    pub(crate) inner: *mut sys::CVastIndexableExpression,
    pub(crate) parent: Arc<Mutex<VastFilePtr>>,
}

pub struct VastModule {
    pub(crate) inner: *mut sys::CVastModule,
    pub(crate) parent: Arc<Mutex<VastFilePtr>>,
}

pub struct Slice {
    pub(crate) inner: *mut sys::CVastSlice,
    pub(crate) parent: Arc<Mutex<VastFilePtr>>,
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
    pub(crate) inner: *mut sys::CVastIndex,
    pub(crate) parent: Arc<Mutex<VastFilePtr>>,
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
    pub(crate) inner: *mut sys::CVastInstantiation,
    pub(crate) parent: Arc<Mutex<VastFilePtr>>,
}

pub struct ContinuousAssignment {
    pub(crate) inner: *mut sys::CVastContinuousAssignment,
    pub(crate) parent: Arc<Mutex<VastFilePtr>>,
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
    pub(crate) ptr: Arc<Mutex<VastFilePtr>>,
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
        connection_expressions: &[&Expr],
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

        let c_param_expr_ptrs = to_expr_ptrs(parameter_expressions);
        let c_conn_expr_ptrs = to_expr_ptrs(connection_expressions);

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
        let v = crate::xls_parse_typed_value(s).unwrap();
        let mut fmt = crate::xls_format_preference_from_string(fmt.to_string()).unwrap();

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vast() {
        let mut file = VastFile::new(VastFileType::Verilog);
        let mut module = file.add_module("main");
        let input_type = file.make_bit_vector_type(32, false);
        let output_type = file.make_scalar_type();
        module.add_input("in", &input_type);
        module.add_output("out", &output_type);
        let verilog = file.emit();
        let want = "module main(\n  input wire [31:0] in,\n  output wire out\n);\n\nendmodule\n";
        assert_eq!(verilog, want);
    }

    #[test]
    fn test_continuous_assignment_of_slice() {
        let mut file = VastFile::new(VastFileType::Verilog);
        let mut module = file.add_module("my_module");
        let input_type = file.make_bit_vector_type(8, false);
        let output_type = file.make_bit_vector_type(4, false);
        let input = module.add_input("my_input", &input_type);
        let output = module.add_output("my_output", &output_type);
        let slice = file.make_slice(&input.to_indexable_expr(), 3, 0);
        let assignment = file.make_continuous_assignment(&output.to_expr(), &slice.to_expr());
        module.add_member_continuous_assignment(assignment);
        let verilog = file.emit();
        let want = "module my_module(
  input wire [7:0] my_input,
  output wire [3:0] my_output
);
  assign my_output = my_input[3:0];
endmodule
";
        assert_eq!(verilog, want);
    }

    #[test]
    fn test_instantiation() {
        let mut file = VastFile::new(VastFileType::Verilog);

        let data_type = file.make_bit_vector_type(8, false);

        let mut a_module = file.add_module("A");
        a_module.add_output("bus", &data_type);

        let mut b_module = file.add_module("B");
        let bus = b_module.add_wire("bus", &data_type);

        let param_value = file
            .make_literal("bits[32]:42", &IrFormatPreference::UnsignedDecimal)
            .unwrap();

        b_module.add_member_instantiation(file.make_instantiation(
            "A",
            "a_i",
            &["a_param"],
            &[&param_value],
            &["bus"],
            &[&bus.to_expr()],
        ));

        let verilog = file.emit();
        let want = "module A(
  output wire [7:0] bus
);

endmodule
module B;
  wire [7:0] bus;
  A #(
    .a_param(32'd42)
  ) a_i (
    .bus(bus)
  );
endmodule
";
        assert_eq!(verilog, want);
    }

    #[test]
    fn test_literal() {
        let mut file = VastFile::new(VastFileType::Verilog);
        let mut module = file.add_module("my_module");
        let wire = module.add_wire("bus", &file.make_bit_vector_type(128, false));
        let literal = file
            .make_literal(
                "bits[128]:0xFFEEDDCCBBAA99887766554433221100",
                &IrFormatPreference::Hex,
            )
            .unwrap();
        let assignment = file.make_continuous_assignment(&wire.to_expr(), &literal);
        module.add_member_continuous_assignment(assignment);
        let verilog = file.emit();
        let want = "module my_module;
  wire [127:0] bus;
  assign bus = 128'hffee_ddcc_bbaa_9988_7766_5544_3322_1100;
endmodule
";
        assert_eq!(verilog, want);
    }

    /// Tests that we can make a port with an external-package-defined struct as
    /// the type, and we also place it in a packed array.
    #[test]
    fn test_port_with_external_package_struct() {
        let mut file = VastFile::new(VastFileType::Verilog);
        let mut module = file.add_module("my_module");
        let my_struct = file.make_extern_package_type("mypack", "mystruct_t");
        let input_type = file.make_packed_array_type(my_struct, &[2, 3, 4]);
        module.add_input("my_input", &input_type);
        let want = "module my_module(
  input mypack::mystruct_t [1:0][2:0][3:0] my_input
);

endmodule
";
        assert_eq!(file.emit(), want);
    }

    /// Tests that we can build a module with a simple concatenation.
    #[test]
    fn test_simple_concat() {
        let mut file = VastFile::new(VastFileType::Verilog);
        let mut module = file.add_module("my_module");
        let input_type = file.make_bit_vector_type(8, false);
        let output_type = file.make_bit_vector_type(16, false);
        let input = module.add_input("my_input", &input_type);
        let output = module.add_output("my_output", &output_type);
        let concat = file.make_concat(&[&input.to_expr(), &input.to_expr()]);
        let assignment = file.make_continuous_assignment(&output.to_expr(), &concat);
        module.add_member_continuous_assignment(assignment);
        let verilog = file.emit();
        let want = "module my_module(
  input wire [7:0] my_input,
  output wire [15:0] my_output
);
  assign my_output = {my_input, my_input};
endmodule
";
        assert_eq!(verilog, want);
    }

    /// Tests that we can reference a slice of a multidimensional packed array
    /// on the LHS or RHS of an assign statement.
    #[test]
    fn test_slice_on_both_sides_of_assignment() {
        let want = "module my_module;
  wire [1:0][2:0][4:0] a;
  wire [1:0] b;
  wire [2:0] c;
  assign a[1][2][3:4] = b[1:0];
  assign a[3:4] = c[2:1];
endmodule
";

        let mut file = VastFile::new(VastFileType::Verilog);
        let mut module = file.add_module("my_module");
        let u2 = file.make_bit_vector_type(2, false);
        let a_type = file.make_packed_array_type(u2, &[3, 5]);
        let b_type = file.make_bit_vector_type(2, false);
        let c_type = file.make_bit_vector_type(3, false);
        let a = module.add_wire("a", &a_type);
        let b = module.add_wire("b", &b_type);
        let c = module.add_wire("c", &c_type);

        // First assignment.
        {
            let a_1 = file.make_index(&a.to_indexable_expr(), 1);
            let a_2 = file.make_index(&a_1.to_indexable_expr(), 2);
            let a_lhs = file.make_slice(&a_2.to_indexable_expr(), 3, 4);
            let b_slice = file.make_slice(&b.to_indexable_expr(), 1, 0);
            let assignment = file.make_continuous_assignment(&a_lhs.to_expr(), &b_slice.to_expr());
            module.add_member_continuous_assignment(assignment);
        }

        // Second assignment.
        {
            let a_lhs = file.make_slice(&a.to_indexable_expr(), 3, 4);
            let c_slice = file.make_slice(&c.to_indexable_expr(), 2, 1);
            let assignment = file.make_continuous_assignment(&a_lhs.to_expr(), &c_slice.to_expr());
            module.add_member_continuous_assignment(assignment);
        }

        let verilog = file.emit();
        assert_eq!(verilog, want);
    }
}
