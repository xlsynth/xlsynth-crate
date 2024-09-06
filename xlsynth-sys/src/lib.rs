// SPDX-License-Identifier: Apache-2.0

//! Declarations for the C API for the XLS dynamic shared object.

extern crate libc;

#[repr(C)]
pub struct CIrValue {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CIrPackage {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CIrFunction {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CIrType {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CIrFunctionType {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

// "VAST" is the "Verilog AST" API which
#[repr(C)]
pub struct CVastFile {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CVastModule {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CVastDataType {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CVastLogicRef {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

pub type XlsFormatPreference = i32;

pub type VastFileType = i32;

extern "C" {
    pub fn xls_convert_dslx_to_ir(
        dslx: *const std::os::raw::c_char,
        path: *const std::os::raw::c_char,
        module_name: *const std::os::raw::c_char,
        dslx_stdlib_path: *const std::os::raw::c_char,
        additional_search_paths: *const *const std::os::raw::c_char,
        additional_search_paths_count: libc::size_t,
        error_out: *mut *mut std::os::raw::c_char,
        ir_out: *mut *mut std::os::raw::c_char,
    ) -> bool;
    pub fn xls_parse_typed_value(
        text: *const std::os::raw::c_char,
        error_out: *mut *mut std::os::raw::c_char,
        value_out: *mut *mut CIrValue,
    ) -> bool;
    pub fn xls_value_free(value: *mut CIrValue);
    pub fn xls_package_free(package: *mut CIrPackage);
    pub fn xls_c_str_free(c_str: *mut std::os::raw::c_char);
    pub fn xls_value_to_string(
        value: *const CIrValue,
        str_out: *mut *mut std::os::raw::c_char,
    ) -> bool;
    pub fn xls_format_preference_from_string(
        s: *const std::os::raw::c_char,
        error_out: *mut *mut std::os::raw::c_char,
        result_out: *mut XlsFormatPreference,
    ) -> bool;
    pub fn xls_value_to_string_format_preference(
        value: *const CIrValue,
        fmt: XlsFormatPreference,
        error_out: *mut *mut std::os::raw::c_char,
        str_out: *mut *mut std::os::raw::c_char,
    ) -> bool;
    pub fn xls_value_eq(value: *const CIrValue, value: *const CIrValue) -> bool;
    pub fn xls_parse_ir_package(
        ir: *const std::os::raw::c_char,
        filename: *const std::os::raw::c_char,
        error_out: *mut *mut std::os::raw::c_char,
        xls_package_out: *mut *mut CIrPackage,
    ) -> bool;
    pub fn xls_type_to_string(
        t: *const CIrType,
        error_out: *mut *mut std::os::raw::c_char,
        result_out: *mut *mut std::os::raw::c_char,
    ) -> bool;
    pub fn xls_package_get_type_for_value(
        package: *const CIrPackage,
        value: *const CIrValue,
        error_out: *mut *mut std::os::raw::c_char,
        result_out: *mut *mut CIrType,
    ) -> bool;
    pub fn xls_package_get_function(
        package: *const CIrPackage,
        function_name: *const std::os::raw::c_char,
        error_out: *mut *mut std::os::raw::c_char,
        result_out: *mut *mut CIrFunction,
    ) -> bool;
    pub fn xls_function_get_type(
        function: *const CIrFunction,
        error_out: *mut *mut std::os::raw::c_char,
        xls_fn_type_out: *mut *mut CIrFunctionType,
    ) -> bool;
    pub fn xls_function_type_to_string(
        t: *const CIrFunctionType,
        error_out: *mut *mut std::os::raw::c_char,
        string_out: *mut *mut std::os::raw::c_char,
    ) -> bool;
    pub fn xls_function_get_name(
        function: *const CIrFunction,
        error_out: *mut *mut std::os::raw::c_char,
        name_out: *mut *mut std::os::raw::c_char,
    ) -> bool;
    pub fn xls_interpret_function(
        function: *const CIrFunction,
        argc: libc::size_t,
        args: *const *const CIrValue,
        error_out: *mut *mut std::os::raw::c_char,
        result_out: *mut *mut CIrValue,
    ) -> bool;
    pub fn xls_optimize_ir(
        ir: *const std::os::raw::c_char,
        top: *const std::os::raw::c_char,
        error_out: *mut *mut std::os::raw::c_char,
        ir_out: *mut *mut std::os::raw::c_char,
    ) -> bool;
    pub fn xls_mangle_dslx_name(
        module_name: *const std::os::raw::c_char,
        function_name: *const std::os::raw::c_char,
        error_out: *mut *mut std::os::raw::c_char,
        mangled_out: *mut *mut std::os::raw::c_char,
    ) -> bool;
    pub fn xls_package_to_string(
        p: *const CIrPackage,
        string_out: *mut *mut std::os::raw::c_char,
    ) -> bool;

    // -- VAST APIs

    pub fn xls_vast_make_verilog_file(file_type: VastFileType) -> *mut CVastFile;
    pub fn xls_vast_verilog_file_free(f: *mut CVastFile);
    pub fn xls_vast_verilog_file_add_module(
        f: *mut CVastFile,
        name: *const std::os::raw::c_char,
    ) -> *mut CVastModule;
    pub fn xls_vast_verilog_file_make_scalar_type(f: *mut CVastFile) -> *mut CVastDataType;
    pub fn xls_vast_verilog_file_make_bit_vector_type(
        f: *mut CVastFile,
        bit_count: i64,
        is_signed: bool,
    ) -> *mut CVastDataType;
    pub fn xls_vast_verilog_module_add_input(
        m: *mut CVastModule,
        name: *const std::os::raw::c_char,
        type_: *mut CVastDataType,
    ) -> *mut CVastLogicRef;
    pub fn xls_vast_verilog_module_add_output(
        m: *mut CVastModule,
        name: *const std::os::raw::c_char,
        type_: *mut CVastDataType,
    ) -> *mut CVastLogicRef;
    pub fn xls_vast_verilog_module_add_wire(
        m: *mut CVastModule,
        name: *const std::os::raw::c_char,
        type_: *mut CVastDataType,
    ) -> *mut CVastLogicRef;
    pub fn xls_vast_verilog_file_add_include(f: *mut CVastFile, path: *const std::os::raw::c_char);
    pub fn xls_vast_verilog_file_emit(f: *const CVastFile) -> *mut std::os::raw::c_char;
}

pub const DSLX_STDLIB_PATH: &str = env!("DSLX_STDLIB_PATH");
pub const XLS_DSO_PATH: &str = env!("XLS_DSO_PATH");
