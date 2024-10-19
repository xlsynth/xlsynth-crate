// SPDX-License-Identifier: Apache-2.0

//! Declarations for the C API for the XLS dynamic shared object.

extern crate libc;

#[repr(C)]
pub struct CIrValue {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CIrBits {
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
pub struct CVastExpression {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CVastLiteral {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CVastLogicRef {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CVastInstantiation {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CVastContinuousAssignment {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CVastSlice {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CVastIndexableExpression {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

// -- DSLX

#[repr(C)]
pub struct CDslxImportData {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CDslxTypecheckedModule {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CDslxModule {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CDslxTypeInfo {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CDslxStructDef {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CDslxEnumDef {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CDslxTypeAlias {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CDslxType {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CDslxTypeDim {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CDslxExpr {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CDslxInterpValue {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CDslxEnumMember {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CDslxStructMember {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CDslxTypeAnnotation {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

pub type XlsFormatPreference = i32;

pub type VastFileType = i32;

pub type DslxTypeDefinitionKind = i32;

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

    pub fn xls_value_get_bits(
        value: *const CIrValue,
        error_out: *mut *mut std::os::raw::c_char,
        bits_out: *mut *mut CIrBits,
    ) -> bool;
    pub fn xls_bits_free(bits: *mut CIrBits);
    pub fn xls_bits_get_bit_count(bits: *const CIrBits) -> i64;

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

    // - Node creation
    pub fn xls_vast_verilog_file_make_scalar_type(f: *mut CVastFile) -> *mut CVastDataType;
    pub fn xls_vast_verilog_file_make_bit_vector_type(
        f: *mut CVastFile,
        bit_count: i64,
        is_signed: bool,
    ) -> *mut CVastDataType;
    pub fn xls_vast_verilog_file_make_continuous_assignment(
        f: *mut CVastFile,
        lhs: *mut CVastExpression,
        rhs: *mut CVastExpression,
    ) -> *mut CVastContinuousAssignment;
    pub fn xls_vast_verilog_file_make_slice_i64(
        f: *mut CVastFile,
        subject: *mut CVastIndexableExpression,
        hi: i64,
        lo: i64,
    ) -> *mut CVastSlice;
    pub fn xls_vast_verilog_file_make_instantiation(
        f: *mut CVastFile,
        module_name: *const std::os::raw::c_char,
        instance_name: *const std::os::raw::c_char,
        parameter_port_names: *const *const std::os::raw::c_char,
        parameter_expressions: *const *const CVastExpression,
        parameter_count: libc::size_t,
        connection_port_names: *const *const std::os::raw::c_char,
        connection_expressions: *const *const CVastExpression,
        connection_count: libc::size_t,
    ) -> *mut CVastInstantiation;
    pub fn xls_vast_verilog_file_make_literal(
        f: *mut CVastFile,
        bits: *const CIrBits,
        format_preference: XlsFormatPreference,
        emit_bit_count: bool,
        error_out: *mut *mut std::os::raw::c_char,
        literal_out: *mut *mut CVastLiteral,
    ) -> bool;

    // - Module additions
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
    pub fn xls_vast_verilog_module_add_member_instantiation(
        m: *mut CVastModule,
        inst: *mut CVastInstantiation,
    );
    pub fn xls_vast_verilog_module_add_member_continuous_assignment(
        m: *mut CVastModule,
        ca: *mut CVastContinuousAssignment,
    );

    // - Expression conversions
    pub fn xls_vast_logic_ref_as_indexable_expression(
        v: *mut CVastLogicRef,
    ) -> *mut CVastIndexableExpression;
    pub fn xls_vast_literal_as_expression(v: *mut CVastLiteral) -> *mut CVastExpression;
    pub fn xls_vast_logic_ref_as_expression(v: *mut CVastLogicRef) -> *mut CVastExpression;
    pub fn xls_vast_slice_as_expression(v: *mut CVastSlice) -> *mut CVastExpression;

    pub fn xls_vast_verilog_file_add_include(f: *mut CVastFile, path: *const std::os::raw::c_char);
    pub fn xls_vast_verilog_file_emit(f: *const CVastFile) -> *mut std::os::raw::c_char;

    // -- DSLX

    pub fn xls_dslx_import_data_create(
        dslx_stdlib_path: *const std::os::raw::c_char,
        additional_search_paths: *const *const std::os::raw::c_char,
        additional_search_paths_count: libc::size_t,
    ) -> *mut CDslxImportData;
    pub fn xls_dslx_import_data_free(data: *mut CDslxImportData);

    pub fn xls_dslx_parse_and_typecheck(
        text: *const std::os::raw::c_char,
        path: *const std::os::raw::c_char,
        module_name: *const std::os::raw::c_char,
        import_data: *const CDslxImportData,
        error_out: *mut *mut std::os::raw::c_char,
        typechecked_module_out: *mut *mut CDslxTypecheckedModule,
    ) -> bool;

    pub fn xls_dslx_typechecked_module_free(module: *mut CDslxTypecheckedModule);

    pub fn xls_dslx_typechecked_module_get_module(
        module: *mut CDslxTypecheckedModule,
    ) -> *mut CDslxModule;

    pub fn xls_dslx_typechecked_module_get_type_info(
        module: *mut CDslxTypecheckedModule,
    ) -> *mut CDslxTypeInfo;

    pub fn xls_dslx_module_get_type_definition_count(module: *const CDslxModule) -> i64;

    pub fn xls_dslx_module_get_type_definition_kind(
        module: *const CDslxModule,
        i: i64,
    ) -> DslxTypeDefinitionKind;

    pub fn xls_dslx_module_get_type_definition_as_struct_def(
        module: *const CDslxModule,
        i: i64,
    ) -> *mut CDslxStructDef;
    pub fn xls_dslx_module_get_type_definition_as_enum_def(
        module: *const CDslxModule,
        i: i64,
    ) -> *mut CDslxEnumDef;
    pub fn xls_dslx_module_get_type_definition_as_type_alias(
        module: *const CDslxModule,
        i: i64,
    ) -> *mut CDslxTypeAlias;

    pub fn xls_dslx_type_info_get_type_struct_def(
        type_info: *mut CDslxTypeInfo,
        enum_def: *mut CDslxStructDef,
    ) -> *mut CDslxType;
    pub fn xls_dslx_type_info_get_type_enum_def(
        type_info: *mut CDslxTypeInfo,
        enum_def: *mut CDslxEnumDef,
    ) -> *mut CDslxType;
    pub fn xls_dslx_type_info_get_type_struct_member(
        type_info: *mut CDslxTypeInfo,
        member: *mut CDslxStructMember,
    ) -> *mut CDslxType;
    pub fn xls_dslx_type_info_get_type_type_alias(
        type_info: *mut CDslxTypeInfo,
        enum_def: *mut CDslxTypeAlias,
    ) -> *mut CDslxType;

    /// Gets the concrete type for a TypeAnnotation AST node.
    pub fn xls_dslx_type_info_get_type_type_annotation(
        type_info: *mut CDslxTypeInfo,
        type_annotation: *mut CDslxTypeAnnotation,
    ) -> *mut CDslxType;

    // -- StructDef

    pub fn xls_dslx_struct_def_get_identifier(
        struct_def: *const CDslxStructDef,
    ) -> *mut std::os::raw::c_char;

    pub fn xls_dslx_struct_def_is_parametric(struct_def: *const CDslxStructDef) -> bool;

    pub fn xls_dslx_struct_def_get_member_count(struct_def: *const CDslxStructDef) -> i64;

    pub fn xls_dslx_struct_def_get_member(
        struct_def: *const CDslxStructDef,
        i: i64,
    ) -> *mut CDslxStructMember;

    pub fn xls_dslx_struct_member_get_name(
        member: *const CDslxStructMember,
    ) -> *mut std::os::raw::c_char;

    pub fn xls_dslx_struct_member_get_type(
        member: *const CDslxStructMember,
    ) -> *mut CDslxTypeAnnotation;

    // -- EnumDef

    pub fn xls_dslx_enum_def_get_identifier(
        enum_def: *const CDslxEnumDef,
    ) -> *mut std::os::raw::c_char;

    pub fn xls_dslx_enum_def_get_member_count(enum_def: *const CDslxEnumDef) -> i64;

    pub fn xls_dslx_enum_def_get_member(
        enum_def: *const CDslxEnumDef,
        i: i64,
    ) -> *mut CDslxEnumMember;

    pub fn xls_dslx_enum_member_get_name(
        member: *const CDslxEnumMember,
    ) -> *mut std::os::raw::c_char;

    pub fn xls_dslx_enum_member_get_value(member: *const CDslxEnumMember) -> *mut CDslxExpr;

    // --

    pub fn xls_dslx_interp_value_free(value: *mut CDslxInterpValue);

    pub fn xls_dslx_interp_value_convert_to_ir(
        value: *mut CDslxInterpValue,
        error_out: *mut *mut std::os::raw::c_char,
        result_out: *mut *mut CIrValue,
    ) -> bool;

    pub fn xls_dslx_type_to_string(
        type_: *const CDslxType,
        error_out: *mut *mut std::os::raw::c_char,
        result_out: *mut *mut std::os::raw::c_char,
    ) -> bool;

    pub fn xls_dslx_type_info_get_const_expr(
        type_info: *mut CDslxTypeInfo,
        expr: *mut CDslxExpr,
        error_out: *mut *mut std::os::raw::c_char,
        result_out: *mut *mut CDslxInterpValue,
    ) -> bool;

    pub fn xls_dslx_type_get_total_bit_count(
        type_: *const CDslxType,
        error_out: *mut *mut std::os::raw::c_char,
        result_out: *mut i64,
    ) -> bool;

    pub fn xls_dslx_type_is_signed_bits(
        type_: *const CDslxType,
        error_out: *mut *mut std::os::raw::c_char,
        result_out: *mut bool,
    ) -> bool;

    pub fn xls_dslx_type_is_bits_like(
        type_: *const CDslxType,
        is_signed: *mut *mut CDslxTypeDim,
        size: *mut *mut CDslxTypeDim,
    ) -> bool;

    pub fn xls_dslx_type_dim_is_parametric(dim: *const CDslxTypeDim) -> bool;
    pub fn xls_dslx_type_dim_get_as_bool(
        dim: *const CDslxTypeDim,
        error_out: *mut *mut std::os::raw::c_char,
        result_out: *mut bool,
    ) -> bool;
    pub fn xls_dslx_type_dim_get_as_int64(
        dim: *const CDslxTypeDim,
        error_out: *mut *mut std::os::raw::c_char,
        result_out: *mut i64,
    ) -> bool;
    pub fn xls_dslx_type_dim_free(dim: *mut CDslxTypeDim);
}

pub const DSLX_STDLIB_PATH: &str = env!("DSLX_STDLIB_PATH");
pub const XLS_DSO_PATH: &str = env!("XLS_DSO_PATH");
