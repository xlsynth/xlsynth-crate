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
pub struct CIrFunctionJit {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CTraceMessage {
    pub message: *mut std::os::raw::c_char,
    pub verbosity: i64,
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
pub struct CVastIndex {
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

#[repr(C)]
pub struct CDslxTypeRefTypeAnnotation {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CDslxTypeRef {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CDslxImport {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CDslxColonRef {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CDslxTypeDefinition {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CDslxConstantDef {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CDslxModuleMember {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CScheduleAndCodegenResult {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CIrBuilderBase {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CIrBValue {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

#[repr(C)]
pub struct CIrFunctionBuilder {
    _private: [u8; 0], // Ensures the struct cannot be instantiated
}

pub type XlsFormatPreference = i32;

pub type VastFileType = i32;

pub type VastOperatorKind = i32;

pub type DslxTypeDefinitionKind = i32;

pub type DslxModuleMemberKind = i32;

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

    pub fn xls_convert_dslx_to_ir_with_warnings(
        dslx: *const std::os::raw::c_char,
        path: *const std::os::raw::c_char,
        module_name: *const std::os::raw::c_char,
        dslx_stdlib_path: *const std::os::raw::c_char,
        additional_search_paths: *const *const std::os::raw::c_char,
        additional_search_paths_count: libc::size_t,
        enable_warnings: *const *const std::os::raw::c_char,
        enable_warnings_count: libc::size_t,
        disable_warnings: *const *const std::os::raw::c_char,
        disable_warnings_count: libc::size_t,
        warnings_as_errors: bool,
        warnings_out: *mut *mut *mut std::os::raw::c_char,
        warnings_out_count: *mut libc::size_t,
        error_out: *mut *mut std::os::raw::c_char,
        ir_out: *mut *mut std::os::raw::c_char,
    ) -> bool;

    pub fn xls_parse_typed_value(
        text: *const std::os::raw::c_char,
        error_out: *mut *mut std::os::raw::c_char,
        value_out: *mut *mut CIrValue,
    ) -> bool;
    pub fn xls_value_free(value: *mut CIrValue);

    pub fn xls_value_clone(value: *const CIrValue) -> *mut CIrValue;

    // Extracts a bits value from a (boxed) value or gives an error.
    pub fn xls_value_get_bits(
        value: *const CIrValue,
        error_out: *mut *mut std::os::raw::c_char,
        bits_out: *mut *mut CIrBits,
    ) -> bool;

    // Turns a span of IR values into a tuple value.
    pub fn xls_value_make_tuple(
        value_count: libc::size_t,
        values: *const *const CIrValue,
    ) -> *mut CIrValue;

    /// Returns an error:
    /// * if the elements do not all have the same type, or
    /// * if the array is empty (because then we cannot determine the element
    ///   type)
    pub fn xls_value_make_array(
        element_count: libc::size_t,
        elements: *const *const CIrValue,
        error_out: *mut *mut std::os::raw::c_char,
        result_out: *mut *mut CIrValue,
    ) -> bool;

    // Extracts an element from a tuple/array value or gives an error (e.g. if this
    // value is not a tuple/array or the index is out of bounds).
    pub fn xls_value_get_element(
        tuple: *const CIrValue,
        index: libc::size_t,
        error_out: *mut *mut std::os::raw::c_char,
        element_out: *mut *mut CIrValue,
    ) -> bool;

    pub fn xls_value_get_element_count(
        value: *const CIrValue,
        error_out: *mut *mut std::os::raw::c_char,
        count_out: *mut i64,
    ) -> bool;

    // Creates a bits value (via an unsigned integer) that is boxed in an IrValue.
    pub fn xls_value_make_ubits(
        bit_count: i64,
        value: u64,
        error_out: *mut *mut std::os::raw::c_char,
        result_out: *mut *mut CIrValue,
    ) -> bool;

    // Creates a bits value (via a signed integer) that is boxed in an IrValue.
    pub fn xls_value_make_sbits(
        bit_count: i64,
        value: i64,
        error_out: *mut *mut std::os::raw::c_char,
        result_out: *mut *mut CIrValue,
    ) -> bool;

    // Boxes an IR bits object into an IR value.
    pub fn xls_value_from_bits(bits: *const CIrBits) -> *mut CIrValue;

    pub fn xls_bits_make_ubits(
        bit_count: i64,
        value: u64,
        error_out: *mut *mut std::os::raw::c_char,
        bits_out: *mut *mut CIrBits,
    ) -> bool;

    pub fn xls_bits_make_sbits(
        bit_count: i64,
        value: i64,
        error_out: *mut *mut std::os::raw::c_char,
        bits_out: *mut *mut CIrBits,
    ) -> bool;

    pub fn xls_bits_free(bits: *mut CIrBits);
    pub fn xls_bits_get_bit_count(bits: *const CIrBits) -> i64;
    pub fn xls_bits_get_bit(bits: *const CIrBits, index: i64) -> bool;
    pub fn xls_bits_eq(bits: *const CIrBits, other: *const CIrBits) -> bool;
    pub fn xls_bits_to_debug_string(bits: *const CIrBits) -> *mut std::os::raw::c_char;
    pub fn xls_bits_add(lhs: *const CIrBits, rhs: *const CIrBits) -> *mut CIrBits;
    pub fn xls_bits_umul(lhs: *const CIrBits, rhs: *const CIrBits) -> *mut CIrBits;
    pub fn xls_bits_smul(lhs: *const CIrBits, rhs: *const CIrBits) -> *mut CIrBits;
    pub fn xls_bits_negate(bits: *const CIrBits) -> *mut CIrBits;
    pub fn xls_bits_abs(bits: *const CIrBits) -> *mut CIrBits;
    pub fn xls_bits_not(bits: *const CIrBits) -> *mut CIrBits;
    pub fn xls_bits_and(lhs: *const CIrBits, rhs: *const CIrBits) -> *mut CIrBits;
    pub fn xls_bits_or(lhs: *const CIrBits, rhs: *const CIrBits) -> *mut CIrBits;
    pub fn xls_bits_xor(lhs: *const CIrBits, rhs: *const CIrBits) -> *mut CIrBits;

    pub fn xls_bits_shift_left_logical(bits: *const CIrBits, shift_amount: i64) -> *mut CIrBits;
    pub fn xls_bits_shift_right_logical(bits: *const CIrBits, shift_amount: i64) -> *mut CIrBits;
    pub fn xls_bits_shift_right_arithmetic(bits: *const CIrBits, shift_amount: i64)
        -> *mut CIrBits;

    // struct xls_bits* xls_bits_width_slice(const struct xls_bits* bits, int64_t
    // start, int64_t width);

    pub fn xls_bits_width_slice(bits: *const CIrBits, start: i64, width: i64) -> *mut CIrBits;

    pub fn xls_package_free(package: *mut CIrPackage);
    pub fn xls_c_str_free(c_str: *mut std::os::raw::c_char);
    pub fn xls_c_strs_free(c_strs: *mut *mut std::os::raw::c_char, count: libc::size_t);
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
    pub fn xls_bits_to_string(
        bits: *const CIrBits,
        fmt: XlsFormatPreference,
        include_bit_count: bool,
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

    pub fn xls_make_function_jit(
        function: *const CIrFunction,
        error_out: *mut *mut std::os::raw::c_char,
        result_out: *mut *mut CIrFunctionJit,
    ) -> bool;
    pub fn xls_function_jit_free(jit: *mut CIrFunctionJit);
    pub fn xls_function_jit_run(
        jit: *const CIrFunctionJit,
        argc: libc::size_t,
        args: *const *const CIrValue,
        error_out: *mut *mut std::os::raw::c_char,
        trace_messages_out: *mut *mut CTraceMessage,
        trace_messages_count_out: *mut libc::size_t,
        assert_messages_out: *mut *mut *mut std::os::raw::c_char,
        assert_messages_count_out: *mut libc::size_t,
        result_out: *mut *mut CIrValue,
    ) -> bool;
    pub fn xls_trace_messages_free(trace_messages: *mut CTraceMessage, count: libc::size_t);

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
    pub fn xls_vast_verilog_file_make_extern_package_type(
        f: *mut CVastFile,
        package_name: *const std::os::raw::c_char,
        type_name: *const std::os::raw::c_char,
    ) -> *mut CVastDataType;
    pub fn xls_vast_verilog_file_make_packed_array_type(
        f: *mut CVastFile,
        element_type: *mut CVastDataType,
        dims: *const i64,
        dim_count: libc::size_t,
    ) -> *mut CVastDataType;

    pub fn xls_vast_verilog_file_make_continuous_assignment(
        f: *mut CVastFile,
        lhs: *mut CVastExpression,
        rhs: *mut CVastExpression,
    ) -> *mut CVastContinuousAssignment;

    pub fn xls_vast_verilog_file_make_concat(
        f: *mut CVastFile,
        expressions: *mut *mut CVastExpression,
        expression_count: libc::size_t,
    ) -> *mut CVastExpression;

    pub fn xls_vast_verilog_file_make_slice_i64(
        f: *mut CVastFile,
        subject: *mut CVastIndexableExpression,
        hi: i64,
        lo: i64,
    ) -> *mut CVastSlice;

    pub fn xls_vast_verilog_file_make_index(
        f: *mut CVastFile,
        subject: *mut CVastIndexableExpression,
        index: *mut CVastExpression,
    ) -> *mut CVastIndex;
    pub fn xls_vast_verilog_file_make_index_i64(
        f: *mut CVastFile,
        subject: *mut CVastIndexableExpression,
        index: i64,
    ) -> *mut CVastIndex;

    pub fn xls_vast_verilog_file_make_unary(
        f: *mut CVastFile,
        arg: *mut CVastExpression,
        op: VastOperatorKind,
    ) -> *mut CVastExpression;

    pub fn xls_vast_verilog_file_make_binary(
        f: *mut CVastFile,
        lhs: *mut CVastExpression,
        rhs: *mut CVastExpression,
        op: VastOperatorKind,
    ) -> *mut CVastExpression;

    pub fn xls_vast_verilog_file_make_ternary(
        f: *mut CVastFile,
        cond: *mut CVastExpression,
        consequent: *mut CVastExpression,
        alternate: *mut CVastExpression,
    ) -> *mut CVastExpression;

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
    pub fn xls_vast_index_as_indexable_expression(
        v: *mut CVastIndex,
    ) -> *mut CVastIndexableExpression;

    pub fn xls_vast_literal_as_expression(v: *mut CVastLiteral) -> *mut CVastExpression;
    pub fn xls_vast_logic_ref_as_expression(v: *mut CVastLogicRef) -> *mut CVastExpression;
    pub fn xls_vast_slice_as_expression(v: *mut CVastSlice) -> *mut CVastExpression;
    pub fn xls_vast_index_as_expression(v: *mut CVastIndex) -> *mut CVastExpression;

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

    // bool xls_schedule_and_codegen_package(
    // struct xls_package* p, const char* scheduling_options_flags_proto,
    // const char* codegen_flags_proto, bool with_delay_model, char** error_out,
    // struct xls_schedule_and_codegen_result** result_out);
    pub fn xls_schedule_and_codegen_package(
        p: *mut CIrPackage,
        scheduling_options_flags_proto: *const std::os::raw::c_char,
        codegen_flags_proto: *const std::os::raw::c_char,
        with_delay_model: bool,
        error_out: *mut *mut std::os::raw::c_char,
        result_out: *mut *mut CScheduleAndCodegenResult,
    ) -> bool;

    pub fn xls_schedule_and_codegen_result_get_verilog_text(
        result: *mut CScheduleAndCodegenResult,
    ) -> *mut std::os::raw::c_char;

    pub fn xls_schedule_and_codegen_result_free(result: *mut CScheduleAndCodegenResult);

    pub fn xls_dslx_typechecked_module_free(module: *mut CDslxTypecheckedModule);

    pub fn xls_dslx_typechecked_module_get_module(
        module: *mut CDslxTypecheckedModule,
    ) -> *mut CDslxModule;

    pub fn xls_dslx_typechecked_module_get_type_info(
        module: *mut CDslxTypecheckedModule,
    ) -> *mut CDslxTypeInfo;

    pub fn xls_dslx_module_get_name(module: *const CDslxModule) -> *mut std::os::raw::c_char;

    pub fn xls_dslx_module_get_type_definition_count(module: *const CDslxModule) -> i64;

    pub fn xls_dslx_module_get_member_count(module: *const CDslxModule) -> i64;

    pub fn xls_dslx_module_get_member(module: *const CDslxModule, i: i64)
        -> *mut CDslxModuleMember;

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

    // -- xls_dslx_module_member
    pub fn xls_dslx_module_member_get_kind(
        member: *const CDslxModuleMember,
    ) -> DslxModuleMemberKind;
    pub fn xls_dslx_module_member_get_constant_def(
        member: *const CDslxModuleMember,
    ) -> *mut CDslxConstantDef;
    pub fn xls_dslx_module_member_get_type_alias(
        member: *const CDslxModuleMember,
    ) -> *mut CDslxTypeAlias;
    pub fn xls_dslx_module_member_get_struct_def(
        member: *const CDslxModuleMember,
    ) -> *mut CDslxStructDef;
    pub fn xls_dslx_module_member_get_enum_def(
        member: *const CDslxModuleMember,
    ) -> *mut CDslxEnumDef;

    pub fn xls_dslx_colon_ref_get_attr(
        colon_ref: *const CDslxColonRef,
    ) -> *mut std::os::raw::c_char;

    pub fn xls_dslx_type_info_get_type_struct_def(
        type_info: *mut CDslxTypeInfo,
        node: *mut CDslxStructDef,
    ) -> *mut CDslxType;
    pub fn xls_dslx_type_info_get_type_enum_def(
        type_info: *mut CDslxTypeInfo,
        node: *mut CDslxEnumDef,
    ) -> *mut CDslxType;
    pub fn xls_dslx_type_info_get_type_struct_member(
        type_info: *mut CDslxTypeInfo,
        member: *mut CDslxStructMember,
    ) -> *mut CDslxType;
    pub fn xls_dslx_type_info_get_type_type_alias(
        type_info: *mut CDslxTypeInfo,
        node: *mut CDslxTypeAlias,
    ) -> *mut CDslxType;
    pub fn xls_dslx_type_info_get_type_constant_def(
        type_info: *mut CDslxTypeInfo,
        node: *mut CDslxConstantDef,
    ) -> *mut CDslxType;

    /// Gets the concrete type for a TypeAnnotation AST node.
    pub fn xls_dslx_type_info_get_type_type_annotation(
        type_info: *mut CDslxTypeInfo,
        type_annotation: *mut CDslxTypeAnnotation,
    ) -> *mut CDslxType;

    // -- ConstantDef

    pub fn xls_dslx_constant_def_get_name(
        constant_def: *const CDslxConstantDef,
    ) -> *mut std::os::raw::c_char;

    pub fn xls_dslx_constant_def_get_value(constant_def: *const CDslxConstantDef)
        -> *mut CDslxExpr;

    // -- TypeAlias

    pub fn xls_dslx_type_alias_get_identifier(
        type_alias: *const CDslxTypeAlias,
    ) -> *mut std::os::raw::c_char;

    pub fn xls_dslx_type_alias_get_type_annotation(
        type_alias: *const CDslxTypeAlias,
    ) -> *mut CDslxTypeAnnotation;

    // -- TypeAnnotation

    pub fn xls_dslx_type_annotation_get_type_ref_type_annotation(
        type_annotation: *const CDslxTypeAnnotation,
    ) -> *mut CDslxTypeRefTypeAnnotation;

    // -- TypeRef

    pub fn xls_dslx_type_ref_get_type_definition(
        type_ref: *const CDslxTypeRef,
    ) -> *mut CDslxTypeDefinition;

    // -- Import

    pub fn xls_dslx_import_get_subject_count(import: *const CDslxImport) -> i64;
    pub fn xls_dslx_import_get_subject(
        import: *const CDslxImport,
        i: i64,
    ) -> *mut std::os::raw::c_char;

    // -- ColonRef

    pub fn xls_dslx_colon_ref_resolve_import_subject(
        colon_ref: *const CDslxColonRef,
    ) -> *mut CDslxImport;

    // -- TypeDefinition

    pub fn xls_dslx_type_definition_get_colon_ref(
        type_definition: *const CDslxTypeDefinition,
    ) -> *mut CDslxColonRef;
    pub fn xls_dslx_type_definition_get_type_alias(
        type_definition: *const CDslxTypeDefinition,
    ) -> *mut CDslxTypeAlias;

    // -- TypeRefTypeAnnotation

    pub fn xls_dslx_type_ref_type_annotation_get_type_ref(
        type_ref_type_annotation: *const CDslxTypeRefTypeAnnotation,
    ) -> *mut CDslxTypeRef;

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

    pub fn xls_dslx_enum_def_get_underlying(
        enum_def: *const CDslxEnumDef,
    ) -> *mut CDslxTypeAnnotation;

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

    pub fn xls_dslx_type_is_enum(type_: *const CDslxType) -> bool;
    pub fn xls_dslx_type_is_struct(type_: *const CDslxType) -> bool;
    pub fn xls_dslx_type_is_array(type_: *const CDslxType) -> bool;

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

    pub fn xls_dslx_type_get_enum_def(ty: *const CDslxType) -> *mut CDslxEnumDef;

    pub fn xls_dslx_type_get_struct_def(ty: *const CDslxType) -> *mut CDslxStructDef;

    pub fn xls_dslx_type_array_get_element_type(ty: *const CDslxType) -> *mut CDslxType;
    pub fn xls_dslx_type_array_get_size(ty: *const CDslxType) -> *mut CDslxTypeDim;

    // -- IR builder APIs

    pub fn xls_package_create(name: *const std::os::raw::c_char) -> *mut CIrPackage;
    pub fn xls_package_get_bits_type(package: *mut CIrPackage, bit_count: i64) -> *mut CIrType;
    pub fn xls_package_get_tuple_type(
        package: *mut CIrPackage,
        members: *mut *mut CIrType,
        member_count: i64,
    ) -> *mut CIrType;
    pub fn xls_function_builder_create(
        name: *const std::os::raw::c_char,
        package: *mut CIrPackage,
        should_verify: bool,
    ) -> *mut CIrFunctionBuilder;
    pub fn xls_function_builder_as_builder_base(
        builder: *mut CIrFunctionBuilder,
    ) -> *mut CIrBuilderBase;
    pub fn xls_function_builder_free(builder: *mut CIrFunctionBuilder);
    pub fn xls_bvalue_free(bvalue: *mut CIrBValue);
    pub fn xls_function_builder_add_parameter(
        builder: *mut CIrFunctionBuilder,
        name: *const std::os::raw::c_char,
        type_: *mut CIrType,
    ) -> *mut CIrBValue;
    pub fn xls_function_builder_build(
        builder: *mut CIrFunctionBuilder,
        error_out: *mut *mut std::os::raw::c_char,
        function_out: *mut *mut CIrFunction,
    ) -> bool;
    pub fn xls_function_builder_build_with_return_value(
        builder: *mut CIrFunctionBuilder,
        return_value: *mut CIrBValue,
        error_out: *mut *mut std::os::raw::c_char,
        function_out: *mut *mut CIrFunction,
    ) -> bool;
    pub fn xls_builder_base_add_and(
        builder: *mut CIrBuilderBase,
        lhs: *mut CIrBValue,
        rhs: *mut CIrBValue,
        name: *const std::os::raw::c_char,
    ) -> *mut CIrBValue;
    pub fn xls_builder_base_add_or(
        builder: *mut CIrBuilderBase,
        lhs: *mut CIrBValue,
        rhs: *mut CIrBValue,
        name: *const std::os::raw::c_char,
    ) -> *mut CIrBValue;
    pub fn xls_builder_base_add_not(
        builder: *mut CIrBuilderBase,
        value: *mut CIrBValue,
        name: *const std::os::raw::c_char,
    ) -> *mut CIrBValue;
    pub fn xls_builder_base_add_literal(
        builder: *mut CIrBuilderBase,
        value: *mut CIrValue,
        name: *const std::os::raw::c_char,
    ) -> *mut CIrBValue;
    pub fn xls_builder_base_add_tuple(
        builder: *mut CIrBuilderBase,
        operands: *mut *mut CIrBValue,
        operand_count: i64,
        name: *const std::os::raw::c_char,
    ) -> *mut CIrBValue;
    pub fn xls_builder_base_add_tuple_index(
        builder: *mut CIrBuilderBase,
        tuple: *mut CIrBValue,
        index: i64,
        name: *const std::os::raw::c_char,
    ) -> *mut CIrBValue;
}

pub const DSLX_STDLIB_PATH: &str = env!("DSLX_STDLIB_PATH");
pub const XLS_DSO_PATH: &str = env!("XLS_DSO_PATH");
