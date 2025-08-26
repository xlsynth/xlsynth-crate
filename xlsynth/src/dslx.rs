// SPDX-License-Identifier: Apache-2.0

//! APIs that wrap the "DSL X" (DSL) facilities inside of XLS.

#![allow(unused)]

use std::rc::Rc;

use xlsynth_sys::{self as sys, CDslxImportData};

use crate::{c_str_to_rust, IrValue, XlsynthError};

#[derive(Debug, PartialEq, Eq)]
pub enum TypeDefinitionKind {
    TypeAlias = 0,
    StructDef = 1,
    EnumDef = 2,
    ColonRef = 3,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ModuleMemberKind {
    Function = 0,
    Proc = 1,
    TestFunction = 2,
    TestProc = 3,
    Quickcheck = 4,
    TypeAlias = 5,
    StructDef = 6,
    ProcDef = 7,
    EnumDef = 8,
    ConstantDef = 9,
    Import = 10,
    ConstAssert = 11,
    Impl = 12,
    VerbatimNode = 13,
}

impl From<sys::DslxModuleMemberKind> for ModuleMemberKind {
    fn from(kind: sys::DslxModuleMemberKind) -> Self {
        let result = match kind {
            0 => ModuleMemberKind::Function,
            1 => ModuleMemberKind::Proc,
            2 => ModuleMemberKind::TestFunction,
            3 => ModuleMemberKind::TestProc,
            4 => ModuleMemberKind::Quickcheck,
            5 => ModuleMemberKind::TypeAlias,
            6 => ModuleMemberKind::StructDef,
            7 => ModuleMemberKind::ProcDef,
            8 => ModuleMemberKind::EnumDef,
            9 => ModuleMemberKind::ConstantDef,
            10 => ModuleMemberKind::Import,
            11 => ModuleMemberKind::ConstAssert,
            12 => ModuleMemberKind::Impl,
            13 => ModuleMemberKind::VerbatimNode,
            _ => panic!("Unknown module member kind: {}", kind),
        };
        assert_eq!(result as i32, kind);
        result
    }
}

struct ImportDataPtr {
    ptr: *mut CDslxImportData,
}

impl Drop for ImportDataPtr {
    fn drop(&mut self) {
        unsafe {
            sys::xls_dslx_import_data_free(self.ptr);
        }
    }
}

pub struct ImportData {
    ptr: Rc<ImportDataPtr>,
}

impl ImportData {
    pub fn new(
        dslx_stdlib_path: Option<&std::path::Path>,
        additional_search_paths: &[&std::path::Path],
    ) -> Self {
        let default_dslx_stdlib_path = std::path::PathBuf::from(xlsynth_sys::DSLX_STDLIB_PATH);
        let dslx_stdlib_path = dslx_stdlib_path.unwrap_or(&default_dslx_stdlib_path);

        let dslx_stdlib_path_c_str =
            std::ffi::CString::new(dslx_stdlib_path.to_str().unwrap()).unwrap();
        let dslx_stdlib_path_c_ptr = dslx_stdlib_path_c_str.as_ptr();

        // Note: we make sure we collect up the CString values so their lifetime
        // envelopes the import_data_create call.
        let additional_search_paths_c_strs: Vec<std::ffi::CString> = additional_search_paths
            .iter()
            .map(|p| std::ffi::CString::new(p.to_str().unwrap()).unwrap())
            .collect::<Vec<_>>();
        let additional_search_paths_c_ptrs: Vec<*const std::os::raw::c_char> =
            additional_search_paths_c_strs
                .iter()
                .map(|s| s.as_ptr())
                .collect::<Vec<_>>();

        ImportData {
            ptr: Rc::new(ImportDataPtr {
                ptr: unsafe {
                    sys::xls_dslx_import_data_create(
                        dslx_stdlib_path_c_ptr,
                        additional_search_paths_c_ptrs.as_ptr(),
                        additional_search_paths_c_ptrs.len(),
                    )
                },
            }),
        }
    }
}

impl Default for ImportData {
    fn default() -> Self {
        Self::new(None, &[])
    }
}

/// Simple wrapper around the typechecked module entity that has a `Drop`
/// implementation
struct TypecheckedModulePtr {
    parent: Rc<ImportDataPtr>,
    ptr: *mut sys::CDslxTypecheckedModule,
}

impl Drop for TypecheckedModulePtr {
    fn drop(&mut self) {
        unsafe {
            sys::xls_dslx_typechecked_module_free(self.ptr);
        }
    }
}

pub struct TypecheckedModule {
    ptr: Rc<TypecheckedModulePtr>,
}

impl TypecheckedModule {
    pub fn get_module(&self) -> Module {
        Module {
            parent: self.ptr.clone(),
            ptr: unsafe { sys::xls_dslx_typechecked_module_get_module(self.ptr.ptr) },
        }
    }

    pub fn get_type_info(&self) -> TypeInfo {
        TypeInfo {
            parent: self.ptr.clone(),
            ptr: unsafe { sys::xls_dslx_typechecked_module_get_type_info(self.ptr.ptr) },
        }
    }

    pub fn get_type_info_for_module(&self, module: &Module) -> Option<TypeInfo> {
        let self_type_info = self.get_type_info();
        if module.ptr == self.get_module().ptr {
            Some(self_type_info)
        } else {
            self_type_info.get_imported_type_info(module)
        }
    }
}

pub struct ConstantDef {
    parent: Rc<TypecheckedModulePtr>,
    ptr: *mut sys::CDslxConstantDef,
}

impl ConstantDef {
    pub fn to_text(&self) -> String {
        unsafe { crate::c_str_to_rust(sys::xls_dslx_constant_def_to_string(self.ptr)) }
    }
    pub fn get_name(&self) -> String {
        unsafe {
            let c_str = sys::xls_dslx_constant_def_get_name(self.ptr);
            c_str_to_rust(c_str)
        }
    }

    pub fn get_value(&self) -> Expr {
        Expr {
            parent: self.parent.clone(),
            ptr: unsafe { sys::xls_dslx_constant_def_get_value(self.ptr) },
        }
    }
}

impl std::fmt::Display for ConstantDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_text())
    }
}

pub struct ModuleMember {
    parent: Rc<TypecheckedModulePtr>,
    ptr: *mut sys::CDslxModuleMember,
}

pub enum MatchableModuleMember {
    EnumDef(EnumDef),
    StructDef(StructDef),
    TypeAlias(TypeAlias),
    ConstantDef(ConstantDef),
    Function(Function),
    Quickcheck(Quickcheck),
}

impl MatchableModuleMember {
    pub fn to_text(&self) -> String {
        match self {
            MatchableModuleMember::EnumDef(e) => format!("{e}"),
            MatchableModuleMember::StructDef(s) => format!("{s}"),
            MatchableModuleMember::TypeAlias(t) => format!("{t}"),
            MatchableModuleMember::ConstantDef(c) => format!("{c}"),
            MatchableModuleMember::Function(f) => format!("{f}"),
            MatchableModuleMember::Quickcheck(qc) => format!("{qc}"),
        }
    }
}

impl std::fmt::Display for MatchableModuleMember {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_text())
    }
}

impl ModuleMember {
    pub fn to_matchable(&self) -> Option<MatchableModuleMember> {
        let kind = unsafe { sys::xls_dslx_module_member_get_kind(self.ptr) };
        match ModuleMemberKind::from(kind) {
            ModuleMemberKind::EnumDef => {
                let enum_def = unsafe { sys::xls_dslx_module_member_get_enum_def(self.ptr) };
                Some(MatchableModuleMember::EnumDef(EnumDef {
                    parent: self.parent.clone(),
                    ptr: enum_def,
                }))
            }
            ModuleMemberKind::StructDef => {
                let struct_def = unsafe { sys::xls_dslx_module_member_get_struct_def(self.ptr) };
                Some(MatchableModuleMember::StructDef(StructDef {
                    parent: self.parent.clone(),
                    ptr: struct_def,
                }))
            }
            ModuleMemberKind::TypeAlias => {
                let type_alias = unsafe { sys::xls_dslx_module_member_get_type_alias(self.ptr) };
                Some(MatchableModuleMember::TypeAlias(TypeAlias {
                    parent: self.parent.clone(),
                    ptr: type_alias,
                }))
            }
            ModuleMemberKind::ConstantDef => {
                let constant_def =
                    unsafe { sys::xls_dslx_module_member_get_constant_def(self.ptr) };
                Some(MatchableModuleMember::ConstantDef(ConstantDef {
                    parent: self.parent.clone(),
                    ptr: constant_def,
                }))
            }
            ModuleMemberKind::Function => {
                let func_ptr = unsafe { sys::xls_dslx_module_member_get_function(self.ptr) };
                Some(MatchableModuleMember::Function(Function {
                    parent: self.parent.clone(),
                    ptr: func_ptr,
                }))
            }
            ModuleMemberKind::Quickcheck => {
                let qc_ptr = unsafe { sys::xls_dslx_module_member_get_quickcheck(self.ptr) };
                Some(MatchableModuleMember::Quickcheck(Quickcheck {
                    parent: self.parent.clone(),
                    ptr: qc_ptr,
                }))
            }
            _ => None,
        }
    }
}

// -- Quickcheck

pub struct Quickcheck {
    parent: Rc<TypecheckedModulePtr>,
    ptr: *mut sys::CDslxQuickcheck,
}

impl Quickcheck {
    pub fn to_text(&self) -> String {
        unsafe { crate::c_str_to_rust(sys::xls_dslx_quickcheck_to_string(self.ptr)) }
    }
    pub fn get_function(&self) -> Function {
        let func_ptr = unsafe { sys::xls_dslx_quickcheck_get_function(self.ptr) };
        Function {
            parent: self.parent.clone(),
            ptr: func_ptr,
        }
    }

    pub fn is_exhaustive(&self) -> bool {
        unsafe { sys::xls_dslx_quickcheck_is_exhaustive(self.ptr) }
    }

    pub fn get_count(&self) -> Option<i64> {
        let mut result: i64 = 0;
        let has_count = unsafe { sys::xls_dslx_quickcheck_get_count(self.ptr, &mut result) };
        if has_count {
            Some(result)
        } else {
            None
        }
    }
}

impl std::fmt::Display for Quickcheck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_text())
    }
}

pub struct Module {
    parent: Rc<TypecheckedModulePtr>,
    ptr: *mut sys::CDslxModule,
}

impl Module {
    pub fn get_name(&self) -> String {
        unsafe {
            let c_str = sys::xls_dslx_module_get_name(self.ptr);
            c_str_to_rust(c_str)
        }
    }

    pub fn get_member_count(&self) -> usize {
        unsafe { sys::xls_dslx_module_get_member_count(self.ptr) as usize }
    }

    pub fn get_member(&self, idx: usize) -> ModuleMember {
        let member_ptr = unsafe { sys::xls_dslx_module_get_member(self.ptr, idx as i64) };
        if member_ptr.is_null() {
            panic!("Failed to get module member at index {}", idx);
        }
        ModuleMember {
            parent: self.parent.clone(),
            ptr: member_ptr,
        }
    }

    pub fn get_type_definition_count(&self) -> usize {
        unsafe { sys::xls_dslx_module_get_type_definition_count(self.ptr) as usize }
    }

    pub fn get_type_definition_kind(&self, idx: usize) -> TypeDefinitionKind {
        let kind = unsafe { sys::xls_dslx_module_get_type_definition_kind(self.ptr, idx as i64) };
        match kind {
            0 => TypeDefinitionKind::TypeAlias,
            1 => TypeDefinitionKind::StructDef,
            2 => TypeDefinitionKind::EnumDef,
            3 => TypeDefinitionKind::ColonRef,
            _ => panic!("Unknown type definition kind: {}", kind),
        }
    }

    pub fn get_type_definition_as_enum_def(&self, idx: usize) -> Result<EnumDef, XlsynthError> {
        let ptr =
            unsafe { sys::xls_dslx_module_get_type_definition_as_enum_def(self.ptr, idx as i64) };
        if ptr.is_null() {
            return Err(XlsynthError("Failed to get enum def".to_string()));
        }
        Ok(EnumDef {
            parent: self.parent.clone(),
            ptr,
        })
    }

    pub fn get_type_definition_as_struct_def(&self, idx: usize) -> Result<StructDef, XlsynthError> {
        let ptr =
            unsafe { sys::xls_dslx_module_get_type_definition_as_struct_def(self.ptr, idx as i64) };
        if ptr.is_null() {
            return Err(XlsynthError("Failed to get struct def".to_string()));
        }
        Ok(StructDef {
            parent: self.parent.clone(),
            ptr,
        })
    }

    pub fn get_type_definition_as_type_alias(&self, idx: usize) -> Result<TypeAlias, XlsynthError> {
        let ptr =
            unsafe { sys::xls_dslx_module_get_type_definition_as_type_alias(self.ptr, idx as i64) };
        if ptr.is_null() {
            return Err(XlsynthError("Failed to get type alias".to_string()));
        }
        Ok(TypeAlias {
            parent: self.parent.clone(),
            ptr,
        })
    }
}

pub struct EnumMember {
    parent: Rc<TypecheckedModulePtr>,
    ptr: *mut sys::CDslxEnumMember,
}

impl EnumMember {
    pub fn get_name(&self) -> String {
        unsafe {
            let c_str = sys::xls_dslx_enum_member_get_name(self.ptr);
            c_str_to_rust(c_str)
        }
    }

    pub fn get_value(&self) -> Expr {
        Expr {
            parent: self.parent.clone(),
            ptr: unsafe { sys::xls_dslx_enum_member_get_value(self.ptr) },
        }
    }
}

pub struct Expr {
    parent: Rc<TypecheckedModulePtr>,
    ptr: *mut sys::CDslxExpr,
}

impl Expr {
    pub fn get_owner_module(&self) -> Module {
        let module_ptr = unsafe { sys::xls_dslx_expr_get_owner_module(self.ptr) };
        assert!(!module_ptr.is_null());
        Module {
            parent: self.parent.clone(),
            ptr: module_ptr,
        }
    }
}

// -- EnumDef

pub struct EnumDef {
    parent: Rc<TypecheckedModulePtr>,
    ptr: *mut sys::CDslxEnumDef,
}

impl EnumDef {
    pub fn to_text(&self) -> String {
        unsafe { crate::c_str_to_rust(sys::xls_dslx_enum_def_to_string(self.ptr)) }
    }
    pub fn get_identifier(&self) -> String {
        unsafe {
            let c_str = sys::xls_dslx_enum_def_get_identifier(self.ptr);
            c_str_to_rust(c_str)
        }
    }

    pub fn get_member_count(&self) -> usize {
        unsafe { sys::xls_dslx_enum_def_get_member_count(self.ptr) as usize }
    }

    pub fn get_member(&self, idx: usize) -> EnumMember {
        EnumMember {
            parent: self.parent.clone(),
            ptr: unsafe { sys::xls_dslx_enum_def_get_member(self.ptr, idx as i64) },
        }
    }

    pub fn get_underlying(&self) -> TypeAnnotation {
        TypeAnnotation {
            parent: self.parent.clone(),
            ptr: unsafe { sys::xls_dslx_enum_def_get_underlying(self.ptr) },
        }
    }
}

impl std::fmt::Display for EnumDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_text())
    }
}

// -- TypeRefTypeAnnotation (note in C++ this is a subtype of `TypeAnnotation`)

pub struct TypeRefTypeAnnotation {
    parent: Rc<TypecheckedModulePtr>,
    ptr: *mut sys::CDslxTypeRefTypeAnnotation,
}

impl TypeRefTypeAnnotation {
    pub fn get_type_ref(&self) -> TypeRef {
        TypeRef {
            parent: self.parent.clone(),
            ptr: unsafe { sys::xls_dslx_type_ref_type_annotation_get_type_ref(self.ptr) },
        }
    }
}

// -- TypeAnnotation

pub struct TypeAnnotation {
    parent: Rc<TypecheckedModulePtr>,
    ptr: *mut sys::CDslxTypeAnnotation,
}

impl TypeAnnotation {
    pub fn to_type_ref_type_annotation(&self) -> Option<TypeRefTypeAnnotation> {
        let casted =
            unsafe { sys::xls_dslx_type_annotation_get_type_ref_type_annotation(self.ptr) };
        if casted.is_null() {
            return None;
        }
        Some(TypeRefTypeAnnotation {
            parent: self.parent.clone(),
            ptr: casted,
        })
    }
}

// -- Import

pub struct Import {
    parent: Rc<TypecheckedModulePtr>,
    ptr: *mut sys::CDslxImport,
}

impl Import {
    pub fn get_subject(&self) -> Vec<String> {
        let mut result = vec![];
        let subject_count = unsafe { sys::xls_dslx_import_get_subject_count(self.ptr) };
        for i in 0..subject_count {
            let s = unsafe {
                let c_str = sys::xls_dslx_import_get_subject(self.ptr, i);
                c_str_to_rust(c_str)
            };
            result.push(s);
        }
        result
    }
}

// -- ColonRef

pub struct ColonRef {
    parent: Rc<TypecheckedModulePtr>,
    ptr: *mut sys::CDslxColonRef,
}

impl ColonRef {
    pub fn resolve_import_subject(&self) -> Option<Import> {
        let casted = unsafe { sys::xls_dslx_colon_ref_resolve_import_subject(self.ptr) };
        if casted.is_null() {
            return None;
        }
        Some(Import {
            parent: self.parent.clone(),
            ptr: casted,
        })
    }

    pub fn get_attr(&self) -> String {
        unsafe {
            let c_str = sys::xls_dslx_colon_ref_get_attr(self.ptr);
            c_str_to_rust(c_str)
        }
    }
}

// -- TypeDefinition

pub struct TypeDefinition {
    parent: Rc<TypecheckedModulePtr>,
    ptr: *mut sys::CDslxTypeDefinition,
}

impl TypeDefinition {
    pub fn to_colon_ref(&self) -> Option<ColonRef> {
        let casted = unsafe { sys::xls_dslx_type_definition_get_colon_ref(self.ptr) };
        if casted.is_null() {
            return None;
        }
        Some(ColonRef {
            parent: self.parent.clone(),
            ptr: casted,
        })
    }

    pub fn to_type_alias(&self) -> Option<TypeAlias> {
        let casted = unsafe { sys::xls_dslx_type_definition_get_type_alias(self.ptr) };
        if casted.is_null() {
            return None;
        }
        Some(TypeAlias {
            parent: self.parent.clone(),
            ptr: casted,
        })
    }
}

// -- TypeRef

pub struct TypeRef {
    parent: Rc<TypecheckedModulePtr>,
    ptr: *mut sys::CDslxTypeRef,
}

impl TypeRef {
    pub fn get_type_definition(&self) -> TypeDefinition {
        TypeDefinition {
            parent: self.parent.clone(),
            ptr: unsafe { sys::xls_dslx_type_ref_get_type_definition(self.ptr) },
        }
    }
}

// -- StructDef

pub struct StructMember {
    parent: Rc<TypecheckedModulePtr>,
    ptr: *mut sys::CDslxStructMember,
}

impl StructMember {
    pub fn get_name(&self) -> String {
        unsafe {
            let c_str = sys::xls_dslx_struct_member_get_name(self.ptr);
            c_str_to_rust(c_str)
        }
    }

    pub fn get_type(&self) -> TypeAnnotation {
        TypeAnnotation {
            parent: self.parent.clone(),
            ptr: unsafe { sys::xls_dslx_struct_member_get_type(self.ptr) },
        }
    }
}

pub struct StructDef {
    parent: Rc<TypecheckedModulePtr>,
    ptr: *mut sys::CDslxStructDef,
}

impl StructDef {
    pub fn to_text(&self) -> String {
        unsafe { crate::c_str_to_rust(sys::xls_dslx_struct_def_to_string(self.ptr)) }
    }
    pub fn get_identifier(&self) -> String {
        unsafe {
            let c_str = sys::xls_dslx_struct_def_get_identifier(self.ptr);
            c_str_to_rust(c_str)
        }
    }

    pub fn is_parametric(&self) -> bool {
        unsafe { sys::xls_dslx_struct_def_is_parametric(self.ptr) }
    }

    pub fn get_member_count(&self) -> usize {
        unsafe { sys::xls_dslx_struct_def_get_member_count(self.ptr) as usize }
    }

    pub fn get_member(&self, idx: usize) -> StructMember {
        assert!(idx < self.get_member_count(), "member index out of bounds");
        let result = StructMember {
            parent: self.parent.clone(),
            ptr: unsafe { sys::xls_dslx_struct_def_get_member(self.ptr, idx as i64) },
        };
        assert!(!result.ptr.is_null());
        result
    }
}

impl std::fmt::Display for StructDef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_text())
    }
}

pub struct TypeAlias {
    parent: Rc<TypecheckedModulePtr>,
    ptr: *mut sys::CDslxTypeAlias,
}

impl TypeAlias {
    pub fn to_text(&self) -> String {
        unsafe { crate::c_str_to_rust(sys::xls_dslx_type_alias_to_string(self.ptr)) }
    }
    pub fn get_identifier(&self) -> String {
        unsafe {
            let c_str = sys::xls_dslx_type_alias_get_identifier(self.ptr);
            c_str_to_rust(c_str)
        }
    }

    pub fn get_type_annotation(&self) -> TypeAnnotation {
        TypeAnnotation {
            parent: self.parent.clone(),
            ptr: unsafe { sys::xls_dslx_type_alias_get_type_annotation(self.ptr) },
        }
    }
}

impl std::fmt::Display for TypeAlias {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_text())
    }
}

/// Wrapper for a DSLX function definition.
pub struct Function {
    parent: Rc<TypecheckedModulePtr>,
    ptr: *mut sys::CDslxFunction,
}

pub struct Param {
    parent: Rc<TypecheckedModulePtr>,
    ptr: *mut sys::CDslxParam,
}

impl Param {
    pub fn get_name(&self) -> String {
        unsafe { c_str_to_rust(sys::xls_dslx_param_get_name(self.ptr)) }
    }

    pub fn get_type_annotation(&self) -> TypeAnnotation {
        TypeAnnotation {
            parent: self.parent.clone(),
            ptr: unsafe { sys::xls_dslx_param_get_type_annotation(self.ptr) },
        }
    }
}

impl Function {
    pub fn to_text(&self) -> String {
        unsafe { crate::c_str_to_rust(sys::xls_dslx_function_to_string(self.ptr)) }
    }
    pub fn get_identifier(&self) -> String {
        unsafe {
            let c_str = sys::xls_dslx_function_get_identifier(self.ptr);
            crate::c_str_to_rust(c_str)
        }
    }

    pub fn is_parametric(&self) -> bool {
        unsafe { sys::xls_dslx_function_is_parametric(self.ptr) }
    }

    pub fn get_param_count(&self) -> usize {
        unsafe { sys::xls_dslx_function_get_param_count(self.ptr) as usize }
    }

    pub fn get_param(&self, idx: usize) -> Param {
        let p = unsafe { sys::xls_dslx_function_get_param(self.ptr, idx as i64) };
        if p.is_null() {
            panic!("Failed to get function param at index {}", idx);
        }
        Param {
            parent: self.parent.clone(),
            ptr: p,
        }
    }
}

impl std::fmt::Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_text())
    }
}

struct InterpValuePtr {
    ptr: *mut sys::CDslxInterpValue,
}

impl Drop for InterpValuePtr {
    fn drop(&mut self) {
        unsafe {
            sys::xls_dslx_interp_value_free(self.ptr);
        }
    }
}

pub struct InterpValue {
    ptr: Rc<InterpValuePtr>,
}

impl InterpValue {
    pub fn convert_to_ir(&self) -> Result<IrValue, XlsynthError> {
        let mut error_out = std::ptr::null_mut();
        let mut result_out = std::ptr::null_mut();
        let success = unsafe {
            sys::xls_dslx_interp_value_convert_to_ir(self.ptr.ptr, &mut error_out, &mut result_out)
        };
        if success {
            assert!(error_out.is_null());
            assert!(!result_out.is_null());
            Ok(IrValue { ptr: result_out })
        } else {
            assert!(!error_out.is_null());
            let error_out_str: String = unsafe { c_str_to_rust(error_out) };
            Err(XlsynthError(error_out_str))
        }
    }
}

pub struct TypeInfo {
    parent: Rc<TypecheckedModulePtr>,
    ptr: *mut sys::CDslxTypeInfo,
}

impl TypeInfo {
    pub fn get_const_expr(&self, expr: &Expr) -> Result<InterpValue, XlsynthError> {
        let mut error_out = std::ptr::null_mut();
        let mut result_out = std::ptr::null_mut();
        let success = unsafe {
            sys::xls_dslx_type_info_get_const_expr(
                self.ptr,
                expr.ptr,
                &mut error_out,
                &mut result_out,
            )
        };
        if success {
            assert!(error_out.is_null());
            assert!(!result_out.is_null());
            Ok(InterpValue {
                ptr: Rc::new(InterpValuePtr { ptr: result_out }),
            })
        } else {
            assert!(!error_out.is_null());
            let error_out_str: String = unsafe { c_str_to_rust(error_out) };
            Err(XlsynthError(error_out_str))
        }
    }

    pub fn get_type_for_type_annotation(&self, type_annotation: &TypeAnnotation) -> Type {
        Type {
            parent: self.parent.clone(),
            ptr: unsafe {
                sys::xls_dslx_type_info_get_type_type_annotation(self.ptr, type_annotation.ptr)
            },
        }
    }

    pub fn get_type_for_enum_def(&self, enum_def: &EnumDef) -> Type {
        Type {
            parent: self.parent.clone(),
            ptr: unsafe { sys::xls_dslx_type_info_get_type_enum_def(self.ptr, enum_def.ptr) },
        }
    }

    pub fn get_type_for_constant_def(&self, constant_def: &ConstantDef) -> Type {
        Type {
            parent: self.parent.clone(),
            ptr: unsafe {
                sys::xls_dslx_type_info_get_type_constant_def(self.ptr, constant_def.ptr)
            },
        }
    }

    pub fn get_type_for_struct_member(&self, struct_member: &StructMember) -> Type {
        assert!(!self.ptr.is_null());
        assert!(!struct_member.ptr.is_null());
        let result = Type {
            parent: self.parent.clone(),
            ptr: unsafe {
                sys::xls_dslx_type_info_get_type_struct_member(self.ptr, struct_member.ptr)
            },
        };
        assert!(!result.ptr.is_null());
        result
    }

    pub fn requires_implicit_token(&self, function: &Function) -> Result<bool, XlsynthError> {
        let mut error_out = std::ptr::null_mut();
        let mut result_out = false;
        let success = unsafe {
            sys::xls_dslx_type_info_get_requires_implicit_token(
                self.ptr,
                function.ptr,
                &mut error_out,
                &mut result_out,
            )
        };
        if success {
            assert!(error_out.is_null());
            Ok(result_out)
        } else {
            assert!(!error_out.is_null());
            let error_out_str: String = unsafe { c_str_to_rust(error_out) };
            Err(XlsynthError(error_out_str))
        }
    }

    pub fn get_imported_type_info(&self, module: &Module) -> Option<TypeInfo> {
        let ptr = unsafe { sys::xls_dslx_type_info_get_imported_type_info(self.ptr, module.ptr) };
        if ptr.is_null() {
            None
        } else {
            Some(TypeInfo {
                parent: self.parent.clone(),
                ptr,
            })
        }
    }
}

/// RAII-style wrapper around a `CDslxTypeDim` pointer that calls `free` on
/// drop.
struct TypeDimWrapper {
    wrapped: *mut sys::CDslxTypeDim,
}

impl Drop for TypeDimWrapper {
    fn drop(&mut self) {
        unsafe {
            sys::xls_dslx_type_dim_free(self.wrapped);
        }
    }
}

impl TypeDimWrapper {
    fn is_parametric(&self) -> bool {
        unsafe { sys::xls_dslx_type_dim_is_parametric(self.wrapped) }
    }

    fn get_as_bool(&self) -> Result<bool, XlsynthError> {
        assert!(!self.wrapped.is_null());
        let mut error_out = std::ptr::null_mut();
        let mut result_out = false;
        let success = unsafe {
            sys::xls_dslx_type_dim_get_as_bool(self.wrapped, &mut error_out, &mut result_out)
        };
        if success {
            assert!(error_out.is_null());
            Ok(result_out)
        } else {
            assert!(!error_out.is_null());
            let error_out_str: String = unsafe { c_str_to_rust(error_out) };
            Err(XlsynthError(error_out_str))
        }
    }

    fn get_as_i64(&self) -> Result<i64, XlsynthError> {
        assert!(!self.wrapped.is_null());
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut result_out: i64 = 0;
        let success = unsafe {
            sys::xls_dslx_type_dim_get_as_int64(self.wrapped, &mut error_out, &mut result_out)
        };
        if success {
            assert!(error_out.is_null());
            Ok(result_out)
        } else {
            assert!(!error_out.is_null());
            let error_out_str: String = unsafe { c_str_to_rust(error_out) };
            Err(XlsynthError(error_out_str))
        }
    }
}

#[derive(Clone)]
pub struct Type {
    parent: Rc<TypecheckedModulePtr>,
    ptr: *mut sys::CDslxType,
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string().unwrap())
    }
}

impl Type {
    pub fn to_string(&self) -> Result<String, XlsynthError> {
        let mut error_out = std::ptr::null_mut();
        let mut result_out = std::ptr::null_mut();
        let success =
            unsafe { sys::xls_dslx_type_to_string(self.ptr, &mut error_out, &mut result_out) };
        if success {
            assert!(error_out.is_null());
            assert!(!result_out.is_null());
            let result_out_str: String = unsafe { c_str_to_rust(result_out) };
            Ok(result_out_str)
        } else {
            assert!(!error_out.is_null());
            let error_out_str: String = unsafe { c_str_to_rust(error_out) };
            Err(XlsynthError(error_out_str))
        }
    }

    pub fn get_total_bit_count(&self) -> Result<usize, XlsynthError> {
        let mut error_out = std::ptr::null_mut();
        let mut result_out = 0;
        let success = unsafe {
            sys::xls_dslx_type_get_total_bit_count(self.ptr, &mut error_out, &mut result_out)
        };
        if success {
            assert!(error_out.is_null());
            Ok(result_out as usize)
        } else {
            assert!(!error_out.is_null());
            let error_out_str: String = unsafe { c_str_to_rust(error_out) };
            Err(XlsynthError(error_out_str))
        }
    }

    pub fn is_signed_bits(&self) -> Result<bool, XlsynthError> {
        let mut error_out = std::ptr::null_mut();
        let mut result_out = false;
        let success =
            unsafe { sys::xls_dslx_type_is_signed_bits(self.ptr, &mut error_out, &mut result_out) };
        if success {
            assert!(error_out.is_null());
            Ok(result_out)
        } else {
            assert!(!error_out.is_null());
            let error_out_str: String = unsafe { c_str_to_rust(error_out) };
            Err(XlsynthError(error_out_str))
        }
    }

    /// Returns `Some((is_signed, bit_count))` if the type is bits-like,
    /// otherwise returns `None`.
    pub fn is_bits_like(&self) -> Option<(bool, usize)> {
        let mut is_signed = std::ptr::null_mut();
        let mut size = std::ptr::null_mut();
        let success =
            unsafe { sys::xls_dslx_type_is_bits_like(self.ptr, &mut is_signed, &mut size) };

        let is_signed_wrapper = TypeDimWrapper { wrapped: is_signed };
        let size_wrapper = TypeDimWrapper { wrapped: size };

        if success {
            let is_signed = is_signed_wrapper
                .get_as_bool()
                .expect("get_as_bool success");
            let size = size_wrapper.get_as_i64().expect("get_as_i64 success");
            Some((is_signed, size as usize))
        } else {
            None
        }
    }

    pub fn is_enum(&self) -> bool {
        unsafe { sys::xls_dslx_type_is_enum(self.ptr) }
    }

    pub fn get_enum_def(&self) -> Result<EnumDef, XlsynthError> {
        if !self.is_enum() {
            return Err(XlsynthError("Type is not an enum".to_string()));
        }
        let ptr = unsafe { sys::xls_dslx_type_get_enum_def(self.ptr) };
        // Wrap up the pointer as an EnumDef structure.
        Ok(EnumDef {
            parent: self.parent.clone(),
            ptr,
        })
    }

    pub fn is_struct(&self) -> bool {
        unsafe { sys::xls_dslx_type_is_struct(self.ptr) }
    }

    pub fn get_struct_def(&self) -> Result<StructDef, XlsynthError> {
        if !self.is_struct() {
            return Err(XlsynthError("Type is not a struct".to_string()));
        }
        let ptr = unsafe { sys::xls_dslx_type_get_struct_def(self.ptr) };
        // Wrap up the pointer as a StructDef structure.
        Ok(StructDef {
            parent: self.parent.clone(),
            ptr,
        })
    }

    pub fn is_array(&self) -> bool {
        unsafe { sys::xls_dslx_type_is_array(self.ptr) }
    }

    pub fn get_array_element_type(&self) -> Type {
        assert!(self.is_array());
        Type {
            parent: self.parent.clone(),
            ptr: unsafe { sys::xls_dslx_type_array_get_element_type(self.ptr) },
        }
    }

    pub fn get_array_size(&self) -> usize {
        assert!(self.is_array());
        let type_dim = TypeDimWrapper {
            wrapped: unsafe { sys::xls_dslx_type_array_get_size(self.ptr) },
        };
        assert!(!type_dim.wrapped.is_null());
        let size = type_dim.get_as_i64().expect("get_as_i64 success");
        assert!(size >= 0);
        size as usize
    }
}

pub fn parse_and_typecheck(
    dslx: &str,
    path: &str,
    module_name: &str,
    import_data: &mut ImportData,
) -> Result<TypecheckedModule, XlsynthError> {
    let program_c_str = std::ffi::CString::new(dslx).unwrap();
    let path_c_str = std::ffi::CString::new(path).unwrap();
    let module_name_c_str = std::ffi::CString::new(module_name).unwrap();
    unsafe {
        let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
        let mut result_out: *mut sys::CDslxTypecheckedModule = std::ptr::null_mut();
        let success = sys::xls_dslx_parse_and_typecheck(
            program_c_str.as_ptr(),
            path_c_str.as_ptr(),
            module_name_c_str.as_ptr(),
            import_data.ptr.ptr,
            &mut error_out,
            &mut result_out,
        );
        if success {
            assert!(error_out.is_null());
            assert!(!result_out.is_null());
            Ok(TypecheckedModule {
                ptr: Rc::new(TypecheckedModulePtr {
                    parent: import_data.ptr.clone(),
                    ptr: result_out,
                }),
            })
        } else {
            assert!(!error_out.is_null());
            let error_out_str: String = c_str_to_rust(error_out);
            Err(XlsynthError(error_out_str))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::ir_value::IrFormatPreference;

    use super::*;

    #[test]
    fn test_one_enum_def() {
        let dslx = "enum MyEnum : u32 { VALUE = 0xcafef00d }";
        let mut import_data = ImportData::default();
        let typechecked_module =
            parse_and_typecheck(dslx, "/fake/path.x", "my_enum_mod", &mut import_data)
                .expect("parse-and-typecheck success");
        let module = typechecked_module.get_module();
        assert_eq!(module.get_type_definition_count(), 1);
        assert_eq!(
            module.get_type_definition_kind(0),
            TypeDefinitionKind::EnumDef
        );

        let enum_def = module
            .get_type_definition_as_enum_def(0)
            .expect("enum definition");
        assert_eq!(enum_def.get_identifier(), "MyEnum");
        assert_eq!(enum_def.get_member_count(), 1);
        let value_member = enum_def.get_member(0);
        assert_eq!(value_member.get_name(), "VALUE");
        let expr: Expr = value_member.get_value();

        let type_info = typechecked_module.get_type_info();
        let interp_value = type_info
            .get_const_expr(&expr)
            .expect("get_const_expr success");
        let ir_value = interp_value.convert_to_ir().expect("convert_to_ir success");
        assert_eq!(
            ir_value.to_string_fmt(IrFormatPreference::Hex).unwrap(),
            "bits[32]:0xcafe_f00d"
        )
    }

    #[test]
    fn test_one_struct_def() {
        let dslx = "struct MyStruct { a: u32, b: u16 }";
        let mut import_data = ImportData::default();
        let typechecked_module =
            parse_and_typecheck(dslx, "/fake/path.x", "my_struct_mod", &mut import_data)
                .expect("parse-and-typecheck success");
        let module = typechecked_module.get_module();
        let type_info = typechecked_module.get_type_info();
        assert_eq!(module.get_type_definition_count(), 1);
        assert_eq!(
            module.get_type_definition_kind(0),
            TypeDefinitionKind::StructDef
        );

        let struct_def = module
            .get_type_definition_as_struct_def(0)
            .expect("struct definition");
        assert_eq!(struct_def.get_identifier(), "MyStruct");

        assert_eq!(struct_def.get_member_count(), 2);

        let member_a = struct_def.get_member(0);
        assert_eq!(member_a.get_name(), "a");
        let type_a = member_a.get_type();
        // Inspect the inferred type information for the type AST node.
        {
            let concrete_type_a = type_info.get_type_for_type_annotation(&type_a);
            assert_eq!(concrete_type_a.to_string().unwrap(), "uN[32]");
            assert_eq!(concrete_type_a.get_total_bit_count().unwrap(), 32);

            let bits_like = concrete_type_a
                .is_bits_like()
                .expect("u32 should be bits-like");
            assert_eq!(bits_like, (false, 32));
        }

        let member_b = struct_def.get_member(1);
        assert_eq!(member_b.get_name(), "b");
        let type_b = member_b.get_type();
        // Inspect the inferred type information for the type AST node.
        {
            let concrete_type_b = type_info.get_type_for_type_annotation(&type_b);
            assert_eq!(concrete_type_b.to_string().unwrap(), "uN[16]");
            assert_eq!(concrete_type_b.get_total_bit_count().unwrap(), 16);

            let bits_like = concrete_type_b
                .is_bits_like()
                .expect("u16 should be bits-like");
            assert_eq!(bits_like, (false, 16));
        }
    }

    #[test]
    fn test_requires_implicit_token() {
        let dslx = "fn with_assert(a: u32, b: u32) -> u32 {
    assert!(a > b, \"a_greater_than_b\");
    a + b
}

fn without_assert(a: u32, b: u32) -> u32 {
    a + b
}";
        let mut import_data = ImportData::default();
        let typechecked_module = parse_and_typecheck(
            dslx,
            "/fake/implicit_token_test.x",
            "implicit_token_test_mod",
            &mut import_data,
        )
        .expect("parse-and-typecheck success");
        let module = typechecked_module.get_module();
        let type_info = typechecked_module.get_type_info();

        use crate::dslx::MatchableModuleMember;

        let mut with_assert_fn: Option<Function> = None;
        let mut without_assert_fn: Option<Function> = None;
        for i in 0..module.get_member_count() {
            if let Some(MatchableModuleMember::Function(f)) = module.get_member(i).to_matchable() {
                match f.get_identifier().as_str() {
                    "with_assert" => with_assert_fn = Some(f),
                    "without_assert" => without_assert_fn = Some(f),
                    _ => {}
                }
            }
        }

        let with_assert_fn = with_assert_fn.expect("with_assert fn found");
        let without_assert_fn = without_assert_fn.expect("without_assert fn found");

        assert!(type_info
            .requires_implicit_token(&with_assert_fn)
            .expect("requires_implicit_token success (with_assert)"));
        assert!(!type_info
            .requires_implicit_token(&without_assert_fn)
            .expect("requires_implicit_token success (without_assert)"));
    }

    #[test]
    fn test_function_params_exposed() {
        let dslx = r#"
            enum MyEnum : u8 { A = 0, B = 1 }
            fn f(a: u32, b: MyEnum) -> u32 { a }
        "#;
        let mut import_data = ImportData::default();
        let tcm = parse_and_typecheck(
            dslx,
            "/fake/params_test.x",
            "params_test_mod",
            &mut import_data,
        )
        .expect("parse-and-typecheck success");
        let module = tcm.get_module();
        let type_info = tcm.get_type_info();
        use crate::dslx::MatchableModuleMember;
        let mut found = false;
        for i in 0..module.get_member_count() {
            if let Some(MatchableModuleMember::Function(f)) = module.get_member(i).to_matchable() {
                if f.get_identifier() == "f" {
                    found = true;
                    assert_eq!(f.get_param_count(), 2);
                    let p0 = f.get_param(0);
                    assert_eq!(p0.get_name(), "a");
                    let p0_ty = type_info.get_type_for_type_annotation(&p0.get_type_annotation());
                    assert!(p0_ty.is_bits_like().is_some());

                    let p1 = f.get_param(1);
                    assert_eq!(p1.get_name(), "b");
                    let p1_ty = type_info.get_type_for_type_annotation(&p1.get_type_annotation());
                    assert!(p1_ty.is_enum());
                }
            }
        }
        assert!(found, "function f not found");
    }

    #[test]
    fn test_owner_module_and_type_info_for_imported_entity() {
        // Create a temporary directory for the imported and main module files.
        let tmpdir = xlsynth_test_helpers::make_test_tmpdir("xlsynth_dslx_test");
        let tmpdir = tmpdir.path().to_path_buf();

        // Write the imported module file to disk so ImportData can resolve it by name.
        let imported_path = tmpdir.join("imported.x");
        let imported_dslx = r#"
            const TEN = u32:10;
            pub enum ImpEnum : u8 { Z = 0xa }
        "#;
        std::fs::write(&imported_path, imported_dslx).expect("write imported.x");

        // Main module that imports the above and references the enum type.
        let main_path = tmpdir.join("main.x");
        let main_dslx = r#"
            import imported;
            fn f(x: imported::ImpEnum) -> u32 { u32:0 }
        "#;
        std::fs::write(&main_path, main_dslx).expect("write main.x");

        // Use ImportData with the temp directory as a search path.
        let mut import_data = ImportData::new(None, &[tmpdir.as_path()]);

        // Parse and typecheck ONLY the main module; imported module will be resolved by
        // import.
        let main_tcm = parse_and_typecheck(
            main_dslx,
            main_path.to_str().unwrap(),
            "main",
            &mut import_data,
        )
        .expect("parse-and-typecheck main success");
        let main_module = main_tcm.get_module();
        let main_type_info = main_tcm.get_type_info();

        // Find function f and its first parameter's type annotation.
        use crate::dslx::MatchableModuleMember;
        let mut f_opt: Option<Function> = None;
        for i in 0..main_module.get_member_count() {
            if let Some(MatchableModuleMember::Function(f)) =
                main_module.get_member(i).to_matchable()
            {
                if f.get_identifier() == "f" {
                    f_opt = Some(f);
                    break;
                }
            }
        }
        let f = f_opt.expect("function f found");
        assert_eq!(f.get_param_count(), 1);
        let p0 = f.get_param(0);
        let p0_ty = main_type_info.get_type_for_type_annotation(&p0.get_type_annotation());
        assert!(p0_ty.is_enum());

        // Get the EnumDef backing the imported type and grab a value expression from
        // it.
        let enum_def = p0_ty.get_enum_def().expect("enum def for imported type");
        assert_eq!(enum_def.get_identifier(), "ImpEnum");
        assert!(enum_def.get_member_count() >= 1);
        let member_expr = enum_def.get_member(0).get_value();

        // The enum member's value expression is owned by the imported module.
        let owner_module = member_expr.get_owner_module();
        assert_eq!(owner_module.get_name(), "imported");

        // Now fetch the TypeInfo corresponding to that imported module from the main
        // TCM.
        let imported_type_info = main_tcm
            .get_type_info_for_module(&owner_module)
            .expect("imported type info available");

        // Verify we can query type information for the imported EnumDef via the
        // imported TypeInfo.
        let imported_enum_type = imported_type_info.get_type_for_enum_def(&enum_def);
        assert!(imported_enum_type.is_enum());
        assert_eq!(
            imported_enum_type.get_enum_def().unwrap().get_identifier(),
            enum_def.get_identifier()
        );

        // Also ensure we can evaluate a const expr in the imported module via its
        // TypeInfo.
        use crate::ir_value::IrFormatPreference;
        let iv = imported_type_info
            .get_const_expr(&member_expr)
            .expect("const expr evaluation");
        let ir = iv.convert_to_ir().expect("convert_to_ir");
        assert_eq!(
            ir.to_string_fmt(IrFormatPreference::Hex).unwrap(),
            "bits[8]:0xa"
        );
    }
}
