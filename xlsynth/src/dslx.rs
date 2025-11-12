// SPDX-License-Identifier: Apache-2.0

//! APIs that wrap the "DSL X" (DSL) facilities inside of XLS.

#![allow(unused)]

use std::{
    cmp::Ordering,
    convert::TryFrom,
    fmt,
    hash::{Hash, Hasher},
    mem::ManuallyDrop,
    rc::Rc,
};

use log::debug;
use xlsynth_sys::{self as sys, CDslxImportData};

use crate::{c_str_to_rust, c_str_to_rust_no_dealloc, IrValue, XlsynthError};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributeKind {
    Cfg,
    DslxFormatDisable,
    ExternVerilog,
    SvType,
    Test,
    TestProc,
    Quickcheck,
}

impl AttributeKind {
    fn from_raw(kind: sys::DslxAttributeKind) -> Self {
        match kind {
            sys::XLS_DSLX_ATTRIBUTE_KIND_CFG => AttributeKind::Cfg,
            sys::XLS_DSLX_ATTRIBUTE_KIND_DSLX_FORMAT_DISABLE => AttributeKind::DslxFormatDisable,
            sys::XLS_DSLX_ATTRIBUTE_KIND_EXTERN_VERILOG => AttributeKind::ExternVerilog,
            sys::XLS_DSLX_ATTRIBUTE_KIND_SV_TYPE => AttributeKind::SvType,
            sys::XLS_DSLX_ATTRIBUTE_KIND_TEST => AttributeKind::Test,
            sys::XLS_DSLX_ATTRIBUTE_KIND_TEST_PROC => AttributeKind::TestProc,
            sys::XLS_DSLX_ATTRIBUTE_KIND_QUICKCHECK => AttributeKind::Quickcheck,
            _ => panic!("Unknown DSLX attribute kind: {}", kind),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributeArgumentKind {
    String,
    StringKeyValue,
    IntKeyValue,
}

impl AttributeArgumentKind {
    fn from_raw(kind: sys::DslxAttributeArgumentKind) -> Self {
        match kind {
            sys::XLS_DSLX_ATTRIBUTE_ARGUMENT_KIND_STRING => AttributeArgumentKind::String,
            sys::XLS_DSLX_ATTRIBUTE_ARGUMENT_KIND_STRING_KEY_VALUE => {
                AttributeArgumentKind::StringKeyValue
            }
            sys::XLS_DSLX_ATTRIBUTE_ARGUMENT_KIND_INT_KEY_VALUE => {
                AttributeArgumentKind::IntKeyValue
            }
            _ => panic!("Unknown DSLX attribute argument kind: {}", kind),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttributeArgument {
    String(String),
    StringKeyValue { key: String, value: String },
    IntKeyValue { key: String, value: i64 },
}

impl fmt::Display for AttributeArgument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AttributeArgument::String(value) => write!(f, "\"{value}\""),
            AttributeArgument::StringKeyValue { key, value } => {
                write!(f, "{key}=\"{value}\"")
            }
            AttributeArgument::IntKeyValue { key, value } => write!(f, "{key}={value}"),
        }
    }
}

pub struct Attribute {
    parent: Rc<TypecheckedModulePtr>,
    ptr: *mut sys::CDslxAttribute,
}

impl Clone for Attribute {
    fn clone(&self) -> Self {
        Self {
            parent: self.parent.clone(),
            ptr: self.ptr,
        }
    }
}

impl Attribute {
    fn ensure_index(&self, index: usize) {
        assert!(
            index < self.argument_count(),
            "attribute argument index {index} out of bounds (count={})",
            self.argument_count()
        );
    }

    pub fn kind(&self) -> AttributeKind {
        let raw = unsafe { sys::xls_dslx_attribute_get_kind(self.ptr) };
        AttributeKind::from_raw(raw)
    }

    pub fn argument_count(&self) -> usize {
        unsafe { sys::xls_dslx_attribute_get_argument_count(self.ptr) as usize }
    }

    pub fn get_argument_kind(&self, index: usize) -> AttributeArgumentKind {
        self.ensure_index(index);
        let raw = unsafe { sys::xls_dslx_attribute_get_argument_kind(self.ptr, index as i64) };
        AttributeArgumentKind::from_raw(raw)
    }

    pub fn get_string_argument(&self, index: usize) -> Option<String> {
        if self.get_argument_kind(index) != AttributeArgumentKind::String {
            return None;
        }
        let c_str = unsafe { sys::xls_dslx_attribute_get_string_argument(self.ptr, index as i64) };
        Some(unsafe { c_str_to_rust(c_str) })
    }

    pub fn get_key_value_argument_key(&self, index: usize) -> Option<String> {
        match self.get_argument_kind(index) {
            AttributeArgumentKind::StringKeyValue | AttributeArgumentKind::IntKeyValue => {
                let c_str = unsafe {
                    sys::xls_dslx_attribute_get_key_value_argument_key(self.ptr, index as i64)
                };
                Some(unsafe { c_str_to_rust(c_str) })
            }
            _ => None,
        }
    }

    pub fn get_key_value_string_argument_value(&self, index: usize) -> Option<String> {
        if self.get_argument_kind(index) != AttributeArgumentKind::StringKeyValue {
            return None;
        }
        let c_str = unsafe {
            sys::xls_dslx_attribute_get_key_value_string_argument_value(self.ptr, index as i64)
        };
        Some(unsafe { c_str_to_rust(c_str) })
    }

    pub fn get_key_value_int_argument_value(&self, index: usize) -> Option<i64> {
        if self.get_argument_kind(index) != AttributeArgumentKind::IntKeyValue {
            return None;
        }
        let value = unsafe {
            sys::xls_dslx_attribute_get_key_value_int_argument_value(self.ptr, index as i64)
        };
        Some(value)
    }

    pub fn get_argument(&self, index: usize) -> AttributeArgument {
        match self.get_argument_kind(index) {
            AttributeArgumentKind::String => AttributeArgument::String(
                self.get_string_argument(index)
                    .expect("string argument should be present"),
            ),
            AttributeArgumentKind::StringKeyValue => {
                let key = self
                    .get_key_value_argument_key(index)
                    .expect("key should be present for string key/value argument");
                let value = self
                    .get_key_value_string_argument_value(index)
                    .expect("string value should be present");
                AttributeArgument::StringKeyValue { key, value }
            }
            AttributeArgumentKind::IntKeyValue => {
                let key = self
                    .get_key_value_argument_key(index)
                    .expect("key should be present for int key/value argument");
                let value = self
                    .get_key_value_int_argument_value(index)
                    .expect("int value should be present");
                AttributeArgument::IntKeyValue { key, value }
            }
        }
    }

    pub fn arguments(&self) -> Vec<AttributeArgument> {
        (0..self.argument_count())
            .map(|index| self.get_argument(index))
            .collect()
    }

    pub fn to_text(&self) -> String {
        unsafe { c_str_to_rust(sys::xls_dslx_attribute_to_string(self.ptr)) }
    }
}

impl fmt::Display for Attribute {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_text())
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

impl Clone for TypecheckedModule {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr.clone(),
        }
    }
}

pub struct FunctionSpecializationRequest<'a> {
    pub function_name: &'a str,
    pub specialized_name: &'a str,
    pub env: Option<ParametricEnv>,
}

pub struct InvocationRewriteRule<'a> {
    pub from_callee: &'a Function,
    pub to_callee: &'a Function,
    pub match_callee_env: Option<ParametricEnv>,
    pub to_callee_env: Option<ParametricEnv>,
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

    pub fn clone_ignoring_functions(
        &self,
        import_data: &mut ImportData,
        functions: &[&Function],
        install_subject: &str,
    ) -> Result<TypecheckedModule, XlsynthError> {
        let mut member_storage: Vec<ModuleMember> = Vec::with_capacity(functions.len());
        for &function in functions {
            member_storage.push(ModuleMember::try_from(function)?);
        }
        let member_refs: Vec<&ModuleMember> = member_storage.iter().collect();
        self.clone_ignoring_members(import_data, &member_refs, install_subject)
    }

    pub fn clone_ignoring_members(
        &self,
        import_data: &mut ImportData,
        members: &[&ModuleMember],
        install_subject: &str,
    ) -> Result<TypecheckedModule, XlsynthError> {
        let mut member_ptrs: Vec<*mut sys::CDslxModuleMember> =
            members.iter().map(|member| member.ptr).collect();
        let install_subject_cstr = std::ffi::CString::new(install_subject).unwrap();
        unsafe {
            let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
            let mut result_out: *mut sys::CDslxTypecheckedModule = std::ptr::null_mut();
            let success = sys::xls_dslx_typechecked_module_clone_removing_members(
                self.ptr.ptr,
                if member_ptrs.is_empty() {
                    std::ptr::null_mut()
                } else {
                    member_ptrs.as_mut_ptr()
                },
                member_ptrs.len(),
                install_subject_cstr.as_ptr(),
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
                Err(XlsynthError(c_str_to_rust(error_out)))
            }
        }
    }

    pub fn insert_function_specializations(
        &self,
        import_data: &mut ImportData,
        requests: &[FunctionSpecializationRequest<'_>],
        install_subject: &str,
    ) -> Result<TypecheckedModule, XlsynthError> {
        let install_subject_cstr = std::ffi::CString::new(install_subject).unwrap();
        let mut function_name_cstrs: Vec<std::ffi::CString> = Vec::with_capacity(requests.len());
        let mut specialized_name_cstrs: Vec<std::ffi::CString> = Vec::with_capacity(requests.len());
        for request in requests {
            function_name_cstrs.push(std::ffi::CString::new(request.function_name).unwrap());
            specialized_name_cstrs.push(std::ffi::CString::new(request.specialized_name).unwrap());
        }
        let mut ffi_requests: Vec<sys::XlsDslxFunctionSpecializationRequest> =
            Vec::with_capacity(requests.len());
        for (i, request) in requests.iter().enumerate() {
            ffi_requests.push(sys::XlsDslxFunctionSpecializationRequest {
                function_name: function_name_cstrs[i].as_ptr(),
                specialized_name: specialized_name_cstrs[i].as_ptr(),
                env: request
                    .env
                    .as_ref()
                    .map(|env| env.ptr)
                    .unwrap_or(std::ptr::null_mut()),
            });
        }

        unsafe {
            let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
            let mut result_out: *mut sys::CDslxTypecheckedModule = std::ptr::null_mut();
            let success = sys::xls_dslx_typechecked_module_insert_function_specializations(
                self.ptr.ptr,
                if ffi_requests.is_empty() {
                    std::ptr::null()
                } else {
                    ffi_requests.as_ptr()
                },
                ffi_requests.len(),
                import_data.ptr.ptr,
                install_subject_cstr.as_ptr(),
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
                Err(XlsynthError(c_str_to_rust(error_out)))
            }
        }
    }

    pub fn replace_invocations_in_module(
        &self,
        import_data: &mut ImportData,
        callers: &[&Function],
        rules: &[InvocationRewriteRule<'_>],
        install_subject: &str,
    ) -> Result<TypecheckedModule, XlsynthError> {
        let install_subject_cstr = std::ffi::CString::new(install_subject).unwrap();
        let mut caller_ptrs: Vec<*mut sys::CDslxFunction> = callers.iter().map(|f| f.ptr).collect();
        let mut rule_storage: Vec<sys::CDslxInvocationRewriteRule> =
            Vec::with_capacity(rules.len());
        for rule in rules {
            debug!("Replace invocations in module: from_callee: {:?}, to_callee: {:?}, match_callee_env: {:?}, to_callee_env: {:?}", rule.from_callee.get_identifier(), rule.to_callee.get_identifier(), rule.match_callee_env.as_ref().map(|env| env.to_string()), rule.to_callee_env.as_ref().map(|env| env.to_string()));
            rule_storage.push(sys::CDslxInvocationRewriteRule {
                from_callee: rule.from_callee.ptr,
                to_callee: rule.to_callee.ptr,
                match_callee_env: rule
                    .match_callee_env
                    .as_ref()
                    .map(|env| env.ptr as *const _)
                    .unwrap_or(std::ptr::null()),
                to_callee_env: rule
                    .to_callee_env
                    .as_ref()
                    .map(|env| env.ptr as *const _)
                    .unwrap_or(std::ptr::null()),
            });
        }

        debug!(
            "Replace invocations in module: {} callers, {} rules",
            caller_ptrs.len(),
            rule_storage.len()
        );

        unsafe {
            let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
            let mut result_out: *mut sys::CDslxTypecheckedModule = std::ptr::null_mut();
            let success = sys::xls_dslx_replace_invocations_in_module(
                self.ptr.ptr,
                if caller_ptrs.is_empty() {
                    std::ptr::null()
                } else {
                    caller_ptrs.as_mut_ptr()
                },
                caller_ptrs.len(),
                if rule_storage.is_empty() {
                    std::ptr::null()
                } else {
                    rule_storage.as_ptr()
                },
                rule_storage.len(),
                import_data.ptr.ptr,
                install_subject_cstr.as_ptr(),
                &mut error_out,
                &mut result_out,
            );
            debug!("Replace invocations in module success: {}", success);
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
                Err(XlsynthError(c_str_to_rust(error_out)))
            }
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

impl TryFrom<&ConstantDef> for ModuleMember {
    type Error = XlsynthError;

    fn try_from(constant_def: &ConstantDef) -> Result<Self, Self::Error> {
        let ptr = unsafe { sys::xls_dslx_module_member_from_constant_def(constant_def.ptr) };
        ModuleMember::from_raw(&constant_def.parent, ptr).ok_or_else(|| {
            let name = constant_def.get_name();
            XlsynthError(format!("constant `{name}` is not a member of the module"))
        })
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
    fn from_raw(
        parent: &Rc<TypecheckedModulePtr>,
        ptr: *mut sys::CDslxModuleMember,
    ) -> Option<Self> {
        if ptr.is_null() {
            None
        } else {
            Some(ModuleMember {
                parent: parent.clone(),
                ptr,
            })
        }
    }

    pub fn kind(&self) -> ModuleMemberKind {
        let kind = unsafe { sys::xls_dslx_module_member_get_kind(self.ptr) };
        ModuleMemberKind::from(kind)
    }

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

impl TryFrom<&Quickcheck> for ModuleMember {
    type Error = XlsynthError;

    fn try_from(quickcheck: &Quickcheck) -> Result<Self, Self::Error> {
        let ptr = unsafe { sys::xls_dslx_module_member_from_quickcheck(quickcheck.ptr) };
        ModuleMember::from_raw(&quickcheck.parent, ptr).ok_or_else(|| {
            let func_name = quickcheck.get_function().get_identifier();
            XlsynthError(format!(
                "quickcheck for `{func_name}` is not a member of the module"
            ))
        })
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

    pub fn to_text(&self) -> String {
        unsafe {
            let c_str = sys::xls_dslx_module_to_string(self.ptr);
            c_str_to_rust(c_str)
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

impl fmt::Display for Module {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_text())
    }
}

impl fmt::Debug for Module {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_text())
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

    pub fn to_text(&self) -> String {
        unsafe { crate::c_str_to_rust(sys::xls_dslx_expr_to_string(self.ptr)) }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_text())
    }
}

impl fmt::Debug for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_text())
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

impl TryFrom<&EnumDef> for ModuleMember {
    type Error = XlsynthError;

    fn try_from(enum_def: &EnumDef) -> Result<Self, Self::Error> {
        let ptr = unsafe { sys::xls_dslx_module_member_from_enum_def(enum_def.ptr) };
        ModuleMember::from_raw(&enum_def.parent, ptr).ok_or_else(|| {
            let name = enum_def.get_identifier();
            XlsynthError(format!("enum `{name}` is not a member of the module"))
        })
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

impl TryFrom<&StructDef> for ModuleMember {
    type Error = XlsynthError;

    fn try_from(struct_def: &StructDef) -> Result<Self, Self::Error> {
        let ptr = unsafe { sys::xls_dslx_module_member_from_struct_def(struct_def.ptr) };
        ModuleMember::from_raw(&struct_def.parent, ptr).ok_or_else(|| {
            let name = struct_def.get_identifier();
            XlsynthError(format!("struct `{name}` is not a member of the module"))
        })
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

impl TryFrom<&TypeAlias> for ModuleMember {
    type Error = XlsynthError;

    fn try_from(type_alias: &TypeAlias) -> Result<Self, Self::Error> {
        let ptr = unsafe { sys::xls_dslx_module_member_from_type_alias(type_alias.ptr) };
        ModuleMember::from_raw(&type_alias.parent, ptr).ok_or_else(|| {
            let name = type_alias.get_identifier();
            XlsynthError(format!("type alias `{name}` is not a member of the module"))
        })
    }
}

/// Wrapper for a DSLX function definition.
pub struct Function {
    parent: Rc<TypecheckedModulePtr>,
    ptr: *mut sys::CDslxFunction,
}

impl Clone for Function {
    fn clone(&self) -> Self {
        Self {
            parent: self.parent.clone(),
            ptr: self.ptr,
        }
    }
}

pub struct Param {
    parent: Rc<TypecheckedModulePtr>,
    ptr: *mut sys::CDslxParam,
}

pub struct ParametricBinding {
    parent: Rc<TypecheckedModulePtr>,
    ptr: *mut sys::CDslxParametricBinding,
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

impl ParametricBinding {
    pub fn get_identifier(&self) -> String {
        unsafe { c_str_to_rust(sys::xls_dslx_parametric_binding_get_identifier(self.ptr)) }
    }

    pub fn get_type_annotation(&self) -> Option<TypeAnnotation> {
        let ptr = unsafe { sys::xls_dslx_parametric_binding_get_type_annotation(self.ptr) };
        if ptr.is_null() {
            None
        } else {
            Some(TypeAnnotation {
                parent: self.parent.clone(),
                ptr,
            })
        }
    }

    pub fn get_expr(&self) -> Option<Expr> {
        let ptr = unsafe { sys::xls_dslx_parametric_binding_get_expr(self.ptr) };
        if ptr.is_null() {
            None
        } else {
            Some(Expr {
                parent: self.parent.clone(),
                ptr,
            })
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

    pub fn is_public(&self) -> bool {
        unsafe { sys::xls_dslx_function_is_public(self.ptr) }
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

    pub fn get_attribute_count(&self) -> usize {
        unsafe { sys::xls_dslx_function_get_attribute_count(self.ptr) as usize }
    }

    pub fn get_attribute(&self, idx: usize) -> Attribute {
        if idx >= self.get_attribute_count() {
            panic!(
                "attribute index {} out of bounds for function {}",
                idx,
                self.get_identifier()
            );
        }
        let ptr = unsafe { sys::xls_dslx_function_get_attribute(self.ptr, idx as i64) };
        if ptr.is_null() {
            panic!("xls_dslx_function_get_attribute returned null pointer");
        }
        Attribute {
            parent: self.parent.clone(),
            ptr,
        }
    }

    pub fn attributes(&self) -> Vec<Attribute> {
        (0..self.get_attribute_count())
            .map(|idx| self.get_attribute(idx))
            .collect()
    }

    pub fn get_parametric_binding_count(&self) -> usize {
        unsafe { sys::xls_dslx_function_get_parametric_binding_count(self.ptr) as usize }
    }

    pub fn get_parametric_binding(&self, idx: usize) -> ParametricBinding {
        if idx >= self.get_parametric_binding_count() {
            panic!("Failed to get function parametric binding at index {}", idx);
        }
        let ptr = unsafe { sys::xls_dslx_function_get_parametric_binding(self.ptr, idx as i64) };
        if ptr.is_null() {
            panic!(
                "xls_dslx_function_get_parametric_binding returned null for index {}",
                idx
            );
        }
        ParametricBinding {
            parent: self.parent.clone(),
            ptr,
        }
    }

    pub fn parametric_bindings(&self) -> Vec<ParametricBinding> {
        (0..self.get_parametric_binding_count())
            .map(|idx| self.get_parametric_binding(idx))
            .collect()
    }

    pub fn get_body(&self) -> Expr {
        let ptr = unsafe { sys::xls_dslx_function_get_body(self.ptr) };
        if ptr.is_null() {
            panic!("xls_dslx_function_get_body returned null pointer");
        }
        Expr {
            parent: self.parent.clone(),
            ptr,
        }
    }

    pub fn get_return_type(&self) -> TypeAnnotation {
        let ptr = unsafe { sys::xls_dslx_function_get_return_type(self.ptr) };
        if ptr.is_null() {
            panic!("xls_dslx_function_get_return_type returned null pointer");
        }
        TypeAnnotation {
            parent: self.parent.clone(),
            ptr,
        }
    }
}

impl std::fmt::Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_text())
    }
}

impl TryFrom<&Function> for ModuleMember {
    type Error = XlsynthError;

    fn try_from(function: &Function) -> Result<Self, Self::Error> {
        let ptr = unsafe { sys::xls_dslx_module_member_from_function(function.ptr) };
        ModuleMember::from_raw(&function.parent, ptr).ok_or_else(|| {
            let identifier = function.get_identifier();
            XlsynthError(format!(
                "function `{identifier}` is not a member of the module"
            ))
        })
    }
}

pub struct InterpValue {
    ptr: *mut sys::CDslxInterpValue,
}

impl InterpValue {
    pub fn make_ubits(bit_count: i64, value: u64) -> Self {
        let raw = unsafe { sys::xls_dslx_interp_value_make_ubits(bit_count, value) };
        Self::from_raw(raw)
    }

    pub fn make_sbits(bit_count: i64, value: i64) -> Self {
        let raw = unsafe { sys::xls_dslx_interp_value_make_sbits(bit_count, value) };
        Self::from_raw(raw)
    }

    pub fn from_string(text: &str) -> Result<Self, crate::XlsynthError> {
        let c_text = std::ffi::CString::new(text).unwrap();
        let c_stdlib = std::ffi::CString::new(sys::DSLX_STDLIB_PATH).unwrap();
        unsafe {
            let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
            let mut result_out: *mut sys::CDslxInterpValue = std::ptr::null_mut();
            let ok = sys::xls_dslx_interp_value_from_string(
                c_text.as_ptr(),
                c_stdlib.as_ptr(),
                &mut error_out,
                &mut result_out,
            );
            if ok {
                Ok(Self::from_raw(result_out))
            } else {
                Err(crate::XlsynthError(crate::c_str_to_rust(error_out)))
            }
        }
    }
    pub fn convert_to_ir(&self) -> Result<IrValue, XlsynthError> {
        unsafe {
            let mut error_out = std::ptr::null_mut();
            let mut result_out = std::ptr::null_mut();
            let success =
                sys::xls_dslx_interp_value_convert_to_ir(self.ptr, &mut error_out, &mut result_out);
            if success {
                assert!(error_out.is_null());
                assert!(!result_out.is_null());
                Ok(IrValue { ptr: result_out })
            } else {
                assert!(!error_out.is_null());
                Err(XlsynthError(unsafe { c_str_to_rust(error_out) }))
            }
        }
    }

    pub fn to_text(&self) -> String {
        unsafe {
            let c_str = sys::xls_dslx_interp_value_to_string(self.ptr);
            c_str_to_rust(c_str)
        }
    }

    pub fn make_enum(
        def: &EnumDef,
        is_signed: bool,
        bits: &crate::IrBits,
    ) -> Result<Self, crate::XlsynthError> {
        unsafe {
            let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
            let mut result_out: *mut sys::CDslxInterpValue = std::ptr::null_mut();
            let ok = sys::xls_dslx_interp_value_make_enum(
                def.ptr,
                is_signed,
                bits.ptr as *const sys::CIrBits,
                &mut error_out,
                &mut result_out,
            );
            if ok {
                Ok(Self::from_raw(result_out))
            } else {
                Err(crate::XlsynthError(crate::c_str_to_rust(error_out)))
            }
        }
    }

    pub fn make_tuple(elements: &[&InterpValue]) -> Result<Self, crate::XlsynthError> {
        let mut ptrs: Vec<*mut sys::CDslxInterpValue> = elements.iter().map(|v| v.ptr).collect();
        unsafe {
            let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
            let mut result_out: *mut sys::CDslxInterpValue = std::ptr::null_mut();
            let ok = sys::xls_dslx_interp_value_make_tuple(
                ptrs.len(),
                if ptrs.is_empty() {
                    std::ptr::null_mut()
                } else {
                    ptrs.as_mut_ptr()
                },
                &mut error_out,
                &mut result_out,
            );
            if ok {
                Ok(Self::from_raw(result_out))
            } else {
                Err(crate::XlsynthError(crate::c_str_to_rust(error_out)))
            }
        }
    }

    pub fn make_array(elements: &[&InterpValue]) -> Result<Self, crate::XlsynthError> {
        let mut ptrs: Vec<*mut sys::CDslxInterpValue> = elements.iter().map(|v| v.ptr).collect();
        unsafe {
            let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
            let mut result_out: *mut sys::CDslxInterpValue = std::ptr::null_mut();
            let ok = sys::xls_dslx_interp_value_make_array(
                ptrs.len(),
                if ptrs.is_empty() {
                    std::ptr::null_mut()
                } else {
                    ptrs.as_mut_ptr()
                },
                &mut error_out,
                &mut result_out,
            );
            if ok {
                Ok(Self::from_raw(result_out))
            } else {
                Err(crate::XlsynthError(crate::c_str_to_rust(error_out)))
            }
        }
    }

    fn from_raw(ptr: *mut sys::CDslxInterpValue) -> Self {
        assert!(
            !ptr.is_null(),
            "InterpValue::from_raw received null pointer"
        );
        InterpValue { ptr }
    }
}

impl Clone for InterpValue {
    fn clone(&self) -> Self {
        unsafe {
            let cloned = sys::xls_dslx_interp_value_clone(self.ptr);
            assert!(
                !cloned.is_null(),
                "xls_dslx_interp_value_clone returned null pointer"
            );
            InterpValue { ptr: cloned }
        }
    }
}

impl Drop for InterpValue {
    fn drop(&mut self) {
        unsafe {
            sys::xls_dslx_interp_value_free(self.ptr);
        }
    }
}

impl fmt::Display for InterpValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_text())
    }
}

impl fmt::Debug for InterpValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_text())
    }
}

// -- ParametricEnv

pub struct ParametricEnv {
    pub(crate) ptr: *mut sys::CDslxParametricEnv,
}

impl ParametricEnv {
    pub fn new(items: &[(&str, &InterpValue)]) -> Result<Self, crate::XlsynthError> {
        let id_cstrs: Vec<std::ffi::CString> = items
            .iter()
            .map(|(id, _)| std::ffi::CString::new(id.as_bytes()).unwrap())
            .collect();
        let mut ffi_items: Vec<sys::XlsDslxParametricEnvItem> = Vec::with_capacity(items.len());
        for (i, (_id, val)) in items.iter().enumerate() {
            ffi_items.push(sys::XlsDslxParametricEnvItem {
                identifier: id_cstrs[i].as_ptr(),
                value: val.ptr,
            });
        }
        unsafe {
            let mut error_out: *mut std::os::raw::c_char = std::ptr::null_mut();
            let mut env_out: *mut sys::CDslxParametricEnv = std::ptr::null_mut();
            let ok = sys::xls_dslx_parametric_env_create(
                if ffi_items.is_empty() {
                    std::ptr::null()
                } else {
                    ffi_items.as_ptr()
                },
                ffi_items.len(),
                &mut error_out,
                &mut env_out,
            );
            if ok {
                Ok(ParametricEnv { ptr: env_out })
            } else {
                Err(crate::XlsynthError(crate::c_str_to_rust(error_out)))
            }
        }
    }

    pub fn empty() -> Result<Self, crate::XlsynthError> {
        Self::new(&[])
    }

    pub fn binding_count(&self) -> usize {
        if self.ptr.is_null() {
            0
        } else {
            unsafe { sys::xls_dslx_parametric_env_get_binding_count(self.ptr) as usize }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.binding_count() == 0
    }

    pub fn get_binding(&self, index: usize) -> Option<(String, InterpValue)> {
        if self.ptr.is_null() {
            return None;
        }
        if index >= self.binding_count() {
            return None;
        }
        let identifier_ptr =
            unsafe { sys::xls_dslx_parametric_env_get_binding_identifier(self.ptr, index as i64) };
        if identifier_ptr.is_null() {
            return None;
        }
        let identifier =
            unsafe { c_str_to_rust_no_dealloc(identifier_ptr as *mut std::os::raw::c_char) };
        let value_ptr =
            unsafe { sys::xls_dslx_parametric_env_get_binding_value(self.ptr, index as i64) }
                as *const sys::CDslxInterpValue;
        if value_ptr.is_null() {
            return None;
        }
        unsafe {
            let temp = ManuallyDrop::new(InterpValue {
                ptr: value_ptr as *mut sys::CDslxInterpValue,
            });
            Some((identifier, (*temp).clone()))
        }
    }

    pub fn bindings(&self) -> Vec<(String, InterpValue)> {
        (0..self.binding_count())
            .filter_map(|index| self.get_binding(index))
            .collect()
    }

    pub fn to_text(&self) -> String {
        unsafe {
            let c_str = sys::xls_dslx_parametric_env_to_string(self.ptr);
            c_str_to_rust(c_str)
        }
    }
}

impl Clone for ParametricEnv {
    fn clone(&self) -> Self {
        unsafe {
            let env_out = sys::xls_dslx_parametric_env_clone(self.ptr);
            assert!(
                !env_out.is_null(),
                "xls_dslx_parametric_env_clone returned null"
            );
            ParametricEnv { ptr: env_out }
        }
    }
}

impl PartialEq for ParametricEnv {
    fn eq(&self, other: &Self) -> bool {
        unsafe {
            sys::xls_dslx_parametric_env_equals(
                self.ptr as *const sys::CDslxParametricEnv,
                other.ptr as *const sys::CDslxParametricEnv,
            )
        }
    }
}

impl Eq for ParametricEnv {}

impl Hash for ParametricEnv {
    fn hash<H: Hasher>(&self, state: &mut H) {
        unsafe {
            let value =
                sys::xls_dslx_parametric_env_hash(self.ptr as *const sys::CDslxParametricEnv);
            state.write_u64(value);
        }
    }
}

impl PartialOrd for ParametricEnv {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ParametricEnv {
    fn cmp(&self, other: &Self) -> Ordering {
        if self == other {
            Ordering::Equal
        } else {
            unsafe {
                let result = sys::xls_dslx_parametric_env_less_than(
                    self.ptr as *const sys::CDslxParametricEnv,
                    other.ptr as *const sys::CDslxParametricEnv,
                );
                if result {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            }
        }
    }
}

impl Drop for ParametricEnv {
    fn drop(&mut self) {
        unsafe { sys::xls_dslx_parametric_env_free(self.ptr) };
    }
}

impl fmt::Debug for ParametricEnv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_text())
    }
}

impl fmt::Display for ParametricEnv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_text())
    }
}

#[cfg(test)]
mod param_env_and_interp_value_tests {
    use super::*;
    use crate::{mangle_dslx_name_with_env, DslxCallingConvention, IrBits, IrValue};

    #[test]
    fn test_interp_value_make_bits_and_convert() {
        let iv = InterpValue::make_ubits(32, 42);
        let ir = iv.convert_to_ir().expect("convert to IR");
        assert_eq!(ir.to_string(), "bits[32]:42");

        let iv_s = InterpValue::make_sbits(8, -5);
        let ir_s = iv_s.convert_to_ir().expect("convert to IR");
        assert_eq!(ir_s.to_string(), "bits[8]:251");
    }

    #[test]
    fn test_interp_value_from_string_and_convert() {
        let iv = InterpValue::from_string("u32:42").expect("parse success");
        let ir = iv.convert_to_ir().expect("convert to IR");
        assert_eq!(ir.to_string(), "bits[32]:42");
    }

    #[test]
    fn test_interp_value_tuple_and_array_convert() {
        let a = InterpValue::make_ubits(8, 1);
        let b = InterpValue::make_ubits(8, 2);
        let tup = InterpValue::make_tuple(&[&a, &b]).expect("make tuple");
        let ir_tup = tup.convert_to_ir().expect("tup to IR");
        assert_eq!(ir_tup.get_element_count().unwrap(), 2);
        assert_eq!(ir_tup.get_element(0).unwrap().to_string(), "bits[8]:1");
        assert_eq!(ir_tup.get_element(1).unwrap().to_string(), "bits[8]:2");

        let arr = InterpValue::make_array(&[&a, &b]).expect("make array");
        let ir_arr = arr.convert_to_ir().expect("arr to IR");
        assert_eq!(ir_arr.get_element_count().unwrap(), 2);
        assert_eq!(ir_arr.get_element(0).unwrap().to_string(), "bits[8]:1");
        assert_eq!(ir_arr.get_element(1).unwrap().to_string(), "bits[8]:2");
    }

    #[test]
    fn test_interp_value_make_enum_and_convert() {
        // Build a small module with an enum and fetch its EnumDef.
        let dslx = "enum MyE : u8 { A = 0x7 }";
        let mut import_data = ImportData::default();
        let tcm = parse_and_typecheck(dslx, "/memfile/e.x", "e", &mut import_data)
            .expect("typecheck success");
        let module = tcm.get_module();
        let enum_def = module.get_type_definition_as_enum_def(0).expect("enum def");

        let bits = IrBits::make_ubits(8, 7).expect("bits");
        let iv = InterpValue::make_enum(&enum_def, false, &bits).expect("make enum");
        let ir = iv.convert_to_ir().expect("convert enum to IR");
        assert_eq!(ir.to_string(), "bits[8]:7");
    }

    #[test]
    fn test_parametric_env_new_and_mangle() {
        let x = InterpValue::make_ubits(32, 42);
        let y = InterpValue::make_ubits(32, 64);
        let env = ParametricEnv::new(&[("X", &x), ("Y", &y)]).expect("env");
        let mangled = mangle_dslx_name_with_env(
            "my_mod",
            "p",
            DslxCallingConvention::Typical,
            &["X", "Y"],
            Some(&env),
            None,
        )
        .expect("mangle");
        assert_eq!(mangled, "__my_mod__p__42_64");
    }

    #[test]
    fn test_parametric_env_empty_and_mangle() {
        let env = ParametricEnv::empty().expect("empty env");
        let mangled = mangle_dslx_name_with_env(
            "my_mod",
            "f",
            DslxCallingConvention::Typical,
            &[],
            Some(&env),
            Some("Point"),
        )
        .expect("mangle");
        assert_eq!(mangled, "__my_mod__Point__f");
    }

    #[test]
    fn test_insert_function_specializations() {
        let dslx = r#"
fn id<N: u32>(x: bits[N]) -> bits[N] { x }

fn call() -> bits[32] {
    id(bits[32]:0x0)
}
"#;
        let mut import_data = ImportData::default();
        let tm = parse_and_typecheck(
            dslx,
            "/memfile/specialize.x",
            "specialize_module",
            &mut import_data,
        )
        .expect("parse and typecheck");

        // Prepare the specialization environment for N = 32.
        let n_value = InterpValue::make_ubits(32, 32);
        let env = ParametricEnv::new(&[("N", &n_value)]).expect("env");

        let requests = [FunctionSpecializationRequest {
            function_name: "id",
            specialized_name: "id_N32",
            env: Some(env.clone()),
        }];

        let specialized_tm = tm
            .insert_function_specializations(
                &mut import_data,
                &requests,
                "specialize_module.specializations",
            )
            .expect("specialization succeeds");

        let specialized_module = specialized_tm.get_module();
        assert_eq!(specialized_module.get_member_count(), 3);

        let mut function_names = Vec::new();
        for i in 0..specialized_module.get_member_count() {
            if let Some(MatchableModuleMember::Function(f)) =
                specialized_module.get_member(i).to_matchable()
            {
                function_names.push(f.get_identifier());
            }
        }

        assert!(function_names.contains(&"id".to_string()));
        assert!(function_names.contains(&"id_N32".to_string()));
        assert!(function_names.contains(&"call".to_string()));
    }
}

pub struct TypeInfo {
    parent: Rc<TypecheckedModulePtr>,
    ptr: *mut sys::CDslxTypeInfo,
}

pub struct FunctionCallGraph {
    parent: Rc<TypecheckedModulePtr>,
    ptr: *mut sys::CDslxCallGraph,
}

impl Drop for FunctionCallGraph {
    fn drop(&mut self) {
        unsafe {
            sys::xls_dslx_call_graph_free(self.ptr);
        }
    }
}

impl FunctionCallGraph {
    pub fn function_count(&self) -> usize {
        unsafe { sys::xls_dslx_call_graph_get_function_count(self.ptr) as usize }
    }

    pub fn get_function(&self, index: usize) -> Option<Function> {
        let ptr = unsafe { sys::xls_dslx_call_graph_get_function(self.ptr, index as i64) };
        if ptr.is_null() {
            None
        } else {
            Some(Function {
                parent: self.parent.clone(),
                ptr,
            })
        }
    }

    pub fn callee_count(&self, caller: &Function) -> usize {
        unsafe { sys::xls_dslx_call_graph_get_callee_count(self.ptr, caller.ptr) as usize }
    }

    pub fn get_callee(&self, caller: &Function, index: usize) -> Option<Function> {
        let ptr = unsafe {
            sys::xls_dslx_call_graph_get_callee_function(self.ptr, caller.ptr, index as i64)
        };
        if ptr.is_null() {
            None
        } else {
            Some(Function {
                parent: self.parent.clone(),
                ptr,
            })
        }
    }
}

pub struct InvocationCalleeDataArray {
    parent: Rc<TypecheckedModulePtr>,
    ptr: *mut sys::CDslxInvocationCalleeDataArray,
}

impl Drop for InvocationCalleeDataArray {
    fn drop(&mut self) {
        unsafe {
            sys::xls_dslx_invocation_callee_data_array_free(self.ptr);
        }
    }
}

pub struct InvocationCalleeData {
    parent: Rc<TypecheckedModulePtr>,
    ptr: *mut sys::CDslxInvocationCalleeData,
}

impl Drop for InvocationCalleeData {
    fn drop(&mut self) {
        unsafe { sys::xls_dslx_invocation_callee_data_free(self.ptr) };
    }
}

pub struct Invocation {
    parent: Rc<TypecheckedModulePtr>,
    ptr: *mut sys::CDslxInvocation,
}

pub struct InvocationData {
    parent: Rc<TypecheckedModulePtr>,
    ptr: *mut sys::CDslxInvocationData,
}

impl InvocationCalleeDataArray {
    pub fn len(&self) -> usize {
        unsafe { sys::xls_dslx_invocation_callee_data_array_get_count(self.ptr) as usize }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, index: usize) -> Option<InvocationCalleeData> {
        if index >= self.len() {
            return None;
        }
        let ptr = unsafe { sys::xls_dslx_invocation_callee_data_array_get(self.ptr, index as i64) };
        if ptr.is_null() {
            None
        } else {
            let cloned = unsafe { sys::xls_dslx_invocation_callee_data_clone(ptr) };
            if cloned.is_null() {
                return None;
            }
            Some(InvocationCalleeData {
                parent: self.parent.clone(),
                ptr: cloned,
            })
        }
    }
}

impl InvocationCalleeData {
    pub fn callee_bindings(&self) -> Option<ParametricEnv> {
        let ptr = unsafe { sys::xls_dslx_invocation_callee_data_get_callee_bindings(self.ptr) };
        if ptr.is_null() {
            None
        } else {
            unsafe {
                let temp = ManuallyDrop::new(ParametricEnv {
                    ptr: ptr as *mut sys::CDslxParametricEnv,
                });
                Some((*temp).clone())
            }
        }
    }

    pub fn caller_bindings(&self) -> Option<ParametricEnv> {
        let ptr = unsafe { sys::xls_dslx_invocation_callee_data_get_caller_bindings(self.ptr) };
        if ptr.is_null() {
            None
        } else {
            unsafe {
                let temp = ManuallyDrop::new(ParametricEnv {
                    ptr: ptr as *mut sys::CDslxParametricEnv,
                });
                Some((*temp).clone())
            }
        }
    }

    pub fn derived_type_info(&self) -> Option<TypeInfo> {
        let ptr = unsafe { sys::xls_dslx_invocation_callee_data_get_derived_type_info(self.ptr) };
        if ptr.is_null() {
            None
        } else {
            Some(TypeInfo {
                parent: self.parent.clone(),
                ptr,
            })
        }
    }

    pub fn invocation(&self) -> Option<Invocation> {
        let ptr = unsafe { sys::xls_dslx_invocation_callee_data_get_invocation(self.ptr) };
        if ptr.is_null() {
            None
        } else {
            Some(Invocation {
                parent: self.parent.clone(),
                ptr,
            })
        }
    }
}

impl Invocation {
    pub fn as_ptr(&self) -> *mut sys::CDslxInvocation {
        self.ptr
    }
}

impl InvocationData {
    pub fn invocation(&self) -> Option<Invocation> {
        let ptr = unsafe { sys::xls_dslx_invocation_data_get_invocation(self.ptr) };
        if ptr.is_null() {
            None
        } else {
            Some(Invocation {
                parent: self.parent.clone(),
                ptr,
            })
        }
    }

    pub fn callee(&self) -> Option<Function> {
        let ptr = unsafe { sys::xls_dslx_invocation_data_get_callee(self.ptr) };
        if ptr.is_null() {
            None
        } else {
            Some(Function {
                parent: self.parent.clone(),
                ptr,
            })
        }
    }

    pub fn caller(&self) -> Option<Function> {
        let ptr = unsafe { sys::xls_dslx_invocation_data_get_caller(self.ptr) };
        if ptr.is_null() {
            None
        } else {
            Some(Function {
                parent: self.parent.clone(),
                ptr,
            })
        }
    }
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
            Ok(InterpValue::from_raw(result_out))
        } else {
            assert!(!error_out.is_null());
            let error_out_str: String = unsafe { c_str_to_rust(error_out) };
            Err(XlsynthError(error_out_str))
        }
    }

    pub fn get_type_for_type_annotation(&self, type_annotation: &TypeAnnotation) -> Option<Type> {
        let ptr = unsafe {
            sys::xls_dslx_type_info_get_type_type_annotation(self.ptr, type_annotation.ptr)
        };
        if ptr.is_null() {
            None
        } else {
            Some(Type {
                parent: self.parent.clone(),
                ptr,
            })
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

    pub fn get_unique_invocation_callee_data(
        &self,
        function: &Function,
    ) -> Option<InvocationCalleeDataArray> {
        let ptr = unsafe {
            sys::xls_dslx_type_info_get_unique_invocation_callee_data(self.ptr, function.ptr)
        };
        if ptr.is_null() {
            None
        } else {
            Some(InvocationCalleeDataArray {
                parent: self.parent.clone(),
                ptr,
            })
        }
    }

    pub fn get_all_invocation_callee_data(
        &self,
        function: &Function,
    ) -> Option<InvocationCalleeDataArray> {
        let ptr = unsafe {
            sys::xls_dslx_type_info_get_all_invocation_callee_data(self.ptr, function.ptr)
        };
        if ptr.is_null() {
            None
        } else {
            Some(InvocationCalleeDataArray {
                parent: self.parent.clone(),
                ptr,
            })
        }
    }

    pub fn build_function_call_graph(&self) -> Result<FunctionCallGraph, XlsynthError> {
        let mut error_out = std::ptr::null_mut();
        let mut result_out = std::ptr::null_mut();
        let success = unsafe {
            sys::xls_dslx_type_info_build_function_call_graph(
                self.ptr,
                &mut error_out,
                &mut result_out,
            )
        };
        if success {
            assert!(error_out.is_null());
            assert!(!result_out.is_null());
            Ok(FunctionCallGraph {
                parent: self.parent.clone(),
                ptr: result_out,
            })
        } else {
            assert!(!error_out.is_null());
            Err(XlsynthError(unsafe { c_str_to_rust(error_out) }))
        }
    }

    pub fn get_root_invocation_data(&self, invocation: &Invocation) -> Option<InvocationData> {
        let ptr =
            unsafe { sys::xls_dslx_type_info_get_root_invocation_data(self.ptr, invocation.ptr) };
        if ptr.is_null() {
            None
        } else {
            Some(InvocationData {
                parent: self.parent.clone(),
                ptr,
            })
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
            let concrete_type_a = type_info.get_type_for_type_annotation(&type_a).unwrap();
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
            let concrete_type_b = type_info.get_type_for_type_annotation(&type_b).unwrap();
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
    fn test_invocation_callee_data_introspection() {
        let dslx = r#"
fn id<N: u32>(x: bits[N]) -> bits[N] { x }

fn caller<N: u32>(x: bits[N]) -> bits[N] {
    id(x)
}

fn main() -> u32 {
    caller(u32:42)
}
"#;
        let mut import_data = ImportData::default();
        let tcm = parse_and_typecheck(dslx, "/memfile/invoke.x", "invoke_mod", &mut import_data)
            .expect("parse-and-typecheck success");
        let module = tcm.get_module();
        let type_info = tcm.get_type_info();

        use crate::dslx::MatchableModuleMember;

        let mut id_fn: Option<Function> = None;
        let mut caller_fn: Option<Function> = None;
        for i in 0..module.get_member_count() {
            if let Some(MatchableModuleMember::Function(f)) = module.get_member(i).to_matchable() {
                match f.get_identifier().as_str() {
                    "id" => id_fn = Some(f),
                    "caller" => caller_fn = Some(f),
                    _ => {}
                }
            }
        }

        let id_fn = id_fn.expect("id function found");
        let caller_fn = caller_fn.expect("caller function found");

        let callee_data_array = type_info
            .get_unique_invocation_callee_data(&id_fn)
            .expect("callee data array returned");
        assert_eq!(callee_data_array.len(), 1);
        let callee_data = callee_data_array.get(0).expect("callee data entry present");

        let all_callee_data_array = type_info
            .get_all_invocation_callee_data(&id_fn)
            .expect("all callee data array returned");
        assert_eq!(all_callee_data_array.len(), 1);

        let callee_env = callee_data
            .callee_bindings()
            .expect("callee bindings present");
        assert_eq!(callee_env.binding_count(), 1);
        let (callee_identifier, callee_value) =
            callee_env.get_binding(0).expect("callee binding present");
        assert_eq!(callee_identifier, "N");
        assert_eq!(callee_value.to_string(), "u32:32");

        let caller_env = callee_data
            .caller_bindings()
            .expect("caller bindings present");
        let (caller_identifier, caller_value) =
            caller_env.get_binding(0).expect("caller binding present");
        assert_eq!(caller_identifier, "N");

        let invocation = callee_data.invocation().expect("invocation pointer");
        let invocation_data = type_info
            .get_root_invocation_data(&invocation)
            .expect("invocation data present");
        assert_eq!(
            invocation_data
                .callee()
                .expect("invocation callee present")
                .get_identifier(),
            "id"
        );
        assert_eq!(
            invocation_data
                .caller()
                .expect("invocation caller present")
                .get_identifier(),
            "caller"
        );

        // Ensure the invocation returned by the invocation data matches the one we
        // queried with.
        let round_trip_invocation = invocation_data
            .invocation()
            .expect("round-trip invocation present");
        assert_eq!(round_trip_invocation.as_ptr(), invocation.as_ptr());

        // The caller binding value should match the callee binding value for this
        // simple program.
        let caller_value_str = caller_value.to_string();
        assert_eq!(caller_value_str, callee_value.to_string());
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
                    let p0_ty = type_info
                        .get_type_for_type_annotation(&p0.get_type_annotation())
                        .unwrap();
                    assert!(p0_ty.is_bits_like().is_some());

                    let p1 = f.get_param(1);
                    assert_eq!(p1.get_name(), "b");
                    let p1_ty = type_info
                        .get_type_for_type_annotation(&p1.get_type_annotation())
                        .unwrap();
                    assert!(p1_ty.is_enum());
                }
            }
        }
        assert!(found, "function f not found");
    }

    #[test]
    fn test_function_parametric_binding_accessors() {
        let dslx = r#"
pub fn add<N: u32 = {32}>(x: bits[N], y: bits[N]) -> bits[N] {
    x + y
}

fn helper(x: u32) -> u32 { x }
"#;
        let mut import_data = ImportData::default();
        let tcm = parse_and_typecheck(
            dslx,
            "/fake/param_binding_test.x",
            "param_binding_test_mod",
            &mut import_data,
        )
        .expect("parse-and-typecheck success");
        let module = tcm.get_module();
        let type_info = tcm.get_type_info();

        use crate::dslx::MatchableModuleMember;

        let mut add_fn: Option<Function> = None;
        let mut helper_fn: Option<Function> = None;
        for i in 0..module.get_member_count() {
            if let Some(MatchableModuleMember::Function(f)) = module.get_member(i).to_matchable() {
                match f.get_identifier().as_str() {
                    "add" => add_fn = Some(f),
                    "helper" => helper_fn = Some(f),
                    _ => {}
                }
            }
        }

        let add_fn = add_fn.expect("add function present");
        assert!(add_fn.is_public());
        assert!(add_fn.is_parametric());
        assert_eq!(add_fn.get_parametric_binding_count(), 1);
        let bindings = add_fn.parametric_bindings();
        assert_eq!(bindings.len(), 1);
        let binding = &bindings[0];
        assert_eq!(binding.get_identifier(), "N");
        let binding_type_annotation = binding
            .get_type_annotation()
            .expect("binding has type annotation");
        assert!(
            type_info
                .get_type_for_type_annotation(&binding_type_annotation)
                .is_none(),
            "binding type annotation for parametric function cannot be resolved"
        );
        let binding_expr = binding.get_expr().expect("binding has default expr");
        let binding_expr_text = binding_expr.to_text();
        assert!(
            binding_expr_text.contains("32"),
            "{}",
            format!(
                "expected binding expr text to contain `32`, got {}",
                binding_expr_text
            )
        );

        let return_type_annotation = add_fn.get_return_type();
        let return_type = type_info.get_type_for_type_annotation(&return_type_annotation);
        assert!(
            return_type.is_none(),
            "return type for parametric function cannot be resolved"
        );
        let body_text = add_fn.get_body().to_text();
        assert!(
            body_text.contains("x + y"),
            "{}",
            format!("expected body text to contain `x + y`, got {}", body_text)
        );

        let helper_fn = helper_fn.expect("helper function present");
        assert!(!helper_fn.is_public());
        assert!(!helper_fn.is_parametric());
        assert_eq!(helper_fn.get_parametric_binding_count(), 0);
        let helper_return_type_annotation = helper_fn.get_return_type();
        let helper_return_type = type_info
            .get_type_for_type_annotation(&helper_return_type_annotation)
            .unwrap()
            .to_string()
            .expect("helper return type string");
        assert_eq!(helper_return_type, "uN[32]");
    }

    #[test]
    fn test_function_attributes() {
        let dslx = r#"
#[cfg(test)]
fn cfg_fn(x: u32) -> u32 { x }

#[dslx_format_disable("fmt-off")]
fn fmt_fn(x: u32) -> u32 { x }

fn plain_fn(x: u32) -> u32 { x }
"#;
        let mut import_data = ImportData::default();
        let tcm = parse_and_typecheck(dslx, "/fake/attr_test.x", "attr_test_mod", &mut import_data)
            .expect("parse-and-typecheck success");
        let module = tcm.get_module();

        use crate::dslx::MatchableModuleMember;

        let mut cfg_fn: Option<Function> = None;
        let mut fmt_fn: Option<Function> = None;
        let mut plain_fn: Option<Function> = None;
        for i in 0..module.get_member_count() {
            if let Some(MatchableModuleMember::Function(f)) = module.get_member(i).to_matchable() {
                match f.get_identifier().as_str() {
                    "cfg_fn" => cfg_fn = Some(f),
                    "fmt_fn" => fmt_fn = Some(f),
                    "plain_fn" => plain_fn = Some(f),
                    _ => {}
                }
            }
        }

        let cfg_fn = cfg_fn.expect("cfg_fn present");
        assert_eq!(cfg_fn.get_attribute_count(), 1);
        let cfg_attr = cfg_fn.get_attribute(0);
        assert_eq!(cfg_attr.kind(), AttributeKind::Cfg);
        let cfg_args = cfg_attr.arguments();
        assert_eq!(cfg_args.len(), 1);
        match &cfg_args[0] {
            AttributeArgument::String(value) => assert_eq!(value, "test"),
            other => panic!("unexpected cfg attribute argument: {:?}", other),
        }
        assert!(cfg_attr.to_text().contains("cfg"));

        // ensure no unexpected extra functions

        let fmt_fn = fmt_fn.expect("fmt_fn present");
        assert_eq!(fmt_fn.get_attribute_count(), 1);
        let fmt_attr = fmt_fn.get_attribute(0);
        assert_eq!(fmt_attr.kind(), AttributeKind::DslxFormatDisable);
        let fmt_args = fmt_attr.arguments();
        assert_eq!(fmt_args.len(), 1);
        match &fmt_args[0] {
            AttributeArgument::String(value) => assert_eq!(value, "fmt-off"),
            other => panic!("unexpected format-disable attribute argument: {:?}", other),
        }
        assert!(fmt_attr.to_text().contains("dslx_format_disable"));

        let plain_fn = plain_fn.expect("plain_fn present");
        assert_eq!(plain_fn.get_attribute_count(), 0);
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
        let p0_ty = main_type_info
            .get_type_for_type_annotation(&p0.get_type_annotation())
            .unwrap();
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
