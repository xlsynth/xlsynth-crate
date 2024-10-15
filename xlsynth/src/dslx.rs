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
    pub fn default() -> Self {
        Self::new(None, &[])
    }

    pub fn new(
        dslx_stdlib_path: Option<&std::path::Path>,
        additional_search_paths: &[&std::path::Path],
    ) -> Self {
        let dslx_stdlib_path = dslx_stdlib_path.unwrap_or(std::path::Path::new("xls/dslx/stdlib"));

        let dslx_stdlib_path_c_str =
            std::ffi::CString::new(dslx_stdlib_path.to_str().unwrap()).unwrap();
        let dslx_stdlib_path_c_ptr = dslx_stdlib_path_c_str.as_ptr();

        let additional_search_paths_c_str: Vec<std::ffi::CString> = additional_search_paths.into_iter().map(|p| std::ffi::CString::new(p.to_str().unwrap()).unwrap()).collect();
        let additional_search_paths_c_ptr: Vec<*const std::os::raw::c_char> = additional_search_paths_c_str.into_iter().map(|p| p.as_ptr()).collect();
        ImportData {
            ptr: Rc::new(ImportDataPtr {
                ptr: unsafe {
                    sys::xls_dslx_import_data_create(
                        dslx_stdlib_path_c_ptr,
                        additional_search_paths_c_ptr.as_ptr(),
                        additional_search_paths_c_ptr.len(),
                    )
                },
            }),
        }
    }
}

/// Simple wrapper around the typechecked module entity that has a `Drop`
/// implementation
struct TypecheckedModulePtr {
    parent: Rc<ImportDataPtr>,
    pub(crate) ptr: *mut sys::CDslxTypecheckedModule,
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
}

pub struct Module {
    parent: Rc<TypecheckedModulePtr>,
    pub(crate) ptr: *mut sys::CDslxModule,
}

impl Module {
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
}

pub struct EnumMember {
    parent: Rc<TypecheckedModulePtr>,
    pub(crate) ptr: *mut sys::CDslxEnumMember,
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
    pub(crate) ptr: *mut sys::CDslxExpr,
}

// -- EnumDef

pub struct EnumDef {
    parent: Rc<TypecheckedModulePtr>,
    pub(crate) ptr: *mut sys::CDslxEnumDef,
}

impl EnumDef {
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
}

// -- TypeAnnotation

pub struct TypeAnnotation {
    parent: Rc<TypecheckedModulePtr>,
    pub(crate) ptr: *mut sys::CDslxTypeAnnotation,
}

// -- StructDef

pub struct StructMember {
    parent: Rc<TypecheckedModulePtr>,
    pub(crate) ptr: *mut sys::CDslxStructMember,
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
    pub(crate) ptr: *mut sys::CDslxStructDef,
}

impl StructDef {
    pub fn get_identifier(&self) -> String {
        unsafe {
            let c_str = sys::xls_dslx_struct_def_get_identifier(self.ptr);
            c_str_to_rust(c_str)
        }
    }

    pub fn is_parametric(&self) -> bool {
        unsafe { sys::xls_dslx_struct_def_is_parametric(self.ptr) }
    }

    // TODO(cdleary): 2024-10-06 This implementation is missing from the library.
    /*
    pub fn get_member_count(&self) -> usize {
        unsafe { sys::xls_dslx_struct_def_get_member_count(self.ptr) as usize }
    }
    */

    pub fn get_member(&self, idx: usize) -> StructMember {
        StructMember {
            parent: self.parent.clone(),
            ptr: unsafe { sys::xls_dslx_struct_def_get_member(self.ptr, idx as i64) },
        }
    }
}

struct InterpValuePtr {
    pub(crate) ptr: *mut sys::CDslxInterpValue,
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
            let error_out_str: String = unsafe { c_str_to_rust(error_out) };
            Err(XlsynthError(error_out_str))
        }
    }
}

pub struct TypeInfo {
    parent: Rc<TypecheckedModulePtr>,
    pub(crate) ptr: *mut sys::CDslxTypeInfo,
}

impl TypeInfo {
    pub fn get_const_expr(&self, expr: Expr) -> Result<InterpValue, XlsynthError> {
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
            let error_out_str: String = unsafe { c_str_to_rust(error_out) };
            Err(XlsynthError(error_out_str))
        }
    }

    pub fn get_type_for_type_annotation(&self, type_annotation: TypeAnnotation) -> Type {
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
}

pub struct Type {
    parent: Rc<TypecheckedModulePtr>,
    pub(crate) ptr: *mut sys::CDslxType,
}

impl Type {
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
            let error_out_str: String = unsafe { c_str_to_rust(error_out) };
            Err(XlsynthError(error_out_str))
        }
    }
}

pub fn parse_and_typecheck(
    dslx: &str,
    path: &str,
    module_name: &str,
    import_data: &mut ImportData,
) -> Result<TypecheckedModule, XlsynthError> {
    unsafe {
        let program_c_str = std::ffi::CString::new(dslx).unwrap();
        let path_c_str = std::ffi::CString::new(path).unwrap();
        let module_name_c_str = std::ffi::CString::new(module_name).unwrap();
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
            return Ok(TypecheckedModule {
                ptr: Rc::new(TypecheckedModulePtr {
                    parent: import_data.ptr.clone(),
                    ptr: result_out,
                }),
            });
        }

        let error_out_str: String = c_str_to_rust(error_out);
        return Err(XlsynthError(error_out_str));
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
            .get_const_expr(expr)
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

        // TODO(cdleary): 2024-10-06 This implementation is missing from the library.
        // assert_eq!(struct_def.get_member_count(), 2);

        let member_a = struct_def.get_member(0);
        assert_eq!(member_a.get_name(), "a");
        let type_a = member_a.get_type();
        let concrete_type_a = type_info.get_type_for_type_annotation(type_a);
        assert_eq!(concrete_type_a.get_total_bit_count().unwrap(), 32);

        let member_b = struct_def.get_member(1);
        assert_eq!(member_b.get_name(), "b");
        let type_b = member_b.get_type();
        let concrete_type_b = type_info.get_type_for_type_annotation(type_b);
        assert_eq!(concrete_type_b.get_total_bit_count().unwrap(), 16);
    }
}
