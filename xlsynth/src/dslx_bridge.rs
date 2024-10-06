// SPDX-License-Identifier: Apache-2.0

//! Library for generating Rust code that reflects the types and callables in a DSLX module subtree.
//! 
//! We walk the type definitions and callable interfaces present in the DSLX module and generate
//! corresponding Rust code that can be `use`'d into a Rust module.

use crate::dslx;

fn convert_leaf_module(import_data: &mut dslx::ImportData, dslx_program: &str) -> Result<String, XlsynthError> {
    let typechecked_module = dslx::parse_and_typecheck(dslx_program)?;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_leaf_module() {
        let dslx = r#"
        enum MyEnum : u2 { A = 0, B = 3 };
        "#;
        let mut import_data = dslx::ImportData::new();
        let rust = convert_leaf_module(&mut import_data, dslx).unwrap();
        assert_eq!(rust, r#"pub mod my_module {
    enum MyEnum {
        A = 0,
        B = 3,
    }

    impl 
"#);
    }
}