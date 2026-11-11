// SPDX-License-Identifier: Apache-2.0

//! Imported type aliases in another alias's right-hand side.

use crate::{SourceModule, converter_contracts::ContractCase};

/// Checks that an imported scalar alias preserves its width and signedness
/// when another alias names it. Qualified imports and directly imported enums
/// serve as controls for the scalar RHS and `use` syntax respectively.
pub(crate) fn scalar_alias(data: &[u8]) -> ContractCase {
    let flags = data.first().copied().unwrap_or_default();
    let form = flags & 3;
    let widths = [8, 1, 5, 16];
    let width = widths[usize::from(data.get(1).copied().unwrap_or_default()) % widths.len()];
    let (prefix, extension, other) = if flags & 4 == 0 {
        ("u", "zero_ext", 1)
    } else {
        ("s", "sign_ext", -1)
    };
    let ty = format!("{prefix}{width}");
    let library = if form == 2 {
        format!("pub enum Lane : {ty} {{ Zero = 0, Other = {other} }}\n")
    } else if flags & 8 != 0 {
        format!("type Local = {ty};\npub type Lane = Local;\n")
    } else {
        format!("pub type Lane = {ty};\n")
    };
    // The `use_syntax` feature disables `import` in the same module.
    let (imports, alias_rhs) = if form == 1 {
        ("import lib;", "lib::Lane")
    } else {
        ("#![feature(use_syntax)]\nuse lib::Lane;", "Lane")
    };
    let source = format!(
        r#"{imports}
type Word = {alias_rhs};
pub fn main(x: Word) -> u32 {{ x as u32 }}
"#
    );
    let expected_ir = format!(
        r#"package expected

top fn main(x: bits[{width}] id=1) -> bits[32] {{
  ret result: bits[32] = {extension}(x, new_bit_count=32, id=2)
}}
"#
    );
    ContractCase {
        family: "imported_scalar_alias_rhs",
        modules: vec![
            SourceModule {
                name: "lib".into(),
                text: library,
            },
            SourceModule {
                name: "sample".into(),
                text: source,
            },
        ],
        expected_ir,
        max_and_nodes: None,
    }
}

#[cfg(test)]
mod tests {
    use super::scalar_alias;

    #[test]
    fn direct_scalar_alias_preserves_original_rhs() {
        let direct = scalar_alias(&[]);
        let qualified = scalar_alias(&[1]);
        let nominal = scalar_alias(&[2]);
        assert_eq!(direct.modules[0].text, "pub type Lane = u8;\n");
        assert_eq!(
            direct.modules[1].text,
            r#"#![feature(use_syntax)]
use lib::Lane;
type Word = Lane;
pub fn main(x: Word) -> u32 { x as u32 }
"#
        );
        assert_eq!(
            qualified.modules[1].text,
            r#"import lib;
type Word = lib::Lane;
pub fn main(x: Word) -> u32 { x as u32 }
"#
        );
        assert_eq!(direct.modules[1].text, nominal.modules[1].text);
        assert_eq!(direct.expected_ir, qualified.expected_ir);
        assert_eq!(direct.expected_ir, nominal.expected_ir);
    }
}
