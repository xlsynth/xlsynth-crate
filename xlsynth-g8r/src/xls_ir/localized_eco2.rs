// SPDX-License-Identifier: Apache-2.0

//! Localized ECO builder expressed via simple rebasing.
//!
//! This module provides a direct constructor for a patched function that
//! preserves as much of the existing implementation as possible while taking on
//! the behavior of the new implementation. It does not compute edit sequences;
//! instead it leverages structural reuse during a rebase operation.

use crate::xls_ir::ir::Fn as IrFn;
use crate::xls_ir::simple_rebase::rebase_onto;

/// Builds a patched version of `old` that behaves like `new`, reusing all
/// structurally equivalent nodes from `old` and only allocating new node ids
/// for newly synthesized nodes.
///
/// Precondition: `old` and `new` must have identical function types.
pub fn compute_localized_eco(old: &IrFn, new: &IrFn) -> IrFn {
    assert_eq!(
        old.get_type(),
        new.get_type(),
        "Function signatures must match for ECO build",
    );

    // Allocate new text ids deterministically beyond the maximum in `old`.
    let mut next_id: usize = old
        .nodes
        .iter()
        .map(|n| n.text_id)
        .max()
        .unwrap_or(0)
        .saturating_add(1);

    // Use the new function's name for the result. Parameters and any reused
    // nodes are inherited from `old` by the rebase routine.
    rebase_onto(new, old, &new.name, || {
        let id = next_id;
        next_id += 1;
        id
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xls_ir::ir_parser::Parser;

    fn parse_fn(ir_body: &str) -> IrFn {
        let pkg_text = format!("package test\n\n{}\n", ir_body);
        let mut p = Parser::new(&pkg_text);
        let pkg = p.parse_and_validate_package().unwrap();
        pkg.get_top().unwrap().clone()
    }

    #[test]
    fn preserves_existing_when_identical() {
        let f = parse_fn(
            r#"fn id(a: bits[8] id=1) -> bits[8] {
  ret identity.2: bits[8] = identity(a, id=2)
}"#,
        );
        let g = parse_fn(
            r#"fn id(a: bits[8] id=1) -> bits[8] {
  ret identity.2: bits[8] = identity(a, id=2)
}"#,
        );
        let result = compute_localized_eco(&f, &g);
        let expected = r#"fn id(a: bits[8] id=1) -> bits[8] {
  ret identity.2: bits[8] = identity(a, id=2)
}"#;
        assert_eq!(result.to_string(), expected);
        // Ensure no additional nodes were introduced.
        assert_eq!(result.nodes.len(), f.nodes.len());
    }

    #[test]
    fn adds_only_needed_work_and_reuses_old() {
        let old = parse_fn(
            r#"fn f(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  ret add.3: bits[8] = add(a, b, id=3)
}"#,
        );
        let new = parse_fn(
            r#"fn f(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  add.3: bits[8] = add(a, b, id=3)
  ret umul.4: bits[8] = umul(add.3, b, id=4)
}"#,
        );
        let result = compute_localized_eco(&old, &new);
        // Expect one extra node over old.
        assert_eq!(result.nodes.len(), old.nodes.len() + 1);
        // Full IR textual expectation.
        let expected = r#"fn f(a: bits[8] id=1, b: bits[8] id=2) -> bits[8] {
  add.3: bits[8] = add(a, b, id=3)
  ret umul.4: bits[8] = umul(add.3, b, id=4)
}"#;
        assert_eq!(result.to_string(), expected);
        assert_eq!(result.get_type(), new.get_type());
    }
}
