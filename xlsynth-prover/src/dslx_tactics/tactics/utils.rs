// SPDX-License-Identifier: Apache-2.0

//! Shared utilities for proof tactics.

use std::collections::BTreeSet;

use crate::dslx_tactics::obligations::LecObligation;

/// Allocates distinct symbols without identifying new function pairs with
/// assumptions inherited from either side of an enclosing proof.
pub(crate) fn fresh_uf_aliases(
    lec: &LecObligation,
    prefix: &str,
    first_index: usize,
    count: usize,
) -> Vec<String> {
    let inherited: BTreeSet<&str> = lec
        .lhs
        .uf_map
        .values()
        .chain(lec.rhs.uf_map.values())
        .map(String::as_str)
        .collect();
    let mut aliases = Vec::with_capacity(count);
    let mut index = first_index;
    while aliases.len() < count {
        let candidate = format!("{prefix}{index}");
        index += 1;
        if !inherited.contains(candidate.as_str()) {
            aliases.push(candidate);
        }
    }
    aliases
}

/// Returns true if the given string is a valid identifier for our purposes:
/// non-empty and consisting only of ASCII alphanumerics or underscore.
pub(crate) fn is_valid_ident(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}
