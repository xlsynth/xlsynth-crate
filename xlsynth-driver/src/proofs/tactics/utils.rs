// SPDX-License-Identifier: Apache-2.0

//! Shared utilities for proof tactics.

/// Returns true if the given string is a valid identifier for our purposes:
/// non-empty and consisting only of ASCII alphanumerics or underscore.
pub(crate) fn is_valid_ident(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
}
