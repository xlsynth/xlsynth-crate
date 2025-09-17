// SPDX-License-Identifier: Apache-2.0

use regex::Regex;

/// Returns a filtered view of `assertions`, including only those whose labels
/// match the given regex `include`, or all assertions if `include` is `None`.
pub fn filter_assertions<'a, R>(
    assertions: &'a [crate::types::Assertion<'a, R>],
    include: Option<&Regex>,
) -> Vec<&'a crate::types::Assertion<'a, R>> {
    match include {
        None => assertions.iter().collect(),
        Some(regex) => assertions
            .iter()
            .filter(|a| regex.is_match(a.label))
            .collect(),
    }
}
