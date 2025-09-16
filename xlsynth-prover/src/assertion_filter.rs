// SPDX-License-Identifier: Apache-2.0

use regex::RegexSet;

/// Returns a filtered view of `assertions`, including only those whose labels
/// match at least one regex in `include`, or all assertions if `include` is
/// `None`.
pub fn filter_assertions<'a, R>(
    assertions: &'a [crate::types::Assertion<'a, R>],
    include: Option<&RegexSet>,
) -> Vec<&'a crate::types::Assertion<'a, R>> {
    match include {
        None => assertions.iter().collect(),
        Some(set) => assertions
            .iter()
            .filter(|a| set.is_match(a.label))
            .collect(),
    }
}
