// SPDX-License-Identifier: Apache-2.0

use regex::Regex;

use crate::types::Assertion;

pub fn filter_assertions<'a, R>(
    assertions: &'a [Assertion<'a, R>],
    include_filter: Option<&Regex>,
) -> Vec<&'a Assertion<'a, R>> {
    match include_filter {
        None => assertions.iter().collect(),
        Some(regex) => assertions
            .iter()
            .filter(|assertion| regex.is_match(assertion.label))
            .collect(),
    }
}
