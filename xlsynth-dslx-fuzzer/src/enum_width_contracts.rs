// SPDX-License-Identifier: Apache-2.0

//! Casts between integer and nominal enum representations of independent
//! widths.

use crate::converter_contracts::{ContractCase, scalar_identity};

/// Selects between two declared enum members and observes their identities.
/// Both numbers fit both widths, so every cast must succeed even in a constant.
pub(crate) fn generate(data: &[u8]) -> ContractCase {
    let option = |index: usize| data.get(index).copied().unwrap_or_default();
    let flags = option(0);
    let source_width = 1 + option(1) % 8;
    let enum_width = 1 + option(2) % 8;
    let common_width = source_width.min(enum_width);
    let choice = i32::from(option(3));
    let (sign, member) = if flags & 4 != 0 {
        let max_magnitude = 1 << (common_width - 1);
        ('s', -(1 + choice % max_magnitude))
    } else {
        let max_value = (1 << common_width) - 1;
        ('u', 1 + choice % max_value)
    };
    let source_type = format!("{sign}{source_width}");
    let enum_repr = format!("{sign}{enum_width}");
    let cast = |value: &str| {
        if flags & 2 != 0 {
            format!("({value} as {enum_repr}) as State")
        } else {
            format!("{value} as State")
        }
    };
    let (family, body) = if flags & 1 != 0 {
        let cast = cast("raw");
        (
            "enum_width_runtime",
            format!(
                r#"pub fn main(x: bool) -> bool {{
    let raw = if x {{ {source_type}:{member} }} else {{ {source_type}:0 }};
    ({cast}) == State::Active
}}"#
            ),
        )
    } else {
        let cast = cast(&format!("{source_type}:{member}"));
        (
            "enum_width_constexpr",
            format!(
                r#"const ACTIVE = {cast};
pub fn main(x: bool) -> bool {{
    let state = if x {{ ACTIVE }} else {{ State::Idle }};
    state == State::Active
}}"#
            ),
        )
    };
    let source = format!("enum State : {enum_repr} {{ Idle = 0, Active = {member} }}\n{body}\n");
    ContractCase::single(family, source, scalar_identity(1))
}
