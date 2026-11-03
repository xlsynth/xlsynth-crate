// SPDX-License-Identifier: Apache-2.0

//! Recognition of low-transparent integrated clock gates in source Liberty.

use super::cell_formula::{Term, parse_formula};
use super::import::text;
use super::liberty_parser::{Block, BlockMember};
use crate::liberty_model::{Library, Pin};
use crate::liberty_proto::{ClockGate, PinDirection, UnsupportedFeature, UnsupportedSemantics};
use std::collections::{BTreeSet, HashMap, HashSet};

/// Requires one unambiguous role for each external pin in the stored gate.
pub(super) fn validate_pin_roles(
    cell_name: &str,
    pins: &[Pin],
    library: &Library,
    gate: &ClockGate,
) -> Result<(), String> {
    let mut roles = HashMap::new();
    for (name, direction) in std::iter::once((&gate.clock_pin, PinDirection::Input))
        .chain(std::iter::once((&gate.output_pin, PinDirection::Output)))
        .chain(
            gate.enable_pins
                .iter()
                .chain(&gate.test_pins)
                .map(|name| (name, PinDirection::Input)),
        )
    {
        if name.is_empty() || roles.insert(name.as_str(), direction).is_some() {
            return Err(format!(
                "cell '{}' has missing or duplicate clock-gate pin roles",
                cell_name
            ));
        }
    }
    let mut pin_names = HashSet::new();
    let mut internal_seen = false;
    for pin in pins {
        let name = library.resolve_string(&pin.name);
        if name.is_empty()
            || !pin_names.insert(name)
            || !pin.function.is_unset()
            || (pin.is_clocking_pin && name != gate.clock_pin)
        {
            return Err(format!(
                "cell '{}' has a duplicate or conflicting clock-gate pin '{}'",
                cell_name, name
            ));
        }
        if pin.direction == PinDirection::Internal as i32 && !internal_seen {
            // A certified raw Liberty table names its private latch state.
            internal_seen = true;
        } else if roles.remove(name).map(|dir| dir as i32) != Some(pin.direction) {
            return Err(format!(
                "cell '{}' has unexpected clock-gate pin '{}' or direction",
                cell_name, name
            ));
        }
    }
    if !roles.is_empty() {
        return Err(format!(
            "cell '{}' lacks a declared clock-gate pin",
            cell_name
        ));
    }
    Ok(())
}

/// Returns a source attribute only if it occurs exactly once.
fn unique_attr<'a>(block: &'a Block, key: &str) -> Option<&'a str> {
    let mut attributes = block.members.iter().filter_map(|member| match member {
        BlockMember::BlockAttr(attr) if attr.attr_name == key => Some(&attr.value),
        _ => None,
    });
    let value = attributes.next()?;
    if attributes.next().is_some() {
        return None;
    }
    text(value).ok()
}

/// Certifies every binary combination of clock, enable, test and prior latch
/// state.
fn verifies_latch_table(table: &str, columns: &[&str], clock: &str) -> bool {
    let Some(clock_index) = columns.iter().position(|name| *name == clock) else {
        return false;
    };
    let table = table.replace("\\\r\n", "").replace("\\\n", "");
    // The caller has checked for exactly two or three inputs. The remaining
    // bit is the current latch state, so there are at most sixteen cases.
    let mut covered = vec![false; 1 << (columns.len() + 1)];
    for row in table.split(',') {
        let fields: Vec<_> = row.split(':').map(str::trim).collect();
        let [inputs, current, next] = fields.as_slice() else {
            return false;
        };
        let mut pattern: Vec<_> = inputs.split_ascii_whitespace().collect();
        if pattern.len() != columns.len() || !matches!(*next, "L" | "H" | "N") {
            return false;
        }
        pattern.push(*current);
        if !pattern
            .iter()
            .all(|level| matches!(*level, "L" | "H" | "-"))
        {
            return false;
        }
        for (case, covered) in covered.iter_mut().enumerate() {
            let high = |index: usize| case & (1 << index) != 0;
            if !pattern
                .iter()
                .enumerate()
                .all(|(i, &level)| level == "-" || (level == "H") == high(i))
            {
                continue;
            }
            let held = high(columns.len());
            let load = (0..columns.len()).any(|i| i != clock_index && high(i));
            let actual = if *next == "N" { held } else { *next == "H" };
            let expected = if high(clock_index) { held } else { load };
            if actual != expected {
                return false;
            }
            *covered = true;
        }
    }
    covered.into_iter().all(|seen| seen)
}

/// Checks the latch table, internal state and output equation before treating
/// exactly those source features as represented by the normalized gate model.
pub(super) fn certifies_low_latch_gate(
    block: &Block,
    gate: &ClockGate,
    unsupported: &[UnsupportedSemantics],
) -> bool {
    if unique_attr(block, "clock_gating_integrated_cell") != Some("latch_posedge_precontrol")
        || gate.clock_pin.is_empty()
        || gate.output_pin.is_empty()
        || gate.enable_pins.len() != 1
        || gate.test_pins.len() > 1
    {
        return false;
    }
    let blocks = block.members.iter().filter_map(|member| match member {
        BlockMember::SubBlock(child) => Some(child.as_ref()),
        _ => None,
    });
    let mut tables = blocks
        .clone()
        .filter(|child| child.block_type == "statetable");
    let Some(table) = tables.next() else {
        return false;
    };
    if tables.next().is_some() || table.members.len() != 1 {
        return false;
    }
    let [inputs, state] = table.qualifiers.as_slice() else {
        return false;
    };
    let (Ok(inputs), Ok(state)) = (text(inputs), text(state)) else {
        return false;
    };
    let columns: Vec<_> = inputs.split_ascii_whitespace().collect();
    let expected: BTreeSet<_> = std::iter::once(gate.clock_pin.as_str())
        .chain(
            gate.enable_pins
                .iter()
                .chain(&gate.test_pins)
                .map(String::as_str),
        )
        .collect();
    if expected.len() != 2 + gate.test_pins.len()
        || columns.len() != expected.len()
        || columns.iter().copied().collect::<BTreeSet<_>>() != expected
        || !unique_attr(table, "table")
            .is_some_and(|value| verifies_latch_table(value, &columns, &gate.clock_pin))
    {
        return false;
    }
    let pins: Vec<_> = blocks.filter(|child| child.block_type == "pin").collect();
    // Pin normalization uses the last value of several scalar attributes.
    // A contradictory or malformed role must not acquire a certified model.
    for pin in &pins {
        let mut seen = HashSet::new();
        for member in &pin.members {
            let BlockMember::BlockAttr(attr) = member else {
                continue;
            };
            let name = attr.attr_name.as_str();
            if name.starts_with("clock_gate_") || name == "clock" {
                let Ok(value) = text(&attr.value) else {
                    return false;
                };
                if !(value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("false")) {
                    return false;
                }
            } else if !matches!(
                name,
                "direction" | "function" | "state_function" | "internal_node"
            ) {
                continue;
            }
            if !seen.insert(name) {
                return false;
            }
        }
    }
    let find_pin = |name: &str| {
        pins.iter()
            .copied()
            .find(|pin| matches!(pin.qualifiers.as_slice(), [value] if text(value) == Ok(name)))
    };
    let Some(internal) = find_pin(state) else {
        return false;
    };
    let Some(output) = find_pin(&gate.output_pin) else {
        return false;
    };
    if unique_attr(internal, "direction") != Some("internal")
        || unique_attr(internal, "internal_node") != Some(state)
        || unique_attr(output, "direction") != Some("output")
    {
        return false;
    }
    let Some(Term::And(lhs, rhs)) =
        unique_attr(output, "state_function").and_then(|formula| parse_formula(formula).ok())
    else {
        return false;
    };
    let (Term::Input(lhs), Term::Input(rhs)) = (*lhs, *rhs) else {
        return false;
    };
    if !((lhs == gate.clock_pin && rhs == state) || (rhs == gate.clock_pin && lhs == state)) {
        return false;
    }
    let represented = [
        (
            UnsupportedFeature::StateTable,
            format!("statetable({inputs},{state})"),
        ),
        (
            UnsupportedFeature::InternalState,
            format!("pin({state}).internal_node"),
        ),
        (
            UnsupportedFeature::InternalState,
            format!("pin({}).state_function", gate.output_pin),
        ),
    ];
    unsupported.len() == represented.len()
        && represented.iter().all(|(feature, location)| {
            unsupported
                .iter()
                .any(|issue| issue.feature == *feature as i32 && issue.location == *location)
        })
}
