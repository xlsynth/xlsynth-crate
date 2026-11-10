// SPDX-License-Identifier: Apache-2.0

//! Recognition of low-transparent integrated clock gates in source Liberty.

use super::cell_formula::{Term, parse_formula};
use super::import::text;
use super::liberty_parser::{Block, BlockMember};
use crate::liberty_model::{Library, Pin};
use crate::liberty_proto::{
    ClockGate, ClockGateBehavior, PinDirection, UnsupportedFeature, UnsupportedSemantics,
};
use std::collections::{HashMap, HashSet};

/// The clock, functional enable, and optional test input of a declared gate.
fn input_pins(gate: &ClockGate) -> impl Iterator<Item = &str> {
    std::iter::once(&gate.clock_pin)
        .chain(&gate.enable_pins)
        .chain(&gate.test_pins)
        .map(String::as_str)
}

/// Requires one unambiguous role for each external pin in the stored gate.
pub(super) fn validate_pin_roles(
    cell_name: &str,
    pins: &[Pin],
    library: &Library,
    gate: &ClockGate,
) -> Result<(), String> {
    if gate.enable_pins.len() != 1 || gate.test_pins.len() > 1 {
        return Err(format!(
            "cell '{cell_name}' has incompatible clock-gate controls"
        ));
    }
    let mut roles: HashMap<_, _> = input_pins(gate)
        .map(|name| (name, PinDirection::Input))
        .chain([(gate.output_pin.as_str(), PinDirection::Output)])
        .collect();
    if roles.contains_key("") || roles.len() != 2 + gate.enable_pins.len() + gate.test_pins.len() {
        return Err(format!(
            "cell '{cell_name}' has missing or duplicate clock-gate pin roles"
        ));
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
///
/// Liberty rows read `inputs : current state : next state`. `L` and `H` mean
/// 0 and 1. A `-` matches either value, and `N` holds the current state.
/// With inputs `CLK ENA SE`, the low-clock rows set the latch to `ENA | SE`.
/// For example, `L H L : - : H` sets it to 1. `H - - : - : N` holds it
/// while CLK is high, so neither enable can start or shorten a pulse.
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
        let pattern: Vec<_> = inputs.split_ascii_whitespace().chain([*current]).collect();
        if pattern.len() != columns.len() + 1
            || !matches!(*next, "L" | "H" | "N")
            || !pattern
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
pub(super) fn source_gate_behavior(
    block: &Block,
    gate: &ClockGate,
    unsupported: &[UnsupportedSemantics],
) -> Option<ClockGateBehavior> {
    if unique_attr(block, "clock_gating_integrated_cell") != Some("latch_posedge_precontrol") {
        return None;
    }
    let blocks = block.members.iter().filter_map(|member| match member {
        BlockMember::SubBlock(child) => Some(child.as_ref()),
        _ => None,
    });
    let mut tables = blocks
        .clone()
        .filter(|child| child.block_type == "statetable");
    let table = tables.next()?;
    if tables.next().is_some() || table.members.len() != 1 {
        return None;
    }
    let [inputs, states] = table.qualifiers.as_slice() else {
        return None;
    };
    let inputs = text(inputs).ok()?;
    // The qualifier lists state names. Keep its source spelling for the
    // unsupported-semantics location, and use its single name for pin lookup.
    let states = text(states).ok()?;
    let mut state_names = states.split_ascii_whitespace();
    let state = state_names.next()?;
    let columns: Vec<_> = inputs.split_ascii_whitespace().collect();
    // Pin-role validation has already established that the names are distinct.
    if state_names.next().is_some()
        || columns.len() != input_pins(gate).count()
        || !input_pins(gate).all(|pin| columns.contains(&pin))
        || !verifies_latch_table(unique_attr(table, "table")?, &columns, &gate.clock_pin)
    {
        return None;
    }
    let pins = blocks.filter(|child| child.block_type == "pin");
    // Pin normalization uses the last value of several scalar attributes.
    // A contradictory or malformed role must not acquire a certified model.
    for pin in pins.clone() {
        let mut seen = HashSet::new();
        for member in &pin.members {
            let BlockMember::BlockAttr(attr) = member else {
                continue;
            };
            let name = attr.attr_name.as_str();
            if name.starts_with("clock_gate_") || name == "clock" {
                let value = text(&attr.value).ok()?;
                if !(value.eq_ignore_ascii_case("true") || value.eq_ignore_ascii_case("false")) {
                    return None;
                }
            } else if !matches!(
                name,
                "direction" | "function" | "state_function" | "internal_node"
            ) {
                continue;
            }
            if !seen.insert(name) {
                return None;
            }
        }
    }
    let find_pin = |name: &str| {
        pins.clone()
            .find(|pin| matches!(pin.qualifiers.as_slice(), [value] if text(value) == Ok(name)))
    };
    let internal = find_pin(state)?;
    let output = find_pin(&gate.output_pin)?;
    if unique_attr(internal, "direction") != Some("internal")
        || unique_attr(internal, "internal_node") != Some(state)
        || unique_attr(output, "direction") != Some("output")
    {
        return None;
    }
    let Term::And(lhs, rhs) = parse_formula(unique_attr(output, "state_function")?).ok()? else {
        return None;
    };
    let (Term::Input(lhs), Term::Input(rhs)) = (*lhs, *rhs) else {
        return None;
    };
    if !((lhs == gate.clock_pin && rhs == state) || (rhs == gate.clock_pin && lhs == state)) {
        return None;
    }
    let recorded = |feature: UnsupportedFeature, location: String| {
        unsupported
            .iter()
            .any(|issue| issue.feature == feature as i32 && issue.location == location)
    };
    let covered = unsupported.len() == 3
        && recorded(
            UnsupportedFeature::StateTable,
            format!("statetable({inputs},{states})"),
        )
        && recorded(
            UnsupportedFeature::InternalState,
            format!("pin({state}).internal_node"),
        )
        && recorded(
            UnsupportedFeature::InternalState,
            format!("pin({}).state_function", gate.output_pin),
        );
    covered.then_some(ClockGateBehavior::LowLatchEnableOrTest)
}
