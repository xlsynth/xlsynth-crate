// SPDX-License-Identifier: Apache-2.0

//! Source normalization and fail-closed handling of unrepresented semantics.

use crate::liberty::cell_formula::{Term, parse_formula};
use crate::liberty::liberty_parser::{Block, BlockMember, Value};
use crate::liberty_model::{Pin, PinDirection};
use crate::liberty_proto::{UnsupportedFeature, UnsupportedSemantics};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};

// Bound generated state, rather than trusting a compact numeric declaration to
// allocate arbitrarily many scalar equations. This is an importer resource
// limit, not a limit on logical design/register widths.
const MAX_EXPANDED_BANK_BITS_PER_CELL: usize = 4096;

/// Reads a textual attribute without treating malformed values as absent.
pub(super) fn text(value: &Value) -> Result<&str, String> {
    match value {
        Value::String(value) | Value::Identifier(value) => Ok(value),
        _ => Err("expected a Liberty string or identifier".into()),
    }
}

/// Reads a finite, nonnegative electrical value in source-library units.
pub(super) fn electrical_value(value: &Value, name: &str) -> Result<f64, String> {
    match value {
        Value::Number(value) if value.is_finite() && *value >= 0.0 => Ok(*value),
        _ => Err(format!("{name} must be a finite nonnegative number")),
    }
}

#[derive(Default)]
pub(super) struct ElectricalDefaults {
    input_cap: Option<f64>,
    output_cap: Option<f64>,
    inout_cap: Option<f64>,
    max_cap: Option<f64>,
    max_transition: Option<f64>,
    max_fanout: Option<f64>,
    fanout_load: Option<f64>,
}

impl ElectricalDefaults {
    /// Captures defaults before libraries with different defaults are merged.
    pub(super) fn from_library(block: &Block) -> Result<Self, String> {
        let mut result = Self::default();
        for member in &block.members {
            let BlockMember::BlockAttr(attr) = member else {
                continue;
            };
            let slot = match attr.attr_name.as_str() {
                "default_input_pin_cap" => &mut result.input_cap,
                "default_output_pin_cap" => &mut result.output_cap,
                "default_inout_pin_cap" => &mut result.inout_cap,
                "default_max_capacitance" => &mut result.max_cap,
                "default_max_transition" => &mut result.max_transition,
                "default_max_fanout" => &mut result.max_fanout,
                "default_fanout_load" => &mut result.fanout_load,
                _ => continue,
            };
            if slot.is_some() {
                return Err(format!("duplicate {}", attr.attr_name));
            }
            *slot = Some(electrical_value(&attr.value, &attr.attr_name)?);
        }
        Ok(result)
    }

    /// Materializes only applicable defaults, retaining explicit zero values.
    pub(super) fn apply(&self, pin: &mut Pin) {
        let direction = PinDirection::try_from(pin.direction).ok();
        let cap = match direction {
            Some(PinDirection::Input) => self.input_cap,
            Some(PinDirection::Output) => self.output_cap,
            Some(PinDirection::Inout) => self.inout_cap,
            _ => None,
        };
        pin.capacitance = pin.capacitance.or(cap);
        if matches!(direction, Some(PinDirection::Input | PinDirection::Inout)) {
            pin.fanout_load = pin.fanout_load.or(self.fanout_load);
        }
        if matches!(direction, Some(PinDirection::Output | PinDirection::Inout)) {
            pin.max_capacitance = pin.max_capacitance.or(self.max_cap);
            pin.max_transition = pin.max_transition.or(self.max_transition);
            pin.max_fanout = pin.max_fanout.or(self.max_fanout);
        }
    }
}

pub(super) struct PreparedCell<'a> {
    pub block: Cow<'a, Block>,
    pub unsupported: Vec<UnsupportedSemantics>,
}

fn block_location(block: &Block) -> String {
    let names = block
        .qualifiers
        .iter()
        .filter_map(|v| text(v).ok())
        .collect::<Vec<_>>();
    format!("{}({})", block.block_type, names.join(","))
}

/// Records semantic loss before normalization, independently of timing/power
/// flags.
fn find_unsupported(block: &Block, path: &str, issues: &mut Vec<UnsupportedSemantics>) {
    let group_feature = match block.block_type.as_str() {
        "test_cell" => {
            // Test-only descriptions do not override ordinary functional
            // behavior.
            return;
        }
        "statetable" => Some(UnsupportedFeature::StateTable),
        "bus" => Some(UnsupportedFeature::Bus),
        _ => None,
    };
    if let Some(feature) = group_feature {
        issues.push(UnsupportedSemantics {
            feature: feature as i32,
            location: path.into(),
        });
        return;
    }
    let sequential = matches!(
        block.block_type.as_str(),
        "ff" | "latch" | "ff_bank" | "latch_bank"
    );
    let has_attribute = |name: &str| {
        block
            .members
            .iter()
            .any(|member| matches!(member, BlockMember::BlockAttr(attr) if attr.attr_name == name))
    };
    if sequential
        && has_attribute("clear")
        && has_attribute("preset")
        && !has_attribute("clear_preset_var1")
        && !has_attribute("clear_preset_var2")
    {
        // Missing simultaneous-value attributes do not make dual controls safe.
        // Explicit attributes are recorded individually below.
        issues.push(UnsupportedSemantics {
            feature: UnsupportedFeature::SimultaneousClearPreset as i32,
            location: format!("{path}.clear+preset"),
        });
    }
    for member in &block.members {
        match member {
            BlockMember::BlockAttr(attr) => {
                let feature = match attr.attr_name.as_str() {
                    "power_down_function" => {
                        // Consumers assume normal powered-on operation.
                        // Power-off behavior is outside this model.
                        // Power tables do not change that assumption.
                        None
                    }
                    "three_state" => Some(UnsupportedFeature::ThreeState),
                    "x_function" => Some(UnsupportedFeature::XFunction),
                    "driver_type" if block.block_type == "cell" => {
                        // At cell scope this identifies drive strength; pin
                        // scope uses the same name for electrical behavior.
                        None
                    }
                    // Driver types and their programmable forms can alter
                    // signal resolution or produce high impedance or weak
                    // values. The normalized model only supports Boolean pins.
                    "driver_type"
                    | "pull_up_function"
                    | "pull_down_function"
                    | "bus_hold_function"
                    | "open_drain_function"
                    | "open_source_function"
                    | "resistive_function"
                    | "resistive_0_function"
                    | "resistive_1_function" => Some(UnsupportedFeature::DriverType),
                    "clocked_on_also" => Some(UnsupportedFeature::SecondaryClock),
                    "enable_also" => Some(UnsupportedFeature::SecondaryEnable),
                    "clear_preset_var1" | "clear_preset_var2" => {
                        Some(UnsupportedFeature::SimultaneousClearPreset)
                    }
                    "internal_node" | "state_function" => Some(UnsupportedFeature::InternalState),
                    "next_state" | "clocked_on" | "clear" | "preset" | "data_in" | "enable" => None,
                    _ if sequential => Some(UnsupportedFeature::SequentialAttribute),
                    // Characterization and documentation are not functional
                    // equations.
                    _ => None,
                };
                if let Some(feature) = feature {
                    issues.push(UnsupportedSemantics {
                        feature: feature as i32,
                        location: format!("{path}.{}", attr.attr_name),
                    });
                }
            }
            BlockMember::SubBlock(child) => {
                let child_path = if path.is_empty() {
                    block_location(child)
                } else {
                    format!("{path}.{}", block_location(child))
                };
                if sequential {
                    issues.push(UnsupportedSemantics {
                        feature: UnsupportedFeature::SequentialGroup as i32,
                        location: child_path,
                    });
                } else if matches!(
                    child.block_type.as_str(),
                    "pin"
                        | "bundle"
                        | "bus"
                        | "ff"
                        | "latch"
                        | "ff_bank"
                        | "latch_bank"
                        | "statetable"
                        | "test_cell"
                ) {
                    find_unsupported(child, &child_path, issues);
                }
            }
        }
    }
}

fn names(block: &Block) -> Result<Vec<String>, String> {
    block
        .qualifiers
        .iter()
        .map(|value| text(value).map(str::to_owned))
        .collect()
}

fn bundle_members(block: &Block) -> Result<Vec<String>, String> {
    let mut result = None;
    for member in &block.members {
        if let BlockMember::BlockAttr(attr) = member
            && attr.attr_name == "members"
        {
            let Value::Tuple(values) = &attr.value else {
                return Err("bundle members must be a list".into());
            };
            let values = values
                .iter()
                .map(|v| text(v).map(str::to_owned))
                .collect::<Result<Vec<_>, _>>()?;
            if values.is_empty()
                || values.iter().collect::<BTreeSet<_>>().len() != values.len()
                || result.replace(values).is_some()
            {
                return Err("bundle must have one nonempty list of distinct members".into());
            }
        }
    }
    result.ok_or_else(|| "bundle is missing members".into())
}

fn add_vector(
    vectors: &mut BTreeMap<String, Vec<String>>,
    name: String,
    bits: Vec<String>,
) -> Result<(), String> {
    if vectors.insert(name.clone(), bits).is_some() {
        return Err(format!("duplicate vector name '{name}'"));
    }
    Ok(())
}

/// Substitutes whole identifiers, broadcasting scalar operands but not vectors.
fn scalar_formula(
    term: &Term,
    vectors: &BTreeMap<String, Vec<String>>,
    bit: usize,
    width: usize,
) -> Result<String, String> {
    let recurse = |term| scalar_formula(term, vectors, bit, width);
    Ok(match term {
        Term::Input(name) => scalar_name(name, vectors, bit, width)?,
        Term::Constant(value) => if *value { "1" } else { "0" }.into(),
        Term::Negate(inner) => format!("!({})", recurse(inner)?),
        Term::And(lhs, rhs) => format!("({} * {})", recurse(lhs)?, recurse(rhs)?),
        Term::Or(lhs, rhs) => format!("({} + {})", recurse(lhs)?, recurse(rhs)?),
        Term::Xor(lhs, rhs) => format!("({} ^ {})", recurse(lhs)?, recurse(rhs)?),
    })
}

fn scalar_name(
    name: &str,
    vectors: &BTreeMap<String, Vec<String>>,
    bit: usize,
    width: usize,
) -> Result<String, String> {
    match vectors.get(name) {
        Some(bits) if bits.len() == width => Ok(bits[bit].clone()),
        Some(bits) => Err(format!(
            "vector '{name}' has width {}, expected {width}",
            bits.len()
        )),
        None => Ok(name.into()),
    }
}

/// Expands vector references in equations and timing/power pin selectors.
fn scalarize(
    block: &mut Block,
    vectors: &BTreeMap<String, Vec<String>>,
    bit: usize,
    width: usize,
) -> Result<(), String> {
    for member in &mut block.members {
        match member {
            BlockMember::BlockAttr(attr) => match attr.attr_name.as_str() {
                "function" | "next_state" | "data_in" | "clocked_on" | "enable" | "clear"
                | "preset" | "when" => {
                    attr.value = Value::String(scalar_formula(
                        &parse_formula(text(&attr.value)?)?,
                        vectors,
                        bit,
                        width,
                    )?);
                }
                "related_pin" => {
                    attr.value = Value::String(
                        text(&attr.value)?
                            .split_whitespace()
                            .map(|name| scalar_name(name, vectors, bit, width))
                            .collect::<Result<Vec<_>, _>>()?
                            .join(" "),
                    );
                }
                _ => {
                    // Numeric characterization and unrelated attributes are
                    // unchanged.
                }
            },
            BlockMember::SubBlock(child) => scalarize(child, vectors, bit, width)?,
        }
    }
    Ok(())
}

/// Combines bundle defaults with explicit member overrides without duplicating
/// scalar attributes.
fn member_pin(bundle: &Block, name: &str) -> Result<Block, String> {
    let mut result = Block {
        block_type: "pin".into(),
        qualifiers: vec![Value::String(name.into())],
        members: vec![],
    };
    let mut overrides = None;
    for member in &bundle.members {
        match member {
            BlockMember::BlockAttr(attr) if attr.attr_name == "members" => {
                // Membership is structural, not a pin attribute.
            }
            BlockMember::SubBlock(child) if child.block_type == "pin" => {
                if names(child)?.iter().any(|n| n == name)
                    && overrides.replace(child.as_ref()).is_some()
                {
                    return Err(format!("duplicate bundle member override '{name}'"));
                }
            }
            _ => result.members.push(member.clone()),
        }
    }
    if let Some(overrides) = overrides {
        let inherited_groups = result
            .members
            .iter()
            .filter_map(|member| match member {
                BlockMember::SubBlock(block) => Some(block.block_type.clone()),
                BlockMember::BlockAttr(_) => None,
            })
            .collect::<BTreeSet<_>>();
        for member in &overrides.members {
            if let BlockMember::BlockAttr(attr) = member {
                if attr.attr_name == "direction"
                    && result.members.iter().any(|existing| matches!(existing, BlockMember::BlockAttr(existing) if existing.attr_name == "direction" && text(&existing.value).ok() != text(&attr.value).ok()))
                {
                    return Err(format!("bundle member '{name}' changes the bundle direction"));
                }
                result.members.retain(|existing| !matches!(existing, BlockMember::BlockAttr(existing) if existing.attr_name == attr.attr_name));
            } else if let BlockMember::SubBlock(child) = member
                && inherited_groups.contains(&child.block_type)
            {
                // Do not guess precedence between inherited and member-level
                // LUT groups.
                return Err(format!(
                    "bundle member '{name}' overrides inherited {} groups; this form is unsupported",
                    child.block_type
                ));
            }
            result.members.push(member.clone());
        }
    }
    Ok(result)
}

/// Keeps scalar cells borrowed; only vector cells need a normalized AST copy.
pub(super) fn prepare_cell(cell: &Block) -> Result<PreparedCell<'_>, String> {
    let mut unsupported = Vec::new();
    find_unsupported(cell, "", &mut unsupported);
    unsupported.sort_by(|a, b| a.location.cmp(&b.location).then(a.feature.cmp(&b.feature)));
    unsupported.dedup();
    let needs_expansion = cell.members.iter().any(|member| {
        let BlockMember::SubBlock(block) = member else {
            return false;
        };
        matches!(
            block.block_type.as_str(),
            "bundle" | "ff_bank" | "latch_bank"
        ) || (block.block_type == "pin" && block.qualifiers.len() > 1)
    });
    if !needs_expansion
        || unsupported
            .iter()
            .any(|issue| issue.feature == UnsupportedFeature::Bus as i32)
    {
        return Ok(PreparedCell {
            block: Cow::Borrowed(cell),
            unsupported,
        });
    }
    let mut vectors = BTreeMap::new();
    let mut expanded_bank_bits = 0;
    for member in &cell.members {
        let BlockMember::SubBlock(block) = member else {
            continue;
        };
        match block.block_type.as_str() {
            "bundle" => {
                let [name]: [String; 1] = names(block)?
                    .try_into()
                    .map_err(|_| "bundle must have exactly one name")?;
                let members = bundle_members(block)?;
                for child in &block.members {
                    if let BlockMember::SubBlock(child) = child
                        && child.block_type == "pin"
                    {
                        for name in names(child)? {
                            if !members.contains(&name) {
                                return Err(format!("override for unknown bundle member '{name}'"));
                            }
                        }
                    }
                }
                add_vector(&mut vectors, name, members)?;
            }
            "ff_bank" | "latch_bank" => {
                let [q, qn, Value::Number(width)] = block.qualifiers.as_slice() else {
                    return Err("bank requires two state names and a bit count".into());
                };
                if !width.is_finite()
                    || *width < 1.0
                    || width.fract() != 0.0
                    || *width > u32::MAX as f64
                {
                    return Err("invalid bank width".into());
                }
                let width = *width as usize;
                expanded_bank_bits = checked_bank_expansion(expanded_bank_bits, width)?;
                for state in [q, qn] {
                    let state = text(state)?;
                    let bits = (0..width)
                        .map(|i| format!("{state}[{i}]"))
                        .collect::<Vec<_>>();
                    add_vector(&mut vectors, state.into(), bits)?;
                }
            }
            _ => {
                // Scalar groups do not define vector aliases.
            }
        }
    }
    let mut result = Block {
        block_type: cell.block_type.clone(),
        qualifiers: cell.qualifiers.clone(),
        members: vec![],
    };
    for member in &cell.members {
        match member {
            BlockMember::SubBlock(block) if block.block_type == "bundle" => {
                let members = bundle_members(block)?;
                for (bit, name) in members.iter().enumerate() {
                    let mut pin = member_pin(block, name)?;
                    scalarize(&mut pin, &vectors, bit, members.len())?;
                    result.members.push(BlockMember::SubBlock(Box::new(pin)));
                }
            }
            BlockMember::SubBlock(block)
                if matches!(block.block_type.as_str(), "ff_bank" | "latch_bank") =>
            {
                let q = text(&block.qualifiers[0])?;
                let qn = text(&block.qualifiers[1])?;
                for bit in 0..vectors[q].len() {
                    let mut scalar = block.as_ref().clone();
                    scalar.block_type = if block.block_type == "ff_bank" {
                        "ff"
                    } else {
                        "latch"
                    }
                    .into();
                    scalar.qualifiers = vec![
                        Value::String(vectors[q][bit].clone()),
                        Value::String(vectors[qn][bit].clone()),
                    ];
                    scalarize(&mut scalar, &vectors, bit, vectors[q].len())?;
                    result.members.push(BlockMember::SubBlock(Box::new(scalar)));
                }
            }
            BlockMember::SubBlock(block) if block.block_type == "pin" => {
                for name in names(block)? {
                    let mut scalar = block.as_ref().clone();
                    scalar.qualifiers = vec![Value::String(name)];
                    scalarize(&mut scalar, &vectors, 0, 1)?;
                    result.members.push(BlockMember::SubBlock(Box::new(scalar)));
                }
            }
            _ => result.members.push(member.clone()),
        }
    }
    let mut physical_names = BTreeSet::new();
    let mut state_names = BTreeSet::new();
    for member in &result.members {
        if let BlockMember::SubBlock(block) = member {
            if block.block_type == "pin" {
                let name = text(&block.qualifiers[0])?;
                if !physical_names.insert(name) || vectors.contains_key(name) {
                    return Err(format!(
                        "duplicate or vector-colliding physical pin '{name}'"
                    ));
                }
            } else if matches!(block.block_type.as_str(), "ff" | "latch") {
                for name in &block.qualifiers {
                    let name = text(name)?;
                    if !state_names.insert(name) || vectors.contains_key(name) {
                        return Err(format!("duplicate or vector-colliding state '{name}'"));
                    }
                }
            }
        }
    }
    if let Some(name) = physical_names.intersection(&state_names).next() {
        return Err(format!("state-colliding physical pin '{name}'"));
    }
    Ok(PreparedCell {
        block: Cow::Owned(result),
        unsupported,
    })
}

/// Checks the cumulative bank budget before allocating per-bit names or ASTs.
fn checked_bank_expansion(previous: usize, width: usize) -> Result<usize, String> {
    previous.checked_add(width)
        .filter(|total| *total <= MAX_EXPANDED_BANK_BITS_PER_CELL)
        .ok_or_else(|| format!("bank expansion exceeds the per-cell limit of {MAX_EXPANDED_BANK_BITS_PER_CELL} state bits"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bank_expansion_budget_checks_boundaries_and_overflow_without_allocating() {
        let limit = MAX_EXPANDED_BANK_BITS_PER_CELL;
        assert_eq!(checked_bank_expansion(0, limit).unwrap(), limit);
        assert_eq!(checked_bank_expansion(limit - 1, 1).unwrap(), limit);
        assert!(checked_bank_expansion(0, limit + 1).is_err());
        assert!(checked_bank_expansion(limit, 1).is_err());
        assert!(checked_bank_expansion(1, usize::MAX).is_err());
    }
}
