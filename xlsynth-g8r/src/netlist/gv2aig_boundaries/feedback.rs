// SPDX-License-Identifier: Apache-2.0

//! Recognizes direct Q feedback through a hold/update mux in a mapped flop.

use std::collections::HashMap;

use crate::liberty::cell_formula::{Term, parse_formula};
use crate::liberty_model::{PinDirection, SequentialKind};
use crate::netlist::normalized::{BitExpr, BitIndex, BitSource};

use super::{BitDriver, BoundaryExtractor};

#[derive(Debug, PartialEq, Eq)]
enum BoundTerm {
    OldQ,
    Net(BitIndex),
    Literal(bool),
    Not(Box<BoundTerm>),
    And(Box<BoundTerm>, Box<BoundTerm>),
    Or(Box<BoundTerm>, Box<BoundTerm>),
    Xor(Box<BoundTerm>, Box<BoundTerm>),
}

impl BoundTerm {
    fn not(input: Self) -> Self {
        match input {
            Self::Literal(value) => Self::Literal(!value),
            Self::Not(inner) => *inner,
            input => Self::Not(Box::new(input)),
        }
    }

    fn and(lhs: Self, rhs: Self) -> Self {
        match (lhs, rhs) {
            (Self::Literal(false), _) | (_, Self::Literal(false)) => Self::Literal(false),
            (Self::Literal(true), rhs) => rhs,
            (lhs, Self::Literal(true)) => lhs,
            (lhs, rhs) => Self::And(Box::new(lhs), Box::new(rhs)),
        }
    }

    fn or(lhs: Self, rhs: Self) -> Self {
        match (lhs, rhs) {
            (Self::Literal(true), _) | (_, Self::Literal(true)) => Self::Literal(true),
            (Self::Literal(false), rhs) => rhs,
            (lhs, Self::Literal(false)) => lhs,
            (lhs, rhs) => Self::Or(Box::new(lhs), Box::new(rhs)),
        }
    }

    fn xor(lhs: Self, rhs: Self) -> Self {
        match (lhs, rhs) {
            (Self::Literal(false), rhs) => rhs,
            (lhs, Self::Literal(false)) => lhs,
            (Self::Literal(true), rhs) => Self::not(rhs),
            (lhs, Self::Literal(true)) => Self::not(lhs),
            (lhs, rhs) => Self::Xor(Box::new(lhs), Box::new(rhs)),
        }
    }

    fn contains_old_q(&self) -> bool {
        match self {
            Self::OldQ => true,
            Self::Net(_) | Self::Literal(_) => false,
            Self::Not(input) => input.contains_old_q(),
            Self::And(lhs, rhs) | Self::Or(lhs, rhs) | Self::Xor(lhs, rhs) => {
                lhs.contains_old_q() || rhs.contains_old_q()
            }
        }
    }

    fn signal(&self) -> Option<(BitIndex, bool)> {
        match self {
            Self::Net(bit) => Some((*bit, false)),
            Self::Not(input) => input.signal().map(|(bit, inverted)| (bit, !inverted)),
            _ => None,
        }
    }
}

fn bind_term(
    term: &Term,
    inputs: &HashMap<&str, BitSource>,
    output_bit: BitIndex,
) -> Result<BoundTerm, String> {
    match term {
        Term::Input(name) => match inputs.get(name.as_str()) {
            Some(BitSource::Bit(bit)) if *bit == output_bit => Ok(BoundTerm::OldQ),
            Some(BitSource::Bit(bit)) => Ok(BoundTerm::Net(*bit)),
            Some(BitSource::Literal(value)) => Ok(BoundTerm::Literal(*value)),
            Some(BitSource::Unknown) => Err("unknown input in hold-feedback formula".to_string()),
            None => Err(format!("unbound hold-feedback formula input '{name}'")),
        },
        Term::Constant(value) => Ok(BoundTerm::Literal(*value)),
        Term::Negate(input) => Ok(BoundTerm::not(bind_term(input, inputs, output_bit)?)),
        Term::And(lhs, rhs) | Term::Or(lhs, rhs) | Term::Xor(lhs, rhs) => {
            let lhs = bind_term(lhs, inputs, output_bit)?;
            let rhs = bind_term(rhs, inputs, output_bit)?;
            Ok(match term {
                Term::And(_, _) => BoundTerm::and(lhs, rhs),
                Term::Or(_, _) => BoundTerm::or(lhs, rhs),
                Term::Xor(_, _) => BoundTerm::xor(lhs, rhs),
                _ => unreachable!("matched binary term"),
            })
        }
    }
}

/// Returns the input of a directly connected inverter, if this bit has one.
fn inverter_input(extractor: &BoundaryExtractor<'_>, bit: BitIndex) -> Option<BitIndex> {
    if extractor.source_owners[bit].is_some() {
        return None;
    }
    match extractor.drivers[bit].as_slice() {
        [
            BitDriver::Assign {
                assign_index,
                rhs_bit_index,
            },
        ] => match &extractor.normalized.assigns[*assign_index].rhs_bits[*rhs_bit_index] {
            BitExpr::Not(input) => match input.as_ref() {
                BitExpr::Source(BitSource::Bit(input_bit)) => Some(*input_bit),
                _ => None,
            },
            _ => None,
        },
        [
            BitDriver::CellOutput {
                instance_index,
                connection_index,
            },
        ] => {
            let instance = &extractor.normalized.instances[*instance_index];
            let type_name = extractor.interner.resolve(instance.type_name)?;
            let pin_name = extractor
                .interner
                .resolve(instance.connections[*connection_index].port)?;
            let library = extractor.liberty;
            let cell = &library.cells[*extractor.cell_index_by_name.get(type_name)?];
            if !cell.sequential.is_empty() {
                return None;
            }
            let pin = cell.pins.iter().find(|pin| {
                pin.direction == PinDirection::Output as i32
                    && library.resolve_string(&pin.name) == pin_name
            })?;
            let Term::Negate(input) = parse_formula(library.resolve_string(&pin.function)).ok()?
            else {
                return None;
            };
            let Term::Input(input_name) = input.as_ref() else {
                return None;
            };
            let connection = instance.connections.iter().find(|connection| {
                extractor.interner.resolve(connection.port) == Some(input_name.as_str())
            })?;
            match connection.bits.as_slice() {
                [BitSource::Bit(input_bit)] => Some(*input_bit),
                _ => None,
            }
        }
        _ => None,
    }
}

fn complementary_guards(
    extractor: &BoundaryExtractor<'_>,
    lhs: &BoundTerm,
    rhs: &BoundTerm,
) -> bool {
    let (Some((lhs_bit, lhs_inverted)), Some((rhs_bit, rhs_inverted))) =
        (lhs.signal(), rhs.signal())
    else {
        return false;
    };
    if lhs_bit == rhs_bit {
        return lhs_inverted != rhs_inverted;
    }
    (inverter_input(extractor, lhs_bit) == Some(rhs_bit)
        || inverter_input(extractor, rhs_bit) == Some(lhs_bit))
        && lhs_inverted == rhs_inverted
}

fn hold_guard(term: &BoundTerm) -> Option<&BoundTerm> {
    let BoundTerm::And(lhs, rhs) = term else {
        return None;
    };
    match (lhs.as_ref(), rhs.as_ref()) {
        (BoundTerm::OldQ, guard) | (guard, BoundTerm::OldQ) if !guard.contains_old_q() => {
            Some(guard)
        }
        _ => None,
    }
}

fn is_update_arm(
    extractor: &BoundaryExtractor<'_>,
    term: &BoundTerm,
    hold_guard: &BoundTerm,
) -> bool {
    let BoundTerm::And(lhs, rhs) = term else {
        return false;
    };
    (!lhs.contains_old_q() && complementary_guards(extractor, hold_guard, rhs))
        || (!rhs.contains_old_q() && complementary_guards(extractor, hold_guard, lhs))
}

/// Recognizes `(old_q & !enable) | (data & enable)` after binding constant
/// pins. The old-Q connection must feed the mapped flop directly. The two
/// enable signals may differ by one external inverter. Returns the feedback
/// output bit whose prior value is assumed zero during collapse.
pub(super) fn recognize_direct_hold_mux(
    extractor: &BoundaryExtractor<'_>,
    instance_index: usize,
    connection_index: usize,
    formula: &Term,
    inputs: &[(String, BitSource)],
) -> Result<Option<BitIndex>, String> {
    let instance = &extractor.normalized.instances[instance_index];
    let [BitSource::Bit(output_bit)] = instance.connections[connection_index].bits.as_slice()
    else {
        return Ok(None);
    };
    if !inputs
        .iter()
        .any(|(_, source)| *source == BitSource::Bit(*output_bit))
    {
        return Ok(None);
    }
    let type_name = extractor.interner.resolve(instance.type_name).unwrap();
    let output_name = extractor
        .interner
        .resolve(instance.connections[connection_index].port)
        .unwrap();
    let library = extractor.liberty;
    let cell = &library.cells[extractor.cell_index_by_name[type_name]];
    let [seq] = cell.sequential.as_slice() else {
        return Ok(None);
    };
    if seq.kind != SequentialKind::Ff as i32
        || !seq.clear_expr.is_empty()
        || !seq.preset_expr.is_empty()
    {
        return Ok(None);
    }
    let output = cell
        .pins
        .iter()
        .find(|pin| library.resolve_string(&pin.name) == output_name)
        .expect("driver indexing validated output pin");
    if !matches!(
        parse_formula(library.resolve_string(&output.function)),
        Ok(Term::Input(name)) if name == seq.state_var
    ) {
        return Ok(None);
    }
    let bindings = inputs
        .iter()
        .map(|(name, source)| (name.as_str(), *source))
        .collect();
    let bound = bind_term(formula, &bindings, *output_bit)?;
    let BoundTerm::Or(lhs, rhs) = &bound else {
        return Ok(None);
    };
    let recognized = hold_guard(lhs).is_some_and(|guard| is_update_arm(extractor, rhs, guard))
        || hold_guard(rhs).is_some_and(|guard| is_update_arm(extractor, lhs, guard));
    Ok(recognized.then_some(*output_bit))
}
