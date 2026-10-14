// SPDX-License-Identifier: Apache-2.0

//! Library-specific reference buffers; boundary fanout policy lives in
//! consumers.

use std::collections::HashMap;

use crate::liberty::cell_formula::parse_formula;
use crate::liberty_model::{Library, Pin, PinDirection};

/// Finds the input of a usable, ordinary, one-input/one-output buffer.
pub fn buffer_input_pin<'a>(library: &'a Library, name: &str) -> Result<&'a Pin, String> {
    let mut cells = library.cells.iter().filter(|cell| cell.name == name);
    let cell = cells
        .next()
        .ok_or_else(|| format!("reference buffer '{name}' is absent from Liberty"))?;
    if cells.next().is_some() {
        return Err(format!("reference buffer '{name}' is ambiguous"));
    }
    cell.validate_functional_semantics()?;
    if cell.dont_use == Some(true)
        || !cell.sequential.is_empty()
        || cell.clock_gate.is_some()
        || cell.pins.len() != 2
        || cell.pins.iter().any(|pin| pin.is_clocking_pin)
    {
        return Err(format!("'{name}' is not a usable ordinary buffer"));
    }
    let input = cell
        .pins
        .iter()
        .find(|pin| pin.direction == PinDirection::Input as i32)
        .ok_or_else(|| format!("buffer '{name}' must have one input"))?;
    let output = cell
        .pins
        .iter()
        .find(|pin| pin.direction == PinDirection::Output as i32)
        .ok_or_else(|| format!("buffer '{name}' must have one output"))?;
    let input_name = library.resolve_string(&input.name);
    let function = parse_formula(library.resolve_string(&output.function))?;
    if function.inputs().iter().any(|name| name != input_name)
        || ![false, true].into_iter().all(|value| {
            function.evaluate_partial(&HashMap::from([(input_name.to_owned(), value)]))
                == Some(value)
        })
    {
        return Err(format!("'{name}' must implement a non-inverting buffer"));
    }
    Ok(input)
}

/// A validated library reference buffer and its input capacitance in fF.
#[derive(Clone, Debug, PartialEq)]
pub struct ReferenceBuffer {
    pub cell_name: String,
    pub input_capacitance_ff: f64,
}

/// Resolves the reference buffer once, rejecting absent/invalid defaults.
pub fn resolve_reference_buffer(library: &Library) -> Result<ReferenceBuffer, String> {
    let name = library.reference_buffer_cell.as_deref().ok_or(
        "Liberty has no reference_buffer_cell; specify explicit drive/load constraints or select a reference buffer during import",
    )?;
    let pin = buffer_input_pin(library, name)?;
    let capacitance = pin
        .capacitance
        .ok_or_else(|| format!("reference buffer '{name}' has no input capacitance"))?;
    let units = library
        .units
        .as_ref()
        .ok_or("Liberty has no capacitance units")?;
    let result = capacitance * capacitance_unit_in_ff(&units.capacitance_unit)?;
    if !result.is_finite() || result <= 0.0 {
        return Err(format!(
            "reference buffer '{name}' needs positive finite input capacitance"
        ));
    }
    Ok(ReferenceBuffer {
        cell_name: name.to_owned(),
        input_capacitance_ff: result,
    })
}

/// Converts an explicitly scaled Liberty capacitance unit to femtofarads.
fn capacitance_unit_in_ff(unit: &str) -> Result<f64, String> {
    let normalized = unit.trim().to_ascii_lowercase();
    for (suffix, scale) in [
        ("af", 1e-3),
        ("ff", 1.0),
        ("pf", 1e3),
        ("nf", 1e6),
        ("uf", 1e9),
        ("mf", 1e12),
        ("f", 1e15),
    ] {
        if let Some(coefficient) = normalized.strip_suffix(suffix) {
            let value = coefficient
                .trim()
                .parse::<f64>()
                .map_err(|_| format!("invalid Liberty capacitance unit '{unit}'"))?
                * scale;
            if value.is_finite() && value > 0.0 {
                return Ok(value);
            }
            break;
        }
    }
    Err(format!("invalid Liberty capacitance unit '{unit}'"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liberty::model::{library_from_proto, library_to_proto};
    use crate::liberty::test_utils::make_test_library;
    use crate::liberty_proto::LibraryUnits;

    fn library() -> Library {
        let mut lib = make_test_library();
        lib.reference_buffer_cell = Some("BUF".into());
        lib.units = Some(LibraryUnits {
            capacitance_unit: "1pf".into(),
            ..Default::default()
        });
        lib.cells[1].pins[0].capacitance = Some(0.0005);
        lib
    }

    #[test]
    fn reference_buffer_survives_wire_roundtrip_and_converts_units() {
        let lib = library_from_proto(library_to_proto(library()).unwrap()).unwrap();
        assert_eq!(lib.reference_buffer_cell.as_deref(), Some("BUF"));
        assert_eq!(
            resolve_reference_buffer(&lib).unwrap(),
            ReferenceBuffer {
                cell_name: "BUF".into(),
                input_capacitance_ff: 0.5,
            }
        );
        for (unit, expected) in [
            ("1ff", 1.0),
            (" 0.001 pF ", 1.0),
            ("1e-3nf", 1000.0),
            ("1000af", 1.0),
        ] {
            assert_eq!(capacitance_unit_in_ff(unit).unwrap(), expected);
        }
        let mut wire = library_to_proto(lib).unwrap();
        wire.reference_buffer_cell = None;
        assert_eq!(
            library_from_proto(wire).unwrap().reference_buffer_cell,
            None
        );
    }

    #[test]
    fn invalid_or_filtered_reference_buffers_are_rejected() {
        for name in [None, Some(""), Some("ABSENT"), Some("INV"), Some("DFF")] {
            let mut lib = library();
            lib.reference_buffer_cell = name.map(str::to_owned);
            assert!(resolve_reference_buffer(&lib).is_err());
        }
        for cap in [
            None,
            Some(0.0),
            Some(-1.0),
            Some(f64::NAN),
            Some(f64::INFINITY),
        ] {
            let mut lib = library();
            lib.cells[1].pins[0].capacitance = cap;
            assert!(resolve_reference_buffer(&lib).is_err());
        }
        for unit in ["", "pf", "0pf", "-1pf", "NaNpf", "infpf", "1ps"] {
            assert!(capacitance_unit_in_ff(unit).is_err());
        }
        let mut lib = library();
        lib.cells[1].dont_use = Some(true);
        assert!(resolve_reference_buffer(&lib).is_err());
        lib.cells[1].dont_use = None;
        let extra = lib.cells[1].pins[0].clone();
        lib.cells[1].pins.push(extra);
        assert!(resolve_reference_buffer(&lib).is_err());
    }
}
