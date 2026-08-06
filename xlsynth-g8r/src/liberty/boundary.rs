// SPDX-License-Identifier: Apache-2.0

//! Validated Liberty-backed defaults for realistic module-boundary timing.

use crate::liberty::model::{Cell, Library, Pin};
use crate::liberty_proto::{PinDirection, TimingTableKind};
use anyhow::{Result, anyhow, bail};

/// Independent rise/fall capacitance presented at one module output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoundaryOutputLoad {
    pub rise: f64,
    pub fall: f64,
}

impl BoundaryOutputLoad {
    /// Returns the conservative scalar capacitance across both edges.
    pub fn maximum(self) -> f64 {
        self.rise.max(self.fall)
    }
}

/// One characterized unary cell used as a virtual primary-input driver.
#[derive(Clone, Copy, Debug)]
pub struct RepresentativeDriver<'a> {
    pub cell: &'a Cell,
    pub input_pin: &'a Pin,
    pub output_pin: &'a Pin,
}

/// Validates optional boundary defaults against retained Liberty cells.
pub fn validate_boundary_timing_defaults(library: &Library) -> Result<()> {
    let Some(defaults) = &library.boundary_timing_defaults else {
        return Ok(());
    };
    if defaults.representative_driver_cell.trim().is_empty() {
        bail!("representative input-driver cell must not be empty");
    }
    if defaults.representative_load_cell.trim().is_empty() {
        bail!("representative output-load cell must not be empty");
    }
    if defaults.representative_load_count == 0 {
        bail!("representative output-load count must be greater than zero");
    }

    let driver = resolve_unary_cell(
        library,
        defaults.representative_driver_cell.as_str(),
        "input-driver",
        true,
    )?;
    if library
        .cells
        .iter()
        .flat_map(|cell| &cell.pins)
        .any(|pin| !pin.timing_arcs.is_empty())
    {
        validate_driver_timing(library, driver)?;
    }
    representative_output_load(library)?;
    Ok(())
}

/// Resolves the characterized unary driver described by library defaults.
pub fn representative_driver(library: &Library) -> Result<Option<RepresentativeDriver<'_>>> {
    let Some(defaults) = &library.boundary_timing_defaults else {
        return Ok(None);
    };
    let driver = resolve_unary_cell(
        library,
        defaults.representative_driver_cell.as_str(),
        "input-driver",
        true,
    )?;
    validate_driver_timing(library, driver)?;
    Ok(Some(driver))
}

/// Computes edge-specific output loading from representative receiver pins.
pub fn representative_output_load(library: &Library) -> Result<Option<BoundaryOutputLoad>> {
    let Some(defaults) = &library.boundary_timing_defaults else {
        return Ok(None);
    };
    if defaults.representative_load_count == 0 {
        bail!("representative output-load count must be greater than zero");
    }
    let load = resolve_unary_cell(
        library,
        defaults.representative_load_cell.as_str(),
        "output-load",
        false,
    )?;
    let pin = load.input_pin;
    let rise = pin
        .rise_capacitance
        .or(pin.capacitance)
        .or(pin.fall_capacitance)
        .ok_or_else(|| {
            anyhow!(
                "representative output-load cell '{}' input '{}' has no capacitance",
                load.cell.name,
                library.resolve_string(&pin.name)
            )
        })?;
    let fall = pin
        .fall_capacitance
        .or(pin.capacitance)
        .or(pin.rise_capacitance)
        .ok_or_else(|| {
            anyhow!(
                "representative output-load cell '{}' input '{}' has no capacitance",
                load.cell.name,
                library.resolve_string(&pin.name)
            )
        })?;
    if !rise.is_finite() || rise <= 0.0 || !fall.is_finite() || fall <= 0.0 {
        bail!(
            "representative output-load cell '{}' has invalid rise/fall capacitance {rise}/{fall}",
            load.cell.name
        );
    }
    let count = f64::from(defaults.representative_load_count);
    Ok(Some(BoundaryOutputLoad {
        rise: rise * count,
        fall: fall * count,
    }))
}

/// Resolves one usable unary Liberty cell and its unique input/output pins.
fn resolve_unary_cell<'a>(
    library: &'a Library,
    name: &str,
    role: &str,
    require_output: bool,
) -> Result<RepresentativeDriver<'a>> {
    if name.trim().is_empty() {
        bail!("representative {role} cell must not be empty");
    }
    let mut matches = library.cells.iter().filter(|cell| cell.name == name);
    let cell = matches
        .next()
        .ok_or_else(|| anyhow!("representative {role} cell '{name}' is absent from the library"))?;
    if matches.next().is_some() {
        bail!("representative {role} cell '{name}' is ambiguous");
    }
    if cell.dont_use == Some(true) {
        bail!("representative {role} cell '{name}' is marked dont_use");
    }
    if !cell.sequential.is_empty() || cell.clock_gate.is_some() {
        bail!("representative {role} cell '{name}' must be combinational");
    }

    let mut inputs = cell
        .pins
        .iter()
        .filter(|pin| pin.direction == PinDirection::Input as i32);
    let input_pin = inputs
        .next()
        .ok_or_else(|| anyhow!("representative {role} cell '{name}' has no input pin"))?;
    if inputs.next().is_some() || input_pin.is_clocking_pin {
        bail!("representative {role} cell '{name}' must have exactly one nonclock input");
    }

    let mut outputs = cell
        .pins
        .iter()
        .filter(|pin| pin.direction == PinDirection::Output as i32);
    let output_pin = outputs
        .next()
        .ok_or_else(|| anyhow!("representative {role} cell '{name}' has no output pin"))?;
    if require_output && outputs.next().is_some() {
        bail!("representative {role} cell '{name}' must have exactly one output");
    }

    Ok(RepresentativeDriver {
        cell,
        input_pin,
        output_pin,
    })
}

/// Rejects virtual drivers without complete delay and transition timing.
fn validate_driver_timing(library: &Library, driver: RepresentativeDriver<'_>) -> Result<()> {
    let input_name = library.resolve_string(&driver.input_pin.name);
    let usable = driver.output_pin.timing_arcs.iter().any(|arc| {
        library
            .resolve_string(&arc.related_pin)
            .split_whitespace()
            .any(|related| related == input_name)
            && [
                TimingTableKind::CellRise,
                TimingTableKind::CellFall,
                TimingTableKind::RiseTransition,
                TimingTableKind::FallTransition,
            ]
            .into_iter()
            .all(|kind| arc.tables.iter().any(|table| table.kind == kind))
    });
    if !usable {
        bail!(
            "representative input-driver cell '{}' has no complete timing arc from '{}' to '{}'",
            driver.cell.name,
            input_name,
            library.resolve_string(&driver.output_pin.name)
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liberty::test_utils::make_test_library;
    use crate::liberty_proto::BoundaryTimingDefaults;

    #[test]
    fn derives_edge_specific_fanout_load_from_receiver_cell() {
        let mut library = make_test_library();
        let buffer = library
            .cells
            .iter_mut()
            .find(|cell| cell.name == "BUF")
            .unwrap();
        let input = buffer
            .pins
            .iter_mut()
            .find(|pin| pin.direction == PinDirection::Input as i32)
            .unwrap();
        input.capacitance = Some(0.4);
        input.rise_capacitance = Some(0.3);
        input.fall_capacitance = Some(0.5);
        library.boundary_timing_defaults = Some(BoundaryTimingDefaults {
            representative_driver_cell: "BUF".to_string(),
            representative_load_cell: "BUF".to_string(),
            representative_load_count: 2,
        });

        assert_eq!(
            representative_output_load(&library).unwrap(),
            Some(BoundaryOutputLoad {
                rise: 0.6,
                fall: 1.0,
            })
        );
        validate_boundary_timing_defaults(&library).unwrap();
    }

    #[test]
    fn rejects_missing_receiver_capacitance() {
        let mut library = make_test_library();
        library.boundary_timing_defaults = Some(BoundaryTimingDefaults {
            representative_driver_cell: "BUF".to_string(),
            representative_load_cell: "BUF".to_string(),
            representative_load_count: 2,
        });

        let error = validate_boundary_timing_defaults(&library).unwrap_err();
        assert!(error.to_string().contains("has no capacitance"));
    }

    #[test]
    fn rejects_zero_representative_load_count() {
        let mut library = make_test_library();
        library.boundary_timing_defaults = Some(BoundaryTimingDefaults {
            representative_driver_cell: "BUF".to_string(),
            representative_load_cell: "BUF".to_string(),
            representative_load_count: 0,
        });

        let error = validate_boundary_timing_defaults(&library).unwrap_err();
        assert!(error.to_string().contains("must be greater than zero"));
    }

    #[test]
    fn absent_boundary_defaults_preserve_legacy_behavior() {
        let library = make_test_library();

        assert!(representative_driver(&library).unwrap().is_none());
        assert!(representative_output_load(&library).unwrap().is_none());
        validate_boundary_timing_defaults(&library).unwrap();
    }
}
