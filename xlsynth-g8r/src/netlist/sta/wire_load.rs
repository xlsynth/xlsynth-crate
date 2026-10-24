// SPDX-License-Identifier: Apache-2.0

//! Opt-in affine, lumped wire capacitance; no interconnect propagation delay.

use super::EdgeLoadCapacitance;
use crate::liberty::io_defaults::capacitance_unit_in_ff;
use crate::liberty_model::Library;
use anyhow::{Result, anyhow};

/// Fixed overhead plus capacitance per sink pin, in Liberty capacitance units.
/// Output ports do not count as sink pins. An output-only net receives the
/// constant term; a dangling net receives neither term. Primary-input drivers
/// remain ideal sources and are not loaded by this policy.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WireLoadModel {
    pub constant_capacitance: f64,
    pub capacitance_per_sink: f64,
}

impl WireLoadModel {
    /// Converts explicit fF parameters to the loaded library's capacitance
    /// units. Disabled models do not require unit metadata, preserving
    /// zero-wire use.
    pub fn from_femtofarads(constant: f64, per_sink: f64, library: &Library) -> Result<Self> {
        let mut model = Self {
            constant_capacitance: constant,
            capacitance_per_sink: per_sink,
        };
        model.validate()?;
        if model == Self::default() {
            return Ok(model);
        }
        let units = library
            .units
            .as_ref()
            .ok_or_else(|| anyhow!("wire load requires Liberty capacitance units"))?;
        let scale = capacitance_unit_in_ff(&units.capacitance_unit).map_err(anyhow::Error::msg)?;
        model.constant_capacitance /= scale;
        model.capacitance_per_sink /= scale;
        model.validate()?;
        Ok(model)
    }

    /// Rejects negative and non-finite coefficients even on empty designs.
    pub fn validate(self) -> Result<()> {
        for (name, value) in [
            ("wire constant capacitance", self.constant_capacitance),
            ("wire capacitance per sink", self.capacitance_per_sink),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(anyhow!(
                    "{name} must be finite and non-negative; got {value}"
                ));
            }
        }
        Ok(())
    }

    /// Adds wire load once after real pin and external port loads.
    pub(crate) fn add_to(
        self,
        cap: &mut EdgeLoadCapacitance,
        sink_pins: usize,
        output: bool,
    ) -> Result<()> {
        if self == Self::default() || (sink_pins == 0 && !output) {
            return Ok(());
        }
        let wire = self.constant_capacitance + self.capacitance_per_sink * sink_pins as f64;
        let rise = cap.rise + wire;
        let fall = cap.fall + wire;
        if !wire.is_finite() || !rise.is_finite() || !fall.is_finite() {
            return Err(anyhow!(
                "wire load or total net capacitance overflow at {sink_pins} sink pins"
            ));
        }
        cap.rise = rise;
        cap.fall = fall;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liberty::test_utils::make_test_library;
    use crate::liberty_proto::LibraryUnits;

    #[test]
    fn affine_terms_add_once_and_preserve_existing_edge_loads() -> Result<()> {
        for (constant, per_sink, expected) in [
            (0.0, 0.0, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.25, 0.75),
            (0.5, 0.25, 1.25),
        ] {
            let model = WireLoadModel {
                constant_capacitance: constant,
                capacitance_per_sink: per_sink,
            };
            let mut cap = EdgeLoadCapacitance {
                rise: 2.0,
                fall: 3.0,
            };
            model.add_to(&mut cap, 3, true)?;
            assert_eq!(cap.rise, 2.0 + expected);
            assert_eq!(cap.fall, 3.0 + expected);
            let mut unused = EdgeLoadCapacitance::default();
            model.add_to(&mut unused, 0, false)?;
            assert_eq!(unused.rise, 0.0);
            model.add_to(&mut unused, 0, true)?;
            assert_eq!(unused.rise, constant);
        }
        Ok(())
    }

    #[test]
    fn converts_bare_and_scaled_units_without_changing_labels() -> Result<()> {
        let mut lib = make_test_library();
        for (unit, constant, per_sink) in [
            ("pf", 0.5, 0.25),
            ("pF", 0.5, 0.25),
            ("1pf", 0.5, 0.25),
            ("1pF", 0.5, 0.25),
            ("0.5pF", 1.0, 0.5),
            ("2pF", 0.25, 0.125),
            ("fF", 500.0, 250.0),
        ] {
            lib.units = Some(LibraryUnits {
                capacitance_unit: unit.into(),
                ..Default::default()
            });
            assert_eq!(
                WireLoadModel::from_femtofarads(500.0, 250.0, &lib)?,
                WireLoadModel {
                    constant_capacitance: constant,
                    capacitance_per_sink: per_sink,
                }
            );
            assert_eq!(lib.units.as_ref().unwrap().capacitance_unit, unit);
            assert_eq!(
                WireLoadModel::from_femtofarads(0.0, 0.0, &lib)?,
                WireLoadModel::default()
            );
        }
        Ok(())
    }

    #[test]
    fn validates_units_coefficients_and_overflow() -> Result<()> {
        let mut lib = make_test_library();
        lib.units = None;
        assert_eq!(
            WireLoadModel::from_femtofarads(0.0, 0.0, &lib)?,
            WireLoadModel::default()
        );
        assert!(WireLoadModel::from_femtofarads(1.0, 0.0, &lib).is_err());
        lib.units = Some(LibraryUnits {
            capacitance_unit: "1pf".into(),
            ..Default::default()
        });
        assert_eq!(
            WireLoadModel::from_femtofarads(500.0, 250.0, &lib)?,
            WireLoadModel {
                constant_capacitance: 0.5,
                capacitance_per_sink: 0.25
            }
        );
        for invalid in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(WireLoadModel::from_femtofarads(invalid, 0.0, &lib).is_err());
            assert!(WireLoadModel::from_femtofarads(0.0, invalid, &lib).is_err());
        }
        let model = WireLoadModel {
            constant_capacitance: f64::MAX,
            capacitance_per_sink: f64::MAX,
        };
        assert!(
            model
                .add_to(&mut EdgeLoadCapacitance::default(), 2, false)
                .is_err()
        );
        let model = WireLoadModel {
            constant_capacitance: f64::MAX,
            capacitance_per_sink: 0.0,
        };
        assert!(
            model
                .add_to(
                    &mut EdgeLoadCapacitance {
                        rise: f64::MAX,
                        fall: 0.0
                    },
                    1,
                    false
                )
                .is_err()
        );
        lib.units.as_mut().unwrap().capacitance_unit = "1e-300ff".into();
        assert!(WireLoadModel::from_femtofarads(f64::MAX, 0.0, &lib).is_err());
        Ok(())
    }
}
