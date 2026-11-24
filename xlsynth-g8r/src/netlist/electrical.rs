// SPDX-License-Identifier: Apache-2.0

//! Liberty electrical properties shared by timing, sizing and netlist edits.
//! Capacitances, transition times and limits use the selected library's units.

use crate::liberty_model::{Library, Pin};

/// Values for rising and falling signal transitions, respectively.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RiseFall<T> {
    pub rise: T,
    pub fall: T,
}

impl<T: Copy> RiseFall<T> {
    /// Uses the same value for both transitions.
    pub fn both(value: T) -> Self {
        Self {
            rise: value,
            fall: value,
        }
    }
}

/// Nominal rise/fall input load. A maximum-capacitance rule is never a load.
/// Callers that require valid characterization must reject non-finite or
/// negative values rather than silently substitute a different load.
#[inline]
pub fn input_capacitance(pin: &Pin) -> RiseFall<f64> {
    characterized_capacitance(pin).unwrap_or_default()
}

#[inline]
fn characterized_capacitance(pin: &Pin) -> Option<RiseFall<f64>> {
    let nominal = pin
        .capacitance
        .or(pin.rise_capacitance)
        .or(pin.fall_capacitance)?;
    Some(RiseFall {
        rise: pin.rise_capacitance.unwrap_or(nominal),
        fall: pin.fall_capacitance.unwrap_or(nominal),
    })
}

/// Liberty weighted fanout contributed by an input pin.
#[inline]
pub fn input_fanout_load(pin: &Pin) -> f64 {
    pin.fanout_load.unwrap_or(1.0)
}

/// Excess over an optional electrical design rule; an absent limit is
/// unconstrained.
#[inline]
pub(crate) fn rule_excess(measured: f64, limit: Option<f64>) -> f64 {
    limit.map_or(0.0, |limit| (measured - limit).max(0.0))
}

/// Characterized input capacitance and Liberty weighted fanout contribution.
#[derive(Clone, Copy, Debug)]
pub struct InputLoad {
    pub capacitance: RiseFall<f64>,
    pub fanout: f64,
}
impl InputLoad {
    /// At least one capacitance must be characterized, including explicit
    /// zero. Missing fanout_load has the Liberty default weight of one.
    pub fn for_pin(pin: &Pin) -> Option<Self> {
        let capacitance = characterized_capacitance(pin)?;
        let fanout = input_fanout_load(pin);
        ([capacitance.rise, capacitance.fall]
            .into_iter()
            .all(|v| v.is_finite() && v >= 0.0)
            && fanout.is_finite()
            && fanout >= 0.0)
            .then_some(Self {
                capacitance,
                fanout,
            })
    }
}

/// Explicit Liberty design-rule limits on an input or output pin. Timing-table
/// characterization ranges are separate because extrapolation is not itself a
/// design-rule violation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PinLimits {
    pub capacitance: Option<f64>,
    pub transition: Option<f64>,
    pub fanout: Option<f64>,
}
impl PinLimits {
    #[inline]
    pub fn for_pin(pin: &Pin) -> Self {
        Self {
            capacitance: pin.max_capacitance,
            transition: pin.max_transition,
            fanout: pin.max_fanout,
        }
    }

    /// Whether this pin has any explicit electrical design-rule limit.
    #[inline]
    pub fn is_unconstrained(self) -> bool {
        self.capacitance.is_none() && self.transition.is_none() && self.fanout.is_none()
    }

    /// Componentwise rule excess when all three quantities have been measured.
    /// Keeping this path free of optional observations matters for incremental
    /// timing, which assesses every constrained pin when building its cache.
    #[inline]
    pub fn violations_known(
        self,
        capacitance: f64,
        transition: f64,
        fanout: f64,
    ) -> ElectricalViolations {
        ElectricalViolations {
            capacitance: rule_excess(capacitance, self.capacitance),
            transition: rule_excess(transition, self.transition),
            fanout: rule_excess(fanout, self.fanout),
        }
    }
}

/// Componentwise excess above explicit design rules; trial acceptance belongs
/// to the optimizer and may use absolute or relative comparisons.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ElectricalViolations {
    pub capacitance: f64,
    pub transition: f64,
    pub fanout: f64,
}

/// Conservative characterization bounds across an output pin's delay and
/// transition tables. Actual query diagnostics remain the authority on which
/// particular timing arc or table was extrapolated.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct OutputCharacterization {
    pub max_load: Option<f64>,
}
impl OutputCharacterization {
    pub fn for_pin(lib: &Library, pin: &Pin) -> Self {
        let mut capacitance: Option<f64> = None;
        for table in pin.timing_arcs.iter().flat_map(|arc| &arc.tables) {
            if !matches!(
                table.kind_str(),
                "cell_rise" | "cell_fall" | "rise_transition" | "fall_transition"
            ) {
                continue;
            }
            for (axis, (variable, knots)) in crate::liberty::lut::timing_axes::<3>(lib, table)
                .into_iter()
                .take(lib.timing_table_shape(table).dimensions.len())
                .enumerate()
            {
                if matches!(
                    variable,
                    "total_output_net_capacitance"
                        | "related_out_total_output_net_capacitance"
                        | "output_net_pin_cap"
                        | "related_out_output_net_pin_cap"
                ) || (variable.is_empty() && axis == 1)
                {
                    if let Some(&hi) = knots.last() {
                        capacitance = Some(capacitance.map_or(hi, |c| c.min(hi)));
                    }
                }
            }
        }
        Self {
            max_load: capacitance,
        }
    }
}

/// Load carried by cell input pins and module output ports. Wire
/// capacitance is supplied separately because an edit can change the fanout.
#[derive(Clone, Copy, Debug, Default)]
pub struct Load {
    pub capacitance: RiseFall<f64>,
    pub weighted_fanout: f64,
    pub pins: usize,
    pub output_ports: usize,
}
impl Load {
    pub fn with_outputs(output_ports: usize, output_load: f64) -> Self {
        Self {
            capacitance: RiseFall::both(output_load * output_ports as f64),
            output_ports,
            ..Self::default()
        }
    }
    pub fn add_pin(&mut self, pin: InputLoad) {
        self.capacitance.rise += pin.capacitance.rise;
        self.capacitance.fall += pin.capacitance.fall;
        self.weighted_fanout += pin.fanout;
        self.pins += 1;
    }
}

/// Electrical measurements for a signal. Transition is populated by timing
/// analysis, not inferred from the number of sinks.
#[derive(Clone, Copy, Debug)]
pub struct SignalElectrical {
    pub load: Load,
    pub transition: Option<f64>,
    pub limits: PinLimits,
    pub characterization: OutputCharacterization,
}
impl SignalElectrical {
    /// Measure exact indexed sink bindings. An unknown pin direction or an
    /// uncharacterized load makes the result unavailable, rather than zero.
    pub fn analyze(
        pins: &crate::netlist::connectivity::exact::SignalPins,
        output_ports: usize,
        output_load: f64,
        limits: PinLimits,
        characterization: OutputCharacterization,
        transition: Option<f64>,
        mut load_for: impl FnMut(crate::netlist::connectivity::PinConnection) -> Option<InputLoad>,
    ) -> Option<Self> {
        if !pins.unknown.is_empty() || !output_load.is_finite() || output_load < 0.0 {
            return None;
        }
        let mut load = Load::with_outputs(output_ports, output_load);
        for &pin in &pins.sinks {
            load.add_pin(load_for(pin)?);
        }
        Some(Self {
            load,
            transition,
            limits,
            characterization,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A partial improvement must remain visible even if the rule is still
    /// violated.
    #[test]
    fn partial_violation_improvement_remains_measurable() {
        let limits = PinLimits {
            transition: Some(100.0),
            ..PinLimits::default()
        };
        assert_eq!(limits.violations_known(0.0, 150.0, 0.0).transition, 50.0);
        assert_eq!(limits.violations_known(0.0, 110.0, 0.0).transition, 10.0);
    }

    /// A zero electrical limit is a real rule, not an absent limit.
    #[test]
    fn zero_limit_is_constrained() {
        let limit = PinLimits {
            fanout: Some(0.0),
            ..PinLimits::default()
        };
        assert_eq!(limit.violations_known(0.0, 0.0, 2.0).fanout, 2.0);
    }

    /// Module output ports add capacitance but not pin fanout. Rise and fall
    /// loads remain separate, and a table boundary is not an electrical rule.
    #[test]
    fn asymmetric_loads_ports_and_characterization_are_independent() {
        use crate::netlist::connectivity::{PinConnection, exact::SignalPins};
        use crate::netlist::parse::InstIndex;
        let id = |i| PinConnection {
            instance: InstIndex(i),
            connection: 0,
        };
        let pins = SignalPins {
            sinks: vec![id(0), id(1)],
            ..Default::default()
        };
        let limits = PinLimits {
            capacitance: Some(2.0),
            ..Default::default()
        };
        let measured = SignalElectrical::analyze(
            &pins,
            2,
            0.25,
            limits,
            OutputCharacterization {
                max_load: Some(0.75),
            },
            None,
            |pin| {
                Some(if pin == id(0) {
                    InputLoad {
                        capacitance: RiseFall {
                            rise: 0.25,
                            fall: 0.5,
                        },
                        fanout: 2.0,
                    }
                } else {
                    InputLoad {
                        capacitance: RiseFall {
                            rise: 0.5,
                            fall: 0.125,
                        },
                        fanout: 0.5,
                    }
                })
            },
        )
        .unwrap();
        assert_eq!(
            measured.load.capacitance,
            RiseFall {
                rise: 1.25,
                fall: 1.125
            }
        );
        assert_eq!(measured.load.weighted_fanout, 2.5);
        assert_eq!((measured.load.pins, measured.load.output_ports), (2, 2));
        assert!(measured.load.capacitance.rise > measured.characterization.max_load.unwrap());
        assert_eq!(limits.violations_known(1.25, 0.0, 0.0).capacitance, 0.0);

        let unknown = SignalPins {
            unknown: vec![id(2)],
            ..pins
        };
        assert!(
            SignalElectrical::analyze(
                &unknown,
                0,
                0.0,
                limits,
                Default::default(),
                None,
                |_| unreachable!()
            )
            .is_none()
        );
        assert!(
            SignalElectrical::analyze(
                &SignalPins {
                    sinks: vec![id(0)],
                    ..Default::default()
                },
                0,
                0.0,
                limits,
                Default::default(),
                None,
                |_| None
            )
            .is_none()
        );
    }

    /// A missing nominal edge uses the other characterized edge, never the
    /// max-capacitance rule; invalid capacitance or weight cannot price a pin.
    #[test]
    fn input_load_fallback_and_invalid_characterization() {
        let mut pin = Pin {
            max_capacitance: Some(100.0),
            ..Default::default()
        };
        assert!(InputLoad::for_pin(&pin).is_none());
        pin.rise_capacitance = Some(0.0);
        assert_eq!(
            InputLoad::for_pin(&pin).unwrap().capacitance,
            RiseFall::default()
        );
        pin.rise_capacitance = Some(0.25);
        let load = InputLoad::for_pin(&pin).unwrap();
        assert_eq!(load.capacitance, RiseFall::both(0.25));
        assert_eq!(load.fanout, 1.0);
        pin.fall_capacitance = Some(0.5);
        assert_eq!(
            InputLoad::for_pin(&pin).unwrap().capacitance,
            RiseFall {
                rise: 0.25,
                fall: 0.5
            }
        );
        for invalid in [-1.0, f64::NAN, f64::INFINITY] {
            pin.rise_capacitance = Some(invalid);
            assert!(InputLoad::for_pin(&pin).is_none());
        }
        pin.rise_capacitance = Some(0.25);
        pin.fanout_load = Some(-1.0);
        assert!(InputLoad::for_pin(&pin).is_none());
    }
}
