// SPDX-License-Identifier: Apache-2.0

//! Experimental fixed-slew, fixed-load delay-first cover of a shared AIG.
//! Pin-specific worst-edge Liberty delays are frozen during selection. Actual
//! sink capacitance, propagated slew and arrival-edge correlation are omitted.
//! One plan per polarity and fixed shared boundaries make this a heuristic,
//! not a global optimum or a hard timing constraint.
use super::area::{AreaMappedNetlist, AreaMappingLibrary, map_gatefn_to_area_netlist};
use crate::aig::gate::GateFn;
use crate::liberty_model::Library;
use anyhow::{Result, ensure};

/// Reference electrical point in the library's native time/capacitance units.
#[derive(Clone, Copy, Debug)]
pub struct DelayMappingOptions {
    pub reference_transition: f64,
    pub reference_load: f64,
}

/// Timing-preserving binding index; unlike area mode it retains cell/phase/pin
/// alternatives.
#[derive(Debug)]
pub struct DelayMappingLibrary {
    inner: AreaMappingLibrary,
}
impl DelayMappingLibrary {
    /// Prepare all eligible cells at one explicit reference electrical point.
    pub fn new(library: &Library, options: DelayMappingOptions) -> Result<Self> {
        ensure!(
            options.reference_transition.is_finite() && options.reference_transition >= 0.0,
            "reference transition must be finite and nonnegative"
        );
        ensure!(
            options.reference_load.is_finite() && options.reference_load >= 0.0,
            "reference load must be finite and nonnegative"
        );
        Ok(Self {
            inner: AreaMappingLibrary::new_internal(library, Some(options))?,
        })
    }
}

/// Minimize estimated worst arrival, then area, retaining stable deterministic
/// ties.
pub fn map_gatefn_to_delay_netlist(
    aig: &GateFn,
    library: &DelayMappingLibrary,
) -> Result<AreaMappedNetlist> {
    map_gatefn_to_area_netlist(aig, &library.inner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use crate::liberty_model::{Cell, LibraryBuilder, Pin, PinDirection};
    use crate::liberty_proto::TimingTableKind;

    /// Build timing-distinct equivalent cells, including a faster negative
    /// phase.
    fn timing_library() -> Library {
        timing_library_at_pins(false)
    }

    fn timing_library_at_pins(asymmetric: bool) -> Library {
        let mut b = LibraryBuilder::new();
        for (name, inputs, function, area, delay) in [
            ("INV", vec!["A"], "!A", 0.5, 5.0),
            ("A_CHEAP", vec!["A", "B"], "A*B", 1.0, 10.0),
            ("B_STRONG", vec!["A", "B"], "A*B", 2.0, 1.0),
            ("C_EQUAL_AREA_FAST", vec!["A", "B"], "A*B", 2.0, 0.5),
            ("NAND", vec!["A", "B"], "!(A*B)", 2.5, 0.2),
        ] {
            let mut pins = Vec::new();
            let mut arcs = Vec::new();
            for input in inputs {
                pins.push(Pin {
                    name: b.intern_string(input).unwrap(),
                    direction: PinDirection::Input as i32,
                    ..Default::default()
                });
                let mut tables = Vec::new();
                for kind in [
                    TimingTableKind::CellRise,
                    TimingTableKind::CellFall,
                    TimingTableKind::RiseTransition,
                    TimingTableKind::FallTransition,
                ] {
                    tables.push(
                        b.add_timing_table_f64(
                            kind,
                            0,
                            vec![],
                            vec![],
                            vec![],
                            vec![
                                if asymmetric && name == "C_EQUAL_AREA_FAST" && input == "B" {
                                    2.0
                                } else {
                                    delay
                                },
                            ],
                            vec![],
                            "",
                        )
                        .unwrap(),
                    );
                }
                arcs.push(
                    b.add_timing_arc(input, "non_unate", "combinational", "", tables)
                        .unwrap(),
                );
            }
            pins.push(Pin {
                name: b.intern_string("Y").unwrap(),
                direction: PinDirection::Output as i32,
                function: b.intern_string(function).unwrap(),
                timing_arcs: arcs,
                ..Default::default()
            });
            b.cells.push(Cell {
                name: name.into(),
                pins,
                area,
                ..Default::default()
            });
        }
        b.finish()
    }

    #[test]
    fn retains_stronger_and_equal_area_alternatives_and_direct_negative_phase() {
        let lib = timing_library();
        let delay = DelayMappingLibrary::new(
            &lib,
            DelayMappingOptions {
                reference_transition: 10.0,
                reference_load: 1.0,
            },
        )
        .unwrap();
        let original_cells = lib.cells.clone();
        let before_area = format!("{delay:?}");
        let area = AreaMappingLibrary::new(&lib).unwrap();
        assert_eq!(lib.cells, original_cells);
        let after_area = DelayMappingLibrary::new(
            &lib,
            DelayMappingOptions {
                reference_transition: 10.0,
                reference_load: 1.0,
            },
        )
        .unwrap();
        assert_eq!(format!("{after_area:?}"), before_area);
        for negative in [false, true] {
            let mut b = GateBuilder::new("test".into(), GateBuilderOptions::no_opt());
            let x = *b.add_input("x".into(), 1).get_lsb(0);
            let y = *b.add_input("y".into(), 1).get_lsb(0);
            let out = b.add_and_binary(x, y);
            b.add_output(
                "out".into(),
                if negative { out.negate() } else { out }.into(),
            );
            let aig = b.build();
            let mapped = map_gatefn_to_delay_netlist(&aig, &delay).unwrap();
            assert_eq!(mapped.stats.cell_count, 1);
            assert_eq!(
                mapped.stats.cells_by_type.get(if negative {
                    "NAND"
                } else {
                    "C_EQUAL_AREA_FAST"
                }),
                Some(&1)
            );
            if !negative {
                let baseline = map_gatefn_to_area_netlist(&aig, &area).unwrap();
                assert_eq!(baseline.stats.cells_by_type.get("A_CHEAP"), Some(&1));
            }
        }
    }

    #[test]
    fn binds_later_arrival_to_faster_symmetric_pin() {
        use crate::netlist::parse::NetRef;
        let mut lib = timing_library_at_pins(true);
        lib.cells
            .retain(|c| c.name == "INV" || c.name == "C_EQUAL_AREA_FAST");
        let delay = DelayMappingLibrary::new(
            &lib,
            DelayMappingOptions {
                reference_transition: 10.0,
                reference_load: 1.0,
            },
        )
        .unwrap();
        let mut b = GateBuilder::new("test".into(), GateBuilderOptions::no_opt());
        let x = *b.add_input("x".into(), 1).get_lsb(0);
        let y = *b.add_input("y".into(), 1).get_lsb(0);
        let z = *b.add_input("z".into(), 1).get_lsb(0);
        let child = b.add_and_binary(x, y);
        let out = b.add_and_binary(child, z);
        b.add_output("out".into(), out.into());
        let mapped = map_gatefn_to_delay_netlist(&b.build(), &delay).unwrap();
        assert_eq!(mapped.stats.cell_count, 2);
        let m = &mapped.netlist;
        let z_pin = m
            .module
            .instances
            .iter()
            .flat_map(|i| &i.connections)
            .find_map(|(pin, net)| match net {
                NetRef::Simple(idx) if m.interner.resolve(m.nets[idx.0].name) == Some("z") => {
                    m.interner.resolve(*pin)
                }
                _ => None,
            });
        // The early primary input takes the slower B pin; child takes fast A.
        assert_eq!(z_pin, Some("B"));
    }

    #[test]
    fn rejects_invalid_electrical_point_and_missing_timing() {
        let mut lib = timing_library();
        assert!(
            DelayMappingLibrary::new(
                &lib,
                DelayMappingOptions {
                    reference_transition: f64::NAN,
                    reference_load: 1.0
                }
            )
            .is_err()
        );
        for cell in &mut lib.cells {
            for pin in &mut cell.pins {
                pin.timing_arcs.clear();
            }
        }
        assert!(
            DelayMappingLibrary::new(
                &lib,
                DelayMappingOptions {
                    reference_transition: 10.0,
                    reference_load: 1.0
                }
            )
            .is_err()
        );
    }
}
