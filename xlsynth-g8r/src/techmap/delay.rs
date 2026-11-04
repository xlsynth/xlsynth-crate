// SPDX-License-Identifier: Apache-2.0

//! Experimental fixed-slew, fixed-load delay-first cover of a shared AIG.
//! Pin-specific worst-edge Liberty delays are frozen during selection. Actual
//! sink capacitance, propagated slew and arrival-edge correlation are omitted.
//! One plan per polarity and fixed shared boundaries make this a heuristic,
//! not a global optimum or a hard timing constraint.
use super::area::{AreaMappedNetlist, AreaMappingLibrary, map_cover_candidates};
use crate::aig::gate::GateFn;
use crate::liberty_model::Library;
use anyhow::{Result, ensure};

/// Reference electrical point in the library's native time/capacitance units.
#[derive(Clone, Copy, Debug)]
pub struct DelayMappingOptions {
    pub reference_transition: f64,
    pub reference_load: f64,
}

/// Binding index that retains distinct logical leaf polarities and fixed
/// delays.
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
    Ok(map_cover_candidates(aig, &library.inner, false)?.0)
}

/// Covers from the same delay cut search, with alternative choices of shared
/// boundary polarity for comparison under a load-aware timing model.
#[derive(Debug)]
pub struct DelayMappedCandidates {
    /// Implements the polarity requested by the selected consumers when the
    /// fixed-delay or area score improves.
    pub demanded: AreaMappedNetlist,
    /// Uses the original locally chosen shared phases. Present only when the
    /// demand pass changed at least one phase.
    pub local: Option<AreaMappedNetlist>,
}

/// Emits a second candidate only when its shared phases differ from the
/// demand-selected cover. Both candidates use the same plans and emission
/// rules.
pub fn map_gatefn_to_delay_candidates(
    aig: &GateFn,
    library: &DelayMappingLibrary,
) -> Result<DelayMappedCandidates> {
    let (demanded, local) = map_cover_candidates(aig, &library.inner, true)?;
    Ok(DelayMappedCandidates { demanded, local })
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use crate::liberty_model::{Cell, LibraryBuilder, Pin, PinDirection};
    use crate::liberty_proto::TimingTableKind;
    use crate::techmap::area::map_gatefn_to_area_netlist;

    /// Builds single-output cells with constant Liberty tables for each input
    /// pin.
    pub(in crate::techmap) fn timing_library(
        specs: &[(&str, &str, f64, &[(&str, f64)])],
    ) -> Result<Library> {
        let mut b = LibraryBuilder::new();
        for &(name, function, area, inputs) in specs {
            let mut pins = Vec::new();
            let mut arcs = Vec::new();
            for &(input, delay) in inputs {
                pins.push(Pin {
                    name: b.intern_string(input)?,
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
                    tables.push(b.add_timing_table_f64(
                        kind,
                        0,
                        vec![],
                        vec![],
                        vec![],
                        vec![delay],
                        vec![],
                        "",
                    )?);
                }
                arcs.push(b.add_timing_arc(input, "non_unate", "combinational", "", tables)?);
            }
            pins.push(Pin {
                name: b.intern_string("Y")?,
                direction: PinDirection::Output as i32,
                function: b.intern_string(function)?,
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
        Ok(b.finish())
    }

    /// Builds timing-distinct AND cells and a faster direct NAND.
    fn timing_library_at_pins(asymmetric: bool) -> Result<Library> {
        timing_library(&[
            ("INV", "!A", 0.5, &[("A", 5.0)]),
            ("A_CHEAP", "A*B", 1.0, &[("A", 10.0), ("B", 10.0)]),
            ("B_STRONG", "A*B", 2.0, &[("A", 1.0), ("B", 1.0)]),
            (
                "C_EQUAL_AREA_FAST",
                "A*B",
                2.0,
                &[("A", 0.5), ("B", if asymmetric { 2.0 } else { 0.5 })],
            ),
            ("NAND", "!(A*B)", 2.5, &[("A", 0.2), ("B", 0.2)]),
        ])
    }

    #[test]
    fn retains_stronger_and_equal_area_alternatives_and_direct_negative_phase() {
        let lib = timing_library_at_pins(false).unwrap();
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
        let mut lib = timing_library_at_pins(true).unwrap();
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
        let mut lib = timing_library_at_pins(false).unwrap();
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
