// SPDX-License-Identifier: Apache-2.0

//! Indexed view over a Liberty `Library`.
//!
//! This module provides a small wrapper around the prost-generated
//! `crate::liberty_proto::Library` that adds lazily constructed indices for
//! fast lookup by cell name and for grouping pins by direction. The underlying
//! `Library` remains the single source of truth; indices are derived from it
//! on demand.

use crate::liberty_proto::{Cell, Library as ProtoLibrary, Pin, PinDirection};
use std::cell::RefCell;
use std::collections::HashMap;

/// Grouped pin indices for a single `Cell` in `Library.cells`.
///
/// Each `usize` stored here is an index into the `pins` vector of the owning
/// `Cell` (i.e. `Library.cells[cell_idx].pins[pin_idx]`).
struct CellPinsByDir {
    /// All `pins[pin_idx]` entries whose `direction` is `PinDirection::Input`.
    input_pins: Vec<usize>,
    /// All `pins[pin_idx]` entries whose `direction` is `PinDirection::Output`.
    output_pins: Vec<usize>,
}

/// Owning wrapper around a `Library` with lazily built indices for fast lookup.
pub struct IndexedLibrary {
    proto: ProtoLibrary,
    /// Map from cell name to index in `proto.cells`.
    cell_by_name: RefCell<Option<HashMap<String, usize>>>,
    /// Map from cell index to directional pin groupings.
    pins_by_dir: RefCell<Option<HashMap<usize, CellPinsByDir>>>,
}

impl IndexedLibrary {
    /// Constructs a new indexed view over the given `Library`.
    pub fn new<T: Into<ProtoLibrary>>(proto: T) -> Self {
        let proto = proto.into();
        IndexedLibrary {
            proto,
            cell_by_name: RefCell::new(None),
            pins_by_dir: RefCell::new(None),
        }
    }

    /// Returns a shared reference to the underlying `Library`.
    pub fn library(&self) -> &ProtoLibrary {
        &self.proto
    }

    /// Returns the index of the cell with the given name in `proto.cells`.
    fn get_cell_index(&self, name: &str) -> Option<usize> {
        let idx_opt = {
            let mut opt_map = self.cell_by_name.borrow_mut();
            if opt_map.is_none() {
                let mut map: HashMap<String, usize> = HashMap::new();
                for (i, cell) in self.proto.cells.iter().enumerate() {
                    map.insert(cell.name.clone(), i);
                }
                *opt_map = Some(map);
            }
            opt_map.as_ref().and_then(|m| m.get(name).copied())
        };
        idx_opt
    }

    /// Looks up a cell by name.
    pub fn get_cell(&self, name: &str) -> Option<&Cell> {
        let idx = self.get_cell_index(name)?;
        Some(&self.proto.cells[idx])
    }

    /// Returns the pins for the given cell index grouped by direction.
    fn pins_by_dir_for_index(&self, cell_idx: usize) -> Option<CellPinsByDir> {
        {
            let pins_map_ref = self.pins_by_dir.borrow();
            if let Some(map) = pins_map_ref.as_ref() {
                if let Some(entry) = map.get(&cell_idx) {
                    return Some(CellPinsByDir {
                        input_pins: entry.input_pins.clone(),
                        output_pins: entry.output_pins.clone(),
                    });
                }
            }
        }

        if cell_idx >= self.proto.cells.len() {
            return None;
        }

        let cell = &self.proto.cells[cell_idx];
        let mut input_pins: Vec<usize> = Vec::new();
        let mut output_pins: Vec<usize> = Vec::new();
        for (i, pin) in cell.pins.iter().enumerate() {
            let dir_val = pin.direction;
            if dir_val == PinDirection::Input as i32 {
                input_pins.push(i);
            } else if dir_val == PinDirection::Output as i32 {
                output_pins.push(i);
            }
        }

        let grouped = CellPinsByDir {
            input_pins,
            output_pins,
        };

        let mut pins_map_mut = self.pins_by_dir.borrow_mut();
        let map = pins_map_mut.get_or_insert_with(HashMap::new);
        map.insert(
            cell_idx,
            CellPinsByDir {
                input_pins: grouped.input_pins.clone(),
                output_pins: grouped.output_pins.clone(),
            },
        );

        Some(grouped)
    }

    /// Returns the list of pins on `cell_name` that have the given direction.
    ///
    /// The returned slice is backed by a small `Vec` allocated per call.
    pub fn pins_for_dir(&self, cell_name: &str, dir: PinDirection) -> Option<Vec<&Pin>> {
        let cell_idx = self.get_cell_index(cell_name)?;
        let grouped = self.pins_by_dir_for_index(cell_idx)?;
        let cell = &self.proto.cells[cell_idx];

        let indices = match dir {
            PinDirection::Input => &grouped.input_pins,
            PinDirection::Output => &grouped.output_pins,
            PinDirection::Invalid => return None,
        };

        let mut result: Vec<&Pin> = Vec::with_capacity(indices.len());
        for idx in indices {
            if let Some(pin) = cell.pins.get(*idx) {
                result.push(pin);
            }
        }
        Some(result)
    }

    /// Looks up a pin by cell name and pin name.
    pub fn pin_by_name(&self, cell_name: &str, pin_name: &str) -> Option<&Pin> {
        let cell_idx = self.get_cell_index(cell_name)?;
        let cell = &self.proto.cells[cell_idx];
        for pin in &cell.pins {
            if pin.name == pin_name {
                return Some(pin);
            }
        }
        None
    }

    /// Looks up a timing arc for the given output pin and related input pin.
    pub fn timing_arc_for_pin(
        &self,
        cell_name: &str,
        output_pin_name: &str,
        related_pin_name: &str,
    ) -> Option<&crate::liberty_proto::TimingArc> {
        self.timing_arc_for_pin_filtered(
            cell_name,
            output_pin_name,
            related_pin_name,
            /* timing_type= */ None,
            /* timing_sense= */ None,
            /* when= */ None,
        )
    }

    /// Looks up a single timing arc for the given output pin with optional
    /// selectors in addition to `related_pin`.
    ///
    /// Returns `None` when no arc matches, or when multiple arcs match the
    /// selectors (to avoid silently choosing the wrong arc).
    pub fn timing_arc_for_pin_filtered(
        &self,
        cell_name: &str,
        output_pin_name: &str,
        related_pin_name: &str,
        timing_type: Option<&str>,
        timing_sense: Option<&str>,
        when: Option<&str>,
    ) -> Option<&crate::liberty_proto::TimingArc> {
        let pin = self.pin_by_name(cell_name, output_pin_name)?;
        let mut matches = pin.timing_arcs.iter().filter(|arc| {
            if arc.related_pin != related_pin_name {
                return false;
            }
            if let Some(v) = timing_type {
                if arc.timing_type != v {
                    return false;
                }
            }
            if let Some(v) = timing_sense {
                if arc.timing_sense != v {
                    return false;
                }
            }
            if let Some(v) = when {
                if arc.when != v {
                    return false;
                }
            }
            true
        });
        let first = matches.next()?;
        if matches.next().is_some() {
            return None;
        }
        Some(first)
    }

    /// Returns all timing arcs defined on the named pin.
    pub fn timing_arcs_for_pin(
        &self,
        cell_name: &str,
        output_pin_name: &str,
    ) -> Option<&[crate::liberty_proto::TimingArc]> {
        let pin = self.pin_by_name(cell_name, output_pin_name)?;
        Some(&pin.timing_arcs)
    }

    /// Looks up a table template by exact kind and name.
    pub fn lu_table_template_by_name(
        &self,
        template_kind: &str,
        template_name: &str,
    ) -> Option<&crate::liberty_proto::LuTableTemplate> {
        self.proto
            .lu_table_templates
            .iter()
            .find(|tmpl| tmpl.kind == template_kind && tmpl.name == template_name)
    }

    /// Looks up a table template by 1-based template ID as stored in
    /// `TimingTable`, requiring the expected template kind.
    pub fn lu_table_template_by_id(
        &self,
        template_id: u32,
        template_kind: &str,
    ) -> Option<&crate::liberty_proto::LuTableTemplate> {
        if template_id == 0 {
            return None;
        }
        let tmpl = self
            .proto
            .lu_table_templates
            .get((template_id - 1) as usize)?;
        if tmpl.kind == template_kind {
            Some(tmpl)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::liberty::test_utils::make_test_library;
    use crate::liberty_proto::{LuTableTemplate, TimingArc, TimingTable};

    #[test]
    fn get_cell_finds_by_name() {
        let lib = make_test_library();
        let indexed = IndexedLibrary::new(lib);

        let inv = indexed.get_cell("INV").expect("INV cell should exist");
        assert_eq!(inv.name, "INV");

        assert!(indexed.get_cell("NO_SUCH_CELL").is_none());
    }

    #[test]
    fn pins_for_dir_returns_expected_pins() {
        let lib = make_test_library();
        let indexed = IndexedLibrary::new(lib);

        // First call should build indices.
        let inputs = indexed
            .pins_for_dir("INV", PinDirection::Input)
            .expect("inputs for INV");
        let outputs = indexed
            .pins_for_dir("INV", PinDirection::Output)
            .expect("outputs for INV");

        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].name, "A");

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].name, "Y");

        // Second call exercises cached indices.
        let inputs_again = indexed
            .pins_for_dir("INV", PinDirection::Input)
            .expect("inputs for INV (cached)");
        assert_eq!(inputs_again.len(), 1);
        assert_eq!(inputs_again[0].name, "A");
    }

    #[test]
    fn pin_by_name_finds_pin_or_none() {
        let lib = make_test_library();
        let indexed = IndexedLibrary::new(lib);

        let y = indexed
            .pin_by_name("INV", "Y")
            .expect("Y pin should exist on INV");
        assert_eq!(y.name, "Y");

        assert!(indexed.pin_by_name("INV", "NO_SUCH_PIN").is_none());
        assert!(indexed.pin_by_name("NO_SUCH_CELL", "Y").is_none());
    }

    #[test]
    fn timing_lookup_finds_arc_and_template() {
        let lib = ProtoLibrary {
            cells: vec![Cell {
                name: "NAND2".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        function: "".to_string(),
                        name: "A".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Input as i32,
                        function: "".to_string(),
                        name: "B".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        function: "!(A * B)".to_string(),
                        name: "Y".to_string(),
                        is_clocking_pin: false,
                        timing_arcs: vec![TimingArc {
                            related_pin: "A".to_string(),
                            timing_sense: "negative_unate".to_string(),
                            timing_type: "combinational".to_string(),
                            when: String::new(),
                            tables: vec![TimingTable {
                                kind: "cell_rise".to_string(),
                                template_id: 1,
                                template_name: String::new(),
                                index_1: vec![0.01, 0.02],
                                index_2: vec![0.1, 0.2],
                                index_3: vec![],
                                values: vec![1.0, 2.0, 3.0, 4.0],
                                dimensions: vec![2, 2],
                            }],
                        }],
                        ..Default::default()
                    },
                ],
                area: 1.0,
                sequential: vec![],
                clock_gate: None,
            }],
            units: None,
            lu_table_templates: vec![LuTableTemplate {
                kind: "lu_table_template".to_string(),
                name: "tmpl_2x2".to_string(),
                variable_1: "input_net_transition".to_string(),
                variable_2: "total_output_net_capacitance".to_string(),
                variable_3: String::new(),
                index_1: vec![0.01, 0.02],
                index_2: vec![0.1, 0.2],
                index_3: vec![],
            }],
        };
        let indexed = IndexedLibrary::new(lib);

        let arcs = indexed
            .timing_arcs_for_pin("NAND2", "Y")
            .expect("timing arcs should exist for NAND2.Y");
        assert_eq!(arcs.len(), 1);

        let arc = indexed
            .timing_arc_for_pin("NAND2", "Y", "A")
            .expect("arc for related pin A should exist");
        assert_eq!(arc.timing_sense, "negative_unate");
        assert_eq!(arc.tables.len(), 1);
        assert_eq!(arc.tables[0].template_id, 1);

        let tmpl = indexed
            .lu_table_template_by_name("lu_table_template", "tmpl_2x2")
            .expect("template should exist");
        assert_eq!(tmpl.kind, "lu_table_template");
        let tmpl_by_id = indexed
            .lu_table_template_by_id(1, "lu_table_template")
            .expect("template should exist by id");
        assert_eq!(tmpl_by_id.name, "tmpl_2x2");
        assert!(
            indexed
                .lu_table_template_by_name("power_lut_template", "tmpl_2x2")
                .is_none()
        );
        assert!(
            indexed
                .lu_table_template_by_id(1, "power_lut_template")
                .is_none()
        );
    }

    #[test]
    fn template_lookup_is_kind_aware_under_name_collisions() {
        let lib = ProtoLibrary {
            cells: vec![],
            units: None,
            lu_table_templates: vec![
                LuTableTemplate {
                    kind: "lu_table_template".to_string(),
                    name: "shared".to_string(),
                    variable_1: "input_net_transition".to_string(),
                    variable_2: "total_output_net_capacitance".to_string(),
                    variable_3: String::new(),
                    index_1: vec![0.01, 0.02],
                    index_2: vec![0.1, 0.2],
                    index_3: vec![],
                },
                LuTableTemplate {
                    kind: "power_lut_template".to_string(),
                    name: "shared".to_string(),
                    variable_1: "input_transition_time".to_string(),
                    variable_2: "total_output_net_capacitance".to_string(),
                    variable_3: String::new(),
                    index_1: vec![1.0, 2.0],
                    index_2: vec![3.0, 4.0],
                    index_3: vec![],
                },
            ],
        };
        let indexed = IndexedLibrary::new(lib);

        let lu = indexed
            .lu_table_template_by_name("lu_table_template", "shared")
            .expect("lu template should resolve by kind and name");
        assert_eq!(lu.kind, "lu_table_template");
        let power = indexed
            .lu_table_template_by_name("power_lut_template", "shared")
            .expect("power template should resolve by kind and name");
        assert_eq!(power.kind, "power_lut_template");

        let lu_by_id = indexed
            .lu_table_template_by_id(1, "lu_table_template")
            .expect("id 1 should be lu template when kind matches");
        assert_eq!(lu_by_id.name, "shared");
        assert!(
            indexed
                .lu_table_template_by_id(1, "power_lut_template")
                .is_none()
        );
        let power_by_id = indexed
            .lu_table_template_by_id(2, "power_lut_template")
            .expect("id 2 should be power template when kind matches");
        assert_eq!(power_by_id.name, "shared");
    }

    #[test]
    fn timing_arc_lookup_avoids_ambiguous_related_pin_matches() {
        let lib = ProtoLibrary {
            cells: vec![Cell {
                name: "NAND2".to_string(),
                pins: vec![
                    Pin {
                        direction: PinDirection::Input as i32,
                        function: "".to_string(),
                        name: "A".to_string(),
                        is_clocking_pin: false,
                        ..Default::default()
                    },
                    Pin {
                        direction: PinDirection::Output as i32,
                        function: "!(A)".to_string(),
                        name: "Y".to_string(),
                        is_clocking_pin: false,
                        timing_arcs: vec![
                            TimingArc {
                                related_pin: "A".to_string(),
                                timing_sense: "negative_unate".to_string(),
                                timing_type: "combinational".to_string(),
                                when: String::new(),
                                tables: vec![],
                            },
                            TimingArc {
                                related_pin: "A".to_string(),
                                timing_sense: "positive_unate".to_string(),
                                timing_type: "setup_rising".to_string(),
                                when: "B".to_string(),
                                tables: vec![],
                            },
                        ],
                        ..Default::default()
                    },
                ],
                area: 1.0,
                sequential: vec![],
                clock_gate: None,
            }],
            units: None,
            lu_table_templates: vec![],
        };
        let indexed = IndexedLibrary::new(lib);

        assert!(indexed.timing_arc_for_pin("NAND2", "Y", "A").is_none());

        let comb = indexed
            .timing_arc_for_pin_filtered(
                "NAND2",
                "Y",
                "A",
                Some("combinational"),
                Some("negative_unate"),
                Some(""),
            )
            .expect("combinational arc should resolve uniquely");
        assert_eq!(comb.timing_type, "combinational");
        assert_eq!(comb.timing_sense, "negative_unate");
        assert_eq!(comb.when, "");

        let setup = indexed
            .timing_arc_for_pin_filtered(
                "NAND2",
                "Y",
                "A",
                Some("setup_rising"),
                Some("positive_unate"),
                Some("B"),
            )
            .expect("setup arc should resolve uniquely");
        assert_eq!(setup.timing_type, "setup_rising");
        assert_eq!(setup.timing_sense, "positive_unate");
        assert_eq!(setup.when, "B");
    }
}
