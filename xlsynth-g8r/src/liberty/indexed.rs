// SPDX-License-Identifier: Apache-2.0

//! Indexed view over a Liberty `Library`.
//!
//! This module provides a small wrapper around the prost-generated
//! `crate::liberty_proto::Library` that adds lazily constructed indices for
//! fast lookup by cell name and for grouping pins by direction. The underlying
//! `Library` remains the single source of truth; indices are derived from it
//! on demand.

use crate::liberty_proto::{Cell, Library, Pin, PinDirection};
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
    proto: Library,
    /// Map from cell name to index in `proto.cells`.
    cell_by_name: RefCell<Option<HashMap<String, usize>>>,
    /// Map from cell index to directional pin groupings.
    pins_by_dir: RefCell<Option<HashMap<usize, CellPinsByDir>>>,
}

impl IndexedLibrary {
    /// Constructs a new indexed view over the given `Library`.
    pub fn new(proto: Library) -> Self {
        IndexedLibrary {
            proto,
            cell_by_name: RefCell::new(None),
            pins_by_dir: RefCell::new(None),
        }
    }

    /// Returns a shared reference to the underlying `Library`.
    pub fn library(&self) -> &Library {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_library() -> Library {
        Library {
            cells: vec![
                Cell {
                    name: "INVX1".to_string(),
                    pins: vec![
                        Pin {
                            direction: PinDirection::Input as i32,
                            function: "".to_string(),
                            name: "A".to_string(),
                        },
                        Pin {
                            direction: PinDirection::Output as i32,
                            function: "(!A)".to_string(),
                            name: "Y".to_string(),
                        },
                    ],
                    area: 1.0,
                },
                Cell {
                    name: "BUF".to_string(),
                    pins: vec![
                        Pin {
                            direction: PinDirection::Input as i32,
                            function: "".to_string(),
                            name: "I".to_string(),
                        },
                        Pin {
                            direction: PinDirection::Output as i32,
                            function: "I".to_string(),
                            name: "O".to_string(),
                        },
                    ],
                    area: 1.0,
                },
            ],
        }
    }

    #[test]
    fn get_cell_finds_by_name() {
        let lib = make_test_library();
        let indexed = IndexedLibrary::new(lib);

        let inv = indexed.get_cell("INVX1").expect("INVX1 cell should exist");
        assert_eq!(inv.name, "INVX1");

        assert!(indexed.get_cell("NO_SUCH_CELL").is_none());
    }

    #[test]
    fn pins_for_dir_returns_expected_pins() {
        let lib = make_test_library();
        let indexed = IndexedLibrary::new(lib);

        // First call should build indices.
        let inputs = indexed
            .pins_for_dir("INVX1", PinDirection::Input)
            .expect("inputs for INVX1");
        let outputs = indexed
            .pins_for_dir("INVX1", PinDirection::Output)
            .expect("outputs for INVX1");

        assert_eq!(inputs.len(), 1);
        assert_eq!(inputs[0].name, "A");

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].name, "Y");

        // Second call exercises cached indices.
        let inputs_again = indexed
            .pins_for_dir("INVX1", PinDirection::Input)
            .expect("inputs for INVX1 (cached)");
        assert_eq!(inputs_again.len(), 1);
        assert_eq!(inputs_again[0].name, "A");
    }

    #[test]
    fn pin_by_name_finds_pin_or_none() {
        let lib = make_test_library();
        let indexed = IndexedLibrary::new(lib);

        let y = indexed
            .pin_by_name("INVX1", "Y")
            .expect("Y pin should exist on INVX1");
        assert_eq!(y.name, "Y");

        assert!(indexed.pin_by_name("INVX1", "NO_SUCH_PIN").is_none());
        assert!(indexed.pin_by_name("NO_SUCH_CELL", "Y").is_none());
    }
}
