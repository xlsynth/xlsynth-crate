// SPDX-License-Identifier: Apache-2.0

mod ascii_stream;
pub mod cell_formula;
pub mod descriptor;
pub mod indexed;
pub mod liberty_parser;
pub mod liberty_to_proto;
pub mod query;
pub mod util;
pub use indexed::IndexedLibrary;
pub use liberty_parser::{CharReader, LibertyParser};

#[cfg(test)]
pub mod test_utils {
    use crate::liberty_proto::{Cell, Library, Pin, PinDirection};

    /// Small test Liberty library used across unit tests. Contains:
    /// - `INV` with pins `A` (input) and `Y` (output).
    /// - `BUF` with pins `I` (input) and `O` (output).
    /// - `DFF` with pins `D` (input), `CLK` (clocking input), and `Q` (output).
    pub fn make_test_library() -> Library {
        Library {
            cells: vec![
                Cell {
                    name: "INV".to_string(),
                    pins: vec![
                        Pin {
                            direction: PinDirection::Input as i32,
                            function: "".to_string(),
                            name: "A".to_string(),
                            is_clocking_pin: false,
                        },
                        Pin {
                            direction: PinDirection::Output as i32,
                            function: "(!A)".to_string(),
                            name: "Y".to_string(),
                            is_clocking_pin: false,
                        },
                    ],
                    area: 1.0,
                    sequential: vec![],
                },
                Cell {
                    name: "BUF".to_string(),
                    pins: vec![
                        Pin {
                            direction: PinDirection::Input as i32,
                            function: "".to_string(),
                            name: "I".to_string(),
                            is_clocking_pin: false,
                        },
                        Pin {
                            direction: PinDirection::Output as i32,
                            function: "I".to_string(),
                            name: "O".to_string(),
                            is_clocking_pin: false,
                        },
                    ],
                    area: 1.0,
                    sequential: vec![],
                },
                Cell {
                    name: "DFF".to_string(),
                    pins: vec![
                        Pin {
                            direction: PinDirection::Input as i32,
                            function: "".to_string(),
                            name: "D".to_string(),
                            is_clocking_pin: false,
                        },
                        Pin {
                            direction: PinDirection::Input as i32,
                            function: "".to_string(),
                            name: "CLK".to_string(),
                            is_clocking_pin: true,
                        },
                        Pin {
                            direction: PinDirection::Output as i32,
                            function: "D".to_string(),
                            name: "Q".to_string(),
                            is_clocking_pin: false,
                        },
                    ],
                    area: 1.0,
                    sequential: vec![],
                },
            ],
        }
    }
}
