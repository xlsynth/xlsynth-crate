// SPDX-License-Identifier: Apache-2.0

mod ascii_stream;
pub mod cell_formula;
pub mod descriptor;
pub mod indexed;
pub mod liberty_parser;
pub mod liberty_to_proto;
pub mod util;
pub use indexed::IndexedLibrary;
pub use liberty_parser::{CharReader, LibertyParser};

#[cfg(test)]
pub mod test_utils {
    use crate::liberty_proto::{Cell, Library, Pin, PinDirection};

    /// Small test Liberty library used across unit tests. Contains:
    /// - `INVX1` with pins `A` (input) and `Y` (output).
    /// - `BUF` with pins `I` (input) and `O` (output).
    pub fn make_test_library() -> Library {
        Library {
            cells: vec![
                Cell {
                    name: "INVX1".to_string(),
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
                },
            ],
        }
    }
}
