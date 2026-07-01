// SPDX-License-Identifier: Apache-2.0

mod ascii_stream;
pub mod cell_filter;
pub mod cell_formula;
pub mod descriptor;
pub mod indexed;
pub mod liberty_parser;
pub mod load;
pub mod lut;
pub mod model;
pub mod parser;
pub mod proto_info;
pub mod query;
pub mod timing_table;
pub mod util;
pub use cell_filter::{CellFilterAction, CellFilterPolicy, CellFilterRule, CellFilterStats};
pub use indexed::IndexedLibrary;
pub use liberty_parser::{CharReader, LibertyParser};
pub use model::{Library, library_from_proto, library_to_proto};
pub use proto_info::{LibertyProtoInfo, liberty_proto_info_from_path};
pub use timing_table::{TimingTableArrayError, TimingTableArrayView};

#[cfg(test)]
pub mod test_utils {
    use crate::liberty_model::{Cell, Library, LibraryBuilder, Pin, PinDirection};

    fn pin(
        builder: &mut LibraryBuilder,
        direction: PinDirection,
        name: &str,
        function: &str,
        is_clocking_pin: bool,
    ) -> Pin {
        Pin {
            direction: direction as i32,
            function: builder.intern_string(function).unwrap(),
            name: builder.intern_string(name).unwrap(),
            is_clocking_pin,
            ..Default::default()
        }
    }

    /// Small test Liberty library used across unit tests. Contains:
    /// - `INV` with pins `A` (input) and `Y` (output).
    /// - `BUF` with pins `I` (input) and `O` (output).
    /// - `DFF` with pins `D` (input), `CLK` (clocking input), and `Q` (output).
    pub fn make_test_library() -> Library {
        let mut builder = LibraryBuilder::new();
        let cells = vec![
            Cell {
                name: "INV".to_string().into(),
                pins: vec![
                    pin(&mut builder, PinDirection::Input, "A", "", false),
                    pin(&mut builder, PinDirection::Output, "Y", "(!A)", false),
                ],
                area: 1.0,
                sequential: vec![],
                clock_gate: None,
                ..Default::default()
            },
            Cell {
                name: "BUF".to_string().into(),
                pins: vec![
                    pin(&mut builder, PinDirection::Input, "I", "", false),
                    pin(&mut builder, PinDirection::Output, "O", "I", false),
                ],
                area: 1.0,
                sequential: vec![],
                clock_gate: None,
                ..Default::default()
            },
            Cell {
                name: "DFF".to_string().into(),
                pins: vec![
                    pin(&mut builder, PinDirection::Input, "D", "", false),
                    pin(&mut builder, PinDirection::Input, "CLK", "", true),
                    pin(&mut builder, PinDirection::Output, "Q", "D", false),
                ],
                area: 1.0,
                sequential: vec![],
                clock_gate: None,
                ..Default::default()
            },
        ];
        builder.cells = cells;
        builder.finish()
    }
}
