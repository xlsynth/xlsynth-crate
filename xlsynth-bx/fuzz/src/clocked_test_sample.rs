// SPDX-License-Identifier: Apache-2.0

//! Generate a clocked BX block and a DSLX proc over the same input sequence.

use std::fmt::Write;

/// A register machine and the inputs presented on its successive clock edges.
#[derive(Debug)]
pub struct ClockedTestSample {
    width: usize,
    active_low: bool,
    explicit_reset: bool,
    registers: Vec<Register>,
    rows: Vec<Row>,
}

/// A read in the combinational logic that feeds a register's data input.
#[derive(Debug)]
enum Operand {
    A,
    B,
    Register(usize),
}

impl Operand {
    fn from_byte(byte: u8, register_count: usize) -> Self {
        match usize::from(byte) % (register_count + 2) {
            0 => Self::A,
            1 => Self::B,
            index => Self::Register(index - 2),
        }
    }

    fn expression(&self) -> String {
        match self {
            Self::A => "drive.a".into(),
            Self::B => "drive.b".into(),
            Self::Register(i) => format!("q{i}"),
        }
    }
}

/// The reset value, write enable and binary data expression of one register.
#[derive(Debug)]
struct Register {
    reset_value: String,
    enable: &'static str,
    lhs: Operand,
    operator: &'static str,
    rhs: Operand,
}

impl Register {
    fn data_expression(&self) -> String {
        format!(
            "{} {} {}",
            self.lhs.expression(),
            self.operator,
            self.rhs.expression()
        )
    }
}

/// Concrete data, enable and logical reset levels sampled on one clock edge.
#[derive(Debug)]
struct Row {
    a: String,
    b: String,
    en_a: bool,
    en_b: bool,
    reset: bool,
}

impl Row {
    fn drive_literal(&self) -> String {
        format!(
            "Drive {{ a: {}, b: {}, en_a: {}, en_b: {} }}",
            self.a, self.b, self.en_a, self.en_b
        )
    }
}

/// Consume enough bytes to represent every possible value at the given width.
fn word_literal(width: usize, byte: &mut impl FnMut() -> u8) -> String {
    let mut literal = format!("uN[{width}]:0x");
    for i in 0..width.div_ceil(8) {
        let mut value = byte();
        if i == 0 && !width.is_multiple_of(8) {
            value &= (1 << (width % 8)) - 1;
        }
        write!(literal, "{value:02x}").unwrap();
    }
    literal
}

impl ClockedTestSample {
    /// Decode a bounded machine and input table, including from an empty
    /// corpus.
    pub fn from_bytes(data: &[u8]) -> Self {
        let mut bytes = data.iter().copied().cycle().chain(std::iter::repeat(0));
        let mut byte = || bytes.next().unwrap();
        let width = 1 + usize::from(byte() % 128);
        let dimensions = byte();
        let register_count = 1 + usize::from(dimensions & 3);
        let row_count = 2 + usize::from((dimensions >> 2) & 7);
        let active_low = dimensions & 0x80 != 0;
        let registers = (0..register_count)
            .map(|_| {
                let lhs = Operand::from_byte(byte(), register_count);
                let rhs = Operand::from_byte(byte(), register_count);
                let control = byte();
                let operators = ["+", "^", "-", "|", "&"];
                let enables = [
                    "true",
                    "drive.en_a",
                    "drive.en_b",
                    "!drive.en_a",
                    "drive.en_a ^ drive.en_b",
                ];
                Register {
                    lhs,
                    rhs,
                    operator: operators[usize::from(control & 15) % operators.len()],
                    enable: enables[usize::from(control >> 4) % enables.len()],
                    reset_value: word_literal(width, &mut byte),
                }
            })
            .collect();
        let rows = (0..row_count)
            .map(|i| {
                let flags = byte();
                Row {
                    a: word_literal(width, &mut byte),
                    b: word_literal(width, &mut byte),
                    en_a: flags & 1 != 0,
                    en_b: flags & 2 != 0,
                    // The first edge makes all BX power-up unknowns
                    // unobservable.
                    reset: i == 0 || flags & 12 == 12,
                }
            })
            .collect();
        Self {
            width,
            active_low,
            explicit_reset: dimensions & 0x40 != 0,
            registers,
            rows,
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn active_low(&self) -> bool {
        self.active_low
    }

    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    pub fn register_count(&self) -> usize {
        self.registers.len()
    }

    fn declarations(&self) -> String {
        format!(
            "type Word = uN[{}];\nstruct Drive {{ a: Word, b: Word, en_a: bool, en_b: bool }}\n",
            self.width
        )
    }

    /// Trace the state computed by each DSLX proc activation, indexed by edge.
    pub fn dslx_source(&self) -> String {
        let mut source = self.declarations();
        let row_count = self.row_count();
        let register_count = self.register_count();
        writeln!(source, "const ROWS: (Drive, bool)[{row_count}] = [").unwrap();
        for row in &self.rows {
            writeln!(source, "    ({}, {}),", row.drive_literal(), row.reset).unwrap();
        }
        writeln!(
            source,
            r#"];

#[test_proc]
proc clocked_oracle {{
    terminator: chan<bool> out;
    config(terminator: chan<bool> out) {{ (terminator,) }}
    init {{ (u32:0, zero!<Word[{register_count}]>()) }}
    next(state: (u32, Word[{register_count}])) {{
        let (row_index, old) = state;
        let (drive, reset) = ROWS[row_index];"#
        )
        .unwrap();
        for i in 0..register_count {
            writeln!(source, "        let q{i} = old[u32:{i}];").unwrap();
        }
        for (i, reg) in self.registers.iter().enumerate() {
            writeln!(
                source,
                "        let n{i} = if reset {{ {} }} else if {} {{ {} }} else {{ q{i} }};",
                reg.reset_value,
                reg.enable,
                reg.data_expression()
            )
            .unwrap();
        }
        let names = (0..register_count)
            .map(|i| format!("n{i}"))
            .collect::<Vec<_>>()
            .join(", ");
        let fields = " {:u}".repeat(register_count);
        writeln!(
            source,
            r#"        trace_fmt!("BX_OBSERVE {{:u}}{fields}", row_index, {names});
        let _ = send_if(join(), terminator, row_index == u32:{last}, true);
        (row_index + u32:1, [{names}])
    }}
}}"#,
            last = row_count - 1,
        )
        .unwrap();
        source
    }

    /// Run the same edges in BX and compare every register with the proc trace.
    pub fn bx_source(&self, observations: &[Vec<String>]) -> String {
        assert_eq!(observations.len(), self.row_count());
        let mut source = self.declarations();
        let register_count = self.register_count();
        let reset = if self.explicit_reset {
            source.insert_str(0, "#![allow(block_reset_signal)]\n");
            ", rst"
        } else {
            ""
        };
        writeln!(
            source,
            "block machine(clk{reset}, drive: Drive) -> Word[{register_count}] {{"
        )
        .unwrap();
        for (i, reg) in self.registers.iter().enumerate() {
            writeln!(source, "    reg q{i}: Word = {};", reg.reset_value).unwrap();
        }
        for (i, reg) in self.registers.iter().enumerate() {
            writeln!(
                source,
                "    if {} {{ q{i} := {}; }}",
                reg.enable,
                reg.data_expression()
            )
            .unwrap();
        }
        let names = (0..register_count)
            .map(|i| format!("q{i}"))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            source,
            "    [{names}]\n}}\n\nconst CASES: (Drive, bool, Word[{register_count}])[{}] = [",
            self.row_count()
        )
        .unwrap();
        for (row, values) in self.rows.iter().zip(observations) {
            assert_eq!(values.len(), register_count);
            let expected = values
                .iter()
                .map(|v| format!("uN[{}]:{v}", self.width))
                .collect::<Vec<_>>()
                .join(", ");
            writeln!(
                source,
                "    ({}, {}, [{expected}]),",
                row.drive_literal(),
                row.reset
            )
            .unwrap();
        }
        source.push_str(
            r#"];

#[test]
block clocked_rows() {
    reg drive: Drive;
    let observed = machine(drive);
    for (inputs, reset, expected) in CASES {
        drive := inputs;
        tick!(rst: reset);
        assert_eq(observed, expected);
    }
}
"#,
        );
        source
    }
}
