// SPDX-License-Identifier: Apache-2.0

//! Compare structural maps and indexed folds when their bodies instantiate
//! registers.

use std::fmt::Write;

use crate::clocked_test::run_clocked_rows;

const GRID: &str = include_str!("../../examples/grid.bx");

/// Compare every lane with an indexed fold and a host register model after each
/// edge.
pub fn check_unroll_map_clocked(data: &[u8]) -> Result<(), String> {
    let mut header = data.iter().copied().chain(std::iter::repeat(0));
    let rows = 1 + usize::from(header.next().unwrap() % 3);
    let cols = 1 + usize::from(header.next().unwrap() % 3);
    let width = 1 + usize::from(header.next().unwrap() % 128);
    let cycles = 4 + usize::from(header.next().unwrap() % 4);
    let mut stimulus = data
        .get(4..)
        .unwrap_or_default()
        .iter()
        .copied()
        .cycle()
        .chain(std::iter::repeat(0));
    let lanes = rows * cols;
    let mut source = format!(
        r#"{GRID}
type Word = uN[{width}];
type Drive = (Word, bool, bool)[{cols}][{rows}];
type Lanes = Word[{lanes}];
block lane<R: u32, C: u32>(clk, drive: (Word, bool, bool)) -> Word {{
    let (data, enable, clear) = drive;
    let tag = (R * u32:7 + C + u32:1) as Word;
    reg q: Word = zero!<Word>();
    if clear || enable {{ q := if clear {{ tag }} else {{ q + (data ^ tag) }}; }}
    q
}}
block mapped(clk, drive: Drive) -> Lanes {{
    unroll_map! (r, c) in product<{rows}, {cols}>() {{ lane<r, c>(drive[r][c]) }}
}}
block folded(clk, drive: Drive) -> Lanes {{
    unroll_for! (i, acc) in u32:0..u32:{lanes} {{
        let r = i / u32:{cols};
        let c = i % u32:{cols};
        let q = lane<r, c>(drive[r][c]);
        update(acc, i, q)
    }}(zero!<Lanes>())
}}
#[test] block clocked_rows() {{
    reg drive: Drive;
    let collected = mapped(drive);
    let indexed = folded(drive);
"#
    );
    let mask = u128::MAX >> (128 - width);
    let mut state = vec![0u128; lanes];
    for cycle in 0..cycles {
        let control = stimulus.next().unwrap();
        // Establish a known power-up state, then allow later synchronous
        // resets.
        let reset = cycle == 0 || (cycle >= 3 && control & 7 == 7);
        let mut entries = Vec::with_capacity(lanes);
        for (i, q) in state.iter_mut().enumerate() {
            let flags = stimulus.next().unwrap();
            let tag = (i / cols * 7 + i % cols + 1) as u128 & mask;
            let enable = (usize::from(flags) + cycle + i) & 1 != 0;
            let clear = (usize::from(flags >> 1) + cycle + i) % 5 == 0;
            let mut value = 0u128;
            for _ in 0..width.div_ceil(8) {
                value = (value << 8) | u128::from(stimulus.next().unwrap());
            }
            let value = value.wrapping_add((cycle + i) as u128) & mask;
            if reset {
                *q = 0;
            } else if clear {
                *q = tag;
            } else if enable {
                *q = q.wrapping_add(value ^ tag) & mask;
            }
            entries.push(format!("(Word:0x{value:x}, {enable}, {clear})"));
        }
        let drive = entries
            .chunks(cols)
            .map(|row| format!("[{}]", row.join(", ")))
            .collect::<Vec<_>>()
            .join(", ");
        let expected = state
            .iter()
            .map(|q| format!("Word:0x{q:x}"))
            .collect::<Vec<_>>()
            .join(", ");
        writeln!(
            source,
            "    drive := [{drive}]; tick!(rst: {reset});
    assert_eq(collected, indexed); assert_eq(collected, [{expected}]);"
        )
        .unwrap();
    }
    source.push('}');
    run_clocked_rows(&source, false, None).map_err(|e| format!("{e}\nBX:\n{source}"))
}

#[cfg(test)]
mod tests {
    use super::check_unroll_map_clocked;

    #[test]
    fn stateful_maps_match_indexed_folds_edge_by_edge() -> Result<(), String> {
        for (rows, cols) in [(1, 1), (2, 3), (3, 2)] {
            for width in [1, 8, 65, 128] {
                check_unroll_map_clocked(&[rows - 1, cols - 1, width - 1, 3, 7, 42, 205, 127])?;
            }
        }
        Ok(())
    }
}
