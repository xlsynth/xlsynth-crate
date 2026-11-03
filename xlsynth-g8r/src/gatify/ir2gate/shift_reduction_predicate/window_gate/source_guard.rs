// SPDX-License-Identifier: Apache-2.0

//! Split the complete OR observation on a word selector already evaluated by
//! the planner. Each arm needs only the control bits distinguishing its cared
//! rows.

use super::{AigBitVector, AigOperand, BTreeMap, Cell, GateBuilder, HashMap, ReductionKind};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Row {
    True,
    Any(u128),
}

pub(super) struct Split {
    pieces: Vec<(usize, usize, usize)>,
    arms: [Arm; 2],
}

struct Arm {
    bits: Vec<usize>,
    rows: Vec<Row>,
}

impl Split {
    pub(super) fn prepare(
        cells: &[Cell],
        guard: &[bool],
        bounds: &BTreeMap<usize, Vec<usize>>,
    ) -> Option<Self> {
        if cells.len() != guard.len() || !cells.len().is_power_of_two() || cells.len() <= 2 {
            return None;
        }
        let count = bounds
            .values()
            .map(|xs| xs.len().saturating_sub(1))
            .sum::<usize>();
        if count > u128::BITS as usize {
            return None;
        }
        let mut pieces = Vec::with_capacity(count);
        let mut offsets = BTreeMap::new();
        for (&word, cuts) in bounds {
            offsets.insert(word, pieces.len());
            pieces.extend(cuts.windows(2).map(|p| (word, p[0], p[1])));
        }
        let mut rows = Vec::with_capacity(cells.len());
        for cell in cells {
            rows.push(match cell {
                Cell::Const(true) => Row::True,
                Cell::Const(false) => Row::Any(0),
                Cell::Any(spans) => {
                    let mut mask = 0;
                    for &(word, lo, hi) in spans {
                        let cuts = &bounds[&word.index];
                        let start = cuts.binary_search(&lo).ok()?;
                        let end = cuts.binary_search(&hi).ok()?;
                        for i in start..end {
                            mask |= 1u128 << (offsets[&word.index] + i);
                        }
                    }
                    Row::Any(mask)
                }
            });
        }
        Some(Self {
            pieces,
            arms: [
                Arm::project(&rows, guard, false)?,
                Arm::project(&rows, guard, true)?,
            ],
        })
    }

    pub(super) fn widths(&self) -> [usize; 2] {
        [self.arms[0].bits.len(), self.arms[1].bits.len()]
    }

    pub(super) fn emit(
        &self,
        words: &BTreeMap<usize, AigBitVector>,
        controls: &[AigOperand],
        selector: AigOperand,
        gb: &mut GateBuilder,
    ) -> AigOperand {
        let pieces: Vec<_> = self
            .pieces
            .iter()
            .map(|&(word, lo, hi)| {
                gb.add_or_reduce(
                    &words[&word].get_lsb_slice(lo, hi - lo),
                    ReductionKind::Tree,
                )
            })
            .collect();
        let mut terms = HashMap::new();
        let [off, on] = self.arms.each_ref().map(|arm| {
            let selected: Vec<_> = arm.bits.iter().map(|&b| controls[b]).collect();
            arm_value(&arm.rows, &selected, &pieces, &mut terms, gb)
        });
        gb.add_mux2(selector, on, off)
    }
}

impl Arm {
    fn project(rows: &[Row], guard: &[bool], selected: bool) -> Option<Self> {
        let mut table: Vec<_> = rows
            .iter()
            .zip(guard)
            .map(|(&v, &g)| (g == selected).then_some(v))
            .collect();
        if table.iter().all(Option::is_none) {
            return None;
        }
        let width = rows.len().ilog2() as usize;
        let mut bits: Vec<_> = (0..width).collect();
        for bit in (0..width).rev() {
            let stride = 1 << bit;
            let pairs = || {
                table
                    .chunks_exact(2 * stride)
                    .flat_map(|chunk| chunk[..stride].iter().zip(&chunk[stride..]))
            };
            if pairs().any(|(a, b)| a.is_some() && b.is_some() && a != b) {
                continue;
            }
            table = pairs().map(|(&a, &b)| a.or(b)).collect();
            bits.remove(bit);
        }
        (bits.len() < width).then(|| Self {
            bits,
            rows: table
                .into_iter()
                .map(|v| v.unwrap_or(Row::Any(0)))
                .collect(),
        })
    }
}

fn arm_value(
    rows: &[Row],
    inputs: &[AigOperand],
    pieces: &[AigOperand],
    terms: &mut HashMap<u128, AigOperand>,
    gb: &mut GateBuilder,
) -> AigOperand {
    if rows.iter().all(|&v| v == rows[0]) {
        return match rows[0] {
            Row::True => gb.get_true(),
            Row::Any(mask) => data_or(mask, pieces, terms, gb),
        };
    }
    // An always-true row contains every common term. Its residual may remain
    // true.
    let common = rows.iter().fold(u128::MAX, |mask, row| match row {
        Row::True => mask,
        Row::Any(bits) => mask & bits,
    });
    if common != 0 {
        let residual: Vec<_> = rows
            .iter()
            .map(|r| match r {
                Row::True => Row::True,
                Row::Any(bits) => Row::Any(bits & !common),
            })
            .collect();
        let a = data_or(common, pieces, terms, gb);
        let b = arm_value(&residual, inputs, pieces, terms, gb);
        return gb.add_or_binary(a, b);
    }
    let (off, on) = rows.split_at(rows.len() / 2);
    let (last, earlier) = inputs.split_last().unwrap();
    let lo = arm_value(off, earlier, pieces, terms, gb);
    if off == on {
        return lo;
    }
    let hi = arm_value(on, earlier, pieces, terms, gb);
    gb.add_mux2(*last, hi, lo)
}

fn data_or(
    mask: u128,
    pieces: &[AigOperand],
    terms: &mut HashMap<u128, AigOperand>,
    gb: &mut GateBuilder,
) -> AigOperand {
    if mask == 0 {
        return gb.get_false();
    }
    *terms.entry(mask).or_insert_with(|| {
        let mut remaining = mask;
        let mut inputs = Vec::with_capacity(mask.count_ones() as usize);
        while remaining != 0 {
            inputs.push(pieces[remaining.trailing_zeros() as usize]);
            remaining &= remaining - 1;
        }
        gb.add_or_nary(&inputs, ReductionKind::Tree)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig_sim::gate_sim::{self, Collect};
    use crate::gate_builder::GateBuilderOptions;
    use xlsynth_pir::{IrBits, ir::NodeRef};

    #[test]
    fn projected_multiword_arms_preserve_forced_values_holes_and_guard_polarity() {
        let a = NodeRef { index: 1 };
        let b = NodeRef { index: 2 };
        let bounds = BTreeMap::from([(a.index, vec![0, 1, 2, 3]), (b.index, vec![0, 1, 2])]);
        for guard in [
            [false, true, true, true, false, false, false, false],
            [false, true, true, false, true, false, false, true],
        ] {
            let mut gb = GateBuilder::new(
                "guarded".into(),
                GateBuilderOptions {
                    hash: true,
                    fold: true,
                },
            );
            let x = gb.add_input("x".into(), 3);
            let y = gb.add_input("y".into(), 2);
            let c = gb.add_input("c".into(), 3);
            let selector = gb
                .add_input("complement_guard".into(), 1)
                .get_lsb(0)
                .negate();
            let cells: Vec<_> = (0..8)
                .map(|i| match (guard[i], i & 1, (i >> 1) & 1) {
                    (true, _, 0) => Cell::Const(false),
                    (true, _, _) => Cell::Any(vec![(a, 0, 1), (a, 2, 3), (b, 1, 2)]),
                    (false, 0, _) => Cell::Any(vec![(a, 0, 1), (b, 0, 1)]),
                    _ => Cell::Const(true),
                })
                .collect();
            let split = Split::prepare(&cells, &guard, &bounds).unwrap();
            let value = split.emit(
                &BTreeMap::from([(a.index, x), (b.index, y)]),
                c.iter_lsb_to_msb().copied().collect::<Vec<_>>().as_slice(),
                selector,
                &mut gb,
            );
            gb.add_output("value".into(), value.into());
            let gf = gb.build();
            for k in 0..8 {
                for x in 0..8 {
                    for y in 0..4 {
                        let expect = if guard[k] {
                            k & 2 != 0 && (x & 5 != 0 || y & 2 != 0)
                        } else {
                            k & 1 != 0 || x & 1 != 0 || y & 1 != 0
                        };
                        let args = [
                            IrBits::make_ubits(3, x).unwrap(),
                            IrBits::make_ubits(2, y).unwrap(),
                            IrBits::make_ubits(3, k as u64).unwrap(),
                            IrBits::make_ubits(1, u64::from(!guard[k])).unwrap(),
                        ];
                        assert_eq!(
                            gate_sim::eval(&gf, &args, Collect::None).outputs[0]
                                .get_bit(0)
                                .unwrap(),
                            expect
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn uncompressible_and_over_capacity_tables_keep_existing_emission() {
        let word = NodeRef { index: 5 };
        let full = Cell::Any(vec![(word, 0, 1)]);
        let cells = vec![Cell::Const(false), full.clone(), full, Cell::Const(false)];
        let bounds = BTreeMap::from([(word.index, vec![0, 1])]);
        assert!(Split::prepare(&cells, &[true; 4], &bounds).is_none());
        // The off arm distinguishes all inputs on its cared rows; no control
        // can be discarded.
        assert!(Split::prepare(&cells, &[false, false, false, true], &bounds).is_none());
        let too_many = BTreeMap::from([(word.index, (0..u128::BITS as usize + 2).collect())]);
        assert!(Split::prepare(&cells, &[false, true, false, true], &too_many).is_none());
    }
}
