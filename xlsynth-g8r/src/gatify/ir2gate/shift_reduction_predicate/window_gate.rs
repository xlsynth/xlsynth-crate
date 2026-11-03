// SPDX-License-Identifier: Apache-2.0

//! Contract a guarded source-bit observation in a reversible gate region.

use super::super::GateEnv;
use super::window::{Cell, Plan};
use crate::aig::gate::{AigBitVector, AigOperand, AigRef};
use crate::gate_builder::{GateBuilder, ReductionKind};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Data words, partition bounds and control bits resolved in one AIG mapping.
pub(super) struct WindowInputs {
    words: BTreeMap<usize, AigBitVector>,
    bounds: BTreeMap<usize, Vec<usize>>,
    controls: Vec<AigOperand>,
}

impl WindowInputs {
    /// Resolve the inputs and bound the size of the partition membership table.
    pub(super) fn collect(plan: &Plan, env: &GateEnv) -> Option<Self> {
        let mut words = BTreeMap::new();
        let mut bounds = BTreeMap::<usize, Vec<usize>>::new();
        for cell in &plan.cells {
            if let Cell::Any(spans) = cell {
                for &(word, lo, hi) in spans {
                    if let std::collections::btree_map::Entry::Vacant(slot) =
                        words.entry(word.index)
                    {
                        slot.insert(env.get_bit_vector(word).ok()?);
                    }
                    bounds.entry(word.index).or_default().extend([lo, hi]);
                }
            }
        }
        let mut controls = Vec::new();
        for &r in &plan.source {
            controls.extend(env.get_bit_vector(r).ok()?.iter_lsb_to_msb().copied());
        }
        if 1usize.checked_shl(u32::try_from(controls.len()).ok()?)? != plan.cells.len() {
            return None;
        }
        for cuts in bounds.values_mut() {
            cuts.sort_unstable();
            cuts.dedup();
        }
        // Bound the membership table before constructing any alternate gates.
        let partitions: usize = bounds.values().map(|xs| xs.len().saturating_sub(1)).sum();
        if partitions.checked_mul(plan.cells.len())? > 32_768 {
            return None;
        }
        Some(Self {
            words,
            bounds,
            controls,
        })
    }

    /// Ordinary data and control gates at the cut of every window candidate.
    pub(super) fn source_gates(&self) -> HashSet<AigRef> {
        self.words
            .values()
            .flat_map(|word| word.iter_lsb_to_msb())
            .chain(&self.controls)
            .map(|bit| bit.node)
            .collect()
    }

    /// Emit the same candidate in the ordinary probe and the second mapping.
    pub(super) fn emit(&self, plan: &Plan, gb: &mut GateBuilder) -> AigOperand {
        if plan.named {
            let data = self
                .words
                .values()
                .next()
                .expect("named domain has one data source");
            let cells = std::array::from_fn(|i| match &plan.cells[i] {
                Cell::Any(xs) => xs.iter().map(|&(_, l, h)| (l, h)).collect(),
                _ => unreachable!("named domain has a nonempty range for every assignment"),
            });
            emit_named(
                &cells,
                plan.inverted,
                data,
                [self.controls[0], self.controls[1]],
                gb,
            )
        } else {
            let value = emit_shared(&plan.cells, &self.bounds, &self.words, &self.controls, gb);
            if plan.inverted { value.negate() } else { value }
        }
    }
}

pub(super) fn lower(
    plan: &Plan,
    ordinary: AigOperand,
    env: &GateEnv,
    gb: &mut GateBuilder,
) -> Option<AigOperand> {
    let inputs = WindowInputs::collect(plan, env)?;
    let boundary: Vec<_> = inputs
        .words
        .values()
        .flat_map(|v| v.iter_lsb_to_msb().copied())
        .chain(inputs.controls.iter().copied())
        .collect();
    // Price the union with retained outputs so gates in their fanin (including
    // partially shared selector trees) are counted once on each side.
    let mut roots = vec![ordinary];
    for &r in &plan.retained_controls {
        roots.extend(env.get_bit_vector(r).ok()?.iter_lsb_to_msb().copied());
    }
    let ordinary_outputs = AigBitVector::from_lsb_is_index_0(&roots);
    let (result, trial) = gb.try_region_alternative(ordinary_outputs, 0, &boundary, |gb| {
        roots[0] = inputs.emit(plan, gb);
        AigBitVector::from_lsb_is_index_0(&roots)
    });
    log::debug!(
        "shift_window trial node={:?} named={} cost={trial:?}",
        plan.root,
        plan.named
    );
    let value = *result.get_lsb(0);
    (value != ordinary).then_some(value)
}

/// Each condition selects a disjoint set of control assignments. Data segments
/// with the same condition share one reduction and one enable.
fn emit_shared(
    cells: &[Cell],
    bounds: &BTreeMap<usize, Vec<usize>>,
    words: &BTreeMap<usize, AigBitVector>,
    controls: &[AigOperand],
    gb: &mut GateBuilder,
) -> AigOperand {
    let mut group_by_mask = HashMap::new();
    let mut groups: Vec<(Vec<bool>, Vec<AigOperand>)> = Vec::new();
    for (&word, cuts) in bounds {
        for edges in cuts.windows(2) {
            let (lo, hi) = (edges[0], edges[1]);
            let mask: Vec<_> = cells
                .iter()
                .map(|c| match c {
                    Cell::Const(_) => false,
                    Cell::Any(xs) => xs
                        .iter()
                        .any(|&(r, l, h)| r.index == word && l <= lo && hi <= h),
                })
                .collect();
            if !mask.iter().any(|v| *v) {
                continue;
            }
            let pos = *group_by_mask.entry(mask.clone()).or_insert_with(|| {
                groups.push((mask, Vec::new()));
                groups.len() - 1
            });
            let term = reduce_spans(&words[&word], &[(lo, hi)], gb).expect("nonempty interval");
            groups[pos].1.push(term);
        }
    }
    let mut memo = HashMap::new();
    let forced: Vec<_> = cells
        .iter()
        .map(|c| matches!(c, Cell::Const(true)))
        .collect();
    let mut terms = vec![condition(&forced, controls, gb, &mut memo)];
    for (mask, pieces) in groups {
        let enable = condition(&mask, controls, gb, &mut memo);
        let data = gb.add_or_nary(&pieces, ReductionKind::Tree);
        terms.push(gb.add_and_binary(enable, data));
    }
    gb.add_or_nary(&terms, ReductionKind::Tree)
}

fn condition(
    mask: &[bool],
    controls: &[AigOperand],
    gb: &mut GateBuilder,
    memo: &mut HashMap<Vec<bool>, AigOperand>,
) -> AigOperand {
    if mask.iter().all(|&b| b == mask[0]) {
        return if mask[0] {
            gb.get_true()
        } else {
            gb.get_false()
        };
    }
    if let Some(&known) = memo.get(mask) {
        return known;
    }
    let (last, earlier) = controls
        .split_last()
        .expect("nonconstant truth table has a control");
    let (off, on) = mask.split_at(mask.len() / 2);
    let lo = condition(off, earlier, gb, memo);
    let hi = condition(on, earlier, gb, memo);
    let value = if lo == hi {
        lo
    } else {
        gb.add_mux2(*last, hi, lo)
    };
    memo.insert(mask.to_vec(), value);
    value
}

fn emit_named(
    cells: &[Vec<(usize, usize)>; 4],
    inverted: bool,
    data: &AigBitVector,
    [inner, outer]: [AigOperand; 2],
    gb: &mut GateBuilder,
) -> AigOperand {
    let mut cuts: Vec<_> = cells
        .iter()
        .flatten()
        .flat_map(|&(lo, hi)| [lo, hi])
        .collect();
    cuts.sort_unstable();
    cuts.dedup();
    let mut common = Vec::new();
    // For each outer setting, a bit is unconditional or needs one polarity
    // of the inner control. Unobserved bits contribute nothing.
    let mut arm_groups: [[Vec<(usize, usize)>; 3]; 2] = Default::default();
    for ends in cuts.windows(2) {
        let (lo, hi) = (ends[0], ends[1]);
        let on = cells
            .each_ref()
            .map(|cell| cell.iter().any(|&(l, h)| l <= lo && hi <= h));
        if on == [true; 4] {
            append_span(&mut common, lo, hi);
            continue;
        }
        for (groups, pair) in arm_groups.iter_mut().zip(on.chunks_exact(2)) {
            let group = match pair {
                [true, true] => 0,
                [true, false] => 1,
                [false, true] => 2,
                _ => continue,
            };
            append_span(&mut groups[group], lo, hi);
        }
    }
    let arms = arm_groups.map(|[shared, when_zero, when_one]| {
        let mut terms = Vec::new();
        if let Some(term) = reduce_spans(data, &shared, gb) {
            terms.push(term);
        }
        for (spans, enable) in [(when_zero, inner.negate()), (when_one, inner)] {
            if let Some(term) = reduce_spans(data, &spans, gb) {
                terms.push(gb.add_and_binary(enable, term));
            }
        }
        if terms.is_empty() {
            gb.get_false()
        } else {
            gb.add_or_nary(&terms, ReductionKind::Tree)
        }
    });
    let selected = gb.add_mux2(outer, arms[1], arms[0]);
    let result = if let Some(always) = reduce_spans(data, &common, gb) {
        gb.add_or_binary(always, selected)
    } else {
        selected
    };
    if inverted { result.negate() } else { result }
}

fn append_span(spans: &mut Vec<(usize, usize)>, lo: usize, hi: usize) {
    if let Some(last) = spans.last_mut()
        && last.1 == lo
    {
        last.1 = hi;
    } else {
        spans.push((lo, hi));
    }
}

fn reduce_spans(
    data: &AigBitVector,
    spans: &[(usize, usize)],
    gb: &mut GateBuilder,
) -> Option<AigOperand> {
    let mut terms = Vec::with_capacity(spans.len());
    for &(lo, hi) in spans {
        terms.push(gb.add_or_reduce(&data.get_lsb_slice(lo, hi - lo), ReductionKind::Tree));
    }
    (!terms.is_empty()).then(|| gb.add_or_nary(&terms, ReductionKind::Tree))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig_sim::gate_sim::{self, Collect};
    use crate::gate_builder::GateBuilderOptions;
    use xlsynth_pir::IrBits;

    #[test]
    fn cofactored_reductions_preserve_gaps_and_control_order() {
        let cells = [vec![(0, 2), (4, 5)], vec![(1, 3)], vec![], vec![(3, 5)]];
        let mut gb = GateBuilder::new(
            "cofactored".into(),
            GateBuilderOptions {
                hash: true,
                fold: true,
            },
        );
        let data = gb.add_input("x".into(), 5);
        let controls = gb.add_input("c".into(), 2);
        let result = emit_named(
            &cells,
            true,
            &data,
            [*controls.get_lsb(0), *controls.get_lsb(1)],
            &mut gb,
        );
        gb.add_output("result".into(), result.into());
        let gates = gb.build();
        for x in 0..32 {
            for (c, observed) in [0b10011, 0b00110, 0, 0b11000].into_iter().enumerate() {
                let args = [
                    IrBits::make_ubits(5, x).unwrap(),
                    IrBits::make_ubits(2, c as u64).unwrap(),
                ];
                assert_eq!(
                    gate_sim::eval(&gates, &args, Collect::None).outputs[0]
                        .get_bit(0)
                        .unwrap(),
                    x & observed == 0
                );
            }
        }
    }

    #[test]
    fn shared_segments_preserve_multiple_data_words_and_forced_cells() {
        use xlsynth_pir::ir::NodeRef;

        let a = NodeRef { index: 7 };
        let b = NodeRef { index: 19 };
        let mut gb = GateBuilder::new(
            "guarded".into(),
            GateBuilderOptions {
                hash: true,
                fold: true,
            },
        );
        let x = gb.add_input("x".into(), 3);
        let y = gb.add_input("y".into(), 2);
        let control = gb.add_input("control".into(), 2);
        let cells = [
            Cell::Const(false),
            Cell::Any(vec![(a, 0, 2), (b, 0, 1)]),
            Cell::Any(vec![(a, 1, 3), (b, 0, 1)]),
            Cell::Const(true),
        ];
        let bounds = BTreeMap::from([(a.index, vec![0, 1, 2, 3]), (b.index, vec![0, 1])]);
        let words = BTreeMap::from([(a.index, x), (b.index, y)]);
        let result = emit_shared(
            &cells,
            &bounds,
            &words,
            &[*control.get_lsb(0), *control.get_lsb(1)],
            &mut gb,
        );
        gb.add_output("result".into(), result.into());
        let gates = gb.build();
        for x in 0..8 {
            for y in 0..4 {
                for c in 0..4 {
                    let expected = match c {
                        0 => false,
                        1 => (x & 3) != 0 || (y & 1) != 0,
                        2 => (x & 6) != 0 || (y & 1) != 0,
                        _ => true,
                    };
                    let args = [
                        IrBits::make_ubits(3, x).unwrap(),
                        IrBits::make_ubits(2, y).unwrap(),
                        IrBits::make_ubits(2, c).unwrap(),
                    ];
                    assert_eq!(
                        gate_sim::eval(&gates, &args, Collect::None).outputs[0]
                            .get_bit(0)
                            .unwrap(),
                        expected
                    );
                }
            }
        }
    }
}
