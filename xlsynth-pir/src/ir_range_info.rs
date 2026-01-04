// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::sync::Arc;

use xlsynth::{Interval, IrAnalysis, IrBits, KnownBits};

use crate::ir;

pub struct NodeRangeInfo {
    pub known_bits: Option<KnownBits>,
    pub intervals: Option<Vec<Interval>>,
    pub unsigned_min: Option<IrBits>,
    pub unsigned_max: Option<IrBits>,
}

pub struct IrRangeInfo {
    by_text_id: BTreeMap<usize, NodeRangeInfo>,
}

/// Returns `(min, max)` unsigned bounds from an interval set, if any exist.
fn unsigned_bounds_from_intervals(intervals: &[Interval]) -> (Option<IrBits>, Option<IrBits>) {
    let mut min_bits: Option<IrBits> = None;
    let mut max_bits: Option<IrBits> = None;
    for it in intervals {
        if min_bits.as_ref().is_none_or(|cur_min| it.lo.ult(cur_min)) {
            min_bits = Some(it.lo.clone());
        }
        if max_bits.as_ref().is_none_or(|cur_max| it.hi.ugt(cur_max)) {
            max_bits = Some(it.hi.clone());
        }
    }
    (min_bits, max_bits)
}

/// Returns the effective unsigned bit-length of `bits` (0 for zero).
fn bit_len_unsigned(bits: &IrBits) -> usize {
    let w = bits.get_bit_count();
    if w == 0 {
        return 0;
    }
    for i in (0..w).rev() {
        if bits.get_bit(i).unwrap() {
            return i + 1;
        }
    }
    0
}

impl IrRangeInfo {
    /// Looks up all recorded range information for a given IR `text_id`.
    pub fn get(&self, text_id: usize) -> Option<&NodeRangeInfo> {
        self.by_text_id.get(&text_id)
    }

    /// Returns the maximum unsigned value for `text_id`, if interval-derived
    /// bounds exist.
    pub fn unsigned_max(&self, text_id: usize) -> Option<&IrBits> {
        self.get(text_id).and_then(|n| n.unsigned_max.as_ref())
    }

    /// Returns true iff analysis proves the node value is always unsigned-`<
    /// bound` (where `bound` is a structural size like array length/bit-width,
    /// not an IR value).
    pub fn proves_ult(&self, text_id: usize, bound: usize) -> bool {
        if bound == 0 {
            return false;
        }
        let max_bits = match self.unsigned_max(text_id) {
            Some(v) => v,
            None => return false,
        };
        let w = max_bits.get_bit_count();
        if w < 64 {
            let limit = 1u64 << w;
            if (bound as u64) >= limit {
                return true;
            }
        }
        let bound_bits = IrBits::make_ubits(w, bound as u64).unwrap();
        max_bits.ult(&bound_bits)
    }

    /// Returns an upper bound on the needed shift-amount bitwidth given a
    /// proven `ult` bound.
    pub fn effective_amount_bits_for_ult(&self, text_id: usize, bound: usize) -> Option<usize> {
        if !self.proves_ult(text_id, bound) {
            return None;
        }
        let max_bits = self.unsigned_max(text_id)?;
        Some(bit_len_unsigned(max_bits))
    }

    /// Returns true iff analysis proves the node value is never equal to zero.
    pub fn proves_nonzero(&self, text_id: usize) -> bool {
        // Prefer interval reasoning when available: prove `0` is not in the interval
        // set.
        if let Some(node) = self.get(text_id) {
            if let Some(intervals) = node.intervals.as_ref() {
                // Zero-width values are always "zero", so we cannot prove nonzero.
                let w = node
                    .unsigned_min
                    .as_ref()
                    .or(node.unsigned_max.as_ref())
                    .map(|b| b.get_bit_count())
                    .unwrap_or(0);
                if w == 0 {
                    return false;
                }
                let zero = IrBits::make_ubits(w, 0).unwrap();
                let zero_is_possible = intervals
                    .iter()
                    .any(|it| it.lo.ule(&zero) && zero.ule(&it.hi));
                return !zero_is_possible;
            }

            // Fall back to known-bits: any provably-one bit implies nonzero.
            if let Some(k) = node.known_bits.as_ref() {
                let w = k.mask.get_bit_count();
                for i in 0..w {
                    let is_known = k.mask.get_bit(i).unwrap_or(false);
                    if !is_known {
                        continue;
                    }
                    let is_one = k.value.get_bit(i).unwrap_or(false);
                    if is_one {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Builds `IrRangeInfo` for a PIR function by querying libxls analysis for
    /// every node `text_id`.
    pub fn build_from_analysis(analysis: &IrAnalysis, f: &ir::Fn) -> Result<Arc<Self>, String> {
        let mut by_text_id: BTreeMap<usize, NodeRangeInfo> = BTreeMap::new();

        for node in &f.nodes {
            let text_id = node.text_id;

            let is_bits_typed = matches!(node.ty, ir::Type::Bits(_));
            if !is_bits_typed {
                by_text_id.insert(
                    text_id,
                    NodeRangeInfo {
                        known_bits: None,
                        intervals: None,
                        unsigned_min: None,
                        unsigned_max: None,
                    },
                );
                continue;
            }

            let node_id_i64: i64 = i64::try_from(text_id)
                .map_err(|_| format!("node text_id {} does not fit in i64", text_id))?;

            let known_bits = analysis
                .get_known_bits_for_node_id(node_id_i64)
                .map(Some)
                .map_err(|e| format!("known-bits query failed for node id={}: {}", text_id, e))?;

            let interval_set = analysis
                .get_intervals_for_node_id(node_id_i64)
                .map_err(|e| format!("interval query failed for node id={}: {}", text_id, e))?;
            let intervals = interval_set.intervals().map_err(|e| {
                format!("interval enumeration failed for node id={}: {}", text_id, e)
            })?;

            let (unsigned_min, unsigned_max) = unsigned_bounds_from_intervals(&intervals);

            by_text_id.insert(
                text_id,
                NodeRangeInfo {
                    known_bits,
                    intervals: Some(intervals),
                    unsigned_min,
                    unsigned_max,
                },
            );
        }

        Ok(Arc::new(Self { by_text_id }))
    }
}
