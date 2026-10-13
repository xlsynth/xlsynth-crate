// SPDX-License-Identifier: Apache-2.0

//! Shared unresolved-row planning for products and canonical additive clusters.
//!
//! Extraction follows n-ary formation and node compaction, using the final
//! prepared snapshot's references and use counts. Existing n-ary formation
//! handles add/sub/negate chains; this read-only plan extracts product leaves.
//! Shared products and narrow products widened by the sum stay resolved.
//! Once emitted, product rows and additive rows use the same arrival policy.
//! Product operands themselves remain ordinary bit vectors: this does not
//! distribute multiplication over nested arithmetic or absorb bit slices.

use super::{GateEnv, gatify_add_with_mapping};
use crate::aig::gate::{AigBitVector, AigOperand};
use crate::gate_builder::GateBuilder;
use crate::ir2gate_utils::{AdderMapping, array_add_with_carry_out};
use std::collections::HashSet;
use xlsynth_pir::IrBits;
use xlsynth_pir::ir::{self, Binop, NodePayload, NodeRef};

/// Products whose sole consumer can accumulate their rows modulo its width.
#[derive(Default)]
pub(super) struct ArithmeticClusterPlan {
    products: HashSet<NodeRef>,
}

impl ArithmeticClusterPlan {
    /// Extracts from the final prepared snapshot using its cached use counts.
    pub(super) fn extract(f: &ir::Fn, env: &GateEnv) -> Self {
        let mut products = HashSet::new();
        for node in &f.nodes {
            let NodePayload::ExtNaryAdd { terms, .. } = &node.payload else {
                continue;
            };
            let output_width = node.ty.bit_count();
            if output_width == 0 {
                continue;
            }
            for term in terms {
                let product = f.get_node(term.operand);
                if matches!(
                    product.payload,
                    NodePayload::Binop(Binop::Umul | Binop::Smul, _, _)
                ) && env.has_single_use(term.operand)
                    && product.ty.bit_count() >= output_width
                {
                    // Truncating a product commutes with adding modulo this
                    // width. Widening an already-truncated product does not.
                    products.insert(term.operand);
                }
            }
        }
        Self { products }
    }

    pub(super) fn contains_product(&self, product: NodeRef) -> bool {
        self.products.contains(&product)
    }
}

/// A modular arithmetic contribution before its carry-propagation boundary.
/// Each vector contributes at its natural bit weights; repeated bits retain
/// arithmetic multiplicity. Constants/carry-in keep their existing fast paths.
struct ArithmeticRows {
    rows: Vec<AigBitVector>,
    latest_arrival: usize,
}

fn row_arrival(gb: &GateBuilder, row: &AigBitVector) -> usize {
    row.iter_lsb_to_msb()
        .filter_map(|bit| gb.aig_depth(*bit))
        .max()
        .unwrap_or(0)
}

fn estimated_resolution_depth(width: usize, mut height: usize, mapping: AdderMapping) -> usize {
    let mut compression_depth = 0usize;
    while height > 2 {
        height = 2 * (height / 3) + height % 3;
        compression_depth = compression_depth.saturating_add(4);
    }
    let cpa_depth = match mapping {
        AdderMapping::RippleCarry => width.saturating_mul(3),
        AdderMapping::BrentKung | AdderMapping::KoggeStone => {
            xlsynth_pir::math::ceil_log2(width).saturating_mul(4)
        }
    };
    compression_depth.saturating_add(cpa_depth)
}

/// Groups unresolved rows by operand availability, regardless of their IR
/// origin. A prefix that can finish before the next row arrives is resolved
/// once; otherwise rows share a compressor and final CPA. No alternate mapping
/// is constructed. Cached AIG depth is a timing proxy, not a graph-LE bound.
pub(in crate::gatify) fn gatify_arithmetic_cluster(
    gb: &mut GateBuilder,
    rows: Vec<AigBitVector>,
    literal: &IrBits,
    carry_in: Option<AigOperand>,
    adder_mapping: AdderMapping,
) -> AigBitVector {
    let width = literal.get_bit_count();
    let input_rows = rows.len();
    if input_rows == 2 && literal.is_zero() {
        // A binary sum needs no compressor tree.
        // Avoid row scans and sorting when there is no compression choice.
        return array_add_with_carry_out(gb, &rows, carry_in, adder_mapping).sum;
    }
    // Fold complete constant rows before arrival grouping. In particular,
    // complemented CSD rows and their modular sign corrections must travel
    // together through an early resolution boundary.
    let mut constant = literal.clone();
    let mut dynamic_rows = Vec::with_capacity(input_rows + 1);
    for row in rows {
        let bits: Option<Vec<bool>> = row
            .iter_lsb_to_msb()
            .map(|bit| {
                if gb.is_known_true(*bit) {
                    Some(true)
                } else if gb.is_known_false(*bit) {
                    Some(false)
                } else {
                    None
                }
            })
            .collect();
        if let Some(bits) = bits {
            constant = constant.add(&IrBits::from_lsb_is_0(&bits));
        } else {
            dynamic_rows.push(row);
        }
    }
    if !constant.is_zero() {
        dynamic_rows.push(gb.add_literal(&constant));
    }
    let mut ordered: Vec<_> = dynamic_rows
        .into_iter()
        .enumerate()
        .map(|(index, row)| (row_arrival(gb, &row), index, row))
        .collect();
    ordered.sort_by_key(|(arrival, index, _)| (*arrival, *index));
    let mut group = ArithmeticRows {
        rows: Vec::new(),
        latest_arrival: 0,
    };
    let mut boundaries = 0;
    for (arrival, _, row) in ordered {
        if group.rows.len() >= 2 {
            let predicted = group
                .latest_arrival
                .saturating_add(estimated_resolution_depth(
                    width,
                    group.rows.len(),
                    adder_mapping,
                ));
            if predicted <= arrival {
                log::debug!(
                    "arithmetic row boundary: width={width} rows={} latest={} predicted={predicted} next={arrival}",
                    group.rows.len(),
                    group.latest_arrival
                );
                let resolved = array_add_with_carry_out(
                    gb,
                    &std::mem::take(&mut group.rows),
                    None,
                    adder_mapping,
                )
                .sum;
                group.latest_arrival = row_arrival(gb, &resolved);
                group.rows.push(resolved);
                boundaries += 1;
            }
        }
        group.latest_arrival = group.latest_arrival.max(arrival);
        group.rows.push(row);
    }
    log::debug!(
        "arithmetic row plan: width={width} input_rows={input_rows} remaining_rows={} boundaries={boundaries}",
        group.rows.len()
    );
    compress_arithmetic_rows(
        gb,
        group.rows,
        &IrBits::zero(width),
        carry_in,
        adder_mapping,
    )
}

/// Compresses populated bit weights, then emits exactly one CPA.
fn compress_arithmetic_rows(
    gb: &mut GateBuilder,
    rows: Vec<AigBitVector>,
    literal: &IrBits,
    carry_in: Option<AigOperand>,
    adder_mapping: AdderMapping,
) -> AigBitVector {
    let width = literal.get_bit_count();
    let mut columns = vec![Vec::new(); width];
    let mut constants = vec![0usize; width];
    for row in rows {
        assert_eq!(row.get_bit_count(), width);
        for (i, bit) in row.iter_lsb_to_msb().copied().enumerate() {
            if gb.is_known_true(bit) {
                constants[i] += 1;
            } else if !gb.is_known_false(bit) {
                // Keep repeated bits: arithmetic multiplicity is significant.
                columns[i].push(bit);
            }
        }
    }
    let mut constant_carry = 0usize;
    for i in 0..width {
        constant_carry += constants[i] + usize::from(literal.get_bit(i).unwrap());
        if constant_carry & 1 != 0 {
            columns[i].push(gb.get_true());
        }
        constant_carry >>= 1;
    }
    // A repeated bit contributes twice its weight. Pair it into the next
    // column, retaining one copy for odd multiplicities; never Boolean-dedup
    // arithmetic inputs. Hash sets are used only for membership, preserving
    // deterministic insertion order in the surviving columns.
    for i in 0..width {
        let mut odd = HashSet::new();
        let mut carries = Vec::new();
        for bit in &columns[i] {
            if !odd.insert(*bit) {
                odd.remove(bit);
                carries.push(*bit);
            }
        }
        columns[i].retain(|bit| odd.remove(bit));
        if i + 1 < width {
            columns[i + 1].extend(carries);
        }
    }
    let initial_height = columns.iter().map(Vec::len).max().unwrap_or(0);
    let mut targets = vec![2usize];
    while *targets.last().unwrap() < initial_height {
        targets.push(targets.last().unwrap() * 3 / 2);
    }
    targets.pop();
    let stage_count = targets.len();
    let mut full_adders = 0usize;
    let mut half_adders = 0usize;
    for target in targets.into_iter().rev() {
        let mut next = vec![Vec::new(); width];
        for (i, column) in columns.iter_mut().enumerate() {
            // Cached builder depths order early inputs first. Carries already
            // in `next[i]` count toward the target but cannot be consumed in
            // this stage. Half-adders prevent a height-three carry tail.
            column.sort_by_key(|bit| gb.aig_depth(*bit).unwrap_or(0));
            let mut used = 0usize;
            while next[i].len() + column.len() - used > target {
                let (sum, carry) = if column.len() - used >= 3 {
                    let full = gb.add_full_adder(column[used], column[used + 1], column[used + 2]);
                    used += 3;
                    full_adders += 1;
                    (full.sum, full.carry)
                } else {
                    assert!(
                        column.len() - used >= 2,
                        "Dadda target must use only stage inputs"
                    );
                    let half = gb.add_half_adder(column[used], column[used + 1]);
                    used += 2;
                    half_adders += 1;
                    (half.sum, half.carry)
                };
                if !gb.is_known_false(sum) {
                    next[i].push(sum);
                }
                if i + 1 < width && !gb.is_known_false(carry) {
                    next[i + 1].push(carry);
                }
            }
            next[i].extend_from_slice(&column[used..]);
            assert!(next[i].len() <= target);
        }
        columns = next;
    }
    log::debug!(
        "arithmetic cluster: width={width} initial_height={initial_height} stages={stage_count} full_adders={full_adders} half_adders={half_adders}"
    );
    let a = AigBitVector::from_lsb_is_index_0(
        &columns
            .iter()
            .map(|c| c.first().copied().unwrap_or_else(|| gb.get_false()))
            .collect::<Vec<_>>(),
    );
    let b = AigBitVector::from_lsb_is_index_0(
        &columns
            .iter()
            .map(|c| c.get(1).copied().unwrap_or_else(|| gb.get_false()))
            .collect::<Vec<_>>(),
    );
    gatify_add_with_mapping(
        adder_mapping,
        &a,
        &b,
        carry_in.unwrap_or_else(|| gb.get_false()),
        None,
        gb,
    )
    .1
}
