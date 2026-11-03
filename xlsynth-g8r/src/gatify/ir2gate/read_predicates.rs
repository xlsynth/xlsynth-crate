// SPDX-License-Identifier: Apache-2.0
//! Shares a row-local conjunction between different predicates on an indexed
//! value.
use super::word_observations::{Cube, Pin, WordObservations};
use super::*;
use crate::aig::{gate::AigRef, topo::postorder_for_aig_refs_node_only};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet, hash_map::Entry};
use xlsynth_pir::ir::{NodePayload as Op, NodeRef, Type};
use xlsynth_pir::ir_utils::Users;

/// Predicate classes that share row logic and the values bound to their
/// results.
struct Region {
    // Slot order is part of the table, independent of where its results were
    // computed.
    tests: Vec<Cube>,
    outputs: Vec<(NodeRef, usize, bool)>,
    common: Vec<Pin>,
}

/// Word columns and estimated ANDs for applying the predicates after selection.
struct SelectionCost {
    columns: Vec<usize>,
    row_ands: usize,
}

impl Region {
    fn find(f: &ir::Fn, read: NodeRef, users: &Users) -> Option<Self> {
        let observations = WordObservations::find(f, read, users, |_| true)?;
        if !observations.escapes.is_empty() {
            return None;
        }
        let mut tests: Vec<_> = observations
            .boundary_scalars()
            .map(|s| Cube {
                pins: s.cube.pins.clone(),
                inverted: false,
            })
            .collect();
        // Two complementary consumers need only one comparison.
        tests.sort_unstable();
        tests.dedup();
        if tests.len() < 2 {
            return None;
        }
        let mut common = tests[0].pins.to_vec();
        common.retain(|pin| tests.iter().all(|c| c.pins.binary_search(pin).is_ok()));
        // Sharing one wire does not eliminate a row-local logic operation.
        if common.len() < 2 {
            return None;
        }
        // Bind the producer as well as every alias of an output class. The
        // producer's tag and the aliases' origins then reach the same gate.
        // A single-pin view cannot match a class with two common pins.
        let outputs = observations
            .scalars
            .iter()
            .filter_map(|s| {
                let slot = tests.binary_search_by(|t| t.pins.cmp(&s.cube.pins)).ok()?;
                Some((s.value, slot, s.cube.inverted))
            })
            .collect();
        Some(Self {
            tests,
            outputs,
            common,
        })
    }

    /// Compares replicated predicate ANDs with the selection columns saved
    /// across the dynamic dimensions, before folding and structural hashing.
    fn selection_cost(&self, dynamic_sizes: impl Iterator<Item = usize>) -> Option<SelectionCost> {
        let columns = self
            .tests
            .iter()
            .flat_map(|c| c.pins.iter().map(|p| p.0))
            .collect::<std::collections::BTreeSet<_>>();
        let unique_bits = columns.len();
        // Reducing the common pins takes C-1 ANDs. Each remaining pin needs
        // one more AND to include it in a conjunction with the shared term.
        let residual_pins: usize = self
            .tests
            .iter()
            .map(|c| c.pins.len() - self.common.len())
            .sum();
        if self.tests.len() >= unique_bits || self.common.len() + residual_pins > unique_bits {
            return None;
        }
        let row_ands = self.common.len() - 1 + residual_pins;
        let mut rows = 1;
        let mut selection_ands = 0;
        for size in dynamic_sizes {
            // Per result bit a one-hot read masks N rows and ORs their N
            // terms. Each row of this dimension needs the preceding selects.
            rows *= size;
            selection_ands = size * selection_ands + 2 * size - 1;
        }
        ((rows - 1) * row_ands <= selection_ands * (unique_bits - self.tests.len())).then(|| {
            SelectionCost {
                columns: columns.into_iter().collect(),
                row_ands,
            }
        })
    }
}

/// Identifies the source records and predicates of an address-independent
/// table.
#[derive(Hash, PartialEq, Eq)]
struct TableKey {
    source: NodeRef,
    first_bit: usize,
    // (element count, stride in source bits), innermost dynamic dimension
    // first.
    axes: Vec<(usize, usize)>,
    tests: Vec<Cube>,
}

/// Visits just the addressed records in flattened array order.
fn visit_rows(axes: &[(usize, usize)], base: usize, emit: &mut impl FnMut(usize)) {
    if let Some((&(count, stride), inner)) = axes.split_last() {
        for i in 0..count {
            visit_rows(inner, base + i * stride, emit);
        }
    } else {
        emit(base);
    }
}

/// Builds shared row tables and tracks which reads may benefit from ordinary
/// lowering.
#[derive(Default)]
pub(super) struct ReadPredicates {
    users: Option<Users>,
    tables: HashMap<TableKey, PredicateTable>,
    results: HashMap<NodeRef, AigOperand>,
    factored_reads: Vec<FactoredRead>,
    keep_ordinary: HashSet<NodeRef>,
}

/// The boundary rails of one rewrite in the completed factored mapping.
struct FactoredRead {
    read: NodeRef,
    cost: SelectionCost,
    predicates: Vec<AigRef>,
}

/// Cached row results and, when tracking origins, the gates implementing them.
struct PredicateTable {
    bits: AigBitVector,
    predicate_gates: HashSet<AigRef>,
}

impl PredicateTable {
    /// Builds each row from its source pins and records the reduction gates
    /// to which later uses of this table must contribute provenance.
    fn build(
        key: &TableKey,
        data: &AigBitVector,
        common: &[Pin],
        columns: &[usize],
        gb: &mut GateBuilder,
        track_origins: bool,
    ) -> Self {
        let mut table = Vec::new();
        let mut predicate_gates = HashSet::new();
        visit_rows(&key.axes, key.first_bit, &mut |base| {
            let start = table.len();
            let inputs = track_origins.then(|| {
                RefCell::new(
                    columns
                        .iter()
                        .map(|&bit| (data.get_lsb(base + bit).node, ()))
                        .collect::<HashMap<_, _>>(),
                )
            });
            let mut reduce = |terms: &[AigOperand]| {
                let Some(inputs) = &inputs else {
                    return gb.add_and_nary(terms, ReductionKind::Tree);
                };
                gb.add_and_nary_with(terms, ReductionKind::Tree, &|gb, lhs, rhs| {
                    let result = gb.add_and_binary(lhs, rhs);
                    // A row input can also implement a pair in the reduction.
                    // On such a hash hit the provenance walk must cross it.
                    if let AigNode::And2 { a, b, .. } = gb.gates[result.node.id]
                        && ((a == lhs && b == rhs) || (a == rhs && b == lhs))
                    {
                        inputs.borrow_mut().remove(&result.node);
                    }
                    result
                })
            };
            let pin = |(bit, high): Pin| {
                let rail = *data.get_lsb(base + bit);
                if high { rail } else { rail.negate() }
            };
            let shared = reduce(&common.iter().copied().map(pin).collect::<Vec<_>>());
            for c in &key.tests {
                let mut terms = vec![shared];
                terms.extend(
                    c.pins
                        .iter()
                        .filter(|p| common.binary_search(p).is_err())
                        .copied()
                        .map(pin),
                );
                let hit = if terms.len() == 1 {
                    shared
                } else {
                    reduce(&terms)
                };
                // On an invalid exact address the ordinary decoder returns
                // zero at that dimension. Encode each class so its zero row
                // is zero.
                table.push(if c.pins.iter().all(|(_, high)| !high) {
                    hit.negate()
                } else {
                    hit
                });
            }
            if let Some(inputs) = inputs {
                // Attribute only reachable reductions, stopping at data that
                // was used solely as input to this row.
                let roots: Vec<_> = table[start..]
                    .iter()
                    .map(|bit| bit.node)
                    .chain([shared.node])
                    .collect();
                predicate_gates.extend(postorder_for_aig_refs_node_only(
                    &roots,
                    &gb.gates,
                    &inputs.into_inner(),
                ));
                predicate_gates.extend(roots);
            }
        });
        PredicateTable {
            bits: AigBitVector::from_lsb_is_index_0(&table),
            predicate_gates,
        }
    }
}

impl ReadPredicates {
    /// Recheck the estimate against the completed outputs. Reviving the
    /// ordinary word costs only ANDs that are currently dead, even if an
    /// independent PIR expression selected the same data later in the map.
    pub(super) fn prepare_retry(&mut self, env: &GateEnv, gb: &GateBuilder) -> Option<usize> {
        if self.factored_reads.is_empty() {
            return None;
        }
        let mut live = shift_reduction_predicate::LiveUses::from_outputs(gb);
        for FactoredRead {
            read,
            cost,
            predicates,
        } in self.factored_reads.drain(..)
        {
            let GateOrVec::BitVector(word) = &env.ir_to_g8[&read] else {
                unreachable!("multi-bit read is a bit vector")
            };
            let columns = cost.columns.iter().map(|&i| word.get_lsb(i).node);
            if live.replacing_saves_ands(gb, &predicates, columns, cost.row_ands) {
                self.keep_ordinary.insert(read);
            }
        }
        // Cached gate IDs belong to this builder. Only the PIR user map and
        // the set of reads to keep ordinary can be reused in the trial.
        self.tables.clear();
        debug_assert!(self.results.is_empty());
        (!self.keep_ordinary.is_empty()).then_some(live.and_nodes)
    }

    pub(super) fn take(&mut self, n: NodeRef) -> Option<AigOperand> {
        self.results.remove(&n)
    }

    pub(super) fn consider(
        &mut self,
        f: &ir::Fn,
        read: NodeRef,
        gb: &mut GateBuilder,
        env: &GateEnv,
        options: &GatifyOptions,
    ) {
        let node = f.get_node(read);
        let Op::ArrayIndex { array, indices, .. } = &node.payload else {
            return;
        };
        if self.keep_ordinary.contains(&read)
            || indices.is_empty()
            || matches!(node.ty, Type::Array(_))
            || node.ty.bit_count() < 2
            || (!matches!(f.get_node(*array).payload, Op::ArrayIndex { .. })
                && indices
                    .iter()
                    .all(|&i| matches!(f.get_node(i).payload, Op::Literal(_))))
        {
            return;
        }
        let users = self.users.get_or_insert_with(|| ir_utils::compute_users(f));
        let Some(region) = Region::find(f, read, users) else {
            return;
        };
        // Capture every addressed dimension. A shared intermediate can keep a
        // wide read live even when moving the final scalar comparisons.
        let mut source = read;
        let mut dims = Vec::new();
        while let Op::ArrayIndex {
            array,
            indices,
            assumed_in_bounds,
        } = &f.get_node(source).payload
        {
            if source != read && (!env.has_single_use(source) || Some(source) == f.ret_node_ref) {
                return;
            }
            let mut ty = f.get_node_ty(*array);
            let mut group = Vec::new();
            for &index in indices {
                let Type::Array(at) = ty else {
                    return;
                };
                if at.element_count == 0 {
                    return;
                }
                group.push((at.element_count, index, *assumed_in_bounds));
                ty = &at.element_type;
            }
            group.reverse();
            dims.extend(group);
            source = *array;
        }
        let mut first_bit = 0;
        let mut stride = node.ty.bit_count();
        let mut axes = Vec::new();
        let mut dynamic_dims = Vec::new();
        for (size, index, assumed) in dims {
            if let Some(literal) = literal_bits_if_bits_node(f, index) {
                let proven = options
                    .range_info
                    .as_ref()
                    .is_some_and(|ri| ri.proves_ult(f.get_node(index).text_id, size));
                let Some(position) = literal_array_position(size, &literal, assumed || proven)
                else {
                    // Ordinary lowering already supplies a constant zero
                    // record.
                    return;
                };
                first_bit += position * stride;
            } else {
                // Truncate the table, but step past the full source dimension
                // when computing the stride of the enclosing axis.
                let width = f.get_node_ty(index).bit_count();
                let count = if width < usize::BITS as usize {
                    size.min(1 << width)
                } else {
                    size
                };
                axes.push((count, stride));
                dynamic_dims.push((count, index, assumed));
            }
            stride *= size;
        }
        if axes.is_empty() {
            return;
        }
        let Some(cost) = region.selection_cost(axes.iter().rev().map(|&(size, _)| size)) else {
            return;
        };
        let key = TableKey {
            source,
            first_bit,
            axes,
            tests: region.tests.clone(),
        };
        let table = match self.tables.entry(key) {
            Entry::Occupied(entry) => {
                let table = entry.into_mut();
                for &gate in &table.predicate_gates {
                    gb.union_current_pir_node_id_into_operand(gate.into());
                }
                table
            }
            Entry::Vacant(entry) => {
                let GateOrVec::BitVector(data) = &env.ir_to_g8[&source] else {
                    unreachable!("array of multi-bit records is a bit vector")
                };
                let table = PredicateTable::build(
                    entry.key(),
                    data,
                    &region.common,
                    &cost.columns,
                    gb,
                    options.track_pir_node_ids,
                );
                entry.insert(table)
            }
        };
        let mut ty = Type::Bits(region.tests.len());
        for &(size, _, _) in &dynamic_dims {
            ty = Type::Array(ir::ArrayTypeData {
                element_count: size,
                element_type: Box::new(ty),
            });
        }
        let mut bits = table.bits.clone();
        for (_, index, assumed) in dynamic_dims.into_iter().rev() {
            let Type::Array(at) = ty else { unreachable!() };
            bits = gatify_array_index_dimensions(
                f,
                gb,
                env,
                &at,
                &bits,
                &[index],
                assumed,
                options.range_info.as_deref(),
                options.array_index_lowering_strategy,
            );
            ty = *at.element_type;
        }
        let mut predicates: Vec<_> = bits.iter_lsb_to_msb().map(|b| b.node).collect();
        predicates.sort_unstable_by(|a, b| b.cmp(a));
        predicates.dedup();
        self.factored_reads.push(FactoredRead {
            read,
            cost,
            predicates,
        });
        for (n, slot, inverted) in region.outputs {
            let rail = *bits.get_lsb(slot);
            let zero = region.tests[slot].pins.iter().all(|(_, high)| !high);
            self.results
                .insert(n, if inverted ^ zero { rail.negate() } else { rail });
        }
        log::debug!(
            "shared read predicates: read={} classes={} shared_pins={}",
            node.text_id,
            region.tests.len(),
            region.common.len()
        );
    }
}

#[cfg(test)]
mod tests;
