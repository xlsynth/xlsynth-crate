// SPDX-License-Identifier: Apache-2.0
//! Shares a row-local conjunction between different predicates on an indexed
//! value.
use super::word_observations::{Cube, Pin, WordObservations};
use super::*;
use std::collections::HashMap;
use xlsynth_pir::ir::{NodePayload as Op, NodeRef, Type};
use xlsynth_pir::ir_utils::Users;

struct Region {
    // Slot order is part of the table, independent of where its results were
    // computed.
    tests: Vec<Cube>,
    outputs: Vec<(NodeRef, usize, bool)>,
    common: Vec<Pin>,
}

impl Region {
    fn find(f: &ir::Fn, read: NodeRef, users: &Users) -> Option<Self> {
        let observations = WordObservations::find(f, read, users)?;
        if !observations.escapes.is_empty() {
            return None;
        }
        let outputs = observations
            .scalars
            .into_iter()
            .map(|s| (s.value, s.cube))
            .collect::<Vec<_>>();
        // Two complementary consumers of the same predicate need only one
        // comparison.
        let mut tests = outputs
            .iter()
            .map(|(_, c)| Cube {
                pins: c.pins.clone(),
                inverted: false,
            })
            .collect::<Vec<_>>();
        tests.sort_unstable();
        tests.dedup();
        if tests.len() < 2 {
            return None;
        }
        let mut common = tests[0].pins.clone();
        common.retain(|pin| tests.iter().all(|c| c.pins.binary_search(pin).is_ok()));
        // Sharing one wire does not eliminate a row-local logic operation.
        // Keep the work for all factored row terms within one complete scan
        // of the demanded input bits. Repeated tests share their table slot.
        let unique_bits: std::collections::BTreeSet<_> = tests
            .iter()
            .flat_map(|c| c.pins.iter().map(|p| p.0))
            .collect();
        let pin_visits = common.len()
            + tests
                .iter()
                .map(|c| c.pins.len() - common.len())
                .sum::<usize>();
        if tests.len() >= unique_bits.len() || pin_visits > unique_bits.len() || common.len() < 2 {
            return None;
        }
        let mut outputs = outputs
            .into_iter()
            .map(|(n, c)| {
                let inverted = c.inverted;
                let slot = tests
                    .binary_search(&Cube {
                        pins: c.pins,
                        inverted: false,
                    })
                    .unwrap();
                (n, slot, inverted)
            })
            .collect::<Vec<_>>();
        outputs.sort_unstable_by_key(|(n, _, _)| n.index);
        Some(Self {
            tests,
            outputs,
            common,
        })
    }
}

#[derive(Default)]
pub(super) struct ReadPredicates {
    users: Option<Users>,
    tables: HashMap<(NodeRef, usize, Vec<Cube>), AigBitVector>,
    results: HashMap<NodeRef, AigOperand>,
}

impl ReadPredicates {
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
        if indices.is_empty()
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
        if dims
            .iter()
            .all(|&(_, i, _)| matches!(f.get_node(i).payload, Op::Literal(_)))
        {
            return;
        }
        dims.reverse();
        let elem_width = node.ty.bit_count();
        let key = (source, elem_width, region.tests.clone());
        let table = self.tables.entry(key).or_insert_with(|| {
            let data = env.get_bit_vector(source).expect("array source is lowered");
            let mut table = Vec::new();
            for base in (0..data.get_bit_count()).step_by(elem_width) {
                let pin = |(bit, high): Pin| {
                    let rail = *data.get_lsb(base + bit);
                    if high { rail } else { rail.negate() }
                };
                let shared = gb.add_and_nary(
                    &region.common.iter().copied().map(pin).collect::<Vec<_>>(),
                    ReductionKind::Tree,
                );
                for c in &region.tests {
                    let mut terms = vec![shared];
                    terms.extend(
                        c.pins
                            .iter()
                            .filter(|p| region.common.binary_search(p).is_err())
                            .copied()
                            .map(pin),
                    );
                    let hit = if terms.len() == 1 {
                        shared
                    } else {
                        gb.add_and_nary(&terms, ReductionKind::Tree)
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
            }
            AigBitVector::from_lsb_is_index_0(&table)
        });
        let mut ty = Type::Bits(region.tests.len());
        for &(size, _, _) in dims.iter().rev() {
            ty = Type::Array(ir::ArrayTypeData {
                element_count: size,
                element_type: Box::new(ty),
            });
        }
        let mut bits = table.clone();
        for (_, index, assumed) in dims {
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
