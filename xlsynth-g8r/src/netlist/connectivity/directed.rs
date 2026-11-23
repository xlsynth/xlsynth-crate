// SPDX-License-Identifier: Apache-2.0

//! Resolve zero-delay directed assignments without merging electrical groups.

use crate::netlist::normalized::{BitSource, NormalizedBitIndex};
use crate::netlist::parse::NetIndex;
use std::collections::{HashMap, hash_map::Entry};
use std::hash::Hash;

/// Wire IDs whose numbering matches the entries in a dense assignment slice.
pub(crate) trait DenseWireIndex: Copy {
    fn from_index(index: usize) -> Self;
    fn index(self) -> usize;
}

impl DenseWireIndex for NetIndex {
    fn from_index(index: usize) -> Self {
        Self(index)
    }

    fn index(self) -> usize {
        self.0
    }
}

impl DenseWireIndex for NormalizedBitIndex {
    fn from_index(index: usize) -> Self {
        Self::new(index)
    }

    fn index(self) -> usize {
        self.index()
    }
}

/// Resolve assignments stored in wire-index order. A missing assignment
/// resolves to the wire itself. On a cycle, return a wire in the cycle.
pub(crate) fn resolve_dense<WireId: DenseWireIndex>(
    assignments: &[Option<BitSource<WireId>>],
) -> Result<Vec<BitSource<WireId>>, WireId> {
    if assignments.iter().all(Option::is_none) {
        return Ok((0..assignments.len())
            .map(|index| BitSource::Bit(WireId::from_index(index)))
            .collect());
    }
    resolve_slots(
        assignments.len(),
        |slot| assignments[slot.0],
        |wire| Some(AssignmentSlot(wire.index())),
        |slot| WireId::from_index(slot.0),
    )
}

/// Assignments for a subset of the wires. Insertion order determines which
/// cycle is reported first.
pub(crate) struct SparseAssignments<WireId> {
    slots: HashMap<WireId, AssignmentSlot>,
    assignments: Vec<(WireId, BitSource<WireId>)>,
}

impl<WireId: Copy + Eq + Hash> SparseAssignments<WireId> {
    pub(crate) fn new() -> Self {
        Self {
            slots: HashMap::new(),
            assignments: Vec::new(),
        }
    }

    /// Record `destination = source`, returning false for a second assignment.
    pub(crate) fn insert(&mut self, destination: WireId, source: BitSource<WireId>) -> bool {
        match self.slots.entry(destination) {
            Entry::Occupied(_) => false,
            Entry::Vacant(entry) => {
                entry.insert(AssignmentSlot(self.assignments.len()));
                self.assignments.push((destination, source));
                true
            }
        }
    }

    pub(crate) fn resolve(self) -> Result<ResolvedAssignments<WireId>, WireId> {
        let sources = resolve_slots(
            self.assignments.len(),
            |slot| Some(self.assignments[slot.0].1),
            |wire| self.slots.get(&wire).copied(),
            |slot| self.assignments[slot.0].0,
        )?;
        Ok(ResolvedAssignments {
            slots: self.slots,
            sources,
        })
    }
}

/// A resolved source for each destination originally assigned.
pub(crate) struct ResolvedAssignments<WireId> {
    slots: HashMap<WireId, AssignmentSlot>,
    sources: Vec<BitSource<WireId>>,
}

impl<WireId: Copy + Eq + Hash> ResolvedAssignments<WireId> {
    pub(crate) fn get(&self, wire: WireId) -> Option<BitSource<WireId>> {
        self.slots.get(&wire).map(|slot| self.sources[slot.0])
    }

    pub(crate) fn sources(&self) -> &[BitSource<WireId>] {
        &self.sources
    }
}

#[derive(Clone, Copy)]
struct AssignmentSlot(usize);

#[derive(Clone, Copy)]
enum Visit<WireId> {
    Unvisited,
    Visiting,
    Resolved(BitSource<WireId>),
}

/// Use a private slot number for visitation state. Each edge is followed once,
/// including when several assigned wires share the same chain.
fn resolve_slots<WireId: Copy>(
    count: usize,
    source_at: impl Fn(AssignmentSlot) -> Option<BitSource<WireId>>,
    slot_for_wire: impl Fn(WireId) -> Option<AssignmentSlot>,
    wire_at: impl Fn(AssignmentSlot) -> WireId,
) -> Result<Vec<BitSource<WireId>>, WireId> {
    let mut states = vec![Visit::Unvisited; count];
    let mut path = Vec::new();
    for start in (0..count).map(AssignmentSlot) {
        if matches!(states[start.0], Visit::Resolved(_)) {
            continue;
        }
        if source_at(start).is_none() {
            states[start.0] = Visit::Resolved(BitSource::Bit(wire_at(start)));
            continue;
        }
        path.clear();
        let mut current = start;
        let terminal = loop {
            match states[current.0] {
                Visit::Resolved(source) => break source,
                Visit::Visiting => return Err(wire_at(current)),
                Visit::Unvisited => {
                    states[current.0] = Visit::Visiting;
                    path.push(current);
                    match source_at(current) {
                        Some(BitSource::Bit(wire)) => {
                            if let Some(next) = slot_for_wire(wire) {
                                current = next;
                            } else {
                                break BitSource::Bit(wire);
                            }
                        }
                        Some(source) => break source,
                        None => break BitSource::Bit(wire_at(current)),
                    }
                }
            }
        };
        for &slot in &path {
            states[slot.0] = Visit::Resolved(terminal);
        }
    }
    Ok(states
        .into_iter()
        .map(|state| match state {
            Visit::Resolved(source) => source,
            Visit::Unvisited | Visit::Visiting => unreachable!("all assignment paths are resolved"),
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Dense numbering preserves wire identity and distinguishes literals,
    /// unknowns, and wires without assignments.
    #[test]
    fn dense_chains_keep_direction_and_terminal_kind() {
        use BitSource::{Bit, Literal, Unknown};
        assert_eq!(
            resolve_dense(&[
                None,
                Some(Bit(NetIndex(0))),
                Some(Bit(NetIndex(1))),
                Some(Literal(true)),
                Some(Bit(NetIndex(3))),
                Some(Unknown),
            ]),
            Ok(vec![
                Bit(NetIndex(0)),
                Bit(NetIndex(0)),
                Bit(NetIndex(0)),
                Literal(true),
                Literal(true),
                Unknown,
            ])
        );
        assert_eq!(
            resolve_dense(&[Some(Bit(NetIndex(1))), Some(Bit(NetIndex(0)))]),
            Err(NetIndex(0))
        );
    }

    /// Sparse wire IDs can differ from slot numbers. Also exercise duplicate
    /// insertion and report a disconnected cycle by its original wire ID.
    #[test]
    fn sparse_forward_chain_and_disconnected_cycle() {
        use BitSource::Bit;
        let bit = NormalizedBitIndex::new;
        let start = 100;
        let end = start + 32_768;
        let root = bit(1_000_000);
        let make_chain = || {
            let mut assignments = SparseAssignments::new();
            for n in start..end {
                assert!(assignments.insert(bit(n), Bit(bit(n + 1))));
            }
            assert!(assignments.insert(bit(end), Bit(root)));
            assert!(!assignments.insert(bit(end), BitSource::Unknown));
            assignments
        };
        let resolved = make_chain().resolve().expect("long forward chain");
        assert_eq!(resolved.get(bit(start)), Some(Bit(root)));
        assert_eq!(resolved.get(root), None);
        assert!(resolved.sources().iter().all(|&source| source == Bit(root)));

        let mut assignments = make_chain();
        assert!(assignments.insert(bit(end + 1), Bit(bit(end + 2))));
        assert!(assignments.insert(bit(end + 2), Bit(bit(end + 1))));
        assert_eq!(assignments.resolve().err(), Some(bit(end + 1)));
    }
}
