// SPDX-License-Identifier: Apache-2.0

//! Exact six-input matching without eagerly expanding every library binding.

use super::{Binding, Cell, compare_bindings};
use std::collections::{BTreeMap, VecDeque};

const INPUTS: usize = 6;
const CACHE_ENTRIES: usize = 4096;
const VARIABLES: [u64; INPUTS] = [
    0xaaaaaaaaaaaaaaaa,
    0xcccccccccccccccc,
    0xf0f0f0f0f0f0f0f0,
    0xff00ff00ff00ff00,
    0xffff0000ffff0000,
    0xffffffff00000000,
];

/// Input properties unchanged by input complementation or renaming.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct VariableSignature {
    smaller_cofactor: u32,
    larger_cofactor: u32,
    influence: u32,
}

/// An inexpensive necessary condition for two functions to share a binding.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Signature {
    ones: u32,
    variables: [VariableSignature; INPUTS],
}

/// Ordered cofactor counts and invariant signatures for one truth table.
#[derive(Debug)]
struct TableProperties {
    cofactors: [[u32; 2]; INPUTS],
    variables: [VariableSignature; INPUTS],
}

impl TableProperties {
    /// Summarizes each input while preserving its physical or logical position.
    fn new(table: u64) -> Self {
        let cofactors = std::array::from_fn(|i| {
            [
                (table & !VARIABLES[i]).count_ones(),
                (table & VARIABLES[i]).count_ones(),
            ]
        });
        let variables = std::array::from_fn(|i| VariableSignature {
            smaller_cofactor: cofactors[i][0].min(cofactors[i][1]),
            larger_cofactor: cofactors[i][0].max(cofactors[i][1]),
            influence: (table ^ flip_input(table, i)).count_ones() / 2,
        });
        Self {
            cofactors,
            variables,
        }
    }

    /// Discards input order for an invariant library-bucket key.
    fn signature(&self, table: u64) -> Signature {
        let mut variables = self.variables;
        variables.sort();
        Signature {
            ones: table.count_ones(),
            variables,
        }
    }
}

/// One unexpanded library function and its interchangeable physical inputs.
#[derive(Debug)]
struct Function {
    cell: usize,
    table: u64,
    properties: TableProperties,
    symmetry_predecessor: [Option<usize>; INPUTS],
}

/// Raw library functions grouped by invariants, retaining no expanded bindings.
#[derive(Debug, Default)]
pub(super) struct SixInputIndex {
    buckets: BTreeMap<Signature, Vec<Function>>,
}

impl SixInputIndex {
    /// Indexes the selected physical functions and finds interchangeable pins.
    pub(super) fn new(functions: Vec<(usize, u64)>, cells: &[Cell]) -> Self {
        let mut result = Self::default();
        for (cell, table) in functions {
            let properties = TableProperties::new(table);
            // Assign leaves in increasing order across pins whose exchange
            // preserves both the function and the delay at each leaf.
            let symmetry_predecessor = std::array::from_fn(|i| {
                (0..i).rev().find(|j| {
                    if cells[cell].pin_delays[i] != cells[cell].pin_delays[*j] {
                        return false;
                    }
                    let mut permutation = std::array::from_fn(|k| k);
                    permutation.swap(i, *j);
                    permute_inputs(table, &permutation) == table
                })
            });
            result
                .buckets
                .entry(properties.signature(table))
                .or_default()
                .push(Function {
                    cell,
                    table,
                    properties,
                    symmetry_predecessor,
                });
        }
        result
    }

    pub(super) fn is_empty(&self) -> bool {
        self.buckets.is_empty()
    }

    /// Searches at most 6! permutations and 2^6 phases per surviving function.
    fn matches(&self, table: u64, cells: &[Cell]) -> Vec<Binding> {
        let query = TableProperties::new(table);
        let Some(functions) = self.buckets.get(&query.signature(table)) else {
            return Vec::new();
        };
        let mut best = BTreeMap::new();
        for function in functions {
            Search {
                function,
                table,
                query: &query,
                cells,
                best: &mut best,
            }
            .permutations(0, 0, &mut [0; INPUTS]);
        }
        let mut result = best.into_values().collect::<Vec<_>>();
        result.sort_by(|a, b| compare_bindings(a, b, cells));
        result
    }
}

/// One query/function comparison, sharing the cheapest binding for each
/// combination of logical leaf phases and delays.
struct Search<'a> {
    function: &'a Function,
    table: u64,
    query: &'a TableProperties,
    cells: &'a [Cell],
    best: &'a mut BTreeMap<(usize, [u64; INPUTS]), Binding>,
}

impl Search<'_> {
    /// Enumerates only compatible input roles and distinct symmetric bindings.
    fn permutations(&mut self, pin: usize, used: usize, permutation: &mut [usize; INPUTS]) {
        if pin == INPUTS {
            self.phases(permutation);
            return;
        }
        for leaf in 0..INPUTS {
            if used & (1 << leaf) != 0
                || self.function.properties.variables[pin] != self.query.variables[leaf]
                || self.function.symmetry_predecessor[pin]
                    .is_some_and(|previous| permutation[previous] >= leaf)
            {
                continue;
            }
            permutation[pin] = leaf;
            self.permutations(pin + 1, used | (1 << leaf), permutation);
        }
    }

    /// Uses cofactor counts to force phases, then checks every remaining table.
    fn phases(&mut self, permutation: &[usize; INPUTS]) {
        let mut transformed = permute_inputs(self.table, permutation);
        let mut physical_phases = 0;
        let mut optional = [0; INPUTS];
        let mut optional_count = 0;
        for (pin, leaf) in permutation.iter().copied().enumerate() {
            let counts = self.query.cofactors[leaf];
            if counts[0] == counts[1] {
                optional[optional_count] = pin;
                optional_count += 1;
            } else if counts[0] != self.function.properties.cofactors[pin][0] {
                physical_phases |= 1 << pin;
                transformed = flip_input(transformed, pin);
            }
        }
        for step in 0usize..1 << optional_count {
            if step != 0 {
                // A Gray-code traversal changes exactly one optional phase.
                let pin = optional[step.trailing_zeros() as usize];
                physical_phases ^= 1 << pin;
                transformed = flip_input(transformed, pin);
            }
            if transformed != self.function.table {
                continue;
            }
            let phases = permutation.iter().enumerate().fold(0, |mask, (pin, leaf)| {
                mask | (((physical_phases >> pin) & 1) << leaf)
            });
            let candidate = Binding {
                cell: self.function.cell,
                perm: permutation.to_vec(),
                phases,
            };
            let key = (phases, candidate.leaf_delay_key(self.cells));
            if self
                .best
                .get(&key)
                .is_none_or(|current| compare_bindings(&candidate, current, self.cells).is_lt())
            {
                self.best.insert(key, candidate);
            }
        }
    }
}

/// Expresses a leaf-ordered query in physical-pin assignment order.
fn permute_inputs(table: u64, permutation: &[usize; INPUTS]) -> u64 {
    (0..64).fold(0, |result, assignment| {
        let source = permutation
            .iter()
            .enumerate()
            .fold(0, |value, (pin, leaf)| {
                value | (((assignment >> pin) & 1) << leaf)
            });
        result | (((table >> source) & 1) << assignment)
    })
}

/// Complements one input by exchanging alternating truth-table blocks.
fn flip_input(table: u64, input: usize) -> u64 {
    let shift = 1 << input;
    ((table & !VARIABLES[input]) << shift) | ((table & VARIABLES[input]) >> shift)
}

/// A bounded FIFO cache local to one mapping, including unsuccessful queries.
#[derive(Default)]
pub(super) struct SixInputCache {
    entries: BTreeMap<u64, Vec<Binding>>,
    order: VecDeque<u64>,
}

impl SixInputCache {
    /// Reuses exact matches while bounding positive and negative cache entries.
    pub(super) fn lookup(
        &mut self,
        index: &SixInputIndex,
        cells: &[Cell],
        table: u64,
    ) -> &[Binding] {
        if !self.entries.contains_key(&table) {
            if self.entries.len() == CACHE_ENTRIES {
                let oldest = self
                    .order
                    .pop_front()
                    .expect("full cache has an oldest key");
                self.entries.remove(&oldest);
            }
            self.entries.insert(table, index.matches(table, cells));
            self.order.push_back(table);
        }
        self.entries.get(&table).expect("query was inserted")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::techmap::area::permutations;

    /// Supplies physical cell costs independently of the matching algorithms.
    fn cell(name: &str, area: f64) -> Cell {
        Cell {
            name: name.to_string(),
            pins: (0..INPUTS).map(|i| format!("I{i}")).collect(),
            truth_table: 0,
            output: "Y".to_string(),
            area,
            pin_delays: vec![0.0; INPUTS],
        }
    }

    /// Evaluates a binding directly, without the matcher's table transforms.
    fn bound_table(table: u64, permutation: &[usize], phases: usize) -> u64 {
        (0usize..64).fold(0, |result, assignment| {
            let physical = permutation
                .iter()
                .enumerate()
                .fold(0, |value, (pin, leaf)| {
                    value | ((((assignment ^ phases) >> leaf) & 1) << pin)
                });
            result | (((table >> physical) & 1) << assignment)
        })
    }

    /// Reference matcher deliberately visits every permutation and polarity.
    fn exhaustive_matches(table: u64, query: u64, cells: &[Cell]) -> Vec<Binding> {
        let mut best: [Option<Binding>; 64] = std::array::from_fn(|_| None);
        for permutation in permutations(INPUTS) {
            for (phases, slot) in best.iter_mut().enumerate() {
                if bound_table(table, &permutation, phases) != query {
                    continue;
                }
                let candidate = Binding {
                    cell: 0,
                    perm: permutation.clone(),
                    phases,
                };
                if slot.as_ref().is_none_or(|current| candidate < *current) {
                    *slot = Some(candidate);
                }
            }
        }
        let mut result = best.into_iter().flatten().collect::<Vec<_>>();
        result.sort_by(|a, b| compare_bindings(a, b, cells));
        result
    }

    #[test]
    fn asymmetric_matching_agrees_with_exhaustive_binding_search() {
        let cells = vec![cell("ASYMMETRIC", 1.0)];
        let table = 0x935c_2bf1_a760_d84e;
        let permutation = [4, 2, 5, 1, 0, 3];
        let phases = 0b010011;
        let query = bound_table(table, &permutation, phases);
        let index = SixInputIndex::new(vec![(0, table)], &cells);
        let matches = index.matches(query, &cells);
        assert_eq!(matches, exhaustive_matches(table, query, &cells));
        assert!(matches.iter().any(|binding| binding.phases == phases));
    }

    #[test]
    fn timing_index_keeps_one_stable_binding_per_leaf_delay_profile() {
        let mut cells = vec![
            cell("AND_CHEAP", 1.0),
            cell("AND_FAST_A", 2.0),
            cell("AND_FAST_B", 2.0),
            cell("AND_FAST_F", 3.0),
        ];
        for cell in &mut cells {
            cell.pin_delays.fill(2.0);
        }
        cells[1].pin_delays[0] = 1.0;
        cells[2].pin_delays[1] = 1.0;
        cells[3].pin_delays[5] = 1.0;
        let table = 1u64 << 63;
        let index =
            SixInputIndex::new((0..cells.len()).rev().map(|i| (i, table)).collect(), &cells);
        let phases = 0b010101;
        let query = bound_table(table, &[0, 1, 2, 3, 4, 5], phases);
        let mut expected = vec![Binding {
            cell: 0,
            perm: (0..INPUTS).collect(),
            phases,
        }];
        // Every leaf can take the fast pin. The first minimum-area cell wins.
        // Its equal-delay pins receive the remaining leaves in order.
        for fast_leaf in 0..INPUTS {
            let mut perm = vec![fast_leaf];
            perm.extend((0..INPUTS).filter(|leaf| *leaf != fast_leaf));
            expected.push(Binding {
                cell: 1,
                perm,
                phases,
            });
        }
        assert_eq!(index.matches(query, &cells), expected);
    }

    #[test]
    fn symmetric_matching_preserves_every_logical_phase_choice() {
        let cells = vec![cell("PARITY_LARGE", 2.0), cell("PARITY", 1.0)];
        let table = (0usize..64)
            .filter(|assignment| assignment.count_ones() % 2 != 0)
            .fold(0u64, |result, assignment| result | (1 << assignment));
        let index = SixInputIndex::new(vec![(0, table), (1, table)], &cells);
        let matches = index.matches(!table, &cells);
        let expected = (0usize..64)
            .filter(|phases| phases.count_ones() % 2 != 0)
            .map(|phases| Binding {
                cell: 1,
                perm: (0..INPUTS).collect(),
                phases,
            })
            .collect::<Vec<_>>();
        assert_eq!(matches, expected);
        for binding in matches {
            assert_eq!(bound_table(table, &binding.perm, binding.phases), !table);
        }
    }

    #[test]
    fn interchangeable_input_pairs_preserve_asymmetric_phase_bindings() {
        let cells = vec![cell("AOI222", 1.0)];
        let table = (0usize..64)
            .filter(|a| {
                !(a & 0b000011 == 0b000011 || a & 0b001100 == 0b001100 || a & 0b110000 == 0b110000)
            })
            .fold(0u64, |result, assignment| result | (1 << assignment));
        let query = bound_table(table, &[5, 1, 3, 0, 4, 2], 0b010101);
        let index = SixInputIndex::new(vec![(0, table)], &cells);
        assert_eq!(
            index.matches(query, &cells),
            exhaustive_matches(table, query, &cells)
        );
    }

    #[test]
    fn cache_retains_successes_and_failures_with_a_fixed_entry_bound() {
        let cells = vec![cell("AND6", 1.0)];
        let index = SixInputIndex::new(vec![(0, 1u64 << 63)], &cells);
        let mut cache = SixInputCache::default();
        let expected = index.matches(1u64 << 63, &cells);
        assert_eq!(cache.lookup(&index, &cells, 1u64 << 63), expected);
        assert_eq!(cache.lookup(&index, &cells, 1u64 << 63), expected);
        assert_eq!(cache.order.len(), 1);
        assert!(cache.lookup(&index, &cells, 0).is_empty());
        assert!(cache.lookup(&index, &cells, 0).is_empty());
        assert_eq!(cache.order.len(), 2);
        let mut cache = SixInputCache::default();
        let empty_index = SixInputIndex::default();
        for table in 0..=CACHE_ENTRIES as u64 {
            assert!(cache.lookup(&empty_index, &cells, table).is_empty());
        }
        assert_eq!(cache.entries.len(), CACHE_ENTRIES);
        assert_eq!(cache.order.len(), CACHE_ENTRIES);
        assert!(!cache.entries.contains_key(&0));
    }
}
