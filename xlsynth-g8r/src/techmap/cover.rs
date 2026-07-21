// SPDX-License-Identifier: Apache-2.0

//! Choice-aware cut matching and bounded area/delay cover selection.

use crate::aig::{AigNode, AigOperand, AigRef, ChoiceAig};
use crate::techmap::cuts::{ChoiceAnalysis, Cut};
use crate::techmap::liberty_index::{CellBinding, LibertyCellIndex};
use crate::techmap::truth::{MAX_TRUTH_TABLE_INPUTS, complement_truth};
use crate::techmap::{TechMapOptions, TechMapTimingConstraints, scalar_bit_name};
use anyhow::{Result, anyhow};
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// Canonical choice class plus polarity relative to that class's first member.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(super) struct StateKey {
    pub class_id: usize,
    pub polarity: bool,
}

impl StateKey {
    fn polarity_index(self) -> usize {
        usize::from(self.polarity)
    }
}

/// Arena ID for one concrete area/delay solution.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(super) struct SolutionId(pub usize);

/// Leaf source or selected standard-cell implementation for one solution.
#[derive(Clone, Debug)]
pub(super) enum SolutionChoice {
    Source(SourceKind),
    Cell {
        binding: CellBinding,
        /// One child solution per cell input pin.
        inputs: Vec<SolutionId>,
    },
}

/// Zero-area signal source available before technology mapping.
#[derive(Clone, Copy, Debug)]
pub(super) enum SourceKind {
    Input(AigRef),
    Literal(bool),
}

/// One retained area/delay point and its reconstruction choice.
#[derive(Clone, Debug)]
pub(super) struct Solution {
    pub state: StateKey,
    pub area: f64,
    pub arrival: f64,
    pub choice: SolutionChoice,
}

/// Complete selected-cover arena plus output roots.
#[derive(Clone, Debug)]
pub(super) struct CoverPlan {
    pub solutions: Vec<Solution>,
    pub output_solutions: Vec<SolutionId>,
    pub output_arrivals: Vec<f64>,
    pub matched_candidate_count: usize,
}

#[derive(Clone, Debug)]
struct Candidate {
    binding: CellBinding,
    /// One canonical-relative state per cell input pin.
    input_states: SmallVec<[StateKey; MAX_TRUTH_TABLE_INPUTS]>,
}

#[derive(Clone, Debug)]
struct PendingSolution {
    area: f64,
    arrival: f64,
    binding: CellBinding,
    inputs: SmallVec<[SolutionId; MAX_TRUTH_TABLE_INPUTS]>,
}

#[derive(Clone, Debug)]
struct PartialCombination {
    area: f64,
    arrival: f64,
    /// One selected child solution for each unique candidate input state.
    selected: SmallVec<[SolutionId; MAX_TRUTH_TABLE_INPUTS]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum VisitState {
    Unvisited,
    Visiting,
    Done,
}

/// Builds a bounded Pareto cover for each requested output.
pub(super) fn build_cover_plan(
    choice_aig: &ChoiceAig,
    analysis: &ChoiceAnalysis,
    cuts_by_node: &[Vec<Cut>],
    cell_index: &LibertyCellIndex,
    options: &TechMapOptions,
    constraints: &TechMapTimingConstraints,
) -> Result<CoverPlan> {
    let mut builder = CoverBuilder::new(
        choice_aig,
        analysis,
        cuts_by_node,
        cell_index,
        options,
        constraints,
    )?;
    builder.build()
}

struct CoverBuilder<'a> {
    choice_aig: &'a ChoiceAig,
    analysis: &'a ChoiceAnalysis,
    cuts_by_node: &'a [Vec<Cut>],
    cell_index: &'a LibertyCellIndex,
    options: &'a TechMapOptions,
    constraints: &'a TechMapTimingConstraints,
    input_arrival_by_node: HashMap<usize, f64>,
    output_names: Vec<String>,
    frontiers: Vec<[Vec<SolutionId>; 2]>,
    visit_state: Vec<VisitState>,
    solutions: Vec<Solution>,
    matched_candidate_count: usize,
}

impl<'a> CoverBuilder<'a> {
    fn new(
        choice_aig: &'a ChoiceAig,
        analysis: &'a ChoiceAnalysis,
        cuts_by_node: &'a [Vec<Cut>],
        cell_index: &'a LibertyCellIndex,
        options: &'a TechMapOptions,
        constraints: &'a TechMapTimingConstraints,
    ) -> Result<Self> {
        if options.max_frontier_size == 0 {
            return Err(anyhow!("max_frontier_size must be at least 1"));
        }
        let graph = choice_aig.graph();
        let mut input_arrival_by_node = HashMap::new();
        let mut input_names = BTreeSet::new();
        for input in &graph.inputs {
            let bit_count = input.get_bit_count();
            for (bit_index, bit) in input.bit_vector.iter_lsb_to_msb().enumerate() {
                if bit.negated {
                    return Err(anyhow!(
                        "technology mapping does not support negated input-port bindings"
                    ));
                }
                let name = scalar_bit_name(input.name.as_str(), bit_index, bit_count);
                input_names.insert(name.clone());
                let arrival = constraints
                    .primary_input_arrivals
                    .get(&name)
                    .copied()
                    .unwrap_or(0.0);
                input_arrival_by_node.insert(bit.node.id, arrival);
            }
        }

        let mut output_names = Vec::new();
        let mut known_output_names = BTreeSet::new();
        for output in &graph.outputs {
            let bit_count = output.get_bit_count();
            for (bit_index, _) in output.bit_vector.iter_lsb_to_msb().enumerate() {
                let name = scalar_bit_name(output.name.as_str(), bit_index, bit_count);
                known_output_names.insert(name.clone());
                output_names.push(name);
            }
        }
        for name in constraints.primary_input_arrivals.keys() {
            if !input_names.contains(name) {
                return Err(anyhow!(
                    "timing constraint names unknown primary input '{}'",
                    name
                ));
            }
        }
        for name in constraints.primary_output_required.keys() {
            if !known_output_names.contains(name) {
                return Err(anyhow!(
                    "timing constraint names unknown primary output '{}'",
                    name
                ));
            }
        }

        Ok(Self {
            choice_aig,
            analysis,
            cuts_by_node,
            cell_index,
            options,
            constraints,
            input_arrival_by_node,
            output_names,
            frontiers: (0..analysis.classes.len())
                .map(|_| [Vec::new(), Vec::new()])
                .collect(),
            visit_state: vec![VisitState::Unvisited; analysis.classes.len()],
            solutions: Vec::new(),
            matched_candidate_count: 0,
        })
    }

    fn build(&mut self) -> Result<CoverPlan> {
        let graph = self.choice_aig.graph();
        let mut output_solutions = Vec::new();
        let mut output_arrivals = Vec::new();
        let mut output_index = 0usize;
        for output in &graph.outputs {
            for operand in output.bit_vector.iter_lsb_to_msb() {
                let (class_id, polarity) = self.analysis.state_for_operand(*operand);
                let state = StateKey { class_id, polarity };
                if !self.solve_class(class_id)? {
                    return Err(anyhow!(
                        "could not resolve a non-cyclic cover for output '{}'",
                        self.output_names[output_index]
                    ));
                }
                let solution =
                    self.select_output_solution(state, self.output_names[output_index].as_str())?;
                output_arrivals.push(self.solutions[solution.0].arrival);
                output_solutions.push(solution);
                output_index += 1;
            }
        }
        Ok(CoverPlan {
            solutions: std::mem::take(&mut self.solutions),
            output_solutions,
            output_arrivals,
            matched_candidate_count: self.matched_candidate_count,
        })
    }

    fn solve_class(&mut self, class_id: usize) -> Result<bool> {
        match self.visit_state[class_id] {
            VisitState::Done => return Ok(true),
            VisitState::Visiting => return Ok(false),
            VisitState::Unvisited => {}
        }
        self.visit_state[class_id] = VisitState::Visiting;

        let mut direct_frontiers = [Vec::new(), Vec::new()];
        self.add_source_solutions(class_id, &mut direct_frontiers);
        for polarity in [false, true] {
            let state = StateKey { class_id, polarity };
            let candidates = self.candidates_for_state(state);
            let mut pending_frontier = Vec::new();
            for candidate in candidates {
                let mut pending = self.combine_candidate(candidate)?;
                extend_pending_frontier(
                    &mut pending_frontier,
                    &mut pending,
                    self.options.max_frontier_size,
                );
            }
            for solution in
                prune_pending_solutions(pending_frontier, self.options.max_frontier_size)
            {
                let id = self.push_solution(
                    state,
                    solution.area,
                    solution.arrival,
                    SolutionChoice::Cell {
                        binding: solution.binding,
                        inputs: solution.inputs.into_vec(),
                    },
                );
                direct_frontiers[state.polarity_index()].push(id);
            }
            direct_frontiers[state.polarity_index()] = prune_solution_ids(
                std::mem::take(&mut direct_frontiers[state.polarity_index()]),
                self.solutions.as_slice(),
                self.options.max_frontier_size,
            );
        }

        let mut final_frontiers = direct_frontiers.clone();
        let require_timing = self.constraints.has_endpoint_requirements();
        let inverters: Vec<CellBinding> = self
            .cell_index
            .matches(1, 0b01)
            .iter()
            .filter(|binding| {
                !binding.input_negated[0] && (!require_timing || binding.has_complete_timing())
            })
            .cloned()
            .collect();
        for inverter in inverters {
            for polarity in [false, true] {
                let target = StateKey { class_id, polarity };
                let source_ids = direct_frontiers[usize::from(!polarity)].clone();
                for source_id in source_ids {
                    let source = &self.solutions[source_id.0];
                    let delay = inverter.input_delays[0].unwrap_or(0.0);
                    let id = self.push_solution(
                        target,
                        inverter.area + source.area,
                        source.arrival + delay,
                        SolutionChoice::Cell {
                            binding: inverter.clone(),
                            inputs: vec![source_id],
                        },
                    );
                    final_frontiers[target.polarity_index()].push(id);
                }
            }
        }
        for polarity in [false, true] {
            final_frontiers[usize::from(polarity)] = prune_solution_ids(
                std::mem::take(&mut final_frontiers[usize::from(polarity)]),
                self.solutions.as_slice(),
                self.options.max_frontier_size,
            );
        }

        self.frontiers[class_id] = final_frontiers;
        self.visit_state[class_id] = VisitState::Done;
        Ok(!self.frontiers[class_id][0].is_empty() || !self.frontiers[class_id][1].is_empty())
    }

    fn add_source_solutions(&mut self, class_id: usize, frontiers: &mut [Vec<SolutionId>; 2]) {
        let graph = self.choice_aig.graph();
        let class = &self.analysis.classes[class_id];
        for member in &class.members {
            match graph.get(*member) {
                AigNode::Input { .. } => {
                    let (source_class, polarity) = self.analysis.state_for_positive_node(*member);
                    debug_assert_eq!(source_class, class_id);
                    let arrival = self
                        .input_arrival_by_node
                        .get(&member.id)
                        .copied()
                        .unwrap_or(0.0);
                    let state = StateKey { class_id, polarity };
                    let id = self.push_solution(
                        state,
                        0.0,
                        arrival,
                        SolutionChoice::Source(SourceKind::Input(*member)),
                    );
                    frontiers[state.polarity_index()].push(id);
                }
                AigNode::Literal { .. } => {
                    for polarity in [false, true] {
                        let state = StateKey { class_id, polarity };
                        let value = self.analysis.phase_by_node[class.canonical.id] ^ polarity;
                        let id = self.push_solution(
                            state,
                            0.0,
                            0.0,
                            SolutionChoice::Source(SourceKind::Literal(value)),
                        );
                        frontiers[state.polarity_index()].push(id);
                    }
                }
                AigNode::And2 { .. } => {}
            }
        }
        for frontier in frontiers {
            *frontier = prune_solution_ids(
                std::mem::take(frontier),
                self.solutions.as_slice(),
                self.options.max_frontier_size,
            );
        }
    }

    fn candidates_for_state(&mut self, state: StateKey) -> Vec<Candidate> {
        let class = &self.analysis.classes[state.class_id];
        let canonical_phase = self.analysis.phase_by_node[class.canonical.id];
        let require_timing = self.constraints.has_endpoint_requirements();
        let mut candidates = Vec::new();
        let mut area_candidates = BTreeMap::new();
        let mut matched_candidate_count = 0usize;
        for member in &class.members {
            let member_phase = self.analysis.phase_by_node[member.id];
            for cut in &self.cuts_by_node[member.id] {
                if cut
                    .leaves
                    .iter()
                    .any(|leaf| self.analysis.class_by_node[leaf.id] == state.class_id)
                {
                    continue;
                }
                let complement = state.polarity ^ canonical_phase ^ member_phase;
                let truth = if complement {
                    complement_truth(cut.truth, cut.leaves.len())
                } else {
                    cut.truth
                };
                for binding in self.cell_index.matches(cut.leaves.len(), truth) {
                    if require_timing && !binding.has_complete_timing() {
                        continue;
                    }
                    let input_states = input_states_for_binding(self.analysis, cut, binding);
                    matched_candidate_count += 1;
                    if require_timing {
                        candidates.push(Candidate {
                            binding: binding.clone(),
                            input_states,
                        });
                        continue;
                    }
                    insert_area_candidate(&mut area_candidates, binding, input_states);
                }
            }
        }
        self.matched_candidate_count += matched_candidate_count;
        if require_timing {
            candidates
        } else {
            area_candidates.into_values().collect()
        }
    }

    fn combine_candidate(&mut self, candidate: Candidate) -> Result<Vec<PendingSolution>> {
        let mut delay_by_state: SmallVec<[(StateKey, f64); MAX_TRUTH_TABLE_INPUTS]> =
            SmallVec::new();
        for (input_index, state) in candidate.input_states.iter().enumerate() {
            let delay = candidate.binding.input_delays[input_index].unwrap_or(0.0);
            match delay_by_state
                .iter_mut()
                .find(|(known_state, _)| known_state == state)
            {
                Some((_, current)) => *current = current.max(delay),
                None => delay_by_state.push((*state, delay)),
            }
        }
        delay_by_state.sort_by_key(|(state, _)| *state);
        let input_state_slots: SmallVec<[usize; MAX_TRUTH_TABLE_INPUTS]> = candidate
            .input_states
            .iter()
            .map(|state| {
                delay_by_state
                    .iter()
                    .position(|(known_state, _)| known_state == state)
                    .expect("every candidate input state should have one local slot")
            })
            .collect();

        let mut partials = vec![PartialCombination {
            area: candidate.binding.area,
            arrival: 0.0,
            selected: SmallVec::new(),
        }];
        for (state, delay) in delay_by_state {
            if !self.solve_class(state.class_id)? {
                return Ok(Vec::new());
            }
            let child_ids = self.frontiers[state.class_id][state.polarity_index()].clone();
            if child_ids.is_empty() {
                return Ok(Vec::new());
            }
            let mut next = Vec::new();
            for partial in &partials {
                for child_id in &child_ids {
                    let child = &self.solutions[child_id.0];
                    let mut selected = partial.selected.clone();
                    selected.push(*child_id);
                    next.push(PartialCombination {
                        area: partial.area + child.area,
                        arrival: partial.arrival.max(child.arrival + delay),
                        selected,
                    });
                }
            }
            partials = prune_partials(next, self.options.max_frontier_size);
        }

        Ok(partials
            .into_iter()
            .map(|partial| PendingSolution {
                area: partial.area,
                arrival: partial.arrival,
                binding: candidate.binding.clone(),
                inputs: input_state_slots
                    .iter()
                    .map(|slot| partial.selected[*slot])
                    .collect(),
            })
            .collect())
    }

    fn select_output_solution(&self, state: StateKey, output_name: &str) -> Result<SolutionId> {
        let frontier = &self.frontiers[state.class_id][state.polarity_index()];
        debug_assert!(
            frontier
                .iter()
                .all(|id| self.solutions[id.0].state == state)
        );
        if frontier.is_empty() {
            return Err(anyhow!(
                "no Liberty cell cover exists for output '{}' state {:?}",
                output_name,
                state
            ));
        }
        let required = self
            .constraints
            .primary_output_required
            .get(output_name)
            .copied();
        let mut eligible: Vec<SolutionId> = frontier
            .iter()
            .copied()
            .filter(|id| {
                required.map_or(true, |limit| {
                    self.solutions[id.0].arrival <= limit + f64::EPSILON
                })
            })
            .collect();
        if eligible.is_empty() {
            let fastest = frontier
                .iter()
                .copied()
                .min_by(|lhs, rhs| {
                    self.solutions[lhs.0]
                        .arrival
                        .total_cmp(&self.solutions[rhs.0].arrival)
                })
                .unwrap();
            return Err(anyhow!(
                "no cover meets required time {} for output '{}'; fastest estimated arrival is {}",
                required.unwrap(),
                output_name,
                self.solutions[fastest.0].arrival
            ));
        }
        eligible.sort_unstable_by(|lhs, rhs| {
            solution_area_order(*lhs, *rhs, self.solutions.as_slice())
        });
        Ok(eligible[0])
    }

    fn push_solution(
        &mut self,
        state: StateKey,
        area: f64,
        arrival: f64,
        choice: SolutionChoice,
    ) -> SolutionId {
        let id = SolutionId(self.solutions.len());
        self.solutions.push(Solution {
            state,
            area,
            arrival,
            choice,
        });
        id
    }
}

fn prune_solution_ids(
    mut ids: Vec<SolutionId>,
    solutions: &[Solution],
    max_frontier_size: usize,
) -> Vec<SolutionId> {
    ids.sort_unstable_by(|lhs, rhs| solution_arrival_order(*lhs, *rhs, solutions));
    ids.dedup_by(|lhs, rhs| {
        let lhs = &solutions[lhs.0];
        let rhs = &solutions[rhs.0];
        lhs.area == rhs.area && lhs.arrival == rhs.arrival
    });
    let mut frontier = Vec::new();
    let mut best_area = f64::INFINITY;
    for id in ids {
        let solution = &solutions[id.0];
        if solution.area < best_area {
            best_area = solution.area;
            frontier.push(id);
        }
    }
    cap_frontier(frontier, max_frontier_size, |id| {
        (solutions[id.0].arrival, solutions[id.0].area)
    })
}

fn prune_partials(
    mut partials: Vec<PartialCombination>,
    max_frontier_size: usize,
) -> Vec<PartialCombination> {
    partials.sort_unstable_by(|lhs, rhs| {
        lhs.arrival
            .total_cmp(&rhs.arrival)
            .then_with(|| lhs.area.total_cmp(&rhs.area))
            .then_with(|| lhs.selected.cmp(&rhs.selected))
    });
    let mut frontier = Vec::new();
    let mut best_area = f64::INFINITY;
    for partial in partials {
        if partial.area < best_area {
            best_area = partial.area;
            frontier.push(partial);
        }
    }
    cap_frontier(frontier, max_frontier_size, |partial| {
        (partial.arrival, partial.area)
    })
}

/// Merges candidate solutions into a small pending frontier before any arena
/// IDs are allocated for them.
///
/// Pruning in bounded batches prevents a large choice class from allocating
/// one permanent arena entry per matched cut/binding pair. The final prune
/// below still applies the same deterministic area/arrival ordering.
fn extend_pending_frontier(
    frontier: &mut Vec<PendingSolution>,
    additions: &mut Vec<PendingSolution>,
    max_frontier_size: usize,
) {
    frontier.append(additions);
    let prune_threshold = max_frontier_size.saturating_mul(4).max(max_frontier_size);
    if frontier.len() >= prune_threshold {
        *frontier = prune_pending_solutions(std::mem::take(frontier), max_frontier_size);
    }
}

fn prune_pending_solutions(
    mut pending: Vec<PendingSolution>,
    max_frontier_size: usize,
) -> Vec<PendingSolution> {
    pending.sort_unstable_by(pending_solution_order);
    pending.dedup_by(|lhs, rhs| lhs.area == rhs.area && lhs.arrival == rhs.arrival);
    let mut frontier = Vec::new();
    let mut best_area = f64::INFINITY;
    for solution in pending {
        if solution.area < best_area {
            best_area = solution.area;
            frontier.push(solution);
        }
    }
    cap_frontier(frontier, max_frontier_size, |solution| {
        (solution.arrival, solution.area)
    })
}

fn cap_frontier<T, F>(frontier: Vec<T>, limit: usize, _key: F) -> Vec<T>
where
    F: Fn(&T) -> (f64, f64),
{
    if frontier.len() <= limit {
        return frontier;
    }
    if limit == 1 {
        return vec![frontier.into_iter().last().unwrap()];
    }
    let last = frontier.len() - 1;
    let mut retained = Vec::with_capacity(limit);
    let mut previous_index = usize::MAX;
    for slot in 0..limit {
        let index = slot * last / (limit - 1);
        if index != previous_index {
            retained.push(index);
            previous_index = index;
        }
    }
    let mut values: Vec<Option<T>> = frontier.into_iter().map(Some).collect();
    retained
        .into_iter()
        .map(|index| values[index].take().unwrap())
        .collect()
}

fn solution_arrival_order(
    lhs: SolutionId,
    rhs: SolutionId,
    solutions: &[Solution],
) -> std::cmp::Ordering {
    solutions[lhs.0]
        .arrival
        .total_cmp(&solutions[rhs.0].arrival)
        .then_with(|| solutions[lhs.0].area.total_cmp(&solutions[rhs.0].area))
        .then_with(|| lhs.cmp(&rhs))
}

fn solution_area_order(
    lhs: SolutionId,
    rhs: SolutionId,
    solutions: &[Solution],
) -> std::cmp::Ordering {
    solutions[lhs.0]
        .area
        .total_cmp(&solutions[rhs.0].area)
        .then_with(|| {
            solutions[lhs.0]
                .arrival
                .total_cmp(&solutions[rhs.0].arrival)
        })
        .then_with(|| lhs.cmp(&rhs))
}

fn pending_solution_order(lhs: &PendingSolution, rhs: &PendingSolution) -> std::cmp::Ordering {
    lhs.arrival
        .total_cmp(&rhs.arrival)
        .then_with(|| lhs.area.total_cmp(&rhs.area))
        .then_with(|| lhs.binding.stable_key().cmp(&rhs.binding.stable_key()))
        .then_with(|| lhs.inputs.cmp(&rhs.inputs))
}

fn input_states_for_binding(
    analysis: &ChoiceAnalysis,
    cut: &Cut,
    binding: &CellBinding,
) -> SmallVec<[StateKey; MAX_TRUTH_TABLE_INPUTS]> {
    binding
        .input_to_leaf
        .iter()
        .enumerate()
        .map(|(input_index, leaf_index)| {
            let leaf = cut.leaves[*leaf_index];
            let (class_id, polarity) = analysis.state_for_operand(AigOperand {
                node: leaf,
                negated: binding.input_negated[input_index],
            });
            StateKey { class_id, polarity }
        })
        .collect()
}

/// Drops area-only candidates whose child-state multiset is identical before
/// cloning their full Liberty bindings into candidate storage.
///
/// Without endpoint requirements, drive-strength and symmetric pin variants
/// with the same child states cannot change the area recurrence. Keeping only
/// the cheapest stable binding avoids spending most of an unconstrained run on
/// physically interchangeable cell variants. Timing-constrained runs retain
/// the full set above.
fn insert_area_candidate(
    by_child_states: &mut BTreeMap<SmallVec<[StateKey; MAX_TRUTH_TABLE_INPUTS]>, Candidate>,
    binding: &CellBinding,
    input_states: SmallVec<[StateKey; MAX_TRUTH_TABLE_INPUTS]>,
) {
    let mut key = input_states.clone();
    key.sort();
    match by_child_states.get(&key) {
        Some(existing)
            if binding.area > existing.binding.area
                || (binding.area == existing.binding.area
                    && binding.stable_key() >= existing.binding.stable_key()) => {}
        _ => {
            by_child_states.insert(
                key,
                Candidate {
                    binding: binding.clone(),
                    input_states,
                },
            );
        }
    }
}
