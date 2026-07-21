// SPDX-License-Identifier: Apache-2.0

//! NF-style choice-aware cut matching and iterative cover selection.

use crate::aig::{AigNode, AigOperand, AigRef, ChoiceAig};
use crate::liberty_model::Library;
use crate::netlist::sta::{
    CombinationalOutputLoad, EdgeTiming, SignalTiming, TimingQueryDiagnosticCounts,
    evaluate_combinational_cell_output_timing,
};
use crate::techmap::cuts::{ChoiceAnalysis, Cut};
use crate::techmap::liberty_index::{CellBinding, LibertyCellIndex};
use crate::techmap::truth::{MAX_TRUTH_TABLE_INPUTS, complement_truth};
use crate::techmap::{TechMapOptions, TechMapTimingConstraints, scalar_bit_name};
use anyhow::{Result, anyhow};
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

const NF_AREA_FLOW_ROUNDS: usize = 4;
const NF_EXACT_AREA_ROUNDS: usize = 2;
const TIMING_EPSILON: f64 = 1e-9;
const MAX_CELL_VARIANTS_PER_SIGNATURE: usize = 8;

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

/// Arena ID for one concrete selected solution.
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

/// One selected solution retained only for final netlist reconstruction.
#[derive(Clone, Debug)]
pub(super) struct Solution {
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
enum MatchChoice {
    Source(SourceKind),
    Cell {
        binding: CellBinding,
        inputs: SmallVec<[StateKey; MAX_TRUTH_TABLE_INPUTS]>,
    },
}

/// One NF-like match: a timing point, an area-flow score, and reconstruction
/// information. A state keeps one delay match and one area-under-required-time
/// match rather than a large local Pareto frontier.
#[derive(Clone, Debug)]
struct NfMatch {
    timing: SignalTiming,
    flow: f64,
    choice: MatchChoice,
    /// Conservative required-time subtraction for each cell input pin.
    input_delays: SmallVec<[f64; MAX_TRUTH_TABLE_INPUTS]>,
}

impl NfMatch {
    fn worst_arrival(&self) -> f64 {
        worst_signal_arrival(self.timing)
    }
}

#[derive(Clone, Debug, Default)]
struct StateMatches {
    direct_delay: Option<NfMatch>,
    direct_area: Option<NfMatch>,
    delay: Option<NfMatch>,
    area: Option<NfMatch>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum VisitState {
    Unvisited,
    Visiting,
    Done,
}

/// Delay model used while NF chooses a structural cover.
///
/// ABC's generated genlib for the normal `&nf` flow gives every gate input a
/// unit delay. Keep that objective for unconstrained mapping so the cover
/// search behaves like NF; use Liberty-scaled scalar delays once the caller
/// supplies an explicit endpoint timing constraint.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SearchTimingModel {
    Unit,
    LibertyScalar,
}

#[derive(Clone, Debug)]
struct OutputEndpoint {
    name: String,
    state: StateKey,
}

#[derive(Clone)]
struct MappingTrace {
    selected: Vec<[Option<NfMatch>; 2]>,
    requireds: Vec<[f64; 2]>,
    loads: Vec<[CombinationalOutputLoad; 2]>,
    map_refs: Vec<[usize; 2]>,
    output_arrivals: Vec<f64>,
    area: f64,
}

/// Builds an NF-shaped cover: fastest mapping first, then area-flow mapping
/// under backward-propagated required times.
pub(super) fn build_cover_plan(
    choice_aig: &ChoiceAig,
    analysis: &ChoiceAnalysis,
    cuts_by_node: &[Vec<Cut>],
    cell_index: &LibertyCellIndex,
    library: &Library,
    options: &TechMapOptions,
    constraints: &TechMapTimingConstraints,
) -> Result<CoverPlan> {
    let mut builder = CoverBuilder::new(
        choice_aig,
        analysis,
        cuts_by_node,
        cell_index,
        library,
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
    library: &'a Library,
    options: &'a TechMapOptions,
    constraints: &'a TechMapTimingConstraints,
    search_timing_model: SearchTimingModel,
    input_arrival_by_node: HashMap<usize, f64>,
    outputs: Vec<OutputEndpoint>,
    candidates: Vec<[Option<Vec<Candidate>>; 2]>,
    matches: Vec<[StateMatches; 2]>,
    visit_state: Vec<VisitState>,
    matched_candidate_count: usize,
    timing_query_diagnostic_counts: TimingQueryDiagnosticCounts,
    empty_known_pin_values: HashMap<String, bool>,
}

impl<'a> CoverBuilder<'a> {
    fn new(
        choice_aig: &'a ChoiceAig,
        analysis: &'a ChoiceAnalysis,
        cuts_by_node: &'a [Vec<Cut>],
        cell_index: &'a LibertyCellIndex,
        library: &'a Library,
        options: &'a TechMapOptions,
        constraints: &'a TechMapTimingConstraints,
    ) -> Result<Self> {
        if options.max_frontier_size == 0 {
            return Err(anyhow!("max_frontier_size must be at least 1"));
        }
        if !options.primary_input_transition.is_finite() || options.primary_input_transition < 0.0 {
            return Err(anyhow!(
                "primary_input_transition must be non-negative and finite; got {}",
                options.primary_input_transition
            ));
        }
        if !options.module_output_load.is_finite() || options.module_output_load < 0.0 {
            return Err(anyhow!(
                "module_output_load must be non-negative and finite; got {}",
                options.module_output_load
            ));
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

        let mut outputs = Vec::new();
        let mut known_output_names = BTreeSet::new();
        for output in &graph.outputs {
            let bit_count = output.get_bit_count();
            for (bit_index, operand) in output.bit_vector.iter_lsb_to_msb().enumerate() {
                let name = scalar_bit_name(output.name.as_str(), bit_index, bit_count);
                known_output_names.insert(name.clone());
                let (class_id, polarity) = analysis.state_for_operand(*operand);
                outputs.push(OutputEndpoint {
                    name,
                    state: StateKey { class_id, polarity },
                });
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

        let class_count = analysis.classes.len();
        let search_timing_model = if constraints.primary_input_arrivals.is_empty()
            && constraints.primary_output_required.is_empty()
        {
            SearchTimingModel::Unit
        } else {
            SearchTimingModel::LibertyScalar
        };
        Ok(Self {
            choice_aig,
            analysis,
            cuts_by_node,
            cell_index,
            library,
            options,
            constraints,
            search_timing_model,
            input_arrival_by_node,
            outputs,
            candidates: (0..class_count).map(|_| [None, None]).collect(),
            matches: (0..class_count)
                .map(|_| [StateMatches::default(), StateMatches::default()])
                .collect(),
            visit_state: vec![VisitState::Unvisited; class_count],
            matched_candidate_count: 0,
            timing_query_diagnostic_counts: TimingQueryDiagnosticCounts::default(),
            empty_known_pin_values: HashMap::new(),
        })
    }

    fn build(&mut self) -> Result<CoverPlan> {
        let class_count = self.analysis.classes.len();
        let mut loads = pair_vec(class_count, CombinationalOutputLoad::default());
        for output in &self.outputs {
            add_output_load(
                &mut loads[output.state.class_id][output.state.polarity_index()],
                self.options.module_output_load,
            );
        }
        // Choice classes collapse multiple sibling GIA objects into one
        // shared state. Start from a neutral flow reference instead of
        // importing ABC's per-object structural-fanout seed, then blend in
        // actual selected-cover references after each round.
        let mut flow_refs = pair_vec(class_count, 1.0f64);
        let mut requireds = pair_vec(class_count, f64::INFINITY);
        let mut output_requireds = vec![f64::INFINITY; self.outputs.len()];
        let mut final_trace = None;

        for round in 0..NF_AREA_FLOW_ROUNDS {
            self.compute_round_matches(
                requireds.as_slice(),
                flow_refs.as_slice(),
                loads.as_slice(),
            )?;
            let trace = if round == 0 {
                let unbounded_output_requireds = vec![f64::INFINITY; self.outputs.len()];
                let mut delay_trace =
                    self.trace_selected(false, unbounded_output_requireds.as_slice())?;
                if self.search_timing_model == SearchTimingModel::LibertyScalar {
                    self.retime_selected(
                        delay_trace.selected.as_mut_slice(),
                        delay_trace.loads.as_slice(),
                    )?;
                }
                let delay_trace = self.trace_fixed_selected(
                    delay_trace.selected,
                    unbounded_output_requireds.as_slice(),
                )?;
                let global_target = delay_trace
                    .output_arrivals
                    .iter()
                    .copied()
                    .reduce(f64::max)
                    .unwrap_or(0.0);
                output_requireds = self
                    .outputs
                    .iter()
                    .map(|output| {
                        self.constraints
                            .primary_output_required
                            .get(output.name.as_str())
                            .copied()
                            .unwrap_or(global_target)
                    })
                    .collect();
                self.trace_fixed_selected(delay_trace.selected, output_requireds.as_slice())?
            } else {
                let mut area_trace = self.trace_selected(true, output_requireds.as_slice())?;
                if self.search_timing_model == SearchTimingModel::LibertyScalar {
                    self.retime_selected(
                        area_trace.selected.as_mut_slice(),
                        area_trace.loads.as_slice(),
                    )?;
                }
                self.trace_fixed_selected(area_trace.selected, output_requireds.as_slice())?
            };
            requireds = trace.requireds.clone();
            loads = trace.loads.clone();
            blend_flow_refs(flow_refs.as_mut_slice(), trace.map_refs.as_slice(), round);
            final_trace = Some(trace);
        }

        let mut final_trace =
            final_trace.ok_or_else(|| anyhow!("technology mapping ran no rounds"))?;
        for _ in 0..NF_EXACT_AREA_ROUNDS {
            final_trace = self.exact_area_recovery(final_trace, output_requireds.as_slice())?;
        }
        if self.search_timing_model == SearchTimingModel::Unit {
            // Unit delay is only the NF cover-selection objective. Report the
            // finished cover using the same load-aware timing semantics as
            // gv-stats, without comparing Liberty time units to the unit-delay
            // required times used during mapping.
            let mut selected = final_trace.selected;
            self.retime_selected(selected.as_mut_slice(), final_trace.loads.as_slice())?;
            let unbounded_output_requireds = vec![f64::INFINITY; self.outputs.len()];
            final_trace =
                self.trace_fixed_selected(selected, unbounded_output_requireds.as_slice())?;
        }
        for (output_index, output) in self.outputs.iter().enumerate() {
            let required = self
                .constraints
                .primary_output_required
                .get(output.name.as_str());
            if let Some(required) = required
                && final_trace.output_arrivals[output_index] > *required + TIMING_EPSILON
            {
                return Err(anyhow!(
                    "no cover meets required time {} for output '{}'; fastest estimated arrival is {}",
                    required,
                    output.name,
                    final_trace.output_arrivals[output_index]
                ));
            }
        }
        self.materialize_selected(
            final_trace.selected.as_slice(),
            final_trace.output_arrivals.as_slice(),
        )
    }

    fn compute_round_matches(
        &mut self,
        requireds: &[[f64; 2]],
        flow_refs: &[[f64; 2]],
        loads: &[[CombinationalOutputLoad; 2]],
    ) -> Result<()> {
        self.matches = (0..self.analysis.classes.len())
            .map(|_| [StateMatches::default(), StateMatches::default()])
            .collect();
        self.visit_state.fill(VisitState::Unvisited);
        let output_classes: Vec<usize> = self
            .outputs
            .iter()
            .map(|output| output.state.class_id)
            .collect();
        for class_id in output_classes {
            if !self.solve_class(class_id, requireds, flow_refs, loads)? {
                return Err(anyhow!(
                    "could not resolve a non-cyclic cover for choice class {}",
                    class_id
                ));
            }
        }
        Ok(())
    }

    fn solve_class(
        &mut self,
        class_id: usize,
        requireds: &[[f64; 2]],
        flow_refs: &[[f64; 2]],
        loads: &[[CombinationalOutputLoad; 2]],
    ) -> Result<bool> {
        match self.visit_state[class_id] {
            VisitState::Done => return Ok(true),
            VisitState::Visiting => return Ok(false),
            VisitState::Unvisited => {}
        }
        self.visit_state[class_id] = VisitState::Visiting;

        for polarity in [false, true] {
            let state = StateKey { class_id, polarity };
            let mut direct_delay = self.source_match(state);
            let mut direct_area = direct_delay.clone();
            let candidates = self.candidates_for_state(state);
            for candidate in candidates {
                if let Some(candidate_match) =
                    self.evaluate_delay_candidate(state, &candidate, flow_refs, loads, requireds)?
                {
                    retain_best_delay(&mut direct_delay, candidate_match);
                }
                if let Some(candidate_match) =
                    self.evaluate_area_candidate(state, &candidate, requireds, flow_refs, loads)?
                {
                    retain_best_area(&mut direct_area, candidate_match);
                }
            }
            let slot = &mut self.matches[class_id][state.polarity_index()];
            slot.direct_delay = direct_delay.clone();
            slot.direct_area = direct_area.clone();
            slot.delay = direct_delay;
            slot.area = direct_area;
        }

        self.add_inverter_closure(class_id, requireds, flow_refs, loads)?;
        self.visit_state[class_id] = VisitState::Done;
        Ok(self.matches[class_id][0].delay.is_some() || self.matches[class_id][1].delay.is_some())
    }

    fn source_match(&self, state: StateKey) -> Option<NfMatch> {
        let graph = self.choice_aig.graph();
        let class = &self.analysis.classes[state.class_id];
        for member in &class.members {
            match graph.get(*member) {
                AigNode::Input { .. } => {
                    let (_, polarity) = self.analysis.state_for_positive_node(*member);
                    if polarity != state.polarity {
                        continue;
                    }
                    let arrival = self
                        .input_arrival_by_node
                        .get(&member.id)
                        .copied()
                        .unwrap_or(0.0);
                    return Some(source_nf_match(
                        SourceKind::Input(*member),
                        arrival,
                        self.options.primary_input_transition,
                    ));
                }
                AigNode::Literal { .. } => {
                    let value = self.analysis.phase_by_node[class.canonical.id] ^ state.polarity;
                    return Some(source_nf_match(SourceKind::Literal(value), 0.0, 0.0));
                }
                AigNode::And2 { .. } => {}
            }
        }
        None
    }

    fn candidates_for_state(&mut self, state: StateKey) -> Vec<Candidate> {
        if let Some(cached) = self.candidates[state.class_id][state.polarity_index()].as_ref() {
            return cached.clone();
        }
        let class = &self.analysis.classes[state.class_id];
        let canonical_phase = self.analysis.phase_by_node[class.canonical.id];
        let mut by_input_states: BTreeMap<
            SmallVec<[StateKey; MAX_TRUTH_TABLE_INPUTS]>,
            Vec<Candidate>,
        > = BTreeMap::new();
        let mut matched_candidate_count = 0usize;
        let variant_limit = self
            .options
            .max_frontier_size
            .min(MAX_CELL_VARIANTS_PER_SIGNATURE)
            .max(1);
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
                    let input_states = input_states_for_binding(self.analysis, cut, binding);
                    matched_candidate_count += 1;
                    insert_candidate_variant(
                        &mut by_input_states,
                        binding,
                        input_states,
                        variant_limit,
                    );
                }
            }
        }
        self.matched_candidate_count += matched_candidate_count;
        let mut candidates: Vec<Candidate> = by_input_states.into_values().flatten().collect();
        candidates.sort_by(candidate_order);
        self.candidates[state.class_id][state.polarity_index()] = Some(candidates.clone());
        candidates
    }

    fn evaluate_delay_candidate(
        &mut self,
        state: StateKey,
        candidate: &Candidate,
        flow_refs: &[[f64; 2]],
        loads: &[[CombinationalOutputLoad; 2]],
        requireds: &[[f64; 2]],
    ) -> Result<Option<NfMatch>> {
        let unique_states = unique_input_states(candidate.input_states.as_slice());
        let mut child_matches = Vec::with_capacity(unique_states.len());
        for child_state in &unique_states {
            if !self.solve_class(child_state.class_id, requireds, flow_refs, loads)? {
                return Ok(None);
            }
            let Some(child_match) = self.matches[child_state.class_id]
                [child_state.polarity_index()]
            .delay
            .clone() else {
                return Ok(None);
            };
            child_matches.push(child_match);
        }
        self.evaluate_with_child_matches(
            state,
            candidate,
            unique_states.as_slice(),
            child_matches.as_slice(),
            flow_refs,
            loads,
        )
        .map(Some)
    }

    fn evaluate_area_candidate(
        &mut self,
        state: StateKey,
        candidate: &Candidate,
        requireds: &[[f64; 2]],
        flow_refs: &[[f64; 2]],
        loads: &[[CombinationalOutputLoad; 2]],
    ) -> Result<Option<NfMatch>> {
        let unique_states = unique_input_states(candidate.input_states.as_slice());
        let mut selected = Vec::with_capacity(unique_states.len());
        let mut area_alternatives = Vec::with_capacity(unique_states.len());
        for child_state in &unique_states {
            if !self.solve_class(child_state.class_id, requireds, flow_refs, loads)? {
                return Ok(None);
            }
            let child_slot = &self.matches[child_state.class_id][child_state.polarity_index()];
            let Some(delay_match) = child_slot.delay.clone() else {
                return Ok(None);
            };
            let area_match = child_slot
                .area
                .clone()
                .filter(|area_match| !same_match_choice(area_match, &delay_match));
            selected.push(delay_match);
            area_alternatives.push(area_match);
        }
        let required = requireds[state.class_id][state.polarity_index()];
        let delay_only = self.evaluate_with_child_matches(
            state,
            candidate,
            unique_states.as_slice(),
            selected.as_slice(),
            flow_refs,
            loads,
        )?;
        if delay_only.worst_arrival() > required + TIMING_EPSILON {
            return Ok(None);
        }

        let mut best = delay_only;
        for (child_index, area_match) in area_alternatives.into_iter().enumerate() {
            let Some(area_match) = area_match else {
                continue;
            };
            let previous = std::mem::replace(&mut selected[child_index], area_match);
            let trial = self.evaluate_with_child_matches(
                state,
                candidate,
                unique_states.as_slice(),
                selected.as_slice(),
                flow_refs,
                loads,
            )?;
            if trial.worst_arrival() <= required + TIMING_EPSILON {
                best = trial;
            } else {
                selected[child_index] = previous;
            }
        }
        Ok(Some(best))
    }

    fn evaluate_with_child_matches(
        &self,
        state: StateKey,
        candidate: &Candidate,
        unique_states: &[StateKey],
        child_matches: &[NfMatch],
        flow_refs: &[[f64; 2]],
        _loads: &[[CombinationalOutputLoad; 2]],
    ) -> Result<NfMatch> {
        let child_timings: SmallVec<[SignalTiming; MAX_TRUTH_TABLE_INPUTS]> = candidate
            .input_states
            .iter()
            .map(|input_state| {
                let slot = unique_states
                    .iter()
                    .position(|known_state| known_state == input_state)
                    .expect("candidate input state should have one unique slot");
                child_matches[slot].timing
            })
            .collect();
        let (timing, input_delays) = search_binding_timing(
            self.search_timing_model,
            &candidate.binding,
            child_timings.as_slice(),
        );
        let mut flow = candidate.binding.area;
        for child_match in child_matches {
            flow += child_match.flow;
        }
        flow /= flow_refs[state.class_id][state.polarity_index()].max(1.0);
        Ok(NfMatch {
            timing,
            flow,
            choice: MatchChoice::Cell {
                binding: candidate.binding.clone(),
                inputs: candidate.input_states.clone(),
            },
            input_delays,
        })
    }

    /// Evaluates one selected cell with the same rise/fall/load semantics as
    /// gv-stats. Trial matches deliberately use the much cheaper unit or
    /// scalar search model above, keeping the NF inner loop lightweight.
    fn evaluate_selected_binding_timing(
        &mut self,
        binding: &CellBinding,
        child_timings: &[SignalTiming],
        output_load: CombinationalOutputLoad,
    ) -> Result<(SignalTiming, SmallVec<[f64; MAX_TRUTH_TABLE_INPUTS]>)> {
        if binding.has_complete_timing() {
            let input_timings: SmallVec<[(&str, SignalTiming); MAX_TRUTH_TABLE_INPUTS]> = binding
                .input_pin_names
                .iter()
                .zip(child_timings.iter().copied())
                .map(|(pin_name, timing)| (pin_name.as_str(), timing))
                .collect();
            let mut diagnostics = TimingQueryDiagnosticCounts::default();
            let timing = evaluate_combinational_cell_output_timing(
                self.library,
                binding.cell_name.as_str(),
                binding.output_pin(self.library),
                input_timings.as_slice(),
                output_load,
                &self.empty_known_pin_values,
                &mut diagnostics,
            )?;
            let mut input_delays = SmallVec::new();
            for (input_index, child_timing) in child_timings.iter().copied().enumerate() {
                let one_input = [(binding.input_pin_names[input_index].as_str(), child_timing)];
                let delay = match evaluate_combinational_cell_output_timing(
                    self.library,
                    binding.cell_name.as_str(),
                    binding.output_pin(self.library),
                    one_input.as_slice(),
                    output_load,
                    &self.empty_known_pin_values,
                    &mut diagnostics,
                ) {
                    Ok(single_input_timing) => (worst_signal_arrival(single_input_timing)
                        - earliest_signal_arrival(child_timing))
                    .max(0.0),
                    Err(_) => binding.input_delays[input_index].unwrap_or(0.0),
                };
                input_delays.push(delay);
            }
            self.timing_query_diagnostic_counts += diagnostics;
            return Ok((timing, input_delays));
        }
        Ok(fallback_binding_timing(binding, child_timings))
    }

    fn add_inverter_closure(
        &mut self,
        class_id: usize,
        requireds: &[[f64; 2]],
        flow_refs: &[[f64; 2]],
        loads: &[[CombinationalOutputLoad; 2]],
    ) -> Result<()> {
        let inverters: Vec<CellBinding> = self
            .cell_index
            .matches(1, 0b01)
            .iter()
            .filter(|binding| !binding.input_negated[0])
            .cloned()
            .collect();
        if inverters.is_empty() {
            return Ok(());
        }
        for polarity in [false, true] {
            let state = StateKey { class_id, polarity };
            let opposite = StateKey {
                class_id,
                polarity: !polarity,
            };
            let opposite_slot = &self.matches[class_id][opposite.polarity_index()];
            let direct_delay = opposite_slot.direct_delay.clone();
            let direct_area = opposite_slot.direct_area.clone();
            for inverter in &inverters {
                let candidate = Candidate {
                    binding: inverter.clone(),
                    input_states: SmallVec::from_slice(&[opposite]),
                };
                if let Some(child) = direct_delay.clone() {
                    let candidate_match = self.evaluate_with_child_matches(
                        state,
                        &candidate,
                        &[opposite],
                        &[child],
                        flow_refs,
                        loads,
                    )?;
                    retain_strictly_faster_inverter(
                        &mut self.matches[class_id][state.polarity_index()].delay,
                        candidate_match,
                    );
                }
                if let Some(child) = direct_area.clone() {
                    let candidate_match = self.evaluate_with_child_matches(
                        state,
                        &candidate,
                        &[opposite],
                        &[child],
                        flow_refs,
                        loads,
                    )?;
                    if candidate_match.worst_arrival()
                        <= requireds[class_id][state.polarity_index()] + TIMING_EPSILON
                    {
                        retain_strictly_smaller_inverter(
                            &mut self.matches[class_id][state.polarity_index()].area,
                            candidate_match,
                        );
                    }
                }
            }
        }
        self.break_mutual_inverter_closure(class_id);
        Ok(())
    }

    /// Keeps at least one direct implementation when both polarities would
    /// otherwise be implemented as an inverter of the other polarity. ABC NF
    /// has the same invariant for its complemented matches.
    fn break_mutual_inverter_closure(&mut self, class_id: usize) {
        let positive = StateKey {
            class_id,
            polarity: false,
        };
        let negative = StateKey {
            class_id,
            polarity: true,
        };
        let positive_slot = self.matches[class_id][0].clone();
        let negative_slot = self.matches[class_id][1].clone();
        if matches_use_each_other(
            positive_slot.delay.as_ref(),
            positive,
            negative_slot.delay.as_ref(),
            negative,
        ) {
            let (positive_delay, negative_delay) = break_mutual_closure(
                positive_slot.delay,
                positive_slot.direct_delay,
                negative_slot.delay,
                negative_slot.direct_delay,
                nf_delay_order,
            );
            self.matches[class_id][0].delay = positive_delay;
            self.matches[class_id][1].delay = negative_delay;
        }
        if matches_use_each_other(
            positive_slot.area.as_ref(),
            positive,
            negative_slot.area.as_ref(),
            negative,
        ) {
            let (positive_area, negative_area) = break_mutual_closure(
                positive_slot.area,
                positive_slot.direct_area,
                negative_slot.area,
                negative_slot.direct_area,
                nf_area_order,
            );
            self.matches[class_id][0].area = positive_area;
            self.matches[class_id][1].area = negative_area;
        }
    }

    fn trace_selected(&self, prefer_area: bool, output_requireds: &[f64]) -> Result<MappingTrace> {
        let class_count = self.analysis.classes.len();
        let mut selected = option_pair_vec(class_count);
        let mut requireds = pair_vec(class_count, f64::INFINITY);
        for (output, required) in self.outputs.iter().zip(output_requireds.iter().copied()) {
            let mut visiting = HashSet::new();
            self.select_state_recursive(
                output.state,
                required,
                prefer_area,
                &mut selected,
                &mut requireds,
                &mut visiting,
            )?;
        }

        let mut loads = pair_vec(class_count, CombinationalOutputLoad::default());
        let mut map_refs = pair_vec(class_count, 0usize);
        let mut area = 0.0;
        for output in &self.outputs {
            add_output_load(
                &mut loads[output.state.class_id][output.state.polarity_index()],
                self.options.module_output_load,
            );
            reference_selected_state(
                output.state,
                selected.as_slice(),
                &mut map_refs,
                &mut loads,
                &mut area,
            )?;
        }
        let output_arrivals = self
            .outputs
            .iter()
            .map(|output| {
                selected[output.state.class_id][output.state.polarity_index()]
                    .as_ref()
                    .map(NfMatch::worst_arrival)
                    .ok_or_else(|| {
                        anyhow!(
                            "selected mapping is missing output '{}' state {:?}",
                            output.name,
                            output.state
                        )
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(MappingTrace {
            selected,
            requireds,
            loads,
            map_refs,
            output_arrivals,
            area,
        })
    }

    /// Runs one ABC-NF-style exact-area pass over the currently selected
    /// cover. Unlike area flow, this charges a replacement only for fanin
    /// cones that become newly referenced after the old match is removed.
    fn exact_area_recovery(
        &mut self,
        trace: MappingTrace,
        output_requireds: &[f64],
    ) -> Result<MappingTrace> {
        let mut current = if self.search_timing_model == SearchTimingModel::Unit {
            self.trace_fixed_selected(trace.selected, output_requireds)?
        } else {
            let mut selected = trace.selected;
            self.retime_selected(selected.as_mut_slice(), trace.loads.as_slice())?;
            self.trace_fixed_selected(selected, output_requireds)?
        };
        let before = current.clone();
        let unit_flow_refs = pair_vec(self.analysis.classes.len(), 1.0f64);
        let order = self.selected_preorder(current.selected.as_slice())?;
        let mut selected = current.selected;
        let mut map_refs = current.map_refs;
        let mut requireds = current.requireds;
        let loads = current.loads;

        for state in order {
            if map_refs[state.class_id][state.polarity_index()] == 0 {
                continue;
            }
            let current_match = selected[state.class_id][state.polarity_index()]
                .as_ref()
                .ok_or_else(|| anyhow!("exact-area pass is missing state {:?}", state))?
                .clone();
            if matches!(current_match.choice, MatchChoice::Source(_)) {
                continue;
            }

            let mut deref_visiting = HashSet::new();
            deref_visiting.insert(state);
            let area_before = dereference_match_children_exact(
                &current_match,
                map_refs.as_mut_slice(),
                selected.as_slice(),
                &mut deref_visiting,
            )?;
            let required = requireds[state.class_id][state.polarity_index()];
            let candidates = self.exact_area_candidates(
                state,
                selected.as_slice(),
                loads.as_slice(),
                unit_flow_refs.as_slice(),
            )?;
            let mut best_match = current_match.clone();
            let mut best_area = area_before;
            for candidate in candidates {
                if candidate.worst_arrival() > required + TIMING_EPSILON {
                    continue;
                }
                let mut increments = Vec::new();
                let mut newly_selected = Vec::new();
                let mut reference_visiting = HashSet::new();
                reference_visiting.insert(state);
                let area_after = match reference_match_children_exact(
                    &candidate,
                    map_refs.as_mut_slice(),
                    selected.as_slice(),
                    self.matches.as_slice(),
                    &mut increments,
                    &mut newly_selected,
                    &mut reference_visiting,
                ) {
                    Ok(area) => area,
                    Err(_) => {
                        undo_reference_increments(map_refs.as_mut_slice(), increments.as_slice());
                        continue;
                    }
                };
                undo_reference_increments(map_refs.as_mut_slice(), increments.as_slice());
                if exact_area_candidate_is_better(area_after, &candidate, best_area, &best_match) {
                    best_area = area_after;
                    best_match = candidate;
                }
            }

            let mut increments = Vec::new();
            let mut newly_selected = Vec::new();
            let mut reference_visiting = HashSet::new();
            reference_visiting.insert(state);
            reference_match_children_exact(
                &best_match,
                map_refs.as_mut_slice(),
                selected.as_slice(),
                self.matches.as_slice(),
                &mut increments,
                &mut newly_selected,
                &mut reference_visiting,
            )?;
            selected[state.class_id][state.polarity_index()] = Some(best_match.clone());
            for (new_state, new_match) in newly_selected {
                let slot = &mut selected[new_state.class_id][new_state.polarity_index()];
                if slot.is_none() {
                    *slot = Some(new_match);
                }
            }
            propagate_match_requireds(&best_match, required, requireds.as_mut_slice());
            let _ = best_area;
        }

        current = self.trace_fixed_selected(selected, output_requireds)?;
        if self.search_timing_model == SearchTimingModel::LibertyScalar {
            self.retime_selected(current.selected.as_mut_slice(), current.loads.as_slice())?;
            current = self.trace_fixed_selected(current.selected, output_requireds)?;
        }
        if current.area > before.area + TIMING_EPSILON
            || current
                .output_arrivals
                .iter()
                .zip(output_requireds.iter())
                .any(|(arrival, required)| arrival > &(required + TIMING_EPSILON))
        {
            return Ok(before);
        }
        Ok(current)
    }

    /// Returns timing-evaluated alternatives for one exact-area replacement.
    /// Unreferenced fanins use their delay match, matching NF's use of the
    /// current delay slot when a trial replacement makes a cone live.
    fn exact_area_candidates(
        &mut self,
        state: StateKey,
        selected: &[[Option<NfMatch>; 2]],
        loads: &[[CombinationalOutputLoad; 2]],
        flow_refs: &[[f64; 2]],
    ) -> Result<Vec<NfMatch>> {
        let mut matches = Vec::new();
        if let Some(current) = selected[state.class_id][state.polarity_index()].clone() {
            matches.push(current);
        }
        if let Some(delay) = self.matches[state.class_id][state.polarity_index()]
            .delay
            .clone()
        {
            matches.push(delay);
        }
        if let Some(area) = self.matches[state.class_id][state.polarity_index()]
            .area
            .clone()
        {
            matches.push(area);
        }
        for candidate in self.candidates_for_state(state) {
            let unique_states = unique_input_states(candidate.input_states.as_slice());
            let mut child_matches = Vec::with_capacity(unique_states.len());
            let mut available = true;
            for child_state in &unique_states {
                let Some(child_match) =
                    selected_or_delay_match(selected, self.matches.as_slice(), *child_state)
                else {
                    available = false;
                    break;
                };
                child_matches.push(child_match);
            }
            if !available {
                continue;
            }
            let candidate_match = self.evaluate_with_child_matches(
                state,
                &candidate,
                unique_states.as_slice(),
                child_matches.as_slice(),
                flow_refs,
                loads,
            )?;
            matches.push(candidate_match);
        }
        matches.sort_by(|lhs, rhs| {
            match_choice_order(&lhs.choice, &rhs.choice)
                .then_with(|| lhs.worst_arrival().total_cmp(&rhs.worst_arrival()))
        });
        matches.dedup_by(|lhs, rhs| same_match_choice(lhs, rhs));
        Ok(matches)
    }

    /// Re-evaluates every reachable selected match under the currently
    /// accumulated capacitive loads.
    fn retime_selected(
        &mut self,
        selected: &mut [[Option<NfMatch>; 2]],
        loads: &[[CombinationalOutputLoad; 2]],
    ) -> Result<()> {
        let mut visit_state = pair_vec(self.analysis.classes.len(), VisitState::Unvisited);
        let output_states: Vec<StateKey> = self.outputs.iter().map(|output| output.state).collect();
        for state in output_states {
            self.retime_selected_state(state, selected, loads, visit_state.as_mut_slice())?;
        }
        Ok(())
    }

    fn retime_selected_state(
        &mut self,
        state: StateKey,
        selected: &mut [[Option<NfMatch>; 2]],
        loads: &[[CombinationalOutputLoad; 2]],
        visit_state: &mut [[VisitState; 2]],
    ) -> Result<SignalTiming> {
        let state_index = state.polarity_index();
        match visit_state[state.class_id][state_index] {
            VisitState::Done => {
                return selected[state.class_id][state_index]
                    .as_ref()
                    .map(|selected_match| selected_match.timing)
                    .ok_or_else(|| anyhow!("retiming is missing state {:?}", state));
            }
            VisitState::Visiting => {
                return Err(anyhow!(
                    "selected technology mapping contains a timing cycle at {:?}",
                    state
                ));
            }
            VisitState::Unvisited => {}
        }
        visit_state[state.class_id][state_index] = VisitState::Visiting;
        let selected_match = selected[state.class_id][state_index]
            .as_ref()
            .ok_or_else(|| anyhow!("retiming is missing state {:?}", state))?
            .clone();
        let (timing, input_delays) = match &selected_match.choice {
            MatchChoice::Source(_) => (selected_match.timing, selected_match.input_delays.clone()),
            MatchChoice::Cell { binding, inputs } => {
                let mut child_timings: SmallVec<[SignalTiming; MAX_TRUTH_TABLE_INPUTS]> =
                    SmallVec::new();
                for input in inputs {
                    child_timings.push(self.retime_selected_state(
                        *input,
                        selected,
                        loads,
                        visit_state,
                    )?);
                }
                self.evaluate_selected_binding_timing(
                    binding,
                    child_timings.as_slice(),
                    loads[state.class_id][state_index],
                )?
            }
        };
        let slot = selected[state.class_id][state_index]
            .as_mut()
            .expect("selected match existed above");
        slot.timing = timing;
        slot.input_delays = input_delays;
        visit_state[state.class_id][state_index] = VisitState::Done;
        Ok(timing)
    }

    /// Recomputes refs, loads, required times, and output arrivals for a fixed
    /// selected cover without choosing different matches.
    fn trace_fixed_selected(
        &self,
        selected: Vec<[Option<NfMatch>; 2]>,
        output_requireds: &[f64],
    ) -> Result<MappingTrace> {
        if output_requireds.len() != self.outputs.len() {
            return Err(anyhow!(
                "expected {} output required times, got {}",
                self.outputs.len(),
                output_requireds.len()
            ));
        }
        let class_count = self.analysis.classes.len();
        let mut requireds = pair_vec(class_count, f64::INFINITY);
        for (output, required) in self.outputs.iter().zip(output_requireds.iter().copied()) {
            let mut visiting = HashSet::new();
            propagate_fixed_required(
                output.state,
                required,
                selected.as_slice(),
                requireds.as_mut_slice(),
                &mut visiting,
            )?;
        }
        let mut loads = pair_vec(class_count, CombinationalOutputLoad::default());
        let mut map_refs = pair_vec(class_count, 0usize);
        let mut area = 0.0;
        for output in &self.outputs {
            add_output_load(
                &mut loads[output.state.class_id][output.state.polarity_index()],
                self.options.module_output_load,
            );
            reference_selected_state(
                output.state,
                selected.as_slice(),
                map_refs.as_mut_slice(),
                loads.as_mut_slice(),
                &mut area,
            )?;
        }
        let output_arrivals = self
            .outputs
            .iter()
            .map(|output| {
                selected[output.state.class_id][output.state.polarity_index()]
                    .as_ref()
                    .map(NfMatch::worst_arrival)
                    .ok_or_else(|| {
                        anyhow!(
                            "fixed mapping is missing output '{}' state {:?}",
                            output.name,
                            output.state
                        )
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(MappingTrace {
            selected,
            requireds,
            loads,
            map_refs,
            output_arrivals,
            area,
        })
    }

    fn selected_preorder(&self, selected: &[[Option<NfMatch>; 2]]) -> Result<Vec<StateKey>> {
        let mut order = Vec::new();
        let mut visited = HashSet::new();
        let mut visiting = HashSet::new();
        for output in &self.outputs {
            collect_selected_preorder(
                output.state,
                selected,
                &mut order,
                &mut visited,
                &mut visiting,
            )?;
        }
        Ok(order)
    }

    fn select_state_recursive(
        &self,
        state: StateKey,
        required: f64,
        prefer_area: bool,
        selected: &mut [[Option<NfMatch>; 2]],
        requireds: &mut [[f64; 2]],
        visiting: &mut HashSet<StateKey>,
    ) -> Result<()> {
        let previous_required = requireds[state.class_id][state.polarity_index()];
        if selected[state.class_id][state.polarity_index()].is_some()
            && required >= previous_required - TIMING_EPSILON
        {
            return Ok(());
        }
        if !visiting.insert(state) {
            return Err(anyhow!(
                "selected technology mapping contains a choice-state cycle at {:?}",
                state
            ));
        }
        requireds[state.class_id][state.polarity_index()] = previous_required.min(required);
        let effective_required = requireds[state.class_id][state.polarity_index()];
        let state_matches = &self.matches[state.class_id][state.polarity_index()];
        let selected_match = if prefer_area {
            state_matches
                .area
                .as_ref()
                .filter(|candidate| {
                    candidate.worst_arrival() <= effective_required + TIMING_EPSILON
                })
                .or(state_matches.delay.as_ref())
        } else {
            state_matches.delay.as_ref()
        }
        .ok_or_else(|| anyhow!("no selected match exists for state {:?}", state))?
        .clone();
        selected[state.class_id][state.polarity_index()] = Some(selected_match.clone());
        if let MatchChoice::Cell { inputs, .. } = &selected_match.choice {
            for (input_index, input_state) in inputs.iter().copied().enumerate() {
                let child_required = if effective_required.is_finite() {
                    effective_required - selected_match.input_delays[input_index]
                } else {
                    f64::INFINITY
                };
                self.select_state_recursive(
                    input_state,
                    child_required,
                    prefer_area,
                    selected,
                    requireds,
                    visiting,
                )?;
            }
        }
        visiting.remove(&state);
        Ok(())
    }

    fn materialize_selected(
        &self,
        selected: &[[Option<NfMatch>; 2]],
        output_arrivals: &[f64],
    ) -> Result<CoverPlan> {
        let mut solutions = Vec::new();
        let mut memo = option_pair_vec(self.analysis.classes.len());
        let mut output_solutions = Vec::with_capacity(self.outputs.len());
        let mut visiting = HashSet::new();
        for output in &self.outputs {
            output_solutions.push(materialize_state(
                output.state,
                selected,
                &mut memo,
                &mut solutions,
                &mut visiting,
            )?);
        }
        Ok(CoverPlan {
            solutions,
            output_solutions,
            output_arrivals: output_arrivals.to_vec(),
            matched_candidate_count: self.matched_candidate_count,
        })
    }
}

fn source_nf_match(source: SourceKind, arrival: f64, transition: f64) -> NfMatch {
    NfMatch {
        timing: SignalTiming {
            rise: EdgeTiming {
                arrival,
                transition,
            },
            fall: EdgeTiming {
                arrival,
                transition,
            },
        },
        flow: 0.0,
        choice: MatchChoice::Source(source),
        input_delays: SmallVec::new(),
    }
}

/// Evaluates a trial match under the timing objective used by the NF rounds.
///
/// Unit mode intentionally ignores Liberty tables and capacitive load: it
/// models the generated genlib convention used by ABC's normal `&nf` flow.
/// The selected final cover is separately re-evaluated with full Liberty
/// timing before it is returned.
fn search_binding_timing(
    model: SearchTimingModel,
    binding: &CellBinding,
    child_timings: &[SignalTiming],
) -> (SignalTiming, SmallVec<[f64; MAX_TRUTH_TABLE_INPUTS]>) {
    match model {
        SearchTimingModel::Unit => unit_binding_timing(child_timings),
        SearchTimingModel::LibertyScalar => fallback_binding_timing(binding, child_timings),
    }
}

fn unit_binding_timing(
    child_timings: &[SignalTiming],
) -> (SignalTiming, SmallVec<[f64; MAX_TRUTH_TABLE_INPUTS]>) {
    let mut worst_arrival: f64 = 0.0;
    let mut transition: f64 = 0.0;
    let mut input_delays = SmallVec::new();
    for child_timing in child_timings.iter().copied() {
        worst_arrival = worst_arrival.max(worst_signal_arrival(child_timing) + 1.0);
        transition = transition.max(max_signal_transition(child_timing));
        input_delays.push(1.0);
    }
    (
        SignalTiming {
            rise: EdgeTiming {
                arrival: worst_arrival,
                transition,
            },
            fall: EdgeTiming {
                arrival: worst_arrival,
                transition,
            },
        },
        input_delays,
    )
}

fn fallback_binding_timing(
    binding: &CellBinding,
    child_timings: &[SignalTiming],
) -> (SignalTiming, SmallVec<[f64; MAX_TRUTH_TABLE_INPUTS]>) {
    let mut worst_arrival: f64 = 0.0;
    let mut transition: f64 = 0.0;
    let mut input_delays = SmallVec::new();
    for (input_index, child_timing) in child_timings.iter().copied().enumerate() {
        let delay = binding.input_delays[input_index].unwrap_or(0.0);
        worst_arrival = worst_arrival.max(worst_signal_arrival(child_timing) + delay);
        transition = transition.max(max_signal_transition(child_timing));
        input_delays.push(delay);
    }
    (
        SignalTiming {
            rise: EdgeTiming {
                arrival: worst_arrival,
                transition,
            },
            fall: EdgeTiming {
                arrival: worst_arrival,
                transition,
            },
        },
        input_delays,
    )
}

fn unique_input_states(states: &[StateKey]) -> SmallVec<[StateKey; MAX_TRUTH_TABLE_INPUTS]> {
    let mut unique = SmallVec::new();
    for state in states {
        if !unique.contains(state) {
            unique.push(*state);
        }
    }
    unique.sort();
    unique
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

fn insert_candidate_variant(
    by_input_states: &mut BTreeMap<SmallVec<[StateKey; MAX_TRUTH_TABLE_INPUTS]>, Vec<Candidate>>,
    binding: &CellBinding,
    input_states: SmallVec<[StateKey; MAX_TRUTH_TABLE_INPUTS]>,
    limit: usize,
) {
    let variants = by_input_states.entry(input_states.clone()).or_default();
    if variants
        .iter()
        .any(|candidate| binding_dominates(&candidate.binding, binding))
    {
        return;
    }
    variants.retain(|candidate| !binding_dominates(binding, &candidate.binding));
    variants.push(Candidate {
        binding: binding.clone(),
        input_states,
    });
    variants.sort_by(candidate_order);
    variants.truncate(limit);
}

fn binding_dominates(lhs: &CellBinding, rhs: &CellBinding) -> bool {
    let lhs_delay = lhs.worst_nominal_delay();
    let rhs_delay = rhs.worst_nominal_delay();
    lhs.area <= rhs.area
        && lhs_delay <= rhs_delay
        && (lhs.area < rhs.area || lhs_delay < rhs_delay || lhs.stable_key() <= rhs.stable_key())
}

fn candidate_order(lhs: &Candidate, rhs: &Candidate) -> std::cmp::Ordering {
    lhs.input_states
        .cmp(&rhs.input_states)
        .then_with(|| lhs.binding.area.total_cmp(&rhs.binding.area))
        .then_with(|| {
            lhs.binding
                .worst_nominal_delay()
                .total_cmp(&rhs.binding.worst_nominal_delay())
        })
        .then_with(|| lhs.binding.stable_key().cmp(&rhs.binding.stable_key()))
}

fn retain_best_delay(slot: &mut Option<NfMatch>, candidate: NfMatch) {
    match slot {
        Some(existing) if nf_delay_order(existing, &candidate).is_le() => {}
        _ => *slot = Some(candidate),
    }
}

fn retain_best_area(slot: &mut Option<NfMatch>, candidate: NfMatch) {
    match slot {
        Some(existing) if nf_area_order(existing, &candidate).is_le() => {}
        _ => *slot = Some(candidate),
    }
}

/// NF does not replace a direct delay match with an inverter closure on an
/// equal-delay tie; doing so can create gratuitous inverter chains.
fn retain_strictly_faster_inverter(slot: &mut Option<NfMatch>, candidate: NfMatch) {
    match slot {
        Some(existing)
            if candidate.worst_arrival() + TIMING_EPSILON >= existing.worst_arrival() => {}
        _ => *slot = Some(candidate),
    }
}

/// NF only uses an inverter closure for area when it is a strict flow
/// improvement, preserving the direct implementation on equal-area ties.
fn retain_strictly_smaller_inverter(slot: &mut Option<NfMatch>, candidate: NfMatch) {
    match slot {
        Some(existing) if candidate.flow + TIMING_EPSILON >= existing.flow => {}
        _ => *slot = Some(candidate),
    }
}

fn nf_delay_order(lhs: &NfMatch, rhs: &NfMatch) -> std::cmp::Ordering {
    lhs.worst_arrival()
        .total_cmp(&rhs.worst_arrival())
        .then_with(|| lhs.flow.total_cmp(&rhs.flow))
        .then_with(|| match_choice_order(&lhs.choice, &rhs.choice))
}

fn nf_area_order(lhs: &NfMatch, rhs: &NfMatch) -> std::cmp::Ordering {
    lhs.flow
        .total_cmp(&rhs.flow)
        .then_with(|| lhs.worst_arrival().total_cmp(&rhs.worst_arrival()))
        .then_with(|| match_choice_order(&lhs.choice, &rhs.choice))
}

fn same_match_choice(lhs: &NfMatch, rhs: &NfMatch) -> bool {
    match_choice_order(&lhs.choice, &rhs.choice).is_eq()
}

fn matches_use_each_other(
    lhs: Option<&NfMatch>,
    lhs_state: StateKey,
    rhs: Option<&NfMatch>,
    rhs_state: StateKey,
) -> bool {
    lhs.is_some_and(|selected_match| match_uses_only_state(selected_match, rhs_state))
        && rhs.is_some_and(|selected_match| match_uses_only_state(selected_match, lhs_state))
}

fn match_uses_only_state(selected_match: &NfMatch, state: StateKey) -> bool {
    matches!(
        &selected_match.choice,
        MatchChoice::Cell { inputs, .. } if inputs.as_slice() == [state]
    )
}

fn break_mutual_closure(
    positive: Option<NfMatch>,
    positive_direct: Option<NfMatch>,
    negative: Option<NfMatch>,
    negative_direct: Option<NfMatch>,
    order: fn(&NfMatch, &NfMatch) -> std::cmp::Ordering,
) -> (Option<NfMatch>, Option<NfMatch>) {
    match (positive_direct, negative_direct) {
        (Some(positive_direct), Some(negative_direct)) => {
            if order(&positive_direct, &negative_direct).is_le() {
                (Some(positive_direct), negative)
            } else {
                (positive, Some(negative_direct))
            }
        }
        (Some(positive_direct), None) => (Some(positive_direct), negative),
        (None, Some(negative_direct)) => (positive, Some(negative_direct)),
        (None, None) => (positive, negative),
    }
}

fn match_choice_order(lhs: &MatchChoice, rhs: &MatchChoice) -> std::cmp::Ordering {
    match (lhs, rhs) {
        (MatchChoice::Source(lhs), MatchChoice::Source(rhs)) => {
            source_order_key(*lhs).cmp(&source_order_key(*rhs))
        }
        (MatchChoice::Source(_), MatchChoice::Cell { .. }) => std::cmp::Ordering::Less,
        (MatchChoice::Cell { .. }, MatchChoice::Source(_)) => std::cmp::Ordering::Greater,
        (
            MatchChoice::Cell {
                binding: lhs_binding,
                inputs: lhs_inputs,
            },
            MatchChoice::Cell {
                binding: rhs_binding,
                inputs: rhs_inputs,
            },
        ) => lhs_binding
            .stable_key()
            .cmp(&rhs_binding.stable_key())
            .then_with(|| lhs_inputs.cmp(rhs_inputs)),
    }
}

fn source_order_key(source: SourceKind) -> (u8, usize, bool) {
    match source {
        SourceKind::Input(node) => (0, node.id, false),
        SourceKind::Literal(value) => (1, 0, value),
    }
}

fn worst_signal_arrival(timing: SignalTiming) -> f64 {
    timing.rise.arrival.max(timing.fall.arrival)
}

fn earliest_signal_arrival(timing: SignalTiming) -> f64 {
    timing.rise.arrival.min(timing.fall.arrival)
}

fn max_signal_transition(timing: SignalTiming) -> f64 {
    timing.rise.transition.max(timing.fall.transition)
}

fn pair_vec<T: Clone>(len: usize, value: T) -> Vec<[T; 2]> {
    (0..len).map(|_| [value.clone(), value.clone()]).collect()
}

fn option_pair_vec<T>(len: usize) -> Vec<[Option<T>; 2]> {
    (0..len).map(|_| [None, None]).collect()
}

fn add_output_load(load: &mut CombinationalOutputLoad, extra: f64) {
    load.rise += extra;
    load.fall += extra;
}

fn add_pin_load(load: &mut CombinationalOutputLoad, pin_load: CombinationalOutputLoad) {
    load.rise += pin_load.rise;
    load.fall += pin_load.fall;
}

fn blend_flow_refs(flow_refs: &mut [[f64; 2]], map_refs: &[[usize; 2]], round: usize) {
    let coefficient = 1.0 / (1.0 + ((round + 1) * (round + 1)) as f64);
    for class_id in 0..flow_refs.len() {
        for polarity in [false, true] {
            let index = usize::from(polarity);
            let actual = map_refs[class_id][index].max(1) as f64;
            flow_refs[class_id][index] =
                (coefficient * flow_refs[class_id][index] + (1.0 - coefficient) * actual).max(1.0);
        }
    }
}

fn reference_selected_state(
    state: StateKey,
    selected: &[[Option<NfMatch>; 2]],
    map_refs: &mut [[usize; 2]],
    loads: &mut [[CombinationalOutputLoad; 2]],
    area: &mut f64,
) -> Result<()> {
    let ref_count = &mut map_refs[state.class_id][state.polarity_index()];
    *ref_count += 1;
    if *ref_count > 1 {
        return Ok(());
    }
    let selected_match = selected[state.class_id][state.polarity_index()]
        .as_ref()
        .ok_or_else(|| anyhow!("selected mapping is missing state {:?}", state))?;
    let MatchChoice::Cell { binding, inputs } = &selected_match.choice else {
        return Ok(());
    };
    *area += binding.area;
    for (input_index, input_state) in inputs.iter().copied().enumerate() {
        add_pin_load(
            &mut loads[input_state.class_id][input_state.polarity_index()],
            binding.input_capacitances[input_index],
        );
        reference_selected_state(input_state, selected, map_refs, loads, area)?;
    }
    Ok(())
}

fn selected_or_delay_match(
    selected: &[[Option<NfMatch>; 2]],
    matches: &[[StateMatches; 2]],
    state: StateKey,
) -> Option<NfMatch> {
    selected[state.class_id][state.polarity_index()]
        .clone()
        .or_else(|| {
            matches[state.class_id][state.polarity_index()]
                .delay
                .clone()
        })
}

fn exact_area_candidate_is_better(
    candidate_area: f64,
    candidate: &NfMatch,
    best_area: f64,
    best: &NfMatch,
) -> bool {
    candidate_area + TIMING_EPSILON < best_area
        || ((candidate_area - best_area).abs() <= TIMING_EPSILON
            && nf_delay_order(candidate, best).is_lt())
}

fn match_cell_area(selected_match: &NfMatch) -> f64 {
    match &selected_match.choice {
        MatchChoice::Source(_) => 0.0,
        MatchChoice::Cell { binding, .. } => binding.area,
    }
}

/// Removes one selected cell's fanin cone from global refs while leaving the
/// root reference itself in place, like ABC's Nf_MatchDeref_rec.
fn dereference_match_children_exact(
    selected_match: &NfMatch,
    map_refs: &mut [[usize; 2]],
    selected: &[[Option<NfMatch>; 2]],
    visiting: &mut HashSet<StateKey>,
) -> Result<f64> {
    let mut area = match_cell_area(selected_match);
    let MatchChoice::Cell { inputs, .. } = &selected_match.choice else {
        return Ok(area);
    };
    for input in inputs {
        let ref_count = &mut map_refs[input.class_id][input.polarity_index()];
        if *ref_count == 0 {
            return Err(anyhow!(
                "exact-area dereference found zero refs for child {:?}",
                input
            ));
        }
        *ref_count -= 1;
        if *ref_count != 0 {
            continue;
        }
        if !visiting.insert(*input) {
            return Err(anyhow!(
                "selected technology mapping contains an exact-area cycle at {:?}",
                input
            ));
        }
        let child_match = selected[input.class_id][input.polarity_index()]
            .as_ref()
            .ok_or_else(|| anyhow!("exact-area dereference is missing child {:?}", input))?;
        area += dereference_match_children_exact(child_match, map_refs, selected, visiting)?;
        visiting.remove(input);
    }
    Ok(area)
}

/// References one trial cell's fanin cone and returns the exact area newly
/// made live. Every increment is logged so a rejected trial can be undone
/// without cloning the full reference-count vector.
#[allow(clippy::too_many_arguments)]
fn reference_match_children_exact(
    selected_match: &NfMatch,
    map_refs: &mut [[usize; 2]],
    selected: &[[Option<NfMatch>; 2]],
    matches: &[[StateMatches; 2]],
    increments: &mut Vec<StateKey>,
    newly_selected: &mut Vec<(StateKey, NfMatch)>,
    visiting: &mut HashSet<StateKey>,
) -> Result<f64> {
    let mut area = match_cell_area(selected_match);
    let MatchChoice::Cell { inputs, .. } = &selected_match.choice else {
        return Ok(area);
    };
    for input in inputs {
        let ref_count = &mut map_refs[input.class_id][input.polarity_index()];
        let was_unreferenced = *ref_count == 0;
        *ref_count += 1;
        increments.push(*input);
        if !was_unreferenced {
            continue;
        }
        if !visiting.insert(*input) {
            return Err(anyhow!(
                "trial technology mapping contains an exact-area cycle at {:?}",
                input
            ));
        }
        let child_match = selected_or_delay_match(selected, matches, *input)
            .ok_or_else(|| anyhow!("exact-area trial has no child match for {:?}", input))?;
        if selected[input.class_id][input.polarity_index()].is_none() {
            newly_selected.push((*input, child_match.clone()));
        }
        area += reference_match_children_exact(
            &child_match,
            map_refs,
            selected,
            matches,
            increments,
            newly_selected,
            visiting,
        )?;
        visiting.remove(input);
    }
    Ok(area)
}

fn undo_reference_increments(map_refs: &mut [[usize; 2]], increments: &[StateKey]) {
    for state in increments.iter().rev() {
        let ref_count = &mut map_refs[state.class_id][state.polarity_index()];
        debug_assert!(*ref_count > 0);
        *ref_count -= 1;
    }
}

fn propagate_match_requireds(selected_match: &NfMatch, required: f64, requireds: &mut [[f64; 2]]) {
    if !required.is_finite() {
        return;
    }
    let MatchChoice::Cell { inputs, .. } = &selected_match.choice else {
        return;
    };
    for (input_index, input) in inputs.iter().enumerate() {
        let child_required = required - selected_match.input_delays[input_index];
        let slot = &mut requireds[input.class_id][input.polarity_index()];
        *slot = slot.min(child_required);
    }
}

fn propagate_fixed_required(
    state: StateKey,
    required: f64,
    selected: &[[Option<NfMatch>; 2]],
    requireds: &mut [[f64; 2]],
    visiting: &mut HashSet<StateKey>,
) -> Result<()> {
    let previous = requireds[state.class_id][state.polarity_index()];
    if required >= previous - TIMING_EPSILON {
        return Ok(());
    }
    if !visiting.insert(state) {
        return Err(anyhow!(
            "fixed technology mapping contains a required-time cycle at {:?}",
            state
        ));
    }
    requireds[state.class_id][state.polarity_index()] = previous.min(required);
    let effective_required = requireds[state.class_id][state.polarity_index()];
    let selected_match = selected[state.class_id][state.polarity_index()]
        .as_ref()
        .ok_or_else(|| anyhow!("fixed mapping is missing state {:?}", state))?;
    if let MatchChoice::Cell { inputs, .. } = &selected_match.choice {
        for (input_index, input) in inputs.iter().enumerate() {
            let child_required = if effective_required.is_finite() {
                effective_required - selected_match.input_delays[input_index]
            } else {
                f64::INFINITY
            };
            propagate_fixed_required(*input, child_required, selected, requireds, visiting)?;
        }
    }
    visiting.remove(&state);
    Ok(())
}

fn collect_selected_preorder(
    state: StateKey,
    selected: &[[Option<NfMatch>; 2]],
    order: &mut Vec<StateKey>,
    visited: &mut HashSet<StateKey>,
    visiting: &mut HashSet<StateKey>,
) -> Result<()> {
    if visited.contains(&state) {
        return Ok(());
    }
    if !visiting.insert(state) {
        return Err(anyhow!(
            "selected technology mapping contains a traversal cycle at {:?}",
            state
        ));
    }
    order.push(state);
    let selected_match = selected[state.class_id][state.polarity_index()]
        .as_ref()
        .ok_or_else(|| anyhow!("selected mapping is missing state {:?}", state))?;
    if let MatchChoice::Cell { inputs, .. } = &selected_match.choice {
        for input in inputs {
            collect_selected_preorder(*input, selected, order, visited, visiting)?;
        }
    }
    visiting.remove(&state);
    visited.insert(state);
    Ok(())
}

fn materialize_state(
    state: StateKey,
    selected: &[[Option<NfMatch>; 2]],
    memo: &mut [[Option<SolutionId>; 2]],
    solutions: &mut Vec<Solution>,
    visiting: &mut HashSet<StateKey>,
) -> Result<SolutionId> {
    if let Some(id) = memo[state.class_id][state.polarity_index()] {
        return Ok(id);
    }
    if !visiting.insert(state) {
        return Err(anyhow!(
            "selected technology mapping contains a reconstruction cycle at {:?}",
            state
        ));
    }
    let selected_match = selected[state.class_id][state.polarity_index()]
        .as_ref()
        .ok_or_else(|| anyhow!("selected mapping is missing state {:?}", state))?;
    let choice = match &selected_match.choice {
        MatchChoice::Source(source) => SolutionChoice::Source(*source),
        MatchChoice::Cell { binding, inputs } => {
            let mut solution_inputs = Vec::with_capacity(inputs.len());
            for input in inputs {
                solution_inputs.push(materialize_state(
                    *input, selected, memo, solutions, visiting,
                )?);
            }
            SolutionChoice::Cell {
                binding: binding.clone(),
                inputs: solution_inputs,
            }
        }
    };
    let id = SolutionId(solutions.len());
    solutions.push(Solution { choice });
    memo[state.class_id][state.polarity_index()] = Some(id);
    visiting.remove(&state);
    Ok(id)
}
