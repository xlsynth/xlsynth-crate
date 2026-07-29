// SPDX-License-Identifier: Apache-2.0

//! NF-style choice-aware cut matching and iterative cover selection.

use crate::aig::{AigNode, AigRef, ChoiceAig};
use crate::liberty_model::Library;
use crate::netlist::sta::{
    CombinationalOutputLoad, EdgeTiming, SignalTiming, TimingQueryDiagnosticCounts,
    evaluate_combinational_cell_output_timing,
};
use crate::techmap::cuts::{Cut, structural_fanout_estimates};
use crate::techmap::liberty_index::{CellBinding, CellBindingId, LibertyCellIndex};
use crate::techmap::truth::{MAX_TRUTH_TABLE_INPUTS, complement_truth, variable_truth};
use crate::techmap::{
    TechMapOptions, TechMapTimingConstraints, TechMapTimingModel, scalar_bit_name,
};
use anyhow::{Result, anyhow};
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

const NF_AREA_FLOW_ROUNDS: usize = 4;
const NF_EXACT_AREA_ROUNDS: usize = 2;
const TIMING_EPSILON: f64 = 1e-9;
const MAX_CELL_VARIANTS_PER_SIGNATURE: usize = 8;

/// One ABC-like GIA object plus output polarity.
///
/// NF keeps mapping matches on each concrete GIA object. Structural choices
/// contribute sibling cuts to an object, but they do not collapse all sibling
/// objects into one shared mapping state.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(super) struct StateKey {
    pub node_id: usize,
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

/// Child-state layout shared by every Liberty variant of one cut signature.
///
/// Multiple cells can realize the same ordered input-state vector. Keeping
/// the de-duplicated child state information here avoids rebuilding it once
/// per cell variant while NF scans a candidate frontier.
#[derive(Clone, Debug)]
struct CandidateLayout {
    /// One concrete AIG-object state per cell input pin.
    input_states: SmallVec<[StateKey; MAX_TRUTH_TABLE_INPUTS]>,
    /// Sorted de-duplicated child states used by NF's shared-fanin flow.
    unique_input_states: SmallVec<[StateKey; MAX_TRUTH_TABLE_INPUTS]>,
    /// Unique-child slot for each cell input pin.
    input_state_slots: SmallVec<[usize; MAX_TRUTH_TABLE_INPUTS]>,
}

impl CandidateLayout {
    /// Builds the reusable child-state layout for one cut signature.
    fn new(input_states: SmallVec<[StateKey; MAX_TRUTH_TABLE_INPUTS]>) -> Self {
        let unique_input_states = unique_input_states(input_states.as_slice());
        let input_state_slots = input_states
            .iter()
            .map(|input_state| {
                unique_input_states
                    .iter()
                    .position(|known_state| known_state == input_state)
                    .expect("candidate input state should have one unique slot")
            })
            .collect();
        Self {
            input_states,
            unique_input_states,
            input_state_slots,
        }
    }
}

/// Non-dominated Liberty bindings sharing one child-state layout.
#[derive(Clone, Debug)]
struct CandidateGroup {
    layout: CandidateLayout,
    bindings: SmallVec<[CellBindingId; MAX_CELL_VARIANTS_PER_SIGNATURE]>,
}

#[derive(Clone, Debug)]
enum MatchChoice {
    Source(SourceKind),
    Cell {
        binding: CellBindingId,
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

/// Lightweight timing/flow result used while trying child-match variants.
#[derive(Clone, Copy, Debug)]
struct CandidateScore {
    timing: SignalTiming,
    flow: f64,
}

impl CandidateScore {
    fn worst_arrival(&self) -> f64 {
        worst_signal_arrival(self.timing)
    }
}

/// Copyable child timing/flow needed while scoring a candidate.
#[derive(Clone, Copy, Debug)]
struct MatchSummary {
    timing: SignalTiming,
    flow: f64,
}

impl From<&NfMatch> for MatchSummary {
    fn from(value: &NfMatch) -> Self {
        Self {
            timing: value.timing,
            flow: value.flow,
        }
    }
}

/// Delay and area-under-required-time scores from one candidate traversal.
struct EvaluatedCandidate {
    delay: CandidateScore,
    area: Option<CandidateScore>,
    input_delays: SmallVec<[f64; MAX_TRUTH_TABLE_INPUTS]>,
}

/// Child matches resolved once for every binding in a candidate group.
struct CandidateGroupChildren {
    selected: SmallVec<[MatchSummary; MAX_TRUTH_TABLE_INPUTS]>,
    area_alternatives: SmallVec<[Option<MatchSummary>; MAX_TRUTH_TABLE_INPUTS]>,
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
/// Cover search stays scalar; exact rise/fall NLDM is reserved for final STA.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SearchTimingModel {
    Unit,
    BufferedLiberty,
    LibertyScalar,
}

/// Cheap electrical assumptions supplied by the actual later buffer pass.
#[derive(Clone, Copy, Debug)]
struct BufferedSearchParameters {
    target_load: Option<f64>,
    buffer_delay: f64,
    max_fanout: usize,
}

impl BufferedSearchParameters {
    /// Derives the search bound from a real noninverting Liberty buffer.
    fn new(cell_index: &LibertyCellIndex, library: &Library, options: &TechMapOptions) -> Self {
        let Some(buffer_options) = &options.buffer_options else {
            return Self {
                target_load: None,
                buffer_delay: 0.0,
                max_fanout: 8,
            };
        };
        let buffer = cell_index
            .matches(1, variable_truth(1, 0))
            .iter()
            .map(|binding| cell_index.binding(*binding))
            .filter(|binding| !binding.input_negated[0])
            .min_by(|lhs, rhs| {
                lhs.area
                    .total_cmp(&rhs.area)
                    .then_with(|| {
                        lhs.worst_nominal_delay()
                            .total_cmp(&rhs.worst_nominal_delay())
                    })
                    .then_with(|| lhs.stable_key().cmp(&rhs.stable_key()))
            });
        let max_fanout = buffer_options.max_fanout.max(2);
        let target_load = buffer_options.target_load.or_else(|| {
            buffer.and_then(|binding| {
                binding
                    .output_pin(library)
                    .max_capacitance
                    .filter(|limit| limit.is_finite() && *limit > 0.0)
                    .or_else(|| {
                        binding.input_capacitances.first().and_then(|load| {
                            let limit = load.rise.max(load.fall) * max_fanout as f64;
                            (limit.is_finite() && limit > 0.0).then_some(limit)
                        })
                    })
            })
        });
        Self {
            target_load,
            buffer_delay: buffer
                .map(CellBinding::worst_nominal_delay)
                .filter(|delay| delay.is_finite() && *delay > 0.0)
                .unwrap_or(0.0),
            max_fanout,
        }
    }
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

/// Dense recursion-path marks reused by exact-area trials.
///
/// Mapping states are a compact node/polarity domain, so a generation-marked
/// vector is cheaper than allocating and hashing a HashSet<StateKey> for
/// every candidate trial. A new generation also makes an aborted traversal
/// harmless without clearing the whole vector.
struct DenseVisitMarks {
    marks: Vec<u32>,
    generation: u32,
}

impl DenseVisitMarks {
    fn new(node_count: usize) -> Self {
        Self {
            marks: vec![0; node_count * 2],
            generation: 0,
        }
    }

    fn begin(&mut self, root: StateKey) {
        if self.generation == u32::MAX {
            self.marks.fill(0);
            self.generation = 1;
        } else {
            self.generation += 1;
        }
        debug_assert!(self.enter(root));
    }

    fn enter(&mut self, state: StateKey) -> bool {
        let index = state.node_id * 2 + state.polarity_index();
        if self.marks[index] == self.generation {
            return false;
        }
        self.marks[index] = self.generation;
        true
    }

    fn leave(&mut self, state: StateKey) {
        let index = state.node_id * 2 + state.polarity_index();
        debug_assert_eq!(self.marks[index], self.generation);
        self.marks[index] = 0;
    }
}

/// Reusable exact-area trial buffers.
struct ExactAreaScratch {
    increments: Vec<StateKey>,
    newly_selected: Vec<StateKey>,
    visiting: DenseVisitMarks,
}

impl ExactAreaScratch {
    fn new(node_count: usize) -> Self {
        Self {
            increments: Vec::new(),
            newly_selected: Vec::new(),
            visiting: DenseVisitMarks::new(node_count),
        }
    }

    fn begin(&mut self, root: StateKey) {
        self.increments.clear();
        self.newly_selected.clear();
        self.visiting.begin(root);
    }
}

/// Builds an NF-shaped cover: fastest mapping first, then area-flow mapping
/// under backward-propagated required times.
pub(super) fn build_cover_plan(
    choice_aig: &ChoiceAig,
    cuts_by_node: &[Vec<Cut>],
    cell_index: &LibertyCellIndex,
    library: &Library,
    options: &TechMapOptions,
    constraints: &TechMapTimingConstraints,
    retime_approximate_cover_for_fallback: bool,
) -> Result<CoverPlan> {
    let mut builder = CoverBuilder::new(
        choice_aig,
        cuts_by_node,
        cell_index,
        library,
        options,
        constraints,
        retime_approximate_cover_for_fallback,
    )?;
    builder.build()
}

struct CoverBuilder<'a> {
    choice_aig: &'a ChoiceAig,
    cuts_by_node: &'a [Vec<Cut>],
    cell_index: &'a LibertyCellIndex,
    library: &'a Library,
    options: &'a TechMapOptions,
    constraints: &'a TechMapTimingConstraints,
    search_timing_model: SearchTimingModel,
    buffered_search: BufferedSearchParameters,
    /// Whether approximate cover modes need fallback Liberty arrivals.
    ///
    /// Timing-complete emitted netlists run a later full STA pass, so they can
    /// skip this otherwise duplicate traversal.
    retime_approximate_cover_for_fallback: bool,
    input_arrival_by_node: HashMap<usize, f64>,
    outputs: Vec<OutputEndpoint>,
    candidates: Vec<[Option<Vec<CandidateGroup>>; 2]>,
    /// Output-reachable nodes in ascending AIG order, filled with candidates.
    cached_reachable_node_order: Option<Vec<usize>>,
    matches: Vec<[StateMatches; 2]>,
    matched_candidate_count: usize,
    timing_query_diagnostic_counts: TimingQueryDiagnosticCounts,
    empty_known_pin_values: HashMap<String, bool>,
}

impl<'a> CoverBuilder<'a> {
    fn new(
        choice_aig: &'a ChoiceAig,
        cuts_by_node: &'a [Vec<Cut>],
        cell_index: &'a LibertyCellIndex,
        library: &'a Library,
        options: &'a TechMapOptions,
        constraints: &'a TechMapTimingConstraints,
        retime_approximate_cover_for_fallback: bool,
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
                outputs.push(OutputEndpoint {
                    name,
                    state: StateKey {
                        node_id: operand.node.id,
                        polarity: operand.negated,
                    },
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

        let node_count = graph.gates.len();
        let search_timing_model = if constraints.primary_input_arrivals.is_empty()
            && constraints.primary_output_required.is_empty()
        {
            match options.timing_model {
                TechMapTimingModel::Balanced | TechMapTimingModel::NfUnit => {
                    SearchTimingModel::Unit
                }
                TechMapTimingModel::NfLiberty => SearchTimingModel::LibertyScalar,
                TechMapTimingModel::BufferedLiberty => SearchTimingModel::BufferedLiberty,
            }
        } else {
            SearchTimingModel::LibertyScalar
        };
        let buffered_search = BufferedSearchParameters::new(cell_index, library, options);
        Ok(Self {
            choice_aig,
            cuts_by_node,
            cell_index,
            library,
            options,
            constraints,
            search_timing_model,
            buffered_search,
            retime_approximate_cover_for_fallback,
            input_arrival_by_node,
            outputs,
            candidates: (0..node_count).map(|_| [None, None]).collect(),
            cached_reachable_node_order: None,
            matches: (0..node_count)
                .map(|_| [StateMatches::default(), StateMatches::default()])
                .collect(),
            matched_candidate_count: 0,
            timing_query_diagnostic_counts: TimingQueryDiagnosticCounts::default(),
            empty_known_pin_values: HashMap::new(),
        })
    }

    fn build(&mut self) -> Result<CoverPlan> {
        let node_count = self.choice_aig.graph().gates.len();
        let mut loads = pair_vec(node_count, CombinationalOutputLoad::default());
        for output in &self.outputs {
            add_output_load(
                &mut loads[output.state.node_id][output.state.polarity_index()],
                self.options.module_output_load,
            );
        }
        // Match NF's initial flow references: each concrete AIG object starts
        // from its structural fanout estimate, clamped to one, and later
        // rounds blend in references from the selected mapped cover.
        let mut flow_refs: Vec<[f64; 2]> = structural_fanout_estimates(self.choice_aig.graph())
            .into_iter()
            .map(|references| {
                let references = references.max(1) as f64;
                [references, references]
            })
            .collect();
        let mut requireds = pair_vec(node_count, f64::INFINITY);
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
        final_trace = self.fix_output_phase_drivers(
            final_trace,
            output_requireds.as_slice(),
            flow_refs.as_slice(),
        )?;
        if self.search_timing_model != SearchTimingModel::LibertyScalar
            && self.retime_approximate_cover_for_fallback
        {
            // Approximate delays are only a cover-selection objective. Report
            // the finished cover with gv-stats-style load-aware Liberty
            // timing instead of comparing exact time units to search-domain
            // required times.
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
        let reachable_node_order = self.reachable_node_order()?;
        self.matches = (0..self.choice_aig.graph().gates.len())
            .map(|_| [StateMatches::default(), StateMatches::default()])
            .collect();
        for node_id in reachable_node_order {
            self.evaluate_node(node_id, requireds, flow_refs, loads)?;
        }
        for output in &self.outputs {
            let node_id = output.state.node_id;
            if self.matches[node_id][0].delay.is_none() && self.matches[node_id][1].delay.is_none()
            {
                return Err(anyhow!(
                    "could not resolve a non-cyclic cover for AIG node {}",
                    node_id
                ));
            }
        }
        Ok(())
    }

    /// Finds output-reachable mapping nodes once, then reuses AIG order.
    fn reachable_node_order(&mut self) -> Result<Vec<usize>> {
        if let Some(order) = self.cached_reachable_node_order.as_ref() {
            return Ok(order.clone());
        }
        let node_count = self.choice_aig.graph().gates.len();
        let mut reachable = vec![false; node_count];
        let mut pending: Vec<usize> = self
            .outputs
            .iter()
            .map(|output| output.state.node_id)
            .collect();
        while let Some(node_id) = pending.pop() {
            if reachable[node_id] {
                continue;
            }
            reachable[node_id] = true;
            for polarity in [false, true] {
                let state = StateKey { node_id, polarity };
                self.ensure_candidates_for_state(state);
                for group in self.candidate_groups_for_state(state) {
                    for child_state in &group.layout.unique_input_states {
                        if child_state.node_id >= node_id {
                            return Err(anyhow!(
                                "technology mapping candidate for node {} depends on non-earlier node {}",
                                node_id,
                                child_state.node_id
                            ));
                        }
                        if !reachable[child_state.node_id] {
                            pending.push(child_state.node_id);
                        }
                    }
                }
            }
        }
        let order: Vec<usize> = reachable
            .iter()
            .enumerate()
            .filter_map(|(node_id, is_reachable)| is_reachable.then_some(node_id))
            .collect();
        self.cached_reachable_node_order = Some(order.clone());
        Ok(order)
    }

    /// Evaluates both polarities after every candidate fanin is already done.
    fn evaluate_node(
        &mut self,
        node_id: usize,
        requireds: &[[f64; 2]],
        flow_refs: &[[f64; 2]],
        loads: &[[CombinationalOutputLoad; 2]],
    ) -> Result<()> {
        for polarity in [false, true] {
            let state = StateKey { node_id, polarity };
            self.ensure_candidates_for_state(state);
            let mut direct_delay = self.source_match(state);
            let mut direct_area = direct_delay.clone();
            let groups = self.candidate_groups_for_state(state);
            for group in groups {
                let Some(children) = self.resolve_candidate_group_children(&group.layout) else {
                    continue;
                };
                for binding in group.bindings.iter().copied() {
                    if let Some(evaluated) = self.evaluate_candidate(
                        state,
                        &group.layout,
                        binding,
                        &children,
                        requireds,
                        flow_refs,
                        loads,
                    ) {
                        retain_best_delay_score(
                            self.cell_index,
                            &mut direct_delay,
                            &group.layout,
                            binding,
                            evaluated.delay,
                            evaluated.input_delays.as_slice(),
                        );
                        if let Some(area_score) = evaluated.area {
                            retain_best_area_score(
                                self.cell_index,
                                &mut direct_area,
                                &group.layout,
                                binding,
                                area_score,
                                evaluated.input_delays.as_slice(),
                            );
                        }
                    }
                }
            }
            let slot = &mut self.matches[node_id][state.polarity_index()];
            slot.direct_delay = direct_delay.clone();
            slot.direct_area = direct_area.clone();
            slot.delay = direct_delay;
            slot.area = direct_area;
        }

        self.add_inverter_closure(node_id, requireds, flow_refs, loads)?;
        Ok(())
    }

    fn source_match(&self, state: StateKey) -> Option<NfMatch> {
        let graph = self.choice_aig.graph();
        let node = AigRef { id: state.node_id };
        match graph.get(node) {
            AigNode::Input { .. } => {
                if state.polarity {
                    return None;
                }
                let arrival = self
                    .input_arrival_by_node
                    .get(&node.id)
                    .copied()
                    .unwrap_or(0.0);
                Some(source_nf_match(
                    SourceKind::Input(node),
                    arrival,
                    self.options.primary_input_transition,
                ))
            }
            AigNode::Literal { value, .. } => Some(source_nf_match(
                SourceKind::Literal(*value ^ state.polarity),
                0.0,
                0.0,
            )),
            AigNode::And2 { .. } => None,
        }
    }

    /// Populates one candidate cache entry once, before mapping rounds borrow
    /// it.
    fn ensure_candidates_for_state(&mut self, state: StateKey) {
        if self.candidates[state.node_id][state.polarity_index()].is_some() {
            return;
        }
        let mut by_input_states: BTreeMap<
            SmallVec<[StateKey; MAX_TRUTH_TABLE_INPUTS]>,
            CandidateGroup,
        > = BTreeMap::new();
        let mut matched_candidate_count = 0usize;
        let variant_limit = self
            .options
            .max_frontier_size
            .min(MAX_CELL_VARIANTS_PER_SIGNATURE)
            .max(1);
        for cut in &self.cuts_by_node[state.node_id] {
            // The trivial self-cut exists only so later nodes can propagate
            // it. NF does not map a node through itself.
            if cut.leaves.iter().any(|leaf| leaf.id == state.node_id) {
                continue;
            }
            let truth = if state.polarity {
                complement_truth(cut.truth, cut.leaves.len())
            } else {
                cut.truth
            };
            for binding_id in self.cell_index.matches(cut.leaves.len(), truth) {
                let binding = self.cell_index.binding(*binding_id);
                let input_states = input_states_for_binding(cut, binding);
                matched_candidate_count += 1;
                insert_candidate_variant(
                    &mut by_input_states,
                    *binding_id,
                    input_states,
                    variant_limit,
                    self.cell_index,
                );
            }
        }
        self.matched_candidate_count += matched_candidate_count;
        // BTreeMap preserves the same input-state-first order as the prior
        // flattened candidate sort; each group's bindings are already sorted
        // by the remaining area/delay/stable-key keys.
        let groups: Vec<CandidateGroup> = by_input_states.into_values().collect();
        self.candidates[state.node_id][state.polarity_index()] = Some(groups);
    }

    /// Returns precomputed candidate groups without cloning the cached
    /// frontier.
    fn candidate_groups_for_state(&self, state: StateKey) -> &[CandidateGroup] {
        self.candidates[state.node_id][state.polarity_index()]
            .as_deref()
            .expect("mapping candidates should be populated before use")
    }

    /// Resolves one shared child layout before trying its cell variants.
    fn resolve_candidate_group_children(
        &self,
        layout: &CandidateLayout,
    ) -> Option<CandidateGroupChildren> {
        let mut selected: SmallVec<[MatchSummary; MAX_TRUTH_TABLE_INPUTS]> =
            SmallVec::with_capacity(layout.unique_input_states.len());
        let mut area_alternatives: SmallVec<[Option<MatchSummary>; MAX_TRUTH_TABLE_INPUTS]> =
            SmallVec::with_capacity(layout.unique_input_states.len());
        for child_state in &layout.unique_input_states {
            let child_slot = &self.matches[child_state.node_id][child_state.polarity_index()];
            let delay_match = child_slot.delay.as_ref()?;
            let area_match = child_slot
                .area
                .as_ref()
                .filter(|area_match| !same_match_choice(self.cell_index, area_match, delay_match))
                .map(MatchSummary::from);
            selected.push(MatchSummary::from(delay_match));
            area_alternatives.push(area_match);
        }
        Some(CandidateGroupChildren {
            selected,
            area_alternatives,
        })
    }

    /// Derives both NF match slots for one binding of a resolved group.
    fn evaluate_candidate(
        &self,
        state: StateKey,
        layout: &CandidateLayout,
        binding_id: CellBindingId,
        children: &CandidateGroupChildren,
        requireds: &[[f64; 2]],
        flow_refs: &[[f64; 2]],
        loads: &[[CombinationalOutputLoad; 2]],
    ) -> Option<EvaluatedCandidate> {
        let mut selected = children.selected.clone();
        let binding = self.cell_index.binding(binding_id);
        let input_delays = search_input_delays(
            self.search_timing_model,
            binding,
            layout.input_states.len(),
            loads[state.node_id][state.polarity_index()],
            self.buffered_search,
        );
        let delay_score = self.score_candidate_with_child_summaries(
            state,
            layout,
            binding_id,
            selected.as_slice(),
            input_delays.as_slice(),
            flow_refs,
        );
        let required = requireds[state.node_id][state.polarity_index()];
        if delay_score.worst_arrival() > required + TIMING_EPSILON {
            return Some(EvaluatedCandidate {
                delay: delay_score,
                area: None,
                input_delays,
            });
        }

        let mut area_score = delay_score;
        for (child_index, area_match) in children.area_alternatives.iter().copied().enumerate() {
            let Some(area_match) = area_match else {
                continue;
            };
            let previous = std::mem::replace(&mut selected[child_index], area_match);
            let trial_score = self.score_candidate_with_child_summaries(
                state,
                layout,
                binding_id,
                selected.as_slice(),
                input_delays.as_slice(),
                flow_refs,
            );
            if trial_score.worst_arrival() <= required + TIMING_EPSILON {
                area_score = trial_score;
            } else {
                selected[child_index] = previous;
            }
        }
        Some(EvaluatedCandidate {
            delay: delay_score,
            area: Some(area_score),
            input_delays,
        })
    }

    /// Scores one child selection without allocating reconstruction state.
    fn score_candidate_with_child_summaries(
        &self,
        state: StateKey,
        layout: &CandidateLayout,
        binding_id: CellBindingId,
        child_matches: &[MatchSummary],
        input_delays: &[f64],
        flow_refs: &[[f64; 2]],
    ) -> CandidateScore {
        debug_assert_eq!(layout.unique_input_states.len(), child_matches.len());
        debug_assert_eq!(layout.input_states.len(), input_delays.len());
        let mut worst_arrival: f64 = 0.0;
        let mut transition: f64 = 0.0;
        for (input_slot, input_delay) in layout
            .input_state_slots
            .iter()
            .zip(input_delays.iter().copied())
        {
            let child_timing = child_matches[*input_slot].timing;
            worst_arrival = worst_arrival.max(worst_signal_arrival(child_timing) + input_delay);
            transition = transition.max(max_signal_transition(child_timing));
        }
        let mut flow = self.cell_index.binding(binding_id).area;
        for child_match in child_matches {
            flow += child_match.flow;
        }
        flow /= flow_refs[state.node_id][state.polarity_index()].max(1.0);
        CandidateScore {
            timing: SignalTiming {
                rise: EdgeTiming {
                    arrival: worst_arrival,
                    transition,
                },
                fall: EdgeTiming {
                    arrival: worst_arrival,
                    transition,
                },
            },
            flow,
        }
    }

    fn evaluate_with_child_matches(
        &self,
        state: StateKey,
        layout: &CandidateLayout,
        binding_id: CellBindingId,
        child_matches: &[&NfMatch],
        flow_refs: &[[f64; 2]],
        loads: &[[CombinationalOutputLoad; 2]],
    ) -> Result<NfMatch> {
        let child_summaries: SmallVec<[MatchSummary; MAX_TRUTH_TABLE_INPUTS]> = child_matches
            .iter()
            .map(|child| MatchSummary::from(*child))
            .collect();
        let binding = self.cell_index.binding(binding_id);
        let input_delays = search_input_delays(
            self.search_timing_model,
            binding,
            layout.input_states.len(),
            loads[state.node_id][state.polarity_index()],
            self.buffered_search,
        );
        let score = self.score_candidate_with_child_summaries(
            state,
            layout,
            binding_id,
            child_summaries.as_slice(),
            input_delays.as_slice(),
            flow_refs,
        );
        Ok(materialize_candidate_match(
            layout,
            binding_id,
            score,
            input_delays,
        ))
    }

    /// Evaluates one selected cell with the same rise/fall/load semantics as
    /// gv-stats. Trial matches deliberately use the much cheaper unit or
    /// scalar search model above, keeping the NF inner loop lightweight.
    fn evaluate_selected_binding_timing(
        &mut self,
        binding_id: CellBindingId,
        child_timings: &[SignalTiming],
        output_load: CombinationalOutputLoad,
    ) -> Result<(SignalTiming, SmallVec<[f64; MAX_TRUTH_TABLE_INPUTS]>)> {
        let binding = self.cell_index.binding(binding_id);
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
        node_id: usize,
        requireds: &[[f64; 2]],
        flow_refs: &[[f64; 2]],
        loads: &[[CombinationalOutputLoad; 2]],
    ) -> Result<()> {
        let inverters: Vec<CellBindingId> = self
            .cell_index
            .matches(1, 0b01)
            .iter()
            .copied()
            .filter(|binding| !self.cell_index.binding(*binding).input_negated[0])
            .collect();
        if inverters.is_empty() {
            return Ok(());
        }
        for polarity in [false, true] {
            let state = StateKey { node_id, polarity };
            let opposite = StateKey {
                node_id,
                polarity: !polarity,
            };
            let layout = CandidateLayout::new(SmallVec::from_slice(&[opposite]));
            let opposite_slot = &self.matches[node_id][opposite.polarity_index()];
            let direct_delay = opposite_slot.direct_delay.clone();
            let direct_area = opposite_slot.direct_area.clone();
            for inverter in &inverters {
                if let Some(child) = direct_delay.as_ref() {
                    let candidate_match = self.evaluate_with_child_matches(
                        state,
                        &layout,
                        *inverter,
                        &[child],
                        flow_refs,
                        loads,
                    )?;
                    retain_strictly_faster_inverter(
                        &mut self.matches[node_id][state.polarity_index()].delay,
                        candidate_match,
                    );
                }
                if let Some(child) = direct_area.as_ref() {
                    let candidate_match = self.evaluate_with_child_matches(
                        state,
                        &layout,
                        *inverter,
                        &[child],
                        flow_refs,
                        loads,
                    )?;
                    if candidate_match.worst_arrival()
                        <= requireds[node_id][state.polarity_index()] + TIMING_EPSILON
                    {
                        retain_strictly_smaller_inverter(
                            &mut self.matches[node_id][state.polarity_index()].area,
                            candidate_match,
                        );
                    }
                }
            }
        }
        self.break_mutual_inverter_closure(node_id);
        Ok(())
    }

    /// Keeps at least one direct implementation when both polarities would
    /// otherwise be implemented as an inverter of the other polarity. ABC NF
    /// has the same invariant for its complemented matches.
    fn break_mutual_inverter_closure(&mut self, node_id: usize) {
        let positive = StateKey {
            node_id,
            polarity: false,
        };
        let negative = StateKey {
            node_id,
            polarity: true,
        };
        let positive_slot = self.matches[node_id][0].clone();
        let negative_slot = self.matches[node_id][1].clone();
        if matches_use_each_other(
            positive_slot.delay.as_ref(),
            positive,
            negative_slot.delay.as_ref(),
            negative,
        ) {
            let (positive_delay, negative_delay) = break_mutual_closure(
                self.cell_index,
                positive_slot.delay,
                positive_slot.direct_delay,
                negative_slot.delay,
                negative_slot.direct_delay,
                nf_delay_order,
            );
            self.matches[node_id][0].delay = positive_delay;
            self.matches[node_id][1].delay = negative_delay;
        }
        if matches_use_each_other(
            positive_slot.area.as_ref(),
            positive,
            negative_slot.area.as_ref(),
            negative,
        ) {
            let (positive_area, negative_area) = break_mutual_closure(
                self.cell_index,
                positive_slot.area,
                positive_slot.direct_area,
                negative_slot.area,
                negative_slot.direct_area,
                nf_area_order,
            );
            self.matches[node_id][0].area = positive_area;
            self.matches[node_id][1].area = negative_area;
        }
    }

    fn trace_selected(&self, prefer_area: bool, output_requireds: &[f64]) -> Result<MappingTrace> {
        let node_count = self.choice_aig.graph().gates.len();
        let mut selected = option_pair_vec(node_count);
        let mut requireds = pair_vec(node_count, f64::INFINITY);
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

        let mut loads = pair_vec(node_count, CombinationalOutputLoad::default());
        let mut map_refs = pair_vec(node_count, 0usize);
        let mut area = 0.0;
        for output in &self.outputs {
            add_output_load(
                &mut loads[output.state.node_id][output.state.polarity_index()],
                self.options.module_output_load,
            );
            reference_selected_state(
                self.cell_index,
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
                selected[output.state.node_id][output.state.polarity_index()]
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
        let mut current = if self.search_timing_model == SearchTimingModel::LibertyScalar {
            let mut selected = trace.selected;
            self.retime_selected(selected.as_mut_slice(), trace.loads.as_slice())?;
            self.trace_fixed_selected(selected, output_requireds)?
        } else {
            self.trace_fixed_selected(trace.selected, output_requireds)?
        };
        let before = current.clone();
        let node_count = self.choice_aig.graph().gates.len();
        let unit_flow_refs = pair_vec(node_count, 1.0f64);
        let mut scratch = ExactAreaScratch::new(node_count);
        let order = self.selected_preorder(current.selected.as_slice())?;
        let mut selected = current.selected;
        let mut map_refs = current.map_refs;
        let mut requireds = current.requireds;
        let loads = current.loads;

        for state in order {
            if map_refs[state.node_id][state.polarity_index()] == 0 {
                continue;
            }
            let current_match = selected[state.node_id][state.polarity_index()]
                .as_ref()
                .ok_or_else(|| anyhow!("exact-area pass is missing state {:?}", state))?
                .clone();
            if matches!(&current_match.choice, MatchChoice::Source(_)) {
                continue;
            }

            scratch.begin(state);
            let area_before = dereference_match_children_exact(
                self.cell_index,
                &current_match,
                map_refs.as_mut_slice(),
                selected.as_slice(),
                &mut scratch.visiting,
            )?;
            let required = requireds[state.node_id][state.polarity_index()];
            let candidates = self.exact_area_candidates(
                state,
                selected.as_slice(),
                loads.as_slice(),
                unit_flow_refs.as_slice(),
            )?;
            let mut best_match = current_match;
            let mut best_area = area_before;
            for candidate in candidates {
                if candidate.worst_arrival() > required + TIMING_EPSILON {
                    continue;
                }
                scratch.begin(state);
                let area_after = match reference_match_children_exact(
                    self.cell_index,
                    &candidate,
                    map_refs.as_mut_slice(),
                    selected.as_slice(),
                    self.matches.as_slice(),
                    &mut scratch.increments,
                    &mut scratch.newly_selected,
                    &mut scratch.visiting,
                ) {
                    Ok(area) => area,
                    Err(_) => {
                        undo_reference_increments(
                            map_refs.as_mut_slice(),
                            scratch.increments.as_slice(),
                        );
                        continue;
                    }
                };
                undo_reference_increments(map_refs.as_mut_slice(), scratch.increments.as_slice());
                if exact_area_candidate_is_better(
                    self.cell_index,
                    area_after,
                    &candidate,
                    best_area,
                    &best_match,
                ) {
                    best_area = area_after;
                    best_match = candidate;
                }
            }

            scratch.begin(state);
            reference_match_children_exact(
                self.cell_index,
                &best_match,
                map_refs.as_mut_slice(),
                selected.as_slice(),
                self.matches.as_slice(),
                &mut scratch.increments,
                &mut scratch.newly_selected,
                &mut scratch.visiting,
            )?;
            propagate_match_requireds(&best_match, required, requireds.as_mut_slice());
            selected[state.node_id][state.polarity_index()] = Some(best_match);
            for new_state in scratch.newly_selected.drain(..) {
                if selected[new_state.node_id][new_state.polarity_index()].is_none() {
                    let new_match = self.matches[new_state.node_id][new_state.polarity_index()]
                        .delay
                        .as_ref()
                        .ok_or_else(|| {
                            anyhow!("exact-area trial has no delay match for {:?}", new_state)
                        })?
                        .clone();
                    selected[new_state.node_id][new_state.polarity_index()] = Some(new_match);
                }
            }
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

    /// Keeps an output-only implementation from becoming the source of a
    /// high-fanout inverter closure.
    ///
    /// NF coordinates both polarities of a referenced GIA object and finishes
    /// with a PO-driver cleanup. Our recursive final-netlist reconstruction
    /// needs an explicit equivalent: when an output implementation feeds a
    /// multiply-referenced plain inverter, prefer a direct implementation of
    /// the inverter's state and put the inverter on the output side.
    fn fix_output_phase_drivers(
        &self,
        trace: MappingTrace,
        output_requireds: &[f64],
        flow_refs: &[[f64; 2]],
    ) -> Result<MappingTrace> {
        let mut selected = trace.selected.clone();
        let mut closure_users: BTreeMap<StateKey, Vec<StateKey>> = BTreeMap::new();
        for node_id in 0..selected.len() {
            for polarity in [false, true] {
                let closure_state = StateKey { node_id, polarity };
                let Some(selected_match) =
                    selected[node_id][closure_state.polarity_index()].as_ref()
                else {
                    continue;
                };
                let Some(input_state) = self.plain_inverter_input_state(selected_match) else {
                    continue;
                };
                closure_users
                    .entry(input_state)
                    .or_default()
                    .push(closure_state);
            }
        }
        for users in closure_users.values_mut() {
            users.sort_by(|lhs, rhs| {
                trace.map_refs[rhs.node_id][rhs.polarity_index()]
                    .cmp(&trace.map_refs[lhs.node_id][lhs.polarity_index()])
                    .then_with(|| lhs.cmp(rhs))
            });
        }
        let mut visited_outputs = BTreeSet::new();
        let mut changed = false;
        for output in &self.outputs {
            let state = output.state;
            if !visited_outputs.insert(state) {
                continue;
            }
            if trace.map_refs[state.node_id][state.polarity_index()] == 0 {
                continue;
            }
            let Some(closure_state) = closure_users.get(&state).and_then(|users| {
                users
                    .iter()
                    .copied()
                    .find(|user| trace.map_refs[user.node_id][user.polarity_index()] > 1)
            }) else {
                continue;
            };
            let closure_required =
                trace.requireds[closure_state.node_id][closure_state.polarity_index()];
            let Some(direct_closure) =
                self.direct_match_for_required(closure_state, closure_required, Some(state))
            else {
                continue;
            };
            let state_required = trace.requireds[state.node_id][state.polarity_index()];
            let Some(output_closure) = self.inverter_closure_for_state(
                state,
                closure_state,
                &direct_closure,
                flow_refs,
                trace.loads.as_slice(),
            )?
            else {
                continue;
            };
            if output_closure.worst_arrival() > state_required + TIMING_EPSILON {
                continue;
            }
            selected[closure_state.node_id][closure_state.polarity_index()] = Some(direct_closure);
            selected[state.node_id][state.polarity_index()] = Some(output_closure);
            changed = true;
        }
        if !changed {
            return Ok(trace);
        }
        let candidate = match self.trace_fixed_selected(selected, output_requireds) {
            Ok(candidate) => candidate,
            // This cleanup is optional. If a sibling-derived unary cut made
            // the candidate cyclic, keep the already-validated cover.
            Err(_) => return Ok(trace),
        };
        let violates_required = candidate
            .output_arrivals
            .iter()
            .zip(output_requireds.iter())
            .any(|(arrival, required)| arrival > &(required + TIMING_EPSILON));
        if violates_required {
            return Ok(trace);
        }
        Ok(candidate)
    }

    /// Returns a direct, non-inverter match meeting the current requirement
    /// without depending on one state that is about to become its fanout.
    fn direct_match_for_required(
        &self,
        state: StateKey,
        required: f64,
        forbidden_input: Option<StateKey>,
    ) -> Option<NfMatch> {
        let slot = &self.matches[state.node_id][state.polarity_index()];
        slot.direct_area
            .clone()
            .filter(|candidate| {
                candidate.worst_arrival() <= required + TIMING_EPSILON
                    && forbidden_input
                        .is_none_or(|forbidden| !match_uses_state(candidate, forbidden))
            })
            .or_else(|| {
                slot.direct_delay.clone().filter(|candidate| {
                    candidate.worst_arrival() <= required + TIMING_EPSILON
                        && forbidden_input
                            .is_none_or(|forbidden| !match_uses_state(candidate, forbidden))
                })
            })
    }

    /// Recognizes the plain inverter closures added by NF-style phase
    /// completion, including those reached through sibling-derived unary cuts.
    fn plain_inverter_input_state(&self, selected_match: &NfMatch) -> Option<StateKey> {
        let MatchChoice::Cell {
            binding: binding_id,
            inputs,
        } = &selected_match.choice
        else {
            return None;
        };
        let binding = self.cell_index.binding(*binding_id);
        if inputs.len() != 1
            || binding.input_pin_names.len() != 1
            || binding.input_negated[0]
            || !self.cell_index.matches(1, 0b01).contains(binding_id)
        {
            return None;
        }
        Some(inputs[0])
    }

    /// Builds the best ordinary inverter closure for one already-selected
    /// opposite phase.
    fn inverter_closure_for_state(
        &self,
        state: StateKey,
        input_state: StateKey,
        child: &NfMatch,
        flow_refs: &[[f64; 2]],
        loads: &[[CombinationalOutputLoad; 2]],
    ) -> Result<Option<NfMatch>> {
        let mut best = None;
        let layout = CandidateLayout::new(SmallVec::from_slice(&[input_state]));
        for inverter in self
            .cell_index
            .matches(1, 0b01)
            .iter()
            .copied()
            .filter(|binding| !self.cell_index.binding(*binding).input_negated[0])
        {
            let candidate_match = self.evaluate_with_child_matches(
                state,
                &layout,
                inverter,
                &[child],
                flow_refs,
                loads,
            )?;
            retain_best_area(self.cell_index, &mut best, candidate_match);
        }
        Ok(best)
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
        if let Some(current) = selected[state.node_id][state.polarity_index()].clone() {
            matches.push(current);
        }
        if let Some(delay) = self.matches[state.node_id][state.polarity_index()]
            .delay
            .clone()
        {
            matches.push(delay);
        }
        if let Some(area) = self.matches[state.node_id][state.polarity_index()]
            .area
            .clone()
        {
            matches.push(area);
        }
        self.ensure_candidates_for_state(state);
        for group in self.candidate_groups_for_state(state) {
            let mut child_matches: SmallVec<[&NfMatch; MAX_TRUTH_TABLE_INPUTS]> =
                SmallVec::with_capacity(group.layout.unique_input_states.len());
            let mut available = true;
            for child_state in &group.layout.unique_input_states {
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
            for binding in group.bindings.iter().copied() {
                let candidate_match = self.evaluate_with_child_matches(
                    state,
                    &group.layout,
                    binding,
                    child_matches.as_slice(),
                    flow_refs,
                    loads,
                )?;
                matches.push(candidate_match);
            }
        }
        matches.sort_by(|lhs, rhs| {
            match_choice_order(self.cell_index, &lhs.choice, &rhs.choice)
                .then_with(|| lhs.worst_arrival().total_cmp(&rhs.worst_arrival()))
        });
        matches.dedup_by(|lhs, rhs| same_match_choice(self.cell_index, lhs, rhs));
        Ok(matches)
    }

    /// Re-evaluates every reachable selected match under the currently
    /// accumulated capacitive loads.
    fn retime_selected(
        &mut self,
        selected: &mut [[Option<NfMatch>; 2]],
        loads: &[[CombinationalOutputLoad; 2]],
    ) -> Result<()> {
        let mut visit_state = pair_vec(self.choice_aig.graph().gates.len(), VisitState::Unvisited);
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
        match visit_state[state.node_id][state_index] {
            VisitState::Done => {
                return selected[state.node_id][state_index]
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
        visit_state[state.node_id][state_index] = VisitState::Visiting;
        let selected_match = selected[state.node_id][state_index]
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
                    *binding,
                    child_timings.as_slice(),
                    loads[state.node_id][state_index],
                )?
            }
        };
        let slot = selected[state.node_id][state_index]
            .as_mut()
            .expect("selected match existed above");
        slot.timing = timing;
        slot.input_delays = input_delays;
        visit_state[state.node_id][state_index] = VisitState::Done;
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
        let node_count = self.choice_aig.graph().gates.len();
        let mut requireds = pair_vec(node_count, f64::INFINITY);
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
        let mut loads = pair_vec(node_count, CombinationalOutputLoad::default());
        let mut map_refs = pair_vec(node_count, 0usize);
        let mut area = 0.0;
        for output in &self.outputs {
            add_output_load(
                &mut loads[output.state.node_id][output.state.polarity_index()],
                self.options.module_output_load,
            );
            reference_selected_state(
                self.cell_index,
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
                selected[output.state.node_id][output.state.polarity_index()]
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
        let previous_required = requireds[state.node_id][state.polarity_index()];
        if selected[state.node_id][state.polarity_index()].is_some()
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
        requireds[state.node_id][state.polarity_index()] = previous_required.min(required);
        let effective_required = requireds[state.node_id][state.polarity_index()];
        let state_matches = &self.matches[state.node_id][state.polarity_index()];
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
        selected[state.node_id][state.polarity_index()] = Some(selected_match.clone());
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
        let mut memo = option_pair_vec(self.choice_aig.graph().gates.len());
        let mut output_solutions = Vec::with_capacity(self.outputs.len());
        let mut visiting = HashSet::new();
        for output in &self.outputs {
            output_solutions.push(materialize_state(
                self.cell_index,
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

/// Materializes reconstruction state only after a lightweight score wins.
fn materialize_candidate_match(
    layout: &CandidateLayout,
    binding: CellBindingId,
    score: CandidateScore,
    input_delays: SmallVec<[f64; MAX_TRUTH_TABLE_INPUTS]>,
) -> NfMatch {
    NfMatch {
        timing: score.timing,
        flow: score.flow,
        choice: MatchChoice::Cell {
            binding,
            inputs: layout.input_states.clone(),
        },
        input_delays,
    }
}

/// Returns the fixed per-pin delay vector for the NF search objective.
///
/// NF-unit mode preserves ABC's generated genlib objective. Buffered-Liberty
/// mode adds bounded downstream electrical pressure and the estimated levels
/// of the same real buffer tree that is subsequently materialized.
fn search_input_delays(
    model: SearchTimingModel,
    binding: &CellBinding,
    input_count: usize,
    output_load: CombinationalOutputLoad,
    buffered: BufferedSearchParameters,
) -> SmallVec<[f64; MAX_TRUTH_TABLE_INPUTS]> {
    let mut input_delays = SmallVec::with_capacity(input_count);
    match model {
        SearchTimingModel::Unit => {
            for _ in 0..input_count {
                input_delays.push(1.0);
            }
        }
        SearchTimingModel::BufferedLiberty => {
            let load = output_load.rise.max(output_load.fall);
            let (pressure, levels) = buffered.target_load.map_or((0.0, 0.0), |target| {
                let pressure = (load / target).clamp(0.0, 1.0);
                let levels = if load > target {
                    (load / target).log(buffered.max_fanout as f64).ceil()
                } else {
                    0.0
                };
                (pressure, levels)
            });
            for input_index in 0..input_count {
                let native_delay = binding.input_delays[input_index]
                    .filter(|delay| delay.is_finite() && *delay > 0.0)
                    .unwrap_or(1.0);
                input_delays
                    .push(native_delay * (1.0 + 0.25 * pressure) + levels * buffered.buffer_delay);
            }
        }
        SearchTimingModel::LibertyScalar => {
            for input_index in 0..input_count {
                input_delays.push(binding.input_delays[input_index].unwrap_or(0.0));
            }
        }
    }
    input_delays
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
    cut: &Cut,
    binding: &CellBinding,
) -> SmallVec<[StateKey; MAX_TRUTH_TABLE_INPUTS]> {
    binding
        .input_to_leaf
        .iter()
        .enumerate()
        .map(|(input_index, leaf_index)| {
            let leaf = cut.leaves[*leaf_index];
            StateKey {
                node_id: leaf.id,
                polarity: binding.input_negated[input_index],
            }
        })
        .collect()
}

fn insert_candidate_variant(
    by_input_states: &mut BTreeMap<SmallVec<[StateKey; MAX_TRUTH_TABLE_INPUTS]>, CandidateGroup>,
    binding_id: CellBindingId,
    input_states: SmallVec<[StateKey; MAX_TRUTH_TABLE_INPUTS]>,
    limit: usize,
    cell_index: &LibertyCellIndex,
) {
    let group = by_input_states
        .entry(input_states.clone())
        .or_insert_with(|| CandidateGroup {
            layout: CandidateLayout::new(input_states),
            bindings: SmallVec::new(),
        });
    if group
        .bindings
        .iter()
        .copied()
        .any(|existing| binding_dominates(cell_index, existing, binding_id))
    {
        return;
    }
    group
        .bindings
        .retain(|existing| !binding_dominates(cell_index, binding_id, *existing));
    group.bindings.push(binding_id);
    group
        .bindings
        .sort_by(|lhs, rhs| binding_variant_order(cell_index, *lhs, *rhs));
    group.bindings.truncate(limit);
}

fn binding_dominates(
    cell_index: &LibertyCellIndex,
    lhs_id: CellBindingId,
    rhs_id: CellBindingId,
) -> bool {
    let lhs = cell_index.binding(lhs_id);
    let rhs = cell_index.binding(rhs_id);
    let lhs_delay = lhs.worst_nominal_delay();
    let rhs_delay = rhs.worst_nominal_delay();
    lhs.area <= rhs.area
        && lhs_delay <= rhs_delay
        && (lhs.area < rhs.area
            || lhs_delay < rhs_delay
            || cell_index.stable_key_order(lhs_id, rhs_id).is_le())
}

fn binding_variant_order(
    cell_index: &LibertyCellIndex,
    lhs_id: CellBindingId,
    rhs_id: CellBindingId,
) -> std::cmp::Ordering {
    let lhs_binding = cell_index.binding(lhs_id);
    let rhs_binding = cell_index.binding(rhs_id);
    lhs_binding
        .area
        .total_cmp(&rhs_binding.area)
        .then_with(|| {
            lhs_binding
                .worst_nominal_delay()
                .total_cmp(&rhs_binding.worst_nominal_delay())
        })
        .then_with(|| cell_index.stable_key_order(lhs_id, rhs_id))
}

/// Retains a scored delay candidate, materializing reconstruction data only
/// when it beats the current winner.
fn retain_best_delay_score(
    cell_index: &LibertyCellIndex,
    slot: &mut Option<NfMatch>,
    layout: &CandidateLayout,
    binding: CellBindingId,
    score: CandidateScore,
    input_delays: &[f64],
) {
    let should_replace = slot.as_ref().is_none_or(|existing| {
        candidate_delay_order(cell_index, layout, binding, score, existing).is_lt()
    });
    if should_replace {
        *slot = Some(materialize_candidate_match(
            layout,
            binding,
            score,
            SmallVec::from_slice(input_delays),
        ));
    }
}

/// Retains a scored area candidate, materializing reconstruction data only
/// when it beats the current winner.
fn retain_best_area_score(
    cell_index: &LibertyCellIndex,
    slot: &mut Option<NfMatch>,
    layout: &CandidateLayout,
    binding: CellBindingId,
    score: CandidateScore,
    input_delays: &[f64],
) {
    let should_replace = slot.as_ref().is_none_or(|existing| {
        candidate_area_order(cell_index, layout, binding, score, existing).is_lt()
    });
    if should_replace {
        *slot = Some(materialize_candidate_match(
            layout,
            binding,
            score,
            SmallVec::from_slice(input_delays),
        ));
    }
}

fn candidate_delay_order(
    cell_index: &LibertyCellIndex,
    layout: &CandidateLayout,
    binding: CellBindingId,
    score: CandidateScore,
    existing: &NfMatch,
) -> std::cmp::Ordering {
    score
        .worst_arrival()
        .total_cmp(&existing.worst_arrival())
        .then_with(|| score.flow.total_cmp(&existing.flow))
        .then_with(|| candidate_choice_order(cell_index, layout, binding, &existing.choice))
}

fn candidate_area_order(
    cell_index: &LibertyCellIndex,
    layout: &CandidateLayout,
    binding: CellBindingId,
    score: CandidateScore,
    existing: &NfMatch,
) -> std::cmp::Ordering {
    score
        .flow
        .total_cmp(&existing.flow)
        .then_with(|| score.worst_arrival().total_cmp(&existing.worst_arrival()))
        .then_with(|| candidate_choice_order(cell_index, layout, binding, &existing.choice))
}

fn candidate_choice_order(
    cell_index: &LibertyCellIndex,
    layout: &CandidateLayout,
    candidate_binding: CellBindingId,
    existing: &MatchChoice,
) -> std::cmp::Ordering {
    match existing {
        MatchChoice::Source(_) => std::cmp::Ordering::Greater,
        MatchChoice::Cell { binding, inputs } => cell_index
            .stable_key_order(candidate_binding, *binding)
            .then_with(|| layout.input_states.cmp(inputs)),
    }
}

fn retain_best_area(cell_index: &LibertyCellIndex, slot: &mut Option<NfMatch>, candidate: NfMatch) {
    match slot {
        Some(existing) if nf_area_order(cell_index, existing, &candidate).is_le() => {}
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

fn nf_delay_order(
    cell_index: &LibertyCellIndex,
    lhs: &NfMatch,
    rhs: &NfMatch,
) -> std::cmp::Ordering {
    lhs.worst_arrival()
        .total_cmp(&rhs.worst_arrival())
        .then_with(|| lhs.flow.total_cmp(&rhs.flow))
        .then_with(|| match_choice_order(cell_index, &lhs.choice, &rhs.choice))
}

fn nf_area_order(
    cell_index: &LibertyCellIndex,
    lhs: &NfMatch,
    rhs: &NfMatch,
) -> std::cmp::Ordering {
    lhs.flow
        .total_cmp(&rhs.flow)
        .then_with(|| lhs.worst_arrival().total_cmp(&rhs.worst_arrival()))
        .then_with(|| match_choice_order(cell_index, &lhs.choice, &rhs.choice))
}

fn same_match_choice(cell_index: &LibertyCellIndex, lhs: &NfMatch, rhs: &NfMatch) -> bool {
    match_choice_order(cell_index, &lhs.choice, &rhs.choice).is_eq()
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

fn match_uses_state(selected_match: &NfMatch, state: StateKey) -> bool {
    matches!(
        &selected_match.choice,
        MatchChoice::Cell { inputs, .. } if inputs.contains(&state)
    )
}

fn break_mutual_closure(
    cell_index: &LibertyCellIndex,
    positive: Option<NfMatch>,
    positive_direct: Option<NfMatch>,
    negative: Option<NfMatch>,
    negative_direct: Option<NfMatch>,
    order: fn(&LibertyCellIndex, &NfMatch, &NfMatch) -> std::cmp::Ordering,
) -> (Option<NfMatch>, Option<NfMatch>) {
    match (positive_direct, negative_direct) {
        (Some(positive_direct), Some(negative_direct)) => {
            if order(cell_index, &positive_direct, &negative_direct).is_le() {
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

fn match_choice_order(
    cell_index: &LibertyCellIndex,
    lhs: &MatchChoice,
    rhs: &MatchChoice,
) -> std::cmp::Ordering {
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
        ) => cell_index
            .stable_key_order(*lhs_binding, *rhs_binding)
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
    for node_id in 0..flow_refs.len() {
        for polarity in [false, true] {
            let index = usize::from(polarity);
            let actual = map_refs[node_id][index].max(1) as f64;
            flow_refs[node_id][index] =
                (coefficient * flow_refs[node_id][index] + (1.0 - coefficient) * actual).max(1.0);
        }
    }
}

fn reference_selected_state(
    cell_index: &LibertyCellIndex,
    state: StateKey,
    selected: &[[Option<NfMatch>; 2]],
    map_refs: &mut [[usize; 2]],
    loads: &mut [[CombinationalOutputLoad; 2]],
    area: &mut f64,
) -> Result<()> {
    let ref_count = &mut map_refs[state.node_id][state.polarity_index()];
    *ref_count += 1;
    if *ref_count > 1 {
        return Ok(());
    }
    let selected_match = selected[state.node_id][state.polarity_index()]
        .as_ref()
        .ok_or_else(|| anyhow!("selected mapping is missing state {:?}", state))?;
    let MatchChoice::Cell {
        binding: binding_id,
        inputs,
    } = &selected_match.choice
    else {
        return Ok(());
    };
    let binding = cell_index.binding(*binding_id);
    *area += binding.area;
    for (input_index, input_state) in inputs.iter().copied().enumerate() {
        add_pin_load(
            &mut loads[input_state.node_id][input_state.polarity_index()],
            binding.input_capacitances[input_index],
        );
        reference_selected_state(cell_index, input_state, selected, map_refs, loads, area)?;
    }
    Ok(())
}

fn selected_or_delay_match<'a>(
    selected: &'a [[Option<NfMatch>; 2]],
    matches: &'a [[StateMatches; 2]],
    state: StateKey,
) -> Option<&'a NfMatch> {
    selected[state.node_id][state.polarity_index()]
        .as_ref()
        .or_else(|| {
            matches[state.node_id][state.polarity_index()]
                .delay
                .as_ref()
        })
}

fn exact_area_candidate_is_better(
    cell_index: &LibertyCellIndex,
    candidate_area: f64,
    candidate: &NfMatch,
    best_area: f64,
    best: &NfMatch,
) -> bool {
    candidate_area + TIMING_EPSILON < best_area
        || ((candidate_area - best_area).abs() <= TIMING_EPSILON
            && nf_delay_order(cell_index, candidate, best).is_lt())
}

fn match_cell_area(cell_index: &LibertyCellIndex, selected_match: &NfMatch) -> f64 {
    match &selected_match.choice {
        MatchChoice::Source(_) => 0.0,
        MatchChoice::Cell { binding, .. } => cell_index.binding(*binding).area,
    }
}

/// Removes one selected cell's fanin cone from global refs while leaving the
/// root reference itself in place, like ABC's Nf_MatchDeref_rec.
fn dereference_match_children_exact(
    cell_index: &LibertyCellIndex,
    selected_match: &NfMatch,
    map_refs: &mut [[usize; 2]],
    selected: &[[Option<NfMatch>; 2]],
    visiting: &mut DenseVisitMarks,
) -> Result<f64> {
    let mut area = match_cell_area(cell_index, selected_match);
    let MatchChoice::Cell { inputs, .. } = &selected_match.choice else {
        return Ok(area);
    };
    for input in inputs {
        let ref_count = &mut map_refs[input.node_id][input.polarity_index()];
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
        if !visiting.enter(*input) {
            return Err(anyhow!(
                "selected technology mapping contains an exact-area cycle at {:?}",
                input
            ));
        }
        let child_match = selected[input.node_id][input.polarity_index()]
            .as_ref()
            .ok_or_else(|| anyhow!("exact-area dereference is missing child {:?}", input))?;
        area += dereference_match_children_exact(
            cell_index,
            child_match,
            map_refs,
            selected,
            visiting,
        )?;
        visiting.leave(*input);
    }
    Ok(area)
}

/// References one trial cell's fanin cone and returns the exact area newly
/// made live. Every increment is logged so a rejected trial can be undone
/// without cloning the full reference-count vector.
#[allow(clippy::too_many_arguments)]
fn reference_match_children_exact(
    cell_index: &LibertyCellIndex,
    selected_match: &NfMatch,
    map_refs: &mut [[usize; 2]],
    selected: &[[Option<NfMatch>; 2]],
    matches: &[[StateMatches; 2]],
    increments: &mut Vec<StateKey>,
    newly_selected: &mut Vec<StateKey>,
    visiting: &mut DenseVisitMarks,
) -> Result<f64> {
    let mut area = match_cell_area(cell_index, selected_match);
    let MatchChoice::Cell { inputs, .. } = &selected_match.choice else {
        return Ok(area);
    };
    for input in inputs {
        let ref_count = &mut map_refs[input.node_id][input.polarity_index()];
        let was_unreferenced = *ref_count == 0;
        *ref_count += 1;
        increments.push(*input);
        if !was_unreferenced {
            continue;
        }
        if !visiting.enter(*input) {
            return Err(anyhow!(
                "trial technology mapping contains an exact-area cycle at {:?}",
                input
            ));
        }
        let child_match = selected_or_delay_match(selected, matches, *input)
            .ok_or_else(|| anyhow!("exact-area trial has no child match for {:?}", input))?;
        if selected[input.node_id][input.polarity_index()].is_none() {
            newly_selected.push(*input);
        }
        area += reference_match_children_exact(
            cell_index,
            child_match,
            map_refs,
            selected,
            matches,
            increments,
            newly_selected,
            visiting,
        )?;
        visiting.leave(*input);
    }
    Ok(area)
}

fn undo_reference_increments(map_refs: &mut [[usize; 2]], increments: &[StateKey]) {
    for state in increments.iter().rev() {
        let ref_count = &mut map_refs[state.node_id][state.polarity_index()];
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
        let slot = &mut requireds[input.node_id][input.polarity_index()];
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
    let previous = requireds[state.node_id][state.polarity_index()];
    if required >= previous - TIMING_EPSILON {
        return Ok(());
    }
    if !visiting.insert(state) {
        return Err(anyhow!(
            "fixed technology mapping contains a required-time cycle at {:?}",
            state
        ));
    }
    requireds[state.node_id][state.polarity_index()] = previous.min(required);
    let effective_required = requireds[state.node_id][state.polarity_index()];
    let selected_match = selected[state.node_id][state.polarity_index()]
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
    let selected_match = selected[state.node_id][state.polarity_index()]
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
    cell_index: &LibertyCellIndex,
    state: StateKey,
    selected: &[[Option<NfMatch>; 2]],
    memo: &mut [[Option<SolutionId>; 2]],
    solutions: &mut Vec<Solution>,
    visiting: &mut HashSet<StateKey>,
) -> Result<SolutionId> {
    if let Some(id) = memo[state.node_id][state.polarity_index()] {
        return Ok(id);
    }
    if !visiting.insert(state) {
        return Err(anyhow!(
            "selected technology mapping contains a reconstruction cycle at {:?}",
            state
        ));
    }
    let selected_match = selected[state.node_id][state.polarity_index()]
        .as_ref()
        .ok_or_else(|| anyhow!("selected mapping is missing state {:?}", state))?;
    let choice = match &selected_match.choice {
        MatchChoice::Source(source) => SolutionChoice::Source(*source),
        MatchChoice::Cell { binding, inputs } => {
            let mut solution_inputs = Vec::with_capacity(inputs.len());
            for input in inputs {
                solution_inputs.push(materialize_state(
                    cell_index, *input, selected, memo, solutions, visiting,
                )?);
            }
            SolutionChoice::Cell {
                binding: cell_index.binding(*binding).clone(),
                inputs: solution_inputs,
            }
        }
    };
    let id = SolutionId(solutions.len());
    solutions.push(Solution { choice });
    memo[state.node_id][state.polarity_index()] = Some(id);
    visiting.remove(&state);
    Ok(id)
}
