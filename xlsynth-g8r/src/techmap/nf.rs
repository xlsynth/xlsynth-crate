// SPDX-License-Identifier: Apache-2.0

//! Direct, choice-aware implementation of ABC's default `&nf` mapper.
//!
//! Structural cut priority and mapping are deliberately separate. ABC derives
//! its cut frontier from structural flow and unit depth before it considers
//! Liberty cell areas. Cell mapping then keeps exactly one delay match and one
//! area-flow match per concrete AIG object and output polarity. Unlike ABC's
//! intermediate GENLIB conversion, this implementation retains the exact area
//! recorded in the supplied Liberty model.

use crate::aig::{AigNode, AigOperand, AigRef, ChoiceAig, GateFn};
use crate::liberty_model::Library;
use crate::techmap::cover::{CoverPlan, Solution, SolutionChoice, SolutionId, SourceKind};
use crate::techmap::cuts::{ChoiceAnalysis, Cut};
use crate::techmap::liberty_index::{CellBindingId, LibertyCellIndex, RepresentativePinDelayTable};
use crate::techmap::truth::{
    CutLeaves, MAX_TRUTH_TABLE_INPUTS, complement_truth, minimize_support, remap_truth,
    variable_truth,
};
use crate::techmap::{
    TechMapOptions, TechMapTimingConstraints, TechMapTimingModel, scalar_bit_name,
};
use anyhow::{Result, anyhow};
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

const NF_AREA_FLOW_ROUNDS: usize = 4;
const NF_EXACT_AREA_ROUNDS: usize = 2;
const NF_EPSILON: f64 = 0.001;
const NF_TIMING_EPSILON: f64 = 1e-9;
const NF_UNIT_DELAY: f64 = 1.0;

/// The selected NF cover together with its independently enumerated cuts.
pub(super) struct NfCover {
    pub plan: CoverPlan,
    pub enumerated_cut_count: usize,
    pub representative_output_load: Option<f64>,
}

/// Maps using ABC's structural cuts, four flow rounds, and two exact rounds.
pub(super) fn build_cover_plan(
    choice_aig: &ChoiceAig,
    analysis: &ChoiceAnalysis,
    library: &Library,
    cell_index: &LibertyCellIndex,
    options: &TechMapOptions,
    constraints: &TechMapTimingConstraints,
) -> Result<NfCover> {
    let mut mapper = NfMapper::new(
        choice_aig,
        analysis,
        library,
        cell_index,
        options,
        constraints,
    )?;
    mapper.map()
}

/// One concrete AIG object and the desired polarity of its Boolean function.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct NfState {
    node_id: usize,
    polarity: bool,
}

impl NfState {
    /// Returns the dense polarity slot used by ABC's two-phase object state.
    fn polarity_index(self) -> usize {
        usize::from(self.polarity)
    }

    /// Returns the opposite phase of the same concrete AIG object.
    fn opposite(self) -> Self {
        Self {
            node_id: self.node_id,
            polarity: !self.polarity,
        }
    }
}

/// A concrete cell configuration in the original cut and root-library order.
#[derive(Clone, Debug)]
struct NfCandidate {
    binding: CellBindingId,
    inputs: SmallVec<[NfState; MAX_TRUTH_TABLE_INPUTS]>,
}

/// The implementation retained in one compact NF mapping match.
#[derive(Clone, Debug)]
enum NfChoice {
    Source(SourceKind),
    Cell {
        binding: CellBindingId,
        inputs: SmallVec<[NfState; MAX_TRUTH_TABLE_INPUTS]>,
    },
}

/// Scalar unit-delay arrival, shared-cone area flow, and reconstruction data.
#[derive(Clone, Debug)]
struct NfMatch {
    arrival: f64,
    flow: f64,
    choice: NfChoice,
}

/// ABC's direct and inverter-closed delay/area matches for one object phase.
#[derive(Clone, Debug, Default)]
struct NfPhaseMatches {
    direct_delay: Option<NfMatch>,
    direct_area: Option<NfMatch>,
    delay: Option<NfMatch>,
    area: Option<NfMatch>,
}

/// One flattened, ordered primary-output endpoint.
#[derive(Clone, Debug)]
struct NfOutput {
    name: String,
    state: NfState,
    required: Option<f64>,
}

/// Validates flattened primary-input constraints without taxing plain NF.
fn validate_primary_input_arrivals(
    graph: &GateFn,
    constraints: &TechMapTimingConstraints,
) -> Result<Option<Vec<f64>>> {
    if constraints.primary_input_arrivals.is_empty() {
        return Ok(None);
    }

    let mut input_nodes_by_name = BTreeMap::new();
    for input in &graph.inputs {
        let bit_count = input.get_bit_count();
        for (bit_index, operand) in input.bit_vector.iter_lsb_to_msb().enumerate() {
            if operand.negated {
                return Err(anyhow!(
                    "technology mapping does not support negated input-port bindings"
                ));
            }
            let name = scalar_bit_name(input.name.as_str(), bit_index, bit_count);
            if !matches!(
                graph.gates.get(operand.node.id),
                Some(AigNode::Input { .. })
            ) {
                return Err(anyhow!(
                    "primary input '{}' does not bind to AIG input node {}",
                    name,
                    operand.node.id
                ));
            }
            if input_nodes_by_name
                .insert(name.clone(), operand.node.id)
                .is_some()
            {
                return Err(anyhow!("technology mapping has duplicate input '{}'", name));
            }
        }
    }

    let mut arrivals = vec![0.0; graph.gates.len()];
    for (name, arrival) in &constraints.primary_input_arrivals {
        let Some(node_id) = input_nodes_by_name.get(name).copied() else {
            return Err(anyhow!(
                "timing constraint names unknown primary input '{}'",
                name
            ));
        };
        if !arrival.is_finite() || *arrival < 0.0 {
            return Err(anyhow!(
                "timing constraint for primary input '{}' must be non-negative and finite; got {}",
                name,
                arrival
            ));
        }
        arrivals[node_id] = *arrival;
    }
    Ok(Some(arrivals))
}

/// Reports an infeasible named output using the existing cover diagnostic.
fn validate_output_required(output: &NfOutput, arrival: f64) -> Result<()> {
    if let Some(required) = output.required
        && (!arrival.is_finite() || arrival > required + NF_TIMING_EPSILON)
    {
        return Err(anyhow!(
            "no cover meets required time {} for output '{}'; fastest estimated arrival is {}",
            required,
            output.name,
            arrival
        ));
    }
    Ok(())
}

/// A bounded frontier implementing `Nf_SetAddCut`'s insertion semantics.
struct NfCutFrontier {
    cuts: Vec<Cut>,
    limit: usize,
}

impl NfCutFrontier {
    /// Reserves the nontrivial-cut slots; a fanin unit cut is added on demand.
    fn new(max_cuts_per_node: usize) -> Self {
        let limit = max_cuts_per_node.saturating_sub(1).max(1);
        Self {
            cuts: Vec::with_capacity(limit),
            limit,
        }
    }

    /// Tests ABC's early containment check on the unminimized merged support.
    fn contains_subset_of(&self, leaves: &[AigRef]) -> bool {
        self.cuts
            .iter()
            .any(|existing| leaves_are_subset(existing.leaves.as_slice(), leaves))
    }

    /// Inserts a cut, drops all strict supersets, and preserves NF visit ties.
    fn insert(&mut self, cut: Cut) {
        self.cuts.retain(|existing| {
            !(cut.leaves.len() < existing.leaves.len()
                && leaves_are_subset(cut.leaves.as_slice(), existing.leaves.as_slice()))
        });

        // `Nf_SetSortByArea` moves a newly inserted cut before equal existing
        // cuts. Preserve that detail without relying on an unstable sort.
        let position = self
            .cuts
            .iter()
            .position(|existing| !nf_cut_order(existing, &cut).is_lt())
            .unwrap_or(self.cuts.len());
        self.cuts.insert(position, cut);
        if self.cuts.len() > self.limit {
            self.cuts.pop();
        }
    }

    /// Returns the final, already ordered nontrivial structural cut set.
    fn finish(self) -> Vec<Cut> {
        self.cuts
    }
}

/// Compares structural cuts exactly in ABC NF's useful/flow/depth/size order.
fn nf_cut_order(lhs: &Cut, rhs: &Cut) -> Ordering {
    if lhs.useful != rhs.useful {
        return rhs.useful.cmp(&lhs.useful);
    }
    if lhs.flow < rhs.flow - NF_EPSILON {
        return Ordering::Less;
    }
    if lhs.flow > rhs.flow + NF_EPSILON {
        return Ordering::Greater;
    }
    lhs.delay
        .total_cmp(&rhs.delay)
        .then_with(|| lhs.leaves.len().cmp(&rhs.leaves.len()))
}

/// Returns whether the sorted first leaf set is contained in the second.
fn leaves_are_subset(lhs: &[AigRef], rhs: &[AigRef]) -> bool {
    let mut lhs_index = 0;
    let mut rhs_index = 0;
    while lhs_index < lhs.len() && rhs_index < rhs.len() {
        match lhs[lhs_index].cmp(&rhs[rhs_index]) {
            Ordering::Less => return false,
            Ordering::Equal => {
                lhs_index += 1;
                rhs_index += 1;
            }
            Ordering::Greater => rhs_index += 1,
        }
    }
    lhs_index == lhs.len()
}

/// Merges two sorted bounded cut supports without hashing or heap allocation.
fn merge_leaves(lhs: &[AigRef], rhs: &[AigRef]) -> CutLeaves {
    let mut merged = CutLeaves::new();
    let mut lhs_index = 0;
    let mut rhs_index = 0;
    while lhs_index < lhs.len() && rhs_index < rhs.len() {
        match lhs[lhs_index].cmp(&rhs[rhs_index]) {
            Ordering::Less => {
                merged.push(lhs[lhs_index]);
                lhs_index += 1;
            }
            Ordering::Equal => {
                merged.push(lhs[lhs_index]);
                lhs_index += 1;
                rhs_index += 1;
            }
            Ordering::Greater => {
                merged.push(rhs[rhs_index]);
                rhs_index += 1;
            }
        }
    }
    merged.extend_from_slice(&lhs[lhs_index..]);
    merged.extend_from_slice(&rhs[rhs_index..]);
    merged
}

/// Creates the propagation-only unit cut for one concrete AIG object.
fn unit_cut(node: AigRef) -> Cut {
    Cut {
        leaves: CutLeaves::from_slice(&[node]),
        truth: variable_truth(1, 0),
        useful: false,
        flow: 0.0,
        delay: 0.0,
    }
}

/// Reproduces `Mf_ManSetFlowRefs`, including structural mux/XOR discounting.
fn initial_flow_references(graph: &GateFn) -> Vec<f64> {
    let mut references = vec![0_usize; graph.gates.len()];
    for node in &graph.gates {
        let AigNode::And2 { a, b, .. } = node else {
            continue;
        };
        for operand in [a, b] {
            if matches!(graph.gates[operand.node.id], AigNode::And2 { .. }) {
                references[operand.node.id] += 1;
            }
        }

        if let Some((control, data0, data1)) = recognize_structural_mux(graph, *a, *b) {
            if matches!(graph.gates[control.id], AigNode::And2 { .. }) {
                references[control.id] = references[control.id].saturating_sub(1);
            }
            if data0 == data1 && matches!(graph.gates[data0.id], AigNode::And2 { .. }) {
                references[data0.id] = references[data0.id].saturating_sub(1);
            }
        }
    }
    for output in &graph.outputs {
        for operand in output.bit_vector.iter_lsb_to_msb() {
            if matches!(graph.gates[operand.node.id], AigNode::And2 { .. }) {
                references[operand.node.id] += 1;
            }
        }
    }
    references
        .into_iter()
        .map(|count| count.max(1) as f64)
        .collect()
}

/// Finds the complemented-grandchild pattern used by ABC's mux discount.
fn recognize_structural_mux(
    graph: &GateFn,
    lhs: AigOperand,
    rhs: AigOperand,
) -> Option<(AigRef, AigRef, AigRef)> {
    if !lhs.negated || !rhs.negated {
        return None;
    }
    let AigNode::And2 {
        a: lhs0, b: lhs1, ..
    } = &graph.gates[lhs.node.id]
    else {
        return None;
    };
    let AigNode::And2 {
        a: rhs0, b: rhs1, ..
    } = &graph.gates[rhs.node.id]
    else {
        return None;
    };
    let left = [*lhs0, *lhs1];
    let right = [*rhs0, *rhs1];

    // `Gia_ObjRecognizeMux` checks these positions in this precise order.
    for (left_index, right_index) in [(1, 1), (0, 0), (0, 1), (1, 0)] {
        let left_control = left[left_index];
        let right_control = right[right_index];
        if left_control.node == right_control.node && left_control.negated != right_control.negated
        {
            return Some((
                left_control.node,
                left[1 - left_index].node,
                right[1 - right_index].node,
            ));
        }
    }
    None
}

/// Enumerates NF-priority choice cuts without consulting Liberty cell delay.
fn enumerate_nf_cuts(
    choice_aig: &ChoiceAig,
    analysis: &ChoiceAnalysis,
    cell_index: &LibertyCellIndex,
    options: &TechMapOptions,
    flow_references: &[f64],
) -> Result<Vec<Vec<Cut>>> {
    if options.max_cut_size == 0 || options.max_cut_size > MAX_TRUTH_TABLE_INPUTS {
        return Err(anyhow!(
            "max_cut_size must be in 1..={}, got {}",
            MAX_TRUTH_TABLE_INPUTS,
            options.max_cut_size
        ));
    }
    if options.max_cuts_per_node == 0 {
        return Err(anyhow!("max_cuts_per_node must be at least 1"));
    }

    let graph = choice_aig.graph();
    let mut cuts_by_node = vec![Vec::new(); graph.gates.len()];
    let mut cut_flows = vec![0.0; graph.gates.len()];
    let mut cut_delays = vec![0.0; graph.gates.len()];

    for (node_id, node) in graph.gates.iter().enumerate() {
        let node_ref = AigRef { id: node_id };
        match node {
            AigNode::Input { .. } => {
                cuts_by_node[node_id] = vec![unit_cut(node_ref)];
            }
            AigNode::Literal { value, .. } => {
                cuts_by_node[node_id] = vec![Cut {
                    leaves: CutLeaves::new(),
                    truth: u64::from(*value),
                    useful: true,
                    flow: 0.0,
                    delay: 0.0,
                }];
            }
            AigNode::And2 { a, b, .. } => {
                if a.node.id >= node_id || b.node.id >= node_id {
                    return Err(anyhow!(
                        "technology mapping requires topological AIG storage: node {} depends on a non-earlier fanin",
                        node_id
                    ));
                }

                let mut frontier = NfCutFrontier::new(options.max_cuts_per_node);
                if let Some(sibling) = choice_aig.next_sibling(node_ref) {
                    let complemented =
                        analysis.phase_by_node[node_id] ^ analysis.phase_by_node[sibling.id];
                    for sibling_cut in &cuts_by_node[sibling.id] {
                        let mut cut: Cut = sibling_cut.clone();
                        if complemented {
                            cut.truth = complement_truth(cut.truth, cut.leaves.len());
                        }
                        refresh_nf_cut(
                            &mut cut,
                            node_id,
                            cell_index,
                            flow_references,
                            &cut_flows,
                            &cut_delays,
                        );
                        frontier.insert(cut);
                    }
                }

                let lhs_cuts = prepare_fanin_cuts(graph, &cuts_by_node, a.node);
                let rhs_cuts = prepare_fanin_cuts(graph, &cuts_by_node, b.node);
                for lhs in lhs_cuts.iter() {
                    for rhs in rhs_cuts.iter() {
                        if lhs.leaves.len() + rhs.leaves.len() > options.max_cut_size
                            && (leaf_signature(lhs.leaves.as_slice())
                                | leaf_signature(rhs.leaves.as_slice()))
                            .count_ones() as usize
                                > options.max_cut_size
                        {
                            continue;
                        }
                        let merged = merge_leaves(lhs.leaves.as_slice(), rhs.leaves.as_slice());
                        if merged.len() > options.max_cut_size
                            || frontier.contains_subset_of(merged.as_slice())
                        {
                            continue;
                        }

                        let mut lhs_truth =
                            remap_truth(lhs.truth, lhs.leaves.as_slice(), merged.as_slice());
                        let mut rhs_truth =
                            remap_truth(rhs.truth, rhs.leaves.as_slice(), merged.as_slice());
                        if a.negated {
                            lhs_truth = complement_truth(lhs_truth, merged.len());
                        }
                        if b.negated {
                            rhs_truth = complement_truth(rhs_truth, merged.len());
                        }
                        let (truth, leaves) =
                            minimize_support(lhs_truth & rhs_truth, merged.as_slice());
                        let mut cut = Cut {
                            leaves,
                            truth,
                            useful: false,
                            flow: 0.0,
                            delay: 0.0,
                        };
                        refresh_nf_cut(
                            &mut cut,
                            node_id,
                            cell_index,
                            flow_references,
                            &cut_flows,
                            &cut_delays,
                        );
                        frontier.insert(cut);
                    }
                }

                let cuts = frontier.finish();
                if let Some(best) = cuts.first() {
                    cut_flows[node_id] = best.flow;
                    cut_delays[node_id] = best.delay;
                }
                cuts_by_node[node_id] = cuts;
            }
        }
    }
    Ok(cuts_by_node)
}

/// A borrowed NF cut set and its optional, propagation-only unit cut.
struct NfPreparedCuts<'a> {
    stored: &'a [Cut],
    unit: Option<Cut>,
}

impl NfPreparedCuts<'_> {
    /// Visits saved structural cuts before ABC's optional fanin unit cut.
    fn iter(&self) -> impl Iterator<Item = &Cut> {
        self.stored.iter().chain(self.unit.iter())
    }
}

/// Adds a unit fanin cut without cloning the entire stored cut frontier.
fn prepare_fanin_cuts<'a>(
    graph: &GateFn,
    cuts_by_node: &'a [Vec<Cut>],
    node: AigRef,
) -> NfPreparedCuts<'a> {
    let stored = cuts_by_node[node.id].as_slice();
    let needs_unit = stored.is_empty()
        || (matches!(graph.gates[node.id], AigNode::And2 { .. })
            && stored.first().is_some_and(|cut| cut.leaves.len() > 1));
    NfPreparedCuts {
        stored,
        unit: needs_unit.then(|| unit_cut(node)),
    }
}

/// Returns the compact, conservative leaf signature used before cut merging.
fn leaf_signature(leaves: &[AigRef]) -> u64 {
    leaves.iter().fold(0_u64, |signature, leaf| {
        signature | (1_u64 << (leaf.id & 63))
    })
}

/// Assigns NF's structural usefulness, unit delay, and fanout-divided flow.
fn refresh_nf_cut(
    cut: &mut Cut,
    root_id: usize,
    cell_index: &LibertyCellIndex,
    flow_references: &[f64],
    cut_flows: &[f64],
    cut_delays: &[f64],
) {
    cut.useful = cut.leaves.is_empty()
        || !cell_index.matches(cut.leaves.len(), cut.truth).is_empty()
        || !cell_index
            .matches(
                cut.leaves.len(),
                complement_truth(cut.truth, cut.leaves.len()),
            )
            .is_empty();

    let mut flow = if cut.leaves.len() < 2 {
        0.0
    } else {
        cut.leaves.len() as f64
    };
    let mut delay: f64 = 0.0;
    for leaf in &cut.leaves {
        flow += cut_flows[leaf.id];
        delay = delay.max(cut_delays[leaf.id]);
    }
    cut.flow = flow / (2.0 * flow_references[root_id].max(1.0));
    cut.delay = delay + f64::from(cut.leaves.len() > 1);
}

/// Compact, single-objective state for a complete ABC-shaped NF mapping.
struct NfMapper<'a> {
    choice_aig: &'a ChoiceAig,
    cell_index: &'a LibertyCellIndex,
    pin_delays: Option<RepresentativePinDelayTable>,
    input_arrivals: Option<Vec<f64>>,
    cuts_by_node: Vec<Vec<Cut>>,
    candidates: Vec<[Vec<NfCandidate>; 2]>,
    matches: Vec<[NfPhaseMatches; 2]>,
    selected: Vec<[Option<NfMatch>; 2]>,
    best: Vec<[Option<NfMatch>; 2]>,
    requireds: Vec<[f64; 2]>,
    map_refs: Vec<[usize; 2]>,
    flow_refs: Vec<[f64; 2]>,
    outputs: Vec<NfOutput>,
    inverter: Option<CellBindingId>,
    has_endpoint_constraints: bool,
    target_delay: f64,
    matched_candidate_count: usize,
    enumerated_cut_count: usize,
}

impl<'a> NfMapper<'a> {
    /// Builds the structural cut frontier and one deterministic match cache.
    fn new(
        choice_aig: &'a ChoiceAig,
        analysis: &ChoiceAnalysis,
        library: &Library,
        cell_index: &'a LibertyCellIndex,
        options: &TechMapOptions,
        constraints: &TechMapTimingConstraints,
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
        let has_endpoint_constraints = !constraints.primary_input_arrivals.is_empty()
            || !constraints.primary_output_required.is_empty();
        let input_arrivals = validate_primary_input_arrivals(graph, constraints)?;

        let mut outputs = Vec::new();
        let mut output_names = BTreeSet::new();
        for output in &graph.outputs {
            let bit_count = output.get_bit_count();
            for (bit_index, operand) in output.bit_vector.iter_lsb_to_msb().enumerate() {
                let name = scalar_bit_name(output.name.as_str(), bit_index, bit_count);
                if !output_names.insert(name.clone()) {
                    return Err(anyhow!(
                        "technology mapping has duplicate output '{}'",
                        name
                    ));
                }
                outputs.push(NfOutput {
                    required: constraints.primary_output_required.get(&name).copied(),
                    name,
                    state: NfState {
                        node_id: operand.node.id,
                        polarity: operand.negated,
                    },
                });
            }
        }
        for (name, required) in &constraints.primary_output_required {
            if !output_names.contains(name) {
                return Err(anyhow!(
                    "timing constraint names unknown primary output '{}'",
                    name
                ));
            }
            if !required.is_finite() || *required < 0.0 {
                return Err(anyhow!(
                    "timing constraint for primary output '{}' must be non-negative and finite; got {}",
                    name,
                    required
                ));
            }
        }

        // Endpoint constraints are expressed in Liberty time units even when
        // the unconstrained cover objective is ABC's unit-delay NF model.
        let pin_delays = (options.timing_model == TechMapTimingModel::NfLiberty
            || has_endpoint_constraints)
            .then(|| {
                cell_index.representative_pin_delays(library, options.primary_input_transition)
            })
            .transpose()?;
        let structural_refs = initial_flow_references(graph);
        let cuts_by_node = enumerate_nf_cuts(
            choice_aig,
            analysis,
            cell_index,
            options,
            structural_refs.as_slice(),
        )?;
        let enumerated_cut_count = cuts_by_node.iter().map(Vec::len).sum();
        let node_count = graph.gates.len();
        let mut candidates = (0..node_count)
            .map(|_| [Vec::new(), Vec::new()])
            .collect::<Vec<_>>();
        let mut matched_candidate_count = 0;

        for (node_id, node) in graph.gates.iter().enumerate() {
            if !matches!(node, AigNode::And2 { .. }) {
                continue;
            }
            for cut in &cuts_by_node[node_id] {
                if cut.leaves.iter().any(|leaf| leaf.id >= node_id) {
                    continue;
                }
                for polarity in [false, true] {
                    let truth = if polarity {
                        complement_truth(cut.truth, cut.leaves.len())
                    } else {
                        cut.truth
                    };
                    for binding_id in cell_index.matches(cut.leaves.len(), truth) {
                        let binding = cell_index.binding(*binding_id);
                        let inputs = binding
                            .input_to_leaf
                            .iter()
                            .copied()
                            .zip(binding.input_negated.iter().copied())
                            .map(|(leaf_index, negated)| NfState {
                                node_id: cut.leaves[leaf_index].id,
                                polarity: negated,
                            })
                            .collect();
                        candidates[node_id][usize::from(polarity)].push(NfCandidate {
                            binding: *binding_id,
                            inputs,
                        });
                        matched_candidate_count += 1;
                    }
                }
            }
        }

        let inverter = cell_index
            .matches(1, complement_truth(variable_truth(1, 0), 1))
            .iter()
            .copied()
            .find(|binding| !cell_index.binding(*binding).input_negated[0]);
        let flow_refs = structural_refs
            .into_iter()
            .map(|references| [references, references])
            .collect();
        let mut requireds = vec![[f64::INFINITY; 2]; node_count];
        if has_endpoint_constraints {
            for output in &outputs {
                if let Some(required) = output.required {
                    let slot = &mut requireds[output.state.node_id][output.state.polarity_index()];
                    *slot = slot.min(required);
                }
            }
        }

        Ok(Self {
            choice_aig,
            cell_index,
            pin_delays,
            input_arrivals,
            cuts_by_node,
            candidates,
            matches: (0..node_count)
                .map(|_| [NfPhaseMatches::default(), NfPhaseMatches::default()])
                .collect(),
            selected: (0..node_count).map(|_| [None, None]).collect(),
            best: (0..node_count).map(|_| [None, None]).collect(),
            requireds,
            map_refs: vec![[0; 2]; node_count],
            flow_refs,
            outputs,
            inverter,
            has_endpoint_constraints,
            target_delay: 0.0,
            matched_candidate_count,
            enumerated_cut_count,
        })
    }

    /// Executes the four mapping/reference rounds and two exact-area rounds.
    fn map(&mut self) -> Result<NfCover> {
        for round in 0..NF_AREA_FLOW_ROUNDS {
            self.compute_round_matches(round)?;
            self.set_mapping_references(round)?;
        }

        for round in 0..NF_EXACT_AREA_ROUNDS {
            self.recover_exact_area(round)?;
        }
        self.fix_primary_output_drivers()?;
        if self.has_endpoint_constraints {
            // Exact recovery visits roots before their shared children. Bring
            // every selected arrival up to date after the last recovery and
            // any complemented-output cleanup before checking endpoints.
            self.reset_exact_matches(NF_EXACT_AREA_ROUNDS)?;
            self.validate_output_requirements()?;
        }
        self.materialize()
    }

    /// Returns either ABC's unit cost or one characterized cell-pin delay.
    fn pin_delay(&self, binding: CellBindingId, input_index: usize) -> f64 {
        self.pin_delays.as_ref().map_or(NF_UNIT_DELAY, |delays| {
            delays.pin_delay(binding, input_index)
        })
    }

    /// Returns the timing cost of the selected explicit phase inverter.
    fn inverter_delay(&self) -> f64 {
        self.inverter
            .map_or(NF_UNIT_DELAY, |inverter| self.pin_delay(inverter, 0))
    }

    /// Computes one delay and one flow match for every concrete object phase.
    fn compute_round_matches(&mut self, round: usize) -> Result<()> {
        for node_id in 0..self.choice_aig.graph().gates.len() {
            if round != 0 {
                for polarity in [false, true] {
                    let state = NfState { node_id, polarity };
                    if self.requireds[node_id][state.polarity_index()].is_infinite() {
                        self.requireds[node_id][state.polarity_index()] =
                            self.infer_required(state);
                    }
                }
            }

            let mut phases = [NfPhaseMatches::default(), NfPhaseMatches::default()];
            for polarity in [false, true] {
                let state = NfState { node_id, polarity };
                let mut direct_delay = self.source_match(state);
                let mut direct_area = direct_delay.clone();
                let required = self.requireds[node_id][state.polarity_index()];

                for candidate in &self.candidates[node_id][state.polarity_index()] {
                    let Some(candidate_match) =
                        self.score_flow_candidate(state, candidate, required)
                    else {
                        continue;
                    };
                    if candidate_match.arrival > required
                        && direct_delay.is_some()
                        && direct_area.is_some()
                    {
                        continue;
                    }
                    if direct_delay
                        .as_ref()
                        .is_none_or(|best| candidate_match.arrival < best.arrival)
                    {
                        direct_delay = Some(candidate_match.clone());
                    }
                    if direct_area
                        .as_ref()
                        .is_none_or(|best| candidate_match.flow < best.flow - NF_EPSILON)
                    {
                        direct_area = Some(candidate_match);
                    }
                }

                let slot = &mut phases[state.polarity_index()];
                slot.direct_delay = direct_delay.clone();
                slot.direct_area = direct_area.clone();
                slot.delay = direct_delay;
                slot.area = direct_area;
            }

            self.close_inverter_phases(node_id, &mut phases, round);
            self.matches[node_id] = phases;
        }

        for output in &self.outputs {
            if self.matches[output.state.node_id][output.state.polarity_index()]
                .delay
                .is_none()
            {
                return Err(anyhow!(
                    "could not map output '{}' from AIG node {} in polarity {}",
                    output.name,
                    output.state.node_id,
                    usize::from(output.state.polarity)
                ));
            }
        }
        Ok(())
    }

    /// Infers a finite bound for an otherwise unreferenced previous match.
    fn infer_required(&self, state: NfState) -> f64 {
        let Some(previous) = self.matches[state.node_id][state.polarity_index()]
            .delay
            .as_ref()
        else {
            return f64::INFINITY;
        };
        let NfChoice::Cell { binding, inputs } = &previous.choice else {
            return previous.arrival;
        };
        let mut inferred: f64 = 0.0;
        for (input_index, child) in inputs.iter().enumerate() {
            let pin_delay = self.pin_delay(*binding, input_index);
            if let Some(child_match) = self.matches[child.node_id][child.polarity_index()]
                .delay
                .as_ref()
            {
                inferred = inferred.max(child_match.arrival + pin_delay);
            }
            let child_required = self.requireds[child.node_id][child.polarity_index()];
            if child_required.is_finite() {
                inferred = inferred.max(child_required + pin_delay);
            }
        }
        inferred
    }

    /// Creates zero-area primary-input and literal mapping matches.
    fn source_match(&self, state: NfState) -> Option<NfMatch> {
        let node = AigRef { id: state.node_id };
        let source = match &self.choice_aig.graph().gates[state.node_id] {
            AigNode::Input { .. } if !state.polarity => SourceKind::Input(node),
            AigNode::Input { .. } => return None,
            AigNode::Literal { value, .. } => SourceKind::Literal(*value ^ state.polarity),
            AigNode::And2 { .. } => {
                let cut = self.cuts_by_node[state.node_id]
                    .iter()
                    .find(|cut| cut.leaves.is_empty())?;
                SourceKind::Literal((cut.truth & 1 != 0) ^ state.polarity)
            }
        };
        Some(NfMatch {
            arrival: match source {
                SourceKind::Input(node) => self
                    .input_arrivals
                    .as_ref()
                    .map_or(0.0, |arrivals| arrivals[node.id]),
                SourceKind::Literal(_) => 0.0,
            },
            flow: 0.0,
            choice: NfChoice::Source(source),
        })
    }

    /// Applies NF's per-pin area-under-required choice and selected arc delay.
    fn score_flow_candidate(
        &self,
        state: NfState,
        candidate: &NfCandidate,
        required: f64,
    ) -> Option<NfMatch> {
        let binding = self.cell_index.binding(candidate.binding);
        let mut arrival: f64 = 0.0;
        let mut flow = binding.area;

        for (input_index, child) in candidate.inputs.iter().enumerate() {
            let pin_delay = self.pin_delay(candidate.binding, input_index);
            let child_matches = &self.matches[child.node_id][child.polarity_index()];
            let delay_match = child_matches.delay.as_ref()?;
            let chosen = if required.is_finite()
                && child_matches
                    .area
                    .as_ref()
                    .is_some_and(|area| area.arrival + pin_delay <= required)
            {
                child_matches
                    .area
                    .as_ref()
                    .expect("finite feasible area match was just checked")
            } else {
                delay_match
            };
            arrival = arrival.max(chosen.arrival + pin_delay);
            flow += chosen.flow;
        }

        Some(NfMatch {
            arrival,
            flow: flow / self.flow_refs[state.node_id][state.polarity_index()].max(1.0),
            choice: NfChoice::Cell {
                binding: candidate.binding,
                inputs: candidate.inputs.clone(),
            },
        })
    }

    /// Implements ABC's direct-phase-first explicit inverter closure.
    fn close_inverter_phases(
        &self,
        node_id: usize,
        phases: &mut [NfPhaseMatches; 2],
        round: usize,
    ) {
        let Some(inverter) = self.inverter else {
            return;
        };
        let inverter_area = self.cell_index.binding(inverter).area;
        let inverter_delay = self.pin_delay(inverter, 0);

        for polarity in [false, true] {
            let state = NfState { node_id, polarity };
            let opposite = state.opposite();
            let Some(opposite_delay) = phases[opposite.polarity_index()].delay.clone() else {
                continue;
            };
            let candidate = inverter_match(
                state,
                &opposite_delay,
                inverter,
                inverter_area,
                inverter_delay,
            );
            if phases[state.polarity_index()]
                .delay
                .as_ref()
                .is_none_or(|best| candidate.arrival < best.arrival)
            {
                phases[state.polarity_index()].delay = Some(candidate.clone());
                if phases[state.polarity_index()].area.is_none() {
                    phases[state.polarity_index()].area = Some(candidate);
                }
            }
        }

        // Initial CI polarity preparation requires an inverter, but ABC's
        // area-improving opposite-phase substitutions start after round zero.
        if round == 0 {
            return;
        }
        for polarity in [false, true] {
            let state = NfState { node_id, polarity };
            let opposite = state.opposite();
            let Some(opposite_area) = phases[opposite.polarity_index()].area.clone() else {
                continue;
            };
            let candidate = inverter_match(
                state,
                &opposite_area,
                inverter,
                inverter_area,
                inverter_delay,
            );
            let required = self.requireds[node_id][state.polarity_index()];
            if candidate.arrival <= required
                && phases[state.polarity_index()]
                    .area
                    .as_ref()
                    .is_none_or(|best| candidate.flow < best.flow - NF_EPSILON)
            {
                phases[state.polarity_index()].area = Some(candidate);
            }
        }
    }

    /// Selects the live cover in reverse order and propagates its references.
    fn set_mapping_references(&mut self, round: usize) -> Result<()> {
        let mut fastest: f64 = 0.0;
        for output in &self.outputs {
            let selected = self.matches[output.state.node_id][output.state.polarity_index()]
                .delay
                .as_ref()
                .ok_or_else(|| anyhow!("output '{}' has no NF delay match", output.name))?;
            if self.has_endpoint_constraints {
                validate_output_required(output, selected.arrival)?;
            }
            fastest = fastest.max(selected.arrival);
        }
        self.target_delay = self.target_delay.max(fastest);
        self.requireds.fill([f64::INFINITY; 2]);
        self.map_refs.fill([0; 2]);
        for slots in &mut self.selected {
            *slots = [None, None];
        }

        for output in &self.outputs {
            let state = output.state;
            self.map_refs[state.node_id][state.polarity_index()] += 1;
            let required = &mut self.requireds[state.node_id][state.polarity_index()];
            *required = required.min(output.required.unwrap_or(self.target_delay));
        }

        for node_id in (0..self.choice_aig.graph().gates.len()).rev() {
            if self.map_refs[node_id] == [0, 0] {
                continue;
            }
            self.select_referenced_node(node_id)?;
        }

        let coefficient = 1.0 / (1.0 + ((round + 1) * (round + 1)) as f64);
        for (flow, refs) in self.flow_refs.iter_mut().zip(&self.map_refs) {
            for polarity in 0..2 {
                flow[polarity] = (coefficient * flow[polarity]
                    + (1.0 - coefficient) * refs[polarity].max(1) as f64)
                    .max(1.0);
            }
        }
        Ok(())
    }

    /// Picks both referenced polarities without creating an inverter cycle.
    fn select_referenced_node(&mut self, node_id: usize) -> Result<()> {
        let mut chosen: [Option<NfMatch>; 2] = [None, None];
        for polarity in [false, true] {
            let state = NfState { node_id, polarity };
            if self.map_refs[node_id][state.polarity_index()] == 0 {
                continue;
            }
            chosen[state.polarity_index()] = Some(self.match_under_required(state, false)?);
        }

        if chosen[0]
            .as_ref()
            .is_some_and(|selected| self.is_same_node_inverter(node_id, selected))
            && chosen[1]
                .as_ref()
                .is_some_and(|selected| self.is_same_node_inverter(node_id, selected))
        {
            for polarity in [false, true] {
                let state = NfState { node_id, polarity };
                if self.map_refs[node_id][state.polarity_index()] != 0 {
                    chosen[state.polarity_index()] = Some(self.match_under_required(state, true)?);
                }
            }
        }

        for polarity in [false, true] {
            let state = NfState { node_id, polarity };
            let Some(selected) = chosen[state.polarity_index()].as_ref() else {
                continue;
            };
            if self.is_same_node_inverter(node_id, selected) {
                let opposite = state.opposite();
                self.map_refs[opposite.node_id][opposite.polarity_index()] += 1;
                let required =
                    self.requireds[node_id][state.polarity_index()] - self.inverter_delay();
                self.update_required(opposite, required);
                chosen[opposite.polarity_index()] =
                    Some(self.match_under_required(opposite, true)?);
            }
        }

        for polarity in [false, true] {
            let state = NfState { node_id, polarity };
            let Some(selected) = chosen[state.polarity_index()].take() else {
                continue;
            };
            if !self.is_same_node_inverter(node_id, &selected) {
                self.propagate_choice(
                    &selected.choice,
                    self.requireds[node_id][state.polarity_index()],
                );
            }
            self.selected[node_id][state.polarity_index()] = Some(selected);
        }
        Ok(())
    }

    /// Selects the area match when it meets required time, as in `&nf`.
    fn match_under_required(&self, state: NfState, direct_only: bool) -> Result<NfMatch> {
        let matches = &self.matches[state.node_id][state.polarity_index()];
        let required = self.requireds[state.node_id][state.polarity_index()];
        let (area, delay) = if direct_only {
            (&matches.direct_area, &matches.direct_delay)
        } else {
            (&matches.area, &matches.delay)
        };
        if let Some(area_match) = area
            && area_match.arrival <= required
        {
            return Ok(area_match.clone());
        }
        delay.clone().ok_or_else(|| {
            anyhow!(
                "AIG node {} in polarity {} has no {}NF mapping match",
                state.node_id,
                state.polarity_index(),
                if direct_only { "direct " } else { "" }
            )
        })
    }

    /// Reports whether a match is the designated inverter of the same node.
    fn is_same_node_inverter(&self, node_id: usize, selected: &NfMatch) -> bool {
        let NfChoice::Cell { binding, inputs } = &selected.choice else {
            return false;
        };
        Some(*binding) == self.inverter && inputs.len() == 1 && inputs[0].node_id == node_id
    }

    /// Propagates one selected cell's references and per-pin required times.
    fn propagate_choice(&mut self, choice: &NfChoice, required: f64) {
        let NfChoice::Cell { binding, inputs } = choice else {
            return;
        };
        for (input_index, child) in inputs.iter().enumerate() {
            let pin_delay = self.pin_delay(*binding, input_index);
            self.map_refs[child.node_id][child.polarity_index()] += 1;
            self.update_required(*child, required - pin_delay);
        }
    }

    /// Tightens the backward required time of one concrete mapping state.
    fn update_required(&mut self, state: NfState, required: f64) {
        let slot = &mut self.requireds[state.node_id][state.polarity_index()];
        *slot = slot.min(required);
    }

    /// Reselects mapped cones using exact reference/dereference cell areas.
    fn recover_exact_area(&mut self, round: usize) -> Result<()> {
        self.reset_exact_matches(round)?;
        self.requireds.fill([f64::INFINITY; 2]);
        for output in &self.outputs {
            let state = output.state;
            let required = &mut self.requireds[state.node_id][state.polarity_index()];
            *required = required.min(output.required.unwrap_or(self.target_delay));
        }

        for node_id in (0..self.choice_aig.graph().gates.len()).rev() {
            for polarity in [false, true] {
                let state = NfState { node_id, polarity };
                if self.map_refs[node_id][state.polarity_index()] == 0 {
                    continue;
                }
                let required = self.requireds[node_id][state.polarity_index()];
                let current = self.best[node_id][state.polarity_index()]
                    .clone()
                    .ok_or_else(|| anyhow!("live NF state has no exact-area match"))?;

                if self.is_same_node_inverter(node_id, &current) {
                    self.update_required(state.opposite(), required - self.inverter_delay());
                    self.selected[node_id][state.polarity_index()] = Some(current);
                    continue;
                }
                if matches!(current.choice, NfChoice::Source(_)) {
                    self.selected[node_id][state.polarity_index()] = Some(current);
                    continue;
                }

                self.dereference_choice(&current.choice)?;
                let mut winner = current.clone();
                winner.arrival = self.choice_arrival(&winner.choice)?;
                let mut winner_area = self.trial_reference_area(&winner.choice)?;

                let candidate_count = self.candidates[node_id][state.polarity_index()].len();
                for candidate_index in 0..candidate_count {
                    let candidate =
                        self.candidates[node_id][state.polarity_index()][candidate_index].clone();
                    let Some(trial) = self.exact_candidate(&candidate, required) else {
                        continue;
                    };
                    let trial_area = self.trial_reference_area(&trial.choice)?;
                    if trial_area < winner_area - NF_EPSILON
                        || ((trial_area - winner_area).abs() <= NF_EPSILON
                            && trial.arrival < winner.arrival)
                    {
                        winner = trial;
                        winner_area = trial_area;
                    }
                }

                let mut permanent = Vec::new();
                self.reference_choice(&winner.choice, &mut permanent)?;
                self.best[node_id][state.polarity_index()] = Some(winner.clone());
                self.selected[node_id][state.polarity_index()] = Some(winner.clone());
                if let NfChoice::Cell { binding, inputs } = &winner.choice {
                    for (input_index, child) in inputs.iter().enumerate() {
                        let pin_delay = self.pin_delay(*binding, input_index);
                        self.update_required(*child, required - pin_delay);
                        if self.best[child.node_id][child.polarity_index()]
                            .as_ref()
                            .is_some_and(|chosen| self.is_same_node_inverter(child.node_id, chosen))
                        {
                            self.update_required(
                                child.opposite(),
                                required - pin_delay - self.inverter_delay(),
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Promotes the selected cover and recomputes its current arc arrivals.
    fn reset_exact_matches(&mut self, round: usize) -> Result<()> {
        for node_id in 0..self.choice_aig.graph().gates.len() {
            for polarity in [false, true] {
                let state = NfState { node_id, polarity };
                let matches = &self.matches[node_id][state.polarity_index()];
                let selected = if self.map_refs[node_id][state.polarity_index()] > 0 {
                    self.selected[node_id][state.polarity_index()].clone()
                } else if round % 2 == 1 {
                    matches
                        .direct_area
                        .clone()
                        .or_else(|| matches.direct_delay.clone())
                        .or_else(|| matches.delay.clone())
                } else {
                    matches.delay.clone()
                };
                self.best[node_id][state.polarity_index()] = selected;
            }

            for polarity in [false, true] {
                let state = NfState { node_id, polarity };
                let Some(mut selected) = self.best[node_id][state.polarity_index()].clone() else {
                    continue;
                };
                if self.is_same_node_inverter(node_id, &selected) {
                    continue;
                }
                selected.arrival = self.choice_arrival(&selected.choice)?;
                self.best[node_id][state.polarity_index()] = Some(selected.clone());
                if self.map_refs[node_id][state.polarity_index()] != 0 {
                    self.selected[node_id][state.polarity_index()] = Some(selected);
                }
            }

            for polarity in [false, true] {
                let state = NfState { node_id, polarity };
                let Some(mut selected) = self.best[node_id][state.polarity_index()].clone() else {
                    continue;
                };
                if !self.is_same_node_inverter(node_id, &selected) {
                    continue;
                }
                selected.arrival = self.choice_arrival(&selected.choice)?;
                self.best[node_id][state.polarity_index()] = Some(selected.clone());
                if self.map_refs[node_id][state.polarity_index()] != 0 {
                    self.selected[node_id][state.polarity_index()] = Some(selected);
                }
            }
        }
        Ok(())
    }

    /// Computes one selected cell's current arrival from chosen child matches.
    fn choice_arrival(&self, choice: &NfChoice) -> Result<f64> {
        let (binding, inputs) = match choice {
            NfChoice::Source(SourceKind::Input(node)) => {
                return Ok(self
                    .input_arrivals
                    .as_ref()
                    .map_or(0.0, |arrivals| arrivals[node.id]));
            }
            NfChoice::Source(SourceKind::Literal(_)) => return Ok(0.0),
            NfChoice::Cell { binding, inputs } => (binding, inputs),
        };
        let mut arrival: f64 = 0.0;
        for (input_index, child) in inputs.iter().enumerate() {
            let selected = self.best[child.node_id][child.polarity_index()]
                .as_ref()
                .ok_or_else(|| {
                    anyhow!(
                        "NF cell depends on unmapped AIG node {} in polarity {}",
                        child.node_id,
                        child.polarity_index()
                    )
                })?;
            arrival = arrival.max(selected.arrival + self.pin_delay(*binding, input_index));
        }
        Ok(arrival)
    }

    /// Builds a timing-feasible exact-area candidate from current child states.
    fn exact_candidate(&self, candidate: &NfCandidate, required: f64) -> Option<NfMatch> {
        let mut arrival: f64 = 0.0;
        for (input_index, child) in candidate.inputs.iter().enumerate() {
            let child_match = self.best[child.node_id][child.polarity_index()].as_ref()?;
            arrival =
                arrival.max(child_match.arrival + self.pin_delay(candidate.binding, input_index));
            if arrival > required {
                return None;
            }
        }
        Some(NfMatch {
            arrival,
            flow: 0.0,
            choice: NfChoice::Cell {
                binding: candidate.binding,
                inputs: candidate.inputs.clone(),
            },
        })
    }

    /// Charges only cells whose fanin cones become live in this trial.
    fn trial_reference_area(&mut self, choice: &NfChoice) -> Result<f64> {
        let mut increments = Vec::new();
        let area = self.reference_choice(choice, &mut increments)?;
        for state in increments.into_iter().rev() {
            let references = &mut self.map_refs[state.node_id][state.polarity_index()];
            if *references == 0 {
                return Err(anyhow!("NF exact-area trial lost a recorded reference"));
            }
            *references -= 1;
        }
        Ok(area)
    }

    /// Recursively references a cone, charging only first-live children.
    fn reference_choice(&mut self, choice: &NfChoice, backup: &mut Vec<NfState>) -> Result<f64> {
        let NfChoice::Cell { binding, inputs } = choice else {
            return Ok(0.0);
        };
        let mut area = self.cell_index.binding(*binding).area;
        for child in inputs {
            let previous = self.map_refs[child.node_id][child.polarity_index()];
            self.map_refs[child.node_id][child.polarity_index()] += 1;
            backup.push(*child);
            if previous == 0 {
                let child_choice = self.best[child.node_id][child.polarity_index()]
                    .as_ref()
                    .ok_or_else(|| anyhow!("NF exact-area trial references an unmapped child"))?
                    .choice
                    .clone();
                area += self.reference_choice(&child_choice, backup)?;
            }
        }
        Ok(area)
    }

    /// Recursively removes exactly the cone that loses its last reference.
    fn dereference_choice(&mut self, choice: &NfChoice) -> Result<f64> {
        let NfChoice::Cell { binding, inputs } = choice else {
            return Ok(0.0);
        };
        let mut area = self.cell_index.binding(*binding).area;
        for child in inputs {
            let references = &mut self.map_refs[child.node_id][child.polarity_index()];
            if *references == 0 {
                return Err(anyhow!(
                    "NF exact-area recovery encountered an unreferenced child"
                ));
            }
            *references -= 1;
            if *references == 0 {
                let child_choice = self.best[child.node_id][child.polarity_index()]
                    .as_ref()
                    .ok_or_else(|| anyhow!("NF exact-area recovery lost a child match"))?
                    .choice
                    .clone();
                area += self.dereference_choice(&child_choice)?;
            }
        }
        Ok(area)
    }

    /// Moves a legal output inverter off the separately implemented phase.
    fn fix_primary_output_drivers(&mut self) -> Result<()> {
        let Some(inverter) = self.inverter else {
            return Ok(());
        };
        let inverter_area = self.cell_index.binding(inverter).area;
        let inverter_delay = self.pin_delay(inverter, 0);
        for output_index in 0..self.outputs.len() {
            let state = self.outputs[output_index].state;
            if !matches!(
                self.choice_aig.graph().gates[state.node_id],
                AigNode::And2 { .. }
            ) || self.map_refs[state.node_id][0] == 0
                || self.map_refs[state.node_id][1] == 0
            {
                continue;
            }

            let opposite = state.opposite();
            let Some(current) = self.best[state.node_id][state.polarity_index()].clone() else {
                continue;
            };
            let Some(opposite_match) =
                self.best[opposite.node_id][opposite.polarity_index()].clone()
            else {
                continue;
            };
            let required = if self.has_endpoint_constraints {
                self.requireds[state.node_id][state.polarity_index()]
            } else {
                self.target_delay
            };
            if self.is_same_node_inverter(state.node_id, &current)
                || self.is_same_node_inverter(state.node_id, &opposite_match)
                || opposite_match.arrival + inverter_delay > required
            {
                continue;
            }

            self.dereference_choice(&current.choice)?;
            self.map_refs[opposite.node_id][opposite.polarity_index()] += 1;
            let replacement = inverter_match(
                state,
                &opposite_match,
                inverter,
                inverter_area,
                inverter_delay,
            );
            self.best[state.node_id][state.polarity_index()] = Some(replacement.clone());
            self.selected[state.node_id][state.polarity_index()] = Some(replacement);
        }
        Ok(())
    }

    /// Checks each explicitly constrained endpoint after final arrival repair.
    fn validate_output_requirements(&self) -> Result<()> {
        for output in &self.outputs {
            let selected = self.best[output.state.node_id][output.state.polarity_index()]
                .as_ref()
                .or_else(|| {
                    self.selected[output.state.node_id][output.state.polarity_index()].as_ref()
                })
                .ok_or_else(|| anyhow!("output '{}' lost its selected NF match", output.name))?;
            validate_output_required(output, selected.arrival)?;
        }
        Ok(())
    }

    /// Materializes only output-live matches into the shared netlist cover.
    fn materialize(&self) -> Result<NfCover> {
        let mut materializer = NfMaterializer {
            mapper: self,
            solutions: Vec::new(),
            solution_by_state: BTreeMap::new(),
            visiting: BTreeSet::new(),
        };
        let mut output_solutions = Vec::with_capacity(self.outputs.len());
        let mut output_arrivals = Vec::with_capacity(self.outputs.len());
        for output in &self.outputs {
            output_solutions.push(materializer.materialize_state(output.state)?);
            let selected = self.best[output.state.node_id][output.state.polarity_index()]
                .as_ref()
                .or_else(|| {
                    self.selected[output.state.node_id][output.state.polarity_index()].as_ref()
                })
                .ok_or_else(|| anyhow!("output '{}' lost its selected NF match", output.name))?;
            output_arrivals.push(selected.arrival);
        }
        Ok(NfCover {
            plan: CoverPlan {
                solutions: materializer.solutions,
                output_solutions,
                output_arrivals,
                matched_candidate_count: self.matched_candidate_count,
            },
            enumerated_cut_count: self.enumerated_cut_count,
            representative_output_load: self
                .pin_delays
                .as_ref()
                .map(RepresentativePinDelayTable::output_load),
        })
    }
}

/// Creates an explicit inverter without dividing its area by node fanout.
fn inverter_match(
    state: NfState,
    opposite: &NfMatch,
    binding: CellBindingId,
    area: f64,
    delay: f64,
) -> NfMatch {
    NfMatch {
        arrival: opposite.arrival + delay,
        flow: opposite.flow + area,
        choice: NfChoice::Cell {
            binding,
            inputs: SmallVec::from_slice(&[state.opposite()]),
        },
    }
}

/// Deterministically reconstructs one shared, selected NF cover.
struct NfMaterializer<'a, 'lib> {
    mapper: &'a NfMapper<'lib>,
    solutions: Vec<Solution>,
    solution_by_state: BTreeMap<NfState, SolutionId>,
    visiting: BTreeSet<NfState>,
}

impl NfMaterializer<'_, '_> {
    /// Emits children before their selected parent and preserves shared cones.
    fn materialize_state(&mut self, state: NfState) -> Result<SolutionId> {
        if let Some(existing) = self.solution_by_state.get(&state) {
            return Ok(*existing);
        }
        if !self.visiting.insert(state) {
            return Err(anyhow!(
                "NF cover has a cyclic polarity choice at AIG node {}",
                state.node_id
            ));
        }

        let selected = self.mapper.best[state.node_id][state.polarity_index()]
            .as_ref()
            .or_else(|| self.mapper.selected[state.node_id][state.polarity_index()].as_ref())
            .ok_or_else(|| {
                anyhow!(
                    "NF cover references unmapped AIG node {} in polarity {}",
                    state.node_id,
                    state.polarity_index()
                )
            })?
            .clone();
        let choice = match selected.choice {
            NfChoice::Source(source) => SolutionChoice::Source(source),
            NfChoice::Cell { binding, inputs } => {
                let mut child_solutions = Vec::with_capacity(inputs.len());
                for child in inputs {
                    child_solutions.push(self.materialize_state(child)?);
                }
                SolutionChoice::Cell {
                    binding: self.mapper.cell_index.binding(binding).clone(),
                    inputs: child_solutions,
                }
            }
        };

        let id = SolutionId(self.solutions.len());
        self.solutions.push(Solution { choice });
        self.solution_by_state.insert(state, id);
        self.visiting.remove(&state);
        Ok(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::{AigBitVector, GateBuilder, GateBuilderOptions};
    use crate::liberty_model::{Cell, LibraryBuilder, Pin, PinDirection};
    use crate::liberty_proto::TimingTableKind;

    /// Builds small, scalar-timed cells for direct NF endpoint tests.
    fn timed_library(specifications: &[(&str, &[&str], &str, f64, f64)]) -> Library {
        let mut builder = LibraryBuilder::new();
        let mut cells = Vec::with_capacity(specifications.len());
        for &(name, inputs, function, area, delay) in specifications {
            let mut pins = Vec::with_capacity(inputs.len() + 1);
            for input in inputs {
                pins.push(Pin {
                    direction: PinDirection::Input as i32,
                    name: builder.intern_string(input).unwrap(),
                    capacitance: Some(0.1),
                    ..Pin::default()
                });
            }

            let mut timing_arcs = Vec::with_capacity(inputs.len());
            for input in inputs {
                let table = builder
                    .add_timing_table_f64(
                        TimingTableKind::CellRise,
                        0,
                        vec![],
                        vec![],
                        vec![],
                        vec![delay],
                        vec![],
                        "",
                    )
                    .unwrap();
                let timing_sense = if function == "!A" {
                    "negative_unate"
                } else {
                    "positive_unate"
                };
                timing_arcs.push(
                    builder
                        .add_timing_arc(input, timing_sense, "combinational", "", vec![table])
                        .unwrap(),
                );
            }
            pins.push(Pin {
                direction: PinDirection::Output as i32,
                name: builder.intern_string("Y").unwrap(),
                function: builder.intern_string(function).unwrap(),
                max_capacitance: Some(10.0),
                timing_arcs,
                ..Pin::default()
            });
            cells.push(Cell {
                name: name.to_string(),
                pins,
                area,
                ..Cell::default()
            });
        }
        builder.cells = cells;
        builder.finish()
    }

    /// Builds one ordinary two-input AND with selectable output polarities.
    fn two_input_and(outputs: &[(&str, bool)]) -> ChoiceAig {
        let mut builder =
            GateBuilder::new("endpoint_and".to_string(), GateBuilderOptions::no_opt());
        let a: AigOperand = builder.add_input("a".to_string(), 1).try_into().unwrap();
        let b: AigOperand = builder.add_input("b".to_string(), 1).try_into().unwrap();
        let root = builder.add_and_binary(a, b);
        for &(name, complemented) in outputs {
            let operand = if complemented { root.negate() } else { root };
            builder.add_output(name.to_string(), operand.into());
        }
        ChoiceAig::without_choices(builder.build())
    }

    /// Builds a two-level AND that also has a shallow three-input cut.
    fn three_input_and() -> ChoiceAig {
        let mut builder = GateBuilder::new(
            "endpoint_three_input_and".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let a: AigOperand = builder.add_input("a".to_string(), 1).try_into().unwrap();
        let b: AigOperand = builder.add_input("b".to_string(), 1).try_into().unwrap();
        let c: AigOperand = builder.add_input("c".to_string(), 1).try_into().unwrap();
        let first = builder.add_and_binary(a, b);
        let root = builder.add_and_binary(first, c);
        builder.add_output("o".to_string(), root.into());
        ChoiceAig::without_choices(builder.build())
    }

    /// Calls the compact NF engine directly, independent of public dispatch.
    fn endpoint_cover(
        choice_aig: &ChoiceAig,
        library: &Library,
        constraints: &TechMapTimingConstraints,
        options: &TechMapOptions,
    ) -> Result<NfCover> {
        let analysis = crate::techmap::cuts::analyze_choices(choice_aig)?;
        let cell_index = LibertyCellIndex::build_nf(library, options.max_cut_size)?;
        build_cover_plan(
            choice_aig,
            &analysis,
            library,
            &cell_index,
            options,
            constraints,
        )
    }

    /// Returns materialized cell names in their deterministic cover order.
    fn selected_cell_names(cover: &NfCover) -> Vec<&str> {
        cover
            .plan
            .solutions
            .iter()
            .filter_map(|solution| match &solution.choice {
                SolutionChoice::Source(_) => None,
                SolutionChoice::Cell { binding, .. } => Some(binding.cell_name.as_str()),
            })
            .collect()
    }

    /// Constructs one structural cut without depending on Liberty timing.
    fn structural_cut(leaves: &[usize], useful: bool, flow: f64, delay: f64) -> Cut {
        Cut {
            leaves: leaves.iter().map(|id| AigRef { id: *id }).collect(),
            truth: 0,
            useful,
            flow,
            delay,
        }
    }

    #[test]
    fn cut_priority_uses_nf_area_flow_epsilon_before_unit_depth() {
        let earlier = structural_cut(&[1, 2], true, 2.0, 3.0);
        let faster = structural_cut(&[3, 4], true, 2.0005, 2.0);

        assert_eq!(nf_cut_order(&faster, &earlier), Ordering::Less);
    }

    #[test]
    fn newly_visited_equal_priority_cut_precedes_existing_cut() {
        let mut frontier = NfCutFrontier::new(4);
        frontier.insert(structural_cut(&[1, 2], true, 1.0, 2.0));
        frontier.insert(structural_cut(&[3, 4], true, 1.0, 2.0));

        let cuts = frontier.finish();

        assert_eq!(
            cuts[0].leaves,
            CutLeaves::from_slice(&[AigRef { id: 3 }, AigRef { id: 4 }])
        );
        assert_eq!(
            cuts[1].leaves,
            CutLeaves::from_slice(&[AigRef { id: 1 }, AigRef { id: 2 }])
        );
    }

    #[test]
    fn smaller_support_removes_existing_supersets_without_cost_comparison() {
        let mut frontier = NfCutFrontier::new(4);
        frontier.insert(structural_cut(&[1, 2, 3], true, 0.1, 1.0));
        frontier.insert(structural_cut(&[1, 2], false, 100.0, 9.0));

        let cuts = frontier.finish();

        assert_eq!(cuts.len(), 1);
        assert_eq!(
            cuts[0].leaves,
            CutLeaves::from_slice(&[AigRef { id: 1 }, AigRef { id: 2 }])
        );
    }

    #[test]
    fn mux_reference_count_discounts_shared_control_and_identical_data() {
        let mut builder = GateBuilder::new("mux_flow".to_string(), GateBuilderOptions::no_opt());
        let a: AigOperand = builder.add_input("a".to_string(), 1).try_into().unwrap();
        let b: AigOperand = builder.add_input("b".to_string(), 1).try_into().unwrap();
        let c: AigOperand = builder.add_input("c".to_string(), 1).try_into().unwrap();
        let d: AigOperand = builder.add_input("d".to_string(), 1).try_into().unwrap();
        let control = builder.add_and_binary(a, b);
        let data = builder.add_and_binary(c, d);
        let first = builder.add_and_binary(control.into(), data.into());
        let second = builder.add_and_binary(control.negate(), data.into());
        let root = builder.add_and_binary(first.negate(), second.negate());
        builder.add_output("o".to_string(), root.into());
        let graph = builder.build();

        let references = initial_flow_references(&graph);

        assert_eq!(references[control.node.id], 1.0);
        assert_eq!(references[data.node.id], 1.0);
        assert_eq!(references[root.node.id], 1.0);
    }

    #[test]
    fn unconstrained_nf_unit_preserves_the_shallow_unit_delay_cover() {
        let graph = three_input_and();
        let library = timed_library(&[
            ("AND2", &["A", "B"], "A * B", 1.0, 1.0),
            ("AND3", &["A", "B", "C"], "A * B * C", 1.5, 100.0),
        ]);
        let options = TechMapOptions {
            timing_model: TechMapTimingModel::NfUnit,
            ..TechMapOptions::default()
        };
        let constraints = TechMapTimingConstraints::default();
        let analysis = crate::techmap::cuts::analyze_choices(&graph).unwrap();
        let cell_index = LibertyCellIndex::build_nf(&library, options.max_cut_size).unwrap();
        let mapper = NfMapper::new(
            &graph,
            &analysis,
            &library,
            &cell_index,
            &options,
            &constraints,
        )
        .unwrap();

        assert!(!mapper.has_endpoint_constraints);
        assert!(mapper.input_arrivals.is_none());
        assert!(mapper.pin_delays.is_none());

        let cover = endpoint_cover(&graph, &library, &constraints, &options).unwrap();
        assert_eq!(selected_cell_names(&cover), ["AND3"]);
        assert_eq!(cover.plan.output_arrivals, [1.0]);
        assert_eq!(cover.representative_output_load, None);
    }

    #[test]
    fn constrained_nf_unit_uses_liberty_pin_delays_and_input_arrivals() {
        let graph = three_input_and();
        let library = timed_library(&[
            ("AND2", &["A", "B"], "A * B", 1.0, 1.0),
            ("AND3", &["A", "B", "C"], "A * B * C", 1.5, 100.0),
        ]);
        let options = TechMapOptions {
            timing_model: TechMapTimingModel::NfUnit,
            ..TechMapOptions::default()
        };
        let mut constraints = TechMapTimingConstraints::default();
        constraints
            .primary_input_arrivals
            .insert("a".to_string(), 3.0);
        constraints
            .primary_output_required
            .insert("o".to_string(), 5.0);

        let cover = endpoint_cover(&graph, &library, &constraints, &options).unwrap();

        assert_eq!(selected_cell_names(&cover), ["AND2", "AND2"]);
        assert_eq!(cover.plan.output_arrivals, [5.0]);
        assert_eq!(cover.representative_output_load, Some(0.2));
    }

    #[test]
    fn constrained_nf_liberty_preserves_representative_timing() {
        let graph = three_input_and();
        let library = timed_library(&[
            ("AND2", &["A", "B"], "A * B", 1.0, 1.0),
            ("AND3", &["A", "B", "C"], "A * B * C", 1.5, 100.0),
        ]);
        let options = TechMapOptions {
            timing_model: TechMapTimingModel::NfLiberty,
            ..TechMapOptions::default()
        };
        let mut constraints = TechMapTimingConstraints::default();
        constraints
            .primary_input_arrivals
            .insert("a".to_string(), 3.0);
        constraints
            .primary_output_required
            .insert("o".to_string(), 5.0);

        let cover = endpoint_cover(&graph, &library, &constraints, &options).unwrap();

        assert_eq!(selected_cell_names(&cover), ["AND2", "AND2"]);
        assert_eq!(cover.plan.output_arrivals, [5.0]);
        assert_eq!(cover.representative_output_load, Some(0.2));
    }

    #[test]
    fn input_arrivals_without_required_times_seed_unconstrained_outputs() {
        let graph = two_input_and(&[("o", false)]);
        let library = timed_library(&[("AND2", &["A", "B"], "A * B", 1.0, 2.0)]);
        let mut constraints = TechMapTimingConstraints::default();
        constraints
            .primary_input_arrivals
            .insert("a".to_string(), 3.0);

        let cover =
            endpoint_cover(&graph, &library, &constraints, &TechMapOptions::default()).unwrap();

        assert_eq!(selected_cell_names(&cover), ["AND2"]);
        assert_eq!(cover.plan.output_arrivals, [5.0]);
    }

    #[test]
    fn constrained_nf_root_reports_the_existing_infeasibility_diagnostic() {
        let graph = two_input_and(&[("o", false)]);
        let library = timed_library(&[
            ("SLOW_AND2", &["A", "B"], "A * B", 1.0, 5.0),
            ("FAST_AND2", &["A", "B"], "A * B", 2.0, 1.0),
        ]);
        let mut constraints = TechMapTimingConstraints::default();
        constraints
            .primary_output_required
            .insert("o".to_string(), 2.0);

        let error = endpoint_cover(&graph, &library, &constraints, &TechMapOptions::default())
            .err()
            .expect("the retained NF root cannot meet the required time");

        assert_eq!(
            error.to_string(),
            "no cover meets required time 2 for output 'o'; fastest estimated arrival is 5"
        );
    }

    #[test]
    fn relaxed_required_time_preserves_the_cheapest_nf_root() {
        let graph = two_input_and(&[("o", false)]);
        let library = timed_library(&[
            ("SLOW_AND2", &["A", "B"], "A * B", 1.0, 5.0),
            ("FAST_AND2", &["A", "B"], "A * B", 2.0, 1.0),
        ]);
        let mut constraints = TechMapTimingConstraints::default();
        constraints
            .primary_output_required
            .insert("o".to_string(), 10.0);

        let cover =
            endpoint_cover(&graph, &library, &constraints, &TechMapOptions::default()).unwrap();

        assert_eq!(selected_cell_names(&cover), ["SLOW_AND2"]);
        assert_eq!(cover.plan.output_arrivals, [5.0]);
    }

    #[test]
    fn input_arrivals_and_required_times_follow_both_output_polarities() {
        let graph = two_input_and(&[("positive", false), ("negative", true)]);
        let library = timed_library(&[
            ("AND2", &["A", "B"], "A * B", 1.0, 2.0),
            ("INV", &["A"], "!A", 0.25, 0.5),
        ]);
        let mut constraints = TechMapTimingConstraints::default();
        constraints
            .primary_input_arrivals
            .insert("a".to_string(), 3.0);
        constraints
            .primary_output_required
            .insert("positive".to_string(), 5.0);
        constraints
            .primary_output_required
            .insert("negative".to_string(), 5.5);

        let cover =
            endpoint_cover(&graph, &library, &constraints, &TechMapOptions::default()).unwrap();

        assert_eq!(selected_cell_names(&cover), ["AND2", "INV"]);
        assert_eq!(cover.plan.output_arrivals, [5.0, 5.5]);
    }

    #[test]
    fn complemented_output_reports_its_actual_inverter_arrival() {
        let graph = two_input_and(&[("negative", true)]);
        let library = timed_library(&[
            ("AND2", &["A", "B"], "A * B", 1.0, 2.0),
            ("INV", &["A"], "!A", 0.25, 0.5),
        ]);
        let mut constraints = TechMapTimingConstraints::default();
        constraints
            .primary_input_arrivals
            .insert("a".to_string(), 3.0);
        constraints
            .primary_output_required
            .insert("negative".to_string(), 5.25);

        let error = endpoint_cover(&graph, &library, &constraints, &TechMapOptions::default())
            .err()
            .expect("the explicit output inverter must meet the required time");

        assert_eq!(
            error.to_string(),
            "no cover meets required time 5.25 for output 'negative'; fastest estimated arrival is 5.5"
        );
    }

    #[test]
    fn shared_outputs_use_the_tightest_constraint_and_leave_others_unbounded() {
        let graph = two_input_and(&[
            ("relaxed", false),
            ("tight", false),
            ("negative", true),
            ("unconstrained", false),
        ]);
        let library = timed_library(&[
            ("AND2", &["A", "B"], "A * B", 1.0, 2.0),
            ("INV", &["A"], "!A", 0.25, 0.5),
        ]);
        let mut constraints = TechMapTimingConstraints::default();
        constraints
            .primary_input_arrivals
            .insert("a".to_string(), 3.0);
        constraints
            .primary_output_required
            .insert("relaxed".to_string(), 10.0);
        constraints
            .primary_output_required
            .insert("tight".to_string(), 5.0);
        constraints
            .primary_output_required
            .insert("negative".to_string(), 5.5);

        let cover =
            endpoint_cover(&graph, &library, &constraints, &TechMapOptions::default()).unwrap();

        assert_eq!(selected_cell_names(&cover), ["AND2", "INV"]);
        assert_eq!(cover.plan.output_arrivals, [5.0, 5.0, 5.5, 5.0]);
    }

    #[test]
    fn required_times_survive_every_flow_and_exact_area_round() {
        let graph = two_input_and(&[
            ("relaxed", false),
            ("tight", false),
            ("negative", true),
            ("unconstrained", false),
        ]);
        let library = timed_library(&[
            ("AND2", &["A", "B"], "A * B", 1.0, 2.0),
            ("INV", &["A"], "!A", 0.25, 0.5),
        ]);
        let options = TechMapOptions::default();
        let mut constraints = TechMapTimingConstraints::default();
        constraints
            .primary_input_arrivals
            .insert("a".to_string(), 3.0);
        constraints
            .primary_output_required
            .insert("relaxed".to_string(), 10.0);
        constraints
            .primary_output_required
            .insert("tight".to_string(), 5.0);
        constraints
            .primary_output_required
            .insert("negative".to_string(), 5.5);

        let analysis = crate::techmap::cuts::analyze_choices(&graph).unwrap();
        let cell_index = LibertyCellIndex::build_nf(&library, options.max_cut_size).unwrap();
        let mut mapper = NfMapper::new(
            &graph,
            &analysis,
            &library,
            &cell_index,
            &options,
            &constraints,
        )
        .unwrap();
        let root = graph.graph().outputs[0].bit_vector.get_lsb(0).node.id;

        assert_eq!(mapper.requireds[root], [5.0, 5.5]);
        for round in 0..NF_AREA_FLOW_ROUNDS {
            mapper.compute_round_matches(round).unwrap();
            mapper.set_mapping_references(round).unwrap();
            assert_eq!(mapper.requireds[root], [5.0, 5.5]);
        }
        for round in 0..NF_EXACT_AREA_ROUNDS {
            mapper.recover_exact_area(round).unwrap();
            assert_eq!(mapper.requireds[root], [5.0, 5.5]);
        }
        mapper.fix_primary_output_drivers().unwrap();
        mapper.reset_exact_matches(NF_EXACT_AREA_ROUNDS).unwrap();
        mapper.validate_output_requirements().unwrap();

        let cover = mapper.materialize().unwrap();
        assert_eq!(cover.plan.output_arrivals, [5.0, 5.0, 5.5, 5.0]);
    }

    #[test]
    fn endpoint_constraints_use_flattened_scalar_bus_names() {
        let mut builder = GateBuilder::new(
            "flattened_endpoint_bus".to_string(),
            GateBuilderOptions::no_opt(),
        );
        let data = builder.add_input("data".to_string(), 2);
        let enable: AigOperand = builder
            .add_input("enable".to_string(), 1)
            .try_into()
            .unwrap();
        let first = builder.add_and_binary(*data.get_lsb(0), enable);
        let second = builder.add_and_binary(*data.get_lsb(1), enable);
        builder.add_output(
            "result".to_string(),
            AigBitVector::from_lsb_is_index_0(&[first, second.negate()]),
        );
        let graph = ChoiceAig::without_choices(builder.build());
        let library = timed_library(&[
            ("AND2", &["A", "B"], "A * B", 1.0, 1.0),
            ("INV", &["A"], "!A", 0.25, 1.0),
        ]);
        let mut constraints = TechMapTimingConstraints::default();
        constraints
            .primary_input_arrivals
            .insert("data_0".to_string(), 2.0);
        constraints
            .primary_input_arrivals
            .insert("data_1".to_string(), 4.0);
        constraints
            .primary_output_required
            .insert("result_0".to_string(), 3.0);
        constraints
            .primary_output_required
            .insert("result_1".to_string(), 6.0);

        let cover =
            endpoint_cover(&graph, &library, &constraints, &TechMapOptions::default()).unwrap();

        assert_eq!(cover.plan.output_arrivals, [3.0, 6.0]);
    }

    #[test]
    fn unknown_primary_input_constraints_are_reported_deterministically() {
        let graph = two_input_and(&[("o", false)]);
        let library = timed_library(&[("AND2", &["A", "B"], "A * B", 1.0, 1.0)]);
        let mut constraints = TechMapTimingConstraints::default();
        constraints
            .primary_input_arrivals
            .insert("z_missing".to_string(), 1.0);
        constraints
            .primary_input_arrivals
            .insert("a_missing".to_string(), 1.0);

        let error = endpoint_cover(&graph, &library, &constraints, &TechMapOptions::default())
            .err()
            .expect("unknown flattened input constraints should fail");

        assert_eq!(
            error.to_string(),
            "timing constraint names unknown primary input 'a_missing'"
        );
    }

    #[test]
    fn unknown_primary_output_constraints_are_reported_deterministically() {
        let graph = two_input_and(&[("o", false)]);
        let library = timed_library(&[("AND2", &["A", "B"], "A * B", 1.0, 1.0)]);
        let mut constraints = TechMapTimingConstraints::default();
        constraints
            .primary_output_required
            .insert("z_missing".to_string(), 1.0);
        constraints
            .primary_output_required
            .insert("a_missing".to_string(), 1.0);

        let error = endpoint_cover(&graph, &library, &constraints, &TechMapOptions::default())
            .err()
            .expect("unknown flattened output constraints should fail");

        assert_eq!(
            error.to_string(),
            "timing constraint names unknown primary output 'a_missing'"
        );
    }

    #[test]
    fn nonfinite_and_negative_primary_input_arrivals_are_rejected() {
        let graph = two_input_and(&[("o", false)]);
        let library = timed_library(&[("AND2", &["A", "B"], "A * B", 1.0, 1.0)]);

        for arrival in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
            let mut constraints = TechMapTimingConstraints::default();
            constraints
                .primary_input_arrivals
                .insert("a".to_string(), arrival);

            let error = endpoint_cover(&graph, &library, &constraints, &TechMapOptions::default())
                .err()
                .expect("invalid input-arrival constraints should fail");

            assert_eq!(
                error.to_string(),
                format!(
                    "timing constraint for primary input 'a' must be non-negative and finite; got {arrival}"
                )
            );
        }
    }

    #[test]
    fn nonfinite_and_negative_primary_output_required_times_are_rejected() {
        let graph = two_input_and(&[("o", false)]);
        let library = timed_library(&[("AND2", &["A", "B"], "A * B", 1.0, 1.0)]);

        for required in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
            let mut constraints = TechMapTimingConstraints::default();
            constraints
                .primary_output_required
                .insert("o".to_string(), required);

            let error = endpoint_cover(&graph, &library, &constraints, &TechMapOptions::default())
                .err()
                .expect("invalid output required-time constraints should fail");

            assert_eq!(
                error.to_string(),
                format!(
                    "timing constraint for primary output 'o' must be non-negative and finite; got {required}"
                )
            );
        }
    }

    #[test]
    fn constrained_mapping_rejects_negated_primary_input_bindings() {
        let original = two_input_and(&[("o", false)]);
        let mut graph = original.graph().clone();
        let input = *graph.inputs[0].bit_vector.get_lsb(0);
        graph.inputs[0].bit_vector.set_lsb(0, input.negate());
        let graph = ChoiceAig::without_choices(graph);
        let library = timed_library(&[("AND2", &["A", "B"], "A * B", 1.0, 1.0)]);
        let mut constraints = TechMapTimingConstraints::default();
        constraints
            .primary_input_arrivals
            .insert("a".to_string(), 0.0);

        let error = endpoint_cover(&graph, &library, &constraints, &TechMapOptions::default())
            .err()
            .expect("negated primary-input port bindings should fail");

        assert_eq!(
            error.to_string(),
            "technology mapping does not support negated input-port bindings"
        );
    }

    #[test]
    fn constrained_mapping_rejects_duplicate_flattened_input_names() {
        let original = two_input_and(&[("o", false)]);
        let mut graph = original.graph().clone();
        graph.inputs.push(graph.inputs[0].clone());
        let graph = ChoiceAig::without_choices(graph);
        let library = timed_library(&[("AND2", &["A", "B"], "A * B", 1.0, 1.0)]);
        let mut constraints = TechMapTimingConstraints::default();
        constraints
            .primary_input_arrivals
            .insert("a".to_string(), 0.0);

        let error = endpoint_cover(&graph, &library, &constraints, &TechMapOptions::default())
            .err()
            .expect("duplicate flattened primary-input names should fail");

        assert_eq!(
            error.to_string(),
            "technology mapping has duplicate input 'a'"
        );
    }
}
