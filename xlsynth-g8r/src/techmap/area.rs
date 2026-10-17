// SPDX-License-Identifier: Apache-2.0

//! Area-only cell covering of a fixed AIG.
//!
//! Cuts stop at every live fanout > 1, including output uses. A shared node has
//! one locally selected base polarity; all opposite uses share one inverter.
//! No choices, graph rewriting, or logic replication are performed. Constants
//! and wiring aliases are ideal wires, with no physical tie-cell area.
//!
//! The two-polarity recurrence optimizes cell area inside each fanout-free
//! region. Boundary implementations are excluded from its cost, and boundary
//! inverter demands are unioned before costing a candidate. Keeping just one
//! state per polarity, fixing boundary phases locally, and capping cuts are
//! heuristics: this is not a globally optimal DAG-area algorithm. Reported area
//! is always the sum of unique emitted cells, never the recurrence's score.
//! Without an inverter, selected boundary uses must agree on a polarity; the
//! local selection heuristic can reject a graph when another cover would work.

use crate::aig::gate::{AigNode, AigOperand, AigRef, GateFn};
use crate::aig::topo::postorder_for_aig_refs_node_only;
use crate::liberty::cell_formula::parse_formula;
use crate::liberty_model::{Library, PinDirection};
use crate::netlist::mapped::MappedNetlist;
use crate::netlist::parse::{
    AssignExpr, Net, NetIndex, NetRef, NetlistAssign, NetlistAssignKind, NetlistInstance,
    NetlistModule, NetlistPort, PortDirection, Pos, Span,
};
use anyhow::{Result, anyhow, ensure};
use imbl::OrdSet;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use string_interner::StringInterner;
use xlsynth_pir::IrBits;

const MAX_INPUTS: usize = 5;
const MAX_CUTS: usize = 64;

/// Eligible single-output cell, with pins in stable lexical order.
#[derive(Clone, Debug)]
struct Cell {
    name: String,
    pins: Vec<String>,
    output: String,
    area: f64,
}

/// A truth-table match: physical pin i receives leaf perm[i], optionally
/// inverted.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Binding {
    cell: usize,
    perm: Vec<usize>,
    phases: usize,
}

/// Reusable truth-table index for combinational single-output cells of <=5
/// inputs. Cells with dont_use, use_for_size_only, unsupported semantics,
/// clocks, or multiple outputs are excluded, as are cells with unparseable
/// functions. Invalid areas or unknown function inputs are errors.
#[derive(Debug)]
pub struct AreaMappingLibrary {
    cells: Vec<Cell>,
    matches: BTreeMap<(usize, u64), Vec<Binding>>,
    inv: Option<usize>,
    /// A potentially usable cell excluded because its sizing policy is unknown.
    missing_policy_coverage: Option<String>,
    pub excluded_cells: Vec<String>,
}

/// Returns the active truth-table bits for at most five inputs.
fn mask(n: usize) -> u64 {
    (1u64 << (1usize << n)) - 1
}
/// Builds the truth table for one input in assignment-bit order.
fn variable(n: usize, pin: usize) -> u64 {
    (0..1usize << n)
        .filter(|a| a & (1 << pin) != 0)
        .fold(0, |tt, a| tt | (1 << a))
}

/// Enumerates every input-pin permutation, including nonsymmetric cell
/// functions.
fn permutations(n: usize) -> Vec<Vec<usize>> {
    /// Extends a permutation prefix with each unused input index.
    fn visit(prefix: &mut Vec<usize>, n: usize, out: &mut Vec<Vec<usize>>) {
        if prefix.len() == n {
            out.push(prefix.clone());
            return;
        }
        for i in 0..n {
            if !prefix.contains(&i) {
                prefix.push(i);
                visit(prefix, n, out);
                prefix.pop();
            }
        }
    }
    let mut out = Vec::new();
    visit(&mut Vec::new(), n, &mut out);
    out
}

impl AreaMappingLibrary {
    /// Compiles exact cell truth tables and all input permutations/polarities.
    pub fn new(lib: &Library) -> Result<Self> {
        let mut cells: Vec<Cell> = Vec::new();
        let mut matches: BTreeMap<(usize, u64), Vec<Binding>> = BTreeMap::new();
        let mut excluded_cells = Vec::new();
        let mut missing_policy_coverage = None;
        let mut names = BTreeSet::new();
        let mut sorted = lib.cells.iter().collect::<Vec<_>>();
        sorted.sort_by(|a, b| a.name.cmp(&b.name));
        let mut inv = None;
        for cell in sorted {
            ensure!(!cell.name.is_empty(), "cell has an empty name");
            ensure!(
                names.insert(cell.name.clone()),
                "duplicate cell {}",
                cell.name
            );
            let mut pins = cell
                .pins
                .iter()
                .filter(|p| p.direction == PinDirection::Input as i32)
                .map(|p| lib.resolve_string(&p.name).to_string())
                .collect::<Vec<_>>();
            pins.sort();
            let outputs = cell
                .pins
                .iter()
                .filter(|p| p.direction == PinDirection::Output as i32)
                .collect::<Vec<_>>();
            if cell.dont_use == Some(true)
                || cell.use_for_size_only == Some(true)
                || !cell.sequential.is_empty()
                || cell.clock_gate.is_some()
                || cell.validate_functional_semantics().is_err()
                || outputs.len() != 1
                || pins.is_empty()
                || pins.len() > MAX_INPUTS
                || cell.pins.iter().any(|p| {
                    p.is_clocking_pin
                        || ![PinDirection::Input as i32, PinDirection::Output as i32]
                            .contains(&p.direction)
                })
            {
                excluded_cells.push(cell.name.clone());
                continue;
            }
            if let Err(error) = cell.validate_mapping_policy_coverage() {
                // Remember only otherwise plausible cells; excluded legacy
                // sequential or malformed cells do not explain a missing cover.
                let mut names = BTreeSet::new();
                let valid_pins = cell.pins.iter().all(|pin| {
                    let name = lib.resolve_string(&pin.name);
                    !name.is_empty() && names.insert(name)
                });
                let valid_function = parse_formula(lib.resolve_string(&outputs[0].function))
                    .is_ok_and(|term| term.inputs().iter().all(|p| pins.contains(p)));
                if valid_pins && valid_function && cell.area.is_finite() && cell.area > 0.0 {
                    missing_policy_coverage.get_or_insert(error);
                }
                excluded_cells.push(cell.name.clone());
                continue;
            }
            let mut pin_names = BTreeSet::new();
            for pin in &cell.pins {
                let name = lib.resolve_string(&pin.name);
                ensure!(!name.is_empty(), "cell {} has an empty pin name", cell.name);
                ensure!(
                    pin_names.insert(name),
                    "cell {} has duplicate pin name {name}",
                    cell.name
                );
            }
            let Ok(term) = parse_formula(lib.resolve_string(&outputs[0].function)) else {
                // An unsupported function cannot be matched safely, but it
                // need not prevent mapping with the rest of the library.
                excluded_cells.push(cell.name.clone());
                continue;
            };
            ensure!(
                cell.area.is_finite() && cell.area > 0.0,
                "cell {} has invalid area {}",
                cell.name,
                cell.area
            );
            ensure!(
                term.inputs().iter().all(|p| pins.contains(p)),
                "unknown function input in {}",
                cell.name
            );
            let n = pins.len();
            let mut tt = 0u64;
            for assignment in 0..1usize << n {
                let values = pins
                    .iter()
                    .enumerate()
                    .map(|(i, p)| (p.clone(), assignment & (1 << i) != 0))
                    .collect();
                if term
                    .evaluate_partial(&values)
                    .ok_or_else(|| anyhow!("incomplete function {}", cell.name))?
                {
                    tt |= 1 << assignment;
                }
            }
            let index = cells.len();
            if n == 1 && tt == 1 && inv.is_none_or(|i: usize| cell.area < cells[i].area) {
                inv = Some(index);
            }
            cells.push(Cell {
                name: cell.name.clone(),
                pins,
                output: lib.resolve_string(&outputs[0].name).to_string(),
                area: cell.area,
            });
            for perm in permutations(n) {
                for phases in 0..1usize << n {
                    let mut transformed = 0u64;
                    for assignment in 0..1usize << n {
                        let mut physical = 0;
                        for (i, leaf) in perm.iter().enumerate() {
                            physical |= (((assignment >> leaf) ^ (phases >> leaf)) & 1) << i;
                        }
                        transformed |= ((tt >> physical) & 1) << assignment;
                    }
                    matches.entry((n, transformed)).or_default().push(Binding {
                        cell: index,
                        perm: perm.clone(),
                        phases,
                    });
                }
            }
        }
        // Equivalent physical functions with the same leaf polarities have the
        // same recursive cost. Retain the smallest cell and a stable
        // permutation.
        for bindings in matches.values_mut() {
            bindings.sort_by(|a, b| {
                cells[a.cell]
                    .area
                    .total_cmp(&cells[b.cell].area)
                    .then(a.cmp(b))
            });
            let mut seen = BTreeSet::new();
            bindings.retain(|b| seen.insert(b.phases));
        }
        Ok(Self {
            cells,
            matches,
            inv,
            missing_policy_coverage,
            excluded_cells,
        })
    }

    /// Adds a recovery hint when unknown sizing policy excluded library cells.
    fn explain_missing_cover(&self, error: anyhow::Error) -> anyhow::Error {
        match (&self.missing_policy_coverage, self.cells.is_empty()) {
            (Some(reason), true) => error.context(reason.clone()),
            (Some(reason), false) => error.context(format!(
                "some library cells were excluded from area mapping: {reason}"
            )),
            (None, _) => error,
        }
    }
}

/// A bounded structural cut; table bit i uses leaf j = bit j of i.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Cut {
    leaves: Vec<usize>,
    tt: u64,
}

/// Creates the identity cut used to stop covering at a boundary.
fn trivial(id: usize) -> Cut {
    Cut {
        leaves: vec![id],
        tt: 2,
    }
}
/// Expands a cut truth table into a sorted superset of its leaves.
fn embed(cut: &Cut, leaves: &[usize]) -> u64 {
    let positions = cut
        .leaves
        .iter()
        .map(|id| leaves.binary_search(id).unwrap())
        .collect::<Vec<_>>();
    (0..1usize << leaves.len()).fold(0, |tt, a| {
        let old = positions
            .iter()
            .enumerate()
            .fold(0, |x, (i, p)| x | (((a >> p) & 1) << i));
        tt | (((cut.tt >> old) & 1) << a)
    })
}

/// Selected local implementation; boundary demands are charged only once per
/// region.
#[derive(Clone, Debug)]
struct Plan {
    area: f64,
    demands: OrdSet<AigOperand>,
    kind: PlanKind,
}
/// Cell covers consume their cut interior; aliases/constants only consume their
/// root.
#[derive(Clone, Debug)]
enum PlanKind {
    Literal(bool),
    Leaf(AigOperand),
    Cell { binding: Binding, cut: Cut },
    Invert(Box<Plan>),
}
impl Plan {
    /// Adds one inverter cost per distinct demand opposite its boundary phase.
    fn score(&self, inv_area: f64) -> f64 {
        self.area + self.demands.len() as f64 * inv_area
    }
}

/// Actual emitted instance accounting and structural nonduplication audit.
#[derive(Debug, Default, Serialize)]
pub struct AreaMappingStats {
    pub cell_area: f64,
    pub cell_count: usize,
    pub inverter_count: usize,
    pub cells_by_type: BTreeMap<String, usize>,
    pub live_ands: usize,
    pub covered_ands: usize,
    pub duplicated_ands: usize,
    pub shared_and_boundaries: usize,
    pub shared_opposite_polarities: usize,
}
/// Mapped netlist and measured unique cell costs (ideal constants and aliases
/// are free).
#[derive(Debug)]
pub struct AreaMappedNetlist {
    pub netlist: MappedNetlist,
    pub stats: AreaMappingStats,
}

/// Local DP state and fixed boundary phases, computed in dependency order.
struct Cover<'a> {
    aig: &'a GateFn,
    lib: &'a AreaMappingLibrary,
    boundary: Vec<bool>,
    anchors: Vec<bool>,
    plans: Vec<[Option<Plan>; 2]>,
}
impl Cover<'_> {
    /// Returns the positive value of a node whose selected plan is constant.
    fn constant_node(&self, id: usize) -> Option<bool> {
        match &self.plans[id][0].as_ref()?.kind {
            PlanKind::Literal(value) => Some(*value),
            _ => None,
        }
    }

    /// Stops at a shared boundary, substituting an already-known constant.
    fn boundary_cut(&self, id: usize) -> Cut {
        match self.constant_node(id) {
            Some(value) => Cut {
                leaves: vec![],
                tt: u64::from(value),
            },
            None => trivial(id),
        }
    }

    /// Returns a boundary demand or the selected plan for an interior operand.
    fn leaf_plan(&self, operand: AigOperand) -> Option<Plan> {
        if self.boundary[operand.node.id] {
            match self.aig.get(operand.node) {
                AigNode::Input { .. } if operand.negated && self.lib.inv.is_none() => {
                    return None;
                }
                AigNode::And2 { .. } => {
                    let phase = if self.lib.inv.is_some() {
                        self.anchors[operand.node.id]
                    } else {
                        operand.negated
                    };
                    let plan = self.plans[operand.node.id][usize::from(phase)].as_ref()?;
                    if let PlanKind::Literal(value) = &plan.kind {
                        return Some(Plan {
                            area: 0.0,
                            demands: OrdSet::new(),
                            kind: PlanKind::Literal(*value ^ (phase != operand.negated)),
                        });
                    }
                }
                _ => {
                    // Other boundaries already have their requested signal.
                }
            }
            // Only opposite boundary demands incur inverter cost. Without an
            // inverter, input demands cannot conflict because only the positive
            // input is available; shared AND demands still need both phases.
            let track_demand = if self.lib.inv.is_some() {
                operand.negated != self.anchors[operand.node.id]
            } else {
                matches!(self.aig.get(operand.node), AigNode::And2 { .. })
            };
            let mut demands = OrdSet::new();
            if track_demand {
                demands.insert(operand);
            }
            Some(Plan {
                area: 0.0,
                demands,
                kind: PlanKind::Leaf(operand),
            })
        } else {
            self.plans[operand.node.id][usize::from(operand.negated)].clone()
        }
    }
    /// Keeps the first minimum-cost plan to make area ties deterministic.
    fn consider(&self, best: &mut Option<Plan>, candidate: Plan) {
        let inv = self.lib.inv.map_or(0.0, |i| self.lib.cells[i].area);
        if best
            .as_ref()
            .is_none_or(|b| candidate.score(inv) < b.score(inv))
        {
            *best = Some(candidate);
        }
    }
    /// Combines selected child states, unioning shared polarity costs.
    fn cell_plan(&self, cut: &Cut, binding: &Binding) -> Option<Plan> {
        let mut plan = Plan {
            area: self.lib.cells[binding.cell].area,
            demands: OrdSet::new(),
            kind: PlanKind::Cell {
                binding: binding.clone(),
                cut: cut.clone(),
            },
        };
        for (i, id) in cut.leaves.iter().enumerate() {
            let mut child = self.leaf_plan(AigOperand {
                node: AigRef { id: *id },
                negated: binding.phases & (1 << i) != 0,
            })?;
            plan.area += child.area;
            // Insert the smaller set into a structurally shared clone of the
            // larger one, avoiding copies of growing demand prefixes.
            if child.demands.len() > plan.demands.len() {
                std::mem::swap(&mut plan.demands, &mut child.demands);
            }
            for demand in &child.demands {
                if self.lib.inv.is_none() && plan.demands.contains(&demand.negate()) {
                    return None;
                }
                plan.demands.insert(*demand);
            }
        }
        Some(plan)
    }
}

/// Maps a fixed combinational AIG without crossing shared-node boundaries.
/// Only demanded nodes need a cover; folded or absorbed nodes may lack one.
pub fn map_gatefn_to_area_netlist(
    aig: &GateFn,
    lib: &AreaMappingLibrary,
) -> Result<AreaMappedNetlist> {
    let roots = aig
        .outputs
        .iter()
        .flat_map(|o| o.bit_vector.iter_lsb_to_msb())
        .copied()
        .collect::<Vec<_>>()
        .iter()
        .map(|op| op.node)
        .collect::<Vec<_>>();
    let order = postorder_for_aig_refs_node_only(&roots, &aig.gates, &HashMap::<AigRef, ()>::new());
    let mut fanout = vec![0; aig.gates.len()];
    for id in &order {
        for op in aig.get(*id).operands() {
            fanout[op.node.id] += 1;
        }
    }
    for op in aig
        .outputs
        .iter()
        .flat_map(|o| o.bit_vector.iter_lsb_to_msb())
        .copied()
        .collect::<Vec<_>>()
    {
        fanout[op.node.id] += 1;
    }
    let mut cover = Cover {
        aig,
        lib,
        boundary: aig
            .gates
            .iter()
            .enumerate()
            .map(|(i, n)| {
                matches!(n, AigNode::And2 { .. }) && fanout[i] > 1
                    || matches!(n, AigNode::Input { .. })
            })
            .collect(),
        anchors: vec![false; aig.gates.len()],
        plans: vec![[None, None]; aig.gates.len()],
    };
    let mut cuts: Vec<Vec<Cut>> = vec![Vec::new(); aig.gates.len()];
    let mut stats = AreaMappingStats::default();
    for node in order {
        let id = node.id;
        match aig.get(node) {
            AigNode::Literal { value, .. } => {
                cuts[id] = vec![Cut {
                    leaves: vec![],
                    tt: u64::from(*value),
                }];
                for p in 0..2 {
                    cover.plans[id][p] = Some(Plan {
                        area: 0.0,
                        demands: OrdSet::new(),
                        kind: PlanKind::Literal(*value ^ (p != 0)),
                    });
                }
            }
            AigNode::Input { .. } => {
                cuts[id] = vec![trivial(id)];
            }
            AigNode::And2 { a, b, .. } => {
                stats.live_ands += 1;
                if cover.boundary[id] {
                    stats.shared_and_boundaries += 1;
                }
                let ca = if cover.boundary[a.node.id] {
                    vec![cover.boundary_cut(a.node.id)]
                } else {
                    cuts[a.node.id].clone()
                };
                let cb = if cover.boundary[b.node.id] {
                    vec![cover.boundary_cut(b.node.id)]
                } else {
                    cuts[b.node.id].clone()
                };
                let mut candidates = BTreeSet::new();
                for l in &ca {
                    for r in &cb {
                        let leaves = l
                            .leaves
                            .iter()
                            .chain(&r.leaves)
                            .copied()
                            .collect::<BTreeSet<_>>()
                            .into_iter()
                            .collect::<Vec<_>>();
                        if leaves.len() > MAX_INPUTS {
                            continue;
                        }
                        let m = mask(leaves.len());
                        let x = embed(l, &leaves) ^ if a.negated { m } else { 0 };
                        let y = embed(r, &leaves) ^ if b.negated { m } else { 0 };
                        candidates.insert(Cut { leaves, tt: x & y });
                    }
                }
                let mut candidates = candidates.into_iter().collect::<Vec<_>>();
                candidates.sort_by(|a, b| a.leaves.len().cmp(&b.leaves.len()).then(a.cmp(b)));
                candidates.truncate(MAX_CUTS);
                let mut direct: [Option<Plan>; 2] = [None, None];
                for cut in &candidates {
                    for p in 0..2 {
                        let tt = cut.tt ^ if p != 0 { mask(cut.leaves.len()) } else { 0 };
                        if tt == 0 || tt == mask(cut.leaves.len()) {
                            cover.consider(
                                &mut direct[p],
                                Plan {
                                    area: 0.0,
                                    demands: OrdSet::new(),
                                    kind: PlanKind::Literal(tt != 0),
                                },
                            );
                        }
                        for (i, id) in cut.leaves.iter().enumerate() {
                            for negated in [false, true] {
                                if tt
                                    == variable(cut.leaves.len(), i)
                                        ^ if negated { mask(cut.leaves.len()) } else { 0 }
                                {
                                    let op = AigOperand {
                                        node: AigRef { id: *id },
                                        negated,
                                    };
                                    if let Some(mut plan) = cover.leaf_plan(op) {
                                        plan.kind = PlanKind::Leaf(op);
                                        cover.consider(&mut direct[p], plan);
                                    }
                                }
                            }
                        }
                        if let Some(bindings) = lib.matches.get(&(cut.leaves.len(), tt)) {
                            for binding in bindings {
                                if let Some(plan) = cover.cell_plan(cut, binding) {
                                    cover.consider(&mut direct[p], plan);
                                }
                            }
                        }
                    }
                }
                for p in 0..2 {
                    let mut best = direct[p].clone();
                    if let (Some(inv), Some(other)) = (lib.inv, &direct[1 - p]) {
                        let mut inverted = other.clone();
                        inverted.area += lib.cells[inv].area;
                        inverted.kind = PlanKind::Invert(Box::new(other.clone()));
                        cover.consider(&mut best, inverted);
                    }
                    cover.plans[id][p] = best;
                }
                let inv_area = lib.inv.map_or(0.0, |i| lib.cells[i].area);
                cover.anchors[id] = match (&cover.plans[id][0], &cover.plans[id][1]) {
                    (Some(positive), Some(negative)) => {
                        negative.score(inv_area) < positive.score(inv_area)
                    }
                    (None, Some(_)) => true,
                    _ => false,
                };
                candidates.push(trivial(id));
                cuts[id] = candidates;
            }
        }
    }
    Emitter::new(cover, stats)?.emit()
}

/// Emits each demanded signal once and audits ownership of covered AIG ANDs.
struct Emitter<'a> {
    cover: Cover<'a>,
    result: AreaMappedNetlist,
    names: BTreeSet<String>,
    signals: BTreeMap<AigOperand, NetRef>,
    ownership: Vec<usize>,
    outputs: Vec<(NetIndex, AigOperand)>,
}

/// Pending work for depth-first signal emission without call-stack recursion.
enum EmitStep {
    Signal(AigOperand),
    Remember(AigOperand),
    Plan { root: usize, plan: Plan },
    BoundaryInversion { root: usize, cell: usize },
    PlanInversion { cell: usize },
    Cell { cell: usize, inputs: usize },
}

impl<'a> Emitter<'a> {
    /// Allocates the output module and scalar ports before emitting demanded
    /// cells.
    fn new(cover: Cover<'a>, stats: AreaMappingStats) -> Result<Self> {
        let mut interner = StringInterner::new();
        let name = interner.get_or_intern(&cover.aig.name);
        let module = NetlistModule {
            name,
            net_index_range: 0..0,
            ports: vec![],
            wires: vec![],
            assigns: vec![],
            instances: vec![],
        };
        let count = cover.aig.gates.len();
        let mut this = Self {
            cover,
            result: AreaMappedNetlist {
                netlist: MappedNetlist {
                    module,
                    nets: vec![],
                    interner,
                },
                stats,
            },
            names: BTreeSet::new(),
            signals: BTreeMap::new(),
            ownership: vec![0; count],
            outputs: vec![],
        };
        for input in &this.cover.aig.inputs {
            for (i, op) in input.bit_vector.iter_lsb_to_msb().enumerate() {
                ensure!(
                    !op.negated && matches!(this.cover.aig.get(op.node), AigNode::Input { .. }),
                    "invalid input bit"
                );
                let name = bit_name(&input.name, i, input.get_bit_count());
                let net = this.port(&name, PortDirection::Input)?;
                ensure!(
                    this.signals.insert(*op, NetRef::Simple(net)).is_none(),
                    "duplicate input node"
                );
            }
        }
        for output in &this.cover.aig.outputs {
            for (i, op) in output.bit_vector.iter_lsb_to_msb().enumerate() {
                let name = bit_name(&output.name, i, output.get_bit_count());
                let net = this.port(&name, PortDirection::Output)?;
                this.outputs.push((net, *op));
            }
        }
        Ok(this)
    }
    /// Allocates a scalar port and rejects names already used by another port.
    fn port(&mut self, name: &str, direction: PortDirection) -> Result<NetIndex> {
        ensure!(
            self.names.insert(name.to_string()),
            "duplicate scalar port {name}"
        );
        let net = self.net(name);
        self.result.netlist.module.ports.push(NetlistPort {
            direction,
            width: None,
            name: self.result.netlist.nets[net.0].name,
        });
        Ok(net)
    }
    /// Appends a scalar net with the requested interned name.
    fn net(&mut self, name: &str) -> NetIndex {
        let index = NetIndex(self.result.netlist.nets.len());
        let name = self.result.netlist.interner.get_or_intern(name);
        self.result.netlist.nets.push(Net { name, width: None });
        index
    }
    /// Emits one uniquely named cell and charges its area exactly once.
    fn cell(&mut self, index: usize, inputs: Vec<NetRef>) -> NetRef {
        let mut name = format!("tm_{}", self.result.netlist.module.instances.len());
        while self.names.contains(&name) || self.names.contains(&format!("u_{name}")) {
            name.push('_');
        }
        self.names.insert(name.clone());
        self.names.insert(format!("u_{name}"));
        let output = self.net(&name);
        self.result.netlist.module.wires.push(output);
        let mapped = &mut self.result.netlist;
        let cell = &self.cover.lib.cells[index];
        let mut connections = cell
            .pins
            .iter()
            .zip(inputs)
            .map(|(p, n)| (mapped.interner.get_or_intern(p), n))
            .collect::<Vec<_>>();
        connections.push((
            mapped.interner.get_or_intern(&cell.output),
            NetRef::Simple(output),
        ));
        mapped.module.instances.push(NetlistInstance {
            type_name: mapped.interner.get_or_intern(&cell.name),
            instance_name: mapped.interner.get_or_intern(format!("u_{name}")),
            connections,
            inst_lineno: 0,
            inst_colno: 0,
        });
        self.result.stats.cell_area += cell.area;
        self.result.stats.cell_count += 1;
        self.result.stats.inverter_count += usize::from(Some(index) == self.cover.lib.inv);
        *self
            .result
            .stats
            .cells_by_type
            .entry(cell.name.clone())
            .or_default() += 1;
        NetRef::Simple(output)
    }
    /// Memoizes each demanded polarity, sharing boundary implementations and
    /// inverters.
    fn signal(&mut self, op: AigOperand) -> Result<NetRef> {
        let mut work = vec![EmitStep::Signal(op)];
        let mut values = Vec::new();
        while let Some(step) = work.pop() {
            match step {
                EmitStep::Signal(op) => {
                    if let Some(signal) = self.signals.get(&op) {
                        values.push(signal.clone());
                        continue;
                    }
                    let id = op.node.id;
                    work.push(EmitStep::Remember(op));
                    if let AigNode::Literal { value, .. } = self.cover.aig.get(op.node) {
                        values.push(literal(*value ^ op.negated));
                    } else if let Some(inv) =
                        self.cover.lib.inv.filter(|_| {
                            self.cover.boundary[id] && op.negated != self.cover.anchors[id]
                        })
                    {
                        work.push(EmitStep::BoundaryInversion {
                            root: id,
                            cell: inv,
                        });
                        work.push(EmitStep::Signal(op.negate()));
                    } else if let Some(NetRef::Literal(bits)) = self.signals.get(&op.negate()) {
                        values.push(literal(!bits.get_bit(0).expect("one-bit constant")));
                    } else {
                        if self.cover.boundary[id]
                            && self.cover.lib.inv.is_none()
                            && self.signals.contains_key(&op.negate())
                        {
                            return Err(self.cover.lib.explain_missing_cover(anyhow!(
                                "node {id} requires both polarities but the library has no inverter"
                            )));
                        }
                        let plan = self.cover.plans[id][usize::from(op.negated)]
                            .clone()
                            .ok_or_else(|| {
                                self.cover.lib.explain_missing_cover(anyhow!(
                                    "no cover for demanded node {id} polarity {}; library may lack a required cell",
                                    usize::from(op.negated)
                                ))
                            })?;
                        work.push(EmitStep::Plan { root: id, plan });
                    }
                }
                EmitStep::Remember(op) => {
                    let value = values.last().expect("completed signal has a value").clone();
                    self.signals.insert(op, value);
                }
                EmitStep::Plan { root, plan } => match plan.kind {
                    PlanKind::Literal(value) => {
                        self.ownership[root] += 1;
                        values.push(literal(value));
                    }
                    PlanKind::Leaf(op) => {
                        self.ownership[root] += 1;
                        work.push(EmitStep::Signal(op));
                    }
                    PlanKind::Invert(inner) => {
                        let inv = self
                            .cover
                            .lib
                            .inv
                            .ok_or_else(|| anyhow!("selected plan requires an inverter"))?;
                        work.push(EmitStep::PlanInversion { cell: inv });
                        work.push(EmitStep::Plan { root, plan: *inner });
                    }
                    PlanKind::Cell { binding, cut } => {
                        self.own_cut(root, &cut.leaves)?;
                        work.push(EmitStep::Cell {
                            cell: binding.cell,
                            inputs: binding.perm.len(),
                        });
                        for &i in binding.perm.iter().rev() {
                            work.push(EmitStep::Signal(AigOperand {
                                node: AigRef { id: cut.leaves[i] },
                                negated: binding.phases & (1 << i) != 0,
                            }));
                        }
                    }
                },
                EmitStep::BoundaryInversion { root, cell } => {
                    let base = values.pop().expect("inverter has an input");
                    let result = if let NetRef::Literal(bits) = &base {
                        literal(!bits.get_bit(0).expect("one-bit constant"))
                    } else {
                        if matches!(self.cover.aig.gates[root], AigNode::And2 { .. }) {
                            self.result.stats.shared_opposite_polarities += 1;
                        }
                        self.cell(cell, vec![base])
                    };
                    values.push(result);
                }
                EmitStep::PlanInversion { cell } => {
                    let base = values.pop().expect("inverter has an input");
                    values.push(self.cell(cell, vec![base]));
                }
                EmitStep::Cell { cell, inputs } => {
                    let args = values.split_off(values.len() - inputs);
                    values.push(self.cell(cell, args));
                }
            }
        }
        ensure!(values.len() == 1, "invalid emission value stack");
        Ok(values.pop().expect("one emission value"))
    }
    /// Claims a cover interior without crossing a shared-node boundary.
    fn own_cut(&mut self, root: usize, leaves: &[usize]) -> Result<()> {
        let mut todo = vec![root];
        let mut seen = BTreeSet::new();
        while let Some(id) = todo.pop() {
            if leaves.contains(&id) || !seen.insert(id) {
                continue;
            }
            if let AigNode::And2 { a, b, .. } = self.cover.aig.gates[id] {
                if id != root && self.cover.boundary[id] {
                    ensure!(
                        self.cover.constant_node(id).is_some(),
                        "cover crossed fanout boundary"
                    );
                    // A substituted shared constant is implemented separately
                    // when demanded, and is not owned by this cell cover.
                    continue;
                }
                self.ownership[id] += 1;
                todo.extend([a.node.id, b.node.id]);
            }
        }
        Ok(())
    }
    /// Connects outputs and rejects duplicate ownership of any covered AIG AND.
    fn emit(mut self) -> Result<AreaMappedNetlist> {
        for (net, op) in self.outputs.clone() {
            let rhs = self.signal(op)?;
            self.result.netlist.module.assigns.push(NetlistAssign {
                kind: NetlistAssignKind::Continuous,
                lhs: NetRef::Simple(net),
                rhs: AssignExpr::Leaf(rhs),
                span: Span {
                    start: Pos {
                        lineno: 0,
                        colno: 0,
                    },
                    limit: Pos {
                        lineno: 0,
                        colno: 0,
                    },
                },
            });
        }
        self.result.stats.covered_ands = self.ownership.iter().filter(|&&n| n != 0).count();
        self.result.stats.duplicated_ands =
            self.ownership.iter().map(|n| n.saturating_sub(1)).sum();
        ensure!(
            self.result.stats.duplicated_ands == 0,
            "cover duplicated AND nodes"
        );
        self.result.netlist.module.net_index_range = 0..self.result.netlist.nets.len();
        Ok(self.result)
    }
}
/// Names a scalar port, adding a bit suffix only for multi-bit vectors.
fn bit_name(name: &str, bit: usize, count: usize) -> String {
    if count == 1 {
        name.to_string()
    } else {
        format!("{name}_{bit}")
    }
}
/// Creates an ideal one-bit literal without a physical tie cell.
fn literal(value: bool) -> NetRef {
    NetRef::Literal(IrBits::make_ubits(1, u64::from(value)).expect("one bit"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig_sim::gate_sim::{Collect, eval};
    #[cfg(feature = "has-bitwuzla")]
    use crate::check_equivalence::{IrCheckResult, prove_same_gate_fn_via_ir_status_with_solver};
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use crate::liberty::descriptor::LIBERTY_DESCRIPTOR;
    use crate::liberty::model::{library_from_proto, library_to_proto};
    use crate::liberty::parser::parse_liberty_files;
    use crate::liberty_model::{Cell as LibertyCell, LibraryBuilder, Pin};
    use crate::liberty_proto::Library as WireLibrary;
    use crate::netlist::emit::emit_module_as_netlist_text;
    use crate::netlist::gatefn_from_netlist::project_gatefn_from_netlist_and_liberty;
    use crate::netlist::io::read_gv_from_str;
    use prost::Message;
    use prost_reflect::{DescriptorPool, DynamicMessage};
    use rand::{Rng, SeedableRng};
    use std::collections::HashSet;
    use std::io::Write;
    use tempfile::NamedTempFile;
    #[cfg(feature = "has-bitwuzla")]
    use xlsynth_prover::prover::SolverChoice;

    /// Builds a functional test library from named cells, pins, functions, and
    /// areas.
    fn library(specs: &[(&str, &[&str], &str, f64)]) -> Library {
        let mut b = LibraryBuilder::new();
        for (name, inputs, function, area) in specs {
            let mut pins = inputs
                .iter()
                .map(|p| Pin {
                    name: b.intern_string(p).unwrap(),
                    direction: PinDirection::Input as i32,
                    ..Default::default()
                })
                .collect::<Vec<_>>();
            pins.push(Pin {
                name: b.intern_string("Y").unwrap(),
                direction: PinDirection::Output as i32,
                function: b.intern_string(function).unwrap(),
                ..Default::default()
            });
            b.cells.push(LibertyCell {
                name: name.to_string(),
                pins,
                area: *area,
                ..Default::default()
            });
        }
        b.finish()
    }
    /// Provides the small cell basis used by the area-covering tests.
    fn basic_library() -> Library {
        library(&[
            ("INV", &["A"], "!A", 1.0),
            ("NAND2", &["A", "B"], "!(A*B)", 2.0),
            ("AO21", &["A", "B", "C"], "(A*B)+C", 2.5),
            ("NAND5", &["A", "B", "C", "D", "E"], "!(A*B*C*D*E)", 3.0),
        ])
    }
    /// Creates an unoptimized five-input graph for sharing and polarity tests.
    fn builder() -> (GateBuilder, Vec<AigOperand>) {
        let mut b = GateBuilder::new("test".into(), GateBuilderOptions::no_opt());
        let inputs = (0..5)
            .map(|i| *b.add_input(format!("i{i}"), 1).get_lsb(0))
            .collect();
        (b, inputs)
    }
    /// Checks serialized output, unique cell costs, and exhaustive Boolean
    /// semantics.
    fn verify(aig: &GateFn, lib: &Library) -> AreaMappedNetlist {
        let index = AreaMappingLibrary::new(lib).unwrap();
        let mapped = map_gatefn_to_area_netlist(aig, &index).unwrap();
        let text = emit_module_as_netlist_text(
            &mapped.netlist.module,
            &mapped.netlist.nets,
            &mapped.netlist.interner,
        )
        .unwrap();
        let parsed = read_gv_from_str(&text).unwrap();
        let projected = project_gatefn_from_netlist_and_liberty(
            &parsed.modules[0],
            &parsed.nets,
            &parsed.interner,
            lib,
            &HashSet::new(),
            &HashSet::new(),
        )
        .unwrap();
        for stimulus in 0..1usize << aig.inputs.len() {
            let inputs = (0..aig.inputs.len())
                .map(|i| IrBits::make_ubits(1, ((stimulus >> i) & 1) as u64).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(
                eval(aig, &inputs, Collect::None).outputs,
                eval(&projected, &inputs, Collect::None).outputs,
                "{text}"
            );
        }
        #[cfg(feature = "has-bitwuzla")]
        assert_eq!(
            prove_same_gate_fn_via_ir_status_with_solver(
                aig,
                &projected,
                SolverChoice::Bitwuzla,
                None
            ),
            IrCheckResult::Equivalent
        );
        let actual = mapped
            .netlist
            .module
            .instances
            .iter()
            .map(|inst| {
                let name = mapped.netlist.interner.resolve(inst.type_name).unwrap();
                lib.cells.iter().find(|c| c.name == name).unwrap().area
            })
            .sum::<f64>();
        assert_eq!(actual, mapped.stats.cell_area);
        assert_eq!(
            mapped.stats.cell_count,
            mapped.netlist.module.instances.len()
        );
        assert_eq!(mapped.stats.duplicated_ands, 0);
        let again = map_gatefn_to_area_netlist(aig, &index).unwrap();
        assert_eq!(
            text,
            emit_module_as_netlist_text(
                &again.netlist.module,
                &again.netlist.nets,
                &again.netlist.interner
            )
            .unwrap()
        );
        mapped
    }
    #[test]
    fn shared_input_inversion_is_emitted_once() {
        let (mut b, x) = builder();
        let a = b.add_and_binary(x[0].negate(), x[1]);
        let c = b.add_and_binary(x[0].negate(), x[2]);
        b.add_output("a".into(), a.negate().into());
        b.add_output("c".into(), c.negate().into());
        let lib = library(&[
            ("INV", &["A"], "!A", 1.0),
            ("NAND2", &["A", "B"], "!(A*B)", 2.0),
        ]);
        let mapped = verify(&b.build(), &lib);
        assert_eq!(
            (
                mapped.stats.cell_count,
                mapped.stats.inverter_count,
                mapped.stats.cell_area
            ),
            (3, 1, 5.0)
        );
    }
    #[test]
    fn generated_names_do_not_collide_with_ports() {
        let mut b = GateBuilder::new("collision".into(), GateBuilderOptions::no_opt());
        let x = *b.add_input("u_tm_0".into(), 1).get_lsb(0);
        let y = *b.add_input("tm_1".into(), 1).get_lsb(0);
        let z = b.add_and_binary(x, y);
        b.add_output("o".into(), z.into());
        let mapped = verify(&b.build(), &basic_library());
        let mut names = BTreeSet::new();
        for net in &mapped.netlist.nets {
            assert!(names.insert(mapped.netlist.interner.resolve(net.name).unwrap()));
        }
        for inst in &mapped.netlist.module.instances {
            assert!(names.insert(mapped.netlist.interner.resolve(inst.instance_name).unwrap()));
        }
    }
    #[test]
    fn negative_output_uses_nand_without_inverter() {
        let (mut b, x) = builder();
        let y = b.add_and_binary(x[0], x[1]);
        b.add_output("o".into(), y.negate().into());
        let mapped = verify(&b.build(), &basic_library());
        assert_eq!(
            (
                mapped.stats.cell_count,
                mapped.stats.cell_area,
                mapped.stats.inverter_count
            ),
            (1, 2.0, 0)
        );
    }
    #[test]
    fn repeated_multioutput_both_polarities_share_one_implementation() {
        let (mut b, x) = builder();
        let y = b.add_and_binary(x[0], x[1]);
        for (i, op) in [y, y.negate(), y, y.negate()].into_iter().enumerate() {
            b.add_output(format!("o{i}"), op.into());
        }
        let mapped = verify(&b.build(), &basic_library());
        assert_eq!(
            (
                mapped.stats.cell_count,
                mapped.stats.cell_area,
                mapped.stats.shared_opposite_polarities
            ),
            (2, 3.0, 1)
        );
        assert_eq!(mapped.stats.covered_ands, 1);
    }
    #[test]
    fn reconvergent_shared_cone_is_never_absorbed_twice() {
        let (mut b, x) = builder();
        let shared = b.add_and_binary(x[0], x[1]);
        let l = b.add_and_binary(shared, x[2]);
        let r = b.add_and_binary(shared.negate(), x[3]);
        let y = b.add_and_binary(l, r.negate());
        b.add_output("o".into(), y.into());
        b.add_output("shared".into(), shared.into());
        let mapped = verify(&b.build(), &basic_library());
        assert_eq!(mapped.stats.live_ands, 4);
        assert_eq!(mapped.stats.covered_ands, 4);
        assert_eq!(mapped.stats.shared_and_boundaries, 1);
    }
    #[test]
    fn asymmetric_cell_permutation_and_input_polarity_are_matched() {
        let (mut b, x) = builder();
        let inner = b.add_and_binary(x[1], x[2]);
        let y = b.add_or_binary(inner, x[0]);
        b.add_output("o".into(), y.into());
        let mapped = verify(&b.build(), &basic_library());
        assert_eq!(
            mapped.stats.cells_by_type,
            BTreeMap::from([("AO21".into(), 1)])
        );
        let (mut b, x) = builder();
        let inner = b.add_and_binary(x[1], x[2]);
        let y = b.add_or_binary(inner, x[0].negate());
        b.add_output("o".into(), y.into());
        let mapped = verify(&b.build(), &basic_library());
        assert_eq!(mapped.stats.cell_area, 3.5);
    }
    #[test]
    fn five_input_cover_and_ideal_constants_aliases() {
        let (mut b, x) = builder();
        let mut y = x[0];
        for &op in &x[1..] {
            y = b.add_and_binary(y, op);
        }
        b.add_output("o".into(), y.negate().into());
        let mapped = verify(&b.build(), &basic_library());
        assert_eq!(mapped.stats.cell_count, 1);
        assert_eq!(mapped.stats.covered_ands, 4);
        let (mut b, x) = builder();
        let y = b.add_and_binary(x[0], b.get_true());
        let z = b.add_and_binary(y, b.get_false());
        for (i, op) in [y, z, z.negate(), b.get_true(), b.get_false(), x[0]]
            .into_iter()
            .enumerate()
        {
            b.add_output(format!("o{i}"), op.into());
        }
        let mapped = verify(&b.build(), &basic_library());
        assert_eq!(mapped.stats.cell_count, 0);
    }
    #[test]
    fn wider_cell_can_cover_an_intermediate_without_its_own_cover() {
        let (mut b, x) = builder();
        let inner = b.add_and_binary(x[0], x[1]);
        let outer = b.add_and_binary(inner, x[2]);
        b.add_output("o".into(), outer.negate().into());
        let lib = library(&[
            ("INV", &["A"], "!A", 1.0),
            ("NAND3", &["A", "B", "C"], "!(A*B*C)", 2.0),
        ]);
        let mapped = verify(&b.build(), &lib);
        assert_eq!(
            mapped.stats.cells_by_type,
            BTreeMap::from([("NAND3".into(), 1)])
        );
        assert_eq!(mapped.stats.covered_ands, 2);
    }
    #[test]
    fn unused_uncoverable_shared_node_can_fold_to_constants() {
        let (mut b, x) = builder();
        let shared = b.add_and_binary(x[0], x[1]);
        let first = b.add_and_binary(shared, b.get_false());
        let second = b.add_and_binary(shared, b.get_false());
        b.add_output("first".into(), first.into());
        b.add_output("second".into(), second.negate().into());
        let mapped = verify(&b.build(), &library(&[]));
        assert_eq!(mapped.stats.shared_and_boundaries, 1);
        assert_eq!(mapped.stats.cell_count, 0);
        assert_eq!(mapped.stats.live_ands, 3);
        assert_eq!(mapped.stats.covered_ands, 2);
    }
    #[test]
    fn shared_constant_boundary_propagates_to_downstream_outputs() {
        let (mut b, x) = builder();
        let shared = b.add_and_binary(x[0], b.get_false());
        let constant = b.add_and_binary(shared, x[1]);
        let alias = b.add_and_binary(shared.negate(), x[2]);
        b.add_output("shared".into(), shared.into());
        b.add_output("constant".into(), constant.into());
        b.add_output("alias".into(), alias.into());
        let mapped = verify(&b.build(), &library(&[]));
        assert_eq!(mapped.stats.cell_count, 0);
        assert_eq!(mapped.stats.live_ands, 3);
        assert_eq!(mapped.stats.covered_ands, 3);
    }
    #[test]
    fn wider_cell_does_not_claim_a_substituted_shared_constant() {
        let (mut b, x) = builder();
        let shared = b.add_and_binary(x[0], b.get_false());
        let first = b.add_and_binary(x[1], shared.negate());
        let second = b.add_and_binary(first, x[2]);
        let third = b.add_and_binary(second, x[3]);
        b.add_output("shared".into(), shared.into());
        b.add_output("result".into(), third.negate().into());
        let lib = library(&[("NAND3", &["A", "B", "C"], "!(A*B*C)", 2.0)]);
        let mapped = verify(&b.build(), &lib);
        assert_eq!(
            mapped.stats.cells_by_type,
            BTreeMap::from([("NAND3".into(), 1)])
        );
        assert_eq!(mapped.stats.live_ands, 4);
        assert_eq!(mapped.stats.covered_ands, 4);
    }
    #[test]
    fn required_uncoverable_shared_node_reports_an_error() {
        let (mut b, x) = builder();
        let shared = b.add_and_binary(x[0], x[1]);
        let outer = b.add_and_binary(shared, x[2]);
        b.add_output("outer".into(), outer.negate().into());
        b.add_output("shared".into(), shared.into());
        let lib = library(&[
            ("INV", &["A"], "!A", 1.0),
            ("NAND3", &["A", "B", "C"], "!(A*B*C)", 2.0),
        ]);
        let index = AreaMappingLibrary::new(&lib).unwrap();
        let error = map_gatefn_to_area_netlist(&b.build(), &index).unwrap_err();
        assert!(error.to_string().contains("no cover for demanded node"));
    }
    #[test]
    fn mapping_without_an_inverter_uses_only_demanded_polarities() {
        let (mut b, x) = builder();
        let y = b.add_and_binary(x[0], x[1]);
        b.add_output("a".into(), y.into());
        b.add_output("b".into(), y.into());
        b.add_output("wire".into(), x[2].into());
        b.add_output("constant".into(), b.get_true().into());
        let alias = b.add_and_binary(x[3], b.get_true());
        b.add_output("alias".into(), alias.into());
        let lib = library(&[("AND2", &["A", "B"], "A*B", 2.0)]);
        let mapped = verify(&b.build(), &lib);
        assert_eq!(
            mapped.stats.cells_by_type,
            BTreeMap::from([("AND2".into(), 1)])
        );
        assert_eq!(mapped.stats.inverter_count, 0);
        assert_eq!(mapped.stats.covered_ands, 2);

        let (mut b, x) = builder();
        let y = b.add_and_binary(x[0], x[1]);
        b.add_output("a".into(), y.negate().into());
        b.add_output("b".into(), y.negate().into());
        let lib = library(&[("NAND2", &["A", "B"], "!(A*B)", 2.0)]);
        let mapped = verify(&b.build(), &lib);
        assert_eq!(
            mapped.stats.cells_by_type,
            BTreeMap::from([("NAND2".into(), 1)])
        );

        let (mut b, x) = builder();
        let constant = b.add_and_binary(x[0], b.get_false());
        b.add_output("zero".into(), constant.into());
        b.add_output("one".into(), constant.negate().into());
        let mapped = verify(&b.build(), &library(&[]));
        assert_eq!(mapped.stats.cell_count, 0);
        assert_eq!(mapped.stats.covered_ands, 1);
    }
    #[test]
    fn mapping_without_an_inverter_rejects_required_opposite_polarities() {
        let (mut b, x) = builder();
        let y = b.add_and_binary(x[0], x[1]);
        b.add_output("positive".into(), y.into());
        b.add_output("negative".into(), y.negate().into());
        let lib = library(&[
            ("AND2", &["A", "B"], "A*B", 2.0),
            ("NAND2", &["A", "B"], "!(A*B)", 2.0),
        ]);
        let index = AreaMappingLibrary::new(&lib).unwrap();
        let error = map_gatefn_to_area_netlist(&b.build(), &index).unwrap_err();
        assert!(error.to_string().contains("library has no inverter"));

        let (mut b, x) = builder();
        let y = b.add_and_binary(x[0], x[1].negate());
        b.add_output("o".into(), y.into());
        let index =
            AreaMappingLibrary::new(&library(&[("AND2", &["A", "B"], "A*B", 2.0)])).unwrap();
        let error = map_gatefn_to_area_netlist(&b.build(), &index).unwrap_err();
        assert!(error.to_string().contains("no cover for demanded node"));

        // Both polarities of the shared AND are needed by one downstream cone.
        let (mut b, x) = builder();
        let shared = b.add_and_binary(x[0], x[1]);
        let a = b.add_and_binary(shared, x[2]);
        let c = b.add_and_binary(shared.negate(), x[3]);
        let mux = b.add_or_binary(a, c);
        b.add_output("o".into(), mux.into());
        let index = AreaMappingLibrary::new(&lib).unwrap();
        assert!(map_gatefn_to_area_netlist(&b.build(), &index).is_err());
    }
    #[test]
    fn multioutput_and_dont_use_cells_are_excluded_invalid_area_rejected() {
        let mut lib = basic_library();
        let mut multi = lib.cells[1].clone();
        multi.name = "MULTI".into();
        multi.pins.push(multi.pins.last().unwrap().clone());
        lib.cells.push(multi);
        let mut forbidden = lib.cells[1].clone();
        forbidden.name = "FORBIDDEN".into();
        forbidden.dont_use = Some(true);
        lib.cells.push(forbidden);
        assert_eq!(
            AreaMappingLibrary::new(&lib).unwrap().excluded_cells,
            vec!["FORBIDDEN", "MULTI"]
        );
        lib.cells[0].area = f64::NAN;
        assert!(AreaMappingLibrary::new(&lib).is_err());
    }
    #[test]
    fn size_only_cells_are_excluded_after_binary_and_text_roundtrips() {
        let source = r#"
        library(test) {
            cell(AND2_SIZE) {
                area: 1; use_for_size_only: true;
                pin(A) { direction: input; } pin(B) { direction: input; }
                pin(Y) { direction: output; function: "A * B"; }
            }
            cell(AND2_FALSE) {
                area: 2; use_for_size_only: false;
                pin(A) { direction: input; } pin(B) { direction: input; }
                pin(Y) { direction: output; function: "A * B"; }
            }
            cell(AND2_ABSENT) {
                area: 3;
                pin(A) { direction: input; } pin(B) { direction: input; }
                pin(Y) { direction: output; function: "A * B"; }
            }
        }
        "#;
        let mut source_file = NamedTempFile::new().unwrap();
        write!(source_file, "{source}").unwrap();
        let imported = parse_liberty_files(&[source_file.path()]).unwrap();
        let binary = library_to_proto(parse_liberty_files(&[source_file.path()]).unwrap())
            .unwrap()
            .encode_to_vec();
        let binary_roundtrip =
            library_from_proto(WireLibrary::decode(binary.as_slice()).unwrap()).unwrap();
        let descriptor = DescriptorPool::decode(LIBERTY_DESCRIPTOR)
            .unwrap()
            .get_message_by_name("liberty.Library")
            .unwrap();
        let text = DynamicMessage::decode(descriptor.clone(), binary.as_slice())
            .unwrap()
            .to_text_format();
        let text_binary = DynamicMessage::parse_text_format(descriptor, &text)
            .unwrap()
            .encode_to_vec();
        let text_roundtrip =
            library_from_proto(WireLibrary::decode(text_binary.as_slice()).unwrap()).unwrap();
        for mut lib in [imported, binary_roundtrip, text_roundtrip] {
            assert_eq!(
                lib.cells
                    .iter()
                    .map(|c| (c.name.as_str(), c.use_for_size_only))
                    .collect::<Vec<_>>(),
                vec![
                    ("AND2_ABSENT", None),
                    ("AND2_FALSE", Some(false)),
                    ("AND2_SIZE", Some(true))
                ]
            );
            let (mut b, x) = builder();
            let y = b.add_and_binary(x[0], x[1]);
            b.add_output("o".into(), y.into());
            let aig = b.build();
            let index = AreaMappingLibrary::new(&lib).unwrap();
            assert_eq!(index.excluded_cells, vec!["AND2_SIZE"]);
            let mapped = map_gatefn_to_area_netlist(&aig, &index).unwrap();
            assert_eq!(
                mapped.stats.cells_by_type,
                BTreeMap::from([("AND2_FALSE".into(), 1)])
            );
            lib.cells.retain(|c| c.name != "AND2_FALSE");
            let index = AreaMappingLibrary::new(&lib).unwrap();
            let mapped = map_gatefn_to_area_netlist(&aig, &index).unwrap();
            assert_eq!(
                mapped.stats.cells_by_type,
                BTreeMap::from([("AND2_ABSENT".into(), 1)])
            );
        }
    }

    #[test]
    fn legacy_unknown_sizing_policy_is_excluded_from_area_mapping() {
        let legacy = library(&[
            ("LEGACY_AND", &["A", "B"], "A*B", 0.5),
            ("LEGACY_INV", &["A"], "!A", 0.5),
        ]);
        let mut wire = library_to_proto(legacy).unwrap();
        wire.format_magic = crate::liberty::model::LIBERTY_FORMAT_MAGIC_V5;
        let legacy = library_from_proto(wire).unwrap();
        assert!(legacy.cells.iter().all(|cell| {
            cell.validate_functional_semantics().is_ok()
                && cell.validate_mapping_policy_coverage().is_err()
        }));
        let index = AreaMappingLibrary::new(&legacy).unwrap();
        let (mut b, x) = builder();
        let and = b.add_and_binary(x[0], x[1]);
        b.add_output("and_out".into(), and.into());
        let error = map_gatefn_to_area_netlist(&b.build(), &index).unwrap_err();
        assert_eq!(
            error.to_string(),
            "cell 'LEGACY_AND' has unknown use_for_size_only source coverage; regenerate from source Liberty before mapping"
        );

        // Wiring alone requires no cells, even with unknown legacy policies.
        let (mut b, x) = builder();
        b.add_output("wire_out".into(), x[0].into());
        assert!(map_gatefn_to_area_netlist(&b.build(), &index).is_ok());

        let mut mixed = library(&[
            ("AND2_SAFE", &["A", "B"], "A*B", 2.0),
            ("INV_SAFE", &["A"], "!A", 2.0),
        ]);
        mixed.append_cells_and_luts(legacy).unwrap();
        let index = AreaMappingLibrary::new(&mixed).unwrap();
        assert_eq!(index.excluded_cells, vec!["LEGACY_AND", "LEGACY_INV"]);
        assert_eq!(index.cells[index.inv.unwrap()].name, "INV_SAFE");

        let (mut b, x) = builder();
        let and = b.add_and_binary(x[0], x[1]);
        b.add_output("and_out".into(), and.into());
        b.add_output("not_out".into(), x[2].negate().into());
        let mapped = map_gatefn_to_area_netlist(&b.build(), &index).unwrap();
        assert_eq!(
            mapped.stats.cells_by_type,
            BTreeMap::from([("AND2_SAFE".into(), 1), ("INV_SAFE".into(), 1)])
        );
    }

    #[test]
    fn mixed_library_missing_cover_mentions_unknown_legacy_sizing_policy() {
        let legacy = library(&[("INV_LEGACY", &["A"], "!A", 1.0)]);
        let mut wire = library_to_proto(legacy).unwrap();
        wire.format_magic = crate::liberty::model::LIBERTY_FORMAT_MAGIC_V5;
        let legacy = library_from_proto(wire).unwrap();
        let mut mixed = library(&[("AND2_SAFE", &["A", "B"], "A*B", 1.0)]);
        mixed.append_cells_and_luts(legacy).unwrap();
        let index = AreaMappingLibrary::new(&mixed).unwrap();
        assert_eq!(index.excluded_cells, vec!["INV_LEGACY"]);

        let (mut b, x) = builder();
        b.add_output("not_out".into(), x[0].negate().into());
        let error = map_gatefn_to_area_netlist(&b.build(), &index).unwrap_err();
        assert_eq!(
            error.to_string(),
            "some library cells were excluded from area mapping: cell 'INV_LEGACY' has unknown use_for_size_only source coverage; regenerate from source Liberty before mapping"
        );
        assert_eq!(
            error.root_cause().to_string(),
            "node 1 requires both polarities but the library has no inverter"
        );
    }
    #[test]
    fn unsupported_formulas_are_excluded_without_blocking_other_cells() {
        let lib = library(&[
            ("ESCAPED", &["1A", "1B"], "\"1A\" + \"1B\"", 0.0),
            ("AND2", &["A", "B"], "A*B", 1.0),
        ]);
        let index = AreaMappingLibrary::new(&lib).unwrap();
        assert_eq!(index.excluded_cells, vec!["ESCAPED"]);
        let (mut b, x) = builder();
        let y = b.add_and_binary(x[0], x[1]);
        b.add_output("o".into(), y.into());
        let mapped = map_gatefn_to_area_netlist(&b.build(), &index).unwrap();
        assert_eq!(mapped.stats.cell_count, 1);
        assert_eq!(mapped.stats.cell_area, 1.0);
    }
    #[test]
    fn invalid_cell_and_pin_names_are_rejected() {
        let mut lib = basic_library();
        lib.cells[0].name.clear();
        assert!(AreaMappingLibrary::new(&lib).is_err());

        for pin in [0, 2] {
            let mut lib = basic_library();
            lib.cells[1].pins[pin].name = Default::default();
            assert!(AreaMappingLibrary::new(&lib).is_err());
        }
        for duplicate in [1, 2] {
            let mut lib = basic_library();
            lib.cells[1].pins[duplicate].name = lib.cells[1].pins[0].name;
            assert!(AreaMappingLibrary::new(&lib).is_err());
        }
    }
    #[test]
    fn deep_chain_maps_without_recursive_emission() {
        let mut b = GateBuilder::new("deep".into(), GateBuilderOptions::no_opt());
        let mut y = *b.add_input("i0".into(), 1).get_lsb(0);
        for i in 1..=20_000 {
            let input = *b.add_input(format!("i{i}"), 1).get_lsb(0);
            y = b.add_and_binary(y, input);
        }
        b.add_output("o".into(), y.into());
        let lib = library(&[("AND2", &["A", "B"], "A*B", 1.0)]);
        let index = AreaMappingLibrary::new(&lib).unwrap();
        let mapped = map_gatefn_to_area_netlist(&b.build(), &index).unwrap();
        assert_eq!(mapped.stats.cell_count, 20_000);
        assert_eq!(mapped.stats.covered_ands, 20_000);
        assert_eq!(mapped.stats.duplicated_ands, 0);
    }
    #[test]
    fn deep_chain_shares_repeated_opposite_input_demands() {
        let mut b = GateBuilder::new("demands".into(), GateBuilderOptions::no_opt());
        let mut y = *b.add_input("seed".into(), 1).get_lsb(0);
        let inputs = (0..2_000)
            .map(|i| *b.add_input(format!("i{i}"), 1).get_lsb(0))
            .collect::<Vec<_>>();
        for _ in 0..2 {
            for &input in &inputs {
                y = b.add_and_binary(y, input.negate());
            }
        }
        b.add_output("o".into(), y.into());
        let lib = library(&[
            ("INV", &["A"], "!A", 1.0),
            ("AND2", &["A", "B"], "A*B", 1.0),
        ]);
        let index = AreaMappingLibrary::new(&lib).unwrap();
        let mapped = map_gatefn_to_area_netlist(&b.build(), &index).unwrap();
        assert_eq!(mapped.stats.inverter_count, inputs.len());
        assert_eq!(mapped.stats.duplicated_ands, 0);
    }
    #[test]
    fn random_shared_dags_match_exhaustive_simulation_and_bitwuzla() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let lib = basic_library();
        for _ in 0..24 {
            let (mut b, mut nodes) = builder();
            nodes.extend([b.get_false(), b.get_true()]);
            for _ in 0..30 {
                let mut a = nodes[rng.gen_range(0..nodes.len())];
                let mut c = nodes[rng.gen_range(0..nodes.len())];
                if rng.gen_bool(0.5) {
                    a = a.negate();
                }
                if rng.gen_bool(0.5) {
                    c = c.negate();
                }
                nodes.push(b.add_and_binary(a, c));
            }
            for i in 0..6 {
                let op = nodes[nodes.len() - 1 - i];
                b.add_output(format!("o{i}"), op.into());
            }
            verify(&b.build(), &lib);
        }
    }
}
