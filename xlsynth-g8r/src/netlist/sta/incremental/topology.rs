// SPDX-License-Identifier: Apache-2.0

//! Generic combinational topology transactions on the existing timing engine.
//! Connectivity, Boolean coverage and timing use sparse transactional journals.
//! Backward dependencies can require an explicitly counted ordering fallback.

use super::*;
use anyhow::ensure;
use std::collections::BTreeMap;

/// One scalar cell instance; cell indices refer to the session's Liberty
/// library.
#[derive(Clone, Debug)]
pub struct TimingInstance {
    pub name: String,
    pub cell: usize,
    /// Connection order is retained for deterministic capacitance summation.
    pub connections: Vec<(String, BitSource)>,
}

/// Policy-free mutations. Equivalence and optimization profitability belong to
/// callers.
#[derive(Clone, Debug)]
pub enum TimingMutation {
    /// Keeps pin bindings and timing-condition constants unchanged. Equivalence
    /// is the caller's responsibility; use Replace to change those bindings.
    Resize {
        instance: usize,
        cell: usize,
    },
    Replace {
        instance: usize,
        cell: usize,
        connections: Vec<(String, BitSource)>,
    },
    Insert(TimingInstance),
    Rewire {
        instance: usize,
        pin: String,
        source: BitSource,
    },
}

/// An atomic batch; new scalar bits append after the current normalized bit
/// count. Existing aliases and module ports cannot change. Inserted instances
/// append in mutation order. Rewires can refer to bits and instances inserted
/// in this batch.
#[derive(Clone, Debug, Default)]
pub struct TimingEdit {
    pub new_bits: usize,
    pub mutations: Vec<TimingMutation>,
}

/// Bounds and optional timing constraints for one transaction.
#[derive(Clone, Copy, Debug)]
pub struct TrialOptions {
    /// Maximum combined timing and Boolean-coverage cell visits.
    pub cell_budget: usize,
    pub limits: Option<BoundaryDelays>,
}
impl Default for TrialOptions {
    fn default() -> Self {
        Self {
            cell_budget: usize::MAX,
            limits: None,
        }
    }
}

/// Coverage work required by mutation preparation.
enum CoverageUpdate {
    Unchanged,
    Rescan,
    Sparse(BTreeSet<usize>),
}

/// Seeds and invalidated observations produced by mutation preparation.
struct EditPlan {
    queue: BTreeSet<(usize, usize)>,
    affected: BTreeSet<usize>,
    endpoints: BTreeSet<usize>,
    coverage: CoverageUpdate,
}

/// Work performed by a transaction, including rejected trials.
#[derive(Clone, Copy, Debug, Default)]
pub struct TrialResult {
    pub outcome: TrialOutcome,
    /// Instances whose pin bindings were reconstructed.
    pub rebuilt_instances: usize,
    /// Cell visits in Boolean coverage propagation.
    pub coverage_instances: usize,
    pub propagated_instances: usize,
    /// Nets whose connectivity/load records were journaled.
    pub touched_nets: usize,
    /// Instances visited by the exceptional global ordering fallback.
    pub reordered_instances: usize,
}

/// Gaps let ordinary insertion trials retain all existing instance ranks.
pub(super) const RANK_GAP: usize = 1 << 20;

/// Only touched graph records are copied. Rare ordering fallbacks retain the
/// old rank vector; ordinary rejected trials restore in affected-cone time.
struct TransactionUndo {
    bits: usize,
    instances: usize,
    changed: BTreeMap<usize, (Instance, usize, Vec<usize>)>,
    nets: BTreeMap<usize, NetUndo>,
    known: BTreeMap<usize, HashMap<String, bool>>,
    coverage: BTreeMap<usize, HashMap<String, bool>>,
    coverage_values: BTreeMap<usize, Option<bool>>,
    ranks: Option<Vec<usize>>,
    traversal_dirty: bool,
    timing: Undo,
}

/// Connectivity and electrical state for one touched canonical net.
struct NetUndo {
    driver: Option<usize>,
    loads: Vec<(usize, String)>,
    consumers: Vec<usize>,
    capacitance: EdgeLoadCapacitance,
    constant: Option<bool>,
}

impl TransactionUndo {
    fn new(s: &IncrementalSta<'_>) -> Self {
        Self {
            bits: s.bit_count(),
            instances: s.instance_count(),
            changed: BTreeMap::new(),
            nets: BTreeMap::new(),
            known: BTreeMap::new(),
            coverage: BTreeMap::new(),
            coverage_values: BTreeMap::new(),
            ranks: None,
            traversal_dirty: s.traversal_dirty,
            timing: Undo::default(),
        }
    }

    /// Journals a net once, before its first connectivity mutation.
    fn net(&mut self, s: &IncrementalSta<'_>, bit: usize) {
        self.nets.entry(bit).or_insert_with(|| NetUndo {
            driver: s.drivers[bit],
            loads: s.loads[bit].clone(),
            consumers: s.consumers[bit].clone(),
            capacitance: s.capacitances[bit],
            constant: s.timing_constants[bit],
        });
    }

    /// Restores timings before truncating new IDs, then restores graph records.
    fn restore(self, s: &mut IncrementalSta<'_>) {
        for (bit, values) in self.timing.bits.into_iter().rev() {
            [s.timing[0][bit], s.timing[1][bit]] = values;
        }
        for (i, counts) in self.timing.counts.into_iter().rev() {
            s.set_query_counts(i, counts);
        }
        for (endpoint, value) in self.timing.endpoints {
            s.set_endpoint(endpoint, value);
        }
        for (i, known) in self.known {
            s.instances[i].known = known;
        }
        for (i, (inst, cell, order)) in self.changed {
            s.instances[i] = inst;
            s.cells[i] = cell;
            s.orders[i] = order;
        }
        for (bit, old) in self.nets {
            s.drivers[bit] = old.driver;
            s.loads[bit] = old.loads;
            s.consumers[bit] = old.consumers;
            s.capacitances[bit] = old.capacitance;
            s.timing_constants[bit] = old.constant;
        }
        for (i, old) in self.coverage {
            s.coverage_known[i] = old;
        }
        for (bit, old) in self.coverage_values {
            s.coverage_values[bit] = old;
        }
        s.traversal_dirty = self.traversal_dirty;
        if let Some(ranks) = self.ranks {
            s.positions = ranks;
        }
        for inst in &s.instances[self.instances..] {
            s.instance_names.remove(&inst.name);
        }
        s.instances.truncate(self.instances);
        s.cells.truncate(self.instances);
        s.orders.truncate(self.instances);
        s.positions.truncate(self.instances);
        s.capture_endpoint.truncate(self.instances);
        s.electrical.truncate(self.instances);
        s.coverage_known.truncate(self.instances);
        s.query_counts.truncate(self.instances);
        s.bit_names.truncate(self.bits);
        s.bit_sources.truncate(self.bits);
        s.drivers.truncate(self.bits);
        s.loads.truncate(self.bits);
        s.consumers.truncate(self.bits);
        s.capacitances.truncate(self.bits);
        s.external_load_counts.truncate(self.bits);
        s.endpoints_by_bit.truncate(self.bits);
        s.timing_constants.truncate(self.bits);
        s.coverage_values.truncate(self.bits);
        for timing in &mut s.timing {
            timing.truncate(self.bits);
        }
    }
}

impl IncrementalSta<'_> {
    /// Number of stable normalized bits, including previously committed
    /// additions.
    pub fn bit_count(&self) -> usize {
        self.bit_names.len()
    }

    /// Number of stable instance IDs.
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Current cell selection for an instance ID.
    pub fn cell_index(&self, instance: usize) -> Option<usize> {
        self.cells.get(instance).copied()
    }

    /// Canonical source after resolving the original module's wiring aliases.
    pub fn canonical_source(&self, bit: usize) -> Option<BitSource> {
        self.bit_sources.get(bit).copied()
    }

    /// PI-launched edge arrival/slew for a normalized bit; aliases resolve to
    /// roots.
    pub fn timing_for_bit(&self, bit: usize) -> Option<SignalTiming> {
        match *self.bit_sources.get(bit)? {
            BitSource::Bit(root) => self.timing[0][root].as_ref()?.as_report_signal_timing(),
            BitSource::Literal(_) => self.literal.as_report_signal_timing(),
            BitSource::Unknown => None,
        }
    }

    /// Rise/fall capacitance at a canonical bit.
    pub fn load_for_bit(&self, bit: usize) -> Option<[f64; 2]> {
        let BitSource::Bit(root) = *self.bit_sources.get(bit)? else {
            return None;
        };
        let load = self.capacitances.get(root)?;
        Some([load.rise, load.fall])
    }

    /// Current cell delay/slew query diagnostics for a combinational graph.
    /// Registered capture-query accounting is intentionally not represented
    /// by this accessor. Transactional totals make this constant-time.
    pub fn combinational_query_diagnostics(&self) -> Option<TimingQueryDiagnosticCounts> {
        if self.registered {
            return None;
        }
        Some(self.query_totals)
    }

    /// Builds an instance binding through the same pin and timing-order checks.
    fn topology_instance(&self, spec: &TimingInstance) -> Result<(Instance, Vec<usize>)> {
        ensure!(!spec.name.is_empty(), "empty instance name");
        self.lib
            .library
            .cells
            .get(spec.cell)
            .ok_or_else(|| anyhow!("unknown cell index"))?;
        self.lib
            .validate_functional_semantics(spec.cell)
            .map_err(anyhow::Error::msg)?;
        // Connected output validation below and the shared evaluator determine
        // combinational eligibility. Unused sequential arcs do not create a
        // register boundary in an explicitly combinational session.
        self.lib.timing_related_input_pins(spec.cell)?;
        let mut pins = HashMap::new();
        let mut pin_order = Vec::new();
        for (name, source) in &spec.connections {
            let pin = self
                .lib
                .pin(spec.cell, name)
                .ok_or_else(|| anyhow!("unknown pin {name}"))?;
            ensure!(
                pin.direction == PinDirection::Input as i32
                    || pin.direction == PinDirection::Output as i32,
                "unsupported pin direction"
            );
            let source = match *source {
                BitSource::Bit(bit) => {
                    ensure!(bit < self.bit_count(), "unknown bit {bit}");
                    if pin.direction == PinDirection::Input as i32 {
                        self.bit_sources[bit]
                    } else {
                        ensure!(
                            self.bit_sources[bit] == BitSource::Bit(bit),
                            "cannot drive an alias"
                        );
                        BitSource::Bit(bit)
                    }
                }
                value => value,
            };
            if pin.direction == PinDirection::Output as i32 {
                ensure!(
                    matches!(source, BitSource::Bit(_)),
                    "output must drive a bit"
                );
            }
            ensure!(
                pins.insert(name.clone(), vec![source]).is_none(),
                "duplicate pin {name}"
            );
            pin_order.push(name.clone());
        }
        let order = combinational_output_pin_evaluation_order(&self.lib, spec.cell, &pins)?;
        Ok((
            Instance {
                name: spec.name.clone(),
                pin_order,
                pins,
                known: HashMap::new(),
                outputs: Vec::new(),
                sequential: false,
            },
            order,
        ))
    }

    /// Reconstructs cached Boolean outputs after initialization or the existing
    /// resize API's exceptional full coverage pass.
    pub(super) fn refresh_coverage_values(&mut self) {
        self.coverage_values.fill(None);
        for i in 0..self.instance_count() {
            if !self.lib.supports_boolean_analysis(self.cells[i]) {
                continue;
            }
            for &pin in &self.orders[i] {
                let name = self
                    .lib
                    .library
                    .resolve_string(&self.lib.library.cells[self.cells[i]].pins[pin].name);
                if let Some([BitSource::Bit(bit)]) =
                    self.instances[i].pins.get(name).map(Vec::as_slice)
                {
                    self.coverage_values[*bit] = self
                        .lib
                        .constant_function(self.cells[i], pin)
                        .and_then(|f| f.evaluate_partial(&self.coverage_known[i]));
                }
            }
        }
    }

    /// Re-evaluates one cell's least Boolean fixed point. Self-feedback starts
    /// unknown so a prior constant cannot justify itself after a rewire.
    fn topology_coverage(&self, i: usize) -> (HashMap<String, bool>, Vec<(usize, Option<bool>)>) {
        let inst = &self.instances[i];
        let mut known: HashMap<_, _> = inst
            .pins
            .iter()
            .filter_map(|(name, sources)| {
                let value = match sources.as_slice() {
                    [BitSource::Literal(v)] => Some(*v),
                    [BitSource::Bit(bit)] if self.drivers[*bit] != Some(i) => {
                        self.coverage_values[*bit]
                    }
                    _ => None,
                };
                value.map(|v| (name.clone(), v))
            })
            .collect();
        let mut values: BTreeMap<_, _> = inst.outputs.iter().map(|&b| (b, None)).collect();
        if self.lib.supports_boolean_analysis(self.cells[i]) {
            loop {
                let mut progress = false;
                for &pin in &self.orders[i] {
                    let name = self
                        .lib
                        .library
                        .resolve_string(&self.lib.library.cells[self.cells[i]].pins[pin].name);
                    let Some([BitSource::Bit(bit)]) = inst.pins.get(name).map(Vec::as_slice) else {
                        continue;
                    };
                    if values[bit].is_some() {
                        continue;
                    }
                    if let Some(value) = self
                        .lib
                        .constant_function(self.cells[i], pin)
                        .and_then(|f| f.evaluate_partial(&known))
                    {
                        values.insert(*bit, Some(value));
                        progress = true;
                        for (input, sources) in &inst.pins {
                            if sources.as_slice() == [BitSource::Bit(*bit)]
                                && self
                                    .lib
                                    .pin(self.cells[i], input)
                                    .is_some_and(|p| p.direction == PinDirection::Input as i32)
                            {
                                known.insert(input.clone(), value);
                            }
                        }
                    }
                }
                if !progress {
                    break;
                }
            }
        }
        (known, values.into_iter().collect())
    }

    /// Updates coverage only until changed Boolean values stop propagating.
    fn propagate_topology_coverage(
        &mut self,
        changed: &BTreeSet<usize>,
        undo: &mut TransactionUndo,
        affected: &mut BTreeSet<usize>,
        budget: usize,
    ) -> (usize, bool) {
        let mut queue: BTreeSet<_> = changed.iter().map(|&i| (self.positions[i], i)).collect();
        let mut visits = 0;
        while let Some((_, i)) = queue.pop_first() {
            if visits >= budget {
                return (visits, false);
            }
            visits += 1;
            let (known, values) = self.topology_coverage(i);
            if known != self.coverage_known[i] {
                undo.coverage
                    .entry(i)
                    .or_insert_with(|| self.coverage_known[i].clone());
                self.coverage_known[i] = known;
                affected.insert(i);
            }
            for (bit, value) in values {
                if value != self.coverage_values[bit] {
                    undo.coverage_values
                        .entry(bit)
                        .or_insert(self.coverage_values[bit]);
                    self.coverage_values[bit] = value;
                    queue.extend(
                        self.consumers[bit]
                            .iter()
                            .filter(|&&j| j != i)
                            .map(|&j| (self.positions[j], j)),
                    );
                }
            }
        }
        (visits, true)
    }

    /// Fits appended nodes between existing ranks. Only incident edges are
    /// checked; a backward edge falls back to a full cycle check.
    fn update_topology_order(&mut self, undo: &mut TransactionUndo) -> Result<usize> {
        let old = undo.instances;
        let mut pending: BTreeSet<_> = (old..self.instance_count()).collect();
        let mut added = Vec::new();
        while !pending.is_empty() {
            let ready = pending.iter().copied().find(|&i| self.instances[i].pins.values().flatten()
                .all(|s| !matches!(s, BitSource::Bit(b) if self.drivers[*b].is_some_and(|d| d != i && pending.contains(&d)))));
            let Some(i) = ready else {
                break;
            };
            pending.remove(&i);
            added.push(i);
        }
        let mut upper = BTreeMap::new();
        for &i in added.iter().rev() {
            let bound = self.instances[i]
                .outputs
                .iter()
                .flat_map(|b| self.consumers[*b].iter())
                .filter(|&&j| j != i)
                .map(|&j| {
                    if j < old {
                        self.positions[j]
                    } else {
                        *upper.get(&j).unwrap_or(&usize::MAX)
                    }
                })
                .min()
                .unwrap_or(usize::MAX);
            upper.insert(i, bound);
        }
        let mut fits = pending.is_empty();
        for &i in &added {
            let lower = self.instances[i]
                .pins
                .iter()
                .filter(|(name, _)| {
                    self.lib
                        .pin(self.cells[i], name)
                        .is_some_and(|p| p.direction == PinDirection::Input as i32)
                })
                .flat_map(|(_, sources)| sources)
                .filter_map(|s| match s {
                    BitSource::Bit(b) => self.drivers[*b]
                        .filter(|&d| d != i)
                        .map(|d| self.positions[d]),
                    _ => None,
                })
                .max()
                .unwrap_or(0);
            self.positions[i] = lower.saturating_add(1);
            fits &= self.positions[i] < upper[&i];
        }
        fits &= undo.nets.keys().all(|&b| {
            self.drivers[b].is_none_or(|d| {
                self.consumers[b]
                    .iter()
                    .all(|&j| d == j || self.positions[d] < self.positions[j])
            })
        });
        if fits {
            return Ok(0);
        }
        let mut degree = vec![0; self.instance_count()];
        let mut successors = vec![BTreeSet::new(); self.instance_count()];
        for (bit, driver) in self.drivers.iter().enumerate() {
            if let Some(d) = *driver {
                for &j in &self.consumers[bit] {
                    if d != j && successors[d].insert(j) {
                        degree[j] += 1;
                    }
                }
            }
        }
        let mut ready: BTreeSet<_> = degree
            .iter()
            .enumerate()
            .filter(|(_, d)| **d == 0)
            .map(|(i, _)| i)
            .collect();
        let mut ranks = vec![0; self.instance_count()];
        let mut count: usize = 0;
        let rank_gap = RANK_GAP.min(usize::MAX / (self.instance_count() + 1));
        while let Some(i) = ready.pop_first() {
            count += 1;
            ranks[i] = count
                .checked_mul(rank_gap)
                .ok_or_else(|| anyhow!("rank overflow"))?;
            for &j in &successors[i] {
                degree[j] -= 1;
                if degree[j] == 0 {
                    ready.insert(j);
                }
            }
        }
        if count != self.instance_count() {
            return Err(UnsupportedTopology.into());
        }
        undo.ranks = Some(std::mem::replace(&mut self.positions, ranks));
        Ok(count)
    }

    /// Updates connectivity for a combinational batch, journaling only touched
    /// records. Timing, coverage and endpoint updates are coordinated
    /// afterward.
    fn prepare_topology(
        &mut self,
        edit: &TimingEdit,
        undo: &mut TransactionUndo,
        work: &mut TrialResult,
    ) -> Result<EditPlan> {
        ensure!(
            !self.registered,
            "topology transactions currently support combinational modules"
        );
        let old_bits = self.bit_count();
        let n = old_bits
            .checked_add(edit.new_bits)
            .ok_or_else(|| anyhow!("bit count overflow"))?;
        self.bit_names
            .extend((old_bits..n).map(|bit| format!("<timing-bit {bit}>")));
        self.bit_sources.extend((old_bits..n).map(BitSource::Bit));
        self.external_load_counts.resize(n, 0);
        self.endpoints_by_bit.resize_with(n, Vec::new);
        for timing in &mut self.timing {
            timing.resize(n, None);
        }
        self.capacitances.resize(n, EdgeLoadCapacitance::default());
        self.drivers.resize(n, None);
        self.loads.resize_with(n, Vec::new);
        self.consumers.resize_with(n, Vec::new);
        self.timing_constants.resize(n, None);
        self.coverage_values.resize(n, None);
        let mut changed = BTreeSet::new();
        for mutation in &edit.mutations {
            let (i, spec) = match mutation {
                TimingMutation::Insert(spec) => (self.instances.len(), spec.clone()),
                TimingMutation::Replace {
                    instance,
                    cell,
                    connections,
                } => {
                    let old = self
                        .instances
                        .get(*instance)
                        .ok_or_else(|| anyhow!("unknown instance"))?;
                    (
                        *instance,
                        TimingInstance {
                            name: old.name.clone(),
                            cell: *cell,
                            connections: connections.clone(),
                        },
                    )
                }
                TimingMutation::Resize { instance, cell } => {
                    self.validate_resize(*instance, *cell)?;
                    let old = self
                        .instances
                        .get(*instance)
                        .ok_or_else(|| anyhow!("unknown instance"))?;
                    ensure!(
                        old.pins.values().all(|v| v.len() <= 1),
                        "scalar instance bindings required"
                    );
                    (
                        *instance,
                        TimingInstance {
                            name: old.name.clone(),
                            cell: *cell,
                            connections: old
                                .pin_order
                                .iter()
                                .filter_map(|p| {
                                    old.pins[p].first().map(|&source| (p.clone(), source))
                                })
                                .collect(),
                        },
                    )
                }
                TimingMutation::Rewire {
                    instance,
                    pin,
                    source,
                } => {
                    let old = self
                        .instances
                        .get(*instance)
                        .ok_or_else(|| anyhow!("unknown instance"))?;
                    ensure!(
                        self.lib.pin(self.cells[*instance], pin).is_some(),
                        "unknown rewire pin"
                    );
                    ensure!(
                        old.pins.values().all(|v| v.len() <= 1),
                        "scalar instance bindings required"
                    );
                    (
                        *instance,
                        TimingInstance {
                            name: old.name.clone(),
                            cell: self.cells[*instance],
                            connections: old
                                .pin_order
                                .iter()
                                .filter_map(|p| {
                                    if p == pin {
                                        Some((p.clone(), *source))
                                    } else {
                                        old.pins[p].first().map(|&source| (p.clone(), source))
                                    }
                                })
                                .chain(
                                    (!old.pin_order.contains(pin)).then(|| (pin.clone(), *source)),
                                )
                                .collect(),
                        },
                    )
                }
            };
            let (instance, order) = self.topology_instance(&spec)?;
            if i == self.instances.len() {
                ensure!(
                    self.instance_names.insert(instance.name.clone()),
                    "duplicate instance name"
                );
                self.positions.push(0);
                self.coverage_known.push(HashMap::new());
                self.query_counts
                    .push(TimingQueryDiagnosticCounts::default());
                self.instances.push(instance);
                self.cells.push(spec.cell);
                self.orders.push(order);
                self.capture_endpoint.push(None);
                self.electrical.push(Vec::new());
            } else {
                if i < undo.instances {
                    undo.changed.entry(i).or_insert_with(|| {
                        (
                            self.instances[i].clone(),
                            self.cells[i],
                            self.orders[i].clone(),
                        )
                    });
                }
                self.instances[i] = instance;
                self.cells[i] = spec.cell;
                self.orders[i] = order;
            }
            changed.insert(i);
        }
        work.rebuilt_instances = changed.len();
        // Disconnect all old bindings before adding the batch's final
        // bindings.
        let old_bits: BTreeSet<_> = undo
            .changed
            .values()
            .flat_map(|(inst, _, _)| inst.pins.values())
            .flatten()
            .filter_map(|s| {
                if let BitSource::Bit(b) = s {
                    Some(*b)
                } else {
                    None
                }
            })
            .collect();
        for bit in old_bits {
            undo.net(self, bit);
        }
        for (&i, (inst, _, _)) in &undo.changed {
            for sources in inst.pins.values() {
                for source in sources {
                    if let BitSource::Bit(bit) = *source {
                        self.loads[bit].retain(|(j, _)| *j != i);
                        self.consumers[bit].retain(|&j| j != i);
                        if self.drivers[bit] == Some(i) {
                            self.drivers[bit] = None;
                            self.timing_constants[bit] = None;
                        }
                    }
                }
            }
        }
        for &i in &changed {
            let pins: Vec<_> = self.instances[i]
                .pin_order
                .iter()
                .map(|name| (name.clone(), self.instances[i].pins[name][0]))
                .collect();
            for (name, source) in pins {
                let direction = self
                    .lib
                    .pin(self.cells[i], &name)
                    .expect("validated pin")
                    .direction;
                if let BitSource::Bit(bit) = source {
                    undo.net(self, bit);
                    if direction == PinDirection::Input as i32 {
                        self.loads[bit].push((i, name));
                    } else if direction == PinDirection::Output as i32 {
                        ensure!(
                            !self.primary_inputs.contains(&bit),
                            "cell drives a primary input"
                        );
                        ensure!(self.drivers[bit].replace(i).is_none(), "multiple drivers");
                        self.instances[i].outputs.push(bit);
                        let pin = self.lib.pin(self.cells[i], &name).expect("validated pin");
                        if pin.timing_arcs.is_empty() {
                            self.timing_constants[bit] = self.lib.timing_constant(
                                self.cells[i],
                                self.lib
                                    .pin_index(self.cells[i], &name)
                                    .expect("validated pin"),
                            )?;
                        }
                    }
                }
            }
        }
        let mut known_changed = changed.clone();
        let mut coverage_changed = changed.clone();
        for (&bit, old) in &undo.nets {
            self.loads[bit].sort_by_key(|(i, p)| {
                (
                    *i,
                    self.instances[*i]
                        .pin_order
                        .iter()
                        .position(|n| n == p)
                        .expect("validated pin"),
                )
            });
            self.consumers[bit] = self.loads[bit].iter().map(|(i, _)| *i).collect();
            self.consumers[bit].dedup();
            ensure!(
                self.drivers[bit].is_some()
                    || self.primary_inputs.contains(&bit)
                    || (self.loads[bit].is_empty() && self.endpoints_by_bit[bit].is_empty()),
                "undriven bit {bit}"
            );
            if self.timing_constants[bit] != old.constant {
                known_changed.extend(self.consumers[bit].iter().copied());
            }
            if self.drivers[bit] != old.driver {
                coverage_changed.extend(self.consumers[bit].iter().copied());
                if let Some(i) = self.drivers[bit] {
                    changed.insert(i);
                } else {
                    undo.timing.bits.push((
                        bit,
                        [self.timing[0][bit].clone(), self.timing[1][bit].clone()],
                    ));
                    self.timing[0][bit] = None;
                    self.timing[1][bit] = None;
                    undo.coverage_values
                        .entry(bit)
                        .or_insert(self.coverage_values[bit]);
                    self.coverage_values[bit] = None;
                }
            }
            let load = self.compute_load(bit)?;
            if load != old.capacitance {
                if let Some(i) = self.drivers[bit] {
                    changed.insert(i);
                }
            }
            self.capacitances[bit] = load;
        }
        for i in known_changed {
            let known: HashMap<_, _> = self.instances[i]
                .pins
                .iter()
                .filter_map(|(name, sources)| {
                    let value = match sources.as_slice() {
                        [BitSource::Literal(v)] => Some(*v),
                        [BitSource::Bit(b)]
                            if self
                                .lib
                                .pin(self.cells[i], name)
                                .is_some_and(|p| p.direction == PinDirection::Input as i32) =>
                        {
                            self.timing_constants[*b]
                        }
                        _ => None,
                    };
                    value.map(|v| (name.clone(), v))
                })
                .collect();
            if self.instances[i].known != known {
                undo.known
                    .entry(i)
                    .or_insert_with(|| self.instances[i].known.clone());
                self.instances[i].known = known;
                changed.insert(i);
            }
        }
        work.reordered_instances = self.update_topology_order(undo)?;
        self.traversal_dirty |= self.instance_count() != undo.instances || undo.ranks.is_some();
        work.touched_nets = undo.nets.len();
        // Electrical fanout can change without a capacitance/timing change.
        // Invalidate touched drivers without needlessly scheduling retiming.
        let mut affected = changed.clone();
        for (&bit, old) in &undo.nets {
            affected.extend(old.driver);
            affected.extend(self.drivers[bit]);
        }
        let queue = changed
            .into_iter()
            .map(|i| (self.positions[i], i))
            .collect();
        Ok(EditPlan {
            queue,
            affected,
            endpoints: BTreeSet::new(),
            coverage: CoverageUpdate::Sparse(coverage_changed),
        })
    }

    /// Checks the fixed-pin Resize contract before either preparation path.
    fn validate_resize(&self, i: usize, cell: usize) -> Result<usize> {
        let inst = self
            .instances
            .get(i)
            .ok_or_else(|| anyhow!("unknown instance"))?;
        self.lib
            .library
            .cells
            .get(cell)
            .ok_or_else(|| anyhow!("unknown cell index"))?;
        let old_cell = self.cells[i];
        self.lib
            .validate_functional_semantics(cell)
            .map_err(anyhow::Error::msg)?;
        if !inst.sequential {
            self.lib.timing_related_input_pins(cell)?;
        }
        if self.registered {
            ensure!(
                inst.sequential == is_sequential_boundary_cell(&self.lib.library.cells[cell]),
                "substitution changes the register boundary"
            );
        }
        for name in &inst.pin_order {
            let old = self.lib.pin(old_cell, name).expect("validated pin");
            let new = self
                .lib
                .pin(cell, name)
                .ok_or_else(|| anyhow!("missing replacement pin {name}"))?;
            ensure!(
                old.direction == new.direction,
                "replacement pin direction differs"
            );
            if inst.sequential {
                ensure!(
                    old.is_clocking_pin == new.is_clocking_pin,
                    "replacement changes clock/data pin role"
                );
            }
        }
        ensure!(
            !self.changes_timing_constants(i, old_cell, cell)?,
            "replacement changes constants used by conditional timing arcs"
        );
        Ok(old_cell)
    }

    /// Prepares one fixed-connectivity substitution, including launch/capture
    /// register cells. The same journal and coordinator handle all edit kinds.
    fn prepare_resize(
        &mut self,
        i: usize,
        cell: usize,
        undo: &mut TransactionUndo,
        work: &mut TrialResult,
    ) -> Result<EditPlan> {
        let old_cell = self.validate_resize(i, cell)?;
        let constants_may_change = self.changes_coverage_constants(i, old_cell, cell);
        undo.changed.insert(
            i,
            (self.instances[i].clone(), old_cell, self.orders[i].clone()),
        );
        self.cells[i] = cell;
        self.orders[i] = if self.instances[i].sequential {
            self.lib.library.cells[cell]
                .pins
                .iter()
                .enumerate()
                .filter(|(_, p)| {
                    p.direction == PinDirection::Output as i32
                        && self.instances[i]
                            .pins
                            .contains_key(self.lib.library.resolve_string(&p.name))
                })
                .map(|(index, _)| index)
                .collect()
        } else {
            combinational_output_pin_evaluation_order(&self.lib, cell, &self.instances[i].pins)?
        };
        let changed_loads: BTreeSet<_> = self.instances[i]
            .pins
            .iter()
            .filter(|(name, _)| {
                self.lib.pin(cell, name).expect("validated pin").direction
                    == PinDirection::Input as i32
            })
            .flat_map(|(_, sources)| sources)
            .filter_map(|source| match source {
                BitSource::Bit(bit) => Some(*bit),
                _ => None,
            })
            .collect();
        let mut queue = BTreeSet::from([(self.positions[i], i)]);
        let mut affected = BTreeSet::from([i]);
        for bit in changed_loads {
            if let Some(driver) = self.drivers[bit] {
                affected.insert(driver);
            }
            let load = self.compute_load(bit)?;
            if load != self.capacitances[bit] {
                undo.net(self, bit);
                self.capacitances[bit] = load;
                if let Some(driver) = self.drivers[bit] {
                    queue.insert((self.positions[driver], driver));
                }
            }
        }
        work.rebuilt_instances = 1;
        work.touched_nets = undo.nets.len();
        let endpoints = self.capture_endpoint[i].into_iter().collect();
        Ok(EditPlan {
            queue,
            affected,
            endpoints,
            coverage: if constants_may_change {
                CoverageUpdate::Rescan
            } else {
                CoverageUpdate::Unchanged
            },
        })
    }

    /// Applies one atomic timing edit. True commits; false, callback errors,
    /// invalid edits and work limits restore the prior graph and timing.
    /// The budget counts timing and Boolean-coverage cell visits in this trial;
    /// connectivity and ordering work are reported separately. Limits are
    /// optional caller-supplied early-rejection bounds, not optimization
    /// policy. Module ports and existing aliases stay fixed. Sequential
    /// modules support one compatible Resize; combinational modules also
    /// support graph batches.
    pub fn trial_edit(
        &mut self,
        edit: &TimingEdit,
        options: TrialOptions,
        accept: impl FnOnce(&Self, &BTreeSet<usize>) -> Result<bool>,
    ) -> Result<TrialResult> {
        let mut undo = TransactionUndo::new(self);
        let start_work = self.propagated;
        let limit = start_work.saturating_add(options.cell_budget);
        let mut work = TrialResult::default();
        let result = (|| -> Result<TrialOutcome> {
            let plan = match edit.mutations.as_slice() {
                [TimingMutation::Resize { instance, cell }] if edit.new_bits == 0 => {
                    self.prepare_resize(*instance, *cell, &mut undo, &mut work)?
                }
                _ => self.prepare_topology(edit, &mut undo, &mut work)?,
            };
            let EditPlan {
                queue,
                mut affected,
                mut endpoints,
                coverage,
            } = plan;
            let before = self.propagated;
            let outcome = self.propagate_edits(
                queue,
                &mut undo.timing,
                &mut affected,
                &mut endpoints,
                limit,
                options.limits,
            )?;
            work.propagated_instances = self.propagated - before;
            if let Some(outcome) = outcome {
                return Ok(outcome);
            }
            let rescan = matches!(coverage, CoverageUpdate::Rescan);
            match coverage {
                CoverageUpdate::Unchanged => { /* Existing coverage is still valid. */ }
                CoverageUpdate::Rescan => {
                    // Register feedback and changed static Boolean semantics
                    // retain the existing bounded least-fixed-point fallback.
                    let (known, visited) =
                        self.coverage_constants(limit.saturating_sub(self.propagated));
                    self.propagated += visited;
                    work.coverage_instances = visited;
                    let Some(known) = known else {
                        return Ok(TrialOutcome::PropagationLimit);
                    };
                    for (i, next) in known.into_iter().enumerate() {
                        if self.coverage_known[i] != next {
                            undo.coverage
                                .insert(i, std::mem::replace(&mut self.coverage_known[i], next));
                            affected.insert(i);
                        }
                    }
                }
                CoverageUpdate::Sparse(changed) => {
                    let (visited, complete) = self.propagate_topology_coverage(
                        &changed,
                        &mut undo,
                        &mut affected,
                        limit.saturating_sub(self.propagated),
                    );
                    self.propagated += visited;
                    work.coverage_instances = visited;
                    if !complete {
                        return Ok(TrialOutcome::PropagationLimit);
                    }
                }
            }
            for &i in &affected {
                let inst = &self.instances[i];
                if inst.sequential {
                    continue;
                }
                for mode in 0..self.launch_class_count() {
                    validate_instance_functional_timing_coverage(
                        InstanceCoverageInputs {
                            lib: &self.lib,
                            cell_idx: self.cells[i],
                            pin_sources: &inst.pins,
                            known_pin_values: &inst.known,
                            coverage_known_pin_values: &self.coverage_known[i],
                            output_pin_order: &self.orders[i],
                            instance_name: &inst.name,
                            launched_timing: self
                                .registered
                                .then_some(self.timing[mode].as_slice()),
                        },
                        &mut self.evaluation,
                    )?;
                }
            }
            for endpoint in endpoints {
                let value = self.compute_endpoint(endpoint)?;
                if let Some(limits) = options.limits {
                    let capture = usize::from(matches!(
                        self.endpoints[endpoint],
                        TimingEndpoint::Capture(_)
                    ));
                    if value.iter().enumerate().any(|(mode, value)| {
                        value.is_some_and(|arrival| {
                            arrival
                                > limits.values()[mode * 2 + capture] + TIMING_COMPARISON_TOLERANCE
                        })
                    }) {
                        return Ok(TrialOutcome::Rejected);
                    }
                }
                undo.timing
                    .endpoints
                    .push((endpoint, self.endpoint_values[endpoint]));
                self.set_endpoint(endpoint, value);
            }
            if !accept(self, &affected)? {
                return Ok(TrialOutcome::Rejected);
            }
            if rescan {
                self.refresh_coverage_values();
            }
            for i in affected {
                self.electrical[i] = self.electrical_violations(i);
            }
            if self.traversal_dirty {
                self.topological = (0..self.instance_count()).collect();
                self.topological.sort_by_key(|&i| (self.positions[i], i));
                self.traversal_dirty = false;
            }
            Ok(TrialOutcome::Accepted)
        })();
        if !matches!(result, Ok(TrialOutcome::Accepted)) {
            undo.restore(self);
        }
        work.outcome = result?;
        Ok(work)
    }
}

#[cfg(test)]
mod tests;
