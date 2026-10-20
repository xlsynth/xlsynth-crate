// SPDX-License-Identifier: Apache-2.0

//! Fanout policy metadata and an adapter to the public shared timing engine.

use super::*;
use crate::netlist::normalized::BitSource;
pub(super) use crate::netlist::sta::incremental::UnsupportedTopology;
use crate::netlist::sta::incremental::{
    IncrementalSta, TimingEdit, TimingInstance, TimingMutation, TrialOptions,
};

mod metrics;
use metrics::{Contribution, MetricsCache};

/// Scalar policy connection; the generic timing API receives normalized bits.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Source {
    Net(usize),
    Literal(bool),
    Empty,
}

/// Compact instance, preserving original connection order for exact load sums.
#[derive(Clone)]
struct Node {
    cell: usize,
    connections: Vec<(usize, Source)>,
}
/// Cached electrical table bounds and pin constraints.
struct Bounds {
    load: Vec<f64>,
    slew: Vec<f64>,
}
/// Sparse replacements and append-only topology additions for a trial.
#[derive(Clone)]
pub(super) struct Edit {
    changed: BTreeMap<usize, Node>,
    added: Vec<Node>,
    new_nets: usize,
}
/// Work counters exclude the full-STA validation oracle.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Work {
    pub nodes: usize,
    pub load_nets: usize,
    pub metadata_instances: usize,
    pub coverage_instances: usize,
    pub reordered_instances: usize,
    pub transaction_seconds: f64,
    pub metrics_seconds: f64,
}
/// Fanout policy metadata; the standalone STA owns all mutable graph state.
pub(super) struct Engine<'a> {
    library: &'a Library,
    sta: Option<IncrementalSta<'a>>,
    bounds: Vec<Bounds>,
    cell_counts: Vec<usize>,
    lexical_cells: Vec<usize>,
    metrics: MetricsCache,
}
impl<'a> Engine<'a> {
    #[cfg(test)]
    pub fn new(
        netlist: &MappedNetlist,
        library: &'a Library,
        options: &FanoutOptimizationOptions,
        sta: &StaReport,
    ) -> Result<Self> {
        Self::new_prepared(
            netlist,
            library,
            options,
            sta,
            std::sync::Arc::new(crate::netlist::sta::PreparedStaLibrary::new(library)?),
        )
    }

    /// Creates policy indexes around the standalone timing graph.
    pub fn new_prepared(
        netlist: &MappedNetlist,
        library: &'a Library,
        options: &FanoutOptimizationOptions,
        sta: &StaReport,
        prepared: std::sync::Arc<crate::netlist::sta::PreparedStaLibrary<'a>>,
    ) -> Result<Self> {
        let timing = IncrementalSta::from_validated_combinational(
            &netlist.module,
            &netlist.nets,
            &netlist.interner,
            prepared,
            StaOptions {
                primary_input_transition: options.primary_input_transition,
                module_output_load: options.module_output_load,
            },
            sta,
        )?;
        let bounds = library
            .cells
            .iter()
            .map(|c| Bounds {
                load: c
                    .pins
                    .iter()
                    .map(|p| table_bound(library, p, None, true))
                    .collect(),
                slew: c
                    .pins
                    .iter()
                    .map(|p| {
                        c.pins
                            .iter()
                            .filter(|o| o.direction == PinDirection::Output as i32)
                            .map(|o| {
                                table_bound(
                                    library,
                                    o,
                                    Some(library.resolve_string(&p.name)),
                                    false,
                                )
                            })
                            .fold(f64::INFINITY, f64::min)
                    })
                    .collect(),
            })
            .collect();

        let mut lexical_cells = (0..library.cells.len()).collect::<Vec<_>>();
        lexical_cells.sort_by_key(|i| &library.cells[*i].name);
        let mut cell_counts = vec![0; library.cells.len()];
        for &cell in timing.cell_indices() {
            cell_counts[cell] += 1;
        }
        let mut engine = Self {
            library,
            sta: Some(timing),
            bounds,
            cell_counts,
            lexical_cells,
            metrics: MetricsCache::default(),
        };
        for i in 0..engine
            .sta
            .as_ref()
            .expect("timing session")
            .instance_count()
        {
            let contribution =
                engine.contribution_for(engine.sta.as_ref().expect("timing session"), i)?;
            engine.metrics.set(i, contribution);
        }
        engine.check(sta)?;
        Ok(engine)
    }

    /// Copies bindings only for an instance touched by a candidate.
    fn node(&self, i: usize) -> Node {
        let sta = self.sta.as_ref().expect("timing session");
        let inst = sta.instance(i).expect("candidate instance");
        let connections = inst
            .pin_order
            .iter()
            .map(|name| {
                let pin = sta.pin_index(i, name).expect("validated pin");
                let source = match inst.pins[name].first() {
                    Some(BitSource::Bit(bit)) => Source::Net(*bit),
                    Some(BitSource::Literal(value)) => Source::Literal(*value),
                    Some(BitSource::Unknown) | None => Source::Empty,
                };
                (pin, source)
            })
            .collect();
        Node {
            cell: inst.cell,
            connections,
        }
    }

    /// Converts a policy edit into the generic append/replace batch.
    fn timing_edit(&self, edit: &Edit) -> TimingEdit {
        let connections = |node: &Node| {
            node.connections
                .iter()
                .filter(|(_, source)| *source != Source::Empty)
                .map(|&(pin, source)| {
                    let source = match source {
                        Source::Net(n) => BitSource::Bit(n),
                        Source::Literal(v) => BitSource::Literal(v),
                        Source::Empty => BitSource::Unknown,
                    };
                    (
                        self.library
                            .resolve_string(&self.library.cells[node.cell].pins[pin].name)
                            .to_string(),
                        source,
                    )
                })
                .collect()
        };
        let mut mutations: Vec<_> = edit
            .changed
            .iter()
            .map(|(&instance, node)| TimingMutation::Replace {
                instance,
                cell: node.cell,
                connections: connections(node),
            })
            .collect();
        for (j, node) in edit.added.iter().enumerate() {
            let mut name = format!(
                "__timing_fanout_{}",
                self.sta.as_ref().expect("timing session").instance_count() + j
            );
            while self
                .sta
                .as_ref()
                .expect("timing session")
                .contains_instance_name(&name)
            {
                name.push('_');
            }
            mutations.push(TimingMutation::Insert(TimingInstance {
                name,
                cell: node.cell,
                connections: connections(node),
            }));
        }
        TimingEdit {
            new_bits: edit.new_nets,
            mutations,
        }
    }

    /// Forms a sparse candidate without cloning the module, names or library.
    pub fn edit(&self, d: &Driver, kind: &FanoutEdit, choices: &[CellChoice]) -> Result<Edit> {
        let mut edit = Edit {
            changed: BTreeMap::new(),
            added: Vec::new(),
            new_nets: 0,
        };
        let original = self.node(d.instance);
        let output_pin = self
            .sta
            .as_ref()
            .expect("timing session")
            .output_pin_index(d.instance, d.net)
            .ok_or_else(|| anyhow!("missing driver pin"))?;
        let output_connection = original
            .connections
            .iter()
            .position(|(pin, _)| *pin == output_pin)
            .ok_or_else(|| anyhow!("missing driver binding"))?;
        match kind {
            FanoutEdit::Upsize { cell } => {
                let c = choices
                    .iter()
                    .find(|c| &c.name == cell)
                    .ok_or_else(|| anyhow!("replacement missing"))?;
                let mut node = original.clone();
                node.cell = c.index;
                for (j, (p, _)) in node.connections.iter_mut().enumerate() {
                    let name = if j == output_connection {
                        c.output.as_str()
                    } else {
                        self.library
                            .resolve_string(&self.library.cells[original.cell].pins[*p].name)
                    };
                    *p = self.library.cells[c.index]
                        .pins
                        .iter()
                        .position(|p| self.library.resolve_string(&p.name) == name)
                        .ok_or_else(|| anyhow!("replacement pin missing"))?;
                }
                edit.changed.insert(d.instance, node);
            }
            FanoutEdit::Replicate { groups } => {
                for part in partition(&d.sinks, *groups, d.fixed_output_load)
                    .iter()
                    .skip(1)
                {
                    if part.is_empty() {
                        continue;
                    }
                    let n = self.sta.as_ref().expect("timing session").bit_count() + edit.new_nets;
                    edit.new_nets += 1;
                    let mut node = original.clone();
                    node.connections[output_connection].1 = Source::Net(n);
                    edit.added.push(node);
                    self.rewire(&mut edit, part, n);
                }
            }
            FanoutEdit::Buffer {
                cell,
                groups,
                levels,
            } => {
                let c = choices
                    .iter()
                    .find(|c| &c.name == cell)
                    .ok_or_else(|| anyhow!("buffer missing"))?;
                let pin_index = |name: &str| {
                    self.library.cells[c.index]
                        .pins
                        .iter()
                        .position(|p| self.library.resolve_string(&p.name) == name)
                        .unwrap()
                };
                let a = pin_index(&c.inputs[0]);
                let y = pin_index(&c.output);
                let mut parents = vec![d.net; *groups];
                if *levels == 2 {
                    for parity in 0..2 {
                        let n =
                            self.sta.as_ref().expect("timing session").bit_count() + edit.new_nets;
                        edit.new_nets += 1;
                        edit.added.push(Node {
                            cell: c.index,
                            connections: vec![(a, Source::Net(d.net)), (y, Source::Net(n))],
                        });
                        for j in (parity..*groups).step_by(2) {
                            parents[j] = n;
                        }
                    }
                }
                for (j, part) in partition(&d.sinks, *groups, 0.0).iter().enumerate() {
                    if part.is_empty() {
                        continue;
                    }
                    let n = self.sta.as_ref().expect("timing session").bit_count() + edit.new_nets;
                    edit.new_nets += 1;
                    edit.added.push(Node {
                        cell: c.index,
                        connections: vec![(a, Source::Net(parents[j])), (y, Source::Net(n))],
                    });
                    self.rewire(&mut edit, part, n);
                }
            }
        }
        Ok(edit)
    }
    /// Applies replacements and additions to the small per-library cell
    /// histogram.
    fn candidate_counts(&self, edit: &Edit) -> Vec<usize> {
        let mut counts = self.cell_counts.clone();
        for (&i, node) in &edit.changed {
            counts[self.sta.as_ref().expect("timing session").cell_indices()[i]] -= 1;
            counts[node.cell] += 1;
        }
        for node in &edit.added {
            counts[node.cell] += 1;
        }
        counts
    }

    /// Sums candidate cell counts in the same lexical order as area reports.
    pub fn candidate_area(&self, edit: &Edit) -> f64 {
        self.area(&self.candidate_counts(edit))
    }
    /// Copies only changed sink instances into the sparse edit.
    fn rewire(&self, e: &mut Edit, part: &[Sink], n: usize) {
        for sink in part {
            let node = e
                .changed
                .entry(sink.instance)
                .or_insert_with(|| self.node(sink.instance));
            let binding = node
                .connections
                .iter_mut()
                .find(|(pin, _)| *pin == sink.pin)
                .expect("validated sink binding");
            binding.1 = Source::Net(n);
        }
    }

    /// Recomputes reporting values only for one affected instance's outputs.
    fn contribution_for(&self, sta: &IncrementalSta<'_>, i: usize) -> Result<Option<Contribution>> {
        let inst = sta.instance(i).expect("affected instance");
        if inst.outputs.is_empty() {
            return Ok(None);
        }
        let mut value = Contribution::default();
        for &n in inst.outputs {
            let cell = sta.cell_indices()[i];
            let p = sta.output_pin_index(i, n).expect("driven output pin");
            let pin = &self.library.cells[cell].pins[p];
            let timing = sta
                .timing_for_bit(n)
                .ok_or_else(|| anyhow!("missing cached net"))?;
            let slew = timing.rise.transition.max(timing.fall.transition);
            let loads = sta
                .load_for_bit(n)
                .ok_or_else(|| anyhow!("missing cached load"))?;
            let load = loads[0].max(loads[1]);
            let sinks = sta.loads_for_bit(n);
            value.slew = value.slew.max(slew);
            value.load = value.load.max(load);
            value.fanout = value.fanout.max(sinks.len());
            value.counts[0] += usize::from(load > self.bounds[cell].load[p]);
            value.counts[2] += usize::from(pin.max_capacitance.is_some_and(|b| load > b));
            value.counts[3] += usize::from(pin.max_transition.is_some_and(|b| slew > b));
            for (sink, name) in sinks {
                let cell = sta.cell_indices()[*sink];
                let pin = sta.pin_index(*sink, name).expect("validated sink pin");
                value.counts[1] += usize::from(slew > self.bounds[cell].slew[pin]);
            }
        }
        Ok(Some(value))
    }

    /// Uses the core's complete invalidation set, including electrical-only
    /// upstream changes, so this client needs no connectivity journal.
    fn metric_changes(
        &self,
        sta: &IncrementalSta<'_>,
        affected: &BTreeSet<usize>,
    ) -> Result<Vec<(usize, Option<Contribution>)>> {
        affected
            .iter()
            .map(|&i| Ok((i, self.contribution_for(sta, i)?)))
            .collect()
    }

    /// Sums cell counts in the same cell-name order as full reports.
    fn area(&self, counts: &[usize]) -> f64 {
        self.lexical_cells
            .iter()
            .filter(|&&c| counts[c] > 0)
            .map(|&c| self.library.cells[c].area * counts[c] as f64)
            .sum()
    }

    #[cfg(test)]
    pub fn metrics(&self) -> Result<FanoutMetrics> {
        Ok(self.metrics.report(
            self.sta.as_ref().expect("timing session"),
            self.area(&self.cell_counts),
        ))
    }

    /// Trials the core transaction and invokes an optional independent oracle.
    pub fn trial<F>(&mut self, edit: Edit, area: f64, oracle: F) -> Result<(FanoutMetrics, Work)>
    where
        F: FnOnce(&View<'_, '_>, &FanoutMetrics) -> Result<()>,
    {
        let batch = self.timing_edit(&edit);
        let mut sta = self.sta.take().expect("timing session");
        let mut metrics = None;
        let mut metrics_seconds = 0.0;
        let mut reached_callback = false;
        let mut oracle_seconds = 0.0;
        let start = Instant::now();
        let result = sta.trial_edit(&batch, TrialOptions::default(), |sta, affected| {
            reached_callback = true;
            let start = Instant::now();
            let changes = self.metric_changes(sta, affected)?;
            let value = self.metrics.preview(&changes, sta, area);
            metrics_seconds = start.elapsed().as_secs_f64();
            let oracle_start = Instant::now();
            let checked = oracle(&View { sta }, &value);
            oracle_seconds = oracle_start.elapsed().as_secs_f64();
            checked?;
            metrics = Some(value);
            Ok(false)
        });
        let transaction_seconds = start.elapsed().as_secs_f64() - oracle_seconds;
        self.sta = Some(sta);
        let work = result.map_err(|error| {
            if reached_callback {
                error
            } else {
                CandidateTimingError(error).into()
            }
        })?;
        Ok((
            metrics.ok_or_else(|| anyhow!("missing trial metrics"))?,
            Work {
                nodes: work.propagated_instances,
                load_nets: work.touched_nets,
                metadata_instances: work.rebuilt_instances,
                coverage_instances: work.coverage_instances,
                reordered_instances: work.reordered_instances,
                transaction_seconds,
                metrics_seconds,
            },
        ))
    }

    /// Commits the core edit, then updates policy area and metric aggregates.
    pub fn commit(&mut self, edit: Edit) -> Result<()> {
        let batch = self.timing_edit(&edit);
        let counts = self.candidate_counts(&edit);
        let mut sta = self.sta.take().expect("timing session");
        let mut changes = None;
        let result = sta.trial_edit(&batch, TrialOptions::default(), |sta, affected| {
            changes = Some(self.metric_changes(sta, affected)?);
            Ok(true)
        });
        self.sta = Some(sta);
        ensure!(
            result?.outcome == crate::netlist::sta::incremental::TrialOutcome::Accepted,
            "unlimited checked fanout commit was rejected"
        );
        for (i, next) in changes.expect("accepted metric changes") {
            self.metrics.set(i, next);
        }
        self.cell_counts = counts;
        Ok(())
    }

    /// Materializes timing only at accepted search boundaries.
    pub fn report(&self) -> Result<StaReport> {
        let sta = self.sta.as_ref().expect("timing session");
        let metrics = self.metrics.report(sta, self.area(&self.cell_counts));
        Ok(StaReport {
            net_timing: (0..sta.bit_count())
                .map(|bit| sta.timing_for_bit(bit))
                .collect(),
            worst_output_arrival: metrics.worst_delay,
            worst_output_breakdown: None,
            worst_register_input_arrival: 0.0,
            worst_register_input_breakdown: None,
            register_input_arrivals: vec![None; sta.instance_count()],
            register_input_breakdowns: vec![None; sta.instance_count()],
            timing_query_diagnostic_counts: metrics.timing_queries,
            cell_levels: 0,
        })
    }

    pub fn check_loads(&self, graph: &Graph) -> Result<()> {
        View {
            sta: self.sta.as_ref().expect("timing session"),
        }
        .check_loads(graph)
    }
    pub fn check(&self, full: &StaReport) -> Result<()> {
        View {
            sta: self.sta.as_ref().expect("timing session"),
        }
        .check(full)
    }
}

/// Borrowed trial state gives the oracle access without a second timing engine.
pub(super) struct View<'s, 'lib> {
    sta: &'s IncrementalSta<'lib>,
}
impl View<'_, '_> {
    pub fn check_loads(&self, graph: &Graph) -> Result<()> {
        for d in &graph.drivers {
            let actual = self
                .sta
                .load_for_bit(d.net)
                .ok_or_else(|| anyhow!("missing load"))?;
            ensure!(
                actual.iter().zip(d.load).all(|(a, b)| (a - b).abs() < 1e-9),
                "oracle load at net {}",
                d.net
            );
        }
        Ok(())
    }
    pub fn check(&self, full: &StaReport) -> Result<()> {
        ensure!(
            full.net_timing.len() == self.sta.bit_count(),
            "oracle net count"
        );
        for (n, expected) in full.net_timing.iter().enumerate() {
            let actual = self.sta.timing_for_bit(n);
            match (actual, expected) {
                (Some(a), Some(b)) => ensure!(
                    [
                        a.rise.arrival - b.rise.arrival,
                        a.fall.arrival - b.fall.arrival,
                        a.rise.transition - b.rise.transition,
                        a.fall.transition - b.fall.transition,
                    ]
                    .iter()
                    .all(|d| d.abs() < 1e-8),
                    "oracle timing at net {n}: {a:?} vs {b:?}"
                ),
                (None, None) => (),
                _ => return Err(anyhow!("oracle timing presence at net {n}")),
            }
        }
        ensure!(
            (self.sta.score().input_to_output - full.worst_output_arrival).abs() < 1e-8,
            "oracle endpoint"
        );
        ensure!(
            self.sta.combinational_query_diagnostics() == Some(full.timing_query_diagnostic_counts),
            "oracle query diagnostics"
        );
        Ok(())
    }
}
