// SPDX-License-Identifier: Apache-2.0

//! Thin trial adapter around shared STA; rich diagnostics are computed on
//! acceptance.

use super::edit::EditPlan;
use super::*;
pub(super) use crate::netlist::sta::incremental::UnsupportedTopology;
use crate::netlist::sta::incremental::{IncrementalSta, TrialOptions};

/// Work counters exclude the full-STA validation oracle.
#[derive(Clone, Debug, Default, Serialize)]
pub struct Work {
    pub nodes: usize,
    pub load_nets: usize,
    pub metadata_instances: usize,
    pub coverage_instances: usize,
    pub reordered_instances: usize,
    pub transaction_seconds: f64,
}

/// Shared timing session with deterministic area accounting.
pub(super) struct Engine<'a> {
    library: &'a Library,
    sta: IncrementalSta<'a>,
    cell_counts: Vec<usize>,
    lexical_cells: Vec<usize>,
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

    /// Reuses the validated baseline and prepared library without redoing STA.
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
                wire_load: options.wire_load,
            },
            sta,
        )?;
        let mut lexical_cells = (0..library.cells.len()).collect::<Vec<_>>();
        lexical_cells.sort_by_key(|i| &library.cells[*i].name);
        let mut cell_counts = vec![0; library.cells.len()];
        for &cell in timing.cell_indices() {
            cell_counts[cell] += 1;
        }
        let engine = Self {
            library,
            sta: timing,
            cell_counts,
            lexical_cells,
        };
        engine.check(sta)?;
        Ok(engine)
    }

    /// Sums candidate area in the same cell-name order as full reports.
    pub fn candidate_area(&self, edit: &EditPlan) -> f64 {
        let counts = edit.counts(&self.cell_counts, self.sta.cell_indices());
        self.lexical_cells
            .iter()
            .filter(|&&c| counts[c] > 0)
            .map(|&c| self.library.cells[c].area * counts[c] as f64)
            .sum()
    }

    /// Evaluates only the search objective; shared STA owns all trial rollback.
    pub fn trial<F>(
        &mut self,
        edit: &EditPlan,
        graph: &Graph,
        area: f64,
        oracle: F,
    ) -> Result<(FanoutTrialMetrics, Work)>
    where
        F: FnOnce(&View<'_, '_>, &FanoutTrialMetrics) -> Result<()>,
    {
        let batch = edit.timing_edit(graph, &self.sta)?;
        let mut metrics = None;
        let mut reached_callback = false;
        let mut oracle_seconds = 0.0;
        let start = Instant::now();
        let result = self
            .sta
            .trial_edit(&batch, TrialOptions::default(), |sta, _| {
                reached_callback = true;
                let value = FanoutTrialMetrics {
                    area,
                    cell_count: sta.instance_count(),
                    worst_delay: sta.score().input_to_output,
                };
                let oracle_start = Instant::now();
                let checked = oracle(&View { sta }, &value);
                oracle_seconds = oracle_start.elapsed().as_secs_f64();
                checked?;
                metrics = Some(value);
                Ok(false)
            });
        let transaction_seconds = start.elapsed().as_secs_f64() - oracle_seconds;
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
            },
        ))
    }

    /// Commits the same sparse plan and updates only area bookkeeping.
    pub fn commit(&mut self, edit: &EditPlan, graph: &Graph) -> Result<()> {
        let batch = edit.timing_edit(graph, &self.sta)?;
        let counts = edit.counts(&self.cell_counts, self.sta.cell_indices());
        let result = self
            .sta
            .trial_edit(&batch, TrialOptions::default(), |_, _| Ok(true))?;
        ensure!(
            result.outcome == crate::netlist::sta::incremental::TrialOutcome::Accepted,
            "unlimited checked fanout commit was rejected"
        );
        self.cell_counts = counts;
        Ok(())
    }

    /// Materializes timing only at accepted search boundaries.
    pub fn report(&self) -> StaReport {
        let sta = &self.sta;
        StaReport {
            net_timing: (0..sta.bit_count())
                .map(|bit| sta.timing_for_bit(bit))
                .collect(),
            worst_output_arrival: sta.score().input_to_output,
            worst_output_breakdown: None,
            worst_register_input_arrival: 0.0,
            worst_register_input_breakdown: None,
            register_input_arrivals: vec![None; sta.instance_count()],
            register_input_breakdowns: vec![None; sta.instance_count()],
            timing_query_diagnostic_counts: sta
                .combinational_query_diagnostics()
                .expect("combinational optimizer"),
            cell_levels: 0,
        }
    }

    pub fn check_loads(&self, graph: &Graph) -> Result<()> {
        View { sta: &self.sta }.check_loads(graph)
    }
    pub fn check(&self, full: &StaReport) -> Result<()> {
        View { sta: &self.sta }.check(full)
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
