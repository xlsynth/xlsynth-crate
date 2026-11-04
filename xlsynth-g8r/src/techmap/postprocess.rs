// SPDX-License-Identifier: Apache-2.0

//! Apply configured timing stages to combinational covers and compare their
//! final results.

use super::area::AreaMappedNetlist;
use super::delay::DelayMappedCandidates;
use crate::liberty_model::Library;
use crate::netlist::fanout_opt::{
    FanoutOptimizationOptions, FanoutOptimizationReport, optimize_fanout,
};
use crate::netlist::mapped::MappedNetlist;
use crate::netlist::resize::{ResizeOptions, ResizeReport, resize_netlist_with_remaining_budget};
use anyhow::{Context, Result};

/// Electrical optimization stages requested after covering, in execution order.
#[derive(Clone, Default)]
pub struct PostMappingOptions {
    pub fanout: Option<FanoutOptimizationOptions>,
    pub recovery: Option<ResizeOptions>,
}

/// Measurements for the emitted mapping, including the final stage's timing and
/// area.
#[derive(Default)]
pub struct PostMappingReport {
    pub fanout: Option<FanoutOptimizationReport>,
    pub recovery: Option<ResizeReport>,
}

impl PostMappingReport {
    /// The last enabled stage measures the entire combinational netlist.
    fn delay_and_area(&self) -> Option<(f64, f64)> {
        if let Some(r) = &self.recovery {
            Some((r.final_delay.input_to_output, r.final_area))
        } else {
            self.fanout
                .as_ref()
                .map(|r| (r.after.worst_delay, r.after.area))
        }
    }
}

impl PostMappingOptions {
    /// Runs fanout optimization followed by area recovery, when enabled.
    pub fn apply(
        &self,
        mapped: &mut MappedNetlist,
        library: &Library,
    ) -> Result<PostMappingReport> {
        self.clone().apply_with_remaining_budget(mapped, library)
    }

    /// Runs the stages and deducts completed resizing visits from the explicit
    /// budgets shared with subsequent covers.
    fn apply_with_remaining_budget(
        &mut self,
        mapped: &mut MappedNetlist,
        library: &Library,
    ) -> Result<PostMappingReport> {
        let fanout = self.fanout.as_ref().map(|options| {
            optimize_fanout(mapped, library, options.clone()).context(
                "area fanout optimization failed; provide usable timing data or use --fanout-opt=false for raw mapping",
            )
        }).transpose()?;
        let recovery = self
            .recovery
            .as_mut()
            .map(|options| {
                resize_netlist_with_remaining_budget(
                    &mut mapped.module,
                    &mapped.nets,
                    &mut mapped.interner,
                    library,
                    options,
                )
                .context("mapped area recovery failed")
            })
            .transpose()?;
        Ok(PostMappingReport { fanout, recovery })
    }

    /// Selects the faster finished delay cover, breaking measured delay ties by
    /// area. Either cover can still succeed if the other's cells lack
    /// timing coverage. The returned netlist includes the enabled stages.
    /// Its cover statistics describe the mapping before those stages. The
    /// accompanying report gives their measurements.
    pub fn select_delay_cover(
        &self,
        candidates: DelayMappedCandidates,
        library: &Library,
    ) -> Result<(AreaMappedNetlist, PostMappingReport)> {
        let mut remaining = self.clone();
        let mut finish = |mut cover: AreaMappedNetlist| {
            let report = remaining.apply_with_remaining_budget(&mut cover.netlist, library)?;
            Ok((cover, report))
        };
        let Some(local) = candidates.local else {
            return finish(candidates.demanded);
        };
        if self.fanout.is_none() && self.recovery.is_none() {
            return finish(candidates.demanded);
        }
        let local = finish(local);
        match (finish(candidates.demanded), local) {
            (Ok(demanded), Ok(local)) => {
                if local.1.delay_and_area() < demanded.1.delay_and_area() {
                    Ok(local)
                } else {
                    Ok(demanded)
                }
            }
            (Ok(cover), Err(_)) | (Err(_), Ok(cover)) => Ok(cover),
            (Err(d), Err(l)) => Err(d.context(format!("local-polarity cover also failed: {l:#}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use crate::liberty_model::LibraryBuilder;
    use crate::techmap::delay::{
        DelayMappingLibrary, DelayMappingOptions, map_gatefn_to_delay_candidates,
        tests::timing_library,
    };

    #[test]
    fn initial_timing_failure_preserves_recovery_budget_for_the_other_cover() -> Result<()> {
        let mut builder = LibraryBuilder::from_library(timing_library(&[
            ("INV", "!A", 1.0, &[("A", 1.0)]),
            (
                "OAI21",
                "!((A+C)*B)",
                2.0,
                &[("A", 2.0), ("B", 2.0), ("C", 2.0)],
            ),
            ("AND2", "A*B", 2.0, &[("A", 2.5), ("B", 2.5)]),
            ("AND2_SMALL", "A*B", 1.0, &[("A", 2.5), ("B", 2.5)]),
        ])?);
        let when = builder.intern_string("A*C")?;
        let cells = &mut builder.library_mut().cells;
        for arc in &mut cells[1].pins.last_mut().unwrap().timing_arcs {
            arc.when = when;
        }
        cells[3].use_for_size_only = Some(true);
        let library = builder.finish();

        let mut b = GateBuilder::new("shared".into(), GateBuilderOptions::no_opt());
        let x = *b.add_input("x".into(), 1).get_lsb(0);
        let y = *b.add_input("y".into(), 1).get_lsb(0);
        let z = *b.add_input("z".into(), 1).get_lsb(0);
        let zero = b.add_and_binary(z, z.negate());
        let x_or_zero = b.add_or_binary(x, zero);
        let shared = b.add_and_binary(x_or_zero, y);
        b.add_output("first".into(), shared.into());
        b.add_output("second".into(), shared.into());

        let mapping_library = DelayMappingLibrary::new(
            &library,
            DelayMappingOptions {
                reference_transition: 1.0,
                reference_load: 1.0,
            },
        )?;
        let aig = b.build();
        for fanout in [None, Some(FanoutOptimizationOptions::default())] {
            let candidates = map_gatefn_to_delay_candidates(&aig, &mapping_library)?;
            let options = PostMappingOptions {
                fanout,
                recovery: Some(ResizeOptions {
                    max_upsize_rounds: 0,
                    recovery_evaluation_budget: Some(1),
                    ..Default::default()
                }),
            };
            // The local OAI21 + INV ties an OR input to zero, disabling
            // its timing arcs. Initial STA fails in the first enabled stage.
            let mut local = candidates.local.as_ref().unwrap().netlist.clone();
            assert!(options.apply(&mut local, &library).is_err());
            let (_, report) = options.select_delay_cover(candidates, &library)?;
            let recovery = report.recovery.unwrap();
            assert_eq!(recovery.work_limits.recovery_evaluations, 1);
            assert_eq!(recovery.downsize.propagated_instances, 1);
            assert_eq!((recovery.initial_area, recovery.final_area), (2.0, 1.0));
        }
        Ok(())
    }
}
