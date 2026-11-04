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
use crate::netlist::resize::{ResizeOptions, ResizeReport, resize_netlist};
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
        let fanout = self.fanout.as_ref().map(|options| {
            optimize_fanout(mapped, library, options.clone()).context(
                "area fanout optimization failed; provide usable timing data or use --fanout-opt=false for raw mapping",
            )
        }).transpose()?;
        let recovery = self
            .recovery
            .as_ref()
            .map(|options| {
                resize_netlist(
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
        let finish = |mut cover: AreaMappedNetlist, options: &Self| {
            let report = options.apply(&mut cover.netlist, library)?;
            Ok((cover, report))
        };
        let Some(local) = candidates.local else {
            return finish(candidates.demanded, self);
        };
        if self.fanout.is_none() && self.recovery.is_none() {
            return finish(candidates.demanded, self);
        }
        let local = finish(local, self);
        let mut remaining = self.clone();
        if let Some(options) = &mut remaining.recovery {
            // Explicit budgets bound the entire command. Preserve the full
            // allowance for the local cover and spend only unused visits on
            // the alternative. A failed run has no usable visit count.
            let spent = local
                .as_ref()
                .ok()
                .and_then(|(_, stages)| stages.recovery.as_ref());
            if let Some(limit) = &mut options.timing_evaluation_budget {
                *limit = spent.map_or(0, |r| limit.saturating_sub(r.upsize.propagated_instances));
            }
            if let Some(limit) = &mut options.recovery_evaluation_budget {
                *limit = spent.map_or(0, |r| limit.saturating_sub(r.downsize.propagated_instances));
            }
        }
        match (finish(candidates.demanded, &remaining), local) {
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
