// SPDX-License-Identifier: Apache-2.0

//! Apply configured timing stages to combinational covers and compare their
//! final results.

use super::area::{AreaMappedNetlist, AreaMappingLibrary, map_gatefn_to_area_netlist};
use super::delay::{
    DelayMappedCandidates, DelayMappingLibrary, DelayMappingOptions,
    map_gatefn_to_delay_candidates, map_gatefn_to_delay_netlist,
};
use crate::aig::gate::GateFn;
use crate::liberty_model::Library;
use crate::netlist::fanout_opt::{
    FanoutOptimizationOptions, FanoutOptimizationReport, optimize_fanout,
};
use crate::netlist::mapped::MappedNetlist;
use crate::netlist::resize::{
    RecoveryWindowCheckpoint, ResizeOptions, ResizeReport, resize_netlist_with_remaining_budget,
};
use anyhow::{Context, Result};

/// Electrical optimization stages requested after covering, in execution order.
#[derive(Clone, Default)]
pub struct PostMappingOptions {
    pub fanout: Option<FanoutOptimizationOptions>,
    pub recovery: Option<ResizeOptions>,
}

/// Objective for covering the prepared combinational transition graph.
#[derive(Clone, Copy, Debug)]
pub enum CoverStrategy {
    Area,
    Delay(DelayMappingOptions),
}

/// Immutable physical cell bindings for consecutive mappings against one
/// Liberty snapshot. Only the last delay reference point is retained. Cut
/// enumeration, six-input match caches, cover selection and sizing stay local
/// to each graph.
pub struct NativeCoverLibrary<'a> {
    library: &'a Library,
    delay: Option<([u64; 2], DelayMappingLibrary)>,
}

impl<'a> NativeCoverLibrary<'a> {
    /// Bind this cache to the exact library that owns all physical cell
    /// indices.
    pub fn new(library: &'a Library) -> Self {
        Self {
            library,
            delay: None,
        }
    }

    /// Return the immutable snapshot; its lifetime is independent of cache
    /// access.
    pub fn library(&self) -> &'a Library {
        self.library
    }

    /// Compute library cell matches when first requested at this reference slew
    /// and load.
    pub(crate) fn delay_index(
        &mut self,
        point: DelayMappingOptions,
    ) -> Result<&DelayMappingLibrary> {
        // The index prices arcs at a specific slew and load. Reusing it at a
        // different reference point would silently change cover selection.
        let key = [
            point.reference_transition.to_bits(),
            point.reference_load.to_bits(),
        ];
        if self.delay.as_ref().map(|(key, _)| *key) != Some(key) {
            self.delay = Some((key, DelayMappingLibrary::new(self.library, point)?));
        }
        Ok(&self.delay.as_ref().expect("successfully priced bindings").1)
    }
}

/// Covers a graph and applies electrical optimization before attaching
/// sequential cells.
pub fn map_and_optimize_cover(
    graph: &GateFn,
    library: &Library,
    strategy: CoverStrategy,
    post: &PostMappingOptions,
) -> Result<(AreaMappedNetlist, PostMappingReport)> {
    map_with_native_cover(graph, &mut NativeCoverLibrary::new(library), strategy, post)
}

/// Map one graph while retaining only its library-dependent cover bindings.
pub(crate) fn map_with_native_cover(
    graph: &GateFn,
    prepared: &mut NativeCoverLibrary,
    strategy: CoverStrategy,
    post: &PostMappingOptions,
) -> Result<(AreaMappedNetlist, PostMappingReport)> {
    let library = prepared.library();
    match strategy {
        CoverStrategy::Area => {
            let mut cover = map_gatefn_to_area_netlist(graph, &AreaMappingLibrary::new(library)?)?;
            let report = post.apply(&mut cover.netlist, library)?;
            Ok((cover, report))
        }
        CoverStrategy::Delay(options) => {
            let cells = prepared.delay_index(options)?;
            if post.fanout.is_some() || post.recovery.is_some() {
                post.select_delay_cover(map_gatefn_to_delay_candidates(graph, cells)?, library)
            } else {
                Ok((
                    map_gatefn_to_delay_netlist(graph, cells)?,
                    PostMappingReport::default(),
                ))
            }
        }
    }
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
                "fanout optimization failed; provide usable timing data or disable fanout optimization with --fanout-opt=false",
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
                    RecoveryWindowCheckpoint::Latest,
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

    use crate::liberty::parser::parse_liberty_files;
    use crate::netlist::emit::emit_netlist_as_text;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// The two equal-area AND gates exchange timing rank when either coordinate
    /// increases. Identical rise/fall arcs remove polarity and pin-order
    /// effects.
    fn crossing_delay_library() -> Result<Library> {
        let mut text = String::from(
            r#"library(crossing) {
        lu_table_template(grid) { variable_1 : input_net_transition;
            variable_2 : total_output_net_capacitance;
            index_1("0.01,0.04"); index_2("1,4"); }"#,
        );
        for (name, rows) in [
            ("AND_FLAT", r#""3,3","3,3""#),
            ("AND_SLOPED", r#""1,5","5,9""#),
        ] {
            text.push_str(&format!(
                r#"cell({name}) {{ area : 1;
            pin(A) {{ direction : input; }} pin(B) {{ direction : input; }}
            pin(Y) {{ direction : output; function : "A & B";"#
            ));
            for pin in ["A", "B"] {
                text.push_str(&format!(
                    r#"timing() {{ related_pin : "{pin}";
                timing_type : combinational; timing_sense : positive_unate;
                cell_rise(grid) {{ values({rows}); }} cell_fall(grid) {{ values({rows}); }}
                rise_transition(grid) {{ values("1,1","1,1"); }}
                fall_transition(grid) {{ values("1,1","1,1"); }} }}"#
                ));
            }
            text.push_str("} }");
        }
        text.push('}');
        let mut file = NamedTempFile::new()?;
        file.write_all(text.as_bytes())?;
        parse_liberty_files(&[file.path()]).map_err(anyhow::Error::msg)
    }

    // Change slew and load independently, then return to the first point.
    // Every result must match a fresh index at the same reference point.
    #[test]
    fn switching_each_reference_coordinate_and_returning_matches_a_fresh_cover() -> Result<()> {
        let library = crossing_delay_library()?;
        let mut warm = NativeCoverLibrary::new(&library);
        let mut graph = GateBuilder::new("and".into(), GateBuilderOptions::no_opt());
        let a = *graph.add_input("a".into(), 1).get_lsb(0);
        let b = *graph.add_input("b".into(), 1).get_lsb(0);
        let and = graph.add_and_binary(a, b);
        graph.add_output("out".into(), and.into());
        let graph = graph.build();
        let post = PostMappingOptions::default();
        for (transition, load, winner) in [
            (0.01, 1.0, "AND_SLOPED"),
            (0.01, 4.0, "AND_FLAT"),
            (0.01, 1.0, "AND_SLOPED"),
            (0.04, 1.0, "AND_FLAT"),
            (0.04, 1.0, "AND_FLAT"),
            (0.01, 1.0, "AND_SLOPED"),
        ] {
            let strategy = CoverStrategy::Delay(DelayMappingOptions {
                reference_transition: transition,
                reference_load: load,
            });
            let fresh = map_and_optimize_cover(&graph, &library, strategy, &post)?.0;
            let cached = map_with_native_cover(&graph, &mut warm, strategy, &post)?.0;
            assert_eq!(fresh.stats.cell_count, 1);
            assert_eq!(fresh.stats.cells_by_type.get(winner), Some(&1));
            assert_eq!(
                emit_netlist_as_text(&cached.netlist.into())?,
                emit_netlist_as_text(&fresh.netlist.into())?,
            );
        }
        let invalid = DelayMappingOptions {
            reference_transition: 0.01,
            reference_load: f64::NAN,
        };
        let cold_error = DelayMappingLibrary::new(&library, invalid).unwrap_err();
        let warm_error = warm.delay_index(invalid).unwrap_err();
        assert_eq!(format!("{warm_error:#}"), format!("{cold_error:#}"));
        let valid = DelayMappingOptions {
            reference_transition: 0.01,
            reference_load: 1.0,
        };
        let after_error =
            map_with_native_cover(&graph, &mut warm, CoverStrategy::Delay(valid), &post)?.0;
        assert_eq!(after_error.stats.cells_by_type.get("AND_SLOPED"), Some(&1));
        Ok(())
    }
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
