// SPDX-License-Identifier: Apache-2.0

//! Exact-Liberty buffering and sizing for mapped combinational netlists.

use crate::liberty_model::Library;
use crate::netlist::buffer::{BufferOptions, BufferStats};
use crate::netlist::parse::{Net, NetlistModule};
use crate::netlist::report::{build_area_report, build_sta_report};
use crate::netlist::resize::{ResizeOptions, ResizeStats, resize_netlist};
use crate::netlist::sta::{StaOptions, analyze_combinational_max_arrival};
use crate::netlist::timing_buffer::{
    BufferTimingConstraints, consolidate_timing_aware_buffers, has_slow_shared_primary_output,
    insert_speculative_timing_aware_buffers, insert_timing_aware_buffers,
    refresh_timing_buffer_diagnostics,
};
use crate::netlist::timing_resize::recover_final_timing_protected_area;
use anyhow::Result;
use serde::Serialize;
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

/// A smaller exploratory tree often isolates critical sinks after sizing.
const COORDINATED_TIMING_MAX_FANOUT: usize = 6;
/// Avoid duplicating expensive complete-network sizing on very large designs.
const MAX_COORDINATED_INSTANCE_COUNT: usize = 4096;
/// Tiny mapped modules rarely repay a second complete Liberty timing graph.
const MIN_FINAL_AREA_RECOVERY_INSTANCE_COUNT: usize = 128;
/// High-buffer designs would require rebuilding an already extensive tree.
const MAX_COORDINATED_EXISTING_BUFFERS: usize = 256;
/// Bound isolated output-slew diagnosis to inexpensive small mapped designs.
const MAX_SHARED_OUTPUT_INSTANCE_COUNT: usize = 128;

/// Shared timing assumptions and optional mapped-netlist optimization passes.
#[derive(Clone, Debug, PartialEq)]
pub struct NetlistOptimizationOptions {
    /// Exact timing assumptions used before, during, and after optimization.
    pub sta_options: StaOptions,
    /// Exact-Liberty, criticality-aware buffering; `None` disables buffering.
    pub buffer_options: Option<BufferOptions>,
    /// Critical-path cell sizing and area recovery; `None` disables sizing.
    pub resize_options: Option<ResizeOptions>,
}

impl Default for NetlistOptimizationOptions {
    fn default() -> Self {
        Self {
            sta_options: StaOptions::default(),
            buffer_options: Some(BufferOptions::default()),
            resize_options: Some(ResizeOptions::default()),
        }
    }
}

/// Exact, independently recomputed timing and area for the complete flow.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct NetlistOptimizationStats {
    pub initial_area: f64,
    pub final_area: f64,
    pub initial_delay: f64,
    pub final_delay: f64,
    pub buffer_stats: Option<BufferStats>,
    pub resize_stats: Option<ResizeStats>,
}

/// Buffers a mapped module, sizes its critical paths, and verifies full STA.
pub fn optimize_mapped_netlist(
    module: &mut NetlistModule,
    nets: &mut Vec<Net>,
    interner: &mut StringInterner<StringBackend<SymbolU32>>,
    library: &Library,
    options: &NetlistOptimizationOptions,
) -> Result<NetlistOptimizationStats> {
    let initial_area = build_area_report(module, interner, library)?.area;
    let initial_delay = build_sta_report(
        module,
        nets.as_slice(),
        interner,
        library,
        options.sta_options,
    )?
    .delay;

    let mut buffer_stats = if let Some(configured) = &options.buffer_options {
        let mut buffer_options = configured.clone();
        buffer_options.module_output_load = options.sta_options.module_output_load;
        Some(insert_timing_aware_buffers(
            module,
            nets,
            interner,
            library,
            &buffer_options,
            options.sta_options,
            &BufferTimingConstraints::default(),
        )?)
    } else {
        None
    };
    let mut resize_stats = if let Some(configured) = &options.resize_options {
        let mut resize_options = configured.clone();
        resize_options.sta_options = options.sta_options;
        Some(resize_netlist(
            module,
            nets.as_slice(),
            interner,
            library,
            &resize_options,
        )?)
    } else {
        None
    };

    if let (Some(buffer_options), Some(resize_options), Some(previous_sizing)) = (
        options.buffer_options.as_ref(),
        options.resize_options.as_ref(),
        resize_stats.as_ref(),
    ) && (previous_sizing.upsizes > 0
        || previous_sizing.downsizes > 0
        || previous_sizing.pin_swaps > 0)
        && module.instances.len() <= MAX_COORDINATED_INSTANCE_COUNT
        && buffer_stats
            .as_ref()
            .is_some_and(|stats| stats.buffers_inserted <= MAX_COORDINATED_EXISTING_BUFFERS)
        && {
            let stats = buffer_stats
                .as_ref()
                .expect("buffer diagnostics are present");
            if stats.max_fanout_after > COORDINATED_TIMING_MAX_FANOUT
                || stats.unresolved_overloaded_nets > 0
            {
                true
            } else if module.instances.len() <= MAX_SHARED_OUTPUT_INSTANCE_COUNT
                && stats.max_fanout_after >= 2
                && options.sta_options.module_output_load > 0.0
            {
                let mut detection_options = buffer_options.clone();
                detection_options.module_output_load = options.sta_options.module_output_load;
                has_slow_shared_primary_output(
                    module,
                    nets.as_slice(),
                    interner,
                    library,
                    &detection_options,
                    options.sta_options,
                )?
            } else {
                false
            }
        }
    {
        let previous_delay = previous_sizing.final_delay;
        let mut trial_module = module.clone();
        let mut trial_nets = nets.clone();
        let mut trial_interner = interner.clone();
        let mut exploratory_buffer_options = buffer_options.clone();
        exploratory_buffer_options.module_output_load = options.sta_options.module_output_load;
        exploratory_buffer_options.max_fanout = exploratory_buffer_options
            .max_fanout
            .min(COORDINATED_TIMING_MAX_FANOUT);
        let trial_buffer_stats = insert_speculative_timing_aware_buffers(
            &mut trial_module,
            &mut trial_nets,
            &mut trial_interner,
            library,
            &exploratory_buffer_options,
            options.sta_options,
            &BufferTimingConstraints::default(),
        )?;

        if trial_buffer_stats.buffers_inserted > 0 {
            let mut exploratory_resize_options = resize_options.clone();
            exploratory_resize_options.sta_options = options.sta_options;
            let trial_resize_stats = resize_netlist(
                &mut trial_module,
                trial_nets.as_slice(),
                &mut trial_interner,
                library,
                &exploratory_resize_options,
            )?;
            let candidate_delay = build_sta_report(
                &trial_module,
                trial_nets.as_slice(),
                &trial_interner,
                library,
                options.sta_options,
            )?
            .delay;

            if candidate_delay + exploratory_resize_options.improvement_epsilon < previous_delay {
                *module = trial_module;
                *nets = trial_nets;
                *interner = trial_interner;
                if let Some(stats) = buffer_stats.as_mut() {
                    merge_buffer_stats(stats, trial_buffer_stats);
                }
                if let Some(stats) = resize_stats.as_mut() {
                    merge_resize_stats(stats, trial_resize_stats);
                }
            }
        }
    }

    if let (Some(configured), Some(sizing)) =
        (options.buffer_options.as_ref(), resize_stats.as_ref())
        && module.instances.len() <= MAX_COORDINATED_INSTANCE_COUNT
        && buffer_stats.as_ref().is_some_and(|stats| {
            (2..=MAX_COORDINATED_EXISTING_BUFFERS).contains(&stats.buffers_inserted)
        })
    {
        let mut recovery_options = configured.clone();
        recovery_options.module_output_load = options.sta_options.module_output_load;
        let recovered = consolidate_timing_aware_buffers(
            module,
            nets.as_slice(),
            interner,
            library,
            &recovery_options,
            options.sta_options,
            sizing.final_delay,
        )?;
        if recovered.buffers_removed > 0 {
            if let Some(stats) = buffer_stats.as_mut() {
                stats.buffers_inserted = stats
                    .buffers_inserted
                    .saturating_sub(recovered.buffers_removed);
                stats.area_added = (stats.area_added - recovered.area_recovered).max(0.0);
                stats.max_fanout_after = recovered.max_fanout_after;
                stats.max_load_after = recovered.max_load_after;
                stats.unresolved_overloaded_nets = recovered.unresolved_overloaded_nets;
                stats.final_worst_delay = Some(recovered.final_delay);
                stats.timing_evaluations += recovered.timing_evaluations;
            }
            if let Some(stats) = resize_stats.as_mut() {
                stats.final_delay = recovered.final_delay;
                stats.final_area = build_area_report(module, interner, library)?.area;
            }
        }
    }

    if let Some(configured) = &options.resize_options
        && configured.max_area_iterations > 0
        && (MIN_FINAL_AREA_RECOVERY_INSTANCE_COUNT..=MAX_COORDINATED_INSTANCE_COUNT)
            .contains(&module.instances.len())
    {
        let mut recovery_options = configured.clone();
        recovery_options.sta_options = options.sta_options;
        let recovered = recover_final_timing_protected_area(
            module,
            nets.as_slice(),
            interner,
            library,
            &recovery_options,
            &BufferTimingConstraints::default(),
        )?;
        if let Some(stats) = resize_stats.as_mut() {
            merge_resize_stats(stats, recovered);
        }
    }

    let final_area = build_area_report(module, interner, library)?.area;
    let final_timing = analyze_combinational_max_arrival(
        module,
        nets.as_slice(),
        interner,
        library,
        options.sta_options,
    )?;
    let final_delay = final_timing.worst_output_arrival;
    if let (Some(stats), Some(configured)) =
        (buffer_stats.as_mut(), options.buffer_options.as_ref())
    {
        let mut final_options = configured.clone();
        final_options.module_output_load = options.sta_options.module_output_load;
        refresh_timing_buffer_diagnostics(
            module,
            nets.as_slice(),
            interner,
            library,
            &final_options,
            &final_timing,
            stats,
        )?;
    }
    Ok(NetlistOptimizationStats {
        initial_area,
        final_area,
        initial_delay,
        final_delay,
        buffer_stats,
        resize_stats,
    })
}

/// Combines diagnostics from accepted electrically distinct buffer rounds.
fn merge_buffer_stats(initial: &mut BufferStats, subsequent: BufferStats) {
    initial.buffered_nets += subsequent.buffered_nets;
    initial.buffers_inserted += subsequent.buffers_inserted;
    initial.area_added += subsequent.area_added;
    initial.max_fanout_after = subsequent.max_fanout_after;
    initial.max_load_after = subsequent.max_load_after;
    initial.unresolved_overloaded_nets = subsequent.unresolved_overloaded_nets;
    initial.final_worst_delay = subsequent.final_worst_delay;
    initial.timing_evaluations += subsequent.timing_evaluations;
    initial.rejected_timing_batches += subsequent.rejected_timing_batches;
}

/// Preserves complete move accounting across coordinated sizing rounds.
fn merge_resize_stats(initial: &mut ResizeStats, subsequent: ResizeStats) {
    initial.final_delay = subsequent.final_delay;
    initial.final_area = subsequent.final_area;
    initial.outer_iterations += subsequent.outer_iterations;
    initial.evaluations += subsequent.evaluations;
    initial.pin_swap_evaluations += subsequent.pin_swap_evaluations;
    initial.failed_evaluations += subsequent.failed_evaluations;
    initial.recomputed_instances += subsequent.recomputed_instances;
    initial.upsizes += subsequent.upsizes;
    initial.downsizes += subsequent.downsizes;
    initial.register_upsizes += subsequent.register_upsizes;
    initial.register_downsizes += subsequent.register_downsizes;
    initial.pin_swaps += subsequent.pin_swaps;
    initial.final_clock_load = subsequent.final_clock_load;
    initial.replacements.extend(subsequent.replacements);
    initial.pin_swap_steps.extend(subsequent.pin_swap_steps);
}

#[cfg(test)]
mod tests {
    use super::{NetlistOptimizationOptions, optimize_mapped_netlist};
    use crate::liberty_model::PinDirection;
    use crate::netlist::buffer::BufferOptions;
    use crate::netlist::cell_catalog::test_utils::{parse_module, sizing_library};
    use crate::netlist::sta::StaOptions;
    use crate::netlist::timing_buffer::tests::{
        slow_shared_output_library, slow_shared_output_source,
    };

    #[test]
    fn buffers_then_resizes_using_consistent_exact_timing() {
        let source = r#"
module top(a, y0, y1, y2);
  input a;
  output y0, y1, y2;
  wire root;
  BUF driver (.A(a), .Y(root));
  BUF use0 (.A(root), .Y(y0));
  BUF use1 (.A(root), .Y(y1));
  BUF use2 (.A(root), .Y(y2));
endmodule
"#;
        let library = sizing_library();
        let (mut module, mut nets, mut interner) = parse_module(source);
        let stats = optimize_mapped_netlist(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &NetlistOptimizationOptions {
                buffer_options: Some(BufferOptions {
                    max_fanout: 2,
                    ..BufferOptions::default()
                }),
                ..NetlistOptimizationOptions::default()
            },
        )
        .unwrap();

        let buffer_stats = stats
            .buffer_stats
            .expect("timing-aware buffering should report complete diagnostics");
        assert!(buffer_stats.buffers_inserted > 0);
        assert!(buffer_stats.timing_evaluations >= 2);
        assert!(buffer_stats.initial_worst_delay.is_some_and(f64::is_finite));
        let resize_stats = stats
            .resize_stats
            .expect("buffering should be followed by incremental sizing");
        assert!(
            buffer_stats
                .final_worst_delay
                .is_some_and(|delay| (delay - stats.final_delay).abs() < 1e-9)
        );
        assert!(resize_stats.upsizes > 0);
        assert!(stats.final_delay < stats.initial_delay);
    }

    #[test]
    fn disabled_passes_preserve_exact_baseline_metrics() {
        let library = sizing_library();
        let (mut module, mut nets, mut interner) = parse_module(
            r#"
module top(a, y);
  input a;
  output y;
  BUF driver (.A(a), .Y(y));
endmodule
"#,
        );
        let stats = optimize_mapped_netlist(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &NetlistOptimizationOptions {
                buffer_options: None,
                resize_options: None,
                ..NetlistOptimizationOptions::default()
            },
        )
        .unwrap();

        assert_eq!(stats.initial_area, stats.final_area);
        assert_eq!(stats.initial_delay, stats.final_delay);
        assert!(stats.buffer_stats.is_none());
        assert!(stats.resize_stats.is_none());
    }

    #[test]
    fn refreshes_buffer_electrical_diagnostics_after_cell_resizing() {
        let mut library = sizing_library();
        for (name, transition_limit) in [("BUF", 0.05), ("BUF_FAST", 0.2)] {
            let cell = library
                .cells
                .iter()
                .position(|cell| cell.name == name)
                .expect("find electrically characterized buffer variant");
            let output = library.cells[cell]
                .pins
                .iter()
                .position(|pin| pin.direction == PinDirection::Output as i32)
                .expect("find characterized output pin");
            library.cells[cell].pins[output].max_transition = Some(transition_limit);
        }
        let (mut module, mut nets, mut interner) = parse_module(
            r#"
module top(a, y);
  input a;
  output y;
  BUF driver (.A(a), .Y(y));
endmodule
"#,
        );

        let stats = optimize_mapped_netlist(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &NetlistOptimizationOptions::default(),
        )
        .expect("refresh final diagnostics after repairing a real slew violation");
        let buffering = stats
            .buffer_stats
            .expect("retain final buffer electrical diagnostics");

        assert_eq!(
            interner.resolve(module.instances[0].type_name),
            Some("BUF_FAST")
        );
        assert_eq!(buffering.unresolved_overloaded_nets, 0);
        assert_eq!(buffering.final_worst_delay, Some(stats.final_delay));
    }

    #[test]
    fn rejects_speculative_rebuffering_that_cannot_improve_exact_delay() {
        let source = r#"
module top(a, out);
  input a;
  output [6:0] out;
  wire root;
  BUF driver (.A(a), .Y(root));
  BUF sink0 (.A(root), .Y(out[0]));
  BUF sink1 (.A(root), .Y(out[1]));
  BUF sink2 (.A(root), .Y(out[2]));
  BUF sink3 (.A(root), .Y(out[3]));
  BUF sink4 (.A(root), .Y(out[4]));
  BUF sink5 (.A(root), .Y(out[5]));
  BUF sink6 (.A(root), .Y(out[6]));
endmodule
"#;
        let library = sizing_library();
        let (mut module, mut nets, mut interner) = parse_module(source);
        let original_instance_count = module.instances.len();

        let stats = optimize_mapped_netlist(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &NetlistOptimizationOptions::default(),
        )
        .expect("roll back a tighter buffer tree when it cannot improve exact timing");

        assert_eq!(module.instances.len(), original_instance_count);
        assert_eq!(stats.buffer_stats.unwrap().buffers_inserted, 0);
        assert!(stats.resize_stats.unwrap().upsizes > 0);
        assert!(stats.final_delay < stats.initial_delay);
    }

    #[test]
    fn isolates_slow_shared_output_only_after_strict_exact_timing_improvement() {
        let library = slow_shared_output_library();
        let (mut module, mut nets, mut interner) = parse_module(slow_shared_output_source());

        let stats = optimize_mapped_netlist(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &NetlistOptimizationOptions {
                sta_options: StaOptions {
                    module_output_load: 0.6,
                    ..StaOptions::default()
                },
                ..NetlistOptimizationOptions::default()
            },
        )
        .expect("strictly improve full-network timing by isolating a shared output");

        let buffering = stats
            .buffer_stats
            .expect("retain output-buffer diagnostics");
        let sizing = stats
            .resize_stats
            .expect("retain coordinated sizing diagnostics");
        assert_eq!(buffering.max_fanout_before, 2);
        assert_eq!(buffering.buffers_inserted, 1);
        assert!(sizing.upsizes > 0);
        assert!(stats.final_delay < stats.initial_delay);
    }

    #[test]
    fn optimizes_logic_while_preserving_constant_output_assignments() {
        let source = r#"
module top(a, y0, y1, y2, zero);
  input a;
  output y0, y1, y2, zero;
  wire root;
  assign zero = 1'b0;
  BUF driver (.A(a), .Y(root));
  BUF use0 (.A(root), .Y(y0));
  BUF use1 (.A(root), .Y(y1));
  BUF use2 (.A(root), .Y(y2));
endmodule
"#;
        let library = sizing_library();
        let (mut module, mut nets, mut interner) = parse_module(source);
        let stats = optimize_mapped_netlist(
            &mut module,
            &mut nets,
            &mut interner,
            &library,
            &NetlistOptimizationOptions {
                buffer_options: Some(BufferOptions {
                    max_fanout: 2,
                    ..BufferOptions::default()
                }),
                ..NetlistOptimizationOptions::default()
            },
        )
        .expect("literal output tie-offs should survive complete netlist optimization");

        assert_eq!(module.assigns.len(), 1);
        assert!(stats.buffer_stats.unwrap().buffers_inserted > 0);
        assert!(stats.resize_stats.unwrap().upsizes > 0);
        assert!(stats.final_delay < stats.initial_delay);
    }
}
