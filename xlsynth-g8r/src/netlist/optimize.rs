// SPDX-License-Identifier: Apache-2.0

//! Exact-Liberty buffering and sizing for mapped combinational netlists.

use crate::liberty_model::Library;
use crate::netlist::buffer::{BufferOptions, BufferStats, insert_buffers};
use crate::netlist::parse::{Net, NetlistModule};
use crate::netlist::report::{build_area_report, build_sta_report};
use crate::netlist::resize::{ResizeOptions, ResizeStats, resize_netlist};
use crate::netlist::sta::StaOptions;
use anyhow::Result;
use serde::Serialize;
use string_interner::symbol::SymbolU32;
use string_interner::{StringInterner, backend::StringBackend};

/// Shared timing assumptions and optional mapped-netlist optimization passes.
#[derive(Clone, Debug, PartialEq)]
pub struct NetlistOptimizationOptions {
    /// Exact timing assumptions used before, during, and after optimization.
    pub sta_options: StaOptions,
    /// Balanced Liberty-buffer insertion; `None` disables buffering.
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

    let buffer_stats = if let Some(configured) = &options.buffer_options {
        let mut buffer_options = configured.clone();
        buffer_options.module_output_load = options.sta_options.module_output_load;
        Some(insert_buffers(
            module,
            nets,
            interner,
            library,
            &buffer_options,
        )?)
    } else {
        None
    };
    let resize_stats = if let Some(configured) = &options.resize_options {
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

    let final_area = build_area_report(module, interner, library)?.area;
    let final_delay = build_sta_report(
        module,
        nets.as_slice(),
        interner,
        library,
        options.sta_options,
    )?
    .delay;
    Ok(NetlistOptimizationStats {
        initial_area,
        final_area,
        initial_delay,
        final_delay,
        buffer_stats,
        resize_stats,
    })
}

#[cfg(test)]
mod tests {
    use super::{NetlistOptimizationOptions, optimize_mapped_netlist};
    use crate::netlist::buffer::BufferOptions;
    use crate::netlist::cell_catalog::test_utils::{parse_module, sizing_library};

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

        assert!(stats.buffer_stats.unwrap().buffers_inserted > 0);
        assert!(stats.resize_stats.unwrap().upsizes > 0);
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
