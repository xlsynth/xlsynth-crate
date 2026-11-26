// SPDX-License-Identifier: Apache-2.0

//! Repeat timing search and area recovery with fixed registers until the
//! netlist stops changing or the iteration limit is reached.

use anyhow::{Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};

use super::{
    RecoveryCellMenu, RecoveryWindowCheckpoint, RecoveryWorkPolicy, ResizeOptions, ResizeReport,
    TimingCellMenu, resize_netlist,
};
use crate::liberty::Library;
use crate::netlist::emit::emit_module_as_netlist_text;
use crate::netlist::io::{read_gv_from_str, select_module};
use crate::netlist::mapped::MappedNetlist;
use crate::netlist::sta::StaOptions;

/// Constraints and cell shortlists shared by the adaptive sizing passes.
#[derive(Clone, Debug)]
pub struct AdaptiveResizeOptions {
    /// STA constraints for every pass, in the timing library's units.
    pub timing: StaOptions,
    /// Candidate ordering during timing search.
    pub timing_cell_menu: TimingCellMenu,
    /// Candidate ordering during area recovery.
    pub recovery_cell_menu: RecoveryCellMenu,
    /// Maximum shortlisted alternatives for each recovery site.
    pub recovery_cell_limit: usize,
    /// Record SHA-256 of the mapped input and each emitted gate checkpoint.
    pub record_gate_hashes: bool,
}

impl Default for AdaptiveResizeOptions {
    fn default() -> Self {
        RegisteredDelayPolicy::default_options()
    }
}

/// Final gate text and the electrical and search measurements for each pass.
#[derive(Debug)]
pub struct AdaptiveResizeResult {
    pub gate_text: String,
    pub report: AdaptiveResizeReport,
}

/// Sizing checkpoints. The area-only phase is reported as skipped.
#[derive(Debug, Serialize)]
pub struct AdaptiveResizeReport {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mapped_sha256: Option<String>,
    pub area: AdaptivePhaseReport,
    pub mixed: AdaptivePhaseReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub final_sha256: Option<String>,
}

/// Each epoch starts from the preceding epoch's gate output and receives fresh
/// work budgets.
#[derive(Debug, Serialize)]
pub struct AdaptivePhaseReport {
    pub epochs: Vec<ResizeReport>,
    pub stop: AdaptiveStopReason,
    /// Gate SHA-256 in the same order as `epochs`, when requested.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub epoch_gate_sha256: Option<Vec<String>>,
}

/// Reason a sequence of sizing epochs finished.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum AdaptiveStopReason {
    EpochCap,
    Skipped,
    IdenticalGate,
}

/// Finish one flat mapped module with bounded, topology-preserving sizing.
///
/// The initial GV is accepted as emitted: the byte comparison that detects
/// convergence includes the first pass's normalization of declarations. Each
/// subsequent epoch reads exactly the preceding netlist. If multiple modules
/// are present, only the named module is emitted.
pub fn finish_mapped_netlist(
    mut mapped_gv: String,
    module_name: Option<&str>,
    library: &Library,
    options: &AdaptiveResizeOptions,
) -> Result<AdaptiveResizeResult> {
    let hash = |gate: &str| {
        options
            .record_gate_hashes
            .then(|| format!("{:x}", Sha256::digest(gate)))
    };
    let mapped_sha256 = hash(&mapped_gv);
    let policy = RegisteredDelayPolicy::new(options);
    let mixed = run_epochs(
        &mut mapped_gv,
        module_name,
        library,
        &policy,
        options.record_gate_hashes,
    )?;
    let final_sha256 = hash(&mapped_gv);
    Ok(AdaptiveResizeResult {
        gate_text: mapped_gv,
        report: AdaptiveResizeReport {
            mapped_sha256,
            area: AdaptivePhaseReport {
                epochs: Vec::new(),
                stop: AdaptiveStopReason::Skipped,
                epoch_gate_sha256: options.record_gate_hashes.then(Vec::new),
            },
            mixed,
            final_sha256,
        },
    })
}

/// Registered delay mapping repeatedly searches timing and recovers area with
/// fixed register masters. The area and work allowances restart each epoch.
struct RegisteredDelayPolicy {
    max_epochs: usize,
    resize: ResizeOptions,
}

impl RegisteredDelayPolicy {
    /// Defaults exposed to callers before their explicit CLI or API overrides.
    fn default_options() -> AdaptiveResizeOptions {
        AdaptiveResizeOptions {
            timing: StaOptions {
                primary_input_transition: 0.01,
                module_output_load: 0.01,
                ..StaOptions::default()
            },
            timing_cell_menu: TimingCellMenu::DelayBandsFastestFirst,
            recovery_cell_menu: RecoveryCellMenu::CheapestGroupsThenFastest,
            recovery_cell_limit: 12,
            record_gate_hashes: false,
        }
    }

    fn new(options: &AdaptiveResizeOptions) -> Self {
        Self {
            max_epochs: 46,
            resize: ResizeOptions {
                timing_cell_menu: options.timing_cell_menu,
                max_cell_candidates: 8,
                max_timing_rounds: 10,
                max_recovery_rounds: 4,
                max_area_growth: 0.02,
                timing_evaluation_budget: Some(250_000),
                timing: options.timing,
                recovery_cell_menu: options.recovery_cell_menu,
                recovery_cell_limit: Some(options.recovery_cell_limit),
                recovery_work_policy: RecoveryWorkPolicy::Renewable {
                    checkpoint: RecoveryWindowCheckpoint::Tightest,
                },
                resize_registers: false,
                ..ResizeOptions::default()
            },
        }
    }
}

fn run_epochs(
    gate: &mut String,
    module_name: Option<&str>,
    library: &Library,
    policy: &RegisteredDelayPolicy,
    record_gate_hashes: bool,
) -> Result<AdaptivePhaseReport> {
    let mut epochs = Vec::new();
    let mut epoch_gate_sha256 = record_gate_hashes.then(Vec::new);
    let mut stop = AdaptiveStopReason::EpochCap;
    for epoch in 1..=policy.max_epochs {
        let (next_gate, report) = edit_gate(gate, module_name, |mapped| {
            resize_netlist(
                &mut mapped.module,
                &mapped.nets,
                &mut mapped.interner,
                library,
                &policy.resize,
            )
        })
        .with_context(|| format!("Adaptive sizing epoch {epoch}"))?;
        epochs.push(report);
        if let Some(hashes) = &mut epoch_gate_sha256 {
            hashes.push(format!("{:x}", Sha256::digest(&next_gate)));
        }
        if next_gate == *gate {
            stop = AdaptiveStopReason::IdenticalGate;
            break;
        }
        *gate = next_gate;
    }
    Ok(AdaptivePhaseReport {
        epochs,
        stop,
        epoch_gate_sha256,
    })
}

/// Reparse each checkpoint as standalone sizing does. Declarations and
/// instance order determine the symbol and net IDs used by timing search.
fn edit_gate<T>(
    gate: &str,
    module_name: Option<&str>,
    edit: impl FnOnce(&mut MappedNetlist) -> Result<T>,
) -> Result<(String, T)> {
    let parsed = read_gv_from_str(gate)?;
    let name = select_module(&parsed, module_name)?.name;
    let module = parsed
        .modules
        .into_iter()
        .find(|m| m.name == name)
        .expect("module was just selected from this netlist");
    let mut mapped = MappedNetlist {
        module,
        nets: parsed.nets,
        interner: parsed.interner,
    };
    let report = edit(&mut mapped)?;
    let output = emit_module_as_netlist_text(&mapped.module, &mapped.nets, &mapped.interner)?;
    Ok((output, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::netlist::resize::tests::library;

    // The input is compact Verilog containing only a fixed flop. Normalizing
    // its text changes the first checkpoint even though the circuit is stable.
    // The next identical output must end the phase.
    #[test]
    fn fixed_registers_converge_on_identical_emitted_bytes() -> Result<()> {
        let library = library()?;
        let input =
            "module top(input clk,input d,output q); DFF4 ff(.D(d),.CLK(clk),.Q(q)); endmodule";
        let output = finish_mapped_netlist(
            input.into(),
            Some("top"),
            &library,
            &AdaptiveResizeOptions {
                record_gate_hashes: true,
                ..Default::default()
            },
        )?;
        let expected = concat!(
            "module top(clk, d, q);\n",
            "  input clk;\n",
            "  input d;\n",
            "  output q;\n",
            "  DFF4 ff (.D(d), .CLK(clk), .Q(q));\n",
            "endmodule\n",
        );
        assert_eq!(output.gate_text, expected);
        let hash = format!("{:x}", Sha256::digest(expected));
        assert_eq!(
            output.report.mapped_sha256,
            Some(format!("{:x}", Sha256::digest(input)))
        );
        assert_eq!(output.report.area.epoch_gate_sha256, Some(vec![]));
        assert_eq!(
            output.report.mixed.epoch_gate_sha256,
            Some(vec![hash.clone(); 2])
        );
        assert_eq!(output.report.final_sha256, Some(hash));
        assert_eq!(output.report.area.epochs.len(), 0);
        assert_eq!(output.report.area.stop, AdaptiveStopReason::Skipped);
        assert_eq!(output.report.mixed.epochs.len(), 2);
        assert_eq!(output.report.mixed.stop, AdaptiveStopReason::IdenticalGate);
        Ok(())
    }
}
