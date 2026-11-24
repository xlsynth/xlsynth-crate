// SPDX-License-Identifier: Apache-2.0

//! Repeat timing search and area recovery with fixed registers until the
//! netlist stops changing or the iteration limit is reached.

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, ensure};
use serde::Serialize;
use sha2::{Digest, Sha256};

use super::{
    RecoveryCellMenu, RecoveryWindowCheckpoint, ResizeOptions, ResizeReport, TimingCellMenu,
    resize_netlist_with_recovery_checkpoint,
};
use crate::liberty::Library;
use crate::liberty_model::Cell;
use crate::netlist::emit::emit_module_as_netlist_text;
use crate::netlist::io::{ParsedNetlist, read_gv_from_str, select_module};
use crate::netlist::mapped::MappedNetlist;
use crate::netlist::sta::StaOptions;

/// Checks each mapped master and removes conflicting shared masters from
/// the alternate sizing view before the search chooses replacements.
pub fn prepare_sizing_view(
    mapped: &ParsedNetlist,
    mapping: &Library,
    sizing: &mut Library,
) -> Result<()> {
    let module = select_module(mapped, None)?;
    let name = mapped
        .interner
        .resolve(module.name)
        .context("missing module name")?;
    crate::netlist::mapped::validate_module_name(name, sizing).map_err(anyhow::Error::msg)?;
    let mut cells = HashMap::with_capacity(sizing.cells.len());
    for cell in &sizing.cells {
        ensure!(
            cells.insert(cell.name.as_str(), cell).is_none(),
            "duplicate Liberty cell name '{}' in sizing library",
            cell.name
        );
    }
    let mut incompatible = HashSet::new();
    for cell in &mapping.cells {
        if cells
            .get(cell.name.as_str())
            .is_some_and(|other| !compatible_cell(cell, mapping, other, sizing))
        {
            incompatible.insert(cell.name.as_str());
        }
    }
    for instance in &module.instances {
        let name = mapped
            .interner
            .resolve(instance.type_name)
            .context("missing cell name")?;
        ensure!(
            cells.contains_key(name),
            "sizing library is missing mapped cell '{name}'"
        );
        ensure!(
            !incompatible.contains(name),
            "sizing library has incompatible behavior or pins for cell '{name}'"
        );
    }
    sizing
        .cells
        .retain(|cell| !incompatible.contains(cell.name.as_str()));
    Ok(())
}

/// Exact pin equations cover clock gates and large cells outside ordinary
/// sizing families. For different equations use the resizer's Boolean test.
fn compatible_cell(cell: &Cell, mapping: &Library, other: &Cell, sizing: &Library) -> bool {
    if cell.sequential != other.sequential || cell.clock_gate != other.clock_gate {
        return false;
    }
    if cell.unsupported_semantics == other.unsupported_semantics
        && pin_equations(cell, mapping) == pin_equations(other, sizing)
    {
        return true;
    }
    super::cell_signature(cell, mapping)
        .is_some_and(|signature| super::cell_signature(other, sizing) == Some(signature))
}

fn pin_equations<'a>(cell: &'a Cell, library: &'a Library) -> Vec<(&'a str, i32, bool, &'a str)> {
    let mut pins: Vec<_> = cell
        .pins
        .iter()
        .map(|pin| {
            (
                library.resolve_string(&pin.name),
                pin.direction,
                pin.is_clocking_pin,
                library.resolve_string(&pin.function),
            )
        })
        .collect();
    pins.sort_unstable();
    pins
}

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
        Self {
            timing: StaOptions {
                primary_input_transition: 0.01,
                module_output_load: 0.01,
                ..StaOptions::default()
            },
            timing_cell_menu: TimingCellMenu::DelayBandsFastestFirst,
            recovery_cell_menu: RecoveryCellMenu::ElectricalMinArea,
            recovery_cell_limit: 12,
            record_gate_hashes: false,
        }
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
    // Timing trials and recovery both keep the register masters fixed.
    let mixed_options = ResizeOptions {
        timing_cell_menu: options.timing_cell_menu,
        max_upsize_rounds: 10,
        max_downsize_rounds: 4,
        max_area_growth: 0.02,
        timing_evaluation_budget: Some(250_000),
        timing: options.timing,
        recovery_cell_menu: options.recovery_cell_menu,
        recovery_cell_limit: Some(options.recovery_cell_limit),
        resize_registers: false,
        ..ResizeOptions::default()
    };
    let mixed = run_epochs(
        &mut mapped_gv,
        module_name,
        library,
        &mixed_options,
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

fn run_epochs(
    gate: &mut String,
    module_name: Option<&str>,
    library: &Library,
    options: &ResizeOptions,
    record_gate_hashes: bool,
) -> Result<AdaptivePhaseReport> {
    const MAX_EPOCHS: usize = 46;
    let mut epochs = Vec::new();
    let mut epoch_gate_sha256 = record_gate_hashes.then(Vec::new);
    let mut stop = AdaptiveStopReason::EpochCap;
    for epoch in 1..=MAX_EPOCHS {
        let (next_gate, report) = edit_gate(gate, module_name, |mapped| {
            resize_netlist_with_recovery_checkpoint(
                &mut mapped.module,
                &mapped.nets,
                &mut mapped.interner,
                library,
                options,
                RecoveryWindowCheckpoint::Tightest,
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
