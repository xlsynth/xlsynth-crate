// SPDX-License-Identifier: Apache-2.0

//! Finish a mapped netlist with fixed register masters. Recover area first,
//! then alternate timing and area work while registered delay improves. A
//! characterized library may also recover leakage at the final timing and area.

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result, ensure};
use num_bigint::BigUint;
use serde::Serialize;
use sha2::{Digest, Sha256};

use super::{
    LeakageRecoveryReport, RecoveryCellLeakage, RecoveryCellMenu, RecoveryWindowCheckpoint,
    ResizeOptions, ResizeReport, SizingLibrary, TimingCellMenu, recover_leakage_netlist,
    resize_with_recovery_cost,
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
    /// Candidate ordering during area and leakage recovery.
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
            recovery_cell_limit: 8,
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

/// The terminal report is present exactly when leakage characterization was
/// supplied.
#[derive(Debug, Serialize)]
pub struct AdaptiveResizeReport {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mapped_sha256: Option<String>,
    pub area: AdaptivePhaseReport,
    pub mixed: AdaptivePhaseReport,
    pub terminal_leakage: Option<LeakageRecoveryReport>,
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
    IdenticalGate,
    NoPositiveRegisteredDelay,
    InsufficientLateDelayGain,
}

/// Finish one flat mapped module with bounded, topology-preserving sizing.
///
/// The initial GV is accepted as emitted: the byte comparison that detects
/// convergence includes the first pass's normalization of declarations. Each
/// subsequent epoch reads exactly the preceding netlist. If multiple modules
/// are present, only the named module is emitted. Pass `None` for leakage
/// when no characterized per-master values exist for this library.
pub fn finish_mapped_netlist(
    mut mapped_gv: String,
    module_name: Option<&str>,
    library: &Library,
    options: &AdaptiveResizeOptions,
    leakage: Option<&RecoveryCellLeakage>,
) -> Result<AdaptiveResizeResult> {
    let hash = |gate: &str| {
        options
            .record_gate_hashes
            .then(|| format!("{:x}", Sha256::digest(gate)))
    };
    let mapped_sha256 = hash(&mapped_gv);
    // Start by recovering slack left by mapping. This frees area before the
    // timing phase spends its growth allowance on hard paths. Both phases
    // keep the physical registers fixed.
    let area_options = ResizeOptions {
        timing: options.timing,
        recovery_cell_menu: options.recovery_cell_menu,
        recovery_cell_limit: Some(options.recovery_cell_limit),
        resize_registers: false,
        max_upsize_rounds: 0,
        max_downsize_rounds: 2,
        max_area_growth: 0.0,
        timing_evaluation_budget: Some(0),
        // Only this phase renews its recovery budget after a productive window.
        recovery_evaluation_budget: None,
        ..ResizeOptions::default()
    };
    let mut sizing = SizingLibrary::new(library, None);
    let area = run_epochs(
        &mut mapped_gv,
        module_name,
        &mut sizing,
        &area_options,
        Phase::Area,
        options.record_gate_hashes,
    )?;
    let mixed_options = ResizeOptions {
        timing_cell_menu: options.timing_cell_menu,
        max_upsize_rounds: 10,
        max_downsize_rounds: 4,
        max_area_growth: 0.02,
        timing_evaluation_budget: Some(250_000),
        recovery_evaluation_budget: Some(1_300_000),
        ..area_options
    };
    let mixed = run_epochs(
        &mut mapped_gv,
        module_name,
        &mut sizing,
        &mixed_options,
        Phase::Mixed,
        options.record_gate_hashes,
    )?;
    drop(sizing);
    // Leakage recovery follows the last sizing checkpoint. It uses the slack
    // left by sizing without reopening the area budget or giving up a timing
    // gain.
    let mut terminal_leakage = None;
    if let Some(table) = leakage {
        let terminal_options = ResizeOptions {
            max_downsize_rounds: 4,
            timing_evaluation_budget: None,
            recovery_evaluation_budget: Some(1_300_000),
            ..area_options
        };
        let (text, report) = edit_gate(&mapped_gv, module_name, |mapped| {
            recover_leakage_netlist(
                &mut mapped.module,
                &mapped.nets,
                &mut mapped.interner,
                library,
                &terminal_options,
                table,
            )
        })
        .context("terminal leakage recovery")?;
        mapped_gv = text;
        terminal_leakage = Some(report);
    }
    let final_sha256 = hash(&mapped_gv);
    Ok(AdaptiveResizeResult {
        gate_text: mapped_gv,
        report: AdaptiveResizeReport {
            mapped_sha256,
            area,
            mixed,
            terminal_leakage,
            final_sha256,
        },
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Area,
    Mixed,
}

fn run_epochs(
    gate: &mut String,
    module_name: Option<&str>,
    sizing: &mut SizingLibrary<'_>,
    options: &ResizeOptions,
    phase: Phase,
    record_gate_hashes: bool,
) -> Result<AdaptivePhaseReport> {
    let cap = match phase {
        Phase::Area => 7,
        Phase::Mixed => 46,
    };
    let mut epochs = Vec::new();
    let mut epoch_gate_sha256 = record_gate_hashes.then(Vec::new);
    let mut stop = AdaptiveStopReason::EpochCap;
    for epoch in 1..=cap {
        let (next_gate, report) = edit_gate(gate, module_name, |mapped| {
            resize_with_recovery_cost(
                &mut mapped.module,
                &mapped.nets,
                &mut mapped.interner,
                sizing,
                &mut options.clone(),
                RecoveryWindowCheckpoint::Tightest,
            )
        })
        .with_context(|| format!("{phase:?} sizing epoch {epoch}"))?;
        epochs.push(report);
        if let Some(hashes) = &mut epoch_gate_sha256 {
            hashes.push(format!("{:x}", Sha256::digest(&next_gate)));
        }
        if next_gate == *gate {
            stop = AdaptiveStopReason::IdenticalGate;
            break;
        }
        *gate = next_gate;
        //     epochs 1 ... 12 ........ 23 ............... 46 (cap)
        //                    | compare |
        // Spend the extended budget only when the later portion of the search
        // is still improving registered timing. An early gain alone cannot
        // qualify.
        if phase == Phase::Mixed && epoch == 23 {
            let before = epochs[11].final_delay.register_to_register;
            let after = epochs[22].final_delay.register_to_register;
            if let Some(reason) = late_gain_stop(before, after) {
                stop = reason;
                break;
            }
        }
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

/// Continue only for an inclusive 1% gain between the two reported delays.
fn late_gain_stop(before: f64, after: f64) -> Option<AdaptiveStopReason> {
    if !before.is_finite() || !after.is_finite() || before <= 0.0 || after <= 0.0 {
        return Some(AdaptiveStopReason::NoPositiveRegisteredDelay);
    }
    // Compare the exact *reported decimal* values. Binary multiplication can
    // reject an exact 1% gain, for example from 0.005 to 0.00495.
    let (before, e_before) = reported_decimal(before);
    let (after, e_after) = reported_decimal(after);
    let common = e_before.min(e_after);
    let ten = BigUint::from(10_u8);
    let before = before * 99_u8 * ten.pow((e_before - common) as u32);
    let after = after * 100_u8 * ten.pow((e_after - common) as u32);
    (after > before).then_some(AdaptiveStopReason::InsufficientLateDelayGain)
}

/// Extract the significand and base-ten exponent emitted by report JSON.
fn reported_decimal(value: f64) -> (BigUint, i32) {
    let text = serde_json::Number::from_f64(value)
        .expect("finite delay")
        .to_string();
    let (mantissa, exponent) = match text.split_once('e') {
        Some((mantissa, exponent)) => (mantissa, exponent.parse().expect("JSON decimal exponent")),
        None => (text.as_str(), 0),
    };
    let (integer, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let digits = format!("{integer}{fraction}")
        .parse()
        .expect("JSON decimal digits");
    (digits, exponent - fraction.len() as i32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::netlist::resize::tests::library;

    // A gain of exactly 1% qualifies. The adjacent slower float does not.
    // Tiny and huge delays keep the decision independent of numeric scale.
    #[test]
    fn compares_reported_decimals_at_the_inclusive_threshold() {
        assert_eq!(late_gain_stop(0.005, 0.00495), None);
        assert_eq!(
            late_gain_stop(0.005, 0.00495_f64.next_up()),
            Some(AdaptiveStopReason::InsufficientLateDelayGain)
        );
        assert_eq!(late_gain_stop(1e-200, 9.9e-201), None);
        assert_eq!(late_gain_stop(f64::MAX, f64::MIN_POSITIVE), None);
        assert_eq!(
            late_gain_stop(0.0, 0.00495),
            Some(AdaptiveStopReason::NoPositiveRegisteredDelay)
        );
    }

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
            None,
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
        assert_eq!(
            output.report.area.epoch_gate_sha256,
            Some(vec![hash.clone(); 2])
        );
        assert_eq!(
            output.report.mixed.epoch_gate_sha256,
            Some(vec![hash.clone()])
        );
        assert_eq!(output.report.final_sha256, Some(hash));
        assert_eq!(output.report.area.epochs.len(), 2);
        assert_eq!(output.report.area.stop, AdaptiveStopReason::IdenticalGate);
        assert_eq!(output.report.mixed.epochs.len(), 1);
        assert_eq!(output.report.mixed.stop, AdaptiveStopReason::IdenticalGate);
        assert!(output.report.terminal_leakage.is_none());
        Ok(())
    }
}
