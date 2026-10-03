// SPDX-License-Identifier: Apache-2.0

//! Augmented XLS optimization with local PIR rewrites and g8r gate costs.
//!
//! The normal entrypoints supply the gate cost model. Callers can also inject
//! an evaluator for experiments or use [`ir2gates_from_ir_text`] to optimize
//! before g8r's preparation and mapping steps.

pub mod constant_shift_choices;
pub mod cost;
pub mod ir_cost;
mod optimizer;

#[cfg(test)]
mod test_utils;

pub use optimizer::{
    AugOptMode, AugOptOptions, AugOptRewriteStats, AugOptRunResult, run_aug_opt_over_ir_text,
    run_aug_opt_over_ir_text_with_evaluator, run_aug_opt_over_ir_text_with_stats,
};

use xlsynth_g8r::gate_builder::GateBuilderOptions;
use xlsynth_g8r::ir2gates::{Ir2GatesOptions, Ir2GatesOutput};
use xlsynth_pir::ir_parser::Parser;

/// Optimizes before formal array-alias analysis, range analysis, and mapping.
///
/// Local profitability uses the same gate folding and sharing options as the
/// subsequent mapping. With aug-opt disabled, this calls g8r directly.
pub fn ir2gates_from_ir_text(
    ir_text: &str,
    top: Option<&str>,
    aug_options: AugOptOptions,
    gate_options: Ir2GatesOptions,
) -> Result<Ir2GatesOutput, String> {
    if !aug_options.enable {
        return xlsynth_g8r::ir2gates::ir2gates_from_ir_text(ir_text, top, gate_options);
    }
    let top_name = match top {
        Some(name) => name.to_string(),
        None => Parser::new(ir_text)
            .parse_and_validate_package()
            .map_err(|error| format!("PIR parse/validate failed: {error}"))?
            .get_top_fn()
            .ok_or_else(|| "PIR package has no top function".to_string())?
            .name
            .clone(),
    };
    let mut evaluator =
        cost::GateBuilderCostEvaluator::default().with_gate_builder_options(GateBuilderOptions {
            fold: gate_options.fold,
            hash: gate_options.hash,
        });
    let optimized = run_aug_opt_over_ir_text_with_evaluator(
        ir_text,
        Some(&top_name),
        aug_options,
        &mut evaluator,
    )?;
    xlsynth_g8r::ir2gates::ir2gates_from_ir_text(
        &optimized.output_text,
        Some(&top_name),
        gate_options,
    )
}
