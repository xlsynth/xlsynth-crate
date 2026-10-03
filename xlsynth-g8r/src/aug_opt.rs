// SPDX-License-Identifier: Apache-2.0

//! Augmented IR optimization with bounded g8r area and Graph LE costing.

use crate::gatify::ir2gate::GateBuilderCostEvaluator;

pub use xlsynth_pir::aug_opt::{AugOptMode, AugOptOptions, AugOptRewriteStats, AugOptRunResult};

/// Runs aug-opt using g8r's cost model for profitability-gated rewrites.
pub fn run_aug_opt_over_ir_text(
    ir_text: &str,
    top: Option<&str>,
    options: AugOptOptions,
) -> Result<String, String> {
    run_aug_opt_over_ir_text_with_stats(ir_text, top, options).map(|result| result.output_text)
}

/// Runs aug-opt with canonical gate costing and reports applied rewrites.
pub fn run_aug_opt_over_ir_text_with_stats(
    ir_text: &str,
    top: Option<&str>,
    options: AugOptOptions,
) -> Result<AugOptRunResult, String> {
    xlsynth_pir::aug_opt::run_aug_opt_over_ir_text_with_evaluator(
        ir_text,
        top,
        options,
        &mut GateBuilderCostEvaluator::default(),
    )
}
