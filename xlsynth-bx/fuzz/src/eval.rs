// SPDX-License-Identifier: Apache-2.0

#[path = "../../tests/support/expr_eval.rs"]
pub(crate) mod expr_eval;

use crate::generator::ExpressionSample;
use expr_eval::{ExpressionEvaluator, compile_bundled_dslx, dslx_source};
use xlsynth_pir::random_inputs::generate_argument_sets_from_seed;

/// Compare the same generated inputs across both frontends. Also exercise BX
/// constant evaluation on the first input through a register reset.
pub fn check_sample(sample: &ExpressionSample, input_seed: u64) -> Result<(), String> {
    let ExpressionSample {
        params,
        result_type,
        bindings,
        expression,
    } = sample;
    let body = format!("{bindings}\n{expression}");
    let context = |error| {
        format!(
            "{error}\ninput seed {input_seed:#x}\n{}",
            dslx_source(params, result_type, &body)
        )
    };
    let evaluator = ExpressionEvaluator::compile(params, result_type, &body, compile_bundled_dslx)
        .map_err(context)?;
    for (index, args) in generate_argument_sets_from_seed(&evaluator.bx, input_seed, 4)
        .iter()
        .enumerate()
    {
        let expected = evaluator.compare(args).map_err(context)?;
        if index == 0 {
            evaluator
                .compare_reset(args, result_type, bindings, expression, &expected)
                .map_err(context)?;
        }
    }
    Ok(())
}
