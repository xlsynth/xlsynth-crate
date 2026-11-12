// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! Compares full synchronous block traces with the upstream XLS interpreter.

use std::collections::BTreeMap;

use libfuzzer_sys::fuzz_target;
use rand::{SeedableRng, rngs::StdRng};
use xlsynth_pir::IrValue;
use xlsynth_pir::block_eval::{BlockEvalField, BlockEvaluator, RegisterInit};
use xlsynth_pir::ir::Package;
use xlsynth_pir::ir_random::{
    DepletableBytes, EntropySource, RandomBlockOptions, RandomBlockResetTiming, RandomFnOptions,
    StopPolicy, generate_block_package,
};
use xlsynth_pir::random_inputs::generate_mixed_values_with_rng;
use xlsynth_pir_fuzz::upstream_block::{check_oracle, evaluate_trace};
use xlsynth_pir_fuzz::upstream_compatible_fn_options;

const CYCLE_COUNT: usize = 16;

/// Resolves a named stimulus or checkpoint against the PIR evaluator's ordered
/// interface.
fn ordered_values(
    fields: &[BlockEvalField],
    values: &BTreeMap<String, IrValue>,
) -> Result<Vec<IrValue>, String> {
    if fields.len() != values.len() {
        return Err(format!(
            "field count mismatch: PIR fields {fields:?}, supplied values {values:?}"
        ));
    }
    fields
        .iter()
        .map(|f| {
            values
                .get(&f.name)
                .cloned()
                .ok_or_else(|| format!("missing supplied value for {}", f.name))
        })
        .collect()
}

/// Checks the complete named interface; the upstream maps have no positional
/// ordering.
fn compare_values(
    fields: &[BlockEvalField],
    pir: &[IrValue],
    xls: &BTreeMap<String, IrValue>,
) -> Result<(), String> {
    if fields.len() != pir.len() || fields.len() != xls.len() {
        return Err(format!(
            "field count mismatch: PIR fields {fields:?}, PIR values {pir:?}, XLS {xls:?}"
        ));
    }
    for (field, actual) in fields.iter().zip(pir) {
        let expected = xls
            .get(&field.name)
            .ok_or_else(|| format!("missing field {} in XLS result {xls:?}", field.name))?;
        if actual != expected {
            return Err(format!("{}: PIR {actual}, XLS {expected}", field.name));
        }
    }
    Ok(())
}

/// Runs both interpreters from identical register values and advances their
/// states independently.
fn compare_trace(
    package: &Package,
    ir: &str,
    top: &str,
    initial: &BTreeMap<String, IrValue>,
    inputs: &[BTreeMap<String, IrValue>],
) -> Result<(), String> {
    let expected = evaluate_trace(ir, top, initial, inputs)?;
    if inputs.len() != expected.len() {
        return Err(format!(
            "{} input cycles, {} XLS results",
            inputs.len(),
            expected.len()
        ));
    }
    let evaluator = BlockEvaluator::prepare(package, top).map_err(|e| e.to_string())?;
    let initial_values = ordered_values(evaluator.registers(), initial)?;
    let mut state = evaluator
        .initial_state(RegisterInit::Supplied(initial_values))
        .map_err(|e| e.to_string())?;
    for (cycle, (sample, expected)) in inputs.iter().zip(expected).enumerate() {
        let values = ordered_values(evaluator.inputs(), sample)
            .map_err(|e| format!("cycle {cycle} inputs: {e}"))?;
        let result = evaluator
            .step(&state, &values)
            .map_err(|e| format!("cycle {cycle}: {e}"))?;
        compare_values(evaluator.outputs(), &result.outputs, &expected.outputs)
            .map_err(|e| format!("cycle {cycle} output {e}"))?;
        compare_values(
            evaluator.registers(),
            result.next_state.values(),
            &expected.registers,
        )
        .map_err(|e| format!("cycle {cycle} next register {e}"))?;
        state = result.next_state;
    }
    Ok(())
}

/// Generates one graph and its independently mutable initial state and cycle
/// inputs.
fn check_sample(data: &[u8]) {
    let mut entropy = DepletableBytes::new(data);
    let mut rng = StdRng::seed_from_u64(entropy.take_u64());
    let options = RandomBlockOptions {
        // Reserve outputs before generating the body, which may consume the remaining node budget.
        min_output_ports: 1 + (entropy.take_u64() % 3) as usize,
        reset_timing: RandomBlockResetTiming::Synchronous,
        function_options: RandomFnOptions {
            max_bit_width: 129,
            max_div_mod_bit_width: Some(16),
            max_multiply_operand_bit_width: Some(32),
            max_aggregate_leaves: 8,
            ..upstream_compatible_fn_options(48)
        },
        ..RandomBlockOptions::default()
    };
    let package = generate_block_package(&mut entropy, &options, StopPolicy::WhenEntropyDepleted)
        .expect("fixed differential options should generate a valid block")
        .package;
    let block = package
        .get_top_block()
        .expect("generated package has a top block");
    let ir = package.to_string();
    let initial = block
        .registers
        .iter()
        .map(|r| r.name.clone())
        .zip(generate_mixed_values_with_rng(
            &mut rng,
            block.registers.iter().map(|r| &r.ty),
        ))
        .collect();
    let inputs: Vec<_> = (0..CYCLE_COUNT)
        .map(|_| {
            let values = generate_mixed_values_with_rng(
                &mut rng,
                block.input_ports().map(|p| block.port_type(p)),
            );
            block
                .input_ports()
                .map(|p| block.port_name(p).to_owned())
                .zip(values)
                .collect()
        })
        .collect();

    if let Err(error) = compare_trace(&package, &ir, &block.name, &initial, &inputs) {
        panic!(
            "block interpreter comparison failed: {error}\n{ir}\ninitial registers: {initial:?}\ninput trace: {inputs:?}"
        );
    }
}

fuzz_target!(init: {
    if let Err(error) = check_oracle() {
        eprintln!("fuzz setup failed: {error}");
        std::process::exit(2);
    }
}, |data: &[u8]| check_sample(data));
