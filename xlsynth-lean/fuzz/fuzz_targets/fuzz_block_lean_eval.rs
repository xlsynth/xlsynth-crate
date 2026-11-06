// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! Compares each Lean clock's outputs and committed registers with PIR.

use std::sync::OnceLock;

use libfuzzer_sys::fuzz_target;
use rand::{RngCore, SeedableRng, rngs::StdRng};
use xlsynth_g8r_fuzz::random_block::evaluate_block_cycle;
use xlsynth_ir_value::IrValue;
use xlsynth_lean::LeanEvaluator;
use xlsynth_pir::ir_random::{
    DepletableBytes, OperationSet, RandomBlockOptions, RandomBlockResetTiming, RandomFnOptions,
    RandomOperation::{
        BitSlice, Concat, Eq, Literal, Ne, OneHot, OneHotSel, PrioritySel, Sel, SignExt,
    },
    StopPolicy, generate_block_package,
};
use xlsynth_pir::random_inputs::generate_mixed_values_with_rng;

static EVALUATOR: OnceLock<LeanEvaluator> = OnceLock::new();

fn block_options() -> RandomBlockOptions {
    RandomBlockOptions {
        min_input_ports: 1,
        max_input_ports: 3,
        min_output_ports: 1,
        require_reset_on_all_registers: true,
        reset_timing: RandomBlockResetTiming::Synchronous,
        function_options: RandomFnOptions {
            max_bit_width: 16,
            allow_arrays: false,
            allow_tuples: false,
            // Arithmetic and bitwise generation can insert zero_ext when
            // adapting operand widths, outside the raw Lean emitter's subset.
            enabled_operations: OperationSet::new([
                Literal,
                Eq,
                Ne,
                Concat,
                BitSlice,
                SignExt,
                Sel,
                PrioritySel,
                OneHot,
                OneHotSel,
            ]),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn check_case(data: &[u8]) {
    let mut rng = StdRng::from_seed(*blake3::hash(data).as_bytes());
    // The generator consumes its wiring header before it builds any logic.
    // Give short corpus entries a reproducible tail so an empty-corpus run
    // can exercise different register layouts and operations immediately.
    let mut padded = [0u8; 512];
    let graph_bytes = if data.len() < padded.len() {
        padded[..data.len()].copy_from_slice(data);
        rng.fill_bytes(&mut padded[data.len()..]);
        &padded[..]
    } else {
        data
    };
    let package = generate_block_package(
        &mut DepletableBytes::new(graph_bytes),
        &block_options(),
        StopPolicy::WhenEntropyDepleted,
    )
    .expect("bounded block generation")
    .package;
    let block = package.get_top_block().expect("generated top block");
    let reset = block.reset.as_ref().expect("required register reset");
    let reset_index = block.input_ports().position(|p| p == reset.port).unwrap();
    let initial = generate_mixed_values_with_rng(&mut rng, block.registers.iter().map(|r| &r.ty));
    // The first reset observes arbitrary old state. Reassert reset after
    // ordinary clocks to exercise reset priority over enabled/held writes.
    let inputs: Vec<_> = (0..8)
        .map(|clock| {
            let mut values = generate_mixed_values_with_rng(
                &mut rng,
                block.input_ports().map(|p| block.port_type(p)),
            );
            let asserted = matches!(clock, 0 | 4);
            values[reset_index] =
                IrValue::make_ubits(1, u64::from(asserted != reset.active_low)).unwrap();
            values
        })
        .collect();
    let ir = package.to_string();
    let actual = EVALUATOR
        .get()
        .expect("Lean evaluator initialized at startup")
        .eval_block_trace(&package, block, &initial, &inputs)
        .unwrap_or_else(|err| {
            panic!("Lean evaluation failed: {err}\n{ir}\ninitial={initial:?}\ninputs={inputs:?}")
        });
    assert_eq!(actual.len(), inputs.len(), "wrong trace length\n{ir}");
    let mut state = initial;
    for (clock, (inputs, result)) in inputs.iter().zip(actual).enumerate() {
        let (outputs, next) = evaluate_block_cycle(block, inputs, &state, &ir);
        assert_eq!(
            result.outputs, outputs,
            "outputs at clock {clock}\n{ir}\nstate={state:?}\ninputs={inputs:?}"
        );
        assert_eq!(
            result.next_state, next,
            "registers at clock {clock}\n{ir}\nstate={state:?}\ninputs={inputs:?}"
        );
        state = next;
    }
}

fuzz_target!(init: {
    let lean = std::env::var_os("XLSYNTH_LEAN").expect("set XLSYNTH_LEAN to the actual Lean executable");
    EVALUATOR.set(LeanEvaluator::new(lean).expect("initialize Lean evaluator")).unwrap();
}, |data: &[u8]| {
    check_case(data);
});
