// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! Checks only the architecturally defined modular product-pair invariant.

use std::collections::BTreeMap;

use libfuzzer_sys::fuzz_target;
use xlsynth_codegen::BlockCodegenOptions;
use xlsynth_codegen_fuzz::iverilog::interface;
use xlsynth_codegen_fuzz::preflight::{Oracles, validate};
use xlsynth_codegen_fuzz::{
    INPUT_SAMPLE_COUNT, deterministic_rng, emit, generate_inputs, parse_reference, references,
    top_block,
};
use xlsynth_g8r_fuzz::random_block::flatten_value;
use xlsynth_test_helpers::rtl_sim::{Icarus, LogicValue};

fuzz_target!(init: {
    if let Err(error) = validate(Oracles::Icarus) {
        eprintln!("fuzz setup failed: {error}");
        std::process::exit(2);
    }
}, |data: &[u8]| {
    let signed = data.first().copied().unwrap_or_default() & 1 != 0;
    let lhs_width = usize::from(data.get(1).copied().unwrap_or(7) % 16) + 1;
    let rhs_width = usize::from(data.get(2).copied().unwrap_or(11) % 16) + 1;
    let result_width = usize::from(data.get(3).copied().unwrap_or(15) % 32) + 1;
    let ir = references::partial_product(signed, lhs_width, rhs_width, result_width);
    let package = parse_reference(&ir);
    let rtl = emit(&package, &BlockCodegenOptions::default());
    let mut simulator = xlsynth_codegen_fuzz::fuzz_tool!(
        Icarus::new(&rtl, interface(&package, None))
            .map_err(|error| error.with_context(format!("Icarus compilation failed:\n{ir}")))
    );
    let (block, _) = top_block(&package);
    let mut rng = deterministic_rng(&ir);
    for sample in 0..INPUT_SAMPLE_COUNT {
        let inputs = generate_inputs(block, &mut rng);
        let input_map = block
            .params
            .iter()
            .zip(inputs.iter())
            .map(|(param, value)| {
                (
                    param.name.clone(),
                    LogicValue::from_bits(&flatten_value(value, &param.ty)),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let values = xlsynth_codegen_fuzz::fuzz_tool!(simulator
            .evaluate(&input_map)
            .map_err(|error| error.with_context(format!("Icarus sample {sample}:\n{ir}\n{rtl}"))))
            .outputs;
        let first = values
            .get("first")
            .expect("first product part")
            .to_bits()
            .expect("concrete product");
        let second = values
            .get("second")
            .expect("second product part")
            .to_bits()
            .expect("concrete product");
        let combined = values
            .get("combined")
            .expect("combined product")
            .to_bits()
            .expect("concrete product");
        let expected = values
            .get("expected")
            .expect("reference product")
            .to_bits()
            .expect("concrete product");
        assert_eq!(
            first.add(&second),
            expected,
            "product-pair modular sum differs at sample {sample}:\nIR:\n{ir}\nRTL:\n{rtl}"
        );
        assert_eq!(
            combined, expected,
            "observable combined product differs at sample {sample}:\nIR:\n{ir}\nRTL:\n{rtl}"
        );
    }
});
