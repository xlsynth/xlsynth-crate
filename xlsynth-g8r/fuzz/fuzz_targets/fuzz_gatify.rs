// SPDX-License-Identifier: Apache-2.0

#![no_main]
use libfuzzer_sys::fuzz_target;
use xlsynth_g8r::gatify::ir2gate;
use xlsynth_g8r::ir2gate_utils;
use xlsynth_pir::ir_fuzz::{FuzzSample, generate_ir_fn};
use xlsynth_pir::ir_parser;

fuzz_target!(|sample: FuzzSample| {
    // Check for necessary environment variables first.
    let _ = std::env::var("XLSYNTH_TOOLS")
        .expect("XLSYNTH_TOOLS environment variable must be set for fuzzing.");

    // Skip empty operation lists
    if sample.ops.is_empty() {
        return;
    }

    let _ = env_logger::builder().try_init();

    // Generate IR function from fuzz input
    let mut package = xlsynth::IrPackage::new("fuzz_test").unwrap();
    if let Err(e) = generate_ir_fn(sample.ops, &mut package, None) {
        log::info!("Error generating IR function: {}", e);
        return;
    }

    let parsed_package =
        match ir_parser::Parser::new(&package.to_string()).parse_and_validate_package() {
            Ok(parsed_package) => parsed_package,
            Err(e) => {
                log::error!(
                    "Error parsing IR package: {}\npackage:\n{}",
                    e,
                    package.to_string()
                );
                return;
            }
        };
    let parsed_fn = parsed_package.get_top_fn().unwrap();

    // Convert to gates with folding disabled to make less machinery under test.
    let _gate_fn_no_fold = ir2gate::gatify(
        &parsed_fn,
        ir2gate::GatifyOptions {
            fold: false,
            hash: false,
            check_equivalence: true,
            adder_mapping: ir2gate_utils::AdderMapping::default(),
            mul_adder_mapping: None,
            range_info: None,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
        },
    );

    log::info!("unfolded conversion succeeded, attempting folded version...");

    // Now check the folded version is also equivalent.
    let _gate_fn_fold = ir2gate::gatify(
        &parsed_fn,
        ir2gate::GatifyOptions {
            fold: true,
            hash: true,
            check_equivalence: true,
            adder_mapping: ir2gate_utils::AdderMapping::default(),
            mul_adder_mapping: None,
            range_info: None,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
        },
    );

    // If we got here the equivalence checks passed.
    // Note: because of transitivity we know that also the unopt version is
    // equivalent to the opt version.
});
