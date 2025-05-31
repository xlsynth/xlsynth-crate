// SPDX-License-Identifier: Apache-2.0

#![no_main]
use libfuzzer_sys::fuzz_target;
use xlsynth_g8r::ir2gate::gatify;
use xlsynth_g8r::xls_ir::ir_parser;
use xlsynth_test_helpers::ir_fuzz::{generate_ir_fn, FuzzSample};

fuzz_target!(|sample: FuzzSample| {
    // Check for necessary environment variables first.
    let _ = std::env::var("XLSYNTH_TOOLS")
        .expect("XLSYNTH_TOOLS environment variable must be set for fuzzing.");

    // Skip empty operation lists or empty input bits
    if sample.ops.is_empty() || sample.input_bits == 0 {
        return;
    }

    let _ = env_logger::builder().try_init();

    // Generate IR function from fuzz input
    let mut package = xlsynth::IrPackage::new("fuzz_test").unwrap();
    if let Err(e) = generate_ir_fn(sample.input_bits, sample.ops, &mut package) {
        log::info!("Error generating IR function: {}", e);
        return;
    }

    let parsed_package = match ir_parser::Parser::new(&package.to_string()).parse_package() {
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
    let parsed_fn = parsed_package.get_top().unwrap();

    // Convert to gates with folding disabled to make less machinery under test.
    let _gate_fn_no_fold = gatify(
        &parsed_fn,
        xlsynth_g8r::ir2gate::GatifyOptions {
            fold: false,
            hash: false,
            check_equivalence: true,
            adder_mapping: xlsynth_g8r::ir2gate_utils::AdderMapping::default(),
        },
    );

    log::info!("unfolded conversion succeeded, attempting folded version...");

    // Now check the folded version is also equivalent.
    let _gate_fn_fold = gatify(
        &parsed_fn,
        xlsynth_g8r::ir2gate::GatifyOptions {
            fold: true,
            hash: true,
            check_equivalence: true,
            adder_mapping: xlsynth_g8r::ir2gate_utils::AdderMapping::default(),
        },
    );

    // If we got here the equivalence checks passed.
    // Note: because of transitivity we know that also the unopt version is
    // equivalent to the opt version.
});
