// SPDX-License-Identifier: Apache-2.0

//! Helper library for fuzzing xlsynth-pir. Exposes equivalence utilities used
//! by fuzz targets and unit tests.

use xlsynth_pir::ir::{Fn as IrFn, Package};
use xlsynth_pir::ir_random::{
    generate_fn, generate_same_signature_pair, DepletableBytes, RandomFnOptions, StopPolicy,
};

pub mod equiv;

/// Generates a standard random PIR package directly from coverage-guided bytes.
pub fn generate_standard_random_pir_package(data: &[u8], package_name: &str) -> Package {
    let mut entropy = DepletableBytes::new(data);
    generate_fn(
        &mut entropy,
        &RandomFnOptions::default(),
        StopPolicy::WhenEntropyDepleted,
    )
    .expect("fixed standard PIR fuzz options should construct a valid function")
    .into_top_package(package_name)
}

/// Generates two directly constructed PIR functions with one shared signature
/// across the full random-function PIR surface.
pub fn generate_full_random_pir_pair(data: &[u8]) -> (IrFn, IrFn) {
    let options = RandomFnOptions {
        max_nodes: 64,
        allow_zero_width_bits: true,
        allow_arbitrary_width_multiply: true,
        allow_gate: true,
        allow_extension_ops: true,
        ..RandomFnOptions::default()
    };
    let mut entropy = DepletableBytes::new(data);
    let (first, second) =
        generate_same_signature_pair(&mut entropy, &options, StopPolicy::WhenEntropyDepleted)
            .expect("fixed paired PIR fuzz options should construct matching functions");
    (first.function, second.function)
}
