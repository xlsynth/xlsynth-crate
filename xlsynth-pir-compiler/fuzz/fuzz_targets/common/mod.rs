// SPDX-License-Identifier: Apache-2.0

use xlsynth::IrValue;
use xlsynth_pir::ir::Fn;
use xlsynth_pir::random_inputs::generate_argument_sets_from_seed;

pub const RANDOM_ARGUMENT_TRIALS: usize = 32;

/// Generates reproducible argument sets independent of graph decoding.
pub fn random_argument_sets(data: &[u8], function: &Fn) -> Vec<Vec<IrValue>> {
    let mut seed = 0xcbf2_9ce4_8422_2325_u64;
    for byte in data {
        seed ^= u64::from(*byte);
        seed = seed.wrapping_mul(0x0000_0100_0000_01b3);
    }
    generate_argument_sets_from_seed(function, seed, RANDOM_ARGUMENT_TRIALS)
}
