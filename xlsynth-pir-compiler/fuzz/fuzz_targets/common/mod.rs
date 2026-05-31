// SPDX-License-Identifier: Apache-2.0

use rand::SeedableRng;
use rand_pcg::Pcg64Mcg;
use xlsynth_pir::ir_random::RngEntropy;

pub const RANDOM_ARGUMENT_TRIALS: usize = 32;

/// Creates a reproducible pseudorandom argument stream independent of graph
/// decoding.
pub fn random_argument_entropy(data: &[u8], trial: usize) -> RngEntropy<Pcg64Mcg> {
    let mut seed = 0xcbf2_9ce4_8422_2325_u64;
    for byte in data {
        seed ^= u64::from(*byte);
        seed = seed.wrapping_mul(0x0000_0100_0000_01b3);
    }
    seed ^= (trial as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    RngEntropy::new(Pcg64Mcg::seed_from_u64(seed))
}
