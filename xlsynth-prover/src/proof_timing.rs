// SPDX-License-Identifier: Apache-2.0

//! Wall-clock measurements for a single proof or suite invocation.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// Monotonic elapsed time measured by the process doing the proof.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofTiming {
    pub wall_micros: u64,
}

impl ProofTiming {
    /// Measures the interval since entry to the proof routine.
    pub fn elapsed(start: Instant) -> Self {
        start.elapsed().into()
    }
}

impl From<Duration> for ProofTiming {
    fn from(duration: Duration) -> Self {
        Self { wall_micros: micros(duration) }
    }
}

/// Converts a monotonic duration to the report's integer microseconds.
pub(crate) fn micros(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}
