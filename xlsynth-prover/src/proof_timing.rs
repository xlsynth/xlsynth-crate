// SPDX-License-Identifier: Apache-2.0

//! Wall-clock measurements for a single proof or suite invocation.

use std::time::{Duration, Instant};

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::value::RawValue;

/// Monotonic elapsed time measured by the process doing the proof.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofTiming {
    pub wall_micros: u64,
}

impl ProofTiming {
    /// Reads the named timing fields shared by worker and saved reports.
    pub(crate) fn from_json(raw: &RawValue) -> Option<Self> {
        read_timing_object(raw)
    }

    /// Discards an invalid measurement without failing its enclosing report.
    pub(crate) fn deserialize_optional<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<Self>, D::Error> {
        let raw = Box::<RawValue>::deserialize(d)?;
        Ok(Self::from_json(&raw))
    }

    /// Measures the interval since entry to the proof routine.
    pub fn elapsed(start: Instant) -> Self {
        start.elapsed().into()
    }
}

/// Timing records use named fields. Serde also accepts positional arrays for
/// structs, which are not part of the report format.
pub(crate) fn read_timing_object<'a, T: Deserialize<'a>>(raw: &'a RawValue) -> Option<T> {
    let data = raw.get();
    if !data.starts_with('{') {
        return None;
    }
    serde_json::from_str(data).ok()
}

impl From<Duration> for ProofTiming {
    fn from(duration: Duration) -> Self {
        Self {
            wall_micros: micros(duration),
        }
    }
}

/// Converts a monotonic duration to the report's integer microseconds.
pub(crate) fn micros(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}
