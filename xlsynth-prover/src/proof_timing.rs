// SPDX-License-Identifier: Apache-2.0

//! Wall-clock measurements for a single proof or suite invocation.

use std::time::{Duration, Instant};

use serde::de::IgnoredAny;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::value::RawValue;

/// Monotonic elapsed time measured by the process doing the proof.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofTiming {
    pub wall_micros: u64,
}

impl ProofTiming {
    /// Reads only the borrowed timing subtree of a worker report.
    pub(crate) fn from_json(raw: &RawValue) -> Option<Self> {
        read_timing_object(raw)
    }

    /// Discards an invalid measurement without failing its enclosing report.
    pub(crate) fn deserialize_optional<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<Self>, D::Error> {
        Ok(TimingObject::<Self>::deserialize(d)?.into_option())
    }

    /// Measures the interval since entry to the proof routine.
    pub fn elapsed(start: Instant) -> Self {
        start.elapsed().into()
    }
}

/// Reads a measurement from a saved report only if it has named fields.
/// Flattening the fields requires a map and preserves duplicate-field checks.
#[derive(Deserialize)]
#[serde(untagged)]
pub(crate) enum TimingObject<T> {
    Named {
        #[serde(flatten)]
        fields: T,
    },
    Invalid(IgnoredAny),
}

impl<T> TimingObject<T> {
    pub(crate) fn into_option(self) -> Option<T> {
        match self {
            Self::Named { fields } => Some(fields),
            Self::Invalid(_) => None,
        }
    }
}

/// Decodes an optional scalar or list from a saved report, including when
/// its enclosing Serde type has already buffered the input.
pub(crate) fn optional_measurement<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
    d: D,
) -> Result<Option<T>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Measurement<T> {
        Valid(T),
        Invalid(IgnoredAny),
    }
    Ok(match Measurement::deserialize(d)? {
        Measurement::Valid(value) => Some(value),
        Measurement::Invalid(_) => None,
    })
}

/// Timing records use named fields. Serde also accepts positional arrays for
/// structs, which are not part of the report format.
fn read_timing_object<'a, T: Deserialize<'a>>(raw: &'a RawValue) -> Option<T> {
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
