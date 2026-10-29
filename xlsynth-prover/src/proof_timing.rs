// SPDX-License-Identifier: Apache-2.0

//! Wall-clock measurements for a single proof or suite invocation.

use std::fmt;
use std::time::{Duration, Instant};

use serde::de::{self, IgnoredAny, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::value::RawValue;

use crate::proof_phases::ProofPhases;

/// Monotonic elapsed time measured by the process doing the proof.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ProofTiming {
    pub wall_micros: u64,
    /// Work measured at known backend boundaries; unavailable for opaque
    /// workers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phases: Option<ProofPhases>,
}

impl<'de> Deserialize<'de> for ProofTiming {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            WallMicros,
            Phases,
            #[serde(other)]
            Other,
        }

        /// A single wall time is required even if the phase extension is
        /// unusable.
        struct TimingVisitor;
        impl<'de> Visitor<'de> for TimingVisitor {
            type Value = ProofTiming;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a timing object with a wall_micros field")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<ProofTiming, A::Error> {
                let mut wall_micros = None;
                let mut phases = None;
                let mut saw_phases = false;
                while let Some(field) = map.next_key()? {
                    match field {
                        Field::WallMicros => {
                            if wall_micros.is_some() {
                                return Err(de::Error::duplicate_field("wall_micros"));
                            }
                            wall_micros = Some(map.next_value()?);
                        }
                        Field::Phases if !saw_phases => {
                            saw_phases = true;
                            phases = map.next_value::<TimingObject<ProofPhases>>()?.into_option();
                        }
                        Field::Phases => {
                            phases = None;
                            map.next_value::<IgnoredAny>()?;
                        }
                        Field::Other => {
                            map.next_value::<IgnoredAny>()?;
                        }
                    }
                }
                let wall_micros =
                    wall_micros.ok_or_else(|| de::Error::missing_field("wall_micros"))?;
                Ok(ProofTiming {
                    wall_micros,
                    phases: None,
                }
                .with_phases(phases.unwrap_or_default()))
            }
        }
        d.deserialize_map(TimingVisitor)
    }
}

impl ProofTiming {
    /// Reads only the borrowed timing subtree of a worker report.
    pub(crate) fn from_json(raw: &RawValue) -> Option<Self> {
        /// Serde can skip an extension too deeply nested to buffer as a typed
        /// value.
        #[derive(Deserialize)]
        struct WorkerWall {
            wall_micros: u64,
        }
        let json = raw.get();
        if !json.starts_with('{') {
            return None;
        }
        serde_json::from_str(json).ok().or_else(|| {
            let wall: WorkerWall = serde_json::from_str(json).ok()?;
            Some(Self {
                wall_micros: wall.wall_micros,
                phases: None,
            })
        })
    }

    /// Discards an invalid measurement without failing its enclosing report.
    pub(crate) fn deserialize_optional<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<Self>, D::Error> {
        optional_measurement(d)
    }

    /// Measures the interval since entry to the proof routine.
    pub fn elapsed(start: Instant) -> Self {
        start.elapsed().into()
    }

    /// Attaches only the phases actually observed by this invocation.
    pub fn with_phases(mut self, phases: ProofPhases) -> Self {
        self.phases = (phases != ProofPhases::default()).then_some(phases);
        self
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

impl From<Duration> for ProofTiming {
    fn from(duration: Duration) -> Self {
        Self {
            wall_micros: micros(duration),
            phases: None,
        }
    }
}

/// Converts a monotonic duration to the report's integer microseconds.
pub(crate) fn micros(duration: Duration) -> u64 {
    u64::try_from(duration.as_micros()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Untagged enums buffer their input before deserializing each candidate.
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum BufferedTiming {
        Timing(ProofTiming),
    }

    #[test_case::test_case(r#"{"wall_micros":7}"#, Some(None); "absent phases")]
    #[test_case::test_case(r#"{"wall_micros":7,"phases":0.5}"#, Some(None); "fractional phases")]
    #[test_case::test_case(r#"{"wall_micros":7,"phases":{}}"#, Some(None); "empty phases")]
    #[test_case::test_case(r#"{"wall_micros":7,"phases":{"future_measurement":42}}"#, Some(None); "unobserved phases")]
    #[test_case::test_case(r#"{"wall_micros":7,"phases":{"solver_check_count":2}}"#, Some(Some(2)); "named phases")]
    #[test_case::test_case(r#"{"phases":{"solver_check_count":-1},"wall_micros":7}"#, Some(None); "invalid phase before wall")]
    #[test_case::test_case(r#"{"wall_micros":7,"phases":[7]}"#, Some(None); "positional phases")]
    #[test_case::test_case(r#"{"phases":{},"phases":{},"wall_micros":7}"#, Some(None); "duplicate phases")]
    #[test_case::test_case(r#"{"phases":{"solver_check_count":1,"solver_check_count":2},"wall_micros":7}"#, Some(None); "duplicate phase field")]
    #[test_case::test_case(r#"{"phases":{}}"#, None; "missing wall")]
    #[test_case::test_case(r#"{"wall_micros":-1}"#, None; "invalid wall")]
    #[test_case::test_case(r#"{"wall_micros":7,"wall_micros":8}"#, None; "duplicate wall")]
    #[test_case::test_case(r#"[7]"#, None; "positional timing")]
    fn decode_public_timing(raw: &str, expected_checks: Option<Option<u64>>) {
        let expected = expected_checks.map(|checks| ProofTiming {
            wall_micros: 7,
            phases: checks.map(|solver_check_count| ProofPhases {
                solver_check_count,
                ..ProofPhases::default()
            }),
        });
        assert_eq!(serde_json::from_str::<ProofTiming>(raw).ok(), expected);
        assert_eq!(
            serde_json::from_str::<BufferedTiming>(raw)
                .ok()
                .map(|BufferedTiming::Timing(timing)| timing),
            expected
        );
    }

    #[test]
    fn zero_duration_is_an_observed_phase() {
        let raw = r#"{"wall_micros":7,"phases":{"prepare_work_micros":0}}"#;
        let timing: ProofTiming = serde_json::from_str(raw).unwrap();
        assert_eq!(timing.phases.unwrap().prepare_work_micros, Some(0));
    }
}
