// SPDX-License-Identifier: Apache-2.0

//! Compatibility for obsolete inputs, discarded before conversion or proving.

use serde::{Deserialize, Deserializer, de::IgnoredAny};

pub const LOG_TARGET: &str = "xlsynth_driver::obsolete_options";

pub fn warn_type_inference_v2() {
    log::warn!(target: LOG_TARGET, "type_inference_v2 is obsolete and ignored");
}

/// Consume any legacy value without retaining or validating a removed option.
pub fn ignore_type_inference_v2<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<(), D::Error> {
    IgnoredAny::deserialize(deserializer)?;
    warn_type_inference_v2();
    Ok(())
}
