// SPDX-License-Identifier: Apache-2.0

//! Validate deprecated inputs without restoring typechecker selection.

use anyhow::Result;
use serde::{Deserialize, Deserializer};

/// Keep deprecation diagnostics enabled by default without other library logs.
pub const LOG_TARGET: &str = "xlsynth_driver::obsolete_options";

/// Warn for V2 requests and reject V1 requests without retaining a selector.
pub fn accept_type_inference_v2(value: bool) -> Result<()> {
    if value {
        log::warn!(target: LOG_TARGET, "The type_inference_v2 option is deprecated; V2 is always used.");
        Ok(())
    } else {
        anyhow::bail!(
            "type_inference_v2=false requests V1, which is no longer supported; remove the option or set it to true"
        )
    }
}

/// Validate each supplied config value even when a CLI value is also present.
pub fn deserialize_type_inference_v2<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<(), D::Error> {
    let value = bool::deserialize(deserializer)?;
    accept_type_inference_v2(value).map_err(serde::de::Error::custom)
}
