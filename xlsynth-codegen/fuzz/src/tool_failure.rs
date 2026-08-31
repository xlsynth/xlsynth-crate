// SPDX-License-Identifier: Apache-2.0

//! Sample-level recovery, deliberately separate from strict startup validation.

use xlsynth::external_tool::ToolError;

/// Only tool resource interruptions lack a verdict; all other errors are bugs.
pub fn recover<T>(result: Result<T, ToolError>) -> Result<T, ToolError> {
    match result {
        Err(error) if !error.is_resource_failure() => panic!("{error}"),
        other => other,
    }
}

/// Ends a focused fuzz sample without suppressing ordinary tool failures.
#[macro_export]
macro_rules! fuzz_tool {
    ($result:expr) => {
        match $crate::tool_failure::recover($result) {
            Ok(value) => value,
            // A tool resource interruption gives no semantic verdict. It is not
            // evidence of incorrect generated RTL, so continue with a new sample.
            Err(error) => {
                eprintln!("inconclusive-tool-check {}: {error}", error.reason_key());
                return;
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn resource_interruption_does_not_prevent_the_next_sample() {
        let mut checked = 0;
        let mut inconclusive = 0;
        for sample in [Err(ToolError::timeout("yosys", Duration::ZERO)), Ok(())] {
            match recover(sample) {
                Ok(()) => checked += 1,
                Err(_) => inconclusive += 1,
            }
        }
        assert_eq!((checked, inconclusive), (1, 1));
    }

    #[test]
    #[should_panic(expected = "invalid RTL")]
    fn ordinary_errors_remain_sample_failures() {
        let _ = recover::<()>(Err(ToolError::failure("invalid RTL")));
    }
}
