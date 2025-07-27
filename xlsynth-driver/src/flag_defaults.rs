// SPDX-License-Identifier: Apache-2.0

//! Central definition of default flag values we expect the external
//! `codegen_main` tool to use. Keeping them here lets us use these
//! values consistently across the driver code and test that they stay
//! in sync with the underlying tool implementation.
//!
//! NOTE: The single source of truth is still the external tool — these
//! constants must reflect its behaviour. See the unit test at the
//! bottom of this file which automatically checks that the values
//! match the defaults reported by `codegen_main --helpfull` so we do
//! not silently diverge.

// -- Codegen flag defaults

/// If true, `codegen_main` inserts runtime assertions that check IR-level
/// invariants (e.g. that a priority selector's one-hot input really is
/// one-hot). Corresponds to `--add_invariant_assertions`.  Help output snippet:
///
/// ```text
/// --add_invariant_assertions ... default: true;
/// ```
pub const CODEGEN_ADD_INVARIANT_ASSERTIONS: bool = true;

/// Whether to emit an additional `idle` output signal that indicates no active
/// transaction is flowing through the pipeline.  `--add_idle_output`.
pub const CODEGEN_ADD_IDLE_OUTPUT: bool = false;

/// Emit bounds checks for array index operations –
/// `--array_index_bounds_checking`.
pub const CODEGEN_ARRAY_INDEX_BOUNDS_CHECKING: bool = true;

/// When true, Verilog generation uses SystemVerilog features.
/// `--use_system_verilog`.
pub const CODEGEN_USE_SYSTEM_VERILOG: bool = true;

/// Flop (register) all module inputs, `--flop_inputs` (pipeline generator).
pub const CODEGEN_FLOP_INPUTS: bool = true;

/// Flop (register) all module outputs, `--flop_outputs` (pipeline generator).
pub const CODEGEN_FLOP_OUTPUTS: bool = true;

// -- Tests --------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use regex::Regex;
    use std::process::Command;

    /// Panics if preconditions are not met.
    fn get_codegen_help() -> String {
        let tool_path = std::env::var("XLSYNTH_TOOLS")
            .expect("XLSYNTH_TOOLS environment variable must be set for tests");

        let codegen_main_path = std::path::Path::new(&tool_path).join("codegen_main");
        assert!(
            codegen_main_path.exists(),
            "codegen_main not found at {:?}",
            codegen_main_path
        );

        let output = Command::new(&codegen_main_path)
            .arg("--helpfull")
            .output()
            .expect("failed to run codegen_main");

        // absl::flags exits with code 1 after printing help; accept 0 or 1.
        assert!(
            matches!(output.status.code(), Some(0) | Some(1)),
            "unexpected exit code {:?}",
            output.status
        );

        String::from_utf8_lossy(&output.stdout).into_owned()
    }

    /// Extracts the default boolean value for `--<flag_name>` from the help
    /// text. Panics if the flag or its default is not found.
    fn extract_default(help: &str, flag_name: &str) -> bool {
        // Regex in DOTALL mode – match everything lazily until we see a
        // "default: true;" or "default: false;" capture group.
        let pattern = format!(r"--{}[\s\S]*?default: (true|false);", flag_name);
        let re = Regex::new(&pattern).expect("invalid regex");
        let caps = re.captures(help).unwrap_or_else(|| {
            panic!(
                "Flag --{} not found in codegen_main --helpfull output",
                flag_name
            )
        });
        &caps[1] == "true"
    }

    #[test]
    fn defaults_match_codegen_main() {
        let help = get_codegen_help();

        let checks: &[(&str, bool)] = &[
            ("add_invariant_assertions", CODEGEN_ADD_INVARIANT_ASSERTIONS),
            ("add_idle_output", CODEGEN_ADD_IDLE_OUTPUT),
            (
                "array_index_bounds_checking",
                CODEGEN_ARRAY_INDEX_BOUNDS_CHECKING,
            ),
            ("use_system_verilog", CODEGEN_USE_SYSTEM_VERILOG),
            ("flop_inputs", CODEGEN_FLOP_INPUTS),
            ("flop_outputs", CODEGEN_FLOP_OUTPUTS),
        ];

        for (flag, expected) in checks {
            let actual = extract_default(&help, flag);
            assert_eq!(
                actual, *expected,
                "Default mismatch for flag '--{}': expected {}, got {}",
                flag, expected, actual
            );
        }
    }
}
