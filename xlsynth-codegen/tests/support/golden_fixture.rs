// SPDX-License-Identifier: Apache-2.0

//! Typed block-codegen fixtures shared by the library and CLI tests.
//!
//! `OPTIONS` comment lines form a TOML document deserialized directly into
//! [`BlockCodegenOptions`]. `EXPECT-SV` or `EXPECT-ERROR` lines hold an exact
//! expected result. No CLI parser, process, or filesystem access is involved.

use std::path::PathBuf;

use xlsynth_codegen::BlockCodegenOptions;

/// The library result expected from a fixture, independent of stdout/stderr.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExpectedOutput {
    SystemVerilog(String),
    Error(String),
}

/// One independently authored block IR input and its typed codegen settings.
#[derive(Clone, Debug)]
pub struct GoldenFixture {
    pub source: String,
    pub options: BlockCodegenOptions,
    pub expected: ExpectedOutput,
    // Preserve all source/options comments verbatim when updating expectations.
    preamble: String,
}

impl GoldenFixture {
    /// Parses strict, declarative fixture metadata while preserving IR text.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut source = String::new();
        let mut preamble = String::new();
        let mut options = String::new();
        let mut expected = String::new();
        let mut expects_error = None;
        for line in text.lines() {
            let trimmed = line.trim_start();
            let expectation = trimmed
                .strip_prefix("// EXPECT-SV:")
                .map(|v| (false, v))
                .or_else(|| trimmed.strip_prefix("// EXPECT-ERROR:").map(|v| (true, v)));
            if let Some((is_error, value)) = expectation {
                if expects_error.is_some_and(|previous| previous != is_error) {
                    return Err("fixture mixes EXPECT-SV and EXPECT-ERROR directives".into());
                }
                expects_error = Some(is_error);
                expected.push_str(value.strip_prefix(' ').unwrap_or(value));
                expected.push('\n');
                continue;
            }
            if trimmed.starts_with("// EXPECT-") {
                return Err(format!("unknown expectation directive: {trimmed}"));
            }
            if [
                "// RUN:",
                "// RUN-FAIL:",
                "// STDOUT:",
                "// STDERR:",
                "// AUX-FILE(",
            ]
            .iter()
            .any(|prefix| trimmed.starts_with(prefix))
            {
                return Err("CLI fixture directives are not supported; use OPTIONS and EXPECT-SV/EXPECT-ERROR".into());
            }
            preamble.push_str(line);
            preamble.push('\n');
            if let Some(value) = trimmed.strip_prefix("// OPTIONS:") {
                options.push_str(value.strip_prefix(' ').unwrap_or(value));
                options.push('\n');
            } else {
                source.push_str(line);
                source.push('\n');
            }
        }
        let options = toml::from_str(&options)
            .map_err(|error| format!("invalid fixture OPTIONS: {error}"))?;
        let expected = match expects_error {
            Some(false) => ExpectedOutput::SystemVerilog(expected),
            Some(true) => ExpectedOutput::Error(expected),
            None => return Err("fixture requires EXPECT-SV or EXPECT-ERROR".into()),
        };
        Ok(Self {
            source,
            options,
            expected,
            preamble,
        })
    }

    pub fn expects_error(&self) -> bool {
        matches!(self.expected, ExpectedOutput::Error(_))
    }

    pub fn expected_output(&self) -> &str {
        match &self.expected {
            ExpectedOutput::SystemVerilog(text) | ExpectedOutput::Error(text) => text,
        }
    }

    /// Replaces only expected text, preserving the input and options verbatim.
    pub fn with_expected_output(&self, output: &str) -> Result<String, String> {
        if !output.ends_with('\n') {
            return Err("fixture output must end with a newline".into());
        }
        let directive = if self.expects_error() {
            "EXPECT-ERROR"
        } else {
            "EXPECT-SV"
        };
        let mut result = self.preamble.clone();
        for line in output.lines() {
            result.push_str(&format!("// {directive}:"));
            if !line.is_empty() {
                result.push(' ');
                result.push_str(line);
            }
            result.push('\n');
        }
        Ok(result)
    }
}

/// Locates the shared corpus from either crate's integration-test build.
pub fn default_golden_fixture_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../xlsynth-codegen/tests/goldens/block2sv")
}

#[cfg(test)]
mod tests {
    use super::{ExpectedOutput, GoldenFixture};
    use xlsynth_codegen::{BlockCodegenOptions, Layout};

    #[test]
    fn typed_options_and_output_preserve_indentation_and_comments() {
        let text = "package p\n// source comment\n// OPTIONS: top = \"p\"\n// OPTIONS: layout = \"pipeline\"\n// OPTIONS: separate_lines = true\n// EXPECT-SV: module p;\n// EXPECT-SV:   logic x;\n// EXPECT-SV:\n// EXPECT-SV: endmodule\n";
        let fixture = GoldenFixture::parse(text).unwrap();
        assert_eq!(fixture.source, "package p\n// source comment\n");
        assert_eq!(fixture.options.top.as_deref(), Some("p"));
        assert_eq!(fixture.options.layout, Layout::Pipeline);
        assert!(fixture.options.separate_lines);
        assert_eq!(
            fixture.expected_output(),
            "module p;\n  logic x;\n\nendmodule\n"
        );
        assert_eq!(
            fixture
                .with_expected_output(fixture.expected_output())
                .unwrap(),
            text
        );
        assert_eq!(
            fixture.with_expected_output("replacement\n").unwrap(),
            "package p\n// source comment\n// OPTIONS: top = \"p\"\n// OPTIONS: layout = \"pipeline\"\n// OPTIONS: separate_lines = true\n// EXPECT-SV: replacement\n"
        );
    }

    #[test]
    fn omitted_options_use_the_library_defaults() {
        let fixture = GoldenFixture::parse("package p\n// EXPECT-SV: module p;\n").unwrap();
        assert_eq!(
            toml::to_string(&fixture.options).unwrap(),
            toml::to_string(&BlockCodegenOptions::default()).unwrap()
        );
    }

    #[test]
    fn expected_errors_are_library_messages_not_cli_diagnostics() {
        let fixture = GoldenFixture::parse("package p\n// EXPECT-ERROR: missing top\n").unwrap();
        assert_eq!(
            fixture.expected,
            ExpectedOutput::Error("missing top\n".into())
        );
        assert!(fixture.expects_error());
        assert!(fixture.with_expected_output("unterminated").is_err());
    }

    #[test]
    fn register_templates_are_embedded_typed_options() {
        let text = "package p\n// OPTIONS: [register_codegen_options]\n// OPTIONS: reg_template = \"always_ff @(posedge {{clock}}) {{reg}} <= {{next}};\"\n// EXPECT-SV: module p;\n";
        let fixture = GoldenFixture::parse(text).unwrap();
        assert_eq!(
            fixture
                .options
                .register_codegen_options
                .unwrap()
                .reg_template
                .as_deref(),
            Some("always_ff @(posedge {{clock}}) {{reg}} <= {{next}};")
        );
    }

    #[test]
    fn malformed_metadata_is_rejected_not_silently_defaulted() {
        for text in [
            "// OPTIONS: typo = true\n// EXPECT-SV: x\n",
            "// OPTIONS: separate_lines = \"true\"\n// EXPECT-SV: x\n",
            "// OPTIONS: layout = \"invalid\"\n// EXPECT-SV: x\n",
            "// OPTIONS: top = \"a\"\n// OPTIONS: top = \"b\"\n// EXPECT-SV: x\n",
            "// OPTIONS: top = 'unterminated\n// EXPECT-SV: x\n",
            "// EXPECT-SV: x\n// EXPECT-ERROR: y\n",
            "// EXPECT-UNKNOWN: x\n",
            "// RUN: driver block2sv %s\n// EXPECT-SV: x\n",
            "// OPTIONS: [register_codegen_options]\n// OPTIONS: reg_template = \"missing placeholders\"\n// EXPECT-SV: x\n",
            "package p\n",
        ] {
            assert!(
                GoldenFixture::parse(text).is_err(),
                "accepted invalid fixture: {text}"
            );
        }
    }
}
