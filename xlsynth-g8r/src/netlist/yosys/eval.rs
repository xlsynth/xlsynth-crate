// SPDX-License-Identifier: Apache-2.0

//! Two-state combinational evaluation in Yosys, without technology mapping.

use std::collections::BTreeMap;
use std::time::Duration;

use xlsynth::external_tool::ToolError;
use xlsynth_pir::IrFormatPreference;
use xlsynth_pir::{IrBits, IrValue};

use super::{YosysToolchain, format_yosys_invocation_context, is_simple_yosys_identifier};

impl YosysToolchain {
    /// Evaluates batches of concrete inputs on flattened combinational
    /// SystemVerilog.
    ///
    /// Parses and lowers the source once, then returns one named output map per
    /// input vector. `outputs` specifies the requested output port names and
    /// widths. Input bindings must name input ports with matching widths.
    /// Names must be simple Verilog identifiers and widths must be positive.
    /// Unknown output bits are errors, not zeroes or don't-cares. This is not
    /// an event-driven simulator; no clocks, Liberty files, or ABC mapping
    /// are used. The timeout covers the entire batch, including RTL
    /// preparation.
    pub fn eval_combinational(
        &self,
        source: &str,
        top: &str,
        inputs: &[BTreeMap<String, IrBits>],
        outputs: &BTreeMap<String, usize>,
        timeout: Duration,
    ) -> Result<Vec<BTreeMap<String, IrBits>>, ToolError> {
        let program = render_eval_program(top, inputs, outputs)?;
        let directory = tempfile::tempdir()
            .map_err(|error| format!("create temporary Yosys eval directory: {error}"))?;
        std::fs::write(directory.path().join("dut.sv"), source)
            .map_err(|error| format!("write temporary Yosys eval source: {error}"))?;
        let diagnostics = self.run_script(directory.path(), &program, timeout)?;
        (0..inputs.len())
            .map(|sample| {
                let result =
                    std::fs::read_to_string(directory.path().join(format!("eval-{sample}.txt")))
                        .map_err(|error| format!("read Yosys eval result: {error}"))
                        .and_then(|text| parse_eval_results(&text, outputs));
                result.map_err(|error| {
                    ToolError::failure(format!(
                        "Yosys eval sample {sample}: {error}\n{}\nYosys output:\n{diagnostics}",
                        format_yosys_invocation_context(self.path(), &program)
                    ))
                })
            })
            .collect()
    }
}

/// Validates script arguments and emits one isolated result file per vector.
fn render_eval_program(
    top: &str,
    inputs: &[BTreeMap<String, IrBits>],
    outputs: &BTreeMap<String, usize>,
) -> Result<String, String> {
    if !is_simple_yosys_identifier(top) {
        return Err(format!(
            "Yosys eval top must be a simple identifier: {top:?}"
        ));
    }
    if inputs.is_empty() || outputs.is_empty() {
        return Err("Yosys eval requires at least one input vector and requested output".into());
    }
    let mut program = format!(
        "read_verilog -sv dut.sv\nprep -top {top} -flatten\nmemory_map\nopt_clean\nselect -module {top}\n"
    );
    for (name, width) in outputs {
        validate_signal(name, *width)?;
        program.push_str(&format!("select -assert-count 1 o:{name} s:{width} %i\n"));
    }
    for (sample, bindings) in inputs.iter().enumerate() {
        let mut command = format!("tee -o eval-{sample}.txt eval");
        for (name, value) in bindings {
            let width = value.get_bit_count();
            validate_signal(name, width)?;
            program.push_str(&format!("select -assert-count 1 i:{name} s:{width} %i\n"));
            let binary = value.to_string_fmt(IrFormatPreference::Binary, false);
            command.push_str(&format!(
                " -set {name} {width}'b{}",
                binary.trim_start_matches("0b")
            ));
        }
        for name in outputs.keys() {
            command.push_str(&format!(" -show {name}"));
        }
        program.push_str(&command);
        program.push('\n');
    }
    Ok(program)
}

/// Restricts names to literal signals, not Yosys expressions or script syntax.
fn validate_signal(name: &str, width: usize) -> Result<(), String> {
    if !is_simple_yosys_identifier(name) || width == 0 {
        return Err(format!(
            "Yosys eval signal must have a simple identifier and positive width: {name:?} ({width} bits)"
        ));
    }
    Ok(())
}

/// Parses all requested results, rejecting missing, duplicate, or unknown
/// values.
fn parse_eval_results(
    text: &str,
    outputs: &BTreeMap<String, usize>,
) -> Result<BTreeMap<String, IrBits>, String> {
    let mut results = BTreeMap::new();
    for line in text
        .lines()
        .filter_map(|line| line.trim().strip_prefix("Eval result: "))
    {
        let (name, value) = line
            .split_once(" = ")
            .ok_or_else(|| format!("malformed eval result: {line}"))?;
        let name = name.strip_prefix('\\').unwrap_or(name);
        let width = *outputs
            .get(name)
            .ok_or_else(|| format!("unexpected eval output: {name}"))?;
        let value = value
            .strip_suffix('.')
            .ok_or_else(|| format!("malformed eval value: {value}"))?;
        let bits = parse_eval_value(value, width)?;
        if results.insert(name.to_owned(), bits).is_some() {
            return Err(format!("duplicate eval output: {name}"));
        }
    }
    for name in outputs.keys() {
        if !results.contains_key(name) {
            return Err(format!("missing eval output: {name}"));
        }
    }
    Ok(results)
}

/// Yosys prints binary width-tagged constants, or signed decimal for 32 bits.
fn parse_eval_value(value: &str, width: usize) -> Result<IrBits, String> {
    if let Some((actual_width, binary)) = value.split_once('\'') {
        if actual_width.parse::<usize>().ok() != Some(width) || binary.len() != width {
            return Err(format!(
                "eval width mismatch: expected {width} bits, got {value}"
            ));
        }
        if !binary.bytes().all(|bit| matches!(bit, b'0' | b'1')) {
            return Err(format!("eval output is not fully two-state: {value}"));
        }
        Ok(IrBits::from_msb_is_0(
            &binary.bytes().map(|bit| bit == b'1').collect::<Vec<_>>(),
        ))
    } else {
        // Only 32-bit constants use Yosys's untagged decimal representation.
        if width != 32 || value.parse::<i32>().is_err() {
            return Err(format!("invalid {width}-bit eval value: {value}"));
        }
        IrValue::parse_typed(&format!("bits[32]:{value}"))
            .and_then(|value| value.to_bits())
            .map_err(|error| format!("invalid eval value {value}: {error}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_wide_binary_and_signed_decimal_results() {
        let outputs = BTreeMap::from([("wide".into(), 129), ("word".into(), 32)]);
        let text = format!(
            "Eval result: \\word = -2147483648.\nEval result: \\wide = 129'{}.\n",
            "1".repeat(129)
        );
        assert_eq!(
            parse_eval_results(&text, &outputs).unwrap(),
            BTreeMap::from([
                ("wide".into(), IrBits::all_ones(129)),
                ("word".into(), IrBits::signed_min_value(32)),
            ])
        );
    }

    #[test]
    fn rejects_incomplete_or_ambiguous_results() {
        let outputs = BTreeMap::from([("y".into(), 2)]);
        for text in [
            "",
            "Eval result: \\y = 2'0x.",
            "Eval result: \\y = 2'0z.",
            "Eval result: \\y = 1'0.",
            "Eval result: \\y = 0.",
            "Eval result: \\y = 2'00.\nEval result: \\y = 2'00.",
            "Eval result: \\other = 2'00.",
        ] {
            assert!(parse_eval_results(text, &outputs).is_err(), "{text}");
        }
    }

    #[test]
    fn rejects_invalid_or_empty_eval_requests() {
        let inputs = [BTreeMap::from([("x".into(), IrBits::zero(2))])];
        let outputs = BTreeMap::from([("y".into(), 2)]);
        assert!(render_eval_program("top; help", &inputs, &outputs).is_err());
        assert!(render_eval_program("top", &[], &outputs).is_err());
        assert!(render_eval_program("top", &inputs, &BTreeMap::new()).is_err());
        assert!(render_eval_program("top", &inputs, &BTreeMap::from([("y".into(), 0)])).is_err());
        assert!(
            render_eval_program(
                "top",
                &[BTreeMap::from([("x; help".into(), IrBits::zero(2))])],
                &outputs
            )
            .is_err()
        );
        assert!(
            render_eval_program("top", &inputs, &BTreeMap::from([("y; help".into(), 2)])).is_err()
        );
    }
}
