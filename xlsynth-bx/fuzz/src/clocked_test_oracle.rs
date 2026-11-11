// SPDX-License-Identifier: Apache-2.0

//! Observe the state reported by each transaction of a generated DSLX test
//! proc.

use crate::function_oracle::run_dslx_interpreter;

const PASS_SUMMARY: &str = "[===============] 1 test(s) ran; 0 failed; 0 skipped.";

/// Execute the test proc and return the unsigned decimal registers in edge
/// order.
pub fn observe(
    source: &str,
    row_count: usize,
    register_count: usize,
    width: usize,
) -> Result<Vec<Vec<String>>, String> {
    let max_ticks = row_count
        .checked_mul(4)
        .and_then(|ticks| ticks.checked_add(8))
        .ok_or_else(|| format!("DSLX tick limit overflow for {row_count} rows"))?;
    let output = run_dslx_interpreter(
        source,
        &[],
        &["--logtostderr", &format!("--max_ticks={max_ticks}")],
    )?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    let rows = if output.status.success() {
        parse_observations(&stderr, row_count, register_count, width)
    } else {
        Err(format!("DSLX interpreter exited with {}", output.status))
    };
    rows.map_err(|e| {
        let stdout = String::from_utf8_lossy(&output.stdout);
        format!("{e}\nDSLX stdout:\n{stdout}\nDSLX stderr:\n{stderr}")
    })
}

/// Read tagged trace messages from stderr, checking the test summary, shape and
/// bit widths.
fn parse_observations(
    stderr: &str,
    row_count: usize,
    register_count: usize,
    width: usize,
) -> Result<Vec<Vec<String>>, String> {
    if !(1..=128).contains(&width) {
        return Err(format!("expected register width in 1..=128, got {width}"));
    }
    let mut summaries = stderr
        .lines()
        .filter(|line| line.starts_with("[===============]"));
    if summaries.next() != Some(PASS_SUMMARY) || summaries.next().is_some() {
        return Err(format!(
            "expected a single DSLX test summary: {PASS_SUMMARY}"
        ));
    }
    let max_value = u128::MAX >> (128 - width);
    let mut rows = Vec::new();
    for line in stderr.lines() {
        // Abseil prefixes trace_fmt! messages with the timestamp and source
        // location.
        let message = line.split_once("] ").map_or(line, |(_, message)| message);
        let mut fields = message.split_ascii_whitespace();
        if fields.next() != Some("BX_OBSERVE") {
            continue;
        }
        let row = rows.len();
        let index = fields.next().ok_or("BX_OBSERVE missing row index")?;
        if row >= row_count {
            return Err(format!(
                "extra BX_OBSERVE row {index}; expected {row_count} rows"
            ));
        }
        if index != row.to_string() {
            return Err(format!("expected BX_OBSERVE row {row}, got {index:?}"));
        }
        let mut values = Vec::new();
        for (register, field) in fields.enumerate() {
            let value = parse_unsigned_decimal(field).ok_or_else(|| {
                format!("row {row} register {register}: invalid unsigned decimal {field:?}")
            })?;
            if value > max_value {
                return Err(format!(
                    "row {row} register {register}: {value} exceeds {width} bits"
                ));
            }
            values.push(value.to_string());
        }
        if values.len() != register_count {
            return Err(format!(
                "row {row}: expected {register_count} register values, got {}",
                values.len()
            ));
        }
        rows.push(values);
    }
    if rows.len() != row_count {
        return Err(format!(
            "expected {row_count} BX_OBSERVE rows, got {}",
            rows.len()
        ));
    }
    Ok(rows)
}

/// Accept decimal digits with optional underscore separators between groups of
/// digits.
fn parse_unsigned_decimal(field: &str) -> Option<u128> {
    if field
        .split('_')
        .any(|group| group.is_empty() || !group.bytes().all(|b| b.is_ascii_digit()))
    {
        return None;
    }
    field.replace('_', "").parse().ok()
}

#[cfg(test)]
mod tests {
    use super::{PASS_SUMMARY, parse_observations};

    #[test]
    fn preserves_wide_values_and_row_order() {
        let trace = format!(
            "I1111 11:55:40.497280 1234 oracle.x:57] BX_OBSERVE 0 0 18_446_744_073_709_551_616\n\
             I1111 11:55:40.497321 1234 oracle.x:57] BX_OBSERVE 1 {} 42\n{PASS_SUMMARY}",
            u128::MAX
        );
        assert_eq!(
            parse_observations(&trace, 2, 2, 128).unwrap(),
            [
                ["0".to_string(), (1u128 << 64).to_string()],
                [u128::MAX.to_string(), "42".to_string()],
            ]
        );
    }

    #[test]
    fn rejects_missing_repeated_reordered_and_extra_rows() {
        for (trace, expected) in [
            ("BX_OBSERVE 0 0", "expected 2 BX_OBSERVE rows, got 1"),
            (
                "BX_OBSERVE 0 0\nBX_OBSERVE 0 1",
                "expected BX_OBSERVE row 1, got \"0\"",
            ),
            (
                "BX_OBSERVE 1 0\nBX_OBSERVE 0 1",
                "expected BX_OBSERVE row 0, got \"1\"",
            ),
            (
                "BX_OBSERVE 0 0\nBX_OBSERVE 1 1\nBX_OBSERVE 2 1",
                "extra BX_OBSERVE row 2; expected 2 rows",
            ),
        ] {
            assert_eq!(
                parse_observations(&format!("{trace}\n{PASS_SUMMARY}"), 2, 1, 1).unwrap_err(),
                expected
            );
        }
    }

    #[test]
    fn rejects_malformed_observations() {
        for trace in [
            "BX_OBSERVE",
            "BX_OBSERVE zero 0",
            "BX_OBSERVE 0",
            "BX_OBSERVE 0 0 1",
            "BX_OBSERVE 0 +1",
            "BX_OBSERVE 0 -1",
            "BX_OBSERVE 0 0x1",
            "BX_OBSERVE 0 _1",
            "BX_OBSERVE 0 1_",
            "BX_OBSERVE 0 0__1",
            "BX_OBSERVE 0 2",
        ] {
            assert!(
                parse_observations(&format!("{trace}\n{PASS_SUMMARY}"), 1, 1, 1).is_err(),
                "{trace}"
            );
        }
        let overflow =
            format!("BX_OBSERVE 0 340282366920938463463374607431768211456\n{PASS_SUMMARY}");
        assert!(parse_observations(&overflow, 1, 1, 128).is_err());
    }

    #[test]
    fn requires_one_passing_test() {
        for summary in [
            "".to_string(),
            "[===============] 1 test(s) ran; 1 failed; 0 skipped.".to_string(),
            "[===============] 2 test(s) ran; 0 failed; 0 skipped.".to_string(),
            format!("{PASS_SUMMARY}\n{PASS_SUMMARY}"),
        ] {
            assert!(parse_observations(&format!("BX_OBSERVE 0 0\n{summary}"), 1, 1, 1).is_err());
        }
    }
}
