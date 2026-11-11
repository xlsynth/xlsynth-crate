// SPDX-License-Identifier: Apache-2.0

//! Run identical pure test functions through the BX and DSLX test runners.

use std::path::Path;

use xlsynth_bx::{BxTestResult, ImportOptions, ResetBehavior, empty_package, run_tests};

use crate::{pure_test_oracle, pure_test_sample::PureTestSample};

const BX_FILE: &str = "suite.bx";

/// Compare each generated test's name, pass/fail result and first failing
/// assertion, including the result of the test following a deliberate failure.
pub fn check_pure_test(data: &[u8]) -> Result<(), String> {
    let sample = PureTestSample::from_bytes(data);
    let names = sample.expected.each_ref().map(|test| test.name);
    pure_test_oracle::observe(&sample.source, &names)
        .and_then(|dslx| compare_bx_outcomes(&sample, &dslx))
        .map_err(|e| format!("{e}\nBX / DSLX source:\n{}", sample.source))
}

/// Check the BX report against both the interpreter and the generated
/// expectations so an unexpected failure shared by both runners is visible.
fn compare_bx_outcomes(sample: &PureTestSample, dslx: &[Option<usize>]) -> Result<(), String> {
    let bx = run_tests(
        &sample.source,
        Path::new(BX_FILE),
        empty_package("suite"),
        ResetBehavior::default(),
        &ImportOptions::default(),
        true,
    )?;
    let count = sample.expected.len();
    if bx.tests.len() != count || dslx.len() != count {
        return Err(format!(
            "expected {count} tests, BX ran {} and DSLX ran {}",
            bx.tests.len(),
            dslx.len()
        ));
    }
    for ((expected, test), &dslx_line) in sample.expected.iter().zip(&bx.tests).zip(dslx) {
        if test.name != expected.name {
            return Err(format!(
                "expected test '{}', BX ran '{}'",
                expected.name, test.name
            ));
        }
        let bx_line = bx_assertion_line(test)?;
        if bx_line != dslx_line || dslx_line != expected.failure_line {
            return Err(format!(
                "test '{}': first failing assertion line (None means passed): BX {bx_line:?}, DSLX {dslx_line:?}, generated {:?}\nBX diagnostic: {:?}",
                test.name, expected.failure_line, test.failure
            ));
        }
    }
    Ok(())
}

/// Treat only source assertion failures as comparable outcomes. A BX
/// evaluation or harness error must fail the fuzz sample even at the same line.
fn bx_assertion_line(test: &BxTestResult) -> Result<Option<usize>, String> {
    let Some(failure) = &test.failure else {
        return Ok(None);
    };
    let (position, message) = failure
        .split_once(&format!(": test '{}': ", test.name))
        .ok_or_else(|| failure.clone())?;
    if message != "assertion failed"
        && !message.starts_with("assertion failed: ")
        && !message.starts_with("assert_eq failed: ")
    {
        return Err(failure.clone());
    }
    let line = position
        .strip_prefix(BX_FILE)
        .and_then(|position| position.strip_prefix(':'))
        .and_then(|position| position.split_once(':'))
        .and_then(|(line, _)| line.parse().ok())
        .ok_or_else(|| format!("missing BX assertion location: {failure}"))?;
    Ok(Some(line))
}

#[cfg(test)]
mod tests {
    use super::{PureTestSample, check_pure_test, compare_bx_outcomes, pure_test_oracle};

    #[test]
    fn compare_generated_pure_tests() -> Result<(), String> {
        check_pure_test(&[])?;
        for width in [1u8, 8, 65, 128] {
            for flags in [0x00, 0x15, 0x2a, 0x1f] {
                for range in [0x0c, 0x8c] {
                    check_pure_test(&[width - 1, flags, range, 0x83, 0xf4, 0x17])?;
                }
            }
        }
        Ok(())
    }

    #[test]
    fn a_skipped_bx_assertion_is_detected_even_when_both_runners_fail_the_test()
    -> Result<(), String> {
        let mut sample = PureTestSample::from_bytes(&[7, 0x10, 4, 3, 1]);
        let names = sample.expected.each_ref().map(|test| test.name);
        let dslx = pure_test_oracle::observe(&sample.source, &names)?;
        let first = sample.expected[1].failure_line.unwrap();
        let mut lines: Vec<_> = sample.source.lines().collect();
        let later = lines
            .iter()
            .position(|line| line.trim() == "assert_eq(true, false);")
            .unwrap()
            + 1;
        // Disable only the selected assertion in BX. The following sentinel
        // still fails, leaving the suite's pass/fail counts unchanged.
        lines[first - 1] = "        assert_eq(true, true);";
        sample.source = lines.join("\n");
        let error = compare_bx_outcomes(&sample, &dslx).unwrap_err();
        assert_eq!(
            error.lines().next().unwrap(),
            format!(
                "test 'selected_failure': first failing assertion line (None means passed): BX Some({later}), DSLX Some({first}), generated Some({first})"
            )
        );
        Ok(())
    }
}
