// SPDX-License-Identifier: Apache-2.0

//! Differentially execute one clocked BX test against a DSLX test proc.

use std::path::Path;

use xlsynth_bx::{ImportOptions, ResetBehavior, empty_package, run_tests};

use crate::{clocked_test_oracle, clocked_test_sample::ClockedTestSample, input_seed};

/// Compare all registers after each edge, using the DSLX interpreter's trace
/// as the expected rows of the BX test.
pub fn check_clocked_test(data: &[u8]) -> Result<(), String> {
    let sample = ClockedTestSample::from_bytes(data);
    let dslx = sample.dslx_source();
    let mut rows = clocked_test_oracle::observe(
        &dslx,
        sample.row_count(),
        sample.register_count(),
        sample.width(),
    )
    .map_err(|e| format!("{e}\nDSLX:\n{dslx}"))?;
    let run = |rows: &[Vec<String>], expected_failure: Option<usize>| {
        let bx = sample.bx_source(rows);
        run_clocked_rows(&bx, sample.active_low(), expected_failure).map_err(|e| {
            format!(
                "{e}\nBX (active_low={}):\n{bx}\nDSLX:\n{dslx}",
                sample.active_low()
            )
        })
    };
    run(&rows, None)?;

    // Check that a skipped iteration or assertion cannot silently pass. The
    // input selects both the observation to perturb and its expected tick.
    let position = (input_seed(data) % (rows.len() * sample.register_count()) as u64) as usize;
    let row = position / sample.register_count();
    let value = &mut rows[row][position % sample.register_count()];
    *value = if value == "0" { "1" } else { "0" }.to_owned();
    run(&rows, Some(row + 1))
}

/// Require one BX test to pass or to report an equality failure at the
/// selected clock edge.
fn run_clocked_rows(
    source: &str,
    active_low: bool,
    expected_failure: Option<usize>,
) -> Result<(), String> {
    let report = run_tests(
        source,
        Path::new("clocked_rows.bx"),
        empty_package("clocked_rows"),
        ResetBehavior {
            asynchronous: false,
            active_low,
        },
        &ImportOptions::default(),
        true,
    )?;
    let [test] = report.tests.as_slice() else {
        return Err(format!("expected one BX test, got {}", report.tests.len()));
    };
    if test.name != "clocked_rows" {
        return Err(format!(
            "expected BX test 'clocked_rows', got '{}'",
            test.name
        ));
    }
    match (&test.failure, expected_failure) {
        (None, None) => Ok(()),
        (Some(failure), None) => Err(failure.clone()),
        (None, Some(tick)) => Err(format!(
            "BX accepted the incorrect expected value at tick {tick}"
        )),
        (Some(failure), Some(tick)) => {
            let diagnostic = format!("test 'clocked_rows' at tick {tick}: assert_eq failed:");
            if failure.contains(&diagnostic) {
                Ok(())
            } else {
                Err(format!(
                    "expected an assertion failure at tick {tick}, got {failure}"
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_clocked_tests_across_word_widths_and_reset_polarities() -> Result<(), String> {
        check_clocked_test(&[])?;
        for width in [1, 8, 65, 128] {
            for polarity in [0, 0x80] {
                check_clocked_test(&[width - 1, polarity | 0x1f, 7, 9, 31, 0xa5, 2])?;
            }
        }
        Ok(())
    }

    #[test]
    fn proc_trace_detects_reading_the_new_value_on_the_same_edge() -> Result<(), String> {
        // After reset, q0 loads 1 and q1 must see q0's previous value of 0.
        let sample =
            ClockedTestSample::from_bytes(&[0, 1, 0, 1, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 1, 0]);
        let rows = clocked_test_oracle::observe(&sample.dslx_source(), 2, 2, 1)?;
        assert_eq!(rows, [["0", "0"], ["1", "0"]]);
        let bx = sample.bx_source(&rows);
        run_clocked_rows(&bx, sample.active_low(), None)?;
        let incorrect = bx.replace(
            "q1 := q0 + drive.b;",
            "q1 := (drive.a + drive.b) + drive.b;",
        );
        let failure = run_clocked_rows(&incorrect, sample.active_low(), None).unwrap_err();
        assert_eq!(
            failure,
            "clocked_rows.bx:23:9: test 'clocked_rows' at tick 2: assert_eq failed: actual [bits[1]:1, bits[1]:1], expected [bits[1]:1, bits[1]:0]"
        );
        Ok(())
    }
}
