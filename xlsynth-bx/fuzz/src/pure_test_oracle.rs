// SPDX-License-Identifier: Apache-2.0

//! Read the results of native DSLX bytecode execution for a module of pure
//! tests.

use crate::function_oracle::run_dslx_interpreter;

/// Run the module once, returning each test's first failing assertion line
/// (one-based), or `None` when the test passed, in declaration order.
pub(crate) fn observe(source: &str, expected_names: &[&str]) -> Result<Vec<Option<usize>>, String> {
    let output = run_dslx_interpreter(source, &[], &[])?;
    let stderr = std::str::from_utf8(&output.stderr).map_err(|e| e.to_string())?;
    parse_outcomes(source, expected_names, stderr, output.status.code()).map_err(|error| {
        let stdout = String::from_utf8_lossy(&output.stdout);
        format!(
            "{error}\nDSLX status: {}\nDSLX stdout:\n{stdout}\nDSLX stderr:\n{stderr}",
            output.status
        )
    })
}

/// Verify a complete RUN/result pair per expected test, followed by the
/// matching summary and process status.
fn parse_outcomes(
    source: &str,
    expected_names: &[&str],
    stderr: &str,
    code: Option<i32>,
) -> Result<Vec<Option<usize>>, String> {
    let mut lines = stderr.lines();
    let mut outcomes = Vec::with_capacity(expected_names.len());
    for (index, &name) in expected_names.iter().enumerate() {
        if expected_names[..index].contains(&name) {
            return Err(format!("duplicate DSLX test name {name}"));
        }
        let run = lines
            .next()
            .and_then(|line| line.strip_prefix("[ RUN UNITTEST  ] "));
        if run != Some(name) {
            return Err(format!("expected DSLX to run {name}, got {run:?}"));
        }
        let result = lines
            .next()
            .ok_or_else(|| format!("missing DSLX result for {name}"))?;
        outcomes.push(if result == "[            OK ]" {
            None
        } else {
            Some(parse_failure(source, name, result, &mut lines)?)
        });
    }
    let count = outcomes.len();
    let failed = outcomes.iter().flatten().count();
    let summary = format!("[===============] {count} test(s) ran; {failed} failed; 0 skipped.");
    if lines.next() != Some(summary.as_str()) || lines.next().is_some() {
        return Err(format!("expected final DSLX test summary {summary}"));
    }
    if code != Some(i32::from(failed != 0)) {
        return Err(format!(
            "DSLX exit code {code:?} disagrees with {failed} failed tests"
        ));
    }
    Ok(outcomes)
}

/// Match the assertion span, its FailureError diagnostic and the named FAILED
/// record. An internal interpreter error uses a different FAILED record.
fn parse_failure(
    source: &str,
    name: &str,
    location: &str,
    lines: &mut std::str::Lines<'_>,
) -> Result<usize, String> {
    let row = assertion_source_line(source, location)
        .ok_or_else(|| format!("expected an assertion location for {name}, got {location:?}"))?;
    let mut saw_failure = false;
    for line in lines.by_ref() {
        if let Some(reported_name) = line.strip_prefix("[        FAILED ] ") {
            if reported_name == name && saw_failure {
                return Ok(row);
            }
            return Err(format!("unexpected DSLX failure record for {name}: {line}"));
        }
        // The pinned interpreter underlines a source excerpt, then prints the
        // diagnostic and optional value diff. Diff lines vary by value shape.
        if let Some((underline, diagnostic)) = line.split_once("^ ")
            && underline.bytes().all(|b| b" ~^-_|".contains(&b))
        {
            if saw_failure
                || !diagnostic.starts_with("FailureError: The program being interpreted failed!")
            {
                return Err(format!(
                    "unexpected DSLX diagnostic for {name}: {diagnostic}"
                ));
            }
            saw_failure = true;
        } else if !(line.is_empty()
            || ["  ", "< ", "> ", "; first differing index: "]
                .iter()
                .any(|prefix| line.starts_with(prefix))
            || line
                .split_once(": ")
                .is_some_and(|(n, _)| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit())))
        {
            return Err(format!("unexpected DSLX output during {name}: {line}"));
        }
    }
    Err(format!("missing DSLX failure record for {name}"))
}

/// The native span starts at the opening parenthesis of assert! or assert_eq.
fn assertion_source_line(source: &str, location: &str) -> Option<usize> {
    let (directory, range) = location.rsplit_once("oracle.x:")?;
    if !directory.is_empty() && !directory.ends_with(['/', '\\']) {
        return None;
    }
    let (start, end) = range.split_once('-')?;
    let position = |text: &str| {
        let (row, col) = text.split_once(':')?;
        Some((row.parse::<usize>().ok()?, col.parse::<usize>().ok()?))
    };
    let (row, col) = position(start)?;
    if position(end)? <= (row, col) {
        return None;
    }
    let line = source.lines().nth(row.checked_sub(1)?)?;
    let offset = col.checked_sub(1)?;
    let callee = line
        .get(..offset)?
        .trim_end()
        .rsplit(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '!'))
        .next()?;
    (line.get(offset..)?.starts_with('(') && matches!(callee, "assert!" | "assert_eq"))
        .then_some(row)
}

#[cfg(test)]
mod tests {
    use super::parse_outcomes;

    const SOURCE: &str = r#"#[test]
fn passing() {
    assert_eq(u4:0, u4:0);
}
#[test]
fn failing() {
    assert_eq(u4:2, u4:3);
}
#[test]
fn labeled() { assert!(false, "bad"); }
"#;
    const NAMES: &[&str] = &["passing", "failing", "labeled"];
    // Captured from the pinned v0.62.0 bytecode evaluator with compare=none.
    const STDERR: &str = r#"[ RUN UNITTEST  ] passing
[            OK ]
[ RUN UNITTEST  ] failing
/path/to/oracle.x:7:14-7:26
0005: #[test]
0006: fn failing() {
0007:     assert_eq(u4:2, u4:3);
~~~~~~~~~~~~~~~~~~~^----------^ FailureError: The program being interpreted failed!
  lhs: u4:2
  rhs: u4:3
  were not equal
0008: }
0009: #[test]
[        FAILED ] failing
[ RUN UNITTEST  ] labeled
/path/to/oracle.x:10:23-10:37
0008: }
0009: #[test]
0010: fn labeled() { assert!(false, "bad"); }
~~~~~~~~~~~~~~~~~~~~~~~~~~~~^------------^ FailureError: The program being interpreted failed! bad
[        FAILED ] labeled
[===============] 3 test(s) ran; 2 failed; 0 skipped.
"#;

    #[test]
    fn reads_individual_assertion_spans_in_order() {
        assert_eq!(
            parse_outcomes(SOURCE, NAMES, STDERR, Some(1)).unwrap(),
            [None, Some(7), Some(10)]
        );
    }

    #[test]
    fn rejects_incomplete_or_inconsistent_interpreter_results() {
        for (from, to) in [
            ("[ RUN UNITTEST  ] passing", "[ RUN UNITTEST  ] failing"),
            ("[            OK ]", "[       SKIPPED ]"),
            ("[            OK ]", "[            OK ]\n[            OK ]"),
            ("[        FAILED ] failing\n", ""),
            (
                "[        FAILED ] failing",
                "[        FAILED ] failing: internal error: INVALID_ARGUMENT",
            ),
            ("oracle.x:7:14-7:26", "oracle.x:7:15-7:26"),
            ("oracle.x:7:14-7:26", "other.x:7:14-7:26"),
            ("FailureError:", "TypeInferenceError:"),
            ("FailureError:", "ParseError:"),
            ("2 failed; 0 skipped.", "1 failed; 1 skipped."),
            ("0 skipped.\n", "0 skipped.\n[===============] duplicate\n"),
        ] {
            assert!(
                parse_outcomes(SOURCE, NAMES, &STDERR.replace(from, to), Some(1)).is_err(),
                "{from:?} -> {to:?}"
            );
        }
        for status in [Some(0), Some(2), None] {
            assert!(parse_outcomes(SOURCE, NAMES, STDERR, status).is_err());
        }
    }
}
