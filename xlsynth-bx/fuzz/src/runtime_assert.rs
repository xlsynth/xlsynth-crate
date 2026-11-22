// SPDX-License-Identifier: Apache-2.0

//! Compare first failing edges for a BX hardware assertion and native DSLX.

use std::{fmt::Write, num::NonZeroUsize, path::Path};

use xlsynth_bx::{
    ImportOptions, QuickCheckOptions, ResetBehavior, empty_package, run_tests_with_quickcheck,
};

use crate::pure_test_oracle;

/// Run both generated programs and return the zero-based first failing edge.
pub fn check_runtime_assert(data: &[u8]) -> Result<Option<usize>, String> {
    let (&config, trace) = data.split_first().unwrap_or((&0, &[]));
    let reset_value = config & 15;
    let compare = ["==", "!=", "<", ">="][usize::from((config >> 4) & 3)];
    let guard = if config & 0x40 == 0 {
        "d.check"
    } else {
        "!d.check"
    };
    let active_low = config & 0x80 != 0;
    // Start with a reset edge to initialize the BX register. The payload and
    // guard still reach the assertion on that edge if reset masking breaks.
    let edges = std::iter::once(config | 0xc0).chain(trace.iter().copied().take(16));
    let row_count = edges.clone().count();
    let mut common = format!(
        r#"struct Drive {{ value: u4, load: bool, check: bool, reset: bool }}
fn predicate(q: u4, d: Drive) -> bool {{ q {compare} d.value }}
const ROWS: Drive[{row_count}] = [
"#,
    );
    for byte in edges {
        writeln!(
            common,
            "    Drive {{ value: u4:{}, load: {}, check: {}, reset: {} }},",
            byte & 15,
            byte & 0x10 != 0,
            byte & 0x20 != 0,
            byte & 0xc0 == 0xc0,
        )
        .unwrap();
    }
    common.push_str("];\n");
    let bx = format!(
        r#"{common}
block checked(clk, d: Drive) -> u4 {{
    reg q: u4 = u4:{reset_value};
    if d.load {{ q := d.value; }}
    if {guard} {{ assert!(predicate(q, d), "sampled value"); }}
    q
}}
#[quickcheck]
block edges() -> bool {{
    reg drive: Drive;
    let observed = checked(drive);
    for d in ROWS {{
        drive := d;
        tick!(rst: d.reset);
    }}
    true
}}
"#,
    );
    let mut dslx = format!("{common}\n#[test]\nfn edges() {{\n    let q0 = u4:0;\n");
    for tick in 0..row_count {
        writeln!(dslx, "    let d = ROWS[u32:{tick}];").unwrap();
        writeln!(
            dslx,
            r#"    if !d.reset && ({guard}) {{ assert!(predicate(q{tick}, d), "sampled value"); }};"#,
        )
        .unwrap();
        if tick + 1 < row_count {
            writeln!(
                dslx,
                "    let q{} = if d.reset {{ u4:{reset_value} }} else if d.load {{ d.value }} else {{ q{tick} }};",
                tick + 1,
            )
            .unwrap();
        }
    }
    dslx.push_str("}\n");

    compare_first_failure(&bx, &dslx, active_low)
        .map_err(|error| format!("{error}\nBX (active_low={active_low}):\n{bx}\nDSLX:\n{dslx}"))
}

/// Each generated DSLX assertion has its own source line. BX reports its
/// sampled edge in the tick failure from the hardware assertion.
fn compare_first_failure(bx: &str, dslx: &str, active_low: bool) -> Result<Option<usize>, String> {
    let outcomes = pure_test_oracle::observe(dslx, &["edges"])?;
    let expected_tick = outcomes[0]
        .map(|line| {
            dslx.lines()
                .enumerate()
                .filter(|(_, text)| text.contains("assert!("))
                .position(|(i, _)| i + 1 == line)
                .ok_or_else(|| format!("unexpected DSLX assertion on line {line}"))
        })
        .transpose()?;
    let report = run_tests_with_quickcheck(
        bx,
        Path::new("runtime_assert.bx"),
        empty_package("runtime_assert"),
        ResetBehavior {
            asynchronous: false,
            active_low,
        },
        &ImportOptions::default(),
        true,
        QuickCheckOptions {
            trials: NonZeroUsize::MIN,
            cross_check_vastly: true,
            ..Default::default()
        },
    )?;
    let [test] = report.tests.as_slice() else {
        return Err(format!("expected one BX test, got {}", report.tests.len()));
    };
    let expected = expected_tick
        .map(|tick| format!(": test 'edges' at tick {tick}: IR assertion failed: sampled value"));
    let four = test
        .quickcheck
        .as_ref()
        .and_then(|qc| qc.four_valued.as_ref())
        .ok_or("missing Vastly four-valued trial")?;
    let counts = (
        usize::from(expected.is_none()),
        usize::from(expected.is_some()),
        0,
    );
    if (four.passed, four.failed, four.indeterminate) != counts {
        return Err(format!(
            "DSLX first failing edge: {expected_tick:?}; Vastly outcomes: {four:?}"
        ));
    }
    for (backend, failure) in [
        ("PIR/Vastly", &test.failure),
        ("Vastly from X", &four.first_false),
    ] {
        let agrees = match (failure, expected.as_deref()) {
            (None, None) => true,
            (Some(actual), Some(suffix)) => actual
                .lines()
                .next()
                .is_some_and(|line| line.ends_with(suffix)),
            _ => false,
        };
        if !agrees {
            return Err(format!(
                "DSLX first failing edge: {expected_tick:?}; {backend} failure: {failure:?}"
            ));
        }
    }
    Ok(expected_tick)
}

#[cfg(test)]
mod tests {
    use super::check_runtime_assert;

    #[test]
    fn compare_pre_edge_state_across_loads_guards_and_resets() -> Result<(), String> {
        for polarity in [0, 0x80] {
            // Reset to 3, load 4 with the check disabled, then check and load
            // 5 together. Comparing against the newly loaded 5 would pass.
            assert_eq!(check_runtime_assert(&[polarity | 3, 0x14, 0x35])?, Some(2));
            // The same bad comparison on a reset edge is masked, and the next
            // enabled comparison sees the reset value again.
            assert_eq!(
                check_runtime_assert(&[polarity | 3, 0x14, 0xf5, 0x23])?,
                None
            );
        }
        Ok(())
    }
}
