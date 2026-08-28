// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
#[cfg(feature = "has-bitwuzla")]
use xlsynth_prover::prover::default_prover;
use xlsynth_prover::prover::types::{
    BoolPropertyResult, QuickCheckAssertionSemantics, QuickCheckEvent, QuickCheckOptions,
};
use xlsynth_prover::prover::{ExternalProver, Prover};

/// A fake backend requires the start callback to have run before it executes.
/// The finished callback also runs before the next property's start callback.
#[cfg(unix)]
#[test]
fn quickcheck_events_surround_backend_execution() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("qc.x");
    std::fs::write(
        &source,
        r#"
#[quickcheck] fn first(x: u8) -> bool { x == x }
#[quickcheck] fn second(x: u8) -> bool { x == x }
"#,
    )
    .unwrap();
    let script = dir.path().join("prove_quickcheck_main");
    std::fs::write(
        &script,
        r#"#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
script_dir=$(dirname "$0")
test -f "$script_dir/started"
"#,
    )
    .unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    let marker = dir.path().join("started");
    let mut events = Vec::new();
    let runs = ExternalProver::ToolExe(script).prove_dslx_quickcheck_with_options(
        &source,
        None,
        &[],
        None,
        QuickCheckAssertionSemantics::Never,
        None,
        &HashMap::new(),
        QuickCheckOptions::default(),
        &mut |event| match event {
            QuickCheckEvent::Started { name } => {
                assert!(!marker.exists(), "previous property must finish first");
                std::fs::write(&marker, name).unwrap();
                events.push(format!("start:{name}"));
            }
            QuickCheckEvent::Finished(run) => {
                assert_eq!(std::fs::read_to_string(&marker).unwrap(), run.name);
                assert_eq!(run.result, BoolPropertyResult::Proved);
                std::fs::remove_file(&marker).unwrap();
                events.push(format!("finish:{}", run.name));
            }
        },
    );
    assert_eq!(
        events,
        [
            "start:first",
            "finish:first",
            "start:second",
            "finish:second"
        ]
    );
    assert_eq!(runs.len(), 2);
}

#[test]
fn quickcheck_toolchain_rejects_disabling_optimization_with_events() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("qc.x");
    std::fs::write(
        &source,
        "#[quickcheck] fn property(x: u8) -> bool { x == x }",
    )
    .unwrap();
    let mut events = Vec::new();
    let runs = ExternalProver::ToolDir(dir.path().into()).prove_dslx_quickcheck_with_options(
        &source,
        None,
        &[],
        None,
        QuickCheckAssertionSemantics::Never,
        None,
        &HashMap::new(),
        QuickCheckOptions { optimize: false },
        &mut |event| match event {
            QuickCheckEvent::Started { name } => events.push(format!("start:{name}")),
            QuickCheckEvent::Finished(run) => events.push(format!("finish:{}", run.name)),
        },
    );
    assert_eq!(events, ["start:property", "finish:property"]);
    assert_eq!(runs[0].result, BoolPropertyResult::Error("The XLS toolchain always optimizes QuickCheck IR; use an in-process solver to disable optimization".into()));
}

/// Verify that optimization does not discard assertions or weaken filters and
/// assumptions, including an assertion that fails even when the result is true.
#[cfg(feature = "has-bitwuzla")]
#[test]
fn quickcheck_optimization_preserves_assertion_semantics() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("qc.x");
    std::fs::write(
        &source,
        r#"
fn helper(x: u8) -> bool {
    if x == u8:0 { fail!("reject_zero", true) } else { true }
}
#[quickcheck] fn assertion_failure(x: u8) -> bool { helper(x) }
#[quickcheck] fn return_failure(x: u8) -> bool { x != u8:0 }
#[quickcheck] fn passing(x: u8) -> bool { x == x }
"#,
    )
    .unwrap();
    let prover = default_prover();
    for optimize in [true, false] {
        for (semantics, filter, expect_failure) in [
            (QuickCheckAssertionSemantics::Never, None, true),
            (
                QuickCheckAssertionSemantics::Never,
                Some("^reject_zero$"),
                true,
            ),
            (
                QuickCheckAssertionSemantics::Never,
                Some("other_label"),
                false,
            ),
            (QuickCheckAssertionSemantics::Ignore, None, false),
            (QuickCheckAssertionSemantics::Assume, None, false),
        ] {
            let mut events = Vec::new();
            let runs = prover.prove_dslx_quickcheck_with_options(
                &source,
                None,
                &[],
                None,
                semantics,
                filter,
                &HashMap::new(),
                QuickCheckOptions { optimize },
                &mut |event| match event {
                    QuickCheckEvent::Started { name } => events.push(format!("start:{name}")),
                    QuickCheckEvent::Finished(run) => events.push(format!("finish:{}", run.name)),
                },
            );
            assert_eq!(
                events,
                [
                    "start:assertion_failure",
                    "finish:assertion_failure",
                    "start:return_failure",
                    "finish:return_failure",
                    "start:passing",
                    "finish:passing"
                ]
            );
            if expect_failure {
                let BoolPropertyResult::Disproved { inputs, output } = &runs[0].result else {
                    panic!(
                        "expected assertion failure with opt={optimize}: {:?}",
                        runs[0]
                    );
                };
                assert!(runs[0].implicit_token);
                assert!(
                    output
                        .value
                        .get_element(1)
                        .unwrap()
                        .bits_equals_u64_value(1)
                );
                let label = &output.assertion_violation.as_ref().unwrap().label;
                // XLS qualifies assertion labels when inlining. Selection must
                // still use the original label, including anchored regexes.
                assert!(label.ends_with("reject_zero"), "{label}");
                assert_eq!(inputs.len(), 3, "raw IR/JSON retains implicit inputs");
                assert_eq!(inputs[2].name, "x");
                assert!(inputs[2].value.bits_equals_u64_value(0));
            } else {
                assert_eq!(
                    runs[0].result,
                    BoolPropertyResult::Proved,
                    "opt={optimize}, semantics={semantics}"
                );
            }
            assert!(matches!(
                runs[1].result,
                BoolPropertyResult::Disproved { .. }
            ));
            assert_eq!(runs[2].result, BoolPropertyResult::Proved);
        }
    }
}

/// Filtering selects properties, not substring-derived synthetic entries.
#[cfg(feature = "has-bitwuzla")]
#[test]
fn quickcheck_filter_and_empty_selection_have_matching_events() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("qc.x");
    std::fs::write(
        &source,
        "#[quickcheck] fn selected(x: u8) -> bool { x == x }",
    )
    .unwrap();
    for (filter, count) in [("^selected$", 1), ("^missing$", 0)] {
        let mut event_count = 0;
        let runs = default_prover().prove_dslx_quickcheck_with_options(
            &source,
            None,
            &[],
            Some(filter),
            QuickCheckAssertionSemantics::Never,
            None,
            &HashMap::new(),
            QuickCheckOptions::default(),
            &mut |_| event_count += 1,
        );
        assert_eq!(runs.len(), count);
        assert_eq!(event_count, count * 2);
    }
}

/// Exact assertion selection must survive inlining when assertions constrain
/// the proof domain, not just when assertions are themselves the goal.
#[cfg(feature = "has-bitwuzla")]
#[test]
fn quickcheck_optimization_preserves_assumption_domain() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("qc.x");
    std::fs::write(
        &source,
        r#"
fn helper(x: u8) -> bool {
    if x == u8:0 { fail!("reject_zero", false) } else { true }
}
#[quickcheck] fn property(x: u8) -> bool { helper(x) }
"#,
    )
    .unwrap();
    for optimize in [true, false] {
        for (filter, expected) in [
            (Some("^reject_zero$"), true),
            (Some("^other_label$"), false),
            (None, true),
        ] {
            let runs = default_prover().prove_dslx_quickcheck_with_options(
                &source,
                None,
                &[],
                None,
                QuickCheckAssertionSemantics::Assume,
                filter,
                &HashMap::new(),
                QuickCheckOptions { optimize },
                &mut |_| {},
            );
            assert_eq!(
                matches!(runs[0].result, BoolPropertyResult::Proved),
                expected,
                "opt={optimize}, filter={filter:?}: {:?}",
                runs[0]
            );
            if !expected {
                assert!(matches!(
                    runs[0].result,
                    BoolPropertyResult::Disproved { .. }
                ));
            }
        }
    }
}

#[cfg(feature = "has-bitwuzla")]
#[test]
fn quickcheck_library_rejects_optimizing_uf_proofs() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("qc.x");
    std::fs::write(
        &source,
        r#"
fn helper(x: u8) -> u8 { x }
#[quickcheck] fn property(x: u8) -> bool { helper(x) == x }
"#,
    )
    .unwrap();
    let uf_map = HashMap::from([(
        xlsynth::mangle_dslx_name("qc", "helper").unwrap(),
        "F".into(),
    )]);
    let runs = default_prover().prove_dslx_quickcheck_with_options(
        &source,
        None,
        &[],
        None,
        QuickCheckAssertionSemantics::Never,
        None,
        &uf_map,
        QuickCheckOptions::default(),
        &mut |_| {},
    );
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].result, BoolPropertyResult::Error("XLS IR optimization cannot be combined with uninterpreted functions: inlining would erase UF boundaries; set optimize=false".into()));
}
