// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
#[cfg(feature = "has-bitwuzla")]
use xlsynth_prover::prover::default_prover;
use xlsynth_prover::prover::types::{
    BoolPropertyResult, QuickCheckAssertionSemantics, QuickCheckOptions,
};
use xlsynth_prover::prover::{ExternalProver, Prover, QuickCheckErrorKind};

/// Preparation never executes the backend; the caller controls proof order,
/// retries, and reporting boundaries without callbacks.
#[cfg(unix)]
#[test]
fn quickcheck_prepared_suite_is_lazy_and_reusable() {
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
test -f "$script_dir/started" || exit 1
cat "$script_dir/started" >> "$script_dir/invocations"
"#,
    )
    .unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    let marker = dir.path().join("started");
    let invocations = dir.path().join("invocations");
    let prover = ExternalProver::ToolExe(script);
    let suite = prover
        .prepare_dslx_quickchecks(
            &source,
            None,
            &[],
            None,
            QuickCheckAssertionSemantics::Never,
            None,
            &HashMap::new(),
            QuickCheckOptions::default(),
        )
        .unwrap();
    assert!(!invocations.exists(), "preparation must not execute proofs");
    assert_eq!(
        suite
            .properties()
            .iter()
            .map(|p| p.name.as_str())
            .collect::<Vec<_>>(),
        ["first", "second"]
    );
    assert_eq!(
        suite.prove(2).unwrap_err().kind,
        QuickCheckErrorKind::InvalidProperty
    );
    for id in [1, 0, 1] {
        let property = &suite.properties()[id];
        assert_eq!(
            &suite.source()[property.name_span.clone().unwrap()],
            property.name
        );
        std::fs::write(&marker, format!("{}\n", property.name)).unwrap();
        let run = suite.prove(id).unwrap();
        assert_eq!(run.name, property.name);
        assert_eq!(run.result, BoolPropertyResult::Proved);
        std::fs::remove_file(&marker).unwrap();
    }
    assert_eq!(
        std::fs::read_to_string(&invocations).unwrap(),
        "second\nfirst\nsecond\n"
    );
    std::fs::write(&source, "// modified after preparation").unwrap();
    assert_eq!(
        suite.prove(0).unwrap().result,
        BoolPropertyResult::Error(
            "DSLX input changed after QuickCheck preparation; prepare the suite again".into()
        )
    );
    assert_eq!(
        std::fs::read_to_string(&invocations).unwrap(),
        "second\nfirst\nsecond\n"
    );
}

#[test]
fn quickcheck_toolchain_rejects_disabling_optimization_during_preparation() {
    let dir = tempfile::tempdir().unwrap();
    let prover = ExternalProver::ToolDir(dir.path().into());
    let error = prover
        .prepare_dslx_quickchecks(
            &dir.path().join("qc.x"),
            None,
            &[],
            None,
            QuickCheckAssertionSemantics::Never,
            None,
            &HashMap::new(),
            QuickCheckOptions { optimize: false },
        )
        .err()
        .unwrap();
    assert_eq!(error.kind, QuickCheckErrorKind::Configuration);
    assert_eq!(
        error.message,
        "The XLS toolchain always optimizes QuickCheck IR; use an in-process solver to disable optimization"
    );
}

/// Invalid source and filters are recoverable setup errors, not panics or
/// synthetic properties. The collecting API still represents them as errors.
#[test]
fn quickcheck_preparation_errors_are_fallible() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("qc.x");
    let prover = ExternalProver::ToolExe(std::env::current_exe().unwrap());
    for (text, filter, kind) in [
        (None, None, QuickCheckErrorKind::Read),
        (
            Some("#[quickcheck] fn check(x: u8) -> bool { true }"),
            Some("["),
            QuickCheckErrorKind::Configuration,
        ),
        (
            Some("#[quickcheck] fn check( {"),
            None,
            QuickCheckErrorKind::Discovery,
        ),
        (
            Some("#[quickcheck] fn check(x: u8) -> bool { x }"),
            None,
            QuickCheckErrorKind::Discovery,
        ),
    ] {
        if let Some(text) = text {
            std::fs::write(&source, text).unwrap();
        }
        let error = prover
            .prepare_dslx_quickchecks(
                &source,
                None,
                &[],
                filter,
                QuickCheckAssertionSemantics::Never,
                None,
                &HashMap::new(),
                QuickCheckOptions::default(),
            )
            .err()
            .unwrap();
        assert_eq!(error.kind, kind);
        let runs = prover.prove_dslx_quickcheck(
            &source,
            None,
            &[],
            filter,
            QuickCheckAssertionSemantics::Never,
            None,
            &HashMap::new(),
        );
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].result, BoolPropertyResult::Error(error.to_string()));
    }
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
            let runs = prover.prove_dslx_quickcheck_with_options(
                &source,
                None,
                &[],
                None,
                semantics,
                filter,
                &HashMap::new(),
                QuickCheckOptions { optimize },
            );
            assert_eq!(
                runs.iter().map(|run| run.name.as_str()).collect::<Vec<_>>(),
                ["assertion_failure", "return_failure", "passing"]
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
fn quickcheck_filter_and_empty_selection_prepare_only_selected_properties() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("qc.x");
    std::fs::write(
        &source,
        "#[quickcheck] fn selected(x: u8) -> bool { x == x }",
    )
    .unwrap();
    for (filter, count) in [("^selected$", 1), ("^missing$", 0)] {
        let prover = default_prover();
        let suite = prover
            .prepare_dslx_quickchecks(
                &source,
                None,
                &[],
                Some(filter),
                QuickCheckAssertionSemantics::Never,
                None,
                &HashMap::new(),
                QuickCheckOptions::default(),
            )
            .unwrap();
        assert_eq!(suite.properties().len(), count);
        assert_eq!(suite.prove_all().len(), count);
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
    );
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].result, BoolPropertyResult::Error("XLS IR optimization cannot be combined with uninterpreted functions: inlining would erase UF boundaries; set optimize=false".into()));
}

/// In-process proofs and diagnostics use the prepared snapshot, even if the
/// original file is edited or removed before a proof is requested.
#[cfg(feature = "has-bitwuzla")]
#[test]
fn quickcheck_prepared_snapshot_matches_collecting_api() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("qc.x");
    std::fs::write(&source, "#[quickcheck] fn check(x: u8) -> bool { x == x }").unwrap();
    let prover = default_prover();
    let suite = prover
        .prepare_dslx_quickchecks(
            &source,
            None,
            &[],
            None,
            QuickCheckAssertionSemantics::Never,
            None,
            &HashMap::new(),
            QuickCheckOptions::default(),
        )
        .unwrap();
    let collected = prover.prove_dslx_quickcheck(
        &source,
        None,
        &[],
        None,
        QuickCheckAssertionSemantics::Never,
        None,
        &HashMap::new(),
    );
    std::fs::remove_file(&source).unwrap();
    let run = suite.prove(0).unwrap();
    assert_eq!(run.name, collected[0].name);
    assert_eq!(run.result, collected[0].result);
    assert_eq!(run.implicit_token, collected[0].implicit_token);
    assert_eq!(suite.prove(0).unwrap().result, run.result);
}
