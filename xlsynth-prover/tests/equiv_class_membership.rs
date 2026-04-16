// SPDX-License-Identifier: Apache-2.0

use xlsynth_prover::ir_equiv::{
    EquivClassMember, EquivClassNoMatchReason, EquivClassRequest, EquivClassResult,
    EquivClassShortlistOptions, IrModule, run_ir_equiv_class_membership,
};
use xlsynth_prover::prover::types::EquivResult;

const CANDIDATE_IDENTITY_IR: &str = r#"package candidate
top fn candidate(x: bits[1] id=1) -> bits[1] {
  ret identity.2: bits[1] = identity(x, id=2)
}
"#;

const EXACT_IDENTITY_IR: &str = r#"package exact
top fn exact(x: bits[1] id=1) -> bits[1] {
  ret identity.2: bits[1] = identity(x, id=2)
}
"#;

const DOUBLE_NOT_IR: &str = r#"package double_not
top fn double_not(x: bits[1] id=1) -> bits[1] {
  not.2: bits[1] = not(x, id=2)
  ret not.3: bits[1] = not(not.2, id=3)
}
"#;

const INVERT_IR: &str = r#"package invert
top fn invert(x: bits[1] id=1) -> bits[1] {
  ret not.2: bits[1] = not(x, id=2)
}
"#;

const ZERO_IR: &str = r#"package zero
top fn zero(x: bits[1] id=1) -> bits[1] {
  ret literal.2: bits[1] = literal(value=0, id=2)
}
"#;

const QUERY_IR: &str = r#"package query_pkg
top fn query(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  add.3: bits[8] = add(x, y, id=3)
  ret sub.4: bits[8] = sub(add.3, y, id=4)
}
"#;

const QUERY_EXACT_IR: &str = r#"package exact_pkg
top fn exact(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  add.3: bits[8] = add(x, y, id=3)
  ret sub.4: bits[8] = sub(add.3, y, id=4)
}
"#;

const QUERY_CLOSE_IR: &str = r#"package close_pkg
top fn close(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  add.3: bits[8] = add(x, y, id=3)
  ret sub.4: bits[8] = sub(add.3, x, id=4)
}
"#;

const QUERY_FAR_ONE_IR: &str = r#"package far_one_pkg
top fn far_one(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  umul.3: bits[8] = umul(x, y, id=3)
  ret not.4: bits[8] = not(umul.3, id=4)
}
"#;

const QUERY_FAR_TWO_IR: &str = r#"package far_two_pkg
top fn far_two(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  literal.3: bits[8] = literal(value=1, id=3)
  ret xor.4: bits[8] = xor(x, literal.3, id=4)
}
"#;

fn make_module<'a>(source: &'a str, top: &'a str) -> IrModule<'a> {
    IrModule::new(source).with_top(Some(top))
}

fn make_member<'a>(id: &'a str, source: &'a str, top: &'a str) -> EquivClassMember<'a> {
    EquivClassMember::new(id, make_module(source, top))
}

fn shortlisted_ids(report: &xlsynth_prover::ir_equiv::EquivClassReport) -> Vec<String> {
    report
        .shortlisted_members
        .iter()
        .map(|entry| entry.member_id.clone())
        .collect()
}

#[test]
fn test_equiv_class_membership_matches_best_ranked_equivalent_member() {
    let request = EquivClassRequest::new(
        make_module(CANDIDATE_IDENTITY_IR, "candidate"),
        vec![
            make_member("double_not", DOUBLE_NOT_IR, "double_not"),
            make_member("exact", EXACT_IDENTITY_IR, "exact"),
        ],
    )
    .with_max_parallel_proofs(2);

    let report = run_ir_equiv_class_membership(&request).expect("class proof succeeds");

    assert_eq!(
        shortlisted_ids(&report),
        vec!["exact".to_string(), "double_not".to_string()]
    );
    assert_eq!(report.excluded_by_shortlist_count, 0);
    assert_eq!(report.unstarted_shortlist_count, 0);
    assert!(
        report
            .completed_proofs
            .iter()
            .all(|entry| matches!(&entry.report.result, EquivResult::Proved))
    );
    match &report.result {
        EquivClassResult::Matched(matched) => {
            assert_eq!(matched.member_id, "exact");
            assert_eq!(matched.shortlist_rank, 0);
        }
        other => panic!("expected matched result, got {:?}", other),
    }
}

#[test]
fn test_equiv_class_membership_reports_no_match_when_all_members_fail() {
    let request = EquivClassRequest::new(
        make_module(CANDIDATE_IDENTITY_IR, "candidate"),
        vec![
            make_member("invert", INVERT_IR, "invert"),
            make_member("zero", ZERO_IR, "zero"),
        ],
    )
    .with_max_parallel_proofs(1);

    let report = run_ir_equiv_class_membership(&request).expect("class proof succeeds");

    assert_eq!(report.completed_proofs.len(), 2);
    match report.result {
        EquivClassResult::NoMatch { reason } => {
            assert_eq!(reason, EquivClassNoMatchReason::ExhaustedShortlist);
        }
        other => panic!("expected no-match result, got {:?}", other),
    }
}

#[test]
fn test_equiv_class_membership_shortlists_and_stops_after_first_match() {
    let request = EquivClassRequest::new(
        make_module(QUERY_IR, "query"),
        vec![
            make_member("far_one", QUERY_FAR_ONE_IR, "far_one"),
            make_member("exact", QUERY_EXACT_IR, "exact"),
            make_member("far_two", QUERY_FAR_TWO_IR, "far_two"),
            make_member("close", QUERY_CLOSE_IR, "close"),
        ],
    )
    .with_max_parallel_proofs(1)
    .with_shortlist_options(EquivClassShortlistOptions {
        prefilter_multiplier: 2,
    });

    let report = run_ir_equiv_class_membership(&request).expect("class proof succeeds");

    assert_eq!(
        shortlisted_ids(&report),
        vec!["exact".to_string(), "close".to_string()]
    );
    assert_eq!(report.excluded_by_shortlist_count, 2);
    assert_eq!(report.completed_proofs.len(), 1);
    assert_eq!(report.unstarted_shortlist_count, 1);
    match &report.result {
        EquivClassResult::Matched(matched) => {
            assert_eq!(matched.member_id, "exact");
        }
        other => panic!("expected matched result, got {:?}", other),
    }
}
