// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "has-bitwuzla")]

use std::fmt::Write;

use xlsynth_pir::ir::Package;
use xlsynth_pir::ir_parser;
use xlsynth_prover::ir_equiv::{
    EquivClassMember, EquivClassRequest, EquivClassResult, IrModule, run_ir_equiv_class_membership,
};
use xlsynth_prover::prover::SolverChoice;
use xlsynth_prover::prover::ir_equiv::prove_ir_fn_equiv_with_interrupt;
use xlsynth_prover::prover::types::{AssertionSemantics, EquivResult, ProverFn};
use xlsynth_prover::solver::AtomicSolverInterrupt;
use xlsynth_prover::solver::bitwuzla::{Bitwuzla, BitwuzlaOptions};

fn make_param_list(width: usize, aux_param_count: usize) -> String {
    let mut params = format!("x: bits[{width}] id=1");
    for i in 0..aux_param_count {
        let _ = write!(&mut params, ", a{i}: bits[{width}] id={}", i + 2);
    }
    params
}

fn make_exact_projection_ir(
    package_name: &str,
    top_name: &str,
    width: usize,
    aux_param_count: usize,
) -> String {
    format!(
        "package {package_name}\n\
top fn {top_name}({}) -> bits[{width}] {{\n  ret identity.{}: bits[{width}] = identity(x, id={})\n}}\n",
        make_param_list(width, aux_param_count),
        aux_param_count + 2,
        aux_param_count + 2,
    )
}

fn make_canceling_chain_ir(
    package_name: &str,
    top_name: &str,
    width: usize,
    aux_param_count: usize,
    rounds: usize,
) -> String {
    let mut ir_text = String::new();
    let _ = writeln!(&mut ir_text, "package {package_name}");
    let _ = writeln!(
        &mut ir_text,
        "top fn {top_name}({}) -> bits[{width}] {{",
        make_param_list(width, aux_param_count)
    );
    let mut current = "x".to_string();
    let mut node_id = aux_param_count + 2;
    for round in 0..rounds {
        let param_name = format!("a{}", round % aux_param_count);
        let add_name = format!("add.{node_id}");
        let _ = writeln!(
            &mut ir_text,
            "  {add_name}: bits[{width}] = add({current}, {param_name}, id={node_id})"
        );
        node_id += 1;
        let sub_name = format!("sub.{node_id}");
        let _ = writeln!(
            &mut ir_text,
            "  {sub_name}: bits[{width}] = sub({add_name}, {param_name}, id={node_id})"
        );
        node_id += 1;
        current = sub_name;
    }
    let _ = writeln!(
        &mut ir_text,
        "  ret identity.{node_id}: bits[{width}] = identity({current}, id={node_id})"
    );
    let _ = writeln!(&mut ir_text, "}}");
    ir_text
}

fn parse_package(ir_text: &str) -> Package {
    ir_parser::Parser::new(ir_text)
        .parse_package()
        .expect("package parses")
}

#[test]
fn test_bitwuzla_pairwise_equiv_returns_interrupted_when_preinterrupted() {
    let lhs_text = make_exact_projection_ir("lhs_pkg", "lhs", 64, 4);
    let rhs_text = make_exact_projection_ir("rhs_pkg", "rhs", 64, 4);
    let lhs_pkg = parse_package(&lhs_text);
    let rhs_pkg = parse_package(&rhs_text);
    let lhs_top = lhs_pkg.get_top_fn().expect("lhs top exists");
    let rhs_top = rhs_pkg.get_top_fn().expect("rhs top exists");
    let lhs = ProverFn::new(lhs_top, Some(&lhs_pkg));
    let rhs = ProverFn::new(rhs_top, Some(&rhs_pkg));
    let interrupt = AtomicSolverInterrupt::new();
    interrupt.interrupt();

    let result = prove_ir_fn_equiv_with_interrupt::<Bitwuzla>(
        &BitwuzlaOptions::new(),
        &lhs,
        &rhs,
        AssertionSemantics::Same,
        None,
        false,
        Some(interrupt.handle()),
    );

    assert_eq!(result, EquivResult::Interrupted);
}

#[test]
fn test_bitwuzla_pairwise_equiv_still_proves_without_interrupt() {
    let lhs_text = make_exact_projection_ir("lhs_pkg", "lhs", 64, 4);
    let rhs_text = make_exact_projection_ir("rhs_pkg", "rhs", 64, 4);
    let lhs_pkg = parse_package(&lhs_text);
    let rhs_pkg = parse_package(&rhs_text);
    let lhs_top = lhs_pkg.get_top_fn().expect("lhs top exists");
    let rhs_top = rhs_pkg.get_top_fn().expect("rhs top exists");
    let lhs = ProverFn::new(lhs_top, Some(&lhs_pkg));
    let rhs = ProverFn::new(rhs_top, Some(&rhs_pkg));

    let result = prove_ir_fn_equiv_with_interrupt::<Bitwuzla>(
        &BitwuzlaOptions::new(),
        &lhs,
        &rhs,
        AssertionSemantics::Same,
        None,
        false,
        None,
    );

    assert_eq!(result, EquivResult::Proved);
}

#[test]
fn test_bitwuzla_class_membership_interrupts_losing_proofs() {
    let candidate_text = make_exact_projection_ir("candidate_pkg", "candidate", 128, 16);
    let exact_text = make_exact_projection_ir("exact_pkg", "exact", 128, 16);
    let deep_one_text = make_canceling_chain_ir("deep_one_pkg", "deep_one", 128, 16, 512);
    let deep_two_text = make_canceling_chain_ir("deep_two_pkg", "deep_two", 128, 16, 768);

    let request = EquivClassRequest::new(
        IrModule::new(&candidate_text).with_top(Some("candidate")),
        vec![
            EquivClassMember::new("exact", IrModule::new(&exact_text).with_top(Some("exact"))),
            EquivClassMember::new(
                "deep_one",
                IrModule::new(&deep_one_text).with_top(Some("deep_one")),
            ),
            EquivClassMember::new(
                "deep_two",
                IrModule::new(&deep_two_text).with_top(Some("deep_two")),
            ),
        ],
    )
    .with_solver(Some(SolverChoice::Bitwuzla))
    .with_max_parallel_proofs(3);

    let report = run_ir_equiv_class_membership(&request).expect("class membership succeeds");

    match &report.result {
        EquivClassResult::Matched(matched) => assert_eq!(matched.member_id, "exact"),
        other => panic!("expected matched result, got {other:?}"),
    }
    assert!(
        report
            .completed_proofs
            .iter()
            .any(|entry| matches!(entry.report.result, EquivResult::Interrupted)),
        "expected at least one interrupted loser, got {:?}",
        report
            .completed_proofs
            .iter()
            .map(|entry| (&entry.member_id, &entry.report.result))
            .collect::<Vec<_>>()
    );
}
