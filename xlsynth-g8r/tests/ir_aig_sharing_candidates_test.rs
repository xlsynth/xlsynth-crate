// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::gate::{AigBitVector, AigOperand};
use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};
use xlsynth_g8r::gatify::ir2gate::GatifyOptions;
use xlsynth_g8r::ir_aig_sharing::{
    CandidateProofResult, IrAigCandidateRhs, IrAigSharingOptions, get_equivalences,
    prove_equivalence_candidates_varisat,
};
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_pir::ir_parser::Parser as PirParser;

#[test]
fn test_ir_aig_sharing_finds_simple_and_node() {
    let _ = env_logger::builder().is_test(true).try_init();

    let ir_text = "package sample

top fn main(a: bits[1] id=1, b: bits[1] id=2) -> bits[1] {
  ret and.3: bits[1] = and(a, b, id=3)
}
";
    let mut parser = PirParser::new(ir_text);
    let pkg = parser
        .parse_and_validate_package()
        .expect("parse PIR package");
    let pir_fn = pkg.get_top_fn().expect("top fn");

    let mut gb = GateBuilder::new("main".to_string(), GateBuilderOptions::no_opt());
    let a = gb.add_input("a".to_string(), 1);
    let b = gb.add_input("b".to_string(), 1);
    let and_ref = gb.add_and_binary(*a.get_lsb(0), *b.get_lsb(0));
    gb.add_output("out".to_string(), AigBitVector::from_bit(and_ref));
    let gate_fn = gb.build();

    let opts = IrAigSharingOptions {
        sample_count: 128,
        sample_seed: 0,
        exclude_structural_pir_nodes: true,
    };
    let cands = get_equivalences(&pkg, pir_fn, &gate_fn, &opts).expect("get equivalences");

    let hits: Vec<_> = cands
        .into_iter()
        .filter(|c| c.pir_node_text_id == 3 && c.bit_index == 0)
        .collect();
    assert!(
        !hits.is_empty(),
        "expected at least one candidate for pir node id=3"
    );
    assert!(
        hits.iter()
            .any(|c| c.rhs == IrAigCandidateRhs::AigOperand(AigOperand::from(and_ref))),
        "expected a candidate mapping to the AND node"
    );

    let gatify_opts = GatifyOptions {
        fold: true,
        hash: true,
        check_equivalence: false,
        adder_mapping: AdderMapping::default(),
        mul_adder_mapping: None,
        range_info: None,
        enable_rewrite_carry_out: false,
    };
    let proofs = prove_equivalence_candidates_varisat(pir_fn, &gate_fn, &hits, &gatify_opts)
        .expect("prove equivalence candidates");
    assert!(
        proofs
            .iter()
            .any(|p| matches!(p.result, CandidateProofResult::Proved)),
        "expected at least one candidate to be proven equivalent"
    );
}
