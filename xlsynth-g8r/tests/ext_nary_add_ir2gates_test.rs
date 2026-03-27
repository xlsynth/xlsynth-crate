// SPDX-License-Identifier: Apache-2.0

use xlsynth::{IrBits, IrValue};
use xlsynth_g8r::aig::get_summary_stats::get_aig_stats;
use xlsynth_g8r::ir2gate_utils::AdderMapping;
use xlsynth_g8r::ir2gates;
use xlsynth_pir::ir::{
    ExtNaryAddArchitecture, ExtNaryAddTerm, FileTable, MemberType, Node, NodePayload, Package,
    PackageMember, Param, ParamId, Type,
};
use xlsynth_pir::ir_validate;

#[derive(Clone, Copy, Debug)]
enum OperandKind {
    Param,
    Literal(u64),
}

#[derive(Clone, Copy, Debug)]
struct OperandSpec {
    width: usize,
    kind: OperandKind,
    signed: bool,
    negated: bool,
}

/// Builds a literal IR value of the requested width from the masked raw bits.
fn make_bits_value(width: usize, raw_value: u64) -> IrValue {
    let bits = IrBits::make_ubits(width, raw_value & width_mask_u64(width))
        .expect("masked literal bits must fit the requested width");
    IrValue::from_bits(&bits)
}

/// Returns a width-bit mask, saturating to all-ones for widths >= 64.
fn width_mask_u64(width: usize) -> u64 {
    if width == 0 {
        0
    } else if width >= u64::BITS as usize {
        u64::MAX
    } else {
        (1u64 << width) - 1
    }
}

fn build_single_stage_ext_nary_add_ir_text(result_width: usize, terms: &[OperandSpec]) -> String {
    let mut params = Vec::new();
    let mut nodes = vec![Node {
        text_id: 0,
        name: None,
        ty: Type::nil(),
        payload: NodePayload::Nil,
        pos: None,
    }];
    let mut next_text_id = 1usize;
    let mut next_param_index = 0usize;
    let mut ext_terms = Vec::with_capacity(terms.len());

    for (term_index, term) in terms.iter().enumerate() {
        let operand = match term.kind {
            OperandKind::Param => {
                let name = format!("p{next_param_index}");
                let param_id = ParamId::new(next_param_index + 1);
                params.push(Param {
                    name: name.clone(),
                    ty: Type::Bits(term.width),
                    id: param_id,
                });
                let node_ref = xlsynth_pir::ir::NodeRef { index: nodes.len() };
                nodes.push(Node {
                    text_id: next_text_id,
                    name: Some(name),
                    ty: Type::Bits(term.width),
                    payload: NodePayload::GetParam(param_id),
                    pos: None,
                });
                next_param_index += 1;
                next_text_id += 1;
                node_ref
            }
            OperandKind::Literal(value) => {
                let node_ref = xlsynth_pir::ir::NodeRef { index: nodes.len() };
                nodes.push(Node {
                    text_id: next_text_id,
                    name: Some(format!("lit_{term_index}")),
                    ty: Type::Bits(term.width),
                    payload: NodePayload::Literal(make_bits_value(term.width, value)),
                    pos: None,
                });
                next_text_id += 1;
                node_ref
            }
        };
        ext_terms.push(ExtNaryAddTerm {
            operand,
            signed: term.signed,
            negated: term.negated,
        });
    }

    let ret_node_ref = xlsynth_pir::ir::NodeRef { index: nodes.len() };
    nodes.push(Node {
        text_id: next_text_id,
        name: Some("r".to_string()),
        ty: Type::Bits(result_width),
        payload: NodePayload::ExtNaryAdd {
            terms: ext_terms,
            arch: Some(ExtNaryAddArchitecture::BrentKung),
        },
        pos: None,
    });

    let package = Package {
        name: "test".to_string(),
        file_table: FileTable::new(),
        members: vec![PackageMember::Function(xlsynth_pir::ir::Fn {
            name: "f".to_string(),
            params,
            ret_ty: Type::Bits(result_width),
            nodes,
            ret_node_ref: Some(ret_node_ref),
            outer_attrs: Vec::new(),
            inner_attrs: Vec::new(),
        })],
        top: Some(("f".to_string(), MemberType::Function)),
    };

    ir_validate::validate_package(&package)
        .expect("programmatically built ext_nary_add package must validate");
    package.to_string()
}

fn build_two_stage_binary_ext_nary_add_ir_text(width: usize) -> String {
    format!(
        r#"package test

top fn f(p0: bits[{width}] id=1, p1: bits[{width}] id=2, p2: bits[{width}] id=3) -> bits[{width}] {{
  tmp: bits[{width}] = ext_nary_add(p0, p1, signed=[false, false], negated=[false, false], arch=brent_kung, id=4)
  ret r: bits[{width}] = ext_nary_add(tmp, p2, signed=[false, false], negated=[false, false], arch=brent_kung, id=5)
}}
"#
    )
}

fn get_ir_gate_stats(ir_text: &str, fold: bool, hash: bool) -> (usize, usize) {
    let out = ir2gates::ir2gates_from_ir_text(
        ir_text,
        None,
        ir2gates::Ir2GatesOptions {
            fold,
            hash,
            check_equivalence: false,
            enable_rewrite_carry_out: false,
            enable_rewrite_prio_encode: false,
            adder_mapping: AdderMapping::RippleCarry,
            mul_adder_mapping: None,
            aug_opt: Default::default(),
        },
    )
    .expect("ir2gates");
    let stats = get_aig_stats(&out.gatify_output.gate_fn);
    (stats.and_nodes, stats.max_depth)
}

#[test]
fn ext_nary_add_adder_architecture_controls_lowering() {
    fn build_binary_ext_nary_add_ir_text(arch: &str) -> String {
        format!(
            r#"package test

fn f(a: bits[32] id=1, b: bits[32] id=2) -> bits[32] {{
  ret r: bits[32] = ext_nary_add(a, b, signed=[false, false], negated=[false, false], arch={arch}, id=3)
}}
"#
        )
    }

    fn get_ext_nary_add_gate_count(mapping: AdderMapping) -> usize {
        let arch = match mapping {
            AdderMapping::RippleCarry => "ripple_carry",
            AdderMapping::KoggeStone => "kogge_stone",
            AdderMapping::BrentKung => "brent_kung",
        };
        get_ir_gate_stats(
            &build_binary_ext_nary_add_ir_text(arch),
            /* fold= */ false,
            /* hash= */ false,
        )
        .0
    }

    let ripple_carry_gates = get_ext_nary_add_gate_count(AdderMapping::RippleCarry);
    let brent_kung_gates = get_ext_nary_add_gate_count(AdderMapping::BrentKung);
    let kogge_stone_gates = get_ext_nary_add_gate_count(AdderMapping::KoggeStone);

    assert!(
        ripple_carry_gates < brent_kung_gates,
        "expected ripple_carry to use fewer live And2 nodes than brent_kung; got ripple_carry={} brent_kung={}",
        ripple_carry_gates,
        brent_kung_gates
    );
    assert!(
        brent_kung_gates < kogge_stone_gates,
        "expected brent_kung to use fewer live And2 nodes than kogge_stone; got brent_kung={} kogge_stone={}",
        brent_kung_gates,
        kogge_stone_gates
    );
}

#[test]
fn three_input_brent_kung_beats_two_binary_brent_kung_adders() {
    let single_stage = build_single_stage_ext_nary_add_ir_text(
        16,
        &[
            OperandSpec {
                width: 16,
                kind: OperandKind::Param,
                signed: false,
                negated: false,
            },
            OperandSpec {
                width: 16,
                kind: OperandKind::Param,
                signed: false,
                negated: false,
            },
            OperandSpec {
                width: 16,
                kind: OperandKind::Param,
                signed: false,
                negated: false,
            },
        ],
    );
    let binary_chain = build_two_stage_binary_ext_nary_add_ir_text(16);

    let (single_gates, _) = get_ir_gate_stats(
        &single_stage,
        /* fold= */ false,
        /* hash= */ false,
    );
    let (binary_chain_gates, _) = get_ir_gate_stats(
        &binary_chain,
        /* fold= */ false,
        /* hash= */ false,
    );

    assert!(
        single_gates < binary_chain_gates,
        "expected a 3-input brent_kung ext_nary_add to use fewer And2 gates than two consecutive binary brent_kung ext_nary_add nodes; single={} binary_chain={}",
        single_gates,
        binary_chain_gates
    );
}

#[test]
fn narrower_operand_reduces_gate_count() {
    let all_same_width = build_single_stage_ext_nary_add_ir_text(
        16,
        &[
            OperandSpec {
                width: 16,
                kind: OperandKind::Param,
                signed: false,
                negated: false,
            },
            OperandSpec {
                width: 16,
                kind: OperandKind::Param,
                signed: false,
                negated: false,
            },
            OperandSpec {
                width: 16,
                kind: OperandKind::Param,
                signed: false,
                negated: false,
            },
        ],
    );
    let one_narrower = build_single_stage_ext_nary_add_ir_text(
        16,
        &[
            OperandSpec {
                width: 16,
                kind: OperandKind::Param,
                signed: false,
                negated: false,
            },
            OperandSpec {
                width: 16,
                kind: OperandKind::Param,
                signed: false,
                negated: false,
            },
            OperandSpec {
                width: 8,
                kind: OperandKind::Param,
                signed: false,
                negated: false,
            },
        ],
    );

    let (same_width_gates, _) = get_ir_gate_stats(
        &all_same_width,
        /* fold= */ true,
        /* hash= */ true,
    );
    let (narrower_gates, _) =
        get_ir_gate_stats(&one_narrower, /* fold= */ true, /* hash= */ true);

    assert!(
        narrower_gates < same_width_gates,
        "expected a narrower third operand to reduce And2 gate count; all_same_width={} one_narrower={}",
        same_width_gates,
        narrower_gates
    );
}

#[test]
fn constant_operand_reduces_gate_count() {
    let all_params = build_single_stage_ext_nary_add_ir_text(
        16,
        &[
            OperandSpec {
                width: 16,
                kind: OperandKind::Param,
                signed: false,
                negated: false,
            },
            OperandSpec {
                width: 16,
                kind: OperandKind::Param,
                signed: false,
                negated: false,
            },
            OperandSpec {
                width: 16,
                kind: OperandKind::Param,
                signed: false,
                negated: false,
            },
        ],
    );
    let with_constant = build_single_stage_ext_nary_add_ir_text(
        16,
        &[
            OperandSpec {
                width: 16,
                kind: OperandKind::Param,
                signed: false,
                negated: false,
            },
            OperandSpec {
                width: 16,
                kind: OperandKind::Param,
                signed: false,
                negated: false,
            },
            OperandSpec {
                width: 16,
                kind: OperandKind::Literal(0x1234),
                signed: false,
                negated: false,
            },
        ],
    );

    let (all_params_gates, _) =
        get_ir_gate_stats(&all_params, /* fold= */ true, /* hash= */ true);
    let (constant_gates, _) =
        get_ir_gate_stats(&with_constant, /* fold= */ true, /* hash= */ true);

    assert!(
        constant_gates < all_params_gates,
        "expected a nonzero literal operand to reduce And2 gate count; all_params={} constant_operand={}",
        all_params_gates,
        constant_gates
    );
}

#[test]
fn negated_operand_increases_gate_count() {
    let non_negated = build_single_stage_ext_nary_add_ir_text(
        16,
        &[
            OperandSpec {
                width: 16,
                kind: OperandKind::Param,
                signed: false,
                negated: false,
            },
            OperandSpec {
                width: 16,
                kind: OperandKind::Param,
                signed: false,
                negated: false,
            },
            OperandSpec {
                width: 16,
                kind: OperandKind::Param,
                signed: false,
                negated: false,
            },
        ],
    );
    let one_negated = build_single_stage_ext_nary_add_ir_text(
        16,
        &[
            OperandSpec {
                width: 16,
                kind: OperandKind::Param,
                signed: false,
                negated: false,
            },
            OperandSpec {
                width: 16,
                kind: OperandKind::Param,
                signed: false,
                negated: true,
            },
            OperandSpec {
                width: 16,
                kind: OperandKind::Param,
                signed: false,
                negated: false,
            },
        ],
    );

    let (non_negated_gates, _) =
        get_ir_gate_stats(&non_negated, /* fold= */ false, /* hash= */ false);
    let (negated_gates, _) =
        get_ir_gate_stats(&one_negated, /* fold= */ false, /* hash= */ false);

    assert!(
        negated_gates > non_negated_gates,
        "expected a negated operand to increase And2 gate count; non_negated={} negated={}",
        non_negated_gates,
        negated_gates
    );
}

#[test]
fn sixteen_bit_nary_depth_sweep_test() {
    let mut got = Vec::new();
    for operand_count in 2usize..=8 {
        let ir_text = build_single_stage_ext_nary_add_ir_text(
            16,
            &vec![
                OperandSpec {
                    width: 16,
                    kind: OperandKind::Param,
                    signed: false,
                    negated: false,
                };
                operand_count
            ],
        );
        let (_, max_depth) =
            get_ir_gate_stats(&ir_text, /* fold= */ false, /* hash= */ false);
        got.push((operand_count, max_depth));
    }

    // Depth should be log-ish plus a fixed number for the reduction.
    #[rustfmt::skip]
    let want: &[(usize, usize)] = &[
        (2, 17),
        (3, 21),
        (4, 25),
        (5, 29),
        (6, 29),
        (7, 33),
        (8, 33),
    ];
    assert_eq!(got, want);
}
