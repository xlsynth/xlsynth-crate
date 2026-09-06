// SPDX-License-Identifier: Apache-2.0

//! Signature references are independent of node positions and textual IDs.

use std::collections::HashSet;

use xlsynth_pir::block2fn::combinational_block_to_fn;
use xlsynth_pir::dce::remove_dead_nodes;
use xlsynth_pir::ir::{self, Block, NodePayload, NodeRef, Type, Unop};
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn, eval_fn_in_package};
use xlsynth_pir::ir_fn_cone_extract::{SinkSelector, extract_fn_cone_to_params};
use xlsynth_pir::ir_outline::outline;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_rebase_ids::rebase_fn_ids;
use xlsynth_pir::ir_utils::{
    compact_and_toposort_in_place, compact_and_toposort_with_mapping_in_place,
    compact_graph_and_toposort_with_mapping_in_place, remap_payload_with,
};
use xlsynth_pir::ir_verify::{VerifyError, verify_function};
use xlsynth_pir::node_hashing::compute_function_structural_hash;
use xlsynth_pir::{FnBuilder, IrValue};

/// Interleaves a parameter with body nodes and reverses two parameter
/// positions.
fn interleaved_function() -> ir::Fn {
    let mut function = Parser::new(
        r#"fn arithmetic(x: bits[8] id=90, y: bits[8] id=4, unused: bits[8] id=50) -> bits[8] {
  one: bits[8] = literal(value=1, id=12)
  difference: bits[8] = sub(x, y, id=20)
  ret result: bits[8] = add(difference, one, id=21)
}"#,
    )
    .parse_fn()
    .unwrap();
    let order = [0, 2, 4, 1, 5, 3, 6];
    let mut old_to_new = [0; 7];
    for (index, old) in order.iter().enumerate() {
        old_to_new[*old] = index;
    }
    let remap = |node: NodeRef| NodeRef {
        index: old_to_new[node.index],
    };
    function.nodes = order
        .into_iter()
        .map(|index| {
            let mut node = function.nodes[index].clone();
            node.payload = remap_payload_with(&node.payload, |(_, operand)| remap(operand));
            node
        })
        .collect();
    function.params = function.params.iter().copied().map(remap).collect();
    function.ret_node_ref = function.ret_node_ref.map(remap);
    function
}

fn args() -> Vec<IrValue> {
    [17, 3, 99]
        .into_iter()
        .map(|value| IrValue::make_ubits(8, value).unwrap())
        .collect()
}

fn result_value(result: FnEvalResult) -> IrValue {
    match result {
        FnEvalResult::Success(result) => result.value,
        other => panic!("unexpected evaluation failure: {other:?}"),
    }
}

#[test]
fn signature_order_survives_storage_reordering_and_compaction() {
    let mut function = interleaved_function();
    assert_eq!(
        function.params,
        [
            NodeRef { index: 3 },
            NodeRef { index: 1 },
            NodeRef { index: 5 }
        ]
    );
    function.check_pir_layout_invariants().unwrap();
    verify_function(&function).unwrap();
    let expected = IrValue::make_ubits(8, 15).unwrap();
    assert_eq!(result_value(eval_fn(&function, &args())), expected);
    let text = function.to_string();
    let hash = compute_function_structural_hash(&function);
    let reparsed = Parser::new(&text).parse_fn().unwrap();
    assert_eq!(result_value(eval_fn(&reparsed, &args())), expected);
    assert_eq!(compute_function_structural_hash(&reparsed), hash);

    function = remove_dead_nodes(&function);
    compact_and_toposort_in_place(&mut function).unwrap();
    assert_eq!(
        function.params,
        [
            NodeRef { index: 1 },
            NodeRef { index: 2 },
            NodeRef { index: 3 }
        ]
    );
    assert_eq!(function.to_string(), text);
    assert_eq!(compute_function_structural_hash(&function), hash);
    assert_eq!(result_value(eval_fn(&function, &args())), expected);

    let rebased = rebase_fn_ids(&function, 100);
    assert_eq!(rebased.params, function.params);
    assert_eq!(compute_function_structural_hash(&rebased), hash);
    let mut swapped = function.clone();
    swapped.params.swap(0, 1);
    assert_ne!(compute_function_structural_hash(&swapped), hash);
    assert_eq!(
        result_value(eval_fn(&swapped, &args())),
        IrValue::make_ubits(8, 243).unwrap()
    );
}

#[test]
fn graph_and_function_compaction_reject_invalid_operands_without_mutation() {
    let cases = [
        (
            6,
            NodePayload::Unop(Unop::Identity, NodeRef { index: 99 }),
            "node 6 references missing node 99",
        ),
        (2, NodePayload::Nil, "node 6 refers to removed node 2"),
        (
            2,
            NodePayload::Unop(Unop::Identity, NodeRef { index: 6 }),
            "cycle detected: one -> result -> one",
        ),
    ];
    for (index, payload, expected) in cases {
        let mut function = interleaved_function();
        function.nodes[index].payload = payload;
        let mut graph = function.graph.clone();
        let function_before = format!("{function:?}");
        let graph_before = format!("{graph:?}");
        assert_eq!(
            compact_and_toposort_with_mapping_in_place(&mut function).unwrap_err(),
            expected
        );
        assert_eq!(format!("{function:?}"), function_before);
        assert_eq!(
            compact_graph_and_toposort_with_mapping_in_place(&mut graph).unwrap_err(),
            expected
        );
        assert_eq!(format!("{graph:?}"), graph_before);
    }
}

#[test]
fn compaction_rejects_invalid_return_references_without_mutation() {
    for index in [0, 2, usize::MAX] {
        let mut function = interleaved_function();
        function.ret_node_ref = Some(NodeRef { index });
        if index == 2 {
            function.nodes[index].payload = NodePayload::Nil;
        }
        let before = format!("{function:?}");
        assert_eq!(
            compact_and_toposort_in_place(&mut function).unwrap_err(),
            format!("function 'arithmetic' return node {index} is missing or deleted")
        );
        assert_eq!(format!("{function:?}"), before);
    }
}

#[test]
fn compaction_rejects_invalid_signature_references_without_mutation() {
    for index in [0, 1, 2, usize::MAX] {
        let mut function = interleaved_function();
        // Index 1 repeats an existing signature reference; other choices are
        // the sentinel, a computation node, or an out-of-bounds reference.
        function.params[0] = NodeRef { index };
        let before = format!("{function:?}");
        assert!(compact_and_toposort_in_place(&mut function).is_err());
        assert_eq!(format!("{function:?}"), before);
    }
}

#[test]
fn compaction_preserves_metadata_and_reports_every_node_mapping() {
    let mut function = interleaved_function();
    function
        .outer_attrs
        .push("#[ffi_proto(\"opaque\")]".to_string());
    function.inner_attrs.push("#![opaque]".to_string());
    for (index, node) in function.nodes.iter_mut().enumerate() {
        node.pos = Some(vec![ir::Pos {
            fileno: 1,
            lineno: index,
            colno: 2,
        }]);
    }
    let mut removed = function.nodes[0].clone();
    removed.text_id = 100;
    function.nodes.push(removed);
    let before = function.clone();
    let mapping = compact_and_toposort_with_mapping_in_place(&mut function).unwrap();
    assert_eq!(
        mapping,
        [
            Some(0),
            Some(2),
            Some(4),
            Some(1),
            Some(5),
            Some(3),
            Some(6),
            None
        ]
        .map(|index| index.map(|index| NodeRef { index }))
    );
    assert_eq!(function.name, before.name);
    assert_eq!(function.outer_attrs, before.outer_attrs);
    assert_eq!(function.inner_attrs, before.inner_attrs);
    for (index, mapped) in mapping.iter().enumerate() {
        if let Some(mapped) = mapped {
            let original = &before.nodes[index];
            let node = function.get_node(*mapped);
            assert_eq!(node.text_id, original.text_id);
            assert_eq!(node.name, original.name);
            assert_eq!(node.ty, original.ty);
            assert_eq!(node.pos, original.pos);
            assert_eq!(
                node.payload,
                remap_payload_with(&original.payload, |(_, operand)| mapping[operand.index]
                    .unwrap())
            );
        }
    }
    // Function construction may compact an unfinished graph without a return.
    function.ret_node_ref = None;
    compact_and_toposort_in_place(&mut function).unwrap();
    assert_eq!(function.ret_node_ref, None);
}

#[test]
fn node_metadata_is_the_only_signature_metadata() {
    let mut b = FnBuilder::new("identity");
    let value = b.param("before", Type::Bits(8)).unwrap();
    let mut function = b.build(value).unwrap();
    let param_ref = function.params[0];
    let node = function.get_node_mut(param_ref);
    node.name = Some("after".to_string());
    node.ty = Type::Bits(129);
    node.text_id = usize::MAX;
    function.ret_ty = Type::Bits(129);
    verify_function(&function).unwrap();
    let expected = format!(
        r#"fn identity(after: bits[129] id={0}) -> bits[129] {{
  ret after: bits[129] = param(name=after, id={0})
}}"#,
        usize::MAX
    );
    assert_eq!(function.to_string(), expected);
    assert_eq!(function.get_type().param_types, [Type::Bits(129)]);
    let value = IrValue::all_ones_bits(129);
    assert_eq!(result_value(eval_fn(&function, &[value.clone()])), value);
    assert!(function.drop_params(&["after".to_string()]).is_err());
}

#[test]
fn block_conversion_and_cone_extraction_follow_signature_references() {
    let function = interleaved_function();
    let block = Block::from_function(function.clone(), None).unwrap();
    assert_eq!(
        block
            .input_ports()
            .map(|port| block.port_name(port))
            .collect::<Vec<_>>(),
        ["x", "y", "unused"]
    );
    let recovered = combinational_block_to_fn(&block).unwrap();
    assert_eq!(
        result_value(eval_fn(&recovered, &args())),
        IrValue::make_ubits(8, 15).unwrap()
    );
    let extracted =
        extract_fn_cone_to_params(&function, None, SinkSelector::TextId(21), true).unwrap();
    assert_eq!(extracted.used_params, function.params[..2]);
    let cone = extracted.package.get_top_fn().unwrap();
    verify_function(cone).unwrap();
    assert_eq!(
        result_value(eval_fn(cone, &args()[..2])),
        IrValue::make_ubits(8, 15).unwrap()
    );
    let mut dropped = function.drop_params(&["unused".to_string()]).unwrap();
    assert_eq!(dropped.params, [NodeRef { index: 3 }, NodeRef { index: 1 }]);
    // Transform tombstones are removed before interpreter execution.
    compact_and_toposort_in_place(&mut dropped).unwrap();
    assert_eq!(
        result_value(eval_fn(&dropped, &args()[..2])),
        IrValue::make_ubits(8, 15).unwrap()
    );
}

#[test]
fn outlined_and_invoked_functions_keep_argument_order() {
    let function = interleaved_function();
    let mut package = Parser::new("package test").parse_package().unwrap();
    let selected = HashSet::from([NodeRef { index: 4 }]);
    let mut result = outline(&function, &selected, "outer", "inner", &mut package);
    assert_eq!(
        result
            .inner
            .param_nodes()
            .map(|node| node.param_name())
            .collect::<Vec<_>>(),
        ["x", "y"]
    );
    result.outer.check_pir_layout_invariants().unwrap();
    result.inner.check_pir_layout_invariants().unwrap();
    compact_and_toposort_in_place(&mut result.outer).unwrap();
    assert_eq!(
        result_value(eval_fn_in_package(&package, &result.outer, &args())),
        IrValue::make_ubits(8, 15).unwrap()
    );
    let mut builder = FnBuilder::new("caller");
    let arguments = args()
        .into_iter()
        .map(|value| builder.literal(value).unwrap())
        .collect::<Vec<_>>();
    let call = builder.invoke(&function, &arguments).unwrap();
    package.members = vec![ir::PackageMember::Function(function)];
    let caller = builder.build_in_package(call, &package).unwrap();
    assert_eq!(
        result_value(eval_fn_in_package(&package, &caller, &[])),
        IrValue::make_ubits(8, 15).unwrap()
    );
}

#[test]
fn verifier_rejects_invalid_signature_references() {
    let function = interleaved_function();
    for index in [0, 2, function.nodes.len()] {
        let mut invalid = function.clone();
        invalid.params[0] = NodeRef { index };
        assert!(matches!(
            verify_function(&invalid),
            Err(VerifyError::MissingParamNode { .. })
        ));
    }
    let mut repeated = function.clone();
    repeated.params.push(repeated.params[0]);
    assert!(matches!(
        verify_function(&repeated),
        Err(VerifyError::NodeSemanticViolation { .. })
    ));
    let mut unlisted = function.clone();
    unlisted.params.pop();
    assert!(matches!(
        verify_function(&unlisted),
        Err(VerifyError::ExtraParamNode { .. })
    ));
    let mut unnamed = function.clone();
    let param_ref = unnamed.params[0];
    unnamed.get_node_mut(param_ref).name = None;
    assert!(matches!(
        verify_function(&unnamed),
        Err(VerifyError::NodeSemanticViolation { .. })
    ));
    let mut duplicate_id = function.clone();
    duplicate_id.nodes[2].text_id = duplicate_id.get_param(0).text_id;
    assert!(matches!(
        verify_function(&duplicate_id),
        Err(VerifyError::DuplicateTextId { .. })
    ));
    assert!(matches!(function.get_param(0).payload, NodePayload::Param));
}
