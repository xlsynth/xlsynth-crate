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
use xlsynth_pir::ir_verify::{
    VerifyError, verify_block_in_package, verify_function, verify_function_in_package,
    verify_function_signature, verify_package,
};
use xlsynth_pir::matching_ged::{MatchAction, NewNodeRef, OldNodeRef, compute_parameter_matches};
use xlsynth_pir::node_hashing::compute_function_structural_hash;
use xlsynth_pir::{BuilderError, FnBuilder, IrValue};

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

#[test]
fn layout_and_verifier_share_signature_checks() {
    let mutations: [fn(&mut ir::Fn); 8] = [
        |f| f.nodes.clear(),
        |f| f.nodes[0] = f.get_param(0).clone(),
        |f| f.params[0] = NodeRef { index: usize::MAX },
        |f| f.params[0] = NodeRef { index: 0 },
        |f| f.params.push(f.params[0]),
        |f| {
            f.params.pop();
        },
        |f| {
            let parameter = f.params[0];
            f.get_node_mut(parameter).name = None;
        },
        |f| {
            let parameter = f.params[0];
            f.get_node_mut(parameter).name = f.get_param(1).name.clone();
        },
    ];
    for mutate in mutations {
        let mut function = interleaved_function();
        mutate(&mut function);
        let expected = verify_function_signature(&function).unwrap_err();
        assert_eq!(
            function.check_pir_layout_invariants(),
            Err(expected.to_string())
        );
        assert_eq!(verify_function(&function), Err(expected));
    }
    // A signature can be checked independently of body topology or completion.
    let mut unfinished = interleaved_function();
    unfinished.ret_node_ref = None;
    unfinished.nodes[2].payload = NodePayload::Unop(Unop::Identity, NodeRef { index: 6 });
    verify_function_signature(&unfinished).unwrap();
    unfinished.check_pir_layout_invariants().unwrap();
}

#[test]
fn invalid_callee_signatures_return_errors_in_either_member_order() {
    let package = Parser::new(
        r#"package test
top fn caller(x: bits[8] id=1) -> bits[8] {
  ret call: bits[8] = invoke(x, to_apply=callee, id=2)
}
fn callee(y: bits[8] id=3) -> bits[8] {
  ret y: bits[8] = param(name=y, id=3)
}
"#,
    )
    .parse_package()
    .unwrap();
    verify_package(&package).unwrap();
    for index in [0, usize::MAX] {
        let mut invalid = package.clone();
        invalid.get_fn_mut("callee").unwrap().params[0] = NodeRef { index };
        for _ in 0..2 {
            let expected = VerifyError::MissingParamNode {
                func: "callee".to_string(),
                node_ref: NodeRef { index },
            };
            assert_eq!(
                verify_function_in_package(invalid.get_fn("caller").unwrap(), &invalid)
                    .unwrap_err(),
                expected,
            );
            assert_eq!(verify_package(&invalid).unwrap_err(), expected);
            invalid.members.reverse();
        }
    }
}

#[test]
fn invalid_loop_body_signatures_return_errors_before_type_checks() {
    let package = Parser::new(
        r#"package test
top fn caller(x: bits[8] id=1) -> bits[8] {
  ret loop: bits[8] = counted_for(x, trip_count=2, stride=1, body=body, id=2)
}
fn body(i: bits[1] id=3, carry: bits[8] id=4) -> bits[8] {
  ret carry: bits[8] = param(name=carry, id=4)
}
"#,
    )
    .parse_package()
    .unwrap();
    verify_package(&package).unwrap();
    for parameter in 0..2 {
        let mut invalid = package.clone();
        let missing = NodeRef { index: usize::MAX };
        invalid.get_fn_mut("body").unwrap().params[parameter] = missing;
        let expected = VerifyError::MissingParamNode {
            func: "body".to_string(),
            node_ref: missing,
        };
        assert_eq!(
            verify_function_in_package(invalid.get_fn("caller").unwrap(), &invalid).unwrap_err(),
            expected,
        );
        assert_eq!(verify_package(&invalid).unwrap_err(), expected);
    }
}

#[test]
fn invalid_external_signatures_return_errors_before_port_mapping() {
    let mut package = Parser::new(r#"package test
top block wrapper(input: bits[8], result: bits[8]) {
  instantiation external_instance(foreign_function=external_fn, kind=extern)
  input: bits[8] = input_port(name=input, id=1)
  instantiation_input.2: () = instantiation_input(input, instantiation=external_instance, port_name=value, id=2)
  instantiation_output.3: bits[8] = instantiation_output(instantiation=external_instance, port_name=return, id=3)
  result: () = output_port(instantiation_output.3, name=result, id=4)
}
fn external_fn(value: bits[8] id=5) -> bits[8] {
  ret value: bits[8] = param(name=value, id=5)
}
"#).parse_package().unwrap();
    verify_package(&package).unwrap();
    let missing = NodeRef { index: usize::MAX };
    package.get_fn_mut("external_fn").unwrap().params[0] = missing;
    let expected = VerifyError::MissingParamNode {
        func: "external_fn".to_string(),
        node_ref: missing,
    };
    assert_eq!(
        verify_block_in_package(package.get_block("wrapper").unwrap(), &package, 0).unwrap_err(),
        expected,
    );
    assert_eq!(verify_package(&package).unwrap_err(), expected);
}

#[test]
fn builder_rejects_invalid_callee_signatures_without_becoming_unusable() {
    let mut callee = interleaved_function();
    let missing = NodeRef { index: usize::MAX };
    callee.params[0] = missing;
    let expected = BuilderError::InvalidOperation(
        VerifyError::MissingParamNode {
            func: callee.name.clone(),
            node_ref: missing,
        }
        .to_string(),
    );
    let mut builder = FnBuilder::new("caller");
    let arg = builder.param("x", Type::Bits(8)).unwrap();
    assert_eq!(
        builder.invoke(&callee, &[arg, arg, arg]).unwrap_err(),
        expected
    );
    assert_eq!(
        builder.counted_for(arg, 2, 1, &callee, &[arg]).unwrap_err(),
        expected
    );
    builder.build(arg).unwrap();
}

#[test]
fn parameter_matches_follow_names_and_signature_order() {
    let old = interleaved_function();
    let mut new = old.clone();
    compact_and_toposort_in_place(&mut new).unwrap();
    new.params.reverse();
    new.nodes[3].name = Some("new_only".to_string());
    new.ret_node_ref = Some(NodeRef { index: 2 });
    verify_function(&new).unwrap();
    assert_eq!(
        compute_parameter_matches(&old, &new),
        vec![
            MatchAction::MatchNodes {
                old_index: OldNodeRef(3),
                new_index: NewNodeRef(1),
                new_operands: vec![],
                is_new_return: false,
            },
            MatchAction::MatchNodes {
                old_index: OldNodeRef(1),
                new_index: NewNodeRef(2),
                new_operands: vec![],
                is_new_return: true,
            },
        ]
    );
}

#[test]
fn body_param_nodes_only_resolve_matching_signature_nodes() {
    for (body, expected) in [
        (
            "ret y: bits[8] = param(name=y, id=2)",
            "unknown parameter name in param node: y",
        ),
        (
            "ret x: bits[8] = param(name=x, id=2)",
            "param name/id mismatch: name=x id=2",
        ),
        (
            "ret x: bits[16] = param(name=x, id=1)",
            "param id=1 type mismatch: header bits[8] vs node bits[16]",
        ),
        (
            r#"y: bits[8] = identity(x, id=2)
  ret y: bits[8] = param(name=y, id=2)"#,
            "param id=2 does not reference a signature parameter",
        ),
    ] {
        let text = format!("fn f(x: bits[8] id=1) -> bits[8] {{\n  {body}\n}}");
        assert_eq!(
            Parser::new(&text).parse_fn().unwrap_err().to_string(),
            format!("ParseError: {expected}")
        );
    }
    let text = r#"fn f(x: bits[8] id=7) -> bits[8] {
  ret x: bits[8] = param(name=x, id=7)
}"#;
    let function = Parser::new(text).parse_fn().unwrap();
    assert_eq!(function.nodes.len(), 2);
    assert_eq!(function.ret_node_ref, Some(function.params[0]));
    assert_eq!(function.to_string(), text);
}
