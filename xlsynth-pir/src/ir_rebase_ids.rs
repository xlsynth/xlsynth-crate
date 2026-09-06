// SPDX-License-Identifier: Apache-2.0

use crate::ir::{Fn as IrFn, NodePayload, Package, ParamId};

/// A checked ID allocation or rebasing operation exceeded the ID space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdRebaseError;

impl std::fmt::Display for IdRebaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("node ID allocation or rebasing overflows usize")
    }
}

impl std::error::Error for IdRebaseError {}

/// Rebases IDs without cloning the function, leaving it unchanged on overflow.
///
/// A zero base is allowed. Node indices, parameter order, and the reserved Nil
/// node's ID are unchanged; parameter IDs and their GetParam nodes move
/// together.
pub fn rebase_fn_ids_in_place(f: &mut IrFn, base: usize) -> Result<(), IdRebaseError> {
    for param in &f.params {
        param
            .id
            .get_wrapped_id()
            .checked_add(base)
            .ok_or(IdRebaseError)?;
    }
    for node in &f.nodes {
        let id = match node.payload {
            NodePayload::Nil => continue,
            NodePayload::GetParam(id) => id.get_wrapped_id(),
            _ => node.text_id,
        };
        id.checked_add(base).ok_or(IdRebaseError)?;
    }
    for param in &mut f.params {
        param.id = ParamId::new(param.id.get_wrapped_id() + base);
    }
    for node in &mut f.nodes {
        match &mut node.payload {
            NodePayload::GetParam(id) => {
                *id = ParamId::new(id.get_wrapped_id() + base);
                node.text_id = id.get_wrapped_id();
            }
            NodePayload::Nil => {
                // The synthetic sentinel is not emitted and never needs
                // rebasing.
            }
            _ => node.text_id += base,
        }
    }
    Ok(())
}

/// Finds the highest allocated ID across all function and block nodes.
pub fn package_max_emitted_node_id(package: &Package) -> usize {
    package
        .members
        .iter()
        .flat_map(|member| &member.graph().nodes)
        .map(|node| node.text_id)
        .max()
        .unwrap_or(0)
}

/// Rebases all block node IDs without changing graph or interface references.
pub fn rebase_block_ids_in_place(
    block: &mut crate::ir::Block,
    base: usize,
) -> Result<(), IdRebaseError> {
    for node in &block.nodes {
        if !matches!(node.payload, NodePayload::Nil) {
            node.text_id.checked_add(base).ok_or(IdRebaseError)?;
        }
    }
    for node in &mut block.nodes {
        if !matches!(node.payload, NodePayload::Nil) {
            node.text_id += base;
        }
    }
    Ok(())
}

/// Returns a clone of `f` with all ParamIds and node text ids rebased by
/// `base`.
///
/// The function topology and payloads are preserved (except for the adjusted
/// `GetParam` payload ids). All node references remain intact because the node
/// list ordering is unchanged. The reserved Nil node keeps its original
/// `text_id`.
///
/// # Panics
/// Panics if `base` is zero or if rebasing would overflow `usize`.
pub fn rebase_fn_ids(f: &IrFn, base: usize) -> IrFn {
    assert!(base >= 1, "base must be at least 1, got {}", base);

    let mut rebased = f.clone();
    rebase_fn_ids_in_place(&mut rebased, base).expect("rebasing ids overflowed usize");
    rebased
}

#[cfg(test)]
mod tests {
    use super::{rebase_block_ids_in_place, rebase_fn_ids, rebase_fn_ids_in_place};
    use crate::ir::{self, NodePayload};
    use crate::ir_parser::Parser;
    use crate::ir_verify::verify_function_in_package;

    #[test]
    fn block_rebase_includes_port_nodes_and_is_atomic() {
        let mut block = crate::ir::Block::new("b");
        let input = block.add_input_port("x", crate::ir::Type::Bits(8)).unwrap();
        let output = block.add_output_port("y", input).unwrap();
        let original = block.to_string();
        let nodes_ptr = block.nodes.as_ptr();
        assert!(rebase_block_ids_in_place(&mut block, usize::MAX).is_err());
        assert_eq!(block.to_string(), original);
        rebase_block_ids_in_place(&mut block, 10).unwrap();
        assert_eq!(block.nodes.as_ptr(), nodes_ptr);
        assert_eq!(block.get_node(input).text_id, 11);
        assert_eq!(block.get_node(output).text_id, 12);
        assert_eq!(block.nodes[0].text_id, 0);
        assert_eq!(block.output_value(output), input);
        assert_eq!(
            block.ports,
            vec![
                crate::ir::BlockPort::Input(input),
                crate::ir::BlockPort::Output(output)
            ]
        );
    }

    #[test]
    fn package_id_validation_includes_block_output_nodes() {
        let mut builder = crate::FnBuilder::new("f");
        let parameter = builder.param("x", crate::ir::Type::Bits(8)).unwrap();
        let function = builder.build(parameter).unwrap();
        let mut block = crate::ir::Block::new("b");
        let input = block.add_input_port("x", crate::ir::Type::Bits(8)).unwrap();
        let output = block.add_output_port("y", input).unwrap();
        rebase_block_ids_in_place(&mut block, 1).unwrap();
        let mut package = crate::ir::Package {
            name: "p".to_string(),
            file_table: crate::ir::FileTable::new(),
            top: None,
            members: vec![
                crate::ir::PackageMember::Function(function),
                crate::ir::PackageMember::Block(block),
            ],
        };
        crate::ir_verify::verify_package(&package).unwrap();
        package
            .get_block_mut("b")
            .unwrap()
            .get_node_mut(output)
            .text_id = 1;
        assert!(crate::ir_verify::verify_package(&package).is_err());
    }

    fn parse_function(ir: &str) -> ir::Fn {
        let mut parser = Parser::new(ir);
        parser.parse_fn().expect("function should parse")
    }

    fn package_with_function(f: &ir::Fn) -> ir::Package {
        ir::Package {
            name: "test_pkg".to_string(),
            file_table: ir::FileTable::new(),
            members: vec![ir::PackageMember::Function(f.clone())],
            top: Some((f.name.clone(), ir::MemberType::Function)),
        }
    }

    fn sample_two_param_function() -> ir::Fn {
        parse_function(
            r#"
fn sample(lhs: bits[8] id=1, rhs: bits[8] id=3) -> bits[8] {
  sum: bits[8] = add(lhs, rhs, id=30)
  ret negated: bits[8] = neg(sum, id=40)
}
"#,
        )
    }

    fn sparse_param_function() -> ir::Fn {
        parse_function(
            r#"
fn sparse(alpha: bits[8] id=10, omega: bits[8] id=1000) -> bits[8] {
  ret add.6000: bits[8] = add(alpha, omega)
}
"#,
        )
    }

    fn zero_param_function() -> ir::Fn {
        parse_function(
            r#"
fn no_params() -> bits[32] {
  ret literal.7: bits[32] = literal(value=0x2a, id=7)
}
"#,
        )
    }

    #[test]
    fn in_place_rebase_is_atomic_and_does_not_reallocate_nodes() {
        let mut function = sample_two_param_function();
        let before = function.to_string();
        assert!(rebase_fn_ids_in_place(&mut function, usize::MAX).is_err());
        assert_eq!(function.to_string(), before);
        let nodes = function.nodes.as_ptr();
        rebase_fn_ids_in_place(&mut function, 0).unwrap();
        assert_eq!(function.to_string(), before);
        rebase_fn_ids_in_place(&mut function, 7).unwrap();
        assert_eq!(function.nodes.as_ptr(), nodes);
        assert_eq!(function.params[0].id.get_wrapped_id(), 8);
        assert_eq!(function.nodes.last().unwrap().text_id, 47);
    }

    fn assert_structure_preserved_except_ids(original: &ir::Fn, rebased: &ir::Fn, base: usize) {
        assert_eq!(original.name, rebased.name);
        assert_eq!(original.ret_ty, rebased.ret_ty);
        assert_eq!(original.ret_node_ref, rebased.ret_node_ref);

        assert_eq!(original.params.len(), rebased.params.len());
        for (orig, rebased_param) in original.params.iter().zip(&rebased.params) {
            assert_eq!(orig.name, rebased_param.name);
            assert_eq!(orig.ty, rebased_param.ty);
            let expected_id = orig.id.get_wrapped_id() + base;
            assert_eq!(rebased_param.id.get_wrapped_id(), expected_id);
        }

        assert_eq!(original.nodes.len(), rebased.nodes.len());
        for (orig_node, rebased_node) in original.nodes.iter().zip(&rebased.nodes) {
            assert_eq!(orig_node.name, rebased_node.name);
            assert_eq!(orig_node.ty, rebased_node.ty);
            assert_eq!(orig_node.pos, rebased_node.pos);
            match (&orig_node.payload, &rebased_node.payload) {
                (NodePayload::GetParam(orig_pid), NodePayload::GetParam(rebased_pid)) => {
                    assert_eq!(
                        rebased_pid.get_wrapped_id(),
                        orig_pid.get_wrapped_id() + base
                    );
                    assert_eq!(rebased_node.text_id, rebased_pid.get_wrapped_id());
                }
                (NodePayload::Nil, NodePayload::Nil) => {
                    assert_eq!(rebased_node.text_id, orig_node.text_id);
                }
                (lhs, rhs) => {
                    assert_eq!(lhs, rhs);
                    assert_eq!(rebased_node.text_id, orig_node.text_id + base);
                }
            }
        }
    }

    #[test]
    fn rebase_updates_param_ids_and_getparam_nodes() {
        let original = sample_two_param_function();
        let base = 10;
        let rebased = rebase_fn_ids(&original, base);

        for (orig, rebased_param) in original.params.iter().zip(&rebased.params) {
            assert_eq!(orig.name, rebased_param.name);
            assert_eq!(orig.ty, rebased_param.ty);
            assert_eq!(
                rebased_param.id.get_wrapped_id(),
                orig.id.get_wrapped_id() + base
            );
        }

        let mut seen_getparam = 0;
        for (orig_node, rebased_node) in original.nodes.iter().zip(&rebased.nodes) {
            if let (NodePayload::GetParam(orig_pid), NodePayload::GetParam(rebased_pid)) =
                (&orig_node.payload, &rebased_node.payload)
            {
                seen_getparam += 1;
                assert_eq!(
                    rebased_pid.get_wrapped_id(),
                    orig_pid.get_wrapped_id() + base
                );
                assert_eq!(rebased_node.text_id, rebased_pid.get_wrapped_id());
            }
        }
        assert!(seen_getparam > 0, "expected GetParam nodes in fixture");

        // Ensure the original function is unchanged.
        assert_eq!(original.params[0].id.get_wrapped_id(), 1);
    }

    #[test]
    fn rebasing_shifts_non_param_node_ids_and_keeps_nil() {
        let original = sample_two_param_function();
        let base = 25;
        let rebased = rebase_fn_ids(&original, base);

        for (orig_node, rebased_node) in original.nodes.iter().zip(&rebased.nodes) {
            match &orig_node.payload {
                NodePayload::Nil => {
                    assert_eq!(rebased_node.text_id, orig_node.text_id);
                }
                NodePayload::GetParam(_) => {
                    // Checked by other test; ensure invariant holds here.
                    assert_eq!(rebased_node.text_id >= base, true);
                }
                _ => {
                    assert_eq!(rebased_node.text_id, orig_node.text_id + base);
                    assert_eq!(orig_node.payload, rebased_node.payload);
                }
            }
        }
    }

    #[test]
    fn rebased_function_passes_validation() {
        let original = sample_two_param_function();
        let base = 5;
        let rebased = rebase_fn_ids(&original, base);
        let package = package_with_function(&rebased);

        verify_function_in_package(&rebased, &package).expect("rebased function should verify");
    }

    #[test]
    fn round_trip_parse_after_rebasing() {
        let original = sample_two_param_function();
        let base = 100;
        let rebased = rebase_fn_ids(&original, base);

        let printed = format!("{}", rebased);
        let reparsed = parse_function(&printed);

        assert_structure_preserved_except_ids(&rebased, &reparsed, 0);
    }

    #[test]
    fn sparse_param_ids_remain_spaced_after_rebase() {
        let original = sparse_param_function();
        let base = 1_234;
        let rebased = rebase_fn_ids(&original, base);

        assert_eq!(rebased.params.len(), 2);
        let original_gap =
            original.params[1].id.get_wrapped_id() - original.params[0].id.get_wrapped_id();
        let rebased_gap =
            rebased.params[1].id.get_wrapped_id() - rebased.params[0].id.get_wrapped_id();
        assert_eq!(rebased_gap, original_gap);

        let package = package_with_function(&rebased);
        verify_function_in_package(&rebased, &package).expect("sparse ids should still verify");
    }

    #[test]
    fn zero_param_functions_shift_only_non_param_nodes() {
        let original = zero_param_function();
        let base = 77;
        let rebased = rebase_fn_ids(&original, base);

        assert_eq!(rebased.params.len(), 0);
        for (orig_node, rebased_node) in original.nodes.iter().zip(&rebased.nodes) {
            match &orig_node.payload {
                NodePayload::Nil => assert_eq!(rebased_node.text_id, orig_node.text_id),
                _ => assert_eq!(rebased_node.text_id, orig_node.text_id + base),
            }
        }
    }

    #[test]
    fn structure_preserved_except_ids() {
        let original = sample_two_param_function();
        let base = 9;
        let rebased = rebase_fn_ids(&original, base);

        assert_structure_preserved_except_ids(&original, &rebased, base);
    }
}
