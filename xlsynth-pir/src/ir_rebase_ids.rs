// SPDX-License-Identifier: Apache-2.0

use crate::ir::{Fn as IrFn, NodePayload, ParamId};

fn add_base(value: usize, base: usize) -> usize {
    value
        .checked_add(base)
        .expect("rebasing ids overflowed usize")
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

    for param in &mut rebased.params {
        let new_id = add_base(param.id.get_wrapped_id(), base);
        param.id = ParamId::new(new_id);
    }

    for node in &mut rebased.nodes {
        match &mut node.payload {
            NodePayload::GetParam(param_id) => {
                let rebased_id = add_base(param_id.get_wrapped_id(), base);
                *param_id = ParamId::new(rebased_id);
                node.text_id = rebased_id;
            }
            NodePayload::Nil => {
                // Reserved Nil node retains text_id 0.
            }
            _ => {
                node.text_id = add_base(node.text_id, base);
            }
        }
    }

    rebased
}

#[cfg(test)]
mod tests {
    use super::rebase_fn_ids;
    use crate::ir::{self, NodePayload};
    use crate::ir_parser::Parser;
    use crate::ir_validate::validate_fn;

    fn parse_function(ir: &str) -> ir::Fn {
        let mut parser = Parser::new(ir);
        parser.parse_fn().expect("function should parse")
    }

    fn package_with_function(f: &ir::Fn) -> ir::Package {
        ir::Package {
            name: "test_pkg".to_string(),
            file_table: ir::FileTable::new(),
            members: vec![ir::PackageMember::Function(f.clone())],
            top_name: Some(f.name.clone()),
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

        validate_fn(&rebased, &package).expect("rebased function should validate");
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
        validate_fn(&rebased, &package).expect("sparse ids should still validate");
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
