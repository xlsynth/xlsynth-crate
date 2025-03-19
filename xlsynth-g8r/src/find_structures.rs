// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use crate::gate::{AigNode, AigOperand, AigRef, GateFn};
use crate::use_count::get_id_to_use_count;

fn traverse_to_structure(
    root: &AigRef,
    node_to_use_count: &HashMap<AigRef, usize>,
    start_numbering: usize,
    f: &GateFn,
) -> (String, usize) {
    // We traverse down the AIG from the root node through single-use nodes only.
    // This forms a structure description.
    assert!(node_to_use_count.get(root).unwrap() == &1);
    let mut args = Vec::new();
    let mut next_number = start_numbering;

    let mut push_operand = |operand: AigOperand| {
        if let Some(1) = node_to_use_count.get(&operand.node) {
            let (structure, got_next_number) =
                traverse_to_structure(&operand.node, node_to_use_count, next_number, f);
            let structure = if operand.negated {
                format!("not({})", structure)
            } else {
                structure
            };
            args.push(structure);
            next_number = got_next_number;
        } else {
            let structure = format!("x{}", next_number);
            let structure = if operand.negated {
                format!("not({})", structure)
            } else {
                structure
            };
            args.push(structure.clone());
            next_number += 1;
        }
    };

    let structure: String = match f.gates[root.id] {
        AigNode::And2 { a, b, .. } => {
            push_operand(a);
            push_operand(b);
            format!("AND({})", args.join(","))
        }
        AigNode::Input { .. } => {
            let result = format!("x{}", next_number);
            next_number += 1;
            result
        }
        AigNode::Literal(value) => {
            format!("{}", value)
        }
    };
    (structure, next_number)
}

pub fn find_structures(f: &GateFn) -> HashMap<String, usize> {
    let node_to_use_count = get_id_to_use_count(f);
    let mut structure_to_count = HashMap::new();
    for (node, count) in node_to_use_count.iter() {
        if *count == 1 {
            log::info!("node {:?} has use count {}", node, count);
            let (structure, _next_number) = traverse_to_structure(node, &node_to_use_count, 0, f);
            *structure_to_count.entry(structure).or_insert(0) += 1;
        }
    }
    structure_to_count
}

#[cfg(test)]
mod tests {
    use crate::gate::{AigBitVector, GateBuilder};

    use super::*;

    #[test]
    fn test_simple_two_ands() {
        let _ = env_logger::builder().is_test(true).try_init();
        let mut builder = GateBuilder::new("test_fn".to_string(), false);
        let inputs = builder.add_input("in".to_string(), 3);
        let inputs0 = inputs.get_lsb(0);
        let inputs1 = inputs.get_lsb(1);
        let inputs2 = inputs.get_lsb(2);
        let op1 = builder.add_and_binary(*inputs0, *inputs1);
        let op2 = builder.add_and_binary(*inputs1, *inputs2);
        builder.add_output(
            "out".to_string(),
            AigBitVector::from_lsb_is_index_0(&[op1, op2]),
        );
        let f = builder.build();
        let id_to_use_count = get_id_to_use_count(&f);

        let output_node = f.outputs[0].bit_vector.get_lsb(0).node;
        assert_eq!(
            traverse_to_structure(&output_node, &id_to_use_count, 0, &f),
            ("AND(x0,x1)".to_string(), 2)
        );

        let structure_to_count = find_structures(&f);
        assert_eq!(structure_to_count.len(), 2);
        let mut got = structure_to_count
            .iter()
            .map(|(s, c)| (s.clone(), *c))
            .collect::<Vec<_>>();
        got.sort_by_key(|(s, _)| s.clone());
        assert_eq!(
            got,
            vec![("AND(x0,x1)".to_string(), 2), ("x0".to_string(), 2)]
        );
    }
}
