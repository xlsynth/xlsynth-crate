// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::aig::{AigBitVector, AigNode, AigOperand, AigRef, GateFn};
use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};

/// Creates a small three-input AND tree for export and evaluation tests.
pub fn and3() -> GateFn {
    let mut builder = GateBuilder::new("and3".to_string(), GateBuilderOptions::no_opt());
    let a = *builder.add_input("a".to_string(), 1).get_lsb(0);
    let b = *builder.add_input("b".to_string(), 1).get_lsb(0);
    let c = *builder.add_input("c".to_string(), 1).get_lsb(0);
    let ab = builder.add_and_binary(a, b);
    let abc = builder.add_and_binary(ab, c);
    builder.add_output("result".to_string(), AigBitVector::from_bit(abc));
    builder.build()
}

/// Reverses all node slots and remaps references, including source node zero.
pub fn reverse_node_storage(graph: &mut GateFn) {
    let node_count = graph.gates.len();
    let remap = |mut operand: AigOperand| {
        operand.node.id = node_count - 1 - operand.node.id;
        operand
    };
    graph.gates.reverse();
    for node in &mut graph.gates {
        if let AigNode::And2 { a, b, .. } = node {
            *a = remap(*a);
            *b = remap(*b);
        }
    }
    for input in &mut graph.inputs {
        input.bit_vector = AigBitVector::from_lsb_is_index_0(
            &input
                .bit_vector
                .iter_lsb_to_msb()
                .map(|operand| remap(*operand))
                .collect::<Vec<_>>(),
        );
    }
    for output in &mut graph.outputs {
        output.bit_vector = AigBitVector::from_lsb_is_index_0(
            &output
                .bit_vector
                .iter_lsb_to_msb()
                .map(|operand| remap(*operand))
                .collect::<Vec<_>>(),
        );
    }
}

/// Creates a graph exercising wide vectors, polarity, sharing, and empty ports.
pub fn interface_graph() -> GateFn {
    let mut builder = GateBuilder::new("interface_graph".to_string(), GateBuilderOptions::no_opt());
    builder.add_input("empty".to_string(), 0);
    let flag = *builder.add_input("flag".to_string(), 1).get_lsb(0);
    let data = builder.add_input("data".to_string(), 65);
    let unused = builder.add_input("unused".to_string(), 3);
    let shared = builder.add_and_binary(flag, *data.get_lsb(0));
    let combined = builder.add_and_binary(shared, data.get_lsb(64).negate());
    builder.add_and_binary(*unused.get_lsb(0), *unused.get_lsb(1));
    let true_ref = AigOperand {
        node: AigRef {
            id: builder.gates.len(),
        },
        negated: false,
    };
    builder.gates.push(AigNode::Literal {
        value: true,
        pir_node_ids: Default::default(),
    });
    builder.add_output(
        "empty_result".to_string(),
        AigBitVector::from_lsb_is_index_0(&[]),
    );
    let mirrored = data
        .iter_msb_to_lsb()
        .enumerate()
        .map(|(index, operand)| {
            if index % 2 == 0 {
                operand.negate()
            } else {
                *operand
            }
        })
        .collect::<Vec<_>>();
    builder.add_output(
        "mirrored".to_string(),
        AigBitVector::from_lsb_is_index_0(&mirrored),
    );
    builder.add_output(
        "flags".to_string(),
        AigBitVector::from_lsb_is_index_0(&[
            flag,
            flag.negate(),
            shared,
            shared.negate(),
            combined,
            true_ref,
            true_ref.negate(),
        ]),
    );
    let mut graph = builder.build();
    if let AigNode::Input {
        name, lsb_index, ..
    } = &mut graph.gates[flag.node.id]
    {
        *name = "metadata_only".to_string();
        *lsb_index = 17;
    }
    reverse_node_storage(&mut graph);
    graph
}
