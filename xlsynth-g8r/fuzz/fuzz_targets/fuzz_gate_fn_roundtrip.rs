// SPDX-License-Identifier: Apache-2.0
#![no_main]
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use xlsynth_g8r::gate::GateFn;
use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};
use xlsynth_g8r::gate_parser::ParseError;
use xlsynth_g8r::test_utils::structurally_equivalent;
use xlsynth_g8r::gate::{AigBitVector};

#[derive(Debug, Clone, Arbitrary)]
struct FuzzOp {
    lhs: u16,
    rhs: u16,
    lhs_neg: bool,
    rhs_neg: bool,
}

#[derive(Debug, Clone, Arbitrary)]
struct FuzzGraph {
    num_inputs: u8,
    input_width: u8,
    num_ops: u8,
    num_outputs: u8,
    ops: Vec<FuzzOp>,
    use_opt: bool,
}

fn build_graph(sample: &FuzzGraph) -> Option<GateFn> {
    let num_inputs = sample.num_inputs.min(4);
    let width = sample.input_width.min(4);
    let num_ops = sample.num_ops.min(32);
    let opts = if sample.use_opt { GateBuilderOptions::opt() } else { GateBuilderOptions::no_opt() };
    let mut builder = GateBuilder::new("fuzz_rt".to_string(), opts);
    let mut nodes = Vec::new();
    for i in 0..num_inputs {
        let bv = builder.add_input(format!("in{}", i), width as usize);
        for j in 0..width {
            nodes.push(*bv.get_lsb(j as usize));
        }
    }
    if nodes.is_empty() {
        nodes.push(builder.get_false());
    }
    for op in sample.ops.iter().take(num_ops as usize) {
        let a = nodes[(op.lhs as usize) % nodes.len()];
        let b = nodes[(op.rhs as usize) % nodes.len()];
        let a = if op.lhs_neg { builder.add_not(a) } else { a };
        let b = if op.rhs_neg { builder.add_not(b) } else { b };
        let new_node = builder.add_and_binary(a, b);
        nodes.push(new_node);
        if nodes.len() > 256 { break; }
    }
    let outputs = nodes.len().min(sample.num_outputs as usize).max(1);
    for i in 0..outputs {
        builder.add_output(format!("out{}", i), AigBitVector::from_bit(nodes[i]));
    }
    Some(builder.build())
}

fuzz_target!(|graph: FuzzGraph| {
    let _ = env_logger::builder().is_test(true).try_init();
    let Some(g) = build_graph(&graph) else { return; };
    let text = g.to_string();
    let parsed = match GateFn::from_str(&text) {
        Ok(p) => p,
        Err(_) => return,
    };
    assert!(structurally_equivalent(&g, &parsed));
});
