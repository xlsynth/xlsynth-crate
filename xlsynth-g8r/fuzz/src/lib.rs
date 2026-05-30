// SPDX-License-Identifier: Apache-2.0
use arbitrary::Arbitrary;
use xlsynth_g8r::aig::{AigBitVector, GateBuilder, GateBuilderOptions, GateFn};
use xlsynth_pir::ir::{Fn as IrFn, Package};
use xlsynth_pir::ir_random::{
    generate_fn, generate_same_signature_pair, DepletableBytes, OperationSet, RandomFnOptions,
    RandomOperation, StopPolicy,
};

#[derive(Debug, Clone, Arbitrary)]
pub struct FuzzOp {
    pub lhs: u16,
    pub rhs: u16,
    pub lhs_neg: bool,
    pub rhs_neg: bool,
}

#[derive(Debug, Clone, Arbitrary)]
pub struct FuzzGraph {
    pub num_inputs: u8,
    pub input_width: u8,
    pub num_ops: u8,
    pub num_outputs: u8,
    pub ops: Vec<FuzzOp>,
    pub use_opt: bool,
}

pub fn build_graph(sample: &FuzzGraph) -> Option<GateFn> {
    let num_inputs = sample.num_inputs.min(4);
    let width = sample.input_width.min(4);
    let num_ops = sample.num_ops.min(32);
    let opts = if sample.use_opt {
        GateBuilderOptions::opt()
    } else {
        GateBuilderOptions::no_opt()
    };
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
        if nodes.len() > 256 {
            break;
        }
    }
    let outputs = nodes.len().min(sample.num_outputs as usize).max(1);
    for i in 0..outputs {
        builder.add_output(format!("out{}", i), AigBitVector::from_bit(nodes[i]));
    }
    Some(builder.build())
}

fn operations_without_product_pairs_or_extensions() -> OperationSet {
    let operations = OperationSet::new(OperationSet::all_supported().iter().filter(|operation| {
        !matches!(
            operation,
            RandomOperation::Umulp
                | RandomOperation::Smulp
                | RandomOperation::ExtCarryOut
                | RandomOperation::ExtPrioEncode
                | RandomOperation::ExtClz
                | RandomOperation::ExtNormalizeLeft
                | RandomOperation::ExtMaskLow
                | RandomOperation::ExtNaryAdd
        )
    }));
    operations
}

fn upstream_formal_random_pir_options(max_nodes: usize) -> RandomFnOptions {
    RandomFnOptions {
        max_nodes,
        max_bit_width: 8,
        allow_arbitrary_width_multiply: true,
        allow_gate: true,
        enabled_operations: operations_without_product_pairs_or_extensions(),
        ..RandomFnOptions::default()
    }
}

fn gatify_random_pir_options(max_nodes: usize) -> RandomFnOptions {
    let operations = OperationSet::new(
        OperationSet::all_supported()
            .iter()
            .filter(|operation| !matches!(operation, RandomOperation::Umulp | RandomOperation::Smulp)),
    );
    RandomFnOptions {
        max_nodes,
        max_bit_width: 8,
        allow_arbitrary_width_multiply: true,
        allow_extension_ops: true,
        enabled_operations: operations,
        ..RandomFnOptions::default()
    }
}

fn full_random_pir_options(max_nodes: usize) -> RandomFnOptions {
    RandomFnOptions {
        max_nodes,
        allow_zero_width_bits: true,
        allow_arbitrary_width_multiply: true,
        allow_gate: true,
        allow_extension_ops: true,
        ..RandomFnOptions::default()
    }
}

/// Generates standard upstream-compatible PIR for libxls and formal checks.
pub fn generate_upstream_formal_random_pir_package(data: &[u8], package_name: &str) -> Package {
    let options = upstream_formal_random_pir_options(32);
    let mut entropy = DepletableBytes::new(data);
    generate_fn(&mut entropy, &options, StopPolicy::WhenEntropyDepleted)
        .expect("fixed standard PIR fuzz options should construct a valid function")
        .into_top_package(package_name)
}

/// Generates PIR supported by gatify, including PIR extension operations.
pub fn generate_gatify_random_pir_package(data: &[u8], package_name: &str) -> Package {
    let options = gatify_random_pir_options(32);
    let mut entropy = DepletableBytes::new(data);
    generate_fn(&mut entropy, &options, StopPolicy::WhenEntropyDepleted)
        .expect("fixed gatify PIR fuzz options should construct a valid function")
        .into_top_package(package_name)
}

/// Generates upstream-compatible PIR for differential interpreter checks.
pub fn generate_upstream_eval_random_pir_package(data: &[u8], package_name: &str) -> Package {
    generate_upstream_formal_random_pir_package(data, package_name)
}

/// Generates two upstream-compatible functions with matching signatures for
/// toolchain-backed equivalence properties.
pub fn generate_upstream_formal_random_pir_pair(data: &[u8]) -> (IrFn, IrFn) {
    let options = upstream_formal_random_pir_options(64);
    let (mut first_entropy, mut second_entropy) = DepletableBytes::split(data);
    let (first, second) = generate_same_signature_pair(
        &mut first_entropy,
        &mut second_entropy,
        &options,
        StopPolicy::WhenEntropyDepleted,
    )
    .expect("fixed paired PIR fuzz options should construct matching functions");
    (first.function, second.function)
}

/// Generates two functions with matching signatures across the full PIR
/// random-function surface for structural properties.
pub fn generate_full_random_pir_pair(data: &[u8]) -> (IrFn, IrFn) {
    let options = full_random_pir_options(64);
    let (mut first_entropy, mut second_entropy) = DepletableBytes::split(data);
    let (first, second) = generate_same_signature_pair(
        &mut first_entropy,
        &mut second_entropy,
        &options,
        StopPolicy::WhenEntropyDepleted,
    )
    .expect("fixed paired PIR fuzz options should construct matching functions");
    (first.function, second.function)
}
