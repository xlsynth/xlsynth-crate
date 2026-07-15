// SPDX-License-Identifier: Apache-2.0

#![no_main]

use std::collections::BTreeMap;

use libfuzzer_sys::fuzz_target;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use xlsynth::{IrBits, IrValue};
use xlsynth_g8r::aig_sim::sequential::{self, SequentialState};
use xlsynth_g8r::block2sequential::block_package_to_sequential_gate_fn;
use xlsynth_g8r::gatify::ir2gate::GatifyOptions;
use xlsynth_pir::ir::{BlockMetadata, Fn, NodePayload, NodeRef, Type};
use xlsynth_pir::ir_eval::{self, FnEvalResult};
use xlsynth_pir::ir_random::{
    DepletableBytes, OperationSet, RandomBlockOptions, RandomFnOptions, RandomOperation,
    RandomBlockResetTiming, StopPolicy, generate_block_package,
};
use xlsynth_pir::ir_value_utils::flatten_ir_value_to_lsb0_bits_for_type;
use xlsynth_pir::random_inputs::generate_uniform_value_with_rng;

const CYCLE_COUNT: usize = 32;

#[derive(Debug, Clone, Copy)]
struct RegisterWriteRefs {
    arg: NodeRef,
    load_enable: Option<NodeRef>,
    reset: Option<NodeRef>,
}

fn fuzz_block_options() -> RandomBlockOptions {
    let operations = OperationSet::new(
        OperationSet::all_supported()
            .iter()
            .filter(|operation| {
                !matches!(
                    operation,
                    RandomOperation::Umulp | RandomOperation::Smulp
                )
            }),
    );
    RandomBlockOptions {
        max_input_ports: 6,
        max_output_ports: 4,
        max_registers: 4,
        reset_timing: RandomBlockResetTiming::Synchronous,
        function_options: RandomFnOptions {
            max_nodes: 64,
            max_bit_width: 16,
            allow_arbitrary_width_multiply: true,
            enabled_operations: operations,
            ..RandomFnOptions::default()
        },
        ..RandomBlockOptions::default()
    }
}

fn block_output_refs(block: &Fn, metadata: &BlockMetadata) -> Vec<NodeRef> {
    let ret_ref = block
        .ret_node_ref
        .expect("generated block should have a return node");
    match metadata.output_names.len() {
        0 => Vec::new(),
        1 => vec![ret_ref],
        _ => {
            let NodePayload::Tuple(outputs) = &block.get_node(ret_ref).payload else {
                panic!("generated multi-output block should return a tuple");
            };
            outputs.clone()
        }
    }
}

fn block_output_types<'a>(block: &'a Fn, metadata: &BlockMetadata) -> Vec<&'a Type> {
    match metadata.output_names.len() {
        0 => Vec::new(),
        1 => vec![&block.ret_ty],
        _ => {
            let Type::Tuple(types) = &block.ret_ty else {
                panic!("generated multi-output block should return a tuple type");
            };
            types.iter().map(|ty| &**ty).collect()
        }
    }
}

fn collect_register_writes(block: &Fn) -> BTreeMap<String, RegisterWriteRefs> {
    block
        .nodes
        .iter()
        .filter_map(|node| match &node.payload {
            NodePayload::RegisterWrite {
                arg,
                register,
                load_enable,
                reset,
            } => Some((
                register.clone(),
                RegisterWriteRefs {
                    arg: *arg,
                    load_enable: *load_enable,
                    reset: *reset,
                },
            )),
            _ => None,
        })
        .collect()
}

fn cycle_eval_fn(block: &Fn, metadata: &BlockMetadata, state: &[IrValue]) -> Fn {
    let state_by_register: BTreeMap<&str, &IrValue> = metadata
        .registers
        .iter()
        .zip(state)
        .map(|(register, value)| (register.name.as_str(), value))
        .collect();
    let mut result = block.clone();
    for node in &mut result.nodes {
        match &node.payload {
            NodePayload::RegisterRead { register } => {
                node.payload = NodePayload::Literal(
                    (*state_by_register
                        .get(register.as_str())
                        .expect("generated register read should have state"))
                    .clone(),
                );
            }
            NodePayload::RegisterWrite { .. } => {
                node.ty = Type::nil();
                node.payload = NodePayload::Nil;
            }
            _ => {}
        }
    }
    result
}

fn eval_ref(cycle_fn: &mut Fn, node_ref: NodeRef, inputs: &[IrValue], ir_text: &str) -> IrValue {
    cycle_fn.ret_ty = cycle_fn.get_node(node_ref).ty.clone();
    cycle_fn.ret_node_ref = Some(node_ref);
    match ir_eval::eval_fn(cycle_fn, inputs) {
        FnEvalResult::Success(success) => success.value,
        failure @ FnEvalResult::Failure(_) => {
            panic!("block PIR evaluation failed:\nIR:\n{ir_text}\nresult={failure:?}")
        }
    }
}

fn bool_value(value: &IrValue) -> bool {
    value
        .to_bool()
        .expect("generated reset/load-enable value should be bits[1]")
}

fn evaluate_block_cycle(
    block: &Fn,
    metadata: &BlockMetadata,
    inputs: &[IrValue],
    state: &[IrValue],
    ir_text: &str,
) -> (Vec<IrValue>, Vec<IrValue>) {
    let output_refs = block_output_refs(block, metadata);
    let writes = collect_register_writes(block);
    let mut cycle_fn = cycle_eval_fn(block, metadata, state);
    if output_refs.is_empty() {
        // Evaluate the explicit empty-tuple block return even though it does
        // not contribute a visible value. This keeps stateless, outputless
        // blocks in the PIR evaluator portion of the fuzz property.
        let ret_ref = block
            .ret_node_ref
            .expect("generated block should have a return node");
        let _ = eval_ref(&mut cycle_fn, ret_ref, inputs, ir_text);
    }
    let outputs = output_refs
        .into_iter()
        .map(|output_ref| eval_ref(&mut cycle_fn, output_ref, inputs, ir_text))
        .collect();
    let mut next_state = Vec::with_capacity(metadata.registers.len());

    for (register_index, register) in metadata.registers.iter().enumerate() {
        let Some(write) = writes.get(&register.name) else {
            next_state.push(state[register_index].clone());
            continue;
        };
        let mut next_value = eval_ref(&mut cycle_fn, write.arg, inputs, ir_text);
        if let Some(load_enable_ref) = write.load_enable
            && !bool_value(&eval_ref(
                &mut cycle_fn,
                load_enable_ref,
                inputs,
                ir_text,
            ))
        {
            next_value = state[register_index].clone();
        }
        if let (Some(reset_ref), Some(reset_value), Some(reset_metadata)) =
            (write.reset, register.reset_value.as_ref(), metadata.reset.as_ref())
        {
            let reset_signal = bool_value(&eval_ref(&mut cycle_fn, reset_ref, inputs, ir_text));
            let reset_asserted = if reset_metadata.active_low {
                !reset_signal
            } else {
                reset_signal
            };
            if reset_asserted {
                next_value = reset_value.clone();
            }
        }
        next_state.push(next_value);
    }

    (outputs, next_state)
}

fn flatten_value(value: &IrValue, ty: &Type) -> IrBits {
    let mut bits = Vec::with_capacity(ty.bit_count());
    flatten_ir_value_to_lsb0_bits_for_type(value, ty, &mut bits)
        .expect("generated value should match its PIR type");
    IrBits::from_lsb_is_0(&bits)
}

fn generate_initial_state(metadata: &BlockMetadata, rng: &mut StdRng) -> Vec<IrValue> {
    metadata
        .registers
        .iter()
        .map(|register| {
            register
                .reset_value
                .clone()
                .unwrap_or_else(|| generate_uniform_value_with_rng(rng, &register.ty))
        })
        .collect()
}

fn generate_cycle_inputs(
    block: &Fn,
    metadata: &BlockMetadata,
    rng: &mut StdRng,
    cycle: usize,
) -> Vec<IrValue> {
    block
        .params
        .iter()
        .map(|param| {
            if let Some(reset) = metadata.reset.as_ref()
                && param.name == reset.port_name
            {
                let asserted = if cycle == 0 {
                    rng.gen_bool(0.5)
                } else {
                    rng.gen_ratio(1, 10)
                };
                let signal_high = if reset.active_low {
                    !asserted
                } else {
                    asserted
                };
                return IrValue::make_ubits(1, u64::from(signal_high))
                    .expect("bits[1] reset input should construct");
            }
            generate_uniform_value_with_rng(rng, &param.ty)
        })
        .collect()
}

fuzz_target!(|data: &[u8]| {
    let mut entropy = DepletableBytes::new(data);
    let generated = generate_block_package(
        &mut entropy,
        &fuzz_block_options(),
        StopPolicy::WhenEntropyDepleted,
    )
    .expect("fixed random block options should construct a valid package");
    let block_ir = generated.package.to_string();
    let block = generated
        .package
        .get_top_block()
        .expect("generated package should have a top block");
    let xlsynth_pir::ir::PackageMember::Block { func, metadata } = block else {
        unreachable!("generated package top should be a block");
    };

    if metadata
        .reset
        .as_ref()
        .is_some_and(|reset| reset.asynchronous)
    {
        panic!("synchronous-only block generation emitted an asynchronous reset:\n{block_ir}");
    }

    let design = block_package_to_sequential_gate_fn(
        &generated.package,
        GatifyOptions::all_opts_disabled(),
    )
    .unwrap_or_else(|error| panic!("random block G8R lowering failed:\n{block_ir}\n{error}"));
    let mut seed = [0_u8; 32];
    seed.copy_from_slice(blake3::hash(block_ir.as_bytes()).as_bytes());
    let mut rng = StdRng::from_seed(seed);
    let mut block_state = generate_initial_state(metadata, &mut rng);
    let initial_g8r_state = block_state
        .iter()
        .zip(&metadata.registers)
        .map(|(value, register)| flatten_value(value, &register.ty))
        .collect();
    let mut g8r_state = SequentialState::from_register_values(&design, initial_g8r_state)
        .expect("generated initial register state should match lowered G8R");
    let output_types = block_output_types(func, metadata);

    for cycle in 0..CYCLE_COUNT {
        let inputs = generate_cycle_inputs(func, metadata, &mut rng, cycle);
        let g8r_inputs = inputs
            .iter()
            .zip(&func.params)
            .map(|(value, param)| flatten_value(value, &param.ty))
            .collect::<Vec<_>>();
        let (expected_outputs, next_block_state) =
            evaluate_block_cycle(func, metadata, &inputs, &block_state, &block_ir);
        let expected_output_bits = expected_outputs
            .iter()
            .zip(&output_types)
            .map(|(value, ty)| flatten_value(value, ty))
            .collect::<Vec<_>>();
        let trace = sequential::simulate(&design, &[g8r_inputs], g8r_state)
            .unwrap_or_else(|error| panic!("random block G8R simulation failed:\n{block_ir}\n{error}"));
        assert_eq!(
            trace.external_outputs()[0],
            expected_output_bits,
            "random block output mismatch at cycle {cycle}:\n{block_ir}"
        );
        let expected_state_bits = next_block_state
            .iter()
            .zip(&metadata.registers)
            .map(|(value, register)| flatten_value(value, &register.ty))
            .collect::<Vec<_>>();
        assert_eq!(
            trace.final_state().values(),
            expected_state_bits,
            "random block register-state mismatch at cycle {cycle}:\n{block_ir}"
        );
        block_state = next_block_state;
        g8r_state = trace.final_state().clone();
    }
});
