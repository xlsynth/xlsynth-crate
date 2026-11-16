// SPDX-License-Identifier: Apache-2.0

//! Generation options for the word-level Lean differential fuzz targets.
//!
//! Both targets draw from the same pure PIR operations and aggregate types.
//! Blocks can sample a generated function's result on each clock.

use rand::{Rng, RngCore};
use xlsynth_pir::ir::{Package, Type};
use xlsynth_pir::ir_random::{
    ArrayAssumptionMode, DepletableBytes, OperationSet, RandomBlockOptions, RandomBlockResetTiming,
    RandomFnOptions, RandomOperation, StopPolicy, generate_block_package, generate_package,
};
use xlsynth_pir::random_inputs::{BitValuePattern, generate_pattern_value};
use xlsynth_pir::{BlockBuilder, RegisterWriteOptions, ResetBehavior};

/// Configures function generation for comparing pure return values in Lean and
/// PIR.
///
/// Generated packages can call helper functions and use arrays, tuples,
/// zero-width bits, and PIR extension operations. The generator excludes
/// events, counted loops, and partial products from these cases.
pub fn lean_function_options() -> RandomFnOptions {
    RandomFnOptions {
        max_nodes: 32,
        max_functions: 3,
        max_bit_width: 129,
        max_aggregate_leaves: 12,
        allow_gate: true,
        allow_map: true,
        allow_empty_case_sel: true,
        allow_zero_width_bits: true,
        allow_arbitrary_width_multiply: true,
        allow_extension_ops: true,
        allow_events: false,
        max_counted_fors_per_function: 0,
        array_assumption_mode: ArrayAssumptionMode::ProvenSafe,
        enabled_operations: OperationSet::new(OperationSet::all_supported().iter().filter(|op| {
            !matches!(
                op,
                RandomOperation::Smulp
                    | RandomOperation::Umulp
                    | RandomOperation::CountedFor
                    | RandomOperation::AfterAll
                    | RandomOperation::Assert
                    | RandomOperation::Trace
                    | RandomOperation::Cover
            )
        })),
        ..RandomFnOptions::default()
    }
}

/// Configures a single clocked block with synchronous reset on every register.
///
/// Ports and registers may have zero width. The block reuses the function
/// generator's value operations except for `Invoke` and `Map`, which need a
/// separate callee.
pub fn lean_block_options() -> RandomBlockOptions {
    let function = lean_function_options();
    RandomBlockOptions {
        min_input_ports: 1,
        max_input_ports: 3,
        min_output_ports: 1,
        allow_zero_width_ports_and_registers: true,
        require_reset_on_all_registers: true,
        reset_timing: RandomBlockResetTiming::Synchronous,
        function_options: RandomFnOptions {
            max_functions: 1,
            max_invokes_per_function: 0,
            allow_map: false,
            enabled_operations: OperationSet::new(
                function
                    .enabled_operations
                    .iter()
                    .filter(|op| !matches!(op, RandomOperation::Invoke | RandomOperation::Map)),
            ),
            ..function
        },
        ..RandomBlockOptions::default()
    }
}

/// Builds a random sequential block or clocks the result of a random function
/// package into a register. Both use the same eight-clock differential oracle.
pub fn generate_lean_block(data: &[u8], rng: &mut impl RngCore) -> Package {
    // The block generator consumes a wiring header before any body operations.
    // Expand short corpus entries reproducibly so they can exercise both.
    let mut padded = [0u8; 512];
    let graph = if data.len() < padded.len() {
        padded[..data.len()].copy_from_slice(data);
        rng.fill_bytes(&mut padded[data.len()..]);
        &padded[..]
    } else {
        data
    };
    let (mode, graph) = graph.split_first().expect("padded graph");
    let mut source = DepletableBytes::new(graph);
    if mode & 1 == 0 {
        return generate_block_package(
            &mut source,
            &lean_block_options(),
            StopPolicy::WhenEntropyDepleted,
        )
        .expect("bounded block generation")
        .package;
    }

    // The block emitter resolves invokes in helpers. It lowers extensions
    // in the block itself, and does not lower maps or extensions in helpers.
    let options = RandomFnOptions {
        allow_map: false,
        allow_extension_ops: false,
        ..lean_function_options()
    };
    let mut package = generate_package(&mut source, &options, StopPolicy::WhenEntropyDepleted)
        .expect("bounded function generation")
        .package;
    let function = package.get_top_fn().expect("generated top function");
    let mut block = BlockBuilder::new("sample");
    block.clock_port("clk").unwrap();
    let reset = block.input_port("rst", Type::Bits(1)).unwrap();
    block.set_reset(reset, ResetBehavior::default()).unwrap();
    let args: Vec<_> = function
        .param_nodes()
        .map(|p| block.input_port(p.param_name(), p.ty.clone()).unwrap())
        .collect();
    let result = block.invoke(function, &args).unwrap();
    let data = rng.gen_ratio(9, 10).then(|| {
        // A nullary helper (or one with only narrow/aggregate arguments)
        // still needs a varying operand for the output arithmetic.
        args.iter()
            .find_map(|&arg| match block.get_type(arg).unwrap() {
                Type::Bits(width) if *width >= 2 => Some((arg, *width)),
                _ => None,
            })
            .unwrap_or_else(|| (block.input_port("data", Type::Bits(8)).unwrap(), 8))
    });
    // A zero-bit result has just one value. Store the data alongside it
    // so that an enabled write can still change the register.
    let carried_data = data.filter(|_| function.ret_ty.bit_count() == 0);
    let write_value = if let Some((operand, _)) = carried_data {
        block.tuple(&[result, operand]).unwrap()
    } else {
        result
    };
    let reg_ty = block.get_type(write_value).unwrap().clone();
    let zero = generate_pattern_value(&reg_ty, BitValuePattern::Zero);
    let reg = block.register("q", reg_ty, Some(zero.clone())).unwrap();
    let old = block.register_read(reg).unwrap();
    let (out, load_enable) = if let Some((data, data_width)) = data {
        let enable = block.bit_slice(data, 0, 1).unwrap();
        let (state, state_width) = if carried_data.is_some() {
            (block.tuple_index(old, 1).unwrap(), data_width)
        } else if let Type::Bits(width) = function.ret_ty {
            (old, width)
        } else {
            let zero = block.literal(zero).unwrap();
            (block.ne(old, zero).unwrap(), 1)
        };
        // Three bits suffice to distinguish 2 squared from 0 squared.
        let width = data_width.max(state_width).max(3);
        let data = block.zero_extend(data, width).unwrap();
        let state = block.zero_extend(state, width).unwrap();
        let square = block.umul(data, data).unwrap();
        let out = if rng.gen_bool(0.5) {
            block.add(state, square).unwrap()
        } else {
            block.sub(state, square).unwrap()
        };
        (out, Some(enable))
    } else {
        (old, None)
    };
    block.output_port("out", out).unwrap();
    block
        .register_write(
            reg,
            write_value,
            RegisterWriteOptions {
                reset: Some(reset),
                load_enable,
            },
        )
        .unwrap();
    block.build_into_package(&mut package).unwrap();
    package.set_top_block("sample").unwrap();
    package
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};
    use xlsynth_ir_value::IrValue;
    use xlsynth_pir::block_eval::{BlockEvaluator, RegisterInit};
    use xlsynth_pir::ir::NodePayload;

    fn replay_package(seed: u32) -> Package {
        let data = seed.to_le_bytes();
        let mut rng = StdRng::from_seed(*blake3::hash(&data).as_bytes());
        generate_lean_block(&data, &mut rng)
    }

    #[test]
    fn constant_one_bit_helper_has_a_varying_arithmetic_output() {
        // This helper takes no arguments and always returns one.
        let package = replay_package(9);
        let block = package.get_top_block().unwrap();
        let evaluator = BlockEvaluator::prepare(&package, &block.name).unwrap();
        let mut state = evaluator
            .initial_state(RegisterInit::Supplied(vec![IrValue::bool(false)]))
            .unwrap();
        // The output is q - data^2, modulo 256.
        for (reset, data, expected) in [(false, 3, 247), (false, 5, 232), (true, 7, 208)] {
            let inputs = [IrValue::bool(reset), IrValue::make_ubits(8, data).unwrap()];
            let cycle = evaluator.step(&state, &inputs).unwrap();
            assert_eq!(cycle.outputs, [IrValue::make_ubits(8, expected).unwrap()]);
            assert_eq!(cycle.next_state.values(), [IrValue::bool(!reset)]);
            state = cycle.next_state;
        }
    }

    #[test]
    fn zero_bit_helper_registers_the_data_alongside_its_result() {
        let package = replay_package(17);
        let block = package.get_top_block().unwrap();
        let evaluator = BlockEvaluator::prepare(&package, &block.name).unwrap();
        let register = |data| {
            IrValue::make_tuple(&[
                IrValue::make_ubits(0, 0).unwrap(),
                IrValue::make_ubits(8, data).unwrap(),
            ])
        };
        let state = evaluator
            .initial_state(RegisterInit::Supplied(vec![register(10)]))
            .unwrap();
        for (data, next) in [(2, 10), (3, 3)] {
            let inputs: Vec<_> = block
                .input_ports()
                .map(|p| {
                    if block.port_name(p) == "data" {
                        IrValue::make_ubits(8, data).unwrap()
                    } else {
                        generate_pattern_value(block.port_type(p), BitValuePattern::Zero)
                    }
                })
                .collect();
            let cycle = evaluator.step(&state, &inputs).unwrap();
            assert_eq!(cycle.next_state.values(), [register(next)]);
            assert_eq!(
                cycle.outputs,
                [IrValue::make_ubits(8, 10 + data * data).unwrap()]
            );
        }
    }

    #[test]
    fn helper_block_can_still_output_the_previous_register() {
        let package = replay_package(49);
        let block = package.get_top_block().unwrap();
        let output = block.output_value(block.get_output_port("out").unwrap());
        assert!(matches!(
            block.get_node(output).payload,
            NodePayload::RegisterRead { .. }
        ));
    }

    #[test]
    fn nested_invoke_seed_registers_the_leaf_result() {
        // Replay the four bytes 0b 00 00 00 in fuzz_block_lean_eval.
        let package = replay_package(11);
        let block = package.get_top_block().unwrap();
        let write_arg = block
            .nodes
            .iter()
            .find_map(|node| match &node.payload {
                NodePayload::RegisterWrite { arg, .. } => Some(*arg),
                _ => None,
            })
            .unwrap();
        let NodePayload::Invoke { to_apply, .. } = &block.get_node(write_arg).payload else {
            panic!("register must capture a helper call");
        };
        let helper = package.get_fn(to_apply).unwrap();
        assert!(matches!(
            helper.get_node(helper.ret_node_ref.unwrap()).payload,
            NodePayload::Invoke { .. }
        ));

        // The leaf sign-extends p1 from 15 to 48 bits. The output is q + p1^2.
        let evaluator = BlockEvaluator::prepare(&package, &block.name).unwrap();
        let state = evaluator
            .initial_state(RegisterInit::Supplied(vec![
                IrValue::make_ubits(48, 42).unwrap(),
            ]))
            .unwrap();
        for (reset, p1, output, next) in [
            (true, 2, 46, 0),
            (true, 4, 58, 0),
            (false, 2, 46, 42),
            (false, 3, 51, 3),
            (false, 0x4001, 42 + 0x4001 * 0x4001, 0xffff_ffff_c001),
        ] {
            let inputs: Vec<_> = block
                .input_ports()
                .map(|port| match block.port_name(port) {
                    "rst" => IrValue::bool(reset),
                    "p1" => IrValue::make_ubits(15, p1).unwrap(),
                    _ => generate_pattern_value(block.port_type(port), BitValuePattern::Zero),
                })
                .collect();
            let cycle = evaluator.step(&state, &inputs).unwrap();
            assert_eq!(cycle.outputs, [IrValue::make_ubits(48, output).unwrap()]);
            assert_eq!(
                cycle.next_state.values(),
                [IrValue::make_ubits(48, next).unwrap()]
            );
        }
    }
}
