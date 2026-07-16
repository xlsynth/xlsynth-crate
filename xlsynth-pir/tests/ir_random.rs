// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeSet, HashMap, HashSet};

use rand::RngCore;
use rand_pcg::Pcg64Mcg;
use xlsynth_pir::ir::{
    Binop, BlockMetadata, ExtNaryAddArchitecture, FileTable, Fn, MemberType, NaryOp, NodePayload,
    NodeRef, Package, PackageMember, Type, Unop,
};
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn_in_package};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_random::{
    DepletableBytes, FunctionSignature, GenerationError, OperationSet, RandomBlockOptions,
    RandomBlockResetTiming, RandomFnOptions, RandomOperation, RngEntropy, StopPolicy,
    generate_block, generate_block_package, generate_fn, generate_fn_with_signature,
    generate_package, generate_same_signature_pair,
};
use xlsynth_pir::ir_utils::operands;
use xlsynth_pir::ir_verify::verify_package;
use xlsynth_pir::random_inputs::{
    generate_argument_sets_from_seed, generate_biased_arguments, generate_biased_value,
    generate_uniform_value,
};

fn validate_generated(function: &xlsynth_pir::ir::Fn) {
    function.check_pir_layout_invariants().unwrap();
    let package = Package {
        name: "random_package".to_string(),
        file_table: FileTable::new(),
        members: vec![PackageMember::Function(function.clone())],
        top: Some((function.name.clone(), MemberType::Function)),
    };
    verify_package(&package).unwrap();
}

fn validate_generated_block_package(package: &Package) {
    verify_package(package).unwrap();
    let ir_text = package.to_string();
    let reparsed = Parser::new(&ir_text)
        .parse_and_validate_package()
        .unwrap_or_else(|error| {
            panic!("generated block package failed roundtrip:\n{ir_text}\n{error:?}")
        });
    assert_eq!(reparsed.to_string(), ir_text);
}

fn node_depends_on(function: &Fn, start: NodeRef, target: NodeRef) -> bool {
    let mut pending = vec![start];
    let mut seen = HashSet::new();
    while let Some(node_ref) = pending.pop() {
        if node_ref == target {
            return true;
        }
        if seen.insert(node_ref.index) {
            pending.extend(operands(&function.get_node(node_ref).payload));
        }
    }
    false
}

fn block_output_types<'a>(function: &'a Fn, output_count: usize) -> Vec<&'a Type> {
    if output_count == 1 {
        return vec![&function.ret_ty];
    }
    let Type::Tuple(fields) = &function.ret_ty else {
        panic!("multi-output generated block should return a tuple");
    };
    fields.iter().map(|field| &**field).collect()
}

fn assert_generated_block_register_wiring(function: &Fn, metadata: &BlockMetadata) {
    assert_eq!(
        metadata.clock_port_name.is_some(),
        !metadata.registers.is_empty()
    );
    for register in &metadata.registers {
        let read_count = function
            .nodes
            .iter()
            .filter(|node| {
                matches!(
                    &node.payload,
                    NodePayload::RegisterRead { register: read_register }
                        if read_register == &register.name
                )
            })
            .count();
        let writes: Vec<_> = function
            .nodes
            .iter()
            .filter_map(|node| match &node.payload {
                NodePayload::RegisterWrite {
                    arg,
                    register: write_register,
                    reset,
                    ..
                } if write_register == &register.name => Some((*arg, *reset)),
                _ => None,
            })
            .collect();

        assert_eq!(read_count, 1);
        assert_eq!(writes.len(), 1);
        assert_eq!(function.get_node(writes[0].0).ty, register.ty);
        assert_eq!(writes[0].1.is_some(), register.reset_value.is_some());
    }
}

fn type_has_array(ty: &Type) -> bool {
    match ty {
        Type::Array(_) => true,
        Type::Tuple(fields) => fields.iter().any(|field| type_has_array(field)),
        Type::Token | Type::Bits(_) => false,
    }
}

fn type_has_tuple(ty: &Type) -> bool {
    match ty {
        Type::Tuple(_) => true,
        Type::Array(array) => type_has_tuple(&array.element_type),
        Type::Token | Type::Bits(_) => false,
    }
}

fn type_has_empty_tuple(ty: &Type) -> bool {
    match ty {
        Type::Tuple(fields) => {
            fields.is_empty() || fields.iter().any(|field| type_has_empty_tuple(field))
        }
        Type::Array(array) => type_has_empty_tuple(&array.element_type),
        Type::Token | Type::Bits(_) => false,
    }
}

fn type_has_nested_tuple(ty: &Type, inside_tuple: bool) -> bool {
    match ty {
        Type::Tuple(fields) => {
            inside_tuple
                || fields
                    .iter()
                    .any(|field| type_has_nested_tuple(field, /* inside_tuple= */ true))
        }
        Type::Array(array) => type_has_nested_tuple(&array.element_type, inside_tuple),
        Type::Token | Type::Bits(_) => false,
    }
}

fn type_has_multidimensional_array(ty: &Type) -> bool {
    match ty {
        Type::Array(array) => {
            matches!(&*array.element_type, Type::Array(_))
                || type_has_multidimensional_array(&array.element_type)
        }
        Type::Tuple(fields) => fields
            .iter()
            .any(|field| type_has_multidimensional_array(field)),
        Type::Token | Type::Bits(_) => false,
    }
}

fn type_has_array_of_tuple(ty: &Type) -> bool {
    match ty {
        Type::Array(array) => {
            matches!(&*array.element_type, Type::Tuple(_))
                || type_has_array_of_tuple(&array.element_type)
        }
        Type::Tuple(fields) => fields.iter().any(|field| type_has_array_of_tuple(field)),
        Type::Token | Type::Bits(_) => false,
    }
}

fn type_has_tuple_with_array_element(ty: &Type) -> bool {
    match ty {
        Type::Tuple(fields) => {
            fields
                .iter()
                .any(|field| matches!(&**field, Type::Array(_)))
                || fields
                    .iter()
                    .any(|field| type_has_tuple_with_array_element(field))
        }
        Type::Array(array) => type_has_tuple_with_array_element(&array.element_type),
        Type::Token | Type::Bits(_) => false,
    }
}

fn type_depth(ty: &Type) -> usize {
    match ty {
        Type::Token | Type::Bits(_) => 0,
        Type::Tuple(fields) => {
            1 + fields
                .iter()
                .map(|field| type_depth(field))
                .max()
                .unwrap_or(0)
        }
        Type::Array(array) => 1 + type_depth(&array.element_type),
    }
}

fn type_leaf_count(ty: &Type) -> usize {
    match ty {
        Type::Token | Type::Bits(_) => 1,
        Type::Tuple(fields) => fields.iter().map(|field| type_leaf_count(field)).sum(),
        Type::Array(array) => type_leaf_count(&array.element_type) * array.element_count,
    }
}

fn assert_type_obeys_options(ty: &Type, options: &RandomFnOptions) {
    assert!(type_depth(ty) <= options.max_type_depth);
    assert!(type_leaf_count(ty) <= options.max_aggregate_leaves);
    match ty {
        Type::Bits(width) => assert!(*width <= options.max_bit_width),
        Type::Array(array) => {
            assert!(options.allow_arrays);
            assert!(array.element_count > 0);
            assert!(array.element_count <= options.max_array_length);
            assert_type_obeys_options(&array.element_type, options);
        }
        Type::Tuple(fields) => {
            assert!(options.allow_tuples);
            assert!(fields.len() <= options.max_tuple_length);
            for field in fields {
                assert_type_obeys_options(field, options);
            }
        }
        Type::Token => assert!(options.allow_events),
    }
}

fn function_types(function: &xlsynth_pir::ir::Fn) -> impl Iterator<Item = &Type> {
    function
        .params
        .iter()
        .map(|param| &param.ty)
        .chain(function.nodes.iter().skip(1).map(|node| &node.ty))
}

fn assert_signature(function: &xlsynth_pir::ir::Fn, signature: &FunctionSignature) {
    assert_eq!(
        function
            .params
            .iter()
            .map(|param| param.ty.clone())
            .collect::<Vec<_>>(),
        signature.params
    );
    assert_eq!(function.ret_ty, signature.return_type);
}

fn max_nested_counted_for_iterations(
    package: &Package,
    function_name: &str,
    memo: &mut HashMap<String, usize>,
) -> usize {
    if let Some(maximum) = memo.get(function_name) {
        return *maximum;
    }
    let function = package.get_fn(function_name).unwrap();
    let mut maximum = 1;
    for node in &function.nodes {
        let expansion = match &node.payload {
            NodePayload::Invoke { to_apply, .. } => {
                max_nested_counted_for_iterations(package, to_apply, memo)
            }
            NodePayload::CountedFor {
                trip_count, body, ..
            } => trip_count.saturating_mul(max_nested_counted_for_iterations(package, body, memo)),
            _ => 1,
        };
        maximum = maximum.max(expansion);
    }
    memo.insert(function_name.to_string(), maximum);
    maximum
}

#[test]
fn depleted_entropy_constructs_a_minimal_deterministic_function() {
    let options = RandomFnOptions {
        max_params: 5,
        max_nodes: 32,
        ..RandomFnOptions::default()
    };
    let mut first_source = DepletableBytes::new(&[]);
    let mut second_source = DepletableBytes::new(&[]);
    let first = generate_fn(&mut first_source, &options, StopPolicy::WhenEntropyDepleted).unwrap();
    let second = generate_fn(
        &mut second_source,
        &options,
        StopPolicy::WhenEntropyDepleted,
    )
    .unwrap();

    validate_generated(&first.function);
    validate_generated(&second.function);
    assert_eq!(first.stats.emitted_node_count, 1);
    assert!(first.function.params.is_empty());
    assert!(matches!(
        first.function.nodes[1].payload,
        NodePayload::Literal(_)
    ));
    assert_eq!(
        first.function.nodes[1].ty.to_string(),
        second.function.nodes[1].ty.to_string()
    );
    assert_eq!(first.stats, second.stats);
}

#[test]
fn depleted_entropy_constructs_a_minimal_deterministic_block() {
    let options = RandomBlockOptions {
        max_input_ports: 0,
        max_output_ports: 0,
        max_registers: 0,
        function_options: RandomFnOptions {
            max_nodes: 1,
            ..RandomFnOptions::default()
        },
        ..RandomBlockOptions::default()
    };
    let mut first_source = DepletableBytes::new(&[]);
    let mut second_source = DepletableBytes::new(&[]);
    let first =
        generate_block_package(&mut first_source, &options, StopPolicy::WhenEntropyDepleted)
            .unwrap();
    let second = generate_block_package(
        &mut second_source,
        &options,
        StopPolicy::WhenEntropyDepleted,
    )
    .unwrap();

    validate_generated_block_package(&first.package);
    validate_generated_block_package(&second.package);
    assert_eq!(first.package.to_string(), second.package.to_string());
    let PackageMember::Block { metadata, .. } = first.package.get_top_block().unwrap() else {
        unreachable!("generated package top should be a block");
    };
    assert!(metadata.registers.is_empty());
    assert!(metadata.output_names.is_empty());
    let PackageMember::Block { func, .. } = first.package.get_top_block().unwrap() else {
        unreachable!("generated package top should be a block");
    };
    let ret_ref = func.ret_node_ref.unwrap();
    assert!(matches!(
        &func.get_node(ret_ref).payload,
        NodePayload::Tuple(outputs) if outputs.is_empty()
    ));
}

#[test]
fn generated_zero_output_block_does_not_write_synthetic_return() {
    let options = RandomBlockOptions {
        max_input_ports: 0,
        min_registers: 1,
        max_registers: 1,
        max_output_ports: 0,
        allow_zero_width_ports_and_registers: true,
        allow_load_enable: false,
        allow_reset: false,
        function_options: RandomFnOptions {
            max_nodes: 3,
            allow_arrays: false,
            ..RandomFnOptions::default()
        },
        ..RandomBlockOptions::default()
    };
    // Select an empty-tuple register, then entropy that would select the
    // same-typed synthetic zero-output return if it were a D-value candidate.
    let entropy_bytes: Vec<u8> = [1_u64, 0, 1, 1]
        .into_iter()
        .flat_map(u64::to_le_bytes)
        .collect();
    let mut entropy = DepletableBytes::new(&entropy_bytes);
    let generated = generate_block(&mut entropy, &options, StopPolicy::ExactBodyNodes(0)).unwrap();
    let ret_ref = generated.function.ret_node_ref.unwrap();
    let write_arg = generated
        .function
        .nodes
        .iter()
        .find_map(|node| match &node.payload {
            NodePayload::RegisterWrite { arg, .. } => Some(*arg),
            _ => None,
        })
        .unwrap();

    assert_ne!(write_arg, ret_ref);
    validate_generated_block_package(&generated.into_top_package("zero_output_register_package"));
}

#[test]
fn generated_block_populates_multi_output_metadata() {
    let options = RandomBlockOptions {
        min_input_ports: 1,
        max_input_ports: 1,
        min_output_ports: 3,
        max_output_ports: 3,
        max_registers: 0,
        function_options: RandomFnOptions {
            max_nodes: 3,
            max_bit_width: 8,
            ..RandomFnOptions::default()
        },
        ..RandomBlockOptions::default()
    };
    let mut entropy = RngEntropy::new(Pcg64Mcg::new(0x0d70_0003));
    for _ in 0..128 {
        let generated =
            generate_block(&mut entropy, &options, StopPolicy::ExactBodyNodes(0)).unwrap();
        let func = &generated.function;
        let metadata = &generated.metadata;
        assert_eq!(
            metadata.output_names,
            vec!["out0".to_string(), "out1".to_string(), "out2".to_string()]
        );
        assert_eq!(metadata.output_port_ids.len(), 3);
        assert_eq!(
            metadata
                .output_port_ids
                .values()
                .collect::<HashSet<_>>()
                .len(),
            3
        );
        let ret_ref = func.ret_node_ref.unwrap();
        assert!(matches!(
            &func.get_node(ret_ref).payload,
            NodePayload::Tuple(outputs) if outputs.len() == 3
        ));
    }
}

#[test]
fn random_block_reset_timing_option_controls_generated_metadata() {
    let cases = [
        (RandomBlockResetTiming::Synchronous, 0x51ac_0001),
        (RandomBlockResetTiming::Asynchronous, 0x51ac_0002),
        (RandomBlockResetTiming::Either, 0x51ac_0003),
    ];
    for (reset_timing, seed) in cases {
        let options = RandomBlockOptions {
            min_input_ports: 0,
            max_input_ports: 0,
            min_registers: 1,
            max_registers: 1,
            min_output_ports: 1,
            max_output_ports: 1,
            allow_load_enable: false,
            reset_timing,
            function_options: RandomFnOptions {
                max_nodes: 4,
                max_bit_width: 8,
                ..RandomFnOptions::default()
            },
            ..RandomBlockOptions::default()
        };
        let mut entropy = RngEntropy::new(Pcg64Mcg::new(seed));
        let mut saw_reset = false;
        let mut saw_synchronous = false;
        let mut saw_asynchronous = false;
        for _ in 0..256 {
            let generated =
                generate_block(&mut entropy, &options, StopPolicy::ExactBodyNodes(0)).unwrap();
            let Some(reset) = generated.metadata.reset.as_ref() else {
                continue;
            };
            saw_reset = true;
            saw_synchronous |= !reset.asynchronous;
            saw_asynchronous |= reset.asynchronous;
            match reset_timing {
                RandomBlockResetTiming::Synchronous => assert!(!reset.asynchronous),
                RandomBlockResetTiming::Asynchronous => assert!(reset.asynchronous),
                RandomBlockResetTiming::Either => {}
            }
        }
        assert!(saw_reset);
        if reset_timing == RandomBlockResetTiming::Either {
            assert!(saw_synchronous);
            assert!(saw_asynchronous);
        }
    }
}

#[test]
fn random_block_required_register_resets_option_controls_samples() {
    let options = RandomBlockOptions {
        min_input_ports: 0,
        max_input_ports: 3,
        min_registers: 1,
        max_registers: 3,
        require_reset_on_all_registers: true,
        reset_timing: RandomBlockResetTiming::Synchronous,
        function_options: RandomFnOptions {
            max_nodes: 32,
            max_bit_width: 8,
            ..RandomFnOptions::default()
        },
        ..RandomBlockOptions::default()
    };
    let mut entropy = RngEntropy::new(Pcg64Mcg::new(0x51ac_0004));
    for _ in 0..256 {
        let generated =
            generate_block(&mut entropy, &options, StopPolicy::ExactBodyNodes(8)).unwrap();
        assert!(!generated.metadata.registers.is_empty());
        assert!(
            generated
                .metadata
                .reset
                .as_ref()
                .is_some_and(|reset| !reset.asynchronous)
        );
        assert!(
            generated
                .metadata
                .registers
                .iter()
                .all(|register| register.reset_value.is_some())
        );
        assert!(
            generated
                .function
                .nodes
                .iter()
                .filter_map(|node| match &node.payload {
                    NodePayload::RegisterWrite { reset, .. } => Some(reset),
                    _ => None,
                })
                .all(Option::is_some)
        );
    }

    let invalid_options = RandomBlockOptions {
        max_registers: 1,
        allow_reset: false,
        require_reset_on_all_registers: true,
        ..RandomBlockOptions::default()
    };
    let mut invalid_entropy = DepletableBytes::new(&[]);
    let error = generate_block(
        &mut invalid_entropy,
        &invalid_options,
        StopPolicy::WhenEntropyDepleted,
    )
    .expect_err("requiring resets while disabling resets should be invalid");
    assert_eq!(
        error,
        GenerationError::InvalidOptions(
            "require_reset_on_all_registers requires allow_reset".to_string()
        )
    );
}

#[test]
fn random_block_zero_width_interface_option_controls_samples() {
    let cases = [(false, 0x20f7_0001), (true, 0x20f7_0002)];
    for (allow_zero_width, seed) in cases {
        let options = RandomBlockOptions {
            min_input_ports: 1,
            max_input_ports: 1,
            min_output_ports: 0,
            max_output_ports: 1,
            min_registers: 1,
            max_registers: 1,
            allow_zero_width_ports_and_registers: allow_zero_width,
            allow_load_enable: false,
            allow_reset: false,
            function_options: RandomFnOptions {
                max_params: 1,
                max_nodes: 8,
                ..RandomFnOptions::default()
            },
            ..RandomBlockOptions::default()
        };
        let mut entropy = RngEntropy::new(Pcg64Mcg::new(seed));
        let mut saw_zero_width = false;
        let mut saw_zero_width_compound = false;
        for _ in 0..1_024 {
            let generated =
                generate_block(&mut entropy, &options, StopPolicy::ExactBodyNodes(4)).unwrap();
            let function = &generated.function;
            let metadata = &generated.metadata;
            let output_types = block_output_types(function, metadata.output_names.len());
            for ty in function
                .params
                .iter()
                .map(|param| &param.ty)
                .chain(metadata.registers.iter().map(|register| &register.ty))
                .chain(output_types.iter().copied())
            {
                if allow_zero_width {
                    saw_zero_width |= ty.bit_count() == 0;
                    saw_zero_width_compound |= ty.bit_count() == 0 && type_has_array(ty);
                } else {
                    assert_ne!(ty.bit_count(), 0);
                }
            }
        }
        if allow_zero_width {
            assert!(saw_zero_width);
            assert!(saw_zero_width_compound);
        }
    }
}

#[test]
fn probabilistic_default_block_generation_covers_shapes_state_and_types() {
    let options = RandomBlockOptions::default();
    let mut entropy = RngEntropy::new(Pcg64Mcg::new(0xb10c_5eed));
    let mut saw_zero_registers = false;
    let mut saw_max_registers = false;
    let mut saw_feedback = false;
    let mut saw_no_feedback = false;
    let mut saw_load_enable = false;
    let mut saw_no_load_enable = false;
    let mut saw_reset_register = false;
    let mut saw_nonreset_register = false;
    let mut saw_unused_reset_port = false;
    let mut saw_synchronous_reset = false;
    let mut saw_asynchronous_reset = false;
    let mut saw_min_inputs = false;
    let mut saw_max_inputs = false;
    let mut saw_min_outputs = false;
    let mut saw_max_outputs = false;
    let mut saw_tuple_register = false;
    let mut saw_array_register = false;
    let mut saw_tuple_port = false;
    let mut saw_array_port = false;

    for _ in 0..2_000 {
        let generated =
            generate_block(&mut entropy, &options, StopPolicy::ExactBodyNodes(20)).unwrap();
        let function = &generated.function;
        let metadata = &generated.metadata;
        let reset_port_count = usize::from(metadata.reset.is_some());
        let data_input_count = function.params.len() - reset_port_count;
        let output_count = metadata.output_names.len();

        saw_zero_registers |= metadata.registers.is_empty();
        saw_max_registers |= metadata.registers.len() == options.max_registers;
        saw_min_inputs |= data_input_count == options.min_input_ports;
        saw_max_inputs |= data_input_count == options.max_input_ports;
        saw_min_outputs |= output_count == options.min_output_ports;
        saw_max_outputs |= output_count == options.max_output_ports;
        saw_tuple_register |= metadata
            .registers
            .iter()
            .any(|register| type_has_tuple(&register.ty));
        saw_array_register |= metadata
            .registers
            .iter()
            .any(|register| type_has_array(&register.ty));
        saw_reset_register |= metadata
            .registers
            .iter()
            .any(|register| register.reset_value.is_some());
        saw_nonreset_register |= metadata
            .registers
            .iter()
            .any(|register| register.reset_value.is_none());
        saw_unused_reset_port |= metadata.reset.is_some()
            && metadata.registers.len() == 1
            && metadata.registers[0].reset_value.is_none();
        if let Some(reset) = metadata.reset.as_ref() {
            saw_synchronous_reset |= !reset.asynchronous;
            saw_asynchronous_reset |= reset.asynchronous;
            if metadata.registers.len() > 1 {
                let reset_count = metadata
                    .registers
                    .iter()
                    .filter(|register| register.reset_value.is_some())
                    .count();
                assert!(reset_count > 0);
                assert!(reset_count < metadata.registers.len());
            }
        }

        assert_generated_block_register_wiring(function, metadata);
        let output_types = block_output_types(function, output_count);
        assert!(
            function
                .params
                .iter()
                .all(|param| param.ty.bit_count() != 0)
        );
        assert!(output_types.iter().all(|ty| ty.bit_count() != 0));
        assert!(
            metadata
                .registers
                .iter()
                .all(|register| register.ty.bit_count() != 0)
        );
        saw_tuple_port |= function
            .params
            .iter()
            .map(|param| &param.ty)
            .chain(output_types.iter().copied())
            .any(type_has_tuple);
        saw_array_port |= function
            .params
            .iter()
            .map(|param| &param.ty)
            .chain(output_types.iter().copied())
            .any(type_has_array);

        let register_reads: HashMap<&str, NodeRef> = function
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| match &node.payload {
                NodePayload::RegisterRead { register } => {
                    Some((register.as_str(), NodeRef { index }))
                }
                _ => None,
            })
            .collect();
        for node in &function.nodes {
            let NodePayload::RegisterWrite {
                arg,
                register,
                load_enable,
                ..
            } = &node.payload
            else {
                continue;
            };
            saw_load_enable |= load_enable.is_some();
            saw_no_load_enable |= load_enable.is_none();
            if let Some(load_enable_ref) = load_enable {
                assert_eq!(function.get_node(*load_enable_ref).ty, Type::Bits(1));
            }
            let read_ref = register_reads[register.as_str()];
            if node_depends_on(function, *arg, read_ref) {
                saw_feedback = true;
            } else {
                saw_no_feedback = true;
            }
        }

        assert!(data_input_count <= options.max_input_ports);
        assert!(output_count <= options.max_output_ports);
        assert!(metadata.registers.len() <= options.max_registers);
        assert!(generated.stats.emitted_node_count <= options.function_options.max_nodes);
    }

    assert!(saw_zero_registers);
    assert!(saw_max_registers);
    assert!(saw_feedback);
    assert!(saw_no_feedback);
    assert!(saw_load_enable);
    assert!(saw_no_load_enable);
    assert!(saw_reset_register);
    assert!(saw_nonreset_register);
    assert!(saw_unused_reset_port);
    assert!(saw_synchronous_reset);
    assert!(saw_asynchronous_reset);
    assert!(saw_min_inputs);
    assert!(saw_max_inputs);
    assert!(saw_min_outputs);
    assert!(saw_max_outputs);
    assert!(saw_tuple_register);
    assert!(saw_array_register);
    assert!(saw_tuple_port);
    assert!(saw_array_port);
}

#[test]
fn registered_block_feedback_can_feed_next_state() {
    let options = RandomBlockOptions {
        min_input_ports: 0,
        max_input_ports: 0,
        min_registers: 1,
        max_registers: 1,
        min_output_ports: 1,
        max_output_ports: 1,
        allow_load_enable: false,
        allow_reset: false,
        function_options: RandomFnOptions {
            max_nodes: 4,
            max_bit_width: 8,
            enabled_operations: OperationSet::new([RandomOperation::Literal]),
            ..RandomFnOptions::default()
        },
        ..RandomBlockOptions::default()
    };
    let mut entropy = DepletableBytes::new(&[]);
    let generated = generate_block(&mut entropy, &options, StopPolicy::ExactBodyNodes(0)).unwrap();
    let package = generated.clone().into_top_package("feedback_block_package");

    validate_generated_block_package(&package);
    let register = &generated.metadata.registers[0];
    let read_index = generated
        .function
        .nodes
        .iter()
        .position(|node| {
            matches!(
                &node.payload,
                NodePayload::RegisterRead { register: read_register }
                    if read_register == &register.name
            )
        })
        .unwrap();
    let write_arg = generated
        .function
        .nodes
        .iter()
        .find_map(|node| match &node.payload {
            NodePayload::RegisterWrite {
                arg,
                register: write_register,
                ..
            } if write_register == &register.name => Some(*arg),
            _ => None,
        })
        .unwrap();

    assert_eq!(write_arg.index, read_index);
    assert_eq!(
        generated.stats.live_operations.get("register_write"),
        Some(&1)
    );
}

#[test]
fn random_package_generation_is_bounded_acyclic_and_reaches_configured_maximum() {
    let options = RandomFnOptions {
        max_params: 4,
        max_nodes: 24,
        max_functions: 5,
        max_invokes_per_function: 3,
        enabled_operations: OperationSet::new([RandomOperation::Literal, RandomOperation::Invoke]),
        ..RandomFnOptions::default()
    };
    let mut entropy = RngEntropy::new(Pcg64Mcg::new(0x1a70_ca11));
    let mut observed_maximum_function_count = false;
    let mut observed_non_wrapper_signature = false;
    let mut observed_reused_callee = false;
    for _ in 0..100 {
        let generated =
            generate_package(&mut entropy, &options, StopPolicy::ExactBodyNodes(16)).unwrap();
        verify_package(&generated.package).unwrap();
        let function_count = generated.package.members.len();
        observed_maximum_function_count |= function_count == options.max_functions;
        assert!((1..=options.max_functions).contains(&function_count));
        assert_eq!(generated.function_stats.len(), function_count);
        assert!(
            generated
                .function_stats
                .iter()
                .all(|stats| stats.emitted_node_count <= options.max_nodes)
        );
        let mut called_names = HashSet::new();
        let mut invoke_count = 0;
        for member in &generated.package.members {
            let PackageMember::Function(function) = member else {
                unreachable!("generated packages contain only functions")
            };
            let mut function_invoke_count = 0;
            for node in &function.nodes {
                let NodePayload::Invoke { to_apply, .. } = &node.payload else {
                    continue;
                };
                function_invoke_count += 1;
                invoke_count += 1;
                observed_reused_callee |= !called_names.insert(to_apply.clone());
                let callee = generated.package.get_fn(to_apply).unwrap();
                observed_non_wrapper_signature |=
                    FunctionSignature::from_fn(function) != FunctionSignature::from_fn(callee);
            }
            assert!(function_invoke_count <= options.max_invokes_per_function);
        }
        assert!(invoke_count > 0);
    }
    assert!(observed_maximum_function_count);
    assert!(observed_non_wrapper_signature);
    assert!(observed_reused_callee);
}

#[test]
fn random_package_counted_for_generation_is_bounded_and_covers_forms() {
    let options = RandomFnOptions {
        max_params: 5,
        max_nodes: 24,
        max_bit_width: 16,
        allow_zero_width_bits: true,
        max_functions: 8,
        max_invokes_per_function: 0,
        max_counted_fors_per_function: 3,
        max_counted_for_trip_count: 5,
        max_counted_for_stride: 4,
        max_nested_counted_for_iterations: 12,
        enabled_operations: OperationSet::new([
            RandomOperation::Literal,
            RandomOperation::CountedFor,
        ]),
        ..RandomFnOptions::default()
    };
    let mut entropy = RngEntropy::new(Pcg64Mcg::new(0xc0a7_edf0));
    let mut saw_trip_count_zero = false;
    let mut saw_trip_count_one = false;
    let mut saw_trip_count_many = false;
    let mut saw_stride_zero = false;
    let mut saw_stride_one = false;
    let mut saw_stride_many = false;
    let mut saw_no_invariant_args = false;
    let mut saw_invariant_args = false;
    let mut saw_aggregate_carry = false;
    let mut saw_zero_width_induction = false;
    let mut saw_nested_loop = false;
    for _ in 0..100 {
        let generated =
            generate_package(&mut entropy, &options, StopPolicy::ExactBodyNodes(16)).unwrap();
        verify_package(&generated.package).unwrap();
        let ir_text = generated.package.to_string();
        xlsynth::IrPackage::parse_ir(&ir_text, None)
            .unwrap_or_else(|error| panic!("libxls rejected generated PIR:\n{ir_text}\n{error}"));

        let mut memo = HashMap::new();
        for member in &generated.package.members {
            let PackageMember::Function(function) = member else {
                unreachable!("generated packages contain only functions")
            };
            assert!(
                max_nested_counted_for_iterations(&generated.package, &function.name, &mut memo)
                    <= options.max_nested_counted_for_iterations
            );
            let mut function_counted_for_count = 0;
            for node in &function.nodes {
                let NodePayload::CountedFor {
                    init,
                    trip_count,
                    stride,
                    body,
                    invariant_args,
                } = &node.payload
                else {
                    continue;
                };
                function_counted_for_count += 1;
                saw_trip_count_zero |= *trip_count == 0;
                saw_trip_count_one |= *trip_count == 1;
                saw_trip_count_many |= *trip_count > 1;
                saw_stride_zero |= *stride == 0;
                saw_stride_one |= *stride == 1;
                saw_stride_many |= *stride > 1;
                saw_no_invariant_args |= invariant_args.is_empty();
                saw_invariant_args |= !invariant_args.is_empty();
                saw_aggregate_carry |=
                    matches!(function.get_node(*init).ty, Type::Array(_) | Type::Tuple(_));

                let body_fn = generated.package.get_fn(body).unwrap();
                saw_zero_width_induction |= body_fn.params[0].ty == Type::Bits(0);
                saw_nested_loop |= body_fn
                    .nodes
                    .iter()
                    .any(|node| matches!(node.payload, NodePayload::CountedFor { .. }));
            }
            assert!(function_counted_for_count <= options.max_counted_fors_per_function);
        }
    }
    assert!(saw_trip_count_zero);
    assert!(saw_trip_count_one);
    assert!(saw_trip_count_many);
    assert!(saw_stride_zero);
    assert!(saw_stride_one);
    assert!(saw_stride_many);
    assert!(saw_no_invariant_args);
    assert!(saw_invariant_args);
    assert!(saw_aggregate_carry);
    assert!(saw_zero_width_induction);
    assert!(saw_nested_loop);
}

#[test]
fn max_counted_fors_per_function_zero_excludes_counted_for_nodes() {
    let options = RandomFnOptions {
        max_params: 5,
        max_nodes: 24,
        max_functions: 8,
        max_counted_fors_per_function: 0,
        enabled_operations: OperationSet::new([
            RandomOperation::Literal,
            RandomOperation::CountedFor,
        ]),
        ..RandomFnOptions::default()
    };
    let mut entropy = RngEntropy::new(Pcg64Mcg::new(0x0ff1_ce00));
    for _ in 0..25 {
        let generated =
            generate_package(&mut entropy, &options, StopPolicy::ExactBodyNodes(16)).unwrap();
        assert!(generated.package.members.iter().all(|member| {
            let PackageMember::Function(function) = member else {
                unreachable!("generated packages contain only functions")
            };
            function
                .nodes
                .iter()
                .all(|node| !matches!(node.payload, NodePayload::CountedFor { .. }))
        }));
    }
}

#[test]
fn constrained_bits_result_reuses_or_materializes_values_by_width() {
    let options = RandomFnOptions {
        max_params: 1,
        max_nodes: 2,
        max_bit_width: 32,
        allow_arrays: false,
        allow_tuples: false,
        ..RandomFnOptions::default()
    };
    let cases = [
        (
            FunctionSignature {
                params: vec![Type::Bits(8)],
                return_type: Type::Bits(8),
            },
            "get_param",
        ),
        (
            FunctionSignature {
                params: vec![Type::Bits(16)],
                return_type: Type::Bits(8),
            },
            "bit_slice",
        ),
        (
            FunctionSignature {
                params: vec![Type::Bits(8)],
                return_type: Type::Bits(16),
            },
            "zero_ext",
        ),
        (
            FunctionSignature {
                params: vec![],
                return_type: Type::Bits(8),
            },
            "literal",
        ),
    ];

    for (signature, expected_operation) in cases {
        let mut entropy = DepletableBytes::new(&[0; 32]);
        let generated = generate_fn_with_signature(
            &mut entropy,
            &options,
            StopPolicy::ExactBodyNodes(0),
            &signature,
        )
        .unwrap();
        validate_generated(&generated.function);
        assert_signature(&generated.function, &signature);
        let ret_node = generated
            .function
            .get_node(generated.function.ret_node_ref.unwrap());
        assert_eq!(ret_node.payload.get_operator(), expected_operation);
    }
}

#[test]
fn constrained_aggregate_result_materializes_recursively_from_bits_values() {
    let signature = FunctionSignature {
        params: vec![Type::Bits(5)],
        return_type: Type::Tuple(vec![
            Box::new(Type::Bits(5)),
            Box::new(Type::new_array(Type::Bits(5), 3)),
        ]),
    };
    let options = RandomFnOptions {
        max_params: 1,
        max_nodes: 4,
        max_bit_width: 8,
        max_type_depth: 2,
        max_aggregate_leaves: 4,
        max_array_length: 3,
        max_tuple_length: 2,
        allow_arrays: true,
        allow_tuples: true,
        ..RandomFnOptions::default()
    };
    let mut entropy = DepletableBytes::new(&[]);
    let generated = generate_fn_with_signature(
        &mut entropy,
        &options,
        StopPolicy::ExactBodyNodes(0),
        &signature,
    )
    .unwrap();

    validate_generated(&generated.function);
    assert_signature(&generated.function, &signature);
    assert_eq!(generated.stats.emitted_node_count, 3);
    assert_eq!(generated.stats.live_operations.get("array"), Some(&1));
    assert_eq!(generated.stats.live_operations.get("tuple"), Some(&1));
}

#[test]
fn constrained_empty_tuple_result_is_supported_without_array_zero_lengths() {
    let signature = FunctionSignature {
        params: vec![],
        return_type: Type::Tuple(vec![]),
    };
    let options = RandomFnOptions {
        max_params: 0,
        max_nodes: 1,
        max_tuple_length: 0,
        allow_arrays: true,
        allow_tuples: true,
        ..RandomFnOptions::default()
    };
    let mut entropy = DepletableBytes::new(&[]);
    let generated = generate_fn_with_signature(
        &mut entropy,
        &options,
        StopPolicy::ExactBodyNodes(0),
        &signature,
    )
    .unwrap();

    validate_generated(&generated.function);
    assert_signature(&generated.function, &signature);
    assert!(matches!(
        generated.function.nodes[1].payload,
        NodePayload::Tuple(ref elements) if elements.is_empty()
    ));
    let ir_text = generated.into_top_package("empty_tuple").to_string();
    xlsynth::IrPackage::parse_ir(&ir_text, None).unwrap();
}

#[test]
fn zero_width_generated_values_use_the_unique_zero_representation() {
    let mut entropy = DepletableBytes::new(&[]);
    let value = generate_uniform_value(&mut entropy, &Type::Bits(0));
    assert_eq!(value.to_string(), "bits[0]:0");
}

#[test]
fn biased_generated_bits_include_corner_patterns() {
    fn generate(width: usize, words: &[u64]) -> String {
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        let mut entropy = DepletableBytes::new(&bytes);
        generate_biased_value(&mut entropy, &Type::Bits(width)).to_string()
    }

    assert_eq!(generate(8, &[0]), "bits[8]:0");
    assert_eq!(generate(8, &[1]), "bits[8]:255");
    assert_eq!(generate(8, &[2]), "bits[8]:128");
    assert_eq!(generate(8, &[3]), "bits[8]:127");
    assert_eq!(generate(8, &[4, 3]), "bits[8]:8");
    assert_eq!(generate(8, &[5, 3]), "bits[8]:7");
    assert_eq!(generate(8, &[6, 3]), "bits[8]:224");
    assert_eq!(generate(8, &[7, 0x5a]), "bits[8]:90");

    for selector in 0..=6 {
        assert_eq!(generate(0, &[selector, 3]), "bits[0]:0");
    }
}

#[test]
fn argument_sets_start_with_whole_input_corner_patterns() {
    let signature = FunctionSignature {
        params: vec![Type::Bits(8), Type::Bits(3)],
        return_type: Type::Bits(8),
    };
    let options = RandomFnOptions {
        max_params: 2,
        max_nodes: 3,
        max_bit_width: 8,
        ..RandomFnOptions::default()
    };
    let mut entropy = DepletableBytes::new(&[]);
    let generated = generate_fn_with_signature(
        &mut entropy,
        &options,
        StopPolicy::ExactBodyNodes(0),
        &signature,
    )
    .unwrap();

    let sets = generate_argument_sets_from_seed(&generated.function, 0x1234, 3);
    assert_eq!(sets.len(), 3);
    assert_eq!(
        sets[0].iter().map(ToString::to_string).collect::<Vec<_>>(),
        ["bits[8]:0", "bits[3]:0"]
    );
    assert_eq!(
        sets[1].iter().map(ToString::to_string).collect::<Vec<_>>(),
        ["bits[8]:255", "bits[3]:7"]
    );
    assert!(generate_argument_sets_from_seed(&generated.function, 0x1234, 0).is_empty());
}

#[test]
fn constrained_signature_rejects_types_disallowed_by_options() {
    let scalar_options = RandomFnOptions {
        max_params: 1,
        max_nodes: 8,
        max_bit_width: 8,
        allow_arrays: false,
        allow_tuples: false,
        ..RandomFnOptions::default()
    };
    let invalid_signatures = [
        FunctionSignature {
            params: vec![],
            return_type: Type::Bits(9),
        },
        FunctionSignature {
            params: vec![],
            return_type: Type::new_array(Type::Bits(1), 2),
        },
        FunctionSignature {
            params: vec![],
            return_type: Type::Tuple(vec![Box::new(Type::Bits(1))]),
        },
        FunctionSignature {
            params: vec![],
            return_type: Type::Token,
        },
    ];
    for signature in invalid_signatures {
        let mut entropy = DepletableBytes::new(&[]);
        assert!(matches!(
            generate_fn_with_signature(
                &mut entropy,
                &scalar_options,
                StopPolicy::ExactBodyNodes(0),
                &signature
            ),
            Err(GenerationError::InvalidSignature(_))
        ));
    }
}

#[test]
fn probabilistic_same_signature_pairs_are_valid_and_exactly_matched() {
    let options = RandomFnOptions {
        max_params: 5,
        max_nodes: 128,
        max_bit_width: 64,
        max_type_depth: 3,
        max_aggregate_leaves: 32,
        max_array_length: 4,
        max_tuple_length: 4,
        allow_arrays: true,
        allow_tuples: true,
        ..RandomFnOptions::default()
    };
    let mut first_entropy = RngEntropy::new(Pcg64Mcg::new(0x551f_25b7));
    let mut second_entropy = RngEntropy::new(Pcg64Mcg::new(0x93ce_b891));
    let mut saw_array_signature = false;
    let mut saw_tuple_signature = false;
    for _ in 0..1_000 {
        let (first, second) = generate_same_signature_pair(
            &mut first_entropy,
            &mut second_entropy,
            &options,
            StopPolicy::ExactBodyNodes(48),
        )
        .unwrap();
        validate_generated(&first.function);
        validate_generated(&second.function);
        let signature = FunctionSignature::from_fn(&first.function);
        assert_signature(&second.function, &signature);
        saw_array_signature |= signature
            .params
            .iter()
            .chain([&signature.return_type])
            .any(type_has_array);
        saw_tuple_signature |= signature
            .params
            .iter()
            .chain([&signature.return_type])
            .any(type_has_tuple);
        assert!(second.stats.emitted_node_count <= options.max_nodes);
    }
    assert!(saw_array_signature);
    assert!(saw_tuple_signature);
}

#[test]
fn same_signature_pair_uses_entropy_for_both_depletable_functions() {
    let options = RandomFnOptions {
        max_params: 0,
        max_nodes: 64,
        max_bit_width: 8,
        allow_arrays: false,
        allow_tuples: false,
        enabled_operations: OperationSet::new([RandomOperation::Literal]),
        ..RandomFnOptions::default()
    };
    let data = vec![0xa5; 256];
    let (mut first_entropy, mut second_entropy) = DepletableBytes::split(&data);
    let (first, second) = generate_same_signature_pair(
        &mut first_entropy,
        &mut second_entropy,
        &options,
        StopPolicy::WhenEntropyDepleted,
    )
    .unwrap();

    validate_generated(&first.function);
    validate_generated(&second.function);
    assert!(first.stats.emitted_node_count > 1);
    assert!(first.stats.emitted_node_count < options.max_nodes);
    assert!(second.stats.emitted_node_count > 1);
}

#[test]
fn depletable_entropy_samples_are_valid() {
    let options = RandomFnOptions {
        max_params: 5,
        max_nodes: 40,
        max_bit_width: 64,
        allow_arrays: true,
        allow_tuples: true,
        ..RandomFnOptions::default()
    };
    let mut rng = Pcg64Mcg::new(0x68db_6b32);
    for _ in 0..10_000 {
        let length = (rng.next_u64() % 257) as usize;
        let mut data = vec![0_u8; length];
        rng.fill_bytes(&mut data);
        let mut entropy = DepletableBytes::new(&data);
        let generated =
            generate_fn(&mut entropy, &options, StopPolicy::WhenEntropyDepleted).unwrap();
        validate_generated(&generated.function);
    }
}

#[test]
fn exact_body_length_obeys_node_and_width_limits() {
    let options = RandomFnOptions {
        max_params: 0,
        max_nodes: 48,
        max_bit_width: 11,
        allow_arrays: false,
        allow_tuples: false,
        ..RandomFnOptions::default()
    };
    let mut entropy = RngEntropy::new(Pcg64Mcg::new(0xdad0_beef));
    let generated = generate_fn(&mut entropy, &options, StopPolicy::ExactBodyNodes(40)).unwrap();

    validate_generated(&generated.function);
    assert_eq!(generated.stats.emitted_node_count, 40);
    assert!(
        generated
            .stats
            .emitted_bits_widths
            .iter()
            .all(|width| *width <= options.max_bit_width)
    );
    assert!(
        function_types(&generated.function).all(|ty| !type_has_array(ty) && !type_has_tuple(ty))
    );
}

#[test]
fn geometric_stop_policy_produces_bounded_body_size_variety() {
    let options = RandomFnOptions {
        max_params: 0,
        max_nodes: 64,
        allow_arrays: false,
        allow_tuples: false,
        enabled_operations: OperationSet::new([RandomOperation::Literal]),
        ..RandomFnOptions::default()
    };
    let policy = StopPolicy::Geometric {
        min_body_nodes: 3,
        max_body_nodes: 20,
        stop_numerator: 1,
        stop_denominator: 4,
    };
    let mut entropy = RngEntropy::new(Pcg64Mcg::new(0x1a77_5eed));
    let mut lengths = BTreeSet::new();
    for _ in 0..500 {
        let generated = generate_fn(&mut entropy, &options, policy).unwrap();
        validate_generated(&generated.function);
        assert!((3..=20).contains(&generated.stats.emitted_node_count));
        assert!(
            generated
                .stats
                .emitted_operations
                .keys()
                .all(|operation| operation == "literal")
        );
        lengths.insert(generated.stats.emitted_node_count);
    }
    assert!(lengths.len() >= 8);
}

#[test]
fn probabilistic_scalar_generation_covers_enabled_operations_and_liveness() {
    let operations = OperationSet::new([
        RandomOperation::Literal,
        RandomOperation::Identity,
        RandomOperation::Not,
        RandomOperation::Neg,
        RandomOperation::And,
        RandomOperation::Nand,
        RandomOperation::Nor,
        RandomOperation::Or,
        RandomOperation::Xor,
        RandomOperation::Add,
        RandomOperation::Sub,
        RandomOperation::Umul,
        RandomOperation::Smul,
        RandomOperation::Eq,
        RandomOperation::Ne,
        RandomOperation::Ugt,
        RandomOperation::Uge,
        RandomOperation::Ult,
        RandomOperation::Ule,
        RandomOperation::Sgt,
        RandomOperation::Sge,
        RandomOperation::Slt,
        RandomOperation::Sle,
        RandomOperation::Shll,
        RandomOperation::ZeroExt,
        RandomOperation::SignExt,
        RandomOperation::BitSlice,
        RandomOperation::Concat,
    ]);
    let options = RandomFnOptions {
        max_params: 5,
        max_nodes: 36,
        max_bit_width: 64,
        allow_arrays: false,
        allow_tuples: false,
        enabled_operations: operations.clone(),
        ..RandomFnOptions::default()
    };
    let mut entropy = RngEntropy::new(Pcg64Mcg::new(0x719b_2a31));
    let mut emitted = HashSet::new();
    let mut live = HashSet::new();
    let mut widths = BTreeSet::new();
    let mut saw_dead_nodes = false;
    let mut saw_nontrivial_live_cone = false;
    for _ in 0..2_000 {
        let generated =
            generate_fn(&mut entropy, &options, StopPolicy::ExactBodyNodes(30)).unwrap();
        validate_generated(&generated.function);
        emitted.extend(generated.stats.emitted_operations.keys().cloned());
        live.extend(generated.stats.live_operations.keys().cloned());
        widths.extend(generated.stats.emitted_bits_widths.iter().copied());
        saw_dead_nodes |= generated.stats.live_node_count < generated.stats.emitted_node_count;
        saw_nontrivial_live_cone |= generated.stats.live_node_count >= 6;
    }

    for operation in operations.iter() {
        assert!(
            emitted.contains(operation.name()),
            "never emitted {}",
            operation.name()
        );
        assert!(
            live.contains(operation.name()),
            "never emitted live {}",
            operation.name()
        );
    }
    assert!(widths.contains(&1));
    assert!(widths.contains(&64));
    assert!(widths.len() >= 20);
    assert!(saw_dead_nodes);
    assert!(saw_nontrivial_live_cone);
}

#[test]
fn probabilistic_aggregate_options_generate_arrays_and_tuples() {
    let options = RandomFnOptions {
        max_params: 5,
        max_nodes: 40,
        max_bit_width: 37,
        max_type_depth: 3,
        max_aggregate_leaves: 32,
        max_array_length: 4,
        max_tuple_length: 4,
        allow_arrays: true,
        allow_tuples: true,
        allow_assumed_in_bounds: true,
        ..RandomFnOptions::default()
    };
    let mut entropy = RngEntropy::new(Pcg64Mcg::new(0xf0a5_a731));
    let mut operations = HashSet::new();
    let mut saw_array_type = false;
    let mut saw_tuple_type = false;
    let mut saw_empty_tuple = false;
    let mut saw_nested_tuple = false;
    let mut saw_multidimensional_array = false;
    let mut saw_array_of_tuple = false;
    let mut saw_tuple_with_array_element = false;
    let mut saw_assumed_array_index = false;
    let mut saw_assumed_array_update = false;
    let mut saw_zero_params = false;
    let mut saw_max_params = false;
    for _ in 0..2_000 {
        let generated =
            generate_fn(&mut entropy, &options, StopPolicy::ExactBodyNodes(32)).unwrap();
        validate_generated(&generated.function);
        operations.extend(generated.stats.emitted_operations.keys().cloned());
        for node in &generated.function.nodes {
            match &node.payload {
                NodePayload::ArrayIndex {
                    assumed_in_bounds: true,
                    ..
                } => saw_assumed_array_index = true,
                NodePayload::ArrayUpdate {
                    assumed_in_bounds: true,
                    ..
                } => saw_assumed_array_update = true,
                _ => {}
            }
        }
        saw_array_type |= function_types(&generated.function).any(type_has_array);
        saw_tuple_type |= function_types(&generated.function).any(type_has_tuple);
        saw_empty_tuple |= function_types(&generated.function).any(type_has_empty_tuple);
        saw_nested_tuple |= function_types(&generated.function).any(|ty| {
            type_has_nested_tuple(ty, /* inside_tuple= */ false)
        });
        saw_multidimensional_array |=
            function_types(&generated.function).any(type_has_multidimensional_array);
        saw_array_of_tuple |= function_types(&generated.function).any(type_has_array_of_tuple);
        saw_tuple_with_array_element |=
            function_types(&generated.function).any(type_has_tuple_with_array_element);
        saw_zero_params |= generated.function.params.is_empty();
        saw_max_params |= generated.function.params.len() == options.max_params;
        assert!(generated.function.params.len() <= options.max_params);
        assert!(generated.stats.emitted_node_count <= options.max_nodes);
        for ty in function_types(&generated.function) {
            assert_type_obeys_options(ty, &options);
            if let Type::Array(array) = ty {
                assert!(array.element_count > 0);
            }
        }
    }

    assert!(saw_array_type);
    assert!(saw_tuple_type);
    assert!(saw_empty_tuple);
    assert!(saw_nested_tuple);
    assert!(saw_multidimensional_array);
    assert!(saw_array_of_tuple);
    assert!(saw_tuple_with_array_element);
    assert!(saw_assumed_array_index);
    assert!(saw_assumed_array_update);
    assert!(saw_zero_params);
    assert!(saw_max_params);
    assert!(operations.contains("array"));
    assert!(operations.contains("array_index"));
    assert!(operations.contains("tuple"));
    assert!(operations.contains("tuple_index"));
}

#[test]
fn probabilistic_disabled_aggregate_options_exclude_aggregate_nodes() {
    let options = RandomFnOptions {
        max_params: 5,
        max_nodes: 32,
        max_bit_width: 19,
        allow_arrays: false,
        allow_tuples: false,
        ..RandomFnOptions::default()
    };
    let mut entropy = RngEntropy::new(Pcg64Mcg::new(0x12f3_5521));
    for _ in 0..500 {
        let generated =
            generate_fn(&mut entropy, &options, StopPolicy::ExactBodyNodes(24)).unwrap();
        validate_generated(&generated.function);
        assert!(
            function_types(&generated.function)
                .all(|ty| !type_has_array(ty) && !type_has_tuple(ty))
        );
        assert!(!generated.stats.emitted_operations.contains_key("array"));
        assert!(!generated.stats.emitted_operations.contains_key("tuple"));
    }
}

#[test]
fn probabilistic_assumed_in_bounds_option_controls_array_attributes() {
    let options = RandomFnOptions {
        max_params: 5,
        max_nodes: 48,
        max_bit_width: 16,
        allow_arrays: true,
        allow_tuples: false,
        enabled_operations: OperationSet::new([
            RandomOperation::Literal,
            RandomOperation::Array,
            RandomOperation::ArrayIndex,
            RandomOperation::ArrayUpdate,
        ]),
        ..RandomFnOptions::default()
    };
    let mut entropy = RngEntropy::new(Pcg64Mcg::new(0x481b_00dd));
    for _ in 0..500 {
        let generated =
            generate_fn(&mut entropy, &options, StopPolicy::ExactBodyNodes(40)).unwrap();
        validate_generated(&generated.function);
        assert!(generated.function.nodes.iter().all(|node| !matches!(
            &node.payload,
            NodePayload::ArrayIndex {
                assumed_in_bounds: true,
                ..
            } | NodePayload::ArrayUpdate {
                assumed_in_bounds: true,
                ..
            }
        )));
    }

    let enabled = RandomFnOptions {
        allow_assumed_in_bounds: true,
        ..options
    };
    let mut entropy = RngEntropy::new(Pcg64Mcg::new(0x481b_00de));
    let mut saw_index = false;
    let mut saw_update = false;
    for _ in 0..500 {
        let generated =
            generate_fn(&mut entropy, &enabled, StopPolicy::ExactBodyNodes(40)).unwrap();
        validate_generated(&generated.function);
        saw_index |= generated.function.nodes.iter().any(|node| {
            matches!(
                &node.payload,
                NodePayload::ArrayIndex {
                    assumed_in_bounds: true,
                    ..
                }
            )
        });
        saw_update |= generated.function.nodes.iter().any(|node| {
            matches!(
                &node.payload,
                NodePayload::ArrayUpdate {
                    assumed_in_bounds: true,
                    ..
                }
            )
        });
    }
    assert!(saw_index);
    assert!(saw_update);
}

#[test]
fn probabilistic_wide_widths_are_present_but_sparse() {
    let options = RandomFnOptions {
        max_params: 5,
        max_nodes: 32,
        max_bit_width: 1024,
        allow_arrays: false,
        allow_tuples: false,
        enabled_operations: OperationSet::new([RandomOperation::Literal]),
        ..RandomFnOptions::default()
    };
    let mut entropy = RngEntropy::new(Pcg64Mcg::new(0xd195_3328));
    let mut widths = Vec::new();
    for _ in 0..500 {
        let generated =
            generate_fn(&mut entropy, &options, StopPolicy::ExactBodyNodes(24)).unwrap();
        validate_generated(&generated.function);
        widths.extend(
            function_types(&generated.function).filter_map(|ty| match ty {
                Type::Bits(width) => Some(*width),
                _ => None,
            }),
        );
    }
    let wide_count = widths.iter().filter(|width| **width > 64).count();
    assert!(wide_count > widths.len() / 20, "wide types were too rare");
    assert!(
        wide_count < widths.len() / 4,
        "wide types dominated generation"
    );
    assert!(widths.iter().any(|width| *width <= 64));
    assert!(widths.iter().any(|width| *width >= 900));
}

#[test]
fn probabilistic_generation_covers_new_standard_operations_and_gate() {
    let requested = [
        RandomOperation::Reverse,
        RandomOperation::OrReduce,
        RandomOperation::AndReduce,
        RandomOperation::XorReduce,
        RandomOperation::Shrl,
        RandomOperation::Shra,
        RandomOperation::Udiv,
        RandomOperation::Sdiv,
        RandomOperation::Umod,
        RandomOperation::Smod,
        RandomOperation::Umulp,
        RandomOperation::Smulp,
        RandomOperation::Gate,
        RandomOperation::ArrayConcat,
        RandomOperation::ArraySlice,
        RandomOperation::ArrayUpdate,
        RandomOperation::DynamicBitSlice,
        RandomOperation::BitSliceUpdate,
        RandomOperation::Sel,
        RandomOperation::PrioritySel,
        RandomOperation::OneHotSel,
        RandomOperation::OneHot,
        RandomOperation::Encode,
        RandomOperation::Decode,
    ];
    let operations = OperationSet::new(
        [RandomOperation::Literal, RandomOperation::Array]
            .into_iter()
            .chain(requested),
    );
    let options = RandomFnOptions {
        max_params: 5,
        max_nodes: 72,
        max_bit_width: 64,
        max_type_depth: 3,
        max_aggregate_leaves: 64,
        max_array_length: 8,
        max_tuple_length: 4,
        allow_arrays: true,
        allow_tuples: true,
        allow_gate: true,
        enabled_operations: operations,
        ..RandomFnOptions::default()
    };
    let mut entropy = RngEntropy::new(Pcg64Mcg::new(0x2ae3_1f73));
    let mut emitted = HashSet::new();
    let mut live = HashSet::new();
    for _ in 0..4_000 {
        let generated =
            generate_fn(&mut entropy, &options, StopPolicy::ExactBodyNodes(64)).unwrap();
        validate_generated(&generated.function);
        emitted.extend(generated.stats.emitted_operations.keys().cloned());
        live.extend(generated.stats.live_operations.keys().cloned());
        for ty in function_types(&generated.function) {
            assert_type_obeys_options(ty, &options);
        }
    }

    for operation in requested {
        assert!(
            emitted.contains(operation.name()),
            "never emitted {}",
            operation.name()
        );
        assert!(
            live.contains(operation.name()),
            "never emitted live {}",
            operation.name()
        );
    }
}

#[test]
fn probabilistic_select_generation_is_accepted_by_libxls() {
    let selected_operations = [
        RandomOperation::Sel,
        RandomOperation::PrioritySel,
        RandomOperation::OneHotSel,
    ];
    let options = RandomFnOptions {
        max_params: 5,
        max_nodes: 48,
        max_bit_width: 8,
        allow_arrays: false,
        allow_tuples: false,
        enabled_operations: OperationSet::new(
            [RandomOperation::Literal]
                .into_iter()
                .chain(selected_operations),
        ),
        ..RandomFnOptions::default()
    };
    let mut entropy = RngEntropy::new(Pcg64Mcg::new(0x0a42_c195));
    let mut emitted = HashSet::new();
    let mut saw_select_with_default = false;
    for sample in 0..500 {
        let generated =
            generate_fn(&mut entropy, &options, StopPolicy::ExactBodyNodes(40)).unwrap();
        emitted.extend(generated.stats.emitted_operations.keys().cloned());
        saw_select_with_default |= generated.function.nodes.iter().any(|node| {
            matches!(
                &node.payload,
                NodePayload::Sel { cases, default, .. }
                    if !cases.is_empty() && default.is_some()
            )
        });
        let ir_text = generated
            .into_top_package(format!("select_parse_{sample}"))
            .to_string();
        xlsynth::IrPackage::parse_ir(&ir_text, None)
            .unwrap_or_else(|error| panic!("libxls rejected generated PIR:\n{ir_text}\n{error}"));
    }

    for operation in selected_operations {
        assert!(emitted.contains(operation.name()));
    }
    assert!(saw_select_with_default);
}

#[test]
fn probabilistic_new_operand_shapes_are_generated() {
    let options = RandomFnOptions {
        max_params: 5,
        max_nodes: 72,
        max_bit_width: 32,
        max_type_depth: 3,
        max_aggregate_leaves: 48,
        max_array_length: 5,
        max_tuple_length: 4,
        max_nary_operands: 5,
        allow_arrays: true,
        allow_tuples: true,
        allow_zero_width_bits: true,
        allow_arbitrary_width_multiply: true,
        allow_gate: true,
        enabled_operations: OperationSet::new([
            RandomOperation::Literal,
            RandomOperation::Identity,
            RandomOperation::And,
            RandomOperation::Or,
            RandomOperation::Concat,
            RandomOperation::Umul,
            RandomOperation::Smul,
            RandomOperation::Umulp,
            RandomOperation::Smulp,
            RandomOperation::Eq,
            RandomOperation::Ne,
            RandomOperation::Shll,
            RandomOperation::Shrl,
            RandomOperation::Shra,
            RandomOperation::Gate,
            RandomOperation::ZeroExt,
            RandomOperation::SignExt,
            RandomOperation::BitSlice,
            RandomOperation::DynamicBitSlice,
            RandomOperation::Array,
            RandomOperation::ArrayIndex,
            RandomOperation::ArrayConcat,
            RandomOperation::ArrayUpdate,
            RandomOperation::Tuple,
            RandomOperation::Sel,
            RandomOperation::Encode,
            RandomOperation::Decode,
        ]),
        ..RandomFnOptions::default()
    };
    let mut entropy = RngEntropy::new(Pcg64Mcg::new(0x73ae_d04c));
    let mut saw_empty_tuple = false;
    let mut saw_unary_nary = false;
    let mut saw_three_or_more_nary = false;
    let mut saw_empty_concat = false;
    let mut saw_width_varying_multiply = false;
    let mut saw_arbitrary_mulp_result_width = false;
    let mut saw_mixed_width_shift = false;
    let mut saw_noop_extension = false;
    let mut saw_zero_extend_from_zero = false;
    let mut saw_zero_width_concat_operand = false;
    let mut saw_aggregate_literal = false;
    let mut saw_aggregate_identity = false;
    let mut saw_aggregate_equality = false;
    let mut saw_aggregate_gate = false;
    let mut saw_zero_index_array_op = false;
    let mut saw_multidimensional_array_op = false;
    let mut saw_zero_width_operation = false;
    let mut saw_select_with_default = false;
    let mut saw_unary_array_concat = false;
    let mut saw_three_or_more_array_concat = false;

    for sample in 0..1_500 {
        let generated =
            generate_fn(&mut entropy, &options, StopPolicy::ExactBodyNodes(64)).unwrap();
        validate_generated(&generated.function);
        for node in &generated.function.nodes {
            match &node.payload {
                NodePayload::Literal(_) => {
                    saw_aggregate_literal |= !matches!(node.ty, Type::Bits(_));
                }
                NodePayload::Tuple(elements) if elements.is_empty() => saw_empty_tuple = true,
                NodePayload::Nary(
                    NaryOp::And | NaryOp::Or | NaryOp::Xor | NaryOp::Nand | NaryOp::Nor,
                    operands,
                ) => {
                    saw_unary_nary |= operands.len() == 1;
                    saw_three_or_more_nary |= operands.len() >= 3;
                }
                NodePayload::Nary(NaryOp::Concat, operands) => {
                    saw_empty_concat |= operands.is_empty();
                    saw_three_or_more_nary |= operands.len() >= 3;
                    saw_zero_width_concat_operand |= operands
                        .iter()
                        .any(|operand| generated.function.get_node(*operand).ty == Type::Bits(0));
                }
                NodePayload::Binop(Binop::Umul | Binop::Smul, lhs, rhs) => {
                    saw_width_varying_multiply |= generated.function.get_node(*lhs).ty
                        != generated.function.get_node(*rhs).ty
                        || node.ty != generated.function.get_node(*lhs).ty;
                }
                NodePayload::Binop(Binop::Umulp | Binop::Smulp, lhs, rhs) => {
                    let (Type::Bits(lhs_width), Type::Bits(rhs_width), Type::Tuple(fields)) = (
                        &generated.function.get_node(*lhs).ty,
                        &generated.function.get_node(*rhs).ty,
                        &node.ty,
                    ) else {
                        panic!("mulp has invalid generated types");
                    };
                    let Type::Bits(result_width) = fields[0].as_ref() else {
                        panic!("mulp result field is not bits");
                    };
                    saw_arbitrary_mulp_result_width |= *result_width != lhs_width + rhs_width;
                }
                NodePayload::Binop(Binop::Shll | Binop::Shrl | Binop::Shra, lhs, rhs) => {
                    saw_mixed_width_shift |= generated.function.get_node(*lhs).ty
                        != generated.function.get_node(*rhs).ty;
                }
                NodePayload::Unop(Unop::Identity, arg) => {
                    saw_aggregate_identity |=
                        !matches!(generated.function.get_node(*arg).ty, Type::Bits(_));
                }
                NodePayload::Binop(Binop::Eq | Binop::Ne, lhs, _) => {
                    saw_aggregate_equality |=
                        !matches!(generated.function.get_node(*lhs).ty, Type::Bits(_));
                }
                NodePayload::Binop(Binop::Gate, _, value) => {
                    saw_aggregate_gate |=
                        !matches!(generated.function.get_node(*value).ty, Type::Bits(_));
                }
                NodePayload::ZeroExt { arg, .. } => {
                    saw_noop_extension |= node.ty == generated.function.get_node(*arg).ty;
                    saw_zero_extend_from_zero |=
                        generated.function.get_node(*arg).ty == Type::Bits(0);
                }
                NodePayload::SignExt { arg, .. } => {
                    saw_noop_extension |= node.ty == generated.function.get_node(*arg).ty;
                }
                NodePayload::ArrayIndex { indices, .. }
                | NodePayload::ArrayUpdate { indices, .. } => {
                    saw_zero_index_array_op |= indices.is_empty();
                    saw_multidimensional_array_op |= indices.len() >= 2;
                }
                NodePayload::ArrayConcat(operands) => {
                    saw_unary_array_concat |= operands.len() == 1;
                    saw_three_or_more_array_concat |= operands.len() >= 3;
                }
                NodePayload::BitSlice { width, .. }
                | NodePayload::DynamicBitSlice { width, .. }
                | NodePayload::Decode { width, .. } => {
                    saw_zero_width_operation |= *width == 0;
                }
                NodePayload::Sel { cases, default, .. } => {
                    saw_select_with_default |= !cases.is_empty() && default.is_some();
                }
                NodePayload::Array(elements) => assert!(!elements.is_empty()),
                _ => {}
            }
        }
        let ir_text = generated
            .into_top_package(format!("covered_shape_parse_{sample}"))
            .to_string();
        xlsynth::IrPackage::parse_ir(&ir_text, None).unwrap_or_else(|error| {
            panic!("libxls rejected covered PIR shape:\n{ir_text}\n{error}")
        });
    }

    assert!(saw_empty_tuple);
    assert!(saw_unary_nary);
    assert!(saw_three_or_more_nary);
    assert!(saw_empty_concat);
    assert!(saw_width_varying_multiply);
    assert!(saw_arbitrary_mulp_result_width);
    assert!(saw_mixed_width_shift);
    assert!(saw_noop_extension);
    assert!(saw_zero_extend_from_zero);
    assert!(saw_zero_width_concat_operand);
    assert!(saw_aggregate_literal);
    assert!(saw_aggregate_identity);
    assert!(saw_aggregate_equality);
    assert!(saw_aggregate_gate);
    assert!(saw_zero_index_array_op);
    assert!(saw_multidimensional_array_op);
    assert!(saw_unary_array_concat);
    assert!(saw_three_or_more_array_concat);
    assert!(saw_zero_width_operation);
    assert!(saw_select_with_default);
}

#[test]
fn empty_case_sel_generation_is_opt_in_for_pir_consumers() {
    let options = RandomFnOptions {
        max_nodes: 32,
        max_bit_width: 8,
        allow_arrays: false,
        allow_tuples: false,
        allow_empty_case_sel: true,
        enabled_operations: OperationSet::new([RandomOperation::Literal, RandomOperation::Sel]),
        ..RandomFnOptions::default()
    };
    let mut entropy = RngEntropy::new(Pcg64Mcg::new(0x781e_002c));
    let mut saw_default_only_sel = false;
    for _ in 0..200 {
        let generated =
            generate_fn(&mut entropy, &options, StopPolicy::ExactBodyNodes(24)).unwrap();
        validate_generated(&generated.function);
        saw_default_only_sel |= generated.function.nodes.iter().any(|node| {
            matches!(
                &node.payload,
                NodePayload::Sel { cases, default, .. }
                    if cases.is_empty() && default.is_some()
            )
        });
    }
    // XLS evaluates this form, but its text parser currently rejects cases=[].
    assert!(saw_default_only_sel);
}

#[test]
fn probabilistic_upstream_standard_generation_is_accepted_by_libxls() {
    let upstream_operations =
        OperationSet::new(OperationSet::all_supported().iter().filter(|operation| {
            !matches!(
                operation,
                RandomOperation::ExtCarryOut
                    | RandomOperation::ExtPrioEncode
                    | RandomOperation::ExtClz
                    | RandomOperation::ExtNormalizeLeft
                    | RandomOperation::ExtMaskLow
                    | RandomOperation::ExtNaryAdd
            )
        }));
    let options = RandomFnOptions {
        max_params: 5,
        max_nodes: 64,
        max_bit_width: 32,
        max_type_depth: 3,
        max_aggregate_leaves: 48,
        max_array_length: 5,
        max_tuple_length: 4,
        max_nary_operands: 5,
        allow_zero_width_bits: true,
        allow_arbitrary_width_multiply: true,
        allow_gate: true,
        enabled_operations: upstream_operations,
        ..RandomFnOptions::default()
    };
    let mut entropy = RngEntropy::new(Pcg64Mcg::new(0xe8a1_52c7));
    for sample in 0..750 {
        let generated =
            generate_fn(&mut entropy, &options, StopPolicy::ExactBodyNodes(56)).unwrap();
        validate_generated(&generated.function);
        let ir_text = generated
            .into_top_package(format!("upstream_random_parse_{sample}"))
            .to_string();
        xlsynth::IrPackage::parse_ir(&ir_text, None)
            .unwrap_or_else(|error| panic!("libxls rejected generated PIR:\n{ir_text}\n{error}"));
    }
}

#[test]
fn probabilistic_expanded_standard_generation_matches_libxls_interpreter() {
    let options = RandomFnOptions {
        max_params: 4,
        max_nodes: 48,
        max_bit_width: 16,
        max_type_depth: 3,
        max_aggregate_leaves: 32,
        max_array_length: 4,
        max_tuple_length: 4,
        max_nary_operands: 4,
        allow_arrays: true,
        allow_tuples: true,
        allow_zero_width_bits: true,
        allow_arbitrary_width_multiply: true,
        allow_gate: true,
        enabled_operations: OperationSet::new([
            RandomOperation::Literal,
            RandomOperation::Identity,
            RandomOperation::Not,
            RandomOperation::Neg,
            RandomOperation::Reverse,
            RandomOperation::OrReduce,
            RandomOperation::AndReduce,
            RandomOperation::XorReduce,
            RandomOperation::And,
            RandomOperation::Nand,
            RandomOperation::Nor,
            RandomOperation::Or,
            RandomOperation::Xor,
            RandomOperation::Add,
            RandomOperation::Sub,
            RandomOperation::Umul,
            RandomOperation::Smul,
            RandomOperation::Udiv,
            RandomOperation::Sdiv,
            RandomOperation::Umod,
            RandomOperation::Smod,
            RandomOperation::Eq,
            RandomOperation::Ne,
            RandomOperation::Ugt,
            RandomOperation::Uge,
            RandomOperation::Ult,
            RandomOperation::Ule,
            RandomOperation::Sgt,
            RandomOperation::Sge,
            RandomOperation::Slt,
            RandomOperation::Sle,
            RandomOperation::Shll,
            RandomOperation::Shrl,
            RandomOperation::Shra,
            RandomOperation::Gate,
            RandomOperation::ZeroExt,
            RandomOperation::SignExt,
            RandomOperation::BitSlice,
            RandomOperation::Concat,
            RandomOperation::Array,
            RandomOperation::ArrayIndex,
            RandomOperation::ArrayConcat,
            RandomOperation::ArraySlice,
            RandomOperation::ArrayUpdate,
            RandomOperation::Tuple,
            RandomOperation::TupleIndex,
            RandomOperation::DynamicBitSlice,
            RandomOperation::BitSliceUpdate,
            RandomOperation::Sel,
            RandomOperation::PrioritySel,
            RandomOperation::OneHotSel,
            RandomOperation::OneHot,
            RandomOperation::Encode,
            RandomOperation::Decode,
        ]),
        ..RandomFnOptions::default()
    };
    let mut graph_entropy = RngEntropy::new(Pcg64Mcg::new(0xb042_0ca9));
    let mut value_entropy = RngEntropy::new(Pcg64Mcg::new(0x16e4_d4f1));
    for sample in 0..300 {
        let generated =
            generate_fn(&mut graph_entropy, &options, StopPolicy::ExactBodyNodes(40)).unwrap();
        let package = generated.into_top_package(format!("expanded_eval_{sample}"));
        let function = package.get_top_fn().unwrap();
        let args = generate_biased_arguments(&mut value_entropy, function);
        let ir_text = package.to_string();
        let xls_package = xlsynth::IrPackage::parse_ir(&ir_text, None)
            .unwrap_or_else(|error| panic!("libxls rejected generated PIR:\n{ir_text}\n{error}"));
        let xls_function = xls_package.get_function(&function.name).unwrap();
        let expected = xls_function.interpret(&args).unwrap();
        let actual = match eval_fn_in_package(&package, function, &args) {
            FnEvalResult::Success(success) => success.value,
            other => panic!("PIR evaluation failed for generated IR:\n{ir_text}\n{other:?}"),
        };
        assert_eq!(actual, expected, "evaluation mismatch for IR:\n{ir_text}");
    }
}

#[test]
fn probabilistic_extension_option_covers_all_extension_operations() {
    let requested = [
        RandomOperation::ExtCarryOut,
        RandomOperation::ExtPrioEncode,
        RandomOperation::ExtClz,
        RandomOperation::ExtNormalizeLeft,
        RandomOperation::ExtMaskLow,
        RandomOperation::ExtNaryAdd,
    ];
    let options = RandomFnOptions {
        max_params: 4,
        max_nodes: 56,
        max_bit_width: 64,
        allow_arrays: false,
        allow_tuples: true,
        allow_extension_ops: true,
        enabled_operations: OperationSet::new(
            [RandomOperation::Literal].into_iter().chain(requested),
        ),
        ..RandomFnOptions::default()
    };
    let mut entropy = RngEntropy::new(Pcg64Mcg::new(0xe971_31c5));
    let mut emitted = HashSet::new();
    let mut live = HashSet::new();
    for _ in 0..2_000 {
        let generated =
            generate_fn(&mut entropy, &options, StopPolicy::ExactBodyNodes(48)).unwrap();
        validate_generated(&generated.function);
        for node in &generated.function.nodes {
            let NodePayload::ExtNormalizeLeft {
                arg,
                normalized_bit_count,
                ..
            } = &node.payload
            else {
                continue;
            };
            let Type::Bits(arg_width) = &generated.function.get_node(*arg).ty else {
                unreachable!("ext_normalize_left argument must be bits-typed");
            };
            assert!(
                normalized_bit_count >= arg_width,
                "ext_normalize_left result must be at least as wide as its operand"
            );
        }
        emitted.extend(generated.stats.emitted_operations.keys().cloned());
        live.extend(generated.stats.live_operations.keys().cloned());
    }

    for operation in requested {
        assert!(
            emitted.contains(operation.name()),
            "never emitted {}",
            operation.name()
        );
        assert!(
            live.contains(operation.name()),
            "never emitted live {}",
            operation.name()
        );
    }
}

#[test]
fn probabilistic_ext_nary_add_covers_term_and_width_forms() {
    let options = RandomFnOptions {
        max_params: 5,
        max_nodes: 64,
        max_bit_width: 128,
        max_nary_operands: 5,
        allow_arrays: false,
        allow_tuples: false,
        allow_extension_ops: true,
        enabled_operations: OperationSet::new([
            RandomOperation::Literal,
            RandomOperation::ExtNaryAdd,
        ]),
        ..RandomFnOptions::default()
    };
    let mut entropy = RngEntropy::new(Pcg64Mcg::new(0xea34_83b7));
    let mut term_counts = BTreeSet::new();
    let mut architectures = BTreeSet::new();
    let mut saw_signed = false;
    let mut saw_unsigned = false;
    let mut saw_negated = false;
    let mut saw_non_negated = false;
    let mut saw_mixed_term_widths = false;
    let mut saw_independent_result_width = false;
    let mut saw_wide_term_narrow_result = false;
    for _ in 0..1_000 {
        let generated =
            generate_fn(&mut entropy, &options, StopPolicy::ExactBodyNodes(48)).unwrap();
        validate_generated(&generated.function);
        for node in &generated.function.nodes {
            let NodePayload::ExtNaryAdd { terms, arch } = &node.payload else {
                continue;
            };
            term_counts.insert(terms.len());
            architectures.insert(match arch {
                None => "none",
                Some(ExtNaryAddArchitecture::RippleCarry) => "ripple",
                Some(ExtNaryAddArchitecture::KoggeStone) => "kogge_stone",
                Some(ExtNaryAddArchitecture::BrentKung) => "brent_kung",
            });
            saw_signed |= terms.iter().any(|term| term.signed);
            saw_unsigned |= terms.iter().any(|term| !term.signed);
            saw_negated |= terms.iter().any(|term| term.negated);
            saw_non_negated |= terms.iter().any(|term| !term.negated);
            let term_widths = terms
                .iter()
                .map(|term| match &generated.function.get_node(term.operand).ty {
                    Type::Bits(width) => *width,
                    _ => unreachable!("ext_nary_add term is bits-typed"),
                })
                .collect::<BTreeSet<_>>();
            saw_mixed_term_widths |= term_widths.len() > 1;
            let Type::Bits(result_width) = &node.ty else {
                unreachable!("ext_nary_add result is bits-typed");
            };
            saw_independent_result_width |= term_widths.iter().any(|width| width != result_width);
            saw_wide_term_narrow_result |=
                *result_width <= 64 && term_widths.iter().any(|width| *width > 64);
        }
    }
    assert_eq!(term_counts, BTreeSet::from([0, 1, 2, 3, 4, 5]));
    assert_eq!(
        architectures,
        BTreeSet::from(["none", "ripple", "kogge_stone", "brent_kung"])
    );
    assert!(saw_signed);
    assert!(saw_unsigned);
    assert!(saw_negated);
    assert!(saw_non_negated);
    assert!(saw_mixed_term_widths);
    assert!(saw_independent_result_width);
    assert!(saw_wide_term_narrow_result);
}

#[test]
fn probabilistic_gate_and_extension_options_exclude_disabled_operations() {
    let prohibited = [
        RandomOperation::Gate,
        RandomOperation::ExtCarryOut,
        RandomOperation::ExtPrioEncode,
        RandomOperation::ExtClz,
        RandomOperation::ExtNormalizeLeft,
        RandomOperation::ExtMaskLow,
        RandomOperation::ExtNaryAdd,
        RandomOperation::AfterAll,
        RandomOperation::Cover,
        RandomOperation::Assert,
        RandomOperation::Trace,
    ];
    let options = RandomFnOptions {
        max_params: 4,
        max_nodes: 48,
        allow_gate: false,
        allow_extension_ops: false,
        allow_events: false,
        enabled_operations: OperationSet::new(
            [RandomOperation::Literal].into_iter().chain(prohibited),
        ),
        ..RandomFnOptions::default()
    };
    let mut entropy = RngEntropy::new(Pcg64Mcg::new(0xb311_e144));
    for _ in 0..1_000 {
        let generated =
            generate_fn(&mut entropy, &options, StopPolicy::ExactBodyNodes(40)).unwrap();
        validate_generated(&generated.function);
        for operation in prohibited {
            assert!(
                !generated
                    .stats
                    .emitted_operations
                    .contains_key(operation.name())
            );
        }
        assert!(function_types(&generated.function).all(|ty| ty != &Type::Token));
    }
}

#[test]
fn probabilistic_event_generation_emits_tokens_effects_and_valid_xls_ir() {
    let requested = [
        RandomOperation::AfterAll,
        RandomOperation::Cover,
        RandomOperation::Assert,
        RandomOperation::Trace,
    ];
    let options = RandomFnOptions {
        max_params: 5,
        max_nodes: 56,
        max_bit_width: 16,
        allow_arrays: false,
        allow_tuples: true,
        allow_events: true,
        enabled_operations: OperationSet::new(
            [RandomOperation::Literal].into_iter().chain(requested),
        ),
        ..RandomFnOptions::default()
    };
    let mut entropy = RngEntropy::new(Pcg64Mcg::new(0xa55e_12c7));
    let mut emitted = HashSet::new();
    let mut live = HashSet::new();
    let mut saw_token_param = false;
    let mut saw_token_literal = false;
    let mut saw_trace_operand_counts = BTreeSet::new();
    let mut saw_trace_format_specifiers = BTreeSet::new();
    let mut saw_trace_escaped_open_brace = false;
    let trace_format_specifiers = [
        "{}", "{:u}", "{:d}", "{:x}", "{:0x}", "{:#x}", "{:b}", "{:0b}", "{:#b}",
    ];

    for sample in 0..750 {
        let generated =
            generate_fn(&mut entropy, &options, StopPolicy::ExactBodyNodes(48)).unwrap();
        validate_generated(&generated.function);
        emitted.extend(generated.stats.emitted_operations.keys().cloned());
        live.extend(generated.stats.live_operations.keys().cloned());
        saw_token_param |= generated
            .function
            .params
            .iter()
            .any(|param| param.ty == Type::Token);
        for node in &generated.function.nodes {
            match &node.payload {
                NodePayload::Literal(_) if node.ty == Type::Token => saw_token_literal = true,
                NodePayload::Trace {
                    format, operands, ..
                } => {
                    assert!(format.starts_with("random_trace"));
                    assert!(
                        operands
                            .iter()
                            .all(|operand| generated.function.get_node(*operand).ty != Type::Token)
                    );
                    saw_trace_operand_counts.insert(operands.len());
                    saw_trace_escaped_open_brace |= format.contains("{{");
                    for specifier in trace_format_specifiers {
                        if format.contains(specifier) {
                            saw_trace_format_specifiers.insert(specifier);
                        }
                    }
                }
                _ => {}
            }
        }
        for ty in function_types(&generated.function) {
            assert_type_obeys_options(ty, &options);
        }
        let ir_text = generated
            .into_top_package(format!("eventful_random_parse_{sample}"))
            .to_string();
        xlsynth::IrPackage::parse_ir(&ir_text, None)
            .unwrap_or_else(|error| panic!("libxls rejected generated PIR:\n{ir_text}\n{error}"));
    }

    for operation in requested {
        assert!(
            emitted.contains(operation.name()),
            "never emitted {}",
            operation.name()
        );
        assert!(
            live.contains(operation.name()),
            "never emitted live {}",
            operation.name()
        );
    }
    assert!(saw_token_param);
    assert!(saw_token_literal);
    assert_eq!(saw_trace_operand_counts, BTreeSet::from([0, 1, 2, 3]));
    assert_eq!(
        saw_trace_format_specifiers,
        BTreeSet::from(trace_format_specifiers)
    );
    assert!(saw_trace_escaped_open_brace);
}

#[test]
fn probabilistic_cover_requires_tuple_support_even_with_events_enabled() {
    let options = RandomFnOptions {
        max_params: 4,
        max_nodes: 40,
        max_bit_width: 8,
        allow_arrays: false,
        allow_tuples: false,
        allow_events: true,
        enabled_operations: OperationSet::new([RandomOperation::Literal, RandomOperation::Cover]),
        ..RandomFnOptions::default()
    };
    let mut entropy = RngEntropy::new(Pcg64Mcg::new(0xc0ee_71e5));

    for _ in 0..1_000 {
        let generated =
            generate_fn(&mut entropy, &options, StopPolicy::ExactBodyNodes(32)).unwrap();
        validate_generated(&generated.function);
        assert!(!generated.stats.emitted_operations.contains_key("cover"));
        for ty in function_types(&generated.function) {
            assert_type_obeys_options(ty, &options);
        }
    }
}
