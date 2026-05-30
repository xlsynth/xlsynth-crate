// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeSet, HashSet};

use rand::RngCore;
use rand_pcg::Pcg64Mcg;
use xlsynth_pir::ir::{
    Binop, FileTable, MemberType, NaryOp, NodePayload, Package, PackageMember, Type, Unop,
};
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn_in_package};
use xlsynth_pir::ir_random::{
    DepletableBytes, FunctionSignature, GenerationError, OperationSet, RandomFnOptions,
    RandomOperation, RngEntropy, StopPolicy, generate_arguments, generate_fn,
    generate_fn_with_signature, generate_same_signature_pair, generate_value,
};
use xlsynth_pir::ir_verify::verify_package;

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
    let value = generate_value(&mut entropy, &Type::Bits(0));
    assert_eq!(value.to_string(), "bits[0]:0");
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
    let mut saw_zero_params = false;
    let mut saw_max_params = false;
    for _ in 0..2_000 {
        let generated =
            generate_fn(&mut entropy, &options, StopPolicy::ExactBodyNodes(32)).unwrap();
        validate_generated(&generated.function);
        operations.extend(generated.stats.emitted_operations.keys().cloned());
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
        let args = generate_arguments(&mut value_entropy, function);
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
                    let expected_format = match operands.len() {
                        0 => "random_trace".to_string(),
                        count => format!("random_trace={}", vec!["{}"; count].join(",")),
                    };
                    assert_eq!(format, &expected_format);
                    assert!(
                        operands
                            .iter()
                            .all(|operand| generated.function.get_node(*operand).ty != Type::Token)
                    );
                    saw_trace_operand_counts.insert(operands.len());
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
