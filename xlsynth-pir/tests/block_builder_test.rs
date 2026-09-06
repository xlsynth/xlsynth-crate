// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;

use xlsynth_pir::block_inline::inline_all_blocks_in_package;
use xlsynth_pir::block2fn::combinational_block_to_fn;
use xlsynth_pir::desugar_extensions::desugar_extensions_in_package;
use xlsynth_pir::ir::{Block, MemberType, NodePayload, Package, PackageMember, Type};
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn_in_package};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_rebase_ids::package_max_emitted_node_id;
use xlsynth_pir::ir_verify::verify_package;
use xlsynth_pir::{
    BValue, BlockBuilder, BuilderError, FnBuilder, IrValue, NaryAddOptions, NaryAddTerm,
    NormalizeLeftOptions, RegisterWriteOptions, ResetBehavior,
};

fn bits(width: usize, value: u64) -> IrValue {
    IrValue::make_ubits(width, value).unwrap()
}

/// Checks that a builder-produced package verifies and roundtrips through PIR.
fn assert_pir_package(package: &Package) {
    verify_package(package).unwrap();
    let text = package.to_string();
    let parsed = Parser::new(&text).parse_and_validate_package().unwrap();
    assert_eq!(parsed.to_string(), text);
}

/// Checks both PIR roundtripping and compatibility with the pinned XLS parser.
fn assert_standard_package(package: &Package) {
    assert_pir_package(package);
    let text = package.to_string();
    xlsynth::IrPackage::parse_ir(&text, None)
        .unwrap_or_else(|error| panic!("XLS rejected builder output: {error}\n{text}"));
}

/// Evaluates an explicit combinational projection without changing its package.
fn evaluate_block(package: &Package, name: &str, args: &[IrValue]) -> FnEvalResult {
    let function = combinational_block_to_fn(package.get_block(name).unwrap()).unwrap();
    eval_fn_in_package(package, &function, args)
}

fn successful_value(result: FnEvalResult) -> IrValue {
    match result {
        FnEvalResult::Success(success) => success.value,
        FnEvalResult::Failure(failure) => panic!("unexpected evaluation failure: {failure:?}"),
    }
}

#[test]
fn empty_block_has_no_synthetic_return_or_data_ports() {
    let package = BlockBuilder::new("empty").build_package("empty").unwrap();
    assert_eq!(
        package.to_string(),
        include_str!("goldens/block_builder/empty.ir")
    );
    let block = package.get_top_block().unwrap();
    assert_eq!(block.nodes.len(), 1);
    assert!(block.ports.is_empty());
    assert_standard_package(&package);
}

#[test]
fn aggregate_ports_late_inputs_and_output_sink_users_roundtrip() {
    let mut builder = BlockBuilder::new("aggregate");
    let x = builder.input_port("x", Type::Bits(129)).unwrap();
    let byte = builder.bit_slice(x, 64, 8).unwrap();
    let elements = builder
        .input_port("elements", Type::new_array(Type::Bits(3), 2))
        .unwrap();
    let zero = builder.literal(bits(1, 0)).unwrap();
    let element = builder.array_index(elements, zero).unwrap();
    let pair = builder.tuple(&[byte, element]).unwrap();
    let sink = builder.output_port("pair", pair).unwrap();
    assert_eq!(builder.get_type(sink).unwrap(), &Type::nil());
    builder.set_name(sink, "pair_sink").unwrap();
    let observed = builder.tuple(&[sink]).unwrap();
    builder.output_port("unit", observed).unwrap();
    let package = builder.build_package("aggregate_ports").unwrap();
    assert_eq!(
        package.to_string(),
        include_str!("goldens/block_builder/aggregate.ir")
    );
    assert_standard_package(&package);
    let input = IrValue::parse_typed("bits[129]:0xab0000000000000000").unwrap();
    let array = IrValue::parse_typed("[bits[3]:5, bits[3]:6]").unwrap();
    assert_eq!(
        successful_value(evaluate_block(&package, "aggregate", &[input, array])),
        IrValue::make_tuple(&[
            IrValue::make_tuple(&[bits(8, 0xab), bits(3, 5)]),
            IrValue::make_tuple(&[IrValue::make_tuple(&[])]),
        ])
    );
}

#[test]
fn register_feedback_caches_reads_and_supports_reset_and_enabled_multiwrites() {
    let mut builder = BlockBuilder::new("feedback");
    builder.clock_port("clk").unwrap();
    let reset = builder.input_port("rst", Type::Bits(1)).unwrap();
    let enable = builder.input_port("enable", Type::Bits(1)).unwrap();
    let data = builder.input_port("data", Type::Bits(8)).unwrap();
    builder
        .set_reset(
            reset,
            ResetBehavior {
                asynchronous: false,
                active_low: false,
            },
        )
        .unwrap();
    let state = builder
        .register("state", Type::Bits(8), Some(bits(8, 7)))
        .unwrap();
    let current = builder.register_read(state).unwrap();
    assert_eq!(builder.register_read(state).unwrap(), current);
    let next = builder.add(current, data).unwrap();
    let first = builder
        .register_write(
            state,
            next,
            RegisterWriteOptions {
                load_enable: Some(enable),
                reset: Some(reset),
            },
        )
        .unwrap();
    assert_eq!(builder.get_type(first).unwrap(), &Type::nil());
    let disabled = builder.not(enable).unwrap();
    builder
        .register_write(
            state,
            data,
            RegisterWriteOptions {
                load_enable: Some(disabled),
                reset: Some(reset),
            },
        )
        .unwrap();
    builder.output_port("value", current).unwrap();
    let package = builder.build_package("register_feedback").unwrap();
    assert_eq!(
        package.to_string(),
        include_str!("goldens/block_builder/feedback.ir")
    );
    assert_standard_package(&package);
    let block = package.get_top_block().unwrap();
    assert_eq!(
        block
            .nodes
            .iter()
            .filter(|node| matches!(node.payload, NodePayload::RegisterRead { .. }))
            .count(),
        1
    );
    assert_eq!(
        block
            .nodes
            .iter()
            .filter(|node| matches!(node.payload, NodePayload::RegisterWrite { .. }))
            .count(),
        2
    );
}

/// Records the common operation state around a rejected resource request.
fn assert_rejected<T>(
    builder: &mut BlockBuilder,
    request: impl FnOnce(&mut BlockBuilder) -> Result<T, BuilderError>,
) {
    let last = builder.last_value();
    let ty = last.map(|value| builder.get_type(value).unwrap().clone());
    assert!(request(builder).is_err());
    assert_eq!(builder.last_value(), last);
    assert_eq!(
        builder
            .last_value()
            .map(|value| builder.get_type(value).unwrap().clone()),
        ty
    );
}

/// Interleaves failed requests without changing the successfully built graph.
fn build_recoverable(inject_failures: bool) -> Block {
    let mut builder = BlockBuilder::new("recoverable");
    builder.clock_port("clk").unwrap();
    let reset = builder.input_port("rst", Type::Bits(1)).unwrap();
    let data = builder.input_port("data", Type::Bits(8)).unwrap();
    builder
        .set_reset(
            reset,
            ResetBehavior {
                asynchronous: false,
                active_low: false,
            },
        )
        .unwrap();
    if inject_failures {
        assert_rejected(&mut builder, |builder| builder.clock_port("second_clock"));
        assert_rejected(&mut builder, |builder| {
            builder.input_port("data", Type::Bits(8))
        });
        assert_rejected(&mut builder, |builder| {
            builder.input_port("invalid name", Type::Bits(8))
        });
        assert_rejected(&mut builder, |builder| {
            builder.input_port("overflow", Type::new_array(Type::Bits(usize::MAX), 2))
        });
        assert_rejected(&mut builder, |builder| {
            builder.set_reset(
                data,
                ResetBehavior {
                    asynchronous: false,
                    active_low: false,
                },
            )
        });
        assert_rejected(&mut builder, |builder| {
            builder.register("state", Type::Bits(8), Some(bits(7, 0)))
        });
    }
    let state = builder
        .register("state", Type::Bits(8), Some(bits(8, 0)))
        .unwrap();
    let current = builder.register_read(state).unwrap();
    builder.set_name(current, "second_clock").unwrap();
    if inject_failures {
        assert_rejected(&mut builder, |builder| {
            builder.register("state", Type::Bits(8), None)
        });
        assert_rejected(&mut builder, |builder| {
            builder.register_write(
                state,
                reset,
                RegisterWriteOptions {
                    load_enable: None,
                    reset: Some(reset),
                },
            )
        });
        assert_rejected(&mut builder, |builder| {
            builder.register_write(
                state,
                data,
                RegisterWriteOptions {
                    load_enable: Some(data),
                    reset: Some(reset),
                },
            )
        });
        assert_rejected(&mut builder, |builder| builder.bit_slice(data, 7, 2));
        assert_rejected(&mut builder, |builder| {
            builder.output_port("result", foreign_value())
        });
    }
    builder
        .register_write(
            state,
            data,
            RegisterWriteOptions {
                load_enable: None,
                reset: Some(reset),
            },
        )
        .unwrap();
    builder.set_name(data, "data_node").unwrap();
    if inject_failures {
        // Renaming an alias never releases the semantic input port name.
        assert_rejected(&mut builder, |builder| builder.set_name(current, "data"));
    }
    builder.output_port("result", current).unwrap();
    builder.build().unwrap()
}

fn foreign_value() -> BValue {
    FnBuilder::new("foreign").param("x", Type::Bits(8)).unwrap()
}

#[test]
fn rejected_requests_do_not_allocate_ids_reserve_names_or_poison_resources() {
    let baseline = build_recoverable(false);
    let recovered = build_recoverable(true);
    assert_eq!(recovered.to_string(), baseline.to_string());
    assert_eq!(recovered.ports, baseline.ports);
    assert_eq!(recovered.reset, baseline.reset);
    assert_eq!(recovered.nodes.len(), baseline.nodes.len());
}

fn leaf_block() -> Block {
    let mut builder = BlockBuilder::new("leaf");
    let input = builder.input_port("x", Type::Bits(8)).unwrap();
    let result = builder.not(input).unwrap();
    builder.output_port("y", result).unwrap();
    builder.build().unwrap()
}

fn parent_builder(name: &str, leaf: &Block) -> BlockBuilder {
    let mut builder = BlockBuilder::new(name);
    let input = builder.input_port("x", Type::Bits(8)).unwrap();
    let instance = builder.instantiate("child", leaf).unwrap();
    builder.instantiation_input(instance, "x", input).unwrap();
    let output = builder.instantiation_output(instance, "y").unwrap();
    assert_eq!(builder.instantiation_output(instance, "y").unwrap(), output);
    builder.output_port("y", output).unwrap();
    builder
}

#[test]
fn hierarchy_finalization_rebases_ids_and_preserves_package_top() {
    let leaf = leaf_block();
    let mut package = Package {
        name: "hierarchy".to_string(),
        file_table: xlsynth_pir::ir::FileTable::new(),
        members: vec![PackageMember::Block(leaf.clone())],
        top: Some(("leaf".to_string(), MemberType::Block)),
    };
    let maximum = package_max_emitted_node_id(&package);
    let original_top = package.top.clone();
    parent_builder("parent", &leaf)
        .build_into_package(&mut package)
        .unwrap();
    assert_eq!(package.top, original_top);
    assert!(
        package
            .get_block("parent")
            .unwrap()
            .nodes
            .iter()
            .filter(|node| !matches!(node.payload, NodePayload::Nil))
            .all(|node| node.text_id > maximum)
    );
    let before = package.to_string();
    let detached = parent_builder("detached", &leaf)
        .build_in_package(&package)
        .unwrap();
    assert_eq!(package.to_string(), before);
    package.members.push(PackageMember::Block(detached));
    assert_standard_package(&package);
    package.set_top_block("parent").unwrap();
    inline_all_blocks_in_package(&mut package).unwrap();
    assert_eq!(
        successful_value(evaluate_block(&package, "parent", &[bits(8, 0xa5)])),
        bits(8, 0x5a)
    );
}

#[test]
fn foreign_handles_and_invalid_instantiation_connections_are_transactional() {
    let leaf = leaf_block();
    let mut foreign = BlockBuilder::new("foreign");
    let foreign_register = foreign.register("r", Type::Bits(8), None).unwrap();
    let foreign_instance = foreign.instantiate("child", &leaf).unwrap();
    let foreign_input = foreign.input_port("x", Type::Bits(8)).unwrap();
    let mut builder = BlockBuilder::new("parent");
    let input = builder.input_port("x", Type::Bits(8)).unwrap();
    let narrow = builder.input_port("flag", Type::Bits(1)).unwrap();
    let instance = builder.instantiate("child", &leaf).unwrap();
    assert_rejected(&mut builder, |builder| {
        builder.register_read(foreign_register)
    });
    assert_rejected(&mut builder, |builder| {
        builder.register_write(foreign_register, input, RegisterWriteOptions::default())
    });
    assert_rejected(&mut builder, |builder| {
        builder.instantiation_output(foreign_instance, "y")
    });
    assert_rejected(&mut builder, |builder| {
        builder.instantiation_input(foreign_instance, "x", input)
    });
    assert_rejected(&mut builder, |builder| {
        builder.instantiation_input(instance, "x", foreign_input)
    });
    assert_rejected(&mut builder, |builder| {
        builder.instantiation_input(instance, "x", narrow)
    });
    assert_rejected(&mut builder, |builder| {
        builder.instantiation_input(instance, "missing", input)
    });
    assert_rejected(&mut builder, |builder| {
        builder.instantiation_output(instance, "missing")
    });
    assert_rejected(&mut builder, |builder| builder.instantiate("child", &leaf));
    builder.instantiation_input(instance, "x", input).unwrap();
    assert_rejected(&mut builder, |builder| {
        builder.instantiation_input(instance, "x", input)
    });
    let output = builder.instantiation_output(instance, "y").unwrap();
    builder.output_port("y", output).unwrap();
    let mut package = Package {
        name: "connections".to_string(),
        file_table: xlsynth_pir::ir::FileTable::new(),
        members: vec![PackageMember::Block(leaf)],
        top: None,
    };
    builder.build_into_package(&mut package).unwrap();
    assert_standard_package(&package);
    let mut function = FnBuilder::new("function");
    assert_eq!(function.not(input), Err(BuilderError::ForeignValue));
}

#[test]
fn package_finalization_rejects_missing_or_mismatched_declarations_atomically() {
    let leaf = leaf_block();
    assert!(parent_builder("standalone", &leaf).build().is_err());
    let mut package = Package {
        name: "checks".to_string(),
        file_table: xlsynth_pir::ir::FileTable::new(),
        members: Vec::new(),
        top: None,
    };
    let before = package.to_string();
    assert!(
        parent_builder("missing", &leaf)
            .build_into_package(&mut package)
            .is_err()
    );
    assert_eq!(package.to_string(), before);
    package.members.push(PackageMember::Block(leaf.clone()));
    let before = package.to_string();
    let mut duplicate = BlockBuilder::new("leaf");
    duplicate.input_port("x", Type::Bits(8)).unwrap();
    assert!(duplicate.build_into_package(&mut package).is_err());
    assert_eq!(package.to_string(), before);
    let mut changed = BlockBuilder::new("leaf");
    let input = changed.input_port("renamed", Type::Bits(8)).unwrap();
    changed.output_port("y", input).unwrap();
    let changed = changed.build().unwrap();
    package.members[0] = PackageMember::Block(changed);
    let before = package.to_string();
    assert!(
        parent_builder("mismatched", &leaf)
            .build_into_package(&mut package)
            .is_err()
    );
    assert_eq!(package.to_string(), before);
}

#[test]
fn package_id_overflow_leaves_the_destination_unchanged() {
    let mut existing = BlockBuilder::new("existing");
    existing.input_port("x", Type::Bits(1)).unwrap();
    let mut package = existing.build_package("ids").unwrap();
    let block = package.get_block_mut("existing").unwrap();
    let input = block.get_input_port("x").unwrap();
    block.get_node_mut(input).text_id = usize::MAX;
    verify_package(&package).unwrap();
    let before = package.to_string();
    let mut addition = BlockBuilder::new("addition");
    addition.input_port("x", Type::Bits(1)).unwrap();
    assert!(addition.build_into_package(&mut package).is_err());
    assert_eq!(package.to_string(), before);
}

#[test]
fn shared_calls_and_loops_use_package_signatures_and_wide_values() {
    let mut step = FnBuilder::new("step");
    step.param("i", Type::Bits(3)).unwrap();
    let carry = step.param("carry", Type::Bits(129)).unwrap();
    let delta = step.param("delta", Type::Bits(129)).unwrap();
    let result = step.add(carry, delta).unwrap();
    let mut package = step.build_package(result, "calls").unwrap();
    let mut negate = FnBuilder::new("negate");
    let input = negate.param("x", Type::Bits(129)).unwrap();
    let result = negate.neg(input).unwrap();
    negate.build_into_package(result, &mut package).unwrap();
    let mut builder = BlockBuilder::new("wrapper");
    let input = builder.input_port("x", Type::Bits(129)).unwrap();
    let delta = builder.input_port("delta", Type::Bits(129)).unwrap();
    let initial = builder
        .invoke(package.get_fn("negate").unwrap(), &[input])
        .unwrap();
    let result = builder
        .counted_for(initial, 3, 2, package.get_fn("step").unwrap(), &[delta])
        .unwrap();
    builder.output_port("y", result).unwrap();
    builder.build_into_package(&mut package).unwrap();
    assert_standard_package(&package);
    assert_eq!(
        successful_value(evaluate_block(
            &package,
            "wrapper",
            &[bits(129, 5), bits(129, 7)]
        )),
        bits(129, 16)
    );
}

#[test]
fn shared_effects_and_all_extensions_survive_block_desugaring() {
    let mut builder = BlockBuilder::new("extensions");
    let x = builder.input_port("x", Type::Bits(8)).unwrap();
    let y = builder.input_port("y", Type::Bits(8)).unwrap();
    let condition = builder.input_port("condition", Type::Bits(1)).unwrap();
    let token = builder.after_all(&[]).unwrap();
    let token = builder
        .assert(token, condition, "condition must hold", "condition_check")
        .unwrap();
    builder.trace(token, condition, "x={:x}", &[x], 0).unwrap();
    builder.cover(condition, "active").unwrap();
    let gate = builder.gate(condition, x).unwrap();
    let partial = builder.umulp(x, y, 16).unwrap();
    let bitwise = builder.and_all(&[x, y, x]).unwrap();
    let carry = builder.ext_carry_out(x, y, condition).unwrap();
    let encoded = builder.ext_prio_encode(x, true).unwrap();
    let clz = builder.ext_clz(x, 2, 5).unwrap();
    let normalized = builder
        .ext_normalize_left(
            x,
            NormalizeLeftOptions {
                normalized_bit_count: 12,
                shift_offset: 0,
                clz_bit_count: Some(4),
            },
        )
        .unwrap();
    let mask = builder.ext_mask_low(y, 8).unwrap();
    let sum = builder
        .ext_nary_add(
            &[NaryAddTerm::unsigned(x), NaryAddTerm::signed(y).negate()],
            NaryAddOptions::new(9),
        )
        .unwrap();
    let result = builder
        .tuple(&[
            gate, partial, bitwise, carry, encoded, clz, normalized, mask, sum,
        ])
        .unwrap();
    builder.output_port("result", result).unwrap();
    let package = builder.build_package("extensions").unwrap();
    verify_package(&package).unwrap();
    let operators = package
        .get_top_block()
        .unwrap()
        .nodes
        .iter()
        .map(|node| node.payload.get_operator())
        .collect::<BTreeSet<_>>();
    for operation in [
        "ext_carry_out",
        "ext_prio_encode",
        "ext_clz",
        "ext_normalize_left",
        "ext_mask_low",
        "ext_nary_add",
    ] {
        assert!(operators.contains(operation), "missing {operation}");
    }
    let mut desugared = package.clone();
    desugar_extensions_in_package(&mut desugared).unwrap();
    assert_standard_package(&desugared);
    for (x, y, active) in [(0, 0, 1), (0xff, 0x80, 1), (7, 9, 0)] {
        let args = [bits(8, x), bits(8, y), bits(1, active)];
        assert_eq!(
            evaluate_block(&package, "extensions", &args),
            evaluate_block(&desugared, "extensions", &args)
        );
    }
}

#[test]
fn extern_instantiation_uses_typed_signature_components() {
    let mut foreign = FnBuilder::new("foreign");
    let input = foreign
        .param(
            "arg",
            Type::Tuple(vec![Box::new(Type::Bits(8)), Box::new(Type::Bits(1))]),
        )
        .unwrap();
    let mut package = foreign.build_package(input, "externs").unwrap();
    package.get_fn_mut("foreign").unwrap().outer_attrs.push(r#"#[ffi_proto("""code_template: "external_cell {fn} (.value({arg.0}), .result({return.0}));"
""")]"#.to_string());
    let mut builder = BlockBuilder::new("wrapper");
    let input = builder.input_port("data", Type::Bits(8)).unwrap();
    let instance = builder
        .instantiate_extern("external", package.get_fn("foreign").unwrap())
        .unwrap();
    builder
        .instantiation_input(instance, "arg.0", input)
        .unwrap();
    let output = builder.instantiation_output(instance, "return.0").unwrap();
    assert_eq!(builder.get_type(output).unwrap(), &Type::Bits(8));
    builder.output_port("result", output).unwrap();
    builder.build_into_package(&mut package).unwrap();
    // The pinned libxls parser does not accept the existing PIR extern
    // `foreign_function=` declaration syntax; native verification and
    // roundtripping still exercise the builder's typed external interface.
    assert_pir_package(&package);
}

#[test]
fn extern_finalization_rejects_renamed_unconnected_parameters_with_identical_types() {
    let make_function = |parameter_name: &str| {
        let mut builder = FnBuilder::new("external_value");
        builder.param(parameter_name, Type::Bits(8)).unwrap();
        let result = builder.literal(bits(8, 42)).unwrap();
        builder.build(result).unwrap()
    };
    let declared = make_function("before");
    let mut wrapper = BlockBuilder::new("wrapper");
    let instance = wrapper.instantiate_extern("external", &declared).unwrap();
    let output = wrapper.instantiation_output(instance, "return").unwrap();
    wrapper.output_port("result", output).unwrap();
    // Leaving the parameter unconnected ensures the signature snapshot, not
    // ordinary connection validation, catches this named-interface change.
    let mut package = Package {
        name: "changed_interface".to_string(),
        file_table: xlsynth_pir::ir::FileTable::new(),
        members: vec![PackageMember::Function(make_function("after"))],
        top: None,
    };
    verify_package(&package).unwrap();
    let before = package.to_string();
    assert!(wrapper.build_into_package(&mut package).is_err());
    assert_eq!(package.to_string(), before);
}

#[test]
fn instance_feedback_distinguishes_combinational_cycles_from_registered_state() {
    for registered in [false, true] {
        let mut child = BlockBuilder::new("child");
        let input = child.input_port("data", Type::Bits(8)).unwrap();
        let value = if registered {
            child.clock_port("clk").unwrap();
            let state = child.register("state", Type::Bits(8), None).unwrap();
            let value = child.register_read(state).unwrap();
            child
                .register_write(state, input, RegisterWriteOptions::default())
                .unwrap();
            value
        } else {
            child.not(input).unwrap()
        };
        child.output_port("result", value).unwrap();
        let mut package = child.build_package("feedback").unwrap();
        let mut parent = BlockBuilder::new("parent");
        if registered {
            parent.clock_port("clk").unwrap();
        }
        let instance = parent
            .instantiate("child_instance", package.get_block("child").unwrap())
            .unwrap();
        let output = parent.instantiation_output(instance, "result").unwrap();
        parent
            .instantiation_input(instance, "data", output)
            .unwrap();
        parent.output_port("result", output).unwrap();
        let before = package.to_string();
        let built = parent.build_into_package(&mut package);
        if registered {
            built.unwrap();
            package.set_top_block("parent").unwrap();
            assert_standard_package(&package);
        } else {
            assert!(built.is_err());
            assert_eq!(package.to_string(), before);
        }
    }
}

#[test]
fn clocked_instances_require_a_parent_clock_but_not_the_same_clock_name() {
    let mut child = BlockBuilder::new("registered_child");
    child.clock_port("child_clk").unwrap();
    let data = child.input_port("data", Type::Bits(8)).unwrap();
    let state = child.register("state", Type::Bits(8), None).unwrap();
    let current = child.register_read(state).unwrap();
    child
        .register_write(state, data, RegisterWriteOptions::default())
        .unwrap();
    child.output_port("result", current).unwrap();
    let package = child.build_package("clocks").unwrap();
    for clock in [None, Some("parent_clk")] {
        let mut package = package.clone();
        let mut parent = BlockBuilder::new("parent");
        if let Some(clock) = clock {
            parent.clock_port(clock).unwrap();
        }
        let data = parent.input_port("data", Type::Bits(8)).unwrap();
        let instance = parent
            .instantiate("child", package.get_block("registered_child").unwrap())
            .unwrap();
        parent.instantiation_input(instance, "data", data).unwrap();
        let current = parent.instantiation_output(instance, "result").unwrap();
        parent.output_port("result", current).unwrap();
        let before = package.to_string();
        let result = parent.build_into_package(&mut package);
        if clock.is_some() {
            result.unwrap();
            assert_eq!(
                package.get_block("parent").unwrap().clock_port_name(),
                clock
            );
            assert_standard_package(&package);
        } else {
            assert!(result.is_err());
            assert_eq!(package.to_string(), before);
        }
    }
}

#[test]
fn port_annotations_are_checked_without_changing_external_names() {
    let mut builder = BlockBuilder::new("annotations");
    let input = builder.input_port("data", Type::Bits(8)).unwrap();
    let body = builder.not(input).unwrap();
    let output = builder.output_port("result", body).unwrap();
    builder.set_name(input, "input_alias").unwrap();
    builder.set_name(output, "output_alias").unwrap();
    builder.set_port_sv_type(input, Some("input_t")).unwrap();
    builder
        .set_port_sv_type(output, Some("discarded_t"))
        .unwrap();
    builder.set_port_sv_type(output, None).unwrap();
    assert_rejected(&mut builder, |builder| {
        builder.set_port_sv_type(body, Some("invalid_t"))
    });
    assert_rejected(&mut builder, |builder| {
        builder.set_port_sv_type(foreign_value(), None)
    });
    let package = builder.build_package("annotations").unwrap();
    assert_standard_package(&package);
    let block = package.get_top_block().unwrap();
    let input = block.get_input_port("data").unwrap();
    let output = block.get_output_port("result").unwrap();
    assert_eq!(block.get_node(input).name.as_deref(), Some("input_alias"));
    assert_eq!(block.get_node(output).name.as_deref(), Some("output_alias"));
    assert_eq!(
        block.get_node(input).payload,
        NodePayload::InputPort {
            name: "data".to_string(),
            sv_type: Some("input_t".to_string())
        }
    );
    assert!(
        matches!(&block.get_node(output).payload, NodePayload::OutputPort { name, sv_type: None, .. } if name == "result")
    );
}

#[test]
fn stage_keyword_is_rejected_without_poisoning_block_resources() {
    assert!(BlockBuilder::new("stage").build().is_err());
    let leaf = leaf_block();
    let mut builder = BlockBuilder::new("parent");
    let input = builder.input_port("data", Type::Bits(8)).unwrap();
    assert_rejected(&mut builder, |builder| {
        builder.input_port("stage", Type::Bits(8))
    });
    assert_rejected(&mut builder, |builder| builder.output_port("stage", input));
    assert_rejected(&mut builder, |builder| builder.clock_port("stage"));
    assert_rejected(&mut builder, |builder| {
        builder.register("stage", Type::Bits(8), None)
    });
    assert_rejected(&mut builder, |builder| builder.instantiate("stage", &leaf));
    assert_rejected(&mut builder, |builder| builder.set_name(input, "stage"));
    let instance = builder.instantiate("child", &leaf).unwrap();
    builder.instantiation_input(instance, "x", input).unwrap();
    let output = builder.instantiation_output(instance, "y").unwrap();
    builder.output_port("result", output).unwrap();
    let mut package = Package {
        name: "keywords".to_string(),
        file_table: xlsynth_pir::ir::FileTable::new(),
        members: vec![PackageMember::Block(leaf)],
        top: None,
    };
    let maximum = package_max_emitted_node_id(&package);
    builder.build_into_package(&mut package).unwrap();
    assert_standard_package(&package);
    assert_eq!(
        package
            .get_block("parent")
            .unwrap()
            .nodes
            .iter()
            .skip(1)
            .map(|node| node.text_id)
            .collect::<Vec<_>>(),
        (1..=4).map(|id| id + maximum).collect::<Vec<_>>()
    );
}
