// SPDX-License-Identifier: Apache-2.0

#![no_main]

//! Differentially checks sequential block IR against Yosys-mapped gv-eval.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use libfuzzer_sys::fuzz_target;
use rand::rngs::StdRng;
use rand::SeedableRng;
use xlsynth::{IrBits, IrValue};
use xlsynth_g8r::aig::SequentialGateFn;
use xlsynth_g8r::aig_serdes::emit_netlist::{
    NetlistPortStyle, emit_netlist_with_version_and_port_style,
};
use xlsynth_g8r::aig_sim::sequential::{self, SequentialState};
use xlsynth_g8r::block2sequential::block_package_to_sequential_gate_fn;
use xlsynth_g8r::gatify::ir2gate::GatifyOptions;
use xlsynth_g8r::liberty::parser::{
    LibertyPayloadOptions, parse_liberty_files_with_payload_options,
};
use xlsynth_g8r::liberty_model::Library;
use xlsynth_g8r::netlist::gv_eval::{
    GvEvalOptions, load_labeled_sequential_netlist_aig_with_liberty,
};
use xlsynth_g8r::netlist::yosys::YosysEnvironment;
use xlsynth_g8r::verilog_version::VerilogVersion;
use xlsynth_pir::ir::{BlockMetadata, Fn, NodePayload, NodeRef, Type};
use xlsynth_pir::ir_eval::{self, FnEvalResult};
use xlsynth_pir::ir_random::{
    DepletableBytes, OperationSet, RandomBlockOptions, RandomBlockResetTiming, RandomFnOptions,
    RandomOperation, StopPolicy, generate_block_package,
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

struct ExternalYosysContext {
    liberty: Library,
    yosys: YosysEnvironment,
}

static EXTERNAL_YOSYS_CONTEXT: OnceLock<Result<ExternalYosysContext, String>> = OnceLock::new();
static SKIP_REPORTED: OnceLock<()> = OnceLock::new();

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
        min_registers: 1,
        max_registers: 4,
        require_reset_on_all_registers: true,
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

fn external_yosys_context() -> Option<&'static ExternalYosysContext> {
    match EXTERNAL_YOSYS_CONTEXT.get_or_init(build_external_yosys_context) {
        Ok(context) => Some(context),
        Err(error) => {
            if SKIP_REPORTED.set(()).is_ok() {
                eprintln!("skipping external Yosys/Liberty fuzz target: {error}");
            }
            None
        }
    }
}

fn build_external_yosys_context() -> Result<ExternalYosysContext, String> {
    let yosys = YosysEnvironment::from_env()?;
    let liberty = parse_liberty_files_with_payload_options(
        yosys.liberty_files().paths(),
        LibertyPayloadOptions {
            include_timing: false,
            include_power: false,
        },
    )
    .map_err(|e| format!("parse Liberty inputs: {e}"))?;
    Ok(ExternalYosysContext { liberty, yosys })
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
        // Evaluate outputless samples too; the property still checks that the
        // PIR, RTL, Yosys, netlist loader, and sequential evaluator do not
        // reject or crash on the sample.
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
        .map(|register| generate_uniform_value_with_rng(rng, &register.ty))
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
                let asserted = cycle == 0;
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

fn external_input_shapes(design: &SequentialGateFn) -> BTreeMap<String, usize> {
    design
        .inputs
        .iter()
        .map(|input_id| {
            let input = &design.transition.inputs[input_id.index()];
            (input.name.clone(), input.get_bit_count())
        })
        .collect()
}

fn external_output_shapes(design: &SequentialGateFn) -> BTreeMap<String, usize> {
    design
        .outputs
        .iter()
        .map(|output_id| {
            let output = &design.transition.outputs[output_id.index()];
            (output.name.clone(), output.get_bit_count())
        })
        .collect()
}

fn remap_inputs_for_design(
    design: &SequentialGateFn,
    block: &Fn,
    input_bits: &[IrBits],
) -> Vec<IrBits> {
    let by_name = block
        .params
        .iter()
        .zip(input_bits)
        .map(|(param, bits)| (param.name.as_str(), bits))
        .collect::<BTreeMap<_, _>>();
    design
        .inputs
        .iter()
        .map(|input_id| {
            let name = design.transition.inputs[input_id.index()].name.as_str();
            (**by_name
                .get(name)
                .unwrap_or_else(|| panic!("mapped gv-eval input '{name}' is not a block input")))
            .clone()
        })
        .collect()
}

fn remap_outputs_for_design(
    design: &SequentialGateFn,
    metadata: &BlockMetadata,
    output_bits: &[IrBits],
) -> Vec<IrBits> {
    let by_name = metadata
        .output_names
        .iter()
        .zip(output_bits)
        .map(|(name, bits)| (name.as_str(), bits))
        .collect::<BTreeMap<_, _>>();
    design
        .outputs
        .iter()
        .map(|output_id| {
            let name = design.transition.outputs[output_id.index()].name.as_str();
            (**by_name
                .get(name)
                .unwrap_or_else(|| panic!("mapped gv-eval output '{name}' is not a block output")))
            .clone()
        })
        .collect()
}

fuzz_target!(|data: &[u8]| {
    // Missing Yosys or Liberty files are infrastructure conditions, not
    // properties of an individual fuzz sample.
    let Some(context) = external_yosys_context() else {
        return;
    };

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
    assert!(
        metadata
            .reset
            .as_ref()
            .is_some_and(|reset| !reset.asynchronous),
        "sequential gv-eval generation requires a synchronous reset:\n{block_ir}"
    );
    assert!(
        metadata
            .registers
            .iter()
            .all(|register| register.reset_value.is_some()),
        "sequential gv-eval generation requires every register reset value:\n{block_ir}"
    );

    let source_design = block_package_to_sequential_gate_fn(
        &generated.package,
        GatifyOptions::all_opts_disabled(),
    )
    .unwrap_or_else(|error| panic!("random block G8R lowering failed:\n{block_ir}\n{error}"));
    let rtl = emit_netlist_with_version_and_port_style(
        &source_design,
        VerilogVersion::Verilog,
        NetlistPortStyle::PackedBits,
    )
    .unwrap_or_else(|error| panic!("random block RTL emission failed:\n{block_ir}\n{error}"));
    let mapped_gv = context
        .yosys
        .synthesize_sequential_verilog_to_gv(&rtl, &source_design.name)
        .unwrap_or_else(|error| {
            panic!("random block Yosys mapping failed:\nIR:\n{block_ir}\nRTL:\n{rtl}\n{error}")
        });
    let netlist_dir = tempfile::tempdir().expect("create temporary mapped-netlist directory");
    let netlist_path = netlist_dir.path().join("mapped.gv");
    std::fs::write(&netlist_path, &mapped_gv).expect("write temporary mapped netlist");
    let mapped_model = load_labeled_sequential_netlist_aig_with_liberty(
        &netlist_path,
        &context.liberty,
        &GvEvalOptions {
            module_name: Some(source_design.name.clone()),
            clock_port_name: source_design
                .clock
                .as_ref()
                .map(|clock| clock.name.clone()),
        },
    )
    .unwrap_or_else(|error| {
        panic!(
            "random block sequential gv-eval loading failed:\nIR:\n{block_ir}\nRTL:\n{rtl}\nGV:\n{mapped_gv}\n{error}"
        )
    });
    let mapped_design = &mapped_model.sequential_gate_fn;

    let source_input_shapes = func
        .params
        .iter()
        .map(|param| (param.name.clone(), param.ty.bit_count()))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        external_input_shapes(mapped_design),
        source_input_shapes,
        "Yosys changed sequential block input shape:\nIR:\n{block_ir}\nGV:\n{mapped_gv}"
    );
    let output_types = block_output_types(func, metadata);
    let source_output_shapes = metadata
        .output_names
        .iter()
        .zip(&output_types)
        .map(|(name, ty)| (name.clone(), ty.bit_count()))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        external_output_shapes(mapped_design),
        source_output_shapes,
        "Yosys changed sequential block output shape:\nIR:\n{block_ir}\nGV:\n{mapped_gv}"
    );

    let mut seed = [0_u8; 32];
    seed.copy_from_slice(blake3::hash(block_ir.as_bytes()).as_bytes());
    let mut rng = StdRng::from_seed(seed);
    let mut block_state = generate_initial_state(metadata, &mut rng);
    let mut mapped_inputs = Vec::with_capacity(CYCLE_COUNT);
    let mut expected_outputs = Vec::with_capacity(CYCLE_COUNT);
    for cycle in 0..CYCLE_COUNT {
        let inputs = generate_cycle_inputs(func, metadata, &mut rng, cycle);
        let input_bits = inputs
            .iter()
            .zip(&func.params)
            .map(|(value, param)| flatten_value(value, &param.ty))
            .collect::<Vec<_>>();
        mapped_inputs.push(remap_inputs_for_design(mapped_design, func, &input_bits));

        let (outputs, next_state) =
            evaluate_block_cycle(func, metadata, &inputs, &block_state, &block_ir);
        let output_bits = outputs
            .iter()
            .zip(&output_types)
            .map(|(value, ty)| flatten_value(value, ty))
            .collect::<Vec<_>>();
        expected_outputs.push(remap_outputs_for_design(
            mapped_design,
            metadata,
            &output_bits,
        ));
        block_state = next_state;
    }

    let trace = sequential::simulate(
        mapped_design,
        &mapped_inputs,
        SequentialState::all_zeros(mapped_design),
    )
    .unwrap_or_else(|error| {
        panic!(
            "random block sequential gv-eval simulation failed:\nIR:\n{block_ir}\nGV:\n{mapped_gv}\n{error}"
        )
    });
    for cycle in 1..CYCLE_COUNT {
        assert_eq!(
            trace.external_outputs()[cycle],
            expected_outputs[cycle],
            "random block sequential gv-eval mismatch at cycle {cycle}:\nIR:\n{block_ir}\nRTL:\n{rtl}\nGV:\n{mapped_gv}"
        );
    }
});
