// SPDX-License-Identifier: Apache-2.0

//! Reset-reachable equivalence against a complete ABC/native mapped netlist.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Duration;

use rand::{Rng, SeedableRng, rngs::StdRng};
use xlsynth_g8r::aig::{
    ClockPort, GateBuilder, GateBuilderOptions, SequentialGateFn, add_input_registers,
    add_output_registers,
};
use xlsynth_g8r::aig_serdes::g8r::emit_g8r;
use xlsynth_g8r::aig_sim::sequential::{SequentialState, simulate};
use xlsynth_g8r::block2sequential::block_package_to_sequential_gate_fn_with_controls;
use xlsynth_g8r::gatify::ir2gate::GatifyOptions;
use xlsynth_g8r::liberty::load::load_library_from_path;
use xlsynth_g8r::netlist::abc::{
    ABC_SPEED_SCRIPT, AbcMappedDesign, AbcMappingContext, AbcMappingOptions, AbcToolchain,
};
use xlsynth_g8r::netlist::gv_eval::{
    GvEvalOptions, load_labeled_sequential_netlist_aig_with_liberty,
};
use xlsynth_g8r::netlist::normalized::{BitExpr, BitSource, NormalizedNetlistModule};
use xlsynth_pir::IrValue;
use xlsynth_pir::ir_random::{
    DepletableBytes, OperationSet, RandomBlockOptions, RandomBlockResetTiming, RandomFnOptions,
    RandomOperation, StopPolicy, generate_block_package,
};
use xlsynth_pir::random_inputs::{generate_mixed_irbits_with_rng, generate_mixed_values_with_rng};

use crate::random_block::flatten_value;

static CONTEXT: OnceLock<Result<AbcMappingContext, String>> = OnceLock::new();

/// Reads explicit raw/proto inputs once; setup errors are never sample skips.
pub fn context() -> Result<&'static AbcMappingContext, &'static str> {
    CONTEXT
        .get_or_init(|| {
            let raw = std::env::var("XLSYNTH_LIBERTY_FILES").map_err(|e| e.to_string())?;
            let proto = std::env::var("XLSYNTH_LIBERTY_PROTO").map_err(|e| e.to_string())?;
            let files = raw
                .split(',')
                .map(str::trim)
                .map(Path::new)
                .collect::<Vec<_>>();
            let library = load_library_from_path(Path::new(&proto)).map_err(|e| e.to_string())?;
            AbcMappingContext::new(
                AbcToolchain::from_env().map_err(|e| e.to_string())?,
                &files,
                library,
            )
            .map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(String::as_str)
}

/// Requires real ABC mapping and a usable FF before starting the campaign.
pub fn preflight() -> Result<(), String> {
    let mut builder = GateBuilder::new("probe".into(), GateBuilderOptions::no_opt());
    let a = builder.add_input("a".into(), 1);
    builder.add_output("y".into(), a);
    let design = add_output_registers(
        &SequentialGateFn::from_gate_fn(builder.build()),
        ClockPort { name: "clk".into() },
    )?;
    let context = context()?;
    let options = AbcMappingOptions::from_library(context.library())?;
    context
        .map_sequential(&design, &options)
        .map_err(|e| e.to_string())?;
    // The direct identity requires a non-inverting FF and bypasses ABC. Also
    // require an actual combinational mapping before starting the campaign.
    let mut logic = GateBuilder::new("probe_logic".into(), GateBuilderOptions::no_opt());
    let a = logic.add_input("a".into(), 1);
    logic.add_output("y".into(), a.get_lsb(0).negate().into());
    context
        .map_sequential(&SequentialGateFn::from_gate_fn(logic.build()), &options)
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Finds wire equivalence without treating buffers or inverters as aliases.
fn wire_root(parents: &mut [usize], mut bit: usize) -> usize {
    while parents[bit] != bit {
        parents[bit] = parents[parents[bit]];
        bit = parents[bit];
    }
    bit
}

/// Checks protected connections on the emitted netlist, independently of the
/// boundary builder. Functional equivalence alone cannot detect extra buffers.
fn check_direct_connections(mapped: &AbcMappedDesign) {
    let parsed = &mapped.netlist;
    let normalized =
        NormalizedNetlistModule::new(&parsed.modules[0], &parsed.nets, &parsed.interner).unwrap();
    let mut parents = normalized.canonical_bits().to_vec();
    for assign in &normalized.assigns {
        for (lhs, rhs) in assign.lhs_bits.iter().zip(&assign.rhs_bits) {
            if let BitExpr::Source(BitSource::Bit(rhs)) = rhs {
                let lhs = wire_root(&mut parents, *lhs);
                let rhs = wire_root(&mut parents, *rhs);
                parents[lhs] = rhs;
            }
        }
    }
    let ports = normalized
        .ports
        .iter()
        .map(|p| (parsed.interner.resolve(p.name).unwrap(), &p.bits))
        .collect::<BTreeMap<_, _>>();
    let design = &mapped.design;
    let input_bits = design
        .inputs
        .iter()
        .flat_map(|id| {
            let input = &design.transition.inputs[id.index()];
            input
                .bit_vector
                .iter_lsb_to_msb()
                .enumerate()
                .map(|(bit, operand)| (*operand, ports[input.name.as_str()][bit]))
        })
        .collect::<BTreeMap<_, _>>();
    let mut output_bits = BTreeMap::<_, Vec<_>>::new();
    for id in &design.outputs {
        let output = &design.transition.outputs[id.index()];
        for (bit, operand) in output.bit_vector.iter_lsb_to_msb().enumerate() {
            output_bits
                .entry(*operand)
                .or_default()
                .push(ports[output.name.as_str()][bit]);
        }
    }
    // Assembly appends FFs in mapping-plan order; ABC is forbidden to emit
    // sequential cells. Filter by the selected physical cell names.
    let instances = normalized
        .instances
        .iter()
        .filter(|instance| {
            mapped.registers.cells.iter().any(|cell| {
                parsed.interner.resolve(instance.type_name) == Some(cell.cell_name.as_str())
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(instances.len(), mapped.registers.bits.len());
    for (bit, instance) in mapped.registers.bits.iter().zip(instances) {
        let cell = &mapped.registers.cells[bit.cell_index];
        assert_eq!(
            parsed.interner.resolve(instance.type_name),
            Some(cell.cell_name.as_str())
        );
        let pin = |name: &str| {
            let connection = instance
                .connections
                .iter()
                .find(|c| parsed.interner.resolve(c.port) == Some(name))
                .unwrap();
            let [BitSource::Bit(pin)] = connection.bits.as_slice() else {
                panic!("expected a scalar connected FF pin");
            };
            *pin
        };
        let d = design.transition.outputs[bit.d.index()]
            .bit_vector
            .get_lsb(bit.bit_index);
        if let Some(input) = input_bits.get(d) {
            assert_eq!(
                wire_root(&mut parents, pin(&cell.data_pin)),
                wire_root(&mut parents, *input),
                "direct input-to-D wire"
            );
        }
        let q = design.transition.inputs[bit.q.index()]
            .bit_vector
            .get_lsb(bit.bit_index);
        if let Some(outputs) = output_bits.get(q) {
            for output in outputs {
                assert_eq!(
                    wire_root(&mut parents, pin(&cell.output_pin)),
                    wire_root(&mut parents, *output),
                    "direct Q-to-output wire"
                );
            }
        }
    }
}

fn random_state(design: &SequentialGateFn, rng: &mut StdRng) -> SequentialState {
    let widths = design
        .registers
        .iter()
        .map(|r| design.transition.inputs[r.q.index()].get_bit_count())
        .collect::<Vec<_>>();
    SequentialState::from_register_values(design, generate_mixed_irbits_with_rng(rng, &widths))
        .unwrap()
}

/// Checks both cleanup settings against original G8R for reset and data cycles.
pub fn check(data: &[u8]) {
    check_with_script(data, ABC_SPEED_SCRIPT);
}

/// Shares the equivalence oracle with the custom-script integration smoke test.
fn check_with_script(data: &[u8], script: &str) {
    let context = context().expect("ABC preflight must run first");
    let mut rng = StdRng::from_seed(*blake3::hash(data).as_bytes());
    let generated = generate_block_package(
        &mut DepletableBytes::new(data),
        &RandomBlockOptions {
            min_output_ports: 1,
            max_output_ports: 3,
            max_input_ports: 4,
            max_registers: 4,
            require_reset_on_all_registers: true,
            reset_timing: RandomBlockResetTiming::Synchronous,
            function_options: RandomFnOptions {
                max_params: 5,
                max_nodes: 24,
                max_bit_width: 8,
                allow_arbitrary_width_multiply: true,
                enabled_operations: OperationSet::new(
                    OperationSet::all_supported().iter().filter(|op| {
                        !matches!(op, RandomOperation::Umulp | RandomOperation::Smulp)
                    }),
                ),
                ..Default::default()
            },
            ..Default::default()
        },
        StopPolicy::WhenEntropyDepleted,
    )
    .expect("bounded block generator options are valid");
    let block = generated.package.get_top_block().unwrap();
    let reset = block.reset.as_ref().expect("resettable generation profile");
    let mut original = block_package_to_sequential_gate_fn_with_controls(
        &generated.package,
        GatifyOptions::all_opts_enabled(),
    )
    .unwrap();
    assert!(!original.registers.is_empty());
    assert!(original.registers.iter().all(|r| r.reset.is_some()));
    // Generic optional pipeline boundaries exercise unreset input/output
    // registers as well as the resettable state generated by block IR.
    let clock = original.clock.clone().unwrap();
    let flop_inputs = rng.gen_bool(0.5);
    let flop_outputs = rng.gen_bool(0.5);
    if flop_inputs {
        original = add_input_registers(&original, clock.clone()).unwrap();
    }
    if flop_outputs {
        original = add_output_registers(&original, clock).unwrap();
    }
    let warmup_cycles = 1 + usize::from(flop_inputs) + usize::from(flop_outputs);
    let source = emit_g8r(&original).unwrap();
    let mut options = AbcMappingOptions::from_library(context.library())
        .expect("library defaults were validated during ABC preflight");
    options.timeout = Duration::from_secs(30);
    options.script = script.to_owned();
    for cleanup_registers in [false, true] {
        options.cleanup_registers = cleanup_registers;
        let mapped = context
            .map_sequential(&original, &options)
            .unwrap_or_else(|error| panic!("ABC mapping failed: {error}\n{source}"));
        mapped.design.validate_legalized().unwrap();
        check_direct_connections(&mapped);
        let gv = mapped.emit_verilog().unwrap();
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(file.path(), &gv).unwrap();
        let physical = load_labeled_sequential_netlist_aig_with_liberty(
            file.path(),
            context.library(),
            &GvEvalOptions {
                clock_port_name: original.clock.as_ref().map(|c| c.name.clone()),
                ..Default::default()
            },
        )
        .unwrap_or_else(|error| {
            panic!("mapped netlist cannot be evaluated: {error}\n{source}\n{gv}")
        });
        let implementation = &physical.sequential_gate_fn;
        let expected_inputs = original
            .inputs
            .iter()
            .map(|id| &original.transition.inputs[id.index()])
            .filter(|p| p.get_bit_count() != 0)
            .map(|p| (p.name.as_str(), p.get_bit_count()))
            .collect::<BTreeMap<_, _>>();
        let actual_inputs = implementation
            .inputs
            .iter()
            .map(|id| &implementation.transition.inputs[id.index()])
            .map(|p| (p.name.as_str(), p.get_bit_count()))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(expected_inputs, actual_inputs, "mapped input interface");
        let expected_outputs = original
            .outputs
            .iter()
            .map(|id| &original.transition.outputs[id.index()])
            .filter(|p| p.get_bit_count() != 0)
            .map(|p| (p.name.as_str(), p.get_bit_count()))
            .collect::<BTreeMap<_, _>>();
        let actual_outputs = implementation
            .outputs
            .iter()
            .map(|id| &implementation.transition.outputs[id.index()])
            .map(|p| (p.name.as_str(), p.get_bit_count()))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(expected_outputs, actual_outputs, "mapped output interface");
        let mut before = random_state(&original, &mut rng);
        let mut after = random_state(implementation, &mut rng);
        for cycle in 0..24 {
            let mut values = generate_mixed_values_with_rng(
                &mut rng,
                block.input_ports().map(|p| block.port_type(p)),
            );
            let asserted = cycle < warmup_cycles || rng.gen_bool(0.15);
            for (index, port) in block.input_ports().enumerate() {
                if port == reset.port {
                    values[index] =
                        IrValue::make_ubits(1, u64::from(asserted ^ reset.active_low)).unwrap();
                }
            }
            let inputs = block
                .input_ports()
                .zip(&values)
                .map(|(port, value)| {
                    (
                        block.port_name(port),
                        flatten_value(value, block.port_type(port)),
                    )
                })
                .collect::<BTreeMap<_, _>>();
            let source_inputs = original
                .inputs
                .iter()
                .map(|id| inputs[original.transition.inputs[id.index()].name.as_str()].clone())
                .collect::<Vec<_>>();
            let mapped_inputs = implementation
                .inputs
                .iter()
                .map(|id| {
                    inputs[implementation.transition.inputs[id.index()].name.as_str()].clone()
                })
                .collect::<Vec<_>>();
            let expected = simulate(&original, &[source_inputs], before).unwrap();
            let actual = physical.simulate_bits(&[mapped_inputs], after).unwrap();
            if cycle >= warmup_cycles {
                let results = implementation
                    .outputs
                    .iter()
                    .zip(&actual.external_outputs()[0])
                    .map(|(id, value)| {
                        (
                            implementation.transition.outputs[id.index()].name.as_str(),
                            value,
                        )
                    })
                    .collect::<BTreeMap<_, _>>();
                for (id, value) in original.outputs.iter().zip(&expected.external_outputs()[0]) {
                    if value.get_bit_count() != 0 {
                        assert_eq!(
                            &value,
                            &results[original.transition.outputs[id.index()].name.as_str()],
                            "mapped output mismatch: cleanup={cleanup_registers}, cycle={cycle}\n{source}\n{gv}"
                        );
                    }
                }
            }
            before = expected.final_state().clone();
            after = actual.final_state().clone();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngCore;

    #[test]
    fn configured_abc_mapping_smoke() {
        // This optional external-tool test requires an explicitly configured
        // ABC and matching raw/proto libraries; ordinary unit tests need none.
        if std::env::var_os("XLSYNTH_ABC_PATH").is_none() {
            return;
        }
        preflight().unwrap();
        let mut rng = StdRng::seed_from_u64(0xabc);
        for length in [0, 32, 128, 512, 1024] {
            let mut data = vec![0; length];
            rng.fill_bytes(&mut data);
            check(&data);
            check_with_script(&data, "&get -n; &dch; &nf; &put");
        }
    }
}
