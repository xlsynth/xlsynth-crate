// SPDX-License-Identifier: Apache-2.0

//! Reset-reachable equivalence against a complete ABC/native mapped netlist.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Duration;

use rand::{Rng, SeedableRng, rngs::StdRng};
use xlsynth_g8r::aig::{
    ClockPort, GateBuilder, GateBuilderOptions, SequentialGateFn, add_output_registers,
};
use xlsynth_g8r::aig_serdes::g8r::emit_g8r;
use xlsynth_g8r::aig_sim::sequential::{SequentialState, simulate};
use xlsynth_g8r::block2sequential::block_package_to_sequential_gate_fn_with_controls;
use xlsynth_g8r::gatify::ir2gate::GatifyOptions;
use xlsynth_g8r::liberty::load::load_library_from_path;
use xlsynth_g8r::netlist::abc::{AbcMappingContext, AbcMappingOptions, AbcToolchain};
use xlsynth_g8r::netlist::gv_eval::{
    GvEvalOptions, load_labeled_sequential_netlist_aig_with_liberty,
};
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
    context()?
        .map_sequential(&design, &Default::default())
        .map_err(|e| e.to_string())?;
    Ok(())
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
    let original = block_package_to_sequential_gate_fn_with_controls(
        &generated.package,
        GatifyOptions::all_opts_enabled(),
    )
    .unwrap();
    assert!(!original.registers.is_empty());
    assert!(original.registers.iter().all(|r| r.reset.is_some()));
    let source = emit_g8r(&original).unwrap();
    for cleanup_registers in [false, true] {
        let mapped = context
            .map_sequential(
                &original,
                &AbcMappingOptions {
                    cleanup_registers,
                    timeout: Duration::from_secs(30),
                    ..Default::default()
                },
            )
            .unwrap_or_else(|error| panic!("ABC mapping failed: {error}\n{source}"));
        mapped.design.validate_legalized().unwrap();
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
            let asserted = cycle == 0 || (cycle > 1 && rng.gen_bool(0.15));
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
            if cycle > 0 {
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
        }
    }
}
