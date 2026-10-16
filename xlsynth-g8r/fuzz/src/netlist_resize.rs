// SPDX-License-Identifier: Apache-2.0

//! End-to-end resizing checks on the shared direct-G8R graph distribution.

use std::collections::HashSet;
use std::io::Write;
use std::sync::OnceLock;

use rand::{Rng, SeedableRng, rngs::StdRng};
use tempfile::NamedTempFile;
use xlsynth_g8r::aig::SequentialGateFn;
use xlsynth_g8r::aig_sim::sequential::{SequentialState, simulate};
use xlsynth_g8r::gate_random::{RandomGateBuilderMode, RandomGateOptions, generate_gate_design};
use xlsynth_g8r::liberty::Library;
use xlsynth_g8r::liberty::parser::parse_liberty_files;
use xlsynth_g8r::netlist::gatefn_from_netlist::project_labeled_sequential_netlist_aig;
use xlsynth_g8r::netlist::parse::{
    Net, NetIndex, NetRef, NetlistInstance, NetlistPort, PortDirection,
};
use xlsynth_g8r::netlist::resize::{ResizeOptions, ResizeReport, resize_netlist};
use xlsynth_g8r::netlist::sta::StaOptions;
use xlsynth_g8r::netlist::techmap::{
    StructuralMappedNetlist, StructuralTechMapOptions, map_gatefn_to_structural_netlist,
};
use xlsynth_pir::IrBits;
use xlsynth_pir::ir_random::DepletableBytes;
use xlsynth_pir::random_inputs::generate_mixed_irbits_with_rng;

/// Small, asymmetric, slew/load-sensitive cell families; no external PDK or
/// synthesis executable is needed to reproduce a failure.
fn library() -> &'static Library {
    static LIBRARY: OnceLock<Library> = OnceLock::new();
    LIBRARY.get_or_init(|| {
        let mut text = String::from(
            r#"library(test) {
            lu_table_template(delay) {
                variable_1: input_net_transition;
                variable_2: total_output_net_capacitance;
                index_1("0,10"); index_2("0,10");
            }
            "#,
        );
        for kind in ["INV", "NAND2", "DFF"] {
            for size in [1, 2, 4] {
                let d = 1.0 / f64::from(size);
                let cap = f64::from(size) * 0.2;
                let (input, output, function, sense, timing_type) = match kind {
                    "INV" => ("A", "Y", "!A", "negative_unate", "combinational"),
                    "NAND2" => ("A", "Y", "!(A & B)", "negative_unate", "combinational"),
                    "DFF" => ("D", "Q", "IQ", "non_unate", "rising_edge"),
                    _ => unreachable!(),
                };
                text.push_str(&format!(
                    r#"cell({kind}_X{size}) {{ area: {size};
                    pin({input}) {{ direction: input; capacitance: {cap}; "#
                ));
                if kind == "DFF" {
                    text.push_str(&format!(r#"timing() {{ related_pin: "CLK"; timing_type: setup_rising;
                        rise_constraint(scalar) {{ values("{d}"); }}
                        fall_constraint(scalar) {{ values("{d}"); }} }}"#));
                }
                text.push('}');
                if kind == "NAND2" {
                    text.push_str(&format!(
                        r#"pin(B) {{ direction: input;
                        rise_capacitance: {}; fall_capacitance: {}; }}"#,
                        cap * 1.5,
                        cap
                    ));
                }
                if kind == "DFF" {
                    text.push_str(r#"pin(CLK) { direction: input; clock: true; capacitance: 0.1; }
                        ff(IQ, IQN) { clocked_on: "CLK"; next_state: "D"; }"#);
                }
                text.push_str(&format!(
                    r#"pin({output}) {{ direction: output; function: "{function}";"#
                ));
                let pins = match kind {
                    "INV" => &["A"][..],
                    "NAND2" => &["A", "B"][..],
                    "DFF" => &["CLK"][..],
                    _ => unreachable!(),
                };
                for (pin_index, pin) in pins.iter().enumerate() {
                    text.push_str(&format!(
                        r#"timing() {{ related_pin: "{pin}"; timing_sense: {sense}; timing_type: {timing_type};"#
                    ));
                    for (edge_index, name) in [
                        "cell_rise",
                        "cell_fall",
                        "rise_transition",
                        "fall_transition",
                    ]
                    .iter()
                    .enumerate()
                    {
                        let intrinsic =
                            d * (1.0 + pin_index as f64 * 0.3 + edge_index as f64 * 0.1);
                        text.push_str(&format!(
                            r#"{name}(delay) {{ values("{},{}", "{},{}"); }}"#,
                            intrinsic,
                            intrinsic + d * 20.0,
                            intrinsic + 2.0,
                            intrinsic + d * 20.0 + 2.0
                        ));
                    }
                    text.push('}');
                }
                text.push_str("}}");
            }
        }
        text.push('}');
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(text.as_bytes()).unwrap();
        parse_liberty_files(&[file.path()]).unwrap()
    })
}

/// Matches the structural mapper's scalarized port names.
fn scalar_name(name: &str, bit: usize, width: usize) -> String {
    if width == 1 {
        name.to_owned()
    } else {
        format!("{name}_{bit}")
    }
}

/// Splits packed stimulus/state into the mapper's LSB-first scalar bindings.
fn scalar_bits(values: &[IrBits]) -> Vec<IrBits> {
    values
        .iter()
        .flat_map(|value| {
            (0..value.get_bit_count())
                .map(|bit| IrBits::from_lsb_is_0(&[value.get_bit(bit).unwrap()]))
        })
        .collect()
}

/// Maps the generated transition graph and connects its existing Q/D bindings
/// to ordinary positive-edge FFs. This adds no graph-generation distribution.
fn map_design(source: &SequentialGateFn) -> StructuralMappedNetlist {
    source.validate_legalized().unwrap();
    let mut mapped = map_gatefn_to_structural_netlist(
        &source.transition,
        &StructuralTechMapOptions {
            nand2_cell_name: "NAND2_X1".into(),
            inv_cell_name: "INV_X1".into(),
            ..StructuralTechMapOptions::default()
        },
    )
    .unwrap();
    let mut external = HashSet::new();
    for id in &source.inputs {
        let input = &source.transition.inputs[id.index()];
        for bit in 0..input.get_bit_count() {
            external.insert(scalar_name(&input.name, bit, input.get_bit_count()));
        }
    }
    for id in &source.outputs {
        let output = &source.transition.outputs[id.index()];
        for bit in 0..output.bit_vector.get_bit_count() {
            external.insert(scalar_name(
                &output.name,
                bit,
                output.bit_vector.get_bit_count(),
            ));
        }
    }
    for port in &mapped.module.ports {
        if !external.contains(mapped.interner.resolve(port.name).unwrap()) {
            mapped.module.wires.push(
                mapped
                    .module
                    .find_net_index(port.name, &mapped.nets)
                    .unwrap(),
            );
        }
    }
    mapped
        .module
        .ports
        .retain(|port| external.contains(mapped.interner.resolve(port.name).unwrap()));
    if let Some(clock) = &source.clock {
        let clock_name = mapped.interner.get_or_intern(&clock.name);
        let clock_net = NetIndex(mapped.nets.len());
        mapped.nets.push(Net {
            name: clock_name,
            width: None,
        });
        mapped.module.ports.push(NetlistPort {
            name: clock_name,
            width: None,
            direction: PortDirection::Input,
        });
        mapped.module.net_index_range.end = mapped.nets.len();
        for (index, register) in source.registers.iter().enumerate() {
            let q = &source.transition.inputs[register.q.index()];
            let d = &source.transition.outputs[register.d.index()];
            for bit in 0..q.get_bit_count() {
                let q_name = mapped
                    .interner
                    .get(scalar_name(&q.name, bit, q.get_bit_count()))
                    .unwrap();
                let d_name = mapped
                    .interner
                    .get(scalar_name(&d.name, bit, d.bit_vector.get_bit_count()))
                    .unwrap();
                let q_net = mapped.module.find_net_index(q_name, &mapped.nets).unwrap();
                let d_net = mapped.module.find_net_index(d_name, &mapped.nets).unwrap();
                mapped.module.instances.push(NetlistInstance {
                    type_name: mapped.interner.get_or_intern("DFF_X1"),
                    instance_name: mapped
                        .interner
                        .get_or_intern(format!("register_{index}_{bit}")),
                    connections: vec![
                        (
                            mapped.interner.get_or_intern("CLK"),
                            NetRef::Simple(clock_net),
                        ),
                        (mapped.interner.get_or_intern("D"), NetRef::Simple(d_net)),
                        (mapped.interner.get_or_intern("Q"), NetRef::Simple(q_net)),
                    ],
                    inst_lineno: 0,
                    inst_colno: 0,
                });
            }
        }
    }
    mapped
}

/// Checks Boolean equivalence, fixed timing/area guards, and deterministic
/// results; resizing itself independently checks incremental against full STA.
pub fn check(data: &[u8]) -> ResizeReport {
    let sample = generate_gate_design(
        &mut DepletableBytes::new(data),
        &RandomGateOptions {
            input_ports: 1..=4,
            input_width: 1..=4,
            output_width: 1..=4,
            registers: 0..=3,
            register_width: 1..=4,
            enable: None,
            reset: None,
            max_operations: 24,
            // The structural mapper requires literal-free cell inputs/outputs.
            // Keeping expressions unfolded and omitting constants satisfies that
            // contract without filtering particular graph shapes or resize moves.
            builder_mode: RandomGateBuilderMode::Unoptimized,
            allow_constants: false,
            ..RandomGateOptions::default()
        },
    )
    .unwrap();
    let mut mapped = map_design(&sample.design);
    let library = library();
    let mut rng = StdRng::from_seed(*blake3::hash(data).as_bytes());
    for instance in &mut mapped.module.instances {
        let name = mapped.interner.resolve(instance.type_name).unwrap();
        let kind = name.split_once("_X").unwrap().0;
        let size = [1, 2, 4][rng.gen_range(0..3)];
        instance.type_name = mapped.interner.get_or_intern(format!("{kind}_X{size}"));
    }
    let original = mapped.module.clone();
    let clock = sample
        .design
        .clock
        .as_ref()
        .map(|clock| clock.name.as_str());
    let before = project_labeled_sequential_netlist_aig(
        &original,
        &mapped.nets,
        &mapped.interner,
        library,
        clock,
    )
    .unwrap()
    .sequential_gate_fn;
    let options = ResizeOptions {
        timing: StaOptions {
            primary_input_transition: [0.0, 0.01, 1.0, 10.0][rng.gen_range(0..4)],
            module_output_load: [0.0, 0.1, 1.0, 10.0][rng.gen_range(0..4)],
        },
        max_area_growth: [0.0, 0.02, 0.5][rng.gen_range(0..3)],
        max_trials: rng.gen_range(0..=128),
        max_propagated_instances: rng.gen_range(0..=4096),
        ..ResizeOptions::default()
    };
    let report = resize_netlist(
        &mut mapped.module,
        &mapped.nets,
        &mut mapped.interner,
        library,
        &options,
    )
    .unwrap();
    for (initial, upsized, final_delay) in [
        (
            report.initial_delay.input_to_output,
            report.upsized_delay.input_to_output,
            report.final_delay.input_to_output,
        ),
        (
            report.initial_delay.input_to_register,
            report.upsized_delay.input_to_register,
            report.final_delay.input_to_register,
        ),
        (
            report.initial_delay.register_to_output,
            report.upsized_delay.register_to_output,
            report.final_delay.register_to_output,
        ),
        (
            report.initial_delay.register_to_register,
            report.upsized_delay.register_to_register,
            report.final_delay.register_to_register,
        ),
    ] {
        assert!(final_delay <= upsized + 1e-9);
        assert!(upsized <= initial + 1e-9);
    }
    assert!(report.final_area <= report.initial_area * (1.0 + options.max_area_growth) + 1e-9);
    assert!(report.upsize.trials + report.downsize.trials <= options.max_trials);
    assert!(
        report.upsize.propagated_instances + report.downsize.propagated_instances
            <= options.max_propagated_instances
    );
    let after = project_labeled_sequential_netlist_aig(
        &mapped.module,
        &mapped.nets,
        &mapped.interner,
        library,
        clock,
    )
    .unwrap()
    .sequential_gate_fn;
    let widths = sample
        .design
        .inputs
        .iter()
        .map(|i| sample.design.transition.inputs[i.index()].get_bit_count())
        .collect::<Vec<_>>();
    let register_widths = sample
        .design
        .registers
        .iter()
        .map(|r| sample.design.transition.inputs[r.q.index()].get_bit_count())
        .collect::<Vec<_>>();
    let values = generate_mixed_irbits_with_rng(&mut rng, &register_widths);
    let before_state =
        SequentialState::from_register_values(&before, scalar_bits(&values)).unwrap();
    let after_state = SequentialState::from_register_values(&after, scalar_bits(&values)).unwrap();
    let source_state = SequentialState::from_register_values(&sample.design, values).unwrap();
    let stimuli = (0..16)
        .map(|_| generate_mixed_irbits_with_rng(&mut rng, &widths))
        .collect::<Vec<_>>();
    let mapped_stimuli = stimuli
        .iter()
        .map(|values| scalar_bits(values))
        .collect::<Vec<_>>();
    let source_trace = simulate(&sample.design, &stimuli, source_state).unwrap();
    let before_trace = simulate(&before, &mapped_stimuli, before_state).unwrap();
    let after_trace = simulate(&after, &mapped_stimuli, after_state).unwrap();
    assert_eq!(
        source_trace
            .external_outputs()
            .iter()
            .map(|values| scalar_bits(values))
            .collect::<Vec<_>>(),
        before_trace.external_outputs()
    );
    assert_eq!(
        scalar_bits(source_trace.final_state().values()),
        before_trace.final_state().values()
    );
    assert_eq!(
        before_trace.external_outputs(),
        after_trace.external_outputs()
    );
    assert_eq!(before_trace.final_state(), after_trace.final_state());
    let mut repeated = original;
    let repeat_report = resize_netlist(
        &mut repeated,
        &mapped.nets,
        &mut mapped.interner,
        library,
        &options,
    )
    .unwrap();
    assert_eq!(repeated, mapped.module);
    assert_eq!(repeat_report.final_delay, report.final_delay);
    assert_eq!(repeat_report.final_area, report.final_area);
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::RngCore;

    #[test]
    fn seeded_shared_generator_resize_smoke() {
        let mut rng = StdRng::seed_from_u64(0x51e);
        let mut upsize = 0;
        let mut downsize = 0;
        let mut register_changes = 0;
        for length in 0..96 {
            let mut bytes = vec![0; length * 8];
            rng.fill_bytes(&mut bytes);
            let report = check(&bytes);
            upsize += report.upsize.accepted;
            downsize += report.downsize.accepted;
            register_changes += report.upsize.register_changes + report.downsize.register_changes;
        }
        assert!(upsize > 0 && downsize > 0 && register_changes > 0);
    }
}
