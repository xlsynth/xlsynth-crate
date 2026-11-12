// SPDX-License-Identifier: Apache-2.0

use crate::aig::{
    AigBitVector, ClockPort, GateBuilder, GateBuilderOptions, SequentialGateFn,
    add_input_registers, add_output_registers,
};
use crate::aig_sim::sequential::{SequentialState, simulate};
use crate::liberty_model::Library;
use crate::netlist::fanout_opt::FanoutOptimizationOptions;
use crate::netlist::gatefn_from_netlist::project_labeled_sequential_netlist_aig;
use crate::netlist::io::{ParsedNetlist, read_gv_from_str};
use crate::netlist::mapped::MappedNetlist;
use crate::netlist::report::build_area_report;
use crate::netlist::sequential_mapping::{
    SequentialMappingOptions, map_sequential_to_native_netlist,
};
use crate::netlist::sta::{PreparedStaLibrary, StaOptions, analyze_register_boundary_max_arrival};
use crate::techmap::postprocess::{CoverStrategy, PostMappingOptions};
use anyhow::{Error, Result};
use xlsynth_pir::IrBits;

/// Reads the first module of a fixture while retaining its full net table and
/// interner.
pub(super) fn read_mapped_netlist(source: &str) -> Result<MappedNetlist> {
    let mut parsed = read_gv_from_str(source)?;
    Ok(MappedNetlist {
        module: parsed.modules.remove(0),
        nets: parsed.nets,
        interner: parsed.interner,
    })
}

/// A broadcast Q can drive four AND inputs directly. The buffer can drive
/// sixteen.
pub(super) fn library() -> Result<Library> {
    let mut lib = crate::netlist::resize::tests::library()?;
    lib.cells
        .retain(|c| matches!(c.name.as_str(), "DFF1" | "BUF4" | "AND1" | "INV1"));
    for cell in &mut lib.cells {
        for pin in &mut cell.pins {
            if pin.direction == crate::liberty_proto::PinDirection::Output as i32 {
                pin.max_capacitance = Some(if cell.name == "DFF1" { 0.8 } else { 3.2 });
            } else if cell.name == "BUF4" {
                pin.capacitance = Some(0.08);
            }
        }
    }
    Ok(lib)
}

#[test]
fn capture_broadcast_can_be_repaired_only_for_a_proven_data_pin() -> Result<()> {
    use crate::netlist::registered_opt::{
        RegisteredOptimizationOptions, optimize_registered_netlist,
    };
    use std::fmt::Write;

    let mut lib = library()?;
    let mut scan = lib.cells.iter().find(|c| c.name == "DFF1").unwrap().clone();
    scan.name = "SDFF".into();
    let state_dependent = format!("D ^ {}", scan.sequential[0].state_var);
    let and = lib.cells.iter().find(|c| c.name == "AND1").unwrap();
    scan.pins.extend(
        and.pins
            .iter()
            .filter(|p| p.direction == crate::liberty_proto::PinDirection::Input as i32)
            .cloned(),
    );
    let capture = lib.cells.len();
    lib.cells.push(scan);
    for (next_state, select, repairable) in [
        ("!D", "1'b0", true),
        ("(!A & D) | (A & B)", "1'b0", true),
        ("(!A & D) | (A & B)", "en", false),
        ("(!A & !D) | (A & B)", "1'b0", true),
        ("(!A & !D) | (A & B)", "en", false),
        ("(!A & !D) | (A & B)", "1'b1", false),
        (state_dependent.as_str(), "1'b0", false),
    ] {
        lib.cells[capture].sequential[0].next_state = next_state.into();
        let mut gv = String::from(
            "module t(input clk, input d, input en, output [31:0] y);
             wire q, shared;
             DFF1 launch(.CLK(clk),.D(d),.Q(q));
             INV1 inv(.A(q),.Y(shared));",
        );
        for i in 0..32 {
            writeln!(
                gv,
                "SDFF cap{i}(.CLK(clk),.A({select}),.B(1'b0),.D(shared),.Q(y[{i}]));"
            )?;
        }
        gv.push_str("endmodule");
        let report = optimize_registered_netlist(
            &mut read_gv_from_str(&gv)?,
            &lib,
            &RegisteredOptimizationOptions {
                timing: StaOptions::default(),
                fanout: Some(FanoutOptimizationOptions::default()),
                resize: None,
            },
        )?;
        if repairable {
            assert!(
                report.final_delay.register_to_register < report.initial_delay.register_to_register,
                "next_state={next_state}, select={select}"
            );
        } else {
            assert_eq!(
                report.topology_trials, 0,
                "next_state={next_state}, select={select}"
            );
        }
    }
    Ok(())
}

fn broadcast(lanes: usize) -> Result<SequentialGateFn> {
    let mut b = GateBuilder::new("broadcast".into(), GateBuilderOptions::no_opt());
    let enable = *b.add_input("enable".into(), 1).get_lsb(0);
    let data = b.add_input("data".into(), lanes);
    let outputs: Vec<_> = data
        .iter_lsb_to_msb()
        .map(|&bit| b.add_and_binary(enable, bit))
        .collect();
    b.add_output("mask".into(), AigBitVector::from_lsb_is_index_0(&outputs));
    let clock = ClockPort { name: "clk".into() };
    let launched = add_input_registers(&SequentialGateFn::from_gate_fn(b.build()), clock.clone())
        .map_err(Error::msg)?;
    add_output_registers(&launched, clock).map_err(Error::msg)
}

fn metrics(parsed: &ParsedNetlist, lib: &Library) -> Result<(f64, f64)> {
    let module = &parsed.modules[0];
    let flops = module
        .instances
        .iter()
        .enumerate()
        .filter(|(_, inst)| parsed.interner.resolve(inst.type_name) == Some("DFF1"))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let timing = analyze_register_boundary_max_arrival(
        module,
        &parsed.nets,
        &parsed.interner,
        lib,
        StaOptions::default(),
        false,
        &flops,
    )?;
    assert!(timing.worst_register_input_breakdown.is_some());
    Ok((
        timing.worst_register_input_arrival,
        build_area_report(module, &parsed.interner, lib)?.area,
    ))
}

// One launch Q drives a bank of capture paths. Verify that buffering lowers
// the registered delay and that every lane still obeys the clocked enable.
#[test]
fn native_mapping_repairs_the_loaded_launch_q_and_preserves_the_clocked_mask() -> Result<()> {
    let lanes = 32;
    let lib = library()?;
    let design = broadcast(lanes)?;
    let options = SequentialMappingOptions::default();
    let baseline = map_sequential_to_native_netlist(
        &design,
        &lib,
        &options,
        CoverStrategy::Area,
        &PostMappingOptions::default(),
    )?;
    let post = PostMappingOptions {
        fanout: Some(FanoutOptimizationOptions {
            module_output_load: StaOptions::default().module_output_load,
            ..Default::default()
        }),
        ..Default::default()
    };
    let repaired =
        map_sequential_to_native_netlist(&design, &lib, &options, CoverStrategy::Area, &post)?;
    let (old_delay, old_area) = metrics(&baseline, &lib)?;
    let (new_delay, new_area) = metrics(&repaired, &lib)?;
    assert!(
        new_delay < old_delay,
        "registered path: {old_delay} -> {new_delay}"
    );
    assert!(new_area * new_delay < old_area * old_delay);
    let instantiated = |parsed: &ParsedNetlist, name: &str| {
        parsed.modules[0]
            .instances
            .iter()
            .filter(|i| parsed.interner.resolve(i.type_name) == Some(name))
            .count()
    };
    assert_eq!(
        instantiated(&baseline, "DFF1"),
        instantiated(&repaired, "DFF1")
    );
    assert!(instantiated(&repaired, "BUF4") > instantiated(&baseline, "BUF4"));
    let recovered = project_labeled_sequential_netlist_aig(
        &repaired.modules[0],
        &repaired.nets,
        &repaired.interner,
        &lib,
        Some("clk"),
    )
    .map_err(Error::msg)?
    .sequential_gate_fn;
    // Walk each data bit with the broadcast both asserted and deasserted.
    let mut stimulus = Vec::new();
    for i in 0..lanes {
        let word = IrBits::from_lsb_is_0(&(0..lanes).map(|j| j == i).collect::<Vec<_>>());
        for enabled in [true, false] {
            stimulus.push(vec![IrBits::bool(enabled), word.clone()]);
        }
    }
    stimulus.extend_from_within(0..2);
    let expected =
        simulate(&design, &stimulus, SequentialState::all_zeros(&design)).map_err(Error::msg)?;
    let actual = simulate(
        &recovered,
        &stimulus,
        SequentialState::all_zeros(&recovered),
    )
    .map_err(Error::msg)?;
    assert_eq!(expected.external_outputs(), actual.external_outputs());
    Ok(())
}

/// Two independent electrical bottlenecks share a stage; the third broadcast
/// is also an aliased generated clock, so every tap of that net stays fixed.
#[test]
fn joint_candidates_can_buffer_launch_and_internal_data_together() -> Result<()> {
    use std::fmt::Write;
    let mut gv = String::from(
        "module t(input clk,input d,input mask,output [49:0] y);
        wire q, shared, inverted_clk, clock_alias;
        assign clock_alias = inverted_clk;
        DFF1 launch(.CLK(clk),.D(d),.Q(q));
        AND1 shared_gate(.A(d),.B(mask),.Y(shared));
        INV1 phase(.A(clk),.Y(inverted_clk));
        DFF1 other_clock(.CLK(clock_alias),.D(d),.Q(y[49]));",
    );
    for i in 0..49 {
        let input = match i {
            0..=15 => "q",
            16..=31 => "shared",
            _ => "clock_alias",
        };
        writeln!(gv, "DFF1 sink{i}(.CLK(clk),.D({input}),.Q(y[{i}]));")?;
    }
    gv.push_str("endmodule");
    let original = read_mapped_netlist(&gv)?;
    let mut together = false;
    let lib = library()?;
    let timing = StaOptions {
        module_output_load: 0.01,
        ..StaOptions::default()
    };
    let roots = super::graph::collect_roots(&original, &lib, timing.module_output_load, Some(8))?;
    assert!(roots.iter().any(|r| r.instance_name == "launch"));
    assert!(roots.iter().any(|r| r.instance_name == "shared_gate"));
    assert!(roots.iter().all(|r| r.instance_name != "phase"));
    let total = super::visit_topology_candidates(
        &original,
        &lib,
        &PreparedStaLibrary::new(&lib)?,
        timing,
        &FanoutOptimizationOptions {
            fanout_trigger: 8,
            ..Default::default()
        },
        |trial, _| {
            // First four instances include both driver types and the
            // generated-clock capture.
            let moved = |i: usize| {
                trial.module.instances[4 + i].connections
                    != original.module.instances[4 + i].connections
            };
            assert!(!(32..49).any(moved));
            assert_eq!(trial.module.instances[2], original.module.instances[2]);
            assert_eq!(trial.module.instances[3], original.module.instances[3]);
            if trial.module.instances[original.module.instances.len()..]
                .iter()
                .all(|i| trial.interner.resolve(i.type_name) == Some("BUF4"))
            {
                together |= (0..16).any(moved) && (16..32).any(moved);
            }
            Ok(None)
        },
    )?;
    assert!(
        total > 0 && together,
        "missing a mixed launch/internal forest"
    );
    Ok(())
}

// Buffer a combinational driver with an externally visible alias. The
// capture pins can move to a new branch while the port stays on the driver.
#[test]
fn buffering_preserves_registered_logic_and_external_alias() -> Result<()> {
    use std::fmt::Write;
    let mut gv = String::from(
        "module aliased_fanout(input clk,input d,input mask,output tap,output [15:0] y);
        wire q, shared, alias_shared; assign alias_shared=shared; assign tap=alias_shared;
        DFF1 launch(.CLK(clk),.D(d),.Q(q)); AND1 compute(.A(q),.B(mask),.Y(shared));",
    );
    for i in 0..16 {
        writeln!(gv, "DFF1 sink{i}(.CLK(clk),.D(alias_shared),.Q(y[{i}]));")?;
    }
    gv.push_str("endmodule");
    let original = read_mapped_netlist(&gv)?;
    let lib = library()?;
    let stimulus: Vec<_> = (0..8)
        .map(|n| vec![IrBits::bool(n & 1 != 0), IrBits::bool(n & 2 != 0)])
        .collect();
    let run = |net: &MappedNetlist| -> Result<_> {
        let design = project_labeled_sequential_netlist_aig(
            &net.module,
            &net.nets,
            &net.interner,
            &lib,
            Some("clk"),
        )
        .map_err(Error::msg)?
        .sequential_gate_fn;
        simulate(&design, &stimulus, SequentialState::all_zeros(&design)).map_err(Error::msg)
    };
    let expected = run(&original)?;
    let mut buffered = 0;
    let options = FanoutOptimizationOptions {
        fanout_trigger: 8,
        ..Default::default()
    };
    let timing = StaOptions {
        module_output_load: 0.01,
        ..Default::default()
    };
    super::visit_topology_candidates(
        &original,
        &lib,
        &PreparedStaLibrary::new(&lib)?,
        timing,
        &options,
        |trial, _| {
            let added = &trial.module.instances[original.module.instances.len()..];
            assert!(!added.is_empty());
            assert!(
                added
                    .iter()
                    .all(|i| trial.interner.resolve(i.type_name) == Some("BUF4"))
            );
            assert_eq!(trial.module.assigns, original.module.assigns);
            assert_eq!(trial.module.instances[..2], original.module.instances[..2]);
            assert_eq!(run(&trial)?, expected);
            buffered += 1;
            Ok(None)
        },
    )?;
    assert!(buffered > 0);
    Ok(())
}

// An otherwise useful forest requests repair of one additional output.
// Check which wires move and simulate the resulting registered circuit.
#[test]
fn a_forest_retries_only_a_named_driver_output_and_keeps_registered_values() -> Result<()> {
    use super::{ForestRetry, TopologyTrial};
    use std::fmt::Write;

    // Leave a launch outside the high-fanout batch so a trial can identify
    // collateral damage at that output. The two inputs toggle independently.
    let banks = 65;
    let sinks = 16;
    let mut gv = format!(
        "module t(input clk, input a, input b, output [{}:0] y);",
        banks * sinks - 1
    );
    for bank in 0..banks {
        let input = if bank % 2 == 0 { "a" } else { "b" };
        writeln!(
            gv,
            "wire q{bank}; DFF1 root{bank}(.CLK(clk),.D({input}),.Q(q{bank}));"
        )?;
        for i in 0..sinks {
            let output = bank * sinks + i;
            writeln!(
                gv,
                "DFF1 cap{output}(.CLK(clk),.D(q{bank}),.Q(y[{output}]));"
            )?;
        }
    }
    gv.push_str("endmodule");
    let original = read_mapped_netlist(&gv)?;
    let lib = library()?;
    let stimulus = (0..8)
        .map(|i| vec![IrBits::bool(i & 1 != 0), IrBits::bool(i & 2 != 0)])
        .collect::<Vec<_>>();
    let simulate_stage = |net: &MappedNetlist| -> Result<_> {
        let design = project_labeled_sequential_netlist_aig(
            &net.module,
            &net.nets,
            &net.interner,
            &lib,
            Some("clk"),
        )
        .map_err(Error::msg)?
        .sequential_gate_fn;
        simulate(&design, &stimulus, SequentialState::all_zeros(&design)).map_err(Error::msg)
    };
    let expected = simulate_stage(&original)?;
    for output in ["D", "Q"] {
        let mut retries = 0;
        let mut requested = 0;
        super::visit_topology_candidates(
            &original,
            &lib,
            &PreparedStaLibrary::new(&lib)?,
            StaOptions::default(),
            &FanoutOptimizationOptions::default(),
            |trial, kind| {
                let direct = |bank: usize| {
                    let first_sink = bank * (sinks + 1) + 1;
                    original.module.instances[first_sink].connections
                        == trial.module.instances[first_sink].connections
                };
                match kind {
                    TopologyTrial::Forest => {
                        let outside = (0..banks)
                            .find(|&bank| direct(bank))
                            .expect("one driver outside the batch");
                        requested += 1;
                        return Ok(Some(ForestRetry {
                            rank: (0, (0..banks).filter(|&bank| direct(bank)).count() as f64),
                            outputs: vec![(outside * (sinks + 1), output.into())],
                        }));
                    }
                    TopologyTrial::ElectricalRetry => {
                        retries += 1;
                        assert!(!(0..banks).any(direct));
                        assert_eq!(simulate_stage(&trial)?, expected);
                    }
                }
                Ok(None)
            },
        )?;
        assert!(requested > 0);
        assert_eq!(retries, usize::from(output == "Q"));
    }
    Ok(())
}
