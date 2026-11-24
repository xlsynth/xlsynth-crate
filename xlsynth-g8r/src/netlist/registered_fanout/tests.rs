// SPDX-License-Identifier: Apache-2.0

use super::repair_registered_fanout;
use crate::aig::{
    AigBitVector, ClockPort, GateBuilder, GateBuilderOptions, SequentialGateFn,
    add_input_registers, add_output_registers,
};
use crate::aig_sim::sequential::{SequentialState, simulate};
use crate::liberty::model::{library_from_proto, library_to_proto};
use crate::liberty_model::Library;
use crate::liberty_proto::PinDirection;
use crate::netlist::emit::emit_module_as_netlist_text;
use crate::netlist::fanout_opt::FanoutOptimizationOptions;
use crate::netlist::gatefn_from_netlist::project_labeled_sequential_netlist_aig;
use crate::netlist::io::{ParsedNetlist, read_gv_from_str};
use crate::netlist::parse::NetRef;
use crate::netlist::report::build_area_report;
use crate::netlist::sequential_mapping::{
    SequentialMappingOptions, map_sequential_to_native_netlist,
};
use crate::netlist::sta::{StaOptions, analyze_register_boundary_max_arrival};
use crate::techmap::postprocess::{CoverStrategy, PostMappingOptions};
use anyhow::{Error, Result};
use std::fmt::Write;
use xlsynth_pir::IrBits;

/// A broadcast Q can drive four AND inputs directly. The buffer can drive
/// sixteen.
pub(super) fn library() -> Result<Library> {
    let mut lib = crate::netlist::resize::tests::library()?;
    lib.cells
        .retain(|c| matches!(c.name.as_str(), "DFF1" | "BUF4" | "AND1" | "INV1"));
    for cell in &mut lib.cells {
        for pin in &mut cell.pins {
            if pin.direction == PinDirection::Output as i32 {
                pin.max_capacitance = Some(if cell.name == "DFF1" { 0.8 } else { 3.2 });
            } else if cell.name == "BUF4" {
                pin.capacitance = Some(0.08);
            }
        }
    }
    Ok(lib)
}

/// Registers the enable and data inputs, then registers each masked output.
/// The enable register's Q drives one AND gate per lane.
fn registered_mask_broadcast(lanes: usize) -> Result<SequentialGateFn> {
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

/// Measures the worst register-to-register delay and total cell area.
fn registered_delay_and_area(parsed: &ParsedNetlist, lib: &Library) -> Result<(f64, f64)> {
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

// A high-fanout launch Q needs buffering without changing the clocked logic.
#[test]
fn native_mapping_repairs_the_loaded_launch_q_and_preserves_the_clocked_mask() -> Result<()> {
    let lanes = 32;
    let lib = library()?;
    let design = registered_mask_broadcast(lanes)?;
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
    let (old_delay, old_area) = registered_delay_and_area(&baseline, &lib)?;
    let (new_delay, new_area) = registered_delay_and_area(&repaired, &lib)?;
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

#[test]
fn legal_load_and_constant_capture_need_no_repair() -> Result<()> {
    let lib = library()?;
    let mut netlist = read_gv_from_str(
        "module test(input clk, input data, output bit_out, output zero); wire q;
         DFF1 launch(.CLK(clk),.D(data),.Q(q));
         DFF1 capture(.CLK(clk),.D(q),.Q(bit_out));
         DFF1 tied(.CLK(clk),.D(1'b0),.Q(zero)); endmodule",
    )?;
    let before =
        emit_module_as_netlist_text(&netlist.modules[0], &netlist.nets, &netlist.interner)?;
    let r = repair_registered_fanout(&mut netlist, &lib, StaOptions::default(), None)?;
    assert_eq!(r.accepted_trial, None);
    assert_eq!(
        emit_module_as_netlist_text(&netlist.modules[0], &netlist.nets, &netlist.interner)?,
        before
    );
    Ok(())
}

/// One Q drives sixteen D pins with 0.2 capacitance each.
fn capture_broadcast() -> Result<ParsedNetlist> {
    let mut text = String::from(
        "module test(input clk, input d, output [15:0] y); wire q; DFF1 launch(.CLK(clk),.D(d),.Q(q));",
    );
    for lane in 0..16 {
        writeln!(text, "DFF1 sink_{lane}(.CLK(clk),.D(q),.Q(y[{lane}]));")?;
    }
    text.push_str("endmodule");
    read_gv_from_str(&text)
}

// Negative setup times can make registered delay zero or negative; ranking must
// remain well-defined.
#[test]
fn negative_setup_can_make_either_the_trial_or_original_delay_nonpositive() -> Result<()> {
    for setup in [-4.0, -12.0] {
        let mut proto = library_to_proto(library()?)?;
        let flop = proto.cells.iter_mut().find(|c| c.name == "DFF1").unwrap();
        let d = flop
            .pins
            .iter_mut()
            .find(|p| p.direction == PinDirection::Input as i32 && !p.is_clocking_pin)
            .unwrap();
        for table in &mut d.timing_arcs[0].tables {
            table.values.fill(setup);
        }
        let lib = library_from_proto(proto)?;
        let mut netlist = capture_broadcast()?;
        let text =
            |n: &ParsedNetlist| emit_module_as_netlist_text(&n.modules[0], &n.nets, &n.interner);
        let before = text(&netlist)?;
        let r = repair_registered_fanout(&mut netlist, &lib, StaOptions::default(), None)?;
        let delay = r.initial.unwrap().register_to_register_delay;
        if setup == -4.0 {
            assert!(delay > 0.0);
            assert!(r.trials.iter().any(|t| {
                t.score.is_some_and(|s| s.register_to_register_delay < 0.0)
                    && t.disposition.contains("nonpositive")
            }));
        } else {
            assert!(delay < 0.0);
            assert!(r.trials.is_empty());
            assert!(r.outcome.contains("nonpositive"));
        }
        if r.accepted_trial.is_none() {
            assert_eq!(text(&netlist)?, before);
        }
    }
    Ok(())
}

// An unconnected complementary register output must remain unconnected after
// buffering.
#[test]
fn open_complementary_outputs_survive_buffering() -> Result<()> {
    let lib = crate::netlist::resize::tests::with_complementary_outputs(library()?)?;
    let mut netlist = capture_broadcast()?;
    let pin = netlist.interner.get_or_intern("QN");
    let open = (pin, NetRef::Unconnected);
    for inst in &mut netlist.modules[0].instances {
        inst.connections.push(open.clone());
    }
    let report = repair_registered_fanout(&mut netlist, &lib, StaOptions::default(), None)?;
    assert!(report.accepted_trial.is_some(), "{report:?}");
    for inst in &netlist.modules[0].instances[..17] {
        assert!(inst.connections.contains(&open));
    }
    // Skipping an open output for timing must preserve duplicate detection.
    netlist.modules[0].instances[0].connections.push(open);
    assert!(registered_delay_and_area(&netlist, &lib).is_err());
    Ok(())
}

#[test]
fn zero_fanout_limit_requires_a_zero_weight_buffer_input() -> Result<()> {
    for weight in [0.0, 0.001] {
        let mut lib = library()?;
        for cell in &mut lib.cells {
            for pin in &mut cell.pins {
                if cell.name == "DFF1" && pin.direction == PinDirection::Output as i32 {
                    pin.max_capacitance = None; // isolate the weighted-fanout trigger
                    pin.max_fanout = Some(0.0);
                } else if cell.name == "BUF4" {
                    pin.fanout_load = Some(weight);
                }
            }
        }
        let mut netlist = capture_broadcast()?;
        let r = repair_registered_fanout(&mut netlist, &lib, StaOptions::default(), None)?;
        assert_eq!(r.candidate_roots, 1);
        assert_eq!(r.accepted_trial.is_some(), weight == 0.0);
    }
    Ok(())
}

#[test]
fn exact_leaf_capacity_survives_roundoff_but_real_excess_splits_leaves() -> Result<()> {
    // Each eight-pin leaf presents 1.6. Only a real excess should split it.
    for (limit, minimum_buffers) in [(1.6 - 2.0 * f64::EPSILON, 2), (1.599, 4)] {
        let mut lib = library()?;
        let buffer = lib.cells.iter_mut().find(|c| c.name == "BUF4").unwrap();
        for pin in &mut buffer.pins {
            if pin.direction == PinDirection::Output as i32 {
                pin.max_capacitance = Some(limit);
            }
        }
        let mut netlist = capture_broadcast()?;
        let r = repair_registered_fanout(&mut netlist, &lib, StaOptions::default(), None)?;
        assert_eq!(
            r.trials.iter().map(|t| t.buffers_added).min(),
            Some(minimum_buffers)
        );
    }
    Ok(())
}

// An inverted state variable does not turn the flip-flop's physical D pin into
// a control input. Protecting that pin would wrongly exclude every launch in
// its data cone from buffering.
#[test]
fn inverted_next_state_still_identifies_the_capture_data_pin() -> Result<()> {
    let mut lib = library()?;
    lib.cells
        .iter_mut()
        .find(|cell| cell.name == "DFF1")
        .unwrap()
        .sequential[0]
        .next_state = "!D".into();
    let mut netlist = capture_broadcast()?;
    let report = repair_registered_fanout(&mut netlist, &lib, StaOptions::default(), None)?;
    assert_eq!(report.shared_outputs, 1);
    assert_eq!(report.candidate_roots, 1);
    assert!(report.accepted_trial.is_some());
    Ok(())
}

#[test]
fn a_generated_clock_with_data_fanout_keeps_its_original_driver_and_aliases() -> Result<()> {
    let mut text = String::from(
        "module test(input clk, input d, output tap, output held, output [31:0] lanes);
        wire q; wire alias_q; assign alias_q=q; assign tap=alias_q;
        DFF1 launch(.CLK(clk),.D(d),.Q(q));
        DFF1 guard(.CLK(alias_q),.D(1'b0),.Q(held));",
    );
    for lane in 0..32 {
        writeln!(
            text,
            "DFF1 sink_{lane}(.CLK(clk),.D(alias_q),.Q(lanes[{lane}]));"
        )?;
    }
    text.push_str("endmodule");
    let lib = library()?;
    let mut netlist = read_gv_from_str(&text)?;
    let before =
        emit_module_as_netlist_text(&netlist.modules[0], &netlist.nets, &netlist.interner)?;
    let report = repair_registered_fanout(&mut netlist, &lib, StaOptions::default(), None)?;
    assert!(report.accepted_trial.is_none());
    let after = emit_module_as_netlist_text(&netlist.modules[0], &netlist.nets, &netlist.interner)?;
    assert_eq!(before, after);
    Ok(())
}

#[test]
fn equally_critical_launches_can_improve_the_global_register_path_together() -> Result<()> {
    let roots = 36;
    let lanes = 16;
    let mut text = format!(
        "module fabric(input clk, input [{msb}:0] d, output [{out_msb}:0] y);",
        msb = roots - 1,
        out_msb = roots * lanes - 1
    );
    for root in 0..roots {
        writeln!(
            text,
            "wire q{root}; DFF1 root{root}(.CLK(clk),.D(d[{root}]),.Q(q{root}));"
        )?;
        for lane in 0..lanes {
            let output = root * lanes + lane;
            writeln!(
                text,
                "DFF1 cap{output}(.CLK(clk),.D(q{root}),.Q(y[{output}]));"
            )?;
        }
    }
    text.push_str("endmodule");
    let mut parsed = read_gv_from_str(&text)?;
    let lib = library()?;
    let (base_delay, base_area) = registered_delay_and_area(&parsed, &lib)?;
    let report = repair_registered_fanout(&mut parsed, &lib, StaOptions::default(), None)?;
    assert_eq!(report.candidate_roots, roots);
    let (delay, area) = registered_delay_and_area(&parsed, &lib)?;
    assert!(
        delay < base_delay,
        "worst R→R {base_delay} -> {delay} for {roots} tied launches"
    );
    assert!(delay * area < base_delay * base_area);
    Ok(())
}

#[test]
fn an_infeasible_tree_leaves_only_its_driver_without_after_measurements() -> Result<()> {
    let mut text = String::from(
        "module fabric(input clk, input d, output pinned, output [25:0] y);
         wire a, b;
         DFF1 wide(.CLK(clk),.D(d),.Q(a));
         DFF1 fixed(.CLK(clk),.D(d),.Q(pinned));
         DFF1 narrow(.CLK(clk),.D(d),.Q(b));",
    );
    for (net, outputs) in [("a", 0..16), ("pinned", 16..18), ("b", 18..26)] {
        for i in outputs {
            writeln!(text, "DFF1 cap{i}(.CLK(clk),.D({net}),.Q(y[{i}]));")?;
        }
    }
    text.push_str("endmodule");
    // The output port alone uses the fixed driver's entire load allowance.
    let timing = StaOptions {
        module_output_load: 0.8,
        ..StaOptions::default()
    };
    let mut parsed = read_gv_from_str(&text)?;
    let report = repair_registered_fanout(&mut parsed, &library()?, timing, None)?;
    assert_eq!(report.accepted_trial, Some(0), "{report:?}");
    let drivers: Vec<_> = report
        .overloaded
        .iter()
        .map(|driver| {
            (
                driver.instance.as_str(),
                driver.after.as_ref().map(|after| after.data_sink_count),
            )
        })
        .collect();
    assert_eq!(
        drivers,
        [("wide", Some(2)), ("fixed", None), ("narrow", Some(1))]
    );
    Ok(())
}
