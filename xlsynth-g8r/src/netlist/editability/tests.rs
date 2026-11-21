// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::netlist::io::read_gv_from_str;

/// A shared source and its clock-facing tap carry control-fan-in facts;
/// the separate ordinary data tap remains distinguishable.
#[test]
fn control_fanin_protects_shared_data_connections() -> Result<()> {
    let lib = crate::netlist::resize::tests::library()?;
    let mut p = read_gv_from_str(
        "module top(input clk, input d, output y); wire q, gclk, data; DFF1 launch(.CLK(clk),.D(d),.Q(q)); BUF1 clockbuf(.A(q),.Y(gclk)); BUF1 databuf(.A(q),.Y(data)); DFF1 capture(.CLK(gclk),.D(data),.Q(y)); endmodule",
    )?;
    let mapped = MappedNetlist {
        module: p.modules.remove(0),
        nets: p.nets,
        interner: p.interner,
    };
    let report = analyze_editable_outputs(&mapped, &lib, 1.0, None)?;
    assert_eq!(report.launches.len(), 2);
    let launch = report
        .roots
        .iter()
        .find(|r| r.binding.instance.0 == 0)
        .unwrap();
    assert!(launch.in_control_fanin);
    assert!(launch.sinks.iter().any(|p| p.binding.instance.0 == 1));
    assert!(launch.sinks.iter().any(|p| p.binding.instance.0 == 2));
    Ok(())
}

/// Worked example for an optimizer: assemble a register boundary, discover
/// editable taps, measure the same physical Q with STA, and assess its rules.
///
///     input -> launch FF --Q--> capture FF -> output
///                         ^
///                  measured physical load
#[test]
fn assembled_stage_combines_editability_sta_and_electrical_rules() -> Result<()> {
    use crate::netlist::sta::{PreparedStaLibrary, StaOptions};

    let mut lib = crate::netlist::resize::tests::library()?;
    let q = lib
        .cells
        .iter_mut()
        .find(|c| c.name == "DFF1")
        .unwrap()
        .pins
        .iter_mut()
        .find(|p| p.direction == PinDirection::Output as i32)
        .unwrap();
    q.max_capacitance = Some(0.1);
    let (prepared, mut parsed) = stage_cover(&lib)?;
    let registers = prepared
        .assemble(&mut parsed, "native")
        .map_err(anyhow::Error::msg)?;
    let netlist = mapped(parsed);
    let graph = analyze_editable_outputs(&netlist, &lib, 0.0, Some(registers))?;
    assert_eq!(graph.launches, registers.instances().collect::<Vec<_>>());
    let root = graph.roots.iter().find(|r| !r.sinks.is_empty()).unwrap();
    assert_eq!(root.sinks.len(), 1);
    assert_eq!(root.electrical.load.capacitance, [0.2, 0.2]);
    assert_eq!(root.electrical.characterization.max_load, Some(10.0));
    assert_eq!(
        registers
            .initial_mapping(root.binding.instance.0)
            .unwrap()
            .cell_name,
        "DFF1"
    );

    let (_, slews) = PreparedStaLibrary::new(&lib)?.analyze_register_boundary_with_slews(
        &netlist.module,
        &netlist.nets,
        &netlist.interner,
        StaOptions::default(),
        false,
        &graph.launches,
        &[root.source_bit],
    )?;
    let slew = slews[0].expect("STA measured Q slew");
    let cap = root
        .electrical
        .load
        .capacitance
        .into_iter()
        .fold(0.0, f64::max);
    assert_eq!(
        root.electrical
            .limits
            .violations_known(cap, slew, root.electrical.load.weighted_fanout)
            .capacitance,
        0.1
    );
    Ok(())
}

/// Rewiring a tap changes the subject graph. Rebuild snapshot analyses and
/// remeasure STA; an old owned report continues to describe the old graph.
#[test]
fn rebuilding_after_a_rewire_updates_exact_pins_load_and_timing() -> Result<()> {
    use crate::netlist::connectivity::exact::ExactConnectivity;
    use crate::netlist::sta::{PreparedStaLibrary, StaOptions};
    use xlsynth_pir::IrBits;

    let lib = crate::netlist::resize::tests::library()?;
    let (prepared, mut parsed) = stage_cover(&lib)?;
    let registers = prepared
        .assemble(&mut parsed, "native")
        .map_err(anyhow::Error::msg)?;
    let mut netlist = mapped(parsed);
    let before = analyze_editable_outputs(&netlist, &lib, 0.0, Some(registers))?;
    let root = before.roots.iter().find(|r| !r.sinks.is_empty()).unwrap();
    let sink = root.sinks[0].binding;
    let probe = root.source_bit;
    let sta = PreparedStaLibrary::new(&lib)?;
    let time = |n: &MappedNetlist| {
        sta.analyze_register_boundary_with_slews(
            &n.module,
            &n.nets,
            &n.interner,
            StaOptions::default(),
            false,
            &before.launches,
            &[probe],
        )
    };
    let (initial, slew_before) = time(&netlist)?;
    assert!(initial.worst_register_input_breakdown.is_some());
    assert_eq!(indexed_sink_count(&netlist, &lib, probe)?, 1);

    netlist.module.instances[sink.instance.0].connections[sink.connection].1 =
        NetRef::Literal(IrBits::make_ubits(1, 0).map_err(anyhow::Error::msg)?);
    let after = analyze_editable_outputs(&netlist, &lib, 0.0, Some(registers))?;
    let changed = after
        .roots
        .iter()
        .find(|r| r.binding == root.binding)
        .unwrap();
    assert!(changed.sinks.is_empty());
    assert_eq!(changed.electrical.load.capacitance, [0.0; 2]);
    assert_eq!(indexed_sink_count(&netlist, &lib, probe)?, 0);
    let (updated, slew_after) = time(&netlist)?;
    assert!(updated.worst_register_input_breakdown.is_none());
    assert!(slew_after[0].unwrap() < slew_before[0].unwrap());
    assert_eq!(
        root.sinks.len(),
        1,
        "the original report is only a snapshot"
    );
    // Constructing the exact view never changes the directed timing graph.
    let normalized =
        NormalizedNetlistModule::new(&netlist.module, &netlist.nets, &netlist.interner)?;
    let _exact = ExactConnectivity::new(&normalized);
    assert_eq!(time(&netlist)?.0, updated);
    Ok(())
}

/// An output alias contributes physical load to Q while preserving directed
/// value flow. Retargeting the assign moves both the load and the timing path.
#[test]
fn retargeting_an_output_alias_updates_load_and_launch_class() -> Result<()> {
    use crate::netlist::parse::{AssignExpr, NetIndex};
    use crate::netlist::sta::{PreparedStaLibrary, StaOptions};
    let lib = crate::netlist::resize::tests::library()?;
    let mut n = mapped(read_gv_from_str(
        "module top(input clk,input d,output tap,output y); wire q; assign tap=q; DFF1 launch(.CLK(clk),.D(d),.Q(q)); DFF1 capture(.CLK(clk),.D(tap),.Q(y)); endmodule",
    )?);
    let analyze = |n: &MappedNetlist| analyze_editable_outputs(n, &lib, 0.5, None);
    let before = analyze(&n)?;
    let q = first_driver(&before);
    assert_eq!(q.electrical.load.output_ports, 1);
    assert_eq!(q.fixed_load, [0.5; 2]);
    assert_eq!(q.electrical.load.capacitance, [0.7; 2]);
    let sta = PreparedStaLibrary::new(&lib)?;
    let time = |n: &MappedNetlist| {
        sta.analyze_register_boundary_max_arrival(
            &n.module,
            &n.nets,
            &n.interner,
            StaOptions {
                module_output_load: 0.5,
                ..Default::default()
            },
            false,
            &before.launches,
        )
    };
    assert!(time(&n)?.worst_register_input_breakdown.is_some());
    let d = NetIndex(
        n.nets
            .iter()
            .position(|net| n.interner.resolve(net.name) == Some("d"))
            .unwrap(),
    );
    n.module.assigns[0].rhs = AssignExpr::Leaf(NetRef::Simple(d));
    let after = analyze(&n)?;
    assert_eq!(first_driver(&after).electrical.load.capacitance, [0.0; 2]);
    assert_eq!(first_driver(&after).electrical.load.output_ports, 0);
    assert!(time(&n)?.worst_register_input_breakdown.is_none());
    Ok(())
}

/// Assembly indices survive appending instances. A different master, missing
/// instance, or reordered binding invalidates the assembly metadata; consumers
/// must rediscover the live netlist instead of trusting the initial plan.
#[test]
fn assembly_metadata_detects_invalid_graph_edits_and_allows_appends() -> Result<()> {
    let lib = crate::netlist::resize::tests::library()?;
    let (prepared, mut parsed) = stage_cover(&lib)?;
    let registers = prepared
        .assemble(&mut parsed, "native")
        .map_err(anyhow::Error::msg)?;
    let original = mapped(parsed);
    let analyze = |n: &MappedNetlist, a| analyze_editable_outputs(n, &lib, 0.0, a);
    let first = registers.instances().start;
    let mut appended = original.clone();
    let mut extra = appended.module.instances[first].clone();
    extra.instance_name = appended.interner.get_or_intern("appended");
    extra.type_name = appended.interner.get_or_intern("BUF1");
    extra.connections = vec![
        (
            appended.interner.get_or_intern("A"),
            extra.connections[2].1.clone(),
        ),
        (appended.interner.get_or_intern("Y"), NetRef::Unconnected),
    ];
    appended.module.instances.push(extra);
    let extra_load = analyze(&appended, Some(registers))?;
    let launch = extra_load
        .roots
        .iter()
        .find(|r| r.binding.instance.0 == first)
        .unwrap();
    assert_eq!(launch.sinks.len(), 2);
    assert_eq!(launch.electrical.load.capacitance, [0.4; 2]);

    let mut with_ff = original.clone();
    let mut extra_ff = with_ff.module.instances[first].clone();
    extra_ff.instance_name = with_ff.interner.get_or_intern("new_register");
    with_ff.module.instances.push(extra_ff);
    assert!(
        analyze(&with_ff, Some(registers))
            .err()
            .unwrap()
            .to_string()
            .contains("outside assembled boundary")
    );

    let mut resized = original.clone();
    resized.module.instances[first].type_name = resized.interner.get_or_intern("DFF4");
    assert!(
        analyze(&resized, Some(registers))
            .err()
            .unwrap()
            .to_string()
            .contains("master changed")
    );
    assert!(
        analyze(&resized, None).is_ok(),
        "fresh discovery uses the resized master"
    );

    let mut removed = original.clone();
    removed.module.instances.remove(first);
    assert!(analyze(&removed, Some(registers)).is_err());
    let mut swapped = original.clone();
    swapped.module.instances.swap(first, first + 1);
    // Reordering invalidates position-based metadata even if both masters
    // happen to match. Fresh discovery must follow the live instances.
    let fresh = analyze(&swapped, None)?;
    assert!(
        fresh
            .roots
            .iter()
            .any(|r| r.binding.instance.0 == first + 1 && !r.sinks.is_empty())
    );
    let mut reordered = original.clone();
    reordered.module.instances[first].connections.swap(0, 1);
    assert!(
        analyze(&reordered, Some(registers))
            .err()
            .unwrap()
            .to_string()
            .contains("binding changed")
    );
    Ok(())
}

/// The same master can capture D at one site and select scan or hold state
/// at the next. Only the site whose ties prove next_state = D can move D.
#[test]
fn tied_controls_select_the_movable_capture_pin_per_instance() -> Result<()> {
    use crate::liberty::model::LibraryBuilder;
    for (next, dynamic, inactive) in [
        (
            "(!SE & D) | (SE & SI)",
            ".SE(x),.EN(1'b1)",
            ".SE(1'b1),.EN(1'b1)",
        ),
        (
            "(EN & D) | (!EN & IQ)",
            ".SE(1'b0),.EN(x)",
            ".SE(1'b0),.EN(1'b0)",
        ),
        (
            "(SE & SI) | (!SE & ((EN & D) | (!EN & IQ)))",
            ".SE(x),.EN(1'b1)",
            ".SE(1'b0),.EN(1'b0)",
        ),
    ] {
        let mut builder = LibraryBuilder::from_library(crate::netlist::resize::tests::library()?);
        let pins = ["SE", "SI", "EN"].map(|name| builder.intern_string(name).unwrap());
        let cell = builder.cells.iter_mut().find(|c| c.name == "DFF1").unwrap();
        cell.sequential[0].next_state = next.into();
        for name in pins {
            cell.pins.push(crate::liberty_model::Pin {
                name,
                direction: PinDirection::Input as i32,
                capacitance: Some(0.1),
                ..Default::default()
            });
        }
        let netlist = mapped(read_gv_from_str(&format!(
            "module top(input clk,input d,input x,output [2:0] y);
                wire q;
                DFF1 launch(.CLK(clk),.D(d),.Q(q),.SI(d),.SE(1'b0),.EN(1'b1));
                DFF1 selected(.CLK(clk),.D(q),.Q(y[0]),.SI(d),.SE(1'b0),.EN(1'b1));
                DFF1 dynamic(.CLK(clk),.D(q),.Q(y[1]),.SI(d),{dynamic});
                DFF1 inactive(.CLK(clk),.D(q),.Q(y[2]),.SI(d),{inactive});
             endmodule"
        ))?);
        let report = analyze_editable_outputs(&netlist, &builder.finish(), 0.0, None)?;
        let root = first_driver(&report);
        assert_eq!(root.sinks.len(), 1, "{next}");
        assert_eq!(root.sinks[0].binding.instance.0, 1, "{next}");
        assert_eq!(root.fixed_sink_count, 2, "{next}");
    }
    Ok(())
}

/// Register setup coverage depends on the actual tied controls at each site,
/// not just on the cell master. A missing fall constraint must keep D fixed.
#[test]
fn tied_setup_conditions_and_incomplete_edges_keep_unpriced_taps_fixed() -> Result<()> {
    use crate::liberty::model::LibraryBuilder;
    let lib = crate::netlist::resize::tests::library()?;
    let mut builder = LibraryBuilder::from_library(lib);
    let se = builder.intern_string("SE")?;
    let cell = builder.cells.iter_mut().find(|c| c.name == "DFF1").unwrap();
    let mut input = cell
        .pins
        .iter()
        .find(|p| p.direction == PinDirection::Input as i32 && !p.is_clocking_pin)
        .unwrap()
        .clone();
    input.name = se;
    input.timing_arcs.clear();
    cell.pins.push(input);
    let data = cell
        .pins
        .iter_mut()
        .find(|p| {
            p.timing_arcs.iter().any(|a| !a.tables.is_empty())
                && p.direction == PinDirection::Input as i32
        })
        .unwrap();
    data.timing_arcs[0].when = se;
    let mut lib = builder.finish();
    let report = |library: &Library, enable: &str| -> Result<EditableOutputs> {
        let netlist = mapped(read_gv_from_str(&format!(
            "module top(input clk,input d,input en,output y); wire q; DFF1 launch(.CLK(clk),.D(d),.SE(1'b1),.Q(q)); DFF1 capture(.CLK(clk),.D(q),.SE({enable}),.Q(y)); endmodule"
        ))?);
        analyze_editable_outputs(&netlist, library, 0.0, None)
    };
    let yes = report(&lib, "1'b1")?;
    assert_eq!(first_driver(&yes).sinks.len(), 1);
    for enable in ["1'b0", "en"] {
        let no = report(&lib, enable)?;
        assert!(first_driver(&no).sinks.is_empty());
        assert_eq!(first_driver(&no).fixed_sink_count, 1);
    }
    let cell = lib.cells.iter_mut().find(|c| c.name == "DFF1").unwrap();
    let data = cell
        .pins
        .iter_mut()
        .find(|p| p.timing_arcs.first().is_some_and(|a| a.when == se))
        .unwrap();
    data.timing_arcs[0]
        .tables
        .retain(|t| t.kind_str() != "fall_constraint");
    assert!(first_driver(&report(&lib, "1'b1")?).sinks.is_empty());
    // A missing falling clock-to-Q slew table makes the launch itself
    // unpriced, rather than merely making its capture tap fixed.
    let cell = lib.cells.iter_mut().find(|c| c.name == "DFF1").unwrap();
    let q = cell
        .pins
        .iter_mut()
        .find(|p| p.direction == PinDirection::Output as i32)
        .unwrap();
    q.timing_arcs[0]
        .tables
        .retain(|t| t.kind_str() != "fall_transition");
    assert!(
        !report(&lib, "1'b1")?
            .roots
            .iter()
            .any(|r| r.binding.instance.0 == 0)
    );
    Ok(())
}

/// An electrical alias cannot disguise multiple physical drivers, duplicate
/// pin bindings, or unknown pins as an editable single-driver signal.
#[test]
fn malformed_alias_groups_never_become_editable_roots() -> Result<()> {
    let lib = crate::netlist::resize::tests::library()?;
    for extra in [
        "DFF1 second(.CLK(clk),.D(d),.Q(alias));",
        "BUF1 bad(.A(alias),.A(q),.Y(y));",
        "BUF1 bad(.P(alias),.Y(y));",
    ] {
        let text = format!(
            "module top(input clk,input d,output y); wire q,alias; assign alias=q; DFF1 launch(.CLK(clk),.D(d),.Q(q)); {extra} endmodule"
        );
        let netlist = mapped(read_gv_from_str(&text)?);
        let result = analyze_editable_outputs(&netlist, &lib, 0.0, None)?;
        assert!(
            !result.roots.iter().any(|root| root.binding.instance.0 == 0),
            "{extra}"
        );
    }
    Ok(())
}

/// A tran joins the two names electrically. With Q as the only driver we
/// can move the capture pin; adding a gate on the alias rules out the root.
#[test]
fn tran_requires_a_unique_driver_even_when_only_register_outputs_are_selected() -> Result<()> {
    let lib = crate::netlist::resize::tests::library()?;
    for extra in ["", "BUF1 comb(.A(d),.Y(alias));"] {
        let netlist = mapped(read_gv_from_str(&format!(
            "module top(input clk, input d, output y);
                wire q, alias;
                tran(q, alias);
                DFF1 launch(.CLK(clk), .D(d), .Q(q));
                DFF1 capture(.CLK(clk), .D(alias), .Q(y));
                {extra}
             endmodule"
        ))?);
        let result = analyze_editable_outputs(&netlist, &lib, 0.0, None)?;
        let launch = result
            .roots
            .iter()
            .find(|root| root.instance_name == "launch");
        if extra.is_empty() {
            let sinks = &launch.expect("Q drives the alias").sinks;
            assert_eq!(sinks.len(), 1);
            assert_eq!(sinks[0].binding.instance.0, 1);
        } else {
            assert!(launch.is_none(), "Q shares the wire with a gate output");
        }
    }
    Ok(())
}

fn stage_cover(
    lib: &Library,
) -> Result<(
    crate::netlist::sequential_mapping::assembly::PreparedSequentialMapping<'_>,
    crate::netlist::io::ParsedNetlist,
)> {
    use crate::aig::{
        AigBitVector, ClockPort, GateBuilder, GateBuilderOptions, SequentialGateFn,
        add_input_registers, add_output_registers,
    };
    use crate::netlist::sequential_mapping::{
        SequentialMappingOptions, assembly::SequentialMappingCatalog,
    };
    use crate::techmap::postprocess::{CoverStrategy, PostMappingOptions, map_and_optimize_cover};
    let mut b = GateBuilder::new("stage".into(), GateBuilderOptions::no_opt());
    let x = b.add_input("x".into(), 1);
    b.add_output(
        "y".into(),
        AigBitVector::from_lsb_is_index_0(&[*x.get_lsb(0)]),
    );
    let clock = ClockPort { name: "clk".into() };
    let launched = add_input_registers(&SequentialGateFn::from_gate_fn(b.build()), clock.clone())
        .map_err(anyhow::Error::msg)?;
    let stage = add_output_registers(&launched, clock).map_err(anyhow::Error::msg)?;
    let prepared = SequentialMappingCatalog::new(lib)
        .map_err(anyhow::Error::msg)?
        .prepare(lib, &stage, &SequentialMappingOptions::default())
        .map_err(anyhow::Error::msg)?;
    let (cover, _) = map_and_optimize_cover(
        prepared.graph(),
        lib,
        CoverStrategy::Area,
        &PostMappingOptions::default(),
    )?;
    Ok((prepared, cover.netlist.into()))
}

fn mapped(mut p: crate::netlist::io::ParsedNetlist) -> MappedNetlist {
    MappedNetlist {
        module: p.modules.remove(0),
        nets: p.nets,
        interner: p.interner,
    }
}

fn indexed_sink_count(
    n: &MappedNetlist,
    lib: &Library,
    bit: crate::netlist::bit_ref::NetBit,
) -> Result<usize> {
    use crate::netlist::connectivity::exact::ExactConnectivity;
    let normalized = NormalizedNetlistModule::new(&n.module, &n.nets, &n.interner)?;
    let exact = ExactConnectivity::new(&normalized);
    let group = exact.group(normalized.bit_index(bit)?);
    let pins = exact.index_pins(&HashSet::from([group]), |id| {
        let inst = &n.module.instances[id.instance.0];
        let cell_name = n.interner.resolve(inst.type_name)?;
        let pin_name = n.interner.resolve(inst.connections[id.connection].0)?;
        let cell = lib.cells.iter().find(|c| c.name == cell_name)?;
        let pin = cell
            .pins
            .iter()
            .find(|p| lib.resolve_string(&p.name) == pin_name)?;
        PinDirection::try_from(pin.direction).ok()
    });
    Ok(pins.signal(group).map_or(0, |p| p.sinks.len()))
}

fn first_driver(report: &EditableOutputs) -> &EditableOutput {
    report
        .roots
        .iter()
        .find(|root| root.binding.instance.0 == 0)
        .unwrap()
}
