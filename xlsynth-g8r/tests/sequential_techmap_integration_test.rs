// SPDX-License-Identifier: Apache-2.0

//! Cycle-accurate coverage of final-only, register-aware technology mapping.

use std::collections::{BTreeMap, BTreeSet};

use xlsynth::IrBits;
use xlsynth_g8r::aig::{
    ChoiceAig, ClockPort, RegisterBinding, SequentialGateFn, TransitionInputId, TransitionOutputId,
};
use xlsynth_g8r::aig_serdes::emit_aiger::emit_aiger;
use xlsynth_g8r::aig_serdes::load_abc_choice_aiger::load_abc_choice_aiger_auto;
use xlsynth_g8r::aig_sim::sequential::{SequentialState, simulate};
use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};
use xlsynth_g8r::liberty_model::{
    Cell, Library, LibraryBuilder, Pin, PinDirection, Sequential, SequentialKind, TimingArc,
    TimingTable,
};
use xlsynth_g8r::liberty_proto::TimingTableKind;
use xlsynth_g8r::netlist::buffer::BufferOptions;
use xlsynth_g8r::netlist::gatefn_from_netlist::project_labeled_sequential_netlist_aig;
use xlsynth_g8r::netlist::parse::PortDirection;
use xlsynth_g8r::netlist::report::build_netlist_report;
use xlsynth_g8r::netlist::resize::ResizeOptions;
use xlsynth_g8r::netlist::sta::StaOptions;
use xlsynth_g8r::techmap::{
    SequentialTechMapConstraints, TechMapOptions, map_sequential_choice_aig_to_netlist,
};

/// Builds a scalar, fully characterized synthetic Liberty timing table.
fn timing_table(builder: &mut LibraryBuilder, kind: TimingTableKind, value: f64) -> TimingTable {
    builder
        .add_timing_table_f64(kind, 0, vec![], vec![], vec![], vec![value], vec![], "")
        .expect("construct scalar timing table")
}

/// Constructs the complete rise/fall timing for one combinational input.
fn combinational_arc(builder: &mut LibraryBuilder, pin: &str, delay: f64) -> TimingArc {
    let tables = vec![
        timing_table(builder, TimingTableKind::CellRise, delay),
        timing_table(builder, TimingTableKind::CellFall, delay),
        timing_table(builder, TimingTableKind::RiseTransition, 0.1),
        timing_table(builder, TimingTableKind::FallTransition, 0.1),
    ];
    builder
        .add_timing_arc(pin, "negative_unate", "combinational", "", tables)
        .expect("construct combinational timing arc")
}

/// Constructs a positive-edge clock-to-output Liberty timing arc.
fn clock_to_output_arc(builder: &mut LibraryBuilder) -> TimingArc {
    let tables = vec![
        timing_table(builder, TimingTableKind::CellRise, 0.5),
        timing_table(builder, TimingTableKind::CellFall, 0.5),
        timing_table(builder, TimingTableKind::RiseTransition, 0.1),
        timing_table(builder, TimingTableKind::FallTransition, 0.1),
    ];
    builder
        .add_timing_arc("CLK", "non_unate", "rising_edge", "", tables)
        .expect("construct flip-flop clock-to-output arc")
}

/// Constructs a positive-edge Liberty setup constraint on the data input.
fn setup_arc(builder: &mut LibraryBuilder) -> TimingArc {
    let tables = vec![
        timing_table(builder, TimingTableKind::RiseConstraint, 0.25),
        timing_table(builder, TimingTableKind::FallConstraint, 0.25),
    ];
    builder
        .add_timing_arc("CLK", "", "setup_rising", "", tables)
        .expect("construct flip-flop setup constraint")
}

/// Builds a functionally complete Liberty library with either Q or QN FFs.
fn sequential_library(complemented_output: bool) -> Library {
    let mut builder = LibraryBuilder::new();
    let a = builder.intern_string("A").expect("intern A");
    let b = builder.intern_string("B").expect("intern B");
    let y = builder.intern_string("Y").expect("intern Y");
    let d = builder.intern_string("D").expect("intern D");
    let clk = builder.intern_string("CLK").expect("intern CLK");
    let not_a = builder.intern_string("!A").expect("intern inversion");
    let nand = builder
        .intern_string("!(A*B)")
        .expect("intern NAND function");

    let (ff_name, output_name, state_var, complementary_state_var, next_state) =
        if complemented_output {
            ("DFF_QN", "QN", "IQN", Some("IQNN"), "!D")
        } else {
            ("DFF", "Q", "IQ", Some("IQN"), "D")
        };
    let ff_output = builder
        .intern_string(output_name)
        .expect("intern flip-flop output");
    let ff_function = builder
        .intern_string(state_var)
        .expect("intern flip-flop state function");

    let inv_a = combinational_arc(&mut builder, "A", 1.0);
    let nand_a = combinational_arc(&mut builder, "A", 2.0);
    let nand_b = combinational_arc(&mut builder, "B", 2.0);
    let ff_setup = setup_arc(&mut builder);
    let ff_clock_to_output = clock_to_output_arc(&mut builder);

    builder.cells = vec![
        Cell {
            name: "INV".to_string(),
            pins: vec![
                Pin {
                    name: a,
                    direction: PinDirection::Input as i32,
                    capacitance: Some(0.02),
                    ..Default::default()
                },
                Pin {
                    name: y,
                    direction: PinDirection::Output as i32,
                    function: not_a,
                    timing_arcs: vec![inv_a],
                    ..Default::default()
                },
            ],
            area: 1.0,
            ..Default::default()
        },
        Cell {
            name: "NAND2".to_string(),
            pins: vec![
                Pin {
                    name: a,
                    direction: PinDirection::Input as i32,
                    capacitance: Some(0.02),
                    ..Default::default()
                },
                Pin {
                    name: b,
                    direction: PinDirection::Input as i32,
                    capacitance: Some(0.02),
                    ..Default::default()
                },
                Pin {
                    name: y,
                    direction: PinDirection::Output as i32,
                    function: nand,
                    timing_arcs: vec![nand_a, nand_b],
                    ..Default::default()
                },
            ],
            area: 2.0,
            ..Default::default()
        },
        Cell {
            name: ff_name.to_string(),
            pins: vec![
                Pin {
                    name: d,
                    direction: PinDirection::Input as i32,
                    capacitance: Some(0.02),
                    timing_arcs: vec![ff_setup],
                    ..Default::default()
                },
                Pin {
                    name: clk,
                    direction: PinDirection::Input as i32,
                    is_clocking_pin: true,
                    capacitance: Some(0.01),
                    ..Default::default()
                },
                Pin {
                    name: ff_output,
                    direction: PinDirection::Output as i32,
                    function: ff_function,
                    timing_arcs: vec![ff_clock_to_output],
                    ..Default::default()
                },
            ],
            area: 4.0,
            sequential: vec![Sequential {
                state_var: state_var.to_string(),
                complementary_state_var: complementary_state_var.map(str::to_string),
                next_state: next_state.to_string(),
                clock_expr: "CLK".to_string(),
                kind: SequentialKind::Ff as i32,
                ..Default::default()
            }],
            ..Default::default()
        },
    ];
    builder.finish()
}

/// Builds a three-bit transition with synchronous reset and load enable.
fn multibit_synchronous_design() -> SequentialGateFn {
    let mut builder = GateBuilder::new(
        "pipeline__transition".to_string(),
        GateBuilderOptions::no_opt(),
    );
    let data = builder.add_input("data".to_string(), 3);
    let enable = builder.add_input("enable".to_string(), 1);
    let reset = builder.add_input("reset".to_string(), 1);
    let state_q = builder.add_input("state__q".to_string(), 3);
    let output = builder.add_not_vec(&state_q);
    let enabled_next_state = builder.add_mux2_vec(enable.get_lsb(0), &data, &state_q);
    let reset_value =
        builder.add_literal(&IrBits::make_ubits(3, 0).expect("construct synchronous reset value"));
    let next_state = builder.add_mux2_vec(reset.get_lsb(0), &reset_value, &enabled_next_state);
    builder.add_output("out".to_string(), output);
    builder.add_output("state__d".to_string(), next_state);

    SequentialGateFn::new(
        "pipeline".to_string(),
        builder.build(),
        vec![
            TransitionInputId::new(0),
            TransitionInputId::new(1),
            TransitionInputId::new(2),
        ],
        vec![TransitionOutputId::new(0)],
        Some(ClockPort {
            name: "clk".to_string(),
        }),
        vec![RegisterBinding {
            name: "state".to_string(),
            q: TransitionInputId::new(3),
            d: TransitionOutputId::new(1),
            initial_value: None,
        }],
    )
    .expect("construct multi-bit sequential transition")
}

/// Round-trips a transition through AIGER to reproduce ABC's flat interface.
fn flattened_choice_transition(design: &SequentialGateFn) -> ChoiceAig {
    let transition = emit_aiger(&design.transition, true).expect("emit transition AIGER");
    load_abc_choice_aiger_auto(transition.as_bytes())
        .expect("reload flattened, choice-compatible transition AIGER")
}

/// Checks mapping, restored bus ports, exact STA, and cycle-accurate behavior.
fn assert_multibit_mapping(complemented_output: bool) {
    let design = multibit_synchronous_design();
    let library = sequential_library(complemented_output);
    let choices = flattened_choice_transition(&design);

    let mapped = map_sequential_choice_aig_to_netlist(
        &design,
        &choices,
        &library,
        &SequentialTechMapConstraints::default(),
        &TechMapOptions::default(),
    )
    .expect("map physical flip-flops and multi-bit transition");

    let ports = mapped
        .module
        .ports
        .iter()
        .map(|port| {
            (
                mapped
                    .interner
                    .resolve(port.name)
                    .expect("resolve mapped port")
                    .to_string(),
                port.width,
            )
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        ports,
        BTreeSet::from([
            ("clk".to_string(), None),
            ("data".to_string(), Some((2, 0))),
            ("enable".to_string(), None),
            ("out".to_string(), Some((2, 0))),
            ("reset".to_string(), None),
        ]),
        "the final module must restore external buses and hide transition Q/D ports"
    );
    assert_eq!(
        mapped
            .module
            .ports
            .iter()
            .filter(|port| port.direction == PortDirection::Output)
            .count(),
        1
    );

    let expected_ff_name = if complemented_output { "DFF_QN" } else { "DFF" };
    let physical_flip_flops = mapped
        .module
        .instances
        .iter()
        .filter(|instance| mapped.interner.resolve(instance.type_name) == Some(expected_ff_name))
        .count();
    assert_eq!(physical_flip_flops, 3);
    assert_eq!(mapped.stats.sequential_instance_count, 3);
    assert_eq!(mapped.stats.sequential_area, 12.0);

    let report = build_netlist_report(
        &mapped.module,
        &mapped.nets,
        &mapped.interner,
        &library,
        StaOptions::default(),
    )
    .expect("analyze complete mapped sequential netlist");
    assert_eq!(report.sequential_cell_area, 12.0);
    assert_eq!(mapped.stats.selected_area, report.cell_area);
    assert_eq!(
        mapped.stats.worst_input_to_register_arrival,
        report.max_input_to_register_delay
    );
    assert_eq!(
        mapped.stats.worst_register_to_register_arrival,
        report.max_register_to_register_delay
    );
    assert_eq!(
        mapped.stats.worst_register_to_output_arrival,
        report.max_register_to_output_delay
    );

    let projected = project_labeled_sequential_netlist_aig(
        &mapped.module,
        &mapped.nets,
        &mapped.interner,
        &library,
        Some("clk"),
    )
    .expect("project physical flip-flop netlist into a sequential transition");
    let samples = [
        (5, 1, 0),
        (3, 0, 0),
        (6, 1, 0),
        (1, 1, 1),
        (7, 0, 0),
        (2, 1, 0),
        (4, 1, 0),
    ];
    let inputs = samples
        .into_iter()
        .map(|(data, enable, reset)| {
            vec![
                IrBits::make_ubits(3, data).expect("construct data stimulus"),
                IrBits::make_ubits(1, enable).expect("construct enable stimulus"),
                IrBits::make_ubits(1, reset).expect("construct reset stimulus"),
            ]
        })
        .collect::<Vec<_>>();
    let expected = simulate(&design, &inputs, SequentialState::all_zeros(&design))
        .expect("simulate original sequential transition");
    let actual = simulate(
        &projected.sequential_gate_fn,
        &inputs,
        SequentialState::all_zeros(&projected.sequential_gate_fn),
    )
    .expect("simulate projected physical flip-flop netlist");
    assert_eq!(actual.external_outputs(), expected.external_outputs());
}

#[test]
fn sequential_mapping_preserves_multibit_buses_and_cycle_behavior() {
    assert_multibit_mapping(false);
}

#[test]
fn complemented_output_flip_flops_preserve_multibit_cycle_behavior() {
    assert_multibit_mapping(true);
}

#[test]
fn sequential_mapping_rejects_unrepresentable_explicit_initial_state() {
    let mut design = multibit_synchronous_design();
    design.registers[0].initial_value =
        Some(IrBits::make_ubits(3, 0).expect("construct explicit initial state"));
    let choices = flattened_choice_transition(&design);

    let error = map_sequential_choice_aig_to_netlist(
        &design,
        &choices,
        &sequential_library(false),
        &SequentialTechMapConstraints::default(),
        &TechMapOptions::default(),
    )
    .expect_err("a physical flip-flop cannot represent an explicit initial state");
    let diagnostic = error.to_string();
    assert!(
        diagnostic.contains("initial") && diagnostic.contains("state"),
        "unexpected explicit-initial-state diagnostic: {diagnostic}"
    );
}

#[test]
fn sequential_mapping_rejects_combinational_only_buffer_insertion() {
    let design = multibit_synchronous_design();
    let error = map_sequential_choice_aig_to_netlist(
        &design,
        &flattened_choice_transition(&design),
        &sequential_library(false),
        &SequentialTechMapConstraints::default(),
        &TechMapOptions {
            buffer_options: Some(BufferOptions::default()),
            ..TechMapOptions::default()
        },
    )
    .expect_err("register-unaware buffer insertion must be rejected");
    assert!(error.to_string().contains("buffer"));
}

#[test]
fn sequential_mapping_rejects_combinational_only_cell_resizing() {
    let design = multibit_synchronous_design();
    let error = map_sequential_choice_aig_to_netlist(
        &design,
        &flattened_choice_transition(&design),
        &sequential_library(false),
        &SequentialTechMapConstraints::default(),
        &TechMapOptions {
            resize_options: Some(ResizeOptions::default()),
            ..TechMapOptions::default()
        },
    )
    .expect_err("register-unaware cell resizing must be rejected");
    assert!(error.to_string().contains("resize"));
}

#[test]
fn sequential_mapping_rejects_negative_edge_flip_flops() {
    let design = multibit_synchronous_design();
    let mut library = sequential_library(false);
    library
        .cells
        .iter_mut()
        .find(|cell| cell.name == "DFF")
        .expect("find physical flip-flop")
        .sequential[0]
        .clock_expr = "!CLK".to_string();
    let error = map_sequential_choice_aig_to_netlist(
        &design,
        &flattened_choice_transition(&design),
        &library,
        &SequentialTechMapConstraints::default(),
        &TechMapOptions::default(),
    )
    .expect_err("negative-edge flip-flops are outside the initial supported subset");
    assert!(error.to_string().contains("negative clock edge"));
}

#[test]
fn sequential_mapping_rejects_asynchronously_reset_flip_flops() {
    let design = multibit_synchronous_design();
    let mut library = sequential_library(false);
    library
        .cells
        .iter_mut()
        .find(|cell| cell.name == "DFF")
        .expect("find physical flip-flop")
        .sequential[0]
        .clear_expr = "CLR".to_string();
    let error = map_sequential_choice_aig_to_netlist(
        &design,
        &flattened_choice_transition(&design),
        &library,
        &SequentialTechMapConstraints::default(),
        &TechMapOptions::default(),
    )
    .expect_err("asynchronously reset flip-flops are outside the supported subset");
    assert!(error.to_string().contains("asynchronous reset"));
}

#[test]
fn sequential_mapping_rejects_latches() {
    let design = multibit_synchronous_design();
    let mut library = sequential_library(false);
    library
        .cells
        .iter_mut()
        .find(|cell| cell.name == "DFF")
        .expect("find physical flip-flop")
        .sequential[0]
        .kind = SequentialKind::Latch as i32;
    let error = map_sequential_choice_aig_to_netlist(
        &design,
        &flattened_choice_transition(&design),
        &library,
        &SequentialTechMapConstraints::default(),
        &TechMapOptions::default(),
    )
    .expect_err("latches are outside the initial supported subset");
    assert!(error.to_string().contains("unsupported sequential kind"));
}

#[test]
fn sequential_mapping_enforces_flip_flop_setup_constrained_clock_period() {
    let design = multibit_synchronous_design();
    let choices = flattened_choice_transition(&design);
    let library = sequential_library(false);

    let mapped = map_sequential_choice_aig_to_netlist(
        &design,
        &choices,
        &library,
        &SequentialTechMapConstraints {
            clock_period: Some(30.0),
            ..SequentialTechMapConstraints::default()
        },
        &TechMapOptions::default(),
    )
    .expect("map transition under a feasible synchronous clock period");
    assert_eq!(mapped.stats.clock_period, Some(30.0));
    assert!(
        mapped
            .stats
            .worst_register_slack
            .is_some_and(|slack| slack.is_finite() && slack >= 0.0)
    );

    let error = map_sequential_choice_aig_to_netlist(
        &design,
        &choices,
        &library,
        &SequentialTechMapConstraints {
            clock_period: Some(0.5),
            ..SequentialTechMapConstraints::default()
        },
        &TechMapOptions::default(),
    )
    .expect_err("a period shorter than the captured logic path is infeasible");
    let diagnostic = format!("{error:#}");
    assert!(
        diagnostic.contains("required time") || diagnostic.contains("required"),
        "unexpected infeasible-clock-period diagnostic: {diagnostic}"
    );
}

#[test]
fn sequential_mapping_reports_exact_nonzero_primary_input_arrivals() {
    let design = multibit_synchronous_design();
    let library = sequential_library(false);
    let mapped = map_sequential_choice_aig_to_netlist(
        &design,
        &flattened_choice_transition(&design),
        &library,
        &SequentialTechMapConstraints {
            primary_input_arrivals: BTreeMap::from([
                ("data_0".to_string(), 5.0),
                ("data_1".to_string(), 5.0),
                ("data_2".to_string(), 5.0),
                ("enable".to_string(), 5.0),
                ("reset".to_string(), 5.0),
            ]),
            ..SequentialTechMapConstraints::default()
        },
        &TechMapOptions::default(),
    )
    .expect("map transition with nonzero external launch arrivals");
    let zero_arrival_report = build_netlist_report(
        &mapped.module,
        &mapped.nets,
        &mapped.interner,
        &library,
        StaOptions::default(),
    )
    .expect("analyze equivalent zero-arrival physical netlist");

    let zero_arrival = zero_arrival_report
        .max_input_to_register_delay
        .expect("zero-arrival input-to-register timing");
    let constrained_arrival = mapped
        .stats
        .worst_input_to_register_arrival
        .expect("constrained input-to-register timing");
    assert!(
        (constrained_arrival - (zero_arrival + 5.0)).abs() < 1e-9,
        "exact capture timing must include primary-input arrival: \
         zero={zero_arrival}, constrained={constrained_arrival}"
    );
}

#[test]
fn sequential_mapping_rejects_negative_external_timing_constraints() {
    let design = multibit_synchronous_design();
    let choices = flattened_choice_transition(&design);
    let library = sequential_library(false);

    for constraints in [
        SequentialTechMapConstraints {
            primary_input_arrivals: BTreeMap::from([("data_0".to_string(), -1.0)]),
            ..SequentialTechMapConstraints::default()
        },
        SequentialTechMapConstraints {
            primary_output_required: BTreeMap::from([("out_0".to_string(), -1.0)]),
            ..SequentialTechMapConstraints::default()
        },
    ] {
        let error = map_sequential_choice_aig_to_netlist(
            &design,
            &choices,
            &library,
            &constraints,
            &TechMapOptions::default(),
        )
        .expect_err("negative external endpoint timing must be rejected");
        assert!(
            error.to_string().contains("non-negative"),
            "unexpected negative-endpoint diagnostic: {error:#}"
        );
    }
}
