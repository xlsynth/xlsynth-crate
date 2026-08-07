// SPDX-License-Identifier: Apache-2.0

//! End-to-end coverage for representative-driver-aware input buffering.

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;

use prost::Message;
use tempfile::TempDir;
use xlsynth_g8r::aig::AigBitVector;
use xlsynth_g8r::aig_serdes::emit_aiger::emit_aiger;
use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};
use xlsynth_g8r::liberty::model::library_to_proto;
use xlsynth_g8r::liberty_model::{Cell, LibraryBuilder, LuTableTemplate, Pin, PinDirection};
use xlsynth_g8r::liberty_proto::{
    BoundaryTimingDefaults, LibraryUnits, LutTemplateKind, LutVariable, TimingTableKind,
};
use xlsynth_g8r::netlist::parse::{NetRef, Parser as NetlistParser, TokenScanner};

const SINK_COUNT: usize = 8;
const MAX_FANOUT: usize = 3;

struct MappingFixture {
    temp_dir: TempDir,
    aiger_path: PathBuf,
    liberty_path: PathBuf,
    mapped_path: PathBuf,
}

/// Adds a complete load-sensitive test cell with scalar input capacitances.
fn add_timing_cell(
    builder: &mut LibraryBuilder,
    name: &str,
    inputs: &[&str],
    function: &str,
    input_capacitance: f64,
    delay_at_zero_load: f64,
    delay_at_unit_load: f64,
) {
    let mut pins = inputs
        .iter()
        .map(|input| Pin {
            direction: PinDirection::Input as i32,
            name: builder.intern_string(input).expect("intern input pin"),
            capacitance: Some(input_capacitance),
            ..Pin::default()
        })
        .collect::<Vec<_>>();
    let arcs = inputs
        .iter()
        .map(|input| {
            let tables = [
                (
                    TimingTableKind::CellRise,
                    vec![delay_at_zero_load, delay_at_unit_load],
                ),
                (
                    TimingTableKind::CellFall,
                    vec![delay_at_zero_load, delay_at_unit_load],
                ),
                (TimingTableKind::RiseTransition, vec![0.05, 0.55]),
                (TimingTableKind::FallTransition, vec![0.05, 0.55]),
            ]
            .into_iter()
            .map(|(kind, values)| {
                builder
                    .add_timing_table_f64(kind, 1, vec![], vec![], vec![], values, vec![2], "")
                    .expect("build load-sensitive timing table")
            })
            .collect();
            let sense = if function.starts_with('!') {
                "negative_unate"
            } else {
                "positive_unate"
            };
            builder
                .add_timing_arc(input, sense, "combinational", "", tables)
                .expect("build load-sensitive timing arc")
        })
        .collect();
    pins.push(Pin {
        direction: PinDirection::Output as i32,
        name: builder.intern_string("Y").expect("intern output pin"),
        function: builder
            .intern_string(function)
            .expect("intern Boolean cell function"),
        max_capacitance: Some(1.0),
        timing_arcs: arcs,
        ..Pin::default()
    });
    builder.cells.push(Cell {
        name: name.to_string(),
        pins,
        area: 1.0,
        ..Cell::default()
    });
}

/// Creates timing-complete libraries with either real or ideal input drivers.
fn test_library(representative_driver: bool) -> xlsynth_g8r::liberty_proto::Library {
    let mut builder = LibraryBuilder::new();
    builder.units = Some(LibraryUnits {
        time_unit: "1ps".to_string(),
        capacitance_unit: "1pf".to_string(),
        ..LibraryUnits::default()
    });
    builder.lu_table_templates.push(LuTableTemplate {
        name: "output_load".to_string(),
        index_1: vec![0.0, 1.0],
        ..LuTableTemplate::default()
    });
    add_timing_cell(&mut builder, "BUF", &["A"], "A", 0.05, 0.05, 8.05);
    add_timing_cell(&mut builder, "INV", &["A"], "!A", 0.10, 0.20, 1.20);
    add_timing_cell(&mut builder, "AND2", &["A", "B"], "A * B", 0.20, 0.20, 1.20);
    if representative_driver {
        builder.boundary_timing_defaults = Some(BoundaryTimingDefaults {
            representative_driver_cell: "BUF".to_string(),
            representative_load_cell: "BUF".to_string(),
            representative_load_count: 2,
        });
    }
    let mut proto = library_to_proto(builder.finish()).expect("serialize timing-complete library");
    proto.lu_table_templates[0].kind = LutTemplateKind::Timing as i32;
    proto.lu_table_templates[0].variable_1 = LutVariable::TotalOutputNetCapacitance as i32;
    proto
}

/// Writes one shared-input AIG and its selected representative-driver library.
fn make_fixture(representative_driver: bool) -> MappingFixture {
    let temp_dir = tempfile::tempdir().expect("create high-fanout mapping fixture");
    let aiger_path = temp_dir.path().join("fanout.aag");
    let liberty_path = temp_dir.path().join("fanout.liberty.proto");
    let mapped_path = temp_dir.path().join("fanout.mapped.gv");

    let mut builder = GateBuilder::new("fanout".to_string(), GateBuilderOptions::no_opt());
    let control = *builder.add_input("control".to_string(), 1).get_lsb(0);
    let data = builder.add_input("data".to_string(), SINK_COUNT);
    for index in 0..SINK_COUNT {
        let result = builder.add_and_binary(control, *data.get_lsb(index));
        builder.add_output(format!("out_{index}"), AigBitVector::from_bit(result));
    }
    std::fs::write(
        &aiger_path,
        emit_aiger(&builder.build(), true).expect("serialize high-fanout AIGER"),
    )
    .expect("write high-fanout AIGER");
    std::fs::write(
        &liberty_path,
        test_library(representative_driver).encode_to_vec(),
    )
    .expect("write representative-driver Liberty proto");

    MappingFixture {
        temp_dir,
        aiger_path,
        liberty_path,
        mapped_path,
    }
}

/// Executes the complete native mapping command with optional buffering.
fn map_choice_aig(fixture: &MappingFixture, buffering: bool) {
    let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .arg("choice-aig-tech-map")
        .arg(&fixture.aiger_path)
        .arg("--liberty_proto")
        .arg(&fixture.liberty_path)
        .arg("--netlist_out")
        .arg(&fixture.mapped_path)
        .arg("--buffer")
        .arg(if buffering { "true" } else { "false" })
        .arg("--max-fanout")
        .arg(MAX_FANOUT.to_string())
        .output()
        .expect("run native choice-AIG technology mapper");
    assert!(
        output.status.success(),
        "native choice-AIG mapping failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Counts actual cell inputs directly connected to the shared data source.
fn primary_input_fanout(path: &Path) -> (usize, usize) {
    let bytes = std::fs::read(path).expect("read mapped high-fanout netlist");
    let scanner = TokenScanner::with_line_lookup(Cursor::new(bytes), Box::new(|_| None));
    let mut parser = NetlistParser::new(scanner);
    let modules = parser
        .parse_file()
        .expect("parse mapped high-fanout netlist");
    assert_eq!(modules.len(), 1);
    let module = &modules[0];
    let control_symbol = parser
        .interner
        .get("control")
        .expect("resolve shared physical data input");
    let control = module
        .find_net_index(control_symbol, &parser.nets)
        .expect("find shared physical data-input net");
    let fanout = module
        .instances
        .iter()
        .flat_map(|instance| &instance.connections)
        .filter(|(_, net)| matches!(net, NetRef::Simple(net) if *net == control))
        .count();
    let buffers = module
        .instances
        .iter()
        .filter(|instance| parser.interner.resolve(instance.type_name) == Some("BUF"))
        .count();
    (fanout, buffers)
}

#[test]
fn choice_aig_tech_map_automatically_buffers_representative_driver_inputs() {
    let fixture = make_fixture(true);
    map_choice_aig(&fixture, true);
    let (fanout, buffers) = primary_input_fanout(&fixture.mapped_path);

    assert!(
        buffers > 0,
        "expected automatic physical-input buffer trees"
    );
    assert!(
        fanout <= MAX_FANOUT,
        "representative-driver input still directly drives {fanout} cells"
    );
}

#[test]
fn choice_aig_tech_map_preserves_ideal_input_opt_in_behavior() {
    let fixture = make_fixture(false);
    map_choice_aig(&fixture, true);
    let (fanout, buffers) = primary_input_fanout(&fixture.mapped_path);

    assert_eq!(fanout, SINK_COUNT);
    assert_eq!(buffers, 0);
}

#[test]
fn gv_optimize_automatically_buffers_representative_driver_inputs() {
    let fixture = make_fixture(true);
    map_choice_aig(&fixture, false);
    assert_eq!(primary_input_fanout(&fixture.mapped_path), (SINK_COUNT, 0));
    let optimized_path = fixture.temp_dir.path().join("fanout.optimized.gv");
    let output = Command::new(env!("CARGO_BIN_EXE_xlsynth-driver"))
        .arg("gv-optimize")
        .arg("--netlist")
        .arg(&fixture.mapped_path)
        .arg("--liberty_proto")
        .arg(&fixture.liberty_path)
        .arg("--netlist_out")
        .arg(&optimized_path)
        .arg("--max-fanout")
        .arg(MAX_FANOUT.to_string())
        .arg("--resize")
        .arg("false")
        .output()
        .expect("run mapped-netlist optimization");
    assert!(
        output.status.success(),
        "mapped-netlist optimization failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let (fanout, buffers) = primary_input_fanout(&optimized_path);
    assert!(
        buffers > 0,
        "expected automatic physical-input buffer trees"
    );
    assert!(
        fanout <= MAX_FANOUT,
        "representative-driver input still directly drives {fanout} cells"
    );
}
