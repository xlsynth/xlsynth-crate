// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::aig::{
    ClockPort, GateBuilder, GateBuilderOptions, RegisterBinding, TransitionInputId,
    TransitionOutputId,
};
use crate::aig_sim::{
    gate_sim,
    sequential::{SequentialState, simulate},
};
use crate::liberty::parser::parse_liberty_files;
use crate::netlist::gv_eval::{GvEvalOptions, load_labeled_sequential_netlist_aig_with_liberty};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use xlsynth_test_helpers::compare_golden_text;

/// Short alternative flow with a trailing comment to exercise script framing.
const CUSTOM_PASSES: &str = "&get -n; &dch; &nf; &put\n# caller-selected passes";

fn library() -> Library {
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), r#"library(test) {
        capacitive_load_unit(1, pf);
        cell(BUF) { area: 1; pin(A) { direction: input; capacitance: 0.0005; }
            pin(Y) { direction: output; function: "A"; } }
        cell(INV) { area: 1; pin(A) { direction: input; } pin(Y) { direction: output; function: "!A"; } }
        cell(DFFN) { area: 1; pin(D) { direction: input; } pin(CLK) { clock: true; }
            pin(QN) { direction: output; function: "IQN"; }
            ff(IQ,IQN) { clocked_on: "CLK"; next_state: "D"; } }
        cell(DFF) { area: 2; pin(D) { direction: input; } pin(CLK) { clock: true; }
            pin(Q) { direction: output; function: "IQ"; }
            ff(IQ,IQN) { clocked_on: "CLK"; next_state: "D"; } }
    }"#).unwrap();
    let mut library = parse_liberty_files(&[file.path()]).unwrap();
    library.reference_buffer_cell = Some("BUF".into());
    library.default_output_load = Some(0.004);
    library
}

/// Includes a shared D/external-output endpoint and an internal-name collision.
fn design() -> SequentialGateFn {
    let mut b = GateBuilder::new("top".into(), GateBuilderOptions::no_opt());
    let a = b.add_input("__abc_n0".into(), 2);
    let q = b.add_input("q".into(), 2);
    b.add_output("data".into(), a);
    b.add_output("wire".into(), q);
    SequentialGateFn::new(
        "top".into(),
        b.build(),
        vec![TransitionInputId::new(0)],
        vec![TransitionOutputId::new(0), TransitionOutputId::new(1)],
        Some(ClockPort { name: "clk".into() }),
        vec![RegisterBinding {
            name: "r".into(),
            q: TransitionInputId::new(1),
            d: TransitionOutputId::new(0),
            enable: None,
            reset: None,
        }],
    )
    .unwrap()
}

#[test]
fn physical_d_polarity_does_not_invert_shared_external_output() {
    let design = design_with_logic();
    let plan = RegisterMappingLibrary::new(&library())
        .unwrap()
        .map_registers(&design, &RegisterCellSelection::SmallestArea)
        .unwrap();
    assert!(plan.cells[0].invert_data);
    let boundary = Boundary::new(&design, &plan);
    let values = [true, false, false, true].map(|v| IrBits::from_lsb_is_0(&[v]));
    let outputs = gate_sim::eval(&boundary.graph, &values, gate_sim::Collect::None).outputs;
    let expected = [false, true].map(|v| IrBits::from_lsb_is_0(&[v]));
    assert_eq!(outputs, expected);
    assert_eq!(
        boundary.register_data,
        [
            BoundarySignal::Direct("i0".into()),
            BoundarySignal::Direct("i1".into())
        ]
    );
}

/// Keeps actual logic in the transition graph for external-process tests.
fn design_with_logic() -> SequentialGateFn {
    let mut design = design();
    design.transition.outputs[0].bit_vector = AigBitVector::from_lsb_is_index_0(
        &design.transition.outputs[0]
            .bit_vector
            .iter_lsb_to_msb()
            .map(|b| b.negate())
            .collect::<Vec<_>>(),
    );
    design
}

#[test]
fn mixed_boundaries_choose_per_bit_cells_and_rephase_every_state_use() {
    let mut builder = GateBuilder::new("mixed".into(), GateBuilderOptions::no_opt());
    let a = builder.add_input("a".into(), 2);
    let q = builder.add_input("q".into(), 4);
    let shared = builder.add_and_binary(*q.get_lsb(0), *q.get_lsb(1));
    let feedback = builder.add_and_binary(*q.get_lsb(3), *q.get_lsb(1));
    builder.add_output(
        "d".into(),
        AigBitVector::from_lsb_is_index_0(&[*a.get_lsb(0), *a.get_lsb(1), shared, feedback]),
    );
    builder.add_output(
        "y".into(),
        AigBitVector::from_lsb_is_index_0(&[
            *q.get_lsb(0),
            *q.get_lsb(2),
            *q.get_lsb(0),
            q.get_lsb(1).negate(),
            feedback,
        ]),
    );
    let design = SequentialGateFn::new(
        "mixed".into(),
        builder.build(),
        vec![TransitionInputId::new(0)],
        vec![TransitionOutputId::new(1)],
        Some(ClockPort { name: "clk".into() }),
        vec![RegisterBinding {
            name: "r".into(),
            q: TransitionInputId::new(1),
            d: TransitionOutputId::new(0),
            enable: None,
            reset: None,
        }],
    )
    .unwrap();
    let plan = RegisterMappingLibrary::new(&library())
        .unwrap()
        .map_registers(&design, &RegisterCellSelection::SmallestArea)
        .unwrap();
    assert_eq!(
        plan.bits
            .iter()
            .map(|bit| (
                plan.cells[bit.cell_index].cell_name.as_str(),
                bit.output_inverted,
            ))
            .collect::<Vec<_>>(),
        [
            ("DFF", false),
            ("DFFN", true),
            ("DFFN", false),
            ("DFFN", false)
        ]
    );
    let boundary = Boundary::new(&design, &plan);
    assert_eq!(
        &boundary.register_data[..2],
        &[
            BoundarySignal::Direct("i0".into()),
            BoundarySignal::Direct("i1".into())
        ]
    );
    assert_eq!(
        boundary.external_outputs[0][0],
        BoundarySignal::Direct("i2".into())
    );
    assert_eq!(
        boundary.external_outputs[0][1],
        BoundarySignal::Direct("i4".into())
    );
    assert_eq!(
        boundary.external_outputs[0][2],
        BoundarySignal::Direct("i2".into())
    );
    // Exhaust all states/inputs and compare transition and external outputs.
    // This catches missing phase changes on shared fanout and feedback edges.
    for assignment in 0..64 {
        let source = [
            IrBits::make_ubits(2, assignment & 3).unwrap(),
            IrBits::make_ubits(4, assignment >> 2).unwrap(),
        ];
        let expected = gate_sim::eval(&design.transition, &source, gate_sim::Collect::None).outputs;
        let physical = (0..6)
            .map(|index| IrBits::from_lsb_is_0(&[(assignment & (1 << index) != 0) ^ (index == 3)]))
            .collect::<Vec<_>>();
        let actual = gate_sim::eval(&boundary.graph, &physical, gate_sim::Collect::None).outputs;
        let value = |signal: &BoundarySignal| {
            let values = match signal {
                BoundarySignal::Direct(name) => {
                    &physical[boundary
                        .graph
                        .inputs
                        .iter()
                        .position(|p| &p.name == name)
                        .unwrap()]
                }
                BoundarySignal::Mapped(name) => {
                    &actual[boundary
                        .graph
                        .outputs
                        .iter()
                        .position(|p| &p.name == name)
                        .unwrap()]
                }
            };
            values.get_bit(0).unwrap()
        };
        for (index, bit) in plan.bits.iter().enumerate() {
            assert_eq!(
                value(&boundary.register_data[index])
                    ^ plan.cells[bit.cell_index].invert_data
                    ^ bit.output_inverted,
                expected[0].get_bit(index).unwrap()
            );
        }
        for (index, signal) in boundary.external_outputs[0].iter().enumerate() {
            assert_eq!(value(signal), expected[1].get_bit(index).unwrap());
        }
    }
}

#[test]
fn both_boundaries_reject_inverting_only_libraries_and_fixed_cells() {
    let mut lib = library();
    let catalog = RegisterMappingLibrary::new(&lib).unwrap();
    let expected = "register 'r[0]' directly connects a module input to a module output; no compatible non-inverting FF for FixedCell(\"DFFN\")";
    assert_eq!(
        catalog
            .map_registers(&design(), &RegisterCellSelection::FixedCell("DFFN".into()))
            .unwrap_err(),
        expected
    );
    lib.cells.retain(|cell| cell.name != "DFF");
    let catalog = RegisterMappingLibrary::new(&lib).unwrap();
    assert_eq!(
        catalog
            .map_registers(&design(), &RegisterCellSelection::SmallestArea)
            .unwrap_err(),
        "register 'r[0]' directly connects a module input to a module output; no compatible non-inverting FF for SmallestArea"
    );
    // An inverted next-state function does not impose the direct-input
    // invariant.
    assert!(
        catalog
            .map_registers(&design_with_logic(), &RegisterCellSelection::SmallestArea)
            .is_ok()
    );
}

#[cfg(unix)]
#[test]
fn direct_identity_maps_without_invoking_abc_or_adding_cells() {
    let temp = tempfile::tempdir().unwrap();
    let raw = temp.path().join("cells.lib");
    std::fs::write(&raw, "library(test) {}").unwrap();
    // This executable cannot emit a mapped netlist, so success proves that
    // the wire-only design never invokes ABC's mapping script.
    let context = AbcMappingContext::new(
        AbcToolchain::new("/usr/bin/true").unwrap(),
        &[raw],
        library(),
    )
    .unwrap();
    let mut builder = GateBuilder::new("identity".into(), GateBuilderOptions::no_opt());
    let a = builder.add_input("a".into(), 1);
    builder.add_output("y".into(), a);
    let design = crate::aig::add_output_registers(
        &SequentialGateFn::from_gate_fn(builder.build()),
        ClockPort { name: "clk".into() },
    )
    .unwrap();
    let result = context
        .map_sequential(
            &design,
            &AbcMappingOptions::from_library(context.library()).unwrap(),
        )
        .unwrap();
    assert!(result.diagnostics.script.is_empty());
    assert_eq!(result.netlist.modules[0].instances.len(), 1);
    assert_eq!(result.registers.cells[0].cell_name, "DFF");
    compare_golden_text(
        &result.emit_verilog().unwrap(),
        "src/netlist/abc/direct_identity.golden.v",
    );
}

#[test]
fn assembly_is_flat_and_preserves_direct_packed_register_boundaries() {
    let design = design();
    let library = library();
    let plan = RegisterMappingLibrary::new(&library)
        .unwrap()
        .map_registers(&design, &RegisterCellSelection::SmallestArea)
        .unwrap();
    let boundary = Boundary::new(&design, &plan);
    let mut parsed = read_gv_from_str(
        r#"module mapped(i0,i1,i2,i3);
        input i0,i1,i2,i3; endmodule"#,
    )
    .unwrap();
    validate_mapped_cells(&parsed, &library).unwrap();
    assemble(&mut parsed, &design, &plan, &boundary).unwrap();
    let text =
        emit_module_as_netlist_text(&parsed.modules[0], &parsed.nets, &parsed.interner).unwrap();
    compare_golden_text(&text, "src/netlist/abc/assembly.golden.v");
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), text).unwrap();
    let mapped = load_labeled_sequential_netlist_aig_with_liberty(
        file.path(),
        &library,
        &GvEvalOptions {
            clock_port_name: Some("clk".into()),
            ..Default::default()
        },
    )
    .unwrap();
    let stimulus = [0, 1, 3, 2, 0].map(|v| vec![IrBits::make_ubits(2, v).unwrap()]);
    let original = simulate(&design, &stimulus, SequentialState::all_zeros(&design)).unwrap();
    // The both-boundaries case uses a non-inverting FF and unchanged state.
    let initial = SequentialState::from_register_values(
        &mapped.sequential_gate_fn,
        vec![IrBits::make_ubits(1, 0).unwrap(); 2],
    )
    .unwrap();
    let after = mapped.simulate_bits(&stimulus, initial).unwrap();
    assert_eq!(after.external_outputs(), original.external_outputs());
}

#[test]
fn mismatched_cells_pins_and_interfaces_are_rejected() {
    let library = library();
    for text in [
        "module m(a,y); input a; output y; UNKNOWN c(.A(a),.Y(y)); endmodule",
        "module m(a,y); input a; output y; INV c(.BAD(a),.Y(y)); endmodule",
        "module m(a,y); input a; output y; INV c(.Y(y)); endmodule",
        "module m(a,y); input a; output y; DFFN c(.D(a),.CLK(a),.QN(y)); endmodule",
    ] {
        assert!(validate_mapped_cells(&read_gv_from_str(text).unwrap(), &library).is_err());
    }
    let design = design();
    let plan = RegisterMappingLibrary::new(&library)
        .unwrap()
        .map_registers(&design, &RegisterCellSelection::SmallestArea)
        .unwrap();
    let mut parsed = read_gv_from_str("module mapped(); endmodule").unwrap();
    assert!(assemble(&mut parsed, &design, &plan, &Boundary::new(&design, &plan)).is_err());
}

#[test]
fn scripts_keep_library_input_constraint_and_output_handling_fixed() {
    for (count, expected) in [
        (1, "read_lib library_0.lib\n"),
        (2, "read_lib library_0.lib library_1.lib\n"),
        (
            3,
            "read_lib library_0.lib library_1.lib\nread_lib -m library_2.lib\n",
        ),
    ] {
        for passes in [ABC_SPEED_SCRIPT, CUSTOM_PASSES] {
            let script = mapping_script(count, passes);
            assert_eq!(
                script.strip_prefix(expected).unwrap(),
                format!(
                    "read_aiger transition.aig\nstrash\nread_constr timing.constr\n{}\nwrite_verilog mapped.v\n",
                    passes.trim_end_matches('\n'),
                )
            );
        }
    }
    compare_golden_text(
        &mapping_script(3, ABC_SPEED_SCRIPT),
        "src/netlist/abc/default_script.golden.txt",
    );
    compare_golden_text(
        &mapping_script(3, CUSTOM_PASSES),
        "src/netlist/abc/custom_script.golden.txt",
    );
    assert_eq!(
        mapping_script(3, CUSTOM_PASSES),
        mapping_script(3, &format!("{CUSTOM_PASSES}\n")),
    );
}

#[test]
fn explicit_library_defaults_are_positive_and_use_femtofarads() {
    let library = library();
    let constraints = AbcIoConstraints::from_library(&library).unwrap();
    assert_eq!(
        constraints,
        AbcIoConstraints {
            driving_cell: "BUF".into(),
            output_load_ff: 4.0
        }
    );
    assert_eq!(
        constraints.to_abc_text(),
        "set_driving_cell BUF\nset_load 4\n"
    );
    for name in ["ABSENT", "INV", "DFFN", "BUF\nquit", "BUF;quit"] {
        assert!(
            AbcIoConstraints {
                driving_cell: name.into(),
                ..constraints.clone()
            }
            .validate(&library)
            .is_err()
        );
    }
    for load in [
        0.0,
        -1.0,
        f64::NAN,
        f64::INFINITY,
        f64::MAX,
        f64::MIN_POSITIVE,
    ] {
        assert!(
            AbcIoConstraints {
                output_load_ff: load,
                ..constraints.clone()
            }
            .validate(&library)
            .is_err()
        );
    }
}

#[test]
fn library_defaults_still_check_abc_name_and_float_limits() {
    for name in ["BUF\nquit", "BUF;quit"] {
        let mut lib = library();
        lib.cells.iter_mut().find(|c| c.name == "BUF").unwrap().name = name.into();
        lib.reference_buffer_cell = Some(name.into());
        assert!(resolve_reference_buffer(&lib).is_ok());
        assert!(AbcIoConstraints::from_library(&lib).is_err());
        assert!(AbcMappingOptions::from_library(&lib).is_err());
    }
    for cap in [f64::MAX / 4000.0, f64::MIN_POSITIVE] {
        let mut lib = library();
        lib.default_output_load = Some(cap);
        assert!(default_output_load_ff(&lib).is_ok());
        assert!(AbcIoConstraints::from_library(&lib).is_err());
        assert!(AbcMappingOptions::from_library(&lib).is_err());
    }
}

#[test]
fn options_resolve_library_defaults_at_construction() {
    let mut library = library();
    let options = AbcMappingOptions::from_library(&library).unwrap();
    assert_eq!(
        options.register_selection,
        RegisterCellSelection::SmallestArea
    );
    assert!(!options.cleanup_registers);
    assert_eq!(options.timeout, Duration::from_secs(180));
    assert_eq!(options.script, ABC_SPEED_SCRIPT);
    assert_eq!(
        options.io_constraints,
        AbcIoConstraints {
            driving_cell: "BUF".into(),
            output_load_ff: 4.0,
        }
    );
    for load in [None, Some(-1.0), Some(f64::NAN)] {
        library.default_output_load = load;
        assert!(AbcMappingOptions::from_library(&library).is_err());
    }
    library.default_output_load = Some(0.004);
    library.units = None;
    assert!(AbcMappingOptions::from_library(&library).is_err());
    library = self::library();
    library.reference_buffer_cell = None;
    assert!(AbcMappingOptions::from_library(&library).is_err());
    library.reference_buffer_cell = Some("INV".into());
    assert!(AbcMappingOptions::from_library(&library).is_err());
    // The resolved options are independent of later library metadata changes.
    options.io_constraints.validate(&library).unwrap();
    assert_eq!(options.io_constraints.output_load_ff, 4.0);
}

#[cfg(unix)]
#[test]
fn explicit_options_map_without_usable_library_defaults() {
    let temp = tempfile::tempdir().unwrap();
    let executable = temp.path().join("fake abc");
    std::fs::write(
        &executable,
        r#"#!/bin/sh
if [ "$2" = "-c" ]; then exit 0; fi
test "$(cat timing.constr)" = "set_driving_cell BUF
set_load 20" || exit 3
echo 'module mapped(i0,o0); input i0; output o0; INV c(.A(i0),.Y(o0)); endmodule' > mapped.v
"#,
    )
    .unwrap();
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o755)).unwrap();
    let raw = temp.path().join("cells.lib");
    std::fs::write(&raw, "library(test) {}").unwrap();
    let mut context =
        AbcMappingContext::new(AbcToolchain::new(&executable).unwrap(), &[raw], library()).unwrap();
    let mut builder = GateBuilder::new("top".into(), GateBuilderOptions::no_opt());
    let input = builder.add_input("a".into(), 1);
    builder.add_output(
        "y".into(),
        AigBitVector::from_bit(input.get_lsb(0).negate()),
    );
    let design = SequentialGateFn::from_gate_fn(builder.build());
    let explicit = AbcIoConstraints {
        driving_cell: "BUF".into(),
        output_load_ff: 20.0,
    };
    let options = AbcMappingOptions::new(explicit.clone());
    assert_eq!(
        options.register_selection,
        RegisterCellSelection::SmallestArea
    );
    assert!(!options.cleanup_registers);
    assert_eq!(options.timeout, Duration::from_secs(180));
    assert_eq!(options.script, ABC_SPEED_SCRIPT);
    for load in [Some(0.004), None, Some(-1.0), Some(f64::NAN)] {
        context.library.default_output_load = load;
        let result = context.map_sequential(&design, &options).unwrap();
        assert_eq!(result.diagnostics.io_constraints.as_ref(), Some(&explicit));
    }
    context.library.reference_buffer_cell = None;
    context.library.units = None;
    let result = context.map_sequential(&design, &options).unwrap();
    assert_eq!(result.diagnostics.io_constraints.as_ref(), Some(&explicit));

    // A caller can also change already-resolved options before mapping.
    let mut overridden = AbcMappingOptions::from_library(&library()).unwrap();
    overridden.io_constraints = explicit;
    context.map_sequential(&design, &overridden).unwrap();
}

#[cfg(unix)]
#[test]
fn mapping_validates_explicit_and_modified_constraints() {
    let temp = tempfile::tempdir().unwrap();
    let raw = temp.path().join("cells.lib");
    std::fs::write(&raw, "library(test) {}").unwrap();
    let context = AbcMappingContext::new(
        AbcToolchain::new("/usr/bin/true").unwrap(),
        &[raw],
        library(),
    )
    .unwrap();
    for constraints in [
        AbcIoConstraints {
            driving_cell: "INV".into(),
            output_load_ff: 4.0,
        },
        AbcIoConstraints {
            driving_cell: "BUF".into(),
            output_load_ff: 0.0,
        },
    ] {
        let expected = constraints.validate(context.library()).unwrap_err();
        let explicit = AbcMappingOptions::new(constraints.clone());
        let mut modified = AbcMappingOptions::from_library(context.library()).unwrap();
        modified.io_constraints = constraints;
        for options in [explicit, modified] {
            assert_eq!(
                context
                    .map_sequential(&design(), &options)
                    .err()
                    .expect("invalid explicit constraints must be rejected before ABC runs")
                    .to_string(),
                expected
            );
        }
    }
}

#[cfg(unix)]
#[test]
fn constraint_file_is_written_and_missing_raw_driver_warning_is_fatal() {
    let temp = tempfile::tempdir().unwrap();
    let executable = temp.path().join("fake abc");
    std::fs::write(
        &executable,
        r#"#!/bin/sh
if [ "$2" = "-c" ]; then exit 0; fi
test "$(cat timing.constr)" = "set_driving_cell BUF
set_load 4" || exit 3
echo 'Cannot find the default PI driving cell (BUF) in the library.'
echo 'module mapped(); endmodule' > mapped.v
"#,
    )
    .unwrap();
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o755)).unwrap();
    let raw = temp.path().join("raw.lib");
    std::fs::write(&raw, "library(raw) {}").unwrap();
    let context =
        AbcMappingContext::new(AbcToolchain::new(&executable).unwrap(), &[raw], library()).unwrap();
    let error = context
        .map_sequential(
            &design_with_logic(),
            &AbcMappingOptions::from_library(context.library()).unwrap(),
        )
        .err()
        .unwrap();
    assert!(
        error
            .to_string()
            .contains("ABC could not use driving cell 'BUF'"),
        "{error}"
    );
}

#[cfg(unix)]
#[test]
fn mapping_invokes_the_override_and_records_the_complete_script() {
    let temp = tempfile::tempdir().unwrap();
    let executable = temp.path().join("fake abc");
    std::fs::write(
        &executable,
        format!(
            r#"#!/bin/sh
if [ "$2" = "-c" ]; then exit 0; fi
test "$1" = '-s' && test "$2" = '-f' && test "$3" = 'map.abc' || exit 2
cat <<'EXPECTED' | cmp - map.abc || exit 3
{}EXPECTED
test -s transition.aig || exit 4
test "$(cat timing.constr)" = "set_driving_cell BUF
set_load 4" || exit 5
echo 'module mapped(i0,o0); input i0; output o0; INV c(.A(i0),.Y(o0)); endmodule' > mapped.v
"#,
            include_str!("custom_script.golden.txt")
        ),
    )
    .unwrap();
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o755)).unwrap();
    let raw = temp.path().join("cells.lib");
    std::fs::write(&raw, "library(test) {}").unwrap();
    let context = AbcMappingContext::new(
        AbcToolchain::new(&executable).unwrap(),
        &[&raw, &raw, &raw],
        library(),
    )
    .unwrap();
    let mut builder = GateBuilder::new("top".into(), GateBuilderOptions::no_opt());
    let a = builder.add_input("a".into(), 1);
    builder.add_output("y".into(), AigBitVector::from_bit(a.get_lsb(0).negate()));
    let design = SequentialGateFn::from_gate_fn(builder.build());
    let mut options = AbcMappingOptions::from_library(context.library()).unwrap();
    options.script = CUSTOM_PASSES.to_owned();
    let result = context.map_sequential(&design, &options).unwrap();
    compare_golden_text(
        &result.diagnostics.script,
        "src/netlist/abc/custom_script.golden.txt",
    );
    assert_eq!(
        result.diagnostics.io_constraints,
        Some(options.io_constraints)
    );
}

#[cfg(unix)]
#[test]
fn post_abc_validation_and_assembly_errors_preserve_diagnostics() {
    let temp = tempfile::tempdir().unwrap();
    let executable = temp.path().join("fake abc");
    let raw = temp.path().join("raw.lib");
    std::fs::write(&raw, "library(raw) {}").unwrap();
    let mut errors = Vec::new();
    for (label, netlist) in [
        (
            "unknown cell",
            "module mapped(a,y); input a; output y; UNKNOWN c(.A(a),.Y(y)); endmodule",
        ),
        (
            "unknown pin",
            "module mapped(a,y); input a; output y; INV c(.BAD(a),.Y(y)); endmodule",
        ),
        ("boundary mismatch", "module mapped(); endmodule"),
    ] {
        std::fs::write(
            &executable,
            format!(
                r#"#!/bin/sh
if [ "$2" = "-c" ]; then echo 'test ABC'; exit 0; fi
echo 'mapping stdout'
echo 'mapping stderr' >&2
printf '%s\n' '{netlist}' > mapped.v
"#
            ),
        )
        .unwrap();
        std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o755)).unwrap();
        let context =
            AbcMappingContext::new(AbcToolchain::new(&executable).unwrap(), &[&raw], library())
                .unwrap();
        let error = context
            .map_sequential(
                &design_with_logic(),
                &AbcMappingOptions::from_library(context.library()).unwrap(),
            )
            .err()
            .expect("invalid ABC output must be rejected");
        assert!(!error.is_resource_failure());
        // Normalize only the temporary paths and separately tested script.
        let message = error
            .to_string()
            .replace(
                temp.path().canonicalize().unwrap().to_str().unwrap(),
                "<workdir>",
            )
            .replace(temp.path().to_str().unwrap(), "<workdir>")
            .replace(ABC_SPEED_SCRIPT, "<speed-script>\n");
        errors.push(format!("{label}:\n{message}"));
    }
    compare_golden_text(
        &errors.join("\n"),
        "src/netlist/abc/failure_diagnostics.golden.txt",
    );
}

#[test]
fn assembly_escapes_reserved_external_port_names() {
    for name in [
        "reg",
        "logic",
        "always",
        "always_ff",
        "always_comb",
        "always_latch",
        "initial",
        "generate",
        "genvar",
        "case",
        "begin",
        "end",
        "function",
        "buf",
    ] {
        let text = format!("module top(\\{name} ); input \\{name} ; endmodule");
        let parsed = read_gv_from_str(&text).unwrap();
        let emitted =
            emit_module_as_netlist_text(&parsed.modules[0], &parsed.nets, &parsed.interner)
                .unwrap();
        assert_eq!(
            emitted,
            format!("module top(\\{name} );\n  input \\{name} ;\nendmodule\n")
        );
        let reread = read_gv_from_str(&emitted).unwrap();
        assert_eq!(
            reread.interner.resolve(reread.modules[0].ports[0].name),
            Some(name)
        );
    }
}

#[cfg(unix)]
#[test]
fn mapping_rejects_all_physical_port_name_collisions_before_abc() {
    let temp = tempfile::tempdir().unwrap();
    let raw = temp.path().join("cells.lib");
    std::fs::write(&raw, "library(test) {}").unwrap();
    let context = AbcMappingContext::new(
        AbcToolchain::new("/usr/bin/true").unwrap(),
        &[raw],
        library(),
    )
    .unwrap();
    // All pairings: input/input, output/output, input/output, and either with
    // the clock. ABC would fail to emit a netlist if validation let one
    // through.
    for first in 0..5 {
        for second in first + 1..5 {
            let mut names = ["a", "b", "y", "z", "clk"];
            names[second] = names[first];
            let mut b = GateBuilder::new("top".into(), GateBuilderOptions::no_opt());
            let a = b.add_input(names[0].into(), 2);
            let other = b.add_input(names[1].into(), 2);
            b.add_output(names[2].into(), a);
            b.add_output(names[3].into(), other);
            let mut design = SequentialGateFn::from_gate_fn(b.build());
            design.clock = Some(ClockPort {
                name: names[4].into(),
            });
            design.validate_legalized().unwrap();
            for cleanup_registers in [false, true] {
                let error = context
                    .map_sequential(
                        &design,
                        &AbcMappingOptions {
                            cleanup_registers,
                            ..AbcMappingOptions::from_library(context.library()).unwrap()
                        },
                    )
                    .err()
                    .expect("colliding physical ports must be rejected");
                assert_eq!(
                    error.to_string(),
                    format!("duplicate external port or clock name: {:?}", names[first])
                );
            }
        }
    }
}

#[test]
fn name_validation_ignores_nonphysical_names_for_uniqueness() {
    let mut b = GateBuilder::new("a".into(), GateBuilderOptions::no_opt());
    let zero = b.add_input("a".into(), 0);
    let a = b.add_input("a".into(), 2);
    b.add_output("a".into(), zero);
    b.add_output("y".into(), a);
    let combo = SequentialGateFn::from_gate_fn(b.build());
    combo.validate_legalized().unwrap();
    validate_names(&combo).unwrap();

    let mut registered = design();
    registered.transition.inputs[1].name = registered.transition.inputs[0].name.clone();
    registered.name = registered.transition.inputs[0].name.clone();
    registered.validate_legalized().unwrap();
    validate_names(&registered).unwrap();
}

#[cfg(unix)]
#[test]
fn zero_width_and_clock_only_designs_need_no_abc_network() {
    let temp = tempfile::tempdir().unwrap();
    let raw = temp.path().join("cells.lib");
    std::fs::write(&raw, "library(test) {}").unwrap();
    let context = AbcMappingContext::new(
        AbcToolchain::new("/usr/bin/true").unwrap(),
        &[raw],
        library(),
    )
    .unwrap();
    let mut b = GateBuilder::new("empty".into(), GateBuilderOptions::no_opt());
    // Zero-width logical ports may share the physical clock's name.
    let a = b.add_input("clk".into(), 0);
    b.add_output("clk".into(), a);
    let mut design = SequentialGateFn::from_gate_fn(b.build());
    design.clock = Some(ClockPort { name: "clk".into() });
    let result = context
        .map_sequential(
            &design,
            &AbcMappingOptions::from_library(context.library()).unwrap(),
        )
        .unwrap();
    assert!(result.diagnostics.script.is_empty());
    assert_eq!(result.design.transition.inputs[0].get_bit_count(), 0);
    assert_eq!(result.design.transition.outputs[0].get_bit_count(), 0);
    assert_eq!(result.netlist.modules[0].ports.len(), 1);
    assert!(result.netlist.modules[0].instances.is_empty());
}

#[cfg(unix)]
#[test]
fn zero_exit_without_output_is_an_error_and_timeout_is_typed() {
    let temp = tempfile::tempdir().unwrap();
    let executable = temp.path().join("fake abc");
    std::fs::write(&executable, "#!/bin/sh\nexit 0\n").unwrap();
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o755)).unwrap();
    let raw = temp.path().join("raw library; name.lib");
    std::fs::write(&raw, "library(raw) {}").unwrap();
    let context =
        AbcMappingContext::new(AbcToolchain::new(&executable).unwrap(), &[raw], library()).unwrap();
    assert!(
        context
            .map_sequential(
                &design_with_logic(),
                &AbcMappingOptions::from_library(context.library()).unwrap()
            )
            .is_err()
    );
    std::fs::write(&executable, "#!/bin/sh\nsleep 30\n").unwrap();
    let error = context
        .map_sequential(
            &design_with_logic(),
            &AbcMappingOptions {
                timeout: Duration::from_millis(30),
                ..AbcMappingOptions::from_library(context.library()).unwrap()
            },
        )
        .err()
        .unwrap();
    assert!(error.is_resource_failure());
}
