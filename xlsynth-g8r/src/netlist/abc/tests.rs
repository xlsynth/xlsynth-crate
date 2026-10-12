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
    }"#).unwrap();
    let mut library = parse_liberty_files(&[file.path()]).unwrap();
    library.reference_buffer_cell = Some("BUF".into());
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
    let design = design();
    let plan = RegisterMappingLibrary::new(&library())
        .unwrap()
        .map_registers(&design, &RegisterCellSelection::SmallestArea)
        .unwrap();
    assert!(plan.cell.as_ref().unwrap().invert_data);
    let boundary = Boundary::new(&design, &plan);
    let values = [true, false, false, true].map(|v| IrBits::from_lsb_is_0(&[v]));
    let outputs = gate_sim::eval(&boundary.graph, &values, gate_sim::Collect::None).outputs;
    let expected = [true, false, false, true, false, true].map(|v| IrBits::from_lsb_is_0(&[v]));
    assert_eq!(outputs, expected);
}

#[test]
fn assembly_is_flat_and_preserves_packed_ports_and_inverted_state() {
    let design = design();
    let library = library();
    let plan = RegisterMappingLibrary::new(&library)
        .unwrap()
        .map_registers(&design, &RegisterCellSelection::SmallestArea)
        .unwrap();
    let boundary = Boundary::new(&design, &plan);
    let mut parsed = read_gv_from_str(
        r#"module mapped(i0,i1,i2,i3,o0,o1,o2,o3,o4,o5);
        input i0,i1,i2,i3; output o0,o1,o2,o3,o4,o5;
        assign o0=i0; assign o1=i1; assign o2=i2; assign o3=i3;
        INV g0(.A(i0),.Y(o4)); INV g1(.A(i1),.Y(o5)); endmodule"#,
    )
    .unwrap();
    validate_mapped_cells(&parsed, &library).unwrap();
    assemble(&mut parsed, &design, &plan, &boundary).unwrap();
    let text =
        emit_module_as_netlist_text(&parsed.modules[0], &parsed.nets, &parsed.interner).unwrap();
    assert_eq!(text, include_str!("assembly.golden.v"));
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
    // The selected QN output represents logical state; physical IQ starts at 1.
    let initial = SequentialState::from_register_values(
        &mapped.sequential_gate_fn,
        vec![IrBits::make_ubits(1, 1).unwrap(); 2],
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
fn script_only_generates_the_library_prologue() {
    for (count, expected) in [
        (1, "read_lib library_0.lib\n"),
        (2, "read_lib library_0.lib library_1.lib\n"),
        (
            3,
            "read_lib library_0.lib library_1.lib\nread_lib -m library_2.lib\n",
        ),
    ] {
        assert_eq!(
            mapping_script(count).strip_suffix(ABC_SPEED_SCRIPT),
            Some(expected)
        );
    }
}

#[test]
fn fo8_constraints_are_positive_and_use_femtofarads() {
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

#[cfg(unix)]
#[test]
fn missing_defaults_fail_and_explicit_constraints_take_precedence() {
    let temp = tempfile::tempdir().unwrap();
    let raw = temp.path().join("cells.lib");
    std::fs::write(&raw, "library(test) {}").unwrap();
    let mut library = library();
    library.reference_buffer_cell = None;
    let context =
        AbcMappingContext::new(AbcToolchain::new("/usr/bin/true").unwrap(), &[raw], library)
            .unwrap();
    assert!(context.resolve_io_constraints(None).is_err());
    let error = context
        .map_sequential(&design(), &Default::default())
        .err()
        .unwrap();
    assert!(error.to_string().contains("reference_buffer_cell"));
    let explicit = AbcIoConstraints {
        driving_cell: "BUF".into(),
        output_load_ff: 20.0,
    };
    assert_eq!(
        context.resolve_io_constraints(Some(&explicit)).unwrap(),
        explicit
    );
    let with_defaults = AbcMappingContext {
        library: self::library(),
        ..context
    };
    assert_eq!(
        with_defaults
            .resolve_io_constraints(Some(&explicit))
            .unwrap(),
        explicit
    );
    assert_eq!(
        with_defaults
            .resolve_io_constraints(None)
            .unwrap()
            .output_load_ff,
        4.0
    );
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
        .map_sequential(&design(), &Default::default())
        .err()
        .unwrap();
    assert!(
        error
            .to_string()
            .contains("ABC could not use driving cell 'BUF'"),
        "{error}"
    );
}

#[test]
fn assembly_escapes_all_reserved_external_port_names() {
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
    ] {
        let text = format!("module top(\\{name} ); input \\{name} ; endmodule");
        let parsed = read_gv_from_str(&text).unwrap();
        let emitted =
            emit_module_as_netlist_text(&parsed.modules[0], &parsed.nets, &parsed.interner)
                .unwrap();
        let reread = read_gv_from_str(&emitted).unwrap();
        assert_eq!(
            reread.interner.resolve(reread.modules[0].ports[0].name),
            Some(name)
        );
    }
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
    let a = b.add_input("empty_input".into(), 0);
    b.add_output("empty_output".into(), a);
    let mut design = SequentialGateFn::from_gate_fn(b.build());
    design.clock = Some(ClockPort { name: "clk".into() });
    let result = context
        .map_sequential(&design, &Default::default())
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
            .map_sequential(&design(), &Default::default())
            .is_err()
    );
    std::fs::write(&executable, "#!/bin/sh\nsleep 30\n").unwrap();
    let error = context
        .map_sequential(
            &design(),
            &AbcMappingOptions {
                timeout: Duration::from_millis(30),
                ..Default::default()
            },
        )
        .err()
        .unwrap();
    assert!(error.is_resource_failure());
}
