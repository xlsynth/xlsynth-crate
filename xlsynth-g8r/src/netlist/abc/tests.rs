// SPDX-License-Identifier: Apache-2.0

use super::*;
#[cfg(unix)]
use crate::aig::{AigBitVector, ClockPort, GateBuilder, GateBuilderOptions};
use crate::aig_sim::sequential::{SequentialState, simulate};
#[cfg(unix)]
use crate::liberty::parser::parse_liberty_files;
use crate::netlist::gv_eval::{GvEvalOptions, load_labeled_sequential_netlist_aig_with_liberty};
use crate::netlist::register_mapping::{RegisterCellSelection, RegisterMappingLibrary};
use crate::netlist::sequential_mapping::SequentialMappingOptions;
#[cfg(unix)]
use crate::netlist::sequential_mapping::assembly::tests::CLOCK_GATE_LIB;
use crate::netlist::sequential_mapping::assembly::tests::{design, design_with_logic, library};
#[cfg(unix)]
use crate::netlist::{
    emit::emit_netlist_as_text,
    gatefn_from_netlist::{
        PinConnection, SequentialAigSignal, project_labeled_sequential_netlist_aig,
    },
    io::read_gv_for_proof_from_path,
    proto::{decode_netlist, encode_netlist},
};
#[cfg(unix)]
use crate::prove_gate_fn_equiv_sat::{
    EquivResult, GateFormalBackend, prove_gate_fn_equiv_with_backend,
};
use xlsynth_pir::IrBits;
use xlsynth_test_helpers::compare_golden_text;

/// Short alternative flow with a trailing comment to exercise script framing.
const CUSTOM_PASSES: &str = "&get -n; &dch; &nf; &put\n# caller-selected passes";

/// Execute checked-in scripts so parallel tests never hold them open for
/// writing.
#[cfg(unix)]
fn fake_abc(script: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src/netlist/abc/fake_abc")
        .join(script)
}

#[cfg(unix)]
#[test]
fn mapped_shared_enable_preserves_outputs_and_every_next_state() -> anyhow::Result<()> {
    let raw = tempfile::NamedTempFile::new()?;
    let make_context = |lib: &str| -> anyhow::Result<_> {
        std::fs::write(raw.path(), lib)?;
        let library = parse_liberty_files(&[raw.path()]).map_err(anyhow::Error::msg)?;
        // Removing the four feedback muxes leaves only wires at the ABC
        // boundary.
        Ok(AbcMappingContext::new(
            AbcToolchain::new("/usr/bin/true")?,
            &[raw.path()],
            library,
        )?)
    };
    let context = make_context(CLOCK_GATE_LIB)?;
    let gv = tempfile::NamedTempFile::new()?;
    let project = |text: &str| -> anyhow::Result<_> {
        std::fs::write(gv.path(), text)?;
        let parsed = read_gv_for_proof_from_path(gv.path())?;
        project_labeled_sequential_netlist_aig(
            &parsed.modules[0],
            &parsed.nets,
            &parsed.interner,
            context.library(),
            None,
        )
        .map_err(anyhow::Error::msg)
    };
    // Name the physical flops in mapper order so the miter can check each
    // prior-state input and next-state output by the same register identity.
    let feedback = r#"module bank(input clk, input en, input [3:0] data, output [3:0] q);
  wire ready, d0, d1, d2, d3;
  DFF __reg_0 (.CLK(clk), .D(en), .Q(ready));
  MUX2 mux0 (.S(ready), .A(q[0]), .B(data[0]), .Y(d0));
  MUX2 mux1 (.S(ready), .A(q[1]), .B(data[1]), .Y(d1));
  MUX2 mux2 (.S(ready), .A(q[2]), .B(data[2]), .Y(d2));
  MUX2 mux3 (.S(ready), .A(q[3]), .B(data[3]), .Y(d3));
  DFF __reg_1 (.CLK(clk), .D(d0), .Q(q[0]));
  DFF __reg_2 (.CLK(clk), .D(d1), .Q(q[1]));
  DFF __reg_3 (.CLK(clk), .D(d2), .Q(q[2]));
  DFF __reg_4 (.CLK(clk), .D(d3), .Q(q[3]));
endmodule"#;
    let source = project(feedback)?.sequential_gate_fn;
    let mut sequential_options = SequentialMappingOptions::default();
    let abc_options = AbcMappingOptions::new(AbcIoConstraints {
        driving_cell: "BUF".into(),
        output_load_ff: 1.0,
    });
    let mapped = context.map_sequential(&source, &sequential_options, &abc_options)?;
    let text = mapped.emit_verilog().map_err(anyhow::Error::msg)?;
    let structural =
        context.map_sequential_structural(&source, &sequential_options, &abc_options)?;
    let decoded = decode_netlist(&encode_netlist(&structural.netlist)?)?;
    assert_eq!(emit_netlist_as_text(&decoded)?, text);

    let model = project(&text)?;
    assert_eq!(model.instances.len(), 6);
    let gate = model
        .instances
        .iter()
        .find(|inst| inst.cell_type == "ICG")
        .unwrap();
    let pin = |name| gate.pins.iter().find(|pin| pin.pin_name == name).unwrap();
    assert_eq!(pin("CLK").signal, SequentialAigSignal::Clock);
    assert_eq!(pin("SE").connection, PinConnection::Literal(false));
    let gated = pin("GCLK").signal;
    assert!(matches!(gated, SequentialAigSignal::GatedClock(_)));
    let clocks: Vec<_> = model
        .instances
        .iter()
        .filter(|inst| inst.cell_type == "DFF")
        .map(|ff| {
            ff.pins
                .iter()
                .find(|pin| pin.pin_name == "CLK")
                .unwrap()
                .signal
        })
        .collect();
    assert_eq!(
        clocks,
        [SequentialAigSignal::Clock, gated, gated, gated, gated]
    );
    let observed = model.sequential_gate_fn;
    assert_eq!(source.registers, observed.registers);
    assert_eq!(
        source.transition.get_signature(),
        observed.transition.get_signature()
    );
    assert_eq!(source.outputs, observed.outputs);
    assert_eq!(
        prove_gate_fn_equiv_with_backend(
            &source.transition,
            &observed.transition,
            GateFormalBackend::Cadical
        )?,
        EquivResult::Proved
    );

    // With only a QN cell, bypassing these muxes would expose four direct
    // input -> FF -> output paths that require a non-inverting FF.
    let qn = make_context(&CLOCK_GATE_LIB.replace(r#"function: "IQ""#, r#"function: "IQN""#))?;
    // Each preserved feedback mux needs one mapped D boundary output.
    assert_eq!(
        qn.sequential
            .prepare(qn.library(), &source, &sequential_options)
            .map_err(anyhow::Error::msg)?
            .graph()
            .outputs
            .len(),
        4
    );
    sequential_options.clock_gating = None;
    assert_eq!(
        context
            .sequential
            .prepare(context.library(), &source, &sequential_options)
            .map_err(anyhow::Error::msg)?
            .graph()
            .outputs
            .len(),
        4
    );
    Ok(())
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
            &SequentialMappingOptions::default(),
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
    let prepared = SequentialMappingCatalog::new(&library)
        .unwrap()
        .prepare(&library, &design, &SequentialMappingOptions::default())
        .unwrap();
    let mut parsed = read_gv_from_str(
        r#"module mapped(i0,i1,i2,i3);
        input i0,i1,i2,i3; endmodule"#,
    )
    .unwrap();
    validate_mapped_cells(&parsed, &library).unwrap();
    prepared.assemble(&mut parsed, "ABC").unwrap();
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
    let prepared = SequentialMappingCatalog::new(&library)
        .unwrap()
        .prepare(&library, &design, &SequentialMappingOptions::default())
        .unwrap();
    let mut parsed = read_gv_from_str("module mapped(); endmodule").unwrap();
    assert!(prepared.assemble(&mut parsed, "ABC").is_err());
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
    let sequential_options = SequentialMappingOptions::default();
    assert_eq!(
        sequential_options.register_selection,
        RegisterCellSelection::SmallestArea
    );
    assert!(!sequential_options.cleanup_registers);
    let gating = sequential_options
        .clock_gating
        .expect("gating is enabled by default");
    assert_eq!(gating.min_flops, 4);
    assert_eq!(gating.max_enable_depth, 8);
    let options = AbcMappingOptions::from_library(&library).unwrap();
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
    let raw = temp.path().join("cells.lib");
    std::fs::write(&raw, "library(test) {}").unwrap();
    let mut context = AbcMappingContext::new(
        AbcToolchain::new(fake_abc("explicit_constraints.py")).unwrap(),
        &[raw],
        library(),
    )
    .unwrap();
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
    let sequential_options = SequentialMappingOptions::default();
    let options = AbcMappingOptions::new(explicit.clone());
    assert_eq!(options.timeout, Duration::from_secs(180));
    assert_eq!(options.script, ABC_SPEED_SCRIPT);
    for load in [Some(0.004), None, Some(-1.0), Some(f64::NAN)] {
        context.library.default_output_load = load;
        let result = context
            .map_sequential(&design, &sequential_options, &options)
            .unwrap();
        assert_eq!(result.diagnostics.io_constraints.as_ref(), Some(&explicit));
    }
    context.library.reference_buffer_cell = None;
    context.library.units = None;
    let result = context
        .map_sequential(&design, &sequential_options, &options)
        .unwrap();
    assert_eq!(result.diagnostics.io_constraints.as_ref(), Some(&explicit));

    // A caller can also change already-resolved options before mapping.
    let mut overridden = AbcMappingOptions::from_library(&library()).unwrap();
    overridden.io_constraints = explicit;
    context
        .map_sequential(&design, &sequential_options, &overridden)
        .unwrap();
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
    let sequential_options = SequentialMappingOptions::default();
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
                    .map_sequential(&design(), &sequential_options, &options)
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
    let raw = temp.path().join("raw.lib");
    std::fs::write(&raw, "library(raw) {}").unwrap();
    let context = AbcMappingContext::new(
        AbcToolchain::new(fake_abc("missing_driver.py")).unwrap(),
        &[raw],
        library(),
    )
    .unwrap();
    let error = context
        .map_sequential(
            &design_with_logic(),
            &SequentialMappingOptions::default(),
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
    let raw = temp.path().join("cells.lib");
    std::fs::write(&raw, "library(test) {}").unwrap();
    let context = AbcMappingContext::new(
        AbcToolchain::new(fake_abc("custom_script.py")).unwrap(),
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
    let result = context
        .map_sequential(&design, &SequentialMappingOptions::default(), &options)
        .unwrap();
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
    let executable = temp.path().join("fake_abc");
    std::os::unix::fs::symlink(fake_abc("diagnostics.py"), &executable).unwrap();
    let raw = temp.path().join("raw.lib");
    std::fs::write(&raw, "library(raw) {}").unwrap();
    let context =
        AbcMappingContext::new(AbcToolchain::new(&executable).unwrap(), &[raw], library()).unwrap();
    let mut errors = Vec::new();
    let sequential_options = SequentialMappingOptions::default();
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
        std::fs::write(temp.path().join("output.v"), format!("{netlist}\n")).unwrap();
        let error = context
            .map_sequential(
                &design_with_logic(),
                &sequential_options,
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
    let abc_options = AbcMappingOptions::from_library(context.library()).unwrap();
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
                        &SequentialMappingOptions {
                            cleanup_registers,
                            ..Default::default()
                        },
                        &abc_options,
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
            &SequentialMappingOptions::default(),
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
    let raw = temp.path().join("raw.lib");
    std::fs::write(&raw, "library(raw) {}").unwrap();
    let context = AbcMappingContext::new(
        AbcToolchain::new("/usr/bin/true").unwrap(),
        &[&raw],
        library(),
    )
    .unwrap();
    let sequential_options = SequentialMappingOptions::default();
    let error = context
        .map_sequential(
            &design_with_logic(),
            &sequential_options,
            &AbcMappingOptions::from_library(context.library()).unwrap(),
        )
        .err()
        .unwrap();
    let message = error.to_string().replace(
        temp.path().canonicalize().unwrap().to_str().unwrap(),
        "<workdir>",
    );
    compare_golden_text(&message, "src/netlist/abc/missing_mapped_v.golden.txt");
    let context = AbcMappingContext::new(
        AbcToolchain::new(fake_abc("timeout.py")).unwrap(),
        &[raw],
        library(),
    )
    .unwrap();
    let error = context
        .map_sequential(
            &design_with_logic(),
            &sequential_options,
            &AbcMappingOptions {
                timeout: Duration::from_millis(30),
                ..AbcMappingOptions::from_library(context.library()).unwrap()
            },
        )
        .err()
        .unwrap();
    assert_eq!(error.kind, xlsynth::external_tool::ToolFailureKind::Timeout);
}
