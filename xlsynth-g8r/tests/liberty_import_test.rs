// SPDX-License-Identifier: Apache-2.0

//! Exercise the raw Liberty -> normalized model -> binary/text proto boundary.

use prost::Message;
use prost_reflect::DynamicMessage;
use std::collections::{HashMap, HashSet};
use std::io::Write;
use tempfile::NamedTempFile;
use xlsynth_g8r::liberty::cell_filter::CellFilterPolicy;
use xlsynth_g8r::liberty::cell_formula::parse_formula;
use xlsynth_g8r::liberty::descriptor::liberty_descriptor_pool;
use xlsynth_g8r::liberty::load::{
    count_timing_tables, decode_timing_table_summary_skip_values_from_bytes,
    load_library_from_path, load_library_with_timing_data_from_path,
};
use xlsynth_g8r::liberty::model::{
    LIBERTY_FORMAT_MAGIC, LIBERTY_FORMAT_MAGIC_V4, LIBERTY_FORMAT_MAGIC_V5, library_from_proto,
    library_to_proto,
};
use xlsynth_g8r::liberty::parser::{
    LibertyPayloadOptions, parse_liberty_files_with_payload_options,
};
use xlsynth_g8r::liberty::proto_info::liberty_proto_info_from_path;
use xlsynth_g8r::liberty_model::{Library, Pin, PinDirection, SequentialKind};
use xlsynth_g8r::liberty_proto::{Library as WireLibrary, UnsupportedFeature};
use xlsynth_g8r::netlist::gatefn_from_netlist::project_gatefn_from_netlist_and_liberty;
use xlsynth_g8r::netlist::io::read_gv_from_path;
use xlsynth_g8r::netlist::register_mapping::{RegisterCellSelection, RegisterMappingLibrary};

fn parse(text: &str, payload: LibertyPayloadOptions) -> Result<Library, String> {
    let mut file = NamedTempFile::new().unwrap();
    file.write_all(text.as_bytes()).unwrap();
    parse_liberty_files_with_payload_options(&[file.path()], payload)
}

/// Both serialized forms must preserve the exact normalized wire
/// representation.
fn roundtrip(library: Library) -> Library {
    let wire = library_to_proto(library).unwrap();
    let bytes = wire.encode_to_vec();
    let descriptor = liberty_descriptor_pool()
        .get_message_by_name("liberty.Library")
        .unwrap();
    let text = DynamicMessage::decode(descriptor, bytes.as_slice())
        .unwrap()
        .to_text_format();
    for payload in [bytes.as_slice(), text.as_bytes()] {
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(payload).unwrap();
        let loaded = load_library_from_path(file.path()).unwrap();
        assert_eq!(library_to_proto(loaded).unwrap(), wire);
    }
    library_from_proto(WireLibrary::decode(bytes.as_slice()).unwrap()).unwrap()
}

fn pin<'a>(library: &'a Library, cell: &str, pin: &str) -> &'a Pin {
    library
        .cells
        .iter()
        .find(|c| c.name == cell)
        .unwrap()
        .pins
        .iter()
        .find(|p| library.resolve_string(&p.name) == pin)
        .unwrap()
}

#[test]
fn size_only_presence_survives_binary_and_text_proto_roundtrips() {
    // The roundtrip helper checks that both serialized forms preserve the
    // normalized wire representation, including optional field presence.
    let library = roundtrip(
        parse(
            r#"
        library(test) {
            cell(SIZE_ONLY) { use_for_size_only: true; }
            cell(EXPLICIT_FALSE) { use_for_size_only: false; }
            cell(ABSENT) {}
        }
        "#,
            LibertyPayloadOptions::all(),
        )
        .unwrap(),
    );
    for (name, expected) in [
        ("SIZE_ONLY", Some(true)),
        ("EXPLICIT_FALSE", Some(false)),
        ("ABSENT", None),
    ] {
        let cell = library.cells.iter().find(|cell| cell.name == name).unwrap();
        assert_eq!(cell.use_for_size_only, expected, "cell: {name}");
    }
}

#[test]
fn scan_equations_and_explicit_clock_annotations_roundtrip() {
    let library = roundtrip(
        parse(
            r#"
        library (test) {
            cell (SDFF) {
                area: 1;
                pin (CLK) { clock: true; }
                pin (D) { direction: input; }
                pin (SI) { direction: input; }
                pin (SE) { direction: input; }
                pin (QN) { direction: output; function: "IQN"; }
                ff (IQ, IQN) { clocked_on: "CLK"; next_state: "(D * SE') + (SI * SE)"; }
                test_cell () {
                    pin (SI) { direction: input; signal_type: test_scan_in; }
                    pin (SE) { direction: input; signal_type: test_scan_enable; }
                }
            }
        }
    "#,
            LibertyPayloadOptions::all(),
        )
        .unwrap(),
    );
    let cell = &library.cells[0];
    cell.validate_functional_semantics().unwrap();
    assert_eq!(
        pin(&library, "SDFF", "CLK").direction,
        PinDirection::Input as i32
    );
    assert!(pin(&library, "SDFF", "CLK").is_clocking_pin);
    assert_eq!(cell.pins.len(), 5);
    let formula = parse_formula(&cell.sequential[0].next_state).unwrap();
    for data in [false, true] {
        for scan in [false, true] {
            for enable in [false, true] {
                assert_eq!(
                    formula.evaluate_partial(&HashMap::from([
                        ("D".into(), data),
                        ("SI".into(), scan),
                        ("SE".into(), enable)
                    ])),
                    Some(if enable { scan } else { data })
                );
            }
        }
    }
}

#[test]
fn power_down_metadata_does_not_change_powered_on_proto_in_any_payload_mode() {
    for (kind, controls) in [
        ("", ""),
        ("ff", r#"clocked_on: "CLK"; next_state: "D";"#),
        ("latch", r#"enable: "CLK"; data_in: "D";"#),
        ("ff_bank", r#"clocked_on: "CLK"; next_state: "D";"#),
        ("latch_bank", r#"enable: "CLK"; data_in: "D";"#),
    ] {
        let source = |power: &str| {
            let bank = kind.ends_with("_bank");
            let output = if kind.is_empty() { "D" } else { "IQ" };
            let pins = if bank {
                format!(
                    r#"bundle(D) {{ members(D1,D0); direction: input; }}
                       bundle(Q) {{ members(Q1,Q0); direction: output; function: "{output}"; {power} }}"#
                )
            } else {
                format!(
                    r#"pin(D) {{ direction: input; }}
                       pin(Q) {{ direction: output; function: "{output}"; {power} }}"#
                )
            };
            let sequential = if kind.is_empty() {
                String::new()
            } else {
                let width = if bank { ",2" } else { "" };
                format!("{kind}(IQ,IQN{width}) {{ {controls} {power} }}")
            };
            format!(
                r#"library(test) {{ cell(CELL) {{
                    pg_pin(VDD) {{ pg_type: primary_power; }}
                    pg_pin(VSS) {{ pg_type: primary_ground; }}
                    pin(CLK) {{ direction: input; }}
                    {pins}
                    {sequential}
                }} }}"#
            )
        };
        for include_timing in [false, true] {
            for include_power in [false, true] {
                let payload = LibertyPayloadOptions {
                    include_timing,
                    include_power,
                };
                let expected = parse(&source(""), payload).unwrap();
                let actual = roundtrip(
                    parse(&source(r#"power_down_function: "!VDD + VSS";"#), payload).unwrap(),
                );
                assert!(actual.cells[0].unsupported_semantics.is_empty());
                actual.cells[0].validate_functional_semantics().unwrap();
                assert_eq!(
                    library_to_proto(actual).unwrap(),
                    library_to_proto(expected).unwrap(),
                    "power-down metadata changed {kind:?} in payload mode {payload:?}"
                );
            }
        }
    }
}

#[test]
fn unsupported_semantics_survive_all_payload_modes_and_proto_forms() {
    let cases = [
        (
            "pin(Q) { direction: output; function: \"IQ\"; three_state: \"!OE\"; }",
            UnsupportedFeature::ThreeState,
            "pin(Q).three_state",
        ),
        (
            "pin(Q) { direction: output; x_function: \"A\"; }",
            UnsupportedFeature::XFunction,
            "pin(Q).x_function",
        ),
        (
            "ff(IQ, IQN) { clocked_on: \"CLK\"; clocked_on_also: \"CLK2\"; next_state: \"D\"; }",
            UnsupportedFeature::SecondaryClock,
            "ff(IQ,IQN).clocked_on_also",
        ),
        (
            "latch(IQ, IQN) { enable: \"G\"; enable_also: \"G2\"; data_in: \"D\"; }",
            UnsupportedFeature::SecondaryEnable,
            "latch(IQ,IQN).enable_also",
        ),
        (
            "ff(IQ, IQN) { clocked_on: \"CLK\"; next_state: \"D\"; clear: \"R\"; preset: \"S\"; clear_preset_var1: H; }",
            UnsupportedFeature::SimultaneousClearPreset,
            "ff(IQ,IQN).clear_preset_var1",
        ),
        (
            "statetable(\"A\", \"IQ\") { table: \"0 : - : 0\"; }",
            UnsupportedFeature::StateTable,
            "statetable(A,IQ)",
        ),
        (
            "pin(Q) { direction: output; internal_node: IQ; }",
            UnsupportedFeature::InternalState,
            "pin(Q).internal_node",
        ),
        (
            "pin(Q) { direction: output; state_function: \"IQ\"; }",
            UnsupportedFeature::InternalState,
            "pin(Q).state_function",
        ),
        (
            "bus(D) { direction: input; }",
            UnsupportedFeature::Bus,
            "bus(D)",
        ),
        (
            "ff(IQ, IQN) { clocked_on: \"CLK\"; next_state: \"D\"; future_control: \"C\"; }",
            UnsupportedFeature::SequentialAttribute,
            "ff(IQ,IQN).future_control",
        ),
        (
            "ff(IQ, IQN) { clocked_on: \"CLK\"; next_state: \"D\"; future_group() {} }",
            UnsupportedFeature::SequentialGroup,
            "ff(IQ,IQN).future_group()",
        ),
        (
            r#"ff(IQ, IQN) { clocked_on: "CLK"; next_state: "D";
                power_down_function: "!VDD + VSS"; clocked_on_also: "!CLK"; }"#,
            UnsupportedFeature::SecondaryClock,
            "ff(IQ,IQN).clocked_on_also",
        ),
        (
            r#"latch(IQ, IQN) { enable: "G"; data_in: "D";
                power_down_function: "!VDD + VSS"; enable_also: "G2"; }"#,
            UnsupportedFeature::SecondaryEnable,
            "latch(IQ,IQN).enable_also",
        ),
        (
            r#"pin(Q) { direction: output; function: "IQ";
                power_down_function: "!VDD + VSS"; three_state: "!OE"; }"#,
            UnsupportedFeature::ThreeState,
            "pin(Q).three_state",
        ),
        (
            r#"ff(IQ, IQN) { clocked_on: "CLK"; next_state: "D";
                power_down_function: "!VDD + VSS"; future_control: "C"; }"#,
            UnsupportedFeature::SequentialAttribute,
            "ff(IQ,IQN).future_control",
        ),
    ];
    for include_timing in [false, true] {
        for include_power in [false, true] {
            for (body, feature, location) in cases {
                let library = roundtrip(parse(&format!("library(test) {{ cell(SPECIAL) {{ {body} }} cell(BUF) {{ pin(A) {{}} pin(Y) {{ direction: output; function: \"A\"; }} }} }}"), LibertyPayloadOptions { include_timing, include_power }).unwrap());
                let cell = library
                    .cells
                    .iter()
                    .find(|cell| cell.name == "SPECIAL")
                    .unwrap();
                assert_eq!(cell.unsupported_semantics.len(), 1);
                assert_eq!(cell.unsupported_semantics[0].feature, feature as i32);
                assert_eq!(cell.unsupported_semantics[0].location, location);
                assert!(cell.validate_functional_semantics().is_err());
                library
                    .cells
                    .iter()
                    .find(|cell| cell.name == "BUF")
                    .unwrap()
                    .validate_functional_semantics()
                    .unwrap();
            }
        }
    }
}

#[test]
fn driver_types_survive_import_and_are_rejected_for_boolean_use() {
    let cases = [
        ("driver_type: pull_up;", "Y"),
        ("driver_type: pull_down;", "Y"),
        ("driver_type: open_drain;", "Y"),
        ("driver_type: open_source;", "Y"),
        ("driver_type: bus_hold;", "Y"),
        ("driver_type: resistive;", "Y"),
        ("driver_type: resistive_0;", "Y"),
        ("driver_type: resistive_1;", "Y"),
        ("driver_type: \"open_drain pull_up\";", "Y"),
        ("driver_type: future_extension;", "Y"),
        ("driver_type: pull_up;", "A"),
        ("pull_up_function: \"A\";", "Y"),
        ("pull_down_function: \"A\";", "Y"),
        ("bus_hold_function: \"A\";", "Y"),
        ("open_drain_function: \"A\";", "Y"),
        ("open_source_function: \"A\";", "Y"),
        ("resistive_function: \"A\";", "Y"),
        ("resistive_0_function: \"A\";", "Y"),
        ("resistive_1_function: \"A\";", "Y"),
    ];
    for include_timing in [false, true] {
        for include_power in [false, true] {
            for (attribute, pin) in cases {
                let input_attribute = if pin == "A" { attribute } else { "" };
                let output_attribute = if pin == "Y" { attribute } else { "" };
                let source = format!(
                    r#"library(test) {{
                        cell(UNSAFE) {{ area: 1; driver_type: "4";
                            pin(A) {{ direction: input; {input_attribute} }}
                            pin(B) {{ direction: input; }}
                            pin(Y) {{ direction: output; function: "A * B"; {output_attribute} }}
                        }}
                        cell(SAFE) {{ area: 2; driver_type: "4";
                            pin(A, B) {{ direction: input; }}
                            pin(Y) {{ direction: output; function: "A * B"; }}
                        }}
                    }}"#
                );
                let library = roundtrip(
                    parse(
                        &source,
                        LibertyPayloadOptions {
                            include_timing,
                            include_power,
                        },
                    )
                    .unwrap(),
                );
                let unsafe_cell = library
                    .cells
                    .iter()
                    .find(|cell| cell.name == "UNSAFE")
                    .unwrap();
                assert_eq!(unsafe_cell.unsupported_semantics.len(), 1, "{attribute}");
                let issue = &unsafe_cell.unsupported_semantics[0];
                assert_eq!(issue.feature, UnsupportedFeature::DriverType as i32);
                let attr_name = attribute.split(':').next().unwrap();
                assert_eq!(issue.location, format!("pin({pin}).{attr_name}"));
                assert!(unsafe_cell.validate_functional_semantics().is_err());
                let safe_cell = library
                    .cells
                    .iter()
                    .find(|cell| cell.name == "SAFE")
                    .unwrap();
                safe_cell.validate_functional_semantics().unwrap();
            }
        }
    }
}

#[test]
fn internal_and_bidirectional_pins_are_preserved_but_not_boolean_cells() {
    for (direction, expected) in [
        ("internal", PinDirection::Internal),
        ("inout", PinDirection::Inout),
    ] {
        let library = roundtrip(
            parse(
                &format!(
                    "library(test) {{ cell(SPECIAL) {{ pin(P) {{ direction: {direction}; }} }} }}"
                ),
                LibertyPayloadOptions::all(),
            )
            .unwrap(),
        );
        assert_eq!(library.cells[0].pins[0].direction, expected as i32);
        assert!(library.cells[0].validate_functional_semantics().is_err());
    }
}

#[test]
fn dual_async_controls_are_rejected_with_or_without_override_attributes() {
    for (kind, controls) in [
        ("ff", r#"clocked_on: "CLK"; next_state: "D";"#),
        ("latch", r#"enable: "CLK"; data_in: "D";"#),
        ("ff_bank", r#"clocked_on: "CLK"; next_state: "D";"#),
        ("latch_bank", r#"enable: "CLK"; data_in: "D";"#),
    ] {
        for overrides in [
            "",
            "clear_preset_var1: H;",
            "clear_preset_var2: L;",
            "clear_preset_var1: H; clear_preset_var2: H;",
        ] {
            let bank = kind.ends_with("_bank");
            let pins = if bank {
                r#"bundle(D) { members(D0,D1); direction: input; }
                   bundle(Q) { members(Q0,Q1); direction: output; function: "IQ"; }"#
            } else {
                r#"pin(D) { direction: input; } pin(Q) { direction: output; function: "IQ"; }"#
            };
            let width = if bank { ",2" } else { "" };
            let mut library = roundtrip(
                parse(
                    &format!(
                        r#"library(test) {{ cell(STORAGE) {{
                {pins} pin(CLK,R,S) {{ direction: input; }}
                {kind}(IQ,IQN{width}) {{ {controls} clear: "R"; preset: "S"; {overrides} }}
            }} }}"#
                    ),
                    LibertyPayloadOptions::all(),
                )
                .unwrap(),
            );
            let cell = &mut library.cells[0];
            assert!(
                cell.unsupported_semantics.iter().any(
                    |issue| issue.feature == UnsupportedFeature::SimultaneousClearPreset as i32
                )
            );
            assert!(cell.validate_functional_semantics().is_err());
            // Manually constructed normalized/proto models must not bypass the
            // guard.
            cell.unsupported_semantics.clear();
            assert!(cell.validate_functional_semantics().is_err());
        }
    }
}

#[test]
fn banks_expand_in_declared_bundle_order_with_scalar_controls() {
    for width in [2usize, 4, 8] {
        let inputs = (0..width)
            .rev()
            .map(|i| format!("D{i}"))
            .collect::<Vec<_>>();
        let outputs = (0..width).map(|i| format!("Q{i}")).collect::<Vec<_>>();
        let complements = (0..width).map(|i| format!("QN{i}")).collect::<Vec<_>>();
        let library = roundtrip(parse(&format!(r#"
            library(test) {{
                default_input_pin_cap: 0.3;
                cell(DFF_BANK) {{
                    area: 12;
                    bundle(D) {{ members({}); direction: input; capacitance: 0.2; pin({}) {{ capacitance: 0; }} }}
                    bundle(Q) {{ members({}); direction: output; function: "IQ";
                        timing() {{ related_pin: "D"; timing_type: combinational; cell_rise(scalar) {{ values("1"); }} }}
                    }}
                    bundle(QN) {{ members({}); direction: output; function: "IQN"; }}
                    pin(CLK) {{ clock: true; }}
                    pin(EN, RST) {{ direction: input; }}
                    ff_bank(IQ, IQN, {width}) {{ clocked_on: "CLK"; next_state: "!RST * ((EN * D) + (!EN * IQ))"; }}
                }}
            }}
        "#, inputs.join(","), inputs[0], outputs.join(","), complements.join(",")), LibertyPayloadOptions::all()).unwrap());
        let cell = &library.cells[0];
        cell.validate_functional_semantics().unwrap();
        assert_eq!(cell.area, 12.0);
        assert_eq!(cell.sequential.len(), width);
        assert_eq!(cell.pins.len(), 3 * width + 3);
        assert_eq!(pin(&library, "DFF_BANK", &inputs[0]).capacitance, Some(0.0));
        assert_eq!(pin(&library, "DFF_BANK", &inputs[1]).capacitance, Some(0.2));
        assert_eq!(pin(&library, "DFF_BANK", "CLK").capacitance, Some(0.3));
        for (bit, seq) in cell.sequential.iter().enumerate() {
            assert_eq!(seq.kind, SequentialKind::Ff as i32);
            assert_eq!(seq.clock_expr, "CLK");
            let q = pin(&library, "DFF_BANK", &outputs[bit]);
            assert_eq!(
                library.resolve_string(&q.timing_arcs[0].related_pin),
                inputs[bit]
            );
            let qn = pin(&library, "DFF_BANK", &complements[bit]);
            let next = parse_formula(&seq.next_state).unwrap();
            for data in [false, true] {
                for state in [false, true] {
                    for en in [false, true] {
                        for reset in [false, true] {
                            let bindings = HashMap::from([
                                (inputs[bit].clone(), data),
                                (seq.state_var.clone(), state),
                                (seq.complementary_state_var.clone().unwrap(), !state),
                                ("EN".into(), en),
                                ("RST".into(), reset),
                            ]);
                            assert_eq!(
                                next.evaluate_partial(&bindings),
                                Some(!reset && if en { data } else { state })
                            );
                            assert_eq!(
                                parse_formula(library.resolve_string(&q.function))
                                    .unwrap()
                                    .evaluate_partial(&bindings),
                                Some(state)
                            );
                            assert_eq!(
                                parse_formula(library.resolve_string(&qn.function))
                                    .unwrap()
                                    .evaluate_partial(&bindings),
                                Some(!state)
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn incomplete_sequential_equations_are_rejected_by_kind() {
    for (kind, equations) in [
        ("ff", r#"next_state: "D";"#),
        ("ff", r#"clocked_on: "CLK";"#),
        ("ff", ""),
        ("ff_bank", r#"next_state: "D";"#),
        ("ff_bank", r#"clocked_on: "CLK";"#),
        ("latch", r#"data_in: "D";"#),
        ("latch", r#"enable: "CLK";"#),
        ("latch", ""),
        ("latch_bank", r#"data_in: "D";"#),
        ("latch_bank", r#"enable: "CLK";"#),
    ] {
        let width = if kind.ends_with("_bank") { ",2" } else { "" };
        // A clock pin annotation must not rescue a missing FF clock equation.
        let source = format!(
            r#"library(test) {{ cell(BAD) {{
            pin(CLK) {{ clock: true; }} pin(D) {{ direction: input; }}
            {kind}(IQ,IQN{width}) {{ {equations} }}
        }} }}"#
        );
        assert!(
            parse(&source, LibertyPayloadOptions::all()).is_err(),
            "{kind}: {equations}"
        );
    }
}

#[test]
fn set_reset_latches_are_retained_without_poisoning_valid_cells() {
    for kind in ["latch", "latch_bank"] {
        let bank = kind.ends_with("_bank");
        let width = if bank { ",2" } else { "" };
        let output = if bank {
            r#"bundle(Q) { members(Q0,Q1); direction: output; function: "IQ"; }"#
        } else {
            r#"pin(Q) { direction: output; function: "IQ"; }"#
        };
        let library = roundtrip(
            parse(
                &format!(
                    r#"library(test) {{
            cell(SR) {{ pin(R,S) {{ direction: input; }} {output}
                {kind}(IQ,IQN{width}) {{ clear: "R"; preset: "S";
                    clear_preset_var1: L; clear_preset_var2: L;
                }}
            }}
            cell(INV) {{ pin(A) {{}} pin(Y) {{ direction: output; function: "!A"; }} }}
        }}"#
                ),
                LibertyPayloadOptions::all(),
            )
            .unwrap(),
        );
        assert_eq!(library.cells.len(), 2);
        let cell = library.cells.iter().find(|cell| cell.name == "SR").unwrap();
        assert_eq!(cell.sequential.len(), if bank { 2 } else { 1 });
        assert!(cell.validate_functional_semantics().is_err());
        for seq in &cell.sequential {
            assert!(seq.next_state.is_empty());
            assert!(seq.clock_expr.is_empty());
            assert_eq!(seq.clear_expr, "R");
            assert_eq!(seq.preset_expr, "S");
        }
        library
            .cells
            .iter()
            .find(|cell| cell.name == "INV")
            .unwrap()
            .validate_functional_semantics()
            .unwrap();
    }
}

#[test]
fn excessive_bank_expansion_is_rejected_before_allocation() {
    for declarations in [
        r#"ff_bank(IQ,IQN,4294967295) { clocked_on: "CLK"; next_state: "D"; }"#,
        r#"latch_bank(IQ,IQN,4294967295) { enable: "CLK"; data_in: "D"; }"#,
        r#"ff_bank(IQ,IQN,2048) { clocked_on: "CLK"; next_state: "D"; }
           ff_bank(JQ,JQN,2049) { clocked_on: "CLK"; next_state: "D"; }"#,
    ] {
        let error = parse(
            &format!("library(test) {{ cell(BANK) {{ {declarations} }} }}"),
            LibertyPayloadOptions::all(),
        )
        .unwrap_err();
        assert_eq!(
            error,
            "cell 'BANK': bank expansion exceeds the per-cell limit of 4096 state bits"
        );
    }
}

#[test]
fn latch_banks_preserve_unsupported_dual_controls_and_enable() {
    let library = roundtrip(
        parse(
            r#"
        library(test) { cell(LATCH_BANK) {
            bundle(D) { members(D1,D0); direction: input; }
            bundle(Q) { members(Q1,Q0); direction: output; function: "IQ"; }
            pin(G,R,S) { direction: input; }
            latch_bank(IQ,IQN,2) { enable: "G"; data_in: "D"; clear: "R"; preset: "S"; }
        } }
    "#,
            LibertyPayloadOptions::all(),
        )
        .unwrap(),
    );
    assert_eq!(library.cells[0].sequential.len(), 2);
    assert!(library.cells[0].validate_functional_semantics().is_err());
    for (seq, data) in library.cells[0].sequential.iter().zip(["D1", "D0"]) {
        assert_eq!(seq.kind, SequentialKind::Latch as i32);
        assert_eq!(seq.clock_expr, "G");
        assert_eq!(seq.next_state, data);
        assert_eq!(seq.clear_expr, "R");
        assert_eq!(seq.preset_expr, "S");
    }
}

#[test]
fn combinational_bundles_expand_formulas_and_power_conditions() {
    let library = roundtrip(
        parse(
            r#"
        library(test) { cell(INV_BANK) {
            bundle(A) { members("A[3]", "A[2]"); direction: input; fanout_load: 2; }
            bundle(Y) { members(Y0,Y1); direction: output; function: "A'";
                max_transition: 0.5; max_fanout: 8;
                internal_power() { related_pin: "A"; when: "A";
                    rise_power(scalar) { values("0.1"); }
                }
                pin(Y0) { max_transition: 0.25; }
            }
        } }
    "#,
            LibertyPayloadOptions::all(),
        )
        .unwrap(),
    );
    for (index, name) in ["A[3]", "A[2]"].into_iter().enumerate() {
        let output = pin(&library, "INV_BANK", &format!("Y{index}"));
        assert_eq!(
            output.max_transition,
            Some(if index == 0 { 0.25 } else { 0.5 })
        );
        assert_eq!(output.max_fanout, Some(8.0));
        assert_eq!(pin(&library, "INV_BANK", name).fanout_load, Some(2.0));
        let power = &output.internal_power[0];
        assert_eq!(library.resolve_string(&power.related_pins[0]), name);
        for value in [false, true] {
            let bindings = HashMap::from([(name.into(), value)]);
            assert_eq!(
                parse_formula(library.resolve_string(&output.function))
                    .unwrap()
                    .evaluate_partial(&bindings),
                Some(!value)
            );
            assert_eq!(
                parse_formula(library.resolve_string(&power.when))
                    .unwrap()
                    .evaluate_partial(&bindings),
                Some(value)
            );
        }
    }
}

#[test]
fn malformed_or_mismatched_vectors_are_rejected() {
    for body in [
        "bundle(D) { direction: input; }",
        "bundle(D) { members(D0,D0); }",
        "bundle(D) { members(D0,D1); pin(D2) {} }",
        "bundle(D) { members(D0,D1); direction: input; pin(D0) { direction: output; } }",
        "bundle(D) { members(D0,D1); timing() {} pin(D0) { timing() {} } }",
        "bundle(D) { members(D0,D1); } bundle(Q) { members(Q0); function: \"D\"; }",
        "ff_bank(IQ,IQN,1.5) { clocked_on: \"CLK\"; next_state: \"D\"; }",
        "ff() { clocked_on: \"CLK\"; next_state: \"D\"; }",
        "ff(IQ,2) { clocked_on: \"CLK\"; next_state: \"D\"; }",
        "ff(IQ) { clocked_on: \"CLK\"; next_state: 123; }",
        "pin(A) { direction: sideways; }",
        "pin(Y) { direction: output; function: 123; }",
        "area: negative;",
    ] {
        assert!(
            parse(
                &format!("library(test) {{ cell(BAD) {{ {body} }} }}"),
                LibertyPayloadOptions::all()
            )
            .is_err(),
            "{body}"
        );
    }
}

#[test]
fn bundle_members_can_define_multiple_noninherited_timing_groups() {
    let library = roundtrip(
        parse(
            r#"
        library(test) { cell(DFF_BANK) {
            pin(CLK) { clock: true; }
            bundle(D) { members(D0,D1); direction: input;
                pin(D0) {
                    timing() { related_pin: "CLK"; timing_type: setup_rising;
                        rise_constraint(scalar) { values("0.1"); }
                    }
                    timing() { related_pin: "CLK"; timing_type: hold_rising;
                        rise_constraint(scalar) { values("0.2"); }
                    }
                }
            }
            bundle(Q) { members(Q0,Q1); direction: output; function: "IQ"; }
            ff_bank(IQ,IQN,2) { clocked_on: "CLK"; next_state: "D"; }
        } }
    "#,
            LibertyPayloadOptions::all(),
        )
        .unwrap(),
    );
    assert_eq!(pin(&library, "DFF_BANK", "D0").timing_arcs.len(), 2);
    assert_eq!(pin(&library, "DFF_BANK", "D1").timing_arcs.len(), 0);
}

#[test]
fn electrical_defaults_do_not_leak_between_merged_libraries() {
    let mut files = Vec::new();
    for (name, value) in [("FIRST", 1), ("SECOND", 2)] {
        let mut file = NamedTempFile::new().unwrap();
        write!(file, r#"library(test) {{
            time_unit: "1ns"; capacitive_load_unit(1,pf);
            default_input_pin_cap: {value}; default_output_pin_cap: {value};
            default_max_capacitance: {value}; default_max_transition: {value};
            default_max_fanout: {value}; default_fanout_load: {value};
            cell({name}) {{ pin(A) {{}} pin(B) {{ capacitance: 0; fanout_load: 0; }}
                pin(Y) {{ direction: output; function: "A"; }}
                pin(Z) {{ direction: output; function: "A"; max_transition: 0; max_fanout: 0; max_capacitance: 0; }} }}
        }}"#).unwrap();
        files.push(file);
    }
    let paths = files.iter().map(|file| file.path()).collect::<Vec<_>>();
    let library = roundtrip(
        parse_liberty_files_with_payload_options(&paths, LibertyPayloadOptions::all()).unwrap(),
    );
    for (name, value) in [("FIRST", 1.0), ("SECOND", 2.0)] {
        let input = pin(&library, name, "A");
        assert_eq!(input.capacitance, Some(value));
        assert_eq!(input.fanout_load, Some(value));
        assert_eq!(input.max_fanout, None);
        let output = pin(&library, name, "Y");
        assert_eq!(output.capacitance, Some(value));
        assert_eq!(output.max_capacitance, Some(value));
        assert_eq!(output.max_transition, Some(value));
        assert_eq!(output.max_fanout, Some(value));
        assert_eq!(output.fanout_load, None);
        assert_eq!(pin(&library, name, "B").capacitance, Some(0.0));
        assert_eq!(pin(&library, name, "B").fanout_load, Some(0.0));
        let overridden = pin(&library, name, "Z");
        assert_eq!(overridden.max_transition, Some(0.0));
        assert_eq!(overridden.max_fanout, Some(0.0));
        assert_eq!(overridden.max_capacitance, Some(0.0));
    }
}

#[test]
fn invalid_electrical_values_and_incompatible_protos_are_rejected() {
    for attribute in [
        "capacitance",
        "rise_capacitance",
        "fall_capacitance",
        "max_capacitance",
        "max_transition",
        "max_fanout",
        "fanout_load",
    ] {
        for value in ["-1", "bad"] {
            assert!(
                parse(
                    &format!(
                        "library(test) {{ cell(BUF) {{ pin(A) {{ {attribute}: {value}; }} }} }}"
                    ),
                    LibertyPayloadOptions::all()
                )
                .is_err()
            );
        }
    }
    let library = parse(
        "library(test) { cell(BUF) { pin(A) {} } }",
        LibertyPayloadOptions::all(),
    )
    .unwrap();
    let wire = library_to_proto(library).unwrap();
    for value in [-1.0, f64::NAN, f64::INFINITY] {
        let mut bad = wire.clone();
        bad.cells[0].pins[0].max_transition = Some(value);
        assert!(library_from_proto(bad).is_err());
    }
    let mut unknown = wire.clone();
    unknown.cells[0]
        .unsupported_semantics
        .push(xlsynth_g8r::liberty_proto::UnsupportedSemantics {
            feature: 999,
            location: "ff(IQ).future_control".into(),
        });
    assert!(library_from_proto(unknown).is_err());
    let mut legacy = wire.clone();
    legacy.format_magic = 0x4c49_4256_3350_524f;
    let error = library_from_proto(legacy).unwrap_err().to_string();
    assert!(error.contains("regenerate from source Liberty"), "{error}");
}

#[test]
fn legacy_protos_remain_inspectable_and_preserve_distinct_coverage() {
    let source = r#"
        library(test) {
            cell(BUF) {
                pin(A) { direction: input; }
                pin(Y) { direction: output; function: "A";
                    timing() { related_pin: "A"; timing_type: combinational;
                        cell_rise(scalar) { values("1"); }
                    }
                }
            }
            cell(FF) {
                area: 1;
                pin(D) { direction: input; }
                pin(CLK) { direction: input; clock: true; }
                pin(Q) { direction: output; function: "IQ"; }
                ff(IQ,IQN) { clocked_on: "CLK"; next_state: "D"; }
            }
        }
    "#;
    let current = library_to_proto(parse(source, LibertyPayloadOptions::all()).unwrap()).unwrap();
    let mut netlist_file = NamedTempFile::new().unwrap();
    write!(
        netlist_file,
        "module top(input a, output y); BUF u(.A(a),.Y(y)); endmodule"
    )
    .unwrap();
    let parsed = read_gv_from_path(netlist_file.path()).unwrap();

    for (magic, driver_unknown) in [
        (LIBERTY_FORMAT_MAGIC_V4, true),
        (LIBERTY_FORMAT_MAGIC_V5, false),
    ] {
        let mut legacy = current.clone();
        legacy.format_magic = magic;
        let binary = legacy.encode_to_vec();
        let descriptor = liberty_descriptor_pool()
            .get_message_by_name("liberty.Library")
            .unwrap();
        let text = DynamicMessage::decode(descriptor, binary.as_slice())
            .unwrap()
            .to_text_format();
        for payload in [binary.as_slice(), text.as_bytes()] {
            let mut file = NamedTempFile::new().unwrap();
            file.write_all(payload).unwrap();
            let mut library = load_library_with_timing_data_from_path(file.path()).unwrap();
            assert_eq!(count_timing_tables(&library), 1);
            let info = liberty_proto_info_from_path(file.path()).unwrap();
            assert_eq!(info.cells, 2);
            assert_eq!(info.timing_arcs, 1);
            let summary =
                decode_timing_table_summary_skip_values_from_bytes(payload, "test").unwrap();
            assert_eq!(summary.cells, 2);
            assert_eq!(summary.timing_tables, 1);

            let boolean = project_gatefn_from_netlist_and_liberty(
                &parsed.modules[0],
                &parsed.nets,
                &parsed.interner,
                &library,
                &HashSet::new(),
                &HashSet::new(),
            );
            if driver_unknown {
                assert!(
                    boolean
                        .unwrap_err()
                        .contains("driver_type source coverage is unknown")
                );
            } else {
                assert!(boolean.is_ok());
            }
            let catalog = RegisterMappingLibrary::new(&library).unwrap();
            let error = catalog
                .select(&RegisterCellSelection::FixedCell("FF".into()))
                .unwrap_err();
            if driver_unknown {
                assert!(
                    error.contains("driver_type source coverage is unknown"),
                    "{error}"
                );
            } else {
                assert!(
                    error.contains("unknown use_for_size_only source coverage"),
                    "{error}"
                );
            }

            CellFilterPolicy::parse("include .*", "test")
                .unwrap()
                .apply(&mut library);
            let reencoded = library_to_proto(library).unwrap();
            assert_eq!(reencoded.format_magic, LIBERTY_FORMAT_MAGIC);
            for cell in &reencoded.cells {
                let features = cell
                    .unsupported_semantics
                    .iter()
                    .map(|x| x.feature)
                    .collect::<Vec<_>>();
                let mut expected = Vec::new();
                if driver_unknown {
                    expected.push(UnsupportedFeature::IncompleteDriverTypeCoverage as i32);
                }
                expected.push(UnsupportedFeature::IncompleteSizeOnlyCoverage as i32);
                assert_eq!(features, expected);
            }
            let roundtripped = roundtrip(library_from_proto(reencoded).unwrap());
            for cell in &roundtripped.cells {
                assert_eq!(
                    cell.validate_functional_semantics().is_err(),
                    driver_unknown
                );
                assert!(cell.validate_mapping_policy_coverage().is_err());
            }
        }
    }
}

#[test]
fn boolean_netlist_projection_rejects_tristate_behavior() {
    let library = roundtrip(
        parse(
            r#"library(test) { cell(BUF) {
        pin(A,EN) { direction: input; }
        pin(Y) { direction: output; function: "A"; three_state: "!EN"; }
    } }"#,
            LibertyPayloadOptions::all(),
        )
        .unwrap(),
    );
    let mut file = NamedTempFile::new().unwrap();
    write!(
        file,
        "module top(input a, input en, output y); BUF u(.A(a),.EN(en),.Y(y)); endmodule"
    )
    .unwrap();
    let parsed = read_gv_from_path(file.path()).unwrap();
    assert!(
        project_gatefn_from_netlist_and_liberty(
            &parsed.modules[0],
            &parsed.nets,
            &parsed.interner,
            &library,
            &HashSet::new(),
            &HashSet::new()
        )
        .is_err()
    );
}
