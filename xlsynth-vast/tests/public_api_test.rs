// SPDX-License-Identifier: Apache-2.0

//! Standalone public API and ownership-model smoke tests.

use xlsynth_vast::{
    Expr, LiteralFormat, LogicRef, ModulePort, ModulePortDirection, VastDataType, VastFile,
    VastFileType, VastModule,
};

#[test]
fn standalone_file_emits_verilog_without_an_xls_dependency() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("passthrough");
    let data_type = file.make_bit_vector_type(8, false);
    let input = file.add_input(module, "input_data", &data_type);
    let output = file.add_output(module, "output_data", &data_type);
    let assignment = file.make_continuous_assignment(&output.to_expr(), &input.to_expr());
    file.add_member_continuous_assignment(module, assignment);

    let expected = r#"module passthrough(
  input wire [7:0] input_data,
  output wire [7:0] output_data
);
  assign output_data = input_data;
endmodule
"#;
    assert_eq!(file.emit(), expected);
}

#[test]
fn previously_obtained_port_handles_survive_later_port_insertions() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("stable_ports");
    let scalar = file.make_scalar_type();
    let first_logic = file.add_input(module, "first", &scalar);
    let first_port = file.module_ports(module)[0];

    for index in 0..128 {
        file.add_output(module, &format!("output_{index}"), &scalar);
    }

    assert_eq!(file.port_name(first_port), "first");
    assert_eq!(file.port_direction(first_port), ModulePortDirection::Input);
    assert_eq!(file.port_logic_ref(first_port), first_logic);
    assert_eq!(file.module_ports(module).len(), 129);
}

#[test]
fn literals_support_values_wider_than_rust_primitive_integers() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let literal = file
        .make_literal(
            "bits[256]:340282366920938463463374607431768211455",
            &LiteralFormat::UnsignedDecimal,
        )
        .expect("a 128-bit integer fits in a 256-bit literal");

    assert_eq!(
        file.emit_expression(&literal),
        "256'd340282366920938463463374607431768211455"
    );
}

#[test]
fn wide_default_literals_do_not_lose_their_declared_bit_width() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let literal = file
        .make_literal("bits[96]:42", &LiteralFormat::Default)
        .expect("wide default literals remain representable");

    assert_eq!(file.emit_expression(&literal), "96'd42");
}

#[test]
fn signed_decimal_literals_place_negative_signs_before_the_sized_number() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let literal = file
        .make_literal("bits[8]:-1", &LiteralFormat::SignedDecimal)
        .expect("a negative value has a valid two's-complement representation");

    assert_eq!(file.emit_expression(&literal), "-8'sd1");
}

#[test]
fn large_constant_indices_match_the_existing_sized_hex_spelling() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("large_indices");
    let data_type = file.make_bit_vector_type(8, false);
    let signal = file.add_input(module, "signal", &data_type);
    let index = file.make_index(&signal.to_indexable_expr(), 2_147_483_648);

    assert_eq!(
        file.emit_expression(&index.to_expr()),
        "signal[64'h0000_0000_8000_0000]"
    );
}

#[test]
#[should_panic(expected = "bit and part-select indices must be nonnegative")]
fn negative_constant_indices_are_rejected() {
    let mut file = VastFile::new(VastFileType::Verilog);
    let module = file.add_module("negative_indices");
    let data_type = file.make_bit_vector_type(8, false);
    let signal = file.add_input(module, "signal", &data_type);

    file.make_index(&signal.to_indexable_expr(), -1);
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn handles_cannot_be_used_with_another_file() {
    let mut original_file = VastFile::new(VastFileType::SystemVerilog);
    let foreign_type = original_file.make_scalar_type();
    let mut destination_file = VastFile::new(VastFileType::SystemVerilog);
    let module = destination_file.add_module("destination");

    destination_file.add_input(module, "invalid", &foreign_type);
}

#[test]
fn handles_are_copy_and_files_can_move_between_threads() {
    fn assert_copy<T: Copy>() {}
    fn assert_send_sync<T: Send + Sync>() {}

    assert_copy::<VastModule>();
    assert_copy::<VastDataType>();
    assert_copy::<ModulePort>();
    assert_copy::<LogicRef>();
    assert_copy::<Expr>();
    assert_send_sync::<VastFile>();
}
