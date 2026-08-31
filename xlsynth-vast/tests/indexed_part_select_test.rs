// SPDX-License-Identifier: Apache-2.0

use xlsynth_vast::{VastFile, VastFileType};

#[test]
fn indexed_part_select_preserves_start_expression_and_result_precedence() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("slices");
    let byte = file.make_bit_vector_type(8, false);
    let index_type = file.make_bit_vector_type(3, false);
    let data = file.add_input(module, "data", &byte);
    let index = file.add_input(module, "index", &index_type);
    let output = file.add_output(module, "out", &index_type);
    let one = file.make_unsized_decimal_literal(1);
    let start = file.make_add(&index.to_expr(), &one);
    let subject = file.make_indexable_expression(&data.to_expr());
    let selected = file.make_indexed_part_select(&subject, &start, 3);
    let inverted = file.make_bitwise_not(&selected);
    let assignment = file.make_continuous_assignment(&output.to_expr(), &inverted);
    file.add_member_continuous_assignment(module, assignment);
    assert_eq!(
        file.emit(),
        r#"module slices(
  input wire [7:0] data,
  input wire [2:0] index,
  output wire [2:0] out
);
  assign out = ~data[index + 1 +: 3];
endmodule
"#
    );
    let single = file.make_indexed_part_select(&subject, &index.to_expr(), 1);
    assert_eq!(file.emit_expression(&single), "data[index +: 1]");
}

#[test]
#[should_panic(expected = "indexed part-select width must be positive")]
fn indexed_part_select_rejects_zero_width() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("slices");
    let byte = file.make_bit_vector_type(8, false);
    let data = file.add_input(module, "data", &byte);
    let subject = file.make_indexable_expression(&data.to_expr());
    let start = file.make_unsized_decimal_literal(0);
    file.make_indexed_part_select(&subject, &start, 0);
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn indexed_part_select_rejects_foreign_start() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let mut other = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("slices");
    let byte = file.make_bit_vector_type(8, false);
    let data = file.add_input(module, "data", &byte);
    let subject = file.make_indexable_expression(&data.to_expr());
    let start = other.make_unsized_decimal_literal(0);
    file.make_indexed_part_select(&subject, &start, 3);
}

#[test]
#[should_panic(expected = "VAST handle belongs to a different file")]
fn indexed_part_select_rejects_foreign_subject() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let mut other = VastFile::new(VastFileType::SystemVerilog);
    let module = other.add_module("slices");
    let byte = other.make_bit_vector_type(8, false);
    let data = other.add_input(module, "data", &byte);
    let subject = other.make_indexable_expression(&data.to_expr());
    let start = file.make_unsized_decimal_literal(0);
    file.make_indexed_part_select(&subject, &start, 3);
}
