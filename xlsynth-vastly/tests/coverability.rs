// SPDX-License-Identifier: Apache-2.0

use xlsynth_vastly::LineCoverability;
use xlsynth_vastly::SourceText;
use xlsynth_vastly::compute_coverability;
use xlsynth_vastly::compute_coverability_or_fallback;

#[test]
fn decls_and_header_are_structural_skipped() {
    let src = concat!(
        "module m(\n",
        "  input logic clk,\n",
        "  input logic a,\n",
        "  output wire y\n",
        ");\n",
        "  wire t;\n",
        "  assign y = a;\n",
        "endmodule\n",
    );
    let st = SourceText::new(src.to_string());
    let c = compute_coverability_or_fallback(&st);
    assert_eq!(c.line(1), LineCoverability::NonCoverableStructural);
    assert_eq!(c.line(2), LineCoverability::NonCoverableStructural);
    assert_eq!(c.line(5), LineCoverability::NonCoverableStructural);
    assert_eq!(c.line(6), LineCoverability::NonCoverableStructural); // wire decl
    assert_eq!(c.line(7), LineCoverability::CoverableExecutable);
    assert_eq!(c.line(8), LineCoverability::NonCoverableStructural);
}

#[test]
fn skipped_preprocessor_lines_are_unknown_coverable() {
    let src = concat!(
        "module m(input logic clk, input logic a, output wire y);\n",
        "`ifdef FOO\n",
        "  assign y = a;\n",
        "`endif\n",
        "endmodule\n",
    );
    let st = SourceText::new(src.to_string());
    let c = compute_coverability_or_fallback(&st);
    assert_eq!(c.line(1), LineCoverability::NonCoverableStructural);
    // Directives are structural.
    assert_eq!(c.line(2), LineCoverability::NonCoverableStructural);
    // The contents are skipped by the preprocessor when not defined; treat as
    // unknown coverable so users can still see gaps in unparsed regions.
    assert_eq!(c.line(3), LineCoverability::CoverableUnknown);
    assert_eq!(c.line(4), LineCoverability::NonCoverableStructural);
}

#[test]
fn generated_items_are_marked_coverable() {
    let src = concat!(
        "module m(\n",
        "  input logic clk,\n",
        "  input logic a,\n",
        "  output logic y\n",
        ");\n",
        "  logic q;\n",
        "  for (genvar i = 0; i < 1; i = i + 1) begin : g\n",
        "    assign y = q;\n",
        "    always_ff @(posedge clk) begin\n",
        "      q <= a;\n",
        "    end\n",
        "  end\n",
        "endmodule\n",
    );
    let st = SourceText::new(src.to_string());
    let c = compute_coverability_or_fallback(&st);
    assert_eq!(c.line(7), LineCoverability::CoverableUnknown);
    assert_eq!(c.line(8), LineCoverability::CoverableExecutable);
    assert_eq!(c.line(9), LineCoverability::CoverableExecutable);
    assert_eq!(c.line(10), LineCoverability::CoverableExecutable);
}

#[test]
fn named_connection_expression_lines_are_coverable() {
    let src = r#"module top(input wire clk, input wire sel, input wire a, output wire y);
  child inst (
    .x(sel ?
       a : 1'b0),
    .out(y)
  );
endmodule
module child(input wire x, output wire out);
  assign out = x;
endmodule
"#;
    let cover = compute_coverability(&SourceText::new(src.to_owned())).unwrap();
    assert_eq!(cover.line(2), LineCoverability::NonCoverableStructural);
    for line in 3..=5 {
        assert_eq!(cover.line(line), LineCoverability::CoverableExecutable);
    }
    assert_eq!(cover.line(6), LineCoverability::NonCoverableStructural);
}
