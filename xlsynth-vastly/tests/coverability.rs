// SPDX-License-Identifier: Apache-2.0

use xlsynth_vastly::LineCoverability;
use xlsynth_vastly::SourceText;
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
