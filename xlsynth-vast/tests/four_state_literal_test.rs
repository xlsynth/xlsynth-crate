// SPDX-License-Identifier: Apache-2.0

use xlsynth_vast::{VastFile, VastFileType};

#[test]
fn four_state_literals_preserve_width_and_unknown_digits() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("Parameters");
    let value = file.make_four_state_binary_literal("10xZ01Xz").unwrap();
    file.add_parameter_port(module, "VALUE", &value).unwrap();
    assert_eq!(
        file.emit(),
        "module Parameters #(\n  parameter VALUE = 8'b10xZ01Xz\n);\n\nendmodule\n"
    );
}

#[test]
fn four_state_literals_reject_invalid_digits_and_sizes() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    for invalid in ["", "?", "2", "0_1", "x ", "é"] {
        assert!(
            file.make_four_state_binary_literal(invalid).is_err(),
            "{invalid:?}"
        );
    }
    assert!(
        file.make_four_state_binary_literal(&"x".repeat((1 << 20) + 1))
            .is_err()
    );
}

/// Known binary literals retain their value for width validation and
/// inspection.
#[test]
fn known_binary_literal_widths_are_inspected_for_both_vector_states() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let four_state = file.make_four_state_binary_literal("000101").unwrap();
    let pattern = file.make_binary_pattern("101").unwrap();
    for width in [four_state, pattern] {
        let logic = file.make_bit_vector_type_expr(&width, false).unwrap();
        let bit = file
            .make_two_state_bit_vector_type_expr(&width, true)
            .unwrap();
        assert_eq!(file.type_width_as_int64(logic), Ok(5));
        assert_eq!(file.type_width_as_int64(bit), Ok(5));
    }
}

/// A known binary zero cannot become a malformed negative-index port range.
#[test]
fn known_binary_zero_is_rejected_for_logic_vector_widths() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let zero = file.make_four_state_binary_literal("0").unwrap();
    let error = file.make_bit_vector_type_expr(&zero, false).unwrap_err();
    assert_eq!(
        error.to_string(),
        "bit-vector width must be greater than zero"
    );
}

/// The same invariant applies to two-state bit vector widths.
#[test]
fn known_binary_zero_is_rejected_for_bit_vector_widths() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let zero = file.make_binary_pattern("000").unwrap();
    let error = file
        .make_two_state_bit_vector_type_expr(&zero, false)
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "bit-vector width must be greater than zero"
    );
}

/// Unknown and overflowing binary values are not misreported as known widths.
#[test]
fn binary_width_caching_handles_unknowns_overflow_and_leading_zeroes() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let unknown = file.make_four_state_binary_literal("10x1").unwrap();
    let pattern = file.make_binary_pattern("10?1").unwrap();
    let overflow = file
        .make_four_state_binary_literal(&"1".repeat(64))
        .unwrap();
    for width in [unknown, pattern, overflow] {
        let vector = file.make_bit_vector_type_expr(&width, false).unwrap();
        assert!(file.type_width_as_int64(vector).is_err());
    }
    let digits = format!("{}1", "0".repeat((1 << 20) - 1));
    let one = file.make_four_state_binary_literal(&digits).unwrap();
    let vector = file.make_bit_vector_type_expr(&one, false).unwrap();
    assert_eq!(file.type_width_as_int64(vector), Ok(1));
    let max = file
        .make_four_state_binary_literal(&"1".repeat(63))
        .unwrap();
    let vector = file.make_bit_vector_type_expr(&max, false).unwrap();
    assert_eq!(file.type_width_as_int64(vector), Ok(i64::MAX));
}
