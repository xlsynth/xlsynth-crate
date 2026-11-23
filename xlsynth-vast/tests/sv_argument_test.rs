// SPDX-License-Identifier: Apache-2.0

//! Typed SystemVerilog argument validation at mixed syntactic boundaries.

use xlsynth_vast::{SvArgument, VastFile, VastFileType};

#[test]
fn legacy_empty_arguments_keep_their_spelling() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let call = file.make_function_call("empty", &[]);
    let macro_ref = file.make_macro_ref_with_args("EMPTY", &[]);
    assert_eq!(file.emit_expression(&call), "empty()");
    assert_eq!(file.emit_expression(&macro_ref.to_expr()), "`EMPTY()");
}

#[test]
#[should_panic(expected = "ordinary function calls do not accept type arguments")]
fn ordinary_functions_reject_type_arguments() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let data_type = file.make_scalar_type();
    file.make_function_call_with_arguments("ordinary", &[SvArgument::Type(data_type)]);
}

#[test]
fn foreign_types_and_values_are_rejected_at_each_argument_boundary() {
    let mut source = VastFile::new(VastFileType::SystemVerilog);
    let data_type = source.make_scalar_type();
    let value = source.make_unsized_decimal_literal(1);
    for argument in [SvArgument::Type(data_type), SvArgument::Value(value)] {
        for boundary in 0..3 {
            let result = std::panic::catch_unwind(|| {
                let mut destination = VastFile::new(VastFileType::SystemVerilog);
                match boundary {
                    0 => {
                        destination.make_function_call_with_arguments("$bits", &[argument]);
                    }
                    1 => {
                        destination.make_instantiation_with_arguments(
                            "Leaf",
                            "u",
                            &["P"],
                            &[argument],
                            &[],
                            &[],
                        );
                    }
                    _ => {
                        destination.make_macro_ref_with_arguments("MACRO", &[argument]);
                    }
                }
            });
            assert!(
                result.is_err(),
                "foreign argument accepted at boundary {boundary}"
            );
        }
    }
}

#[test]
fn anonymous_unpacked_types_require_a_typedef_at_each_argument_boundary() {
    for boundary in 0..3 {
        let result = std::panic::catch_unwind(|| {
            let mut file = VastFile::new(VastFileType::SystemVerilog);
            let scalar = file.make_scalar_type();
            let array = file.make_unpacked_array_type(scalar, &[4]);
            let argument = SvArgument::Type(array);
            match boundary {
                0 => {
                    file.make_function_call_with_arguments("$bits", &[argument]);
                }
                1 => {
                    file.make_instantiation_with_arguments(
                        "Leaf",
                        "u",
                        &["T"],
                        &[argument],
                        &[],
                        &[],
                    );
                }
                _ => {
                    file.make_macro_ref_with_arguments("MACRO", &[argument]);
                }
            }
        });
        assert!(
            result.is_err(),
            "anonymous unpacked type accepted at boundary {boundary}"
        );
    }
}
