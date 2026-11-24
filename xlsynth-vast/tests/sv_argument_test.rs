// SPDX-License-Identifier: Apache-2.0

//! Typed SystemVerilog argument validation at mixed syntactic boundaries.

use xlsynth_vast::{SvArgument, VastFile, VastFileType};

#[test]
fn legacy_empty_arguments_keep_their_spelling() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let call = file.make_function_call("empty", &[]).unwrap();
    let macro_ref = file.make_macro_ref_with_args("EMPTY", &[]);
    assert_eq!(file.emit_expression(&call), "empty()");
    assert_eq!(file.emit_expression(&macro_ref.to_expr()), "`EMPTY()");
}

#[test]
fn ordinary_functions_reject_type_arguments() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let data_type = file.make_scalar_type();
    let error = file
        .make_function_call_with_arguments("ordinary", &[SvArgument::Type(data_type)])
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "ordinary function calls do not accept type arguments"
    );
    let call = file
        .make_function_call_with_arguments("$bits", &[SvArgument::Type(data_type)])
        .unwrap();
    assert_eq!(file.emit_expression(&call), "$bits(logic)");
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
                        let _ = destination.make_function_call_with_arguments("$bits", &[argument]);
                    }
                    1 => {
                        let _ = destination.make_instantiation_with_arguments(
                            "Leaf",
                            "u",
                            &["P"],
                            &[argument],
                            &[],
                            &[],
                        );
                    }
                    _ => {
                        let _ = destination.make_macro_ref_with_arguments("MACRO", &[argument]);
                    }
                }
            });
            let panic =
                result.expect_err("foreign argument must violate the file ownership invariant");
            let message = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .expect("panic message");
            assert!(
                message.contains("VAST handle belongs to a different file"),
                "unexpected panic at boundary {boundary}: {message}"
            );
        }
    }
}

#[test]
fn anonymous_unpacked_types_require_a_typedef_at_each_argument_boundary() {
    for boundary in 0..3 {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let scalar = file.make_scalar_type();
        let array = file.make_unpacked_array_type(scalar, &[4]).unwrap();
        let invalid = SvArgument::Type(array);
        let error = match boundary {
            0 => file
                .make_function_call_with_arguments("$bits", &[invalid])
                .map(|_| ()),
            1 => file
                .make_instantiation_with_arguments("Leaf", "u", &["T"], &[invalid], &[], &[])
                .map(|_| ()),
            _ => file
                .make_macro_ref_with_arguments("MACRO", &[invalid])
                .map(|_| ()),
        }
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "anonymous unpacked array types require a named typedef"
        );
        let valid = SvArgument::Type(scalar);
        let recovered = match boundary {
            0 => file
                .make_function_call_with_arguments("$bits", &[valid])
                .map(|_| ()),
            1 => file
                .make_instantiation_with_arguments("Leaf", "u", &["T"], &[valid], &[], &[])
                .map(|_| ()),
            _ => file
                .make_macro_ref_with_arguments("MACRO", &[valid])
                .map(|_| ()),
        };
        assert!(
            recovered.is_ok(),
            "valid type was rejected at boundary {boundary}"
        );
    }
}

#[test]
fn mixed_instantiation_rejects_mismatched_names_without_mutating_the_file() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("Top");
    let value = file.make_unsized_decimal_literal(4);
    let before = file.emit();
    let error = file
        .make_instantiation_with_arguments("Leaf", "u", &["W"], &[], &[], &[])
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        "module-instantiation parameter names and values must have equal lengths"
    );
    assert_eq!(file.emit(), before);
    let recovered = file
        .make_instantiation_with_arguments(
            "Leaf",
            "u",
            &["W"],
            &[SvArgument::Value(value)],
            &[],
            &[],
        )
        .unwrap();
    file.add_member_instantiation(module, recovered).unwrap();
}
