// SPDX-License-Identifier: Apache-2.0

//! Recoverable validation for VAST builders.

use xlsynth_vast::{
    LogicRef, SvArgument, VastDataType, VastError, VastFile, VastFileType, VastModule,
};

#[test]
fn invalid_net_types_do_not_mutate_the_module_or_reserve_the_name() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let module = file.add_module("Recovered");
    let bit = file.make_bit_type();
    let logic = file.make_scalar_type();
    let before = file.emit();

    let wire_error = file.add_wire(module, "wire_name", &bit).unwrap_err();
    assert!(
        wire_error
            .to_string()
            .contains("cannot be used as a wire net")
    );
    assert_eq!(file.emit(), before);
    file.add_wire(module, "wire_name", &logic).unwrap();
    let before_inout = file.emit();

    let inout_error = file.add_inout(module, "pin", &bit).unwrap_err();
    assert!(
        inout_error
            .to_string()
            .contains("cannot be used as an inout net")
    );
    assert_eq!(file.emit(), before_inout);
    file.add_inout(module, "pin", &logic).unwrap();
}

#[test]
fn invalid_packed_array_types_can_be_corrected() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let integer = file.make_int_type(true);
    let logic = file.make_scalar_type();
    let error = file.make_packed_array_type(integer, &[2]).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("packed arrays cannot contain integer or int elements")
    );
    let corrected = file.make_packed_array_type(logic, &[2]).unwrap();
    assert_eq!(file.type_flat_bit_count_as_int64(corrected), Ok(2));
}

#[test]
fn two_state_bit_widths_return_errors_and_can_be_corrected() {
    let mut file = VastFile::new(VastFileType::SystemVerilog);
    let error = file.make_two_state_bit_vector_type(0, false).unwrap_err();
    assert_eq!(
        error.to_string(),
        "bit-vector width must be greater than zero"
    );
    let corrected = file.make_two_state_bit_vector_type(1, false).unwrap();
    assert_eq!(file.type_flat_bit_count_as_int64(corrected), Ok(1));
}

#[test]
fn all_port_builders_report_name_collisions_without_mutation() {
    type PortBuilder =
        fn(&mut VastFile, VastModule, &str, &VastDataType) -> Result<LogicRef, VastError>;
    let cases: [(PortBuilder, &str); 5] = [
        (VastFile::add_input, "input"),
        (VastFile::add_output, "output"),
        (VastFile::add_inout, "inout"),
        (VastFile::add_logic_input, "input"),
        (VastFile::add_logic_output, "output"),
    ];
    for (builder, kind) in cases {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let module = file.add_module("Ports");
        let logic = file.make_scalar_type();
        file.add_wire(module, "pin", &logic).unwrap();
        let before = file.emit();
        let result = builder(&mut file, module, "pin", &logic);
        assert_eq!(
            result.unwrap_err().to_string(),
            format!(
                "FAILED_PRECONDITION: Attempted to declare {kind} with name 'pin' multiple times in the same module. Already defined: [pin]"
            )
        );
        assert_eq!(file.emit(), before);
    }
}

#[test]
fn instantiation_builders_reject_duplicate_named_arguments_separately() {
    for typed_arguments in [false, true] {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let module = file.add_module("Assembly");
        let one = file.make_unsized_decimal_literal(1);
        let bit = file.make_bit_type();
        let before = file.emit();

        let parameter_error = if typed_arguments {
            file.make_instantiation_with_arguments(
                "Leaf",
                "u",
                &["W", "W"],
                &[SvArgument::Value(one), SvArgument::Type(bit)],
                &[],
                &[],
            )
        } else {
            file.make_instantiation("Leaf", "u", &["W", "W"], &[&one, &one], &[], &[])
        }
        .unwrap_err();
        assert_eq!(
            parameter_error.to_string(),
            "duplicate module-instantiation parameter name: W"
        );
        assert_eq!(file.emit(), before);

        let port_error = if typed_arguments {
            file.make_instantiation_with_arguments(
                "Leaf",
                "u",
                &[],
                &[],
                &["q", "q"],
                &[Some(&one), None],
            )
        } else {
            file.make_instantiation("Leaf", "u", &[], &[], &["q", "q"], &[Some(&one), None])
        }
        .unwrap_err();
        assert_eq!(
            port_error.to_string(),
            "duplicate module-instantiation port name: q"
        );
        assert_eq!(file.emit(), before);

        let corrected = if typed_arguments {
            file.make_instantiation_with_arguments(
                "Leaf",
                "u",
                &["W"],
                &[SvArgument::Value(one)],
                &["W"],
                &[None],
            )
        } else {
            file.make_instantiation("Leaf", "u", &["W"], &[&one], &["W"], &[None])
        }
        .unwrap();
        file.add_member_instantiation(module, corrected).unwrap();
        let output = file.emit();
        assert!(output.contains(".W(1)"));
        assert!(output.contains(".W()"));
    }
}
