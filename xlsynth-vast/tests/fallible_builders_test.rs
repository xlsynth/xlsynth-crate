// SPDX-License-Identifier: Apache-2.0

//! Recoverable validation for VAST builders.

use xlsynth_vast::{VastFile, VastFileType};

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
    for direction in ["input", "output", "logic input", "logic output"] {
        let mut file = VastFile::new(VastFileType::SystemVerilog);
        let module = file.add_module("Ports");
        let logic = file.make_scalar_type();
        file.add_wire(module, "pin", &logic).unwrap();
        let before = file.emit();
        let result = match direction {
            "input" => file.add_input(module, "pin", &logic),
            "output" => file.add_output(module, "pin", &logic),
            "logic input" => file.add_logic_input(module, "pin", &logic),
            "logic output" => file.add_logic_output(module, "pin", &logic),
            _ => unreachable!(),
        };
        let kind = if direction.ends_with("input") {
            "input"
        } else {
            "output"
        };
        assert_eq!(
            result.unwrap_err().to_string(),
            format!(
                "FAILED_PRECONDITION: Attempted to declare {kind} with name 'pin' multiple times in the same module. Already defined: [pin]"
            )
        );
        assert_eq!(file.emit(), before);
    }
}
