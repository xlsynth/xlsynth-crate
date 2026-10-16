// SPDX-License-Identifier: Apache-2.0

use xlsynth_g8r::gate_builder::{GateBuilder, GateBuilderOptions};
use xlsynth_g8r::netlist::gv_eval::LabeledNetlistAig;
use xlsynth_pir::{IrBits, IrValue};

/// Builds an inverter for each input port without involving text parsing.
fn inverter_model(widths: &[usize]) -> LabeledNetlistAig {
    let mut builder = GateBuilder::new("conversion".into(), GateBuilderOptions::no_opt());
    for (i, &width) in widths.iter().enumerate() {
        let input = builder.add_input(format!("in{i}"), width);
        let inverted = builder.add_not_vec(&input);
        builder.add_output(format!("out{i}"), inverted);
    }
    LabeledNetlistAig {
        module_name: "conversion".into(),
        gate_fn: builder.build(),
        module_ports: vec![],
        module_boundaries: vec![],
        instances: vec![],
    }
}

#[test]
fn typed_batch_preserves_output_shapes_and_wide_bits() {
    for widths in [vec![], vec![0], vec![129], vec![0, 1, 65, 129], vec![1; 64]] {
        let model = inverter_model(&widths);
        let mut args = Vec::new();
        let mut expected = Vec::new();
        for sample in 0..513 {
            let mut fields = Vec::new();
            let mut inverted = Vec::new();
            for (port, &width) in widths.iter().enumerate() {
                let bit_at = |bit| (sample * 17 + port * 11 + bit * 7) % 13 < 6;
                fields.push(IrValue::from_bits(&IrBits::from_lsb_fn(width, bit_at)));
                inverted.push(IrValue::from_bits(&IrBits::from_lsb_fn(width, |bit| {
                    !bit_at(bit)
                })));
            }
            args.push(IrValue::make_tuple(&fields));
            expected.push(if widths.len() == 1 {
                inverted[0].clone()
            } else {
                IrValue::make_tuple(&inverted)
            });
        }
        let original_args = args.clone();
        for n in [0, 1, 2, 255, 256, 257, 511, 512, 513] {
            assert_eq!(model.evaluate_ir_values(&args[..n]).unwrap(), expected[..n]);
        }
        assert_eq!(model.evaluate_ir_value(&args[0]).unwrap(), expected[0]);
        assert_eq!(args, original_args);
    }
}

#[test]
fn typed_batch_preserves_conversion_error_precedence_and_array_acceptance() {
    let model = inverter_model(&[1, 1]);
    let valid_fields = [
        IrValue::make_ubits(1, 0).unwrap(),
        IrValue::make_ubits(1, 1).unwrap(),
    ];
    let valid = IrValue::make_tuple(&valid_fields);
    let malformed = IrValue::make_tuple(&[
        IrValue::make_ubits(2, 0).unwrap(),
        valid_fields[1].clone(),
        IrValue::make_token(),
    ]);
    assert_eq!(
        model
            .evaluate_ir_values(&[valid.clone(), malformed])
            .unwrap_err(),
        "input sample 2: argument 2 for input '<extra>' is not bits-typed: value of type token is not bits-typed"
    );
    assert_eq!(
        model
            .evaluate_ir_value(&IrValue::make_array(&valid_fields).unwrap())
            .unwrap(),
        model.evaluate_ir_value(&valid).unwrap()
    );
}
