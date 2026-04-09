// SPDX-License-Identifier: Apache-2.0

use xlsynth::{IrBits, IrPackage, IrValue};
use xlsynth_g8r::aig_sim::gate_sim::{self, Collect};
use xlsynth_g8r::test_utils::{
    interesting_ir_output_ordering_cases, load_interesting_ir_output_ordering_case,
};
use xlsynth_pir::ir;
use xlsynth_pir::ir_value_utils::{
    flatten_ir_value_to_lsb0_bits_for_type, ir_value_from_lsb0_bits_with_layout,
};

fn flatten_gate_outputs_lsb0(outputs: &[IrBits]) -> IrBits {
    let total_bits: usize = outputs.iter().map(IrBits::get_bit_count).sum();
    let mut flat_bits = Vec::with_capacity(total_bits);
    for output in outputs {
        for bit_index in 0..output.get_bit_count() {
            flat_bits.push(output.get_bit(bit_index).unwrap());
        }
    }
    IrBits::from_lsb_is_0(&flat_bits)
}

fn flatten_value_for_verilog_layout(value: &IrValue, ty: &ir::Type) -> IrBits {
    let mut flat_bits = Vec::with_capacity(ty.bit_count());
    flatten_ir_value_to_lsb0_bits_for_type(value, ty, &mut flat_bits)
        .expect("value should match type for verilog-layout flattening");
    IrBits::from_lsb_is_0(&flat_bits)
}

fn make_one_hot_param_samples(params: &[ir::Param]) -> Vec<Vec<IrValue>> {
    let zero_args: Vec<IrValue> = params
        .iter()
        .map(|param| {
            ir_value_from_lsb0_bits_with_layout(&param.ty, &vec![false; param.ty.bit_count()])
                .expect("zero bits should rebuild for parameter type")
        })
        .collect();

    let mut samples = vec![zero_args.clone()];
    for (param_index, param) in params.iter().enumerate() {
        for hot_bit in 0..param.ty.bit_count() {
            let mut bits = vec![false; param.ty.bit_count()];
            bits[hot_bit] = true;
            let mut args = zero_args.clone();
            args[param_index] = ir_value_from_lsb0_bits_with_layout(&param.ty, &bits)
                .expect("one-hot bits should rebuild for parameter type");
            samples.push(args);
        }
    }
    samples
}

#[test]
fn test_bit_blast_outputs_follow_verilog_layout_for_interesting_signatures() {
    for case in interesting_ir_output_ordering_cases() {
        let sample = load_interesting_ir_output_ordering_case(&case);
        let xlsynth_pkg =
            IrPackage::parse_ir(&case.ir_text, Some(case.name)).expect("IR should parse");
        let xlsynth_fn = xlsynth_pkg.get_function("main").expect("main should exist");

        for args in make_one_hot_param_samples(&sample.g8r_fn.params) {
            let gate_inputs: Vec<IrBits> = args
                .iter()
                .zip(sample.g8r_fn.params.iter())
                .map(|(arg, param)| flatten_value_for_verilog_layout(arg, &param.ty))
                .collect();
            let gate_result = gate_sim::eval(&sample.gate_fn, &gate_inputs, Collect::None);
            let got_bits = flatten_gate_outputs_lsb0(&gate_result.outputs);

            let expected_value = xlsynth_fn
                .interpret(&args)
                .expect("IR interpret should succeed");
            let expected_bits =
                flatten_value_for_verilog_layout(&expected_value, &sample.g8r_fn.ret_ty);

            assert_eq!(
                got_bits,
                expected_bits,
                "bit-blasted output ordering should match Verilog layout for case {} and args {}",
                sample.case.name,
                IrValue::make_tuple(&args)
            );
        }
    }
}
