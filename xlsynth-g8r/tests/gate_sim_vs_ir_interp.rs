// SPDX-License-Identifier: Apache-2.0

//! Tests that compare the results of g8r gate simulation with the results of
//! XLS IR interpretation.
//!
//! This is useful for ensuring the gate simulation is correct.

use half::bf16;
use rand::Rng;
use xlsynth::{IrBits, IrValue};
use xlsynth_g8r::gate_sim;
use xlsynth_g8r::test_utils::load_bf16_mul_sample;

const BF16_EXPONENT_BITS: usize = 8;
const BF16_EXPONENT_MASK: usize = (1 << BF16_EXPONENT_BITS) - 1;
const BF16_FRACTION_BITS: usize = 7;
const BF16_FRACTION_MASK: usize = (1 << BF16_FRACTION_BITS) - 1;
const BF16_TOTAL_BITS: usize = 16;

fn make_bf16(value: bf16) -> IrValue {
    let bits = value.to_bits();
    let sign_val = (bits >> (BF16_EXPONENT_BITS + BF16_FRACTION_BITS)) & 1;
    let exponent_val = (bits >> BF16_FRACTION_BITS) & (BF16_EXPONENT_MASK as u16);
    let fraction_val = bits & (BF16_FRACTION_MASK as u16);

    let sign = IrValue::bool(sign_val == 1);
    let exponent = IrValue::make_ubits(BF16_EXPONENT_BITS as usize, exponent_val as u64).unwrap();
    let fraction = IrValue::make_ubits(BF16_FRACTION_BITS as usize, fraction_val as u64).unwrap();
    IrValue::make_tuple(&[sign, exponent, fraction])
}

// Convert a structured BF16 IrValue (sign, exponent, fraction) into a flat
// IrBits.
fn ir_value_bf16_to_flat_ir_bits(value: &IrValue) -> IrBits {
    let tuple_elements = value.get_elements().unwrap();
    let sign = tuple_elements[0].to_bool().unwrap();
    let exponent = tuple_elements[1].to_u64().unwrap();
    let fraction = tuple_elements[2].to_u64().unwrap();

    let mut bits: u16 = 0;
    if sign {
        bits |= 1 << (BF16_EXPONENT_BITS + BF16_FRACTION_BITS);
    }
    bits |= (exponent as u16) << BF16_FRACTION_BITS;
    bits |= fraction as u16;

    IrBits::make_ubits(BF16_TOTAL_BITS as usize, bits as u64).unwrap()
}

// Convert flat IrBits into a structured BF16 IrValue (sign, exponent,
// fraction).
fn flat_ir_bits_to_ir_value_bf16(bits_value: &IrBits) -> IrValue {
    assert_eq!(bits_value.get_bit_count(), BF16_TOTAL_BITS as usize);
    let temp_value = IrValue::from_bits(bits_value);
    let bits = temp_value.to_u64().unwrap() as u16;

    let sign_bit = (bits >> (BF16_EXPONENT_BITS + BF16_FRACTION_BITS)) & 1;
    let exponent_bits = (bits >> BF16_FRACTION_BITS) & (BF16_EXPONENT_MASK as u16);
    let fraction_bits = bits & (BF16_FRACTION_MASK as u16);

    let sign = IrValue::bool(sign_bit == 1);
    let exponent = IrValue::make_ubits(BF16_EXPONENT_BITS as usize, exponent_bits as u64).unwrap();
    let fraction = IrValue::make_ubits(BF16_FRACTION_BITS as usize, fraction_bits as u64).unwrap();

    IrValue::make_tuple(&[sign, exponent, fraction])
}

#[test]
fn test_make_bf16_one() {
    let one = bf16::from_f32(1.0);
    let ir_value = make_bf16(one);
    assert_eq!(
        ir_value,
        IrValue::make_tuple(&[
            IrValue::bool(false),                                  // sign
            IrValue::make_ubits(BF16_EXPONENT_BITS, 127).unwrap(), // biased exponent
            IrValue::make_ubits(BF16_FRACTION_BITS, 0).unwrap(),   // fraction
        ])
    );
}

#[test]
fn test_bf16_mul_zero_zero() {
    let _ = env_logger::builder().is_test(true).try_init();
    let loaded_sample = load_bf16_mul_sample();
    let ir_fn = loaded_sample.ir_fn;
    let gate_fn = loaded_sample.gate_fn;

    let arg0 = make_bf16(bf16::from_f32(0.0));
    let arg1 = make_bf16(bf16::from_f32(0.0));

    let ir_result = ir_fn.interpret(&[arg0.clone(), arg1.clone()]).unwrap();

    let gate_arg0_bits = ir_value_bf16_to_flat_ir_bits(&arg0);
    let gate_arg1_bits = ir_value_bf16_to_flat_ir_bits(&arg1);
    let gate_result_sim = gate_sim::eval(&gate_fn, &[gate_arg0_bits, gate_arg1_bits], false);

    // GateFn outputs are flattened. The mul_bf16 returns a single BF16 tuple.
    assert_eq!(gate_result_sim.outputs.len(), 1);
    let gate_result_bits = &gate_result_sim.outputs[0];
    let gate_result = flat_ir_bits_to_ir_value_bf16(gate_result_bits);

    assert_eq!(ir_result, gate_result);
}

#[test]
fn test_bf16_mul_random() {
    let _ = env_logger::builder().is_test(true).try_init();
    let loaded_sample = load_bf16_mul_sample();
    let ir_fn = loaded_sample.ir_fn;
    let gate_fn = loaded_sample.gate_fn;

    let mut rng = rand::thread_rng();

    for i in 0..256 {
        let f0_bits: u16 = rng.gen();
        let f1_bits: u16 = rng.gen();

        let f0_bf16 = bf16::from_bits(f0_bits);
        let f1_bf16 = bf16::from_bits(f1_bits);

        log::debug!(
            "Testing iter {} with input bits: {:#06x}, {:#06x} (bf16: {}, {})",
            i,
            f0_bits,
            f1_bits,
            f0_bf16,
            f1_bf16
        );

        let arg0 = make_bf16(f0_bf16);
        let arg1 = make_bf16(f1_bf16);

        let ir_result = ir_fn.interpret(&[arg0.clone(), arg1.clone()]).unwrap();

        let gate_arg0_bits = ir_value_bf16_to_flat_ir_bits(&arg0);
        let gate_arg1_bits = ir_value_bf16_to_flat_ir_bits(&arg1);
        let gate_result_sim = gate_sim::eval(&gate_fn, &[gate_arg0_bits, gate_arg1_bits], false);

        assert_eq!(gate_result_sim.outputs.len(), 1);
        let gate_result_bits = &gate_result_sim.outputs[0];
        let gate_result = flat_ir_bits_to_ir_value_bf16(gate_result_bits);

        assert_eq!(
            ir_result, gate_result,
            "Mismatch at iteration {} with input bits: {:#06x}, {:#06x} (bf16: {}, {})",
            i, f0_bits, f1_bits, f0_bf16, f1_bf16
        );
    }
}
