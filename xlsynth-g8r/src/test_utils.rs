// SPDX-License-Identifier: Apache-2.0

use crate::gate::GateFn;
use crate::ir2gate;
use crate::xls_ir::ir::Package as CrateIrPackage;
use crate::xls_ir::ir_parser;
use half::bf16;
use std::path::Path;
use xlsynth::{mangle_dslx_name, IrBits, IrFunction, IrPackage, IrValue};

// BF16 Constants
pub const BF16_EXPONENT_BITS: usize = 8;
pub const BF16_EXPONENT_MASK: usize = (1 << BF16_EXPONENT_BITS) - 1;
pub const BF16_FRACTION_BITS: usize = 7;
pub const BF16_FRACTION_MASK: usize = (1 << BF16_FRACTION_BITS) - 1;
pub const BF16_TOTAL_BITS: usize = 16;

pub struct LoadedSample {
    pub ir_package: IrPackage,
    pub ir_fn: IrFunction,
    pub g8r_pkg: CrateIrPackage,
    pub gate_fn: GateFn,
    pub mangled_fn_name: String,
}

// BF16 Helper Functions
pub fn make_bf16(value: bf16) -> IrValue {
    let bits = value.to_bits();
    let sign_val = (bits >> (BF16_EXPONENT_BITS + BF16_FRACTION_BITS)) & 1;
    let exponent_val = (bits >> BF16_FRACTION_BITS) & (BF16_EXPONENT_MASK as u16);
    let fraction_val = bits & (BF16_FRACTION_MASK as u16);

    let sign = IrValue::bool(sign_val == 1);
    let exponent = IrValue::make_ubits(BF16_EXPONENT_BITS as usize, exponent_val as u64).unwrap();
    let fraction = IrValue::make_ubits(BF16_FRACTION_BITS as usize, fraction_val as u64).unwrap();
    IrValue::make_tuple(&[sign, exponent, fraction])
}

pub fn ir_value_bf16_to_flat_ir_bits(value: &IrValue) -> IrBits {
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

pub fn flat_ir_bits_to_ir_value_bf16(bits_value: &IrBits) -> IrValue {
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

pub fn load_bf16_mul_sample() -> LoadedSample {
    let dslx_text = "import bfloat16;

fn mul_bf16_bf16(x: bfloat16::BF16, y: bfloat16::BF16) -> bfloat16::BF16 {
        bfloat16::mul(x, y)
}";
    let fake_path = Path::new("test_utils.x");
    let ir_result = xlsynth::convert_dslx_to_ir(
        dslx_text,
        fake_path,
        &xlsynth::DslxConvertOptions::default(),
    )
    .unwrap();
    let fn_name = "mul_bf16_bf16";
    let module_name = "test_utils";
    let mangled_name = mangle_dslx_name(module_name, fn_name).unwrap();

    let opt_ir = xlsynth::optimize_ir(&ir_result.ir, &mangled_name).unwrap();
    let ir_text = opt_ir.to_string();

    // Parse with xlsynth-g8r's parser to get its internal IR representation
    let mut parser = ir_parser::Parser::new(&ir_text);
    let g8r_ir_package = parser.parse_package().unwrap();
    let g8r_ir_fn = g8r_ir_package.get_fn(&mangled_name).unwrap();

    // Convert the internal IR function to a GateFn
    let gatify_output = ir2gate::gatify(
        &g8r_ir_fn,
        ir2gate::GatifyOptions {
            fold: true,
            hash: true,
            check_equivalence: false,
        },
    )
    .unwrap();
    let gate_fn = gatify_output.gate_fn;

    // Get the final IrFunction from the optimized package
    let ir_fn = opt_ir.get_function(&mangled_name).unwrap();

    LoadedSample {
        ir_package: opt_ir,
        ir_fn,
        g8r_pkg: g8r_ir_package,
        gate_fn,
        mangled_fn_name: mangled_name,
    }
}

pub fn load_bf16_add_sample() -> LoadedSample {
    let dslx_text = "import bfloat16;\n\nfn add_bf16_bf16(x: bfloat16::BF16, y: bfloat16::BF16) -> bfloat16::BF16 {\n        bfloat16::add(x, y)\n}";
    let fake_path = Path::new("test_utils.x");
    let ir_result = xlsynth::convert_dslx_to_ir(
        dslx_text,
        fake_path,
        &xlsynth::DslxConvertOptions::default(),
    )
    .unwrap();
    let fn_name = "add_bf16_bf16";
    let module_name = "test_utils";
    let mangled_name = mangle_dslx_name(module_name, fn_name).unwrap();

    let opt_ir = xlsynth::optimize_ir(&ir_result.ir, &mangled_name).unwrap();
    let ir_text = opt_ir.to_string();

    // Parse with xlsynth-g8r's parser to get its internal IR representation
    let mut parser = ir_parser::Parser::new(&ir_text);
    let g8r_ir_package = parser.parse_package().unwrap();
    let g8r_ir_fn = g8r_ir_package.get_fn(&mangled_name).unwrap();

    // Convert the internal IR function to a GateFn
    let gatify_output = ir2gate::gatify(
        &g8r_ir_fn,
        ir2gate::GatifyOptions {
            fold: true,
            hash: true,
            check_equivalence: false,
        },
    )
    .unwrap();
    let gate_fn = gatify_output.gate_fn;

    // Get the final IrFunction from the optimized package
    let ir_fn = opt_ir.get_function(&mangled_name).unwrap();

    LoadedSample {
        ir_package: opt_ir,
        ir_fn,
        g8r_pkg: g8r_ir_package,
        gate_fn,
        mangled_fn_name: mangled_name,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
