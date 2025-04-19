// SPDX-License-Identifier: Apache-2.0

use crate::gate::{AigOperand, GateFn};
use crate::gate_builder::{GateBuilder, GateBuilderOptions};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opt {
    Yes,
    No,
}

pub fn load_bf16_mul_sample(opt: Opt) -> LoadedSample {
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
            fold: if opt == Opt::Yes { true } else { false },
            hash: if opt == Opt::Yes { true } else { false },
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

pub fn load_bf16_add_sample(opt: Opt) -> LoadedSample {
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
            fold: if opt == Opt::Yes { true } else { false },
            hash: if opt == Opt::Yes { true } else { false },
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

pub struct TestGraph {
    pub g: GateFn,
    pub i0: AigOperand,
    pub i1: AigOperand,
    pub i2: AigOperand,
    pub i3: AigOperand,
    pub a: AigOperand,
    pub b: AigOperand,
    pub c: AigOperand,
    pub o: AigOperand,
}

/// Creates a common graph structure for testing e.g. cone extraction.
/// There are no redundancies in this graph.
/// ```text
/// fn g(i0, i1, i2, i3) -> (o, c) {
///     a = i0 & i1
///     b = i1 & i2
///     c = i2 & i3
///     o = a & b
///     return (o, c)
/// }
/// ```
pub fn setup_simple_graph() -> TestGraph {
    let mut gb = GateBuilder::new("g".to_string(), GateBuilderOptions::no_opt());
    let i0: AigOperand = gb.add_input("i0".to_string(), 1).try_into().unwrap();
    let i1: AigOperand = gb.add_input("i1".to_string(), 1).try_into().unwrap();
    let i2: AigOperand = gb.add_input("i2".to_string(), 1).try_into().unwrap();
    let i3: AigOperand = gb.add_input("i3".to_string(), 1).try_into().unwrap();

    let a = gb.add_and_binary(i0, i1);
    let b = gb.add_and_binary(i1, i2);
    let c = gb.add_and_binary(i2, i3);

    let o = gb.add_and_binary(a, b);
    gb.add_output("o".to_string(), o.into());
    gb.add_output("c".to_string(), c.into()); // Add c as an output too

    let g = gb.build();
    TestGraph {
        g,
        i0,
        i1,
        i2,
        i3,
        a,
        b,
        c,
        o,
    }
}

pub struct TestGraphWithRedundancies {
    pub g: GateFn,
    pub i0: AigOperand,
    pub i1: AigOperand,
    pub i2: AigOperand,
    pub inner0: AigOperand,
    pub inner1: AigOperand,
    pub outer0: AigOperand,
    pub outer1: AigOperand,
}

pub fn setup_graph_with_redundancies() -> TestGraphWithRedundancies {
    let mut gb = GateBuilder::new("redundant_graph".to_string(), GateBuilderOptions::no_opt());
    let i0 = gb.add_input("i0".to_string(), 1).try_into().unwrap();
    let i1 = gb.add_input("i1".to_string(), 1).try_into().unwrap();
    let i2 = gb.add_input("i2".to_string(), 1).try_into().unwrap();
    let inner0 = gb.add_and_binary(i0, i1);
    let inner1 = gb.add_and_binary(i0, i1);
    let outer0 = gb.add_and_binary(inner0, i2);
    let outer1 = gb.add_and_binary(inner1, i2);
    gb.add_output("o0".to_string(), outer0.into());
    gb.add_output("o1".to_string(), outer1.into());

    let g = gb.build();
    TestGraphWithRedundancies {
        g,
        i0,
        i1,
        i2,
        inner0,
        inner1,
        outer0,
        outer1,
    }
}

pub struct TestGraphWithMoreRedundancies {
    pub g: GateFn,
    pub i0: AigOperand,
    pub i1: AigOperand,
    pub i2: AigOperand,
    pub inner0: AigOperand,
    pub inner1: AigOperand,
    pub inner2: AigOperand,
    pub outer0: AigOperand,
    pub outer1: AigOperand,
    pub outer2: AigOperand,
}

pub fn setup_graph_with_more_redundancies() -> TestGraphWithMoreRedundancies {
    let mut gb = GateBuilder::new(
        "more_redundant_graph".to_string(),
        GateBuilderOptions::no_opt(),
    );
    let i0 = gb.add_input("i0".to_string(), 1).try_into().unwrap();
    let i1 = gb.add_input("i1".to_string(), 1).try_into().unwrap();
    let i2 = gb.add_input("i2".to_string(), 1).try_into().unwrap();

    let inner0 = gb.add_and_binary(i0, i1);
    let inner1 = gb.add_and_binary(i0, i1);
    let inner2 = gb.add_and_binary(i0, i1);

    let outer0 = gb.add_and_binary(inner0, i2);
    let outer1 = gb.add_and_binary(inner1, i2);
    let outer2 = gb.add_and_binary(inner2, i2);

    gb.add_output("o0".to_string(), outer0.into());
    gb.add_output("o1".to_string(), outer1.into());
    gb.add_output("o2".to_string(), outer2.into());

    let g = gb.build();
    TestGraphWithMoreRedundancies {
        g,
        i0,
        i1,
        i2,
        inner0,
        inner1,
        inner2,
        outer0,
        outer1,
        outer2,
    }
}

pub struct TestPartiallyEquivGraph {
    pub g: GateFn,
    pub i0: AigOperand,
    pub i1: AigOperand,
    pub i2: AigOperand,
    pub a: AigOperand,
    pub b: AigOperand,
    pub c: AigOperand,
}

/// Creates a graph where 'a' and 'b' are equivalent, but 'c' is not.
/// Graph:
/// i0 --\\ AND(a) [output]
/// i1 --/
/// i0 --\\ AND(b) [output]
/// i1 --/
/// i0 --\\ AND(c) [output]
/// i2 --/
pub fn setup_partially_equiv_graph() -> TestPartiallyEquivGraph {
    let mut gb = GateBuilder::new("partial_equiv".to_string(), GateBuilderOptions::no_opt());
    let i0 = gb.add_input("i0".to_string(), 1).try_into().unwrap();
    let i1 = gb.add_input("i1".to_string(), 1).try_into().unwrap();
    let i2 = gb.add_input("i2".to_string(), 1).try_into().unwrap();

    let a = gb.add_and_binary(i0, i1);
    let b = gb.add_and_binary(i0, i1); // Identical to 'a'
    let c = gb.add_and_binary(i0, i2); // Different from 'a' and 'b'

    gb.add_output("a".to_string(), a.into());
    gb.add_output("b".to_string(), b.into());
    gb.add_output("c".to_string(), c.into());

    let g = gb.build();
    TestPartiallyEquivGraph {
        g,
        i0,
        i1,
        i2,
        a,
        b,
        c,
    }
}

pub struct TestGraphForNonEquivReplace {
    pub g: GateFn,
    pub i0: AigOperand,   // Represents A
    pub i1: AigOperand,   // Represents B
    pub i2: AigOperand,   // Represents C
    pub a_op: AigOperand, // Operand for A (same as i0)
    pub b_op: AigOperand, // Operand for B (same as i1)
    pub c_op: AigOperand, // Operand for C (same as i2)
    pub o_op: AigOperand, // Operand for O = AND(A, B)
}

pub fn setup_graph_for_nonequiv_replace() -> TestGraphForNonEquivReplace {
    let mut gb = GateBuilder::new("nonequiv_replace".to_string(), GateBuilderOptions::no_opt());
    let i0 = gb.add_input("i0".to_string(), 1).try_into().unwrap(); // A (%1)
    let i1 = gb.add_input("i1".to_string(), 1).try_into().unwrap(); // B (%2)
    let i2 = gb.add_input("i2".to_string(), 1).try_into().unwrap(); // C (%3)
    let o = gb.add_and_binary(i0, i1); // O = AND(A, B) (%4)
    gb.add_output("o".to_string(), o.into());

    let g = gb.build();
    TestGraphForNonEquivReplace {
        g,
        i0,
        i1,
        i2,
        a_op: i0,
        b_op: i1,
        c_op: i2,
        o_op: o,
    }
}

pub struct TestGraphForLivenessCheck {
    pub g: GateFn,
    pub a_in: AigOperand,  // Input A
    pub b_in: AigOperand,  // Input B
    pub c_in: AigOperand,  // Input C
    pub o1_op: AigOperand, // O1 = AND(A, B)
    pub o2_op: AigOperand, // O2 = AND(B, C)
}

pub fn setup_graph_for_liveness_check() -> TestGraphForLivenessCheck {
    let mut gb = GateBuilder::new("liveness_check".to_string(), GateBuilderOptions::no_opt());
    let a_in = gb.add_input("a_in".to_string(), 1).try_into().unwrap(); // %1
    let b_in = gb.add_input("b_in".to_string(), 1).try_into().unwrap(); // %2
    let c_in = gb.add_input("c_in".to_string(), 1).try_into().unwrap(); // %3
    let o1 = gb.add_and_binary(a_in, b_in); // %4
    let o2 = gb.add_and_binary(b_in, c_in); // %5
    gb.add_output("o1".to_string(), o1.into());
    gb.add_output("o2".to_string(), o2.into());

    let g = gb.build();
    TestGraphForLivenessCheck {
        g,
        a_in,
        b_in,
        c_in,
        o1_op: o1,
        o2_op: o2,
    }
}

pub struct TestGraphForConstantReplace {
    pub g: GateFn,
    pub const_true: AigOperand,
    pub const_false: AigOperand,
    pub and_true_true: AigOperand, // And2(true, true) node
}

pub fn setup_graph_for_constant_replace() -> TestGraphForConstantReplace {
    let mut gb = GateBuilder::new("constant_replace".to_string(), GateBuilderOptions::no_opt());
    let const_true = gb.get_true();
    let const_false = gb.get_false();
    let and_true_true = gb.add_and_binary(const_true, const_true);
    gb.add_output("out".to_string(), and_true_true.into());

    let g = gb.build();

    TestGraphForConstantReplace {
        g,
        const_true,
        const_false,
        and_true_true,
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
