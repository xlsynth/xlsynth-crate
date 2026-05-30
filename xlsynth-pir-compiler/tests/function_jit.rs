// SPDX-License-Identifier: Apache-2.0

use xlsynth::IrValue;
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir_compiler::{JitError, NativeValueLayout, PirFunctionJit, ScalarLayout};

fn compile(ir: &str) -> PirFunctionJit {
    let package = Parser::new(ir)
        .parse_and_validate_package()
        .expect("test PIR should parse and validate");
    let function = package.get_fn("f").expect("function f should exist");
    PirFunctionJit::compile(function).expect("function should JIT compile")
}

#[test]
fn native_carrier_storage_is_accessed_without_argument_copying() {
    let jit = compile(
        r#"package test

fn f(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  ret sum: bits[8] = add(x, y, id=3)
}
"#,
    );
    let x: u8 = 7;
    let y: u8 = 9;
    let mut output: u8 = 0;
    let inputs = [
        std::ptr::from_ref(&x).cast::<u8>(),
        std::ptr::from_ref(&y).cast::<u8>(),
    ];

    // SAFETY: the JIT signature describes two bits[8] (`u8`) inputs and one
    // bits[8] (`u8`) output, all alive and properly aligned for this call.
    unsafe {
        jit.run_native(&inputs, std::ptr::from_mut(&mut output).cast())
            .expect("native execution should succeed");
    }
    assert_eq!(output, 16);
}

#[test]
fn odd_width_arithmetic_masks_to_pir_width() {
    let jit = compile(
        r#"package test

fn f(x: bits[3] id=1, y: bits[3] id=2) -> bits[3] {
  ret sum: bits[3] = add(x, y, id=3)
}
"#,
    );
    assert_eq!(jit.run_u64(&[7, 3]).expect("execute"), 2);
}

#[test]
fn lowers_bitwise_comparison_extension_and_slice_nodes() {
    let jit = compile(
        r#"package test

fn f(x: bits[5] id=1, y: bits[5] id=2) -> bits[5] {
  both: bits[5] = and(x, y, id=3)
  low: bits[3] = bit_slice(both, start=1, width=3, id=4)
  ret widened: bits[5] = zero_ext(low, new_bit_count=5, id=5)
}
"#,
    );
    assert_eq!(
        jit.run_u64(&[0b1_1110, 0b0_1111]).expect("execute"),
        0b0_0111
    );
    assert_eq!(
        jit.run_u64(&[0b1_0100, 0b0_1111]).expect("execute"),
        0b0_0010
    );
}

#[test]
fn signed_comparisons_use_logical_bit_width() {
    let jit = compile(
        r#"package test

fn f(x: bits[3] id=1, y: bits[3] id=2) -> bits[1] {
  ret result: bits[1] = slt(x, y, id=3)
}
"#,
    );
    assert_eq!(jit.run_u64(&[0b111, 0b001]).expect("execute"), 1);
    assert_eq!(jit.run_u64(&[0b010, 0b111]).expect("execute"), 0);
}

#[test]
fn logical_left_shift_returns_zero_on_pir_overshift() {
    let jit = compile(
        r#"package test

fn f(x: bits[8] id=1, amount: bits[4] id=2) -> bits[8] {
  ret shifted: bits[8] = shll(x, amount, id=3)
}
"#,
    );
    assert_eq!(jit.run_u64(&[3, 2]).expect("execute"), 12);
    assert_eq!(jit.run_u64(&[3, 8]).expect("execute"), 0);
    assert_eq!(jit.run_u64(&[3, 15]).expect("execute"), 0);
}

#[test]
fn concat_combines_scalar_values_of_different_widths() {
    let jit = compile(
        r#"package test

fn f(high: bits[5] id=1, low: bits[3] id=2) -> bits[8] {
  ret joined: bits[8] = concat(high, low, id=3)
}
"#,
    );
    assert_eq!(
        jit.run_u64(&[0b10101, 0b011]).expect("execute"),
        0b1010_1011
    );
}

#[test]
fn literal_array_index_uses_xls_out_of_bounds_clamping() {
    let jit = compile(
        r#"package test

fn f(index: bits[4] id=1) -> bits[3] {
  values: bits[3][4] = literal(value=[0, 2, 4, 6], id=2)
  ret selected: bits[3] = array_index(values, indices=[index], id=3)
}
"#,
    );
    assert_eq!(jit.run_u64(&[0]).expect("execute"), 0);
    assert_eq!(jit.run_u64(&[2]).expect("execute"), 4);
    assert_eq!(jit.run_u64(&[15]).expect("execute"), 6);
}

#[test]
fn bounded_literal_array_index_accepts_assumed_in_bounds() {
    let jit = compile(
        r#"package test

fn f(index: bits[2] id=1) -> bits[3] {
  values: bits[3][4] = literal(value=[0, 2, 4, 6], id=2)
  ret selected: bits[3] = array_index(values, indices=[index], assumed_in_bounds=true, id=3)
}
"#,
    );
    assert_eq!(jit.run_u64(&[3]).expect("execute"), 6);
}

#[test]
fn native_scalar_array_input_uses_c_array_layout() {
    let jit = compile(
        r#"package test

fn f(values: bits[9][4] id=1, index: bits[3] id=2) -> bits[9] {
  ret selected: bits[9] = array_index(values, indices=[index], id=3)
}
"#,
    );
    assert_eq!(
        jit.param_layouts()[0],
        NativeValueLayout::Array {
            element: Box::new(NativeValueLayout::Scalar(ScalarLayout {
                bit_count: 9,
                byte_count: 2,
            })),
            element_count: 4,
        }
    );
    assert_eq!(
        jit.param_layouts()[0].byte_count(),
        std::mem::size_of::<[u16; 4]>()
    );
    assert_eq!(
        jit.param_layouts()[0].alignment(),
        std::mem::align_of::<[u16; 4]>()
    );
    assert_eq!(
        jit.param_layouts()[0].element_stride(),
        Some(std::mem::size_of::<u16>())
    );

    let values: [u16; 4] = [3, 101, 255, 509];
    let index: u8 = 2;
    let mut output: u16 = 0;
    let inputs = [
        std::ptr::from_ref(&values).cast::<u8>(),
        std::ptr::from_ref(&index).cast::<u8>(),
    ];
    // SAFETY: `[u16; 4]` is exactly the native layout for `bits[9][4]`;
    // `index` and `output` use the scalar layouts selected by the JIT.
    unsafe {
        jit.run_native(&inputs, std::ptr::from_mut(&mut output).cast())
            .expect("native array indexing should execute");
    }
    assert_eq!(output, 255);

    let index: u8 = 7;
    let inputs = [
        std::ptr::from_ref(&values).cast::<u8>(),
        std::ptr::from_ref(&index).cast::<u8>(),
    ];
    // SAFETY: same native storage contract as the call above.
    unsafe {
        jit.run_native(&inputs, std::ptr::from_mut(&mut output).cast())
            .expect("out-of-bounds index should clamp");
    }
    assert_eq!(output, 509);
}

#[test]
fn native_array_result_is_written_to_c_array_storage() {
    let jit = compile(
        r#"package test

fn f(x: bits[9] id=1, y: bits[9] id=2) -> bits[9][3] {
  ret made: bits[9][3] = array(x, y, x, id=3)
}
"#,
    );
    let x: u16 = 12;
    let y: u16 = 300;
    let mut output: [u16; 3] = [0; 3];
    let inputs = [
        std::ptr::from_ref(&x).cast::<u8>(),
        std::ptr::from_ref(&y).cast::<u8>(),
    ];
    // SAFETY: the inputs and output use the published native layouts; the
    // output is directly writable as a C-compatible array of `u16`.
    unsafe {
        jit.run_native(&inputs, std::ptr::from_mut(&mut output).cast())
            .expect("native array construction should execute");
    }
    assert_eq!(output, [12, 300, 12]);
}

#[test]
fn nested_native_arrays_use_recursive_c_array_strides() {
    let jit = compile(
        r#"package test

fn f(values: bits[8][2][2] id=1, i: bits[2] id=2, j: bits[2] id=3) -> bits[8] {
  ret selected: bits[8] = array_index(values, indices=[i, j], id=4)
}
"#,
    );
    let values: [[u8; 2]; 2] = [[1, 2], [3, 4]];
    let i: u8 = 1;
    let j: u8 = 0;
    let mut output: u8 = 0;
    let inputs = [
        std::ptr::from_ref(&values).cast::<u8>(),
        std::ptr::from_ref(&i).cast::<u8>(),
        std::ptr::from_ref(&j).cast::<u8>(),
    ];
    // SAFETY: nested Rust arrays are contiguous recursive native array
    // storage corresponding to `bits[8][2][2]`.
    unsafe {
        jit.run_native(&inputs, std::ptr::from_mut(&mut output).cast())
            .expect("nested native array indexing should execute");
    }
    assert_eq!(output, 3);
}

#[test]
fn native_subarray_result_copies_from_nested_array_storage() {
    let jit = compile(
        r#"package test

fn f(values: bits[9][2][2] id=1, index: bits[1] id=2) -> bits[9][2] {
  ret selected: bits[9][2] = array_index(values, indices=[index], id=3)
}
"#,
    );
    let values: [[u16; 2]; 2] = [[10, 11], [400, 401]];
    let index: u8 = 1;
    let mut output: [u16; 2] = [0; 2];
    let inputs = [
        std::ptr::from_ref(&values).cast::<u8>(),
        std::ptr::from_ref(&index).cast::<u8>(),
    ];
    // SAFETY: nested input and separate output arrays obey the recursive
    // native layout contract and do not overlap.
    unsafe {
        jit.run_native(&inputs, std::ptr::from_mut(&mut output).cast())
            .expect("subarray result should execute");
    }
    assert_eq!(output, [400, 401]);
}

#[test]
fn intermediate_array_construction_uses_scratch_storage() {
    let jit = compile(
        r#"package test

fn f(x: bits[8] id=1, y: bits[8] id=2, index: bits[2] id=3) -> bits[8] {
  made: bits[8][2] = array(x, y, id=4)
  ret selected: bits[8] = array_index(made, indices=[index], id=5)
}
"#,
    );
    assert_eq!(jit.run_u64(&[7, 13, 0]).expect("execute"), 7);
    assert_eq!(jit.run_u64(&[7, 13, 1]).expect("execute"), 13);
    assert_eq!(jit.run_u64(&[7, 13, 3]).expect("execute"), 13);
}

#[test]
fn ir_value_adapter_matches_native_result() {
    let jit = compile(
        r#"package test

fn f(x: bits[8] id=1, y: bits[8] id=2) -> bits[8] {
  ret difference: bits[8] = sub(x, y, id=3)
}
"#,
    );
    let result = jit
        .run_ir_values(&[
            IrValue::make_ubits(8, 3).unwrap(),
            IrValue::make_ubits(8, 5).unwrap(),
        ])
        .expect("execute");
    assert_eq!(result, IrValue::make_ubits(8, 254).unwrap());
}

#[test]
fn rejects_aggregate_functions_until_native_layout_support_is_added() {
    let package = Parser::new(
        r#"package test

fn f(x: bits[8] id=1) -> (bits[8], bits[8]) {
  ret pair: (bits[8], bits[8]) = tuple(x, x, id=2)
}
"#,
    )
    .parse_and_validate_package()
    .expect("test PIR should parse and validate");
    let error = match PirFunctionJit::compile(package.get_fn("f").unwrap()) {
        Ok(_) => panic!("aggregate result should not JIT compile yet"),
        Err(error) => error,
    };
    assert!(matches!(error, JitError::UnsupportedType(_)));
}
