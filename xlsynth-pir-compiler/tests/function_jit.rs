// SPDX-License-Identifier: Apache-2.0

use xlsynth::IrValue;
use xlsynth_pir::ir_eval::{FnEvalResult, eval_fn};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir_compiler::{
    JitError, NativeTupleFieldLayout, NativeValueLayout, PirFunctionJit, ScalarLayout,
};

fn compile(ir: &str) -> PirFunctionJit {
    let package = Parser::new(ir)
        .parse_and_validate_package()
        .expect("test PIR should parse and validate");
    let function = package.get_fn("f").expect("function f should exist");
    PirFunctionJit::compile(function).expect("function should JIT compile")
}

fn assert_matches_evaluator(ir: &str, argument_sets: &[Vec<IrValue>]) {
    let package = Parser::new(ir)
        .parse_and_validate_package()
        .expect("test PIR should parse and validate");
    let function = package.get_fn("f").expect("function f should exist");
    let jit = PirFunctionJit::compile(function).expect("function should JIT compile");
    for args in argument_sets {
        let expected = match eval_fn(function, args) {
            FnEvalResult::Success(success) => success.value,
            FnEvalResult::Failure(_) => panic!("PIR evaluation failed for arguments {args:?}"),
        };
        let actual = jit
            .run_ir_values(args)
            .expect("JIT execution should succeed");
        assert_eq!(actual, expected, "JIT mismatch for arguments {args:?}");
    }
}

fn bits(width: usize, value: u64) -> IrValue {
    IrValue::make_ubits(width, value).unwrap()
}

fn array(width: usize, values: &[u64]) -> IrValue {
    IrValue::make_array(
        &values
            .iter()
            .map(|value| bits(width, *value))
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

fn tuple(elements: &[IrValue]) -> IrValue {
    IrValue::make_tuple(elements)
}

#[test]
fn compile_rejects_unvalidated_invalid_xls_node_semantics() {
    let package = Parser::new(
        r#"package test

fn f(x: bits[8] id=1, y: bits[7] id=2) -> bits[8] {
  ret add.3: bits[8] = add(x, y, id=3)
}
"#,
    )
    .parse_package()
    .expect("invalid semantics should still parse structurally");
    let error = match PirFunctionJit::compile(package.get_fn("f").unwrap()) {
        Err(error) => error,
        Ok(_) => panic!("invalid XLS node semantics should not compile"),
    };
    assert!(matches!(
        error,
        JitError::InvalidFunction(message) if message.contains("right operand")
    ));
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
fn rejects_unvalidated_out_of_bounds_static_bit_slice() {
    let package = Parser::new(
        r#"package test

fn f(x: bits[8] id=1) -> bits[2] {
  ret bit_slice.2: bits[2] = bit_slice(x, start=7, width=2, id=2)
}
"#,
    )
    .parse_package()
    .expect("unvalidated function should parse as PIR");
    let function = package.get_fn("f").expect("function f should exist");
    let error = match PirFunctionJit::compile(function) {
        Ok(_) => panic!("out-of-bounds bit_slice should not JIT compile"),
        Err(error) => error,
    };
    match error {
        JitError::InvalidFunction(message) => assert!(
            message.contains("bit_slice start 7") && message.contains("exceeds operand width 8"),
            "unexpected validation diagnostic: {message}"
        ),
        error => panic!("expected InvalidFunction, got {error}"),
    }
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
fn lowers_reverse_and_reduction_operations() {
    let reverse = compile(
        r#"package test

fn f(x: bits[5] id=1) -> bits[5] {
  ret reversed: bits[5] = reverse(x, id=2)
}
"#,
    );
    assert_eq!(reverse.run_u64(&[0b1_0010]).expect("execute"), 0b0_1001);

    let reductions = compile(
        r#"package test

fn f(x: bits[5] id=1) -> bits[3] {
  any: bits[1] = or_reduce(x, id=2)
  all: bits[1] = and_reduce(x, id=3)
  parity: bits[1] = xor_reduce(x, id=4)
  ret combined: bits[3] = concat(any, all, parity, id=5)
}
"#,
    );
    assert_eq!(reductions.run_u64(&[0]).expect("execute"), 0b000);
    assert_eq!(reductions.run_u64(&[0b1_1111]).expect("execute"), 0b111);
    assert_eq!(reductions.run_u64(&[0b1_0011]).expect("execute"), 0b101);
}

#[test]
fn right_shifts_and_divmod_match_pir_evaluator_on_edge_values() {
    for op in ["shrl", "shra", "udiv", "sdiv", "umod", "smod"] {
        let ir = format!(
            r#"package test

fn f(x: bits[5] id=1, y: bits[5] id=2) -> bits[5] {{
  ret result: bits[5] = {op}(x, y, id=3)
}}
"#
        );
        let args = (0..32)
            .flat_map(|x| (0..32).map(move |y| vec![bits(5, x), bits(5, y)]))
            .collect::<Vec<_>>();
        assert_matches_evaluator(&ir, &args);
    }
    for width in [8, 16, 32, 64] {
        let minimum = 1u64 << (width - 1);
        let mask = if width == 64 {
            u64::MAX
        } else {
            (1u64 << width) - 1
        };
        let ir = format!(
            r#"package test

fn f(x: bits[{width}] id=1, y: bits[{width}] id=2) -> bits[{width}] {{
  ret result: bits[{width}] = sdiv(x, y, id=3)
}}
"#
        );
        assert_matches_evaluator(
            &ir,
            &[
                vec![bits(width, minimum), bits(width, mask)],
                vec![bits(width, minimum), bits(width, 0)],
                vec![bits(width, mask >> 1), bits(width, 0)],
            ],
        );
    }
}

#[test]
fn arbitrary_width_multiply_matches_pir_evaluator() {
    assert_matches_evaluator(
        r#"package test

fn f(lhs: bits[24] id=1, rhs: bits[24] id=2) -> bits[48] {
  ret result: bits[48] = umul(lhs, rhs, id=3)
}
"#,
        &[
            vec![bits(24, 0), bits(24, 0x12_3456)],
            vec![bits(24, 0xff_ffff), bits(24, 0xab_cdef)],
        ],
    );
    assert_matches_evaluator(
        r#"package test

fn f(lhs: bits[7] id=1, rhs: bits[32] id=2) -> bits[32] {
  ret result: bits[32] = umul(lhs, rhs, id=3)
}
"#,
        &[
            vec![bits(7, 0x7f), bits(32, 0x1234_5678)],
            vec![bits(7, 3), bits(32, 17)],
        ],
    );
    assert_matches_evaluator(
        r#"package test

fn f(lhs: bits[5] id=1, rhs: bits[9] id=2) -> bits[20] {
  ret result: bits[20] = smul(lhs, rhs, id=3)
}
"#,
        &[
            vec![bits(5, 0x1f), bits(9, 3)],
            vec![bits(5, 0x11), bits(9, 0x1ff)],
        ],
    );
    assert_matches_evaluator(
        r#"package test

fn f(lhs: bits[10] id=1, rhs: bits[9] id=2) -> bits[6] {
  ret result: bits[6] = smul(lhs, rhs, id=3)
}
"#,
        &[
            vec![bits(10, 0x3ff), bits(9, 3)],
            vec![bits(10, 0x201), bits(9, 0x1ff)],
        ],
    );
}

#[test]
fn gate_select_and_encoding_operations_match_pir_evaluator() {
    assert_matches_evaluator(
        r#"package test

fn f(pred: bits[1] id=1, value: bits[8] id=2) -> bits[8] {
  ret result: bits[8] = gate(pred, value, id=3)
}
"#,
        &[
            vec![bits(1, 0), bits(8, 0xab)],
            vec![bits(1, 1), bits(8, 0xab)],
        ],
    );
    assert_matches_evaluator(
        r#"package test

fn f(selector: bits[3] id=1, a: bits[8] id=2, b: bits[8] id=3, d: bits[8] id=4) -> bits[8] {
  selected: bits[8] = sel(selector, cases=[a, b], default=d, id=5)
  priority: bits[8] = priority_sel(selector, cases=[a, b, selected], default=d, id=6)
  ret result: bits[8] = one_hot_sel(selector, cases=[priority, b, a], id=7)
}
"#,
        &(0..8)
            .map(|selector| {
                vec![
                    bits(3, selector),
                    bits(8, 0x12),
                    bits(8, 0x41),
                    bits(8, 0x80),
                ]
            })
            .collect::<Vec<_>>(),
    );
    assert_matches_evaluator(
        r#"package test

fn f(x: bits[5] id=1) -> bits[3] {
  hot: bits[6] = one_hot(x, lsb_prio=true, id=2)
  ret encoded: bits[3] = encode(hot, id=3)
}
"#,
        &(0..32).map(|x| vec![bits(5, x)]).collect::<Vec<_>>(),
    );
    assert_matches_evaluator(
        r#"package test

fn f(x: bits[3] id=1) -> bits[6] {
  ret decoded: bits[6] = decode(x, width=6, id=2)
}
"#,
        &(0..8).map(|x| vec![bits(3, x)]).collect::<Vec<_>>(),
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

    let x: u8 = 7;
    let y: u8 = 13;
    let index: u8 = 1;
    let mut output = 0u8;
    let inputs = [
        std::ptr::from_ref(&x).cast::<u8>(),
        std::ptr::from_ref(&y).cast::<u8>(),
        std::ptr::from_ref(&index).cast::<u8>(),
    ];
    let mut scratch = vec![
        0u64;
        jit.scratch_byte_count()
            .div_ceil(std::mem::size_of::<u64>())
    ];
    assert!(jit.scratch_byte_count() > 0);
    assert!(jit.scratch_alignment() <= std::mem::align_of::<u64>());
    // SAFETY: all values use their native scalar layouts, and `scratch` is
    // sufficiently sized and aligned for the published scratch requirement.
    unsafe {
        jit.run_native_with_scratch(
            &inputs,
            std::ptr::from_mut(&mut output).cast(),
            scratch.as_mut_ptr().cast(),
            scratch.len() * std::mem::size_of::<u64>(),
        )
        .expect("execution with caller-owned scratch should succeed");
    }
    assert_eq!(output, 13);
}

#[test]
fn zero_index_array_index_preserves_native_array_value() {
    assert_matches_evaluator(
        r#"package test

fn f(values: bits[8][2] id=1) -> bits[8][2] {
  ret result: bits[8][2] = array_index(values, indices=[], id=2)
}
"#,
        &[vec![array(8, &[9, 27])]],
    );
    assert_matches_evaluator(
        r#"package test

fn f(values: bits[8][2] id=1, replacement: bits[8][2] id=2) -> bits[8][2] {
  ret result: bits[8][2] = array_update(values, replacement, indices=[], id=3)
}
"#,
        &[vec![array(8, &[9, 27]), array(8, &[41, 63])]],
    );
}

#[test]
fn native_array_concat_slice_and_update_match_pir_evaluator() {
    assert_matches_evaluator(
        r#"package test

fn f(lhs: bits[8][1] id=1, middle: bits[8][2] id=2, rhs: bits[8][1] id=3) -> bits[8][4] {
  ret joined: bits[8][4] = array_concat(lhs, middle, rhs, id=4)
}
"#,
        &[vec![array(8, &[1]), array(8, &[2, 3]), array(8, &[4])]],
    );
    assert_matches_evaluator(
        r#"package test

fn f(values: bits[8][2] id=1) -> bits[8][2] {
  ret joined: bits[8][2] = array_concat(values, id=2)
}
"#,
        &[vec![array(8, &[7, 9])]],
    );
    assert_matches_evaluator(
        r#"package test

fn f(values: bits[8][3] id=1, start: bits[3] id=2) -> bits[8][4] {
  ret sliced: bits[8][4] = array_slice(values, start, width=4, id=3)
}
"#,
        &(0..8)
            .map(|start| vec![array(8, &[10, 20, 30]), bits(3, start)])
            .collect::<Vec<_>>(),
    );
    assert_matches_evaluator(
        r#"package test

fn f(values: bits[8][2][2] id=1, replacement: bits[8] id=2, i: bits[2] id=3, j: bits[2] id=4) -> bits[8][2][2] {
  ret updated: bits[8][2][2] = array_update(values, replacement, indices=[i, j], id=5)
}
"#,
        &(0..4)
            .flat_map(|i| {
                (0..4).map(move |j| {
                    vec![
                        IrValue::make_array(&[array(8, &[1, 2]), array(8, &[3, 4])]).unwrap(),
                        bits(8, 99),
                        bits(2, i),
                        bits(2, j),
                    ]
                })
            })
            .collect::<Vec<_>>(),
    );
}

#[test]
fn selection_and_gate_operate_on_native_arrays() {
    assert_matches_evaluator(
        r#"package test

fn f(selector: bits[2] id=1, pred: bits[1] id=2, a: bits[8][2] id=3, b: bits[8][2] id=4, d: bits[8][2] id=5) -> bits[8][2] {
  chosen: bits[8][2] = sel(selector, cases=[a, b], default=d, id=6)
  gated: bits[8][2] = gate(pred, chosen, id=7)
  ret result: bits[8][2] = one_hot_sel(selector, cases=[gated, b], id=8)
}
"#,
        &[
            vec![
                bits(2, 0),
                bits(1, 0),
                array(8, &[1, 2]),
                array(8, &[4, 8]),
                array(8, &[16, 32]),
            ],
            vec![
                bits(2, 1),
                bits(1, 1),
                array(8, &[1, 2]),
                array(8, &[4, 8]),
                array(8, &[16, 32]),
            ],
            vec![
                bits(2, 3),
                bits(1, 1),
                array(8, &[1, 2]),
                array(8, &[4, 8]),
                array(8, &[16, 32]),
            ],
        ],
    );
}

#[test]
fn aggregate_comparisons_and_default_only_sel_match_pir_evaluator() {
    assert_matches_evaluator(
        r#"package test

fn f(lhs: (bits[8], bits[4][2]) id=1, rhs: (bits[8], bits[4][2]) id=2) -> (bits[1], bits[1]) {
  same: bits[1] = eq(lhs, rhs, id=3)
  different: bits[1] = ne(lhs, rhs, id=4)
  ret result: (bits[1], bits[1]) = tuple(same, different, id=5)
}
"#,
        &[
            vec![
                tuple(&[bits(8, 3), array(4, &[1, 2])]),
                tuple(&[bits(8, 3), array(4, &[1, 2])]),
            ],
            vec![
                tuple(&[bits(8, 3), array(4, &[1, 2])]),
                tuple(&[bits(8, 3), array(4, &[1, 4])]),
            ],
        ],
    );
    assert_matches_evaluator(
        r#"package test

fn f(selector: bits[2] id=1, fallback: bits[8][2] id=2) -> bits[8][2] {
  ret result: bits[8][2] = sel(selector, cases=[], default=fallback, id=3)
}
"#,
        &[vec![bits(2, 3), array(8, &[11, 27])]],
    );
}

#[test]
fn native_tuple_layout_matches_repr_c_storage() {
    #[repr(C)]
    struct Pair {
        low: u8,
        high: u32,
    }

    let jit = compile(
        r#"package test

fn f(low: bits[8] id=1, high: bits[17] id=2) -> (bits[8], bits[17]) {
  ret pair: (bits[8], bits[17]) = tuple(low, high, id=3)
}
"#,
    );
    assert_eq!(
        jit.result_layout(),
        &NativeValueLayout::Tuple {
            fields: vec![
                NativeTupleFieldLayout {
                    layout: Box::new(NativeValueLayout::Scalar(ScalarLayout {
                        bit_count: 8,
                        byte_count: 1,
                    })),
                    offset: std::mem::offset_of!(Pair, low),
                },
                NativeTupleFieldLayout {
                    layout: Box::new(NativeValueLayout::Scalar(ScalarLayout {
                        bit_count: 17,
                        byte_count: 4,
                    })),
                    offset: std::mem::offset_of!(Pair, high),
                },
            ],
            byte_count: std::mem::size_of::<Pair>(),
            alignment: std::mem::align_of::<Pair>(),
        }
    );

    let low: u8 = 0x91;
    let high: u32 = 0x1_2345;
    let mut output = Pair { low: 0, high: 0 };
    let inputs = [
        std::ptr::from_ref(&low).cast::<u8>(),
        std::ptr::from_ref(&high).cast::<u8>(),
    ];
    // SAFETY: `Pair` has the `#[repr(C)]` tuple layout published by the JIT,
    // and the scalar inputs use their respective native carriers.
    unsafe {
        jit.run_native(&inputs, std::ptr::from_mut(&mut output).cast())
            .expect("native tuple construction should execute");
    }
    assert_eq!(output.low, low);
    assert_eq!(output.high, high);
}

#[test]
fn nested_tuples_arrays_and_tuple_selection_match_pir_evaluator() {
    assert_matches_evaluator(
        r#"package test

fn f(selector: bits[1] id=1, x: bits[8] id=2, y: bits[8] id=3) -> ((bits[8], bits[8][2]), bits[8]) {
  values: bits[8][2] = array(x, y, id=4)
  inner: (bits[8], bits[8][2]) = tuple(x, values, id=5)
  selected: (bits[8], bits[8][2]) = sel(selector, cases=[inner], default=inner, id=6)
  gated: (bits[8], bits[8][2]) = gate(selector, selected, id=7)
  ret result: ((bits[8], bits[8][2]), bits[8]) = tuple(gated, y, id=8)
}
"#,
        &[
            vec![bits(1, 0), bits(8, 3), bits(8, 9)],
            vec![bits(1, 1), bits(8, 3), bits(8, 9)],
        ],
    );
    assert_matches_evaluator(
        r#"package test

fn f(value: ((bits[8], bits[8][2]), bits[8]) id=1) -> bits[8] {
  inner: (bits[8], bits[8][2]) = tuple_index(value, index=0, id=2)
  values: bits[8][2] = tuple_index(inner, index=1, id=3)
  index: bits[1] = literal(value=1, id=4)
  ret result: bits[8] = array_index(values, indices=[index], id=5)
}
"#,
        &[vec![tuple(&[
            tuple(&[bits(8, 4), array(8, &[7, 11])]),
            bits(8, 17),
        ])]],
    );
}

#[test]
fn dynamic_bit_slice_and_update_match_pir_evaluator() {
    assert_matches_evaluator(
        r#"package test

fn f(x: bits[8] id=1, start: bits[4] id=2, replacement: bits[5] id=3) -> (bits[5], bits[8]) {
  sliced: bits[5] = dynamic_bit_slice(x, start, width=5, id=4)
  updated: bits[8] = bit_slice_update(x, start, replacement, id=5)
  ret result: (bits[5], bits[8]) = tuple(sliced, updated, id=6)
}
"#,
        &(0..16)
            .map(|start| vec![bits(8, 0b1010_0110), bits(4, start), bits(5, 0b1_1101)])
            .collect::<Vec<_>>(),
    );
}

#[test]
fn extension_operations_match_pir_evaluator() {
    assert_matches_evaluator(
        r#"package test

fn f(x: bits[5] id=1, y: bits[5] id=2, c_in: bits[1] id=3, count: bits[4] id=4) -> (bits[1], bits[3], bits[3], bits[4], (bits[8], bits[4]), bits[8], bits[8]) {
  carry: bits[1] = ext_carry_out(x, y, c_in, id=5)
  prio_lsb: bits[3] = ext_prio_encode(x,lsb_prio=true,id=6)
  prio_msb: bits[3] = ext_prio_encode(x,lsb_prio=false,id=7)
  zeroes: bits[4] = ext_clz(x, offset=2, new_bit_count=4, id=8)
  normalized: (bits[8], bits[4]) = ext_normalize_left(x, shift_offset=1, normalized_bit_count=8, clz_bit_count=4, id=9)
  mask: bits[8] = ext_mask_low(count, id=10)
  sum: bits[8] = ext_nary_add(x, y, count, signed=[true, false, false], negated=[false, true, false], arch=brent_kung, id=11)
  ret result: (bits[1], bits[3], bits[3], bits[4], (bits[8], bits[4]), bits[8], bits[8]) = tuple(carry, prio_lsb, prio_msb, zeroes, normalized, mask, sum, id=12)
}
"#,
        &(0..32)
            .flat_map(|x| {
                [
                    vec![bits(5, x), bits(5, 0), bits(1, 0), bits(4, 0)],
                    vec![bits(5, x), bits(5, 31), bits(1, 1), bits(4, 9)],
                ]
            })
            .collect::<Vec<_>>(),
    );
    assert_matches_evaluator(
        r#"package test

fn f(lhs: bits[64] id=1, rhs: bits[64] id=2, c_in: bits[1] id=3) -> bits[1] {
  ret carry: bits[1] = ext_carry_out(lhs, rhs, c_in, id=4)
}
"#,
        &[
            vec![bits(64, u64::MAX), bits(64, 0), bits(1, 1)],
            vec![bits(64, u64::MAX), bits(64, 1), bits(1, 0)],
            vec![bits(64, 1), bits(64, 1), bits(1, 0)],
        ],
    );
    assert_matches_evaluator(
        r#"package test

fn f() -> bits[7] {
  ret sum: bits[7] = ext_nary_add(signed=[], negated=[], id=1)
}
"#,
        &[vec![]],
    );
}

#[test]
fn partial_product_multiply_tuples_match_the_llvm_jit_convention() {
    let unsigned = compile(
        r#"package test

fn f(lhs: bits[4] id=1, rhs: bits[4] id=2) -> (bits[8], bits[8]) {
  ret result: (bits[8], bits[8]) = umulp(lhs, rhs, id=3)
}
"#,
    );
    assert_eq!(
        unsigned
            .run_ir_values(&[bits(4, 3), bits(4, 5)])
            .expect("unsigned partial multiply should execute"),
        tuple(&[bits(8, 0x47), bits(8, 0xc8)])
    );
    assert_matches_evaluator(
        r#"package test

fn f(lhs: bits[4] id=1, rhs: bits[4] id=2) -> (bits[8], bits[8]) {
  ret result: (bits[8], bits[8]) = umulp(lhs, rhs, id=3)
}
"#,
        &(0..16)
            .flat_map(|lhs| (0..16).map(move |rhs| vec![bits(4, lhs), bits(4, rhs)]))
            .collect::<Vec<_>>(),
    );
    assert_matches_evaluator(
        r#"package test

fn f(lhs: bits[5] id=1, rhs: bits[4] id=2) -> (bits[8], bits[8]) {
  ret result: (bits[8], bits[8]) = smulp(lhs, rhs, id=3)
}
"#,
        &(0..32)
            .flat_map(|lhs| (0..16).map(move |rhs| vec![bits(5, lhs), bits(4, rhs)]))
            .collect::<Vec<_>>(),
    );
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
