// SPDX-License-Identifier: Apache-2.0

use arbitrary::Unstructured;
use xlsynth::IrPackage;
use xlsynth_test_helpers::ir_fuzz::{generate_ir_fn, FuzzBinop, FuzzOp, FuzzSample};

fn build_concat_literals_sample(literals: usize) -> FuzzSample {
    assert!(literals >= 2);
    let mut ops: Vec<FuzzOp> = Vec::new();
    // Push N literals of 8 bits each at indices 1..=N.
    for _ in 0..literals {
        ops.push(FuzzOp::Literal { bits: 8, value: 0 });
    }
    // After pushing N literals, available_nodes = 1 + N, indices 0..N present.
    // First concat: (1, 2) -> new node at index 1+N.
    let mut current_idx: u8 = (1 + literals) as u8; // index of the next created node
    ops.push(FuzzOp::Binop(FuzzBinop::Concat, 1, 2));
    // Now iteratively concat the running result with literal i.
    // The running result is always at `current_idx` after each push.
    for lit_idx in 3..=literals as u8 {
        ops.push(FuzzOp::Binop(FuzzBinop::Concat, current_idx, lit_idx));
        current_idx = current_idx + 1;
    }
    FuzzSample {
        input_bits: 1,
        ops,
    }
}

#[test]
/// Ensures `FuzzSample::gen_with_same_signature` preserves return bit-widths
/// when the original function's flat width fits in the fuzz op's `u8` fields.
///
/// We construct a function returning 248 bits by concatenating 31 literals of
/// 8 bits each (31 * 8 = 248). The paired sample generated should have the
/// exact same return width; this would have previously mismatched if any
/// intermediate width calculations were truncated or miscomputed.
fn gen_with_same_signature_matches_under_threshold() {
    // 31 literals * 8 bits = 248 bits (<= 255)
    let orig = build_concat_literals_sample(31);
    let mut pkg1 = IrPackage::new("first").expect("IrPackage::new");
    let f1 = generate_ir_fn(orig.input_bits, orig.ops.clone(), &mut pkg1).expect("generate_ir_fn");
    let ret_w1 = f1.get_type().unwrap().return_type().get_flat_bit_count();
    assert_eq!(ret_w1, 248);

    let mut bytes = [0u8; 64];
    let mut u = Unstructured::new(&bytes);
    let sample = FuzzSample::gen_with_same_signature(&orig, &mut u).expect("gen_with_same_signature");

    let mut pkg2 = IrPackage::new("second").expect("IrPackage::new");
    let f2 = generate_ir_fn(sample.input_bits, sample.ops.clone(), &mut pkg2).expect("generate_ir_fn");
    let ret_w2 = f2.get_type().unwrap().return_type().get_flat_bit_count();
    assert_eq!(ret_w1, ret_w2);
}

#[test]
/// Ensures `FuzzSample::gen_with_same_signature` rejects oversized widths
/// instead of silently truncating when they do not fit in `u8`.
///
/// We construct a function returning 264 bits by concatenating 33 literals of
/// 8 bits each (33 * 8 = 264). Prior to the fix, the generator truncated the
/// width to `u8`, causing mismatched signatures (e.g. 264 -> 8-bit overflow to
/// an unexpected value). Now the generator fails the sample instead of
/// producing an inconsistent pair.
fn gen_with_same_signature_errs_over_threshold() {
    // 33 literals * 8 bits = 264 bits (> 255); generation should now error rather than truncate widths.
    let orig = build_concat_literals_sample(33);
    let mut pkg1 = IrPackage::new("first").expect("IrPackage::new");
    let f1 = generate_ir_fn(orig.input_bits, orig.ops.clone(), &mut pkg1).expect("generate_ir_fn");
    let ret_w1 = f1.get_type().unwrap().return_type().get_flat_bit_count();
    assert_eq!(ret_w1, 264);

    let mut bytes = [0u8; 64];
    let mut u = Unstructured::new(&bytes);
    let res = FuzzSample::gen_with_same_signature(&orig, &mut u);
    assert!(res.is_err(), "expected gen_with_same_signature to error for large widths");
}
