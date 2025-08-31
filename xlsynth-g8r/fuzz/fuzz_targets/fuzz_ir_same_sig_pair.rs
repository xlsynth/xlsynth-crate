// SPDX-License-Identifier: Apache-2.0

#![no_main]

use libfuzzer_sys::fuzz_target;
use xlsynth::{IrPackage, IrType};
use xlsynth_g8r::xls_ir::ir_fuzz::{generate_ir_fn, FuzzSampleSameTypedPair};

fuzz_target!(|pair: FuzzSampleSameTypedPair| {
    // Skip degenerate samples early.
    if pair.first.ops.is_empty()
        || pair.second.ops.is_empty()
        || pair.first.input_bits == 0
        || pair.second.input_bits == 0
    {
        // Degenerate generator inputs (no ops or zero-width inputs) are not
        // interesting for this target and can arise frequently. We intentionally
        // skip rather than crash to avoid biasing the corpus toward trivial cases.
        return;
    }

    let _ = env_logger::builder().is_test(true).try_init();

    let mut pkg1 = IrPackage::new("first")
        .expect("IrPackage::new should not fail; treat as infra error");
    let func1 = match generate_ir_fn(
        pair.first.input_bits,
        pair.first.ops.clone(),
        &mut pkg1,
        None,
    ) {
        Ok(f) => f,
        Err(_) => {
            // The generator can produce constructs not yet supported; skip such cases.
            return;
        }
    };

    let mut pkg2 = IrPackage::new("second")
        .expect("IrPackage::new should not fail; treat as infra error");
    let func2 = match generate_ir_fn(
        pair.second.input_bits,
        pair.second.ops.clone(),
        &mut pkg2,
        None,
    ) {
        Ok(f) => f,
        Err(_) => {
            // The generator can produce constructs not yet supported; skip such cases.
            return;
        }
    };

    let t1 = func1
        .get_type()
        .expect("function type query should succeed");
    let t2 = func2
        .get_type()
        .expect("function type query should succeed");

    assert_eq!(t1.param_count(), t2.param_count());
    for i in 0..t1.param_count() {
        let p1: IrType = t1.param_type(i).expect("param type");
        let p2: IrType = t2.param_type(i).expect("param type");
        assert_eq!(
            p1.to_string(),
            p2.to_string(),
            "param {i} width mismatch"
        );
    }
    assert_eq!(
        t1.return_type().to_string(),
        t2.return_type().to_string(),
        "return width mismatch"
    );
});
