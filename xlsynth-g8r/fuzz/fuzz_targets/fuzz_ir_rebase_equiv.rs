// SPDX-License-Identifier: Apache-2.0

#![no_main]
use libfuzzer_sys::fuzz_target;

use xlsynth_g8r::equiv::types::EquivResult;
use xlsynth_g8r::equiv::prove_equiv_via_toolchain::{self};
use xlsynth_pir::ir_fuzz::{FuzzSampleSameTypedPair, generate_ir_fn};
use xlsynth_pir::ir_validate::validate_fn;
use xlsynth_pir::simple_rebase::rebase_onto;
use xlsynth_pir::{ir, ir_parser};

fn max_text_id(f: &ir::Fn) -> usize {
    f.nodes.iter().map(|n| n.text_id).max().unwrap_or(0)
}

fuzz_target!(|pair: FuzzSampleSameTypedPair| {
    // Require toolchain path for equivalence checking; without it this target
    // cannot check the property under test.
    if std::env::var("XLSYNTH_TOOLS").is_err() {
        panic!("XLSYNTH_TOOLS environment variable must be set for fuzzing.");
    }

    // Skip degenerate samples early.
    if pair.first.ops.is_empty()
        || pair.second.ops.is_empty()
        || pair.first.input_bits == 0
        || pair.second.input_bits == 0
    {
        // Degenerate generator inputs (no ops or zero-width inputs) are not
        // informative for this property.
        return;
    }

    let _ = env_logger::builder().is_test(true).try_init();

    // 1) Build orig IR from the first sample
    let mut pkg_orig = xlsynth::IrPackage::new("fuzz_pkg_orig").unwrap();
    if let Err(_) = generate_ir_fn(
        pair.first.input_bits,
        pair.first.ops.clone(),
        &mut pkg_orig,
        None,
    ) {
        // Generator can produce temporarily unsupported constructs; not a sample
        // failure.
        return;
    }

    // 2) Build desired IR from the second sample
    let mut pkg_desired = xlsynth::IrPackage::new("fuzz_pkg_desired").unwrap();
    if let Err(_) = generate_ir_fn(
        pair.second.input_bits,
        pair.second.ops.clone(),
        &mut pkg_desired,
        None,
    ) {
        // Generator can produce temporarily unsupported constructs; not a sample
        // failure.
        return;
    }

    // 3) Parse both packages into Rust IR and obtain tops
    let orig_pkg = ir_parser::Parser::new(&pkg_orig.to_string())
        .parse_and_validate_package()
        .unwrap();
    let desired_pkg = ir_parser::Parser::new(&pkg_desired.to_string())
        .parse_and_validate_package()
        .unwrap();
    let orig = orig_pkg.get_top().unwrap();
    let desired = desired_pkg.get_top().unwrap().clone();

    // 4) Precondition: signatures must match; guaranteed by FuzzSampleSameTypedPair
    assert_eq!(
        orig.get_type(),
        desired.get_type(),
        "FuzzSampleSameTypedPair must produce functions with identical signatures"
    );

    // 5) Rebase desired onto orig and prove semantic equivalence: rebase_onto(orig,
    //    desired) == desired
    let mut next_id = max_text_id(orig) + 1;
    let rebased: ir::Fn = rebase_onto(&desired, &orig, "rebased", || {
        let id = next_id;
        next_id += 1;
        id
    });

    // 5a) Verify the rebased function via the composite validator
    let pkg = ir::Package {
        name: "rebased_pkg".to_string(),
        file_table: ir::FileTable::new(),
        members: vec![ir::PackageMember::Function(rebased.clone())],
        top_name: Some("rebased".to_string()),
    };
    if let Err(e) = validate_fn(&rebased, &pkg) {
        panic!("rebased IR failed composite validation: {}", e);
    }

    match prove_equiv_via_toolchain::prove_ir_fn_equiv_via_toolchain(&desired, &rebased) {
        EquivResult::Proved => {}
        other => {
            // Treat tool infra failure as non-sample failure; but equivalence disproved
            // must panic.
            match other {
                EquivResult::Error(_) => return,
                // No other variant currently, but keep match exhaustive for clarity.
                _ => panic!("rebase_onto failed equivalence: {:?}", other),
            }
        }
    }
});
