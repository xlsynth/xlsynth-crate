// SPDX-License-Identifier: Apache-2.0

#![no_main]
use libfuzzer_sys::fuzz_target;

use xlsynth_g8r_fuzz::generate_upstream_formal_random_pir_pair;
use xlsynth_pir::ir;
use xlsynth_pir::ir_validate::validate_fn;
use xlsynth_pir::prove_equiv_via_toolchain::{
    prove_ir_fn_equiv_via_toolchain, ToolchainEquivResult,
};
use xlsynth_pir::simple_rebase::rebase_onto;

fn max_text_id(f: &ir::Fn) -> usize {
    f.nodes.iter().map(|n| n.text_id).max().unwrap_or(0)
}

fuzz_target!(|data: &[u8]| {
    // Require toolchain path for equivalence checking; without it this target
    // cannot check the property under test.
    if std::env::var("XLSYNTH_TOOLS").is_err() {
        panic!("XLSYNTH_TOOLS environment variable must be set for fuzzing.");
    }

    let _ = env_logger::builder().is_test(true).try_init();

    let (orig, desired) = generate_upstream_formal_random_pir_pair(data);

    // Precondition: signatures must match; guaranteed by constrained generation.
    assert_eq!(
        orig.get_type(),
        desired.get_type(),
        "paired random PIR generation must produce identical signatures"
    );

    // Rebase desired onto orig and prove semantic equivalence: rebased == desired.
    let mut next_id = max_text_id(&orig) + 1;
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
        top: Some(("rebased".to_string(), ir::MemberType::Function)),
    };
    if let Err(e) = validate_fn(&rebased, &pkg) {
        panic!("rebased IR failed composite validation: {}", e);
    }

    match prove_ir_fn_equiv_via_toolchain(&desired, &rebased) {
        ToolchainEquivResult::Proved => {}
        other => {
            // Treat tool infra failure as non-sample failure; but equivalence disproved
            // must panic.
            match other {
                ToolchainEquivResult::Error(_) => return,
                // No other variant currently, but keep match exhaustive for clarity.
                _ => panic!("rebase_onto failed equivalence: {:?}", other),
            }
        }
    }
});
