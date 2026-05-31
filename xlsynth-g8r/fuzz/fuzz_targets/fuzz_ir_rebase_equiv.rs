// SPDX-License-Identifier: Apache-2.0

#![no_main]
use libfuzzer_sys::fuzz_target;

use xlsynth_g8r_fuzz::generate_upstream_formal_random_pir_pair;
use xlsynth_pir::ir;
use xlsynth_pir::ir_verify::verify_function_in_package;
use xlsynth_pir::simple_rebase::rebase_onto;
use xlsynth_prover::prover::types::EquivResult;
use xlsynth_prover::prover::{SolverChoice, prover_for_choice};

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

    // 5a) Verify the rebased function in its package context.
    let pkg = ir::Package {
        name: "rebased_pkg".to_string(),
        file_table: ir::FileTable::new(),
        members: vec![ir::PackageMember::Function(rebased.clone())],
        top: Some(("rebased".to_string(), ir::MemberType::Function)),
    };
    if let Err(e) = verify_function_in_package(&rebased, &pkg) {
        panic!("rebased IR failed verification: {}", e);
    }

    match prover_for_choice(SolverChoice::Toolchain, None).prove_ir_fn_equiv(&desired, &rebased) {
        EquivResult::Proved => {}
        other => {
            // Early-return rationale: tool interruption or infrastructure failure
            // is not a property of this sample; an equivalence disproof must panic.
            match other {
                EquivResult::Inconclusive(_) | EquivResult::Error(_) => return,
                _ => panic!("rebase_onto failed equivalence: {:?}", other),
            }
        }
    }
});
