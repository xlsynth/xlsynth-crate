// SPDX-License-Identifier: Apache-2.0

#![no_main]

use std::sync::Once;

use libfuzzer_sys::fuzz_target;
use rand::rngs::StdRng;
use rand::seq::IteratorRandom;
use rand::SeedableRng;
use xlsynth::IrPackage;
use xlsynth_mcmc_pir::transforms::get_all_pir_transforms;
use xlsynth_pir::ir;
use xlsynth_pir::ir_fuzz::{generate_ir_fn, FuzzSample};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_utils::compact_and_toposort_in_place;
use xlsynth_pir::ir_validate;
use xlsynth_prover::prover::types::{
    AssertionSemantics, EquivParallelism, EquivResult, ProverFn,
};
use xlsynth_prover::prover::{prover_for_choice, Prover, SolverChoice};

const NUM_STEPS: usize = 32;
const MAX_TRANSFORM_DRAWS: usize = NUM_STEPS * 32;

static INIT_LOGGER: Once = Once::new();

fuzz_target!(|sample: FuzzSample| {
    INIT_LOGGER.call_once(|| {
        let _ = env_logger::Builder::from_env(env_logger::Env::default())
            .is_test(true)
            .try_init();
    });

    let mut xls_pkg =
        IrPackage::new("fuzz_pir_transform_arbitrary").expect("IrPackage::new should succeed");
    if let Err(e) = generate_ir_fn(sample.ops.clone(), &mut xls_pkg, None) {
        // Early-return rationale: unsupported generator outputs are outside
        // this target's transform-soundness scope; see FUZZ.md.
        log::debug!("generate_ir_fn failed, skipping sample: {e}");
        return;
    }

    let initial_ir_text = xls_pkg.to_string();
    let mut cur_pkg = match Parser::new(&initial_ir_text).parse_and_validate_package() {
        Ok(pkg) => pkg,
        Err(e) => {
            // Early-return rationale: this sample does not give us a valid
            // starting PIR package, so transform application is not being
            // exercised. See FUZZ.md.
            log::debug!("initial parse/validate failed, skipping sample: {e}");
            return;
        }
    };
    if cur_pkg.get_top_fn().is_none() {
        // Early-return rationale: generated package has no function to mutate;
        // this is a degenerate harness input rather than a transform bug.
        return;
    }

    let mut transforms = get_all_pir_transforms();
    if transforms.is_empty() {
        panic!("get_all_pir_transforms unexpectedly returned an empty transform set");
    }
    let mut rng = make_rng(&initial_ir_text);
    let prover = make_in_process_prover();

    let mut successful_steps = 0usize;
    for _ in 0..MAX_TRANSFORM_DRAWS {
        if successful_steps >= NUM_STEPS {
            break;
        }

        let transform = transforms
            .iter_mut()
            .choose(&mut rng)
            .expect("transform set should be non-empty");
        let cur_fn = cur_pkg
            .get_top_fn()
            .expect("current package should retain a top function");
        let candidates = transform.find_candidates(cur_fn);
        let Some(candidate) = candidates.into_iter().choose(&mut rng) else {
            log::debug!("transform {:?}: no candidates", transform.kind());
            continue;
        };

        let mut next_pkg = cur_pkg.clone();
        let next_fn = next_pkg
            .get_top_fn_mut()
            .expect("cloned package should retain a top function");
        if let Err(e) = transform.apply(next_fn, &candidate.location) {
            log::debug!(
                "transform {:?} apply failed at {:?}, skipping candidate: {e}",
                transform.kind(),
                candidate.location,
            );
            continue;
        }
        if let Err(e) = compact_and_toposort_in_place(next_fn) {
            log::debug!(
                "transform {:?} produced non-toposortable PIR at {:?}, skipping candidate: {e}",
                transform.kind(),
                candidate.location,
            );
            continue;
        }
        if let Err(e) = ir_validate::validate_package(&next_pkg) {
            panic!(
                "transform {:?} produced invalid PIR at {:?}: {e:?}\n\nbefore:\n{}\n\nafter:\n{}",
                transform.kind(),
                candidate.location,
                cur_pkg,
                next_pkg,
            );
        }

        if candidate.always_equivalent {
            prove_candidate_equivalent(
                prover.as_ref(),
                &cur_pkg,
                &next_pkg,
                transform.kind(),
                &candidate.location,
            );
        }

        cur_pkg = next_pkg;
        successful_steps += 1;
    }
});

fn make_rng(initial_ir_text: &str) -> StdRng {
    StdRng::from_seed(*blake3::hash(initial_ir_text.as_bytes()).as_bytes())
}

fn prove_candidate_equivalent(
    prover: &dyn Prover,
    cur_pkg: &ir::Package,
    next_pkg: &ir::Package,
    kind: xlsynth_mcmc_pir::transforms::PirTransformKind,
    location: &xlsynth_mcmc_pir::transforms::TransformLocation,
) {
    prove_candidate_equivalent_with_result(
        cur_pkg,
        next_pkg,
        kind,
        location,
        prover.prove_ir_equiv(
            &ProverFn::new(expect_top_fn(cur_pkg), Some(cur_pkg)),
            &ProverFn::new(expect_top_fn(next_pkg), Some(next_pkg)),
            EquivParallelism::SingleThreaded,
            AssertionSemantics::Same,
            None,
            false,
        ),
    )
}

fn make_in_process_prover() -> Box<dyn Prover> {
    if cfg!(not(any(feature = "has-bitwuzla", feature = "has-boolector"))) {
        panic!(
            "fuzz_pir_transform_arbitrary refuses SolverChoice::Auto when it \
             would fall back to Toolchain; enable with-bitwuzla-system, \
             with-bitwuzla-built, with-boolector-system, or with-boolector-built"
        );
    }
    prover_for_choice(SolverChoice::Auto, None)
}

fn expect_top_fn(pkg: &ir::Package) -> &ir::Fn {
    pkg.get_top_fn().expect("package should have a top fn")
}

fn prove_candidate_equivalent_with_result(
    cur_pkg: &ir::Package,
    next_pkg: &ir::Package,
    kind: xlsynth_mcmc_pir::transforms::PirTransformKind,
    location: &xlsynth_mcmc_pir::transforms::TransformLocation,
    equiv_result: EquivResult,
) {
    match equiv_result {
        EquivResult::Proved => {}
        EquivResult::Disproved {
            lhs_inputs,
            rhs_inputs,
            lhs_output,
            rhs_output,
        } => {
            panic!(
                "transform {:?} at {:?} was marked always_equivalent but was disproved\n\
                 lhs_inputs={lhs_inputs:?}\n\
                 rhs_inputs={rhs_inputs:?}\n\
                 lhs_output={lhs_output:?}\n\
                 rhs_output={rhs_output:?}\n\n\
                 before:\n{}\n\n\
                 after:\n{}",
                kind, location, cur_pkg, next_pkg,
            );
        }
        EquivResult::ToolchainDisproved(msg) | EquivResult::Error(msg) => {
            panic!(
                "formal equivalence failed for transform {:?} at {:?}: {msg}\n\nbefore:\n{}\n\n\
                 after:\n{}",
                kind, location, cur_pkg, next_pkg,
            );
        }
    }
}
