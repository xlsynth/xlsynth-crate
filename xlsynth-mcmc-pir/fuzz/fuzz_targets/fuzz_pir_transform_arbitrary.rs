// SPDX-License-Identifier: Apache-2.0

#![no_main]

use std::sync::Once;

use libfuzzer_sys::fuzz_target;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::IteratorRandom;
use xlsynth_mcmc_pir::transforms::get_all_pir_transforms;
use xlsynth_pir::ir;
use xlsynth_pir::ir_random::{
    DepletableBytes, OperationSet, RandomFnOptions, RandomOperation, StopPolicy, generate_fn,
};
use xlsynth_pir::ir_utils::compact_and_toposort_in_place;
use xlsynth_pir::ir_validate;
use xlsynth_prover::prover::types::{AssertionSemantics, EquivParallelism, EquivResult, ProverFn};
use xlsynth_prover::prover::{
    Prover, SolverChoice, SolverLimits, prover_for_choice_with_limits,
};

const NUM_STEPS: usize = 32;
const MAX_TRANSFORM_DRAWS: usize = NUM_STEPS * 32;
const FUZZ_SOLVER_TIME_LIMIT_PER_MS: u64 = 10_000;

static INIT_LOGGER: Once = Once::new();

fuzz_target!(|data: &[u8]| {
    INIT_LOGGER.call_once(|| {
        let _ = env_logger::Builder::from_env(env_logger::Env::default())
            .is_test(true)
            .try_init();
    });

    let options = RandomFnOptions {
        max_nodes: 20,
        max_bit_width: 8,
        enabled_operations: OperationSet::new(
            OperationSet::all_supported().iter().filter(|operation| {
                !matches!(operation, RandomOperation::Umulp | RandomOperation::Smulp)
            }),
        ),
        ..RandomFnOptions::default()
    };
    let mut entropy = DepletableBytes::new(data);
    let generated = generate_fn(&mut entropy, &options, StopPolicy::WhenEntropyDepleted)
        .expect("fixed random PIR options should always construct a valid function");
    let mut cur_pkg = generated.into_top_package("fuzz_pir_transform_arbitrary");
    ir_validate::validate_package(&cur_pkg)
        .expect("directly generated PIR package should start valid");
    let initial_ir_text = cur_pkg.to_string();

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

        if candidate.always_equivalent
            && !prove_candidate_equivalent(
                prover.as_ref(),
                &cur_pkg,
                &next_pkg,
                transform.kind(),
                &candidate.location,
            )
        {
            // A configured solver limit is expected to make some fuzz samples
            // inconclusive; those samples are not transform failures.
            return;
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
) -> bool {
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
    if cfg!(not(feature = "has-bitwuzla")) {
        panic!(
            "fuzz_pir_transform_arbitrary requires in-process Bitwuzla; enable \
             with-bitwuzla-system or with-bitwuzla-built"
        );
    }
    prover_for_choice_with_limits(
        SolverChoice::Bitwuzla,
        None,
        SolverLimits::with_time_limit_per_ms(FUZZ_SOLVER_TIME_LIMIT_PER_MS),
    )
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
) -> bool {
    match equiv_result {
        EquivResult::Proved => true,
        EquivResult::Inconclusive(msg) => {
            log::debug!(
                "formal equivalence inconclusive for transform {:?} at {:?}: {msg}",
                kind,
                location,
            );
            false
        }
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
