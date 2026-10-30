// SPDX-License-Identifier: Apache-2.0

//! Bitwuzla proofs comparing behavioral stage IR with mapped cell logic.

use std::collections::BTreeMap;
use std::ops::Range;
use std::time::Duration;

use xlsynth_pir::{IrBits, ir};
use xlsynth_prover::solver::bitwuzla::{Bitwuzla, BitwuzlaOptions, BitwuzlaTerm};
use xlsynth_prover::solver::{BitVec, Response, Solver};

use crate::aig::SequentialGateFn;

use super::behavioral::BehavioralPipeline;
use super::mapped_smt::{MappedStages, unroll_mapped};
use super::stage_script::LeafResult;
use super::{StitchCounterexample, StitchEquivResult, replay};

const INCONCLUSIVE: &str =
    "stitch equivalence inconclusive: Bitwuzla interrupted by a timeout or resource limit";

/// Builds a single-worker solver, or reports an already expired time budget.
fn new_solver(timeout: Option<Duration>, models: bool) -> Result<Option<Bitwuzla>, String> {
    let mut options = BitwuzlaOptions::new();
    if let Some(duration) = timeout {
        // Bitwuzla interprets a zero setting as unlimited.
        if duration.is_zero() {
            return Ok(None);
        }
        let millis = u64::try_from(duration.as_nanos().div_ceil(1_000_000))
            .map_err(|_| "Bitwuzla timeout exceeds the 64-bit millisecond limit")?;
        options.set_time_limit_per(millis);
    }
    options.set_nthreads(1);
    if models {
        options.enable_produce_models();
    } else {
        options.disable_produce_models();
    }
    Bitwuzla::new(&options).map(Some).map_err(|e| e.to_string())
}

/// Declares one independent frame, binding its words by external port name.
fn input_frame(
    solver: &mut Bitwuzla,
    design: &SequentialGateFn,
    frame: usize,
) -> Result<BTreeMap<String, BitVec<BitwuzlaTerm>>, String> {
    design
        .inputs
        .iter()
        .map(|id| {
            let port = &design.transition.inputs[id.index()];
            let symbol = format!("frame_{frame}_{}", id.index());
            let term = solver
                .declare(&symbol, port.get_bit_count())
                .map_err(|e| e.to_string())?;
            Ok((port.name.clone(), term))
        })
        .collect()
}

/// Allocates arbitrary Q words in the design's register order.
fn arbitrary_state(
    solver: &mut Bitwuzla,
    design: &SequentialGateFn,
) -> Result<Vec<BitVec<BitwuzlaTerm>>, String> {
    design
        .registers
        .iter()
        .enumerate()
        .map(|(i, reg)| {
            let width = design.transition.inputs[reg.q.index()].get_bit_count();
            solver
                .declare(&format!("state_{i}"), width)
                .map_err(|e| e.to_string())
        })
        .collect()
}

/// Asserts an observable difference, including the vacuous zero-width case.
fn check_mismatch(
    solver: &mut Bitwuzla,
    expected: &BitVec<BitwuzlaTerm>,
    mapped: &BitVec<BitwuzlaTerm>,
) -> Result<Response, String> {
    if expected.get_width() != mapped.get_width() {
        return Err(format!(
            "behavioral and mapped widths differ ({} and {} bits)",
            expected.get_width(),
            mapped.get_width()
        ));
    }
    let mismatch = solver.ne(expected, mapped);
    solver
        .assert(&mismatch)
        .and_then(|()| solver.check())
        .map_err(|e| e.to_string())
}

/// Checks a selected capture/output slice over arbitrary inputs and logical Q.
pub(super) fn prove_stage_leaf(
    behavior: &BehavioralPipeline,
    mapped: &MappedStages<'_>,
    reference: &SequentialGateFn,
    stage: usize,
    bits: Range<usize>,
    timeout: Option<Duration>,
) -> Result<LeafResult, String> {
    let Some(mut solver) = new_solver(timeout, false)? else {
        return Ok(LeafResult::Inconclusive);
    };
    let inputs = input_frame(&mut solver, reference, 0)?;
    let q = arbitrary_state(&mut solver, reference)?;
    let expected = behavior.stage_expected(&mut solver, stage, &inputs, &q)?;
    if expected.get_width() != mapped.width(stage) {
        return Err(format!(
            "stage {stage} behavioral width {} differs from mapped width {}",
            expected.get_width(),
            mapped.width(stage)
        ));
    }
    // MappedStages validates the selected range before slicing the IR result.
    let actual = mapped.evaluate(&mut solver, &inputs, &q, stage, bits.clone())?;
    let expected_slice = solver.slice(&expected, bits.start, bits.len());
    let response = check_mismatch(&mut solver, &expected_slice, &actual)?;
    Ok(match response {
        Response::Unsat => LeafResult::Proved,
        Response::Sat => LeafResult::Counterexample,
        Response::Unknown => LeafResult::Inconclusive,
    })
}

/// Decodes a packed solver word at its full width for concrete replay.
fn model_bits(solver: &mut Bitwuzla, word: &BitVec<BitwuzlaTerm>) -> Result<IrBits, String> {
    solver
        .get_value(word, &ir::Type::Bits(word.get_width()))
        .map_err(|e| e.to_string())?
        .to_bits()
        .map_err(|e| e.to_string())
}

/// Compares the composed frame-zero IR result with the post-fill mapped output.
pub(super) fn prove_post_fill(
    behavior: &BehavioralPipeline,
    reference: &SequentialGateFn,
    mapped: &SequentialGateFn,
    latency: usize,
    timeout: Option<Duration>,
) -> Result<StitchEquivResult, String> {
    let frame_count = latency.checked_add(1).ok_or("pipeline latency overflow")?;
    let Some(mut solver) = new_solver(timeout, true)? else {
        return Err(INCONCLUSIVE.into());
    };
    let frames = (0..frame_count)
        .map(|index| input_frame(&mut solver, reference, index))
        .collect::<Result<Vec<_>, _>>()?;
    let q = arbitrary_state(&mut solver, mapped)?;
    let expected = behavior.compose_smt(&mut solver, &frames[0])?;
    let actual = unroll_mapped(&mut solver, mapped, &frames, &q)?;
    // Free-running stitch specs have one packed data output and no controls.
    let actual_data = actual
        .get("out")
        .filter(|_| actual.len() == 1)
        .ok_or("behavioral stitch equivalence requires a single mapped data output 'out'")?;
    match check_mismatch(&mut solver, &expected, actual_data)? {
        Response::Unsat => return Ok(StitchEquivResult::Proved),
        Response::Unknown => return Err(INCONCLUSIVE.into()),
        Response::Sat => {
            // Replay the full witness before reporting a data mismatch.
        }
    }
    let inputs_by_frame = frames
        .iter()
        .map(|frame| {
            frame
                .iter()
                .map(|(name, term)| Ok((name.clone(), model_bits(&mut solver, term)?)))
                .collect::<Result<BTreeMap<_, _>, String>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mapped_initial_state = mapped
        .registers
        .iter()
        .zip(&q)
        .map(|(reg, term)| Ok((reg.name.clone(), model_bits(&mut solver, term)?)))
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    // All logical power-up data has been flushed. Zero is a concrete initial
    // state consistent with the behavioral counterexample.
    let reference_initial_state = reference
        .registers
        .iter()
        .map(|reg| {
            let width = reference.transition.inputs[reg.q.index()].get_bit_count();
            (reg.name.clone(), IrBits::zero(width))
        })
        .collect();
    let reference_outputs = BTreeMap::from([(
        "out".to_owned(),
        behavior.compose_concrete(&inputs_by_frame[0])?,
    )]);
    let mapped_outputs = replay(mapped, &mapped_initial_state, &inputs_by_frame)?;
    if reference_outputs == mapped_outputs
        || reference_outputs["out"] != model_bits(&mut solver, &expected)?
        || mapped_outputs["out"] != model_bits(&mut solver, actual_data)?
    {
        return Err(
            "Bitwuzla counterexample did not replay in the behavioral IR and mapped simulators"
                .into(),
        );
    }
    Ok(StitchEquivResult::Disproved(StitchCounterexample {
        reference_initial_state,
        mapped_initial_state,
        inputs_by_frame,
        reference_outputs,
        mapped_outputs,
    }))
}
