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
use super::{StitchCounterexample, StitchEquivResult, StitchObservation, replay_trace};

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

/// Compares the composed IR payload when qualified, plus every defined output
/// valid bit, starting from arbitrary mapped power-up state.
pub(super) fn prove_post_fill(
    behavior: &BehavioralPipeline,
    reference: &SequentialGateFn,
    mapped: &SequentialGateFn,
    latency: usize,
    controls: StitchObservation<'_>,
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
    let actual = unroll_mapped(
        &mut solver,
        mapped,
        &frames,
        &q,
        controls.reset.and(controls.output_valid),
    )?;
    let actual_data = &actual.final_outputs["out"];
    if expected.get_width() != actual_data.get_width() {
        return Err("behavioral and mapped payload widths differ".into());
    }
    // Any sampled reset defines every valid register, even before the launch
    // transaction could reach the output. The observation frame has no edge.
    let mut flushed = solver.numerical(1, 0);
    let mut early_mismatches = Vec::new();
    if let Some(reset) = controls.reset {
        for (edge, frame) in frames[..latency].iter().enumerate() {
            let active = if controls.reset_active_low {
                solver.not(&frame[reset])
            } else {
                frame[reset].clone()
            };
            flushed = solver.or(&flushed, &active);
            if let Some(mapped_valid) = actual.prefill_valid.get(edge) {
                // The observation just after this edge is still before fill.
                early_mismatches.push(solver.and(&flushed, mapped_valid));
            }
        }
    }
    let mut valid = solver.not(&flushed);
    if let Some(input_valid) = controls.input_valid {
        valid = solver.and(&valid, &frames[0][input_valid]);
    }
    let data_differs = solver.ne(&expected, actual_data);
    let mut mismatch = solver.and(&valid, &data_differs);
    if let Some(port) = controls.output_valid {
        let valid_differs = solver.ne(&valid, &actual.final_outputs[port]);
        mismatch = solver.or(&mismatch, &valid_differs);
    }
    for early in &early_mismatches {
        mismatch = solver.or(&mismatch, early);
    }
    match solver
        .assert(&mismatch)
        .and_then(|()| solver.check())
        .map_err(|e| e.to_string())?
    {
        Response::Unsat => return Ok(StitchEquivResult::Proved),
        Response::Unknown => return Err(INCONCLUSIVE.into()),
        Response::Sat => {
            // Decode and independently replay the first mismatching
            // observation.
        }
    }
    let mut cycle = latency;
    for (index, early) in early_mismatches.iter().enumerate() {
        if !model_bits(&mut solver, early)?.is_zero() {
            cycle = index + 1;
            break;
        }
    }
    let inputs_by_frame = frames[..=cycle]
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
    // An observed transaction depends only on its launch data. A sampled
    // reset likewise makes validity independent of the initial logical state.
    let reference_initial_state = reference
        .registers
        .iter()
        .map(|reg| {
            let width = reference.transition.inputs[reg.q.index()].get_bit_count();
            (reg.name.clone(), IrBits::zero(width))
        })
        .collect();
    let qualified = controls
        .valid_at(latency, &inputs_by_frame)
        .ok_or("Bitwuzla counterexample selected an undefined observation")?;
    let mut reference_outputs = BTreeMap::new();
    let mut mapped_outputs = replay_trace(mapped, &mapped_initial_state, &inputs_by_frame)?
        .pop()
        .ok_or("missing mapped observation frame")?;
    if let Some(port) = controls.output_valid {
        reference_outputs.insert(port.to_owned(), IrBits::from_lsb_is_0(&[qualified]));
        let mapped_valid = if cycle < latency {
            &actual.prefill_valid[cycle - 1]
        } else {
            &actual.final_outputs[port]
        };
        if mapped_outputs[port] != model_bits(&mut solver, mapped_valid)? {
            return Err(
                "Bitwuzla valid counterexample did not replay in the mapped simulator".into(),
            );
        }
    }
    if qualified {
        let data = behavior.compose_concrete(&inputs_by_frame[0])?;
        if data != model_bits(&mut solver, &expected)?
            || mapped_outputs["out"] != model_bits(&mut solver, actual_data)?
        {
            return Err(
                "Bitwuzla counterexample did not replay in the behavioral IR and mapped simulators"
                    .into(),
            );
        }
        reference_outputs.insert("out".to_owned(), data);
    } else {
        mapped_outputs.remove("out");
    }
    if reference_outputs == mapped_outputs {
        return Err("Bitwuzla counterexample did not reproduce an observable mismatch".into());
    }
    Ok(StitchEquivResult::Disproved(StitchCounterexample {
        reference_initial_state,
        mapped_initial_state,
        inputs_by_frame,
        reference_outputs,
        mapped_outputs,
    }))
}
