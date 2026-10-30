// SPDX-License-Identifier: Apache-2.0

//! Formal equivalence of a stitched pipeline and a mapped netlist.

use std::collections::BTreeMap;
use std::fmt;
use std::num::NonZeroUsize;
use std::time::Duration;

use xlsynth_pir::IrBits;

use crate::aig::import::import_gate_fn;
use crate::aig::sequential_pipeline::StitchedGatePipeline;
use crate::aig::topo::extract_cone;
use crate::aig::{AigBitVector, SequentialGateFn, SequentialPipelineOptions};
use crate::aig_sim::sequential::{SequentialState, simulate};
use crate::gate_builder::{GateBuilder, GateBuilderOptions, ReductionKind};
use crate::netlist::gv_eval::{
    LabeledSequentialNetlistAig, SequentialAigSignal, SequentialClockEdge,
};
use crate::prove_gate_fn_equiv_sat::{
    CadicalSat, GateFormalOptions, IncrementalSat, SatModel, SatSolveResult, ValidationError,
    build_sat_clauses,
};

mod correspondence;
mod stage_graph;
mod stage_script;

#[cfg(feature = "has-bitwuzla")]
mod behavioral;
#[cfg(feature = "has-bitwuzla")]
mod bitwuzla;
#[cfg(feature = "has-bitwuzla")]
mod mapped_smt;

#[cfg(feature = "has-bitwuzla")]
pub(crate) use behavioral::BehavioralPipeline;

pub use stage_script::StageProofScript;

/// The controls that define which outputs are observable in a stitch window.
#[derive(Clone, Copy, Default)]
pub(crate) struct StitchObservation<'a> {
    pub(crate) input_valid: Option<&'a str>,
    pub(crate) output_valid: Option<&'a str>,
    pub(crate) reset: Option<&'a str>,
    pub(crate) reset_active_low: bool,
}

impl<'a> From<&'a SequentialPipelineOptions> for StitchObservation<'a> {
    fn from(boundary: &'a SequentialPipelineOptions) -> Self {
        Self {
            input_valid: boundary.input_valid_signal.as_deref(),
            output_valid: boundary.output_valid_signal.as_deref(),
            reset: boundary.reset_signal.as_deref(),
            reset_active_low: boundary.reset_active_low,
        }
    }
}

impl StitchObservation<'_> {
    /// Returns the specified output validity, or None while power-up state
    /// can still determine it. The last frame supplies the observation inputs.
    pub(crate) fn valid_at(
        &self,
        latency: usize,
        frames: &[BTreeMap<String, IrBits>],
    ) -> Option<bool> {
        let (_, edges) = frames.split_last()?;
        let start = edges.len().saturating_sub(latency);
        if let Some(reset) = self.reset
            && edges[start..]
                .iter()
                .any(|frame| frame[reset].is_zero() == self.reset_active_low)
        {
            // Synchronous reset clears every valid register at this edge.
            return Some(false);
        }
        (edges.len() >= latency).then(|| {
            self.input_valid
                .is_none_or(|valid| !frames[start][valid].is_zero())
        })
    }

    /// Tests the named outputs that are observable at the end of this trace.
    pub(crate) fn mismatch(
        &self,
        latency: usize,
        frames: &[BTreeMap<String, IrBits>],
        reference: &BTreeMap<String, IrBits>,
        mapped: &BTreeMap<String, IrBits>,
    ) -> bool {
        self.valid_at(latency, frames).is_some_and(|data_valid| {
            reference.iter().any(|(name, value)| {
                (data_valid || Some(name.as_str()) == self.output_valid) && *value != mapped[name]
            })
        })
    }
}

/// Selects when to use the direct register correspondence proof.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StitchProofMode {
    /// Try stage obligations first, then check the external outputs if needed.
    #[default]
    Auto,
    /// Require a complete, proved correspondence between the two state vectors.
    Stages,
    /// Compare defined valid indications and qualified data directly.
    WholePipeline,
}

/// Selects the reference semantics and solver for every proof and fallback.
#[derive(Clone, Copy, Debug)]
pub enum StitchProofSolver {
    /// Compare behavioral XLS IR against Boolean mapped-cell formulas.
    Bitwuzla { timeout: Option<Duration> },
    /// Compare the gatified reference against the mapped cells.
    Cadical(GateFormalOptions),
}

impl Default for StitchProofSolver {
    fn default() -> Self {
        if cfg!(feature = "has-bitwuzla") {
            Self::Bitwuzla { timeout: None }
        } else {
            Self::Cadical(GateFormalOptions::default())
        }
    }
}

/// A prepared reference for the selected proof backend.
#[derive(Clone, Copy)]
pub(crate) enum ProofReference<'a> {
    Gates(&'a GateFormalOptions),
    #[cfg(feature = "has-bitwuzla")]
    Behavioral {
        stages: &'a BehavioralPipeline,
        timeout: Option<Duration>,
        /// Simulates candidate pairs. The proof uses the behavioral functions.
        lower_for_pairing: &'a dyn Fn() -> Result<StitchedGatePipeline, String>,
    },
}

/// Stage scripts override the default single Solve for the named DSLX stages.
pub struct StitchProofOptions {
    pub mode: StitchProofMode,
    pub jobs: NonZeroUsize,
    pub stage_scripts: BTreeMap<String, StageProofScript>,
    pub solver: StitchProofSolver,
}

impl Default for StitchProofOptions {
    fn default() -> Self {
        Self {
            mode: StitchProofMode::Auto,
            jobs: std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN),
            stage_scripts: BTreeMap::new(),
            solver: StitchProofSolver::default(),
        }
    }
}

/// Identifies which proof established the result and what it checked.
#[derive(Debug, PartialEq, Eq)]
pub enum StitchProofMethod {
    Stages { stages: Vec<StageProofSummary> },
    WholePipeline { fallback_reason: Option<String> },
}

/// Number of discharged leaves for one stage, in stitch specification order.
#[derive(Debug, PartialEq, Eq)]
pub struct StageProofSummary {
    pub name: String,
    pub leaves: usize,
}

/// The external equivalence result together with the proof that produced it.
#[derive(Debug, PartialEq, Eq)]
pub struct StitchProofReport {
    pub result: StitchEquivResult,
    pub method: StitchProofMethod,
}

/// Conclusion of checking every defined valid indication and qualified output.
#[derive(Debug, PartialEq, Eq)]
pub enum StitchEquivResult {
    Proved,
    Disproved(StitchCounterexample),
}

/// A concrete trace from arbitrary, independent initial states. The last input
/// record is the observation frame after all specified clock edges.
#[derive(Debug, PartialEq, Eq)]
pub struct StitchCounterexample {
    pub reference_initial_state: BTreeMap<String, IrBits>,
    pub mapped_initial_state: BTreeMap<String, IrBits>,
    pub inputs_by_frame: Vec<BTreeMap<String, IrBits>>,
    pub reference_outputs: BTreeMap<String, IrBits>,
    pub mapped_outputs: BTreeMap<String, IrBits>,
}

impl fmt::Display for StitchCounterexample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "Reference initial state {:?}",
            self.reference_initial_state
        )?;
        writeln!(f, "Mapped initial state {:?}", self.mapped_initial_state)?;
        for (index, inputs) in self.inputs_by_frame.iter().enumerate() {
            writeln!(f, "Frame {index} inputs {inputs:?}")?;
        }
        writeln!(f, "Reference outputs {:?}", self.reference_outputs)?;
        write!(f, "Mapped outputs {:?}", self.mapped_outputs)
    }
}

/// External port widths in the common named ABI.
struct Interface<'a> {
    inputs: BTreeMap<&'a str, usize>,
    outputs: BTreeMap<&'a str, usize>,
}

/// Uses the stitched RTL data output name for its native g8r counterpart.
fn output_name(name: &str) -> &str {
    if name == "output_value" { "out" } else { name }
}

/// Describes the external ports, rejecting ambiguous logical names.
fn interface(design: &SequentialGateFn) -> Result<Interface<'_>, String> {
    let mut inputs = BTreeMap::new();
    for id in &design.inputs {
        let port = &design.transition.inputs[id.index()];
        if inputs
            .insert(port.name.as_str(), port.get_bit_count())
            .is_some()
        {
            return Err(format!("duplicate external input '{}'", port.name));
        }
    }
    let mut outputs = BTreeMap::new();
    for id in &design.outputs {
        let port = &design.transition.outputs[id.index()];
        if outputs
            .insert(output_name(&port.name), port.get_bit_count())
            .is_some()
        {
            return Err(format!(
                "duplicate external output role '{}'",
                output_name(&port.name)
            ));
        }
    }
    if outputs.is_empty() {
        return Err("pipeline has no observable output".into());
    }
    Ok(Interface { inputs, outputs })
}

/// Validates the clock edge and bit labels retained by the mapped GV reader.
fn check_mapped_abi(
    reference: &SequentialGateFn,
    mapped: &LabeledSequentialNetlistAig,
) -> Result<(), String> {
    if let Some(expected) = &reference.clock {
        let actual = mapped.clock.as_ref().ok_or_else(|| {
            format!(
                "mapped module '{}' is missing the stitch clock input '{}'",
                mapped.module_name, expected.name
            )
        })?;
        if expected.name != actual.port_name {
            return Err(format!(
                "expected clock '{}', got '{}'",
                expected.name, actual.port_name
            ));
        }
    }
    if !mapped.sequential_gate_fn.registers.is_empty() {
        let clock = mapped
            .clock
            .as_ref()
            .ok_or("mapped registers have no clock metadata")?;
        match clock.active_edge {
            Some(SequentialClockEdge::Rising) => {
                // Each unrolled transition advances on this edge.
            }
            Some(SequentialClockEdge::Falling) => {
                return Err(format!(
                    "mapped registers use the falling edge of '{}'. stitch-equiv requires rising-edge registers",
                    clock.port_name
                ));
            }
            None => {
                return Err(format!(
                    "mapped registers have no active edge specified for clock '{}'",
                    clock.port_name
                ));
            }
        }
    }
    for port in &mapped.module_ports {
        for (index, bit) in port.bits_lsb_to_msb.iter().enumerate() {
            if bit.signal == SequentialAigSignal::Clock
                && mapped.clock.as_ref().map(|clock| &clock.port_name) != Some(&port.name)
            {
                return Err(format!(
                    "mapped port '{}' bit {index} aliases the pipeline clock",
                    port.name
                ));
            }
            if index != bit.bit_number as usize {
                return Err(format!(
                    "mapped port '{}' has Verilog index {} at bit offset {index}. stitch-equiv requires packed range [{}:0] with index 0 as the LSB",
                    port.name,
                    bit.bit_number,
                    port.bits_lsb_to_msb.len() - 1
                ));
            }
        }
    }
    Ok(())
}

/// Unrolls one transition graph, sampling its outputs before each edge.
fn unroll(
    design: &SequentialGateFn,
    frames: &[BTreeMap<&str, AigBitVector>],
    initial_state: &[AigBitVector],
    builder: &mut GateBuilder,
) -> Result<Vec<BTreeMap<String, AigBitVector>>, String> {
    let mut bindings = vec![AigBitVector::zeros(0); design.transition.inputs.len()];
    let mut outputs = Vec::with_capacity(frames.len());
    for (register, value) in design.registers.iter().zip(initial_state) {
        bindings[register.q.index()] = value.clone();
    }
    for frame in frames {
        for id in &design.inputs {
            let name = &design.transition.inputs[id.index()].name;
            bindings[id.index()] = frame[name.as_str()].clone();
        }
        let values = import_gate_fn(&design.transition, &bindings, builder)?;
        outputs.push(
            design
                .outputs
                .iter()
                .map(|id| {
                    let name = output_name(&design.transition.outputs[id.index()].name);
                    (name.to_string(), values[id.index()].clone())
                })
                .collect(),
        );
        for register in &design.registers {
            bindings[register.q.index()] = values[register.d.index()].clone();
        }
    }
    Ok(outputs)
}

/// Allocates independent symbolic power-up bits even for equally named flops.
fn initial_state(
    design: &SequentialGateFn,
    side: &str,
    builder: &mut GateBuilder,
) -> Vec<AigBitVector> {
    design
        .registers
        .iter()
        .enumerate()
        .map(|(index, register)| {
            builder.add_input(
                format!("{side}_{index}"),
                design.transition.inputs[register.q.index()].get_bit_count(),
            )
        })
        .collect()
}

/// Replays the named input frames, sampling all external outputs before each
/// edge.
fn replay_trace(
    design: &SequentialGateFn,
    state: &BTreeMap<String, IrBits>,
    frames: &[BTreeMap<String, IrBits>],
) -> Result<Vec<BTreeMap<String, IrBits>>, String> {
    let initial = SequentialState::from_register_values(
        design,
        design
            .registers
            .iter()
            .map(|reg| state[&reg.name].clone())
            .collect(),
    )?;
    let inputs = frames
        .iter()
        .map(|frame| {
            design
                .inputs
                .iter()
                .map(|id| frame[&design.transition.inputs[id.index()].name].clone())
                .collect()
        })
        .collect::<Vec<_>>();
    let trace = simulate(design, &inputs, initial)?;
    Ok(trace
        .external_outputs()
        .iter()
        .map(|values| {
            design
                .outputs
                .iter()
                .zip(values)
                .map(|(id, value)| {
                    (
                        output_name(&design.transition.outputs[id.index()].name).to_owned(),
                        value.clone(),
                    )
                })
                .collect()
        })
        .collect())
}

/// Checks the named external interface and clock before either proof strategy.
fn validate_interfaces<'a>(
    reference: &'a SequentialGateFn,
    mapped: &LabeledSequentialNetlistAig,
) -> Result<Interface<'a>, String> {
    reference.validate_legalized()?;
    let implementation = &mapped.sequential_gate_fn;
    implementation.validate_legalized()?;
    check_mapped_abi(reference, mapped)?;
    matching_interface(reference, implementation)
}

/// Returns the common named data interface, with packed widths checked.
fn matching_interface<'a>(
    reference: &'a SequentialGateFn,
    implementation: &SequentialGateFn,
) -> Result<Interface<'a>, String> {
    let expected = interface(reference)?;
    let actual = interface(implementation)?;
    if expected.inputs != actual.inputs {
        return Err(format!(
            "input ports differ: reference {:?}, mapped {:?}",
            expected.inputs, actual.inputs
        ));
    }
    if expected.outputs != actual.outputs {
        return Err(format!(
            "output ports differ: reference {:?}, mapped {:?}",
            expected.outputs, actual.outputs
        ));
    }
    Ok(expected)
}

/// Whether the proposed register relation establishes external equivalence.
enum StageAttempt {
    Proved(Vec<StageProofSummary>),
    Unresolved(String),
}

/// Proves the Q relation inductive by checking every D bit and external output.
/// The bijection gives every mapped power-up state a corresponding reference
/// state. At a defined observation the reference valid and qualified data
/// are independent of that initial state.
fn prove_stages(
    pipeline: &StitchedGatePipeline,
    implementation: &SequentialGateFn,
    latency: usize,
    observation: StitchObservation<'_>,
    stage_names: &[String],
    options: &StitchProofOptions,
    model: ProofReference<'_>,
) -> Result<StageAttempt, String> {
    let reference = &pipeline.design;
    // Auto can go straight to the post-fill check on unequal state widths.
    // Strict and scripted requests still identify an unmatched register bit.
    if options.mode == StitchProofMode::Auto && options.stage_scripts.is_empty() {
        let reference_bits = reference.register_bit_count();
        let mapped_bits = implementation.register_bit_count();
        if reference_bits != mapped_bits {
            return Ok(StageAttempt::Unresolved(format!(
                "no direct register correspondence: register bit counts differ (reference {reference_bits}, mapped {mapped_bits})"
            )));
        }
    }
    let proposal = match model {
        _ if reference.registers.is_empty() && implementation.registers.is_empty() => {
            // A combinational design has an empty state relation.
            Ok(Vec::new())
        }
        ProofReference::Gates(_) => {
            correspondence::direct_register_pairs(reference, implementation, latency, observation)
        }
        #[cfg(feature = "has-bitwuzla")]
        ProofReference::Behavioral {
            lower_for_pairing, ..
        } => lower_for_pairing().and_then(|candidate| {
            correspondence::direct_register_pairs(
                &candidate.design,
                implementation,
                latency,
                observation,
            )
        }),
    };
    let pairs = match proposal {
        Ok(pairs) => pairs,
        Err(reason) if options.stage_scripts.is_empty() => {
            return Ok(StageAttempt::Unresolved(format!(
                "no direct register correspondence: {reason}"
            )));
        }
        Err(reason) => {
            return Err(format!(
                "cannot run per-stage scripts without a direct register correspondence: {reason}"
            ));
        }
    };
    let (outcome, solver_name) = match model {
        ProofReference::Gates(formal) => {
            let miters = stage_graph::build_stage_miters(
                reference,
                implementation,
                &pairs,
                &pipeline.register_stages,
                stage_names.len(),
            )?;
            (
                stage_script::run_stage_proofs(
                    &miters,
                    stage_names,
                    &options.stage_scripts,
                    options.jobs.get(),
                    *formal,
                )?,
                "CaDiCaL",
            )
        }
        #[cfg(feature = "has-bitwuzla")]
        ProofReference::Behavioral {
            stages: behavior,
            timeout,
            ..
        } => {
            let mapped = mapped_smt::MappedStages::new(
                reference,
                implementation,
                &pairs,
                &pipeline.register_stages,
                stage_names.len(),
            )?;
            let widths = (0..stage_names.len())
                .map(|i| behavior.stage_width(i))
                .collect::<Result<Vec<_>, _>>()?;
            for (i, &width) in widths.iter().enumerate() {
                if width != mapped.width(i) {
                    return Err(format!(
                        "stage '{}' has {width} behavioral bits but {} mapped bits",
                        stage_names[i],
                        mapped.width(i)
                    ));
                }
            }
            (
                stage_script::run_stage_leaves(
                    stage_names,
                    &widths,
                    &options.stage_scripts,
                    options.jobs.get(),
                    |stage, bits| {
                        bitwuzla::prove_stage_leaf(
                            behavior, &mapped, reference, stage, bits, timeout,
                        )
                    },
                )?,
                "Bitwuzla",
            )
        }
    };
    Ok(match outcome {
        stage_script::StageSolveOutcome::Proved { leaf_counts } => StageAttempt::Proved(
            stage_names
                .iter()
                .zip(leaf_counts)
                .map(|(name, leaves)| StageProofSummary {
                    name: name.clone(),
                    leaves,
                })
                .collect(),
        ),
        stage_script::StageSolveOutcome::Counterexample { location } => StageAttempt::Unresolved(
            format!("the proposed correspondence fails the one-step obligation for {location}"),
        ),
        stage_script::StageSolveOutcome::Inconclusive { location } => {
            StageAttempt::Unresolved(format!(
                "stage proof inconclusive for {location}: {solver_name} interrupted by a timeout or resource limit"
            ))
        }
    })
}

/// Selects the proof strategy and, when needed, checks the defined external
/// observations independently of the proposed register relation.
pub(crate) fn prove_with_stage_options(
    pipeline: &StitchedGatePipeline,
    mapped: &LabeledSequentialNetlistAig,
    latency: usize,
    observation: StitchObservation<'_>,
    stage_names: &[String],
    options: &StitchProofOptions,
    model: ProofReference<'_>,
) -> Result<StitchProofReport, String> {
    let reference = &pipeline.design;
    let implementation = &mapped.sequential_gate_fn;
    let expected = validate_interfaces(reference, mapped)?;
    for name in options.stage_scripts.keys() {
        if !stage_names.contains(name) {
            return Err(format!(
                "unknown scripted stage '{name}'. Available stages: {}",
                stage_names.join(", ")
            ));
        }
    }
    let fallback_reason = match options.mode {
        StitchProofMode::WholePipeline => {
            if !options.stage_scripts.is_empty() {
                return Err("whole-pipeline proof mode cannot run per-stage scripts".into());
            }
            None
        }
        StitchProofMode::Auto | StitchProofMode::Stages => {
            match prove_stages(
                pipeline,
                implementation,
                latency,
                observation,
                stage_names,
                options,
                model,
            )? {
                StageAttempt::Proved(stages) => {
                    return Ok(StitchProofReport {
                        result: StitchEquivResult::Proved,
                        method: StitchProofMethod::Stages { stages },
                    });
                }
                StageAttempt::Unresolved(reason) if options.mode == StitchProofMode::Stages => {
                    return Err(format!(
                        "{reason}. Use whole-pipeline mode to check the external outputs after fill"
                    ));
                }
                StageAttempt::Unresolved(reason) => Some(reason),
            }
        }
    };
    let result = match model {
        ProofReference::Gates(formal) => prove_post_fill(
            reference,
            implementation,
            &expected,
            latency,
            observation,
            *formal,
        ),
        #[cfg(feature = "has-bitwuzla")]
        ProofReference::Behavioral {
            stages: behavior,
            timeout,
            ..
        } => bitwuzla::prove_post_fill(
            behavior,
            reference,
            implementation,
            latency,
            observation,
            timeout,
        ),
    }
    .map_err(|error| match &fallback_reason {
        Some(reason) => format!("{reason}; whole-pipeline proof: {error}"),
        None => error,
    })?;
    Ok(StitchProofReport {
        result,
        method: StitchProofMethod::WholePipeline { fallback_reason },
    })
}

/// Checks post-fill outputs and pre-fill validity after a sampled reset.
/// An arbitrary state at the start of the window covers every reset edge.
fn prove_post_fill(
    reference: &SequentialGateFn,
    implementation: &SequentialGateFn,
    expected: &Interface,
    latency: usize,
    observation: StitchObservation<'_>,
    options: GateFormalOptions,
) -> Result<StitchEquivResult, String> {
    let frame_count = latency.checked_add(1).ok_or("pipeline latency overflow")?;
    let mut builder = GateBuilder::new("stitch_equiv".to_string(), GateBuilderOptions::opt());
    let frames = (0..frame_count)
        .map(|frame_index| {
            expected
                .inputs
                .iter()
                .enumerate()
                .map(|(port_index, (&name, &width))| {
                    (
                        name,
                        builder.add_input(format!("frame_{frame_index}_{port_index}"), width),
                    )
                })
                .collect()
        })
        .collect::<Vec<_>>();
    let ref_state = initial_state(reference, "reference", &mut builder);
    let mapped_state = initial_state(implementation, "mapped", &mut builder);
    let ref_outputs = unroll(reference, &frames, &ref_state, &mut builder)?;
    let mapped_outputs = unroll(implementation, &frames, &mapped_state, &mut builder)?;
    let mut mismatch = builder.get_false();
    let mut flushed = builder.get_false();
    let accepted = observation
        .input_valid
        .map_or(builder.get_true(), |valid| *frames[0][valid].get_lsb(0));
    for (i, (expected, actual)) in ref_outputs.iter().zip(&mapped_outputs).enumerate() {
        if i == latency {
            let data_valid = builder.add_and_binary(accepted, flushed.negate());
            for (name, value) in expected {
                let mut differs = builder.add_ne_vec(value, &actual[name], ReductionKind::Tree);
                if Some(name.as_str()) != observation.output_valid {
                    differs = builder.add_and_binary(data_valid, differs);
                }
                mismatch = builder.add_or_binary(mismatch, differs);
            }
        } else {
            if let (Some(valid), Some(_)) = (observation.output_valid, observation.reset) {
                // Before fill, a sampled reset makes the valid output defined.
                let differs =
                    builder.add_ne_vec(&expected[valid], &actual[valid], ReductionKind::Tree);
                let observable = builder.add_and_binary(flushed, differs);
                mismatch = builder.add_or_binary(mismatch, observable);
            }
            if let Some(reset) = observation.reset {
                let bit = *frames[i][reset].get_lsb(0);
                let active = if observation.reset_active_low {
                    bit.negate()
                } else {
                    bit
                };
                flushed = builder.add_or_binary(flushed, active);
            }
        }
    }
    builder.add_output("mismatch".to_string(), mismatch.into());
    let graph = builder.build();
    let mut solver = CadicalSat::new_with_options(options).map_err(|e| e.to_string())?;
    let (cone, cone_inputs) = extract_cone(&[mismatch.node], &graph.gates);
    let literals = build_sat_clauses(&mut solver, &cone, &cone_inputs, &graph.gates);
    let lit = literals[&mismatch.node];
    let differs = if mismatch.negated { !lit } else { lit };
    if solver
        .sat_solve_assuming(&[differs])
        .map_err(|error| match error {
            ValidationError::CadicalSolveInterrupted => {
                "stitch equivalence inconclusive: CaDiCaL interrupted by a timeout or resource limit".to_owned()
            }
            other => other.to_string(),
        })?
        == SatSolveResult::Unsat
    {
        return Ok(StitchEquivResult::Proved);
    }

    let model = solver.sat_model().map_err(|e| e.to_string())?;
    // All symbols here are positive primary inputs. Inputs outside the
    // mismatch cone are unconstrained and may be set to zero for replay.
    let decode = |value: &AigBitVector| {
        IrBits::from_lsb_is_0(
            &value
                .iter_lsb_to_msb()
                .map(|bit| {
                    literals
                        .get(&bit.node)
                        .is_some_and(|lit| model.lit_value(*lit))
                })
                .collect::<Vec<_>>(),
        )
    };
    let decode_state = |design: &SequentialGateFn, values: &[AigBitVector]| {
        design
            .registers
            .iter()
            .zip(values)
            .map(|(reg, value)| (reg.name.clone(), decode(value)))
            .collect()
    };
    let reference_initial_state = decode_state(reference, &ref_state);
    let mapped_initial_state = decode_state(implementation, &mapped_state);
    let mut inputs_by_frame = frames
        .iter()
        .map(|frame| {
            frame
                .iter()
                .map(|(&name, value)| (name.to_owned(), decode(value)))
                .collect()
        })
        .collect::<Vec<_>>();
    let reference_trace = replay_trace(reference, &reference_initial_state, &inputs_by_frame)?;
    let mapped_trace = replay_trace(implementation, &mapped_initial_state, &inputs_by_frame)?;
    let (index, (mut reference_outputs, mut mapped_outputs)) = reference_trace
        .into_iter()
        .zip(mapped_trace)
        .enumerate()
        .find(|(i, (reference, mapped))| {
            observation.mismatch(latency, &inputs_by_frame[..=*i], reference, mapped)
        })
        .ok_or("SAT counterexample did not replay in the sequential simulator")?;
    inputs_by_frame.truncate(index + 1);
    if observation.valid_at(latency, &inputs_by_frame) == Some(false) {
        reference_outputs.retain(|name, _| Some(name.as_str()) == observation.output_valid);
        mapped_outputs.retain(|name, _| Some(name.as_str()) == observation.output_valid);
    }
    Ok(StitchEquivResult::Disproved(StitchCounterexample {
        reference_initial_state,
        mapped_initial_state,
        inputs_by_frame,
        reference_outputs,
        mapped_outputs,
    }))
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::aig::{ClockPort, add_output_registers};
    use crate::liberty_model::Library;
    use crate::netlist::gv_eval::{
        GvEvalOptions, LabeledSequentialClock, load_labeled_sequential_netlist_aig_with_liberty,
        project_sequential_module,
    };
    use crate::netlist::io::read_gv_from_str;
    #[cfg(feature = "has-bitwuzla")]
    use xlsynth_pir::ir_parser::Parser;

    fn prove_stitched_pipeline_equiv(
        reference: &SequentialGateFn,
        mapped: &LabeledSequentialNetlistAig,
        latency: usize,
        options: GateFormalOptions,
    ) -> Result<StitchEquivResult, String> {
        let interface = validate_interfaces(reference, mapped)?;
        prove_post_fill(
            reference,
            &mapped.sequential_gate_fn,
            &interface,
            latency,
            StitchObservation::default(),
            options,
        )
    }

    /// A two-port design that exposes only x, optionally pipelined.
    fn delayed_x(depth: usize) -> Result<SequentialGateFn, String> {
        let mut builder = GateBuilder::new("delayed_x".to_string(), GateBuilderOptions::opt());
        let x = builder.add_input("x".to_string(), 4);
        builder.add_input("y".to_string(), 4);
        builder.add_output("output_value".to_string(), x);
        let mut design = SequentialGateFn::from_gate_fn(builder.build());
        for _ in 0..depth {
            design = add_output_registers(&design, ClockPort { name: "clk".into() })?;
        }
        Ok(design)
    }

    /// Carries the mapped clock information used by the GV frontend.
    fn mapped(design: SequentialGateFn) -> LabeledSequentialNetlistAig {
        let active_edge = (!design.registers.is_empty()).then_some(SequentialClockEdge::Rising);
        LabeledSequentialNetlistAig {
            module_name: design.name.clone(),
            clock: Some(LabeledSequentialClock {
                port_name: "clk".into(),
                active_edge,
            }),
            sequential_gate_fn: design,
            module_ports: Vec::new(),
            module_boundaries: Vec::new(),
            instances: Vec::new(),
        }
    }

    #[test]
    fn a_reset_flushes_exactly_the_transactions_crossing_its_edge() {
        let observation = StitchObservation {
            input_valid: Some("vin"),
            reset: Some("rst_n"),
            reset_active_low: true,
            output_valid: None,
        };
        let frames = [
            (false, false), // reset edge, flushing unknown power-up validity
            (true, true),   // launch transaction
            (false, true),  // bubble
            (true, false),  // observe transaction, then reset on this edge
            (true, true),   // launch post-reset transaction
            (false, true),
            (false, true), // observe post-reset transaction
        ]
        .map(|(vin, rst_n)| {
            BTreeMap::from([
                ("vin".to_owned(), IrBits::bool(vin)),
                ("rst_n".to_owned(), IrBits::bool(rst_n)),
            ])
        });
        let expected = [
            None,
            Some(false),
            Some(false),
            Some(true),
            Some(false),
            Some(false),
            Some(true),
        ];
        for (i, valid) in expected.iter().enumerate() {
            assert_eq!(observation.valid_at(2, &frames[..=i]), *valid);
        }
    }

    #[test]
    fn aligns_reordered_ports_and_rtl_output_name() {
        let reference = delayed_x(2).unwrap();
        let mut implementation = reference.clone();
        implementation.inputs.reverse();
        implementation.transition.outputs[implementation.outputs[0].index()].name = "out".into();
        for (index, register) in implementation.registers.iter_mut().enumerate() {
            register.name = format!("mapped_flop_{index}");
        }
        assert_eq!(
            prove_stitched_pipeline_equiv(
                &reference,
                &mapped(implementation),
                2,
                GateFormalOptions::default()
            )
            .unwrap(),
            StitchEquivResult::Proved
        );
    }

    #[test]
    fn detects_latency_loss_with_fresh_observation_inputs() {
        let reference = delayed_x(1).unwrap();
        let implementation = mapped(delayed_x(0).unwrap());
        let StitchEquivResult::Disproved(trace) = prove_stitched_pipeline_equiv(
            &reference,
            &implementation,
            1,
            GateFormalOptions::default(),
        )
        .unwrap() else {
            panic!("a missing pipeline register must change observable behavior");
        };
        assert_eq!(trace.inputs_by_frame.len(), 2);
        assert_eq!(
            trace.inputs_by_frame[0]["x"],
            trace.reference_outputs["out"]
        );
        assert_eq!(trace.inputs_by_frame[1]["x"], trace.mapped_outputs["out"]);
        assert_ne!(trace.inputs_by_frame[0]["x"], trace.inputs_by_frame[1]["x"]);
    }

    #[cfg(feature = "has-bitwuzla")]
    #[test]
    fn behavioral_proofs_detect_a_shared_error_in_the_gate_lowering() {
        // Both the candidate graph and the mapped cells capture x. The XLS
        // stage specifies !x. Pair discovery can succeed despite the bug.
        let stages = ["invert".to_owned()];
        let package = Parser::new(
            r#"package wrong_lowering
top fn invert(x: bits[4] id=1, y: bits[4] id=2) -> bits[4] {
  ret complement: bits[4] = not(x, id=3)
}"#,
        )
        .parse_package()
        .unwrap();
        let behavior = BehavioralPipeline::new(
            package,
            &[(stages[0].clone(), stages[0].clone())],
            &SequentialPipelineOptions {
                name: "invert".into(),
                clock: ClockPort { name: "clk".into() },
                flop_inputs: false,
                flop_outputs: true,
                input_valid_signal: None,
                output_valid_signal: None,
                reset_signal: None,
                reset_active_low: false,
            },
        )
        .unwrap();
        let lower = || {
            delayed_x(1).map(|design| StitchedGatePipeline {
                design,
                register_stages: vec![0],
            })
        };
        let pipeline = lower().unwrap();
        let implementation = mapped(delayed_x(1).unwrap());
        let formal = GateFormalOptions::default();
        let mut options = StitchProofOptions {
            mode: StitchProofMode::Stages,
            solver: StitchProofSolver::Cadical(formal),
            ..StitchProofOptions::default()
        };
        let gate_report = prove_with_stage_options(
            &pipeline,
            &implementation,
            1,
            StitchObservation::default(),
            &stages,
            &options,
            ProofReference::Gates(&formal),
        )
        .unwrap();
        assert_eq!(gate_report.result, StitchEquivResult::Proved);

        options.solver = StitchProofSolver::Bitwuzla { timeout: None };
        let model = ProofReference::Behavioral {
            stages: &behavior,
            timeout: None,
            lower_for_pairing: &lower,
        };
        assert_eq!(
            prove_with_stage_options(
                &pipeline,
                &implementation,
                1,
                StitchObservation::default(),
                &stages,
                &options,
                model
            )
            .unwrap_err(),
            "the proposed correspondence fails the one-step obligation for 'invert'. Use whole-pipeline mode to check the external outputs after fill"
        );
        for mode in [StitchProofMode::Auto, StitchProofMode::WholePipeline] {
            options.mode = mode;
            let model = if mode == StitchProofMode::WholePipeline {
                ProofReference::Behavioral {
                    stages: &behavior,
                    timeout: None,
                    lower_for_pairing: &|| panic!("whole-pipeline proof attempted gatification"),
                }
            } else {
                model
            };
            let report = prove_with_stage_options(
                &pipeline,
                &implementation,
                1,
                StitchObservation::default(),
                &stages,
                &options,
                model,
            )
            .unwrap();
            let StitchEquivResult::Disproved(trace) = report.result else {
                panic!("behavioral comparison must disprove the faulty lowering")
            };
            let x = &trace.inputs_by_frame[0]["x"];
            assert_eq!(&trace.mapped_outputs["out"], x);
            assert_eq!(trace.reference_outputs["out"], x.not());
        }
    }

    #[test]
    fn interrupted_solve_reports_inconclusive() {
        let reference = delayed_x(1).unwrap();
        let implementation = mapped(delayed_x(0).unwrap());
        assert_eq!(
            prove_stitched_pipeline_equiv(
                &reference,
                &implementation,
                1,
                GateFormalOptions::default().with_cadical_timeout(Duration::ZERO),
            )
            .unwrap_err(),
            "stitch equivalence inconclusive: CaDiCaL interrupted by a timeout or resource limit"
        );
    }

    #[test]
    fn interrupted_stage_leaf_reports_its_selector_and_auto_fallback() {
        let pipeline = StitchedGatePipeline {
            design: delayed_x(1).unwrap(),
            register_stages: vec![0],
        };
        let mut implementation = pipeline.design.clone();
        // Both designs capture x, but the mapped output bypasses the register
        // and exposes y. Only the high leaf (the four output bits) can differ.
        implementation.transition.outputs[implementation.outputs[0].index()].bit_vector =
            implementation.transition.inputs[implementation.inputs[1].index()]
                .bit_vector
                .clone();
        let implementation = mapped(implementation);
        let stage_names = ["capture".to_owned()];
        let split = StageProofScript::from_json(
            r#"[
                {"selector":["root"], "command":{"Apply":"SplitOutputs"}},
                {"selector":["root","low"], "command":"Solve"},
                {"selector":["root","high"], "command":"Solve"}
            ]"#,
        )
        .unwrap();
        let formal = GateFormalOptions::default().with_cadical_timeout(Duration::ZERO);
        let mut options = StitchProofOptions {
            mode: StitchProofMode::Stages,
            jobs: NonZeroUsize::MIN,
            stage_scripts: BTreeMap::from([(stage_names[0].clone(), split)]),
            solver: StitchProofSolver::Cadical(formal),
        };
        let stage_error = "stage proof inconclusive for 'capture' at root/high: CaDiCaL interrupted by a timeout or resource limit";
        assert_eq!(
            prove_with_stage_options(
                &pipeline,
                &implementation,
                1,
                StitchObservation::default(),
                &stage_names,
                &options,
                ProofReference::Gates(&formal)
            )
            .unwrap_err(),
            format!(
                "{stage_error}. Use whole-pipeline mode to check the external outputs after fill"
            )
        );
        options.mode = StitchProofMode::Auto;
        assert_eq!(
            prove_with_stage_options(
                &pipeline,
                &implementation,
                1,
                StitchObservation::default(),
                &stage_names,
                &options,
                ProofReference::Gates(&formal)
            )
            .unwrap_err(),
            format!(
                "{stage_error}; whole-pipeline proof: stitch equivalence inconclusive: CaDiCaL interrupted by a timeout or resource limit"
            )
        );
    }

    #[test]
    fn power_up_states_are_independent_and_not_assumed_zero() {
        let reference = delayed_x(1).unwrap();
        let implementation = mapped(reference.clone());
        let StitchEquivResult::Disproved(trace) = prove_stitched_pipeline_equiv(
            &reference,
            &implementation,
            0,
            GateFormalOptions::default(),
        )
        .unwrap() else {
            panic!("independent power-up values can differ before fill");
        };
        let name = &reference.registers[0].name;
        assert_ne!(
            trace.reference_initial_state[name],
            trace.mapped_initial_state[name]
        );
        assert_eq!(
            trace.reference_outputs["out"],
            trace.reference_initial_state[name]
        );
        assert_eq!(
            trace.mapped_outputs["out"],
            trace.mapped_initial_state[name]
        );
    }

    #[test]
    fn rejects_scalar_and_vector_ports_aliased_to_the_clock() {
        let reference = delayed_x(0).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("alias.gv");
        for (extra_port, alias_bit) in [("input extra", "extra"), ("input [1:0] extra", "extra[0]")]
        {
            std::fs::write(
                &path,
                format!(
                    r#"
module top(input clk, input [3:0] x, input [3:0] y, {extra_port}, output [3:0] out);
  tran(clk, {alias_bit});
  assign out = x;
endmodule
"#
                ),
            )
            .unwrap();
            let mapped = load_labeled_sequential_netlist_aig_with_liberty(
                &path,
                &Library::default(),
                &GvEvalOptions {
                    module_name: Some("top".into()),
                    clock_port_name: Some("clk".into()),
                },
            )
            .unwrap();
            assert_eq!(
                prove_stitched_pipeline_equiv(&reference, &mapped, 0, GateFormalOptions::default())
                    .unwrap_err(),
                "mapped port 'extra' bit 0 aliases the pipeline clock"
            );
        }
    }

    #[test]
    fn identifies_ascending_and_shifted_packed_ranges() {
        let reference = delayed_x(0).unwrap();
        for (range, lsb) in [("[0:3]", 3), ("[7:4]", 4)] {
            let source = format!(
                "module top(input [3:0] x, input [3:0] y, output {range} out); assign out = x; endmodule"
            );
            let parsed = read_gv_from_str(&source).unwrap();
            let mapped =
                project_sequential_module(&parsed, &parsed.modules[0], &Library::default(), None)
                    .unwrap();
            assert_eq!(
                prove_stitched_pipeline_equiv(&reference, &mapped, 0, GateFormalOptions::default())
                    .unwrap_err(),
                format!(
                    "mapped port 'out' has Verilog index {lsb} at bit offset 0. stitch-equiv requires packed range [3:0] with index 0 as the LSB"
                )
            );
        }
    }

    #[test]
    fn requires_a_compatible_clock_and_complete_port_abi() {
        let reference = delayed_x(1).unwrap();
        let mut implementation = mapped(reference.clone());
        implementation.clock.as_mut().unwrap().active_edge = Some(SequentialClockEdge::Falling);
        assert_eq!(
            prove_stitched_pipeline_equiv(
                &reference,
                &implementation,
                1,
                GateFormalOptions::default()
            )
            .unwrap_err(),
            "mapped registers use the falling edge of 'clk'. stitch-equiv requires rising-edge registers"
        );
        let mut implementation = mapped(reference.clone());
        implementation.sequential_gate_fn.transition.inputs[0].name = "z".into();
        assert_eq!(
            prove_stitched_pipeline_equiv(
                &reference,
                &implementation,
                1,
                GateFormalOptions::default()
            )
            .unwrap_err(),
            r#"input ports differ: reference {"x": 4, "y": 4}, mapped {"y": 4, "z": 4}"#
        );
    }
}
