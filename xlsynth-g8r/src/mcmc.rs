// SPDX-License-Identifier: Apache-2.0

use crate::aig::gate::GateFn;
use crate::aig::{dce, get_summary_stats};
use crate::aig_sim::gate_simd::{self, Vec256};
use crate::aig_serdes::ir2gate::{self, GatifyOptions};
use crate::test_utils::{
    Opt as SampleOpt, load_bf16_add_sample, load_bf16_mul_sample, make_ripple_carry_adder,
};
use crate::transforms::get_all_transforms;
use crate::transforms::transform_trait::{Transform, TransformDirection, TransformKind};
use clap::ValueEnum;
use core::simd::u64x4;
use rand::distributions::WeightedIndex;
use rand::prelude::{Rng, SliceRandom};
use rand_pcg::Pcg64Mcg;
use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use xlsynth_mcmc as engine;
use xlsynth_pir::ir_parser;

/// Calculates the cost of a `GateFn` (nodes, depth).
pub fn cost(g: &GateFn) -> Cost {
    let stats = get_summary_stats::get_summary_stats(g);
    Cost { nodes: stats.live_nodes, depth: stats.deepest_path }
}

/// Nodes/depth cost for objective selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cost {
    /// Number of live nodes in the netlist.
    pub nodes: usize,
    /// Deepest path length in the netlist.
    pub depth: usize,
}

/// Objective used to evaluate cost improvements.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Objective {
    /// Minimize node count.
    Nodes,
    /// Minimize circuit depth.
    Depth,
    /// Minimize product of nodes and depth.
    Product,
}

impl Objective {
    fn metric(self, c: &Cost) -> u64 {
        match self {
            Objective::Nodes => c.nodes as u64,
            Objective::Depth => c.depth as u64,
            Objective::Product => (c.nodes as u64) * (c.depth as u64),
        }
    }
}

/// Checks equivalence of two `GateFn`s using Z3 or IR-based checker.
pub fn oracle_equiv_sat(lhs: &GateFn, rhs: &GateFn) -> bool {
    #[cfg(any(feature = "with-z3-system", feature = "with-z3-built"))]
    {
        use crate::prove_gate_fn_equiv_common::EquivResult;
        use crate::prove_gate_fn_equiv_z3::{self, prove_gate_fn_equiv as prove_gate_fn_equiv_z3};
        let mut ctx = prove_gate_fn_equiv_z3::Ctx::new();
        return matches!(prove_gate_fn_equiv_z3(lhs, rhs, &mut ctx), EquivResult::Proved);
    }

    #[cfg(not(any(feature = "with-z3-system", feature = "with-z3-built")))]
    {
        return crate::check_equivalence::prove_same_gate_fn_via_ir(lhs, rhs).is_ok();
    }
}

/// G8R proposer over transforms with weighted selection and bidirectional moves.
pub struct G8rProposer {
    all_transforms: Vec<Box<dyn Transform>>,
    weights: WeightedIndex<f64>,
}

impl G8rProposer {
    /// Build a proposer from available transforms and objective-specific weights.
    pub fn new(transforms: Vec<Box<dyn Transform>>, weights: Vec<f64>) -> Self {
        let weights_dist = WeightedIndex::new(weights).expect("non-empty weights");
        Self { all_transforms: transforms, weights: weights_dist }
    }
}

impl engine::Proposer<GateFn, TransformKind, Pcg64Mcg> for G8rProposer {
    fn propose(&mut self, current: &GateFn, rng: &mut Pcg64Mcg) -> engine::ProposeResult<GateFn, TransformKind> {
        if self.all_transforms.is_empty() {
            return engine::ProposeResult::NoCandidates;
        }
        let idx = self.weights.sample(rng);
        let transform = &mut self.all_transforms[idx];
        let kind = transform.kind();
        let direction = if rng.r#gen::<bool>() { TransformDirection::Forward } else { TransformDirection::Backward };
        let candidates = transform.find_candidates(current, direction);
        if candidates.is_empty() {
            return engine::ProposeResult::NoCandidates;
        }
        let loc = candidates.choose(rng).unwrap();
        let mut next = current.clone();
        match transform.apply(&mut next, loc, direction) {
            Ok(()) => {
                let next = dce::dce(&next);
                engine::ProposeResult::Proposed { candidate: next, kind, always_equivalent: transform.always_equivalent() }
            }
            Err(_e) => engine::ProposeResult::ApplyFailed { kind },
        }
    }
}

/// Validator that performs SIMD baseline checking, then SAT/IR equivalence.
pub struct G8rValidator {
    simd_inputs: Vec<Vec256>,
    baseline_outputs: Vec<Vec256>,
    paranoid: bool,
}

impl G8rValidator {
    /// Create a validator using the original baseline `GateFn` and RNG for inputs.
    pub fn new(original: &GateFn, rng: &mut Pcg64Mcg, paranoid: bool) -> Self {
        let simd_inputs = generate_simd_inputs(original, rng);
        let baseline_outputs = gate_simd::eval(original, &simd_inputs).outputs;
        Self { simd_inputs, baseline_outputs, paranoid }
    }
}

impl engine::Validator<GateFn> for G8rValidator {
    fn validate(&self, before: &GateFn, after: &GateFn) -> engine::ValidationVerdict {
        let sim_start = Instant::now();
        let cand_out = gate_simd::eval(after, &self.simd_inputs).outputs;
        let sim_time = sim_start.elapsed().as_micros();
        if self.baseline_outputs != cand_out {
            return engine::ValidationVerdict::Rejected { stage: engine::ValidationStage::Cheap, time_micros: sim_time };
        }
        let sat_start = Instant::now();
        let sat_ok = oracle_equiv_sat(before, after);
        let sat_time = sat_start.elapsed().as_micros();
        if self.paranoid {
            let external_ok = crate::check_equivalence::prove_same_gate_fn_via_ir(before, after).is_ok();
            if sat_ok != external_ok {
                panic!(
                    "[mcmc] ERROR: SAT oracle and external check_equivalence_with_top DISAGREE: SAT oracle: {}, external: {}",
                    sat_ok, external_ok
                );
            }
        }
        if sat_ok {
            engine::ValidationVerdict::Equivalent { cheap_micros: sim_time, oracle_micros: sat_time }
        } else {
            engine::ValidationVerdict::Rejected { stage: engine::ValidationStage::Expensive, time_micros: sat_time }
        }
    }
}

/// Returns a vector of weights for the given transforms and objective.
pub fn build_transform_weights<T: AsRef<[Box<dyn Transform>]>>(transforms: T, objective: Objective) -> Vec<f64> {
    fn weight_for_kind(k: TransformKind, obj: Objective) -> f64 {
        use TransformKind::*;
        match obj {
            Objective::Nodes | Objective::Product => match k {
                RemoveRedundantAnd | RemoveFalseAnd | RemoveTrueAnd | UnduplicateGate | MergeFanout => 3.0,
                InsertRedundantAnd | InsertFalseAnd | InsertTrueAnd | DuplicateGate | SplitFanout | UnfactorSharedAnd => 0.5,
                _ => 1.0,
            },
            Objective::Depth => match k {
                RotateAndRight | RotateAndLeft | BalanceAndTree | UnbalanceAndTree | SplitFanout | FactorSharedAnd => 3.0,
                _ => 1.0,
            },
        }
    }
    transforms
        .as_ref()
        .iter()
        .map(|t| weight_for_kind(t.kind(), objective))
        .collect()
}

/// Options mirror for g8r, including SAT reset interval for compatibility.
#[derive(Clone, Debug)]
pub struct McmcOptions {
    /// Interval to reset SAT context (currently unused in generic engine).
    pub sat_reset_interval: u64,
    /// Starting temperature.
    pub initial_temperature: f64,
    /// Global iteration offset.
    pub start_iteration: u64,
    /// Planned total iterations (for cooling schedule).
    pub total_iters: Option<u64>,
}

/// Shared best wrapper re-exported from the engine.
pub type Best = engine::Best<GateFn>;

const MIN_TEMPERATURE_RATIO: f64 = 1e-5;

/// Load the starting `GateFn` from sample, `.g8r`, or IR file.
pub fn load_start<P: AsRef<Path>>(p_generic: P) -> anyhow::Result<GateFn> {
    let p_str = p_generic.as_ref().to_str().unwrap_or_default();

    if p_str.starts_with("sample://") {
        let sample_name = p_str.trim_start_matches("sample://");
        match sample_name {
            "bf16_add" => {
                let loaded_sample = load_bf16_add_sample(SampleOpt::Yes);
                let sample_cost = cost(&loaded_sample.gate_fn);
                println!(
                    "Sample '{}' loaded. Initial stats: nodes={}, depth={}",
                    sample_name, sample_cost.nodes, sample_cost.depth
                );
                Ok(loaded_sample.gate_fn)
            }
            "bf16_mul" => {
                let loaded_sample = load_bf16_mul_sample(SampleOpt::Yes);
                let sample_cost = cost(&loaded_sample.gate_fn);
                println!(
                    "Sample '{}' loaded. Initial stats: nodes={}, depth={}",
                    sample_name, sample_cost.nodes, sample_cost.depth
                );
                Ok(loaded_sample.gate_fn)
            }
            _ if sample_name.starts_with("ripple_carry_adder:") => {
                let bits_str = sample_name.trim_start_matches("ripple_carry_adder:");
                let bits: usize = bits_str.parse().map_err(|_| anyhow::anyhow!("Invalid bit width '{}'", bits_str))?;
                let gfn = make_ripple_carry_adder(bits);
                let sample_cost = cost(&gfn);
                println!(
                    "Sample '{}' loaded. Initial stats: nodes={}, depth={}",
                    sample_name, sample_cost.nodes, sample_cost.depth
                );
                Ok(gfn)
            }
            _ => Err(anyhow::anyhow!("Unknown sample name: {}", sample_name)),
        }
    } else {
        let path = p_generic.as_ref();
        match path.extension().and_then(|e| e.to_str()) {
            Some("g8r") => {
                println!("Loading GateFn from path: {}", p_str);
                let contents = std::fs::read_to_string(path)
                    .map_err(|e| anyhow::anyhow!("Failed to read GateFn file '{}': {}", p_str, e))?;
                let gfn = GateFn::try_from(contents.as_str())
                    .map_err(|e| anyhow::anyhow!("Failed to parse GateFn from '{}': {}", p_str, e))?;
                let g_cost = cost(&gfn);
                println!(
                    "Loaded GateFn. Initial stats: nodes={}, depth={}",
                    g_cost.nodes, g_cost.depth
                );
                Ok(gfn)
            }
            _ => {
                println!("Loading IR from path: {}", p_str);
                let package = ir_parser::parse_and_validate_path_to_package(path)
                    .map_err(|e| anyhow::anyhow!("Failed to parse IR package '{}': {:?}", p_str, e))?;
                let top_entity = package
                    .get_top_fn()
                    .ok_or_else(|| anyhow::anyhow!("No top entity found in IR package '{}'", p_str))?;
                println!("Found top function: {}", top_entity.name);
                let gatify_options = GatifyOptions {
                    fold: true,
                    hash: true,
                    check_equivalence: false,
                    adder_mapping: crate::ir2gate_utils::AdderMapping::default(),
                };
                let gatify_output = ir2gate::gatify(top_entity, gatify_options)
                    .map_err(|e| anyhow::anyhow!("Failed to gatify IR from '{}': {}", p_str, e))?;
                println!("Successfully gatified main function into GateFn.");
                Ok(gatify_output.gate_fn)
            }
        }
    }
}

/// Generates fixed 256-wide random inputs matching `gate_fn` input widths.
fn generate_simd_inputs(gate_fn: &GateFn, rng: &mut impl rand::Rng) -> Vec<Vec256> {
    const LANES: usize = 256;
    let total_bits: usize = gate_fn.inputs.iter().map(|i| i.get_bit_count()).sum();
    let mut words_per_bit = vec![[0u64; 4]; total_bits];

    for lane in 0..LANES {
        let mut bit_cursor = 0;
        for input in &gate_fn.inputs {
            let rand_val = xlsynth_pir::fuzz_utils::arbitrary_irbits(rng, input.bit_vector.get_bit_count());
            for bit_idx in 0..input.bit_vector.get_bit_count() {
                if rand_val.get_bit(bit_idx).unwrap() {
                    let limb = lane / 64;
                    let offset = lane % 64;
                    words_per_bit[bit_cursor + bit_idx][limb] |= 1u64 << offset;
                }
            }
            bit_cursor += input.bit_vector.get_bit_count();
        }
    }

    words_per_bit.into_iter().map(|w| Vec256(u64x4::from_array(w))).collect()
}

/// Write a checkpoint (best g8r + stats JSON) and verify equivalence.
pub fn write_checkpoint(
    g8r_path: &Path,
    stats_path: &Path,
    original_gfn: &GateFn,
    best_gfn: &GateFn,
    iter: u64,
    context: &str,
) -> anyhow::Result<()> {
    let equiv_ok_sat = oracle_equiv_sat(original_gfn, best_gfn);
    use crate::check_equivalence::{IrCheckResult, prove_same_gate_fn_via_ir_status};

    let ir_status = prove_same_gate_fn_via_ir_status(original_gfn, best_gfn);
    let equiv_ok_external = matches!(ir_status, IrCheckResult::Equivalent);

    if equiv_ok_sat != equiv_ok_external || !equiv_ok_sat {
        if let Some(parent_dir) = g8r_path.parent() {
            let dump_dir = parent_dir.join("equiv_failures");
            let _ = std::fs::create_dir_all(&dump_dir);
            let orig_dump = dump_dir.join(format!("orig_iter_{}.g8r", iter));
            let cand_dump = dump_dir.join(format!("cand_iter_{}.g8r", iter));
            let orig_bin_dump = dump_dir.join(format!("orig_iter_{}.g8rbin", iter));
            let cand_bin_dump = dump_dir.join(format!("cand_iter_{}.g8rbin", iter));
            let _ = std::fs::write(&orig_dump, original_gfn.to_string());
            let _ = std::fs::write(&cand_dump, best_gfn.to_string());
            let _ = std::fs::write(&orig_bin_dump, bincode::serialize(original_gfn).unwrap());
            let _ = std::fs::write(&cand_bin_dump, bincode::serialize(best_gfn).unwrap());
            eprintln!(
                "[mcmc] Disagreeing GateFns dumped to {} and {} (text), {} and {} (bincode)",
                orig_dump.display(),
                cand_dump.display(),
                orig_bin_dump.display(),
                cand_bin_dump.display()
            );
        }
        return Err(anyhow::anyhow!(
            "[mcmc] ERROR: Equivalence disagreement at iter {} (sat:{}, external:{})",
            iter, equiv_ok_sat, equiv_ok_external
        ));
    }

    match ir_status {
        crate::check_equivalence::IrCheckResult::Equivalent => {}
        crate::check_equivalence::IrCheckResult::TimedOutOrInterrupted => {
            eprintln!(
                "[mcmc] Warning: External IR equivalence check timed out or was interrupted (iteration {}). Proceeding with SAT oracle result only.",
                iter
            );
        }
        crate::check_equivalence::IrCheckResult::OtherProcessError(ref msg) => {
            eprintln!(
                "[mcmc] Warning: External IR equivalence checker failed at iter {}: {}",
                iter, msg
            );
        }
        crate::check_equivalence::IrCheckResult::NotEquivalent => {}
    }

    if let Err(e) = std::fs::write(g8r_path, best_gfn.to_string()) {
        eprintln!(
            "[mcmc] Warning: Failed to write {} checkpoint to {}: {:?}",
            context,
            g8r_path.display(),
            e
        );
    }
    let stats = get_summary_stats::get_summary_stats(best_gfn);
    match serde_json::to_string_pretty(&stats) {
        Ok(json) => {
            if let Err(e) = std::fs::write(stats_path, json) {
                eprintln!(
                    "[mcmc] Warning: Failed to write {} stats checkpoint to {}: {:?}",
                    context,
                    stats_path.display(),
                    e
                );
            } else {
                println!(
                    "[mcmc] {}: checkpoint written to {} and {} | Equivalence: OK",
                    context,
                    g8r_path.display(),
                    stats_path.display()
                );
            }
        }
        Err(e) => {
            eprintln!(
                "[mcmc] Warning: Failed to serialize stats for {} checkpoint at iteration {}: {:?}",
                context, iter, e
            );
        }
    }

    Ok(())
}
