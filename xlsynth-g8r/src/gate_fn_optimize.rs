// SPDX-License-Identifier: Apache-2.0

//! Reusable post-gatification optimization pipeline for combinational GateFns.

use std::sync::Arc;

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::aig::GateFn;
use crate::aig::cut_db_rewrite::{self, CutDbRewriteMode};
use crate::aig::dce;
use crate::aig::fraig::{self, FraigPassStat};
use crate::aig::reassociation;
use crate::cut_db::loader::CutDb;
use crate::prove_gate_fn_equiv_common::GateFormalBackend;
use crate::prove_gate_fn_equiv_sat::{DEFAULT_CADICAL_TERMINATE_LIMIT, GateFormalOptions};
use crate::use_count::get_id_to_use_count;

pub const DEFAULT_MAX_FRAIG_SIM_SAMPLES: usize = 8 * 1024;

/// Configures the optimization stages that run after gatification.
#[derive(Clone)]
pub struct GateFnOptimizeOptions {
    pub fraig: bool,
    pub reassociation: bool,
    pub max_fraig_sim_samples: Option<usize>,
    pub gate_formal_backend: GateFormalBackend,
    pub cadical_terminate_limit: u32,
    pub cut_db: Option<Arc<CutDb>>,
    pub cut_db_rewrite_max_iterations: usize,
    pub cut_db_rewrite_max_cuts_per_node: usize,
    pub cut_db_enable_large_cone_rewrite: bool,
    pub cut_db_rewrite_mode: CutDbRewriteMode,
}

impl GateFnOptimizeOptions {
    /// Returns options that retain only the mandatory initial dead-code pass.
    pub fn all_disabled() -> Self {
        Self {
            fraig: false,
            reassociation: false,
            max_fraig_sim_samples: Some(DEFAULT_MAX_FRAIG_SIM_SAMPLES),
            gate_formal_backend: GateFormalBackend::default(),
            cadical_terminate_limit: DEFAULT_CADICAL_TERMINATE_LIMIT,
            cut_db: None,
            cut_db_rewrite_max_iterations:
                crate::cut_db_cli_defaults::CUT_DB_REWRITE_MAX_ITERATIONS_CLI,
            cut_db_rewrite_max_cuts_per_node:
                crate::cut_db_cli_defaults::CUT_DB_REWRITE_MAX_CUTS_PER_NODE_CLI,
            cut_db_enable_large_cone_rewrite: true,
            cut_db_rewrite_mode: CutDbRewriteMode::default(),
        }
    }
}

impl Default for GateFnOptimizeOptions {
    fn default() -> Self {
        Self {
            fraig: true,
            reassociation: true,
            cut_db: Some(CutDb::load_default()),
            ..Self::all_disabled()
        }
    }
}

/// Holds the optimized graph and statistics produced by optional stages.
pub struct GateFnOptimizeOutcome {
    pub gate_fn: GateFn,
    pub fraig_pass_stat: Option<FraigPassStat>,
}

/// Runs the canonical post-gatification optimization sequence on `gate_fn`.
pub fn optimize_gate_fn(
    gate_fn: GateFn,
    options: &GateFnOptimizeOptions,
) -> Result<GateFnOptimizeOutcome, String> {
    let pre_dce_gate_count = gate_fn.gates.len();
    let mut gate_fn = dce::dce(&gate_fn);
    log::info!(
        "pre-fraig dce: gate count {} -> {}",
        pre_dce_gate_count,
        gate_fn.gates.len()
    );
    gate_fn.check_invariants_with_debug_assert();

    let mut fraig_pass_stat = None;
    if options.fraig {
        log::info!(
            "fraig is enabled for GateFn {:?} with {} nodes",
            gate_fn.name,
            gate_fn.gates.len()
        );
        let live_node_count = get_id_to_use_count(&gate_fn).len().max(1);
        let scaled = (live_node_count as f64 / 8.0).ceil() as usize;
        let uncapped = round_up_to_nearest_multiple(scaled, 256);
        let sim_samples = options
            .max_fraig_sim_samples
            .map_or(uncapped, |max_samples| uncapped.min(max_samples));
        log::info!(
            "fraig sim samples; live node count: {}, scaled: {}, uncapped: {}, max: {:?}, result: {}",
            live_node_count,
            scaled,
            uncapped,
            options.max_fraig_sim_samples,
            sim_samples
        );
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0);
        let gate_formal_options = GateFormalOptions::default()
            .with_cadical_terminate_limit(options.cadical_terminate_limit);
        let result = fraig::fraig_optimize_with_backend_and_options(
            &gate_fn,
            sim_samples,
            options.gate_formal_backend,
            gate_formal_options,
            &mut rng,
        )
        .map_err(|error| format!("Fraig optimization failed: {error}"))?;
        gate_fn = result.optimized_fn;
        fraig_pass_stat = Some(result.stat);
    }

    if options.reassociation {
        log::info!("reassociation enabled");
        gate_fn = reassociation::reassociate_gatefn(&gate_fn);
    }

    if let Some(db) = options.cut_db.as_ref() {
        log::info!("cut-db rewrite enabled");
        gate_fn = cut_db_rewrite::rewrite_gatefn_with_cut_db(
            &gate_fn,
            db.as_ref(),
            cut_db_rewrite::RewriteOptions {
                max_cuts_per_node: options.cut_db_rewrite_max_cuts_per_node,
                max_iterations: options.cut_db_rewrite_max_iterations,
                verify_area_costing: false,
                verify_delay_costing: false,
                enable_large_cone_rewrite: options.cut_db_enable_large_cone_rewrite,
                mode: options.cut_db_rewrite_mode,
            },
        );
        if options.reassociation {
            log::info!("post-cut-db reassociation enabled");
            gate_fn = reassociation::reassociate_gatefn(&gate_fn);
        }
    }

    Ok(GateFnOptimizeOutcome {
        gate_fn,
        fraig_pass_stat,
    })
}

fn round_up_to_nearest_multiple(x: usize, y: usize) -> usize {
    ((x + y - 1) / y) * y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::aig::get_summary_stats::get_aig_stats;
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};

    fn build_linear_and4() -> GateFn {
        let mut builder = GateBuilder::new("and4".to_string(), GateBuilderOptions::opt());
        let a = *builder.add_input("a".to_string(), 1).get_lsb(0);
        let b = *builder.add_input("b".to_string(), 1).get_lsb(0);
        let c = *builder.add_input("c".to_string(), 1).get_lsb(0);
        let d = *builder.add_input("d".to_string(), 1).get_lsb(0);
        let ab = builder.add_and_binary(a, b);
        let abc = builder.add_and_binary(ab, c);
        let abcd = builder.add_and_binary(abc, d);
        builder.add_output("o".to_string(), abcd.into());
        builder.build()
    }

    #[test]
    fn can_run_reassociation_in_isolation() {
        let input = build_linear_and4();
        let mut options = GateFnOptimizeOptions::all_disabled();
        options.reassociation = true;

        let outcome = optimize_gate_fn(input, &options).unwrap();
        let stats = get_aig_stats(&outcome.gate_fn);
        assert_eq!(stats.and_nodes, 3);
        assert_eq!(stats.max_depth, 2);
        assert!(outcome.fraig_pass_stat.is_none());
    }
}
