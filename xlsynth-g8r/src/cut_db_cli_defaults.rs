// SPDX-License-Identifier: Apache-2.0
//! Consolidated defaults for cut-db rewriting as used by CLI entrypoints.
//!
//! These are *policy* knobs: we keep them deterministically bounded to avoid
//! unpredictable runtimes in tests/CI. Library callers can override via
//! `process_ir_path::Options` or directly via
//! `aig::cut_db_rewrite::RewriteOptions`.

/// Default max outer iterations for cut-db rewriting in CLI entrypoints.
pub const CUT_DB_REWRITE_MAX_ITERATIONS_CLI: usize = 4;

/// Default max cheap candidate depth evaluations per global recompute round.
pub const CUT_DB_REWRITE_MAX_CANDIDATE_EVALS_PER_ROUND_CLI: usize = 4096;

/// Default max accepted cut-db rewrites per global recompute round.
pub const CUT_DB_REWRITE_MAX_REWRITES_PER_ROUND_CLI: usize = 64;

/// Default max cuts retained per node during cut enumeration in CLI
/// entrypoints.
pub const CUT_DB_REWRITE_MAX_CUTS_PER_NODE_CLI: usize = 16;
