// SPDX-License-Identifier: Apache-2.0

#![feature(map_try_insert)]

//! Functionality that is purely related to the XLS IR, i.e. parsing,
//! representing, querying/manipulating, etc.

pub mod dce;
pub mod edit_distance;
pub mod fuzz_utils;
pub mod greedy_matching_ged;
pub mod ir;
pub mod ir_deduce;
pub mod ir_eval;
pub mod ir_fuzz;
pub mod ir_node_env;
pub mod ir_outline;
pub mod ir_parser;
pub mod ir_rebase_ids;
pub mod ir_utils;
pub mod ir_validate;
pub mod ir_value_utils;
pub mod ir_verify;
pub mod ir_verify_parity;
pub mod localized_eco2;
pub mod matching_ged;
pub mod node_hashing;
pub mod prove_equiv_via_toolchain;
pub mod simple_rebase;
pub mod structural_similarity;
