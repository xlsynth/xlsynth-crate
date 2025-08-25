// SPDX-License-Identifier: Apache-2.0

//! Functionality that is purely related to the XLS IR, i.e. parsing,
//! representing, querying/manipulating, etc.

pub mod edit_distance;
pub mod ir;
pub mod ir2dslx;
pub mod ir_node_env;
pub mod ir_parser;
pub mod ir_utils;
pub mod ir_verify;
pub mod localized_eco;
pub mod node_hashing;
pub mod structural_similarity;
