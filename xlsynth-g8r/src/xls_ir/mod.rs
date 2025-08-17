// SPDX-License-Identifier: Apache-2.0

//! Functionality that is purely related to the XLS IR, i.e. parsing,
//! representing, querying/manipulating, etc.

pub mod edit_distance;
pub mod ir;
pub mod ir_node_env;
pub mod ir_parser;
pub mod ir_utils;
pub mod structural_similarity;
