// SPDX-License-Identifier: Apache-2.0

#![feature(map_try_insert)]
#![feature(let_chains)]
#![feature(custom_test_frameworks)]

mod aig_simplify;
pub mod check_equivalence;
mod emit_netlist;
pub mod find_structures;
pub mod gate;
pub mod gate2ir;
mod gate_sim;
pub mod ir;
pub mod ir2gate;
mod ir2gate_utils;
mod ir_node_env;
pub mod ir_parser;
mod ir_utils;
pub mod use_count;
