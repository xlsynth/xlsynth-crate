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
pub mod ir2gate;
mod ir2gate_utils;
pub mod liberty;
pub mod process_ir_path;
pub mod use_count;
pub mod xls_ir;
