// SPDX-License-Identifier: Apache-2.0

#![no_main]
use libfuzzer_sys::fuzz_target;

use std::collections::{HashMap, HashSet, VecDeque};

use xlsynth_g8r::equiv::types::{AssertionSemantics, EquivResult, IrFn as EqIrFn};
use xlsynth_g8r::equiv::prove_equiv::prove_ir_fn_equiv;
#[cfg(feature = "has-bitwuzla")]
use xlsynth_g8r::equiv::bitwuzla_backend::{Bitwuzla, BitwuzlaOptions};
#[cfg(feature = "has-boolector")]
use xlsynth_g8r::equiv::boolector_backend::{Boolector, BoolectorConfig};
#[cfg(feature = "has-easy-smt")]
use xlsynth_g8r::equiv::easy_smt_backend::{EasySmtConfig, EasySmtSolver};

use xlsynth_pir::ir::{Fn as IrFn, NodePayload, NodeRef, PackageMember};
use xlsynth_pir::ir_fuzz::{FuzzSample, generate_ir_fn};
use xlsynth_pir::ir_outline::{compute_default_ordering, outline_with_ordering, OutlineOrdering};
use xlsynth_pir::ir_parser::Parser;
use xlsynth_pir::ir_utils::operands;

fn build_users(f: &IrFn) -> HashMap<usize, Vec<usize>> {
    let mut users: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, n) in f.nodes.iter().enumerate() {
        for dep in operands(&n.payload).into_iter() {
            users.entry(dep.index).or_default().push(i);
        }
    }
    users
}

fn pick_connected_region(f: &IrFn, seed: usize, max_size: usize) -> HashSet<usize> {
    let users = build_users(f);
    let mut visited: HashSet<usize> = HashSet::new();
    let mut q: VecDeque<usize> = VecDeque::new();
    q.push_back(seed);
    visited.insert(seed);
    while let Some(cur) = q.pop_front() {
        if visited.len() >= max_size { break; }
        // Neighbors: operands and users
        for dep in operands(&f.nodes[cur].payload).into_iter().map(|r| r.index) {
            if visited.len() >= max_size { break; }
            if visited.insert(dep) { q.push_back(dep); }
        }
        if let Some(us) = users.get(&cur) {
            for &u in us.iter() {
                if visited.len() >= max_size { break; }
                if visited.insert(u) { q.push_back(u); }
            }
        }
    }
    visited
}

fn has_boundary_output(f: &IrFn, sel: &HashSet<usize>) -> bool {
    // If selection includes return, it's a boundary output.
    if let Some(ret) = f.ret_node_ref { if sel.contains(&ret.index) { return true; } }
    let users = build_users(f);
    for &idx in sel.iter() {
        if let Some(us) = users.get(&idx) {
            if us.iter().any(|u| !sel.contains(u)) { return true; }
        }
    }
    false
}

fn stable_hash_u64(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut h);
    h.finish()
}

#[derive(Clone, Debug)]
enum ParamOrderMode {
    Default,
    NonDefault,
}

#[derive(Clone, Debug)]
enum ReturnOrderMode {
    Default,
    NonDefault,
}

fn xorshift64(mut x: u64) -> u64 {
    // Stateless xorshift step for deterministic pseudo-random numbers.
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

fn shuffle_indices_deterministic(indices: &mut [usize], mut seed: u64) {
    if indices.len() <= 1 { return; }
    for i in (1..indices.len()).rev() {
        seed = xorshift64(seed);
        let j = (seed as usize) % (i + 1);
        indices.swap(i, j);
    }
}

fn permute_vec_deterministic<T: Clone>(v: &mut Vec<T>, seed: u64) {
    if v.len() <= 1 { return; }
    let mut idx: Vec<usize> = (0..v.len()).collect();
    shuffle_indices_deterministic(&mut idx, seed);
    let old = v.clone();
    for (k, &i) in idx.iter().enumerate() {
        v[k] = old[i].clone();
    }
}

fuzz_target!(|sample: FuzzSample| {
    let _ = env_logger::builder().is_test(true).try_init();

    // Degenerate fuzz samples are not informative for outlining, skip.
    if sample.ops.is_empty() || sample.input_bits == 0 {
        return; // Early-return: degenerate input (no ops or zero-width inputs)
    }

    // Build an XLS IR package via C++ bindings
    let mut cpp_pkg = match xlsynth::IrPackage::new("outline_fuzz_pkg") {
        Ok(p) => p,
        Err(_) => return, // Early-return: infra issue constructing package
    };
    if let Err(_e) = generate_ir_fn(sample.input_bits, sample.ops.clone(), &mut cpp_pkg, None) {
        return; // Early-return: generator could not build function with given ops
    }

    // Parse into Rust IR (g8r)
    let orig_pkg_text = cpp_pkg.to_string();
    let orig_pkg = match Parser::new(&orig_pkg_text).parse_and_validate_package() {
        Ok(p) => p,
        Err(_e) => {
            // Unexpected: our own pretty-printed text should parse; flag failure.
            panic!("Rust IR parser failed on C++-emitted package text");
        }
    };

    log::info!("orig_pkg_text:\n{}", orig_pkg_text);

    let mut work_pkg = match Parser::new(&orig_pkg_text).parse_and_validate_package() {
        Ok(p) => p,
        Err(_e) => panic!("Rust IR parser failed on C++-emitted package text (clone)")
    };

    let orig_fn = orig_pkg.get_top().expect("missing top function").clone();
    let n = orig_fn.nodes.len();
    if n < 2 {
        return; // Early-return: not enough nodes to outline a nontrivial region
    }

    // Derive deterministic seed and target size from the IR text to avoid external RNG
    let h = stable_hash_u64(&orig_pkg_text);
    let seed = (h as usize) % n;
    // Aim for a small connected region size in [1, min(n-1, 16)]
    let max_region = std::cmp::min(n.saturating_sub(1), 16usize);
    if max_region == 0 { return; }
    let target = 1 + ((h.rotate_left(13) as usize) % max_region);

    let region = pick_connected_region(&orig_fn, seed, target);
    if region.is_empty() { return; }
    if !has_boundary_output(&orig_fn, &region) {
        return; // Early-return: selected region produces no boundary outputs (uninformative)
    }

    // Convert region into NodeRef set
    let to_outline: HashSet<NodeRef> = region.into_iter().map(|i| NodeRef { index: i }).collect();

    // Perform outlining on a mutable clone of the package using the original function
    let work_fn = work_pkg.get_top().unwrap().clone();
    // Choose parameter and return ordering modes from hash bits for reproducibility
    let param_mode = if (h & 1) == 0 { ParamOrderMode::Default } else { ParamOrderMode::NonDefault };
    let ret_mode = if (h & 2) == 0 { ReturnOrderMode::Default } else { ReturnOrderMode::NonDefault };

    // Build ordering and optionally permute deterministically
    let mut ordering: OutlineOrdering = compute_default_ordering(&work_fn, &to_outline);
    if let ParamOrderMode::NonDefault = param_mode { permute_vec_deterministic(&mut ordering.params, h.rotate_right(17)); }
    if let ReturnOrderMode::NonDefault = ret_mode { permute_vec_deterministic(&mut ordering.returns, h.rotate_left(9)); }

    let res = outline_with_ordering(
        &work_fn,
        &to_outline,
        "outlined_outer",
        "outlined_inner",
        &ordering,
        &mut work_pkg,
    );

    log::info!("work_pkg_text:\n{}", work_pkg.to_string());

    // Prove equivalence: original function == outlined outer function
    let lhs = EqIrFn::new(&orig_fn, None);
    let rhs = EqIrFn::new(&res.outer, None);

    #[cfg(feature = "has-bitwuzla")]
    {
        let r = prove_ir_fn_equiv::<Bitwuzla>(&BitwuzlaOptions::new(), &lhs, &rhs, AssertionSemantics::Same, false);
        if let EquivResult::Disproved { lhs_inputs, rhs_inputs, .. } = r {
            // Provide context in logs on failure
            if let Some(f) = orig_pkg.get_top() {
                log::info!("Original IR fn:\n{}", f);
            }
            if let Some(PackageMember::Function(f)) = work_pkg.members.iter().find(|m| matches!(m, PackageMember::Function(f) if f.name == "outlined_outer")) {
                log::info!("Outlined outer IR fn:\n{}", f);
            }
            panic!("Outline equivalence failed (Bitwuzla): lhs_inputs={:?} rhs_inputs={:?}", lhs_inputs, rhs_inputs);
        }
    }

    #[cfg(feature = "has-boolector")]
    {
        let r = prove_ir_fn_equiv::<Boolector>(&BoolectorConfig::new(), &lhs, &rhs, AssertionSemantics::Same, false);
        if let EquivResult::Disproved { lhs_inputs, rhs_inputs, .. } = r {
            if let Some(f) = orig_pkg.get_top() {
                log::info!("Original IR fn:\n{}", f);
            }
            if let Some(PackageMember::Function(f)) = work_pkg.members.iter().find(|m| matches!(m, PackageMember::Function(f) if f.name == "outlined_outer")) {
                log::info!("Outlined outer IR fn:\n{}", f);
            }
            panic!("Outline equivalence failed (Boolector): lhs_inputs={:?} rhs_inputs={:?}", lhs_inputs, rhs_inputs);
        }
    }

    #[cfg(feature = "with-z3-binary-test")]
    {
        let r = prove_ir_fn_equiv::<EasySmtSolver>(&EasySmtConfig::z3(), &lhs, &rhs, AssertionSemantics::Same, false);
        if let EquivResult::Disproved { lhs_inputs, rhs_inputs, .. } = r {
            panic!("Outline equivalence failed (Z3 binary): lhs_inputs={:?} rhs_inputs={:?}", lhs_inputs, rhs_inputs);
        }
    }
});
