// SPDX-License-Identifier: Apache-2.0

#![no_main]
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashMap;
use std::sync::Once;

use xlsynth_g8r::aig::bulk_replace::{SubstitutionMap, bulk_replace};
use xlsynth_g8r::aig::dce::dce;
use xlsynth_g8r::aig::{AigOperand, AigRef};
use xlsynth_g8r::aig_sim::gate_sim::{Collect, eval};
use xlsynth_g8r::gate_random::{RandomGateOptions, RandomGateShapes};
use xlsynth_g8r_fuzz::direct_g8r::generate_combinational;

use xlsynth_pir::random_inputs::generate_mixed_irbits_with_rng;

static LOGGER_INIT: Once = Once::new();

#[derive(Debug, Clone, Arbitrary)]
struct FuzzSubstitutions {
    // Indices into the node list to substitute, and their replacements
    subs: Vec<(u16, u16, bool)>, // (from_idx, to_idx, negated)
}

/// Seeds simulation reproducibly from the graph bytes and edit sequence.
fn make_rng(graph: &[u8], substitutions: &FuzzSubstitutions) -> StdRng {
    let mut hasher = blake3::Hasher::new();
    hasher.update(graph);
    for (from, to, negated) in &substitutions.subs {
        hasher.update(&from.to_le_bytes());
        hasher.update(&to.to_le_bytes());
        hasher.update(&[u8::from(*negated)]);
    }
    StdRng::from_seed(*hasher.finalize().as_bytes())
}

fuzz_target!(|data: (Vec<u8>, FuzzSubstitutions)| {
    // Ensure XLSYNTH_TOOLS is set up front for equivalence checking
    if std::env::var("XLSYNTH_TOOLS").is_err() {
        panic!(
            "XLSYNTH_TOOLS environment variable must be set for equivalence checking in this fuzz target"
        );
    }
    LOGGER_INIT.call_once(|| {
        let _ = env_logger::Builder::from_env(env_logger::Env::default())
            .is_test(true)
            .try_init();
    });
    let mut rng = make_rng(&data.0, &data.1);
    // Explicit structural stress is confined to this substitution campaign.
    let options = RandomGateOptions {
        input_ports: 0..=16,
        input_width: 1..=16,
        output_ports: 1..=16,
        shapes: RandomGateShapes {
            max_chain_length: 8,
            max_redundant_pairs: 8,
            max_fanout_branches: 5,
        },
        ..RandomGateOptions::default()
    };
    let generated = generate_combinational(&data.0, &options)
        .expect("valid bounded substitution generator options");
    let gate_fn = generated.graph;
    let opts = generated.builder_options;
    let mut subs_map = HashMap::new();
    let node_count = gate_fn.gates.len();
    // Collect all input node IDs
    let mut input_node_ids = std::collections::HashSet::new();
    for input in &gate_fn.inputs {
        for i in 0..input.get_bit_count() {
            input_node_ids.insert(input.bit_vector.get_lsb(i).node.id);
        }
    }
    // Model the cone/cut replacements used by production callers. AIG node IDs
    // are topological, so replacing a node only with an earlier node cannot
    // introduce a dependency cycle.
    let mut targets = std::collections::HashSet::<usize>::new();
    let mut any_valid_sub = false;
    for (from_idx, to_idx, negated) in &data.1.subs {
        if *from_idx as usize >= node_count || *to_idx as usize >= node_count {
            continue;
        }
        if input_node_ids.contains(&(*from_idx as usize)) {
            continue; // Don't substitute input nodes!
        }
        if *from_idx as usize == 0 {
            continue; // Don't substitute the constant literal node!
        }
        let from_idx_usize = *from_idx as usize;
        let to_idx_usize = *to_idx as usize;
        if to_idx_usize >= from_idx_usize {
            continue;
        }
        if targets.contains(&from_idx_usize) {
            continue;
        }
        // Avoid chaining: don't substitute to a node that's also a target
        if targets.contains(&to_idx_usize) {
            continue;
        }
        let from_ref = AigRef { id: from_idx_usize };
        let to_op = AigOperand {
            node: AigRef { id: to_idx_usize },
            negated: *negated,
        };
        subs_map.insert(from_ref, to_op);
        targets.insert(from_idx_usize);
        any_valid_sub = true;
    }
    // If all substitutions were vetoed, skip this fuzz case
    if !any_valid_sub && !subs_map.is_empty() {
        return;
    }
    // Veto samples with substitution chains (a node is both a key and a target)
    let keys: std::collections::HashSet<_> = subs_map.keys().map(|k| k.id).collect();
    let targets: std::collections::HashSet<_> = subs_map.values().map(|v| v.node.id).collect();
    if !keys.is_disjoint(&targets) {
        // Substitution chain detected, skip this fuzz case
        return;
    }
    // Defensive: ensure postorder traversal completes and is not too large
    let _postorder = gate_fn.post_order_operands(false);
    let substitutions = {
        let mut s = SubstitutionMap::new();
        for (k, v) in subs_map.into_iter() {
            s.add(k, v);
        }
        s
    };
    let new_fn = bulk_replace(&gate_fn, &substitutions, opts);
    // Assert that substitution does not increase node count
    assert!(
        new_fn.gates.len() <= gate_fn.gates.len(),
        "bulk_replace increased node count: before = {}, after = {}",
        gate_fn.gates.len(),
        new_fn.gates.len()
    );
    // Check invariants and for cycles
    new_fn.check_invariants_with_debug_assert();
    let _dce_fn = dce(&new_fn);
    // If we get here, the graph is valid and passes invariants.

    // 1. Structural checks: output count and widths
    assert_eq!(
        gate_fn.outputs.len(),
        new_fn.outputs.len(),
        "Output count changed after bulk_replace"
    );
    for (orig, new) in gate_fn.outputs.iter().zip(new_fn.outputs.iter()) {
        assert_eq!(
            orig.get_bit_count(),
            new.get_bit_count(),
            "Output width changed for {}",
            orig.name
        );
    }

    // 2. No-op rebuild: the original substitution map is expressed in terms of
    //    `gate_fn` node IDs, but `bulk_replace` runs DCE and renumbers nodes in
    //    `new_fn`. Reapplying that stale map would be invalid, so rebuild
    //    `new_fn` with an empty map instead and check output shape stability.
    let no_op_substitutions = SubstitutionMap::new();
    let new_fn2 = bulk_replace(&new_fn, &no_op_substitutions, opts);
    assert_eq!(
        new_fn.outputs.len(),
        new_fn2.outputs.len(),
        "Idempotence: output count changed"
    );
    for (a, b) in new_fn.outputs.iter().zip(new_fn2.outputs.iter()) {
        assert_eq!(
            a.get_bit_count(),
            b.get_bit_count(),
            "Idempotence: output width changed"
        );
    }

    // 3. No-op DCE: DCE should not change the simulation outputs
    let dce_fn = dce(&gate_fn);
    assert_eq!(
        gate_fn.outputs.len(),
        dce_fn.outputs.len(),
        "No-op DCE: output count changed"
    );
    for (orig, new) in gate_fn.outputs.iter().zip(dce_fn.outputs.iter()) {
        assert_eq!(
            orig.get_bit_count(),
            new.get_bit_count(),
            "No-op DCE: output width changed"
        );
    }
    // Optionally, check simulation outputs for a random input vector
    let widths = gate_fn
        .inputs
        .iter()
        .map(|input| input.get_bit_count())
        .collect::<Vec<_>>();
    let input_vecs = generate_mixed_irbits_with_rng(&mut rng, &widths);
    let orig_sim = eval(&gate_fn, &input_vecs, Collect::None);
    let dce_sim = eval(&dce_fn, &input_vecs, Collect::None);
    assert_eq!(
        orig_sim.outputs, dce_sim.outputs,
        "No-op DCE: simulation outputs changed"
    );
});
