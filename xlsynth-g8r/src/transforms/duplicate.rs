use crate::gate::{AigNode, AigRef, GateFn};
use crate::topo::topo_sort_refs;
use rand::seq::SliceRandom;
use std::collections::HashMap;

/// Duplicate (replicate) a gate inside a `GateFn`.
///
/// The newly created gate is structurally *identical* to the one referenced by
/// `which`, i.e. it is created with the same kind and **re-uses** the very same
/// fan-ins.  The function deliberately leaves the list of designated outputs
/// untouched so that callers can decide on their own whether or not the fresh
/// gate should become observable.
///
/// Attempting to duplicate primary inputs or constants has no semantic value
/// and is therefore rejected with an error.
pub fn duplicate(g: &mut GateFn, which: AigRef) -> Result<AigRef, &'static str> {
    match &g.gates[which.id] {
        AigNode::Literal(_) => return Err("cannot duplicate literal"),
        AigNode::Input { .. } => return Err("cannot duplicate input"),
        AigNode::And2 { a, b, .. } => {
            let new_gate = AigNode::And2 {
                a: *a,
                b: *b,
                tags: None,
            };
            let new_ref = AigRef { id: g.gates.len() };
            g.gates.push(new_gate);
            Ok(new_ref)
        }
    }
}

/// Attempts to unduplicate a structurally redundant node in the graph.
/// Returns the AigRef that should now be dead (all references replaced), or
/// None if no unduplication was possible.
pub fn unduplicate<R: rand::Rng + ?Sized>(g: &mut GateFn, rng: &mut R) -> Option<AigRef> {
    // 1. Compute topo order
    let topo = topo_sort_refs(&g.gates);
    // 2. Build content hash -> Vec<AigRef> (excluding primary inputs)
    let mut buckets: HashMap<String, Vec<AigRef>> = HashMap::new();
    for &node_ref in &topo {
        match &g.gates[node_ref.id] {
            AigNode::Input { .. } => continue, // never deduplicate inputs
            AigNode::Literal(val) => {
                let key = format!("Literal({})", val);
                buckets.entry(key).or_default().push(node_ref);
            }
            AigNode::And2 { a, b, .. } => {
                let key = format!("And2({:?},{:?})", a, b);
                buckets.entry(key).or_default().push(node_ref);
            }
        }
    }
    // 3. Find a bucket with 2+ nodes
    let candidates: Vec<_> = buckets.values().filter(|v| v.len() >= 2).collect();
    if candidates.is_empty() {
        return None;
    }
    let bucket = candidates.choose(rng).unwrap();
    debug_assert!(bucket.len() >= 2, "Bucket should have at least 2 nodes");
    // 4. Pick two distinct nodes randomly
    let mut pair = bucket.iter().copied().collect::<Vec<_>>();
    pair.shuffle(rng);
    let (keep, kill) = (pair[0], pair[1]);
    // 5. Rewrite all references to kill -> keep (except for inputs)
    for node in &mut g.gates {
        match node {
            AigNode::And2 { a, b, .. } => {
                if a.node == kill {
                    a.node = keep;
                }
                if b.node == kill {
                    b.node = keep;
                }
            }
            _ => {}
        }
    }
    // 6. Return the node that should now be dead
    Some(kill)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::{AigBitVector, AigNode, AigOperand, AigRef};
    use crate::gate_builder::{GateBuilder, GateBuilderOptions};
    use crate::test_utils::setup_simple_graph;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_duplicate_and_gate() {
        let mut test_graph = setup_simple_graph();
        let orig_gate = test_graph.a.node; // 'a' is an AND gate
        let orig_fanins = match &test_graph.g.gates[orig_gate.id] {
            AigNode::And2 { a, b, .. } => (a.node, b.node),
            _ => panic!("Not an AND node"),
        };
        let new_ref = duplicate(&mut test_graph.g, orig_gate).expect("should duplicate AND");
        assert_ne!(new_ref, orig_gate);
        let new_gate = &test_graph.g.gates[new_ref.id];
        match new_gate {
            AigNode::And2 { a, b, .. } => {
                assert_eq!((a.node, b.node), orig_fanins);
            }
            _ => panic!("Duplicated node is not an AND"),
        }
    }

    #[test]
    fn test_duplicate_input_fails() {
        let mut test_graph = setup_simple_graph();
        let input_ref = test_graph.i0.node;
        let result = duplicate(&mut test_graph.g, input_ref);
        assert!(result.is_err());
    }

    #[test]
    fn test_duplicate_literal_fails() {
        let mut test_graph = setup_simple_graph();
        let literal_ref = AigRef { id: 0 };
        let result = duplicate(&mut test_graph.g, literal_ref);
        assert!(result.is_err());
    }

    #[test]
    fn test_unduplicate_after_duplicate() {
        // Build a minimal graph: a = AND(i0, i1)
        let mut gb = GateBuilder::new("g".to_string(), GateBuilderOptions::no_opt());
        let i0 = gb.add_input("i0".to_string(), 1).get_lsb(0).clone();
        let i1 = gb.add_input("i1".to_string(), 1).get_lsb(0).clone();
        let a = gb.add_and_binary(i0, i1);
        gb.add_output("o".to_string(), a.into());
        let mut g = gb.build();
        // Duplicate 'a'
        let dup = super::duplicate(&mut g, a.node).expect("should duplicate");
        // Add the duplicate as an output to keep it alive
        g.outputs.push(crate::gate::Output {
            name: "dup".to_string(),
            bit_vector: AigBitVector::from_bit(AigOperand::from(dup)),
        });
        // Now unduplicate with deterministic RNG
        let mut rng = StdRng::seed_from_u64(42);
        let dead = unduplicate(&mut g, &mut rng).expect("should unduplicate");
        // After unduplication, all references to 'dead' should be replaced
        for node in &g.gates {
            match node {
                AigNode::And2 { a, b, .. } => {
                    assert!(
                        a.node != dead && b.node != dead,
                        "Reference to dead node remains"
                    );
                }
                _ => {}
            }
        }
    }
}
