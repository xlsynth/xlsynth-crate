// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A transform that rewires one operand of a node to a different node with the
/// same type.
///
/// This is not semantics-preserving in general. It is intended to be used with
/// an equivalence oracle in the MCMC loop.
#[derive(Debug)]
pub struct RewireOperandToSameTypeTransform;

impl RewireOperandToSameTypeTransform {
    const MAX_CANDIDATES: usize = 2000;

    fn operand_pairs(payload: &NodePayload) -> Vec<(usize, NodeRef)> {
        let mut pairs: Vec<(usize, NodeRef)> = Vec::new();
        let _ = remap_payload_with(payload, |(slot, dep)| {
            pairs.push((slot, dep));
            dep
        });
        pairs
    }

    fn node_type(f: &IrFn, nr: NodeRef) -> Type {
        f.get_node(nr).ty.clone()
    }

    fn compute_fanout_cone(
        users_map: &std::collections::HashMap<NodeRef, HashSet<NodeRef>>,
        root: NodeRef,
    ) -> HashSet<NodeRef> {
        let mut visited: HashSet<NodeRef> = HashSet::new();
        let mut work: VecDeque<NodeRef> = VecDeque::new();
        visited.insert(root);
        work.push_back(root);

        while let Some(cur) = work.pop_front() {
            if let Some(users) = users_map.get(&cur) {
                for u in users {
                    if visited.insert(*u) {
                        work.push_back(*u);
                    }
                }
            }
        }
        visited
    }
}

impl PirTransform for RewireOperandToSameTypeTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::RewireOperandToSameType
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let users_map = compute_users(f);

        // Deterministic enumeration order:
        //   - node index ascending
        //   - operand slot ascending (as reported by remap_payload_with)
        //   - replacement node index ascending
        let mut out: Vec<TransformLocation> = Vec::new();

        for i in 0..f.nodes.len() {
            if out.len() >= Self::MAX_CANDIDATES {
                break;
            }

            let node_ref = NodeRef { index: i };
            let node = f.get_node(node_ref);
            let pairs = Self::operand_pairs(&node.payload);
            if pairs.is_empty() {
                continue;
            }

            // To avoid introducing cycles, we disallow rewiring a node's operand to any
            // node in the node's fanout cone (i.e., any node that depends on
            // `node_ref`).
            //
            // If `new_operand` depends on `node_ref`, then adding an edge `node_ref ->
            // new_operand` would create a cycle.
            let fanout_cone = Self::compute_fanout_cone(&users_map, node_ref);

            for (slot, old_dep) in pairs {
                if out.len() >= Self::MAX_CANDIDATES {
                    break;
                }
                let old_ty = Self::node_type(f, old_dep);
                for repl_i in 0..f.nodes.len() {
                    if out.len() >= Self::MAX_CANDIDATES {
                        break;
                    }
                    let new_operand = NodeRef { index: repl_i };
                    if new_operand == node_ref {
                        // Self-dependency is always a cycle.
                        continue;
                    }
                    if new_operand == old_dep {
                        continue;
                    }
                    if fanout_cone.contains(&new_operand) {
                        // Would introduce a cycle.
                        continue;
                    }
                    if Self::node_type(f, new_operand) != old_ty {
                        continue;
                    }
                    out.push(TransformLocation::RewireOperand {
                        node: node_ref,
                        operand_slot: slot,
                        new_operand,
                    });
                }
            }
        }

        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let (node_ref, operand_slot, new_operand) = match loc {
            TransformLocation::RewireOperand {
                node,
                operand_slot,
                new_operand,
            } => (*node, *operand_slot, *new_operand),
            TransformLocation::Node(_) => {
                return Err(
                    "RewireOperandToSameType: expected TransformLocation::RewireOperand, got Node"
                        .to_string(),
                );
            }
        };

        if node_ref.index >= f.nodes.len() || new_operand.index >= f.nodes.len() {
            return Err("RewireOperandToSameType: node ref out of bounds".to_string());
        }
        if node_ref == new_operand {
            return Err(
                "RewireOperandToSameType: cannot rewire an operand to the node itself".to_string(),
            );
        }

        // Safety check: reject rewires that would introduce cycles.
        let users_map = compute_users(f);
        let fanout_cone = Self::compute_fanout_cone(&users_map, node_ref);
        if fanout_cone.contains(&new_operand) {
            return Err("RewireOperandToSameType: rewire would introduce a cycle".to_string());
        }

        let old_payload = f.get_node(node_ref).payload.clone();
        let new_payload = remap_payload_with(&old_payload, |(slot, dep)| {
            if slot == operand_slot {
                new_operand
            } else {
                dep
            }
        });

        f.get_node_mut(node_ref).payload = new_payload;
        Ok(())
    }

    fn always_equivalent(&self) -> bool {
        false
    }
}
