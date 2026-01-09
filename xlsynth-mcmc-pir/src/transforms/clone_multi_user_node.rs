// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A transform that clones a node with multiple users so that afterwards each
/// user refers to a distinct replica of the node.
///
/// This is semantics-preserving: all replicas compute the same value as the
/// original node, they are simply duplicated to reduce fanout.
#[derive(Debug)]
pub struct CloneMultiUserNodeTransform;

impl PirTransform for CloneMultiUserNodeTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::CloneMultiUserNode
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let users_map = compute_users(f);
        users_map
            .iter()
            .filter_map(|(nr, users)| {
                // Do not clone parameter nodes; they are expected to always
                // have names, and cloning them can violate invariants used by
                // pretty-printing and other utilities.
                let node = f.get_node(*nr);
                match &node.payload {
                    NodePayload::GetParam(_) => None,
                    NodePayload::Nil => None,
                    _ if users.len() > 1 => Some(TransformLocation::Node(*nr)),
                    _ => None,
                }
            })
            .collect()
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "CloneMultiUserNode: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let users_map = compute_users(f);
        let users = users_map.get(&target_ref).ok_or_else(|| {
            format!(
                "CloneMultiUserNode: NodeRef {:?} not found in users map",
                target_ref
            )
        })?;

        if users.len() <= 1 {
            return Err("CloneMultiUserNode: target node does not have multiple users".to_string());
        }

        // Stable ordering of users by index for determinism.
        let mut users_vec: Vec<NodeRef> = users.iter().copied().collect();
        users_vec.sort_by_key(|nr| nr.index);

        let original_node = f.get_node(target_ref).clone();
        let mut next_text_id = f
            .nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1);

        for (i, user_nr) in users_vec.into_iter().enumerate() {
            let replacement_ref = if i == 0 {
                // The first user keeps using the original node.
                target_ref
            } else {
                // Subsequent users each get their own cloned node appended to `f.nodes`.
                let mut cloned = original_node.clone();
                cloned.text_id = next_text_id;
                next_text_id = next_text_id.saturating_add(1);
                // Clear the name so that textual IR emission does not produce
                // duplicate node identifiers. The text_id still uniquely
                // identifies the clone, and semantics are unchanged.
                cloned.name = None;
                let new_index = f.nodes.len();
                f.nodes.push(cloned);
                NodeRef { index: new_index }
            };

            let user_node = f.get_node_mut(user_nr);
            user_node.payload = remap_payload_with(&user_node.payload, |(_slot, dep)| {
                if dep == target_ref {
                    replacement_ref
                } else {
                    dep
                }
            });
        }

        Ok(())
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}
