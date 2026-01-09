// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform implementing:
///
/// `sel(p, cases=[a, a]) ↔ a`
///
/// Since there is no "alias" node payload in PIR, the fold direction rewrites
/// to `identity(a)`, which is semantics-preserving and still removes the `sel`.
#[derive(Debug)]
pub struct SelSameArmsFoldTransform;

impl SelSameArmsFoldTransform {
    fn next_text_id(f: &IrFn) -> usize {
        f.nodes
            .iter()
            .map(|n| n.text_id)
            .max()
            .unwrap_or(0)
            .saturating_add(1)
    }

    fn is_u1_selector(f: &IrFn, selector: NodeRef) -> bool {
        matches!(f.get_node(selector).ty, Type::Bits(1))
    }

    fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }

    fn compute_fanout_cone(f: &IrFn, root: NodeRef) -> HashSet<NodeRef> {
        let users_map = compute_users(f);
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

impl PirTransform for SelSameArmsFoldTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::SelSameArmsFold
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();

        for nr in f.node_refs() {
            let node = f.get_node(nr);
            match &node.payload {
                // Fold: sel(p, [a,a]) -> identity(a)
                NodePayload::Sel {
                    selector,
                    cases,
                    default,
                } => {
                    if cases.len() != 2 || default.is_some() {
                        continue;
                    }
                    if cases[0] != cases[1] {
                        continue;
                    }
                    if !Self::is_u1_selector(f, *selector) {
                        continue;
                    }
                    // Must be able to represent result as identity(bits[w]).
                    let w = Self::bits_width(f, cases[0]);
                    if w.is_some() && w == Self::bits_width(f, nr) {
                        out.push(TransformLocation::Node(nr));
                    }
                }

                // Expand: a -> sel(p, [a,a]) for some safe predicate p (bits[1]).
                //
                // We avoid GetParam/Nil targets (they have structural invariants) and
                // avoid choosing a selector that depends on the target (cycle risk).
                NodePayload::GetParam(_) | NodePayload::Nil => {}
                _ => {
                    let Some(w) = Self::bits_width(f, nr) else {
                        continue;
                    };

                    let fanout_cone = Self::compute_fanout_cone(f, nr);
                    let mut chosen_p: Option<NodeRef> = None;
                    for cand in f.node_refs() {
                        if cand == nr {
                            continue;
                        }
                        if !Self::is_u1_selector(f, cand) {
                            continue;
                        }
                        if fanout_cone.contains(&cand) {
                            // Would introduce a dependency cycle: nr -> cand and cand ->* nr.
                            continue;
                        }
                        chosen_p = Some(cand);
                        break;
                    }
                    if chosen_p.is_some() && w > 0 {
                        out.push(TransformLocation::Node(nr));
                    }
                }
            }
        }

        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "SelSameArmsFoldTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            // Fold: sel(p, [a,a]) -> identity(a)
            NodePayload::Sel {
                selector,
                cases,
                default,
            } => {
                if cases.len() != 2 || default.is_some() || cases[0] != cases[1] {
                    return Err(
                        "SelSameArmsFoldTransform: expected 2-case sel with identical arms"
                            .to_string(),
                    );
                }
                if !Self::is_u1_selector(f, selector) {
                    return Err("SelSameArmsFoldTransform: selector must be bits[1]".to_string());
                }
                let a = cases[0];
                let w = Self::bits_width(f, a)
                    .filter(|wa| Some(*wa) == Self::bits_width(f, target_ref))
                    .ok_or_else(|| {
                        "SelSameArmsFoldTransform: sel arms/output must be bits[w] with same width"
                            .to_string()
                    })?;
                let _ = w;

                f.get_node_mut(target_ref).payload = NodePayload::Unop(Unop::Identity, a);
                Ok(())
            }

            // Expand: a -> sel(p, [a,a]) by cloning the current node into a new node,
            // then selecting between that new node twice.
            NodePayload::GetParam(_) | NodePayload::Nil => {
                Err("SelSameArmsFoldTransform: refusing to wrap GetParam/Nil nodes".to_string())
            }
            _ => {
                let Some(w) = Self::bits_width(f, target_ref) else {
                    return Err("SelSameArmsFoldTransform: only supports bits[w] nodes".to_string());
                };

                let fanout_cone = Self::compute_fanout_cone(f, target_ref);
                let mut chosen_p: Option<NodeRef> = None;
                for cand in f.node_refs() {
                    if cand == target_ref {
                        continue;
                    }
                    if !Self::is_u1_selector(f, cand) {
                        continue;
                    }
                    if fanout_cone.contains(&cand) {
                        continue;
                    }
                    chosen_p = Some(cand);
                    break;
                }
                let p = chosen_p.ok_or_else(|| {
                    "SelSameArmsFoldTransform: no safe bits[1] selector available".to_string()
                })?;

                // Clone the current node as a new node `a_clone`.
                let mut cloned = f.get_node(target_ref).clone();
                cloned.text_id = Self::next_text_id(f);
                cloned.name = None;
                let a_clone = NodeRef {
                    index: f.nodes.len(),
                };
                f.nodes.push(cloned);

                f.get_node_mut(target_ref).payload = NodePayload::Sel {
                    selector: p,
                    cases: vec![a_clone, a_clone],
                    default: None,
                };

                // Ensure the output type remains bits[w] (it does, since we didn't change ty).
                let _ = w;
                Ok(())
            }
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}
