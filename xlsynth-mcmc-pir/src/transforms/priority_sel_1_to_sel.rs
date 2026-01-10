// SPDX-License-Identifier: Apache-2.0

use super::*;

/// A semantics-preserving transform implementing:
///
/// `priority_sel(p:bits[1], cases=[a], default=b) ↔ sel(p, cases=[b, a])`
#[derive(Debug)]
pub struct PrioritySel1ToSelTransform;

impl PrioritySel1ToSelTransform {
    fn is_u1_selector(f: &IrFn, selector: NodeRef) -> bool {
        matches!(f.get_node(selector).ty, Type::Bits(1))
    }

    fn bits_width(f: &IrFn, r: NodeRef) -> Option<usize> {
        match f.get_node(r).ty {
            Type::Bits(w) => Some(w),
            _ => None,
        }
    }
}

impl PirTransform for PrioritySel1ToSelTransform {
    fn kind(&self) -> PirTransformKind {
        PirTransformKind::PrioritySel1ToSel
    }

    fn find_candidates(&mut self, f: &IrFn) -> Vec<TransformLocation> {
        let mut out: Vec<TransformLocation> = Vec::new();
        for nr in f.node_refs() {
            match &f.get_node(nr).payload {
                // priority_sel(p, cases=[a], default=b) -> sel(p, cases=[b,a])
                NodePayload::PrioritySel {
                    selector,
                    cases,
                    default,
                } => {
                    if cases.len() != 1 {
                        continue;
                    }
                    let Some(default_ref) = default else {
                        continue;
                    };
                    if !Self::is_u1_selector(f, *selector) {
                        continue;
                    }
                    let a = cases[0];
                    let b = *default_ref;
                    let wa = Self::bits_width(f, a);
                    let wb = Self::bits_width(f, b);
                    let wout = Self::bits_width(f, nr);
                    if wa.is_some() && wa == wb && wa == wout {
                        out.push(TransformLocation::Node(nr));
                    }
                }
                // sel(p, cases=[b,a]) -> priority_sel(p, cases=[a], default=b)
                NodePayload::Sel {
                    selector,
                    cases,
                    default,
                } => {
                    if cases.len() != 2 || default.is_some() {
                        continue;
                    }
                    if !Self::is_u1_selector(f, *selector) {
                        continue;
                    }
                    let b = cases[0];
                    let a = cases[1];
                    let wa = Self::bits_width(f, a);
                    let wb = Self::bits_width(f, b);
                    let wout = Self::bits_width(f, nr);
                    if wa.is_some() && wa == wb && wa == wout {
                        out.push(TransformLocation::Node(nr));
                    }
                }
                _ => {}
            }
        }
        out
    }

    fn apply(&self, f: &mut IrFn, loc: &TransformLocation) -> Result<(), String> {
        let target_ref = match loc {
            TransformLocation::Node(nr) => *nr,
            TransformLocation::RewireOperand { .. } => {
                return Err(
                    "PrioritySel1ToSelTransform: expected TransformLocation::Node, got RewireOperand"
                        .to_string(),
                );
            }
        };

        let target_payload = f.get_node(target_ref).payload.clone();
        match target_payload {
            NodePayload::PrioritySel {
                selector,
                cases,
                default,
            } => {
                if cases.len() != 1 {
                    return Err(
                        "PrioritySel1ToSelTransform: expected priority_sel with 1 case"
                            .to_string(),
                    );
                }
                let Some(b) = default else {
                    return Err(
                        "PrioritySel1ToSelTransform: expected priority_sel to have a default"
                            .to_string(),
                    );
                };
                if !Self::is_u1_selector(f, selector) {
                    return Err(
                        "PrioritySel1ToSelTransform: selector must be bits[1]".to_string(),
                    );
                }
                let a = cases[0];
                let w = Self::bits_width(f, a)
                    .filter(|wa| Some(*wa) == Self::bits_width(f, b))
                    .filter(|wa| Some(*wa) == Self::bits_width(f, target_ref))
                    .ok_or_else(|| {
                        "PrioritySel1ToSelTransform: cases/default/output must be bits[w] with same width"
                            .to_string()
                    })?;
                let _ = w;

                f.get_node_mut(target_ref).payload = NodePayload::Sel {
                    selector,
                    cases: vec![b, a],
                    default: None,
                };
                Ok(())
            }
            NodePayload::Sel {
                selector,
                cases,
                default,
            } => {
                if cases.len() != 2 || default.is_some() {
                    return Err(
                        "PrioritySel1ToSelTransform: expected 2-case sel without default"
                            .to_string(),
                    );
                }
                if !Self::is_u1_selector(f, selector) {
                    return Err(
                        "PrioritySel1ToSelTransform: selector must be bits[1]".to_string(),
                    );
                }
                let b = cases[0];
                let a = cases[1];
                let w = Self::bits_width(f, a)
                    .filter(|wa| Some(*wa) == Self::bits_width(f, b))
                    .filter(|wa| Some(*wa) == Self::bits_width(f, target_ref))
                    .ok_or_else(|| {
                        "PrioritySel1ToSelTransform: cases/output must be bits[w] with same width"
                            .to_string()
                    })?;
                let _ = w;

                f.get_node_mut(target_ref).payload = NodePayload::PrioritySel {
                    selector,
                    cases: vec![a],
                    default: Some(b),
                };
                Ok(())
            }
            _ => Err(
                "PrioritySel1ToSelTransform: expected priority_sel or sel payload at target location"
                    .to_string(),
            ),
        }
    }

    fn always_equivalent(&self) -> bool {
        true
    }
}
